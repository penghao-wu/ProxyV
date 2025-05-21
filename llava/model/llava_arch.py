#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import math
import re
import time
import numpy as np
import torch
import torch.nn as nn
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector
from .vision_mlp import VisionMLP, ProxyInitializer

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import random

class LlavaMetaModel:

	def __init__(self, config):
		super(LlavaMetaModel, self).__init__(config)

		if hasattr(config, "mm_vision_tower"):
			delay_load = getattr(config, "delay_load", False)
			self.vision_tower = build_vision_tower(config, delay_load=delay_load)
			self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
			self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

			if config.proxyv:
				hidden_size_reduce_factor = 4 if config.hidden_size >= 1024 else 2
				config.num_of_vision_mlp_layers = config.num_hidden_layers - config.proxyv_start_layer
				self.vision_mlp_layers = nn.ModuleList(
						[VisionMLP(self.config, self.config.hidden_size//hidden_size_reduce_factor) for layer_idx in range(0, config.num_of_vision_mlp_layers)]
						)
				learn_proxy = getattr(config, 'learn_proxy', False)
				if learn_proxy:
					self.proxyv_initializer = ProxyInitializer(config.hidden_size, config.hidden_size//hidden_size_reduce_factor, n=(config.per_crop_token_len//(config.proxyv_reduce_factor**2)))

			self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

	def get_vision_tower(self):
		vision_tower = getattr(self, "vision_tower", None)
		if type(vision_tower) is list:
			vision_tower = vision_tower[0]
		return vision_tower

	def initialize_vision_modules(self, model_args, fsdp=None):
		vision_tower = model_args.vision_tower
		mm_vision_select_layer = model_args.mm_vision_select_layer
		mm_vision_select_feature = model_args.mm_vision_select_feature
		pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
		mm_patch_merge_type = model_args.mm_patch_merge_type
		hidden_size_reduce_factor = model_args.hidden_size_reduce_factor

		per_crop_token_len = model_args.per_crop_token_len
		proxyv_reduce_factor = model_args.proxyv_reduce_factor
		proxyv = model_args.proxyv
		learn_proxy = model_args.learn_proxy
		proxyv_start_layer = model_args.proxyv_start_layer

		self.config.mm_vision_tower = vision_tower
		self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")
		self.config.hidden_size_reduce_factor = hidden_size_reduce_factor
		self.config.per_crop_token_len = per_crop_token_len
		self.config.proxyv_reduce_factor = proxyv_reduce_factor
		self.config.proxyv = proxyv
		self.config.learn_proxy = learn_proxy
		self.config.proxyv_start_layer = proxyv_start_layer

		if self.get_vision_tower() is None:
			vision_tower = build_vision_tower(model_args)
			vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
			for k, v in vision_resampler.config.items():
				setattr(self.config, k, v)

			if fsdp is not None and len(fsdp) > 0:
				self.vision_tower = [vision_tower]
				self.vision_resampler = [vision_resampler]
			else:
				self.vision_tower = vision_tower
				self.vision_resampler = vision_resampler
		else:
			if fsdp is not None and len(fsdp) > 0:
				vision_resampler = self.vision_resampler[0]
				vision_tower = self.vision_tower[0]
			else:
				vision_resampler = self.vision_resampler
				vision_tower = self.vision_tower
			vision_tower.load_model()

			# In case it is frozen by LoRA
			for p in self.vision_resampler.parameters():
				p.requires_grad = True

		self.config.use_mm_proj = True
		self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
		self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
		self.config.mm_vision_select_layer = mm_vision_select_layer
		self.config.mm_vision_select_feature = mm_vision_select_feature
		self.config.mm_patch_merge_type = mm_patch_merge_type
		self.config.image_token_len_per_side = int(model_args.per_crop_token_len**0.5)

		if not hasattr(self.config, 'add_faster_video'):
			if model_args.add_faster_video:
				embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
				self.faster_token = nn.Parameter(
					torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
				)

		if getattr(self, "mm_projector", None) is None:
			self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)
			num_of_vision_mlp_layers = self.config.num_hidden_layers - proxyv_start_layer
			self.config.num_of_vision_mlp_layers = num_of_vision_mlp_layers
			hidden_size_reduce_factor = hidden_size_reduce_factor
			if proxyv:
				num_of_vision_mlp_layers = self.config.num_hidden_layers - proxyv_start_layer
				self.config.num_of_vision_mlp_layers = num_of_vision_mlp_layers
				hidden_size_reduce_factor = 4 if self.config.hidden_size >= 1024 else 2
				self.vision_mlp_layers = nn.ModuleList(
					[VisionMLP(self.config, self.config.hidden_size//hidden_size_reduce_factor) for layer_idx in range(0, num_of_vision_mlp_layers)]
					)

			# if "unpad" in mm_patch_merge_type:
			embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
			self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
			if learn_proxy:
				self.proxyv_initializer = ProxyInitializer(self.config.hidden_size, self.config.hidden_size//4, n=(per_crop_token_len//(proxyv_reduce_factor**2)))
		else:
			# In case it is frozen by LoRA
			for p in self.mm_projector.parameters():
				p.requires_grad = True

		if pretrain_mm_mlp_adapter is not None:
			mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

			def get_w(weights, keyword):
				return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}
			self.image_newline.data = mm_projector_weights['model.image_newline']
			incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
			rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
			incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
			rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
			if proxyv:
				incompatible_keys = self.vision_mlp_layers.load_state_dict(get_w(mm_projector_weights, "vision_mlp_layers"), strict=False)
				rank0_print(f"Loaded vision mlp weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
				if learn_proxy:
					incompatible_keys = self.proxyv_initializer.load_state_dict(get_w(mm_projector_weights, "proxyv_initializer"), strict=False)
					rank0_print(f"Loaded vision mlp weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")



def unpad_image(tensor, original_size):
	"""
	Unpads a PyTorch tensor of a padded and resized image.

	Args:
	tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
	original_size (tuple): The original size of the image (height, width).

	Returns:
	torch.Tensor: The unpadded image tensor.
	"""
	original_width, original_height = original_size
	current_height, current_width = tensor.shape[1:]

	# Compute aspect ratios
	original_aspect_ratio = original_width / original_height
	current_aspect_ratio = current_width / current_height

	# Determine padding size and direction
	if original_aspect_ratio > current_aspect_ratio:
		# Padding was added to the height
		scale_factor = current_width / original_width
		new_height = int(original_height * scale_factor)
		padding = (current_height - new_height) // 2
		unpadded_tensor = tensor[:, padding : current_height - padding, :]
	else:
		# Padding was added to the width
		scale_factor = current_height / original_height
		new_width = int(original_width * scale_factor)
		padding = (current_width - new_width) // 2
		unpadded_tensor = tensor[:, :, padding : current_width - padding]

	return unpadded_tensor



def get_padding_offset(cur_size, original_size):
	cur_w, cur_h = cur_size
	original_w, original_h = original_size

	original_aspect_ratio = original_w / original_h
	current_aspect_ratio = cur_w / cur_h

	if original_aspect_ratio > current_aspect_ratio:
		scale_factor = cur_w / original_w
		new_height = int(np.ceil(original_h * scale_factor))
		padding = (cur_h - new_height) // 2
		return 0, 0, padding, padding
	else:
		scale_factor = cur_h / original_h
		new_width = int(np.ceil(original_w * scale_factor))
		padding = (cur_w - new_width) // 2
		return padding, padding, 0, 0


def calculate_causal_attention_mask(position_ids_q, position_ids_kv, attention_mask, dtype=torch.bfloat16):
	min_dtype = torch.finfo(dtype).min
	bs = position_ids_q.shape[0]
	position_ids_q = position_ids_q.view(bs, -1, 1)
	position_ids_kv = position_ids_kv.view(bs, -1, 1)
	causal_mask = position_ids_q >= position_ids_kv.transpose(1, 2)
	causal_mask = causal_mask.to(dtype).view(bs, 1, position_ids_q.shape[1], position_ids_kv.shape[1])
	attention_mask_4d = attention_mask.view(bs, 1, 1, -1).repeat(1, 1, position_ids_q.shape[1], 1)

	causal_mask = causal_mask.masked_fill(causal_mask == 0, min_dtype)
	causal_mask = causal_mask.masked_fill(causal_mask == 1, 0.0)
	causal_mask = causal_mask.masked_fill(attention_mask_4d == 0, min_dtype)

	return causal_mask
	

class LlavaMetaForCausalLM(ABC):

	@abstractmethod
	def get_model(self):
		pass

	def get_vision_tower(self):
		return self.get_model().get_vision_tower()

	def get_2dPool(self, image_feature, stride=2):
		height = width = self.get_vision_tower().num_patches_per_side
		num_frames, num_tokens, num_dim = image_feature.shape
		image_feature = image_feature.view(num_frames, height, width, -1)
		image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
		if self.config.mm_spatial_pool_mode == "average":
			image_feature = nn.functional.avg_pool2d(image_feature, stride)
		elif self.config.mm_spatial_pool_mode == "max":
			image_feature = nn.functional.max_pool2d(image_feature, stride)
		elif self.config.mm_spatial_pool_mode == "bilinear":
			height, weight = image_feature.shape[2:]
			scaled_shape = [math.ceil(height / stride), math.ceil(weight / stride)]
			image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

		else:
			raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
		image_feature = image_feature.permute(0, 2, 3, 1)
		image_feature = image_feature.view(num_frames, -1, num_dim)
		return image_feature

	def encode_images(self, images):
		image_features = self.get_model().get_vision_tower()(images)
		image_features = self.get_model().mm_projector(image_features)
		return image_features
	
	def prepare_image_information(self, image_feature, image_size=None, is_dummy=False, dummy_num=1, modality='image'):
		height = width = int(self.get_model().config.per_crop_token_len**0.5)
		proxyv_reduce_factor = self.get_model().config.proxyv_reduce_factor
		height_proxy = height//proxyv_reduce_factor
		width_proxy = width//proxyv_reduce_factor

		if is_dummy:
			attention_mask_image_full = torch.zeros((dummy_num*(height*width), 1), device=image_feature.device, dtype=torch.bool)
			attention_mask_newline = torch.zeros((dummy_num, 1), device=image_feature.device, dtype=torch.bool)
			position_ids_image_full = torch.zeros((dummy_num*(height*width), 1), device=image_feature.device, dtype=torch.long)
			position_ids_newline = torch.zeros((dummy_num, 1), device=image_feature.device, dtype=torch.long)

			attention_mask_image_proxy = torch.zeros((dummy_num*height_proxy*width_proxy, 1), device=image_feature.device, dtype=torch.bool)
			position_ids_image_proxy = torch.zeros((dummy_num*height_proxy*width_proxy, 1), device=image_feature.device, dtype=torch.long)

			newline_feature = self.model.image_newline[None].repeat(dummy_num, 1)
			image_feature = image_feature.flatten(0,1)[:height*width].repeat(dummy_num, 1)

		else:
			if modality =='video':
				num_frames = image_feature.shape[0]
				image_feature = image_feature.contiguous()
				image_feature = image_feature.view(num_frames * height * width, -1)
				newline_feature = self.model.image_newline[None, :].repeat(num_frames, 1)

				attention_mask_image_full_withnewline = torch.ones((num_frames, height * width+1, 1), device=image_feature.device, dtype=torch.bool)
				position_ids_image_full_withnewline = attention_mask_image_full_withnewline.flatten(0,1).cumsum(0)-1
				position_ids_image_full_withnewline = position_ids_image_full_withnewline.view(num_frames, height * width+1, 1)

				attention_mask_image_full = attention_mask_image_full_withnewline[:, :-1].flatten(0,1)
				attention_mask_newline = attention_mask_image_full_withnewline[:, -1:].flatten(0,1)
				position_ids_image_full = position_ids_image_full_withnewline[:, :-1].flatten(0,1)
				position_ids_newline = position_ids_image_full_withnewline[:, -1:].flatten(0,1)

				attention_mask_image_proxy = torch.ones((num_frames*height_proxy*width_proxy, 1), device=image_feature.device, dtype=torch.bool)
				position_ids_image_proxy = attention_mask_image_proxy.cumsum(0)-1
			else:
				if image_feature.shape[0] > 1:
					base_image_feature = image_feature[0]
					image_feature = image_feature[1:]
					assert height * width == base_image_feature.shape[0]

					if hasattr(self.get_vision_tower(), "image_size"):
						vision_tower_image_size = self.get_vision_tower().image_size
					else:
						raise ValueError("vision_tower_image_size is not found in the vision tower.")
					try:
						num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_size, self.config.image_grid_pinpoints, vision_tower_image_size)
					except Exception as e:
						rank0_print(f"Error: {e}")
						num_patch_width, num_patch_height = 2, 2
					image_feature = image_feature.view(num_patch_height * num_patch_width * height * width, -1)

					newline_feature = self.model.image_newline[None, :].repeat(num_patch_height * num_patch_width+1, 1)

					attention_mask_image_full_withnewline = torch.ones((num_patch_height * num_patch_width, height * width+1, 1), device=image_feature.device, dtype=torch.bool)
					position_ids_image_full_withnewline = attention_mask_image_full_withnewline.flatten(0,1).cumsum(0)-1
					position_ids_image_full_withnewline = position_ids_image_full_withnewline.view(num_patch_height * num_patch_width, height * width+1, 1)

					attention_mask_image_full = attention_mask_image_full_withnewline[:, :-1].flatten(0,1)
					attention_mask_newline = attention_mask_image_full_withnewline[:, -1:].flatten(0,1)
					position_ids_image_full = position_ids_image_full_withnewline[:, :-1].flatten(0,1)
					position_ids_newline = position_ids_image_full_withnewline[:, -1:].flatten(0,1)

					# add base
					attention_mask_image_full = torch.cat([torch.ones((height * width, 1), device=image_feature.device, dtype=torch.bool), attention_mask_image_full])
					attention_mask_newline = torch.cat([torch.ones((1, 1), device=image_feature.device, dtype=torch.bool), attention_mask_newline])
					position_ids_image_full = position_ids_image_full + height * width + 1
					# position_ids_newline = position_ids_newline + height * width + 1
					position_ids_image_full = torch.cat([torch.arange(height * width, device=image_feature.device, dtype=torch.long).view(-1,1), position_ids_image_full])
					position_ids_newline = torch.cat([torch.full((1, 1), height * width, device=image_feature.device, dtype=torch.long), position_ids_newline])

					image_feature = torch.cat([base_image_feature.view(height * width, -1), image_feature])

					attention_mask_image_proxy = torch.ones(((1+num_patch_height*num_patch_width)*height_proxy*width_proxy, 1), device=image_feature.device, dtype=torch.bool)
					position_ids_image_proxy = attention_mask_image_proxy.cumsum(0)-1
				
				else:
					attention_mask_image_full_withnewline = torch.ones((height * width+1, 1), device=image_feature.device, dtype=torch.bool)
					position_ids_image_full_withnewline = attention_mask_image_full_withnewline.cumsum(0)-1

					attention_mask_image_full = attention_mask_image_full_withnewline[:-1]
					attention_mask_newline = attention_mask_image_full_withnewline[-1:]
					position_ids_image_full = position_ids_image_full_withnewline[:-1]
					position_ids_newline = position_ids_image_full_withnewline[-1:]

					attention_mask_image_proxy = torch.ones((height_proxy*width_proxy, 1), device=image_feature.device, dtype=torch.bool)
					position_ids_image_proxy = attention_mask_image_proxy.cumsum(0)-1

					newline_feature = self.model.image_newline[None]
					image_feature = image_feature.flatten(0,1)
		
		image_info = {}
		image_info['image_feature'] = image_feature
		image_info['newline_feature'] = newline_feature
		image_info['attention_mask_image_full'] = attention_mask_image_full
		image_info['position_ids_image_full'] = position_ids_image_full
		image_info['attention_mask_newline'] = attention_mask_newline
		image_info['position_ids_newline'] = position_ids_newline
		image_info['attention_mask_image_proxy'] = attention_mask_image_proxy
		image_info['position_ids_image_proxy'] = position_ids_image_proxy
		
		return image_info

	def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None):
		vision_tower = self.get_vision_tower()
		if vision_tower is None or images is None or input_ids.shape[1] == 1:
			return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None, None, None
		
		if not (type(images) is list or images.ndim == 5):
			images = [images]

		if type(images) is list:
			images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

		video_idx_in_batch = []
		for _ in range(len(modalities)):
			if modalities[_] == "video":
				video_idx_in_batch.append(_)

		images_list = []
		for image in images:
			if image.ndim == 4:
				images_list.append(image)
			else:
				images_list.append(image.unsqueeze(0))

		concat_images = torch.cat([image for image in images_list], dim=0)
		split_sizes = [image.shape[0] for image in images_list]
		encoded_image_features = self.encode_images(concat_images)

		encoded_image_features = torch.split(encoded_image_features, split_sizes)
		image_features = []
		for idx, image_feat in enumerate(encoded_image_features):
			if idx in video_idx_in_batch:
				image_features.append(self.get_2dPool(image_feat, self.config.mm_spatial_pool_stride))

			else:
				image_features.append(image_feat)
		mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

		assert mm_newline_position == "one_token"

		height = width = self.get_vision_tower().num_patches_per_side

		# TODO: image start / end is not implemented here to support pretraining.
		if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
			raise NotImplementedError
		# rank_print(f"Total images : {len(image_features)}")
		tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)

		# Let's just add dummy tensors if they do not exist,
		# it is a headache to deal with None all the time.
		# But it is not ideal, and if you have a better idea,
		# please open an issue / submit a PR, thanks.
		_labels = labels
		_position_ids = position_ids
		_attention_mask = attention_mask
		if attention_mask is None:
			attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
		else:
			attention_mask = attention_mask.bool()
		if position_ids is None:
			position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
		if labels is None:
			labels = torch.full_like(input_ids, IGNORE_INDEX)

		# remove the padding using attention_mask -- FIXME
		_input_ids = input_ids
		input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
		labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

		input_embeds_text = []
		input_embeds_image_full = []
		input_embeds_newline = []

		attention_mask_text = []
		attention_mask_image_full = []
		attention_mask_image_proxy = []
		attention_mask_newline = []
		
		position_ids_text = []
		position_ids_image_full = []
		position_ids_image_proxy = []
		position_ids_newline = []

		# this is directly the next token
		labels_text = []
		labels_image_full = []
		labels_newline = []

		cur_image_idx = 0

		max_num_image_crops = 0
		per_crop_token_len = self.get_model().config.per_crop_token_len
		proxyv_reduce_factor = self.get_model().config.proxyv_reduce_factor

		for batch_idx, cur_input_ids in enumerate(input_ids):
			num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
			num_image_crops = 0
			if num_images > 0:
				num_image_crops = sum([image.shape[0] for image in images_list][cur_image_idx:cur_image_idx+num_images])
				cur_image_idx += num_images
				max_num_image_crops = max(num_image_crops, max_num_image_crops)

		max_num_image_crops = min(max_num_image_crops, self.get_model().config.max_num_image_crops)

		cur_image_idx = 0
		# rank_print("Inserting Images embedding")
		for batch_idx, cur_input_ids in enumerate(input_ids):
			num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
			num_image_crops = 0
			if num_images > 0:
				num_image_crops = sum([image.shape[0] for image in images_list][cur_image_idx:cur_image_idx+num_images])	
			cur_seq_len = 0
			if num_images==0:
				cur_image_features = image_features[cur_image_idx]
				cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)

				image_info = self.prepare_image_information(cur_image_features, image_sizes[cur_image_idx], is_dummy=True, dummy_num=max_num_image_crops)

				input_embeds_text.append(cur_input_embeds_1)
				input_embeds_image_full.append(image_info['image_feature'])
				input_embeds_newline.append(image_info['newline_feature'])

				attention_mask_text.append(torch.ones((cur_input_ids.shape[0], 1), dtype=torch.bool, device=self.device))
				attention_mask_image_full.append(image_info['attention_mask_image_full'])
				attention_mask_image_proxy.append(image_info['attention_mask_image_proxy'])
				attention_mask_newline.append(image_info['attention_mask_newline'])


				position_ids_text.append(torch.arange(0, cur_input_ids.shape[0], dtype=torch.long, device=self.device).view(-1, 1))
				position_ids_image_full.append(image_info['position_ids_image_full'])
				position_ids_image_proxy.append(image_info['position_ids_image_proxy'])
				position_ids_newline.append(image_info['position_ids_newline'])

				labels_text.append(torch.cat([labels[batch_idx][1:], torch.full((1, ), IGNORE_INDEX, device=self.device, dtype=torch.long)]))
				labels_image_full.append(torch.full((len(image_info['position_ids_image_full']), ), IGNORE_INDEX, device=self.device, dtype=torch.long))
				labels_newline.append(torch.full((len(image_info['position_ids_newline']), ), IGNORE_INDEX, device=self.device, dtype=torch.long))
				cur_image_idx += 1
				continue

			image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
			cur_input_ids_noim = []
			cur_labels = labels[batch_idx]
			cur_labels_noim = []
			cur_input_embeds_image_full = []
			cur_input_embeds_newline = []

			cur_attention_mask_image_full = []
			cur_attention_mask_image_proxy = []
			cur_attention_mask_newline = []

			cur_position_ids_text = []
			cur_position_ids_image_full = []
			cur_position_ids_image_proxy = []
			cur_position_ids_newline = []

			cur_labels_text = []
			cur_labels_image_full = []
			cur_labels_newline = []

			for i in range(len(image_token_indices) - 1):
				cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
				cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
			split_sizes = [x.shape[0] for x in cur_labels_noim]
			cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
			cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
			input_embeds_text.append(cur_input_embeds)
			attention_mask_text.append(torch.ones((cur_input_embeds.shape[0], 1), dtype=torch.bool, device=self.device))
			
			for i in range(num_images + 1):
				if cur_input_embeds_no_im[i].shape[0] > 0:
					cur_position_ids_text.append(torch.arange(0, cur_input_embeds_no_im[i].shape[0], dtype=torch.long, device=self.device).view(-1, 1)+cur_seq_len)
				if cur_labels_noim[i].shape[0] == 1:
					cur_labels_text.append(torch.full((1, ), IGNORE_INDEX, device=self.device, dtype=torch.long))
				elif cur_labels_noim[i].shape[0] > 1:
					cur_labels_text.append(torch.cat([cur_labels_noim[i][1:], torch.full((1, ), IGNORE_INDEX, device=self.device, dtype=torch.long)]))
				cur_seq_len += cur_input_embeds_no_im[i].shape[0]
				if i < num_images:
					try:
						cur_image_features = image_features[cur_image_idx]
					except IndexError:
						cur_image_features = image_features[cur_image_idx - 1]
					if modalities[batch_idx] == 'video':
						cur_image_info = self.prepare_image_information(cur_image_features, None, modality=modalities[batch_idx])
					else:
						cur_image_info = self.prepare_image_information(cur_image_features, image_sizes[cur_image_idx], modality=modalities[batch_idx])
					cur_image_idx += 1

					cur_input_embeds_image_full.append(cur_image_info['image_feature'])
					cur_input_embeds_newline.append(cur_image_info['newline_feature'])

					cur_attention_mask_image_full.append(cur_image_info['attention_mask_image_full'])
					cur_attention_mask_image_proxy.append(cur_image_info['attention_mask_image_proxy'])
					cur_attention_mask_newline.append(cur_image_info['attention_mask_newline'])

					cur_position_ids_image_full.append(cur_image_info['position_ids_image_full'] + cur_seq_len)
					cur_position_ids_image_proxy.append(cur_image_info['position_ids_image_proxy'] + cur_seq_len)
					cur_position_ids_newline.append(cur_image_info['position_ids_newline'] + cur_seq_len)

					cur_seq_len += cur_image_info['image_feature'].shape[0] + cur_image_info['newline_feature'].shape[0]

					cur_labels_image_full.append(torch.full((len(cur_image_info['position_ids_image_full']), ), IGNORE_INDEX, device=self.device, dtype=torch.long))

					# next token is text
					if i == num_images-1 or image_token_indices[i] != image_token_indices[i+1]:
						cur_labels_newline.append(torch.cat([torch.full((len(cur_image_info['position_ids_newline'])-1, ), IGNORE_INDEX, device=self.device, dtype=torch.long), cur_labels_noim[i+1][:1]]))
					# next token is image
					else:
						cur_labels_newline.append(torch.full((len(cur_image_info['position_ids_newline']), ), IGNORE_INDEX, device=self.device, dtype=torch.long))


			# adding dummy image crops
			if num_image_crops < max_num_image_crops:
				num_dummy_image_crops = max_num_image_crops - num_image_crops
				cur_image_features = torch.zeros_like(image_features[0])[:1]
				image_info = self.prepare_image_information(cur_image_features, image_sizes[-1], is_dummy=True, dummy_num=num_dummy_image_crops)

				cur_input_embeds_image_full.append(image_info['image_feature'])
				cur_input_embeds_newline.append(image_info['newline_feature'])

				cur_attention_mask_image_full.append(image_info['attention_mask_image_full'])
				cur_attention_mask_image_proxy.append(image_info['attention_mask_image_proxy'])
				cur_attention_mask_newline.append(image_info['attention_mask_newline'])

				cur_position_ids_image_full.append(image_info['position_ids_image_full'])
				cur_position_ids_image_proxy.append(image_info['position_ids_image_proxy'])
				cur_position_ids_newline.append(image_info['position_ids_newline'])

				cur_labels_image_full.append(torch.full((len(image_info['position_ids_image_full']), ), IGNORE_INDEX, device=self.device, dtype=torch.long))
				cur_labels_newline.append(torch.full((len(image_info['position_ids_newline']), ), IGNORE_INDEX, device=self.device, dtype=torch.long))

			input_embeds_image_full.append(torch.cat(cur_input_embeds_image_full))
			input_embeds_newline.append(torch.cat(cur_input_embeds_newline))

			attention_mask_image_full.append(torch.cat(cur_attention_mask_image_full))
			attention_mask_image_proxy.append(torch.cat(cur_attention_mask_image_proxy))
			attention_mask_newline.append(torch.cat(cur_attention_mask_newline))

			position_ids_text.append(torch.cat(cur_position_ids_text))
			position_ids_image_full.append(torch.cat(cur_position_ids_image_full))
			position_ids_image_proxy.append(torch.cat(cur_position_ids_image_proxy))
			position_ids_newline.append(torch.cat(cur_position_ids_newline))

			labels_text.append(torch.cat(cur_labels_text))
			labels_image_full.append(torch.cat(cur_labels_image_full))
			labels_newline.append(torch.cat(cur_labels_newline))


		image_full_len = max_num_image_crops * per_crop_token_len
		image_proxyv_len = max_num_image_crops * (per_crop_token_len//proxyv_reduce_factor**2)
		newline_len = max_num_image_crops

		non_text_len = image_full_len + newline_len
		text_len = min(max([len(input_embeds_text[batch_idx]) for batch_idx in range(len(input_ids))]), tokenizer_model_max_length-non_text_len)

		assert getattr(self.config, "tokenizer_padding_side", "right") == "right"
		
		# pad to the max length
		for batch_idx in range(len(input_ids)):
			if len(input_embeds_text[batch_idx]) > text_len:
				input_embeds_text[batch_idx] = input_embeds_text[batch_idx][:text_len]
				attention_mask_text[batch_idx] = attention_mask_text[batch_idx][:text_len]
				position_ids_text[batch_idx] = position_ids_text[batch_idx][:text_len]
				labels_text[batch_idx] = labels_text[batch_idx][:text_len]
			elif len(input_embeds_text[batch_idx]) < text_len:
				padding_len = text_len - len(input_embeds_text[batch_idx])
				input_embeds_text[batch_idx] = torch.cat([input_embeds_text[batch_idx], torch.zeros((padding_len, input_embeds_text[batch_idx].shape[-1]), device=self.device, dtype=input_embeds_text[batch_idx].dtype)])
				attention_mask_text[batch_idx] = torch.cat([attention_mask_text[batch_idx], torch.zeros((padding_len, 1), device=self.device, dtype=torch.bool)])
				position_ids_text[batch_idx] = torch.cat([position_ids_text[batch_idx], torch.arange(position_ids_text[batch_idx].max().item()+1, position_ids_text[batch_idx].max().item()+1+padding_len, device=self.device, dtype=torch.long).view(-1, 1)])
				labels_text[batch_idx] = torch.cat([labels_text[batch_idx], torch.full((padding_len, ), IGNORE_INDEX, device=self.device, dtype=torch.long)])

		
		input_embeds_image_full = torch.stack(input_embeds_image_full)
		input_embeds_newline = torch.stack(input_embeds_newline)
		input_embeds_text = torch.stack(input_embeds_text)

		attention_mask_image_full = torch.stack(attention_mask_image_full)
		attention_mask_image_proxy = torch.stack(attention_mask_image_proxy)
		attention_mask_newline = torch.stack(attention_mask_newline)
		attention_mask_text = torch.stack(attention_mask_text)

		position_ids_image_full = torch.stack(position_ids_image_full)
		position_ids_image_proxy = torch.stack(position_ids_image_proxy)
		position_ids_newline = torch.stack(position_ids_newline)
		position_ids_text = torch.stack(position_ids_text)

		labels_image_full = torch.stack(labels_image_full)
		labels_newline = torch.stack(labels_newline)
		labels_text = torch.stack(labels_text)

		input_embeds = torch.cat([input_embeds_image_full, input_embeds_newline, input_embeds_text], 1)
		labels = torch.cat([labels_image_full, labels_newline, labels_text], 1)
		attention_mask = torch.cat([attention_mask_image_full, attention_mask_newline, attention_mask_text], 1)
		position_ids = torch.cat([position_ids_image_full, position_ids_newline, position_ids_text], 1)

		# prepare the 4D attention masks for regular attention and proxyv attention

		# regular attention: Q=[image_full, newline_full, text], KV=[image_full, newline_full, text]
		attention_mask_regular_4d = calculate_causal_attention_mask(position_ids, position_ids, attention_mask, input_embeds.dtype)

		# proxyv attention: Q=[image_proxy, newline_full, text], KV=[image_full, image_proxy, newline_full, text]
		attention_mask_proxyv_4d = calculate_causal_attention_mask(torch.cat([position_ids_image_proxy, position_ids_newline, position_ids_text], 1), torch.cat([position_ids_image_full, position_ids_image_proxy, position_ids_newline, position_ids_text], 1), torch.cat([attention_mask_image_full, attention_mask_image_proxy, attention_mask_newline, attention_mask_text], 1), input_embeds.dtype)

		min_dtype = torch.finfo(input_embeds.dtype).min
		# proxy can't attend full
		attention_mask_proxyv_4d[:, :, :image_proxyv_len, :image_full_len] = min_dtype
		# others can't attend proxy
		attention_mask_proxyv_4d[:, :, image_proxyv_len:, image_full_len:image_full_len+image_proxyv_len] = min_dtype

		attention_mask = attention_mask.view(attention_mask.shape[0], -1)
		position_ids = position_ids.view(position_ids.shape[0], -1)
		position_ids_image_proxy = position_ids_image_proxy.view(position_ids_image_proxy.shape[0], -1)
		return None, position_ids, attention_mask, past_key_values, input_embeds, labels, attention_mask_regular_4d, attention_mask_proxyv_4d, position_ids_image_proxy, max_num_image_crops

	def initialize_vision_tokenizer(self, model_args, tokenizer):
		if model_args.mm_use_im_patch_token:
			tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
			self.resize_token_embeddings(len(tokenizer))

		if model_args.mm_use_im_start_end:
			num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
			self.resize_token_embeddings(len(tokenizer))

			if num_new_tokens > 0:
				input_embeddings = self.get_input_embeddings().weight.data
				output_embeddings = self.get_output_embeddings().weight.data

				input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
				output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

				input_embeddings[-num_new_tokens:] = input_embeddings_avg
				output_embeddings[-num_new_tokens:] = output_embeddings_avg

			if model_args.tune_mm_mlp_adapter:
				for p in self.get_input_embeddings().parameters():
					p.requires_grad = True
				for p in self.get_output_embeddings().parameters():
					p.requires_grad = False

			if model_args.pretrain_mm_mlp_adapter:
				mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
				embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
				assert num_new_tokens == 2
				if input_embeddings.shape == embed_tokens_weight.shape:
					input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
				elif embed_tokens_weight.shape[0] == num_new_tokens:
					input_embeddings[-num_new_tokens:] = embed_tokens_weight
				else:
					raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
		elif model_args.mm_use_im_patch_token:
			if model_args.tune_mm_mlp_adapter:
				for p in self.get_input_embeddings().parameters():
					p.requires_grad = False
				for p in self.get_output_embeddings().parameters():
					p.requires_grad = False
