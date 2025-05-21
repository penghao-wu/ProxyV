import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm

import numpy as np

class ProxyInitializer(nn.Module):
	def __init__(self, d_model, d_hidden, n):
		"""
		d_model: embedding dimension for both full tokens and proxy tokens
		"""
		super().__init__()
		self.Wk = nn.Linear(d_model, d_hidden, bias=False)

		self.proxyv_tokens = nn.Parameter(torch.randn((n, d_hidden))/torch.sqrt(torch.tensor(d_hidden*4)))

	def forward(self, full_tokens, compress_reduce_factor, single_crop_len):
		"""
		full_tokens: Tensor of shape (batch_size, N, d_model)
		proxyv_tokens: Tensor of shape (batch_size, n, d_model)

		Returns:
		  proxyv_out: updated proxy tokens (batch_size, n, d_model)
		  attn: attention matrix from proxy->full (batch_size, n, N)
		"""
		proxyv_tokens = self.proxyv_tokens.unsqueeze(0).repeat(full_tokens.shape[0], 1, 1)
		Q = proxyv_tokens
		K = self.Wk(full_tokens)
		V = full_tokens

		d_model = Q.shape[-1]
		attn_logits = torch.bmm(Q, K.transpose(1, 2)) / (d_model ** 0.5)

		attn = nn.functional.softmax(attn_logits, dim=-1)

		attn_logits_T = attn_logits.transpose(1, 2)
		attn_T = nn.functional.softmax(attn_logits_T, dim=-1)

		# Update the proxy tokens by pooling full tokens
		proxyv_out = torch.bmm(attn, V)
		return proxyv_out, attn_T

def splat_proxyv_tokens(proxyv_tokens, attn):
	return torch.bmm(attn, proxyv_tokens)


class VisionMLP(nn.Module):
	def __init__(self, config, intermediate_size):
		super().__init__()
		self.context_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
		self.input_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
		self.proj = nn.Sequential(
			nn.Linear(intermediate_size*2, intermediate_size, bias=False),
			nn.SiLU(),
			nn.Linear(intermediate_size, config.hidden_size, bias=False)
		)
		self.layernorm_post = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

	def forward(self, image_full, image_compress, compress_reduce_factor, per_crop_token_len=576, learn_proxy=False, proxyv_attn=None):
		side_len_full = int(per_crop_token_len**0.5)
		side_len_compress = side_len_full // compress_reduce_factor

		num_image_crops = image_full.shape[1]//per_crop_token_len
		bs = image_full.shape[0]

		if learn_proxy:
			image_full = image_full.view(bs*num_image_crops, side_len_full*side_len_full, -1)
			image_compress = image_compress.view(bs*num_image_crops, side_len_compress*side_len_compress, -1)
			image_compress = splat_proxyv_tokens(image_compress, proxyv_attn)
			image_compress = self.context_proj(image_compress)
		else:
			image_full = image_full.view(bs*num_image_crops, side_len_full, side_len_full, -1)
			image_compress = image_compress.view(bs*num_image_crops, side_len_compress, side_len_compress, -1)
			image_compress = self.context_proj(image_compress)
			image_compress = image_compress.repeat_interleave(compress_reduce_factor, 1).repeat_interleave(compress_reduce_factor, 2)
		residual = image_full
		image_full = self.input_proj(image_full)
		image_full = torch.cat([image_full, image_compress], -1)
		image_full = self.layernorm_post(self.proj(image_full) + residual) 

		image_full = image_full.view(bs, num_image_crops*side_len_full*side_len_full, -1)

		return image_full



