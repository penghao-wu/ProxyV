from PIL import Image
import torch


from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path



def process(image, question, tokenizer, image_processor, model_config):
    qs = question
    if model_config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print(prompt)
    
    image_size = [image.size]
    image_tensor = process_images([image], image_processor, model_config).to(torch.float16)
    # image_tensor = process_images([image], image_processor, model_config).to(torch.bfloat16)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    return input_ids, image_tensor, image_size, prompt


conv_mode = "v1"
temperature = 0.0
model_path = "craigwu/proxyv_vicuna_7b_layer12"
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, multimodal=True,attn_implementation='sdpa')

while True:
    image_path = input("image path: ")
    image = Image.open(image_path).convert('RGB')
    question = input("question: ")

    input_ids, image_tensor, image_sizes, prompt = process(image, question, tokenizer, image_processor, model.config)
    input_ids = input_ids.to(device='cuda', non_blocking=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            num_beams=1,
            max_new_tokens=512,
            use_cache=True)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    print(outputs)