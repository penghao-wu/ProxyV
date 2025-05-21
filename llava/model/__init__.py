import os

AVAILABLE_MODELS = {
    "llava_llama": "LlavaLlamaForCausalLM, LlavaConfig",
    "llava_qwen": "LlavaQwenForCausalLM, LlavaQwenConfig",
    "llava_phi3": "LlavaPhiForCausalLM, LlavaPhiConfig",
    # "llava_qwen_moe": "LlavaQwenMoeForCausalLM, LlavaQwenMoeConfig",    
    # Add other models as needed
}

for model_name, model_classes in AVAILABLE_MODELS.items():
    exec(f"from .language_model.{model_name} import {model_classes}")
