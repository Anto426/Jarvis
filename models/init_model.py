import torch
from transformers import GPTNeoXForCausalLM
from .architecture import load_model_config


def initialize_model(device=None, attn_implementation=None):

    config = load_model_config()
    if attn_implementation:
        config._attn_implementation = attn_implementation

    model = GPTNeoXForCausalLM(config)

    if device:
        model.to(device)

    model.config.use_cache = False

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {total_params / 1e6:.2f}M parameters")

    return model
