import torch
from transformers import GPTNeoXForCausalLM
from .architecture import load_model_config


def initialize_model(device="cuda"):

    config = load_model_config()
    model = GPTNeoXForCausalLM(config)

    model.to(device)

    model.config.use_cache = False

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {total_params / 1e6:.2f}M parameters")

    return model