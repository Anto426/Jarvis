import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def clip_gradients(model, max_norm):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)