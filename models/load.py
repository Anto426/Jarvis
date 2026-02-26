import os
import yaml
import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer


def load_model_and_tokenizer(checkpoint_path=None):

    with open("config/paths.yaml", "r") as f:
        paths = yaml.safe_load(f)["paths"]

    tokenizer_path = paths["tokenizer_dir"]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if checkpoint_path is None:
        checkpoint_path = paths["model_output_dir"]

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Checkpoint non trovato.")

    model = GPTNeoXForCausalLM.from_pretrained(checkpoint_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model, tokenizer