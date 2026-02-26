import yaml
from transformers import GPTNeoXConfig


def load_model_config(config_path="config/model.yaml"):

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)["model"]

    config = GPTNeoXConfig(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        intermediate_size=cfg["intermediate_size"],
        max_position_embeddings=cfg["max_position_embeddings"],
        rotary_pct=cfg["rotary_pct"],
        rotary_emb_base=cfg["rotary_emb_base"],
        use_parallel_residual=cfg["use_parallel_residual"],
        hidden_dropout=cfg["hidden_dropout"],
        attention_dropout=cfg["attention_dropout"],
        layer_norm_eps=cfg["layer_norm_eps"],
        initializer_range=cfg["initializer_range"],
        tie_word_embeddings=cfg["tie_word_embeddings"],
        hidden_act=cfg["hidden_act"]
    )

    return config