import yaml
from transformers import GPTNeoXConfig


def load_model_config(config_path="config/model.yaml"):

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)["model"]

    config = GPTNeoXConfig(
        vocab_size=int(cfg["vocab_size"]),
        hidden_size=int(cfg["hidden_size"]),
        num_hidden_layers=int(cfg["num_hidden_layers"]),
        num_attention_heads=int(cfg["num_attention_heads"]),
        intermediate_size=int(cfg["intermediate_size"]),
        max_position_embeddings=int(cfg["max_position_embeddings"]),

        rotary_pct=float(cfg["rotary_pct"]),
        rotary_emb_base=int(cfg["rotary_emb_base"]),

        use_parallel_residual=bool(cfg["use_parallel_residual"]),

        hidden_dropout=float(cfg["hidden_dropout"]),
        attention_dropout=float(cfg["attention_dropout"]),
        layer_norm_eps=float(cfg["layer_norm_eps"]),
        initializer_range=float(cfg["initializer_range"]),

        tie_word_embeddings=bool(cfg["tie_word_embeddings"]),
        hidden_act=str(cfg["hidden_act"])
    )

    return config
