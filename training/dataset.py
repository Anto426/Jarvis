import os
from training.paths import configure_cache_env, get_path


configure_cache_env()
from datasets import load_dataset

def load_training_dataset(split_validation=False, val_ratio=0.02):

    data_path = get_path("shards_dir", create=True)
    data_pattern = os.path.join(str(data_path), "packed_*.parquet")

    print("CWD:", os.getcwd())
    print("Looking for:", data_pattern)
    print("Exists:", data_path.exists())

    if not list(data_path.glob("packed_*.parquet")):
        raise FileNotFoundError(
            f"Nessuno shard packed trovato in {data_path}. "
            "Esegui prima data_pipeline/pack_blocks.py."
        )

    dataset = load_dataset(
        "parquet",
        data_files={
            "train": data_pattern
        },
        cache_dir=os.environ.get("HF_DATASETS_CACHE"),
    )["train"]

    # ✅ SOLO input_ids
    dataset.set_format(
        type="torch",
        columns=["input_ids"]
    )

    if not split_validation:
        return dataset

    split = dataset.train_test_split(
        test_size=val_ratio,
        seed=42
    )

    split["train"].set_format(type="torch", columns=["input_ids"])
    split["test"].set_format(type="torch", columns=["input_ids"])

    return split["train"], split["test"]
