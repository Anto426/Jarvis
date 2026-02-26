import os
import yaml
from datasets import load_dataset

def load_training_dataset():

    with open("config/paths.yaml", "r") as f:
        paths = yaml.safe_load(f)["paths"]

    shard_dir = paths["shards_dir"]

    files = [
        os.path.join(shard_dir, f)
        for f in os.listdir(shard_dir)
        if f.startswith("packed_") and f.endswith(".parquet")
    ]

    dataset = load_dataset(
        "parquet",
        data_files=files,
        split="train"
    )

    dataset = dataset.with_format("torch")

    return dataset