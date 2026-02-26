import os
from datasets import load_dataset

def load_training_dataset(split_validation=False, val_ratio=0.02):

    data_path = r"data\shards"

    print("CWD:", os.getcwd())
    print("Looking for:", os.path.abspath(data_path))
    print("Exists:", os.path.exists(data_path))

    dataset = load_dataset(
        "parquet",
        data_files={
            "train": os.path.join(data_path, "packed_*.parquet")
        }
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