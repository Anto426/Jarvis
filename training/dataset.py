import os

import torch

from data_pipeline.sample_format import IGNORE_INDEX, PAD_ID
from training.paths import configure_cache_env, get_path


configure_cache_env()
from datasets import load_dataset


def load_training_dataset(split_validation=False, val_ratio=0.02):

    data_path = get_path("shards_dir", create=True)
    data_pattern = os.path.join(str(data_path), "packed_*.parquet")

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

    keep_columns = [
        column
        for column in ("input_ids", "labels", "attention_mask")
        if column in dataset.column_names
    ]
    dataset.set_format(type=None, columns=keep_columns)

    if not split_validation:
        return dataset

    split = dataset.train_test_split(
        test_size=val_ratio,
        seed=42
    )

    split["train"].set_format(type=None, columns=keep_columns)
    split["test"].set_format(type=None, columns=keep_columns)

    return split["train"], split["test"]


def _as_list(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return list(value)


def _truncate(values, max_length=None, keep="tail"):
    if max_length is None or len(values) <= max_length:
        return values
    if keep == "head":
        return values[:max_length]
    return values[-max_length:]


def causal_lm_collate(
    batch,
    pad_token_id=PAD_ID,
    label_pad_id=IGNORE_INDEX,
    max_sequence_length=None,
    truncation_keep="tail",
):
    max_length = max(
        min(len(item["input_ids"]), max_sequence_length or len(item["input_ids"]))
        for item in batch
    )

    input_ids = []
    labels = []
    attention_mask = []

    for item in batch:
        ids = _truncate(_as_list(item["input_ids"]), max_sequence_length, truncation_keep)
        item_labels = _truncate(
            _as_list(item.get("labels", ids)),
            max_sequence_length,
            truncation_keep,
        )
        mask = _truncate(
            _as_list(item.get("attention_mask", [1] * len(ids))),
            max_sequence_length,
            truncation_keep,
        )
        padding = max_length - len(ids)

        input_ids.append(ids + [pad_token_id] * padding)
        labels.append(item_labels + [label_pad_id] * padding)
        attention_mask.append(mask + [0] * padding)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }
