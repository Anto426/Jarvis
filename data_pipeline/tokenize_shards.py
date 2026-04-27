import json
import os
import re

import pyarrow as pa
import pyarrow.parquet as pq
import sentencepiece as spm
from tqdm import tqdm

from data_pipeline.sample_format import (
    BOS_ID,
    EOS_ID,
    IGNORE_INDEX,
    MAX_SEQUENCE_LENGTH,
    normalize_messages,
    normalize_sample,
    render_messages,
    render_prompt_completion,
    render_training_text,
)
from training.paths import get_path


DEDUP_DIR = os.path.join(get_path("cleaned_data_dir", create=True), "deduplicated")
SHARD_DIR = get_path("shards_dir", create=True)
TOKENIZER_DIR = get_path("tokenizer_dir", create=True)
DEDUP_INCLUDE = [
    name.strip()
    for name in os.environ.get("JARVIS_DEDUP_INCLUDE", "").split(";")
    if name.strip()
]
ALLOWED_FORMATS = {
    item.strip().lower()
    for item in os.environ.get("JARVIS_ALLOWED_FORMATS", "").split(",")
    if item.strip()
}
DATA_PROFILE = os.environ.get("JARVIS_DATA_PROFILE", "auto")

MAX_LENGTH = int(os.environ.get("JARVIS_SEQUENCE_LENGTH", MAX_SEQUENCE_LENGTH))
SHARD_RECORDS = int(os.environ.get("JARVIS_TOKENIZED_SHARD_RECORDS", 5000))
MIN_SEQUENCE_TOKENS = int(os.environ.get("JARVIS_MIN_SEQUENCE_TOKENS", 16))

SOURCE_TOKEN_PROFILES = {
    "piqa_italian": {
        "mode": "sft",
        "max_length": 512,
        "min_tokens": 12,
        "prompt_keep": "head",
        "format": "choice_qa",
    },
    "squad_it": {
        "mode": "sft",
        "max_length": 1536,
        "min_tokens": 24,
        "prompt_keep": "tail",
        "format": "qa",
    },
    "evol_instruct_italian": {
        "mode": "chat",
        "max_length": MAX_LENGTH,
        "min_tokens": 24,
        "assistant_turns": "all",
        "prompt_keep": "tail",
        "format": "chat",
    },
    "wikipedia_it": {
        "mode": "lm",
        "max_length": MAX_LENGTH,
        "min_tokens": 64,
        "stride": MAX_LENGTH,
        "format": "text",
    },
    "wikipedia_it_wikimedia_direct": {
        "mode": "lm",
        "max_length": MAX_LENGTH,
        "min_tokens": 64,
        "stride": MAX_LENGTH,
        "format": "text",
    },
    "fineweb2_it": {
        "mode": "lm",
        "max_length": MAX_LENGTH,
        "min_tokens": 48,
        "stride": MAX_LENGTH,
        "format": "text",
    },
    "stackexchange_auto": {
        "mode": "sft",
        "max_length": 1536,
        "min_tokens": 24,
        "prompt_keep": "tail",
        "format": "qa",
    },
    "local_import": {
        "mode": "auto",
        "max_length": MAX_LENGTH,
        "min_tokens": 16,
        "prompt_keep": "tail",
    },
}

os.makedirs(SHARD_DIR, exist_ok=True)

sp = None


def get_tokenizer():
    global sp
    if sp is None:
        sp = spm.SentencePieceProcessor()
        sp.load(os.path.join(str(TOKENIZER_DIR), "jarvis.model"))
    return sp


def safe_name(path):
    name = os.path.splitext(os.path.basename(path))[0]
    name = name.replace("_dedup", "")
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_") or "dataset"


def encode(text):
    return get_tokenizer().encode(text, out_type=int)


def source_profile(sample, fallback_source=None):
    source = sample.get("source") or fallback_source or "unknown"
    return SOURCE_TOKEN_PROFILES.get(source, {"mode": "auto", "max_length": MAX_LENGTH, "min_tokens": MIN_SEQUENCE_TOKENS})


def clamp_ids(ids, budget, keep="tail"):
    if budget <= 0:
        return []
    if len(ids) <= budget:
        return ids
    if keep == "head":
        return ids[:budget]
    return ids[-budget:]


def build_sft_from_parts(prompt, completion, profile):
    if not prompt or not completion:
        return None

    prompt_ids = encode(prompt)
    completion_ids = encode("\n" + completion)
    max_length = int(profile.get("max_length", MAX_LENGTH))
    min_tokens = int(profile.get("min_tokens", MIN_SEQUENCE_TOKENS))
    prompt_keep = profile.get("prompt_keep", "tail")
    available = max_length - 2

    if available <= 0:
        raise ValueError("JARVIS_SEQUENCE_LENGTH deve essere almeno 3.")

    if len(completion_ids) >= available:
        prompt_ids = []
        completion_ids = completion_ids[:available]
    else:
        prompt_budget = available - len(completion_ids)
        prompt_ids = clamp_ids(prompt_ids, prompt_budget, keep=prompt_keep)

    input_ids = [BOS_ID] + prompt_ids + completion_ids + [EOS_ID]
    labels = [IGNORE_INDEX] * (1 + len(prompt_ids)) + completion_ids + [EOS_ID]

    if len(input_ids) < min_tokens:
        return None

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
        "loss_mode": "assistant_only",
    }


def build_sft_example(sample, profile):
    prompt, completion = render_prompt_completion(sample)
    return build_sft_from_parts(prompt, completion, profile)


def iter_chat_examples(sample, profile):
    messages = normalize_messages(sample.get("messages", []))
    if len(messages) < 2:
        return

    assistant_turns = profile.get("assistant_turns", "last")
    assistant_indexes = [
        index for index, message in enumerate(messages)
        if message.get("role") == "assistant"
    ]
    if assistant_turns != "all" and assistant_indexes:
        assistant_indexes = [assistant_indexes[-1]]

    for index in assistant_indexes:
        prompt_messages = messages[:index]
        completion_message = messages[index]
        if not prompt_messages:
            continue

        prompt = render_messages(prompt_messages, include_assistant_answers=True)
        completion = f"<|assistant|>\n{completion_message['content']}\n<|end|>"
        example = build_sft_from_parts(prompt, completion, profile)
        if example:
            yield example


def build_lm_examples(sample, profile):
    text = render_training_text(sample)
    if not text:
        return

    max_length = int(profile.get("max_length", MAX_LENGTH))
    min_tokens = int(profile.get("min_tokens", MIN_SEQUENCE_TOKENS))
    stride = int(profile.get("stride", max_length))
    stride = max(1, stride)
    ids = [BOS_ID] + encode(text) + [EOS_ID]
    for start in range(0, len(ids), stride):
        block = ids[start:start + max_length]
        if len(block) < min_tokens:
            continue
        yield {
            "input_ids": block,
            "labels": list(block),
            "attention_mask": [1] * len(block),
            "loss_mode": "full_text",
        }


def iter_training_examples(input_path):
    fallback_source = safe_name(input_path)

    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=os.path.basename(input_path)):
            try:
                sample = normalize_sample(json.loads(line))
            except (json.JSONDecodeError, TypeError):
                continue

            sample_format = sample.get("format", "text")
            source = sample.get("source", fallback_source)
            profile = source_profile(sample, fallback_source=source)
            mode = profile.get("mode", "auto")

            if ALLOWED_FORMATS and sample_format not in ALLOWED_FORMATS:
                continue

            if sample_format == "chat" or mode == "chat":
                for example in iter_chat_examples(sample, profile):
                    yield {
                        **example,
                        "source": source,
                        "format": profile.get("format", sample_format),
                        "tokenization": "chat" if mode == "auto" else mode,
                    }
                continue

            if sample_format in {"qa", "choice_qa", "instruction"} or mode == "sft":
                example = build_sft_example(sample, profile)
                if example:
                    yield {
                        **example,
                        "source": source,
                        "format": profile.get("format", sample_format),
                        "tokenization": "sft" if mode == "auto" else mode,
                    }
                continue

            for example in build_lm_examples(sample, profile):
                yield {
                    **example,
                    "source": source,
                    "format": profile.get("format", sample_format),
                    "tokenization": mode if mode != "auto" else "lm",
                }


def write_shard(records, prefix, shard_id):
    output_path = os.path.join(SHARD_DIR, f"tokenized_{prefix}_{shard_id:05d}.parquet")
    table = pa.Table.from_pylist(records)
    pq.write_table(table, output_path)
    return output_path


def main():

    files = [f for f in os.listdir(DEDUP_DIR) if f.endswith(".jsonl")]
    if DEDUP_INCLUDE:
        files = [f for f in files if f in DEDUP_INCLUDE]
    files = [os.path.join(DEDUP_DIR, f) for f in files]

    manifest = {
        "schema_version": 2,
        "mode": "source_profiled_causal_lm",
        "data_profile": DATA_PROFILE,
        "allowed_formats": sorted(ALLOWED_FORMATS),
        "max_length": MAX_LENGTH,
        "columns": ["input_ids", "labels", "attention_mask", "source", "format", "loss_mode", "tokenization"],
        "source_profiles": SOURCE_TOKEN_PROFILES,
        "files": [],
    }

    total_records = 0

    for file in files:
        prefix = safe_name(file)
        records = []
        shard_id = 0
        file_records = 0
        format_counts = {}
        written_files = []

        for example in iter_training_examples(file):
            records.append(example)
            file_records += 1
            total_records += 1
            format_counts[example["format"]] = format_counts.get(example["format"], 0) + 1

            if len(records) >= SHARD_RECORDS:
                written_files.append(write_shard(records, prefix, shard_id))
                records = []
                shard_id += 1

        if records:
            written_files.append(write_shard(records, prefix, shard_id))

        manifest["files"].append(
            {
                "source_file": file,
                "records": file_records,
                "format_counts": format_counts,
                "shards": written_files,
            }
        )

    with open(os.path.join(SHARD_DIR, "tokenized_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if total_records == 0:
        raise RuntimeError("Tokenizzazione completata senza esempi validi.")

    print(f"Tokenizzazione strutturata completata: {total_records} esempi.")


if __name__ == "__main__":
    main()
