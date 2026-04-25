import os
import json
import sentencepiece as spm
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from training.paths import get_path

DEDUP_DIR = os.path.join(get_path("cleaned_data_dir", create=True), "deduplicated")
SHARD_DIR = get_path("shards_dir", create=True)
TOKENIZER_DIR = get_path("tokenizer_dir", create=True)
DEDUP_INCLUDE = [
    name.strip()
    for name in os.environ.get("JARVIS_DEDUP_INCLUDE", "").split(";")
    if name.strip()
]

os.makedirs(SHARD_DIR, exist_ok=True)

sp = spm.SentencePieceProcessor()
sp.load(os.path.join(str(TOKENIZER_DIR), "jarvis.model"))

def main():

    shard_id = 0
    records = []

    files = [f for f in os.listdir(DEDUP_DIR) if f.endswith(".jsonl")]
    if DEDUP_INCLUDE:
        files = [f for f in files if f in DEDUP_INCLUDE]
    files = [os.path.join(DEDUP_DIR, f) for f in files]

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                sample = json.loads(line)
                ids = sp.encode(sample["text"], out_type=int)
                records.append({"input_ids": ids})

                if len(records) >= 10000:
                    table = pa.Table.from_pylist(records)
                    pq.write_table(table, os.path.join(SHARD_DIR, f"shard_{shard_id}.parquet"))
                    records = []
                    shard_id += 1

    if records:
        table = pa.Table.from_pylist(records)
        pq.write_table(table, os.path.join(SHARD_DIR, f"shard_{shard_id}.parquet"))

    print("Tokenizzazione completata.")

if __name__ == "__main__":
    main()
