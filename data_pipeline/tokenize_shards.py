import os
import json
import yaml
import sentencepiece as spm
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

with open("config/paths.yaml", "r") as f:
    paths = yaml.safe_load(f)["paths"]

DEDUP_DIR = os.path.join(paths["cleaned_data_dir"], "deduplicated")
SHARD_DIR = paths["shards_dir"]
TOKENIZER_DIR = paths["tokenizer_dir"]

os.makedirs(SHARD_DIR, exist_ok=True)

sp = spm.SentencePieceProcessor()
sp.load(os.path.join(TOKENIZER_DIR, "jarvis.model"))

def main():

    shard_id = 0
    records = []

    files = [os.path.join(DEDUP_DIR, f) for f in os.listdir(DEDUP_DIR) if f.endswith(".jsonl")]

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