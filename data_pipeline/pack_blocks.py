import json
import os

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from training.paths import get_path


SHARD_DIR = get_path("shards_dir", create=True)
PACKED_RECORDS = int(os.environ.get("JARVIS_PACKED_SHARD_RECORDS", 5000))


def packed_name(source_name, shard_id):
    stem = os.path.splitext(os.path.basename(source_name))[0]
    if stem.startswith("tokenized_"):
        stem = stem[len("tokenized_"):]
    return f"packed_{stem}_{shard_id:05d}.parquet"


def write_packed(records, source_name, shard_id):
    output_path = os.path.join(SHARD_DIR, packed_name(source_name, shard_id))
    table = pa.Table.from_pylist(records)
    pq.write_table(table, output_path)
    return output_path


def main():

    files = [
        os.path.join(SHARD_DIR, f)
        for f in os.listdir(SHARD_DIR)
        if f.endswith(".parquet") and f.startswith("tokenized_")
    ]

    if not files:
        raise FileNotFoundError(f"Nessuno shard tokenized trovato in {SHARD_DIR}")

    manifest = {
        "schema_version": 2,
        "mode": "structured_causal_lm",
        "packed_files": [],
        "total_records": 0,
    }

    for file in files:
        table = pq.read_table(file)
        data = table.to_pylist()
        records = []
        shard_id = 0
        file_records = 0

        for row in tqdm(data, desc=os.path.basename(file)):
            records.append(row)
            file_records += 1
            manifest["total_records"] += 1

            if len(records) >= PACKED_RECORDS:
                output_path = write_packed(records, file, shard_id)
                manifest["packed_files"].append(output_path)
                records = []
                shard_id += 1

        if records:
            output_path = write_packed(records, file, shard_id)
            manifest["packed_files"].append(output_path)

        print(f"Shard finale da {os.path.basename(file)}: {file_records} record.")

    with open(os.path.join(SHARD_DIR, "packed_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Packing strutturato completato: {manifest['total_records']} record.")


if __name__ == "__main__":
    main()

