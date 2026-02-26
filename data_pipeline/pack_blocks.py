import os
import yaml
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm

with open("config/paths.yaml", "r") as f:
    paths = yaml.safe_load(f)["paths"]

SHARD_DIR = paths["shards_dir"]
BLOCK_SIZE = 2048

def main():

    files = [os.path.join(SHARD_DIR, f) for f in os.listdir(SHARD_DIR) if f.endswith(".parquet")]

    buffer = []
    packed_records = []
    shard_id = 0

    for file in files:
        table = pq.read_table(file)
        data = table.to_pylist()

        for row in tqdm(data):
            buffer.extend(row["input_ids"])

            while len(buffer) >= BLOCK_SIZE:
                block = buffer[:BLOCK_SIZE]
                buffer = buffer[BLOCK_SIZE:]
                packed_records.append({"input_ids": block})

                if len(packed_records) >= 5000:
                    table_out = pa.Table.from_pylist(packed_records)
                    pq.write_table(table_out, os.path.join(SHARD_DIR, f"packed_{shard_id}.parquet"))
                    packed_records = []
                    shard_id += 1

    if packed_records:
        table_out = pa.Table.from_pylist(packed_records)
        pq.write_table(table_out, os.path.join(SHARD_DIR, f"packed_{shard_id}.parquet"))

    print("Packing completato.")

if __name__ == "__main__":
    main()