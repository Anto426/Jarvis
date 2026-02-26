import os
import json
import hashlib
import multiprocessing as mp
from tqdm import tqdm
import yaml

# =========================
# LOAD PATHS
# =========================

with open("config/paths.yaml", "r") as f:
    paths = yaml.safe_load(f)["paths"]

CLEAN_DIR = paths["cleaned_data_dir"]
DEDUP_DIR = os.path.join(CLEAN_DIR, "deduplicated")
os.makedirs(DEDUP_DIR, exist_ok=True)

NUM_CORES = max(1, mp.cpu_count() - 1)

# =========================
# HASH FUNCTION
# =========================

def hash_text(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

# =========================
# WORKER
# =========================

def process_line(args):
    line, seen = args

    try:
        sample = json.loads(line)
        text = sample["text"]
    except:
        return None

    h = hash_text(text)

    if h in seen:
        return None

    seen[h] = True
    return sample

# =========================
# MAIN
# =========================

def main():

    files = [f for f in os.listdir(CLEAN_DIR) if f.endswith("_clean.jsonl")]

    if not files:
        print("Nessun file clean trovato.")
        return

    manager = mp.Manager()
    seen = manager.dict()

    for file in files:

        input_path = os.path.join(CLEAN_DIR, file)
        output_path = os.path.join(
            DEDUP_DIR,
            file.replace("_clean", "_dedup")
        )

        print(f"Deduplicating {file} con {NUM_CORES} core...")

        with open(input_path, "r", encoding="utf-8") as fin:
            lines = fin.readlines()

        with mp.Pool(NUM_CORES) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(
                        process_line,
                        [(line, seen) for line in lines]
                    ),
                    total=len(lines)
                )
            )

        with open(output_path, "w", encoding="utf-8") as fout:
            for result in results:
                if result:
                    json.dump(result, fout, ensure_ascii=False)
                    fout.write("\n")

    print("Deduplica completata.")
    print(f"Totale unici: {len(seen)}")


if __name__ == "__main__":
    mp.freeze_support()
    main()