import os
import json
import hashlib
import multiprocessing as mp
from tqdm import tqdm
from training.paths import get_path

# =========================
# LOAD PATHS
# =========================

CLEAN_DIR = get_path("cleaned_data_dir", create=True)
DEDUP_DIR = os.path.join(CLEAN_DIR, "deduplicated")
os.makedirs(DEDUP_DIR, exist_ok=True)

NUM_CORES = max(1, mp.cpu_count() - 1)
CLEAN_INCLUDE = [
    name.strip()
    for name in os.environ.get("JARVIS_CLEAN_INCLUDE", "").split(";")
    if name.strip()
]

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
    if CLEAN_INCLUDE:
        files = [f for f in files if f in CLEAN_INCLUDE]

    if not files:
        print("Nessun file clean trovato.")
        return

    seen = set()

    for file in files:

        input_path = os.path.join(CLEAN_DIR, file)
        output_path = os.path.join(
            DEDUP_DIR,
            file.replace("_clean", "_dedup")
        )

        print(f"Deduplicating {file} con {NUM_CORES} core...")

        saved = 0
        with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
            for line in tqdm(fin):
                try:
                    sample = json.loads(line)
                    text = sample["text"]
                except Exception:
                    continue

                h = hash_text(text)
                if h in seen:
                    continue

                seen.add(h)
                json.dump(sample, fout, ensure_ascii=False)
                fout.write("\n")
                saved += 1

        print(f"Salvati {saved} record deduplicati in {output_path}")

    print("Deduplica completata.")
    print(f"Totale unici: {len(seen)}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
