import os
import json
from datasets import load_dataset
from tqdm import tqdm
import yaml

# =========================
# LOAD PATHS
# =========================

with open("config/paths.yaml", "r") as f:
    paths = yaml.safe_load(f)["paths"]

RAW_DIR = paths["raw_data_dir"]
os.makedirs(RAW_DIR, exist_ok=True)

# =========================
# CONFIG
# =========================

USE_WIKIPEDIA = True
USE_MC4 = True
MC4_LIMIT = 500_000

# =========================
# HELPERS
# =========================

def save_jsonl(dataset_iterator, output_file, text_field="text", limit=None):

    with open(output_file, "w", encoding="utf-8") as f:
        for i, sample in enumerate(tqdm(dataset_iterator)):
            if limit and i >= limit:
                break

            text = sample.get(text_field, "")
            if text and len(text) > 200:
                json.dump({"text": text}, f, ensure_ascii=False)
                f.write("\n")

# =========================
# MAIN
# =========================

def collect_wikipedia():

    print("Scarico Wikipedia IT...")

    dataset = load_dataset(
        "wikipedia",
        "20220301.it",
        split="train",
        streaming=True
    )

    output_file = os.path.join(RAW_DIR, "wikipedia_it.jsonl")

    save_jsonl(dataset, output_file)

    print("Wikipedia salvata.")


def collect_mc4():

    print("Scarico MC4 IT...")

    dataset = load_dataset(
        "mc4",
        "it",
        split="train",
        streaming=True
    )

    output_file = os.path.join(RAW_DIR, "mc4_it.jsonl")

    save_jsonl(dataset, output_file, limit=MC4_LIMIT)

    print("MC4 salvato.")


def main():

    if USE_WIKIPEDIA:
        collect_wikipedia()

    if USE_MC4:
        collect_mc4()

    print("Collect completato.")


if __name__ == "__main__":
    main()