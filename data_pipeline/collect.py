import os
import json
from tqdm import tqdm
from training.paths import configure_cache_env, get_path

# =========================
# LOAD PATHS
# =========================

configure_cache_env()
from datasets import load_dataset

RAW_DIR = get_path("raw_data_dir", create=True)

# =========================
# CONFIG
# =========================

USE_WIKIPEDIA = True
USE_MC4 = True
USE_STACKEXCHANGE = True

MC4_LIMIT = 500_000
STACK_LIMIT = 300_000

MIN_TEXT_LENGTH = 200

# =========================
# HELPERS
# =========================

def save_jsonl(iterator, output_file, text_extractor, limit=None):

    with open(output_file, "w", encoding="utf-8") as f:

        for i, sample in enumerate(tqdm(iterator)):

            if limit and i >= limit:
                break

            text = text_extractor(sample)

            if text and len(text) >= MIN_TEXT_LENGTH:
                json.dump({"text": text}, f, ensure_ascii=False)
                f.write("\n")


# =========================
# WIKIPEDIA
# =========================

def collect_wikipedia():

    print("Scarico Wikipedia IT...")

    dataset = load_dataset(
        "wikipedia",
        "20220301.it",
        split="train",
        streaming=True,
        cache_dir=os.environ.get("HF_DATASETS_CACHE")
    )

    output_file = os.path.join(RAW_DIR, "wikipedia_it.jsonl")

    save_jsonl(
        dataset,
        output_file,
        text_extractor=lambda x: x.get("text", "")
    )

    print("Wikipedia salvata.")


# =========================
# MC4
# =========================

def collect_mc4():

    print("Scarico MC4 IT...")

    dataset = load_dataset(
        "mc4",
        "it",
        split="train",
        streaming=True,
        cache_dir=os.environ.get("HF_DATASETS_CACHE")
    )

    output_file = os.path.join(RAW_DIR, "mc4_it.jsonl")

    save_jsonl(
        dataset,
        output_file,
        text_extractor=lambda x: x.get("text", ""),
        limit=MC4_LIMIT
    )

    print("MC4 salvato.")


# =========================
# STACKEXCHANGE AUTOMOTIVE
# =========================

def collect_stackexchange():

    print("Scarico StackExchange...")

    dataset = load_dataset(
        "HuggingFaceH4/stack-exchange-preferences",
        split="train",
        streaming=True,
        cache_dir=os.environ.get("HF_DATASETS_CACHE")
    )

    output_file = os.path.join(RAW_DIR, "stackexchange_auto.jsonl")

    automotive_keywords = [
        # Inglese tecnico
        "engine", "car", "vehicle", "OBD", "ECU",
        "sensor", "brake", "battery", "transmission",
        "diesel", "petrol", "hybrid", "electric",
        "torque", "throttle", "gearbox", "clutch",
        # Italiano
        "motore", "automobile", "veicolo", "benzina",
        "freni", "batteria", "centralina",
        "cambio", "frizione", "sensore"
    ]

    def extract_text(sample):

        question = sample.get("question", "")
        response = sample.get("response", "")

        text = question + "\n" + response
        text_lower = text.lower()

        if any(k in text_lower for k in automotive_keywords):
            return text

        return ""

    save_jsonl(
        dataset,
        output_file,
        text_extractor=extract_text,
        limit=STACK_LIMIT
    )

    print("StackExchange automotive salvato.")


# =========================
# MAIN
# =========================

def main():

    if USE_WIKIPEDIA:
        collect_wikipedia()

    if USE_MC4:
        collect_mc4()

    if USE_STACKEXCHANGE:
        collect_stackexchange()

    print("Collect completato.")


if __name__ == "__main__":
    main()
