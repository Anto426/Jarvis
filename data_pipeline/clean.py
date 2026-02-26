import os
import json
import re
import unicodedata
import yaml
import multiprocessing as mp
from tqdm import tqdm
from langdetect import detect, LangDetectException

# =========================
# LOAD PATHS
# =========================

with open("config/paths.yaml", "r") as f:
    paths = yaml.safe_load(f)["paths"]

RAW_DIR = paths["raw_data_dir"]
CLEAN_DIR = paths["cleaned_data_dir"]

os.makedirs(CLEAN_DIR, exist_ok=True)

NUM_CORES = max(1, mp.cpu_count() - 1)

# =========================
# QUALITY THRESHOLDS
# =========================

MIN_LENGTH = 400
MIN_ALPHA_RATIO = 0.6
MAX_DIGIT_RATIO = 0.2

# =========================
# CLEAN FUNCTIONS
# =========================

def basic_clean(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def is_valid_text(text):
    if len(text) < MIN_LENGTH:
        return False

    alpha = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)

    if alpha / len(text) < MIN_ALPHA_RATIO:
        return False

    if digits / len(text) > MAX_DIGIT_RATIO:
        return False

    return True

def is_italian(text):
    try:
        return detect(text) == "it"
    except LangDetectException:
        return False

def process_line(line):
    try:
        sample = json.loads(line)
        text = sample["text"]
    except:
        return None

    text = basic_clean(text)

    if not is_valid_text(text):
        return None

    if not is_italian(text):
        return None

    return {"text": text}

# =========================
# CLEAN FILE
# =========================

def clean_file(input_path, output_path):

    with open(input_path, "r", encoding="utf-8") as fin:
        lines = fin.readlines()

    with mp.Pool(NUM_CORES) as pool:
        results = list(tqdm(pool.imap_unordered(process_line, lines), total=len(lines)))

    with open(output_path, "w", encoding="utf-8") as fout:
        for result in results:
            if result:
                json.dump(result, fout, ensure_ascii=False)
                fout.write("\n")

# =========================
# MAIN
# =========================

def main():

    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".jsonl")]

    if not files:
        print("Nessun file raw trovato.")
        return

    for file in files:
        input_path = os.path.join(RAW_DIR, file)
        output_path = os.path.join(CLEAN_DIR, file.replace(".jsonl", "_clean.jsonl"))

        print(f"Pulizia {file} con {NUM_CORES} core...")
        clean_file(input_path, output_path)

    print("Clean completato.")

if __name__ == "__main__":
    mp.freeze_support()
    main()