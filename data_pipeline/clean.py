import os
import json
import re
import unicodedata
import multiprocessing as mp
from tqdm import tqdm
from langdetect import detect, LangDetectException
from training.paths import get_path

# =========================
# LOAD PATHS
# =========================

RAW_DIR = get_path("raw_data_dir", create=True)
CLEAN_DIR = get_path("cleaned_data_dir", create=True)

NUM_CORES = max(1, mp.cpu_count() - 1)
RAW_INCLUDE = [
    name.strip()
    for name in os.environ.get("JARVIS_RAW_INCLUDE", "").split(";")
    if name.strip()
]

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
    saved = 0

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        with mp.Pool(NUM_CORES) as pool:
            for result in tqdm(pool.imap_unordered(process_line, fin, chunksize=1000)):
                if not result:
                    continue

                json.dump(result, fout, ensure_ascii=False)
                fout.write("\n")
                saved += 1

    print(f"Salvati {saved} record puliti in {output_path}")

# =========================
# MAIN
# =========================

def main():

    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".jsonl")]
    if RAW_INCLUDE:
        files = [f for f in files if f in RAW_INCLUDE]

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
