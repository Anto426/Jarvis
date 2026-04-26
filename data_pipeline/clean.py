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
    
    # 3. Sezioni di Servizio: Tronca il testo all'inizio di queste sezioni
    service_keywords = [
        "Note", "Bibliografia", "Voci correlate", "Altri progetti", "Collegamenti esterni"
    ]
    section_pattern = re.compile(
        r'(?:\n|^)(?:==+ *|)(?:' + '|'.join(service_keywords) + r')(?: *==+|\n|:)', 
        flags=re.IGNORECASE
    )
    match = section_pattern.search(text)
    if match:
        text = text[:match.start()]
        
    # 1. Firme e Timestamp
    text = re.sub(r'--[^\n]*?(?:\d{1,2}:\d{2}|\d{1,2}\s+[a-zA-Z]+\s+\d{4})[^\n]*', ' ', text)
    
    # 2. Codice Wiki e Template
    # Template {{...}} (gestisce fino a 2 livelli di annidamento)
    text = re.sub(r'\{\{(?:[^{}]|\{[^{}]*\})*\}\}', ' ', text)
    text = re.sub(r'\{\{.*?\}\}', ' ', text) # Fallback
    
    # Link interni: [[Target|Testo]] -> Testo, [[Testo]] -> Testo
    text = re.sub(r'\[\[(?:[^\]\|\n]*\|)?([^\]\|\n]+)\]\]', r'\1', text)
    text = re.sub(r'\[\[.*?\]\]', ' ', text) # Rimanenti non validi
    
    # Simboli ripetuti
    text = re.sub(r'\^{2,}', ' ', text)
    text = re.sub(r'-{3,}', ' ', text)
    
    # 4. Identificativi di Sistema
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    # Stringhe alfanumeriche isolate (es. hash, ID revisione di almeno 8 caratteri)
    text = re.sub(r'\b(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{8,}\b', ' ', text)
    
    # 5. Regole di Formattazione
    # Rimuovi spazi multipli ma preserva i newline per i paragrafi
    text = re.sub(r'[ \t]+', ' ', text)
    text = '\n'.join(line.strip() for line in text.split('\n'))
    text = re.sub(r'\n{3,}', '\n\n', text)
    
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
