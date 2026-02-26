import os
import re
import sqlite3
import hashlib
import unicodedata
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
from datasets import load_dataset

# =====================================================
# CONFIG
# =====================================================

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "raw", "db")
DB_PATH = os.path.join(BASE_DIR, "dataset.db")

CHUNK_SIZE = 1200
MIN_LENGTH = 400
COMMIT_INTERVAL = 5000

USE_WIKIPEDIA = True
USE_MC4 = True
MC4_MAX_SAMPLES = 100000

NUM_CORES = max(1, mp.cpu_count() - 1)

os.makedirs(BASE_DIR, exist_ok=True)

# =====================================================
# UTILS
# =====================================================

def hash_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def clean_text(text):
    if not text:
        return ""

    # Fix encoding tipo Ã²
    try:
        text = text.encode("latin1").decode("utf-8")
    except:
        pass

    text = unicodedata.normalize("NFC", text)

    # Rimozione markup
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\[\[.*?\]\]", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

def is_valid_text(text):
    if len(text) < MIN_LENGTH:
        return False

    letters = sum(c.isalpha() for c in text)
    ratio = letters / max(len(text), 1)

    if ratio < 0.7:
        return False

    digits = sum(c.isdigit() for c in text)
    if digits > len(text) * 0.2:
        return False

    return True

def chunk_by_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunk = ""
    for s in sentences:
        if len(chunk) + len(s) < CHUNK_SIZE:
            chunk += " " + s
        else:
            if is_valid_text(chunk):
                yield chunk.strip()
            chunk = s

    if is_valid_text(chunk):
        yield chunk.strip()

def process_text_worker(text, source):

    clean = clean_text(text)

    if not is_valid_text(clean):
        return []

    h = hash_text(clean)

    return [(h, clean, len(clean), source)]

# =====================================================
# DATABASE
# =====================================================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA temp_store=MEMORY;")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS samples (
            hash TEXT PRIMARY KEY,
            text TEXT,
            length INTEGER,
            source TEXT
        )
    """)

    conn.commit()
    return conn, cur

# =====================================================
# WIKIPEDIA
# =====================================================

def process_wikipedia(cur, conn):

    print("Caricamento Wikipedia dataset HF...")

    dataset = load_dataset(
        "wikipedia",
        "20220301.it",
        split="train",
        streaming=True
    )

    total = 0
    batch = []

    worker = partial(process_text_worker, source="wikipedia")

    with mp.Pool(NUM_CORES) as pool:
        for results in tqdm(pool.imap_unordered(worker, (x["text"] for x in dataset), chunksize=100), desc="Wikipedia"):

            if not results:
                continue

            batch.extend(results)

            if len(batch) >= COMMIT_INTERVAL:
                cur.executemany("INSERT OR IGNORE INTO samples VALUES (?, ?, ?, ?)", batch)
                total += cur.rowcount
                conn.commit()
                batch.clear()

    if batch:
        cur.executemany("INSERT OR IGNORE INTO samples VALUES (?, ?, ?, ?)", batch)
        total += cur.rowcount
        conn.commit()

    print(f"Wikipedia completata ({total} record)\n")
    return total

# =====================================================
# MC4 FILTRATO
# =====================================================

def process_mc4(cur, conn):

    print("Caricamento MC4 streaming...")

    dataset = load_dataset(
        "mc4",
        "it",
        split="train",
        streaming=True
    )

    total = 0
    batch = []
    processed = 0

    worker = partial(process_text_worker, source="mc4")

    with mp.Pool(NUM_CORES) as pool:
        for results in tqdm(pool.imap_unordered(worker, (x["text"] for x in dataset), chunksize=500), desc="MC4"):

            if processed >= MC4_MAX_SAMPLES:
                break

            processed += 1

            if not results:
                continue

            batch.extend(results)

            if len(batch) >= COMMIT_INTERVAL:
                cur.executemany("INSERT OR IGNORE INTO samples VALUES (?, ?, ?, ?)", batch)
                total += cur.rowcount
                conn.commit()
                batch.clear()

    if batch:
        cur.executemany("INSERT OR IGNORE INTO samples VALUES (?, ?, ?, ?)", batch)
        total += cur.rowcount
        conn.commit()

    print(f"MC4 completato ({total} record)\n")
    return total

# =====================================================
# MAIN
# =====================================================

def main():

    print("=== JARVIS CLEAN DATA PIPELINE ===")
    print(f"Multiprocessing su {NUM_CORES} core\n")

    conn, cur = init_db()

    total = 0

    if USE_WIKIPEDIA:
        total += process_wikipedia(cur, conn)

    if USE_MC4:
        total += process_mc4(cur, conn)

    conn.close()

    print("Totale record inseriti:", total)
    print("Pipeline completata.")

# =====================================================

if __name__ == "__main__":
    mp.freeze_support()
    main()