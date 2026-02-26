import os
import sys
import sqlite3
import numpy as np
from transformers import AutoTokenizer

# =====================================================
# CONFIG
# =====================================================

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TOKENS_DIR = os.path.join(BASE_DIR, "data", "tokens")
TOKENIZER_PATH = os.path.join(BASE_DIR, "data", "tokenizer")

BLOCK_SIZE = 512

os.makedirs(TOKENS_DIR, exist_ok=True)

# =====================================================
# ARGUMENT
# =====================================================

if len(sys.argv) != 2:
    print("Uso: python tokenize_dataset.py [base|chat|personal]")
    sys.exit(1)

dataset_type = sys.argv[1]

DB_FILE = os.path.join(PROCESSED_DIR, f"{dataset_type}.db")
TOKENS_DB = os.path.join(TOKENS_DIR, f"{dataset_type}_tokens.db")

# =====================================================
# MAIN
# =====================================================

def main():

    if not os.path.exists(DB_FILE):
        print(f"{DB_FILE} non trovato.")
        sys.exit(1)

    if os.path.exists(TOKENS_DB):
        os.remove(TOKENS_DB)

    print(f"Tokenizzazione dataset: {dataset_type}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    conn_out = sqlite3.connect(TOKENS_DB)
    cur_out = conn_out.cursor()

    cur_out.execute("""
        CREATE TABLE token_blocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_ids BLOB
        )
    """)

    conn_in = sqlite3.connect(DB_FILE)
    cur_in = conn_in.cursor()
    cur_in.execute("SELECT text FROM samples")

    buffer = []
    total_blocks = 0

    for row in cur_in:
        text = row[0]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        buffer.extend(tokens)

        while len(buffer) >= BLOCK_SIZE:
            block = buffer[:BLOCK_SIZE]
            buffer = buffer[BLOCK_SIZE:]
            arr = np.array(block, dtype=np.uint16)
            cur_out.execute(
                "INSERT INTO token_blocks (input_ids) VALUES (?)",
                (arr.tobytes(),)
            )
            total_blocks += 1

    conn_out.commit()
    conn_out.close()
    conn_in.close()

    print(f"Completato. Blocchi creati: {total_blocks}")

if __name__ == "__main__":
    main()