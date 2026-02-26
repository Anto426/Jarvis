import os
import sys
import sqlite3
import numpy as np
from tqdm import tqdm
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

# =====================================================
# CONFIG
# =====================================================

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TOKENS_DIR = os.path.join(BASE_DIR, "data", "tokens")
TOKENIZER_DIR = os.path.join(BASE_DIR, "data", "tokenizer")

VOCAB_SIZE = 32000
BLOCK_SIZE = 512
TRAIN_LIMIT = None

os.makedirs(TOKENS_DIR, exist_ok=True)
os.makedirs(TOKENIZER_DIR, exist_ok=True)

# =====================================================
# TOKENIZER CREATION (AUTO)
# =====================================================

def train_tokenizer():

    vocab_file = os.path.join(TOKENIZER_DIR, "vocab.json")

    if os.path.exists(vocab_file):
        print("Tokenizer già presente.")
        return

    base_db = os.path.join(PROCESSED_DIR, "base.db")

    if not os.path.exists(base_db):
        raise FileNotFoundError("base.db necessario per creare il tokenizer.")

    print("Training tokenizer base...")

    def text_iterator(batch_size=10000):
        conn = sqlite3.connect(base_db)
        cur = conn.cursor()
        cur.execute("SELECT text FROM samples")

        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            yield [r[0] for r in rows]

        conn.close()

    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train_from_iterator(
        text_iterator(),
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"]
    )

    tokenizer.save_model(TOKENIZER_DIR)

    print("Tokenizer creato.")

# =====================================================
# LOAD TOKENIZER
# =====================================================

def load_tokenizer():

    base = ByteLevelBPETokenizer(
        vocab=os.path.join(TOKENIZER_DIR, "vocab.json"),
        merges=os.path.join(TOKENIZER_DIR, "merges.txt")
    )

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=base._tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>"
    )

    return tokenizer

# =====================================================
# TOKENIZE DB
# =====================================================

def tokenize_db(dataset_name):

    input_db = os.path.join(PROCESSED_DIR, f"{dataset_name}.db")
    output_db = os.path.join(TOKENS_DIR, f"{dataset_name}_tokens.db")

    if not os.path.exists(input_db):
        raise FileNotFoundError(f"{input_db} non trovato.")

    if os.path.exists(output_db):
        os.remove(output_db)

    tokenizer = load_tokenizer()

    with sqlite3.connect(output_db) as conn_out:

        cur_out = conn_out.cursor()
        cur_out.execute("""
            CREATE TABLE token_blocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_ids BLOB
            )
        """)
        conn_out.commit()

        with sqlite3.connect(input_db) as conn_in:

            cur_in = conn_in.cursor()
            cur_in.execute("SELECT COUNT(*) FROM samples")
            total = cur_in.fetchone()[0]

            cur_in.execute("SELECT text FROM samples")

            print(f"Tokenizzazione {dataset_name} ({total} campioni)...")

            buffer = []
            batch_insert = []
            batch_size = 5000

            for row in tqdm(cur_in, total=total):

                text = row[0]
                tokens = tokenizer.encode(text, add_special_tokens=False)

                buffer.extend(tokens)

                while len(buffer) >= BLOCK_SIZE:
                    block = buffer[:BLOCK_SIZE]
                    buffer = buffer[BLOCK_SIZE:]

                    arr = np.array(block, dtype=np.uint16)
                    batch_insert.append((arr.tobytes(),))

                if len(batch_insert) >= batch_size:
                    cur_out.executemany(
                        "INSERT INTO token_blocks (input_ids) VALUES (?)",
                        batch_insert
                    )
                    conn_out.commit()
                    batch_insert.clear()

            if batch_insert:
                cur_out.executemany(
                    "INSERT INTO token_blocks (input_ids) VALUES (?)",
                    batch_insert
                )
                conn_out.commit()

    print(f"{dataset_name} tokenizzato.")

# =====================================================
# MAIN
# =====================================================

def main():

    if len(sys.argv) != 2:
        print("Uso: python tokenize_dataset.py [base|chat|personal]")
        return

    dataset_name = sys.argv[1]

    # 1️⃣ Assicura tokenizer
    train_tokenizer()

    # 2️⃣ Tokenizza
    tokenize_db(dataset_name)

    print("Completato.")

if __name__ == "__main__":
    main()