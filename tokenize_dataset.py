import os
import sqlite3
import numpy as np
from tqdm import tqdm
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

# ==========================
# CONFIG
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DB = os.path.join(BASE_DIR, "data", "dataset.db")
PERSONA_DB = os.path.join(BASE_DIR, "data", "persona.db")

TOKENS_DB = os.path.join(BASE_DIR, "data", "tokens.db")
PERSONA_TOKENS_DB = os.path.join(BASE_DIR, "data", "persona_tokens.db")

TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")

VOCAB_SIZE = 32000
BLOCK_SIZE = 512
TRAIN_LIMIT = 250000

# ==========================
# STREAM ITERATOR
# ==========================

def text_iterator(db_path, table="samples", limit=None, batch_size=10000):

    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.cursor()

    query = f"SELECT text FROM {table}"
    if limit:
        query += f" LIMIT {limit}"

    cur.execute(query)

    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break
        yield [r[0] for r in rows]

    conn.close()

# ==========================
# TRAIN TOKENIZER
# ==========================

def train_tokenizer():

    if os.path.exists(os.path.join(TOKENIZER_DIR, "vocab.json")):
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    print("Training tokenizer (streaming)...")

    trainer = ByteLevelBPETokenizer()

    trainer.train_from_iterator(
        text_iterator(DATASET_DB, limit=TRAIN_LIMIT),
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"]
    )

    trainer.save_model(TOKENIZER_DIR)

# ==========================
# LOAD TOKENIZER
# ==========================

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

    if tokenizer.vocab_size >= 65536:
        raise ValueError("Vocab troppo grande per uint16.")

    return tokenizer

# ==========================
# TOKENIZE DATABASE
# ==========================

def tokenize_db(input_db, output_db, is_persona=False):

    if not os.path.exists(input_db):
        print(f"{input_db} non trovato, salto.")
        return

    if os.path.exists(output_db):
        os.remove(output_db)

    tokenizer = load_tokenizer()

    with sqlite3.connect(output_db) as conn_out:

        cur_out = conn_out.cursor()

        cur_out.execute("PRAGMA journal_mode=WAL;")
        cur_out.execute("PRAGMA synchronous=NORMAL;")
        cur_out.execute("PRAGMA temp_store=MEMORY;")

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

            print(f"Tokenizzazione {input_db} ({total} campioni)...")

            buffer = []
            insert_batch = []
            batch_size = 5000

            for row in tqdm(cur_in, total=total):

                text = row[0]

                tokens = tokenizer.encode(text, add_special_tokens=False)

                if is_persona:
                    # Per persona NON facciamo packing continuo
                    tokens = tokens[:BLOCK_SIZE]
                    tokens += [tokenizer.pad_token_id] * (BLOCK_SIZE - len(tokens))
                    arr = np.array(tokens, dtype=np.uint16)
                    insert_batch.append((arr.tobytes(),))
                else:
                    buffer.extend(tokens)

                    while len(buffer) >= BLOCK_SIZE:
                        block = buffer[:BLOCK_SIZE]
                        buffer = buffer[BLOCK_SIZE:]
                        arr = np.array(block, dtype=np.uint16)
                        insert_batch.append((arr.tobytes(),))

                if len(insert_batch) >= batch_size:
                    cur_out.executemany(
                        "INSERT INTO token_blocks (input_ids) VALUES (?)",
                        insert_batch
                    )
                    conn_out.commit()
                    insert_batch.clear()

            if insert_batch:
                cur_out.executemany(
                    "INSERT INTO token_blocks (input_ids) VALUES (?)",
                    insert_batch
                )
                conn_out.commit()

    print(f"Salvato in {output_db}")

# ==========================
# MAIN
# ==========================

def main():

    if not os.path.exists(DATASET_DB):
        raise FileNotFoundError("dataset.db non trovato.")

    train_tokenizer()

    print("\n=== BASE TOKENIZATION ===")
    tokenize_db(DATASET_DB, TOKENS_DB, is_persona=False)

    print("\n=== PERSONA TOKENIZATION ===")
    tokenize_db(PERSONA_DB, PERSONA_TOKENS_DB, is_persona=True)

    print("\nTokenizzazione completata.")

if __name__ == "__main__":
    main()