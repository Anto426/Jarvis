from datasets import load_from_disk
import sqlite3
import os
from transformers import GPT2TokenizerFast
from tokenizers import ByteLevelBPETokenizer

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dataset.db")
TOKENIZER_DIR = "tokenizer"

def check():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*), sum(length(text)) FROM samples")
    row = cur.fetchone()
    print("DB Stats:", row)
    
    tokenizer = GPT2TokenizerFast(vocab_file=os.path.join(TOKENIZER_DIR, "vocab.json"), merges_file=os.path.join(TOKENIZER_DIR, "merges.txt"))
    print("Tokenizer vocab size:", tokenizer.vocab_size)
    
    cur.execute("SELECT text FROM samples LIMIT 2")
    texts = [r[0] for r in cur.fetchall()]
    tokens = tokenizer(texts)["input_ids"]
    print("Tokenized sample lengths:", [len(t) for t in tokens])

if __name__ == "__main__":
    check()
