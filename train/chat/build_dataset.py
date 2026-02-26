import os
import json
import sqlite3

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

RAW_FILE = os.path.join(BASE_DIR, "data", "raw", "chat", "synthetic_chat.json")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
DB_FILE = os.path.join(PROCESSED_DIR, "chat.db")

os.makedirs(PROCESSED_DIR, exist_ok=True)

def format_sample(entry):
    return f"""### Instruction:
{entry["instruction"]}

### Input:
{json.dumps(entry["input"], ensure_ascii=False)}

### Response:
{entry["output"]}"""

def main():

    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)

    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT
        )
    """)

    with open(RAW_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        formatted = format_sample(entry)
        cur.execute("INSERT INTO samples (text) VALUES (?)", (formatted,))

    conn.commit()
    conn.close()

    print("Chat DB creato in processed/")

if __name__ == "__main__":
    main()