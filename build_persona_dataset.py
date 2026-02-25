import os
import re
import json
import hashlib
import sqlite3
import random
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# =====================================================
# CONFIG
# =====================================================

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(BASE_DIR, "dataset.db")

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
SYNTHETIC_SAMPLES = 2000
COMMIT_INTERVAL = 200

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(BASE_DIR, exist_ok=True)

# =====================================================
# UTIL
# =====================================================

def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()

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
# SYSTEM PROMPT
# =====================================================

SYSTEM_PROMPT = """
Sei J.A.R.V.I.S. (Just A Rather Very Intelligent System).

Rispondi ESCLUSIVAMENTE con la tua risposta, che deve essere una frase narrativa in stile cinematografico.
Non usare markdown, non usare tag, non giustificare il tuo ragionamento, non usare JSON. Genera SOLAMENTE il testo della risposta.
"""

# =====================================================
# TELEMETRY
# =====================================================

def random_telemetry():
    return {
        "rpm": random.randint(800, 6500),
        "velocita_kmh": random.randint(0, 160),
        "temp_liquido_c": random.randint(70, 115),
        "carburante_pct": random.randint(5, 100),
        "carico_motore_pct": random.randint(10, 100),
        "dtc_rilevati": random.choice(["Nessuno", "P0171", "P0300", "P0420"])
    }

question_templates = [
    "Esegui un check completo dei sistemi.",
    "Analizza lo stato del propulsore.",
    "Ci sono DTC attivi?",
    "Attiva Protocollo Ares.",
    "Avvia Protocollo Hermes."
]

# =====================================================
# GENERAZIONE SINTETICA
# =====================================================

def process_synthetic(cur, conn):

    print(f"Caricamento modello {MODEL_NAME} su {DEVICE}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model.eval()

    total = 0
    batch = []

    for i in tqdm(range(SYNTHETIC_SAMPLES), desc="Generazione DeepSeek"):

        telemetry = random_telemetry()
        question = random.choice(question_templates)

        prompt = f"""
Sei J.A.R.V.I.S. (Just A Rather Very Intelligent System).
Rispondi sempre analizzando il veicolo ma sii tagliente e sintetico.

<INPUT>
Dati veicolo: {json.dumps(telemetry, ensure_ascii=False)}
Richiesta: {question}
</INPUT>
"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        print("\n================ RAW MODEL OUTPUT ================")
        print(raw_text)
        print("==================================================\n")

        if "</think>" in raw_text:
            output_text = raw_text.split("</think>")[-1].strip()
        else:
            output_text = raw_text.strip()

        if not output_text:
            print("⚠ Nessun testo generato.\n")
            continue

        try:
            # Formattiamo come stringa Prompt/Risposta
            formatted = f"[INST] {prompt.strip()} [/INST]\n{output_text}\n"

            print("\n" + "="*80)
            print(f"ESEMPIO {i+1}")
            print("="*80)
            print(formatted)
            print("="*80 + "\n")

        except Exception as e:
            print("⚠ Errore formattazione testo:", e)
            continue

        h = hash_text(formatted)
        batch.append((h, formatted, len(formatted), "synthetic_deepseek"))

        if len(batch) >= COMMIT_INTERVAL:
            cur.executemany(
                "INSERT OR IGNORE INTO samples VALUES (?, ?, ?, ?)",
                batch
            )
            conn.commit()
            total += cur.rowcount
            batch.clear()

        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if batch:
        cur.executemany(
            "INSERT OR IGNORE INTO samples VALUES (?, ?, ?, ?)",
            batch
        )
        conn.commit()
        total += cur.rowcount

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Inseriti {total} campioni sintetici.\n")
    return total

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    print("=== JARVIS DATASET PIPELINE – DEEPSEEK DEBUG MODE ===\n")

    conn, cur = init_db()

    total = process_synthetic(cur, conn)

    conn.close()

    print("Totale inseriti:", total)
    print("Pipeline completata con successo.")