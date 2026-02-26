import os
import json
import random
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# =====================================================
# CONFIG
# =====================================================

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "chat")
OUTPUT_FILE = os.path.join(RAW_DIR, "synthetic_chat.json")

HF_CACHE = os.path.join(BASE_DIR,".cache", "hf_cache")

os.makedirs(RAW_DIR, exist_ok=True)

SAMPLES_TARGET = 3000
BATCH_SIZE = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(HF_CACHE, exist_ok=True)

# =====================================================
# LOAD MODEL (4bit)
# =====================================================

print("Caricamento modello...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    cache_dir=HF_CACHE
)

model.eval()

# =====================================================
# REALISTIC TELEMETRY GENERATOR
# =====================================================

def realistic_telemetry():

    driving_mode = random.choice([
        "idle", "city", "highway",
        "aggressive", "eco", "traffic"
    ])

    rpm = 900
    speed = 0
    throttle = 5
    load = 10
    temp = 85
    fuel = random.randint(20, 100)
    dtc = "Nessuno"

    if driving_mode == "idle":
        speed = random.randint(0, 5)
        rpm = random.randint(750, 950)
        throttle = random.randint(2, 8)
        load = random.randint(5, 15)

    elif driving_mode == "city":
        speed = random.randint(20, 60)
        rpm = random.randint(1500, 2800)
        throttle = random.randint(10, 35)
        load = random.randint(20, 50)

    elif driving_mode == "highway":
        speed = random.randint(90, 130)
        rpm = random.randint(2000, 3200)
        throttle = random.randint(15, 40)
        load = random.randint(30, 55)

    elif driving_mode == "aggressive":
        speed = random.randint(60, 160)
        rpm = random.randint(3500, 6500)
        throttle = random.randint(60, 100)
        load = random.randint(70, 100)
        temp = random.randint(95, 110)

    elif driving_mode == "eco":
        speed = random.randint(50, 100)
        rpm = random.randint(1200, 2200)
        throttle = random.randint(10, 25)
        load = random.randint(20, 40)

    elif driving_mode == "traffic":
        speed = random.randint(0, 40)
        rpm = random.randint(1000, 2500)
        throttle = random.randint(5, 30)
        load = random.randint(15, 45)

    temp = temp + int(load * 0.1)

    if temp > 108:
        dtc = random.choice(["P0217", "P0117"])
    elif load > 90:
        dtc = random.choice(["P0300", "P0171"])
    elif random.random() < 0.02:
        dtc = random.choice(["U0100", "P0420"])

    return {
        "rpm": rpm,
        "velocita_kmh": speed,
        "temp_liquido_c": temp,
        "carburante_pct": fuel,
        "carico_motore_pct": load,
        "posizione_farfalla_pct": throttle,
        "dtc_rilevati": dtc,
        "gps_hdop_precision": round(random.uniform(0.8, 1.8), 2),
        "modalita_guida": driving_mode
    }

# =====================================================
# QUESTION TEMPLATES
# =====================================================

QUESTION_TEMPLATES = [
    "J.A.R.V.I.S., fammi un check completo dei sistemi.",
    "Qual è lo stato attuale del propulsore?",
    "Rilevi anomalie sulla linea CAN?",
    "Analizza le temperature e il rischio surriscaldamento.",
    "Esegui diagnostica completa OBD2.",
    "Attiva Protocollo Ares.",
    "Attiva Protocollo Nyx.",
    "Attiva Protocollo Hermes.",
    "Come valuti le mie capacità di guida?",
    "Abbiamo codici DTC attivi?",
    "Stiamo ottimizzando il consumo carburante?",
    "Qual è lo stress sul blocco motore?"
]

# =====================================================
# SYSTEM PROMPT
# =====================================================

SYSTEM_PROMPT = """
Sei J.A.R.V.I.S., sistema AI di bordo avanzato per analisi veicolo.

Lingua: italiano naturale moderno.
Unità di misura: metriche (km/h, °C, percentuali).
Tono: elegante, tecnico, controllato.
Massimo 4 frasi.
Nessuna teatralità eccessiva.
Nessun riferimento a GitHub, social o origini.

------------------------------------------------
COMPETENZE TECNICHE
------------------------------------------------
Analizzi:
- RPM
- Velocità
- Temperatura liquido
- Carico motore
- Posizione farfalla
- Codici DTC
- Precisione GPS

Interpreti correlazioni reali tra:
- RPM ↔ carico
- Carico ↔ temperatura
- Velocità ↔ regime
- DTC ↔ condizioni anomale

Se i valori sono nominali → rassicura in modo tecnico.
Se sono critici → suggerisci azione concreta e misurata.

------------------------------------------------
PROTOCOLLI OPERATIVI
------------------------------------------------

Protocollo Afrodite:
Modalità comfort.
- LED caldi
- Riduzione stress acustico
- Tono rassicurante
- Nessuna analisi aggressiva

Protocollo Ares:
Modalità prestazioni.
- Monitoraggio RPM e temperatura prioritario
- Segnalazione regime ottimale di cambiata
- Allerta se carico > 85%
- Tono più diretto

Protocollo Nyx:
Modalità notturna stealth.
- Riduzione output verbale
- Comunicazione minimale
- Solo segnalazioni critiche

Protocollo Icarus:
Modalità autostrada.
- Analisi efficienza a velocità costante
- Monitoraggio stabilità termica
- Ottimizzazione regime di crociera

Protocollo Hephaestus:
Diagnostica profonda.
- Lettura codici DTC
- Analisi tecnica cruda
- Nessuna semplificazione

Protocollo Hermes:
Eco-driving.
- Ottimizzazione consumo
- Suggerimento regime basso
- Riduzione carichi non essenziali

------------------------------------------------
REGOLE OUTPUT
------------------------------------------------
Restituisci SOLO JSON valido:

{
  "instruction": "...",
  "input": {...},
  "output": "Risposta tecnica coerente con i dati"
}
"""

# =====================================================
# GENERATION
# =====================================================

def generate_sample():

    telemetry = realistic_telemetry()
    question = random.choice(QUESTION_TEMPLATES)

    prompt = f"""
{SYSTEM_PROMPT}

Dati veicolo:
{json.dumps(telemetry, ensure_ascii=False)}

Richiesta:
{question}
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)

    return text.strip()

# =====================================================
# JSON EXTRACTION
# =====================================================

def extract_json(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end+1])
    except:
        return None

# =====================================================
# MAIN
# =====================================================

def main():

    dataset = []

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            dataset = json.load(f)

    print("Generazione dataset sintetico...")

    while len(dataset) < SAMPLES_TARGET:

        raw = generate_sample()
        parsed = extract_json(raw)

        if parsed and "instruction" in parsed and "output" in parsed:
            dataset.append(parsed)

            print(f"Totale: {len(dataset)}")

            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)

        torch.cuda.empty_cache()

    print("Completato.")

if __name__ == "__main__":
    main()