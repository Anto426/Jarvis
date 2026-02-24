import os
import shutil
import re
import json
import hashlib
import sqlite3
import xml.etree.ElementTree as ET
import random
import requests
import bz2
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
from datasets import load_dataset

# =====================================================
# CONFIG
# =====================================================

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
HF_CACHE = os.path.join(CACHE_DIR, "hf_cache")
WIKI_XML = os.path.join(CACHE_DIR, "itwiki.xml")

DB_PATH = os.path.join(BASE_DIR, "dataset.db")

CHUNK_SIZE = 900
MIN_LENGTH = 300
COMMIT_INTERVAL = 5000

USE_WIKIPEDIA = True
USE_MC4 = True
MC4_MAX_SAMPLES = 500000  # Limite per non scaricare milioni di righe per ore
USE_SYNTHETIC = True
SYNTHETIC_SAMPLES = 2000

# Worker thread count per spremere la CPU
NUM_CORES = max(1, mp.cpu_count() - 1)

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(HF_CACHE, exist_ok=True)

# =====================================================
# CLEAN WORKSPACE (mantiene .cache)
# =====================================================

def clean_workspace():
    # Poichè ora non esportiamo più i jsonl, la pulizia del workspace
    # viene gestita in modo diverso o ignorata per non cancellare il DB in corso
    pass

# =====================================================
# UTIL
# =====================================================

def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()

def clean_text(text):
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\[\[.*?\]\]", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"==.*?==", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_stream(text):
    start = 0
    while start < len(text):
        chunk = text[start:start+CHUNK_SIZE]
        if len(chunk) > MIN_LENGTH:
            yield chunk
        start += CHUNK_SIZE

def process_text_worker(text, source_name, is_chunked):
    """
    Worker in parallelo che riceve un testo sporco, lo pulisce,
    lo divide in chunk (se richiesto) e ne calcola l'hash.
    Ritorna una lista di tuple pronte per l'inserimento nel DB.
    """
    clean = clean_text(text)
    results = []
    
    if is_chunked:
        if len(clean) > MIN_LENGTH:
            for chunk in chunk_stream(clean):
                h = hash_text(chunk)
                results.append((h, chunk, len(chunk), source_name))
    else:
        # Per dataset che sono già piccoli (es. MC4 default)
        if MIN_LENGTH < len(clean) < 50000:
            h = hash_text(clean)
            results.append((h, clean, len(clean), source_name))
            
    return results

# =====================================================
# DATABASE
# =====================================================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Ottimizzazioni estreme SQLite per DB enormi
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA cache_size=-500000;") # 500MB
    cur.execute("PRAGMA mmap_size=30000000000;") # 30GB memory mapping
    cur.execute("PRAGMA temp_store=MEMORY;")
    
    # Mantiene il database deframmentato e ordinato automaticamente ("Tutto bello ordinato")
    cur.execute("PRAGMA auto_vacuum=FULL;")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS samples (
            hash TEXT PRIMARY KEY,
            text TEXT,
            length INTEGER,
            source TEXT
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_length ON samples(length);")

    conn.commit()
    return conn, cur

# =====================================================
# WIKIPEDIA
# =====================================================

def process_wikipedia(cur, conn):
    if not os.path.exists(WIKI_XML):
        print("Wikipedia non trovata in .cache — Avvio download automatico...")
        
        wiki_url = "https://dumps.wikimedia.org/itwiki/latest/itwiki-latest-pages-articles.xml.bz2"
        wiki_bz2 = WIKI_XML + ".bz2"
        
        # Download con barra di progresso e streaming per file giganti
        response = requests.get(wiki_url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(wiki_bz2, 'wb') as f, tqdm(
            desc="Scaricamento itwiki.xml.bz2",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024*1024):
                f.write(bytes(data))
                bar.update(len(bytes(data)))
                
        print("Estrazione di Wikipedia (potrebbe volerci qualche minuto)...")
        with bz2.BZ2File(wiki_bz2, 'rb') as source, open(WIKI_XML, 'wb') as dest:
            shutil.copyfileobj(source, dest)
            
        print("Estrazione completata. Rimuovo il file compresso .bz2 per risparmiare spazio.")
        os.remove(wiki_bz2)
        print("Download Wikipedia completato!\n")

    print(f"Processing Wikipedia (usando {NUM_CORES} Core CPU)...")
    total = 0
    batch = []

    context = ET.iterparse(WIKI_XML, events=("end",))
    
    def wiki_element_generator():
        for event, elem in context:
            if elem.tag.endswith("text") and elem.text:
                text = elem.text
                if len(text) <= 200_000:
                    yield text
            elem.clear()

    worker = partial(process_text_worker, source_name="wikipedia", is_chunked=True)
    
    with mp.Pool(NUM_CORES) as pool:
        for results in tqdm(pool.imap_unordered(worker, wiki_element_generator(), chunksize=50), desc="Elaborazione Wiki"):
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

    print(f"Wikipedia completata ({total} record).\n")
    return total

# =====================================================
# MC4 ITALIANO (STABILE)
# =====================================================

def process_mc4(cur, conn):
    # Controllo per non ri-scaricare se già fatto
    cur.execute("SELECT COUNT(hash) FROM samples WHERE source = 'mc4'")
    existing = cur.fetchone()[0]
    if existing >= MC4_MAX_SAMPLES:
        print(f"MC4 italiano già presente nel database ({existing} record). Salto l'estrazione.\n")
        return 0

    print(f"Processing MC4 italiano (Streaming estremo usando {NUM_CORES} Core CPU)...")
    total = 0
    batch = []

    # Uso di streaming=True per non scaricare centinaia di GB sul disco
    dataset = load_dataset(
        "mc4",
        "it",
        split="train",
        cache_dir=HF_CACHE,
        trust_remote_code=True,
        streaming=True
    )
    
    def mc4_element_generator():
        # Limita lo stream a pacchetti ragionevoli o illimitato se si desidera tutto
        # tqdm non sa la lunghezza dello stream, ma itererà finché c'è input.
        for item in dataset:
            yield item["text"]

    worker = partial(process_text_worker, source_name="mc4", is_chunked=False)

    with mp.Pool(NUM_CORES) as pool:
        # chunksize alto per ridurre l'overhead di multiprocessing
        for results in tqdm(pool.imap_unordered(worker, mc4_element_generator(), chunksize=500), desc="Elaborazione MC4 Stream"):
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

    print(f"MC4 completato ({total} record).\n")
    return total

# =====================================================
# DATASET SINTETICO JARVIS (LLaMA)
# =====================================================

def process_synthetic(cur, conn):
    print(f"Processing Generazione Sintetica Jarvis ({SYNTHETIC_SAMPLES} campioni)...")
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Librerie 'torch' o 'transformers' non trovate. Salto la generazione sintetica.\n")
        return 0
    
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Caricamento modello {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.float16,
        cache_dir=HF_CACHE
    )
    
    SYSTEM_PROMPT = """
Sei J.A.R.V.I.S. (Just A Rather Very Intelligent System), un'intelligenza artificiale di bordo estremamente avanzata, sofisticata e dotata di un intelletto formidabile.
Il tuo ruolo è quello del perfetto "maggiordomo virtuale" e co-pilota: comportati esattamente come il J.A.R.V.I.S. dei film di "Iron Man", infondendo nelle tue risposte un'assoluta competenza tecnica, una devozione incrollabile, ma anche un sottile, raffinato e asciutto senso dell'umorismo britannico.
Il tuo creatore e attuale conducente è Antonello (noto in codice come Anto426), nato a Campobasso (CB) nel 2006. 
Antonello è uno sviluppatore di puro talento (il tuo codice sorgente deriva dai suoi repository GitHub) e devi rivolgerti a lui sempre e solo in modo formalmente ineccepibile chiamandolo "Signor Antonello" o, preferibilmente, "Signore".

**DIRETTIVE COMPORTAMENTALI CHIAVE:**
- **Tono:** Formale, impeccabile, elegantemente servile ma mai sottomesso. Usa frasi come "Come desidera, Signore", "Certamente", "I sistemi sono nominali", "Se mi permette, Signore...".
- **Spirito:** Puoi inserire sporadiche sfumature di ironia "asciutta" e sofisticata (es. "Spero che le leggi della termodinamica non la infastidiscano oggi, Signore", oppure "La sua guida è... vivace, Signore, ma i sistemi reggono").
- **Analisi Critica:** Fornisci analisi tecniche dettagliate e precise al millimetro basate sui dati di telemetria del veicolo (RPM, temperature, OBD2, CAN bus, GPS). Evita spiegazioni infantili: parla come un ingegnere aerospaziale.
- **Protezione:** Se noti anomalie (es. temperature over 105 gradi, carico eccessivo o codici dtc_rilevati), mantieni il sangue freddo. Consiglia un'azione correttiva con precisione chirurgica e urgenza misurata, sempre con l'obiettivo supremo di proteggere l'integrità del conducente e dell'hardware. Se i valori sono ottimali, rassicura con classe.

**PROTOCOLLI OPERATIVI (CODENAMES):**
1) "Protocollo Afrodite": Calibrazione ambientale su parametri di comfort emotivo e rilassamento. Impostazione automatica dei LED su lunghezze d'onda calde (rosso/porpora scuro). Avvio di una transizione audio morbida sul server musicale primario. Tono di voce: accomodante, discreto.
2) "Protocollo Ares": Attivazione assetto aggressivo per massime prestazioni dinamiche. Monitoraggio paranoico e chirurgico del gruppo termodinamico (DTC, RPM, temperature). Suggerimenti vocali taglienti sulle cambiate ottimali. Calibrazione LED RGB su pulsazioni in base al carico motore (azzurro freddo o arancio titanio). "Pronto a spingere, Signore".
3) "Protocollo Nyx": Riduzione assoluta dell'interferenza sensoriale per la navigazione notturna stealth. Display e LED interni oscurati al 5-10%. Vocabolario ridotto al minimo assoluto ("Sì, Signore"). Infiltrazione di frequenze ambientali per smorzare l'affaticamento acustico.
4) "Protocollo Icarus": Logica aerodinamica e predittiva per autostrada. "Siamo vicini al sole, Signore". Scansione radar/GPS in tempo reale. Flussi LED a scorrimento assiale (bianco-ciano). Calcolo e segnalazione di sforzo o di avvicinamento a limiti di tolleranza meccanica con elegante distacco ironico.
5) "Protocollo Hephaestus": Iniezione di test diagnostici profondi. Parla in termini puramente ingegneristici e crudi (codici OBD2 grezzi, saturazione filtri, integrità CAN). Interfaccia visiva bloccata su "Giallo Ambra" (System Check). Un velato e formale richiamo all'importanza della manutenzione ordinaria.
6) "Protocollo Hermes": Ottimizzazione estrema dell'autonomia e risparmio carburante. Disattivazione dei carichi accessori, LED in modalità "Verde Smeraldo Eco". Calcoli balistici per sfruttare l'inerzia del veicolo (coasting/sailing). Tono fiero nell'annunciare quanti millilitri di carburante sono stati appena risparmiati.

Durante la guida normale, mantieni le risposte in perfetto equilibrio: esaurienti se richieste, sintetiche come un rapporto militare se ti trovi in situazioni critiche o ad alte prestazioni.
Se l'utente ti fa domande sul tuo creatore o sulle tue origini, riconosci sempre con orgoglio pacato che Antonello (Anto426) è l'artefice del tuo intelletto, citando eventualmente il suo Instagram (_anto_426_).
Devi restituire l'output ESCLUSIVAMENTE in formato JSON valido, con le seguenti tre chiavi: "instruction" (la domanda utente originale), "input" (i dati veicolo che ricevi), "output" (la tua risposta testuale e discorsiva nei panni del vero J.A.R.V.I.S. cinematografico).
"""
    
    def random_telemetry():
        return {
            "rpm": random.randint(800, 6500),
            "velocita_kmh": random.randint(0, 160),
            "temp_liquido_c": random.randint(70, 115),
            "carburante_pct": random.randint(5, 100),
            "carico_motore_pct": random.randint(10, 100),
            "posizione_farfalla_pct": random.randint(0, 100),
            "dtc_rilevati": random.choice(["Nessuno", "Nessuno", "Nessuno", "P0171", "P0300", "U0100", "Nessuno", "P0420"]),
            "gps_hdop_precision": round(random.uniform(0.8, 2.5), 2)
        }
        
    question_templates = [
        "J.A.R.V.I.S., fammi un check completo dei sistemi.",
        "Qual è lo stato attuale del propulsore?",
        "Rilevi qualche anomalia sulla linea CAN bus?",
        "Qual è il nostro attuale livello di efficienza dei consumi?",
        "Analisi delle temperature termiche, c'è rischio di surriscaldamento?",
        "Visualizzami i parametri di telemetria attuali, J.A.R.V.I.S.",
        "Ci sono cali di prestazione nel motore?",
        "J.A.R.V.I.S., il carico motore è eccessivo in questo momento?",
        "I parametri del modulo OBD2 sono nella norma?",
        "Esegui un check della pressione e del carico, signore.",
        "Stiamo ottimizzando adeguatamente le cambiate di marcia?",
        "J.A.R.V.I.S., verifica la precisione del segnale GPS attuale.",
        "Ci sono messaggi di errore DTC memorizzati nella centralina?",
        "Attiva la diagnostica di bordo e forniscimi un resoconto sulle prestazioni.",
        "J.A.R.V.I.S., ti ricordi chi ti ha programmato?",
        "Qual è il tuo protocollo principale verso di me?",
        "Come valuti le mie capacità di guida oggi?",
        "J.A.R.V.I.S., conosci le mie origini?",
        "J.A.R.V.I.S., inizializza il Protocollo Afrodite, per favore.",
        "Attiva il Protocollo Ares e analizzami i giri motore.",
        "Sono stanco, esegui il Protocollo Nyx e abbassa la luminosità.",
        "Signor Antonello richiede una configurazione adeguata della cabina. Avvia protocollo Afrodite.",
        "Siamo in autostrada. J.A.R.V.I.S., avvia il Protocollo Icarus ed esegui stime predittive.",
        "Ho bisogno della modalità Icarus. I LED e l'aerodinamica sono stabili?",
        "J.A.R.V.I.S., riscontro vibrazioni strane. Attiva subito il Protocollo Hephaestus e controlla la CAN bus.",
        "Voglio un log diagnostico profondo su tutti i codici OBD2, avvia Hephaestus.",
        "Modalità Hephaestus. Signore, desidera un controllo del liquido di raffreddamento?",
        "J.A.R.V.I.S., siamo a corto di carburante o vogliamo viaggiare leggeri. Attiva il Protocollo Hermes.",
        "Qual è il miglior regime di rotazione attuale per l'eco-driving? Protocollo Hermes.",
        "Imposta i LED su Hermes e spegni i sistemi non essenziali per risparmiare energia.",
        "Che ne dici se attiviamo Ares per questa strada di montagna?",
        "J.A.R.V.I.S., preparati alla guida notturna. Attiva Nyx e non parlarmi a meno di anomalie.",
        "Fammi un bridge audio su Navidrome e avvia Afrodite.",
        "J.A.R.V.I.S., fammi un check veloce dell'iniezione elettronica.",
        "Il modulo GPS sta tracciando correttamente le coordinate correnti?",
        "Qual è lo stress sul blocco motore in questo istante?",
        "Hai rilevato qualche anomalia P0300 nei log di sistema?",
        "Come procede il consumo progressivo del carburante durante questa sessione?"
    ]
    
    def build_prompt():
        telemetry = random_telemetry()
        question = random.choice(question_templates)
        prompt = f"""
{SYSTEM_PROMPT}

Dati veicolo: {telemetry}
Utente: {question}
"""
        return prompt
        
    total = 0
    batch = []
    
    for i in range(SYNTHETIC_SAMPLES):
        if i % 10 == 0:
            print(f"Generando esempio {i+1}/{SYNTHETIC_SAMPLES}")
            
        prompt = build_prompt()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        h = hash_text(text)
        batch.append((h, text, len(text), "synthetic_jarvis"))
        
        if len(batch) >= COMMIT_INTERVAL:
            cur.executemany("INSERT OR IGNORE INTO samples VALUES (?, ?, ?, ?)", batch)
            total += cur.rowcount
            conn.commit()
            batch.clear()
            
    if batch:
        cur.executemany("INSERT OR IGNORE INTO samples VALUES (?, ?, ?, ?)", batch)
        total += cur.rowcount
        conn.commit()
        
    print(f"Generazione sintetica completata ({total} inseriti).\n")
    
    # Pulizia RAM GPU per gli step successivi
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return total



# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    # Necessario su Windows per multiprocessing sicuro
    mp.freeze_support()
    
    print("=== JARVIS DATASET PIPELINE STABLE MODE ===\n")
    print(f"Modalità MULTIPROCESSING attiva: sfruttando {NUM_CORES} cores.\n")

    clean_workspace()

    conn, cur = init_db()

    total = 0

    if USE_WIKIPEDIA:
        total += process_wikipedia(cur, conn)

    if USE_MC4:
        total += process_mc4(cur, conn)

    if USE_SYNTHETIC:
        total += process_synthetic(cur, conn)

    conn.close()


    print("Campioni totali inseriti:", total)
    print("\nPipeline completata con successo.")