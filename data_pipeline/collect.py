import os
import json
import time
import bz2
import re
import xml.etree.ElementTree as ET
from tqdm import tqdm
from training.paths import configure_cache_env, get_path

# =========================
# LOAD PATHS
# =========================

configure_cache_env()

RAW_DIR = get_path("raw_data_dir", create=True)
CACHE_DIR = get_path("cache_dir", create=True)

# =========================
# CONFIG
# =========================

USE_HUGGINGFACE = os.environ.get("JARVIS_USE_HUGGINGFACE", "0") == "1"
USE_WIKIPEDIA = USE_HUGGINGFACE
USE_MC4 = USE_HUGGINGFACE
USE_STACKEXCHANGE = USE_HUGGINGFACE
USE_WIKIMEDIA_DIRECT = True

WIKIPEDIA_SOURCES = [
    ("wikimedia/wikipedia", "20231101.it"),
    ("OpenLLM-France/wikipedia", "it"),
]
MC4_SOURCES = [
    ("mc4", "it"),
    ("bertin-project/mc4-sampling", "it"),
]

MC4_LIMIT = 500_000
STACK_LIMIT = 300_000
WIKIMEDIA_DIRECT_LIMIT = 100_000
WIKIMEDIA_ARTICLES_URL = (
    "https://dumps.wikimedia.org/itwiki/latest/"
    "itwiki-latest-pages-articles1.xml-p1p316052.bz2"
)

MIN_TEXT_LENGTH = 200
DOWNLOAD_RETRIES = 3
RETRY_SLEEP_SECONDS = 10

# =========================
# HELPERS
# =========================

def save_jsonl(iterator, output_file, text_extractor, limit=None):
    saved = 0

    with open(output_file, "w", encoding="utf-8") as f:

        for i, sample in enumerate(tqdm(iterator)):

            if limit and i >= limit:
                break

            text = text_extractor(sample)

            if text and len(text) >= MIN_TEXT_LENGTH:
                json.dump({"text": text}, f, ensure_ascii=False)
                f.write("\n")
                saved += 1

    return saved


def load_streaming_dataset(path, *args, **kwargs):
    from datasets import load_dataset

    last_error = None

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            load_kwargs = {
                "streaming": True,
                "cache_dir": os.environ.get("HF_DATASETS_CACHE"),
                **kwargs,
            }
            if token:
                load_kwargs["token"] = token

            return load_dataset(
                path,
                *args,
                **load_kwargs,
            )
        except Exception as exc:
            last_error = exc
            print(
                f"Download fallito per {path} "
                f"({attempt}/{DOWNLOAD_RETRIES}): {type(exc).__name__}: {exc}"
            )
            if attempt < DOWNLOAD_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)

    raise last_error


def load_first_available(name, sources, **kwargs):
    last_error = None

    for source in sources:
        path = source[0]
        args = source[1:]
        try:
            print(f"Provo sorgente {name}: {path} {' '.join(args)}")
            return load_streaming_dataset(path, *args, **kwargs)
        except Exception as exc:
            last_error = exc
            print(f"Sorgente non disponibile: {path} -> {type(exc).__name__}: {exc}")

    raise last_error


def run_source(name, collector):
    try:
        return collector()
    except Exception as exc:
        print(f"Sorgente saltata: {name} -> {type(exc).__name__}: {exc}")
        return 0


def download_file(url, output_path):
    import requests

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"File gia presente: {output_path}")
        return output_path

    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    print(f"Scarico da {url}")

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(tmp_path, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                progress.update(len(chunk))

    tmp_path.replace(output_path)
    return output_path


def local_name(tag):
    return tag.rsplit("}", 1)[-1]


def find_child_text(element, child_name):
    for child in element:
        if local_name(child.tag) == child_name:
            return child.text or ""
    return ""


def find_revision_text(page):
    for child in page:
        if local_name(child.tag) != "revision":
            continue

        for revision_child in child:
            if local_name(revision_child.tag) == "text":
                return revision_child.text or ""

    return ""


def basic_wiki_text_cleanup(text):
    text = re.sub(r"\{\{.*?\}\}", " ", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[\[File:.*?\]\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[Immagine:.*?\]\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[[^|\]]+\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"={2,}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def collect_wikimedia_direct():
    print("Scarico Wikipedia IT diretto da Wikimedia dumps...")

    dump_path = CACHE_DIR / "downloads" / "itwiki-latest-pages-articles1.xml-p1p316052.bz2"
    output_file = RAW_DIR / "wikipedia_it_wikimedia_direct.jsonl"
    download_file(WIKIMEDIA_ARTICLES_URL, dump_path)

    saved = 0
    with bz2.open(dump_path, "rb") as source, open(output_file, "w", encoding="utf-8") as out:
        for event, elem in tqdm(ET.iterparse(source, events=("end",))):
            if local_name(elem.tag) != "page":
                continue

            title = find_child_text(elem, "title")
            text = basic_wiki_text_cleanup(find_revision_text(elem))
            full_text = f"{title}\n{text}".strip()

            if len(full_text) >= MIN_TEXT_LENGTH:
                json.dump({"text": full_text}, out, ensure_ascii=False)
                out.write("\n")
                saved += 1

            elem.clear()

            if WIKIMEDIA_DIRECT_LIMIT and saved >= WIKIMEDIA_DIRECT_LIMIT:
                break

    print(f"Wikimedia diretto salvato: {saved} record.")
    return saved


# =========================
# WIKIPEDIA
# =========================

def collect_wikipedia():

    print("Scarico Wikipedia IT...")

    dataset = load_first_available(
        "Wikipedia IT",
        WIKIPEDIA_SOURCES,
        split="train"
    )

    output_file = os.path.join(RAW_DIR, "wikipedia_it.jsonl")

    saved = save_jsonl(
        dataset,
        output_file,
        text_extractor=lambda x: x.get("text", "")
    )

    print(f"Wikipedia salvata: {saved} record.")
    return saved


# =========================
# MC4
# =========================

def collect_mc4():

    print("Scarico MC4 IT...")

    dataset = load_first_available(
        "MC4 IT",
        MC4_SOURCES,
        split="train"
    )

    output_file = os.path.join(RAW_DIR, "mc4_it.jsonl")

    saved = save_jsonl(
        dataset,
        output_file,
        text_extractor=lambda x: x.get("text", ""),
        limit=MC4_LIMIT
    )

    print(f"MC4 salvato: {saved} record.")
    return saved


# =========================
# STACKEXCHANGE AUTOMOTIVE
# =========================

def collect_stackexchange():

    print("Scarico StackExchange...")

    dataset = load_streaming_dataset(
        "HuggingFaceH4/stack-exchange-preferences",
        split="train"
    )

    output_file = os.path.join(RAW_DIR, "stackexchange_auto.jsonl")

    automotive_keywords = [
        # Inglese tecnico
        "engine", "car", "vehicle", "OBD", "ECU",
        "sensor", "brake", "battery", "transmission",
        "diesel", "petrol", "hybrid", "electric",
        "torque", "throttle", "gearbox", "clutch",
        # Italiano
        "motore", "automobile", "veicolo", "benzina",
        "freni", "batteria", "centralina",
        "cambio", "frizione", "sensore"
    ]

    def extract_text(sample):

        question = sample.get("question", "")
        response = sample.get("response", "")

        text = question + "\n" + response
        text_lower = text.lower()

        if any(k in text_lower for k in automotive_keywords):
            return text

        return ""

    saved = save_jsonl(
        dataset,
        output_file,
        text_extractor=extract_text,
        limit=STACK_LIMIT
    )

    print(f"StackExchange automotive salvato: {saved} record.")
    return saved


# =========================
# MAIN
# =========================

def main():
    total_saved = 0
    existing_raw_files = list(RAW_DIR.glob("*.jsonl"))

    if not USE_HUGGINGFACE:
        print("Hugging Face disattivato: uso sorgenti dirette/locali.")

    if USE_WIKIPEDIA:
        total_saved += run_source("Wikipedia IT", collect_wikipedia)

    if USE_MC4:
        total_saved += run_source("MC4 IT", collect_mc4)

    if USE_STACKEXCHANGE:
        total_saved += run_source("StackExchange", collect_stackexchange)

    if total_saved == 0 and USE_WIKIMEDIA_DIRECT:
        total_saved += run_source("Wikimedia direct", collect_wikimedia_direct)

    if total_saved == 0 and not existing_raw_files:
        raise RuntimeError(
            "Nessun dato scaricato. La connessione a Hugging Face sembra bloccata "
            "o instabile. Riprova piu tardi, usa una VPN/proxy, oppure metti file "
            ".jsonl locali in data/raw/ con una colonna/campo 'text'."
        )

    if total_saved == 0:
        print("Nessun nuovo download, uso i file raw gia presenti.")
        return

    print(f"Collect completato: {total_saved} record totali.")


if __name__ == "__main__":
    main()
