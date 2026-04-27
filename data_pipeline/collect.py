import os
import json
import time
import bz2
import re
from urllib.parse import urljoin
import xml.etree.ElementTree as ET
from tqdm import tqdm
from training.paths import configure_cache_env, get_path
from data_pipeline.sample_format import (
    first_answer_text,
    normalize_sample,
    render_training_text,
)

# =========================
# LOAD PATHS
# =========================

configure_cache_env()

RAW_DIR = get_path("raw_data_dir", create=True)
CACHE_DIR = get_path("cache_dir", create=True)

# =========================
# CONFIG
# =========================

HUGGINGFACE_MODE = os.environ.get("JARVIS_USE_HUGGINGFACE", "auto").lower()
RAW_INCLUDE = {
    name.strip()
    for name in os.environ.get("JARVIS_RAW_INCLUDE", "").split(";")
    if name.strip()
}
HF_PREFLIGHT_URL = (
    "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/"
    "20231101.it/train-00000-of-00010.parquet"
)
HF_PREFLIGHT_BYTES = 16 * 1024 * 1024


def env_int(name, default):
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def huggingface_preflight():
    if HUGGINGFACE_MODE in {"0", "false", "no", "off"}:
        return False

    if HUGGINGFACE_MODE in {"force", "forced"}:
        return True

    import requests

    print("Controllo rapido Hugging Face...")
    try:
        with requests.get(
            HF_PREFLIGHT_URL,
            stream=True,
            timeout=30,
        ) as response:
            response.raise_for_status()
            downloaded = 0
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded >= HF_PREFLIGHT_BYTES:
                    break
    except Exception as exc:
        print(f"Hugging Face non stabile ({type(exc).__name__}: {exc}). Lo salto per questa run.")
        return False

    print("Hugging Face disponibile.")
    return True


def wants_raw_file(name):
    return not RAW_INCLUDE or name in RAW_INCLUDE


NEEDS_HUGGINGFACE = (
    wants_raw_file("wikipedia_it.jsonl")
    or wants_raw_file("fineweb2_it.jsonl")
    or wants_raw_file("stackexchange_auto.jsonl")
    or wants_raw_file("squad_it.jsonl")
    or wants_raw_file("piqa_italian.jsonl")
    or wants_raw_file("evol_instruct_italian.jsonl")
)
USE_HUGGINGFACE = NEEDS_HUGGINGFACE and huggingface_preflight()
USE_WIKIPEDIA = (
    os.environ.get("JARVIS_USE_HF_WIKIPEDIA", "0") == "1"
    and USE_HUGGINGFACE
    and wants_raw_file("wikipedia_it.jsonl")
)
USE_FINEWEB = USE_HUGGINGFACE and wants_raw_file("fineweb2_it.jsonl")
USE_STACKEXCHANGE = USE_HUGGINGFACE and wants_raw_file("stackexchange_auto.jsonl")
USE_SQUAD_IT = USE_HUGGINGFACE and wants_raw_file("squad_it.jsonl")
USE_PIQA_ITALIAN = USE_HUGGINGFACE and wants_raw_file("piqa_italian.jsonl")
USE_EVOL_INSTRUCT_ITALIAN = USE_HUGGINGFACE and wants_raw_file("evol_instruct_italian.jsonl")
USE_WIKIMEDIA_DIRECT = wants_raw_file("wikipedia_it_wikimedia_direct.jsonl")

WIKIPEDIA_SOURCES = [
    ("wikimedia/wikipedia", "20231101.it"),
    ("OpenLLM-France/wikipedia", "it"),
]
FINEWEB_SOURCES = [
    ("HuggingFaceFW/fineweb-2", "ita_Latn"),
]

WIKIPEDIA_LIMIT = env_int("JARVIS_WIKIPEDIA_LIMIT", 100_000)
FINEWEB_LIMIT = env_int("JARVIS_FINEWEB_LIMIT", 200_000)
STACK_LIMIT = env_int("JARVIS_STACK_LIMIT", 300_000)
SQUAD_IT_LIMIT = env_int("JARVIS_SQUAD_IT_LIMIT", 0)
PIQA_IT_LIMIT = env_int("JARVIS_PIQA_IT_LIMIT", 0)
EVOL_INSTRUCT_IT_LIMIT = env_int("JARVIS_EVOL_INSTRUCT_IT_LIMIT", 0)
WIKIMEDIA_DIRECT_LIMIT = env_int("JARVIS_WIKIMEDIA_DIRECT_LIMIT", 100_000)
WIKIMEDIA_DIRECT_ALL_PARTS = env_flag("JARVIS_WIKIMEDIA_DIRECT_ALL_PARTS", False)
WIKIMEDIA_DUMP_INDEX_URL = "https://dumps.wikimedia.org/itwiki/latest/"
WIKIMEDIA_ARTICLES_URL = (
    WIKIMEDIA_DUMP_INDEX_URL +
    "itwiki-latest-pages-articles1.xml-p1p316052.bz2"
)

MIN_TEXT_LENGTH = 200
MIN_STRUCTURED_TEXT_LENGTH = 40
DOWNLOAD_RETRIES = 3
RETRY_SLEEP_SECONDS = 10

# =========================
# HELPERS
# =========================

def save_jsonl(iterator, output_file, sample_builder, limit=None):
    saved = 0

    with open(output_file, "w", encoding="utf-8") as f:

        for i, sample in enumerate(tqdm(iterator)):

            if limit and i >= limit:
                break

            record = sample_builder(sample)
            if isinstance(record, str):
                record = {"text": record, "format": "text"}
            if not record:
                continue

            record = normalize_sample(record)
            text = render_training_text(record)
            min_length = (
                MIN_TEXT_LENGTH
                if record.get("format") == "text"
                else MIN_STRUCTURED_TEXT_LENGTH
            )

            if text and len(text) >= min_length:
                if not record.get("text"):
                    record["text"] = text
                json.dump(record, f, ensure_ascii=False)
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


def load_first_split_available(name, path, splits=("train", "validation", "test")):
    last_error = None

    for split in splits:
        try:
            print(f"Provo sorgente {name}: {path} split={split}")
            return load_streaming_dataset(path, split=split)
        except Exception as exc:
            last_error = exc
            print(f"Split non disponibile: {path}/{split} -> {type(exc).__name__}: {exc}")

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


def discover_wikimedia_article_urls():
    if not WIKIMEDIA_DIRECT_ALL_PARTS:
        return [WIKIMEDIA_ARTICLES_URL]

    import requests

    print("Cerco tutti i dump Wikipedia IT pages-articles disponibili...")
    response = requests.get(WIKIMEDIA_DUMP_INDEX_URL, timeout=60)
    response.raise_for_status()

    names = sorted(
        set(
            re.findall(
                r'href="(itwiki-latest-pages-articles\d+\.xml-p\d+p\d+\.bz2)"',
                response.text,
            )
        )
    )
    if not names:
        print("Indice Wikimedia senza lista completa: uso il primo dump noto.")
        return [WIKIMEDIA_ARTICLES_URL]

    urls = [urljoin(WIKIMEDIA_DUMP_INDEX_URL, name) for name in names]
    print(f"Dump Wikipedia IT trovati: {len(urls)} file.")
    return urls


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

    output_file = RAW_DIR / "wikipedia_it_wikimedia_direct.jsonl"
    urls = discover_wikimedia_article_urls()
    saved = 0

    with open(output_file, "w", encoding="utf-8") as out:
        for url in urls:
            dump_name = url.rsplit("/", 1)[-1]
            dump_path = CACHE_DIR / "downloads" / dump_name
            download_file(url, dump_path)

            with bz2.open(dump_path, "rb") as source:
                for event, elem in tqdm(ET.iterparse(source, events=("end",)), desc=dump_name):
                    if local_name(elem.tag) != "page":
                        continue

                    title = find_child_text(elem, "title")
                    text = basic_wiki_text_cleanup(find_revision_text(elem))
                    full_text = f"{title}\n{text}".strip()

                    if len(full_text) >= MIN_TEXT_LENGTH:
                        json.dump(
                            {
                                "format": "text",
                                "source": "wikipedia_it_wikimedia_direct",
                                "language": "it",
                                "text": full_text,
                            },
                            out,
                            ensure_ascii=False,
                        )
                        out.write("\n")
                        saved += 1

                    elem.clear()

                    if WIKIMEDIA_DIRECT_LIMIT and saved >= WIKIMEDIA_DIRECT_LIMIT:
                        break

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
        sample_builder=lambda x: {
            "format": "text",
            "source": "wikipedia_it",
            "language": "it",
            "text": x.get("text", ""),
        },
        limit=WIKIPEDIA_LIMIT
    )

    print(f"Wikipedia salvata: {saved} record.")
    return saved


# =========================
# FINEWEB
# =========================

def collect_fineweb():

    print("Scarico FineWeb2 IT...")

    dataset = load_first_available(
        "FineWeb2 IT",
        FINEWEB_SOURCES,
        split="train"
    )

    output_file = os.path.join(RAW_DIR, "fineweb2_it.jsonl")

    saved = save_jsonl(
        dataset,
        output_file,
        sample_builder=lambda x: {
            "format": "text",
            "source": "fineweb2_it",
            "language": "it",
            "text": x.get("text", ""),
        },
        limit=FINEWEB_LIMIT
    )

    print(f"FineWeb2 salvato: {saved} record.")
    return saved


# =========================
# STRUCTURED ITALIAN Q/A AND LOGIC
# =========================

def collect_squad_it():

    print("Scarico SQuAD IT per domande/risposte contestuali...")

    dataset = load_first_split_available(
        "SQuAD IT",
        "crux82/squad_it",
    )

    output_file = os.path.join(RAW_DIR, "squad_it.jsonl")

    def build_sample(sample):
        context = sample.get("context", "")
        question = sample.get("question", "")
        answer = first_answer_text(sample.get("answers"))

        if not context or not question or not answer:
            return None

        prompt = (
            "Leggi il contesto e rispondi in italiano in modo breve e preciso.\n\n"
            f"<|context|>\n{context}\n\nDomanda:\n{question}"
        )

        return {
            "format": "qa",
            "source": "squad_it",
            "language": "it",
            "prompt": prompt,
            "completion": answer,
            "meta": {
                "id": sample.get("id"),
                "task": "extractive_qa",
            },
        }

    saved = save_jsonl(
        dataset,
        output_file,
        sample_builder=build_sample,
        limit=SQUAD_IT_LIMIT
    )

    print(f"SQuAD IT salvato: {saved} record.")
    return saved


def collect_piqa_italian():

    print("Scarico PIQA Italian per logica fisica e scelta corretta...")

    dataset = load_first_split_available(
        "PIQA Italian",
        "sapienzanlp/piqa_italian",
    )

    output_file = os.path.join(RAW_DIR, "piqa_italian.jsonl")

    def build_sample(sample):
        question = (
            sample.get("input_translation")
            or sample.get("input")
            or sample.get("goal")
            or ""
        )
        choices = sample.get("choices_translation") or sample.get("choices") or []
        if not choices and (sample.get("sol1") or sample.get("sol2")):
            choices = [sample.get("sol1", ""), sample.get("sol2", "")]

        try:
            gold_index = int(sample.get("gold_index", sample.get("label", 0)))
        except (TypeError, ValueError):
            gold_index = 0

        if not question or not choices or gold_index < 0 or gold_index >= len(choices):
            return None

        option_lines = []
        for index, choice in enumerate(choices):
            letter = chr(ord("A") + index)
            option_lines.append(f"{letter}) {choice}")

        answer = f"{chr(ord('A') + gold_index)}) {choices[gold_index]}"
        prompt = (
            "Scegli l'opzione logicamente corretta e rispondi solo con la scelta migliore.\n\n"
            f"Situazione:\n{question}\n\nOpzioni:\n" + "\n".join(option_lines)
        )

        metadata = sample.get("metadata") or {}
        category = metadata.get("category", "physical_reasoning") if isinstance(metadata, dict) else "physical_reasoning"

        return {
            "format": "choice_qa",
            "source": "piqa_italian",
            "language": "it",
            "prompt": prompt,
            "completion": answer,
            "meta": {
                "id": sample.get("id"),
                "task": category,
            },
        }

    saved = save_jsonl(
        dataset,
        output_file,
        sample_builder=build_sample,
        limit=PIQA_IT_LIMIT
    )

    print(f"PIQA Italian salvato: {saved} record.")
    return saved


def collect_evol_instruct_italian():

    print("Scarico Evol-Instruct Italian per chat e istruzioni...")

    dataset = load_first_split_available(
        "Evol-Instruct Italian",
        "FreedomIntelligence/evol-instruct-italian",
    )

    output_file = os.path.join(RAW_DIR, "evol_instruct_italian.jsonl")

    def build_sample(sample):
        messages = (
            sample.get("messages")
            or sample.get("conversations")
            or sample.get("conversation")
        )
        if messages:
            return {
                "format": "chat",
                "source": "evol_instruct_italian",
                "language": "it",
                "messages": messages,
                "meta": {
                    "id": sample.get("id"),
                    "task": "instruction_chat",
                },
            }

        instruction = (
            sample.get("instruction")
            or sample.get("prompt")
            or sample.get("input")
            or sample.get("question")
            or ""
        )
        output = (
            sample.get("output")
            or sample.get("response")
            or sample.get("answer")
            or sample.get("completion")
            or ""
        )
        if not instruction or not output:
            text = sample.get("text", "")
            if text:
                return {
                    "format": "text",
                    "source": "evol_instruct_italian",
                    "language": "it",
                    "text": text,
                }
            return None

        return {
            "format": "instruction",
            "source": "evol_instruct_italian",
            "language": "it",
            "prompt": instruction,
            "completion": output,
            "meta": {
                "id": sample.get("id"),
                "task": "instruction_response",
            },
        }

    saved = save_jsonl(
        dataset,
        output_file,
        sample_builder=build_sample,
        limit=EVOL_INSTRUCT_IT_LIMIT
    )

    print(f"Evol-Instruct Italian salvato: {saved} record.")
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
    automotive_pattern = re.compile(
        r"(?<![a-z0-9])("
        + "|".join(re.escape(keyword.lower()) for keyword in automotive_keywords)
        + r")(?![a-z0-9])",
        flags=re.IGNORECASE,
    )

    def build_sample(sample):

        question = sample.get("question", "")
        answers = sample.get("answers", [])
        response = first_answer_text(answers)

        text = question + "\n" + response
        text_lower = text.lower()

        if automotive_pattern.search(text_lower):
            return {
                "format": "qa",
                "source": "stackexchange_auto",
                "language": "en",
                "prompt": question,
                "completion": response,
                "meta": {
                    "task": "automotive_qa",
                },
            }

        return None

    saved = save_jsonl(
        dataset,
        output_file,
        sample_builder=build_sample,
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

    if NEEDS_HUGGINGFACE and not USE_HUGGINGFACE:
        print("Hugging Face disattivato: uso sorgenti dirette/locali.")

    if USE_WIKIPEDIA:
        total_saved += run_source("Wikipedia IT", collect_wikipedia)

    if USE_FINEWEB:
        total_saved += run_source("FineWeb2 IT", collect_fineweb)

    if USE_SQUAD_IT:
        total_saved += run_source("SQuAD IT", collect_squad_it)

    if USE_PIQA_ITALIAN:
        total_saved += run_source("PIQA Italian", collect_piqa_italian)

    if USE_EVOL_INSTRUCT_ITALIAN:
        total_saved += run_source("Evol-Instruct Italian", collect_evol_instruct_italian)

    if USE_STACKEXCHANGE:
        total_saved += run_source("StackExchange", collect_stackexchange)

    if USE_WIKIMEDIA_DIRECT:
        total_saved += run_source("Wikimedia direct", collect_wikimedia_direct)

    if RAW_INCLUDE:
        missing_requested = [
            name
            for name in sorted(RAW_INCLUDE)
            if name != "local_import.jsonl"
            and (
                not (RAW_DIR / name).exists()
                or (RAW_DIR / name).stat().st_size == 0
            )
        ]
        if missing_requested:
            raise RuntimeError(
                "Mancano fonti raw richieste per questo step: "
                + ", ".join(missing_requested)
                + ". Abilita Hugging Face, aggiungi file locali, oppure correggi config/training_pipeline.yaml."
            )

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
