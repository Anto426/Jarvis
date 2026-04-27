import os
import json
import re
import html
import unicodedata
import multiprocessing as mp
from tqdm import tqdm
from langdetect import detect, LangDetectException
from training.paths import get_path
from data_pipeline.sample_format import (
    normalize_messages,
    normalize_sample,
    render_training_text,
    sample_language,
)

# =========================
# LOAD PATHS
# =========================

RAW_DIR = get_path("raw_data_dir", create=True)
CLEAN_DIR = get_path("cleaned_data_dir", create=True)

NUM_CORES = max(1, mp.cpu_count() - 1)
RAW_INCLUDE = [
    name.strip()
    for name in os.environ.get("JARVIS_RAW_INCLUDE", "").split(";")
    if name.strip()
]
ALLOWED_LANGS = {
    lang.strip()
    for lang in os.environ.get("JARVIS_ALLOWED_LANGS", "it").split(",")
    if lang.strip()
}
ALLOWED_FORMATS = {
    item.strip().lower()
    for item in os.environ.get("JARVIS_ALLOWED_FORMATS", "").split(",")
    if item.strip()
}
DATA_PROFILE = os.environ.get("JARVIS_DATA_PROFILE", "auto").strip().lower()

# =========================
# QUALITY THRESHOLDS
# =========================

MIN_LENGTH = 400
MIN_STRUCTURED_LENGTH = 40
MIN_ALPHA_RATIO = 0.6
MAX_DIGIT_RATIO = 0.2

SOURCE_PROFILES = {
    "piqa_italian": {
        "min_length": 40,
        "min_alpha_ratio": 0.35,
        "max_digit_ratio": 0.45,
        "detect_language": False,
    },
    "squad_it": {
        "min_length": 40,
        "min_alpha_ratio": 0.35,
        "max_digit_ratio": 0.55,
        "detect_language": False,
    },
    "evol_instruct_italian": {
        "min_length": 40,
        "min_alpha_ratio": 0.30,
        "max_digit_ratio": 0.55,
        "detect_language": False,
    },
    "wikipedia_it": {
        "min_length": 350,
        "min_alpha_ratio": 0.55,
        "max_digit_ratio": 0.25,
        "detect_language": True,
    },
    "wikipedia_it_wikimedia_direct": {
        "min_length": 350,
        "min_alpha_ratio": 0.55,
        "max_digit_ratio": 0.25,
        "detect_language": True,
    },
    "fineweb2_it": {
        "min_length": 260,
        "min_alpha_ratio": 0.55,
        "max_digit_ratio": 0.25,
        "detect_language": True,
    },
    "stackexchange_auto": {
        "min_length": 80,
        "min_alpha_ratio": 0.35,
        "max_digit_ratio": 0.45,
        "detect_language": False,
    },
    "local_import": {
        "min_length": 40,
        "min_alpha_ratio": 0.30,
        "max_digit_ratio": 0.55,
        "detect_language": True,
    },
}

AUTOMOTIVE_TERMS = [
    "abs", "airbag", "alternator", "automobile", "battery", "benzina",
    "brake", "brakes", "cambio", "car", "cars", "centralina", "clutch",
    "diesel", "dtc", "ecu", "engine", "freno", "freni", "fuel", "gearbox",
    "hybrid", "ignition", "manutenzione", "mechanic", "motore", "obd",
    "obd2", "oil", "petrol", "sensore", "sensor", "throttle", "torque",
    "transmission", "vehicle", "veicolo",
]
AUTOMOTIVE_RE = re.compile(
    r"(?<![a-z0-9])(" + "|".join(re.escape(term) for term in AUTOMOTIVE_TERMS) + r")(?![a-z0-9])",
    flags=re.IGNORECASE,
)

STACK_META_RE = re.compile(
    r"\b(private beta|public beta|stack exchange|stackexchange|area51|reputation|upvote|downvote|moderator)\b",
    flags=re.IGNORECASE,
)

SEPARATOR_LINE_RE = re.compile(r"^\s*(?:[\*\-=~_#\.]\s*){8,}\s*$")
LONG_SYMBOL_RUN_RE = re.compile(r"(?:[\*\-=~_#\.]\s*){10,}")
REPEATED_TOKEN_RE = re.compile(r"\b([A-Za-zÀ-ÿ]{3,})(?:\s+\1\b){5,}", flags=re.IGNORECASE)
WEB_BOILERPLATE_RE = re.compile(
    r"\b("
    r"accetta tutti|archivio|categorie|commenti|continua a leggere|cookie|copyright|"
    r"iscriviti|lascia un commento|leggi tutto|newsletter|posted by|privacy policy|"
    r"pubblicato da|registrati|termini e condizioni|tag:"
    r")\b",
    flags=re.IGNORECASE,
)
NAVIGATION_LINE_RE = re.compile(
    r"\b(home|menu|login|logout|cerca|search|rss|feed|precedente|successivo|"
    r"condividi|facebook|twitter|instagram)\b",
    flags=re.IGNORECASE,
)
SENTENCE_END_RE = re.compile(r"[.!?](?:\s|$)")
SPECIAL_TOKEN_RE = re.compile(r"<\|(?:system|user|assistant|context|answer|end)\|>")

# =========================
# CLEAN FUNCTIONS
# =========================

def source_profile(sample):
    source = (sample.get("source") or "").strip()
    return SOURCE_PROFILES.get(source, {})


def normalize_text_surface(text, preserve_newlines=True):
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"\u00a0", " ", text)
    if preserve_newlines:
        text = re.sub(r"[ \t]+", " ", text)
        text = "\n".join(line.strip() for line in text.split("\n"))
        text = re.sub(r"\n{3,}", "\n\n", text)
    else:
        text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_urls(text):
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def strip_html(text):
    text = html.unescape(text)
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p\s*>", "\n\n", text)
    text = re.sub(r"(?i)</li\s*>", "\n", text)
    text = re.sub(r"(?i)<li\s*>", "- ", text)
    text = re.sub(r"(?i)</h[1-6]\s*>", "\n\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return normalize_text_surface(text)


def strip_repeated_separators(text):
    kept_lines = []
    for line in text.split("\n"):
        if SEPARATOR_LINE_RE.match(line):
            continue
        kept_lines.append(line)

    text = "\n".join(kept_lines)
    text = LONG_SYMBOL_RUN_RE.sub(" ", text)
    return normalize_text_surface(text)


def strip_web_boilerplate_lines(text):
    kept_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            kept_lines.append(line)
            continue

        if len(stripped) <= 180 and (WEB_BOILERPLATE_RE.search(stripped) or NAVIGATION_LINE_RE.search(stripped)):
            continue

        kept_lines.append(line)

    return normalize_text_surface("\n".join(kept_lines))


def clean_wiki_text(text):
    text = normalize_text_surface(text)

    # 3. Sezioni di Servizio: Tronca il testo all'inizio di queste sezioni
    service_keywords = [
        "Note", "Bibliografia", "Voci correlate", "Altri progetti", "Collegamenti esterni"
    ]
    section_pattern = re.compile(
        r'(?:\n|^)(?:==+ *|)(?:' + '|'.join(service_keywords) + r')(?: *==+|\n|:)', 
        flags=re.IGNORECASE
    )
    match = section_pattern.search(text)
    if match:
        text = text[:match.start()]

    # 1. Firme e Timestamp
    text = re.sub(r'--[^\n]*?(?:\d{1,2}:\d{2}|\d{1,2}\s+[a-zA-Z]+\s+\d{4})[^\n]*', ' ', text)

    # 2. Codice Wiki e Template
    # Template {{...}} (gestisce fino a 2 livelli di annidamento)
    text = re.sub(r'\{\{(?:[^{}]|\{[^{}]*\})*\}\}', ' ', text)
    text = re.sub(r'\{\{.*?\}\}', ' ', text) # Fallback
    text = re.sub(r"\s\|[A-Za-zÀ-ÿ0-9 _-]{1,40}\s*=\s*[^|{}]*", " ", text)

    # Link interni: [[Target|Testo]] -> Testo, [[Testo]] -> Testo
    text = re.sub(r'\[\[(?:[^\]\|\n]*\|)?([^\]\|\n]+)\]\]', r'\1', text)
    text = re.sub(r'\[\[.*?\]\]', ' ', text) # Rimanenti non validi

    # Simboli ripetuti
    text = re.sub(r'\^{2,}', ' ', text)
    text = strip_repeated_separators(text)

    # 4. Identificativi di Sistema
    text = re.sub(r"<(?!\|).*?>", " ", text)
    text = strip_urls(text)
    text = re.sub(r"\S+\.(?:ogg|oga|mp3|jpg|jpeg|png|svg|gif|webm|pdf)\b", " ", text, flags=re.IGNORECASE)
    # Stringhe alfanumeriche isolate (es. hash, ID revisione di almeno 8 caratteri)
    text = re.sub(r'\b(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{8,}\b', ' ', text)

    text = re.sub(r"\bCategoria:[^\n]+", " ", text)
    text = re.sub(r"\{\{|\}\}|\[\[|\]\]|\s\|[A-Za-zÀ-ÿ0-9 _-]{1,40}\s*=", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = trim_wiki_lead_noise(text)
    return normalize_text_surface(text)


def trim_wiki_lead_noise(text):
    lines = text.split("\n", 1)
    if len(lines) != 2:
        return text

    title = lines[0].strip()
    body = lines[1].strip()
    if not title or not body:
        return text

    title_key = re.escape(title.lower())
    match = re.search(r"(?<![a-z0-9à-ÿ])" + title_key + r"(?![a-z0-9à-ÿ])", body.lower())
    if not match:
        return text

    idx = match.start()
    if idx <= 20 or idx >= 1500:
        return text

    prefix = body[:idx]
    prefix_is_noise = (
        "|" in prefix
        or "}}" in prefix
        or "]]" in prefix
        or re.search(r"\.(?:ogg|oga|mp3|jpg|jpeg|png|svg|gif|webm|pdf)\b", prefix, re.IGNORECASE)
        or prefix.lstrip().startswith("(")
        or (idx < 300 and prefix.count(".") < 2)
    )
    if not prefix_is_noise:
        return text

    return f"{title}\n{body[idx:]}"


def clean_web_text(text):
    text = strip_html(text)
    text = strip_urls(text)
    text = strip_repeated_separators(text)
    text = re.sub(
        r"\b(cookie|privacy policy|termini e condizioni|accetta tutti|newsletter|"
        r"javascript|browser non supportato)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = strip_web_boilerplate_lines(text)
    return normalize_text_surface(text)


def clean_structured_text(text):
    text = strip_repeated_separators(text)
    text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
    return text


def clean_chat_text(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    # 5. Regole di Formattazione
    text = re.sub(r'[ \t]+', ' ', text)
    text = '\n'.join(line.strip() for line in text.split('\n'))
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def clean_stackexchange_text(text):
    return strip_html(text)


def clean_text_for_sample(text, sample):
    source = sample.get("source")
    sample_format = sample.get("format", "text")

    if source in {"wikipedia_it", "wikipedia_it_wikimedia_direct"}:
        return clean_wiki_text(text)
    if source == "fineweb2_it":
        return clean_web_text(text)
    if source == "stackexchange_auto":
        return clean_stackexchange_text(text)
    if sample_format == "chat":
        return clean_chat_text(text)
    if sample_format in {"qa", "choice_qa", "instruction"}:
        return clean_structured_text(text)
    return clean_web_text(text)


def is_valid_text(text, sample, structured=False):
    profile = source_profile(sample)
    min_length = profile.get("min_length", MIN_STRUCTURED_LENGTH if structured else MIN_LENGTH)
    min_alpha_ratio = profile.get("min_alpha_ratio", 0.4 if structured else MIN_ALPHA_RATIO)
    max_digit_ratio = profile.get("max_digit_ratio", 0.35 if structured else MAX_DIGIT_RATIO)

    if len(text) < min_length:
        return False

    alpha = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)

    if alpha / len(text) < min_alpha_ratio:
        return False

    if digits / len(text) > max_digit_ratio:
        return False

    return True


def sentence_count(text):
    return len(SENTENCE_END_RE.findall(text))


def symbol_ratio(text):
    if not text:
        return 1.0
    noisy = sum(1 for char in text if char in "*=_~#|{}[]<>")
    return noisy / len(text)


def looks_like_low_quality_text(text, sample, structured=False):
    source = sample.get("source")
    sample_format = sample.get("format", "text")
    strict_lm = DATA_PROFILE == "italian_lm" and sample_format == "text"
    quality_text = SPECIAL_TOKEN_RE.sub("", text)

    if LONG_SYMBOL_RUN_RE.search(text) or REPEATED_TOKEN_RE.search(text):
        return True

    if symbol_ratio(quality_text) > (0.12 if structured else 0.04):
        return True

    if source == "fineweb2_it":
        if WEB_BOILERPLATE_RE.search(text):
            return True

        if strict_lm and sentence_count(text) < 4:
            return True

        lines = [line for line in text.split("\n") if line.strip()]
        if strict_lm and lines:
            short_lines = sum(1 for line in lines if len(line.strip()) < 80)
            if short_lines / len(lines) > 0.75:
                return True

    return False


def passes_source_filter(sample, text):
    source = sample.get("source")

    if source == "piqa_italian":
        prompt = sample.get("prompt", "")
        completion = sample.get("completion", "")
        return (
            "Situazione:" in prompt
            and "Opzioni:" in prompt
            and bool(re.match(r"^[A-Z]\)", completion.strip()))
        )

    if source == "squad_it":
        return bool(sample.get("prompt") and sample.get("completion"))

    if source == "stackexchange_auto":
        return bool(AUTOMOTIVE_RE.search(text)) and not STACK_META_RE.search(text)

    return True


def is_allowed_language(text, sample=None):
    profile = source_profile(sample or {})
    language = sample_language(sample or {})
    if language and language not in ALLOWED_LANGS:
        return False

    if profile.get("detect_language") is False:
        return True

    if language and language in ALLOWED_LANGS and profile.get("detect_language") is not True:
        return True

    try:
        return detect(text) in ALLOWED_LANGS
    except LangDetectException:
        return False


def clean_sample_fields(sample):
    sample = normalize_sample(sample)
    sample_format = sample.get("format", "text")

    if sample_format == "chat":
        cleaned_messages = []
        for message in normalize_messages(sample.get("messages", [])):
            content = clean_text_for_sample(message.get("content", ""), sample)
            if content:
                cleaned_messages.append({**message, "content": content})
        sample["messages"] = cleaned_messages
    else:
        for field in ("text", "prompt", "completion", "instruction", "question", "answer", "output"):
            if field in sample:
                sample[field] = clean_text_for_sample(sample.get(field, ""), sample)

    sample["text"] = render_training_text(sample)
    return sample


def process_line(line):
    try:
        sample = json.loads(line)
    except:
        return None

    sample = clean_sample_fields(sample)
    text = render_training_text(sample)
    structured = sample.get("format") != "text"

    if ALLOWED_FORMATS and sample.get("format", "text") not in ALLOWED_FORMATS:
        return None

    if not passes_source_filter(sample, text):
        return None

    if looks_like_low_quality_text(text, sample=sample, structured=structured):
        return None

    if not is_valid_text(text, sample=sample, structured=structured):
        return None

    if not is_allowed_language(text, sample=sample):
        return None

    return sample

# =========================
# CLEAN FILE
# =========================

def clean_file(input_path, output_path):
    saved = 0

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        with mp.Pool(NUM_CORES) as pool:
            for result in tqdm(pool.imap_unordered(process_line, fin, chunksize=1000)):
                if not result:
                    continue

                json.dump(result, fout, ensure_ascii=False)
                fout.write("\n")
                saved += 1

    print(f"Salvati {saved} record puliti in {output_path}")

# =========================
# MAIN
# =========================

def main():

    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".jsonl")]
    if RAW_INCLUDE:
        files = [f for f in files if f in RAW_INCLUDE]

    if not files:
        print("Nessun file raw trovato.")
        return

    for file in files:
        input_path = os.path.join(RAW_DIR, file)
        output_path = os.path.join(CLEAN_DIR, file.replace(".jsonl", "_clean.jsonl"))

        print(f"Pulizia {file} con {NUM_CORES} core...")
        clean_file(input_path, output_path)

    print("Clean completato.")

if __name__ == "__main__":
    mp.freeze_support()
    main()
