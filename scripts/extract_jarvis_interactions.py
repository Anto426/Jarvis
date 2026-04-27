import argparse
import html
import json
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SUBTITLES_DIR = DATA_DIR / "subtitles" / "iron_man"
RAW_OUTPUT = DATA_DIR / "raw" / "jarvis_personality.jsonl"
REVIEW_OUTPUT = DATA_DIR / "jarvis_interactions.txt"
REPORT_OUTPUT = DATA_DIR / "jarvis_interactions_report.json"

SUBTITLE_EXTENSIONS = {".srt", ".sub", ".ass"}
JARVIS_RE = re.compile(r"\bjarvis\b|j\.?\s*a\.?\s*r\.?\s*v\.?\s*i\.?\s*s\.?", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
ASS_PREFIX_RE = re.compile(r"^Dialogue:[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,")
SRT_BLOCK_RE = re.compile(r"\n\s*\n+")
TIMING_RE = re.compile(r"\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}")

JARVIS_REPLY_CUES = {
    "analisi",
    "armatura",
    "calcolo",
    "caricamento",
    "completato",
    "diagnostica",
    "energia",
    "interfaccia",
    "online",
    "potenza",
    "protocollo",
    "reattore",
    "rilevo",
    "scansione",
    "signor stark",
    "signore",
    "sistema",
    "sto",
    "traiettoria",
}

TONY_CUES = {
    "jarvis",
    "ehi",
    "fammi",
    "metti",
    "mostrami",
    "prepara",
    "senti",
}


def movie_slug(path):
    stem = path.stem.lower()
    if "3" in stem:
        return "iron_man_3"
    if "2" in stem:
        return "iron_man_2"
    return "iron_man_1"


def decode_subtitle(data):
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="ignore")


def clean_line(value):
    value = html.unescape(str(value or ""))
    value = ASS_PREFIX_RE.sub("", value)
    value = value.replace("\\N", " ").replace("{\\i1}", "").replace("{\\i0}", "")
    value = TAG_RE.sub("", value)
    value = re.sub(r"\{[^}]+\}", " ", value)
    value = re.sub(r"\[[^\]]+\]", " ", value)
    value = re.sub(r"^\s*[-–]\s*", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def parse_subtitle_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = SRT_BLOCK_RE.split(text.strip())
    lines = []

    for block in blocks:
        raw_lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not raw_lines:
            continue

        payload = []
        for line in raw_lines:
            if line.isdigit() or TIMING_RE.search(line):
                continue
            if line.startswith("[Script Info]") or line.startswith("[V4+ Styles]"):
                continue
            cleaned = clean_line(line)
            if cleaned:
                payload.append(cleaned)

        text_line = clean_line(" ".join(payload))
        if text_line:
            lines.append(text_line)

    return lines


def safe_extract_subtitles(zip_path):
    extracted = []
    target_dir = SUBTITLES_DIR / movie_slug(zip_path)
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as archive:
        for info in archive.infolist():
            suffix = Path(info.filename).suffix.lower()
            if suffix not in SUBTITLE_EXTENSIONS:
                continue

            target = target_dir / Path(info.filename).name
            target.write_bytes(archive.read(info))
            extracted.append(target)

    return extracted


def merge_windows(windows):
    if not windows:
        return []

    windows = sorted(windows)
    merged = [windows[0]]
    for start, end in windows[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def scene_windows(lines, before=5, after=12):
    windows = []
    for index, line in enumerate(lines):
        if JARVIS_RE.search(line):
            windows.append((max(0, index - before), min(len(lines), index + after + 1)))
    return merge_windows(windows)


def looks_like_jarvis_reply(line):
    lower = line.lower()
    return any(cue in lower for cue in JARVIS_REPLY_CUES)


def looks_like_tony_command(line):
    lower = line.lower()
    return bool(JARVIS_RE.search(line)) or any(cue in lower for cue in TONY_CUES)


def append_turn(messages, role, content):
    content = clean_line(content)
    if not content:
        return

    if messages and messages[-1]["role"] == role:
        messages[-1]["content"] = f"{messages[-1]['content']}\n{content}"
    else:
        messages.append({"role": role, "content": content})


def build_scene_messages(lines):
    messages = []
    awaiting_reply = False

    for line in lines:
        if looks_like_tony_command(line):
            append_turn(messages, "user", line)
            awaiting_reply = True
            continue

        if looks_like_jarvis_reply(line) or awaiting_reply:
            append_turn(messages, "assistant", line)
            awaiting_reply = False

    has_user = any(message["role"] == "user" for message in messages)
    has_assistant = any(message["role"] == "assistant" for message in messages)
    if has_user and has_assistant:
        return messages
    return []


def iter_interactions(zip_paths):
    for zip_path in zip_paths:
        movie = movie_slug(zip_path)
        for subtitle_path in safe_extract_subtitles(zip_path):
            lines = parse_subtitle_text(decode_subtitle(subtitle_path.read_bytes()))
            for scene_index, (start, end) in enumerate(scene_windows(lines), start=1):
                scene_lines = lines[start:end]
                messages = build_scene_messages(scene_lines)
                if not messages:
                    continue
                yield {
                    "format": "chat",
                    "source": "jarvis_personality",
                    "language": "it",
                    "messages": messages,
                    "meta": {
                        "movie": movie,
                        "subtitle_file": subtitle_path.name,
                        "scene_index": scene_index,
                        "source_type": "subtitle_interaction",
                        "review_required": True,
                    },
                }


def write_outputs(records):
    RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with RAW_OUTPUT.open("w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    with REVIEW_OUTPUT.open("w", encoding="utf-8") as f:
        for index, record in enumerate(records, start=1):
            meta = record["meta"]
            f.write(f"\n# {index} | {meta['movie']} | {meta['subtitle_file']} | scene {meta['scene_index']}\n")
            for message in record["messages"]:
                speaker = "TONY" if message["role"] == "user" else "JARVIS"
                f.write(f"{speaker}: {message['content']}\n")

    report = {
        "records": len(records),
        "raw_output": str(RAW_OUTPUT),
        "review_output": str(REVIEW_OUTPUT),
        "subtitles_dir": str(SUBTITLES_DIR),
    }
    REPORT_OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Estrae interazioni Tony/JARVIS dagli ZIP dei sottotitoli.")
    parser.add_argument(
        "--zip-glob",
        default="iron.man*.zip",
        help="Glob degli archivi da leggere dalla root del progetto.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    zip_paths = sorted(ROOT.glob(args.zip_glob))
    if not zip_paths:
        raise FileNotFoundError(f"Nessuno ZIP trovato con glob: {args.zip_glob}")

    records = list(iter_interactions(zip_paths))
    report = write_outputs(records)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
