import json
from training.paths import get_path
from data_pipeline.sample_format import normalize_sample, render_training_text


LOCAL_DIR = get_path("local_data_dir", create=True)
RAW_DIR = get_path("raw_data_dir", create=True)
OUTPUT_FILE = RAW_DIR / "local_import.jsonl"
MIN_TEXT_LENGTH = 200
MIN_STRUCTURED_TEXT_LENGTH = 40


def iter_local_texts():
    for path in sorted(LOCAL_DIR.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if len(text) >= MIN_TEXT_LENGTH:
                yield {
                    "format": "text",
                    "source": "local_import",
                    "language": "it",
                    "text": text,
                }

        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        sample = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    sample = normalize_sample(
                        sample,
                        source=sample.get("source", "local_import"),
                        language=sample.get("language", "it"),
                    )
                    text = render_training_text(sample)
                    min_length = (
                        MIN_TEXT_LENGTH
                        if sample.get("format") == "text"
                        else MIN_STRUCTURED_TEXT_LENGTH
                    )
                    if len(text) >= min_length:
                        if not sample.get("text"):
                            sample["text"] = text
                        yield sample


def main():
    count = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for sample in iter_local_texts():
            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")
            count += 1

    if count == 0:
        print(f"Nessun file locale importato da {LOCAL_DIR}")
        return

    print(f"Import locale completato: {count} record in {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
