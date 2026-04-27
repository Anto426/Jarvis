import os
import json
import sentencepiece as spm
from training.paths import get_path
from data_pipeline.sample_format import SPECIAL_TOKENS, normalize_sample, render_training_text

DEDUP_DIR = os.path.join(get_path("cleaned_data_dir", create=True), "deduplicated")
TOKENIZER_DIR = get_path("tokenizer_dir", create=True)
DEDUP_INCLUDE = [
    name.strip()
    for name in os.environ.get("JARVIS_DEDUP_INCLUDE", "").split(";")
    if name.strip()
]

VOCAB_SIZE = 32000

def collect_files():
    files = [f for f in os.listdir(DEDUP_DIR) if f.endswith(".jsonl")]
    if DEDUP_INCLUDE:
        files = [f for f in files if f in DEDUP_INCLUDE]
    files = [os.path.join(DEDUP_DIR, f) for f in files]
    return files


def build_tokenizer_corpus(files):
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    corpus_path = os.path.join(str(TOKENIZER_DIR), "tokenizer_corpus.txt")
    written = 0

    with open(corpus_path, "w", encoding="utf-8") as out:
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        sample = normalize_sample(json.loads(line))
                    except (json.JSONDecodeError, TypeError):
                        continue

                    text = render_training_text(sample)
                    if not text:
                        continue

                    out.write(text.replace("\n", " "))
                    out.write("\n")
                    written += 1

    if written == 0:
        raise RuntimeError("Corpus tokenizer vuoto dopo la normalizzazione strutturata.")

    return corpus_path

def main():

    files = collect_files()
    if not files:
        raise FileNotFoundError(f"Nessun file deduplicato trovato in {DEDUP_DIR}")

    corpus_path = build_tokenizer_corpus(files)

    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=os.path.join(str(TOKENIZER_DIR), "jarvis"),
        vocab_size=VOCAB_SIZE,
        model_type="unigram",
        character_coverage=0.9995,
        user_defined_symbols=SPECIAL_TOKENS,
        hard_vocab_limit=False,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=3
    )

    print("Tokenizer creato.")

if __name__ == "__main__":
    main()
