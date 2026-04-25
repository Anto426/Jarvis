import os
import sentencepiece as spm
from training.paths import get_path

DEDUP_DIR = os.path.join(get_path("cleaned_data_dir", create=True), "deduplicated")
TOKENIZER_DIR = get_path("tokenizer_dir", create=True)

VOCAB_SIZE = 32000

def collect_files():
    files = [os.path.join(DEDUP_DIR, f) for f in os.listdir(DEDUP_DIR) if f.endswith(".jsonl")]
    return files

def main():

    files = collect_files()
    if not files:
        raise FileNotFoundError(f"Nessun file deduplicato trovato in {DEDUP_DIR}")

    input_files = ",".join(files)

    spm.SentencePieceTrainer.train(
        input=input_files,
        model_prefix=os.path.join(str(TOKENIZER_DIR), "jarvis"),
        vocab_size=VOCAB_SIZE,
        model_type="unigram",
        character_coverage=0.9995,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=3
    )

    print("Tokenizer creato.")

if __name__ == "__main__":
    main()
