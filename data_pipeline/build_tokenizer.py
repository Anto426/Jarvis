import os
import yaml
import sentencepiece as spm

with open("config/paths.yaml", "r") as f:
    paths = yaml.safe_load(f)["paths"]

DEDUP_DIR = os.path.join(paths["cleaned_data_dir"], "deduplicated")
TOKENIZER_DIR = paths["tokenizer_dir"]
os.makedirs(TOKENIZER_DIR, exist_ok=True)

VOCAB_SIZE = 32000

def collect_files():
    files = [os.path.join(DEDUP_DIR, f) for f in os.listdir(DEDUP_DIR) if f.endswith(".jsonl")]
    return files

def main():

    input_files = ",".join(collect_files())

    spm.SentencePieceTrainer.train(
        input=input_files,
        model_prefix=os.path.join(TOKENIZER_DIR, "jarvis"),
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