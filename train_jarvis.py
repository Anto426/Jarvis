import os
import torch
from datasets import load_dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    GPT2TokenizerFast
)
from tokenizers import ByteLevelBPETokenizer

from datasets import load_dataset, Dataset
import sqlite3

# ==========================
# CONFIG
# ==========================

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dataset.db")

TOKENIZER_DIR = "tokenizer"
OUTPUT_DIR = "./jarvis_model"

if not os.path.exists(DB_PATH):
    raise FileNotFoundError(f"Database {DB_PATH} non trovato. Esegui prima jarvis_pipeline.py!")

VOCAB_SIZE = 32000
BLOCK_SIZE = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if not os.path.exists(TOKENIZER_DIR):
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    print("Training tokenizer (ByteLevelBPE) su SQLite dataset.db...\n")

    def db_text_iterator():
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cur = conn.cursor()
        cur.execute("SELECT text FROM samples")
        while True:
            rows = cur.fetchmany(10000)
            if not rows:
                break
            yield [r[0] for r in rows]
        conn.close()

    tokenizer_trainer = ByteLevelBPETokenizer()
    tokenizer_trainer.train_from_iterator(
        db_text_iterator(),
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"]
    )
    tokenizer_trainer.save_model(TOKENIZER_DIR)

print("Loading tokenizer...")

tokenizer = GPT2TokenizerFast(
    vocab_file=os.path.join(TOKENIZER_DIR, "vocab.json"),
    merges_file=os.path.join(TOKENIZER_DIR, "merges.txt"),
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>"
)

# ==========================
# DATASET
# ==========================

print("Loading dataset da SQLite...")

def hf_dataset_generator():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("SELECT text FROM samples")
    for row in cur:
        yield {"text": row[0]}
    conn.close()

dataset = Dataset.from_generator(hf_dataset_generator)

if len(dataset) == 0:
    raise ValueError("Dataset vuoto")

dataset = dataset.train_test_split(test_size=0.02)

def tokenize_function(examples):
    return tokenizer(examples["text"], return_attention_mask=False)

print(f"Tokenizing multiprocessing usando {max(os.cpu_count() - 1, 1)} cores...")
tokenized = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    num_proc=max(os.cpu_count() - 1, 1)
)

def group_texts(examples):
    concatenated = sum(examples["input_ids"], [])
    total_length = (len(concatenated) // BLOCK_SIZE) * BLOCK_SIZE

    input_ids = [
        concatenated[i:i + BLOCK_SIZE]
        for i in range(0, total_length, BLOCK_SIZE)
    ]

    return {
        "input_ids": input_ids,
        "labels": input_ids.copy()
    }

print("Grouping sequences for training (LM packaging)...")
lm_dataset = tokenized.map(
    group_texts,
    batched=True,
    num_proc=max(os.cpu_count() - 1, 1)
)

lm_dataset.set_format(type="torch")

# ==========================
# MODEL
# ==========================

print("Initializing model...")

config = GPT2Config(
    vocab_size=VOCAB_SIZE,
    n_positions=BLOCK_SIZE,
    n_ctx=BLOCK_SIZE,
    n_embd=1024,
    n_layer=16,
    n_head=16,
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1
)

model = GPT2LMHeadModel(config).to(DEVICE)

if torch.__version__.startswith("2"):
    model = torch.compile(model)

# ==========================
# TRAINING SETTINGS
# ==========================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=4,
    per_device_train_batch_size=4,   # Adattabile, dipende dalla GPU
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    eval_strategy="steps",           # eval_strategy standard in Transformers >=4.40
    eval_steps=1000,
    save_steps=1000,
    save_total_limit=2,
    logging_steps=50,
    warmup_ratio=0.03,
    learning_rate=2e-4,
    weight_decay=0.1,
    lr_scheduler_type="cosine",
    bf16=True,                       # Tensor Core accelerati FP16
    gradient_checkpointing=True,     # Fondamentale per contesti grandi su schede standard
    dataloader_num_workers=max(os.cpu_count() - 1, 1), # Core massimi in I/O
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator
)

# ==========================
# TRAIN
# ==========================

print("Starting training...")
trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training completato.")