import os
import torch
import sqlite3
import numpy as np
import multiprocessing
from datasets import Dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments
)

# ==========================
# CONFIG
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TOKENS_DB = os.path.join(BASE_DIR, "data", "tokens.db")
PERSONA_TOKENS_DB = os.path.join(BASE_DIR, "data", "persona_tokens.db")

OUTPUT_DIR = os.path.join(BASE_DIR, "jarvis_model")

VOCAB_SIZE = 32000
BLOCK_SIZE = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ==========================
# DATASET GENERATOR
# ==========================

def db_generator(db_path):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT input_ids FROM token_blocks")

        for row in cur:
            ids = np.frombuffer(row[0], dtype=np.uint16).astype(np.int64)
            yield {
                "input_ids": ids,
                "labels": ids
            }

# ==========================
# TRAIN STAGE FUNCTION
# ==========================

def train_stage(model, db_path, epochs, lr):

    print(f"\nLoading dataset from {db_path}...")

    dataset = Dataset.from_generator(lambda: db_generator(db_path))

    if len(dataset) == 0:
        raise ValueError(f"{db_path} vuoto.")

    dataset = dataset.train_test_split(test_size=0.02, seed=42)
    dataset.set_format("torch")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=50,
        warmup_steps=100,
        learning_rate=lr,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        dataloader_num_workers=0,  # Windows stabile
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"]
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)

    return model

# ==========================
# MAIN
# ==========================

def main():

    if not os.path.exists(TOKENS_DB):
        raise FileNotFoundError("tokens.db non trovato.")

    print("Initializing base model...")

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

    # ======================
    # STAGE 1 — BASE TRAIN
    # ======================

    print("\n=== STAGE 1: BASE PRETRAINING ===")
    model = train_stage(
        model,
        TOKENS_DB,
        epochs=4,
        lr=2e-4
    )

    # ======================
    # STAGE 2 — PERSONA
    # ======================

    if os.path.exists(PERSONA_TOKENS_DB):

        print("\n=== STAGE 2: PERSONA FINE-TUNING ===")

        model = train_stage(
            model,
            PERSONA_TOKENS_DB,
            epochs=2,
            lr=5e-5  # LR più basso per non distruggere il linguaggio
        )

    print("\nTraining completato.")
    print(f"Modello salvato in: {OUTPUT_DIR}")

# ==========================
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()