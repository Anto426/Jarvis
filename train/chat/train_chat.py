import os
import torch
import sqlite3
import numpy as np
from datasets import Dataset
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

TOKENS_DB = os.path.join(BASE_DIR, "data", "tokens","raw", "chat_tokens.db")
MODEL_DIR = os.path.join(BASE_DIR, "jarvis_model")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def db_generator():
    with sqlite3.connect(TOKENS_DB) as conn:
        cur = conn.cursor()
        cur.execute("SELECT input_ids FROM token_blocks")
        for row in cur:
            ids = np.frombuffer(row[0], dtype=np.uint16).astype(np.int64)
            yield {"input_ids": ids, "labels": ids}

def main():

    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(DEVICE)
    model.config.use_cache = False

    dataset = Dataset.from_generator(db_generator)
    dataset.set_format("torch")

    args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        logging_steps=50,
        save_steps=1000,
        save_total_limit=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset
    )

    trainer.train()
    trainer.save_model(MODEL_DIR)

    print("Fine-tuning chat completato.")

if __name__ == "__main__":
    main()