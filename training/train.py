import yaml
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AdamW
from model.init_model import initialize_model
from training.dataset import load_training_dataset
from training.scheduler import build_scheduler
from training.utils import clip_gradients, count_parameters


def main():

    accelerator = Accelerator(mixed_precision="fp16")

    with open("config/training.yaml", "r") as f:
        train_cfg = yaml.safe_load(f)["training"]

    dataset = load_training_dataset()

    model = initialize_model()

    print(f"Total parameters: {count_parameters(model)/1e6:.2f}M")

    model.gradient_checkpointing_enable()

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["per_device_batch_size"],
        shuffle=True,
        num_workers=train_cfg["dataloader_num_workers"],
        pin_memory=train_cfg["pin_memory"]
    )

    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        betas=tuple(train_cfg["betas"]),
        eps=train_cfg["eps"],
        weight_decay=train_cfg["weight_decay"]
    )

    total_steps = len(dataloader) * train_cfg["num_train_epochs"]
    scheduler = build_scheduler(
        optimizer,
        total_steps,
        train_cfg["warmup_ratio"]
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    global_step = 0

    for epoch in range(train_cfg["num_train_epochs"]):

        model.train()

        for step, batch in enumerate(dataloader):

            outputs = model(
                input_ids=batch["input_ids"],
                labels=batch["input_ids"]
            )

            loss = outputs.loss
            accelerator.backward(loss)

            if (step + 1) % train_cfg["gradient_accumulation_steps"] == 0:
                clip_gradients(model, train_cfg["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % train_cfg["logging_steps"] == 0:
                    accelerator.print(f"Step {global_step} | Loss {loss.item():.4f}")

                if global_step % train_cfg["save_steps"] == 0:
                    accelerator.save_state("checkpoints/")

    accelerator.save_state("checkpoints/")
    accelerator.print("Training completato.")


if __name__ == "__main__":
    main()