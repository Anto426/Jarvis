import os
import math
import yaml
import time
import torch
import shutil
from tqdm import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import AdamW

from models.init_model import initialize_model
from training.dataset import load_training_dataset
from training.scheduler import build_scheduler
from training.utils import count_parameters
from training.paths import configure_cache_env, get_path


SAVE_EVERY_MINUTES = 5
SAVE_INTERVAL = SAVE_EVERY_MINUTES * 60


def rotate_checkpoints(checkpoint_dir, max_checkpoints):
    checkpoints = sorted(
        [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda x: int(x.name.split("_")[1])
    )

    while len(checkpoints) > max_checkpoints:
        old = checkpoints.pop(0)
        shutil.rmtree(old)


def get_latest_checkpoint(checkpoint_dir):
    if not checkpoint_dir.exists():
        return None

    checkpoints = [
        d for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.startswith("step_")
    ]

    if not checkpoints:
        return None

    checkpoints = sorted(
        checkpoints,
        key=lambda x: int(x.name.split("_")[1])
    )

    return checkpoints[-1]


def save_checkpoint(accelerator, step, checkpoint_dir, max_checkpoints):
    save_path = checkpoint_dir / f"step_{step}"
    save_path.mkdir(parents=True, exist_ok=True)

    accelerator.save_state(str(save_path))
    rotate_checkpoints(checkpoint_dir, max_checkpoints)

    accelerator.print(f"\n💾 Checkpoint salvato: {save_path}\n")


def main():

    configure_cache_env()
    with open("config/training.yaml", "r") as f:
        train_cfg = yaml.safe_load(f)["training"]

    accelerator = Accelerator(mixed_precision=train_cfg["mixed_precision"])
    checkpoint_dir = get_path("model_output_dir", create=True)
    max_checkpoints = int(train_cfg.get("save_total_limit", 2))

    dataset, val_dataset = load_training_dataset(split_validation=True)

    model = initialize_model(device=None)
    if train_cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    accelerator.print(f"\n🚀 Model: {count_parameters(model)/1e6:.2f}M parameters\n")

    train_loader = DataLoader(
        dataset,
        batch_size=train_cfg["per_device_batch_size"],
        shuffle=True,
        num_workers=train_cfg["dataloader_num_workers"],
        pin_memory=train_cfg["pin_memory"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["eval_batch_size"],
        shuffle=False
    )

    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        betas=tuple(train_cfg["betas"]),
        eps=float(train_cfg["eps"]),
        weight_decay=float(train_cfg["weight_decay"])
    )

    total_steps = (
        len(train_loader) //
        train_cfg["gradient_accumulation_steps"]
    ) * train_cfg["num_train_epochs"]

    scheduler = build_scheduler(
        optimizer,
        total_steps,
        train_cfg["warmup_ratio"]
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # 🔥 AUTO RESUME
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    global_step = 0

    if latest_checkpoint:
        accelerator.print(f"\n🔄 Ripristino da {latest_checkpoint}\n")
        accelerator.load_state(str(latest_checkpoint))
        global_step = int(latest_checkpoint.name.split("_")[-1])

    accumulation_steps = train_cfg["gradient_accumulation_steps"]
    last_save_time = time.time()

    try:

        for epoch in range(train_cfg["num_train_epochs"]):

            model.train()
            progress = tqdm(
                train_loader,
                disable=not accelerator.is_local_main_process,
                dynamic_ncols=True
            )

            for step, batch in enumerate(progress):

                with accelerator.autocast():

                    outputs = model(
                        input_ids=batch["input_ids"],
                        labels=batch["input_ids"]
                    )

                    loss = outputs.loss / accumulation_steps

                accelerator.backward(loss)

                if (step + 1) % accumulation_steps == 0:

                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        train_cfg["max_grad_norm"]
                    )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    global_step += 1

                    avg_loss = loss.item() * accumulation_steps
                    perplexity = math.exp(min(avg_loss, 20))
                    current_lr = scheduler.get_last_lr()[0]

                    tokens = batch["input_ids"].numel()
                    gpu_mem = (
                        torch.cuda.memory_allocated() / 1024**3
                        if torch.cuda.is_available() else 0
                    )

                    progress.set_description(
                        f"E{epoch+1} | "
                        f"GS {global_step}/{total_steps} | "
                        f"Loss {avg_loss:.4f} | "
                        f"PPL {perplexity:.1f} | "
                        f"LR {current_lr:.2e} | "
                        f"VRAM {gpu_mem:.1f}GB"
                    )

                    # 🔥 SAVE A TEMPO
                    current_time = time.time()
                    if current_time - last_save_time >= SAVE_INTERVAL:
                        if accelerator.is_main_process:
                            save_checkpoint(
                                accelerator,
                                global_step,
                                checkpoint_dir,
                                max_checkpoints
                            )
                        last_save_time = current_time

                    # 🔥 VALIDATION
                    if global_step % train_cfg["eval_steps"] == 0:

                        model.eval()
                        eval_loss = 0
                        eval_steps = 0

                        with torch.no_grad():
                            for val_batch in val_loader:
                                with accelerator.autocast():
                                    val_out = model(
                                        input_ids=val_batch["input_ids"],
                                        labels=val_batch["input_ids"]
                                    )
                                eval_loss += val_out.loss.item()
                                eval_steps += 1

                        eval_loss /= eval_steps
                        eval_ppl = math.exp(min(eval_loss, 20))

                        accelerator.print(
                            f"\n📊 Validation | "
                            f"Loss {eval_loss:.4f} | "
                            f"PPL {eval_ppl:.2f}\n"
                        )

                        model.train()

    finally:
        if accelerator.is_main_process:
            save_checkpoint(
                accelerator,
                global_step,
                checkpoint_dir,
                max_checkpoints
            )

    accelerator.print("\n🔥 Training completato.")


if __name__ == "__main__":
    main()
