import os
import math
import yaml
import time
import torch
import shutil
import importlib.util
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
INTERRUPT_REQUESTED = False


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def format_duration(seconds):
    if seconds is None or seconds <= 0 or math.isinf(seconds):
        return "--"

    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def cuda_memory_gb():
    if not torch.cuda.is_available():
        return 0.0, 0.0

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    return allocated, reserved


def resolve_mixed_precision(config_value):
    value = os.environ.get("JARVIS_MIXED_PRECISION", config_value)
    if value == "bf16_if_available":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return "bf16"
        return "fp16"
    return value


def override_from_env(train_cfg):
    overrides = {
        "JARVIS_BATCH_SIZE": "per_device_batch_size",
        "JARVIS_GRAD_ACCUM": "gradient_accumulation_steps",
        "JARVIS_DATALOADER_WORKERS": "dataloader_num_workers",
        "JARVIS_MAX_STEPS": "max_steps",
    }
    for env_name, cfg_name in overrides.items():
        value = os.environ.get(env_name)
        if value:
            train_cfg[cfg_name] = int(value)


def configure_cuda_fast_path(train_cfg):
    if not torch.cuda.is_available():
        return

    use_tf32 = env_flag("JARVIS_TF32", bool(train_cfg.get("tf32", True)))
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    torch.backends.cudnn.benchmark = True

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    cuda_backend = getattr(torch.backends, "cuda", None)
    if cuda_backend:
        for name in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_math_sdp"):
            fn = getattr(cuda_backend, name, None)
            if fn:
                fn(True)


def build_adamw(model, train_cfg, optimizer_kwargs):
    return AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        betas=tuple(train_cfg["betas"]),
        eps=float(train_cfg["eps"]),
        weight_decay=float(train_cfg["weight_decay"]),
        **optimizer_kwargs,
    )


def get_autocast_dtype(mixed_precision):
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    return None


def triton_available():
    return importlib.util.find_spec("triton") is not None


def flash_attention_available():
    return importlib.util.find_spec("flash_attn") is not None


def resolve_attention_implementation(train_cfg):
    requested = os.environ.get(
        "JARVIS_ATTENTION_IMPL",
        train_cfg.get("attention_implementation", "sdpa"),
    )

    if requested == "auto":
        if env_flag("JARVIS_USE_FLASH_ATTENTION", False) and flash_attention_available():
            return "flash_attention_2"
        return "sdpa"

    if requested == "flash_attention_2" and not flash_attention_available():
        if env_flag("JARVIS_REQUIRE_FLASH_ATTENTION", False):
            raise RuntimeError("FlashAttention richiesta ma flash_attn non e installato/importabile.")
        print("FlashAttention non disponibile: fallback a SDPA.")
        return "sdpa"

    return requested


def autotune_batch_size(model, train_cfg, mixed_precision):
    if not env_flag("JARVIS_AUTO_BATCH", bool(train_cfg.get("auto_batch_size", False))):
        return
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    model.to(device)
    model.train()

    start_batch = int(train_cfg["per_device_batch_size"])
    max_batch = int(os.environ.get("JARVIS_MAX_BATCH_SIZE", train_cfg.get("max_auto_batch_size", 8)))
    target_vram = float(os.environ.get("JARVIS_TARGET_VRAM", train_cfg.get("target_vram_fraction", 0.92)))
    sequence_length = int(os.environ.get("JARVIS_TUNE_SEQUENCE_LENGTH", 2048))
    repeats = int(os.environ.get("JARVIS_TUNE_REPEATS", train_cfg.get("auto_batch_repeats", 2)))
    vocab_size = int(getattr(model.config, "vocab_size", 32000))
    dtype = get_autocast_dtype(mixed_precision)

    original_effective_batch = start_batch * int(train_cfg["gradient_accumulation_steps"])
    best_batch = start_batch
    best_tokens_per_second = 0.0
    candidate = start_batch

    print("")
    print("CUDA autotune batch attivo")

    while candidate <= max_batch:
        input_ids = None
        output = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model.zero_grad(set_to_none=True)

            input_ids = torch.randint(
                low=0,
                high=vocab_size,
                size=(candidate, sequence_length),
                dtype=torch.long,
                device=device,
            )

            # Warmup non cronometrato: evita di scegliere batch grandi solo per effetto cache/JIT.
            if dtype is None:
                output = model(input_ids=input_ids, labels=input_ids)
            else:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    output = model(input_ids=input_ids, labels=input_ids)
            output.loss.backward()
            torch.cuda.synchronize()
            model.zero_grad(set_to_none=True)
            del output

            elapsed = 0.0
            for _ in range(repeats):
                started_at = time.time()
                if dtype is None:
                    output = model(input_ids=input_ids, labels=input_ids)
                else:
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        output = model(input_ids=input_ids, labels=input_ids)

                output.loss.backward()
                torch.cuda.synchronize()
                elapsed += time.time() - started_at
                model.zero_grad(set_to_none=True)
                del output

            peak = torch.cuda.max_memory_reserved() / torch.cuda.get_device_properties(0).total_memory
            tokens_per_second = (candidate * sequence_length * repeats) / max(elapsed, 1e-6)
            print(
                f"  batch {candidate}: ok, "
                f"{tokens_per_second:.0f} tok/s, peak VRAM {peak * 100:.1f}%"
            )

            if tokens_per_second > best_tokens_per_second:
                best_tokens_per_second = tokens_per_second
                best_batch = candidate

            del input_ids
            torch.cuda.empty_cache()

            if peak >= target_vram:
                break

            candidate *= 2
        except torch.cuda.OutOfMemoryError:
            print(f"  batch {candidate}: OOM, uso batch {best_batch}")
            model.zero_grad(set_to_none=True)
            if input_ids is not None:
                del input_ids
            if output is not None:
                del output
            torch.cuda.empty_cache()
            break

    if best_batch != start_batch:
        new_accumulation = max(1, math.ceil(original_effective_batch / best_batch))
        train_cfg["per_device_batch_size"] = best_batch
        train_cfg["gradient_accumulation_steps"] = new_accumulation
        train_cfg["eval_batch_size"] = max(int(train_cfg.get("eval_batch_size", 1)), best_batch)
        print(
            "CUDA autotune scelto: "
            f"micro_batch={best_batch}, grad_accum={new_accumulation}, "
            f"stima={best_tokens_per_second:.0f} tok/s"
        )
    else:
        print("CUDA autotune: tengo il batch configurato")


def rotate_checkpoints(checkpoint_dir, max_checkpoints):
    checkpoints = sorted(
        [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda x: int(x.name.split("_")[1])
    )

    while len(checkpoints) > max_checkpoints:
        old = checkpoints.pop(0)
        shutil.rmtree(old)


def cleanup_staging_checkpoints(checkpoint_dir):
    if not checkpoint_dir.exists():
        return
    for staging in checkpoint_dir.glob("step_*.tmp"):
        if staging.is_dir():
            shutil.rmtree(staging, ignore_errors=True)


def get_latest_checkpoint(checkpoint_dir):
    checkpoints = get_candidate_checkpoints(checkpoint_dir)
    return checkpoints[0] if checkpoints else None


def get_candidate_checkpoints(checkpoint_dir):
    if not checkpoint_dir.exists():
        return []

    checkpoints = [
        d for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.startswith("step_") and not (d / ".bad_checkpoint").exists()
    ]

    if not checkpoints:
        return []

    checkpoints = sorted(
        checkpoints,
        key=lambda x: int(x.name.split("_")[1]),
        reverse=True,
    )

    valid_checkpoints = []
    for checkpoint in checkpoints:
        has_model = (
            (checkpoint / "pytorch_model.bin").exists()
            or (checkpoint / "model.safetensors").exists()
        )
        has_training_state = (
            (checkpoint / "optimizer.bin").exists()
            and (checkpoint / "scheduler.bin").exists()
        )

        if has_model and has_training_state:
            valid_checkpoints.append(checkpoint)
            continue

        print(f"Checkpoint incompleto ignorato: {checkpoint}")

    return valid_checkpoints


def save_checkpoint(accelerator, step, checkpoint_dir, max_checkpoints):
    save_path = checkpoint_dir / f"step_{step}"
    tmp_path = checkpoint_dir / f"step_{step}.tmp"

    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)
    if save_path.exists():
        shutil.rmtree(save_path, ignore_errors=True)

    tmp_path.mkdir(parents=True, exist_ok=True)

    accelerator.save_state(str(tmp_path), safe_serialization=False)
    tmp_path.replace(save_path)
    rotate_checkpoints(checkpoint_dir, max_checkpoints)

    accelerator.print(f"\nCheckpoint salvato: {save_path}\n")


def request_stop():
    global INTERRUPT_REQUESTED
    INTERRUPT_REQUESTED = True


def main():

    configure_cache_env()
    with open("config/training.yaml", "r") as f:
        train_cfg = yaml.safe_load(f)["training"]

    override_from_env(train_cfg)
    configure_cuda_fast_path(train_cfg)

    mixed_precision = resolve_mixed_precision(train_cfg["mixed_precision"])
    checkpoint_dir = get_path("model_output_dir", create=True)
    max_checkpoints = int(train_cfg.get("save_total_limit", 2))
    cleanup_staging_checkpoints(checkpoint_dir)

    dataset, val_dataset = load_training_dataset(split_validation=True)
    attention_impl = resolve_attention_implementation(train_cfg)

    model = initialize_model(
        device=None,
        attn_implementation=attention_impl,
    )
    if train_cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    autotune_batch_size(model, train_cfg, mixed_precision)

    accumulation_steps = int(train_cfg["gradient_accumulation_steps"])
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=accumulation_steps,
    )

    fused_optimizer = env_flag(
        "JARVIS_FUSED_OPTIMIZER",
        bool(train_cfg.get("fused_optimizer", True)),
    )
    if fused_optimizer and torch.cuda.is_available():
        model.to(accelerator.device)

    compile_model = env_flag("JARVIS_TORCH_COMPILE", bool(train_cfg.get("torch_compile", False)))
    if compile_model and hasattr(torch, "compile"):
        if not triton_available() and not env_flag("JARVIS_FORCE_COMPILE", False):
            accelerator.print(
                "\ntorch.compile disattivato: Triton non disponibile su questo ambiente. "
                "CUDA/BF16/TF32/fused optimizer restano attivi.\n"
            )
        else:
            try:
                torch_dynamo = importlib.import_module("torch._dynamo")

                torch_dynamo.config.suppress_errors = True
                accelerator.print("\ntorch.compile attivo (first step piu lento, poi piu veloce).\n")
                model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            except Exception as exc:
                accelerator.print(f"\ntorch.compile disattivato: {exc}\n")

    accelerator.print(
        f"\nModel: {count_parameters(model)/1e6:.2f}M parameters | "
        f"attention={attention_impl}\n"
    )

    num_workers = int(train_cfg["dataloader_num_workers"])
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(train_cfg["pin_memory"]),
        "drop_last": bool(train_cfg.get("drop_last", True)),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(train_cfg.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 4))

    train_loader = DataLoader(
        dataset,
        batch_size=train_cfg["per_device_batch_size"],
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["eval_batch_size"],
        shuffle=False,
        num_workers=max(0, min(num_workers, 2)),
        pin_memory=bool(train_cfg["pin_memory"]),
    )

    optimizer_kwargs = {}
    if fused_optimizer and torch.cuda.is_available():
        optimizer_kwargs["fused"] = True

    try:
        optimizer = build_adamw(model, train_cfg, optimizer_kwargs)
    except (TypeError, RuntimeError) as exc:
        optimizer_kwargs.pop("fused", None)
        accelerator.print(f"\nFused AdamW disattivato: {exc}\n")
        optimizer = build_adamw(model, train_cfg, optimizer_kwargs)

    total_steps = math.ceil(len(train_loader) / accumulation_steps) * train_cfg["num_train_epochs"]
    if train_cfg.get("max_steps") is not None:
        total_steps = min(total_steps, int(train_cfg["max_steps"]))

    scheduler = build_scheduler(
        optimizer,
        total_steps,
        train_cfg["warmup_ratio"]
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    global_step = 0

    for checkpoint in get_candidate_checkpoints(checkpoint_dir):
        accelerator.print(f"\nRipristino da {checkpoint}\n")
        try:
            accelerator.load_state(str(checkpoint))
            global_step = int(checkpoint.name.split("_")[-1])
            break
        except Exception as exc:
            accelerator.print(f"Checkpoint non caricabile, lo salto: {checkpoint}")
            accelerator.print(f"Motivo: {exc}\n")
            if accelerator.is_main_process:
                marker = checkpoint / ".bad_checkpoint"
                marker.write_text(str(exc), encoding="utf-8")
    else:
        accelerator.print("\nNessun checkpoint valido trovato: training da zero.\n")

    last_save_time = time.time()
    status_start_time = last_save_time
    last_status_time = last_save_time
    last_status_tokens = 0
    last_status_micro_steps = 0
    session_tokens = 0
    session_micro_steps = 0
    session_start_global_step = global_step
    last_loss = None
    last_ppl = None
    last_lr = scheduler.get_last_lr()[0]
    status_update_seconds = float(
        os.environ.get("JARVIS_STATUS_SECONDS", train_cfg.get("status_update_seconds", 5))
    )
    max_steps = train_cfg.get("max_steps")
    max_steps = int(max_steps) if max_steps is not None else None

    try:

        for epoch in range(train_cfg["num_train_epochs"]):

            model.train()
            progress = tqdm(
                train_loader,
                disable=not accelerator.is_local_main_process,
                dynamic_ncols=True
            )

            for step, batch in enumerate(progress):
                if INTERRUPT_REQUESTED:
                    accelerator.print("\nStop richiesto: salvo checkpoint e chiudo.\n")
                    return

                with accelerator.accumulate(model):
                    with accelerator.autocast():

                        outputs = model(
                            input_ids=batch["input_ids"],
                            labels=batch["input_ids"]
                        )

                        loss = outputs.loss

                    accelerator.backward(loss)
                    batch_tokens = batch["input_ids"].numel()
                    session_tokens += batch_tokens
                    session_micro_steps += 1
                    current_time = time.time()

                    if (
                        current_time - last_status_time >= status_update_seconds
                        or accelerator.sync_gradients
                    ):
                        last_loss = loss.detach().float().item()
                        last_ppl = math.exp(min(last_loss, 20))
                        elapsed_window = max(current_time - last_status_time, 1e-6)
                        total_elapsed = max(current_time - status_start_time, 1e-6)
                        window_tokens = session_tokens - last_status_tokens
                        window_micro_steps = session_micro_steps - last_status_micro_steps
                        tokens_per_second = window_tokens / elapsed_window
                        micro_steps_per_second = window_micro_steps / elapsed_window
                        completed_global_steps = max(0, global_step - session_start_global_step)
                        global_steps_per_second = completed_global_steps / total_elapsed
                        eta_seconds = (
                            (total_steps - global_step) / global_steps_per_second
                            if global_steps_per_second > 0 else None
                        )
                        allocated_gb, reserved_gb = cuda_memory_gb()

                        progress.set_description(
                            f"E{epoch+1} | "
                            f"GS {global_step}/{total_steps} | "
                            f"Loss {last_loss:.4f} | "
                            f"PPL {last_ppl:.1f} | "
                            f"LR {last_lr:.2e} | "
                            f"{tokens_per_second:.0f} tok/s | "
                            f"{micro_steps_per_second:.2f} it/s | "
                            f"ETA {format_duration(eta_seconds)} | "
                            f"VRAM {allocated_gb:.1f}/{reserved_gb:.1f}GB",
                            refresh=True,
                        )

                        last_status_time = current_time
                        last_status_tokens = session_tokens
                        last_status_micro_steps = session_micro_steps

                    if accelerator.sync_gradients:

                        accelerator.clip_grad_norm_(
                            model.parameters(),
                            train_cfg["max_grad_norm"]
                        )

                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

                        global_step += 1

                        avg_loss = loss.item()
                        perplexity = math.exp(min(avg_loss, 20))
                        current_lr = scheduler.get_last_lr()[0]
                        last_lr = current_lr

                        tokens = batch["input_ids"].numel() * accumulation_steps
                        allocated_gb, reserved_gb = cuda_memory_gb()

                        progress.set_description(
                            f"E{epoch+1} | "
                            f"GS {global_step}/{total_steps} | "
                            f"Loss {avg_loss:.4f} | "
                            f"PPL {perplexity:.1f} | "
                            f"LR {current_lr:.2e} | "
                            f"Tok {tokens} | "
                            f"VRAM {allocated_gb:.1f}/{reserved_gb:.1f}GB",
                            refresh=True,
                        )

                        # SAVE A TEMPO
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

                        # VALIDATION
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
                                f"\nValidation | "
                                f"Loss {eval_loss:.4f} | "
                                f"PPL {eval_ppl:.2f}\n"
                            )

                            model.train()

                        if max_steps and global_step >= max_steps:
                            return

    except KeyboardInterrupt:
        request_stop()
        accelerator.print("\nInterruzione ricevuta. Salvo checkpoint prima di uscire...\n")

    finally:
        if accelerator.is_main_process:
            try:
                save_checkpoint(
                    accelerator,
                    global_step,
                    checkpoint_dir,
                    max_checkpoints
                )
            except KeyboardInterrupt:
                accelerator.print("\nUscita forzata durante il salvataggio. Checkpoint temporaneo ignorato.\n")
                cleanup_staging_checkpoints(checkpoint_dir)

    accelerator.print("\nTraining completato.")


if __name__ == "__main__":
    main()
