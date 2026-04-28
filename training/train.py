import os
import math
import yaml
import time
import torch
import json
import queue
import ctypes
import shutil
import importlib.util
import sys
import threading
from datetime import datetime, timezone
from functools import partial
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import send_to_device
from torch.utils.data import DataLoader
from torch.optim import AdamW
from safetensors.torch import load_file as load_safetensors

from models.init_model import initialize_model
from training.cpu_topology import (
    detect_cpu_topology,
    select_logical_processors,
    set_process_affinity,
    set_process_priority,
    summarize_cpu_topology,
)
from training.dataset import causal_lm_collate, load_training_dataset
from training.optim import CPUAdamW
from training.scheduler import build_scheduler
from training.utils import count_parameters
from training.paths import configure_cache_env, get_path


INTERRUPT_REQUESTED = False
JSON_WRITE_FAILURES = {}
CONTROL_CACHE = {"checked_at": 0.0, "payload": None, "handled_save_request": None}


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name, default=None):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def env_int(name, default=None):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def available_system_memory_bytes():
    if os.name == "nt":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except (AttributeError, OSError):
            return None

    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages * page_size)
        except (AttributeError, OSError, ValueError):
            return None

    return None


def estimate_prefetch_batch_bytes(train_cfg):
    batch_size = max(1, int(train_cfg.get("per_device_batch_size", 1)))
    sequence_length = train_cfg.get("max_sequence_length") or 2048
    sequence_length = max(1, int(sequence_length))
    tensor_count = 3
    int64_bytes = 8
    overhead = max(1.0, float(train_cfg.get("async_prefetch_memory_overhead", 4.0)))
    return int(batch_size * sequence_length * tensor_count * int64_bytes * overhead)


def auto_async_prefetch_batches(train_cfg, cpu_runtime=None):
    max_batches = max(0, int(train_cfg.get("async_prefetch_max_batches", 8)))
    if max_batches <= 0:
        return 0

    min_batches = max(1, int(train_cfg.get("async_prefetch_min_batches", 2)))
    selected_cores = os.cpu_count() or 1
    if cpu_runtime:
        topology = cpu_runtime.get("cpu_topology") or {}
        selected_cores = int(topology.get("selected_count") or selected_cores)

    core_limited = max(1, selected_cores // 2)
    available_memory = available_system_memory_bytes()
    if available_memory:
        ram_fraction = float(train_cfg.get("async_prefetch_ram_fraction", 0.05))
        ram_budget = max(1, int(available_memory * max(0.0, ram_fraction)))
        batch_bytes = max(1, estimate_prefetch_batch_bytes(train_cfg))
        ram_limited = max(1, ram_budget // batch_bytes)
    else:
        ram_limited = max_batches

    resolved = min(max_batches, core_limited, ram_limited)
    if resolved < min_batches:
        return max(1, resolved)
    return resolved


class AsyncPrefetchLoader:
    def __init__(self, loader, buffer_size):
        self.loader = loader
        self.base_dataloader = loader
        self.buffer_size = max(1, int(buffer_size))
        self.async_prefetch_batches = self.buffer_size
        self.num_workers = getattr(loader, "num_workers", None)

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __iter__(self):
        batch_queue = queue.Queue(maxsize=self.buffer_size)
        stop_event = threading.Event()
        sentinel = object()
        errors = []

        def put_until_ready(item):
            while not stop_event.is_set():
                try:
                    batch_queue.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def producer():
            try:
                for batch in self.loader:
                    if not put_until_ready(batch):
                        break
            except BaseException as exc:
                errors.append(exc)
            finally:
                put_until_ready(sentinel)

        thread = threading.Thread(
            target=producer,
            name="jarvis-dataloader-prefetch",
            daemon=True,
        )
        thread.start()

        try:
            while True:
                item = batch_queue.get()
                if item is sentinel:
                    if errors:
                        raise errors[0]
                    break
                yield item
        finally:
            stop_event.set()
            while True:
                try:
                    batch_queue.get_nowait()
                except queue.Empty:
                    break
            thread.join(timeout=1.0)


def async_prefetch_batches(train_cfg, cpu_runtime=None):
    value = os.environ.get(
        "JARVIS_ASYNC_PREFETCH_BATCHES",
        train_cfg.get("async_prefetch_batches", 0),
    )
    normalized = str(value).strip().lower()
    if normalized in {"", "0", "false", "no", "off", "none", "disabled"}:
        return 0
    if normalized in {"auto", "dynamic", "ram", "cores"}:
        return auto_async_prefetch_batches(train_cfg, cpu_runtime)
    return max(0, int(value))


def resolve_cpu_thread_value(value, selected_count, total_logical, default=None):
    if value is None:
        return default

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "default", "none", "off"}:
            return default
        if normalized in {"auto", "selected", "policy"}:
            return max(1, selected_count)
        if normalized in {"all", "max"}:
            return max(1, total_logical)
        value = normalized

    value = int(value)
    if value <= 0:
        return default
    return value


def cpu_core_policy(train_cfg):
    return os.environ.get(
        "JARVIS_CPU_CORE_POLICY",
        train_cfg.get("cpu_core_policy", "all_cores"),
    )


def load_pipeline_step_metadata():
    step_id = os.environ.get("JARVIS_PIPELINE_STEP_ID")
    if not step_id:
        return {}

    return {
        "id": step_id,
        "order": os.environ.get("JARVIS_PIPELINE_STEP_ORDER"),
        "title": os.environ.get("JARVIS_PIPELINE_STEP_TITLE"),
        "objective": os.environ.get("JARVIS_PIPELINE_STEP_OBJECTIVE"),
        "requires": os.environ.get("JARVIS_PIPELINE_STEP_REQUIRES"),
        "data_profile": os.environ.get("JARVIS_DATA_PROFILE"),
        "allowed_formats": os.environ.get("JARVIS_ALLOWED_FORMATS"),
        "raw_include": os.environ.get("JARVIS_RAW_INCLUDE"),
        "shards_dir": os.environ.get("JARVIS_PATH_SHARDS_DIR"),
    }


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


def cuda_reserved_fraction():
    if not torch.cuda.is_available():
        return 0.0

    total = torch.cuda.get_device_properties(0).total_memory
    if total <= 0:
        return 0.0
    return torch.cuda.memory_reserved() / total


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    last_error = None
    for attempt in range(40):
        try:
            tmp_path.replace(path)
            return True
        except OSError as exc:
            is_windows_lock = isinstance(exc, PermissionError) or getattr(exc, "winerror", None) == 5
            if not is_windows_lock:
                raise
            last_error = exc
            time.sleep(min(0.05 * (attempt + 1), 1.0))

    try:
        tmp_path.unlink(missing_ok=True)
    except OSError:
        pass

    key = str(path)
    JSON_WRITE_FAILURES[key] = JSON_WRITE_FAILURES.get(key, 0) + 1
    if JSON_WRITE_FAILURES[key] <= 3 or JSON_WRITE_FAILURES[key] % 20 == 0:
        print(
            f"Warning: metriche non aggiornate per lock Windows su {path}: {last_error}",
            flush=True,
        )
    return False


def console_print(message=""):
    print(message, flush=True)


def early_startup_banner():
    console_print("")
    console_print("=" * 78)
    console_print("JARVIS TRAIN.PY BOOT")
    console_print("=" * 78)
    console_print(f"Time UTC: {utc_now_iso()}")
    console_print(f"PID: {os.getpid()} | PPID: {os.getppid() if hasattr(os, 'getppid') else '-'}")
    console_print(f"Python: {sys.executable}")
    console_print(f"Working dir: {os.getcwd()}")
    console_print("train.py entrato in main(): carico config, dataset, modello e checkpoint...")
    console_print("=" * 78)
    console_print("")


def yaml_file_payload(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as exc:
        return {"_error": str(exc)}


def make_yaml_safe(value):
    if isinstance(value, dict):
        return {str(key): make_yaml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_yaml_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def jarvis_environment_snapshot():
    prefixes = (
        "JARVIS_",
        "PYTORCH_",
        "CUDA_",
        "HF_",
        "TRANSFORMERS_",
        "OMP_",
        "MKL_",
        "NUMEXPR_",
    )
    return {
        key: os.environ[key]
        for key in sorted(os.environ)
        if key.startswith(prefixes)
    }


def resolved_paths_snapshot():
    keys = (
        "cache_dir",
        "hf_home",
        "datasets_cache_dir",
        "torch_home",
        "raw_data_dir",
        "local_data_dir",
        "cleaned_data_dir",
        "shards_dir",
        "tokenizer_dir",
        "model_output_dir",
        "final_model_dir",
        "logs_dir",
    )
    return {key: str(get_path(key, create=False)) for key in keys}


def effective_config_payload(
    train_cfg,
    pipeline_step,
    cpu_runtime=None,
    mixed_precision=None,
    attention_impl=None,
    checkpoint_dir=None,
    logs_dir=None,
    total_steps=None,
):
    return make_yaml_safe(
        {
            "config_files": {
                "training": {"training": train_cfg},
                "model": yaml_file_payload("config/model.yaml"),
                "paths": yaml_file_payload("config/paths.yaml"),
                "training_pipeline": yaml_file_payload("config/training_pipeline.yaml"),
            },
            "active_pipeline_step": pipeline_step or {},
            "resolved": {
                "mixed_precision": mixed_precision,
                "attention_implementation": attention_impl,
                "checkpoint_dir": checkpoint_dir,
                "logs_dir": logs_dir,
                "total_steps": total_steps,
                "paths": resolved_paths_snapshot(),
                "cpu_runtime": cpu_runtime or {},
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
            "environment": jarvis_environment_snapshot(),
        }
    )


def effective_config_text(
    train_cfg,
    pipeline_step,
    cpu_runtime=None,
    mixed_precision=None,
    attention_impl=None,
    checkpoint_dir=None,
    logs_dir=None,
    total_steps=None,
):
    payload = effective_config_payload(
        train_cfg=train_cfg,
        pipeline_step=pipeline_step,
        cpu_runtime=cpu_runtime,
        mixed_precision=mixed_precision,
        attention_impl=attention_impl,
        checkpoint_dir=checkpoint_dir,
        logs_dir=logs_dir,
        total_steps=total_steps,
    )
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=120)


def print_effective_config(
    train_cfg,
    pipeline_step,
    cpu_runtime=None,
    mixed_precision=None,
    attention_impl=None,
    checkpoint_dir=None,
    logs_dir=None,
    total_steps=None,
    emit_console=True,
):
    text = effective_config_text(
        train_cfg=train_cfg,
        pipeline_step=pipeline_step,
        cpu_runtime=cpu_runtime,
        mixed_precision=mixed_precision,
        attention_impl=attention_impl,
        checkpoint_dir=checkpoint_dir,
        logs_dir=logs_dir,
        total_steps=total_steps,
    )
    if emit_console:
        console_print("")
        console_print("=" * 78)
        console_print("JARVIS EFFECTIVE CONFIG")
        console_print("=" * 78)
        console_print(text.rstrip())
        console_print("=" * 78)
        console_print("")

    if logs_dir:
        try:
            config_path = logs_dir / "training_effective_config.yaml"
            config_path.write_text(text, encoding="utf-8")
        except OSError:
            pass
    return text


def init_metrics(logs_dir, train_cfg, total_steps, start_global_step, pipeline_step=None):
    metrics_path = logs_dir / "training_metrics.json"
    previous = {}
    if metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                previous = json.load(f)
        except (OSError, json.JSONDecodeError):
            previous = {}

    previous_step_id = (previous.get("pipeline_step") or {}).get("id")
    current_step_id = (pipeline_step or {}).get("id")
    same_pipeline_step = bool(current_step_id and previous_step_id == current_step_id)
    timestamp = utc_now_iso()
    history = previous.get("history", []) if same_pipeline_step else []
    epochs = previous.get("epochs", []) if same_pipeline_step else []
    if not isinstance(history, list):
        history = []
    if not isinstance(epochs, list):
        epochs = []

    current = previous.get("current", {}) if same_pipeline_step else {}
    if not isinstance(current, dict):
        current = {}
    current = {
        **current,
        "type": current.get("type", "resume"),
        "timestamp": timestamp,
        "global_step": start_global_step,
        "total_steps": total_steps,
    }

    metrics = {
        "schema_version": 1,
        "created_at": previous.get("created_at", timestamp) if same_pipeline_step else timestamp,
        "updated_at": timestamp,
        "status": "running",
        "total_steps": total_steps,
        "start_global_step": start_global_step,
        "pipeline_step": pipeline_step or previous.get("pipeline_step", {}),
        "current": current,
        "config": {
            "num_train_epochs": int(train_cfg["num_train_epochs"]),
            "per_device_batch_size": int(train_cfg["per_device_batch_size"]),
            "gradient_accumulation_steps": int(train_cfg["gradient_accumulation_steps"]),
            "learning_rate": float(train_cfg["learning_rate"]),
            "eval_steps": int(train_cfg["eval_steps"]),
            "eval_max_batches": train_cfg.get("eval_max_batches"),
            "logging_steps": int(train_cfg.get("logging_steps", 50)),
            "max_steps": train_cfg.get("max_steps"),
            "save_steps": train_cfg.get("save_steps"),
            "save_every_minutes": train_cfg.get("save_every_minutes"),
            "cpu_optimizer": cpu_optimizer_mode(train_cfg),
            "cpu_optimizer_vram_threshold": train_cfg.get("cpu_optimizer_vram_threshold"),
            "cpu_threads": train_cfg.get("cpu_threads"),
            "cpu_interop_threads": train_cfg.get("cpu_interop_threads"),
            "cpu_core_policy": train_cfg.get("cpu_core_policy"),
            "cpu_priority": train_cfg.get("cpu_priority"),
            "async_prefetch_batches": train_cfg.get("async_prefetch_batches"),
            "resolved_async_prefetch_batches": train_cfg.get("resolved_async_prefetch_batches"),
            "pipeline_step": (pipeline_step or {}).get("id"),
        },
        "history": history,
        "epochs": epochs,
    }
    write_json_atomic(metrics_path, metrics)
    return metrics_path, metrics


def record_metric(metrics_path, metrics, kind, values):
    entry = {"type": kind, "timestamp": utc_now_iso(), **values}
    metrics["updated_at"] = entry["timestamp"]
    metrics["current"] = entry
    metrics["history"].append(entry)
    write_json_atomic(metrics_path, metrics)


def record_epoch_metrics(metrics_path, metrics, values):
    entry = {"timestamp": utc_now_iso(), **values}
    metrics["updated_at"] = entry["timestamp"]
    metrics["epochs"].append(entry)
    write_json_atomic(metrics_path, metrics)


def update_metrics_status(metrics_path, metrics, status, global_step):
    metrics["updated_at"] = utc_now_iso()
    metrics["status"] = status
    metrics["current"] = {**metrics.get("current", {}), "global_step": global_step}
    write_json_atomic(metrics_path, metrics)


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
        "JARVIS_DATALOADER_TIMEOUT": "dataloader_timeout",
        "JARVIS_EVAL_MAX_BATCHES": "eval_max_batches",
        "JARVIS_MAX_STEPS": "max_steps",
        "JARVIS_MAX_SEQUENCE_LENGTH": "max_sequence_length",
        "JARVIS_SAVE_STEPS": "save_steps",
    }
    for env_name, cfg_name in overrides.items():
        value = os.environ.get(env_name)
        if value:
            train_cfg[cfg_name] = int(value)

    value = os.environ.get("JARVIS_ASYNC_PREFETCH_BATCHES")
    if value:
        train_cfg["async_prefetch_batches"] = value.strip()


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


def configure_cpu_fast_path(train_cfg):
    topology = detect_cpu_topology()
    selected_logical, applied_policy = select_logical_processors(
        topology,
        cpu_core_policy(train_cfg),
    )
    total_logical = int(topology.get("total_logical", os.cpu_count() or 1))
    affinity_applied = env_flag(
        "JARVIS_CPU_AFFINITY",
        bool(train_cfg.get("cpu_affinity", True)),
    ) and set_process_affinity(selected_logical)

    cpu_threads = resolve_cpu_thread_value(
        os.environ.get("JARVIS_CPU_THREADS", train_cfg.get("cpu_threads")),
        selected_count=len(selected_logical),
        total_logical=total_logical,
        default=None,
    )
    cpu_interop_threads = resolve_cpu_thread_value(
        os.environ.get("JARVIS_CPU_INTEROP_THREADS", train_cfg.get("cpu_interop_threads")),
        selected_count=len(selected_logical),
        total_logical=total_logical,
        default=None,
    )

    if cpu_threads and cpu_threads > 0:
        cpu_threads = min(cpu_threads, max(1, len(selected_logical)))
        torch.set_num_threads(cpu_threads)
        for env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(env_name, str(cpu_threads))

    if cpu_interop_threads and cpu_interop_threads > 0:
        try:
            torch.set_num_interop_threads(cpu_interop_threads)
        except RuntimeError:
            pass

    priority = os.environ.get("JARVIS_CPU_PRIORITY", train_cfg.get("cpu_priority", "normal"))
    priority_applied = set_process_priority(priority)
    summary = summarize_cpu_topology(
        topology,
        selected_logical,
        applied_policy,
        affinity_applied,
        priority,
    )

    return {
        "cpu_threads": torch.get_num_threads(),
        "cpu_interop_threads": torch.get_num_interop_threads(),
        "cpu_topology": summary,
        "priority_applied": priority_applied,
    }


def cpu_optimizer_mode(train_cfg):
    value = os.environ.get("JARVIS_CPU_OPTIMIZER", train_cfg.get("cpu_optimizer", False))
    if isinstance(value, bool):
        return "always" if value else "off"

    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "on", "always"}:
        return "always"
    if value in {"auto", "threshold", "limit"}:
        return "auto"
    return "off"


def cpu_optimizer_vram_threshold(train_cfg):
    return env_float(
        "JARVIS_CPU_OPTIMIZER_VRAM_THRESHOLD",
        float(train_cfg.get("cpu_optimizer_vram_threshold", 0.9)),
    )


def cpu_optimizer_check_steps(train_cfg):
    return max(
        1,
        env_int(
            "JARVIS_CPU_OPTIMIZER_CHECK_STEPS",
            int(train_cfg.get("cpu_optimizer_check_steps", 1)),
        ),
    )


def cpu_optimizer_auto_due(train_cfg, global_step):
    mode = cpu_optimizer_mode(train_cfg)
    if mode != "auto" or not torch.cuda.is_available():
        return False

    check_steps = cpu_optimizer_check_steps(train_cfg)
    if global_step > 0 and global_step % check_steps != 0:
        return False

    return cuda_reserved_fraction() >= cpu_optimizer_vram_threshold(train_cfg)


def build_adamw(model, train_cfg, optimizer_kwargs):
    if cpu_optimizer_mode(train_cfg) == "always":
        return CPUAdamW(
            model.parameters(),
            lr=float(train_cfg["learning_rate"]),
            betas=tuple(train_cfg["betas"]),
            eps=float(train_cfg["eps"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )

    return AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        betas=tuple(train_cfg["betas"]),
        eps=float(train_cfg["eps"]),
        weight_decay=float(train_cfg["weight_decay"]),
        **optimizer_kwargs,
    )


def unwrap_optimizer(optimizer):
    return getattr(optimizer, "optimizer", optimizer)


def cpu_optimizer_from_existing(raw_optimizer, train_cfg):
    param_groups = []
    for group in raw_optimizer.param_groups:
        copied = {key: value for key, value in group.items() if key != "params"}
        copied["params"] = group["params"]
        param_groups.append(copied)

    cpu_optimizer = CPUAdamW(
        param_groups,
        lr=float(train_cfg["learning_rate"]),
        betas=tuple(train_cfg["betas"]),
        eps=float(train_cfg["eps"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    cpu_optimizer.load_state_dict(raw_optimizer.state_dict())
    return cpu_optimizer


def register_accelerator_optimizer(accelerator, old_optimizer, new_optimizer):
    optimizers = getattr(accelerator, "_optimizers", None)
    if optimizers is None:
        return

    accelerator._optimizers = [
        optimizer
        for optimizer in optimizers
        if optimizer is not old_optimizer and optimizer is not new_optimizer
    ]
    accelerator._optimizers.append(new_optimizer)


def bind_scheduler_to_optimizer(scheduler, optimizer):
    raw_optimizer = unwrap_optimizer(optimizer)
    raw_scheduler = getattr(scheduler, "scheduler", scheduler)
    if hasattr(raw_scheduler, "optimizer"):
        raw_scheduler.optimizer = raw_optimizer
    if hasattr(scheduler, "optimizers"):
        scheduler.optimizers = [optimizer]


def switch_to_cpu_optimizer(accelerator, optimizer, scheduler, train_cfg):
    raw_optimizer = unwrap_optimizer(optimizer)
    cpu_optimizer = cpu_optimizer_from_existing(raw_optimizer, train_cfg)

    if hasattr(accelerator, "_optimizers"):
        accelerator._optimizers = [
            registered
            for registered in accelerator._optimizers
            if registered is not optimizer
        ]

    prepared_optimizer = accelerator.prepare_optimizer(
        cpu_optimizer,
        device_placement=False,
    )
    register_accelerator_optimizer(accelerator, optimizer, prepared_optimizer)
    bind_scheduler_to_optimizer(scheduler, prepared_optimizer)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return prepared_optimizer, scheduler


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


def clear_active_checkpoints(checkpoint_dir):
    if not checkpoint_dir.exists():
        return
    for checkpoint in checkpoint_dir.glob("step_*"):
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint, ignore_errors=True)


def get_latest_checkpoint(checkpoint_dir):
    checkpoints = get_candidate_checkpoints(checkpoint_dir)
    return checkpoints[0] if checkpoints else None


def checkpoint_step_number(checkpoint):
    try:
        return int(checkpoint.name.split("_")[1])
    except (IndexError, ValueError):
        return -1


def get_candidate_checkpoints(checkpoint_dir):
    if not checkpoint_dir.exists():
        return []

    checkpoints = [
        d for d in checkpoint_dir.iterdir()
        if d.is_dir()
        and d.name.startswith("step_")
        and checkpoint_step_number(d) >= 0
        and not (d / ".bad_checkpoint").exists()
    ]

    if not checkpoints:
        return []

    checkpoints = sorted(
        checkpoints,
        key=checkpoint_step_number,
        reverse=True,
    )

    valid_checkpoints = []
    for checkpoint in checkpoints:
        has_model = checkpoint_has_model(checkpoint)
        has_training_state = checkpoint_has_full_training_state(checkpoint)

        if has_model and has_training_state:
            valid_checkpoints.append(checkpoint)
            continue

        print(f"Checkpoint incompleto ignorato: {checkpoint}")

    return valid_checkpoints


def get_resumable_checkpoints(checkpoint_dir):
    if not checkpoint_dir.exists():
        return []

    checkpoints = [
        d for d in checkpoint_dir.iterdir()
        if d.is_dir()
        and d.name.startswith("step_")
        and checkpoint_step_number(d) >= 0
        and not (d / ".bad_checkpoint").exists()
    ]

    checkpoints = sorted(
        checkpoints,
        key=checkpoint_step_number,
        reverse=True,
    )

    return [checkpoint for checkpoint in checkpoints if checkpoint_has_model(checkpoint)]


def checkpoint_has_model(checkpoint):
    return (
        (checkpoint / "pytorch_model.bin").exists()
        or (checkpoint / "model.safetensors").exists()
    )


def checkpoint_has_full_training_state(checkpoint):
    return (
        checkpoint_has_model(checkpoint)
        and (checkpoint / "optimizer.bin").exists()
        and (checkpoint / "scheduler.bin").exists()
    )


def checkpoint_has_scheduler(checkpoint):
    return (checkpoint / "scheduler.bin").exists()


def checkpoint_meta_path(checkpoint):
    return checkpoint / "jarvis_checkpoint_meta.json"


def read_checkpoint_meta(checkpoint):
    path = checkpoint_meta_path(checkpoint)
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def checkpoint_pipeline_step_id(checkpoint):
    meta = read_checkpoint_meta(checkpoint)
    pipeline_step = meta.get("pipeline_step") if isinstance(meta, dict) else None
    if isinstance(pipeline_step, dict):
        return pipeline_step.get("id")
    return None


def write_checkpoint_meta(checkpoint, step, pipeline_step=None, checkpoint_mode="full_state"):
    meta = {
        "schema_version": 1,
        "checkpoint_step": step,
        "checkpoint_mode": checkpoint_mode,
        "saved_at": utc_now_iso(),
        "pipeline_step": pipeline_step or {},
    }
    with checkpoint_meta_path(checkpoint).open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_model_weights_from_checkpoint(model, checkpoint):
    safetensors_path = checkpoint / "model.safetensors"
    torch_path = checkpoint / "pytorch_model.bin"

    if safetensors_path.exists():
        state_dict = load_safetensors(str(safetensors_path), device="cpu")
    elif torch_path.exists():
        state_dict = torch.load(torch_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"Nessun file modello trovato in {checkpoint}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return {
        "missing": len(missing),
        "unexpected": len(unexpected),
    }


def load_scheduler_from_checkpoint(scheduler, checkpoint):
    scheduler_path = checkpoint / "scheduler.bin"
    if not scheduler_path.exists():
        return False

    state = torch.load(scheduler_path, map_location="cpu")
    scheduler.load_state_dict(state)
    return True


def save_checkpoint(accelerator, step, checkpoint_dir, max_checkpoints, pipeline_step=None):
    save_path = checkpoint_dir / f"step_{step}"
    tmp_path = checkpoint_dir / f"step_{step}.tmp"

    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)
    if save_path.exists():
        shutil.rmtree(save_path, ignore_errors=True)

    tmp_path.mkdir(parents=True, exist_ok=True)

    accelerator.save_state(str(tmp_path), safe_serialization=False)
    write_checkpoint_meta(tmp_path, step, pipeline_step=pipeline_step)
    tmp_path.replace(save_path)
    rotate_checkpoints(checkpoint_dir, max_checkpoints)

    accelerator.print(f"\nCheckpoint salvato: {save_path}\n")
    return save_path


def checkpoint_due(global_step, last_saved_step, current_time, last_save_time, train_cfg):
    if global_step <= 0 or global_step == last_saved_step:
        return False

    save_steps = train_cfg.get("save_steps")
    if save_steps is not None:
        save_steps = int(save_steps)
        if save_steps > 0 and global_step % save_steps == 0:
            return True

    save_every_minutes = env_float(
        "JARVIS_SAVE_EVERY_MINUTES",
        train_cfg.get("save_every_minutes"),
    )
    if save_every_minutes is not None and save_every_minutes > 0:
        return current_time - last_save_time >= save_every_minutes * 60

    return False


def next_checkpoint_step(global_step, train_cfg):
    save_steps = train_cfg.get("save_steps")
    if save_steps is None:
        return None

    save_steps = int(save_steps)
    if save_steps <= 0:
        return None

    return ((global_step // save_steps) + 1) * save_steps


def cuda_startup_lines():
    if not torch.cuda.is_available():
        return ["CUDA: non disponibile"]

    lines = [
        f"CUDA: torch={torch.version.cuda} | devices={torch.cuda.device_count()} | "
        f"current={torch.cuda.current_device()}",
        f"CUDA bf16 support: {torch.cuda.is_bf16_supported()}",
    ]
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        total_gb = props.total_memory / 1024**3
        lines.append(
            f"GPU {index}: {props.name} | total_vram={total_gb:.1f}GB | "
            f"sm={props.major}.{props.minor}"
        )

    allocated_gb, reserved_gb = cuda_memory_gb()
    lines.append(
        f"CUDA memory now: allocated={allocated_gb:.2f}GB | "
        f"reserved={reserved_gb:.2f}GB | reserved_ratio={cuda_reserved_fraction() * 100:.1f}%"
    )
    return lines


def dataloader_workers(loader):
    current = loader
    seen = set()
    for _ in range(5):
        if id(current) in seen:
            break
        seen.add(id(current))

        value = getattr(current, "num_workers", None)
        if value is not None:
            return value

        for name in ("base_dataloader", "dataloader", "loader"):
            nested = getattr(current, name, None)
            if nested is not None and nested is not current:
                current = nested
                break
        else:
            break

    return "?"


def dataloader_async_prefetch(loader):
    current = loader
    seen = set()
    for _ in range(5):
        if id(current) in seen:
            break
        seen.add(id(current))

        value = getattr(current, "async_prefetch_batches", None)
        if value is not None:
            return value

        for name in ("base_dataloader", "dataloader", "loader"):
            nested = getattr(current, name, None)
            if nested is not None and nested is not current:
                current = nested
                break
        else:
            break

    return 0


def startup_summary(
    accelerator,
    train_cfg,
    pipeline_step,
    dataset,
    val_dataset,
    train_loader,
    val_loader,
    model,
    optimizer,
    checkpoint_dir,
    logs_dir,
    metrics_path,
    total_steps,
    global_step,
    accumulation_steps,
    mixed_precision,
    attention_impl,
    cpu_runtime,
    cpu_optimizer_setting,
    cpu_optimizer_active,
    fused_optimizer,
    compile_model,
    restore_weights_only,
    restored_checkpoint,
    loaded_weights_checkpoint,
    max_sequence_length,
    truncation_keep,
    eval_max_batches,
    slow_step_seconds,
    status_update_seconds,
    max_checkpoints,
):
    if not accelerator.is_main_process:
        return

    raw_optimizer = unwrap_optimizer(optimizer)
    model_params = count_parameters(model)
    effective_batch = (
        int(train_cfg["per_device_batch_size"])
        * int(accumulation_steps)
        * int(accelerator.num_processes)
    )
    next_save = next_checkpoint_step(global_step, train_cfg)
    cpu_topology = cpu_runtime["cpu_topology"]
    selected_logical = ",".join(str(item) for item in cpu_topology["selected_logical"])
    data_path = get_path("shards_dir", create=False)

    lines = [
        "",
        "=" * 78,
        "JARVIS TRAINING STARTUP",
        "=" * 78,
        f"Time UTC: {utc_now_iso()}",
        f"Project root: {get_path('logs_dir').parents[0]}",
        f"Stage: {(pipeline_step or {}).get('id', 'manual')} | "
        f"{(pipeline_step or {}).get('title', 'training diretto')}",
        f"Objective: {(pipeline_step or {}).get('objective', '-')}",
        f"Data profile: {(pipeline_step or {}).get('data_profile', '-')} | "
        f"allowed_formats={(pipeline_step or {}).get('allowed_formats', '-')}",
        f"Shards dir: {data_path}",
        f"Checkpoint dir: {checkpoint_dir}",
        f"Logs dir: {logs_dir}",
        f"Metrics file: {metrics_path}",
        f"Effective config file: {logs_dir / 'training_effective_config.yaml'}",
        "-" * 78,
        "HARDWARE",
        *cuda_startup_lines(),
        f"CPU: hybrid={cpu_topology['hybrid']} | policy={cpu_topology['policy']} | "
        f"selected={cpu_topology['selected_count']}/{cpu_topology['total_logical']} | "
        f"P={cpu_topology['performance_count']} | E={cpu_topology['efficient_count']} | "
        f"parked={cpu_topology['parked_count']}",
        f"CPU affinity={cpu_topology['affinity_applied']} | priority={cpu_topology['priority']} | "
        f"priority_applied={cpu_runtime['priority_applied']}",
        f"CPU threads={cpu_runtime['cpu_threads']} | interop={cpu_runtime['cpu_interop_threads']} | "
        f"selected_logical=[{selected_logical}]",
        "-" * 78,
        "MODEL / PRECISION",
        f"Model params: {model_params / 1e6:.2f}M",
        f"Attention: {attention_impl}",
        f"Mixed precision: {mixed_precision}",
        f"Gradient checkpointing: {bool(train_cfg['gradient_checkpointing'])}",
        f"TF32: {bool(train_cfg.get('tf32', True))}",
        f"torch.compile requested: {compile_model}",
        f"CUDA fast optimizer fused: {bool(fused_optimizer)}",
        "-" * 78,
        "OPTIMIZER / SCHEDULER",
        f"Optimizer: {raw_optimizer.__class__.__name__}",
        f"CPU optimizer mode: {cpu_optimizer_setting} | active_now={cpu_optimizer_active} | "
        f"threshold={float(train_cfg.get('cpu_optimizer_vram_threshold', 0.9)) * 100:.1f}%",
        f"LR: {float(train_cfg['learning_rate']):.3e} | betas={train_cfg['betas']} | "
        f"eps={train_cfg['eps']} | weight_decay={train_cfg['weight_decay']}",
        f"Scheduler: {train_cfg.get('scheduler', 'cosine')} | warmup_ratio={train_cfg['warmup_ratio']}",
        f"Max grad norm: {train_cfg['max_grad_norm']}",
        "-" * 78,
        "DATA / BATCH",
        f"Train records: {len(dataset)} | Val records: {len(val_dataset)}",
        f"Train loader batches: {len(train_loader)} | Val loader batches: {len(val_loader)}",
        f"Micro batch per device: {train_cfg['per_device_batch_size']} | "
        f"grad_accum={accumulation_steps} | effective_batch={effective_batch}",
        f"Eval batch size: {train_cfg['eval_batch_size']}",
        f"Sequence cap: {max_sequence_length or 'off'} | truncation_keep={truncation_keep}",
        f"Dataloader workers: train={dataloader_workers(train_loader)} | "
        f"val={dataloader_workers(val_loader)} | "
        f"pin_memory={bool(train_cfg['pin_memory'])} | drop_last={bool(train_cfg.get('drop_last', True))}",
        f"Async prefetch batches: train={dataloader_async_prefetch(train_loader)} | "
        f"config={train_cfg.get('async_prefetch_batches', 0)}",
        "-" * 78,
        "RUN CONTROL",
        f"Epochs: {train_cfg['num_train_epochs']} | max_steps={train_cfg.get('max_steps')}",
        f"Global step: {global_step}/{total_steps} | remaining={max(0, total_steps - global_step)}",
        f"Save every steps: {train_cfg.get('save_steps')} | next_save={next_save} | "
        f"save_total_limit={max_checkpoints}",
        f"Save every minutes: {train_cfg.get('save_every_minutes')}",
        f"Eval steps: {train_cfg['eval_steps']} | eval_max_batches={eval_max_batches}",
        f"Status update seconds: {status_update_seconds} | slow_step_seconds={slow_step_seconds}",
        "-" * 78,
        "RESUME",
        f"restore_weights_only={restore_weights_only}",
        f"restored_state_checkpoint={restored_checkpoint or '-'}",
        f"loaded_weights_checkpoint={loaded_weights_checkpoint or '-'}",
        "-" * 78,
        "ENV",
        f"JARVIS_MIXED_PRECISION={os.environ.get('JARVIS_MIXED_PRECISION', '-')}",
        f"JARVIS_CPU_OPTIMIZER={os.environ.get('JARVIS_CPU_OPTIMIZER', '-')}",
        f"JARVIS_CPU_CORE_POLICY={os.environ.get('JARVIS_CPU_CORE_POLICY', '-')}",
        f"JARVIS_MAX_SEQUENCE_LENGTH={os.environ.get('JARVIS_MAX_SEQUENCE_LENGTH', '-')}",
        f"JARVIS_SAVE_STEPS={os.environ.get('JARVIS_SAVE_STEPS', '-')}",
        f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '-')}",
        "=" * 78,
        "",
    ]
    text = "\n".join(lines)
    console_print(text)

    try:
        startup_path = logs_dir / "training_startup.txt"
        startup_path.write_text(text + "\n", encoding="utf-8")
    except OSError:
        pass


def request_stop():
    global INTERRUPT_REQUESTED
    INTERRUPT_REQUESTED = True


def dashboard_control(logs_dir):
    now = time.time()
    if now - CONTROL_CACHE["checked_at"] < 2.0:
        return CONTROL_CACHE["payload"]

    CONTROL_CACHE["checked_at"] = now
    CONTROL_CACHE["payload"] = None
    control_path = logs_dir / "training_control.json"
    if not control_path.exists():
        return None

    try:
        with control_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    action = str(payload.get("action", "")).strip().lower()
    if action in {"save"}:
        CONTROL_CACHE["payload"] = payload
    return CONTROL_CACHE["payload"]


def complete_dashboard_save_request(logs_dir, payload, step, checkpoint_path):
    control_path = logs_dir / "training_control.json"
    completed = {
        **payload,
        "action": "save_done",
        "completed_at": utc_now_iso(),
        "checkpoint_step": step,
        "checkpoint_path": str(checkpoint_path),
    }
    try:
        write_json_atomic(control_path, completed)
    except OSError:
        pass
    CONTROL_CACHE["payload"] = completed


def main():

    early_startup_banner()
    configure_cache_env()
    with open("config/training.yaml", "r") as f:
        train_cfg = yaml.safe_load(f)["training"]
    pipeline_step = load_pipeline_step_metadata()

    override_from_env(train_cfg)
    configure_cuda_fast_path(train_cfg)
    cpu_runtime = configure_cpu_fast_path(train_cfg)
    train_cfg["resolved_async_prefetch_batches"] = async_prefetch_batches(
        train_cfg,
        cpu_runtime,
    )

    mixed_precision = resolve_mixed_precision(train_cfg["mixed_precision"])
    checkpoint_dir = get_path("model_output_dir", create=True)
    logs_dir = get_path("logs_dir", create=True)
    max_checkpoints = int(train_cfg.get("save_total_limit", 2))
    cleanup_staging_checkpoints(checkpoint_dir)

    attention_impl = resolve_attention_implementation(train_cfg)
    print_effective_config(
        train_cfg=train_cfg,
        pipeline_step=pipeline_step,
        cpu_runtime=cpu_runtime,
        mixed_precision=mixed_precision,
        attention_impl=attention_impl,
        checkpoint_dir=checkpoint_dir,
        logs_dir=logs_dir,
    )

    dataset, val_dataset = load_training_dataset(split_validation=True)

    model = initialize_model(
        device=None,
        attn_implementation=attention_impl,
    )
    if train_cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    restore_weights_only = env_flag("JARVIS_RESTORE_WEIGHTS_ONLY", False)
    loaded_weights_checkpoint = None
    if restore_weights_only:
        loaded_previous_weights = False
        for checkpoint in get_resumable_checkpoints(checkpoint_dir):
            print(f"\nCarico solo i pesi modello da {checkpoint}\n")
            try:
                result = load_model_weights_from_checkpoint(model, checkpoint)
                loaded_previous_weights = True
                loaded_weights_checkpoint = str(checkpoint)
                print(
                    "Pesi caricati, optimizer/scheduler/global_step resettati "
                    f"(missing={result['missing']}, unexpected={result['unexpected']}).\n"
                )
                break
            except Exception as exc:
                print(f"Checkpoint non caricabile come pesi modello, lo salto: {checkpoint}")
                print(f"Motivo: {exc}\n")
        else:
            print("\nNessun checkpoint precedente valido: training da zero.\n")
        if loaded_previous_weights:
            clear_active_checkpoints(checkpoint_dir)

    autotune_batch_size(model, train_cfg, mixed_precision)
    print_effective_config(
        train_cfg=train_cfg,
        pipeline_step=pipeline_step,
        cpu_runtime=cpu_runtime,
        mixed_precision=mixed_precision,
        attention_impl=attention_impl,
        checkpoint_dir=checkpoint_dir,
        logs_dir=logs_dir,
        emit_console=False,
    )

    accumulation_steps = int(train_cfg["gradient_accumulation_steps"])
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=accumulation_steps,
    )

    cpu_optimizer_setting = cpu_optimizer_mode(train_cfg)
    cpu_optimizer_active = cpu_optimizer_setting == "always"
    cpu_optimizer_auto = cpu_optimizer_setting == "auto"
    fused_optimizer = (
        env_flag(
            "JARVIS_FUSED_OPTIMIZER",
            bool(train_cfg.get("fused_optimizer", True)),
        )
        and not cpu_optimizer_active
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
    if pipeline_step:
        accelerator.print(
            f"Pipeline step: {pipeline_step.get('id')} | {pipeline_step.get('title')}\n"
        )
    if cpu_optimizer_active:
        accelerator.print(
            "\nCPU optimizer attivo: stati AdamW su RAM di sistema. "
            "Riduce VRAM ma rende optimizer.step piu lento.\n"
        )
    elif cpu_optimizer_auto:
        accelerator.print(
            "\nCPU optimizer auto: partenza su GPU, switch su CPU se VRAM reserved >= "
            f"{cpu_optimizer_vram_threshold(train_cfg) * 100:.0f}%.\n"
        )
    accelerator.print(
        "CPU runtime: "
        f"threads={cpu_runtime['cpu_threads']} | "
        f"interop={cpu_runtime['cpu_interop_threads']}\n"
    )
    cpu_topology = cpu_runtime["cpu_topology"]
    accelerator.print(
        "CPU topology: "
        f"hybrid={cpu_topology['hybrid']} | "
        f"policy={cpu_topology['policy']} | "
        f"selected={cpu_topology['selected_count']}/{cpu_topology['total_logical']} | "
        f"P={cpu_topology['performance_count']} | "
        f"E={cpu_topology['efficient_count']} | "
        f"parked={cpu_topology['parked_count']} | "
        f"affinity={cpu_topology['affinity_applied']} | "
        f"priority={cpu_topology['priority']}\n"
    )

    num_workers = int(train_cfg["dataloader_num_workers"])
    windows_safe_loader = env_flag("JARVIS_WINDOWS_SAFE_DATALOADER", os.name == "nt")
    if windows_safe_loader and os.name == "nt" and num_workers > 0:
        accelerator.print(
            "\nWindows safe dataloader attivo: uso num_workers=0 per evitare stalli "
            "multiprocessing con dataset Parquet/HF. Imposta JARVIS_WINDOWS_SAFE_DATALOADER=0 "
            "per riabilitare i worker.\n"
        )
        num_workers = 0
        train_cfg["dataloader_num_workers"] = 0

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(train_cfg["pin_memory"]),
        "drop_last": bool(train_cfg.get("drop_last", True)),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(train_cfg.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 4))
        loader_kwargs["timeout"] = int(train_cfg.get("dataloader_timeout", 0))

    max_sequence_length = train_cfg.get("max_sequence_length")
    max_sequence_length = int(max_sequence_length) if max_sequence_length else None
    truncation_keep = str(train_cfg.get("sequence_truncation_keep", "tail"))
    collate_fn = partial(
        causal_lm_collate,
        max_sequence_length=max_sequence_length,
        truncation_keep=truncation_keep,
    )
    if max_sequence_length:
        accelerator.print(
            "\nVRAM safe sequence cap attivo: "
            f"max_sequence_length={max_sequence_length}, keep={truncation_keep}.\n"
        )

    train_loader = DataLoader(
        dataset,
        batch_size=train_cfg["per_device_batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        **loader_kwargs,
    )

    val_workers = max(0, min(num_workers, 2))
    val_loader_kwargs = {
        "num_workers": val_workers,
        "pin_memory": bool(train_cfg["pin_memory"]),
    }
    if val_workers > 0:
        val_loader_kwargs["timeout"] = int(train_cfg.get("dataloader_timeout", 0))

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["eval_batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        **val_loader_kwargs,
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
    print_effective_config(
        train_cfg=train_cfg,
        pipeline_step=pipeline_step,
        cpu_runtime=cpu_runtime,
        mixed_precision=mixed_precision,
        attention_impl=attention_impl,
        checkpoint_dir=checkpoint_dir,
        logs_dir=logs_dir,
        total_steps=total_steps,
        emit_console=False,
    )

    scheduler = build_scheduler(
        optimizer,
        total_steps,
        train_cfg["warmup_ratio"]
    )

    prefetch_batches = int(train_cfg.get("resolved_async_prefetch_batches", 0))
    manual_device_placement = prefetch_batches > 0
    if manual_device_placement:
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
        train_loader, val_loader = accelerator.prepare(
            train_loader,
            val_loader,
            device_placement=[False, False],
        )
        train_loader = AsyncPrefetchLoader(train_loader, prefetch_batches)
        accelerator.print(
            "\nAsync dataloader prefetch attivo: "
            f"{prefetch_batches} batch in coda "
            f"(config={train_cfg.get('async_prefetch_batches')}), "
            "device move nel thread principale.\n"
        )
    else:
        model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler
        )

    global_step = 0
    restored_checkpoint = None

    if restore_weights_only:
        accelerator.print("\nNuovo stage: non ripristino optimizer, scheduler o global_step.\n")
    else:
        current_step_id = pipeline_step.get("id") if pipeline_step else None
        for checkpoint in get_resumable_checkpoints(checkpoint_dir):
            checkpoint_step_id = checkpoint_pipeline_step_id(checkpoint)
            if current_step_id and checkpoint_step_id and checkpoint_step_id != current_step_id:
                accelerator.print(
                    f"\nCheckpoint di un altro stage ignorato: {checkpoint} "
                    f"({checkpoint_step_id} != {current_step_id})\n"
                )
                continue

            if checkpoint_has_full_training_state(checkpoint):
                accelerator.print(f"\nRipristino stato completo da {checkpoint}\n")
                try:
                    accelerator.load_state(str(checkpoint))
                    global_step = checkpoint_step_number(checkpoint)
                    restored_checkpoint = str(checkpoint)
                    break
                except Exception as exc:
                    accelerator.print(f"Checkpoint non caricabile, lo salto: {checkpoint}")
                    accelerator.print(f"Motivo: {exc}\n")
                    if accelerator.is_main_process:
                        marker = checkpoint / ".bad_checkpoint"
                        marker.write_text(str(exc), encoding="utf-8")
                    continue

            accelerator.print(
                f"\nCheckpoint senza optimizer.bin: ripristino pesi modello da {checkpoint} "
                "e continuo con optimizer nuovo.\n"
            )
            try:
                result = load_model_weights_from_checkpoint(accelerator.unwrap_model(model), checkpoint)
                scheduler_loaded = load_scheduler_from_checkpoint(scheduler, checkpoint)
                global_step = checkpoint_step_number(checkpoint)
                restored_checkpoint = (
                    f"{checkpoint} (pesi"
                    f"{' + scheduler' if scheduler_loaded else ''}; optimizer reset)"
                )
                accelerator.print(
                    "Resume parziale riuscito "
                    f"(missing={result['missing']}, unexpected={result['unexpected']}, "
                    f"scheduler_loaded={scheduler_loaded}).\n"
                )
                break
            except Exception as exc:
                accelerator.print(f"Checkpoint non caricabile come resume parziale, lo salto: {checkpoint}")
                accelerator.print(f"Motivo: {exc}\n")
                if accelerator.is_main_process:
                    marker = checkpoint / ".bad_checkpoint"
                    marker.write_text(str(exc), encoding="utf-8")
        else:
            accelerator.print("\nNessun checkpoint valido trovato: training da zero.\n")

    last_save_time = time.time()
    last_saved_step = global_step
    status_start_time = last_save_time
    last_status_time = last_save_time
    last_status_tokens = 0
    last_status_micro_steps = 0
    session_tokens = 0
    session_micro_steps = 0
    accumulation_tokens = 0
    session_start_global_step = global_step
    last_loss = None
    last_ppl = None
    last_lr = scheduler.get_last_lr()[0]
    status_update_seconds = float(
        os.environ.get("JARVIS_STATUS_SECONDS", train_cfg.get("status_update_seconds", 5))
    )
    max_steps = train_cfg.get("max_steps")
    max_steps = int(max_steps) if max_steps is not None else None
    logging_steps = max(1, int(train_cfg.get("logging_steps", 50)))
    eval_max_batches = train_cfg.get("eval_max_batches")
    eval_max_batches = int(eval_max_batches) if eval_max_batches is not None else None
    slow_step_seconds = float(train_cfg.get("slow_step_seconds", 60))
    slow_step_seconds = env_float("JARVIS_SLOW_STEP_SECONDS", slow_step_seconds)

    metrics_path = None
    metrics = None
    if accelerator.is_main_process:
        metrics_path, metrics = init_metrics(
            logs_dir,
            train_cfg,
            total_steps,
            global_step,
            pipeline_step=pipeline_step,
        )
        accelerator.print(f"\nMetriche training: {metrics_path}\n")
    metrics_final_status = "completed"

    startup_summary(
        accelerator=accelerator,
        train_cfg=train_cfg,
        pipeline_step=pipeline_step,
        dataset=dataset,
        val_dataset=val_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        checkpoint_dir=checkpoint_dir,
        logs_dir=logs_dir,
        metrics_path=metrics_path,
        total_steps=total_steps,
        global_step=global_step,
        accumulation_steps=accumulation_steps,
        mixed_precision=mixed_precision,
        attention_impl=attention_impl,
        cpu_runtime=cpu_runtime,
        cpu_optimizer_setting=cpu_optimizer_setting,
        cpu_optimizer_active=cpu_optimizer_active,
        fused_optimizer=fused_optimizer,
        compile_model=compile_model,
        restore_weights_only=restore_weights_only,
        restored_checkpoint=restored_checkpoint,
        loaded_weights_checkpoint=loaded_weights_checkpoint,
        max_sequence_length=max_sequence_length,
        truncation_keep=truncation_keep,
        eval_max_batches=eval_max_batches,
        slow_step_seconds=slow_step_seconds,
        status_update_seconds=status_update_seconds,
        max_checkpoints=max_checkpoints,
    )

    try:

        for epoch in range(train_cfg["num_train_epochs"]):

            model.train()
            epoch_start_time = time.time()
            epoch_start_step = global_step
            epoch_loss_total = 0.0
            epoch_loss_count = 0
            progress = tqdm(
                train_loader,
                disable=not accelerator.is_local_main_process,
                dynamic_ncols=True
            )

            last_batch_end_time = time.time()
            for step, batch in enumerate(progress):
                batch_ready_time = time.time()
                fetch_seconds = batch_ready_time - last_batch_end_time
                if manual_device_placement:
                    batch = send_to_device(
                        batch,
                        accelerator.device,
                        non_blocking=bool(train_cfg["pin_memory"]),
                    )
                compute_start_time = time.time()

                if INTERRUPT_REQUESTED:
                    accelerator.print("\nStop richiesto: salvo checkpoint e chiudo.\n")
                    metrics_final_status = "interrupted"
                    raise KeyboardInterrupt

                with accelerator.accumulate(model):
                    with accelerator.autocast():

                        outputs = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch.get("attention_mask"),
                            labels=batch.get("labels", batch["input_ids"]),
                        )

                        loss = outputs.loss

                    accelerator.backward(loss)
                    compute_seconds = time.time() - compute_start_time
                    batch_tokens = (
                        batch["attention_mask"].sum().item()
                        if "attention_mask" in batch
                        else batch["input_ids"].numel()
                    )
                    session_tokens += batch_tokens
                    accumulation_tokens += batch_tokens
                    session_micro_steps += 1
                    current_time = time.time()

                    if slow_step_seconds and (
                        fetch_seconds >= slow_step_seconds
                        or compute_seconds >= slow_step_seconds
                    ):
                        seq_len = int(batch["input_ids"].shape[-1])
                        batch_size = int(batch["input_ids"].shape[0])
                        accelerator.print(
                            "\nBatch lento rilevato | "
                            f"GS {global_step} | micro {step} | "
                            f"fetch {fetch_seconds:.1f}s | compute {compute_seconds:.1f}s | "
                            f"batch {batch_size} | seq {seq_len} | tokens {batch_tokens}\n"
                        )

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

                        if (
                            cpu_optimizer_auto
                            and not cpu_optimizer_active
                            and cpu_optimizer_auto_due(train_cfg, global_step)
                        ):
                            reserved_fraction = cuda_reserved_fraction()
                            accelerator.print(
                                "\nSoglia VRAM raggiunta: passo a CPU optimizer | "
                                f"reserved {reserved_fraction * 100:.1f}% >= "
                                f"{cpu_optimizer_vram_threshold(train_cfg) * 100:.1f}%\n"
                            )
                            optimizer, scheduler = switch_to_cpu_optimizer(
                                accelerator,
                                optimizer,
                                scheduler,
                                train_cfg,
                            )
                            cpu_optimizer_active = True

                        optimizer_started = time.time()
                        accelerator.clip_grad_norm_(
                            model.parameters(),
                            train_cfg["max_grad_norm"]
                        )

                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        optimizer_seconds = time.time() - optimizer_started

                        if slow_step_seconds and optimizer_seconds >= slow_step_seconds:
                            accelerator.print(
                                "\nOptimizer lento rilevato | "
                                f"GS {global_step} | optimizer {optimizer_seconds:.1f}s | "
                                f"cpu_optimizer={cpu_optimizer_active}\n"
                            )

                        global_step += 1

                        avg_loss = loss.item()
                        perplexity = math.exp(min(avg_loss, 20))
                        current_lr = scheduler.get_last_lr()[0]
                        last_lr = current_lr
                        epoch_loss_total += avg_loss
                        epoch_loss_count += 1

                        tokens = accumulation_tokens
                        accumulation_tokens = 0
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

                        if accelerator.is_main_process:
                            record_metric(
                                metrics_path,
                                metrics,
                                "train",
                                {
                                    "epoch": epoch + 1,
                                    "global_step": global_step,
                                    "total_steps": total_steps,
                                    "loss": avg_loss,
                                    "perplexity": perplexity,
                                    "learning_rate": current_lr,
                                    "tokens": tokens,
                                    "vram_allocated_gb": allocated_gb,
                                    "vram_reserved_gb": reserved_gb,
                                    "optimizer_step_seconds": optimizer_seconds,
                                    "cpu_optimizer_active": cpu_optimizer_active,
                                },
                            )

                        # CHECKPOINT
                        current_time = time.time()
                        control_payload = dashboard_control(logs_dir)
                        save_request_id = None
                        if control_payload and control_payload.get("action") == "save":
                            save_request_id = str(
                                control_payload.get("request_id")
                                or control_payload.get("requested_at")
                                or "manual"
                            )
                        save_requested = (
                            save_request_id is not None
                            and save_request_id != CONTROL_CACHE.get("handled_save_request")
                        )
                        due_checkpoint = checkpoint_due(
                            global_step,
                            last_saved_step,
                            current_time,
                            last_save_time,
                            train_cfg,
                        )
                        if save_requested or due_checkpoint:
                            if accelerator.is_main_process:
                                checkpoint_started = time.time()
                                saved_path = save_checkpoint(
                                    accelerator,
                                    global_step,
                                    checkpoint_dir,
                                    max_checkpoints,
                                    pipeline_step=pipeline_step,
                                )
                                checkpoint_seconds = time.time() - checkpoint_started
                                accelerator.print(
                                    f"Checkpoint step {global_step} completato in "
                                    f"{format_duration(checkpoint_seconds)}.\n"
                                )
                                if save_requested:
                                    complete_dashboard_save_request(
                                        logs_dir,
                                        control_payload,
                                        global_step,
                                        saved_path,
                                    )
                                    accelerator.print(
                                        f"Checkpoint manuale dashboard salvato: {saved_path}\n"
                                    )
                            if save_requested:
                                CONTROL_CACHE["handled_save_request"] = save_request_id
                            last_save_time = time.time()
                            last_saved_step = global_step

                        # VALIDATION
                        if global_step % train_cfg["eval_steps"] == 0:

                            model.eval()
                            eval_loss = 0
                            eval_steps = 0
                            eval_started = time.time()

                            with torch.no_grad():
                                for val_batch in val_loader:
                                    if eval_max_batches and eval_steps >= eval_max_batches:
                                        break
                                    if manual_device_placement:
                                        val_batch = send_to_device(
                                            val_batch,
                                            accelerator.device,
                                            non_blocking=bool(train_cfg["pin_memory"]),
                                        )
                                    with accelerator.autocast():
                                        val_out = model(
                                            input_ids=val_batch["input_ids"],
                                            attention_mask=val_batch.get("attention_mask"),
                                            labels=val_batch.get("labels", val_batch["input_ids"]),
                                        )
                                    eval_loss += val_out.loss.item()
                                    eval_steps += 1

                            eval_loss /= max(eval_steps, 1)
                            eval_ppl = math.exp(min(eval_loss, 20))
                            eval_seconds = time.time() - eval_started
                            eval_limit = ""
                            if eval_max_batches:
                                eval_limit = f" | batches {eval_steps}/{eval_max_batches}"

                            accelerator.print(
                                f"\nValidation | "
                                f"Loss {eval_loss:.4f} | "
                                f"PPL {eval_ppl:.2f} | "
                                f"Durata {format_duration(eval_seconds)}"
                                f"{eval_limit}\n"
                            )

                            if accelerator.is_main_process:
                                record_metric(
                                    metrics_path,
                                    metrics,
                                    "eval",
                                    {
                                        "epoch": epoch + 1,
                                        "global_step": global_step,
                                        "total_steps": total_steps,
                                        "loss": eval_loss,
                                        "perplexity": eval_ppl,
                                        "learning_rate": current_lr,
                                        "eval_batches": eval_steps,
                                        "duration_seconds": eval_seconds,
                                    },
                                )

                            model.train()

                        if max_steps and global_step >= max_steps:
                            if accelerator.is_main_process:
                                avg_epoch_loss = epoch_loss_total / max(epoch_loss_count, 1)
                                record_epoch_metrics(
                                    metrics_path,
                                    metrics,
                                    {
                                        "epoch": epoch + 1,
                                        "complete": False,
                                        "start_global_step": epoch_start_step,
                                        "end_global_step": global_step,
                                        "duration_seconds": time.time() - epoch_start_time,
                                        "train_loss": avg_epoch_loss,
                                        "train_perplexity": math.exp(min(avg_epoch_loss, 20)),
                                    },
                                )
                                metrics_final_status = "stopped_at_max_steps"
                            return

                last_batch_end_time = time.time()

            if accelerator.is_main_process:
                avg_epoch_loss = epoch_loss_total / max(epoch_loss_count, 1)
                record_epoch_metrics(
                    metrics_path,
                    metrics,
                    {
                        "epoch": epoch + 1,
                        "complete": True,
                        "start_global_step": epoch_start_step,
                        "end_global_step": global_step,
                        "duration_seconds": time.time() - epoch_start_time,
                        "train_loss": avg_epoch_loss,
                        "train_perplexity": math.exp(min(avg_epoch_loss, 20)),
                    },
                )

    except KeyboardInterrupt:
        request_stop()
        metrics_final_status = "interrupted"
        accelerator.print("\nInterruzione ricevuta. Salvo checkpoint prima di uscire...\n")

    except Exception:
        metrics_final_status = "failed"
        raise

    finally:
        if accelerator.is_main_process:
            if metrics_path and metrics:
                update_metrics_status(metrics_path, metrics, metrics_final_status, global_step)
            try:
                save_checkpoint(
                    accelerator,
                    global_step,
                    checkpoint_dir,
                    max_checkpoints,
                    pipeline_step=pipeline_step,
                )
            except KeyboardInterrupt:
                accelerator.print("\nUscita forzata durante il salvataggio. Checkpoint temporaneo ignorato.\n")
                cleanup_staging_checkpoints(checkpoint_dir)

    if metrics_final_status == "interrupted":
        accelerator.print("\nTraining interrotto dopo salvataggio checkpoint. Riavvia lo stesso step per riprendere.\n")
        sys.exit(130)

    accelerator.print("\nTraining completato.")


if __name__ == "__main__":
    main()
