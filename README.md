<p align="center">
  <h1 align="center">Jarvis</h1>
</p>

<p align="center">
  <img src="./asset/divider.gif" width="440" height="40" />
</p>

## <img src="./asset/icon.gif" width="44px" /> About

```sh
jarvis@garage:~$ whoami
Italian LLM training lab

jarvis@garage:~$ echo "focus"
Automotive reasoning, diagnostics, OBD/DTC knowledge, and workshop-style answers

jarvis@garage:~$ echo "base"
GPT-NeoX style causal language model trained from local and public data
```

Jarvis is a local training pipeline for building an Italian language model with a strong automotive direction.
The project is designed to keep heavy data, caches, tokenizer files, and checkpoints on this repository disk instead of filling the main Windows drive.

<p align="center">
  <img src="./asset/divider.gif" width="440" height="40" />
</p>

## <img src="./asset/icon2.gif" width="48px" /> Highlights

- Python 3.11 virtual environment in `.venv/`
- CUDA training path verified with Torch on NVIDIA GPUs
- Hugging Face, datasets, Torch, and pip caches under `.cache/`
- Local data import from `data/local/`
- Wikimedia direct fallback when Hugging Face downloads are blocked
- SentencePiece tokenizer build
- Parquet shard packing for language-model pretraining
- Accelerate-based BF16/FP16 training with checkpoint auto-resume
- Smart launcher that skips data regeneration when packed shards are already valid

<p align="center">
  <img src="./asset/divider.gif" width="440" height="40" />
</p>

## Toolchain

Recommended setup:

- Windows PowerShell
- Python: `3.11.x`
- GPU: NVIDIA CUDA-capable card
- PyTorch: CUDA build installed by `scripts/setup_env.ps1`
- Main dependencies: `transformers`, `datasets`, `accelerate`, `sentencepiece`, `pyarrow`

The setup script creates the virtual environment and installs the pinned dependency set from `requirements.txt`.

<p align="center">
  <img src="./asset/divider.gif" width="440" height="40" />
</p>

## <img src="./asset/icon3.gif" width="48px" /> Quick Start

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
```

Run Jarvis. The launcher checks/configures the Python environment, builds the dataset only when needed, otherwise it resumes/trains directly:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\jarvis.ps1
```

Update and check dependencies before running:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\jarvis.ps1 -UpdateDeps
```

Use CUDA turbo mode with `torch.compile`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\jarvis.ps1 -Turbo
```

Use max performance mode. This enables CUDA compile plus automatic micro-batch tuning against available VRAM:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\jarvis.ps1 -MaxPerf
```

Start from zero and rebuild generated data:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\jarvis.ps1 -FromScratch -RebuildData
```

<p align="center">
  <img src="./asset/divider.gif" width="440" height="40" />
</p>

## Data Pipeline

The full pipeline runs these stages:

```sh
data/local -> data/raw -> data/cleaned -> data/cleaned/deduplicated
          -> data/tokenizer -> data/shards -> checkpoints
```

Local files can be placed in:

```text
data/local/
```

Supported local formats:

- `.txt`: one document per file
- `.jsonl`: one JSON object per line with a `text` field

The smart launcher uses Hugging Face in `auto` mode by default: it runs a quick download preflight and skips Hub sources if the network resets.
Hugging Face is used for modern data-only sources such as FineWeb2 Italian and StackExchange; Wikimedia direct remains the default Wikipedia source.
To force direct/local-only sources:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\jarvis.ps1 -HuggingFace 0
```

The launcher stores a data fingerprint next to packed shards. If `data/local/`, `data_pipeline/`, or data path config changes, Jarvis rebuilds the generated dataset only when no checkpoint is being resumed or when you explicitly run from scratch. Existing checkpoints keep using the existing packed dataset until you choose:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\jarvis.ps1 -FromScratch -RebuildData
```

<p align="center">
  <img src="./asset/divider.gif" width="440" height="40" />
</p>

## Model

The default model configuration lives in `config/model.yaml`.

Current target:

- Architecture: GPT-NeoX causal LM
- Vocabulary: `32000`
- Context: `2048`
- Hidden size: `1536`
- Layers: `24`
- Attention heads: `16`
- Approximate scale: `700M`

Training behavior is configured in `config/training.yaml`.

Key defaults:

- Precision: BF16 when supported, otherwise FP16
- Batch size per device: `1`
- Gradient accumulation: `16`
- Gradient checkpointing: enabled
- CUDA fast path: TF32, SDPA attention, fused optimizer
- Max performance mode: `torch.compile` and automatic VRAM-based batch tuning
- Checkpoint limit: `3`

<p align="center">
  <img src="./asset/divider.gif" width="440" height="40" />
</p>

## Cache And Storage

Heavy paths are configured in `config/paths.yaml`.

```text
.cache/
data/
checkpoints/
model/
logs/
```

This keeps downloads, datasets, Torch cache, tokenizer files, shards, and checkpoints inside the project disk.

The generated directories are intentionally ignored by Git.

<p align="center">
  <img src="./asset/divider.gif" width="440" height="40" />
</p>

## Project Layout

- `config/`: paths, model, and training configuration
- `data_pipeline/`: collection, import, cleaning, deduplication, tokenizer build, tokenization, packing
- `models/`: GPT-NeoX configuration and model initialization
- `training/`: dataset loading, scheduler, training loop, checkpoint helpers
- `scripts/`: smart launcher, setup, checks, and Hugging Face login helpers
- `asset/`: README visual assets

<p align="center">
  <img src="./asset/divider.gif" width="440" height="40" />
</p>

## <img src="./asset/icon4.gif" width="48px" /> Automotive Direction

The first objective is to make the full training path complete reliably.
After that, Jarvis should move from generic Italian text toward high-value automotive data:

- OBD and DTC code explanations
- ECU, sensors, actuators, and diagnostic flows
- Maintenance procedures
- Fault symptoms, likely causes, and validation tests
- Workshop notes and structured Q/A examples

The strongest version of Jarvis will likely come from a general Italian base plus a focused automotive fine-tuning pass.

<p align="center">
  <img src="./asset/divider.gif" width="440" height="40" />
</p>

## Useful Checks

Verify the environment:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Check dependency consistency:

```powershell
.\.venv\Scripts\python.exe -m pip check
```

Check Hugging Face connectivity:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\test_hf.ps1
```

Compile Python files:

```powershell
.\.venv\Scripts\python.exe -m compileall training models data_pipeline scripts
```

<p align="center">
  <img src="./asset/divider.gif" width="440" height="40" />
</p>

## Notes

If Hugging Face fails with `WinError 10054`, it is usually a network reset during Hub downloads, not a Python error.
Use `-HuggingFace 0` to skip Hub sources and use direct/local sources.
Use `-HuggingFace force` only when the network is known to handle Hub downloads reliably.

On some Windows networks Python may prefer IPv6 for Hugging Face while IPv4 works correctly.
Jarvis sets `JARVIS_FORCE_IPV4=1` in the PowerShell entrypoints, and `sitecustomize.py` forces Python networking to IPv4 for those runs.
