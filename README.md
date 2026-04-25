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
- Accelerate-based fp16 training with checkpoint auto-resume

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

Run the full data pipeline and training:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_full_training.ps1
```

Run only pretraining when packed shards already exist:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_pretraining.ps1
```

Resume training from the latest checkpoint:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\resume_training.ps1
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

When Hugging Face is unreliable on the network, the pipeline uses Wikimedia direct downloads by default. Hugging Face can be re-enabled by setting:

```powershell
$env:JARVIS_USE_HUGGINGFACE="1"
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

- Precision: `fp16`
- Batch size per device: `1`
- Gradient accumulation: `16`
- Gradient checkpointing: enabled
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
- `scripts/`: PowerShell entrypoints for setup, full training, pretraining, resume, and Hugging Face login
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

Compile Python files:

```powershell
.\.venv\Scripts\python.exe -m compileall training models data_pipeline scripts
```

<p align="center">
  <img src="./asset/divider.gif" width="440" height="40" />
</p>

## Notes

If Hugging Face fails with `WinError 10054`, it is usually a network reset during Hub downloads, not a Python error.
The default full-training script disables Hugging Face and uses direct/local sources so the pipeline can continue.

