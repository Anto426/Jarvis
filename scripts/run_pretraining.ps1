Write-Host "========================================"
Write-Host "   JARVIS 700M PRETRAINING START"
Write-Host "========================================"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Cache = Join-Path $Root ".cache"

$env:CUDA_VISIBLE_DEVICES="0"
$env:HF_HOME=(Join-Path $Cache "huggingface")
$env:HF_DATASETS_CACHE=(Join-Path $Cache "huggingface\datasets")
$env:TRANSFORMERS_CACHE=(Join-Path $Cache "huggingface\transformers")
$env:TORCH_HOME=(Join-Path $Cache "torch")
$env:XDG_CACHE_HOME=$Cache
$env:PYTHONPATH=$Root
$env:HF_HUB_DISABLE_XET="1"

New-Item -ItemType Directory -Force -Path $env:HF_HOME, $env:HF_DATASETS_CACHE, $env:TRANSFORMERS_CACHE, $env:TORCH_HOME | Out-Null
$Accelerate = Join-Path $Root ".venv\Scripts\accelerate.exe"
if (-not (Test-Path $Accelerate)) { $Accelerate = "accelerate" }

Push-Location $Root
& $Accelerate launch --mixed_precision=fp16 training/train.py
Pop-Location

Write-Host "========================================"
Write-Host "   PRETRAINING FINISHED"
Write-Host "========================================"
