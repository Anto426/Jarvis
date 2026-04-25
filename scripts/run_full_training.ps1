Write-Host "========================================"
Write-Host "   JARVIS DATA PIPELINE + TRAINING"
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
$env:JARVIS_USE_HUGGINGFACE="0"

New-Item -ItemType Directory -Force -Path $env:HF_HOME, $env:HF_DATASETS_CACHE, $env:TRANSFORMERS_CACHE, $env:TORCH_HOME | Out-Null
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Accelerate = Join-Path $Root ".venv\Scripts\accelerate.exe"
if (-not (Test-Path $Python)) { $Python = "python" }
if (-not (Test-Path $Accelerate)) { $Accelerate = "accelerate" }

Push-Location $Root

& $Python data_pipeline/import_local.py
if ($LASTEXITCODE -ne 0) { Pop-Location; exit $LASTEXITCODE }

& $Python data_pipeline/collect.py
if ($LASTEXITCODE -ne 0) { Pop-Location; exit $LASTEXITCODE }

& $Python data_pipeline/clean.py
if ($LASTEXITCODE -ne 0) { Pop-Location; exit $LASTEXITCODE }

& $Python data_pipeline/deduplicate.py
if ($LASTEXITCODE -ne 0) { Pop-Location; exit $LASTEXITCODE }

& $Python data_pipeline/build_tokenizer.py
if ($LASTEXITCODE -ne 0) { Pop-Location; exit $LASTEXITCODE }

& $Python data_pipeline/tokenize_shards.py
if ($LASTEXITCODE -ne 0) { Pop-Location; exit $LASTEXITCODE }

& $Python data_pipeline/pack_blocks.py
if ($LASTEXITCODE -ne 0) { Pop-Location; exit $LASTEXITCODE }

& $Accelerate launch --mixed_precision=fp16 training/train.py
$ExitCode = $LASTEXITCODE

Pop-Location

Write-Host "========================================"
Write-Host "   JARVIS RUN FINISHED"
Write-Host "========================================"

exit $ExitCode
