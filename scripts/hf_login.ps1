Write-Host "========================================"
Write-Host "   HUGGING FACE LOGIN"
Write-Host "========================================"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Cache = Join-Path $Root ".cache"
$Hf = Join-Path $Root ".venv\Scripts\hf.exe"

$env:HF_HOME=(Join-Path $Cache "huggingface")
$env:HF_DATASETS_CACHE=(Join-Path $Cache "huggingface\datasets")
$env:TRANSFORMERS_CACHE=(Join-Path $Cache "huggingface\transformers")
$env:HF_HUB_DISABLE_XET="1"

New-Item -ItemType Directory -Force -Path $env:HF_HOME, $env:HF_DATASETS_CACHE, $env:TRANSFORMERS_CACHE | Out-Null

if (-not (Test-Path $Hf)) {
    Write-Host "hf.exe non trovato. Esegui prima: powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1"
    exit 1
}

& $Hf auth login
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "========================================"
Write-Host "   LOGIN FINISHED"
Write-Host "========================================"
