Write-Host "========================================"
Write-Host "   HUGGING FACE LOGIN"
Write-Host "========================================"

. "$PSScriptRoot\lib\jarvis.ps1"

$Root = Get-JarvisRoot
Initialize-JarvisEnvironment -Root $Root
$Hf = Get-JarvisHf -Root $Root

if (-not (Test-Path $Hf)) {
    Write-Host "hf.exe non trovato. Esegui prima: powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1"
    exit 1
}

& $Hf auth login
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "========================================"
Write-Host "   LOGIN FINISHED"
Write-Host "========================================"
