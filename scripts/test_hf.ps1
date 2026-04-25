Write-Host "========================================"
Write-Host "   HUGGING FACE CONNECTIVITY TEST"
Write-Host "========================================"

. "$PSScriptRoot\lib\jarvis.ps1"

$Root = Get-JarvisRoot
Initialize-JarvisEnvironment -Root $Root
$Python = Get-JarvisPython -Root $Root
$Hf = Get-JarvisHf -Root $Root

if ($Python -eq "python") {
    Write-Host "Python venv non trovato. Esegui prima scripts\setup_env.ps1"
    exit 1
}

if (Test-Path $Hf) {
    & $Hf auth whoami
}

@'
import requests
import socket
import time

url = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.it/train-00000-of-00010.parquet"
print("first_addr", socket.getaddrinfo("huggingface.co", 443)[0][4])

start = time.time()
size = 0
with requests.get(url, stream=True, timeout=60) as response:
    print("status", response.status_code)
    response.raise_for_status()
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        if not chunk:
            continue
        size += len(chunk)
        if size >= 16 * 1024 * 1024:
            break

print(f"download_test ok bytes={size} seconds={time.time() - start:.2f}")
'@ | & $Python -

exit $LASTEXITCODE
