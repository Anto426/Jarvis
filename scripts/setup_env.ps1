Write-Host "========================================"
Write-Host "   JARVIS ENV SETUP"
Write-Host "========================================"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Venv = Join-Path $Root ".venv"
$Cache = Join-Path $Root ".cache"

$env:PIP_CACHE_DIR=(Join-Path $Cache "pip")
$env:HF_HOME=(Join-Path $Cache "huggingface")
$env:HF_DATASETS_CACHE=(Join-Path $Cache "huggingface\datasets")
$env:TRANSFORMERS_CACHE=(Join-Path $Cache "huggingface\transformers")
$env:TORCH_HOME=(Join-Path $Cache "torch")
$env:XDG_CACHE_HOME=$Cache
$env:HF_HUB_DISABLE_XET="1"

New-Item -ItemType Directory -Force -Path $env:PIP_CACHE_DIR, $env:HF_HOME, $env:HF_DATASETS_CACHE, $env:TRANSFORMERS_CACHE, $env:TORCH_HOME | Out-Null

function Find-CompatiblePython {
    foreach ($Version in @("3.12", "3.11", "3.10")) {
        $Arg = "-$Version"
        & py $Arg --version *> $null
        if ($LASTEXITCODE -eq 0) {
            return $Arg
        }
    }

    return $null
}

$PythonArg = Find-CompatiblePython

if (-not $PythonArg) {
    Write-Host "Python 3.10, 3.11 o 3.12 non trovato. Installo Python 3.11 con Python Manager..."
    & py install 3.11 -y
    if ($LASTEXITCODE -eq 0) {
        $PythonArg = Find-CompatiblePython
    }
}

if (-not $PythonArg) {
    Write-Host "Non riesco a trovare/installare Python 3.11. Controlla Python Manager con: py list --online"
    exit 1
}

Push-Location $Root

& py $PythonArg -m venv "$Venv"
if ($LASTEXITCODE -ne 0) { Pop-Location; exit $LASTEXITCODE }

$Py = Join-Path $Venv "Scripts\python.exe"

& $Py -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { Pop-Location; exit $LASTEXITCODE }

& $Py -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) { Pop-Location; exit $LASTEXITCODE }

& $Py -m pip install -r requirements.txt
$ExitCode = $LASTEXITCODE

Pop-Location

Write-Host "========================================"
Write-Host "   ENV SETUP FINISHED"
Write-Host "========================================"

exit $ExitCode
