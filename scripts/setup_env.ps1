Write-Host "========================================"
Write-Host "   JARVIS ENV SETUP"
Write-Host "========================================"

. "$PSScriptRoot\lib\jarvis.ps1"

$Root = Get-JarvisRoot
$Venv = Join-Path $Root ".venv"

Initialize-JarvisEnvironment -Root $Root

function Find-CompatiblePython {
    foreach ($Version in @("3.12", "3.11", "3.10")) {
        $Arg = "-$Version"
        $PreviousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & py $Arg --version *> $null
            if ($LASTEXITCODE -eq 0) {
                return $Arg
            }
        }
        finally {
            $ErrorActionPreference = $PreviousErrorActionPreference
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

$Py = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path $Py)) {
    & py $PythonArg -m venv "$Venv"
    if ($LASTEXITCODE -ne 0) { Pop-Location; exit $LASTEXITCODE }
} else {
    Write-Host "Venv esistente trovata: aggiorno dipendenze senza ricrearla."
}

& $Py -m pip install --upgrade pip wheel "setuptools<82"
if ($LASTEXITCODE -ne 0) { Pop-Location; exit $LASTEXITCODE }

$TorchIndex = if ($env:JARVIS_TORCH_INDEX) { $env:JARVIS_TORCH_INDEX } else { "https://download.pytorch.org/whl/cu128" }
Write-Host "Installo PyTorch CUDA da: $TorchIndex"

& $Py -m pip install --upgrade torch --index-url $TorchIndex
if ($LASTEXITCODE -ne 0 -and -not $env:JARVIS_TORCH_INDEX) {
    Write-Host "CUDA 12.8 non disponibile per questa combinazione. Provo CUDA 12.6..."
    & $Py -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu126
}
if ($LASTEXITCODE -ne 0) { Pop-Location; exit $LASTEXITCODE }

& $Py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Pop-Location; exit $LASTEXITCODE }

& $Py -m pip install "setuptools<82"
if ($LASTEXITCODE -ne 0) { Pop-Location; exit $LASTEXITCODE }

& $Py -m pip check
$ExitCode = $LASTEXITCODE

Pop-Location

Write-Host "========================================"
Write-Host "   ENV SETUP FINISHED"
Write-Host "========================================"

exit $ExitCode
