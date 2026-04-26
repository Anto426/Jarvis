$ErrorActionPreference = "Stop"

function Get-JarvisRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

function Initialize-JarvisEnvironment {
    param(
        [string]$Root = (Get-JarvisRoot),
        [string]$CudaDevices = "0"
    )

    $Cache = Join-Path $Root ".cache"

    $env:CUDA_VISIBLE_DEVICES = $CudaDevices
    $env:PIP_CACHE_DIR = Join-Path $Cache "pip"
    $env:HF_HOME = Join-Path $Cache "huggingface"
    $env:HF_DATASETS_CACHE = Join-Path $Cache "huggingface\datasets"
    $env:TORCH_HOME = Join-Path $Cache "torch"
    $env:XDG_CACHE_HOME = $Cache
    $env:PYTHONPATH = $Root
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    $env:JARVIS_FORCE_IPV4 = "1"
    $env:HF_HUB_DISABLE_XET = "1"

    New-Item -ItemType Directory -Force -Path `
        $env:PIP_CACHE_DIR, `
        $env:HF_HOME, `
        $env:HF_DATASETS_CACHE, `
        $env:TORCH_HOME | Out-Null
}

function Get-JarvisPython {
    param([string]$Root = (Get-JarvisRoot))

    $Python = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path $Python) {
        return $Python
    }

    return "python"
}

function Get-JarvisAccelerate {
    param([string]$Root = (Get-JarvisRoot))

    $Accelerate = Join-Path $Root ".venv\Scripts\accelerate.exe"
    if (Test-Path $Accelerate) {
        return $Accelerate
    }

    return "accelerate"
}

function Get-JarvisHf {
    param([string]$Root = (Get-JarvisRoot))

    return (Join-Path $Root ".venv\Scripts\hf.exe")
}

function Invoke-JarvisSetup {
    param([string]$Root = (Get-JarvisRoot))

    $SetupScript = Join-Path $Root "scripts\setup_env.ps1"
    Write-Host ""
    Write-Host ">>> Configuro/aggiorno ambiente Jarvis"
    $SetupOutput = & powershell -NoProfile -ExecutionPolicy Bypass -File $SetupScript 2>&1
    $SetupExit = $LASTEXITCODE
    $SetupOutput | ForEach-Object { Write-Host $_ }
    return ($SetupExit -eq 0)
}

function Test-JarvisPythonModule {
    param(
        [string]$Root = (Get-JarvisRoot),
        [Parameter(Mandatory = $true)]
        [string]$ModuleName
    )

    $Python = Get-JarvisPython -Root $Root
    if ($Python -eq "python") {
        return $false
    }

    & $Python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$ModuleName') is not None else 1)" *> $null
    return ($LASTEXITCODE -eq 0)
}

function Get-JarvisTorchCudaVersion {
    param([string]$Root = (Get-JarvisRoot))

    $Python = Get-JarvisPython -Root $Root
    if ($Python -eq "python") {
        return $null
    }

    $Version = & $Python -c "import torch; print(torch.version.cuda or '')" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $Version) {
        return $null
    }

    return $Version.Trim()
}

function Get-JarvisNvccCudaVersion {
    $Nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
    if (-not $Nvcc) {
        return $null
    }

    $Output = & nvcc --version 2>$null
    $Match = ($Output | Select-String -Pattern "release\s+([0-9]+\.[0-9]+)" | Select-Object -First 1)
    if (-not $Match) {
        return $null
    }

    return $Match.Matches[0].Groups[1].Value
}

function Test-JarvisFlashAttentionPrerequisites {
    param(
        [string]$Root = (Get-JarvisRoot),
        [switch]$AllowExperimentalWindows
    )

    $IsWindows = $PSVersionTable.Platform -eq "Win32NT" -or $env:OS -eq "Windows_NT"
    $TorchCuda = Get-JarvisTorchCudaVersion -Root $Root
    $NvccCuda = Get-JarvisNvccCudaVersion
    $HasCl = [bool](Get-Command cl.exe -ErrorAction SilentlyContinue)

    Write-Host ""
    Write-Host ">>> Preflight FlashAttention"
    Write-Host "OS Windows: $IsWindows"
    Write-Host "Torch CUDA: $TorchCuda"
    Write-Host "nvcc CUDA:  $NvccCuda"
    Write-Host "cl.exe:     $HasCl"

    if ($IsWindows -and -not $AllowExperimentalWindows) {
        Write-Host "FlashAttention ufficiale e pensata per Linux/WSL. Windows nativo resta sperimentale."
        return $false
    }

    if (-not $TorchCuda) {
        Write-Host "Torch non espone una build CUDA valida."
        return $false
    }

    if (-not $NvccCuda) {
        Write-Host "nvcc non trovato. Per compilare flash-attn serve CUDA Toolkit, non solo il driver NVIDIA."
        return $false
    }

    if ($TorchCuda -ne $NvccCuda) {
        Write-Host "CUDA mismatch: PyTorch e compilato con CUDA $TorchCuda, ma nvcc e CUDA $NvccCuda."
        Write-Host "Installa CUDA Toolkit $TorchCuda e mettilo prima nel PATH, oppure usa WSL/Linux con toolkit allineato."
        return $false
    }

    if ($IsWindows -and -not $HasCl) {
        Write-Host "cl.exe non trovato. Su Windows serve Visual Studio Build Tools con workload C++."
        Write-Host "Anche con cl.exe, flash-attn su Windows nativo resta sperimentale; WSL2/Ubuntu e il percorso consigliato."
        return $false
    }

    return $true
}

function Install-JarvisFlashAttention {
    param(
        [string]$Root = (Get-JarvisRoot),
        [switch]$AllowExperimentalWindows
    )

    $Python = Get-JarvisPython -Root $Root
    if ($Python -eq "python") {
        return $false
    }

    $IsWindows = $PSVersionTable.Platform -eq "Win32NT" -or $env:OS -eq "Windows_NT"
    if (-not (Test-JarvisFlashAttentionPrerequisites -Root $Root -AllowExperimentalWindows:$AllowExperimentalWindows)) {
        return $false
    }

    Write-Host ""
    Write-Host ">>> Installo FlashAttention 2 (flash-attn)"
    $VenvScripts = Join-Path $Root ".venv\Scripts"
    if (Test-Path $VenvScripts) {
        $env:PATH = "$VenvScripts;$env:PATH"
    }

    & $Python -m pip install ninja packaging psutil
    if ($LASTEXITCODE -ne 0) {
        return $false
    }

    $env:MAX_JOBS = if ($env:MAX_JOBS) { $env:MAX_JOBS } else { "4" }
    & $Python -m pip install flash-attn --no-build-isolation
    return ($LASTEXITCODE -eq 0)
}

function Invoke-JarvisStep {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,

        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    Write-Host ""
    Write-Host ">>> $Label"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Label (exit $LASTEXITCODE)"
    }
}

function Invoke-JarvisTraining {
    param(
        [string]$Root = (Get-JarvisRoot),
        [string]$MixedPrecision = "fp16",
        [string]$DynamoBackend = "no"
    )

    $Accelerate = Get-JarvisAccelerate -Root $Root
    Push-Location $Root
    try {
        & $Accelerate launch `
            --mixed_precision=$MixedPrecision `
            --num_processes=1 `
            --num_machines=1 `
            --dynamo_backend=$DynamoBackend `
            training/train.py

        return $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
}

function Get-JarvisBestMixedPrecision {
    param([string]$Root = (Get-JarvisRoot))

    $Python = Get-JarvisPython -Root $Root
    if ($Python -eq "python") {
        return "fp16"
    }

    $Result = & $Python -c "import torch; print('bf16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'fp16')" 2>$null
    if ($LASTEXITCODE -eq 0 -and $Result) {
        return $Result.Trim()
    }

    return "fp16"
}

function Test-JarvisDependencies {
    param([string]$Root = (Get-JarvisRoot))

    $Python = Get-JarvisPython -Root $Root
    if ($Python -eq "python") {
        Write-Host "Python venv non trovato. Esegui prima: powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1"
        return $false
    }

    Write-Host ""
    Write-Host ">>> Check dipendenze Python"
    $PipCheck = & $Python -m pip check 2>&1
    $PipExit = $LASTEXITCODE
    $PipCheck | ForEach-Object { Write-Host $_ }
    if ($PipExit -ne 0) {
        Write-Host "Dipendenze non coerenti. Prova: powershell -ExecutionPolicy Bypass -File .\scripts\jarvis.ps1 -UpdateDeps"
        return $false
    }

    Write-Host ""
    Write-Host ">>> Check CUDA / librerie training"
    $ImportCheck = & $Python -c "import sys, importlib.util, torch, accelerate, transformers, datasets; print('python', sys.version.split()[0]); print('torch', torch.__version__, 'cuda_runtime', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False); print('triton_available', importlib.util.find_spec('triton') is not None); print('flash_attn_available', importlib.util.find_spec('flash_attn') is not None); print('accelerate', accelerate.__version__); print('transformers', transformers.__version__); print('datasets', datasets.__version__)" 2>&1
    $ImportExit = $LASTEXITCODE
    $ImportCheck | ForEach-Object { Write-Host $_ }
    if ($ImportExit -ne 0) {
        Write-Host "Import dipendenze fallito. Prova: powershell -ExecutionPolicy Bypass -File .\scripts\jarvis.ps1 -UpdateDeps"
        return $false
    }

    return $true
}

function Ensure-JarvisDependencies {
    param(
        [string]$Root = (Get-JarvisRoot),
        [switch]$ForceUpdate
    )

    if ($ForceUpdate) {
        if (-not (Invoke-JarvisSetup -Root $Root)) {
            return $false
        }
        Initialize-JarvisEnvironment -Root $Root
    }

    if (Test-JarvisDependencies -Root $Root) {
        return $true
    }

    Write-Host ""
    Write-Host ">>> Ambiente non pronto: provo setup/update automatico"
    if (-not (Invoke-JarvisSetup -Root $Root)) {
        return $false
    }

    Initialize-JarvisEnvironment -Root $Root
    return (Test-JarvisDependencies -Root $Root)
}
