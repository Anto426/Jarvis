param(
    [switch]$FromScratch,
    [switch]$RebuildData,
    [switch]$Turbo,
    [switch]$MaxPerf,
    [switch]$FlashAttention,
    [switch]$InstallFlashAttention,
    [switch]$ForceExperimentalFlashAttentionInstall,
    [switch]$UpdateDeps,
    [switch]$SkipDependencyCheck,
    [switch]$SkipTraining,
    [ValidateSet("auto", "0", "force")]
    [string]$HuggingFace = "auto"
)

Write-Host "========================================"
Write-Host "   JARVIS SMART LAUNCHER"
Write-Host "========================================"

. "$PSScriptRoot\lib\jarvis.ps1"

$Root = Get-JarvisRoot
Initialize-JarvisEnvironment -Root $Root

$DataDir = Join-Path $Root "data"
$RawDir = Join-Path $DataDir "raw"
$CleanedDir = Join-Path $DataDir "cleaned"
$ShardsDir = Join-Path $DataDir "shards"
$TokenizerDir = Join-Path $DataDir "tokenizer"
$CheckpointDir = Join-Path $Root "checkpoints"
$ManifestPath = Join-Path $ShardsDir ".jarvis_data_manifest.json"

$env:JARVIS_USE_HUGGINGFACE = $HuggingFace
$env:JARVIS_USE_HF_WIKIPEDIA = "0"
$env:JARVIS_RAW_INCLUDE = "local_import.jsonl;wikipedia_it_wikimedia_direct.jsonl;fineweb2_it.jsonl;stackexchange_auto.jsonl"
$env:JARVIS_CLEAN_INCLUDE = "local_import_clean.jsonl;wikipedia_it_wikimedia_direct_clean.jsonl;fineweb2_it_clean.jsonl;stackexchange_auto_clean.jsonl"
$env:JARVIS_DEDUP_INCLUDE = "local_import_dedup.jsonl;wikipedia_it_wikimedia_direct_dedup.jsonl;fineweb2_it_dedup.jsonl;stackexchange_auto_dedup.jsonl"
$env:JARVIS_MIXED_PRECISION = "bf16_if_available"
$env:JARVIS_FUSED_OPTIMIZER = "1"
$env:JARVIS_TF32 = "1"
$env:CUDA_MODULE_LOADING = "LAZY"
$env:TORCH_CUDNN_V8_API_ENABLED = "1"

if ($Turbo -or $MaxPerf) {
    $env:JARVIS_TORCH_COMPILE = "1"
}

if ($MaxPerf) {
    $env:JARVIS_AUTO_BATCH = "1"
    $env:JARVIS_TARGET_VRAM = "0.92"
    $env:JARVIS_MAX_BATCH_SIZE = "4"
    $env:JARVIS_TUNE_REPEATS = "2"
}

if ($FlashAttention -or $MaxPerf) {
    $env:JARVIS_USE_FLASH_ATTENTION = "1"
    $env:JARVIS_ATTENTION_IMPL = "auto"
}

if (-not $SkipDependencyCheck) {
    if (-not (Ensure-JarvisDependencies -Root $Root -ForceUpdate:$UpdateDeps)) {
        Write-Host "Setup automatico fallito: controlla l'output sopra."
        exit 1
    }
}

$Python = Get-JarvisPython -Root $Root

if ($InstallFlashAttention) {
    if (-not (Install-JarvisFlashAttention -Root $Root -AllowExperimentalWindows:$ForceExperimentalFlashAttentionInstall)) {
        Write-Host "Install FlashAttention fallita o non supportata su questo ambiente."
        Write-Host "Percorso consigliato per FlashAttention stabile: WSL2/Ubuntu o Linux nativo."
    }
}

if (($FlashAttention -or $MaxPerf) -and -not (Test-JarvisPythonModule -Root $Root -ModuleName "flash_attn")) {
    Write-Host "FlashAttention richiesta, ma flash_attn non e installato/importabile: usero SDPA."
    Write-Host "Per tentare install ufficiale: powershell -ExecutionPolicy Bypass -File .\scripts\jarvis.ps1 -InstallFlashAttention -FlashAttention"
}

function Get-RelativePath {
    param([string]$Path)

    $RootPath = (Resolve-Path $Root).Path
    if (-not $RootPath.EndsWith("\")) {
        $RootPath = "$RootPath\"
    }

    $RootUri = [System.Uri]::new($RootPath)
    $PathUri = [System.Uri]::new((Resolve-Path $Path).Path)
    return [System.Uri]::UnescapeDataString(
        $RootUri.MakeRelativeUri($PathUri).ToString()
    ).Replace("\", "/")
}

function Get-DataFingerprint {
    $Items = @()
    $Sources = @(
        (Join-Path $Root "config\paths.yaml"),
        (Join-Path $Root "data_pipeline")
    )

    $LocalData = Join-Path $Root "data\local"
    if (Test-Path $LocalData) {
        $Sources += $LocalData
    }

    foreach ($Source in $Sources) {
        if (-not (Test-Path $Source)) {
            continue
        }

        $Files = if ((Get-Item $Source).PSIsContainer) {
            Get-ChildItem -LiteralPath $Source -Recurse -File | Sort-Object FullName
        } else {
            @(Get-Item $Source)
        }

        foreach ($File in $Files) {
            $Hash = Get-FileHash -Algorithm SHA256 -LiteralPath $File.FullName
            $Items += "{0}:{1}:{2}" -f (Get-RelativePath $File.FullName), $File.Length, $Hash.Hash
        }
    }

    $Joined = $Items -join "`n"
    $Bytes = [System.Text.Encoding]::UTF8.GetBytes($Joined)
    $Sha = [System.Security.Cryptography.SHA256]::Create()
    return [System.BitConverter]::ToString($Sha.ComputeHash($Bytes)).Replace("-", "").ToLowerInvariant()
}

function Get-SavedFingerprint {
    if (-not (Test-Path $ManifestPath)) {
        return $null
    }

    try {
        return (Get-Content -Raw -LiteralPath $ManifestPath | ConvertFrom-Json).fingerprint
    }
    catch {
        return $null
    }
}

function Save-DataFingerprint {
    param([string]$Fingerprint)

    New-Item -ItemType Directory -Force -Path $ShardsDir | Out-Null
    [pscustomobject]@{
        fingerprint = $Fingerprint
        generated_at = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath $ManifestPath
}

function Test-PackedDatasetReady {
    return (Test-Path $ShardsDir) -and [bool](Get-ChildItem -LiteralPath $ShardsDir -Filter "packed_*.parquet" -File -ErrorAction SilentlyContinue)
}

function Test-CheckpointReady {
    return (Test-Path $CheckpointDir) -and [bool](Get-ChildItem -LiteralPath $CheckpointDir -Directory -Filter "step_*" -ErrorAction SilentlyContinue)
}

function Remove-GeneratedPath {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return
    }

    $ResolvedRoot = (Resolve-Path $Root).Path
    $ResolvedPath = (Resolve-Path $Path).Path
    if (-not $ResolvedPath.StartsWith($ResolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Percorso fuori dal progetto, pulizia bloccata: $ResolvedPath"
    }

    Remove-Item -LiteralPath $ResolvedPath -Recurse -Force
}

function Clear-GeneratedData {
    Write-Host "Pulisco dati generati: raw, cleaned, shards, tokenizer"
    Remove-GeneratedPath $RawDir
    Remove-GeneratedPath $CleanedDir
    Remove-GeneratedPath $ShardsDir
    Remove-GeneratedPath $TokenizerDir
}

function Clear-Checkpoints {
    Write-Host "Pulisco checkpoint per ripartenza da zero"
    Remove-GeneratedPath $CheckpointDir
}

function Invoke-DataPipeline {
    Push-Location $Root
    try {
        Invoke-JarvisStep "Import local data" { & $Python data_pipeline/import_local.py }
        Invoke-JarvisStep "Collect public data" { & $Python data_pipeline/collect.py }
        Invoke-JarvisStep "Clean data" { & $Python data_pipeline/clean.py }
        Invoke-JarvisStep "Deduplicate data" { & $Python data_pipeline/deduplicate.py }
        Invoke-JarvisStep "Build tokenizer" { & $Python data_pipeline/build_tokenizer.py }
        Invoke-JarvisStep "Tokenize shards" { & $Python data_pipeline/tokenize_shards.py }
        Invoke-JarvisStep "Pack blocks" { & $Python data_pipeline/pack_blocks.py }
    }
    finally {
        Pop-Location
    }
}

$CurrentFingerprint = Get-DataFingerprint
$SavedFingerprint = Get-SavedFingerprint
$DatasetReady = Test-PackedDatasetReady
$CheckpointReady = Test-CheckpointReady
$DataChanged = ($SavedFingerprint -ne $CurrentFingerprint)

if ($RebuildData -and $CheckpointReady -and -not $FromScratch) {
    Write-Host "RebuildData bloccato: ci sono checkpoint esistenti."
    Write-Host "Per rigenerare il dataset senza mischiare stati vecchi e dati nuovi usa: -FromScratch -RebuildData"
    exit 1
}

if ($FromScratch) {
    Clear-Checkpoints
    $CheckpointReady = $false
}

$ShouldBuildData = $false
if ($RebuildData) {
    $ShouldBuildData = $true
} elseif (-not $DatasetReady) {
    $ShouldBuildData = $true
} elseif ($DataChanged -and -not $CheckpointReady) {
    $ShouldBuildData = $true
}

if ($ShouldBuildData) {
    if ($RebuildData -or $FromScratch -or $DataChanged) {
        Clear-GeneratedData
    }

    Invoke-DataPipeline
    Save-DataFingerprint -Fingerprint $CurrentFingerprint
} elseif ($DataChanged -and $CheckpointReady) {
    Write-Host "Dataset pronto e checkpoint presenti: riprendo training senza rigenerare i dati."
    Write-Host "La pipeline dati e cambiata; usa -FromScratch -RebuildData quando vuoi ricostruire tutto da zero."
} else {
    Write-Host "Dataset packed gia pronto: salto pipeline dati."
}

if ($SkipTraining) {
    Write-Host "SkipTraining attivo: mi fermo dopo la verifica/generazione dati."
    exit 0
}

$MixedPrecision = Get-JarvisBestMixedPrecision -Root $Root
$ExitCode = Invoke-JarvisTraining -Root $Root -MixedPrecision $MixedPrecision

Write-Host "========================================"
Write-Host "   JARVIS FINISHED"
Write-Host "========================================"

exit $ExitCode
