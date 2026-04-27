param(
    [Alias("ResetAll", "ResettaTutto", "RipristinaTutto")]
    [switch]$FromScratch,
    [switch]$RebuildSources,
    [switch]$RebuildStageData,
    [switch]$ResetActiveCheckpoints,
    [Alias("RipulisciStage", "CleanStageRestart")]
    [switch]$CleanRestartStage,
    [string]$Step = "",
    [string]$StartStep = "",
    [string]$StopAfterStep = "",
    [switch]$Turbo,
    [switch]$MaxPerf,
    [switch]$FlashAttention,
    [switch]$InstallFlashAttention,
    [switch]$ForceExperimentalFlashAttentionInstall,
    [switch]$UpdateDeps,
    [switch]$SkipDependencyCheck,
    [switch]$NoDashboard,
    [int]$DashboardPort = 8765,
    [ValidateSet("dev", "full", "max")]
    [string]$DataScale = "full",
    [ValidateSet("auto", "0", "force")]
    [string]$HuggingFace = "auto"
)

Write-Host "========================================"
Write-Host "   JARVIS STAGED PIPELINE"
Write-Host "========================================"

. "$PSScriptRoot\lib\jarvis.ps1"

$Root = Get-JarvisRoot
Initialize-JarvisEnvironment -Root $Root

$Python = Get-JarvisPython -Root $Root
$DataDir = Join-Path $Root "data"
$RawDir = Join-Path $DataDir "raw"
$StagesDir = Join-Path $DataDir "stages"
$TokenizerDir = Join-Path $DataDir "tokenizer"
$CheckpointDir = Join-Path $Root "checkpoints"
$OfficialDir = Join-Path $CheckpointDir "official"
$LogsDir = Join-Path $Root "logs"

$env:JARVIS_USE_HUGGINGFACE = $HuggingFace
$env:JARVIS_USE_HF_WIKIPEDIA = "0"
$env:JARVIS_MIXED_PRECISION = "bf16_if_available"
$env:JARVIS_FUSED_OPTIMIZER = "1"
$env:JARVIS_TF32 = "1"
$env:CUDA_MODULE_LOADING = "LAZY"
$env:TORCH_CUDNN_V8_API_ENABLED = "1"
if ($env:OS -eq "Windows_NT") {
    Remove-Item Env:PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue
} else {
    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
}
$CpuThreads = [Environment]::ProcessorCount
$env:JARVIS_CPU_CORE_POLICY = "all_cores"
$env:JARVIS_CPU_AFFINITY = "1"
$env:JARVIS_CPU_PRIORITY = "above_normal"
$env:JARVIS_CPU_THREADS = "auto"
$env:JARVIS_CPU_INTEROP_THREADS = "2"
$env:OMP_NUM_THREADS = "$CpuThreads"
$env:MKL_NUM_THREADS = "$CpuThreads"
$env:NUMEXPR_NUM_THREADS = "$CpuThreads"
$env:NUMEXPR_MAX_THREADS = "$CpuThreads"
$env:OMP_DYNAMIC = "FALSE"
$env:MKL_DYNAMIC = "FALSE"
Remove-Item Env:OMP_PROC_BIND -ErrorAction SilentlyContinue
Remove-Item Env:OMP_PLACES -ErrorAction SilentlyContinue
$env:KMP_AFFINITY = "granularity=fine,compact,1,0"
$env:KMP_BLOCKTIME = "1"
$env:KMP_SETTINGS = "0"

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

function Set-DataScaleEnvironment {
    param([string]$Scale)

    if ($Scale -eq "dev") {
        $env:JARVIS_WIKIPEDIA_LIMIT = "100000"
        $env:JARVIS_FINEWEB_LIMIT = "200000"
        $env:JARVIS_STACK_LIMIT = "300000"
        $env:JARVIS_SQUAD_IT_LIMIT = "20000"
        $env:JARVIS_PIQA_IT_LIMIT = "10000"
        $env:JARVIS_EVOL_INSTRUCT_IT_LIMIT = "20000"
        $env:JARVIS_WIKIMEDIA_DIRECT_LIMIT = "100000"
        $env:JARVIS_WIKIMEDIA_DIRECT_ALL_PARTS = "0"
        return
    }

    if ($Scale -eq "full") {
        $env:JARVIS_WIKIPEDIA_LIMIT = "90000"
        $env:JARVIS_FINEWEB_LIMIT = "9000000"
        $env:JARVIS_STACK_LIMIT = "9000000000"
        $env:JARVIS_SQUAD_IT_LIMIT = "9000000000"
        $env:JARVIS_PIQA_IT_LIMIT = "9000000000"
        $env:JARVIS_EVOL_INSTRUCT_IT_LIMIT = "9000000000"
        $env:JARVIS_WIKIMEDIA_DIRECT_LIMIT = "9000000000"
        $env:JARVIS_WIKIMEDIA_DIRECT_ALL_PARTS = "1"
        return
    }

    if ($Scale -eq "max") {
        $env:JARVIS_WIKIPEDIA_LIMIT = "0"
        $env:JARVIS_FINEWEB_LIMIT = "0"
        $env:JARVIS_STACK_LIMIT = "0"
        $env:JARVIS_SQUAD_IT_LIMIT = "0"
        $env:JARVIS_PIQA_IT_LIMIT = "0"
        $env:JARVIS_EVOL_INSTRUCT_IT_LIMIT = "0"
        $env:JARVIS_WIKIMEDIA_DIRECT_LIMIT = "0"
        $env:JARVIS_WIKIMEDIA_DIRECT_ALL_PARTS = "1"
    }
}

Set-DataScaleEnvironment -Scale $DataScale
Write-Host "Data scale: $DataScale"

function Resolve-InProjectPath {
    param([string]$Path)

    $RootPath = (Resolve-Path $Root).Path
    $Candidate = if ([System.IO.Path]::IsPathRooted($Path)) {
        $Path
    } else {
        Join-Path $Root $Path
    }

    $FullPath = [System.IO.Path]::GetFullPath($Candidate)
    if (-not $FullPath.StartsWith($RootPath, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Percorso fuori dal progetto, operazione bloccata: $FullPath"
    }

    return $FullPath
}

function Remove-ProjectPath {
    param([string]$Path)

    $FullPath = Resolve-InProjectPath $Path
    if (Test-Path $FullPath) {
        Remove-Item -LiteralPath $FullPath -Recurse -Force
    }
}

function Invoke-PipelineJson {
    param([string[]]$CommandArgs)

    Push-Location $Root
    try {
        $PreviousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $Output = & $Python training/pipeline.py @CommandArgs 2>&1
            $ExitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $PreviousErrorActionPreference
        }

        if ($ExitCode -ne 0) {
            $Message = ($Output | ForEach-Object { "$_" }) -join "`n"
            throw "Pipeline command failed: $($CommandArgs -join ' ')`n$Message"
        }
        return (($Output -join "`n") | ConvertFrom-Json)
    }
    finally {
        Pop-Location
    }
}

function Mark-PipelineStep {
    param(
        [string]$StepId,
        [string]$Status,
        [string]$Checkpoint = "",
        [string]$Message = ""
    )

    $MarkArgs = @("mark", "--step", $StepId, "--status", $Status)
    if ($Checkpoint) {
        $MarkArgs += @("--checkpoint", $Checkpoint)
    }
    if ($Message) {
        $MarkArgs += @("--message", $Message)
    }
    Invoke-PipelineJson -CommandArgs $MarkArgs | Out-Null
}

function Test-PipelineStepStarted {
    param([pscustomobject]$Status)

    if (-not $Status) {
        return $false
    }

    return (
        $Status.status -ne "pending" -or
        -not [string]::IsNullOrWhiteSpace([string]$Status.started_at) -or
        -not [string]::IsNullOrWhiteSpace([string]$Status.completed_at)
    )
}

function Get-LatestActiveCheckpoint {
    if (-not (Test-Path $CheckpointDir)) {
        return $null
    }

    $Candidates = Get-ChildItem -LiteralPath $CheckpointDir -Directory -Filter "step_*" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^step_[0-9]+$' } |
        Sort-Object { [int]($_.Name -replace '^step_', '') } -Descending

    return $Candidates | Select-Object -First 1
}

function Test-CheckpointComplete {
    param([System.IO.DirectoryInfo]$Checkpoint)

    if (-not $Checkpoint) {
        return $false
    }

    $HasModel = (Test-Path (Join-Path $Checkpoint.FullName "pytorch_model.bin")) -or
        (Test-Path (Join-Path $Checkpoint.FullName "model.safetensors"))
    $HasState = Test-Path (Join-Path $Checkpoint.FullName "scheduler.bin")

    return ($HasModel -and $HasState)
}

function Get-PipelineStepById {
    param(
        [object[]]$PipelineSteps,
        [string]$StepId
    )

    return @($PipelineSteps | Where-Object { $_.id -eq $StepId } | Select-Object -First 1)[0]
}

function Get-LatestOfficialCheckpointPath {
    param([pscustomobject]$Stage)

    $StageRoot = Join-Path $OfficialDir $Stage.id
    if (-not (Test-Path $StageRoot)) {
        return $null
    }

    if ($Stage.kind -eq "prepare") {
        $TokenizerModel = Join-Path $StageRoot "jarvis.model"
        if ((Test-Path $TokenizerModel) -and (Get-Item -LiteralPath $TokenizerModel).Length -gt 0) {
            return $StageRoot
        }
        return $null
    }

    return Get-ChildItem -LiteralPath $StageRoot -Directory -Filter "step_*" -ErrorAction SilentlyContinue |
        Where-Object { Test-CheckpointComplete $_ } |
        Sort-Object { [int]($_.Name -replace '^step_', '') } -Descending |
        Select-Object -First 1
}

function Test-OfficialCheckpointForStage {
    param([pscustomobject]$Stage)

    if (-not $Stage) {
        return $false
    }

    $Manifest = Join-Path (Join-Path $OfficialDir $Stage.id) "official.json"
    return ((Test-Path $Manifest) -and [bool](Get-LatestOfficialCheckpointPath -Stage $Stage))
}

function Get-LatestOfficialStage {
    param([object[]]$PipelineSteps)

    foreach ($Item in ($PipelineSteps | Sort-Object order -Descending)) {
        if (Test-OfficialCheckpointForStage -Stage $Item) {
            return $Item
        }
    }

    return $null
}

function Write-LatestOfficialPointer {
    param(
        [pscustomobject]$Stage,
        [string]$ArtifactPath
    )

    [pscustomobject]@{
        step_id = $Stage.id
        order = [int]$Stage.order
        kind = $Stage.kind
        title = $Stage.title
        artifact_path = $ArtifactPath
        updated_at = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $OfficialDir "latest.json")
}

function Sync-LatestOfficialPointer {
    param([object[]]$PipelineSteps)

    $Latest = Get-LatestOfficialStage -PipelineSteps $PipelineSteps
    if (-not $Latest) {
        return
    }

    $Artifact = Get-LatestOfficialCheckpointPath -Stage $Latest
    if ($Artifact) {
        $ArtifactPath = if ($Artifact -is [System.IO.FileSystemInfo]) {
            $Artifact.FullName
        } else {
            [string]$Artifact
        }
        Write-LatestOfficialPointer -Stage $Latest -ArtifactPath $ArtifactPath
    }
}

function Assert-ForwardOnlyStage {
    param(
        [pscustomobject]$Stage,
        [object[]]$PipelineSteps
    )

    $Latest = Get-LatestOfficialStage -PipelineSteps $PipelineSteps
    if (-not $Latest) {
        if ([int]$Stage.order -ne 0) {
            throw "Nessun checkpoint ufficiale trovato. Devi partire da step_0_prepare oppure usare -FromScratch."
        }
        return
    }

    $LatestOrder = [int]$Latest.order
    $StageOrder = [int]$Stage.order

    if ($StageOrder -lt $LatestOrder) {
        throw "Indietro bloccato: ultimo checkpoint ufficiale = $($Latest.id), richiesto = $($Stage.id). Usa solo lo stesso step o il successivo."
    }

    if ($StageOrder -gt ($LatestOrder + 1)) {
        throw "Salto bloccato: ultimo checkpoint ufficiale = $($Latest.id), richiesto = $($Stage.id). Devi completare prima lo step successivo."
    }
}

function Assert-RequiredOfficialCheckpoint {
    param(
        [pscustomobject]$Stage,
        [object[]]$PipelineSteps
    )

    if (-not $Stage.requires) {
        return
    }

    $Required = Get-PipelineStepById -PipelineSteps $PipelineSteps -StepId $Stage.requires
    if (-not (Test-OfficialCheckpointForStage -Stage $Required)) {
        throw "Checkpoint ufficiale richiesto mancante: $($Stage.requires). Completa e pubblica prima quello step."
    }
}

function Clear-ActiveCheckpoints {
    if (-not (Test-Path $CheckpointDir)) {
        return
    }

    Get-ChildItem -LiteralPath $CheckpointDir -Directory -Filter "step_*" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^step_[0-9]+(\.tmp)?$' } |
        ForEach-Object { Remove-ProjectPath $_.FullName }
}

function Restore-OfficialCheckpoint {
    param([string]$StepId)

    $SourceRoot = Join-Path $OfficialDir $StepId
    if (-not (Test-Path $SourceRoot)) {
        throw "Checkpoint ufficiale mancante per ${StepId}: $SourceRoot"
    }

    $Source = Get-ChildItem -LiteralPath $SourceRoot -Directory -Filter "step_*" |
        Sort-Object { [int]($_.Name -replace '^step_', '') } -Descending |
        Select-Object -First 1

    if (-not (Test-CheckpointComplete $Source)) {
        throw "Checkpoint ufficiale incompleto per ${StepId}: $($Source.FullName)"
    }

    Clear-ActiveCheckpoints
    New-Item -ItemType Directory -Force -Path $CheckpointDir | Out-Null
    Copy-Item -LiteralPath $Source.FullName -Destination (Join-Path $CheckpointDir $Source.Name) -Recurse -Force
    Write-Host "Ripristinato checkpoint ufficiale: $StepId -> $($Source.Name)"
}

function Publish-OfficialCheckpoint {
    param([pscustomobject]$Stage)

    $Latest = Get-LatestActiveCheckpoint
    if (-not (Test-CheckpointComplete $Latest)) {
        throw "Nessun checkpoint completo da pubblicare per $($Stage.id)."
    }

    $TargetRoot = Join-Path $OfficialDir $Stage.id
    Remove-ProjectPath $TargetRoot
    New-Item -ItemType Directory -Force -Path $TargetRoot | Out-Null
    $Target = Join-Path $TargetRoot $Latest.Name
    Copy-Item -LiteralPath $Latest.FullName -Destination $Target -Recurse -Force

    [pscustomobject]@{
        step_id = $Stage.id
        title = $Stage.title
        checkpoint = $Latest.Name
        published_at = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $TargetRoot "official.json")

    Write-LatestOfficialPointer -Stage $Stage -ArtifactPath $Target

    return $Target
}

function Publish-PrepareCheckpoint {
    param([pscustomobject]$Stage)

    $TokenizerModel = Join-Path $TokenizerDir "jarvis.model"
    $TokenizerVocab = Join-Path $TokenizerDir "jarvis.vocab"
    if (-not (Test-Path $TokenizerModel)) {
        throw "Tokenizer non trovato dopo $($Stage.id): $TokenizerModel"
    }

    $TargetRoot = Join-Path $OfficialDir $Stage.id
    Remove-ProjectPath $TargetRoot
    New-Item -ItemType Directory -Force -Path $TargetRoot | Out-Null
    Copy-Item -LiteralPath $TokenizerModel -Destination (Join-Path $TargetRoot "jarvis.model") -Force
    if (Test-Path $TokenizerVocab) {
        Copy-Item -LiteralPath $TokenizerVocab -Destination (Join-Path $TargetRoot "jarvis.vocab") -Force
    }

    [pscustomobject]@{
        step_id = $Stage.id
        title = $Stage.title
        tokenizer = "jarvis.model"
        published_at = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $TargetRoot "official.json")

    Write-LatestOfficialPointer -Stage $Stage -ArtifactPath $TargetRoot

    return $TargetRoot
}

function Test-TokenizerReady {
    $TokenizerModel = Join-Path $TokenizerDir "jarvis.model"
    $TokenizerVocab = Join-Path $TokenizerDir "jarvis.vocab"
    return (
        (Test-Path $TokenizerModel) -and
        (Get-Item -LiteralPath $TokenizerModel).Length -gt 0 -and
        (Test-Path $TokenizerVocab) -and
        (Get-Item -LiteralPath $TokenizerVocab).Length -gt 0
    )
}

function Set-StageEnvironment {
    param([pscustomobject]$Stage)

    $env:JARVIS_PIPELINE_STEP_ID = $Stage.id
    $env:JARVIS_PIPELINE_STEP_ORDER = "$($Stage.order)"
    $env:JARVIS_PIPELINE_STEP_TITLE = $Stage.title
    $env:JARVIS_PIPELINE_STEP_OBJECTIVE = $Stage.objective
    $env:JARVIS_PIPELINE_STEP_REQUIRES = $Stage.requires
    $env:JARVIS_RAW_INCLUDE = $Stage.raw_include
    $env:JARVIS_CLEAN_INCLUDE = $Stage.clean_include
    $env:JARVIS_DEDUP_INCLUDE = $Stage.dedup_include
    $env:JARVIS_ALLOWED_LANGS = $Stage.allowed_languages
    $env:JARVIS_DATA_PROFILE = $Stage.data_profile
    $env:JARVIS_ALLOWED_FORMATS = $Stage.allowed_formats
    $env:JARVIS_PATH_CLEANED_DATA_DIR = $Stage.cleaned_data_dir
    $env:JARVIS_PATH_SHARDS_DIR = $Stage.shards_dir
    $env:JARVIS_RESTORE_WEIGHTS_ONLY = "0"
}

function Clear-StageGeneratedData {
    param([pscustomobject]$Stage)

    Remove-ProjectPath $Stage.cleaned_data_dir
    Remove-ProjectPath $Stage.shards_dir
}

function Test-StageHasData {
    param([pscustomobject]$Stage)

    $DedupDir = Resolve-InProjectPath (Join-Path $Stage.cleaned_data_dir "deduplicated")
    if (-not (Test-Path $DedupDir)) {
        return $false
    }

    $Files = Get-ChildItem -LiteralPath $DedupDir -Filter "*_dedup.jsonl" -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Length -gt 0 }
    return [bool]($Files | Select-Object -First 1)
}

function Test-StageHasPackedData {
    param([pscustomobject]$Stage)

    $ShardDir = Resolve-InProjectPath $Stage.shards_dir
    if (-not (Test-Path $ShardDir)) {
        return $false
    }

    $Files = Get-ChildItem -LiteralPath $ShardDir -Filter "packed_*.parquet" -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Length -gt 0 }
    return [bool]($Files | Select-Object -First 1)
}

function Assert-StageHasPackedData {
    param([pscustomobject]$Stage)

    if (Test-StageHasPackedData -Stage $Stage) {
        return
    }

    throw "Dataset tokenizzato mancante per $($Stage.id). Esegui prima step_0_prepare: powershell -ExecutionPolicy Bypass -File .\scripts\jarvis_pipeline.ps1 -Step step_0_prepare"
}

function Test-ActiveCheckpointForStage {
    param([string]$StepId)

    if (-not (Test-Path $CheckpointDir)) {
        return $false
    }

    $Candidates = Get-ChildItem -LiteralPath $CheckpointDir -Directory -Filter "step_*" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^step_[0-9]+$' } |
        Sort-Object { [int]($_.Name -replace '^step_', '') } -Descending

    foreach ($Candidate in $Candidates) {
        if (-not (Test-CheckpointComplete $Candidate)) {
            continue
        }

        $MetaPath = Join-Path $Candidate.FullName "jarvis_checkpoint_meta.json"
        if (-not (Test-Path $MetaPath)) {
            continue
        }

        try {
            $Meta = Get-Content -Raw -LiteralPath $MetaPath | ConvertFrom-Json
            if ($Meta.pipeline_step.id -eq $StepId) {
                return $true
            }
        }
        catch {
            continue
        }
    }

    return $false
}

function Test-StageHasRawDataCandidate {
    param([pscustomobject]$Stage)

    foreach ($RawFile in @($Stage.raw_files)) {
        $RawPath = Resolve-InProjectPath (Join-Path $RawDir $RawFile)
        if ((Test-Path $RawPath) -and (Get-Item -LiteralPath $RawPath).Length -gt 0) {
            return $true
        }
    }

    return $false
}

function Invoke-StageDataBuild {
    param(
        [pscustomobject]$Stage,
        [switch]$BuildTokenizer
    )

    Clear-StageGeneratedData -Stage $Stage

    Push-Location $Root
    try {
        Invoke-JarvisStep "[$($Stage.id)] Import local data" { & $Python data_pipeline/import_local.py }
        Invoke-JarvisStep "[$($Stage.id)] Collect public data" { & $Python data_pipeline/collect.py }
        Invoke-JarvisStep "[$($Stage.id)] Clean data" { & $Python data_pipeline/clean.py }
        Invoke-JarvisStep "[$($Stage.id)] Deduplicate data" { & $Python data_pipeline/deduplicate.py }

        if (-not (Test-StageHasData -Stage $Stage)) {
            throw "Nessun dato valido per $($Stage.id). Aggiungi dataset locali o abilita le fonti richieste."
        }

        if ($BuildTokenizer) {
            Invoke-JarvisStep "[$($Stage.id)] Build tokenizer once" { & $Python data_pipeline/build_tokenizer.py }
        }
    }
    finally {
        Pop-Location
    }
}

function Invoke-StageShardBuild {
    param([pscustomobject]$Stage)

    Clear-StageGeneratedData -Stage $Stage

    Push-Location $Root
    try {
        Invoke-JarvisStep "[$($Stage.id)] Clean data" { & $Python data_pipeline/clean.py }
        Invoke-JarvisStep "[$($Stage.id)] Deduplicate data" { & $Python data_pipeline/deduplicate.py }

        if (-not (Test-StageHasData -Stage $Stage)) {
            throw "Nessun dato valido per $($Stage.id)."
        }

        if (-not (Test-Path (Join-Path $TokenizerDir "jarvis.model"))) {
            throw "Tokenizer mancante. Completa prima step_0_prepare."
        }

        Invoke-JarvisStep "[$($Stage.id)] Tokenize shards" { & $Python data_pipeline/tokenize_shards.py }
        Invoke-JarvisStep "[$($Stage.id)] Pack blocks" { & $Python data_pipeline/pack_blocks.py }
    }
    finally {
        Pop-Location
    }
}

function Invoke-TrainingDatasetPreparation {
    param([object[]]$PipelineSteps)

    foreach ($Item in ($PipelineSteps | Where-Object { $_.kind -eq "train" } | Sort-Object order)) {
        $TrainStage = Invoke-PipelineJson -CommandArgs @("env", "--step", $Item.id)
        Set-StageEnvironment -Stage $TrainStage

        if (-not (Test-StageHasRawDataCandidate -Stage $TrainStage)) {
            Write-Host "Dataset saltato per $($TrainStage.id): nessuna sorgente raw disponibile."
            continue
        }

        try {
            Invoke-StageShardBuild -Stage $TrainStage
        }
        catch {
            Write-Host "Dataset non preparato per $($TrainStage.id): $_"
            Write-Host "Lo step di training restera bloccato finche step_0_prepare non genera i suoi shard."
        }
    }
}

function Test-TrainingDatasetsReady {
    param([object[]]$PipelineSteps)

    foreach ($Item in ($PipelineSteps | Where-Object { $_.kind -eq "train" } | Sort-Object order)) {
        $TrainStage = Invoke-PipelineJson -CommandArgs @("env", "--step", $Item.id)
        if (-not (Test-StageHasRawDataCandidate -Stage $TrainStage)) {
            continue
        }
        if (-not (Test-StageHasPackedData -Stage $TrainStage)) {
            return $false
        }
    }

    return $true
}

if (-not $SkipDependencyCheck) {
    if (-not (Ensure-JarvisDependencies -Root $Root -ForceUpdate:$UpdateDeps)) {
        Write-Host "Setup automatico fallito: controlla l'output sopra."
        exit 1
    }
}

if ($InstallFlashAttention) {
    if (-not (Install-JarvisFlashAttention -Root $Root -AllowExperimentalWindows:$ForceExperimentalFlashAttentionInstall)) {
        Write-Host "Install FlashAttention fallita o non supportata su questo ambiente."
        Write-Host "Percorso consigliato per FlashAttention stabile: WSL2/Ubuntu o Linux nativo."
    }
}

if (($FlashAttention -or $MaxPerf) -and -not (Test-JarvisPythonModule -Root $Root -ModuleName "flash_attn")) {
    Write-Host "FlashAttention richiesta, ma flash_attn non e installato/importabile: usero SDPA."
}

$SelectedStep = $Step
if ([string]::IsNullOrWhiteSpace($SelectedStep) -and -not [string]::IsNullOrWhiteSpace($StartStep)) {
    $SelectedStep = $StartStep
    Write-Host "StartStep e deprecato: uso come Step singolo richiesto."
}

if ([string]::IsNullOrWhiteSpace($SelectedStep)) {
    $PipelinePreview = Invoke-PipelineJson -CommandArgs @("list")
    $PreviewSteps = @($PipelinePreview.steps | Sort-Object order)

    Write-Host ""
    Write-Host "Nessuno step selezionato. La pipeline non avanza automaticamente."
    Write-Host "Scegli esplicitamente uno step con -Step:"
    foreach ($Item in $PreviewSteps) {
        Write-Host "  $($Item.id) - $($Item.title)"
    }
    Write-Host ""
    Write-Host "Esempio:"
    Write-Host "powershell -ExecutionPolicy Bypass -File .\scripts\jarvis_pipeline.ps1 -FromScratch -Step step_0_prepare"
    exit 1
}

if ($StopAfterStep) {
    Write-Host "StopAfterStep ignorato: ora ogni esecuzione processa un solo step."
}

if ($FromScratch -and $SelectedStep -ne "step_0_prepare") {
    Write-Host "FromScratch puo partire solo da step_0_prepare."
    Write-Host "Comando corretto: powershell -ExecutionPolicy Bypass -File .\scripts\jarvis_pipeline.ps1 -FromScratch -Step step_0_prepare"
    exit 1
}

if ($FromScratch) {
    Write-Host "Ripartenza pipeline da zero: pulisco checkpoint, log, stage e tokenizer."
    Remove-ProjectPath $CheckpointDir
    Remove-ProjectPath $LogsDir
    Remove-ProjectPath $StagesDir
    Remove-ProjectPath $TokenizerDir
}

New-Item -ItemType Directory -Force -Path $LogsDir, $CheckpointDir, $OfficialDir | Out-Null
Invoke-PipelineJson -CommandArgs @("init-db") | Out-Null
$Pipeline = Invoke-PipelineJson -CommandArgs @("list")
$Steps = @($Pipeline.steps | Sort-Object order)
Sync-LatestOfficialPointer -PipelineSteps $Steps

$Stage = Invoke-PipelineJson -CommandArgs @("env", "--step", $SelectedStep)
Set-StageEnvironment -Stage $Stage
$ShouldRebuildStageData = [bool]$RebuildStageData -or [bool]$CleanRestartStage
$ShouldResetActiveCheckpoints = [bool]$ResetActiveCheckpoints -or [bool]$CleanRestartStage

$ExistingStatus = Invoke-PipelineJson -CommandArgs @("status", "--step", $Stage.id)
if (
    -not $FromScratch -and
    $Stage.id -eq "step_0_prepare" -and
    (Test-PipelineStepStarted -Status $ExistingStatus)
) {
    Write-Host "step_0_prepare e gia stato avviato: stato $($ExistingStatus.status)."
    if ($ExistingStatus.started_at) {
        Write-Host "Avvio registrato: $($ExistingStatus.started_at)"
    }
    Write-Host "Blocco la riesecuzione per evitare di ricreare dataset, tokenizer e shard."
    Write-Host "Per azzerare tutto usa:"
    Write-Host "powershell -ExecutionPolicy Bypass -File .\scripts\jarvis_pipeline.ps1 -FromScratch -Step step_0_prepare"
    Write-Host "Alias equivalente: -ResetAll"
    exit 1
}

if (-not $FromScratch) {
    try {
        Assert-ForwardOnlyStage -Stage $Stage -PipelineSteps $Steps
    }
    catch {
        Write-Host $_
        exit 1
    }
}

if ($RebuildSources) {
    Write-Host "RebuildSources attivo: pulisco anche data/raw."
    Remove-ProjectPath $RawDir
}

if (-not $FromScratch -and $ExistingStatus.status -eq "completed" -and -not $ShouldRebuildStageData -and -not $ShouldResetActiveCheckpoints) {
    if ($Stage.kind -eq "prepare" -and -not (Test-TrainingDatasetsReady -PipelineSteps $Steps)) {
        Write-Host "Step prepare gia completato, ma mancano shard tokenizzati: rigenero solo la preparazione dati."
    } else {
        Write-Host "Step gia completato: $($Stage.id)"
        Write-Host "Non rieseguo uno step chiuso senza una ripartenza esplicita."
        exit 0
    }
}

Write-Host ""
Write-Host "========================================"
Write-Host "$($Stage.id) - $($Stage.title)"
Write-Host "$($Stage.objective)"
Write-Host "========================================"

$StageStarted = $false
try {
    Invoke-PipelineJson -CommandArgs @("assert-ready", "--step", $Stage.id) | Out-Null
    Assert-RequiredOfficialCheckpoint -Stage $Stage -PipelineSteps $Steps
    Mark-PipelineStep -StepId $Stage.id -Status "running" -Message "Stage avviato"
    $StageStarted = $true

    if ($Stage.kind -eq "prepare") {
        if (Test-TokenizerReady) {
            Write-Host "Tokenizer gia pronto: salto import/clean/dedup/tokenizer dello step prepare."
        } else {
            Invoke-StageDataBuild -Stage $Stage -BuildTokenizer
        }
        Invoke-TrainingDatasetPreparation -PipelineSteps $Steps
        Set-StageEnvironment -Stage $Stage

        $PrepareCheckpoint = Publish-PrepareCheckpoint -Stage $Stage
        Mark-PipelineStep `
            -StepId $Stage.id `
            -Status "completed" `
            -Checkpoint $PrepareCheckpoint `
            -Message "Tokenizer e dataset tokenizzati pronti"
    } else {
        if ($ShouldRebuildStageData) {
            Write-Host "RebuildStageData attivo: ricostruisco clean/dedup/shard per $($Stage.id)."
            Invoke-StageShardBuild -Stage $Stage
            Set-StageEnvironment -Stage $Stage
        }

        Assert-StageHasPackedData -Stage $Stage

        if ($ShouldResetActiveCheckpoints) {
            Write-Host "ResetActiveCheckpoints attivo: elimino gli snapshot attivi step_* prima del training."
            Clear-ActiveCheckpoints
        }

        if ($Stage.requires -and $Stage.requires -ne "step_0_prepare") {
            if (Test-ActiveCheckpointForStage -StepId $Stage.id) {
                Write-Host "Checkpoint attivo dello stesso stage trovato: riprendo $($Stage.id)."
                $env:JARVIS_RESTORE_WEIGHTS_ONLY = "0"
            } else {
                Restore-OfficialCheckpoint -StepId $Stage.requires
                $env:JARVIS_RESTORE_WEIGHTS_ONLY = "1"
            }
        } elseif ([int]$Stage.order -eq 1) {
            if (Test-ActiveCheckpointForStage -StepId $Stage.id) {
                Write-Host "Checkpoint attivo dello stesso stage trovato: riprendo $($Stage.id)."
            } else {
                Clear-ActiveCheckpoints
            }
            $env:JARVIS_RESTORE_WEIGHTS_ONLY = "0"
        }

        if (-not $NoDashboard) {
            $DashboardProcess = Start-JarvisDashboard -Root $Root -Port $DashboardPort
            if ($DashboardProcess) {
                Write-Host "Dashboard avviata/in uso (PID $($DashboardProcess.Id))."
            }
        }

        $MixedPrecision = Get-JarvisBestMixedPrecision -Root $Root
        $ExitCode = Invoke-JarvisTraining -Root $Root -MixedPrecision $MixedPrecision
        if ($ExitCode -ne 0) {
            throw "Training fallito per $($Stage.id) (exit $ExitCode)."
        }

        $OfficialCheckpoint = Publish-OfficialCheckpoint -Stage $Stage
        Mark-PipelineStep `
            -StepId $Stage.id `
            -Status "completed" `
            -Checkpoint $OfficialCheckpoint `
            -Message "Checkpoint ufficiale pubblicato"
    }
}
catch {
    if ($StageStarted) {
        Mark-PipelineStep -StepId $Stage.id -Status "failed" -Message "$_"
    }
    Write-Host "Stage fallito: $($Stage.id)"
    Write-Host $_
    Write-Host "Riparti dall'ultimo checkpoint ufficiale stabile dopo aver corretto dati o parametri."
    exit 1
}

Write-Host "========================================"
Write-Host "   STEP COMPLETATO: $($Stage.id)"
Write-Host "   La pipeline si ferma qui. Avvia manualmente lo step successivo."
Write-Host "========================================"
