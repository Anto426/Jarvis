param(
    [switch]$SkipDependencyCheck
)

Write-Host "========================================"
Write-Host "   JARVIS MODEL TEST"
Write-Host "========================================"

. "$PSScriptRoot\lib\jarvis.ps1"

$Root = Get-JarvisRoot
Initialize-JarvisEnvironment -Root $Root

if (-not $SkipDependencyCheck) {
    if (-not (Ensure-JarvisDependencies -Root $Root)) {
        Write-Host "Setup automatico fallito: controlla l'output sopra."
        exit 1
    }
}

$Python = Get-JarvisPython -Root $Root
if ($Python -eq "python") {
    Write-Host "Python venv non trovato. Esegui: powershell -ExecutionPolicy Bypass -File .\scripts\jarvis.ps1 -UpdateDeps"
    exit 1
}

Push-Location $Root
try {
    & $Python scripts/test_model.py
    $ExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

Write-Host "========================================"
Write-Host "   MODEL TEST FINISHED"
Write-Host "========================================"

exit $ExitCode
