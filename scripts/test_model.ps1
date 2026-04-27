param(
    [switch]$SkipDependencyCheck,
    [switch]$Cli,
    [switch]$NoBrowser,
    [int]$DashboardPort = 8765
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

if ($Cli) {
    Push-Location $Root
    try {
        & $Python scripts/test_model.py
        $ExitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
} else {
    $TestModelUrl = "http://127.0.0.1:$DashboardPort/test-model"

    Write-Host ""
    Write-Host "Avvio dashboard Next.js per il test modello"
    Write-Host "URL: $TestModelUrl"
    Write-Host "Per la vecchia console: powershell -ExecutionPolicy Bypass -File .\scripts\test_model.ps1 -Cli"

    if (-not $NoBrowser) {
        $EscapedUrl = $TestModelUrl.Replace("'", "''")
        Start-Process `
            -FilePath "powershell" `
            -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "Start-Sleep -Seconds 3; Start-Process '$EscapedUrl'") `
            -WindowStyle Hidden | Out-Null
    }

    Start-JarvisDashboard -Root $Root -Port $DashboardPort -Foreground | Out-Null
    $ExitCode = 0
}

Write-Host "========================================"
Write-Host "   MODEL TEST FINISHED"
Write-Host "========================================"

exit $ExitCode
