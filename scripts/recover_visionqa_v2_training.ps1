$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ReportDir = Join-Path $ProjectRoot "reports\visionqa\v2"
$StatusPath = Join-Path $ReportDir "training_status.json"
$LockPath = Join-Path $ReportDir "external_recovery_lock.json"
$Python = Join-Path $ProjectRoot ".venv-visionqa\Scripts\python.exe"
$Controller = Join-Path $ProjectRoot "scripts\run_visionqa_v2_experiment.py"
$TaskName = "QASense-VisionQA-v2-Recovery"

if (Test-Path -LiteralPath $StatusPath) {
    $Status = Get-Content -Raw -LiteralPath $StatusPath | ConvertFrom-Json
    if ($Status.status -in @("completed", "stopped")) {
        schtasks.exe /Delete /TN $TaskName /F 2>$null | Out-Null
        exit 0
    }
}

if (Test-Path -LiteralPath $LockPath) {
    $Lock = Get-Content -Raw -LiteralPath $LockPath | ConvertFrom-Json
    foreach ($ProcessId in @($Lock.process_ids)) {
        if (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) {
            exit 0
        }
    }
}

@{
    process_ids = @($PID)
    started_at = (Get-Date).ToString("o")
    mode = "scheduled_recovery"
} | ConvertTo-Json | Set-Content -LiteralPath $LockPath -Encoding utf8

try {
    & $Python $Controller
    exit $LASTEXITCODE
}
finally {
    if (Test-Path -LiteralPath $StatusPath) {
        $FinalStatus = Get-Content -Raw -LiteralPath $StatusPath | ConvertFrom-Json
        if ($FinalStatus.status -in @("completed", "stopped")) {
            schtasks.exe /Delete /TN $TaskName /F 2>$null | Out-Null
        }
    }
}
