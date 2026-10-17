param(
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv-visionqa\Scripts\python.exe"
$Controller = Join-Path $ProjectRoot "scripts\run_visionqa_v2_experiment.py"
$ReportDir = Join-Path $ProjectRoot "reports\visionqa\v2"

if (-not (Test-Path -LiteralPath $Python)) { throw "VisionQA Python is missing: $Python" }
if (-not (Test-Path -LiteralPath $Controller)) { throw "Training controller is missing: $Controller" }
New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null

if ($Foreground) {
    & $Python $Controller
    exit $LASTEXITCODE
}

$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$Stdout = Join-Path $ReportDir "controller-$Timestamp.stdout.log"
$Stderr = Join-Path $ReportDir "controller-$Timestamp.stderr.log"
$Process = Start-Process -FilePath $Python `
    -ArgumentList @($Controller) `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr `
    -PassThru

@{
    process_id = $Process.Id
    started_at = (Get-Date).ToString("o")
    stdout = $Stdout
    stderr = $Stderr
    controller = $Controller
    note = "Re-running this launcher resumes the latest complete per-epoch checkpoint for an interrupted stage."
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $ReportDir "controller_process.json") -Encoding utf8

Get-Content -Raw -LiteralPath (Join-Path $ReportDir "controller_process.json")
