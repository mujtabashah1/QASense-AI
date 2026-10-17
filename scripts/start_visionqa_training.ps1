param(
    [string]$Checkpoint = "runs\visionqa\atomblock_yolo11n_main_b2\weights\last.pt",
    [string]$MetricsOutput = "reports\visionqa\metrics\training_status.json"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $ProjectRoot ".venv-visionqa\Scripts\python.exe"
$TrainingScript = Join-Path $ProjectRoot "scripts\train_visionqa_yolo.py"
$CheckpointPath = (Resolve-Path (Join-Path $ProjectRoot $Checkpoint)).Path
$MetricsPath = Join-Path $ProjectRoot $MetricsOutput
$LogDirectory = Join-Path $ProjectRoot "runs\visionqa\process_logs"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "VisionQA Python executable not found: $Python"
}
if (-not (Test-Path -LiteralPath $TrainingScript)) {
    throw "Training script not found: $TrainingScript"
}

New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$StdoutLog = Join-Path $LogDirectory "main_resume_${Timestamp}.stdout.log"
$StderrLog = Join-Path $LogDirectory "main_resume_${Timestamp}.stderr.log"

$Arguments = @(
    $TrainingScript,
    "--resume", $CheckpointPath,
    "--device", "0",
    "--workers", "2",
    "--metrics-output", $MetricsPath
)

$Process = Start-Process `
    -FilePath $Python `
    -ArgumentList $Arguments `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $StdoutLog `
    -RedirectStandardError $StderrLog `
    -PassThru

# The Python app launcher may create a separate pythoncore process. Capture the
# CUDA compute PID as well as the launcher PID so status checks remain reliable.
Start-Sleep -Seconds 5
$ComputePids = @(
    & nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>$null |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ -match '^\d+$' } |
        ForEach-Object { [int]$_ }
)

$ProcessState = [ordered]@{
    pid = $Process.Id
    cuda_process_ids = $ComputePids
    status = "started"
    detached = $true
    started_utc = (Get-Date).ToUniversalTime().ToString("o")
    project_root = $ProjectRoot
    checkpoint = $CheckpointPath
    metrics_output = $MetricsPath
    stdout_log = $StdoutLog
    stderr_log = $StderrLog
    resume_command = ".\.venv-visionqa\Scripts\python.exe scripts\train_visionqa_yolo.py --resume runs\visionqa\atomblock_yolo11n_main_b2\weights\last.pt --device 0 --workers 2 --metrics-output reports\visionqa\metrics\training_status.json"
}

$ProcessStatePath = Join-Path $ProjectRoot "reports\visionqa\metrics\training_process.json"
$ProcessState | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $ProcessStatePath -Encoding UTF8
$ProcessState | ConvertTo-Json -Depth 4
