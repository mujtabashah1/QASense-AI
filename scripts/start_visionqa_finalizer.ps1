param(
    [Parameter(Mandatory = $true)]
    [int]$TrainingPid
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Finalizer = Join-Path $ProjectRoot "scripts\finalize_visionqa_after_training.ps1"
$LogDirectory = Join-Path $ProjectRoot "runs\visionqa\process_logs"
$Stdout = Join-Path $LogDirectory "finalizer_launcher.stdout.log"
$Stderr = Join-Path $LogDirectory "finalizer_launcher.stderr.log"
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null

$Process = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $Finalizer,
        "-TrainingPid", $TrainingPid
    ) `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr `
    -PassThru

[ordered]@{
    finalizer_pid = $Process.Id
    training_pid = $TrainingPid
    detached = $true
    started_utc = (Get-Date).ToUniversalTime().ToString("o")
    status_file = (Join-Path $ProjectRoot "reports\visionqa\metrics\postprocess_status.json")
    stdout_log = $Stdout
    stderr_log = $Stderr
} | ConvertTo-Json -Depth 4
