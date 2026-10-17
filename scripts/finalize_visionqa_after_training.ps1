param(
    [Parameter(Mandatory = $true)]
    [int]$TrainingPid
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $ProjectRoot ".venv-visionqa\Scripts\python.exe"
$MetricsDirectory = Join-Path $ProjectRoot "reports\visionqa\metrics"
$StatePath = Join-Path $MetricsDirectory "postprocess_status.json"
$MasterLog = Join-Path $ProjectRoot "runs\visionqa\process_logs\postprocess.log"
$StableCheckpoint = Join-Path $ProjectRoot "models\visionqa\atomblock_yolo11n\best.pt"
$SourceCheckpoint = Join-Path $ProjectRoot "runs\visionqa\atomblock_yolo11n_main_b2\weights\best.pt"
$CompletedSteps = [System.Collections.Generic.List[string]]::new()

function Write-State {
    param([string]$Status, [string]$CurrentStep = "", [string]$ErrorMessage = "")
    $Payload = [ordered]@{
        status = $Status
        updated_utc = (Get-Date).ToUniversalTime().ToString("o")
        training_pid = $TrainingPid
        current_step = $CurrentStep
        completed_steps = @($CompletedSteps)
        log = $MasterLog
        error = $ErrorMessage
    }
    $Temporary = "$StatePath.tmp"
    $Payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $Temporary -Encoding UTF8
    Move-Item -LiteralPath $Temporary -Destination $StatePath -Force
}

function Invoke-Step {
    param([string]$Name, [string[]]$Arguments)
    Write-State -Status "running" -CurrentStep $Name
    $StepOut = Join-Path $env:TEMP "visionqa_${Name}_$PID.stdout.log"
    $StepErr = Join-Path $env:TEMP "visionqa_${Name}_$PID.stderr.log"
    $Process = Start-Process `
        -FilePath $Python `
        -ArgumentList $Arguments `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $StepOut `
        -RedirectStandardError $StepErr `
        -Wait `
        -PassThru
    Add-Content -LiteralPath $MasterLog -Value "`n===== $Name stdout ====="
    if (Test-Path -LiteralPath $StepOut) {
        Get-Content -LiteralPath $StepOut | Add-Content -LiteralPath $MasterLog
        Remove-Item -LiteralPath $StepOut -Force
    }
    Add-Content -LiteralPath $MasterLog -Value "`n===== $Name stderr ====="
    if (Test-Path -LiteralPath $StepErr) {
        Get-Content -LiteralPath $StepErr | Add-Content -LiteralPath $MasterLog
        Remove-Item -LiteralPath $StepErr -Force
    }
    if ($Process.ExitCode -ne 0) {
        throw "$Name failed with exit code $($Process.ExitCode). See $MasterLog"
    }
    $CompletedSteps.Add($Name)
    Write-State -Status "running" -CurrentStep $Name
}

New-Item -ItemType Directory -Path $MetricsDirectory -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path $MasterLog) -Force | Out-Null
Write-State -Status "waiting_for_training" -CurrentStep "training"

try {
    if (Get-Process -Id $TrainingPid -ErrorAction SilentlyContinue) {
        Wait-Process -Id $TrainingPid
    }
    Start-Sleep -Seconds 5
    $TrainingStatus = Get-Content -LiteralPath (Join-Path $MetricsDirectory "training_status.json") -Raw | ConvertFrom-Json
    if ($TrainingStatus.status -ne "completed") {
        throw "Training process ended with status '$($TrainingStatus.status)'. Final evaluation was not started."
    }
    if (-not (Test-Path -LiteralPath $SourceCheckpoint)) {
        throw "Validation-selected checkpoint is missing: $SourceCheckpoint"
    }

    Invoke-Step -Name "live_snapshot" -Arguments @("scripts\get_visionqa_training_status.py")
    Invoke-Step -Name "progress_review" -Arguments @("scripts\review_visionqa_training_progress.py")
    Invoke-Step -Name "promote" -Arguments @(
        "scripts\promote_visionqa_model.py", "--source", $SourceCheckpoint
    )
    Invoke-Step -Name "test_evaluation" -Arguments @(
        "scripts\evaluate_visionqa_yolo.py",
        "--checkpoint", $StableCheckpoint,
        "--split", "test",
        "--batch", "2",
        "--imgsz", "640",
        "--max-det", "1500",
        "--name", "atomblock_yolo11n_test"
    )
    Invoke-Step -Name "loader_summary" -Arguments @("scripts\summarize_visionqa_loader.py")
    Invoke-Step -Name "visual_examples" -Arguments @(
        "scripts\generate_visionqa_examples.py", "--checkpoint", $StableCheckpoint
    )

    $SampleImage = Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "datasets\processed\visionqa_atomblock\test\images") -File | Select-Object -First 1
    if (-not $SampleImage) {
        throw "No held-out test image is available for inference verification."
    }
    Invoke-Step -Name "foundation_check" -Arguments @(
        "scripts\check_visionqa_foundation.py",
        "--checkpoint", $StableCheckpoint,
        "--sample-image", $SampleImage.FullName
    )
    Invoke-Step -Name "inference_sample" -Arguments @(
        "scripts\visionqa_inference.py",
        $SampleImage.FullName,
        "--checkpoint", $StableCheckpoint,
        "--output", (Join-Path $MetricsDirectory "inference_sample.json")
    )
    Invoke-Step -Name "final_report" -Arguments @("scripts\build_visionqa_report.py")
    Write-State -Status "completed"
}
catch {
    Write-State -Status "failed" -CurrentStep "postprocess" -ErrorMessage $_.Exception.Message
    Add-Content -LiteralPath $MasterLog -Value "`nFAILED: $($_.Exception.Message)"
    exit 1
}

