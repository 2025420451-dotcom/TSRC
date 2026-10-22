[CmdletBinding()]
param(
    [ValidateSet("yelp", "fashion", "beauty")]
    [string[]]$Datasets = @("yelp", "fashion", "beauty"),

    [ValidateSet("llmesr_sasrec", "llmesr_bert4rec", "llmesr_gru4rec")]
    [string[]]$Models = @(
        "llmesr_sasrec",
        "llmesr_bert4rec",
        "llmesr_gru4rec"
    ),

    [int[]]$Seeds = @(41, 42, 43, 44, 45),
    [string]$SeedsCsv = "",
    [int[]]$GpuIds = @(0),
    [string]$GpuIdsCsv = "",
    [int]$Epochs = 200,
    [int]$NumWorkers = 8,
    [int]$EvalBatchSize = 16,
    [ValidateSet("full", "sampled")]
    [string]$CheckpointEvalMode = "full",
    [ValidateSet("full", "sampled")]
    [string]$EvalMode = "sampled",
    [string]$PythonCommand = "python",
    [string]$BaseOutputRoot = "saved_mcrr_base",
    [string]$AblationOutputRoot = "output/mcrr_ablation",
    [switch]$ForceRetrain,
    [switch]$AblationOnly,
    [switch]$NoCrossAttention,
    [switch]$ContinueOnError,
    [switch]$DryRun,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

if ($SeedsCsv.Trim()) {
    try {
        $Seeds = @(
            $SeedsCsv.Split(",") |
                ForEach-Object { [int]$_.Trim() }
        )
    }
    catch {
        throw "SeedsCsv must contain comma-separated integers"
    }
}
if ($GpuIdsCsv.Trim()) {
    try {
        $GpuIds = @(
            $GpuIdsCsv.Split(",") |
                ForEach-Object { [int]$_.Trim() }
        )
    }
    catch {
        throw "GpuIdsCsv must contain comma-separated integers"
    }
}
if ($Seeds.Count -eq 0) {
    throw "Seeds cannot be empty"
}
if ($GpuIds.Count -eq 0) {
    throw "GpuIds cannot be empty"
}
if ($ForceRetrain -and $AblationOnly) {
    throw "ForceRetrain and AblationOnly cannot be used together"
}

$trainScript = Join-Path $PSScriptRoot "train_mcrr_base.ps1"
$ablationScript = Join-Path $PSScriptRoot "run_mcrr_ablation.ps1"
if (-not (Test-Path -LiteralPath $trainScript)) {
    throw "Missing child script: $trainScript"
}
if (-not (Test-Path -LiteralPath $ablationScript)) {
    throw "Missing child script: $ablationScript"
}

$totalBaseJobs = $Datasets.Count * $Models.Count * $Seeds.Count
$totalEvaluationRuns = $totalBaseJobs * 10
$estimatedNewBaseJobs = 0
$jobIndex = 0
$failures = [System.Collections.Generic.List[object]]::new()

foreach ($dataset in $Datasets) {
    foreach ($model in $Models) {
        foreach ($seed in $Seeds) {
            $checkpoint = Join-Path $BaseOutputRoot (
                Join-Path $dataset (
                    Join-Path $model (
                        Join-Path "base_seed$seed" "pytorch_model.bin"
                    )
                )
            )
            if ($ForceRetrain -or -not (Test-Path -LiteralPath $checkpoint)) {
                $estimatedNewBaseJobs++
            }
        }
    }
}

Write-Host "============================================================"
Write-Host "MCRR complete experiment plan"
Write-Host "Datasets             : $($Datasets -join ', ')"
Write-Host "Models               : $($Models -join ', ')"
Write-Host "Seeds                : $($Seeds -join ', ')"
Write-Host "GPU ids              : $($GpuIds -join ', ')"
Write-Host "Dataset-model-seeds  : $totalBaseJobs"
Write-Host "New Base training    : $estimatedNewBaseJobs"
Write-Host "Ablation evaluations : $totalEvaluationRuns"
Write-Host "Checkpoint selection : $CheckpointEvalMode"
Write-Host "Test / calibration   : $EvalMode"
Write-Host "============================================================"

if ($DryRun) {
    Write-Host "DryRun enabled: no training or evaluation was started."
    exit 0
}

$runStarted = Get-Date
foreach ($dataset in $Datasets) {
    foreach ($model in $Models) {
        foreach ($seed in $Seeds) {
            $gpuId = $GpuIds[$jobIndex % $GpuIds.Count]
            $jobIndex++
            $jobName = "$dataset/$model/seed$seed"
            $checkpoint = Join-Path $BaseOutputRoot (
                Join-Path $dataset (
                    Join-Path $model (
                        Join-Path "base_seed$seed" "pytorch_model.bin"
                    )
                )
            )

            try {
                if ($AblationOnly) {
                    if (-not (Test-Path -LiteralPath $checkpoint)) {
                        throw "AblationOnly requires checkpoint: $checkpoint"
                    }
                    Write-Host "[$jobIndex/$totalBaseJobs] Reusing Base: $jobName"
                }
                elseif ($ForceRetrain -or -not (Test-Path -LiteralPath $checkpoint)) {
                    Write-Host "[$jobIndex/$totalBaseJobs] Training Base: $jobName on GPU $gpuId"
                    $trainArgs = @{
                        Dataset = $dataset
                        ModelName = $model
                        Seed = $seed
                        GpuId = $gpuId
                        NumWorkers = $NumWorkers
                        Epochs = $Epochs
                        CheckpointEvalMode = $CheckpointEvalMode
                        EvalMode = $EvalMode
                        PythonCommand = $PythonCommand
                        OutputRoot = $BaseOutputRoot
                        ExtraArgs = $ExtraArgs
                    }
                    if ($NoCrossAttention) {
                        $trainArgs.NoCrossAttention = $true
                    }
                    & $trainScript @trainArgs
                    if ($LASTEXITCODE -ne 0) {
                        throw "Base training returned exit code $LASTEXITCODE"
                    }
                    if (-not (Test-Path -LiteralPath $checkpoint)) {
                        throw "Training finished but checkpoint is missing: $checkpoint"
                    }
                }
                else {
                    Write-Host "[$jobIndex/$totalBaseJobs] Reusing existing Base: $jobName"
                }

                Write-Host "[$jobIndex/$totalBaseJobs] Running 10 ablations: $jobName"
                $ablationArgs = @{
                    Dataset = $dataset
                    ModelName = $model
                    SourceCheckpoint = $checkpoint
                    Seed = $seed
                    GpuId = $gpuId
                    EvalBatchSize = $EvalBatchSize
                    CheckpointEvalMode = $CheckpointEvalMode
                    EvalMode = $EvalMode
                    PythonCommand = $PythonCommand
                    OutputRoot = $AblationOutputRoot
                    ExtraArgs = $ExtraArgs
                }
                if ($NoCrossAttention) {
                    $ablationArgs.NoCrossAttention = $true
                }
                & $ablationScript @ablationArgs
                if ($LASTEXITCODE -ne 0) {
                    throw "Ablation evaluation returned exit code $LASTEXITCODE"
                }
            }
            catch {
                $failure = [PSCustomObject]@{
                    dataset = $dataset
                    model = $model
                    seed = $seed
                    gpu_id = $gpuId
                    message = $_.Exception.Message
                }
                $failures.Add($failure)
                Write-Error "Experiment failed: $jobName - $($failure.message)" -ErrorAction Continue
                if (-not $ContinueOnError) {
                    throw
                }
            }
        }
    }
}

$resultFiles = Get-ChildItem -LiteralPath $AblationOutputRoot -Filter "*.json" -File -Recurse -ErrorAction SilentlyContinue
$summaryRows = foreach ($resultFile in $resultFiles) {
    try {
        $result = Get-Content -LiteralPath $resultFile.FullName -Raw | ConvertFrom-Json
        [PSCustomObject]@{
            dataset = $result.dataset
            backbone = $result.backbone
            seed = $result.seed
            variant = $result.variant
            mcrr_enabled = $result.mcrr.enabled
            center_C = $result.mcrr.catalog_centering_C
            recency_R = $result.mcrr.recency_pooling_R
            standardize_Z = $result.mcrr.candidate_standardization_Z
            automatic_weight = $result.mcrr.automatic_weight
            selected_weight = $result.mcrr.selected_weight
            ndcg_at_10 = $result.metrics.'NDCG@10'
            hr_at_10 = $result.metrics.'HR@10'
            checkpoint_sha256 = $result.source_checkpoint_sha256
            runtime_seconds = $result.runtime_seconds
            result_file = $resultFile.FullName
        }
    }
    catch {
        Write-Warning "Cannot aggregate result file: $($resultFile.FullName)"
    }
}

New-Item -ItemType Directory -Force -Path $AblationOutputRoot | Out-Null
$summaryPath = Join-Path $AblationOutputRoot "mcrr_ablation_summary.csv"
$summaryRows |
    Sort-Object dataset, backbone, seed, variant |
    Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding UTF8

if ($failures.Count -gt 0) {
    $failurePath = Join-Path $AblationOutputRoot "failed_jobs.csv"
    $failures | Export-Csv -LiteralPath $failurePath -NoTypeInformation -Encoding UTF8
    Write-Warning "$($failures.Count) jobs failed. Details: $failurePath"
}

$elapsed = (Get-Date) - $runStarted
Write-Host "============================================================"
Write-Host "All requested MCRR experiments finished."
Write-Host "Successful result files: $($summaryRows.Count)"
Write-Host "Failed jobs           : $($failures.Count)"
Write-Host "Elapsed               : $elapsed"
Write-Host "Summary               : $summaryPath"
Write-Host "============================================================"
