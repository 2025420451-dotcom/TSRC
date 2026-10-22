[CmdletBinding()]
param(
    [ValidateSet("yelp", "fashion", "beauty")]
    [string[]]$Datasets = @("yelp", "fashion", "beauty"),

    [ValidateSet("llmesr_sasrec", "llmesr_bert4rec", "llmesr_gru4rec")]
    [string[]]$Models = @("llmesr_sasrec", "llmesr_bert4rec", "llmesr_gru4rec"),

    [string]$SeedsCsv = "41,42,43,44,45",
    [string]$GpuIdsCsv = "0",
    [int]$EvalBatchSize = 16,
    [string]$PythonCommand = "python",
    [string]$CheckpointRoot = "saved_mcrr_base",
    [string]$OutputRoot = "log(重新评测)",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

try {
    $seeds = @($SeedsCsv.Split(",") | ForEach-Object { [int]$_.Trim() })
}
catch {
    throw "SeedsCsv must contain comma-separated integers"
}
if (-not $SeedsCsv.Trim() -or $seeds.Count -eq 0) {
    throw "SeedsCsv cannot be empty"
}
if (@($seeds | Select-Object -Unique).Count -ne $seeds.Count) {
    throw "SeedsCsv contains duplicate seeds"
}
if (-not [System.IO.Path]::IsPathRooted($CheckpointRoot)) {
    $CheckpointRoot = Join-Path $repoRoot $CheckpointRoot
}
if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
    $OutputRoot = Join-Path $repoRoot $OutputRoot
}
Get-Command $PythonCommand -ErrorAction Stop | Out-Null

# Check every source before starting. This entry point never trains a model.
foreach ($dataset in $Datasets) {
    foreach ($model in $Models) {
        foreach ($seed in $seeds) {
            $checkpoint = Join-Path $CheckpointRoot "$dataset/$model/base_seed$seed/pytorch_model.bin"
            if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
                throw "Missing checkpoint; reevaluation will not train a replacement: $checkpoint"
            }
        }
    }
}

# A new directory keeps earlier logs/results intact, including prior reruns.
$runRoot = Join-Path $OutputRoot (Get-Date -Format "yyyyMMdd_HHmmss_fff")
$textLogRoot = Join-Path $runRoot "logs"
$resultRoot = Join-Path $runRoot "results"
Write-Host "Reevaluation output: $runRoot"
Write-Host "Protocol: sampled, 1 positive + 100 negatives per user"
Write-Host "Checkpoints are read only; every checkpoint runs Base + 9 MCRR variants."

$runArgs = @{
    Datasets = $Datasets
    Models = $Models
    SeedsCsv = ($seeds -join ",")
    GpuIdsCsv = $GpuIdsCsv
    NumWorkers = 0
    EvalBatchSize = $EvalBatchSize
    EvalMode = "sampled"
    PythonCommand = $PythonCommand
    BaseOutputRoot = $CheckpointRoot
    AblationOutputRoot = $resultRoot
    AblationOnly = $true
    ExtraArgs = @("--log_root", $textLogRoot, "--test_neg", "100")
    DryRun = [bool]$DryRun
}
& (Join-Path $PSScriptRoot "run_all_mcrr_experiments.ps1") @runArgs
if ($LASTEXITCODE) {
    throw "Reevaluation returned exit code $LASTEXITCODE"
}
