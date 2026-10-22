[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("yelp", "fashion", "beauty")]
    [string]$Dataset,

    [Parameter(Mandatory = $true)]
    [ValidateSet("llmesr_sasrec", "llmesr_bert4rec", "llmesr_gru4rec")]
    [string]$ModelName,

    [int]$Seed = 42,
    [int]$GpuId = 0,
    [int]$NumWorkers = 8,
    [int]$Epochs = 200,
    [ValidateSet("full", "sampled")]
    [string]$CheckpointEvalMode = "full",
    [ValidateSet("full", "sampled")]
    [string]$EvalMode = "sampled",
    [string]$PythonCommand = "python",
    [string]$OutputRoot = "saved_mcrr_base",
    [switch]$NoCrossAttention,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

switch ($Dataset) {
    "yelp" {
        $tsUser = 12
        $tsItem = 13
    }
    "fashion" {
        $tsUser = 3
        $tsItem = 4
    }
    "beauty" {
        $tsUser = 9
        $tsItem = 4
    }
}

$checkPath = "base_seed$Seed"
$resultDir = Join-Path "output/mcrr_base" (Join-Path $Dataset $ModelName)
New-Item -ItemType Directory -Force -Path $resultDir | Out-Null
$runArgs = @(
    "main.py",
    "--dataset", $Dataset,
    "--model_name", $ModelName,
    "--output_dir", $OutputRoot,
    "--check_path", $checkPath,
    "--experiment_variant", $checkPath,
    "--result_json", (Join-Path $resultDir "$checkPath.json"),
    "--rank_output", (Join-Path $resultDir "$checkPath.ranks.npz"),
    "--disable_semantic_rerank",
    "--checkpoint_eval_mode", $CheckpointEvalMode,
    "--eval_mode", $EvalMode,
    "--test_neg", 100,
    "--eval_batch_size", 16,
    "--hidden_size", 64,
    "--train_batch_size", 128,
    "--lr", 0.001,
    "--max_len", 200,
    "--gpu_id", $GpuId,
    "--num_workers", $NumWorkers,
    "--num_train_epochs", $Epochs,
    "--seed", $Seed,
    "--eval_seed", $Seed,
    "--patience", 20,
    "--watch_metric", "NDCG@10",
    "--ts_user", $tsUser,
    "--ts_item", $tsItem,
    "--freeze",
    "--log",
    "--user_sim_func", "kd",
    "--alpha", 0.1
)
if ($ModelName -eq "llmesr_bert4rec") {
    $runArgs += @("--mask_prob", 0.6)
}
if (-not $NoCrossAttention) {
    $runArgs += "--use_cross_att"
}
$runArgs += $ExtraArgs

& $PythonCommand @runArgs
if ($LASTEXITCODE -ne 0) {
    throw "Base training failed with exit code $LASTEXITCODE"
}

$checkpoint = Join-Path $OutputRoot (Join-Path $Dataset (Join-Path $ModelName (Join-Path $checkPath "pytorch_model.bin")))
Write-Host "Base checkpoint: $checkpoint"
