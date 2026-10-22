[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("yelp", "fashion", "beauty")]
    [string]$Dataset,

    [Parameter(Mandatory = $true)]
    [ValidateSet("llmesr_sasrec", "llmesr_bert4rec", "llmesr_gru4rec")]
    [string]$ModelName,

    [Parameter(Mandatory = $true)]
    [string]$SourceCheckpoint,

    [int]$Seed = 42,
    [int]$GpuId = 0,
    [int]$EvalBatchSize = 16,
    [ValidateSet("full", "sampled")]
    [string]$CheckpointEvalMode = "full",
    [ValidateSet("full", "sampled")]
    [string]$EvalMode = "sampled",
    [string]$PythonCommand = "python",
    [string]$OutputRoot = "output/mcrr_ablation",
    [switch]$NoCrossAttention,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
$resolvedCheckpoint = (Resolve-Path -LiteralPath $SourceCheckpoint).Path

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

# C = catalog-mean centering, R = recency pooling,
# Z = candidate-wise standardization.  000 is still a semantic residual model;
# only Base disables the entire MCRR branch.
$variants = @(
    @{ Name = "base"; Flags = @("--disable_semantic_rerank") },
    @{ Name = "mcrr_c0r0z0"; Flags = @("--disable_semantic_centering", "--semantic_pooling", "mean", "--disable_candidate_standardization") },
    @{ Name = "mcrr_c1r0z0"; Flags = @("--semantic_pooling", "mean", "--disable_candidate_standardization") },
    @{ Name = "mcrr_c0r1z0"; Flags = @("--disable_semantic_centering", "--disable_candidate_standardization") },
    @{ Name = "mcrr_c0r0z1"; Flags = @("--disable_semantic_centering", "--semantic_pooling", "mean") },
    @{ Name = "mcrr_c1r1z0"; Flags = @("--disable_candidate_standardization") },
    @{ Name = "mcrr_c1r0z1"; Flags = @("--semantic_pooling", "mean") },
    @{ Name = "mcrr_c0r1z1"; Flags = @("--disable_semantic_centering") },
    @{ Name = "mcrr_c1r1z1_full"; Flags = @() },
    @{ Name = "mcrr_c1r1z1_fixed_w1"; Flags = @("--disable_auto_semantic_weight", "--semantic_rerank_weight", "1.0") }
)

$commonArgs = @(
    "main.py",
    "--do_test",
    "--dataset", $Dataset,
    "--model_name", $ModelName,
    "--source_checkpoint", $resolvedCheckpoint,
    "--output_dir", (Join-Path $OutputRoot "checkpoints"),
    "--eval_batch_size", $EvalBatchSize,
    "--checkpoint_eval_mode", $CheckpointEvalMode,
    "--eval_mode", $EvalMode,
    "--test_neg", 100,
    "--seed", $Seed,
    "--eval_seed", $Seed,
    "--gpu_id", $GpuId,
    "--num_workers", 0,
    "--hidden_size", 64,
    "--max_len", 200,
    "--semantic_pooling", "recency",
    "--semantic_recency_decay", 0.2,
    "--semantic_weight_max", 3.0,
    "--semantic_weight_step", 0.1,
    "--semantic_raw_weight_grid", "0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100",
    "--ts_user", $tsUser,
    "--ts_item", $tsItem,
    "--freeze",
    "--log"
)
if ($ModelName -eq "llmesr_bert4rec") {
    $commonArgs += @("--mask_prob", 0.6)
}
if (-not $NoCrossAttention) {
    $commonArgs += "--use_cross_att"
}

foreach ($variant in $variants) {
    $variantName = $variant.Name
    $resultDir = Join-Path $OutputRoot (Join-Path $Dataset (Join-Path $ModelName "seed$Seed"))
    New-Item -ItemType Directory -Force -Path $resultDir | Out-Null
    $resultPath = Join-Path $resultDir "$variantName.json"
    $rankPath = Join-Path $resultDir "$variantName.ranks.npz"
    $runArgs = $commonArgs + @(
        "--check_path", (Join-Path "ablation" (Join-Path $variantName "seed$Seed")),
        "--experiment_variant", $variantName,
        "--result_json", $resultPath,
        "--rank_output", $rankPath
    ) + $variant.Flags + $ExtraArgs

    Write-Host "Running $Dataset / $ModelName / seed $Seed / $variantName"
    & $PythonCommand @runArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Variant $variantName failed with exit code $LASTEXITCODE"
    }
}

Write-Host "All MCRR variants completed. Results: $OutputRoot"
