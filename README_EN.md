# TSRC: Test-Time Semantic Residual Calibration for Long-Tail Sequential Recommendation

This repository accompanies the paper **Unlocking LLM Semantics: Test-Time Semantic Residual Calibration for Long-Tail Sequential Recommendation**.

## Overview

TSRC leverages offline item semantics to calibrate the scores of sequential recommendation models at test time, providing supplementary recommendation signals for long-tail items and users with short interaction histories.

- **No additional training required for TSRC**: Train an LLM-ESR backbone first, then freeze the backbone and rerank candidates at test time.
- **No online LLM calls**: Use cached item semantic embeddings directly.
- **Recency-aware interests**: Aggregate user interests according to the recency of historical interactions.
- **Multiple backbone architectures**: Compatible with LLM-ESR using SASRec, BERT4Rec, or GRU4Rec as the sequence encoder.

![TSRC architecture](framework.jpeg)

The semantic branch centers and normalizes item embeddings, then computes matching scores based on recent user interests. The semantic matching scores and backbone scores are standardized separately and combined through residual fusion to rerank candidates. The fusion weight is selected on the validation set and kept fixed during testing.

## Datasets

- **Datasets**: Yelp, Amazon Fashion, and Amazon Beauty.
- **Download**: [Preprocessed datasets](https://drive.google.com/file/d/1MpBUjCDLiFIEODTnopSCzDAnS8RzO9aV/view?usp=sharing), provided by the [official LLM-ESR repository](https://github.com/Applied-Machine-Learning-Lab/LLM-ESR).
- **Data directories**: `data/yelp/handled/`, `data/fashion/handled/`, and `data/beauty/handled/`.
- **Required files**: `inter.txt`, `itm_emb_np.pkl`, `pca64_itm_emb_np.pkl`, and `sim_user_100.pkl`.

## Environment

The reference experimental environment is listed below. See [environment.yml](environment.yml) for dependency versions.

| Component | Configuration |
|---|---|
| Operating system | Windows x64 |
| CPU / RAM | AMD Ryzen 7 9800X3D / 16 GB |
| GPU / Driver | NVIDIA RTX 5060 Ti 16 GB / 610.62 |
| Python / pip | 3.10.20 / 26.1.2 |
| PyTorch / CUDA runtime | 2.11.0+cu128 / 12.8 |
| NumPy / pandas / tqdm | 1.24.3 / 2.0.1 / 4.65.0 |
| TensorBoard | 2.20.0 |

## Paper Hyperparameters

The following settings correspond to Sections 4.3 and 4.6 of the paper. The evaluation modes used for checkpoint selection and testing can be configured independently through command-line arguments.

| Parameter | Setting |
|---|---|
| Offline item semantics | text-embedding-ada-002, 1536 dimensions; the TSRC semantic cache remains frozen |
| Backbone hidden dimension per view | 64 |
| Optimizer / learning rate | Adam / 0.001 |
| Training batch size | 128 |
| Maximum sequence length | 200 |
| Maximum epochs / early-stopping patience | 200 / 20 |
| Backbone checkpoint selection | Validation `NDCG@10`; TSRC disabled during backbone training |
| Recency decay coefficient γ | 0.2 |
| Numerical floor for vector normalization δ | 10⁻⁸ |
| Numerical floor for candidate score standardization ε | 10⁻⁶; population standard deviation |
| Fusion weight λ for full TSRC | 0.0 to 3.0 in steps of 0.1; selected by validation `NDCG@10`, choosing the smaller value in case of a tie |
| λ grid without score standardization | `{0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100}` |
| Fixed-weight ablation | λ=1; all other full TSRC settings unchanged |
| Sampled evaluation candidate set | 1 ground-truth target + 100 negatives sampled uniformly without replacement, excluding the complete input history and ground-truth target |
| Random seeds for the full experiments | 41, 42, 43, 44, 45; evaluation sampling seeds are determined by the run seed |

Long-tail grouping thresholds are fixed across all backbones and ablation configurations for each dataset. Users or items with values strictly below the corresponding threshold are assigned to the tail group.

| Dataset | User history length threshold | Item interaction-user count threshold |
|---|---:|---:|
| Yelp | 12 | 13 |
| Fashion | 3 | 4 |
| Beauty | 9 | 4 |

## Running the Experiments

Run the following commands from the **repository root directory (the directory containing `main.py`)**. First, create the environment and download and place the four required preprocessed files for the selected dataset as described above. The commands train the backbone from scratch and then automatically run the tests.

### Create the Environment

```powershell
conda env create -f environment.yml
conda activate tsrc
```

### Single Configuration: Fashion–GRU4Rec, with Automatic Testing After Training

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\run_all_mcrr_experiments.ps1 -Datasets fashion -Models llmesr_gru4rec -SeedsCsv "42" -GpuIdsCsv "0" -NumWorkers 0 -Epochs 200 -CheckpointEvalMode full -EvalMode sampled -ForceRetrain
```

This command performs the following steps in sequence:

1. Trains the Fashion–GRU4Rec backbone from scratch for up to 200 epochs, with early stopping if the validation metric does not improve for 20 consecutive epochs. The best checkpoint is saved based on full-catalog validation `NDCG@10`.
2. Uses **the best checkpoint produced by this training run** to automatically evaluate the Base model, full TSRC, and the ablation configurations. The TSRC fusion weight is selected on the validation set.
3. Saves test metrics, per-user rankings, and summary CSV files. Testing uses a sampled evaluation protocol with one positive item and 100 negative items per user.

The `-ForceRetrain` flag ensures that training is performed again on every run, even if a checkpoint with the same name already exists locally. Once the environment and data are ready, you can run the full workflow using this single command.

### All Configurations: 3 Datasets × 3 Backbones × 5 Random Seeds

Prepare all three datasets (Yelp, Fashion, and Beauty) before running:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\run_all_mcrr_experiments.ps1 -SeedsCsv "41,42,43,44,45" -GpuIdsCsv "0" -NumWorkers 0 -Epochs 200 -CheckpointEvalMode full -EvalMode sampled -ForceRetrain
```

This command trains 45 backbones from scratch in sequence. After each backbone has been trained, it automatically evaluates the Base model and nine TSRC configurations.

### Choose the Evaluation Modes

The two options are independent and both support `full` (full-catalog evaluation) and `sampled` (sampled evaluation):

| Purpose | PowerShell argument | `main.py` argument | Default |
|---|---|---|---|
| Validation during training and backbone checkpoint selection | `-CheckpointEvalMode` | `--checkpoint_eval_mode` | `full` |
| TSRC fusion-weight selection on validation data and final testing | `-EvalMode` | `--eval_mode` | `sampled` |

For full-catalog testing, change `-EvalMode sampled` to `-EvalMode full` in the command. To select backbone checkpoints using sampled validation, change `-CheckpointEvalMode full` to `-CheckpointEvalMode sampled`. Test data are never used to select either the backbone checkpoint or the fusion weight.

### Output Locations

- **Best checkpoint from the current training run**: `saved_mcrr_base/<dataset>/<model>/base_seed<seed>/pytorch_model.bin`.
- **Full TSRC test metrics**: `output/mcrr_ablation/<dataset>/<model>/seed<seed>/mcrr_c1r1z1_full.json`.
- **Full TSRC per-user rankings**: `mcrr_c1r1z1_full.ranks.npz` in the same directory.
- **Summary of all test results**: `output/mcrr_ablation/mcrr_ablation_summary.csv`.
- **Training and testing logs**: `log/`.

The code is based on the [official LLM-ESR repository](https://github.com/Applied-Machine-Learning-Lab/LLM-ESR) and retains the original MIT license. See [LICENSE](LICENSE).
