# Onboarding Guide for Future AI Agents

## Project Summary
This project implements a **Diffusion-based Time Series Forecasting (TSF)** model, inspired by the ViTime paper. It treats time series forecasting as a computer vision problem by converting 1D time series into 2D "stripe" images and using a U-Net backbone with DDPM/DDIM diffusion to generate future predictions.

## Key Goals
- **Multivariate Forecasting:** Supports forecasting multiple variables simultaneously.
- **Universal Pre-training:** Pretrain on 1M synthetic samples (7 variates), then fine-tune on any dataset.
- **Channel Clustering (CCM):** For datasets with >7 variates, uses Channel Clustering Module to cluster channels into 7 "super-channels" compatible with pretrained model.
- **Rich Synthetic Data:** Uses a sophisticated multivariate augmentation system to generate realistic synthetic pre-training data.

## File Structure & Key Components

### Models
- `models/diffusion_tsf/`: Core diffusion model logic.
    - `diffusion_model.py`: Main `DiffusionTSF` class. Handles 2D encoding/decoding, forward pass, generation.
    - `unet.py`: Conditional U-Net backbone (2D).
    - `ccm_adapter.py`: **Channel Clustering Module** - clusters >7 variates into 7 super-channels for pretrained model.
    - `dataset.py`: `ElectricityDataset` for real data and `get_synthetic_dataloader`.
    - `realts.py`: Synthetic data generator classes.
    - `train_universal_v2.py`: **Primary Entry Point**. Pretrain on synthetic → fine-tune on all datasets.
    - `train_electricity.py`: Single-dataset training with Optuna HP search.
    - `config.py`: Configuration dataclasses.
    - `tests/test_ccm_adapter.py`: Unit tests for CCM module.

- `models/iTransformer/`: iTransformer model (Stage 1 predictor/guidance).
- `models/TimeSeriesCCM-main/`: Reference implementation of CCM paper.

### Scripts
- `run_all_datasets.sh`: **Master shell script** - runs full pipeline on all 9 datasets.
    ```bash
    ./run_all_datasets.sh --smoke-test    # Quick validation
    ./run_all_datasets.sh                  # Full training (all datasets)
    ./run_all_datasets.sh --dataset ETTh1  # Single dataset
    ```
- `train_universal_v2.py`: Python entry point.
    ```bash
    python -m models.diffusion_tsf.train_universal_v2 --smoke-test
    python -m models.diffusion_tsf.train_universal_v2 --mode pretrain --synthetic-samples 1000000
    python -m models.diffusion_tsf.train_universal_v2 --mode finetune --dataset electricity
    ```

## Architecture Highlights

### 1. Unified L+F Channel Scheme
Instead of stacking "guidance" and "lookback" as separate channels with misaligned time axes, we now use a single time axis of length `Lookback + Forecast`.
- **Input Canvas:** Shape `(Batch, Channels, Height, L+F)`.
- **Past Part (0 to L):** Contains the ground truth past data.
- **Future Part (L to L+F):** Contains the noisy future (during training/inference).
- **Auxiliary Channels:** Coordinate grids, time ramps, and sine waves are injected across the full `L+F` width.
- **Guidance:** Stage 1 forecasts are converted to 2D and placed in the "Future" part of the canvas.

### 2. Multivariate Augmentation
Located in `models/diffusion_tsf/augmentation.py`.
- Generates synthetic multivariate time series by coupling independent random processes.
- **Updated (2026-01-27):** Now uses "Organic" generators (Random Walks, Inverse FFT, Seasonal patterns, etc. from `realts.py`) as the base behavior for each variable, ensuring realistic textures.
- Uses "Impact Functions" to model causal effects between variables.
- Used by `RealTS` dataset when `num_variables > 1`.
- **Visualization:** Use `models/diffusion_tsf/visualize_synthetic.py` to generate sample plots of the synthetic data.

### 3. Channel Clustering Module (CCM)
For datasets with >7 variates (electricity=321, traffic=862, weather=21, exchange_rate=8):
- **ClusterAssigner**: Projects channels to embedding space, computes similarity to 7 cluster prototypes
- **Aggregate**: Weighted average of channels per cluster → (B, 7, L)
- **Expand**: Distribute predictions back using learned cluster probabilities
- **ClusterLoss**: Encourages similar channels to cluster together (RBF similarity + sinkhorn)

### Dataset Categories
| Category | Datasets | Approach |
|----------|----------|----------|
| 7-variate | ETTh1/h2/m1/m2, illness | Direct fine-tune |
| >7-variate | electricity, weather, exchange_rate, traffic | CCM → 7 clusters |

**Note:** `traffic.csv` is auto-combined from `traffic_part1.csv` + `traffic_part2.csv` by the shell script.

## Gotchas & Tips
- **Module Imports:** Run from project root: `python -m models.diffusion_tsf.script_name`
- **Smoke Test:** Always run `--smoke-test` before full training to verify pipeline.
- **OOM Issues:** Use `attention_levels=[2]` (bottleneck only) and small batch sizes.
- **CCM for small variates:** CCM adapter rejects datasets with ≤7 variates (use direct fine-tuning).

## Recent Changes
- **2026-02-04:** Added CCM adapter for >7-variate datasets, `train_universal_v2.py`, `run_all_datasets.sh`
- Renamed `model.py` to `diffusion_model.py` to avoid conflict with `iTransformer/model`.
- Implemented `train_universal.py` (legacy 4-phase approach).

## 7-Variate Pipeline (NEW)

Alternative to CCM approach: train separate models on non-overlapping 7-variate subsets.

### Scripts
- `run_7var_pipeline.sh`: Master shell script for 7-variate training
    ```bash
    ./run_7var_pipeline.sh --smoke-test    # Quick validation
    ./run_7var_pipeline.sh                  # Full pipeline (train + eval)
    ./run_7var_pipeline.sh --train          # Training only
    ./run_7var_pipeline.sh --evaluate       # Evaluation only
    ./run_7var_pipeline.sh --resume         # Resume interrupted training
    ./run_7var_pipeline.sh --status         # Show progress
    ```

- `train_7var_pipeline.py`: Training script
    ```bash
    python -m models.diffusion_tsf.train_7var_pipeline --smoke-test
    python -m models.diffusion_tsf.train_7var_pipeline --resume
    python -m models.diffusion_tsf.train_7var_pipeline --list-subsets
    python -m models.diffusion_tsf.train_7var_pipeline --only traffic-5
    ```

- `evaluate_7var.py`: Standalone evaluation (can run anytime on completed models)
    ```bash
    python -m models.diffusion_tsf.evaluate_7var --smoke-test
    python -m models.diffusion_tsf.evaluate_7var --n-samples 50
    ```

### How it Works
1. **Pretrain once** on 1M synthetic 7-variate samples
2. **Fine-tune** on each dataset:
   - 7-variate datasets (ETTh1/h2/m1/m2, illness): Direct fine-tune
   - >7-variate datasets: Random shuffle → partition into non-overlapping 7-variate subsets
     - electricity (321 vars) → 45 subsets: electricity-0, ..., electricity-44
     - traffic (862 vars) → 123 subsets: traffic-0, ..., traffic-122
     - weather (21 vars) → 3 subsets: weather-0, weather-1, weather-2
     - exchange_rate (8 vars) → 1 subset: exchange_rate-0
3. **Evaluate** with single sample AND averaged (30 samples) metrics

### Key Features
- **Resumable:** Progress saved in `training_manifest.json`. Ctrl+C safe.
- **Variate tracking:** Each subset's variate indices saved in `metadata.json`
- **Separate eval:** Run evaluation anytime on completed models
- **Early stopping:** patience=25, max_epochs=200

### Output Structure
```
checkpoints_7var/
├── pretrained_7var.pt
├── training_manifest.json
├── ETTh1/
│   ├── best.pt
│   ├── latest.pt
│   └── metadata.json
├── traffic-0/
│   ├── best.pt
│   └── metadata.json
...

results_7var/
├── summary.csv
├── all_results.json
└── {subset_id}_results.json
```

### Syncing Results to Local Machine

After starting training on remote GPU, sync results back to local:

```bash
# On LOCAL machine (not the GPU server)
./sync_results.sh user@gpu-server           # Use default remote path
./sync_results.sh user@gpu-server /path     # Custom remote path
```

This syncs:
- `results_7var/` - All evaluation results and summary.csv
- `checkpoints_7var/*.json` - Metadata only (NOT .pt weights)
- Training logs

**Results are saved incrementally** - you can sync while training is still running to see progress.

## 7-Variate Pipeline with Multi-GPU Support

### New Training Pipeline: `train_7var_pipeline.py`
A complete pipeline for training on all datasets with HP tuning:

```bash
# Single GPU
python -m models.diffusion_tsf.train_7var_pipeline --smoke-test  # Quick test
python -m models.diffusion_tsf.train_7var_pipeline --resume      # Full training

# Multi-GPU (DDP)
torchrun --nproc_per_node=4 -m models.diffusion_tsf.train_7var_pipeline --ddp
torchrun --nproc_per_node=2 -m models.diffusion_tsf.train_7var_pipeline --ddp --resume
```

### Pipeline Phases
1. **Phase 1A:** iTransformer HP Tuning (20 trials on 100k synthetic samples)
2. **Phase 1C-1:** Full iTransformer Pretraining (1M samples, 200 epochs)
3. **Phase 1B:** Diffusion HP Tuning with iTransformer guidance (8 trials, 10k samples)
4. **Phase 1C-2:** Full Diffusion Pretraining (1M samples, 200 epochs)
5. **Phase 2:** Per-dataset HP tuning + fine-tuning + evaluation

### Multi-GPU (DDP) Features
- Uses `DistributedDataParallel` for efficient multi-GPU training
- HP tuning runs on main process only (Optuna not DDP-aware)
- Pretraining/fine-tuning uses all GPUs
- Evaluation runs on main process only
- Batch size automatically divided across GPUs
- Loss averaged across GPUs for consistent logging

### Resumability
- `TrainingManifest` tracks progress across all phases
- Safe to interrupt (Ctrl+C) and resume with `--resume`
- Checkpoints saved in `models/diffusion_tsf/checkpoints_7var/`

---

## Digital Research Alliance (Canada HPC) Deployment

### Quick Start on Alliance Clusters

```bash
# 1. Clone/upload repo to your home directory
cd ~
git clone <your-repo> ts-sandbox
cd ts-sandbox

# 2. Run setup (creates persistent storage in $PROJECT)
./alliance_setup.sh

# 3. Edit slurm script with your account
nano slurm_train_7var.sh
# Change: --account=def-YOURPI
# Change: --mail-user=YOUR@EMAIL

# 4. Submit job
./submit_train.sh --smoke-test   # Test first
./submit_train.sh                # Full training
```

### Cluster-Specific GPU Settings

| Cluster | GPU Model | VRAM | Flag |
|---------|-----------|------|------|
| Narval | A100 | 40GB | `--gpus-per-node=a100:4` |
| Fir | H100 | 80GB | `--gpus-per-node=h100:4` |
| Nibi | H100 | 80GB | `--gpus-per-node=h100:4` |
| Rorqual | H100 | 80GB | `--gpus-per-node=h100:4` |

### Storage Strategy
- **PROJECT** (`$PROJECT/diffusion-tsf/`): Checkpoints, results, datasets - backed up, persistent
- **SCRATCH**: Only for temporary files during job execution
- **HOME**: Just the code repo

### Wandb Logging
The pipeline logs comprehensive metrics to wandb including:
- Git commit, branch, dirty status
- All hyperparameters and constants
- System info (GPU names, CUDA version, etc.)
- Per-epoch training/validation losses
- HP search results and best params
- Final evaluation metrics per dataset
- Model checkpoints as artifacts

```bash
# Setup wandb
pip install wandb
wandb login

# Enable in training
python -m models.diffusion_tsf.train_7var_pipeline --wandb
```

### Monitoring Jobs
```bash
sq                    # Check job queue
./check_status.sh     # Training progress
tail -f *.out         # Watch output
scancel <jobid>       # Cancel job
```

### Syncing Results to Local
```bash
# On your LOCAL machine:
./sync_from_cluster.sh user@narval.alliancecan.ca
```
