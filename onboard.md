# onboarding guide for this thing

## summary
diffusion for time series. basically treats time series like 2D images (stripes) and uses a U-Net + DDPM/DDIM to generate predictions.

## key stuff
- multivariate: handles multiple variables at once.
- universal pre-train: train on 1M fake samples (7 vars), then fine-tune on real stuff.
- CCM: for datasets with more than 7 variables. clusters them into 7 "super-channels".
- multivariate augs: generates realistic synthetic data for pre-training.

## file layout

### models
- `models/diffusion_tsf/`: core diffusion logic.
    - `diffusion_model.py`: main `DiffusionTSF` class. encoding/decoding, forward pass, generation.
    - `unet.py`: the U-Net backbone (2D).
    - `ccm_adapter.py`: clusters >7 vars into 7 super-channels.
    - `dataset.py`: dataset stuff (Electricity + synthetic).
    - `realts.py`: synthetic data generators.
    - `train_universal_v2.py`: **THE MAIN SCRIPT**. pretrain on synthetic → fine-tune everything.
    - `train_electricity.py`: single dataset training + optuna.
    - `config.py`: config dataclasses.
    - `tests/`: some tests for CCM etc.

- `models/iTransformer/`: iTransformer (Stage 1 predictor).
- `models/TimeSeriesCCM-main/`: reference code for CCM.

### scripts
- `run_all_datasets.sh`: **master script** - runs everything on 9 datasets.
    ```bash
    ./run_all_datasets.sh --smoke-test    # quick check
    ./run_all_datasets.sh                  # do everything
    ./run_all_datasets.sh --dataset ETTh1  # just one dataset
    ```
- `train_universal_v2.py`: entry point.
    ```bash
    python -m models.diffusion_tsf.train_universal_v2 --smoke-test
    python -m models.diffusion_tsf.train_universal_v2 --mode pretrain --synthetic-samples 1000000
    python -m models.diffusion_tsf.train_universal_v2 --mode finetune --dataset electricity
    ```

## architecture notes

### 1. Unified L+F scheme
we use a single time axis of length `Lookback + Forecast`.
- Input: `(Batch, Channels, Height, L+F)`.
- Past (0 to L): ground truth.
- Future (L to L+F): noisy future.
- Auxiliary: coordinate grids, time ramps, sine waves.
- Guidance: Stage 1 forecasts as "ghost images" in the future part.

### 2. multivariate augs
in `models/diffusion_tsf/augmentation.py`.
- generates fake data by coupling random processes.
- uses "Organic" generators (Random Walks, FFT, etc.) from `realts.py`.
- periodic generators support irregular periods (50% of samples).
- impact functions for causal effects.
- use `models/diffusion_tsf/visualize_synthetic.py` to see what it looks like.
mple plots of the synthetic data.

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
- **2026-02-20:** 2D representation doubled to 128×1216 (was 64×608). >7-variate subsets capped at 5. Synthetic pretraining uses 100k samples (regenerated, not cached). Irregular periodicity added to synthetic generators.
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
1. **Pretrain once** on 100k synthetic 7-variate samples (regenerated each run; no cache reuse)
2. **Fine-tune** on each dataset:
   - 7-variate datasets (ETTh1/h2/m1/m2, illness): Direct fine-tune
   - >7-variate datasets: Random shuffle → partition into non-overlapping 7-variate subsets, **capped at 5 subsets per dataset**
     - electricity (321 vars) → 5 subsets (of 45 possible)
     - traffic (862 vars) → 5 subsets (of 123 possible)
     - weather (21 vars) → 3 subsets: weather-0, weather-1, weather-2
     - exchange_rate (8 vars) → 1 subset: exchange_rate-0
3. **Evaluate** with single sample AND averaged (30 samples) metrics

### 2D Representation Size
- **Image dimensions:** 128 × 1216 (height × (lookback + forecast))
- Lookback: 1024, Forecast: 192, Height: 128

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

### Visualizations

**Comparison plots (iTransformer vs Diffusion):**
```bash
# On cluster:
sbatch slurm_viz_comparison.sh
# Then sync locally:
rsync -avz user@narval:~/projects/def-*/diffusion-tsf/results/viz/comparison/ ./synced_results/viz/comparison/
```

Generates per-dataset PNGs (`comparison_{dataset}.png`) with ground truth, iTransformer-only, and diffusion overlays. 3 samples × 3 variables per dataset.

**Detailed per-subset visualizations (2D representations, etc.):**
```bash
sbatch slurm_visualize.sh
```

### Syncing Results to Local
```bash
# On your LOCAL machine:
./sync_from_cluster.sh user@narval.alliancecan.ca
```
