# onboarding guide

## summary
diffusion for time series. treats time series like 2D images (stripes/occupancy maps) and uses a U-Net + DDPM/DDIM to generate predictions. iTransformer provides "ghost image" guidance for the diffusion model.

## key stuff
- multivariate: handles multiple variables as separate image channels.
- dimensionality groups: each unique column count gets its own pretrained model. high-variate datasets (>32 cols) are split into 32-dim subsets.
- synthetic pretrain + real fine-tune: for each dim group, pretrain on 100k synthetic samples, then fine-tune on real data.
- iTransformer guidance: Stage 1 coarse forecast → diffusion refines it.
- multi-GPU parallel: high-variate subset fine-tuning distributed across GPUs.

## file layout

### root scripts
- `slurm_unet_fullvar.sh`: **primary Slurm training entry** for full-variate U-Net (no 32-dim subset splitting). Self-`sbatch` from login node; job runs `train_multivariate_pipeline` with bf16, H=96, 75K synth pool; storage under `$PROJECT/$USER/diffusion-tsf-fullvar`.
    ```bash
    ./slurm_unet_fullvar.sh --smoke-test
    ./slurm_unet_fullvar.sh --dataset electricity
    ```
- `slurm_latent_experiment.sh`: Slurm wrapper for 1-var latent diffusion (`train_latent_experiment.py`). Uses `$PROJECT/$USER/diffusion-tsf/venv` like the other GPU jobs.
- `slurm_ci_latent_multidataset.sh`: **one Slurm job** — reuses shared stages 0–2 under `checkpoints_ci_etth2/`, then stages 3–4 for ETTh1, ETTh2, ETTm1, ETTm2, and `exchange_rate` (7 of 8 columns, seed 42). Stage 4 uses **12 Optuna trials** (wide search space). Wall time default 7 days.
    ```bash
    sbatch slurm_ci_latent_multidataset.sh
    sbatch slurm_ci_latent_multidataset.sh -- --smoke-test
    ```
- `submit_ci_latent_multidataset_jobs.sh` (login node): submits **bootstrap** `slurm_ci_latent_bootstrap.sh` (stages 0–2, 2-day cap) then **five** `slurm_ci_latent_finetune_dataset.sh` jobs in parallel (`afterok` bootstrap), each 3-day cap — one job per dataset. Shared setup lives in `slurm_ci_latent_common.inc.sh`. Optional: `SBATCH_EXTRA="--time=24:00:00"` or `SBATCH_ACCOUNT=...` when submitting.
    ```bash
    ./submit_ci_latent_multidataset_jobs.sh
    ./submit_ci_latent_multidataset_jobs.sh -- --smoke-test
    ```
- `slurm_ci_latent_etth2.sh`: Single-dataset ETTh2 full pipeline (passes `--shared-ckpt-dir` / `--run-ckpt-dir` like the multi script).
    ```bash
    sbatch slurm_ci_latent_etth2.sh                              # full run
    sbatch --job-name=ci-etth2-smoke slurm_ci_latent_etth2.sh -- --smoke-test
    ```
- `find_traffic_results.sh`: search cluster storage for traffic experiment artifacts (run over SSH on the cluster).
- `setup/alliance_setup_killarney.sh`: one-time Killarney cluster setup (creates `sync_from_killarney.sh`, etc.).

### models
- `models/diffusion_tsf/`: core diffusion logic.
    - `diffusion_model.py`: main `DiffusionTSF` class.
    - `unet.py`: U-Net backbone.
    - `transformer.py`: DiT-style backbone option.
    - `diffusion.py`: `DiffusionScheduler` — DDPM/DDIM forward and reverse processes.
    - `preprocessing.py`: `Standardizer`, `TimeSeriesTo2D` (encode/decode), `VerticalGaussianBlur`.
    - `train_multivariate_pipeline.py`: **the Python entry point** — pretrain, finetune, eval, baseline.
    - `storage_paths.py`: default checkpoint/results dirs (`checkpoints_multivariate` / `results_multivariate`); falls back to legacy `checkpoints_7var` / `results_7var` if those exist and new dirs do not.
    - `train_latent_experiment.py`: 1-var latent diffusion experiment (VAE → iTransformer → LDM → ETTh1).
    - `train_ci_latent_etth1.py`: CI latent diffusion ablation on ETTh1 7-var. Tests guided (iTransformer ghost) vs unguided diffusion, both vs iTransformer-only baseline. Stage 3 saves `checkpoints_ci_latent/{guided,unguided}_finetuned_H{96|128}.pt` for viz.
    - `train_ci_latent_etth2.py`: Full 4-stage CI latent pipeline for ETTh2. Stage 1: pretrain iTrans on synth. Stage 2: pretrain diffusion with pretrained iTrans guidance. Stage 3: finetune iTrans on ETTh2. Stage 4: finetune diffusion with *finetuned* iTrans guidance + eval vs finetuned iTrans. Checkpoints in `checkpoints_ci_etth2/`.
    - `latent_diffusion_model.py`: `LatentDiffusionTSF` — latent-space diffusion with frozen VAE.
    - `vae.py`: `TimeSeriesVAE` — convolutional VAE for 2D time-series images (4× spatial compression).
    - `latent_experiment_common.py`: shared helpers for latent experiment scripts.
    - `config.py`: config dataclasses.
    - `dataset.py`: dataset loading + synthetic data.
    - `realts.py`: synthetic data generators.
    - `augmentation.py`: multivariate augmentation.
    - `metrics.py`: shape-preservation + standard TSF metrics (MSE/MAE).
    - `visualize_comparison.py`: **only viz script** — multivariate overlays (GT vs iTransformer vs diffusion). Default: scans both `checkpoints_multivariate` and legacy `checkpoints_7var` if present. `python -m models.diffusion_tsf.visualize_comparison --output-dir ... --num-samples 3 --vars 3` (optional `--checkpoint-dir` for a single root).
    - `guidance.py`: iTransformerGuidance + other guidance wrappers.
    - `tests/`: unit tests.

- `models/iTransformer/`: vendored **minimal** iTransformer: `model/iTransformer.py`, `layers/*`, `utils/masking.py` only (upstream `run.py`, `experiments/`, `data_provider/`, extra model variants removed).

## datasets

| Dataset | Cols | Sampling | Seasonal period |
|---------|------|----------|-----------------|
| ETTh1/h2 | 7 | Hourly | 24 |
| ETTm1/m2 | 7 | 15-min | 96 |
| exchange_rate | 8 | Daily | 5 |
| illness | 7 | Weekly | 52 |
| weather | 21 | 10-min | 144 |
| electricity | 321 | Hourly | 96 |
| traffic | 861 | Hourly | 24 |

## pipeline overview

### dimensionality groups
| Dim | Datasets | Approach |
|-----|----------|----------|
| 7   | ETTh1/h2/m1/m2, illness | Direct fine-tune |
| 8   | exchange_rate | Direct fine-tune |
| 21  | weather | Direct fine-tune |
| 32  | electricity (321 cols → 10 subsets), traffic (861 cols → ~26 subsets) | Split into 32-dim subsets |

### phases
1. **Phase 1 — Pretrain** (per unique dim): generate 100k synthetic samples → HP tune → pretrain iTransformer + Diffusion.
2. **Phase 2 — Fine-tune** (per dataset): HP tune → fine-tune → evaluate. High-variate subsets run in parallel across GPUs.
3. **Phase 3 — Baselines** (high-variate only): train a full-dim iTransformer on ALL columns for comparison.
4. **Phase 4 — Viz**: run `python -m models.diffusion_tsf.visualize_comparison` on the checkpoint tree (GT vs iTransformer vs diffusion).

### Python CLI modes
```bash
python -m models.diffusion_tsf.train_multivariate_pipeline --mode pretrain --n-variates 7
python -m models.diffusion_tsf.train_multivariate_pipeline --mode finetune --dataset ETTh1 --n-variates 7
python -m models.diffusion_tsf.train_multivariate_pipeline --mode finetune-subset --dataset electricity --subset-id electricity-0 --variate-indices 0,1,2,...,31 --n-variates 32
python -m models.diffusion_tsf.train_multivariate_pipeline --mode baseline --dataset electricity
python -m models.diffusion_tsf.train_multivariate_pipeline --mode list-subsets --dataset electricity --n-variates 32
python -m models.diffusion_tsf.train_multivariate_pipeline --mode full  # legacy: run everything in one go
```

### checkpoint structure
Default on-disk names: `checkpoints_multivariate/` (same layout as before). Legacy `checkpoints_7var/` is still read if the new dir is absent.
```
checkpoints_multivariate/   # or checkpoints_7var (legacy)
├── pretrained_dim7/
│   ├── itransformer.pt
│   └── diffusion.pt
├── pretrained_dim8/
│   ├── itransformer.pt
│   └── diffusion.pt
├── pretrained_dim21/ ...
├── pretrained_dim32/ ...
├── ETTh1/
│   ├── best.pt
│   └── metadata.json
├── electricity-0/
│   ├── best.pt
│   └── metadata.json
├── electricity-baseline/
│   └── itransformer_full.pt
├── training_manifest.json
```

## architecture notes

### Lookback overlap (boundary smoothing)
The diffusion model predicts the last K=8 lookback timesteps in addition to the H=192 forecast. Total prediction width = K+H = 200. Loss is weighted: 0.3× for the overlap region, 1.0× for the forecast. During inference, the overlap is trimmed — only the H-step forecast is returned. iTransformer still predicts H steps; its guidance is concatenated with actual observed values for the overlap region.

Config: `lookback_overlap=8`, `past_loss_weight=0.3`. Constants: `LOOKBACK_OVERLAP`, `PAST_LOSS_WEIGHT` in train_multivariate_pipeline.py.

### Unified L+F scheme
Single time axis of length `Lookback + Forecast` (1024 + 200 = 1224).
- Input: `(Batch, Channels, Height=128, L+F)`.
- Past (0..L): ground truth stripe/occupancy.
- Future (L..L+F): noisy future (K overlap + H forecast).
- Aux channels: coordinate grid, time ramp, time sine.
- Guidance channels: iTransformer "ghost images" in the future part.

### 2D Representation
- **Image:** 128 × 1224 (height × width, with K=8 overlap).
- **Encoding:** occupancy map — bin normalized value, fill rows `0..y` to 1.0 (MS=3.5); vertical Gaussian blur (31×1) along the value axis only.
- **Decode:** column sum → normalized level, or optional `expectation` decoder (vertical-gradient mass → expected bin); maps back to `[-max_scale, max_scale]`.

### Channel order (full)
```
[Var_0..Var_N (noisy),  Vertical_Coord,  Time_Ramp,  Time_Sine,  Guide_Var_0..Guide_Var_N (if guidance)]
```
- Multivariate: each variable = one image channel; aux channels shared across all vars.
- Backbone in-channels: `num_variables + num_aux + (num_variables if guidance)`.
- Backbone out-channels: `num_variables` (predicts noise per variable).

### Hybrid forecasting flow
```
Past → [iTransformer] → coarse 1D forecast
                              ↓
                       [TimeSeriesTo2D + Blur]
                              ↓
                         "ghost image" (2D)
                              ↓
[Noisy future + Aux + Ghost] → [U-Net] → refined forecast
```
- iTransformer trained on chronological 70/10/20 split to avoid leakage.
- Ghost image guidance only populates the future portion (L..L+F) of the input tensor.

### Backbone options
- **U-Net (default):** residual blocks, GroupNorm, skip connections — best for local spatial hierarchies. `conditioning_mode=visual_concat` directly prepends past 2D image (no extra encoder).
- **DiT:** patch-based transformer; global context injected as a prepended context token.
- Diffusion: DDPM (T=1000, linear/cosine/sigmoid schedule), DDIM (50 steps, `eta` controls stochasticity), CFG support.

## Alliance HPC deployment

**Killarney** (primary cluster): account `aip-boyuwang`, H100 SXM 80GB. Must submit from `/scratch`, not `/home`.

```bash
# 1. SSH in, move repo to scratch
ssh ccao87@killarney.alliancecan.ca
cp -r ~/ts-sandbox /scratch/$USER/ts-sandbox   # or git clone there
cd /scratch/$USER/ts-sandbox

# 2. One-time setup (venv, datasets, generates slurm scripts)
./setup/alliance_setup_killarney.sh

# 3. Smoke test (pick one)
./slurm_unet_fullvar.sh --smoke-test
# or: python -m models.diffusion_tsf.train_multivariate_pipeline --smoke-test  # from an interactive GPU session

# 4. Monitor
sq && tail -f diffusion-tsf-*.out

# 5. Full-variate U-Net (typical production)
./slurm_unet_fullvar.sh

# 6. Sync results locally (from your WSL machine; file created by alliance_setup)
./sync_from_killarney.sh ccao87@killarney.alliancecan.ca

# Latent 1-var experiment
sbatch slurm_latent_experiment.sh
```

**Killarney venv:** GPU jobs load `python/3.11` modules, then use **`$PROJECT/$USER/diffusion-tsf/venv`** (not the repo-local `venv/`). Slurm scripts (`slurm_unet_fullvar.sh`, `slurm_latent_experiment.sh`, `slurm_ci_latent_*.sh`) create or reuse that venv and install CUDA PyTorch + `reformer_pytorch`. Prefer **`$VENV/bin/python`** in jobs so Slurm never uses the wrong interpreter.

| Cluster | GPU | VRAM | Account prefix | Partitions |
|---------|-----|------|----------------|------------|
| Killarney | H100 SXM | 80GB | `aip-` | `gpubase_h100_b1`(3h)..`b5`(7d) |
| Narval  | A100 | 40GB | `def-` | default |
| Fir     | H100 | 80GB | `def-` | default |

## Gotchas
- **Killarney torch missing:** if a job dies with `No module named 'torch'`, the PROJECT venv is empty or broken. Delete `$PROJECT/$USER/diffusion-tsf/venv` and resubmit a smoke job, or run `./slurm_unet_fullvar.sh --smoke-test` so the script recreates the venv.
- **Imports:** always run from project root: `python -m models.diffusion_tsf.script_name`
- **Smoke test first:** `./slurm_unet_fullvar.sh --smoke-test` before long runs.
- **OOM:** use `attention_levels=[2]` and small batch sizes for 32-dim models.
- **traffic.csv:** auto-combined from part1+part2 by `slurm_unet_fullvar.sh` (and the same logic if you run `train_multivariate_pipeline` after merging parts manually).

## Recent Changes
- **2026-03-31:** Checkpoint/results default dirs renamed to `checkpoints_multivariate` / `results_multivariate`; `storage_paths.py` still resolves legacy `checkpoints_7var` / `results_7var` when the new folders are missing. `visualize_comparison` defaults to scanning both trees.
- **2026-03-31:** Renamed `train_7var_pipeline.py` → `train_multivariate_pipeline.py` (any `--n-variates`; default 7 is only ETT-style default).
- **2026-03-31 (cleanup):** Removed `legacy/` tree. Dropped extra viz scripts; kept `visualize_comparison.py` only. Trimmed `models/iTransformer/` to `iTransformer.py` + `layers/` + `utils/masking.py`.
- **2026-03-31:** Full-variate training is `slurm_unet_fullvar.sh` only (inlined former `run_unet_fullvar.sh`). Removed `setup/setup.sh`.
- **2026-03-29:** CI latent diffusion ETTh2 full pipeline. `train_ci_latent_etth2.py` + `slurm_ci_latent_etth2.sh` — 4-stage pipeline (pretrain iTrans on synth → pretrain diffusion with pretrained iTrans guidance → finetune iTrans on ETTh2 → finetune diffusion with finetuned iTrans + eval). Both eval baselines and diffusion guidance use the dataset-finetuned iTransformer, not the synthetic-pretrained one. Added ETTh2 to `latent_experiment_common.py` DATASET_REGISTRY.
- **2026-04-04:** `slurm_ci_latent_multidataset.sh` + multi-`--dataset` support in `train_ci_latent_etth2.py`; removed `slurm_ci_latent_etth1.sh` (ETTh1 ablation still: `python -m models.diffusion_tsf.train_ci_latent_etth1`).
- **2026-03-27:** CI latent ETTh1 ablation script `train_ci_latent_etth1.py` (guided vs `--no-guidance`). `CIiTransformerGuidance` batching; `LatentDiffusionTSF` `use_guidance_channel` fix.
- **2026-03-15:** Full-variate U-Net path. `slurm_unet_fullvar.sh` trains U-Net directly on native-dim datasets (traffic=861, electricity=321) with bf16, H=96, 75K synth pool, 12 iTransformer HP trials. Cross-var augmentation auto-skipped for V>32. Synthetic pool disk caching via `cache_dir`. New CLI flags: `--synthetic-samples`, `--itransformer-trials`, `--subset-threshold`.
- **2026-03-06:** Lookback overlap: diffusion model now predicts K=8 past steps alongside H=192 forecast to smooth boundary. Weighted loss (0.3× overlap, 1.0× forecast), trimmed at inference.
- **2026-03-06:** Unified `train_multivariate_pipeline` modes: dimensionality groups (7/8/21/32), per-dim pretraining, multi-GPU subset fine-tuning, full-dim iTransformer baseline for high-variate comparison.
- **2026-02-20:** 2D representation doubled to 128×1216. Synthetic pretraining uses 100k samples. Irregular periodicity added.
- **2026-02-04:** Added CCM adapter, train_universal_v2.py.
