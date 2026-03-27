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
- `pipeline.sh`: **THE master script** — runs everything (splits high-variate into 32-dim subsets).
    ```bash
    ./pipeline.sh                          # full pipeline
    ./pipeline.sh --smoke-test             # quick validation
    ./pipeline.sh --gpus 4                 # parallel fine-tuning
    ./pipeline.sh --dataset electricity    # single dataset
    ./pipeline.sh --pretrain-only          # just Phase 1
    ```
- `run_unet_fullvar.sh`: trains U-Net on full-variate datasets (no subset splitting). bf16, H=96, 75K synth pool.
    ```bash
    ./run_unet_fullvar.sh                          # default: traffic (861-var)
    ./run_unet_fullvar.sh --smoke-test             # quick validation
    ./run_unet_fullvar.sh --dataset electricity    # 321-var
    ```
- `slurm_pipeline.sh`: Alliance HPC Slurm wrapper for pipeline.sh.
- `slurm_latent_experiment.sh`: Slurm wrapper for 1-var latent diffusion (`train_latent_experiment.py`). Same `$PROJECT/$USER/diffusion-tsf/venv` + module stack as `slurm_pipeline.sh`.
- `slurm_ci_latent_etth1.sh`: CI (channel-independent) latent diffusion ablation on ETTh1 7-var. Submit guided vs unguided:
    ```bash
    sbatch --job-name=ci-guided   slurm_ci_latent_etth1.sh
    sbatch --job-name=ci-unguided slurm_ci_latent_etth1.sh -- --no-guidance
    ```
- `slurm_unet_fullvar.sh`: self-resubmitting Slurm wrapper for run_unet_fullvar.sh (Killarney).
- `summarize_results.py`: generate markdown report from eval results.
- `setup/setup.sh`: one-time local env setup (venv, CUDA, deps).
- `setup/alliance_setup_killarney.sh`: one-time Killarney cluster setup.

### models
- `models/diffusion_tsf/`: core diffusion logic.
    - `diffusion_model.py`: main `DiffusionTSF` class.
    - `unet.py`: U-Net backbone.
    - `transformer.py`: DiT-style backbone option.
    - `diffusion.py`: `DiffusionScheduler` — DDPM/DDIM forward and reverse processes.
    - `preprocessing.py`: `Standardizer`, `TimeSeriesTo2D` (encode/decode), `VerticalGaussianBlur`.
    - `train_7var_pipeline.py`: **the Python entry point** — pretrain, finetune, eval, baseline.
    - `train_latent_experiment.py`: 1-var latent diffusion experiment (VAE → iTransformer → LDM → ETTh1).
    - `train_ci_latent_etth1.py`: CI latent diffusion ablation on ETTh1 7-var. Tests guided (iTransformer ghost) vs unguided diffusion, both vs iTransformer-only baseline.
    - `latent_diffusion_model.py`: `LatentDiffusionTSF` — latent-space diffusion with frozen VAE.
    - `vae.py`: `TimeSeriesVAE` — convolutional VAE for 2D time-series images (4× spatial compression).
    - `latent_experiment_common.py`: shared helpers for latent experiment scripts.
    - `train_electricity.py`: single-dataset training + optuna (used internally).
    - `train_universal_v2.py`: older universal training script.
    - `config.py`: config dataclasses.
    - `dataset.py`: dataset loading + synthetic data.
    - `realts.py`: synthetic data generators.
    - `augmentation.py`: multivariate augmentation.
    - `metrics.py`: shape-preservation + standard TSF metrics (MSE/MAE).
    - `visualize.py`: general plotting utilities for 2D representations.
    - `visualize_comparison.py`: iTransformer vs Diffusion comparison plots.
    - `visualize_7var.py`: per-subset visualizations.
    - `evaluate_7var.py`: standalone evaluation.
    - `ccm_adapter.py`: channel clustering module (legacy approach).
    - `guidance.py`: iTransformerGuidance wrapper.
    - `tests/`: unit tests.

- `models/iTransformer/`: iTransformer (Stage 1 predictor).

### legacy
- `legacy/scripts/`: old shell scripts (run_all_datasets.sh, run_7var_pipeline.sh, etc.)
- `legacy/`: old Python scripts (evaluate_latest.py, remote_evaluate.py, etc.)

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
4. **Phase 4 — Viz**: comparison plots (ground truth vs iTransformer vs diffusion).

### Python CLI modes
```bash
python -m models.diffusion_tsf.train_7var_pipeline --mode pretrain --n-variates 7
python -m models.diffusion_tsf.train_7var_pipeline --mode finetune --dataset ETTh1 --n-variates 7
python -m models.diffusion_tsf.train_7var_pipeline --mode finetune-subset --dataset electricity --subset-id electricity-0 --variate-indices 0,1,2,...,31 --n-variates 32
python -m models.diffusion_tsf.train_7var_pipeline --mode baseline --dataset electricity
python -m models.diffusion_tsf.train_7var_pipeline --mode list-subsets --dataset electricity --n-variates 32
python -m models.diffusion_tsf.train_7var_pipeline --mode full  # legacy: run everything in one go
```

### checkpoint structure
```
checkpoints_7var/
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

Config: `lookback_overlap=8`, `past_loss_weight=0.3`. Constants: `LOOKBACK_OVERLAP`, `PAST_LOSS_WEIGHT` in train_7var_pipeline.py.

### Unified L+F scheme
Single time axis of length `Lookback + Forecast` (1024 + 200 = 1224).
- Input: `(Batch, Channels, Height=128, L+F)`.
- Past (0..L): ground truth stripe/occupancy.
- Future (L..L+F): noisy future (K overlap + H forecast).
- Aux channels: coordinate grid, time ramp, time sine.
- Guidance channels: iTransformer "ghost images" in the future part.

### 2D Representation
- **Image:** 128 × 1224 (height × width, with K=8 overlap).
- **PDF mode (default):** one-hot stripe — pixel index `y = clip((x-μ)/σ * MS/2 * H + H/2, 0, H-1)` where MS=3.5.
- **CDF mode:** occupancy map — fill pixels `0..y` as 1.0 (cumulative); softens into a sigmoid boundary after blur.
- Vertical Gaussian blur (31×1 kernel) blurs only along the value axis, preserving temporal patterns.
- Decode: PDF → softmax-expectation; CDF → column sum normalized to `[-max_scale, max_scale]`.

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

# 3. Smoke test
sbatch slurm_pipeline.sh --smoke-test

# 4. Monitor
sq && tail -f diffusion-tsf-*.out

# 5. Full run
sbatch slurm_pipeline.sh

# 6. Sync results locally (from your WSL machine)
./sync_from_killarney.sh ccao87@killarney.alliancecan.ca

# Latent 1-var experiment (uses same venv path as pipeline; script creates/repairs venv if torch missing)
sbatch slurm_latent_experiment.sh
```

**Killarney venv:** GPU jobs load `python/3.11` modules, then use **`$PROJECT/$USER/diffusion-tsf/venv`** (not the repo-local `venv/`). `slurm_pipeline.sh` and `slurm_latent_experiment.sh` both create that venv on first run and install CUDA PyTorch + `reformer_pytorch`. The latent script calls **`$VENV/bin/python`** so Slurm never picks system Python by mistake.

| Cluster | GPU | VRAM | Account prefix | Partitions |
|---------|-----|------|----------------|------------|
| Killarney | H100 SXM | 80GB | `aip-` | `gpubase_h100_b1`(3h)..`b5`(7d) |
| Narval  | A100 | 40GB | `def-` | default |
| Fir     | H100 | 80GB | `def-` | default |

## Gotchas
- **Killarney torch missing:** if a job dies with `No module named 'torch'`, the PROJECT venv is empty or broken. Run `sbatch slurm_pipeline.sh --smoke-test` once, or delete `$PROJECT/$USER/diffusion-tsf/venv` and resubmit so the Slurm script recreates it.
- **Imports:** always run from project root: `python -m models.diffusion_tsf.script_name`
- **Smoke test first:** `./pipeline.sh --smoke-test` before committing to a full run.
- **OOM:** use `attention_levels=[2]` and small batch sizes for 32-dim models.
- **traffic.csv:** auto-combined from part1+part2 by pipeline.sh.

## Recent Changes
- **2026-03-27:** CI latent diffusion ablation. `train_ci_latent_etth1.py` + `slurm_ci_latent_etth1.sh` test channel-independent latent diffusion on ETTh1 (7-var). Each variate processed independently through shared univariate VAE + U-Net. Two variants: `--no-guidance` (pure diffusion) vs guided (iTransformer ghost images via `CIiTransformerGuidance` wrapper that unflattens batch for multivariate iTransformer). Fixed `in_ch` bug in `LatentDiffusionTSF` for `use_guidance_channel=False`.
- **2026-03-15:** Full-variate U-Net path. `run_unet_fullvar.sh` + `slurm_unet_fullvar.sh` train U-Net directly on native-dim datasets (traffic=861, electricity=321) with bf16, H=96, 75K synth pool, 3 iTransformer HP trials. Cross-var augmentation auto-skipped for V>32. Synthetic pool disk caching via `cache_dir`. New CLI flags: `--synthetic-samples`, `--itransformer-trials`, `--subset-threshold`.
- **2026-03-06:** Lookback overlap: diffusion model now predicts K=8 past steps alongside H=192 forecast to smooth boundary. Weighted loss (0.3× overlap, 1.0× forecast), trimmed at inference.
- **2026-03-06:** Unified pipeline. Single `pipeline.sh` replaces scattered scripts. Dimensionality groups (7/8/21/32) with per-dim pretraining. Multi-GPU parallel subset fine-tuning. Full-dim iTransformer baseline for high-variate comparison. Old scripts moved to `legacy/`.
- **2026-02-20:** 2D representation doubled to 128×1216. Synthetic pretraining uses 100k samples. Irregular periodicity added.
- **2026-02-04:** Added CCM adapter, train_universal_v2.py.
