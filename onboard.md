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
- `pipeline.sh`: **THE master script** — runs everything.
    ```bash
    ./pipeline.sh                          # full pipeline
    ./pipeline.sh --smoke-test             # quick validation
    ./pipeline.sh --gpus 4                 # parallel fine-tuning
    ./pipeline.sh --dataset electricity    # single dataset
    ./pipeline.sh --pretrain-only          # just Phase 1
    ```
- `slurm_pipeline.sh`: Alliance HPC Slurm wrapper for pipeline.sh.
- `setup.sh`: one-time env setup (venv, CUDA, deps).
- `alliance_setup.sh`, `alliance_setup_killarney.sh`: cluster-specific setup.
- `sync_results.sh`: pull results from remote cluster.
- `summarize_results.py`: generate markdown report from eval results.

### models
- `models/diffusion_tsf/`: core diffusion logic.
    - `diffusion_model.py`: main `DiffusionTSF` class.
    - `unet.py`: U-Net backbone.
    - `train_7var_pipeline.py`: **the Python entry point** — pretrain, finetune, eval, baseline.
    - `train_electricity.py`: single-dataset training + optuna (used internally).
    - `train_universal_v2.py`: older universal training script.
    - `config.py`: config dataclasses.
    - `dataset.py`: dataset loading + synthetic data.
    - `realts.py`: synthetic data generators.
    - `augmentation.py`: multivariate augmentation.
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
single time axis of length `Lookback + Forecast` (1024 + 200 = 1224).
- Input: `(Batch, Channels, Height=128, L+F)`.
- Past (0..L): ground truth stripe/occupancy.
- Future (L..L+F): noisy future (K overlap + H forecast).
- Aux channels: coordinate grid, time ramp, time sine.
- Guidance channels: iTransformer "ghost images" in the future part.

### 2D Representation
- **Image:** 128 × 1224 (height × width, with K=8 overlap).
- PDF mode: one-hot stripe per timestep.
- CDF mode: occupancy map (cumulative fill).
- Vertical Gaussian blur softens the representation for diffusion.

## Alliance HPC deployment

```bash
# 1. Setup
./alliance_setup.sh

# 2. Edit slurm_pipeline.sh: --account, --mail-user, --gpus-per-node

# 3. Submit
sbatch slurm_pipeline.sh --smoke-test   # test
sbatch slurm_pipeline.sh                # full run

# 4. Monitor
sq && tail -f diffusion-tsf-*.out

# 5. Sync results locally
./sync_results.sh user@narval.alliancecan.ca
```

| Cluster | GPU | VRAM | Flag |
|---------|-----|------|------|
| Narval  | A100 | 40GB | `--gpus-per-node=a100:4` |
| Fir     | H100 | 80GB | `--gpus-per-node=h100:4` |

## Gotchas
- **Imports:** always run from project root: `python -m models.diffusion_tsf.script_name`
- **Smoke test first:** `./pipeline.sh --smoke-test` before committing to a full run.
- **OOM:** use `attention_levels=[2]` and small batch sizes for 32-dim models.
- **traffic.csv:** auto-combined from part1+part2 by pipeline.sh.

## Recent Changes
- **2026-03-06:** Lookback overlap: diffusion model now predicts K=8 past steps alongside H=192 forecast to smooth boundary. Weighted loss (0.3× overlap, 1.0× forecast), trimmed at inference.
- **2026-03-06:** Unified pipeline. Single `pipeline.sh` replaces scattered scripts. Dimensionality groups (7/8/21/32) with per-dim pretraining. Multi-GPU parallel subset fine-tuning. Full-dim iTransformer baseline for high-variate comparison. Old scripts moved to `legacy/`.
- **2026-02-20:** 2D representation doubled to 128×1216. Synthetic pretraining uses 100k samples. Irregular periodicity added.
- **2026-02-04:** Added CCM adapter, train_universal_v2.py.
