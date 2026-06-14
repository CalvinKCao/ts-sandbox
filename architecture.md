# Architecture (ts-sandbox)

Implementation-oriented map for the **current default** binary forecasting stack. Tensor-level detail for the legacy joint dual-scale path is summarized at the end only.

---

## Current default (production)

**What we run:** staged coarse→fine binary diffusion on hard CDF maps, **stationary-flat anchor** (`0.5` canvas, not random bits), **EMA 0.99** on diffusion weights during finetune, **anchor** training loss (`λ=0.99`), eval with **DPM-Solver++** (20 steps, 20 samples for sweeps).

| Knob | Value | Config source |
|------|-------|----------------|
| Pipeline | YAML `phases` list | `configs/base/binary_staged.yaml` |
| Leaf experiment | `binary_anchor_stationary_flat_subsets_ema099` (or `_flat` / `_flat_subsets`) | `configs/binary_anchor_stationary_flat*.yaml` |
| Representation | Staged coarse + fine CDF (`image_height=16` each) | `use_dual_scale: false`, separate checkpoints per stage |
| `binary_anchor_input_mode` | `stationary_flat` | flat `0.5` XOR anchor |
| `diffusion_ema_decay` | `0.99` | `training:` section |
| `deterministic_anchor_lambda` | `0.99` | anchor BCE mixed into train loss |
| `deterministic_anchor_alpha` | `0.0` | unused for binary |
| Diffusion finetune LR | `3e-5` fixed | `configs/base/fixed_lr_pipeline_base.yaml` |
| `max_scale` | per-dataset | `max_scale_by_dataset` in `binary_staged.yaml` |
| Variate policy | ETTh1-capped subsets | `data_subset` in `binary_staged.yaml` |
| CFG / guidance channel | off | `use_guidance_channel: false`, `cfg_dropout: 0.0` |
| Eval sampler | `dpmpp` (sweep) / `anchor` (loss) | `staged_eval` phase overrides |

**Not the default:** joint `use_dual_scale: true` single model (`configs/binary_dual_scale.yaml`) — one forward interleaves coarse+fine rows; production uses **two denoisers** (`diffusion_stage: coarse` / `fine`).

---

## File map

| Area | Files |
|------|--------|
| Orchestration | `models/diffusion_tsf/pipeline/` (`orchestrator.py`, `state.py`, `phases/*`) |
| CLI entry | `models/diffusion_tsf/train_multivariate_pipeline.py` |
| Model | `models/diffusion_tsf/diffusion_model.py`, `dit.py`, `diffusion.py`, `preprocessing.py` |
| Config | `models/diffusion_tsf/pipeline/config.py`, `configs/base/binary_staged.yaml` |
| Submit | `submit_grid.sh`, leaf YAML under `configs/` |
| iTransformer | `models/diffusion_tsf/guidance.py`, `models/iTransformer/` |

---

## Pipeline (YAML-driven)

`PipelineState` holds paths and knobs; `Pipeline` runs registered `PipelinePhase` classes in order (`models/diffusion_tsf/pipeline/phases/`). Add steps by subclassing `PipelinePhase`, registering in `phases/__init__.py`, and listing the phase in YAML — avoid one-off shell DAGs.

Sweep / flat-subset reuse configs skip Phase 1 (+ iTrans) via `reuse_pretrain_from_config` / `reuse_checkpoint_from_config` and only re-run Phases 2–4.

### Phase map (doc numbering)

| Doc | YAML `phase` | Implementation |
|-----|--------------|----------------|
| **1** | `staged_diffusion_pretrain` | `StagedDiffusionPretrainPhase` |
| *(iTrans)* | `itrans_finetune_hp` | `ITransFinetuneHPPhase` — runs after Phase 1, before Phase 2 |
| **2** | `diffusion_coarse_finetune_hp` | `CoarseDiffusionFinetuneHPPhase` |
| **3** | `diffusion_fine_finetune_hp` | `FineDiffusionFinetuneHPPhase` |
| **4** | `staged_eval` | `StagedEvalPhase` |

Default trial/epoch counts below come from `configs/base/binary_staged.yaml`. Sweep leaves (`fixed_lr_pipeline_base.yaml`) override to `n_trials: 1`, fixed LR `3e-5`, `search_space: lr_only`.

---

### Phase encyclopedia

#### Phase 1 — Synthetic staged pretrain (`staged_diffusion_pretrain`)

Trains **separate** coarse and fine denoisers on synthetic `RealTS` windows (no Optuna in this phase — fixed HP from `use_hardcoded_synthetic_hp` / Phase-1 source config).

- **Entry:** `StagedDiffusionPretrainPhase.execute` → `pretrain_diffusion` per stage.
- **Stages:** `coarse`, then `fine` (add `finer` when `use_triple_scale: true`).
- **YAML defaults:** `n_samples: 10000`, `epochs: 20`, `patience: 4`, `phase1_config_name: binary_dual_scale`.
- **Guidance:** frozen synthetic-pretrain iTransformer from Phase-1 source dir (`itrans_hp_best.pt` lineage).
- **Diffusion HP:** reuses `diff_hp.json` / hardcoded synthetic params from the same source (not re-searched here).
- **Outputs:**
  - `pretrained_coarse/pretrained_diffusion.pt`
  - `pretrained_fine/pretrained_diffusion.pt`
  - Shared cache under `_shared_staged_pretrain/<signature>/<stage>/` when `shared_cache: true`.
- **Skip when:** both stage ckpts exist locally, in shared cache, or `reuse_pretrain_from_config` copies from a prior run config.
- **Smoke:** `n_samples ≤ 4`, `epochs = patience = 1`.

#### iTrans finetune (`itrans_finetune_hp`) — between Phase 1 and Phase 2

Real-data iTransformer HP search; **cold start by default** (`cold_start: true`) — synthetic pretrain is skipped because RealTS pretrain tends toward a trivial mean predictor.

- **Entry:** `ITransFinetuneHPPhase` → `run_itransformer_finetune_hp_tuning`.
- **YAML defaults:** `n_trials: 10`, `max_epochs: 10`.
- **Search space:** `learning_rate` categorical over `itrans_paper_lr_grid` (`[1e-3, 5e-4, 1e-4]`); batch size fixed at `32`, dropout fixed at `0.1` (paper-faithful); Optuna `TPESampler`, `MedianPruner`.
- **Data:** 70/10/20 train/val/test windows on the target dataset/subset.
- **Outputs:** `{subset_id}_itrans_ft_hp_best.pt` promoted to `{subset_id}_itransformer_finetuned.pt`; `{subset_id}_itrans_ft_hp.json`.
- **Skip when:** finetuned ckpt exists, or `reuse_checkpoint_from_config` copies from a sibling config.
- **Downstream:** Phases 2–4 load this ckpt for cross-variate context tokens (guidance channel stays off in default binary flat runs).

#### Phase 2 — Coarse diffusion finetune HP (`diffusion_coarse_finetune_hp`)

Optuna-tunes the **coarse** staged DiT on real data; **best trial checkpoint is final** (no extra full retrain after HP).

- **Entry:** `CoarseDiffusionFinetuneHPPhase` (`diffusion_stage: coarse`).
- **YAML defaults:** `n_trials: 20`, `max_epochs: 20`, `patience: 8`, `search_space: default`.
- **Warm-start:** `pretrained_coarse/pretrained_diffusion.pt` from Phase 1.
- **Context:** finetuned iTransformer from iTrans step (cross-attn tokens only when enabled).
- **Search space `default`:** LR `3e-6`–`8e-4` log, batch from probed grid, `ema_decay` ∈ `{0, 0.99, 0.995, 0.999}`, noise schedule, loss weighting, prediction target; optional `max_scale` tune when `training.max_scale_tuning: true`.
- **Search space `lr_only`:** LR only (sweep default: fixed `3e-5`); `ema_decay` taken from `training.diffusion_ema_decay` (default **0.99**).
- **Optuna:** `TPESampler`, `HyperbandPruner`; EMA shadow weights updated during training when `ema_decay > 0`; promoted `best.pt` uses EMA weights.
- **Outputs:** `{checkpoint_dir}/{subset_id}/coarse/best.pt` + `metadata.json` (`tuned_params`).
- **Skip when:** `best.pt` + `metadata.json` exist, or `reuse_tuned_params_from` copies HP from another config (still retrains with current policy `max_scale`).

#### Phase 3 — Fine diffusion finetune HP (`diffusion_fine_finetune_hp`)

Same machinery as Phase 2 for the **fine** residual denoiser.

- **Entry:** `FineDiffusionFinetuneHPPhase` (`diffusion_stage: fine`).
- **YAML defaults:** same as Phase 2 (`n_trials: 20`, `max_epochs: 20`, `patience: 8`).
- **Requires:** completed Phase 2 coarse `best.pt` (fine conditions on coarse at inference; training uses **GT** coarse channel).
- **Warm-start:** `pretrained_fine/pretrained_diffusion.pt` from Phase 1.
- **Outputs:** `{subset_id}/fine/best.pt` + `metadata.json`.
- **Triple-scale:** optional `diffusion_finer_finetune_hp` after Phase 3 when `use_triple_scale: true` (not in current flat-subset defaults).

#### Phase 4 — Staged eval (`staged_eval`)

Loads coarse + fine `best.pt`, runs chained sampling, writes metrics and optional viz.

- **Entry:** `StagedEvalPhase`.
- **YAML defaults:** `probabilistic_sampler: dpmpp`, `probabilistic_num_inference_steps: 20`, `probabilistic_n_samples: 20`, `tune_sampler: false`, `eval_test_fraction: 1.0`, `test_stride: 4`, `batch_size: 8`.
- **Inference:** sample coarse → sample fine conditioned on sampled coarse → `decode_dual` to normalized values → denormalize.
- **Metrics:** CRPS / top-k from DPM++ sample ensemble; `sample_mean_mse/mae`; separate **anchor** one-shot metrics (`anchor_mse`, `anchor_mae`).
- **Outputs:** `results/partials/{dataset}_staged_anchor.json`, `{subset_id}/staged_results.json`, raw NPZ under `results/raw/`.
- **Skip when:** partial JSON has full metric set and raw NPZ artifacts exist (re-runs if anchor or sample-mean fields missing).
- **Baselines:** finetuned iTransformer evaluated alongside diffusion when enabled in phase.

### Caching and resume

| Artifact | Phase |
|----------|-------|
| `pretrained_{coarse,fine}/pretrained_diffusion.pt` | 1 |
| `{subset_id}_itransformer_finetuned.pt` | iTrans |
| `{subset_id}/coarse/best.pt`, `.../fine/best.pt` | 2, 3 |
| `results/partials/*_staged_anchor.json` | 4 |

`should_skip` on each phase checks these paths. Reuse flags (`reuse_pretrain_from_config`, `reuse_checkpoint_from_config`, `reuse_tuned_params_from`) symlink/copy from a donor config so EMA sweeps and grad-accum ablations only rerun Phases 2–4.

---

## Data normalization (two layers)

1. **Dataset z-score** — train-split mean/std per variate (`load_dataset`).
2. **Per-window norm** — past mean/std applied to past+future when `use_window_normalization: true` (`_normalize_sequence`).

Synthetic pretrain: `RealTS` + `augmentation.py` (mixed generators, optional cache).

---

## Representation: hard binary CDF

Values clipped to `[-max_scale, max_scale]`, binned into `H=16` rows; occupancy map is a monotone staircase in `{0,1}` (no Gaussian blur).

**Staged (default):** same dual decomposition as joint dual-scale, but **separate models**:

- **Coarse:** full-range binning → coarse CDF map.
- **Fine:** residual within coarse bin → fine CDF map.
- **Decode:** `decode_dual(coarse, fine)` = sum of decoded coarse + fine, clamped.

**Training:** coarse stage predicts future coarse map from past maps; fine stage conditions on **GT** future coarse (not model prediction). **Inference:** sample coarse, then sample fine conditioned on sampled coarse, then decode.

---

## Binary diffusion

- Schedule: `sqrt_linear` β from `1e-5` → `0.5`, `num_steps=1000`.
- Forward: independent Bernoulli XOR flips per pixel (`BinaryDiffusionScheduler`).
- Loss: BCE on clean-bit and flip-mask heads (`out_channels=2`).
- **Stationary-flat anchor:** at `t=T−1`, anchor canvas is **constant 0.5** (not `Bernoulli(0.5)`); see `_binary_anchor_canvas_like` when `binary_anchor_input_mode=stationary_flat`.
- Combined train loss: `λ·L_reg + (1−λ)·L_anchor` with `λ=0.99`.

**Eval:** `staged_eval` uses `probabilistic_sampler: dpmpp`, `probabilistic_num_inference_steps: 20`, `probabilistic_n_samples: 20` for sweep metrics; anchor path is one-shot at max noise for the anchor loss only.

---

## Model: FactorizedDiT (staged path)

- `model_type: dit` → `FactorizedDiT` (`dit.py`), patch size `(8,8)`, `embed_dim=384`, `depth=8`, `heads=6`.
- One variate = one batch row (`BV`); self-attention is **spatial patches only** (no variate axis in DiT).
- Cross-variate signal: bottleneck cross-attention to `V` iTransformer tokens (`iTransformerTokenAdapter` in `unet.py`) when `disable_cross_attention=false`.
- With `use_guidance_channel=false` (default), no ghost forecast channel on the canvas.
- Fine-stage `cond` includes **GT coarse CDF channel** during training; coarse channel from sampled coarse at inference.
- Optional **EMA** shadow weights during finetune when `training.diffusion_ema_decay > 0` (default **0.99**).

Chunking: `unet_max_chunk_size` caps `BV` through the denoiser for memory.

---

## iTransformer

Frozen encoder; finetuned per dataset/subset. Provides bottleneck context tokens when cross-attention is enabled. With guidance channel off, it does **not** add a pixel ghost map. Cold-start finetune on real data (`itrans_finetune_hp`, `cold_start: true`).

---

## Hyperparameters

Read merged YAML — do not rely on stale `pipeline_config.py` module defaults.

- **Base:** `configs/base/binary_staged.yaml` (experiment + training + phases).
- **Fixed LR sweep base:** `configs/base/fixed_lr_pipeline_base.yaml` (`3e-5`, `lr_only` HP phases).
- **Flat anchor leaf:** `configs/binary_anchor_stationary_flat.yaml` sets `binary_anchor_input_mode: stationary_flat`.
- **EMA 0.99 leaf:** `configs/binary_anchor_stationary_flat_subsets_ema099.yaml` sets `diffusion_ema_decay: 0.99`.

Optuna LR range when tuning: `finetune_hp_lr_min/max` = **`3e-6` – `2e-4`** log-uniform (`lr_only` search space).

---

## Pitfalls

1. **Staged ≠ joint dual-scale** — `use_dual_scale: false` in yaml; two checkpoints, chained eval.
2. **Double normalization** — dataset z-score then per-window norm.
3. **`image_height` must divide patch size** (16 / 8 = 2 patches tall).
4. **Training flags must reach `PipelineState`** — `training.*` keys need wiring (`training_value()` / `apply_training_section_to_state`); module globals alone are not enough for Optuna paths.
5. **Subset ckpt paths** use `subset_id` (e.g. `weather_4v_s2`), not bare dataset name.

---

## Legacy: joint dual-scale (`use_dual_scale: true`)

Single `FactorizedDiT` forward on interleaved coarse/fine batch rows (`BV×2`), `dual_scale_fine_weight` (e.g. 0.75), `dual_scale_independent_timesteps`, cross-scale attention at DiT bottleneck, guidance ghost on canvas when `use_guidance_channel: true`. Documented in git history and `configs/binary_dual_scale.yaml`; kept for ablations, not current flat-subset sweeps.

**Slurm / venv / classic Phase 0A–2C** monolithic pretrain paths still exist in `train_multivariate_pipeline.py` for old manifests; new work uses YAML phases only.

---

## Trimmed from older versions (intentionally)

- Full Slurm DAG and `$STORE/venv` preamble — see `submit_grid.sh` / cluster scripts.
- Classic monolithic phases (1A/1B/2A/2B/2C joint model) — see Legacy section; staged pipeline above is canonical.
- Step-by-step `iTransformerTokenAdapter` tensor algebra — see `unet.py` + `guidance.py`.
- Full FactorizedDiT AdaLN block diagram — see `dit.py`.
- Duplicate hyperparameter tables mirroring yaml — single source: `configs/base/binary_staged.yaml`.
- Gaussian / U-Net / ordinal paths unless you enable those `diffusion_type`s.
