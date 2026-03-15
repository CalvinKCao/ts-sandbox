# Task: U-Net Full-Variate Training with bf16, H=96, and Optimized Data Gen

## Goal

Modify the existing pipeline to train the **U-Net** backbone directly on **full-variate datasets** (traffic=861, electricity=321, etc.) WITHOUT splitting into 32-dim subsets. Use bf16 mixed precision, image height=96, and optimized synthetic data generation.

This is a new training path alongside the existing pipeline — create a new shell script `run_unet_fullvar.sh` and a new Slurm wrapper `slurm_unet_fullvar.sh`.

---

## Context

The codebase already has:
- `pipeline.sh` — the master script that orchestrates everything via `train_7var_pipeline.py`
- `slurm_pipeline.sh` — Slurm wrapper for `pipeline.sh`
- `slurm_ci_dit_test.sh` — self-resubmitting Slurm script for CI-DiT experiments (good template)
- `train_7var_pipeline.py` — Python entry point that already supports `--model-type`, `--amp`, `--image-height` CLI flags
- `models/diffusion_tsf/augmentation.py` — contains `generate_multivariate_synthetic_data()` which is the data gen bottleneck

The key insight from prior analysis: the U-Net can handle 861 variates directly because its body (64/128/256 channels) is V-independent — only the first/last conv scales with V. Training is only ~2.9× slower per step vs V=7. The CI-DiT approach (running a full transformer per variate) is ~15× slower.

---

## Changes Required

### 1. Optimize data generation in `augmentation.py`

**File:** `models/diffusion_tsf/augmentation.py`, function `generate_multivariate_synthetic_data()`

**Problem:** The current code has a double loop:
```python
for s in range(num_samples):          # 100K
    for v in range(num_vars):          # 861 — generate base series
        base = gen_func(length)
    for v in range(num_vars):          # 861 — augment with cross-var coupling
        others = [series_list[j] for j in range(num_vars) if j != v]  # builds 860-item list
        y_aug = informative_covariate_augmentation(y, others, ...)
```

For V=861 this takes ~65 hours for 100K samples. The cross-variate augmentation loop is O(V²) in list operations and provides minimal benefit for synthetic pretraining.

**Fix:** Add a `skip_cross_var_aug` parameter (default `False`). When `True` (or when `num_vars > 32` as a heuristic), skip the second augmentation loop entirely — just generate independent base series, normalize, and return. This reduces V=861 generation from ~65h to ~14h per 100K samples.

```python
def generate_multivariate_synthetic_data(
    num_samples, num_vars, length,
    hyperparams=None, seed=None,
    skip_cross_var_aug=False          # NEW PARAM
):
```

When `skip_cross_var_aug` is True OR `num_vars > 32`:
- Generate all V base series per sample (keep the first loop)
- Skip the `informative_covariate_augmentation` loop entirely
- Just normalize and stack directly into the output array

**GOTCHA:** Don't break the existing V=7 and V=32 paths. They should still use cross-var augmentation by default. Only skip for high-V.

### 2. Enable disk caching for synthetic data pool

The `RealTS` class in `realts.py` already supports `cache_dir` — it saves the pool as a `.npy` file and loads via mmap. But `train_7var_pipeline.py` doesn't pass `cache_dir` when creating the dataloader.

**File:** `models/diffusion_tsf/train_7var_pipeline.py`

Wire up `cache_dir` in the `get_synthetic_dataloader()` calls. Use `{checkpoint_dir}/synth_cache/` as the cache directory. This way the 100K pool is generated once and reused for both iTransformer HP tuning and diffusion pretraining.

Also pass `skip_cross_var_aug=True` through to `generate_multivariate_synthetic_data` when the variate count exceeds 32. You'll need to thread this parameter through `RealTS.__init__` → `generate_multivariate_synthetic_data()`.

### 3. Reduce synthetic pool to 75K for high-variate

In `train_7var_pipeline.py`, the constants `SYNTHETIC_SAMPLES` and `SYNTHETIC_SAMPLES_HP` control pool sizes. For the full-var script, use 75K instead of 100K for the main pool. This can be done via a new CLI flag `--synthetic-samples` or just by setting it in the shell script wrapper.

Currently the pipeline has:
```python
SYNTHETIC_SAMPLES = 100000
SYNTHETIC_SAMPLES_HP = 10000
```

Add a CLI arg `--synthetic-samples` that overrides `SYNTHETIC_SAMPLES`. The shell script will pass `--synthetic-samples 75000` for high-variate runs.

### 4. Transfer iTransformer HPs from V=7

For high-variate runs, the iTransformer HP search (20 trials) is extremely slow because iTransformer attention scales with V². 

Add a CLI flag `--itransformer-trials` to override the number of iTransformer HP trials. The shell script will pass `--itransformer-trials 3` for high-variate datasets (just validate that the transferred HPs work, don't do full search).

Currently `ITRANSFORMER_HP_TRIALS = 20` is hardcoded. Make it overridable.

### 5. Create `run_unet_fullvar.sh`

New shell script that invokes `train_7var_pipeline.py` with the right flags for full-variate U-Net training. Model this after how the existing `pipeline.sh` works.

Key settings:
```bash
# Force U-Net backbone
MODEL_TYPE="unet"

# bf16 mixed precision
AMP_FLAG="--amp"

# Reduced image height
IMAGE_HEIGHT=96

# Full-variate: set SUBSET_THRESHOLD very high so nothing gets split
SUBSET_THRESHOLD=999999

# Reduced synthetic pool for high-V
SYNTHETIC_SAMPLES=75000

# Fewer iTransformer HP trials for high-V
ITRANSFORMER_TRIALS=3
```

The script should accept the same CLI interface as `pipeline.sh` (--dataset, --smoke-test, --checkpoint-dir, --results-dir, etc.).

**IMPORTANT:** Set `SUBSET_THRESHOLD=999999` in the pipeline so that `discover_dims()` treats traffic (861 cols) and electricity (321 cols) as their native dimensionality instead of splitting into 32-dim subsets. This means `dim=861` for traffic and `dim=321` for electricity. Each gets its own pretrained model at that dimensionality.

### 6. Create `slurm_unet_fullvar.sh`

Self-resubmitting Slurm script for Killarney, modeled after `slurm_ci_dit_test.sh`.

**Cluster details (CRITICAL — these caused errors before):**
- Account: `aip-boyuwang`
- Killarney ONLY has H100 partitions: `gpubase_h100_b1` (3h), `gpubase_h100_b2` (12h), `gpubase_h100_b3` (24h), `gpubase_h100_b4` (3 days), `gpubase_h100_b5` (7 days)
- There is NO `gpubase_bygpu_b1` or generic GPU partition — those don't exist and will error
- GPU spec in sbatch: `--gpus-per-node=h100:1`
- Email: `ccao87@uwo.ca`
- Must run from `/scratch/$USER/ts-sandbox`
- Storage: `$PROJECT/$USER/diffusion-tsf-fullvar` (use a separate storage root so it doesn't conflict with the existing pipeline or CI-DiT checkpoints)

Two modes:
- `--smoke-test`: partition `gpubase_h100_b1`, time 0:30:00, mem 20G, 4 CPUs
- Full run: partition `gpubase_h100_b4`, time 3-00:00:00, mem 60G, 6 CPUs (traffic needs ~4 days)

When no `--dataset` is specified, default to running traffic (861-var) as the primary test.

**Environment setup** (copy from `slurm_ci_dit_test.sh`):
```bash
module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9
```

Venv: reuse `$PROJECT/$USER/diffusion-tsf/venv` if it exists (shared with main pipeline).

### 7. Multiprocess data generation (optional but recommended)

In `train_7var_pipeline.py`, when creating the synthetic dataloader, pass `num_workers=4` to the DataLoader (it currently defaults to 0). This parallelizes batch preparation across CPU cores.

Also, in the `generate_multivariate_synthetic_data()` function, consider using `multiprocessing.Pool` to parallelize the outer sample loop. But this is secondary — the cross-var aug skip is the biggest win.

---

## What NOT to change

- Don't modify the existing `pipeline.sh` or `slurm_pipeline.sh` — those are the production scripts for the current approach
- Don't change the U-Net architecture in `unet.py` — it already handles arbitrary channel counts
- Don't change `diffusion_model.py`'s standard (non-CI-DiT) forward path — it already works with any V
- Don't change `preprocessing.py` — the 2D encoding already handles multivariate data vectorized
- Don't touch anything in `ci_dit.py` — that's a separate experiment running on the cluster right now

---

## Verification

After implementing, verify with a local quick test (if GPU available) or a smoke test on Killarney:

```bash
# Local (if you have a GPU):
python -m models.diffusion_tsf.train_7var_pipeline \
    --mode full --dataset ETTh1 --smoke-test \
    --model-type unet --amp --image-height 96

# Killarney:
./slurm_unet_fullvar.sh --smoke-test --dataset ETTh1
```

The smoke test should complete in <5 minutes and validate the full pipeline end-to-end (1 HP trial, 1 epoch, few samples).

---

## Expected Training Time (H100, full run)

For traffic (V=861) with all optimizations:
| Phase | Time |
|---|---|
| Data gen (75K, cached, no cross-var aug) | ~2-3h |
| iTransformer HP (3 trials) | ~8h |
| Diffusion HP (8 trials, 10K samples) | ~9h |
| Diffusion pretrain (~55 epochs w/ patience) | ~77h |
| **Total** | **~96h (~4 days)** |

This is why the Slurm script requests `gpubase_h100_b4` (3-day partition). If it doesn't finish, resubmit with `--resume` on `gpubase_h100_b5` (7-day).

---

## Files to create/modify (summary)

| File | Action |
|---|---|
| `models/diffusion_tsf/augmentation.py` | Add `skip_cross_var_aug` param to `generate_multivariate_synthetic_data()` |
| `models/diffusion_tsf/realts.py` | Thread `skip_cross_var_aug` through `RealTS.__init__` |
| `models/diffusion_tsf/dataset.py` | Thread `skip_cross_var_aug` through `get_synthetic_dataloader()` |
| `models/diffusion_tsf/train_7var_pipeline.py` | Add CLI flags: `--synthetic-samples`, `--itransformer-trials`, wire up `cache_dir` and `skip_cross_var_aug` |
| `run_unet_fullvar.sh` | **NEW** — shell script wrapper |
| `slurm_unet_fullvar.sh` | **NEW** — self-resubmitting Slurm script for Killarney |
