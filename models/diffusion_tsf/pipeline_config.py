"""Single source of truth for diffusion-TSF training pipeline knobs.

Edit this file directly to change defaults. The pipeline (`train_multivariate_pipeline.py`)
and the Slurm driver (`run.sh`) both read from here; nothing in either script overrides
these values via CLI any more.

The values currently baked in reproduce the behavior of running:

    ./run.sh --variant default \\
        --itrans-finetune-max-epochs 100 \\
        --itrans-finetune-patience 15 \\
        --itransformer-trials 7

at the time this file was introduced.

What this file covers:
    - sequence/window lengths (lookback, forecast, iTransformer seq_len, overlap)
    - 2D map height + U-Net topology + AMP/checkpointing flags
    - synthetic pool sizing
    - per-phase trial counts, max epochs, and early-stop patience
    - batch-size search grids per phase

What this file does NOT cover (still picked per run on the CLI):
    - dispatch (`--mode`, `--dataset`, `--variate-indices`, `--subset-id`, `--n-variates`)
    - paths (`--checkpoint-dir`, `--results-dir`, `--synth-cache-dir`)
    - run mode (`--smoke-test`, `--seed`, `--resume`, `--ddp`, `--wandb`, `--fresh`)
"""

from typing import List, Optional


# ---- Sequence / window lengths ---------------------------------------------------

LOOKBACK_LENGTH: int = 512
FORECAST_LENGTH: int = 96

# iTransformer's inverted embedding uses seq_len as the Linear input width. Papers
# typically benchmark on ≤336 for hourly ETT; the diffusion model still sees the full
# LOOKBACK_LENGTH and slices for iTransformer guidance.
ITRANSFORMER_SEQ_LEN: int = 336

# Predict the last K observed steps alongside the forecast horizon. The diffusion
# model denoises a (K + H)-wide region; the K steps are discarded at inference.
LOOKBACK_OVERLAP: int = 8
PAST_LOSS_WEIGHT: float = 0.3


# ---- 2D map / U-Net topology ------------------------------------------------------

IMAGE_HEIGHT: int = 64
UNET_CHANNELS: List[int] = [64, 128, 256]
ATTENTION_LEVELS: List[int] = [2]

USE_AMP: bool = True               # bfloat16 mixed precision
USE_GRADIENT_CHECKPOINTING: bool = False


# ---- Default variate count (only used when --n-variates is not supplied) ----------

N_VARIATES_DEFAULT: int = 7


# ---- Synthetic data pool sizes ----------------------------------------------------

SYNTHETIC_SAMPLES_FULL: int = 100_000
SYNTHETIC_SAMPLES_HP_TUNE: int = 20_000   # Phase 1A iTransformer HP pool
SYNTHETIC_SAMPLES_DIFF_TUNE: int = 10_000  # Phase 1B diffusion HP pool
SYNTHETIC_SAMPLES_MIN: int = 4_096
SYNTHETIC_SAMPLES_CAP: int = 100_000

# Optional fixed pretrain pool size; when None, the pipeline auto-sizes from PRETRAIN_EPOCHS.
PRETRAIN_SYNTHETIC_SAMPLES_OVERRIDE: Optional[int] = None


# ---- Phase 1 (synthetic pretrain / HP) --------------------------------------------

# Phase 1A: iTransformer HP tuning on synthetic data. Best HP-tuning model is saved
# directly as `itransformer.pt` (no separate "full" pretrain step).
N_ITRANS_HP_TRIALS: int = 7
ITRANS_HP_PRETRAIN_MAX_EPOCHS: int = 30
ITRANS_HP_PRETRAIN_PATIENCE: int = 5

# Fallback pretrain (only runs if HP cache is missing).
PRETRAIN_EPOCHS: int = 10
PRETRAIN_PATIENCE: int = 5

# Phase 1B: Diffusion HP tuning on synthetic data. Best HP-tuning model is saved
# directly as `diffusion.pt` (no separate "full" pretrain step).
N_DIFFUSION_HP_TRIALS: int = 3
DIFFUSION_HP_MAX_EPOCHS: int = 15
DIFFUSION_HP_PATIENCE: int = 10

# Fallback diffusion pretrain (only runs if HP cache is missing).
PRETRAIN_DIFFUSION_EPOCHS: int = 3
PRETRAIN_DIFFUSION_MAX_EPOCHS: int = 15


# ---- Phase 2 (real-data fine-tune / HP) -------------------------------------------

# Phase 2A: iTransformer HP tuning on real data. Best HP model is saved directly as
# `{subset}_itransformer_finetuned.pt`.
ITRANS_HP_FINETUNE_MAX_EPOCHS: int = 100
ITRANS_HP_FINETUNE_PATIENCE: int = 15
# Synthetic pretrain on RealTS data tends to plateau at the unit-variance mean
# predictor, which makes warm-started fine-tunes barely move. Cold-start gives
# Phase 2A a fair shot.
ITRANS_REAL_COLD_START: bool = True

# Phase 2B: Diffusion HP tuning on real data (LR-only search; batch size auto-probed).
# The best trial's checkpoint is reused as the final fine-tuned model — there is no
# separate "Phase 2C" full retrain.
N_FINETUNE_HP_TRIALS: int = 3
HP_TUNE_EPOCHS: int = 10
HP_TUNE_PATIENCE: int = 5


# ---- Batch-size search grids ------------------------------------------------------

# iTransformer HP tuning adapts these based on N_VARIATES at runtime.
ITRANS_BATCH_SIZES: List[int] = [64, 128, 256]
DIFFUSION_BATCH_SIZES: List[int] = [16]
FINETUNE_BATCH_SIZES: List[int] = [4, 8, 16]


# ---- Evaluation -------------------------------------------------------------------

EVAL_NUM_SAMPLES: int = 30  # diffusion eval averages this many samples per window
