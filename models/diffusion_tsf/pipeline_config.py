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
    - iTransformer batch grids + diffusion probe constants (``diffusion_probe_max_candidate``)

What this file does NOT cover (still picked per run on the CLI):
    - dispatch (`--mode`, `--dataset`, `--variate-indices`, `--subset-id`, `--n-variates`)
    - paths (`--checkpoint-dir`, `--results-dir`, `--synth-cache-dir`)
    - run mode (`--smoke-test`, `--seed`, `--resume`, `--ddp`, `--wandb`, `--fresh`)
"""

from typing import List, Optional


# ---- Sequence / window lengths ---------------------------------------------------

LOOKBACK_LENGTH: int = 1024
FORECAST_LENGTH: int = 192

# iTransformer paper benchmarks anchor at T=96 for ETTh1; we use the same lookback
# for both the diffusion model and the iTransformer guidance.
ITRANSFORMER_SEQ_LEN: int = 1024

# Predict the last K observed steps alongside the forecast horizon. The diffusion
# model denoises a (K + H)-wide region; the K steps are discarded at inference.
LOOKBACK_OVERLAP: int = 8
PAST_LOSS_WEIGHT: float = 0.3


# ---- Backbone selection ------------------------------------------------------------

# "unet" -> ConditionalUNet2D (default), "dit" -> FactorizedDiT.
# Both backbones share the same factorized per-variate call site.
MODEL_TYPE: str = "unet"


# ---- 2D map / U-Net topology ------------------------------------------------------

IMAGE_HEIGHT: int = 64
UNET_CHANNELS: List[int] = [64, 128, 256]
ATTENTION_LEVELS: List[int] = [2]
DISABLE_CROSS_ATTENTION: bool = False


# ---- DiT topology (used when MODEL_TYPE == "dit") ----------------------------------

DIT_PATCH_SIZE: tuple = (8, 8)
DIT_EMBED_DIM: int = 384
DIT_DEPTH: int = 8
DIT_NUM_HEADS: int = 6
DIT_MLP_RATIO: float = 4.0
DIT_DROPOUT: float = 0.0

USE_AMP: bool = True               # bfloat16 mixed precision
USE_GRADIENT_CHECKPOINTING: bool = True
UNET_MAX_CHUNK_SIZE: int = 128     # To prevent OOM by chunking variates through U-Net


# ---- Default variate count (only used when --n-variates is not supplied) ----------

N_VARIATES_DEFAULT: int = 7


# ---- Synthetic data (RealTS) ------------------------------------------------------
#
# Virtual dataset length ``num_samples`` is what DataLoader uses as ``len(dataset)``
# (one "epoch" of indices 0 .. num_samples-1). The on-disk pool can be larger.
#
# SYNTHETIC_SAMPLES_HP_TUNE / SYNTHETIC_SAMPLES_DIFF_TUNE
#     Virtual lengths for Phase 1A (iTransformer HP) and Phase 1B (diffusion HP).
#
# SYNTHETIC_SAMPLES_MIN / SYNTHETIC_SAMPLES_CAP
#     Floor / optional ceiling for the TOTAL synthetic pool size.
#     CAP=None means no upper limit.
#
# PRETRAIN_SYNTHETIC_SAMPLES_OVERRIDE
#     If set, skip auto-sizing and use exactly this many virtual samples per epoch.
#
# With a disk ``cache_dir``, RealTS uses an epoch-strided pool: ``train_n * max_epochs
# + val_tail`` rows so each synthetic training epoch draws a fresh block (validation
# tail is fixed at the end). Smoke / no-cache runs use a single block.
#
PRETRAIN_SYNTHETIC_SAMPLES_OVERRIDE: Optional[int] = None

SYNTHETIC_SAMPLES_HP_TUNE: int = 20_000   # Phase 1A: virtual len (per epoch)
SYNTHETIC_SAMPLES_DIFF_TUNE: int = 10_000  # Phase 1B: virtual len (per epoch)
SYNTHETIC_SAMPLES_MIN: int = 4_096
SYNTHETIC_SAMPLES_CAP: Optional[int] = 50000  # None = no cap on TOTAL pool size


# ---- Phase 1 (synthetic pretrain / HP) --------------------------------------------
# ... rest of file (Phase 1/2 constants) ...
# I will actually replace the helper functions at the end too.


# ---- Phase 1 (synthetic pretrain / HP) --------------------------------------------

# Phase 1A: iTransformer HP tuning on synthetic data. Best HP-tuning model is saved
# directly as `itransformer.pt` (no separate "full" pretrain step).
#
# Paper-faithful setup: fixed 10 epochs, no early stopping. Only LR is searched.
N_ITRANS_HP_TRIALS: int = 3
ITRANS_HP_PRETRAIN_MAX_EPOCHS: int = 10

# Fallback pretrain (only runs if HP cache is missing).
PRETRAIN_EPOCHS: int = 10

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
# `{subset}_itransformer_finetuned.pt`. Paper-faithful: 10 epochs, no early stopping.
ITRANS_HP_FINETUNE_MAX_EPOCHS: int = 10
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
# Optuna log-uniform bounds for Phase 2B only (finetune_hp_objective).
FINETUNE_HP_LR_MIN: float = 3e-6
FINETUNE_HP_LR_MAX: float = 2e-4


# ---- Batch-size search grids ------------------------------------------------------

# iTransformer paper uses a fixed batch size of 32 and does not tune it.
ITRANS_PAPER_BATCH_SIZE: int = 32
# iTransformer paper LR grid (categorical). Used as the only iTrans HP search axis.
ITRANS_PAPER_LR_GRID: List[float] = [1e-3, 5e-4, 1e-4]
# iTransformer paper dropout default; not tuned.
ITRANS_PAPER_DROPOUT: float = 0.1

# Legacy Optuna categorical grids — only used if ``fixed_batch_size`` is omitted in
# ``diffusion_hp_objective`` / ``finetune_hp_objective``. Current pipeline always
# probes once and passes that size, so these lists do not affect Phase 1B / 2B.
DIFFUSION_BATCH_SIZES: List[int] = [16]
FINETUNE_BATCH_SIZES: List[int] = [4, 8, 16]


# ---- Diffusion batch probing (Phase 1B HP, fallback pretrain, Phase 2B HP) -----

# Factorized DiffusionTSF flattens batch×variates as one U-Net batch (effective ~ B·V).
# ``select_diffusion_batch_size`` binary-searches up to ``diffusion_probe_max_candidate(V)``;
# OOM during the probe lowers B automatically (then a 0.8 safety margin is applied).
DIFFUSION_PROBE_TARGET_EFFECTIVE_BATCH: int = 512
DIFFUSION_PROBE_MAX_BATCH_CAP: int = 128   # upper bound on per-device batch B
DIFFUSION_PROBE_MIN_BATCH: int = 1


def diffusion_probe_max_candidate(n_variates: int, smoke_test: bool) -> int:
    """Upper bound (even, ≥2) for diffusion batch-size binary search for ``n_variates``.

    Larger ``n_variates`` ⇒ smaller candidate hi so the first probes do not try
    catastrophic B·V. Small-V datasets can use larger B up to ``DIFFUSION_PROBE_MAX_BATCH_CAP``.
    """
    if smoke_test:
        return 8
    V = max(int(n_variates), 1)
    raw = DIFFUSION_PROBE_TARGET_EFFECTIVE_BATCH // V
    hi = min(DIFFUSION_PROBE_MAX_BATCH_CAP, max(DIFFUSION_PROBE_MIN_BATCH, raw))
    if hi % 2 != 0:
        hi -= 1
    return max(DIFFUSION_PROBE_MIN_BATCH, hi)


# ---- Loss terms -------------------------------------------------------------------

EMD_LAMBDA: float = 0.2
GUIDANCE_PENALTY_WEIGHT: float = 0.0  # small penalty for deviating from iTransformer guidance

# ---- Evaluation -------------------------------------------------------------------

EVAL_NUM_SAMPLES: int = 30  # diffusion eval averages this many samples per window


# ---- Synthetic helpers (after phase constants) ------------------------------------

def resolve_synthetic_params(
    requested_n: int, 
    requested_cap: int, 
    smoke_test: bool
) -> tuple[int, int]:
    """Resolve (num_samples, capacity) respecting SYNTHETIC_SAMPLES_CAP as total budget."""
    if smoke_test:
        return 4, 1
        
    n = requested_n
    cap = requested_cap
    
    if SYNTHETIC_SAMPLES_CAP is not None:
        total = n * cap
        if total > SYNTHETIC_SAMPLES_CAP:
            # Scale down n first while keeping it at least SYNTHETIC_SAMPLES_MIN
            n = max(SYNTHETIC_SAMPLES_MIN, SYNTHETIC_SAMPLES_CAP // cap)
            # If total is still over, scale down cap
            if n * cap > SYNTHETIC_SAMPLES_CAP:
                cap = max(1, SYNTHETIC_SAMPLES_CAP // n)
                
    return int(n), int(cap)


def synthetic_epoch_capacity_itrans_hp() -> int:
    return ITRANS_HP_PRETRAIN_MAX_EPOCHS


def synthetic_epoch_capacity_diff_hp() -> int:
    return DIFFUSION_HP_MAX_EPOCHS


def synthetic_epoch_capacity_pretrain_itrans() -> int:
    return PRETRAIN_EPOCHS


def synthetic_epoch_capacity_pretrain_diffusion() -> int:
    return PRETRAIN_DIFFUSION_MAX_EPOCHS


def resolve_pretrain_virtual_dataset_size(smoke_test: bool) -> int:
    """Virtual ``len`` of the synthetic dataset for fallback full pretrain (iTrans / diffusion).

    This is indices per epoch (not the on-disk pool row count). With a disk cache,
    RealTS allocates ``train_n * max_epochs + val_tail`` rows for non-repeating
    synthetic epochs (see ``synthetic_epoch_capacity_pretrain_*``).
    """
    if smoke_test:
        return 4
    if PRETRAIN_SYNTHETIC_SAMPLES_OVERRIDE is not None:
        return max(4, int(PRETRAIN_SYNTHETIC_SAMPLES_OVERRIDE))
    
    steps = 32 + 48 * PRETRAIN_EPOCHS
    steps = max(64, steps)
    ref_bs = 8
    requested_n = steps * ref_bs
    
    # Use the larger of the two possible capacities to ensure we stay under cap
    max_cap = max(PRETRAIN_EPOCHS, PRETRAIN_DIFFUSION_MAX_EPOCHS)
    n, _ = resolve_synthetic_params(requested_n, max_cap, smoke_test)
    return n
