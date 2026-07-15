
"""
Usage:
    python -m models.diffusion_tsf.train_multivariate_pipeline --config configs/foo.yaml --dataset ETTh1
    python -m models.diffusion_tsf.train_multivariate_pipeline --smoke-test ...
"""

import argparse
import errno
import gc
import importlib.util
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import optuna
from optuna.samplers import TPESampler
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

# Setup path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.diffusion_tsf.config import DiffusionTSFConfig
from models.diffusion_tsf.diffusion_model import DiffusionTSF
from models.diffusion_tsf.realts import get_synthetic_dataloader
from models.diffusion_tsf.guidance import iTransformerGuidance, PatchDecoderGuidance
from models.diffusion_tsf.patch_guidance_stack import PatchGuidanceStack, PatchGuidanceStackConfig
from models.diffusion_tsf.ordinal_window_norm import (
    build_global_ladder_from_training,
    ordinal_encode,
    ranks_to_unit,
)
from models.diffusion_tsf.storage_paths import resolve_checkpoint_dir, resolve_results_dir
from models.diffusion_tsf.pipeline.data_subset import resolve_data_subset

DATASETS_DIR = os.path.join(project_root, "datasets")
CHECKPOINT_DIR = resolve_checkpoint_dir(script_dir)
RESULTS_DIR = resolve_results_dir(script_dir)
SYNTH_CACHE_DIR: Optional[str] = None

def is_main_process() -> bool:
    """True on the coordinator process (not an Optuna child worker)."""
    from models.diffusion_tsf.pipeline.optuna_parallel import is_optuna_child_worker
    return not is_optuna_child_worker()


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def unwrap_model(model: nn.Module) -> nn.Module:
    return model


def require_tuned_param(params: Dict, key: str, stage_name: str):
    """Fail fast when a required tuned hyperparameter is missing."""
    if params is None:
        raise RuntimeError(f"{stage_name} requires tuned params, got None.")
    if key not in params:
        raise RuntimeError(
            f"{stage_name} requires tuned param '{key}', but tuning output is missing it."
        )
    return params[key]


def fixed_deterministic_anchor_hp() -> Tuple[float, float]:
    """Fixed anchor hyperparameters from YAML (not Optuna-tuned)."""
    return DETERMINISTIC_ANCHOR_LAMBDA, DETERMINISTIC_ANCHOR_ALPHA


def anchor_kwargs_from_params(params: Optional[Dict] = None) -> Dict:
    """Kwargs for create_diffusion_model from CLI anchor settings."""
    del params  # kept for call-site compatibility; anchor HP is not tuned
    if not DETERMINISTIC_ANCHOR_LOSS:
        return {}
    anchor_lambda, anchor_alpha = fixed_deterministic_anchor_hp()
    return {
        'use_deterministic_anchor_loss': True,
        'deterministic_anchor_lambda': anchor_lambda,
        'deterministic_anchor_alpha': anchor_alpha,
    }


def diffusion_arch_config_dict() -> Dict[str, Any]:
    """Architecture/runtime flags needed to reconstruct diffusion checkpoints."""
    return {
        'image_height': IMAGE_HEIGHT,
        'coarse_image_height': COARSE_IMAGE_HEIGHT,
        'fine_image_height': FINE_IMAGE_HEIGHT,
        'finer_image_height': FINER_IMAGE_HEIGHT,
        'max_scale': MAX_SCALE,
        'staged_representation': STAGED_REPRESENTATION,
        'window_norm_std_floor': WINDOW_NORM_STD_FLOOR,
        'window_norm_low_var_threshold': WINDOW_NORM_LOW_VAR_THRESHOLD,
        'window_norm_low_var_unit_std': WINDOW_NORM_LOW_VAR_UNIT_STD,
        'window_norm_low_var_unit_std_per_variate': WINDOW_NORM_LOW_VAR_UNIT_STD_PER_VARIATE,
        'lookback_overlap_center_shift': LOOKBACK_OVERLAP_CENTER_SHIFT,
        'window_norm_center': WINDOW_NORM_CENTER,
        'use_triple_scale': USE_TRIPLE_SCALE,
        'diffusion_stage': DIFFUSION_STAGE,
        'use_guidance_channel': USE_GUIDANCE_CHANNEL,
        'cfg_dropout': CFG_DROPOUT,
        'disable_cross_attention': DISABLE_CROSS_ATTENTION,
        'cross_variate_context_bias': CROSS_VARIATE_CONTEXT_BIAS,
        'model_type': MODEL_TYPE,
        'diffusion_type': DIFFUSION_TYPE,
        'use_ordinal_window_norm': USE_ORDINAL_WINDOW_NORM,
        'ordinal_tie_atol': ORDINAL_TIE_ATOL,
        'binary_anchor_input_mode': BINARY_ANCHOR_INPUT_MODE,
        'dit_patch_size': DIT_PATCH_SIZE,
        'dit_embed_dim': DIT_EMBED_DIM,
        'dit_depth': DIT_DEPTH,
        'dit_num_heads': DIT_NUM_HEADS,
        'dit_mlp_ratio': DIT_MLP_RATIO,
        'dit_dropout': DIT_DROPOUT,
        'use_window_normalization': USE_WINDOW_NORMALIZATION,
        'window_norm_center': WINDOW_NORM_CENTER,
        'zero_guidance_forecast': ZERO_GUIDANCE_FORECAST,
        'window_stride': WINDOW_STRIDE,
        'binary_noise_schedule': BINARY_NOISE_SCHEDULE,
        'prediction_target': PREDICTION_TARGET,
        'loss_weighting': LOSS_WEIGHTING,
        'min_snr_gamma': MIN_SNR_GAMMA,
        'use_coordinate_channel': USE_COORDINATE_CHANNEL,
        'use_raw_lookback_cond_channel': USE_RAW_LOOKBACK_COND_CHANNEL,
        'representation_time_stride': REPRESENTATION_TIME_STRIDE,
        'past_cond_resize_to_horizon': PAST_COND_RESIZE_TO_HORIZON,
        'itrans_d_model': ITRANS_D_MODEL,
    }


# Logging - coordinator only (Optuna child workers stay quiet)
def setup_logging():
    """Setup logging - only coordinator logs to file/stdout."""
    from models.diffusion_tsf.pipeline.logging_utils import configure_pipeline_logging

    is_main = is_main_process()
    level = logging.INFO if is_main else logging.WARNING
    handlers = []
    if is_main:
        handlers.append(logging.StreamHandler(sys.stdout))
        handlers.append(logging.FileHandler(os.path.join(script_dir, 'train_multivariate.log')))

    if handlers:
        configure_pipeline_logging(level=level, handlers=handlers)
    else:
        logging.basicConfig(level=level, handlers=[logging.NullHandler()], force=True)
        configure_pipeline_logging(level=level)

    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.ERROR)
    except ImportError:
        pass

    return logging.getLogger(__name__)


# Deferred logger initialization (called after DDP setup).
# Falls back to module-level logger when imported by other scripts.
logger = logging.getLogger(__name__)


def get_git_info() -> dict:
    """Get git commit info for reproducibility."""
    import subprocess
    info = {}
    try:
        info['git_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=project_root, stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        info['git_branch'] = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=project_root, stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        info['git_dirty'] = len(subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=project_root, stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()) > 0
        # Get diff if dirty
        if info['git_dirty']:
            diff = subprocess.check_output(
                ['git', 'diff', '--stat'], cwd=project_root, stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
            info['git_diff_summary'] = diff[:1000] if len(diff) > 1000 else diff
    except Exception:
        info['git_commit'] = 'unknown'
        info['git_branch'] = 'unknown'
        info['git_dirty'] = False
    return info


def get_system_info() -> dict:
    """Get system info for reproducibility."""
    import platform
    info = {
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
        'cudnn_version': str(torch.backends.cudnn.version()) if torch.cuda.is_available() else 'N/A',
        'num_gpus': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'hostname': platform.node(),
        'platform': platform.platform(),
    }
    if torch.cuda.is_available():
        info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info['gpu_memory_gb'] = [
            round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2) 
            for i in range(torch.cuda.device_count())
        ]
    return info


# ============================================================================
# Runtime knobs — populated from YAML via patch_globals / apply_training_config.
# ============================================================================

from models.diffusion_tsf.pipeline import training_helpers as _training_helpers

LOOKBACK_LENGTH = 96
FORECAST_LENGTH = 96
ITRANSFORMER_SEQ_LEN = 96
DIFFUSION_LOOKBACK_CAP = 0
DIFFUSION_CHUNK_HORIZON = 0
REPRESENTATION_TIME_STRIDE = 1
PAST_COND_RESIZE_TO_HORIZON = True
ITRANS_LOOKBACK_LENGTH = None
IMAGE_HEIGHT = 16
COARSE_IMAGE_HEIGHT = 16
FINE_IMAGE_HEIGHT = 16
FINER_IMAGE_HEIGHT = 16
MAX_SCALE = 3.5
STAGED_REPRESENTATION = "value_precision"
WINDOW_NORM_STD_FLOOR = 1e-8
WINDOW_NORM_LOW_VAR_THRESHOLD = 0.0
WINDOW_NORM_LOW_VAR_UNIT_STD = 1.0
WINDOW_NORM_LOW_VAR_UNIT_STD_PER_VARIATE: Optional[List[float]] = None
LOOKBACK_OVERLAP_CENTER_SHIFT = False
LOOKBACK_OVERLAP = 8
PAST_LOSS_WEIGHT = 0.3
PRETRAIN_EPOCHS = 10
PRETRAIN_DIFFUSION_EPOCHS = 20
PRETRAIN_DIFFUSION_MAX_EPOCHS = 20
DIFFUSION_HP_PATIENCE = 4
SYNTHETIC_SAMPLES_HP_TUNE = 20_000
SYNTHETIC_SAMPLES_DIFF_TUNE = 10_000
SYNTHETIC_SAMPLES_MIN = 4_096
SYNTHETIC_SAMPLES_CAP = 50_000
PRETRAIN_SYNTHETIC_SAMPLES_OVERRIDE = None
HP_TUNE_EPOCHS = 20
HP_TUNE_PATIENCE = 15
N_ITRANS_HP_TRIALS = 10
N_DIFFUSION_HP_TRIALS = 8
N_FINETUNE_HP_TRIALS = 5
ITRANS_HP_PRETRAIN_MAX_EPOCHS = 10
ITRANS_HP_FINETUNE_MAX_EPOCHS = 10
ITRANS_REAL_COLD_START = True
ITRANS_PAPER_BATCH_SIZE = 32
ITRANS_PAPER_LR_GRID = [1e-3, 5e-4, 1e-4]
ITRANS_PAPER_DROPOUT = 0.1
ITRANS_D_MODEL = 512
ITRANS_D_FF = 512
ITRANS_E_LAYERS = 4
ITRANS_N_HEADS = 8
GUIDANCE_TYPE = "patch_decoder"
MMPD_PATCH_SIZE = 12
PATCH_GUIDANCE_HP_FINETUNE_MAX_EPOCHS = 10
BINARY_NOISE_SCHEDULE = "linear"
BINARY_LENGTH_MODE = "none"
BINARY_LENGTH_G = 1.0
BINARY_LENGTH_SCALE = 1.0
PREDICTION_TARGET = "x0"
LOSS_WEIGHTING = "none"
MIN_SNR_GAMMA = 5.0
BINARY_NUM_STEPS = 1000
BINARY_BETA_START = 1e-5
BINARY_BETA_END = 0.5
LR_SCHEDULER_TYPE = "none"
LR_WARMUP_EPOCHS = 0
MAX_SCALE_TUNING = False
MAX_SCALE_TUNING_RANGE = [2.5, 14.0]
USE_COORDINATE_CHANNEL = True
USE_RAW_LOOKBACK_COND_CHANNEL = False
DIFFUSION_BATCH_SIZE = 32
DIFFUSION_BATCH_SIZES = [16]
FINETUNE_BATCH_SIZES = [4, 8, 16]
FINETUNE_HP_LR_MIN = 3e-6
FINETUNE_HP_LR_MAX = 2e-4
USE_AMP = True
USE_GRADIENT_CHECKPOINTING = True
UNET_MAX_CHUNK_SIZE = 128
DISABLE_CROSS_ATTENTION = False
USE_TRIPLE_SCALE = False
DIFFUSION_STAGE = "coarse"
USE_GUIDANCE_CHANNEL = False
CFG_DROPOUT = 0.0
MODEL_TYPE = "dit"
DIFFUSION_TYPE = "binary"
USE_ORDINAL_WINDOW_NORM = False
ORDINAL_TIE_ATOL = 1e-6
GLOBAL_ORDINAL_LADDER = None
TRAIN_WINDOW_AUG = {}
PIPELINE_SEED = 42
DIT_PATCH_SIZE = (8, 8)
DIT_EMBED_DIM = 384
DIT_DEPTH = 8
DIT_NUM_HEADS = 6
DIT_MLP_RATIO = 4.0
DIT_DROPOUT = 0.0
CROSS_VARIATE_CONTEXT_BIAS = 0.0
DETERMINISTIC_ANCHOR_LOSS = True
DETERMINISTIC_ANCHOR_LAMBDA = 0.99
DETERMINISTIC_ANCHOR_ALPHA = 0.5
BINARY_ANCHOR_INPUT_MODE = "stationary_flat"
BINARY_USE_BOUNDARY_WEIGHTED_BCE = False
BINARY_CDF_DISTANCE_ALPHA = 1.0
ANCHOR_MSE_PROXY_LAMBDA = 0.5
USE_VERTICAL_DUAL_CONCAT = False
USE_WINDOW_NORMALIZATION = True
WINDOW_NORM_CENTER = "mean"
ZERO_GUIDANCE_FORECAST = False
WINDOW_STRIDE = 1
EVAL_NUM_SAMPLES = 30
EVAL_SAMPLER = "dpmpp"

N_VARIATES = 7


def resolve_synthetic_params(requested_n: int, requested_cap: int, smoke_test: bool):
    return _training_helpers.resolve_synthetic_params(
        requested_n,
        requested_cap,
        smoke_test,
        samples_cap=SYNTHETIC_SAMPLES_CAP,
        samples_min=SYNTHETIC_SAMPLES_MIN,
    )


def resolve_pretrain_virtual_dataset_size(smoke_test: bool) -> int:
    return _training_helpers.resolve_pretrain_virtual_dataset_size(
        smoke_test,
        pretrain_epochs=PRETRAIN_EPOCHS,
        pretrain_diffusion_max_epochs=PRETRAIN_DIFFUSION_MAX_EPOCHS,
        pretrain_synthetic_override=PRETRAIN_SYNTHETIC_SAMPLES_OVERRIDE,
        samples_cap=SYNTHETIC_SAMPLES_CAP,
        samples_min=SYNTHETIC_SAMPLES_MIN,
    )


def synthetic_epoch_capacity_itrans_hp() -> int:
    return ITRANS_HP_PRETRAIN_MAX_EPOCHS


def synthetic_epoch_capacity_diff_hp() -> int:
    return PRETRAIN_DIFFUSION_MAX_EPOCHS


def synthetic_epoch_capacity_pretrain_diffusion() -> int:
    return PRETRAIN_DIFFUSION_MAX_EPOCHS

# Dataset registry: name -> (path, date_col, seasonal_period)
DATASET_REGISTRY = {
    'ETTh1': ('ETT-small/ETTh1.csv', 'date', 24),
    'ETTh2': ('ETT-small/ETTh2.csv', 'date', 24),
    'ETTm1': ('ETT-small/ETTm1.csv', 'date', 96),
    'ETTm2': ('ETT-small/ETTm2.csv', 'date', 96),
    'illness': ('illness/national_illness.csv', 'date', 52),
    'exchange_rate': ('exchange_rate/exchange_rate.csv', 'date', 5),
    'weather': ('weather/weather.csv', 'date', 144),
    'electricity': ('electricity/electricity.csv', 'date', 96),
    'traffic': ('traffic/traffic.csv', 'date', 24),
    # PeMS benchmarks ship as NPZ (iTransformer Dataset_PEMS); see scripts/fetch_pems_solar.sh
    'PeMS': ('PeMS/PEMS04.npz', None, 24),
    'solar_Alabama': ('solar_Alabama/solar_Alabama.csv', 'Unnamed: 0', 96),
    # First 500k timesteps only (see datasets/dynamic/dynamic_500K.csv).
    'dynamic': ('dynamic/dynamic_500K.csv', 'date', 96),
}


def _datasets_root() -> str:
    return os.path.abspath(os.path.expanduser(DATASETS_DIR))


def _path_is_file(path: str, retries: int = 3, delay_s: float = 0.5) -> bool:
    """os.path.isfile with brief retries for NFS stale-file-handle (errno 116)."""
    for attempt in range(retries):
        try:
            return os.path.isfile(path)
        except OSError as exc:
            if exc.errno == errno.ESTALE and attempt + 1 < retries:
                time.sleep(delay_s)
                continue
            raise
    return False


def _resolve_registry_path(dataset_name: str) -> Tuple[str, Optional[str]]:
    """Return (absolute path, date_col or None for NPZ/headerless)."""
    rel, date_col, _ = DATASET_REGISTRY[dataset_name]
    path = os.path.join(_datasets_root(), rel)
    if not _path_is_file(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return path, date_col


def _load_pems_npz(path: str) -> np.ndarray:
    raw = np.load(path, allow_pickle=True)
    data = raw['data']
    if data.ndim == 3:
        data = data[:, :, 0]
    return np.asarray(data, dtype=np.float32)


def _load_solar_lines(path: str) -> np.ndarray:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split(',')])
    if not rows:
        raise ValueError(f"Empty solar dataset: {path}")
    return np.stack(rows, axis=0).astype(np.float32)


def _load_dataset_array(path: str, date_col: Optional[str]) -> np.ndarray:
    if path.endswith('.npz'):
        return _load_pems_npz(path)
    try:
        df_head = pd.read_csv(path, nrows=1)
    except Exception:
        df_head = None
    if df_head is not None and date_col and date_col in df_head.columns:
        df = pd.read_csv(path)
        cols = [c for c in df.columns if c != date_col]
        return df[cols].values.astype(np.float32)
    if df_head is not None and len(df_head.columns) > 1:
        df = pd.read_csv(path)
        cols = list(df.columns)
        if date_col and date_col in cols:
            cols = [c for c in cols if c != date_col]
        return df[cols].values.astype(np.float32)
    return _load_solar_lines(path)


def _dataset_variate_names(path: str, date_col: Optional[str], n_cols: int) -> List[str]:
    if path.endswith('.npz'):
        return [f"var_{i}" for i in range(n_cols)]
    try:
        df = pd.read_csv(path, nrows=1)
        if date_col and date_col in df.columns:
            return [c for c in df.columns if c != date_col]
        return list(df.columns)
    except Exception:
        return [f"var_{i}" for i in range(n_cols)]


def dataset_window_lengths(dataset_name: str) -> Tuple[int, int]:
    """Per-dataset (lookback, forecast) for finetune/eval; pretrain stays on pipeline defaults."""
    return LOOKBACK_LENGTH, FORECAST_LENGTH


def itrans_model_lengths(dataset_lookback: int, dataset_horizon: int) -> Tuple[int, int]:
    """iTrans seq_len / pred_len decoupled from diffusion AR chunk canvas."""
    seq_len = int(ITRANSFORMER_SEQ_LEN) if ITRANSFORMER_SEQ_LEN else dataset_lookback
    chunk_hz = int(DIFFUSION_CHUNK_HORIZON or 0)
    pred_len = min(dataset_horizon, chunk_hz) if chunk_hz > 0 else dataset_horizon
    return seq_len, pred_len


def wrap_itrans_guidance(
    model: nn.Module,
    *,
    seq_len: Optional[int] = None,
    pred_len: Optional[int] = None,
):
    """Attach iTransformer with explicit seq/pred lens (full LB, chunk forecast)."""
    from models.diffusion_tsf.guidance import iTransformerGuidance

    if seq_len is None:
        seq_len = int(ITRANSFORMER_SEQ_LEN or getattr(model, "seq_len", 96))
    if pred_len is None:
        chunk = int(DIFFUSION_CHUNK_HORIZON or 0)
        pred_len = chunk if chunk > 0 else int(FORECAST_LENGTH)
    return iTransformerGuidance(model, seq_len=int(seq_len), pred_len=int(pred_len))


def _patch_guidance_out_len() -> int:
    """Native decoder forecast length (dataset horizon, not diffusion AR chunk)."""
    return int(FORECAST_LENGTH)


def _patch_guidance_pred_len() -> int:
    chunk = int(DIFFUSION_CHUNK_HORIZON or 0)
    return chunk if chunk > 0 else int(FORECAST_LENGTH)


def _checkpoint_is_patch_guidance(ckpt: dict) -> bool:
    cfg = ckpt.get("config")
    if isinstance(cfg, dict) and cfg.get("in_len") and cfg.get("out_len") and cfg.get("patch_size"):
        return True
    sd = ckpt.get("model_state_dict")
    if not isinstance(sd, dict):
        return False
    return any(k.startswith("decoder.") or k.startswith("mixer.") for k in sd)


def create_patch_guidance_stack(
    num_vars: int,
    *,
    in_len: Optional[int] = None,
    out_len: Optional[int] = None,
    patch_size: Optional[int] = None,
) -> PatchGuidanceStack:
    cfg = PatchGuidanceStackConfig(
        in_len=int(in_len or LOOKBACK_LENGTH),
        out_len=int(out_len or _patch_guidance_out_len()),
        patch_size=int(patch_size or MMPD_PATCH_SIZE),
        data_dim=int(num_vars),
    )
    return PatchGuidanceStack(cfg)


def wrap_patch_guidance(stack: PatchGuidanceStack) -> PatchDecoderGuidance:
    ordinal_ladder = GLOBAL_ORDINAL_LADDER if USE_ORDINAL_WINDOW_NORM else None
    return PatchDecoderGuidance(
        stack,
        chunk_horizon=_patch_guidance_pred_len(),
        ordinal_ladder=ordinal_ladder,
    )


def load_patch_guidance_from_checkpoint(
    path: str,
    num_vars: int,
    device: torch.device,
    ckpt: Optional[dict] = None,
) -> PatchGuidanceStack:
    if ckpt is None:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = dict(ckpt.get("config") or {})
    cfg_dict.setdefault("data_dim", num_vars)
    cfg_dict.setdefault("in_len", LOOKBACK_LENGTH)
    cfg_dict.setdefault("out_len", _patch_guidance_out_len())
    cfg_dict.setdefault("patch_size", MMPD_PATCH_SIZE)
    cfg = PatchGuidanceStackConfig(**cfg_dict)
    stack = PatchGuidanceStack(cfg).to(device)
    stack.load_state_dict(ckpt["model_state_dict"], strict=True)
    stack.eval()
    return stack


def load_wrapped_guidance(
    ckpt_path: str,
    num_vars: int,
    device: torch.device,
    *,
    guidance_type: Optional[str] = None,
    dataset_lookback: Optional[int] = None,
    dataset_horizon: Optional[int] = None,
):
    """Load finetuned patch_decoder guidance."""
    gtype = guidance_type or GUIDANCE_TYPE
    if gtype != "patch_decoder":
        raise ValueError(f"Only patch_decoder guidance is supported; got {gtype!r}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not _checkpoint_is_patch_guidance(ckpt):
        raise ValueError(
            f"guidance_type=patch_decoder but checkpoint is not patch guidance: {ckpt_path}"
        )
    if USE_ORDINAL_WINDOW_NORM:
        if GLOBAL_ORDINAL_LADDER is None:
            raise ValueError(
                "GLOBAL_ORDINAL_LADDER must be set before loading ordinal patch guidance"
            )
        if not bool(ckpt.get("ordinal_patch_guidance_unit_ranks", False)):
            raise ValueError(
                "Patch guidance checkpoint was not trained with unit-rank ordinal "
                "targets. Delete this patch guidance checkpoint and retrain it."
            )
    stack = load_patch_guidance_from_checkpoint(
        ckpt_path, num_vars, device, ckpt=ckpt,
    )
    return wrap_patch_guidance(stack)


def _window_norm_past_future(
    past: torch.Tensor,
    future: torch.Tensor,
    *,
    apply_ood_shift: bool = False,
    data_is_ranked: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if USE_ORDINAL_WINDOW_NORM:
        if GLOBAL_ORDINAL_LADDER is None:
            raise ValueError("GLOBAL_ORDINAL_LADDER must be set before ordinal encoding")
        if data_is_ranked:
            return past, future
        past_ord, future_ord, _ladder, _ood_shift = ordinal_encode(
            past,
            future,
            ladder=GLOBAL_ORDINAL_LADDER,
            apply_ood_shift=apply_ood_shift,
        )
        return past_ord, future_ord
    if not USE_WINDOW_NORMALIZATION:
        return past, future
    if WINDOW_NORM_CENTER == "last":
        center = past[..., -1:]
    elif WINDOW_NORM_CENTER == "mean":
        center = past.mean(dim=-1, keepdim=True)
    else:
        raise ValueError(f"unknown window_norm_center {WINDOW_NORM_CENTER!r}")
    past_std = past.std(dim=-1, keepdim=True)
    if WINDOW_NORM_LOW_VAR_THRESHOLD > 0.0:
        std_floor = past_std.clamp_min(WINDOW_NORM_STD_FLOOR)
        unit = torch.full_like(past_std, WINDOW_NORM_LOW_VAR_UNIT_STD)
        low_var = past_std < WINDOW_NORM_LOW_VAR_THRESHOLD
        flat = past_std <= WINDOW_NORM_STD_FLOOR
        std = torch.where(flat | low_var, unit, std_floor)
    else:
        std = past_std.clamp_min(WINDOW_NORM_STD_FLOOR)
    return (past - center) / std, (future - center) / std


def _set_ordinal_loader_mode(model, loader, *, eval_mode: bool = False) -> None:
    """Configure per-batch ordinal flags on the diffusion model."""
    if not USE_ORDINAL_WINDOW_NORM:
        return
    ranked = _dataset_yields_ordinal_ranks(loader.dataset)
    model._ordinal_input_is_ranked = ranked
    model._ordinal_apply_ood_shift = bool(eval_mode and not ranked)


def _dataset_yields_ordinal_ranks(dataset) -> bool:
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return bool(getattr(dataset, "yields_ordinal_ranks", False))


def _patch_guidance_batch(
    past: torch.Tensor,
    future: torch.Tensor,
    device: torch.device,
    *,
    apply_ood_shift: bool = False,
    data_is_ranked: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if _itrans_ar_enabled(future.shape[-1]):
        past, future = _sample_itrans_ar_chunk(past, future)
    past = past.to(device)
    future = future.to(device)
    past_norm, future_norm = _window_norm_past_future(
        past,
        future,
        apply_ood_shift=apply_ood_shift,
        data_is_ranked=data_is_ranked,
    )
    avail = future_norm.shape[-1] - LOOKBACK_OVERLAP if LOOKBACK_OVERLAP > 0 else future_norm.shape[-1]
    pred_len = min(_patch_guidance_out_len(), avail)
    if LOOKBACK_OVERLAP > 0:
        target = future_norm[..., LOOKBACK_OVERLAP : LOOKBACK_OVERLAP + pred_len]
    else:
        target = future_norm[..., :pred_len]
    if USE_ORDINAL_WINDOW_NORM:
        if GLOBAL_ORDINAL_LADDER is None:
            raise ValueError("GLOBAL_ORDINAL_LADDER must be set before ordinal patch guidance")
        ladder_past = GLOBAL_ORDINAL_LADDER.expand_batch(past_norm.shape[0])
        ladder_target = GLOBAL_ORDINAL_LADDER.expand_batch(target.shape[0])
        past_norm = ranks_to_unit(past_norm, ladder_past)
        target = ranks_to_unit(target, ladder_target)
    return past_norm, target


def train_patch_guidance_epoch(stack, loader, optimizer, device, scheduler=None):
    stack.train()
    total_loss = 0.0
    n_batches = 0
    data_is_ranked = _dataset_yields_ordinal_ranks(loader.dataset)
    from models.diffusion_tsf.train_window_aug import set_train_window_aug_epoch

    # Epoch counter lives on the dataset; bump once per call (one epoch).
    ds = loader.dataset
    while hasattr(ds, "dataset") and not hasattr(ds, "set_epoch"):
        ds = ds.dataset
    if hasattr(ds, "set_epoch"):
        set_train_window_aug_epoch(loader, int(getattr(ds, "_epoch", 0)) + 1)
    for past, future in loader:
        past_norm, y_true = _patch_guidance_batch(
            past, future, device, data_is_ranked=data_is_ranked,
        )
        optimizer.zero_grad()
        loss = stack.finetune_loss(past_norm, y_true)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def validate_patch_guidance(stack, loader, device):
    stack.eval()
    total_loss = 0.0
    n_batches = 0
    data_is_ranked = _dataset_yields_ordinal_ranks(loader.dataset)
    with torch.no_grad():
        for past, future in loader:
            past_norm, y_true = _patch_guidance_batch(
                past,
                future,
                device,
                apply_ood_shift=USE_ORDINAL_WINDOW_NORM,
                data_is_ranked=data_is_ranked,
            )
            loss = stack.finetune_loss(past_norm, y_true)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def patch_guidance_hp_objective(
    trial,
    train_loader,
    val_loader,
    num_vars: int,
    device,
    smoke_test=False,
    fixed_batch_size: Optional[int] = None,
    max_epochs: int = PATCH_GUIDANCE_HP_FINETUNE_MAX_EPOCHS,
    trial_ckpt_dir: Optional[str] = None,
):
    lr = trial.suggest_categorical("learning_rate", ITRANS_PAPER_LR_GRID)
    batch_size = fixed_batch_size if fixed_batch_size is not None else ITRANS_PAPER_BATCH_SIZE

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    stack = create_patch_guidance_stack(num_vars).to(device)

    train_loader_local = DataLoader(
        train_loader.dataset, batch_size=batch_size, shuffle=True, num_workers=0,
    )
    val_bs = min(batch_size, 32)
    val_loader_local = DataLoader(
        val_loader.dataset, batch_size=val_bs, shuffle=False, num_workers=0,
    )

    optimizer = torch.optim.Adam(stack.parameters(), lr=lr)
    epochs = max_epochs if not smoke_test else 1
    best_val_loss = float("inf")
    trial_ckpt_path = None
    if trial_ckpt_dir is not None:
        os.makedirs(trial_ckpt_dir, exist_ok=True)
        trial_ckpt_path = os.path.join(
            trial_ckpt_dir, f"patch_guidance_hp_trial_{trial.number}.pt",
        )

    try:
        for epoch in range(epochs):
            train_patch_guidance_epoch(stack, train_loader_local, optimizer, device)
            val_loss = validate_patch_guidance(stack, val_loader_local, device)
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                tuned = {"learning_rate": lr, "batch_size": batch_size}
                if trial_ckpt_path is not None:
                    torch.save(
                        {
                            "model_state_dict": stack.state_dict(),
                            "config": stack.config.to_dict(),
                            "best_params": tuned,
                            "val_loss": val_loss,
                            "ordinal_patch_guidance_unit_ranks": bool(USE_ORDINAL_WINDOW_NORM),
                            "patch_guidance_target_space": (
                                "ordinal_unit_rank"
                                if USE_ORDINAL_WINDOW_NORM
                                else "window_normalized"
                            ),
                        },
                        trial_ckpt_path,
                    )
                    trial.set_user_attr("ckpt_path", trial_ckpt_path)
    except torch.OutOfMemoryError:
        logger.warning(
            "[Patch guidance HP] OOM at batch_size=%s; pruning trial %s.",
            batch_size, trial.number,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise optuna.TrialPruned()
    return best_val_loss


def run_patch_guidance_finetune_hp_tuning(
    dataset_name: str,
    variate_indices: List[int],
    n_trials: int,
    device: torch.device,
    smoke_test: bool = False,
    checkpoint_dir: Optional[str] = None,
    subset_id: Optional[str] = None,
    train_stride: Optional[int] = None,
    test_stride: Optional[int] = None,
    parallel_workers: int = 1,
) -> Tuple[Dict, Optional[str]]:
    """HP tune patch decoder + mixer on real data (window-norm MSE)."""
    label = subset_id or dataset_name
    n_vars = len(variate_indices)
    logger.info("=" * 60)
    logger.info(
        "Patch guidance finetune HP: %s (%d trials, %d workers)",
        label, n_trials, parallel_workers,
    )
    logger.info("=" * 60)

    train_ds, val_ds, _, norm_stats = load_dataset(
        dataset_name, variate_indices,
        stride=train_stride or WINDOW_STRIDE,
        test_stride=1 if test_stride is None else test_stride,
    )
    from models.diffusion_tsf.train_window_aug import maybe_wrap_train_window_aug

    aug_cfg = globals().get("TRAIN_WINDOW_AUG") or {}
    if not isinstance(aug_cfg, dict):
        aug_cfg = {}
    # Prefer patched pipeline globals when present (set via training.train_window_aug).
    train_ds = maybe_wrap_train_window_aug(
        train_ds,
        enabled=bool(aug_cfg.get("enabled", False)),
        apply_prob=float(aug_cfg.get("apply_prob", 0.5)),
        seed=int(globals().get("PIPELINE_SEED", 42)),
        ladder=norm_stats.get("ordinal_ladder"),
        acf_threshold=float(aug_cfg.get("acf_threshold", 0.35)),
    )
    if smoke_test:
        train_ds = Subset(train_ds, list(range(min(2, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(2, len(val_ds)))))

    train_bs = ITRANS_PAPER_BATCH_SIZE
    train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=min(train_bs, 32), shuffle=False, num_workers=0)

    trial_dir = checkpoint_dir or CHECKPOINT_DIR
    os.makedirs(trial_dir, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from models.diffusion_tsf.pipeline.optuna_parallel import run_optuna_study

    def objective_builder(_worker_id: int):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def objective(trial):
            return patch_guidance_hp_objective(
                trial, train_loader, val_loader, n_vars, dev, smoke_test,
                fixed_batch_size=train_bs,
                max_epochs=PATCH_GUIDANCE_HP_FINETUNE_MAX_EPOCHS,
                trial_ckpt_dir=trial_dir,
            )

        return objective

    study = run_optuna_study(
        study_name=f"patch-guidance-ft-{label}",
        checkpoint_dir=trial_dir,
        n_trials=n_trials,
        parallel_workers=parallel_workers,
        direction="minimize",
        objective_builder=objective_builder,
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
        show_progress_bar=not smoke_test,
        sampler_seed=42,
    )

    best_params = dict(study.best_params)
    best_params["batch_size"] = train_bs
    logger.info(
        "Best patch guidance FT params for %s: lr=%.2e → val_loss=%.4f",
        label, best_params["learning_rate"], study.best_value,
    )

    ckpt_path = None
    if checkpoint_dir is not None:
        ckpt_path = os.path.join(checkpoint_dir, f"{label}_patch_guidance_hp_best.pt")
        _promote_trial_ckpt(
            study, trial_dir, "patch_guidance_hp_trial_{trial}.pt", ckpt_path,
        )
        logger.info("  Saved best patch guidance FT HP model → %s", ckpt_path)
    return best_params, ckpt_path


def _itrans_chunk_horizon() -> int:
    chunk = int(DIFFUSION_CHUNK_HORIZON or 0)
    if chunk > 0:
        return chunk
    return int(FORECAST_LENGTH)


def _itrans_ar_enabled(future_len: int) -> bool:
    chunk = _itrans_chunk_horizon()
    if chunk <= 0:
        return False
    dataset_h = future_len - int(LOOKBACK_OVERLAP)
    return dataset_h > chunk


def _itrans_ar_num_chunks(dataset_horizon: int) -> int:
    K = int(LOOKBACK_OVERLAP)
    C = _itrans_chunk_horizon()
    if dataset_horizon <= C:
        return 1
    stride = max(1, C - K)
    return int(math.ceil((dataset_horizon - K) / stride))


def _sample_itrans_ar_chunk(
    past: torch.Tensor,
    future: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random AR chunk for iTrans: full-seq past window, 96-step target."""
    K = int(LOOKBACK_OVERLAP)
    C = _itrans_chunk_horizon()
    dataset_h = future.shape[-1] - K
    n_chunks = _itrans_ar_num_chunks(dataset_h)
    if n_chunks <= 1:
        return past, future

    device = past.device
    past_out = []
    future_out = []
    for b in range(past.shape[0]):
        c = int(torch.randint(0, n_chunks, (1,), device=device).item())
        offset = c * max(1, C - K)
        end = min(offset + K + C, future.shape[-1])
        fut_b = future[b : b + 1, ..., offset:end]
        if c == 0:
            past_b = past[b : b + 1]
        else:
            hist = future[b : b + 1, ..., K : K + offset]
            past_b = torch.cat([past[b : b + 1], hist], dim=-1)
        past_out.append(past_b)
        future_out.append(fut_b)
    max_past = max(p.shape[-1] for p in past_out)
    max_fut = max(f.shape[-1] for f in future_out)
    past_pad, future_pad = [], []
    for p, f in zip(past_out, future_out):
        if p.shape[-1] < max_past:
            pad = max_past - p.shape[-1]
            p = torch.cat([p[..., :1].expand(*p.shape[:-1], pad), p], dim=-1)
        if f.shape[-1] < max_fut:
            f = torch.nn.functional.pad(f, (0, max_fut - f.shape[-1]))
        past_pad.append(p)
        future_pad.append(f)
    return torch.cat(past_pad, dim=0), torch.cat(future_pad, dim=0)


def set_realts_training_epoch(loader_or_subset_or_dataset, epoch: int) -> None:
    """If ``loader_or_subset_or_dataset`` wraps a RealTS, set its strided synthetic epoch."""
    ds = loader_or_subset_or_dataset
    if isinstance(ds, DataLoader):
        ds = ds.dataset
    if isinstance(ds, Subset):
        ds = ds.dataset
    if hasattr(ds, 'set_synthetic_epoch'):
        ds.set_synthetic_epoch(epoch)


def get_synth_cache_dir(checkpoint_dir: Optional[str] = None, smoke_test: bool = False) -> Optional[str]:
    """Resolve synthetic cache dir; prefer shared cache when configured."""
    if smoke_test:
        return None
    if SYNTH_CACHE_DIR:
        os.makedirs(SYNTH_CACHE_DIR, exist_ok=True)
        return SYNTH_CACHE_DIR
    path = os.path.join(project_root, 'synth_data')
    os.makedirs(path, exist_ok=True)
    return path


# ============================================================================
# Dimensionality Helpers
# ============================================================================

def get_dataset_n_cols(dataset_name: str) -> int:
    """Return the number of numeric columns in a dataset (excluding date)."""
    path, date_col = _resolve_registry_path(dataset_name)
    if path.endswith('.npz'):
        data = _load_pems_npz(path)
        return int(data.shape[1])
    try:
        df = pd.read_csv(path, nrows=1)
        if date_col and date_col in df.columns:
            return sum(1 for c in df.columns if c != date_col)
        return len(df.columns)
    except Exception:
        return int(_load_solar_lines(path).shape[1])


_DATASET_SHAPE_CACHE: Dict[Tuple[str, str], Tuple[int, int]] = {}


def get_dataset_shape(dataset_name: str) -> Tuple[int, int]:
    """Return raw row/variate counts without materializing the full numeric array."""
    key = (DATASETS_DIR, dataset_name)
    if key in _DATASET_SHAPE_CACHE:
        return _DATASET_SHAPE_CACHE[key]
    path, date_col = _resolve_registry_path(dataset_name)
    data = _load_dataset_array(path, date_col)
    shape = (int(data.shape[0]), int(data.shape[1]))
    _DATASET_SHAPE_CACHE[key] = shape
    return shape


def resolve_pipeline_data_subset(state) -> Dict[str, Any]:
    """Resolve state.data_subset and write concrete variates/strides to state."""
    base_indices = state.variate_indices
    if base_indices is None:
        base_indices = generate_dataset_job(state.dataset)["variate_indices"]
    raw_rows, raw_variates = get_dataset_shape(state.dataset)
    policy = dict(state.data_subset or {})
    target_dataset = policy.get("target_dataset")
    target_rows = target_variates = None
    if target_dataset:
        try:
            target_rows, target_variates = get_dataset_shape(str(target_dataset))
        except Exception as exc:
            raise ValueError(f"Could not resolve data_subset target_dataset={target_dataset!r}: {exc}") from exc
    resolved = resolve_data_subset(
        dataset_name=state.dataset,
        raw_rows=raw_rows,
        raw_variates=raw_variates,
        base_variate_indices=list(base_indices),
        default_subset_id=state.subset_id,
        default_window_stride=state.window_stride,
        seed=state.seed,
        policy=policy,
        target_rows=target_rows,
        target_variates=target_variates,
    )
    state.variate_indices = list(resolved["variate_indices"])
    state.n_variates = int(resolved["n_variates"])
    state.subset_id = str(resolved["subset_id"])
    state.data_subset_resolved = resolved
    print(
        f"[data_subset] {state.dataset}: subset_id={resolved['subset_id']} "
        f"n_variates={resolved['n_variates']} sample_stride={resolved['sample_stride']} "
        f"raw_mb={resolved['raw_size_mb']:.3f} reduced_mb={resolved['reduced_size_mb']:.3f} "
        f"target_mb={resolved.get('target_size_mb')} reason={resolved.get('reason')}"
    )
    return resolved


def get_dim_for_dataset(dataset_name: str) -> int:
    """Return native dataset dimensionality (always full variates)."""
    return get_dataset_n_cols(dataset_name)


def pretrain_dir_for_dim(dim: int, base_dir: str = None) -> str:
    """Checkpoint subdirectory for a specific pretrain dimensionality."""
    base = base_dir or CHECKPOINT_DIR
    return os.path.join(base, f'pretrained_dim{dim}')


# ============================================================================
# iTransformer Model Creation
# ============================================================================

def _purge_repo_utils_for_itransformer() -> None:
    """Repo ``utils/`` (eval scripts) shadows iTransformer's ``utils.masking`` imports."""
    repo_utils = Path(__file__).resolve().parents[2] / "utils"
    drop: List[str] = []
    for name, mod in list(sys.modules.items()):
        if name != "utils" and not name.startswith("utils."):
            continue
        mod_file = getattr(mod, "__file__", None)
        if mod_file is None:
            if name == "utils":
                drop.append(name)
            continue
        try:
            mod_path = Path(mod_file).resolve()
        except OSError:
            continue
        if "iTransformer" in mod_path.parts:
            continue
        if mod_path == repo_utils or repo_utils in mod_path.parents:
            drop.append(name)
    for name in drop:
        sys.modules.pop(name, None)


def get_itransformer_class():
    """Dynamically load iTransformer model class."""
    itrans_path = os.path.join(script_dir, '..', 'iTransformer', 'model', 'iTransformer.py')
    itrans_path = os.path.abspath(itrans_path)
    
    # Add iTransformer directory to path for internal imports
    itrans_dir = os.path.join(script_dir, '..', 'iTransformer')
    itrans_dir = os.path.abspath(itrans_dir)
    _purge_repo_utils_for_itransformer()
    sys.path = [p for p in sys.path if os.path.abspath(p) != itrans_dir]
    sys.path.insert(0, itrans_dir)
    
    spec = importlib.util.spec_from_file_location("iTransformer_module", itrans_path)
    itrans_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(itrans_module)
    return itrans_module.Model


def create_itransformer_config(
    *,
    seq_len: Optional[int] = None,
    pred_len: Optional[int] = None,
    num_vars: Optional[int] = None,
    dropout: Optional[float] = None,
):
    """Create iTransformer config object from YAML-patched module globals."""
    seq_len = ITRANSFORMER_SEQ_LEN if seq_len is None else seq_len
    pred_len = FORECAST_LENGTH if pred_len is None else pred_len
    num_vars = N_VARIATES if num_vars is None else num_vars
    dropout = ITRANS_PAPER_DROPOUT if dropout is None else dropout
    class iTransConfig:
        def __init__(self):
            self.seq_len = seq_len
            self.pred_len = pred_len
            self.output_attention = False
            self.use_norm = True
            self.d_model = ITRANS_D_MODEL
            self.d_ff = ITRANS_D_FF
            self.e_layers = ITRANS_E_LAYERS
            self.n_heads = ITRANS_N_HEADS
            self.dropout = dropout
            self.activation = 'gelu'
            self.embed = 'fixed'
            self.freq = 'h'
            self.factor = 1
            self.enc_in = num_vars
            self.class_strategy = 'projection'
    return iTransConfig()


def create_itransformer(
    *,
    seq_len: Optional[int] = None,
    pred_len: Optional[int] = None,
    num_vars: Optional[int] = None,
    dropout: Optional[float] = None,
) -> nn.Module:
    """Create iTransformer model from YAML-patched module globals."""
    iTransformerModel = get_itransformer_class()
    config = create_itransformer_config(
        seq_len=seq_len,
        pred_len=pred_len,
        num_vars=num_vars,
        dropout=dropout,
    )
    return iTransformerModel(config)


def load_itransformer_from_checkpoint(
    path: str,
    num_vars: int,
    device: torch.device,
) -> nn.Module:
    """Load iTransformer weights, inferring ``seq_len`` from the checkpoint itself.

    The inverted-embedding layer stores a Linear of shape ``[d_model, seq_len]`` at
    key ``enc_embedding.value_embedding.weight``. Reading that shape lets us build
    a model that matches whatever seq_len the checkpoint was saved with — no
    hardcoded fallback list.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt['model_state_dict']
    weight_key = 'enc_embedding.value_embedding.weight'
    if weight_key not in state:
        raise RuntimeError(
            f"iTransformer checkpoint {path} is missing key {weight_key!r}; "
            f"cannot infer seq_len."
        )
    ckpt_seq_len = int(state[weight_key].shape[1])
    proj_key = 'projector.weight'
    if proj_key in state:
        ckpt_pred_len = int(state[proj_key].shape[0])
    else:
        ckpt_pred_len = FORECAST_LENGTH

    model = create_itransformer(
        seq_len=ckpt_seq_len,
        pred_len=ckpt_pred_len,
        num_vars=num_vars,
    ).to(device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"Cannot load iTransformer checkpoint {path} "
            f"(inferred seq_len={ckpt_seq_len}): {e}"
        ) from e
    model.eval()
    if ckpt_seq_len != ITRANSFORMER_SEQ_LEN:
        logger.warning(
            f"iTransformer checkpoint {path} has seq_len={ckpt_seq_len}, "
            f"differs from current ITRANSFORMER_SEQ_LEN={ITRANSFORMER_SEQ_LEN}. "
            f"Loaded with checkpoint's seq_len; ensure callers slice past inputs accordingly."
        )
    return model


def load_diffusion_state_keep_attached_guidance(model: nn.Module, ckpt_state: Dict) -> None:
    """Load a diffusion checkpoint while preserving the guidance submodule that the
    caller already attached on the model.

    The diffusion checkpoint's ``model_state_dict`` includes ``guidance_model.*``
    keys (PyTorch saves all submodules). Reloading those would overwrite the
    freshly-attached guidance — and breaks loudly when the saved guidance has a
    different ``seq_len`` than the attached one (e.g. synthetic-pretrain vs
    real-finetuned iTransformer). We always want to keep the attached guidance
    and only restore the diffusion backbone weights.
    """
    model_state = model.state_dict()
    filtered = {}
    for k, v in ckpt_state.items():
        if k.startswith('guidance_model.'):
            continue
        if k in model_state and model_state[k].shape != v.shape:
            logger.warning(f"Skipping {k} due to shape mismatch: ckpt {v.shape} vs model {model_state[k].shape}")
            continue
        filtered[k] = v
        
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    leaked = [k for k in (missing or []) if not k.startswith('guidance_model.')]
    if leaked:
        logger.warning(f"Diffusion ckpt missing non-guidance keys: {leaked[:5]}...")
    real_unexpected = [k for k in (unexpected or []) if not k.startswith('guidance_model.')]
    if real_unexpected:
        logger.warning(f"Diffusion ckpt has unexpected keys: {real_unexpected[:5]}...")


# ============================================================================
# Diffusion Model Creation (with guidance support)
# ============================================================================

def _resolve_guidance_type(guidance_model, override: Optional[str] = None) -> str:
    """Match DiffusionTSF routing to the attached guidance, not YAML alone."""
    if override is not None:
        return str(override)
    if isinstance(guidance_model, PatchDecoderGuidance):
        return "patch_decoder"
    if guidance_model is not None:
        return "itransformer"
    return GUIDANCE_TYPE


def create_diffusion_model(
    *,
    guidance_model=None,
    n_variates: Optional[int] = None,
    lookback: Optional[int] = None,
    horizon: Optional[int] = None,
    diffusion_stage: Optional[str] = None,
    **overrides: Any,
) -> DiffusionTSF:
    """Build DiffusionTSF from YAML-patched module globals.

    Pass explicit lookback/horizon/n_variates for per-dataset geometry, diffusion_stage
    for staged training, and overrides only for runtime exceptions (HP search, checkpoints).
    """
    def o(key: str, default: Any) -> Any:
        if key not in overrides:
            return default
        val = overrides[key]
        return default if val is None else val

    lb = LOOKBACK_LENGTH if lookback is None else lookback
    hz = FORECAST_LENGTH if horizon is None else horizon
    stage = DIFFUSION_STAGE if diffusion_stage is None else diffusion_stage
    chunk_hz = int(DIFFUSION_CHUNK_HORIZON or 0)
    if chunk_hz > 0 and hz > chunk_hz:
        model_hz = chunk_hz + LOOKBACK_OVERLAP
    else:
        model_hz = hz + LOOKBACK_OVERLAP

    config = DiffusionTSFConfig(
        num_variables=N_VARIATES if n_variates is None else n_variates,
        lookback_length=lb,
        forecast_length=model_hz,
        dataset_forecast_length=hz,
        lookback_overlap=LOOKBACK_OVERLAP,
        diffusion_lookback_cap=int(DIFFUSION_LOOKBACK_CAP or 0),
        diffusion_chunk_horizon=chunk_hz,
        representation_time_stride=int(REPRESENTATION_TIME_STRIDE),
        past_cond_resize_to_horizon=bool(PAST_COND_RESIZE_TO_HORIZON),
        itrans_lookback_length=ITRANS_LOOKBACK_LENGTH,
        past_loss_weight=PAST_LOSS_WEIGHT,
        image_height=IMAGE_HEIGHT,
        coarse_image_height=COARSE_IMAGE_HEIGHT,
        fine_image_height=FINE_IMAGE_HEIGHT,
        finer_image_height=FINER_IMAGE_HEIGHT,
        max_scale=o("max_scale", MAX_SCALE),
        staged_representation=o("staged_representation", STAGED_REPRESENTATION),
        binary_noise_schedule=o("binary_noise_schedule", BINARY_NOISE_SCHEDULE),
        binary_length_mode=o(
            "binary_length_mode",
            globals().get("BINARY_LENGTH_MODE", "none"),
        ),
        binary_length_g=float(
            o("binary_length_g", globals().get("BINARY_LENGTH_G", 1.0))
        ),
        binary_length_scale=float(
            o("binary_length_scale", globals().get("BINARY_LENGTH_SCALE", 1.0))
        ),
        prediction_target=o("prediction_target", PREDICTION_TARGET),
        loss_weighting=o("loss_weighting", LOSS_WEIGHTING),
        min_snr_gamma=o("min_snr_gamma", MIN_SNR_GAMMA),
        use_coordinate_channel=USE_COORDINATE_CHANNEL,
        use_raw_lookback_cond_channel=o(
            "use_raw_lookback_cond_channel", USE_RAW_LOOKBACK_COND_CHANNEL,
        ),
        use_guidance_channel=o("use_guidance_channel", USE_GUIDANCE_CHANNEL),
        guidance_penalty_weight=0.0,
        model_type=o("model_type", MODEL_TYPE),
        disable_cross_attention=DISABLE_CROSS_ATTENTION,
        diffusion_stage=stage,
        use_triple_scale=USE_TRIPLE_SCALE,
        dit_patch_size=DIT_PATCH_SIZE,
        dit_embed_dim=DIT_EMBED_DIM,
        dit_depth=DIT_DEPTH,
        dit_num_heads=DIT_NUM_HEADS,
        dit_mlp_ratio=DIT_MLP_RATIO,
        dit_dropout=o("dit_dropout", DIT_DROPOUT),
        use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        unet_max_chunk_size=UNET_MAX_CHUNK_SIZE,
        use_amp=USE_AMP,
        diffusion_type=o("diffusion_type", DIFFUSION_TYPE),
        use_ordinal_window_norm=o("use_ordinal_window_norm", USE_ORDINAL_WINDOW_NORM),
        ordinal_tie_atol=o("ordinal_tie_atol", ORDINAL_TIE_ATOL),
        ordinal_ladder=o("ordinal_ladder", GLOBAL_ORDINAL_LADDER),
        use_deterministic_anchor_loss=o("use_deterministic_anchor_loss", DETERMINISTIC_ANCHOR_LOSS),
        deterministic_anchor_lambda=o("deterministic_anchor_lambda", DETERMINISTIC_ANCHOR_LAMBDA),
        deterministic_anchor_alpha=o("deterministic_anchor_alpha", DETERMINISTIC_ANCHOR_ALPHA),
        binary_anchor_input_mode=o("binary_anchor_input_mode", BINARY_ANCHOR_INPUT_MODE),
        binary_use_boundary_weighted_bce=o(
            "binary_use_boundary_weighted_bce", BINARY_USE_BOUNDARY_WEIGHTED_BCE,
        ),
        binary_cdf_distance_alpha=float(
            o("binary_cdf_distance_alpha", BINARY_CDF_DISTANCE_ALPHA)
        ),
        anchor_mse_proxy_lambda=float(
            o("anchor_mse_proxy_lambda", ANCHOR_MSE_PROXY_LAMBDA)
        ),
        cross_variate_context_bias=CROSS_VARIATE_CONTEXT_BIAS,
        cfg_dropout=CFG_DROPOUT,
        binary_num_steps=o("binary_num_steps", BINARY_NUM_STEPS),
        binary_beta_start=o("binary_beta_start", BINARY_BETA_START),
        binary_beta_end=o("binary_beta_end", BINARY_BETA_END),
        use_window_normalization=USE_WINDOW_NORMALIZATION,
        window_norm_center=WINDOW_NORM_CENTER,
        window_norm_std_floor=WINDOW_NORM_STD_FLOOR,
        window_norm_low_var_threshold=WINDOW_NORM_LOW_VAR_THRESHOLD,
        window_norm_low_var_unit_std=WINDOW_NORM_LOW_VAR_UNIT_STD,
        window_norm_low_var_unit_std_per_variate=WINDOW_NORM_LOW_VAR_UNIT_STD_PER_VARIATE,
        lookback_overlap_center_shift=LOOKBACK_OVERLAP_CENTER_SHIFT,
        zero_guidance_forecast=ZERO_GUIDANCE_FORECAST,
        itrans_d_model=ITRANS_D_MODEL,
        guidance_type=_resolve_guidance_type(
            guidance_model, o("guidance_type", None),
        ),
        mmpd_patch_size=MMPD_PATCH_SIZE,
    )
    return DiffusionTSF(config, guidance_model=guidance_model)


# ============================================================================
# Dataset Classes
# ============================================================================

class TimeSeriesDataset(Dataset):
    """Dataset for multivariate time series forecasting."""

    def __init__(
        self,
        data: np.ndarray,
        lookback: int,
        horizon: int,
        stride: int,
        lookback_overlap: int,
        *,
        rank_data: Optional[np.ndarray] = None,
    ):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.rank_data = (
            torch.tensor(rank_data, dtype=torch.float32) if rank_data is not None else None
        )
        self.yields_ordinal_ranks = self.rank_data is not None
        self.lookback = lookback
        self.horizon = horizon
        self.stride = stride
        self.lookback_overlap = lookback_overlap
        total_len = lookback + horizon
        self.n_samples = max(0, (len(data) - total_len) // stride + 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        source = self.rank_data if self.rank_data is not None else self.data
        past = source[start:start + self.lookback].T
        target_start = start + self.lookback - self.lookback_overlap
        target_end = start + self.lookback + self.horizon
        future = source[target_start:target_end].T
        return past, future


def _paper_split_borders(dataset_name: str, n: int, seq_len: int) -> Tuple[List[int], List[int]]:
    """Return (border1s, border2s) following the iTransformer / TimesNet protocol.

    Each split's window-construction array runs from ``border1s[i]`` to
    ``border2s[i]``. ``border1s[i] = boundary - seq_len`` for val/test so the
    first val/test lookback reaches back into the previous split (no "dead
    zone" of unevaluated steps right after the train boundary).

    ETTh{1,2} and ETTm{1,2} use fixed month-based boundaries. Every other
    dataset uses the 70/10/20 length-based convention with the same overlap
    trick — this mirrors ``Dataset_Custom`` in the upstream iTransformer repo.
    """
    if dataset_name in ('ETTh1', 'ETTh2'):
        b2 = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    elif dataset_name in ('ETTm1', 'ETTm2'):
        b2 = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    elif dataset_name == 'PeMS':
        n_train = int(n * 0.6)
        n_val = int(n * 0.2)
        n_test = n - n_train - n_val
        b2 = [n_train, n_train + n_val, n_train + n_val + n_test]
    else:
        n_train = int(n * 0.7)
        n_test = int(n * 0.2)
        n_val = n - n_train - n_test
        b2 = [n_train, n_train + n_val, n_train + n_val + n_test]
    b1 = [0, b2[0] - seq_len, b2[1] - seq_len]
    return b1, b2


def load_dataset(
    dataset_name: str,
    variate_indices: List[int] = None,
    lookback: Optional[int] = None,
    horizon: Optional[int] = None,
    stride: Optional[int] = None,
    test_stride: Optional[int] = None,
    lookback_overlap: Optional[int] = None,
    ordinal_tie_atol: float = 1e-6,
    use_ordinal_window_norm: Optional[bool] = None,
) -> Tuple[Dataset, Dataset, Dataset, Dict]:
    """Load dataset and return train/val/test splits matching iTransformer paper.

    Splits follow the upstream iTransformer / TimesNet data loaders: fixed
    month-based boundaries for ETT* and 70/10/20 length-based otherwise, with
    the standard overlap trick (val/test windows can reach back into the
    previous split by ``lookback`` steps). Train/val use ``stride``; test uses
    ``test_stride`` (default: same as ``stride``). Finetune jobs pass
    ``test_stride=1`` so eval keeps the dense paper protocol while train/val
  can use a larger stride to cut redundant overlap.
    """
    if stride is None:
        stride = WINDOW_STRIDE
    if test_stride is None:
        test_stride = stride
    if lookback_overlap is None:
        lookback_overlap = LOOKBACK_OVERLAP
    if lookback is None:
        lookback = LOOKBACK_LENGTH
    if horizon is None:
        horizon = FORECAST_LENGTH
    path, date_col = _resolve_registry_path(dataset_name)
    data = _load_dataset_array(path, date_col)

    if variate_indices is not None:
        data = data[:, variate_indices]

    n = len(data)
    total_window = lookback + horizon
    if n < total_window:
        raise ValueError(
            f"Dataset '{dataset_name}' has {n} rows but needs at least "
            f"{total_window} (lookback={lookback} + horizon={horizon}). "
            f"Skipping this dataset."
        )

    border1s, border2s = _paper_split_borders(dataset_name, n, lookback)
    train_end = border2s[0]

    train_slice = data[:train_end]
    mean = train_slice.mean(axis=0, keepdims=True)
    std = train_slice.std(axis=0, keepdims=True) + 1e-8
    data = (data - mean) / std

    ordinal_ladder = None
    rank_full = None
    use_ord = USE_ORDINAL_WINDOW_NORM if use_ordinal_window_norm is None else use_ordinal_window_norm
    if use_ord:
        ordinal_ladder = build_global_ladder_from_training(
            data[border1s[0]:border2s[0]],
            tie_atol=float(ordinal_tie_atol),
            precompute_ranks_for=data,
        )
        rank_full = ordinal_ladder.precomputed_ranks.numpy()
        global GLOBAL_ORDINAL_LADDER
        GLOBAL_ORDINAL_LADDER = ordinal_ladder

    train_rank = rank_full[border1s[0]:border2s[0]] if rank_full is not None else None
    train_ds = TimeSeriesDataset(
        data[border1s[0]:border2s[0]], lookback, horizon, stride,
        lookback_overlap=lookback_overlap,
        rank_data=train_rank,
    )
    val_ds = TimeSeriesDataset(
        data[border1s[1]:border2s[1]], lookback, horizon, stride,
        lookback_overlap=lookback_overlap,
    )
    test_ds = TimeSeriesDataset(
        data[border1s[2]:border2s[2]], lookback, horizon, test_stride,
        lookback_overlap=lookback_overlap,
    )

    stats: Dict = {'mean': mean, 'std': std}
    if ordinal_ladder is not None:
        stats['ordinal_ladder'] = ordinal_ladder
    return train_ds, val_ds, test_ds, stats


# ============================================================================
# Variate Subset Management
# ============================================================================

def generate_dataset_job(dataset_name: str, n_variates: int = None, seed: int = 42) -> Dict:
    """Return one full-dataset training job (no variate partitioning)."""
    path, date_col = _resolve_registry_path(dataset_name)
    n_cols = get_dataset_n_cols(dataset_name)
    all_cols = _dataset_variate_names(path, date_col, n_cols)
    indices = list(range(len(all_cols)))
    return {'dataset_id': dataset_name, 'variate_indices': indices, 'variate_names': all_cols}


# ============================================================================
# Early Stopping & Checkpointing
# ============================================================================

from models.diffusion_tsf.pipeline.train.checkpointing import (
    EarlyStopping,
    amp_context,
    ensure_checkpoint_dir,
    save_checkpoint,
)
from models.diffusion_tsf.pipeline.train.diffusion_loop import (
    train_diffusion_epoch,
    validate_diffusion_epoch,
)

# ============================================================================
# PHASE 1A: iTransformer HP Tuning
# ============================================================================

def _itrans_targets(future: torch.Tensor, model: nn.Module, device: torch.device) -> torch.Tensor:
    """Align supervised horizon with iTransformer pred_len (AR may use H>pred_len)."""
    y_true = future.permute(0, 2, 1).to(device)
    if LOOKBACK_OVERLAP > 0:
        y_true = y_true[:, LOOKBACK_OVERLAP:, :]
    pred_len = int(getattr(model, "pred_len", 0) or 0)
    if pred_len > 0:
        if y_true.shape[1] < pred_len:
            raise ValueError(
                f"iTrans target length {y_true.shape[1]} < model pred_len {pred_len}"
            )
        y_true = y_true[:, :pred_len, :]
    return y_true


def _itrans_batch(
    past: torch.Tensor,
    future: torch.Tensor,
    model: nn.Module,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare iTrans past/target: full seq_len lookback, 96-step chunk target."""
    if _itrans_ar_enabled(future.shape[-1]):
        past, future = _sample_itrans_ar_chunk(past, future)
    past = past.to(device)
    future = future.to(device)
    x_enc = past.permute(0, 2, 1)
    seq_sl = int(getattr(model, "seq_len", x_enc.shape[1]) or x_enc.shape[1])
    if x_enc.shape[1] > seq_sl:
        x_enc = x_enc[:, -seq_sl:, :]
    y_true = _itrans_targets(future, model, device)
    return x_enc, y_true


def train_itransformer_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    """Train iTransformer for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for past, future in loader:
        x_enc, y_true = _itrans_batch(past, future, model, device)
        
        optimizer.zero_grad()
        y_pred = model(x_enc, None, None, None)
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def validate_itransformer(model, loader, criterion, device):
    """Validate iTransformer."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for past, future in loader:
            x_enc, y_true = _itrans_batch(past, future, model, device)
            y_pred = model(x_enc, None, None, None)
            loss = criterion(y_pred, y_true)
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / max(n_batches, 1)


def itrans_hp_objective(
    trial,
    synthetic_loader,
    val_loader,
    device,
    smoke_test=False,
    fixed_batch_size: Optional[int] = None,
    best_state: Optional[dict] = None,
    pretrained_ckpt: Optional[str] = None,
    max_epochs: int = ITRANS_HP_PRETRAIN_MAX_EPOCHS,
    seq_len: Optional[int] = None,
    pred_len: Optional[int] = None,
    trial_ckpt_dir: Optional[str] = None,
):
    """Optuna objective for iTransformer HP search.

    Paper-faithful setup: only learning rate is searched, over the categorical
    grid {1e-3, 5e-4, 1e-4}. Batch size is fixed at 32, dropout at 0.1, optimizer
    is plain Adam, no gradient clipping, no early stopping, exactly 10 epochs.

    pretrained_ckpt: if provided, warm-starts each trial from those weights
        (used for finetune HP search on real data).
    best_state: shared mutable dict; updated with best cross-trial model state
        whenever a new minimum val loss is achieved.
    """
    lr = trial.suggest_categorical('learning_rate', ITRANS_PAPER_LR_GRID)
    batch_size = fixed_batch_size if fixed_batch_size is not None else ITRANS_PAPER_BATCH_SIZE

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = create_itransformer(seq_len=seq_len, pred_len=pred_len).to(device)
    if pretrained_ckpt is not None and os.path.exists(pretrained_ckpt):
        ckpt = torch.load(pretrained_ckpt, map_location=device, weights_only=False)
        try:
            model.load_state_dict(ckpt['model_state_dict'], strict=True)
        except RuntimeError as e:
            logger.warning(
                "iTransformer warm-start failed (%s); training this trial from scratch.",
                e,
            )

    train_loader = DataLoader(synthetic_loader.dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_bs = min(batch_size, 32)
    val_loader_local = DataLoader(val_loader.dataset, batch_size=val_bs, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    epochs = max_epochs if not smoke_test else 1
    best_val_loss = float('inf')
    trial_ckpt_path = None
    if trial_ckpt_dir is not None:
        os.makedirs(trial_ckpt_dir, exist_ok=True)
        trial_ckpt_path = os.path.join(trial_ckpt_dir, f"itrans_hp_trial_{trial.number}.pt")

    try:
        for epoch in range(epochs):
            set_realts_training_epoch(synthetic_loader, epoch)
            train_itransformer_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = validate_itransformer(model, val_loader_local, criterion, device)

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                tuned = {
                    'learning_rate': lr,
                    'batch_size': batch_size,
                    'dropout': ITRANS_PAPER_DROPOUT,
                }
                if trial_ckpt_path is not None:
                    torch.save(
                        {'model_state_dict': model.state_dict(), 'best_params': tuned, 'val_loss': val_loss},
                        trial_ckpt_path,
                    )
                    trial.set_user_attr('ckpt_path', trial_ckpt_path)
                elif best_state is not None and val_loss < best_state.get('val_loss', float('inf')):
                    best_state['model_state'] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_state['val_loss'] = val_loss
    except torch.OutOfMemoryError:
        logger.warning(f"[iTransformer HP] OOM at batch_size={batch_size}; pruning trial {trial.number}.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise optuna.TrialPruned()

    return best_val_loss


def _promote_trial_ckpt(study, trial_dir: str, trial_filename: str, dest: str) -> None:
    import shutil
    if study.best_trial is None:
        raise RuntimeError("Optuna study has no successful trials")
    src = os.path.join(trial_dir, trial_filename.format(trial=study.best_trial.number))
    if not os.path.exists(src):
        raise RuntimeError(f"Best trial checkpoint missing: {src}")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    shutil.copy2(src, dest)


def run_itransformer_hp_tuning(
    n_trials: int,
    smoke_test: bool = False,
    checkpoint_dir: Optional[str] = None,
    parallel_workers: int = 1,
) -> Tuple[Dict, Optional[str]]:
    """Run Optuna HP search for iTransformer.

    Returns (best_params, path_to_best_model_or_None). The best model state
    across all trials is saved to itrans_hp_best.pt in checkpoint_dir so the
    caller can use it directly without a separate full-pretrain step.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1A: iTransformer HP Tuning")
    logger.info(f"Trials: {n_trials}")
    logger.info(
        f"iTransformer seq_len={ITRANSFORMER_SEQ_LEN} (diffusion lookback={LOOKBACK_LENGTH})"
    )
    logger.info("=" * 60)

    requested_n = SYNTHETIC_SAMPLES_HP_TUNE
    requested_cap = synthetic_epoch_capacity_itrans_hp()
    n_samples, epoch_cap = resolve_synthetic_params(requested_n, requested_cap, smoke_test)

    n_val = 0 if smoke_test else min(n_samples // 10, 1000)
    synth_cache = get_synth_cache_dir(smoke_test=smoke_test)
    synthetic_loader = get_synthetic_dataloader(
        batch_size=64,
        lookback_length=LOOKBACK_LENGTH,
        forecast_length=FORECAST_LENGTH,
        num_variables=N_VARIATES,
        num_samples=n_samples,
        num_workers=0,
        lookback_overlap=LOOKBACK_OVERLAP,
        cache_dir=synth_cache,
        skip_cross_var_aug=(N_VARIATES > 32),
        val_tail_n=n_val,
        synthetic_epoch_capacity=epoch_cap,
    )

    dataset = synthetic_loader.dataset
    if smoke_test:
        n_val = max(1, min(len(dataset) // 4, len(dataset) - 1))
    else:
        n_val = min(len(dataset) // 10, 1000)
    train_subset = Subset(dataset, list(range(len(dataset) - n_val)))
    val_subset   = Subset(dataset, list(range(len(dataset) - n_val, len(dataset))))

    train_bs = ITRANS_PAPER_BATCH_SIZE
    train_loader = DataLoader(train_subset, batch_size=train_bs, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_subset,   batch_size=min(train_bs, 32), shuffle=False, num_workers=0)

    trial_dir = checkpoint_dir or CHECKPOINT_DIR
    os.makedirs(trial_dir, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from models.diffusion_tsf.pipeline.optuna_parallel import run_optuna_study

    def objective_builder(_worker_id: int):
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        def objective(trial):
            return itrans_hp_objective(
                trial, train_loader, val_loader, dev, smoke_test,
                fixed_batch_size=train_bs,
                max_epochs=ITRANS_HP_PRETRAIN_MAX_EPOCHS,
                trial_ckpt_dir=trial_dir,
            )
        return objective

    logger.info(
        "Starting iTransformer HP search: %d trials (%d workers)",
        n_trials, parallel_workers,
    )
    study = run_optuna_study(
        study_name="itrans-hp-pretrain",
        checkpoint_dir=trial_dir,
        n_trials=n_trials,
        parallel_workers=parallel_workers,
        direction="minimize",
        objective_builder=objective_builder,
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
        show_progress_bar=not smoke_test,
        sampler_seed=42,
    )

    best_params = dict(study.best_params)
    best_params['batch_size'] = train_bs
    best_params['dropout'] = ITRANS_PAPER_DROPOUT
    logger.info(
        "Best iTransformer params: lr=%.2e bs=%s dropout=%.3f val=%.4f",
        best_params['learning_rate'], best_params['batch_size'],
        best_params['dropout'], study.best_value,
    )

    ckpt_path = None
    if checkpoint_dir is not None:
        ckpt_path = os.path.join(checkpoint_dir, 'itrans_hp_best.pt')
        _promote_trial_ckpt(study, trial_dir, "itrans_hp_trial_{trial}.pt", ckpt_path)
        logger.info("  Saved best iTrans HP model → %s", ckpt_path)

    return best_params, ckpt_path


def run_patch_guidance_synthetic_tuning(
    n_trials: int,
    smoke_test: bool = False,
    checkpoint_dir: Optional[str] = None,
    parallel_workers: int = 1,
) -> Tuple[Dict, Optional[str]]:
    """Optuna HP search for patch guidance on synthetic data (staged pretrain fallback)."""
    logger.info("=" * 60)
    logger.info("Patch guidance synthetic HP tuning")
    logger.info("Trials: %s", n_trials)
    logger.info("=" * 60)

    requested_n = SYNTHETIC_SAMPLES_HP_TUNE
    requested_cap = synthetic_epoch_capacity_itrans_hp()
    n_samples, epoch_cap = resolve_synthetic_params(requested_n, requested_cap, smoke_test)

    n_val = 0 if smoke_test else min(n_samples // 10, 1000)
    synth_cache = get_synth_cache_dir(smoke_test=smoke_test)
    synthetic_loader = get_synthetic_dataloader(
        batch_size=64,
        lookback_length=LOOKBACK_LENGTH,
        forecast_length=FORECAST_LENGTH,
        num_variables=N_VARIATES,
        num_samples=n_samples,
        num_workers=0,
        lookback_overlap=LOOKBACK_OVERLAP,
        cache_dir=synth_cache,
        skip_cross_var_aug=(N_VARIATES > 32),
        val_tail_n=n_val,
        synthetic_epoch_capacity=epoch_cap,
    )

    dataset = synthetic_loader.dataset
    if smoke_test:
        n_val = max(1, min(len(dataset) // 4, len(dataset) - 1))
    else:
        n_val = min(len(dataset) // 10, 1000)
    train_subset = Subset(dataset, list(range(len(dataset) - n_val)))
    val_subset = Subset(dataset, list(range(len(dataset) - n_val, len(dataset))))

    train_bs = ITRANS_PAPER_BATCH_SIZE
    train_loader = DataLoader(train_subset, batch_size=train_bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=min(train_bs, 32), shuffle=False, num_workers=0)

    trial_dir = checkpoint_dir or CHECKPOINT_DIR
    os.makedirs(trial_dir, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from models.diffusion_tsf.pipeline.optuna_parallel import run_optuna_study

    def objective_builder(_worker_id: int):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def objective(trial):
            return patch_guidance_hp_objective(
                trial, train_loader, val_loader, N_VARIATES, dev, smoke_test,
                fixed_batch_size=train_bs,
                max_epochs=ITRANS_HP_PRETRAIN_MAX_EPOCHS,
                trial_ckpt_dir=trial_dir,
            )

        return objective

    study = run_optuna_study(
        study_name="patch-guidance-synthetic-pretrain",
        checkpoint_dir=trial_dir,
        n_trials=n_trials,
        parallel_workers=parallel_workers,
        direction="minimize",
        objective_builder=objective_builder,
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
        show_progress_bar=not smoke_test,
        sampler_seed=42,
    )

    best_params = dict(study.best_params)
    best_params["batch_size"] = train_bs
    logger.info(
        "Best patch guidance synthetic params: lr=%.2e bs=%s val=%.4f",
        best_params["learning_rate"], best_params["batch_size"], study.best_value,
    )

    ckpt_path = None
    if checkpoint_dir is not None:
        ckpt_path = os.path.join(checkpoint_dir, "patch_guidance_synthetic_hp_best.pt")
        _promote_trial_ckpt(
            study, trial_dir, "patch_guidance_hp_trial_{trial}.pt", ckpt_path,
        )
        logger.info("  Saved best patch guidance synthetic HP model → %s", ckpt_path)

    return best_params, ckpt_path


def run_itransformer_finetune_hp_tuning(
    dataset_name: str,
    variate_indices: List[int],
    pretrained_ckpt: str,
    n_trials: int,
    device: torch.device,
    smoke_test: bool = False,
    checkpoint_dir: Optional[str] = None,
    subset_id: Optional[str] = None,
    train_stride: Optional[int] = None,
    test_stride: Optional[int] = None,
    parallel_workers: int = 1,
) -> Tuple[Dict, Optional[str]]:
    """HP tune iTransformer on real data using Optuna parallel workers."""
    label = subset_id or dataset_name
    warm = (None if ITRANS_REAL_COLD_START else pretrained_ckpt)
    logger.info("=" * 60)
    logger.info(f"iTrans Finetune HP Tuning: {label} ({n_trials} trials, {parallel_workers} workers)")
    logger.info(
        f"{ITRANS_HP_FINETUNE_MAX_EPOCHS} epochs per trial, "
        f"warm_start={'no (cold start)' if warm is None else os.path.basename(warm)}"
    )
    logger.info("=" * 60)

    train_ds, val_ds, _, _ = load_dataset(
        dataset_name, variate_indices,
        stride=train_stride or WINDOW_STRIDE,
        test_stride=1 if test_stride is None else test_stride,
    )
    if smoke_test:
        train_ds = Subset(train_ds, list(range(min(2, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(2, len(val_ds)))))

    train_bs = ITRANS_PAPER_BATCH_SIZE
    train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=min(train_bs, 32), shuffle=False, num_workers=0)

    trial_dir = checkpoint_dir or CHECKPOINT_DIR
    os.makedirs(trial_dir, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    ds_lb, ds_hz = dataset_window_lengths(dataset_name)
    itrans_seq, itrans_pred = itrans_model_lengths(ds_lb, ds_hz)

    from models.diffusion_tsf.pipeline.optuna_parallel import run_optuna_study

    def objective_builder(_worker_id: int):
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        def objective(trial):
            return itrans_hp_objective(
                trial, train_loader, val_loader, dev, smoke_test,
                fixed_batch_size=train_bs,
                pretrained_ckpt=warm,
                max_epochs=ITRANS_HP_FINETUNE_MAX_EPOCHS,
                trial_ckpt_dir=trial_dir,
                seq_len=itrans_seq,
                pred_len=itrans_pred,
            )
        return objective

    study = run_optuna_study(
        study_name=f"itrans-ft-{label}",
        checkpoint_dir=trial_dir,
        n_trials=n_trials,
        parallel_workers=parallel_workers,
        direction="minimize",
        objective_builder=objective_builder,
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
        show_progress_bar=not smoke_test,
        sampler_seed=42,
    )

    best_params = dict(study.best_params)
    best_params['batch_size'] = train_bs
    best_params['dropout'] = ITRANS_PAPER_DROPOUT
    best_params['lookback_length'] = itrans_seq
    best_params['forecast_length'] = itrans_pred

    logger.info(
        "Best iTrans FT params for %s: lr=%.2e dropout=%.3f → val_loss=%.4f",
        label, best_params['learning_rate'], best_params['dropout'], study.best_value
    )

    ckpt_path = None
    if checkpoint_dir is not None:
        ckpt_path = os.path.join(checkpoint_dir, f'{label}_itrans_ft_hp_best.pt')
        _promote_trial_ckpt(study, trial_dir, "itrans_hp_trial_{trial}.pt", ckpt_path)
        logger.info("  Saved best iTrans FT HP model → %s", ckpt_path)

    return best_params, ckpt_path


if __name__ == "__main__":
    # Import here: cli imports this module, so a module-level import cycles.
    from models.diffusion_tsf.pipeline.train.cli import main

    main()
