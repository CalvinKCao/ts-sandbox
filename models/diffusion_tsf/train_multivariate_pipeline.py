
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
import os
import sys
import time
from datetime import datetime
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
from models.diffusion_tsf.guidance import iTransformerGuidance
from models.diffusion_tsf.storage_paths import resolve_checkpoint_dir, resolve_results_dir
from models.diffusion_tsf.dalia_data import (
    DALIA_CHANNEL_NAMES,
    DALIA_DEFAULT_FORECAST,
    DALIA_DEFAULT_LOOKBACK,
    DALIA_N_VARS,
    dalia_window_count,
    dalia_window_lengths,
    ensure_dalia_csv,
    load_dalia_dataset,
)
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
        'window_norm_std_floor': WINDOW_NORM_STD_FLOOR,
        'use_dual_scale': USE_DUAL_SCALE,
        'use_triple_scale': USE_TRIPLE_SCALE,
        'diffusion_stage': DIFFUSION_STAGE,
        'dual_scale_fine_weight': DUAL_SCALE_FINE_WEIGHT,
        'dual_scale_independent_timesteps': DUAL_SCALE_INDEPENDENT_TIMESTEPS,
        'use_guidance_channel': USE_GUIDANCE_CHANNEL,
        'cfg_dropout': CFG_DROPOUT,
        'disable_cross_attention': DISABLE_CROSS_ATTENTION,
        'cross_variate_context_bias': CROSS_VARIATE_CONTEXT_BIAS,
        'model_type': MODEL_TYPE,
        'diffusion_type': DIFFUSION_TYPE,
        'd3pm_transition_max': D3PM_TRANSITION_MAX,
        'd3pm_transition_min': D3PM_TRANSITION_MIN,
        'd3pm_neighbor_kernel': D3PM_NEIGHBOR_KERNEL,
        'd3pm_noise_schedule': D3PM_NOISE_SCHEDULE,
        'd3pm_loss_type': D3PM_LOSS_TYPE,
        'binary_anchor_input_mode': BINARY_ANCHOR_INPUT_MODE,
        'dit_patch_size': DIT_PATCH_SIZE,
        'dit_embed_dim': DIT_EMBED_DIM,
        'dit_depth': DIT_DEPTH,
        'dit_num_heads': DIT_NUM_HEADS,
        'dit_mlp_ratio': DIT_MLP_RATIO,
        'dit_dropout': DIT_DROPOUT,
        'use_window_normalization': USE_WINDOW_NORMALIZATION,
        'zero_guidance_forecast': ZERO_GUIDANCE_FORECAST,
        'window_stride': WINDOW_STRIDE,
        'binary_noise_schedule': BINARY_NOISE_SCHEDULE,
        'prediction_target': PREDICTION_TARGET,
        'loss_weighting': LOSS_WEIGHTING,
        'min_snr_gamma': MIN_SNR_GAMMA,
        'use_coordinate_channel': USE_COORDINATE_CHANNEL,
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
ITRANS_LOOKBACK_LENGTH = None
IMAGE_HEIGHT = 16
COARSE_IMAGE_HEIGHT = 16
FINE_IMAGE_HEIGHT = 16
FINER_IMAGE_HEIGHT = 16
MAX_SCALE = 3.5
WINDOW_NORM_STD_FLOOR = 1e-8
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
BINARY_NOISE_SCHEDULE = "sqrt_linear"
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
DIFFUSION_BATCH_SIZE = 32
DIFFUSION_BATCH_SIZES = [16]
FINETUNE_BATCH_SIZES = [4, 8, 16]
DIFFUSION_PROBE_TARGET_EFFECTIVE_BATCH = 512
DIFFUSION_PROBE_MAX_BATCH_CAP = 128
DIFFUSION_PROBE_MIN_BATCH = 1
FINETUNE_HP_LR_MIN = 3e-6
FINETUNE_HP_LR_MAX = 2e-4
USE_AMP = True
USE_GRADIENT_CHECKPOINTING = True
UNET_MAX_CHUNK_SIZE = 128
DISABLE_CROSS_ATTENTION = False
USE_DUAL_SCALE = False
USE_TRIPLE_SCALE = False
DIFFUSION_STAGE = "joint"
DUAL_SCALE_FINE_WEIGHT = 0.5
DUAL_SCALE_INDEPENDENT_TIMESTEPS = True
USE_GUIDANCE_CHANNEL = True
CFG_DROPOUT = 0.1
MODEL_TYPE = "dit"
DIFFUSION_TYPE = "binary"
D3PM_TRANSITION_MAX = 0.3
D3PM_TRANSITION_MIN = 1e-5
D3PM_NEIGHBOR_KERNEL = "gaussian"
D3PM_NOISE_SCHEDULE = "sqrt_linear"
D3PM_LOSS_TYPE = "cross_entropy"
DIT_PATCH_SIZE = (8, 8)
DIT_EMBED_DIM = 384
DIT_DEPTH = 8
DIT_NUM_HEADS = 6
DIT_MLP_RATIO = 4.0
DIT_DROPOUT = 0.0
CROSS_VARIATE_CONTEXT_BIAS = 0.0
GUIDANCE_PENALTY_WEIGHT = 0.0
EMD_LAMBDA = 0.2
DETERMINISTIC_ANCHOR_LOSS = False
DETERMINISTIC_ANCHOR_LAMBDA = 0.99
DETERMINISTIC_ANCHOR_ALPHA = 0.5
BINARY_ANCHOR_INPUT_MODE = "stationary_flat"
USE_WINDOW_NORMALIZATION = True
ZERO_GUIDANCE_FORECAST = False
WINDOW_STRIDE = 1
ANCHOR_HP_LAMBDA_MIN = 0.90
ANCHOR_HP_LAMBDA_MAX = 0.995
ANCHOR_HP_ALPHA_MIN = 0.35
ANCHOR_HP_ALPHA_MAX = 0.65
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


def diffusion_probe_max_candidate(n_variates: int, smoke_test: bool) -> int:
    return _training_helpers.diffusion_probe_max_candidate(
        n_variates,
        smoke_test,
        target_effective_batch=DIFFUSION_PROBE_TARGET_EFFECTIVE_BATCH,
        max_batch_cap=DIFFUSION_PROBE_MAX_BATCH_CAP,
        min_batch=DIFFUSION_PROBE_MIN_BATCH,
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
    'dalia': ('dalia/dalia.csv', 'window_id', 96),
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
    if dataset_name == 'dalia':
        return dalia_window_lengths()
    return LOOKBACK_LENGTH, FORECAST_LENGTH


def itrans_model_lengths(dataset_lookback: int, dataset_horizon: int) -> Tuple[int, int]:
    """iTrans seq_len / pred_len decoupled from diffusion AR chunk canvas."""
    seq_len = int(ITRANSFORMER_SEQ_LEN) if ITRANSFORMER_SEQ_LEN else dataset_lookback
    chunk_hz = int(DIFFUSION_CHUNK_HORIZON or 0)
    pred_len = min(dataset_horizon, chunk_hz) if chunk_hz > 0 else dataset_horizon
    return seq_len, pred_len


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
    if dataset_name == 'dalia':
        ensure_dalia_csv(DATASETS_DIR)
        return DALIA_N_VARS
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
    if dataset_name == 'dalia':
        ensure_dalia_csv(DATASETS_DIR)
        n_win = dalia_window_count(DATASETS_DIR)
        shape = (n_win, DALIA_N_VARS)
    else:
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
        itrans_lookback_length=ITRANS_LOOKBACK_LENGTH,
        past_loss_weight=PAST_LOSS_WEIGHT,
        image_height=IMAGE_HEIGHT,
        coarse_image_height=COARSE_IMAGE_HEIGHT,
        fine_image_height=FINE_IMAGE_HEIGHT,
        finer_image_height=FINER_IMAGE_HEIGHT,
        max_scale=o("max_scale", MAX_SCALE),
        binary_noise_schedule=o("binary_noise_schedule", BINARY_NOISE_SCHEDULE),
        prediction_target=o(
            "prediction_target",
            "x0" if DIFFUSION_TYPE == "ordinal_d3pm" else PREDICTION_TARGET,
        ),
        loss_weighting=o(
            "loss_weighting",
            "none" if DIFFUSION_TYPE == "ordinal_d3pm" else LOSS_WEIGHTING,
        ),
        min_snr_gamma=o("min_snr_gamma", MIN_SNR_GAMMA),
        use_coordinate_channel=USE_COORDINATE_CHANNEL,
        use_guidance_channel=o("use_guidance_channel", USE_GUIDANCE_CHANNEL),
        guidance_penalty_weight=GUIDANCE_PENALTY_WEIGHT,
        model_type=o("model_type", MODEL_TYPE),
        disable_cross_attention=DISABLE_CROSS_ATTENTION,
        diffusion_stage=stage,
        use_dual_scale=USE_DUAL_SCALE,
        use_triple_scale=USE_TRIPLE_SCALE,
        dual_scale_fine_weight=DUAL_SCALE_FINE_WEIGHT,
        dual_scale_independent_timesteps=DUAL_SCALE_INDEPENDENT_TIMESTEPS,
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
        d3pm_transition_max=o("d3pm_transition_max", D3PM_TRANSITION_MAX),
        d3pm_transition_min=o("d3pm_transition_min", D3PM_TRANSITION_MIN),
        d3pm_neighbor_kernel=o("d3pm_neighbor_kernel", D3PM_NEIGHBOR_KERNEL),
        d3pm_noise_schedule=o("d3pm_noise_schedule", D3PM_NOISE_SCHEDULE),
        d3pm_loss_type=o("d3pm_loss_type", D3PM_LOSS_TYPE),
        use_deterministic_anchor_loss=o("use_deterministic_anchor_loss", DETERMINISTIC_ANCHOR_LOSS),
        deterministic_anchor_lambda=o("deterministic_anchor_lambda", DETERMINISTIC_ANCHOR_LAMBDA),
        deterministic_anchor_alpha=o("deterministic_anchor_alpha", DETERMINISTIC_ANCHOR_ALPHA),
        binary_anchor_input_mode=o("binary_anchor_input_mode", BINARY_ANCHOR_INPUT_MODE),
        cross_variate_context_bias=CROSS_VARIATE_CONTEXT_BIAS,
        cfg_dropout=CFG_DROPOUT,
        binary_num_steps=o("binary_num_steps", BINARY_NUM_STEPS),
        binary_beta_start=o("binary_beta_start", BINARY_BETA_START),
        binary_beta_end=o("binary_beta_end", BINARY_BETA_END),
        use_window_normalization=USE_WINDOW_NORMALIZATION,
        window_norm_std_floor=WINDOW_NORM_STD_FLOOR,
        zero_guidance_forecast=ZERO_GUIDANCE_FORECAST,
        itrans_d_model=ITRANS_D_MODEL,
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
    ):
        self.data = torch.tensor(data, dtype=torch.float32)
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
        past = self.data[start:start + self.lookback].T
        # Target includes last K observed steps + H forecast steps
        target_start = start + self.lookback - self.lookback_overlap
        target_end = start + self.lookback + self.horizon
        future = self.data[target_start:target_end].T
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
        lookback = DALIA_DEFAULT_LOOKBACK if dataset_name == 'dalia' else LOOKBACK_LENGTH
    if horizon is None:
        horizon = DALIA_DEFAULT_FORECAST if dataset_name == 'dalia' else FORECAST_LENGTH
    if dataset_name == 'dalia':
        return load_dalia_dataset(
            variate_indices=variate_indices,
            lookback=lookback,
            horizon=horizon,
            stride=stride,
            test_stride=test_stride,
            lookback_overlap=lookback_overlap,
            datasets_dir=DATASETS_DIR,
        )
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

    train_ds = TimeSeriesDataset(
        data[border1s[0]:border2s[0]], lookback, horizon, stride,
        lookback_overlap=lookback_overlap,
    )
    val_ds = TimeSeriesDataset(
        data[border1s[1]:border2s[1]], lookback, horizon, stride,
        lookback_overlap=lookback_overlap,
    )
    test_ds = TimeSeriesDataset(
        data[border1s[2]:border2s[2]], lookback, horizon, test_stride,
        lookback_overlap=lookback_overlap,
    )

    return train_ds, val_ds, test_ds, {'mean': mean, 'std': std}


# ============================================================================
# Variate Subset Management
# ============================================================================

def generate_dataset_job(dataset_name: str, n_variates: int = None, seed: int = 42) -> Dict:
    """Return one full-dataset training job (no variate partitioning)."""
    if dataset_name == 'dalia':
        indices = list(range(DALIA_N_VARS))
        return {
            'dataset_id': dataset_name,
            'variate_indices': indices,
            'variate_names': DALIA_CHANNEL_NAMES,
        }
    path, date_col = _resolve_registry_path(dataset_name)
    n_cols = get_dataset_n_cols(dataset_name)
    all_cols = _dataset_variate_names(path, date_col, n_cols)
    indices = list(range(len(all_cols)))
    return {'dataset_id': dataset_name, 'variate_indices': indices, 'variate_names': all_cols}


# ============================================================================
# Early Stopping & Checkpointing
# ============================================================================

class EarlyStopping:
    def __init__(self, patience: int = 25, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def amp_context():
    """Return the appropriate autocast context for mixed precision."""
    if USE_AMP and torch.cuda.is_available():
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    from contextlib import nullcontext
    return nullcontext()


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, config, path, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


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


def train_itransformer_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    """Train iTransformer for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for past, future in loader:
        x_enc = past.permute(0, 2, 1).to(device)
        seq_sl = getattr(model, 'seq_len', x_enc.shape[1])
        if x_enc.shape[1] > seq_sl:
            x_enc = x_enc[:, -seq_sl:, :]
        y_true = _itrans_targets(future, model, device)
        
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
            x_enc = past.permute(0, 2, 1).to(device)
            seq_sl = getattr(model, 'seq_len', x_enc.shape[1])
            if x_enc.shape[1] > seq_sl:
                x_enc = x_enc[:, -seq_sl:, :]
            y_true = _itrans_targets(future, model, device)
            y_pred = model(x_enc, None, None, None)
            loss = criterion(y_pred, y_true)
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / max(n_batches, 1)


def _even_batch_size(n: int, *, floor: int = 1) -> int:
    n = max(floor, int(n))
    if n > 1 and n % 2:
        n -= 1
    return n


def _probe_step_ok(try_step_fn, batch_size: int) -> bool:
    try:
        return bool(try_step_fn(batch_size))
    except torch.OutOfMemoryError:
        return False
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            return False
        raise


def auto_select_max_even_batch_size(
    phase_name: str,
    max_candidate: int,
    try_step_fn,
    min_candidate: int = 2,
) -> int:
    """Pick the largest even batch size that passes ``try_step_fn`` without OOM."""
    min_bs = max(1, min_candidate)
    lo = min_bs
    hi = probe_max = _even_batch_size(max_candidate, floor=min_bs)
    best = min_bs

    while lo <= hi:
        mid = _even_batch_size((lo + hi) // 2, floor=min_bs)
        if _probe_step_ok(try_step_fn, mid):
            best = mid
            lo = mid + (1 if mid == 1 else 2)
        else:
            hi = mid - (1 if mid == 1 else 2)

    safe = _even_batch_size(int(best * 0.8), floor=min_bs)
    logger.info(
        "[AutoBS] %s: selected batch_size=%s (probe_max=%s, tested_max=%s)",
        phase_name, safe, best, probe_max,
    )
    return safe


def select_diffusion_batch_size(
    phase_name: str,
    dataset,
    device: torch.device,
    itrans_guidance: iTransformerGuidance,
    max_candidate: int,
    smoke_test: bool = False,
) -> int:
    """Probe diffusion memory with one train step and pick largest safe even batch."""
    if smoke_test:
        return min(4, _even_batch_size(max_candidate, floor=DIFFUSION_PROBE_MIN_BATCH))

    sample_past, sample_future = dataset[0]

    def _try(batch_size: int) -> bool:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = create_diffusion_model(guidance_model=itrans_guidance).to(device)
        try:
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            past = sample_past.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
            future = sample_future.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context():
                loss = model.get_loss(past, future)
            loss.backward()
            optimizer.step()
            return True
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    return auto_select_max_even_batch_size(
        phase_name, max_candidate, _try, min_candidate=DIFFUSION_PROBE_MIN_BATCH,
    )


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



# ============================================================================
# PHASE 1B: Diffusion HP Tuning (with iTransformer guidance)
# ============================================================================

def diffusion_hp_objective(
    trial,
    synthetic_loader,
    val_loader,
    itrans_guidance: iTransformerGuidance,
    device,
    smoke_test=False,
    fixed_batch_size: Optional[int] = None,
    best_state: Optional[dict] = None,
    disable_anchor_loss: bool = False,
    trial_ckpt_dir: Optional[str] = None,
):
    """Optuna objective for Diffusion HP search.

    disable_anchor_loss: skip the anchor forward pass during HP search to
        halve per-step cost. The anchor regularizer doesn't help rank LR
        candidates on synthetic data.
    """
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    if fixed_batch_size is None:
        batch_size = min(4, DIFFUSION_BATCH_SIZE) if smoke_test else DIFFUSION_BATCH_SIZE
    else:
        batch_size = fixed_batch_size

    diff_kw: Dict[str, Any] = {}
    if disable_anchor_loss:
        diff_kw["use_deterministic_anchor_loss"] = False
    model = create_diffusion_model(guidance_model=itrans_guidance, **diff_kw).to(device)

    train_loader = DataLoader(synthetic_loader.dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    epochs = PRETRAIN_DIFFUSION_MAX_EPOCHS if not smoke_test else 1
    patience = DIFFUSION_HP_PATIENCE if not smoke_test else 1
    early_stop = EarlyStopping(patience=patience)
    best_val_loss = float('inf')
    trial_ckpt_path = None
    if trial_ckpt_dir is not None:
        os.makedirs(trial_ckpt_dir, exist_ok=True)
        trial_ckpt_path = os.path.join(trial_ckpt_dir, f"diff_hp_trial_{trial.number}.pt")

    for epoch in range(epochs):
        set_realts_training_epoch(synthetic_loader, epoch)
        model.train()
        for past, future in train_loader:
            past, future = past.to(device), future.to(device)
            optimizer.zero_grad()
            with amp_context():
                loss = model.get_loss(past, future)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for past, future in val_loader:
                past, future = past.to(device), future.to(device)
                with amp_context():
                    loss = model.get_loss(past, future)
                val_loss += loss.item()
                n_batches += 1
        val_loss /= max(n_batches, 1)

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            tuned = {'learning_rate': lr, 'batch_size': batch_size}
            if trial_ckpt_path is not None:
                torch.save(
                    {'model_state_dict': model.state_dict(), 'best_params': tuned, 'val_loss': val_loss},
                    trial_ckpt_path,
                )
                trial.set_user_attr('ckpt_path', trial_ckpt_path)
            elif best_state is not None and val_loss < best_state.get('val_loss', float('inf')):
                best_state['model_state'] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_state['val_loss'] = val_loss

        if early_stop(val_loss):
            break

    return best_val_loss


def run_diffusion_hp_tuning(
    itrans_checkpoint: str,
    n_trials: int,
    smoke_test: bool = False,
    checkpoint_dir: Optional[str] = None,
    parallel_workers: int = 1,
) -> Tuple[Dict, Optional[str]]:
    """Run Optuna HP search for Diffusion model."""
    logger.info("=" * 60)
    logger.info("PHASE 1B: Diffusion HP Tuning (with iTransformer guidance)")
    logger.info(f"Trials: {n_trials}")
    logger.info("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load iTransformer as guidance (supports legacy 512-wide and current shorter seq checkpoints)
    itrans_model = load_itransformer_from_checkpoint(itrans_checkpoint, N_VARIATES, device)
    itrans_guidance = iTransformerGuidance(itrans_model)
    
    # Create small synthetic dataset for fast iteration
    requested_n = SYNTHETIC_SAMPLES_DIFF_TUNE
    requested_cap = synthetic_epoch_capacity_diff_hp()
    n_samples, epoch_cap = resolve_synthetic_params(requested_n, requested_cap, smoke_test)

    n_val = 0 if smoke_test else min(n_samples // 10, 500)
    synth_cache = get_synth_cache_dir(smoke_test=smoke_test)
    synthetic_loader = get_synthetic_dataloader(
        batch_size=32,
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
        n_val = min(len(dataset) // 10, 500)
    train_subset = Subset(dataset, list(range(len(dataset) - n_val)))
    val_subset = Subset(dataset, list(range(len(dataset) - n_val, len(dataset))))
    
    train_bs = min(4, DIFFUSION_BATCH_SIZE) if smoke_test else DIFFUSION_BATCH_SIZE
    train_loader = DataLoader(train_subset, batch_size=train_bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=min(train_bs, 16), shuffle=False, num_workers=0)
    
    trial_dir = checkpoint_dir or CHECKPOINT_DIR
    os.makedirs(trial_dir, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    skip_anchor = DETERMINISTIC_ANCHOR_LOSS
    if skip_anchor:
        logger.info("Phase 1B: anchor loss disabled for HP search (2× speedup)")

    from models.diffusion_tsf.pipeline.optuna_parallel import run_optuna_study

    def objective_builder(_worker_id: int):
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        def objective(trial):
            return diffusion_hp_objective(
                trial, train_loader, val_loader, itrans_guidance, dev, smoke_test,
                fixed_batch_size=train_bs,
                disable_anchor_loss=skip_anchor,
                trial_ckpt_dir=trial_dir,
            )
        return objective

    logger.info(
        "Starting Diffusion HP search: %d trials (%d workers)",
        n_trials, parallel_workers,
    )
    study = run_optuna_study(
        study_name="diffusion-hp-pretrain",
        checkpoint_dir=trial_dir,
        n_trials=n_trials,
        parallel_workers=parallel_workers,
        direction="minimize",
        objective_builder=objective_builder,
        sampler=TPESampler(seed=42),
        show_progress_bar=not smoke_test,
        sampler_seed=42,
    )

    best_params = dict(study.best_params)
    best_params['batch_size'] = train_bs
    logger.info(
        "Best Diffusion params: lr=%.2e bs=%s val=%.4f",
        best_params['learning_rate'], best_params['batch_size'], study.best_value,
    )

    ckpt_path = None
    if checkpoint_dir is not None:
        ckpt_path = os.path.join(checkpoint_dir, 'diff_hp_best.pt')
        _promote_trial_ckpt(study, trial_dir, "diff_hp_trial_{trial}.pt", ckpt_path)
        logger.info("  Saved best diffusion HP model → %s", ckpt_path)

    return best_params, ckpt_path


# ============================================================================
# Staged synthetic diffusion pretrain (coarse/fine checkpoints)
# ============================================================================

def pretrain_diffusion(
    best_params: Dict,
    itrans_checkpoint: str,
    n_samples: int,
    epochs: int,
    patience: int,
    checkpoint_dir: str,
    smoke_test: bool = False,
) -> str:
    """Train one staged diffusion checkpoint on synthetic data (not post-HP retrain)."""
    logger.info("=" * 60)
    logger.info("Staged synthetic diffusion pretrain (with iTransformer guidance)")
    logger.info(f"Samples: {n_samples}, Epochs: {epochs}, Patience: {patience}")
    logger.info(f"Params: {best_params}")
    logger.info("=" * 60)
    
    device = get_device()
    
    lr = require_tuned_param(best_params, 'learning_rate', 'Diffusion pretraining')
    tuned_batch_size = require_tuned_param(best_params, 'batch_size', 'Diffusion pretraining')
    batch_size = tuned_batch_size
    
    # Load iTransformer as guidance (not wrapped in DDP - used in eval mode only)
    itrans_model = load_itransformer_from_checkpoint(itrans_checkpoint, N_VARIATES, device)
    itrans_guidance = iTransformerGuidance(itrans_model)
    
    # Create data
    synth_cache = get_synth_cache_dir(checkpoint_dir=checkpoint_dir, smoke_test=smoke_test)
    n_val = 0 if smoke_test else min(n_samples // 10, 5000)
    epoch_cap = 1 if smoke_test else synthetic_epoch_capacity_pretrain_diffusion()
    synthetic_loader = get_synthetic_dataloader(
        batch_size=min(16, max(2, tuned_batch_size)),
        lookback_length=LOOKBACK_LENGTH,
        forecast_length=FORECAST_LENGTH,
        num_variables=N_VARIATES,
        num_samples=n_samples,
        num_workers=0 if smoke_test else 4,
        lookback_overlap=LOOKBACK_OVERLAP,
        cache_dir=synth_cache,
        skip_cross_var_aug=(N_VARIATES > 32),
        val_tail_n=n_val,
        synthetic_epoch_capacity=epoch_cap,
    )
    
    dataset = synthetic_loader.dataset
    train_subset = Subset(dataset, list(range(len(dataset) - n_val)))
    val_subset = Subset(dataset, list(range(len(dataset) - n_val, len(dataset))))
    batch_size = tuned_batch_size or (min(4, DIFFUSION_BATCH_SIZE) if smoke_test else DIFFUSION_BATCH_SIZE)
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=0 if smoke_test else 4,
    )
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    model_kwargs = anchor_kwargs_from_params(best_params)
    for key in (
        "max_scale",
        "d3pm_transition_max",
        "d3pm_transition_min",
        "dit_dropout",
        "prediction_target",
        "loss_weighting",
    ):
        if key in best_params:
            model_kwargs[key] = best_params[key]
    model = create_diffusion_model(
        guidance_model=itrans_guidance,
        **model_kwargs,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    
    early_stop = EarlyStopping(patience=patience)
    best_val_loss = float('inf')
    ckpt_path = os.path.join(checkpoint_dir, 'pretrained_diffusion.pt')
    
    for epoch in range(epochs):
        set_realts_training_epoch(train_loader, epoch)
        t0 = time.time()

        model.train()
        total_loss = 0.0
        n_batches = 0
        for past, future in train_loader:
            past, future = past.to(device), future.to(device)
            optimizer.zero_grad()
            with amp_context():
                loss = model.get_loss(past, future)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for past, future in val_loader:
                past, future = past.to(device), future.to(device)
                with amp_context():
                    loss = model.get_loss(past, future)
                total_loss += loss.item()
                n_batches += 1
        val_loss = total_loss / max(n_batches, 1)

        scheduler.step()
        logger.info(
            "[Diffusion] Epoch %d/%d | Train: %.4f | Val: %.4f | LR: %.2e | Time: %.1fs",
            epoch + 1, epochs, train_loss, val_loss, scheduler.get_last_lr()[0], time.time() - t0,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                {'diffusion_params': best_params, 'itrans_checkpoint': itrans_checkpoint},
                ckpt_path,
            )
            logger.info("  -> New best! Saved to %s", ckpt_path)

        if early_stop(val_loss):
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    logger.info(f"Diffusion pretraining complete. Best val loss: {best_val_loss:.4f}")
    return ckpt_path


# ============================================================================
# PHASE 2: Fine-tuning HP Search & Training
# ============================================================================

def finetune_hp_objective(
    trial,
    dataset_name: str,
    variate_indices: List[int],
    pretrained_path: str,
    itrans_checkpoint: str,
    device: torch.device,
    smoke_test: bool = False,
    fixed_batch_size: Optional[int] = None,
    trial_ckpt_dir: Optional[str] = None,
    train_stride: Optional[int] = None,
    test_stride: Optional[int] = None,
    train_ds: Any = None,
    val_ds: Any = None,
) -> float:
    """Optuna objective for fine-tuning HP search (lr only; batch_size auto-probed or fixed).

    If ``trial_ckpt_dir`` is provided, this trial's best-epoch model state is saved
    to ``{trial_ckpt_dir}/_diff_ft_trial_{trial.number}_best.pt``. The caller picks
    the best study trial and promotes its file to the final ``best.pt`` — no
    separate "Phase 2C" retrain is performed.

    When ``train_ds`` / ``val_ds`` are passed (recommended), the caller loads real data
    once before Optuna — avoids re-reading CSVs every trial and NFS stale-handle flakes.
    """
    lr = trial.suggest_float(
        'learning_rate', FINETUNE_HP_LR_MIN, FINETUNE_HP_LR_MAX, log=True,
    )
    if fixed_batch_size is not None:
        batch_size = fixed_batch_size
    else:
        batch_size = min(4, DIFFUSION_BATCH_SIZE) if smoke_test else DIFFUSION_BATCH_SIZE

    anchor_lambda, anchor_alpha = fixed_deterministic_anchor_hp()
    
    if train_ds is None or val_ds is None:
        train_ds, val_ds, _, _ = load_dataset(
            dataset_name, variate_indices,
            stride=train_stride or WINDOW_STRIDE,
            test_stride=1 if test_stride is None else test_stride,
        )
    
    if smoke_test:
        train_ds = Subset(train_ds, list(range(min(2, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(2, len(val_ds)))))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Load iTransformer guidance
    n_iv = len(variate_indices)
    itrans_model = load_itransformer_from_checkpoint(itrans_checkpoint, n_iv, device)
    itrans_guidance = iTransformerGuidance(itrans_model)
    
    # Load pretrained diffusion (skip guidance keys — keep the attached one)
    ds_lb, ds_hz = dataset_window_lengths(dataset_name)
    model = create_diffusion_model(
        n_variates=n_iv,
        lookback=ds_lb,
        horizon=ds_hz,
        guidance_model=itrans_guidance,
        **anchor_kwargs_from_params(),
    ).to(device)
    ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
    load_diffusion_state_keep_attached_guidance(model, ckpt['model_state_dict'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    epochs = HP_TUNE_EPOCHS if not smoke_test else 1
    patience = HP_TUNE_PATIENCE if not smoke_test else 1
    early_stop = EarlyStopping(patience=patience)
    best_val_loss = float('inf')
    trial_tuned_params = {'learning_rate': lr, 'batch_size': batch_size}
    if DETERMINISTIC_ANCHOR_LOSS:
        trial_tuned_params['deterministic_anchor_lambda'] = anchor_lambda
        trial_tuned_params['deterministic_anchor_alpha'] = anchor_alpha

    trial_ckpt_path: Optional[str] = None
    if trial_ckpt_dir is not None:
        os.makedirs(trial_ckpt_dir, exist_ok=True)
        trial_ckpt_path = os.path.join(trial_ckpt_dir, f'_diff_ft_trial_{trial.number}_best.pt')

    for epoch in range(epochs):
        model.train()
        for past, future in train_loader:
            past, future = past.to(device), future.to(device)
            optimizer.zero_grad()
            with amp_context():
                loss = model.get_loss(past, future)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        model.eval()
        val_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for past, future in val_loader:
                past, future = past.to(device), future.to(device)
                with amp_context():
                    loss = model.get_loss(past, future)
                val_loss += loss.item()
                n_batches += 1
        val_loss /= max(n_batches, 1)
        
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if trial_ckpt_path is not None and is_main_process():
                ckpt_config = diffusion_arch_config_dict()
                ckpt_config.update({
                    'tuned_params': trial_tuned_params,
                    'trial_number': trial.number,
                })
                save_checkpoint(
                    unwrap_model(model), optimizer, epoch, float('nan'), val_loss,
                    ckpt_config,
                    trial_ckpt_path,
                )

        if early_stop(val_loss):
            break

    return best_val_loss


def _promote_best_trial_to_final(
    study,
    subset_dir: str,
    subset_info: Dict,
    dataset_name: str,
    norm_stats: Dict,
    fixed_batch_size: int,
    pretrained_path: str,
    itrans_checkpoint: str,
    device: torch.device,
    smoke_test: bool = False,
) -> Tuple[str, Dict]:
    """Copy the best Phase 2B trial checkpoint to best.pt (no extra retrain)."""
    del pretrained_path, itrans_checkpoint, device, smoke_test  # kept for call-site compatibility
    if study.best_trial is None:
        raise RuntimeError("Optuna study has no successful trials")

    subset_id = subset_info['subset_id']
    variate_indices = subset_info['variate_indices']
    best_num = study.best_trial.number
    tuned_params = dict(study.best_params)
    tuned_params['batch_size'] = fixed_batch_size

    src = os.path.join(subset_dir, f'_diff_ft_trial_{best_num}_best.pt')
    if not os.path.exists(src):
        raise RuntimeError(f"Best trial checkpoint missing: {src}")

    dst = os.path.join(subset_dir, 'best.pt')
    logger.info(
        f"Promoting trial {best_num} to {dst} for {subset_id} "
        f"(lr={float(tuned_params['learning_rate']):.2e}, batch_size={int(tuned_params['batch_size'])})"
    )

    import shutil
    if is_main_process():
        shutil.copy2(src, dst)
        trial_ckpt = torch.load(src, map_location='cpu', weights_only=False)
        ckpt_config = trial_ckpt.get('config', {})
        if isinstance(ckpt_config, dict) and 'tuned_params' in ckpt_config:
            tuned_params = dict(ckpt_config['tuned_params'])
            tuned_params['batch_size'] = fixed_batch_size
        best_val_loss = float(trial_ckpt.get('val_loss', study.best_value))
        best_epoch = int(trial_ckpt.get('epoch', 0)) + 1
        if DETERMINISTIC_ANCHOR_LOSS:
            lam, alpha = fixed_deterministic_anchor_hp()
            tuned_params['deterministic_anchor_lambda'] = lam
            tuned_params['deterministic_anchor_alpha'] = alpha
        os.makedirs(subset_dir, exist_ok=True)
        with open(os.path.join(subset_dir, 'metadata.json'), 'w') as f:
            json.dump({
                'subset_id': subset_id,
                'dataset_name': dataset_name,
                'variate_indices': variate_indices,
                'data_subset': subset_info.get('data_subset', {}),
                'variate_names': subset_info.get('variate_names', []),
                'norm_mean': norm_stats['mean'].tolist(),
                'norm_std': norm_stats['std'].tolist(),
                'tuned_params': tuned_params,
                'best_trial': best_num,
                'hp_best_val_loss': float(study.best_value),
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'promoted_from_trial_ckpt': True,
                'diffusion_type': DIFFUSION_TYPE,
                'image_height': IMAGE_HEIGHT,
                'max_scale': MAX_SCALE,
                'window_norm_std_floor': WINDOW_NORM_STD_FLOOR,
                'use_dual_scale': USE_DUAL_SCALE,
                'diffusion_stage': DIFFUSION_STAGE,
                'dual_scale_fine_weight': DUAL_SCALE_FINE_WEIGHT,
                'dual_scale_independent_timesteps': DUAL_SCALE_INDEPENDENT_TIMESTEPS,
                'use_guidance_channel': USE_GUIDANCE_CHANNEL,
                'cfg_dropout': CFG_DROPOUT,
                'disable_cross_attention': DISABLE_CROSS_ATTENTION,
                'cross_variate_context_bias': CROSS_VARIATE_CONTEXT_BIAS,
                'use_window_normalization': USE_WINDOW_NORMALIZATION,
                'zero_guidance_forecast': ZERO_GUIDANCE_FORECAST,
                'window_stride': WINDOW_STRIDE,
                'lookback_length': LOOKBACK_LENGTH,
                'forecast_length': FORECAST_LENGTH,
                'dit_patch_size': list(DIT_PATCH_SIZE),
            }, f, indent=2)
        for fn in os.listdir(subset_dir):
            if fn.startswith('_diff_ft_trial_') and fn.endswith('_best.pt'):
                try:
                    os.remove(os.path.join(subset_dir, fn))
                except OSError:
                    pass
    else:
        best_val_loss = float(study.best_value)
        best_epoch = 0

    return dst, {
        'best_val_loss': best_val_loss,
        'best_trial': best_num,
        'hp_best_val_loss': float(study.best_value),
        'best_epoch': best_epoch,
    }

# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(
    model: DiffusionTSF,
    test_loader: DataLoader,
    device: torch.device,
    n_samples: int = 3,
    probabilistic_n_samples: Optional[int] = None,
    probabilistic_sampler: Optional[str] = None,
    probabilistic_num_inference_steps: Optional[int] = None,
    smoke_test: bool = False,
) -> Dict:
    """Evaluate model on test set.

    Uses a single deterministic anchor decode for deterministic metrics when
    EVAL_SAMPLER is anchor / deterministic_anchor. Probabilistic CRPS/top-k and
    sample texture always come from a non-anchor stochastic sampler.
    Logs periodic progress (batch index, throughput, ETA) for Slurm logs.
    Note: ``test_loader`` should already be subsetted by the caller if a
    half-test sweep is desired (see eval call site).

    Returns:
        ``single``: MSE/MAE/trend + texture on the first draw (anchor pred when
            ``eval_sampler`` is anchor).
        ``averaged``: texture_* keys are the mean of per-draw texture metrics
            (not texture of the mean). Mean-forecast MSE/MAE is intentionally
            omitted for now because the current MMPD matrix reports
            deterministic-output MSE/MAE in its full profile.
    """
    from tqdm import tqdm
    model.eval()

    all_preds_single = []
    all_preds_avg = []
    all_samples = []
    all_prob_samples = []
    all_targets = []

    n_batches = min(1, len(test_loader)) if smoke_test else len(test_loader)
    batch_size = getattr(test_loader, 'batch_size', None) or 1
    ds = getattr(test_loader, 'dataset', None)
    n_windows = len(ds) if ds is not None else None

    def _gen_kwargs_for_sampler(sampler_name: str, *, default_steps: int) -> Dict[str, Any]:
        if sampler_name in ("anchor", "deterministic_anchor"):
            return {'sampler': 'anchor'}
        if sampler_name == "ddpm":
            return {'sampler': 'ddpm', 'use_ddim': False}
        return {'sampler': sampler_name, 'num_inference_steps': default_steps}

    eval_sampler = EVAL_SAMPLER
    det_steps = 1 if eval_sampler in ("anchor", "deterministic_anchor") else (5 if smoke_test else 20)
    det_gen_kwargs = _gen_kwargs_for_sampler(eval_sampler, default_steps=det_steps)
    anchor_sampler = det_gen_kwargs.get('sampler') == 'anchor'
    prob_sampler = probabilistic_sampler or ("dpmpp" if anchor_sampler else eval_sampler)
    prob_steps = probabilistic_num_inference_steps or (5 if smoke_test else 20)
    prob_gen_kwargs = _gen_kwargs_for_sampler(prob_sampler, default_steps=prob_steps)
    if prob_gen_kwargs.get('sampler') == 'anchor':
        raise ValueError("probabilistic_sampler must not be anchor/deterministic_anchor")

    K = getattr(model.config, 'lookback_overlap', 0)
    if probabilistic_n_samples is None:
        probabilistic_n_samples = n_samples
    effective_avg_samples = 1 if (smoke_test or anchor_sampler) else n_samples
    effective_prob_samples = 1 if (smoke_test or anchor_sampler) else probabilistic_n_samples
    if anchor_sampler and not smoke_test:
        effective_prob_samples = probabilistic_n_samples
    effective_n_samples = max(effective_avg_samples, effective_prob_samples)
    det_steps_for_log = det_gen_kwargs.get('num_inference_steps', 1 if anchor_sampler else 20)
    prob_steps_for_log = prob_gen_kwargs.get(
        'num_inference_steps',
        1 if prob_gen_kwargs.get('sampler') == 'anchor' else 20,
    )
    nfe_total = n_batches * (det_steps_for_log + effective_prob_samples * prob_steps_for_log)

    logger.info(
        "eval: start | windows=%s batches=%d batch_size=%d avg_samples=%d prob_samples=%d "
        "sampler=%s steps=%d lookback_overlap=%d device=%s "
        "(~%d U-Net forward passes across eval)",
        n_windows if n_windows is not None else '?',
        n_batches,
        batch_size,
        effective_avg_samples,
        effective_prob_samples,
        f"{det_gen_kwargs.get('sampler')}+prob:{prob_gen_kwargs.get('sampler')}",
        prob_steps_for_log,
        K,
        device,
        nfe_total,
    )

    use_tqdm = sys.stdout.isatty()
    pbar = tqdm(
        enumerate(test_loader),
        total=n_batches,
        desc='eval',
        mininterval=2.0,
        disable=not use_tqdm,
        file=sys.stdout,
    )
    log_every = max(1, min(50, n_batches // 40 or 1))
    t0 = time.perf_counter()

    with torch.no_grad():
        for batch_idx, (past, future) in pbar:
            if batch_idx >= n_batches:
                break

            past = past.to(device)
            t_batch = time.perf_counter()

            torch.manual_seed(42 + batch_idx)
            result = model.generate(past, **det_gen_kwargs)
            all_preds_single.append(result.get('prediction_global_norm', result['prediction']).cpu())

            if smoke_test:
                pred_cpu = result.get('prediction_global_norm', result['prediction']).cpu()
                all_preds_avg.append(pred_cpu)
                all_samples.append(pred_cpu.unsqueeze(0))
                all_prob_samples.append(pred_cpu.unsqueeze(0))
            else:
                samples = []
                for s_idx in range(effective_n_samples):
                    torch.manual_seed(1000 + s_idx * 17 + batch_idx)
                    result = model.generate(past, **prob_gen_kwargs)
                    samples.append(result.get('prediction_global_norm', result['prediction']).cpu())
                stacked_samples = torch.stack(samples)
                all_preds_avg.append(stacked_samples[:effective_avg_samples].mean(dim=0))
                all_samples.append(stacked_samples[:effective_avg_samples])
                all_prob_samples.append(stacked_samples[:effective_prob_samples])

            if K > 0:
                future = future[..., K:]
            all_targets.append(future)

            done = batch_idx + 1
            elapsed = time.perf_counter() - t0
            batch_wall = time.perf_counter() - t_batch
            rate = done / elapsed if elapsed > 1e-6 else 0.0
            eta = (n_batches - done) / rate if rate > 1e-9 else float('nan')
            mem_mb = ''
            if torch.cuda.is_available() and device.type == 'cuda':
                try:
                    mem_mb = f" cuda_alloc_MiB={torch.cuda.memory_allocated() / (1024 ** 2):.0f}"
                except Exception:
                    mem_mb = ''

            if (
                batch_idx < 3
                or batch_idx % log_every == 0
                or done == n_batches
            ):
                logger.info(
                    "eval: batch %d/%d (%.1f%%) | last_batch_wall=%.2fs | "
                    "avg_rate=%.3f batch/s | elapsed=%.1fs | eta=%.1fs%s",
                    done,
                    n_batches,
                    100.0 * done / n_batches,
                    batch_wall,
                    rate,
                    elapsed,
                    eta,
                    mem_mb,
                )

    logger.info(
        "eval: done | total_wall=%.1fs | batches=%d",
        time.perf_counter() - t0,
        n_batches,
    )

    preds_single = torch.cat(all_preds_single, dim=0)
    preds_avg = torch.cat(all_preds_avg, dim=0)
    samples_tensor = torch.cat(all_samples, dim=1) if len(all_samples) > 0 else preds_avg.unsqueeze(0)
    prob_samples_tensor = (
        torch.cat(all_prob_samples, dim=1)
        if len(all_prob_samples) > 0
        else samples_tensor
    )
    targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics in the dataset's global z-scored space. Diffusion generate()
    # returns prediction_global_norm after undoing only the per-window model norm.
    def compute_metrics(pred, target):
        mse = torch.nn.functional.mse_loss(pred, target).item()
        mae = torch.nn.functional.l1_loss(pred, target).item()
        
        return {'mse': mse, 'mae': mae}
    
    from models.diffusion_tsf.metrics import (
        aggregate_texture_per_sample,
        probabilistic_forecast_metrics,
        texture_metrics,
    )

    n_series = int(targets.shape[0] * targets.shape[1])
    n_draws = int(samples_tensor.shape[0]) if samples_tensor.ndim > 3 else 1
    n_prob_draws = int(prob_samples_tensor.shape[0]) if prob_samples_tensor.ndim > 3 else 1
    logger.info(
        "eval: computing MSE/MAE/trend + texture on %d windows, %d variate-series, "
        "%d avg draw(s), %d paper probabilistic draw(s) "
        "(CPU texture/CRPS; no batch logs — can take a long time on ETTm-scale data)",
        int(targets.shape[0]),
        n_series,
        n_draws,
        n_prob_draws,
    )
    t_metrics = time.perf_counter()

    t_np = targets.numpy()
    single_metrics = compute_metrics(preds_single, targets)
    # First draw (anchor when eval_sampler=anchor, else one stochastic sample).
    single_metrics.update(texture_metrics(t_np, preds_single.numpy()))

    avg_metrics = {
        "n_samples": float(n_draws),
        "point_metrics_disabled": True,
        "point_metrics_note": (
            "Mean-sample MSE/MAE disabled; MMPD full-profile mse/mae are "
            "deterministic-output metrics, not sample-mean metrics."
        ),
    }
    prob_samples_np = prob_samples_tensor.numpy()
    samples_bvsl = np.moveaxis(prob_samples_np, 0, 2)
    prob_metrics = {}
    prob_metrics.update(
        probabilistic_forecast_metrics(
            t_np,
            samples_bvsl,
            gmm_components=10,
            topk_max=3,
            seed=42,
        )
    )
    prob_texture = aggregate_texture_per_sample(t_np, prob_samples_np, max_draws=3)
    for key, val in prob_texture.items():
        prob_metrics[f"prob_{key}"] = val

    logger.info(
        "eval: metrics done | wall=%.1fs (includes texture)",
        time.perf_counter() - t_metrics,
    )

    return {
        'single': single_metrics,
        'averaged': avg_metrics,
        'deterministic_anchor': single_metrics,
        'probabilistic_averaged': avg_metrics,
        'probabilistic_avg30': avg_metrics,
        'probabilistic': prob_metrics,
    }


def _subset_results_path(results_dir: str, subset_id: str) -> str:
    """Return path to the canonical results.json for a subset."""
    return os.path.join(results_dir, subset_id, 'results.json')


def _load_subset_results(results_dir: str, subset_id: str) -> dict:
    path = _subset_results_path(results_dir, subset_id)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_subset_results(results_dir: str, subset_id: str, data: dict):
    subset_dir = os.path.join(results_dir, subset_id)
    os.makedirs(subset_dir, exist_ok=True)
    with open(os.path.join(subset_dir, 'results.json'), 'w') as f:
        json.dump(data, f, indent=2)


def save_eval_results(
    subset_id,
    dataset_name,
    variate_indices,
    train_metrics,
    eval_results,
    results_dir,
    data_subset: Optional[Dict] = None,
):
    """Save diffusion evaluation results to per-subset subdirectory."""
    data = _load_subset_results(results_dir, subset_id)
    data.update({
        'subset_id': subset_id,
        'dataset': dataset_name,
        'variate_indices': variate_indices,
        'data_subset': data_subset or {},
        'train_metrics': train_metrics,
        'eval_metrics': eval_results,
        'evaluated_at': datetime.now().isoformat(),
    })
    _save_subset_results(results_dir, subset_id, data)
    update_summary_csv(results_dir)


def update_summary_csv(results_dir):
    """Rebuild summary CSV by walking per-subset subdirectories."""
    rows = []
    results_path = Path(results_dir)
    for subset_dir in sorted(results_path.iterdir()):
        if not subset_dir.is_dir():
            continue
        rfile = subset_dir / 'results.json'
        if not rfile.exists():
            continue
        try:
            with open(rfile) as f:
                data = json.load(f)
            if 'eval_metrics' not in data:
                continue
            m = data['eval_metrics']
            itrans = data.get('itransformer_metrics', {})
            row = {
                'subset_id': data['subset_id'],
                'dataset': data['dataset'],
                'best_val_loss': data.get('train_metrics', {}).get('best_val_loss'),
                'single_mse': m['single']['mse'],
                'single_mae': m['single']['mae'],
                'avg_mse': m.get('averaged', {}).get('mse'),
                'avg_mae': m.get('averaged', {}).get('mae'),
                'itrans_mse': itrans.get('mse'),
                'itrans_mae': itrans.get('mae'),
            }
            for src, pfx in ((m['single'], 'single'), (m.get('averaged', {}), 'avg')):
                for key, val in src.items():
                    if key.startswith('texture_'):
                        row[f'{pfx}_{key}'] = val
            rows.append(row)
        except Exception:
            continue

    if rows:
        df = pd.DataFrame(rows).sort_values(['dataset', 'subset_id'])
        df.to_csv(os.path.join(results_dir, 'summary.csv'), index=False)


# ============================================================================
# iTransformer Baseline Evaluation
# ============================================================================

def train_subset_itransformer_full_baseline(
    dataset_name: str,
    variate_indices: List[int],
    subset_id: str,
    device: torch.device,
    smoke_test: bool = False,
    epochs: int = None,
    patience: int = None,
    train_stride: Optional[int] = None,
    test_stride: Optional[int] = None,
    data_subset: Optional[Dict] = None,
) -> str:
    """Train iTransformer from scratch on the full train split (no diffusion, no warm-start).

    This is the fair ``iTrans-only'' comparison: same variates and train/val windows as the
    diffusion finetune job, but a separate model not used as guidance.
    """
    ds_lb, ds_hz = dataset_window_lengths(dataset_name)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'{subset_id}_itrans_full_dataset.pt')
    if os.path.exists(ckpt_path) and not smoke_test:
        logger.info(f"  Using cached full-dataset iTransformer baseline: {ckpt_path}")
        return ckpt_path

    global LOOKBACK_LENGTH, FORECAST_LENGTH, ITRANSFORMER_SEQ_LEN
    saved_lens = (LOOKBACK_LENGTH, FORECAST_LENGTH, ITRANSFORMER_SEQ_LEN)
    LOOKBACK_LENGTH, FORECAST_LENGTH, ITRANSFORMER_SEQ_LEN = ds_lb, ds_hz, ds_lb
    try:
        train_ds, val_ds, _, _ = load_dataset(
            dataset_name, variate_indices,
            stride=train_stride or WINDOW_STRIDE,
            test_stride=1 if test_stride is None else test_stride,
        )
        if smoke_test:
            train_ds = Subset(train_ds, list(range(min(4, len(train_ds)))))
            val_ds = Subset(val_ds, list(range(min(2, len(val_ds)))))

        train_loader = DataLoader(
            train_ds, batch_size=ITRANS_PAPER_BATCH_SIZE, shuffle=True, num_workers=0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=min(ITRANS_PAPER_BATCH_SIZE, 32), shuffle=False, num_workers=0,
        )

        n_iv = len(variate_indices)
        itrans_seq, itrans_pred = itrans_model_lengths(ds_lb, ds_hz)
        model = create_itransformer(seq_len=itrans_seq, pred_len=itrans_pred, num_vars=n_iv).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()

        max_epochs = 1 if smoke_test else (epochs if epochs is not None else ITRANS_HP_FINETUNE_MAX_EPOCHS)
        patience_val = 1 if smoke_test else (patience if patience is not None else 5)
        early_stop = EarlyStopping(patience=patience_val)
        best_val = float('inf')

        logger.info(
            f"[{subset_id}] Training full-dataset iTransformer baseline "
            f"({max_epochs} epochs, lookback={ds_lb}, forecast={ds_hz}, n={n_iv})..."
        )
        for epoch in range(max_epochs):
            train_loss = train_itransformer_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = validate_itransformer(model, val_loader, criterion, device)
            logger.info(
                f"[{subset_id}] iTrans full-baseline epoch {epoch + 1}/{max_epochs} "
                f"train={train_loss:.4f} val={val_loss:.4f}"
            )
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    model, optimizer, epoch, train_loss, val_loss,
                    {
                        'subset_id': subset_id,
                        'dataset_name': dataset_name,
                        'variate_indices': variate_indices,
                        'data_subset': data_subset or {},
                        'lookback_length': ds_lb,
                        'forecast_length': ds_hz,
                        'type': 'itrans_full_dataset_baseline',
                    },
                    ckpt_path,
                )
            if early_stop(val_loss):
                break
        logger.info(f"  Full-dataset iTransformer baseline saved → {ckpt_path} (val={best_val:.4f})")
        return ckpt_path
    finally:
        LOOKBACK_LENGTH, FORECAST_LENGTH, ITRANSFORMER_SEQ_LEN = saved_lens


def evaluate_itransformer_baseline(
    subset_id: str,
    dataset_name: str,
    variate_indices: List[int],
    itrans_checkpoint: str,
    results_dir: str,
    device: torch.device,
    smoke_test: bool = False,
    test_indices: Optional[List[int]] = None,
    test_stride: Optional[int] = None,
    data_subset: Optional[Dict] = None,
) -> Dict:
    """Run iTransformer-only forecast on the test split (same windows as diffusion eval).

    Results are merged into the per-subset ``results.json`` for summary tables.
    """
    ds_lb, ds_hz = dataset_window_lengths(dataset_name)
    global LOOKBACK_LENGTH, FORECAST_LENGTH, ITRANSFORMER_SEQ_LEN
    saved_lens = (LOOKBACK_LENGTH, FORECAST_LENGTH, ITRANSFORMER_SEQ_LEN)
    LOOKBACK_LENGTH, FORECAST_LENGTH, ITRANSFORMER_SEQ_LEN = ds_lb, ds_hz, ds_lb
    try:
        _, _, test_ds, _ = load_dataset(
            dataset_name, variate_indices,
            stride=1,
            test_stride=1 if test_stride is None else test_stride,
        )
        if test_indices is not None:
            test_ds = Subset(test_ds, list(test_indices))
        elif smoke_test:
            test_ds = Subset(test_ds, list(range(min(2, len(test_ds)))))
        test_loader = DataLoader(test_ds, batch_size=8 if not smoke_test else 2, shuffle=False)

        n_iv = len(variate_indices)
        itrans_model = load_itransformer_from_checkpoint(itrans_checkpoint, n_iv, device)

        all_preds, all_targets = [], []
        with torch.no_grad():
            for past, future in test_loader:
                past = past.to(device)
                B, C, L = past.shape
                x_enc = past.permute(0, 2, 1)
                seq_sl = getattr(itrans_model, 'seq_len', L)
                if x_enc.shape[1] > seq_sl:
                    x_enc = x_enc[:, -seq_sl:, :]
                x_dec = torch.zeros(B, ds_hz, C, device=device, dtype=past.dtype)
                output = itrans_model(x_enc, None, x_dec, None)
                if isinstance(output, tuple):
                    output = output[0]
                all_preds.append(output.permute(0, 2, 1).cpu())
                if LOOKBACK_OVERLAP > 0:
                    future = future[..., LOOKBACK_OVERLAP:]
                all_targets.append(future)

        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)

        mse = torch.nn.functional.mse_loss(preds, targets).item()
        mae = torch.nn.functional.l1_loss(preds, targets).item()
        pred_diff = preds[:, :, 1:] - preds[:, :, :-1]
        tgt_diff = targets[:, :, 1:] - targets[:, :, :-1]
        trend_acc = ((pred_diff > 0) == (tgt_diff > 0)).float().mean().item()

        metrics = {'mse': mse, 'mae': mae, 'trend_accuracy': trend_acc}
        logger.info(
            f"[{subset_id}] iTransformer full-dataset baseline: "
            f"MSE={mse:.4f}, MAE={mae:.4f}, trend={trend_acc:.3f}"
        )

        data = _load_subset_results(results_dir, subset_id)
        data.setdefault('subset_id', subset_id)
        data.setdefault('dataset', dataset_name)
        data.setdefault('variate_indices', variate_indices)
        data.setdefault('data_subset', data_subset or {})
        data['itransformer_metrics'] = metrics
        data['itransformer_baseline_ckpt'] = itrans_checkpoint
        data['itransformer_evaluated_at'] = datetime.now().isoformat()
        _save_subset_results(results_dir, subset_id, data)
        update_summary_csv(results_dir)

        return metrics
    finally:
        LOOKBACK_LENGTH, FORECAST_LENGTH, ITRANSFORMER_SEQ_LEN = saved_lens


# ============================================================================
# CLI
# ============================================================================

def main():
    global logger, N_VARIATES, CHECKPOINT_DIR, RESULTS_DIR, SYNTH_CACHE_DIR, DATASETS_DIR

    parser = argparse.ArgumentParser(description="Diffusion TSF Training Pipeline")
    parser.add_argument("--config", type=str, required=True, help="YAML experiment config")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset from YAML")
    parser.add_argument("--n-variates", type=int, default=None, help="Override variate count")
    parser.add_argument("--variate-indices", type=str, default=None, help="Comma-separated variate indices")
    parser.add_argument("--subset-id", type=str, default=None, help="Optional subset id label")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--smoke-test", action="store_true", help="Quick validation run")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed from YAML")
    parser.add_argument("--parallel-optuna-workers", type=int, default=1, help="Parallel Optuna workers")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Override checkpoint directory")
    parser.add_argument("--results-dir", type=str, default=None, help="Override results directory")
    parser.add_argument("--datasets-dir", type=str, default=None, help="Benchmark CSV/NPZ root")
    parser.add_argument("--synth-cache-dir", type=str, default=None, help="Shared synthetic pool cache")
    parser.add_argument("--fresh", action="store_true", help="Wipe manifest and checkpoints")
    args = parser.parse_args()

    logger = setup_logging()

    from models.diffusion_tsf.pipeline.config import apply_cli_state_overrides
    from models.diffusion_tsf.pipeline import load_experiment_config, PipelineState, Pipeline
    from models.diffusion_tsf.pipeline.phases import PHASE_REGISTRY

    cli_overrides = {}
    if args.dataset:
        cli_overrides["dataset"] = args.dataset

    nv = args.n_variates
    variate_indices = None
    if args.variate_indices:
        variate_indices = [int(x.strip()) for x in args.variate_indices.split(",") if x.strip()]
        cli_overrides["variate_indices"] = variate_indices
        if not nv:
            nv = len(variate_indices)

    if not nv and args.dataset:
        try:
            nv = get_dim_for_dataset(args.dataset)
        except Exception:
            pass
    if nv:
        cli_overrides["n_variates"] = nv

    if args.seed is not None:
        cli_overrides["seed"] = args.seed
    if args.smoke_test:
        cli_overrides["smoke_test"] = True
    if args.checkpoint_dir:
        cli_overrides["checkpoint_dir"] = args.checkpoint_dir
    if args.results_dir:
        cli_overrides["results_dir"] = args.results_dir
    if args.datasets_dir:
        cli_overrides["datasets_dir"] = os.path.abspath(args.datasets_dir)
    if args.synth_cache_dir:
        cli_overrides["synth_cache_dir"] = args.synth_cache_dir
    if args.fresh:
        cli_overrides["fresh"] = True
    if args.resume:
        cli_overrides["resume"] = True
    if args.subset_id:
        cli_overrides["subset_id"] = args.subset_id

    parallel_workers = 1 if args.smoke_test else max(1, int(args.parallel_optuna_workers))
    cli_overrides["parallel_optuna_workers"] = parallel_workers

    cfg = load_experiment_config(args.config, cli_overrides)
    state = PipelineState.from_config(cfg)
    apply_cli_state_overrides(state, cfg)

    if args.checkpoint_dir:
        CHECKPOINT_DIR = args.checkpoint_dir
    if args.results_dir:
        RESULTS_DIR = args.results_dir
    if args.synth_cache_dir:
        SYNTH_CACHE_DIR = args.synth_cache_dir
    if nv:
        N_VARIATES = nv

    subset_meta = resolve_pipeline_data_subset(state)
    if subset_meta.get("enabled"):
        logger.info(
            "Data subset resolved: %s -> %s vars, train_stride=%s, test_stride=%s, "
            "raw=%.2f MiB, reduced≈%.2f MiB",
            state.subset_id,
            subset_meta.get("n_variates"),
            subset_meta.get("train_stride"),
            subset_meta.get("test_stride"),
            float(subset_meta.get("raw_size_mb") or 0.0),
            float(subset_meta.get("reduced_size_mb") or 0.0),
        )

    phases = []
    for p in cfg["phases"]:
        p_class = PHASE_REGISTRY.get(p["phase"])
        if not p_class:
            logger.error("Unknown phase: %s", p["phase"])
            sys.exit(1)
        phases.append(p_class(**p))

    try:
        Pipeline(phases, state, merged_config=cfg).run()
    finally:
        if state.wandb_enabled:
            from models.diffusion_tsf.pipeline import wandb_utils
            wandb_utils.finish_phase_run()




if __name__ == '__main__':
    main()
