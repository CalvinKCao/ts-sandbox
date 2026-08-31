
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
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.diffusion_model import DiffusionTSF
from models.diffusion_tsf.realts import get_synthetic_dataloader
from models.diffusion_tsf.guidance import iTransformerGuidance, PatchDecoderGuidance
from models.diffusion_tsf.patch_guidance_stack import PatchGuidanceStack, PatchGuidanceStackConfig
from models.diffusion_tsf.ordinal_window_norm import (
    build_global_ladder_from_training,
    ordinal_encode,
    ranks_to_unit,
)
from models.diffusion_tsf.pipeline.data_subset import resolve_data_subset


def is_main_process() -> bool:
    """True on the coordinator process (not an Optuna child worker)."""
    from models.diffusion_tsf.pipeline.optuna_parallel import is_optuna_child_worker
    return not is_optuna_child_worker()


def get_device(state: PipelineState) -> torch.device:
    return state.resolve_device()


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


def fixed_deterministic_anchor_hp(state: PipelineState) -> Tuple[float, float]:
    """Fixed anchor hyperparameters from YAML (not Optuna-tuned)."""
    return state.deterministic_anchor_lambda, state.deterministic_anchor_alpha


def anchor_kwargs_from_params(state: PipelineState, params: Optional[Dict] = None) -> Dict:
    """Kwargs for create_diffusion_model from immutable state settings."""
    del params  # kept for call-site compatibility; anchor HP is not tuned
    if not state.deterministic_anchor_loss:
        return {}
    anchor_lambda, anchor_alpha = fixed_deterministic_anchor_hp(state)
    return {
        'use_deterministic_anchor_loss': True,
        'deterministic_anchor_lambda': anchor_lambda,
        'deterministic_anchor_alpha': anchor_alpha,
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
from models.diffusion_tsf.pipeline import training_helpers as _training_helpers


def resolve_synthetic_params(state: PipelineState, requested_n: int, requested_cap: int, smoke_test: bool):
    return _training_helpers.resolve_synthetic_params(
        requested_n,
        requested_cap,
        smoke_test,
        samples_cap=state.synthetic_samples_full_cap,
        samples_min=state.synthetic_samples_min,
    )


def resolve_pretrain_virtual_dataset_size(state: PipelineState, smoke_test: bool) -> int:
    return _training_helpers.resolve_pretrain_virtual_dataset_size(
        smoke_test,
        pretrain_epochs=state.pretrain_epochs,
        pretrain_diffusion_max_epochs=state.pretrain_diffusion_max_epochs,
        pretrain_synthetic_override=state.pretrain_synthetic_override,
        samples_cap=state.synthetic_samples_full_cap,
        samples_min=state.synthetic_samples_min,
    )


def synthetic_epoch_capacity_itrans_hp(state: PipelineState) -> int:
    return state.itrans_hp_pretrain_max_epochs


def synthetic_epoch_capacity_diff_hp(state: PipelineState) -> int:
    return state.pretrain_diffusion_max_epochs


def synthetic_epoch_capacity_pretrain_diffusion(state: PipelineState) -> int:
    return state.pretrain_diffusion_max_epochs

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
    # PeMS benchmarks ship as NPZ (iTransformer Dataset_PEMS).
    # `PeMS` stays PEMS04 for back-compat; 03/07/08 live in sibling folders.
    'PeMS': ('PeMS/PEMS04.npz', None, 24),
    'PEMS03': ('PEMS03/PEMS03.npz', None, 24),
    'PEMS07': ('PEMS07/PEMS07.npz', None, 24),
    'PEMS08': ('PEMS08/PEMS08.npz', None, 24),
    'solar_Alabama': ('solar_Alabama/solar_Alabama.csv', 'Unnamed: 0', 96),
    # First 500k timesteps only (see datasets/dynamic/dynamic_500K.csv).
    'dynamic': ('dynamic/dynamic_500K.csv', 'date', 96),
    # Tiny synthetic series for coverage / dead-code probes (lb336/hz96 capable).
    'coverage_synth': ('coverage_synth/coverage_synth.csv', 'date', 24),
}


def _datasets_root(state: PipelineState) -> str:
    return os.path.abspath(os.path.expanduser(state.datasets_dir))


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


def _resolve_registry_path(state: PipelineState, dataset_name: str) -> Tuple[str, Optional[str]]:
    """Return (absolute path, date_col or None for NPZ/headerless)."""
    rel, date_col, _ = DATASET_REGISTRY[dataset_name]
    path = os.path.join(_datasets_root(state), rel)
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


def dataset_window_lengths(state: PipelineState, dataset_name: str) -> Tuple[int, int]:
    """Per-dataset (lookback, forecast) for finetune/eval; pretrain stays on pipeline defaults."""
    return state.lookback_length, state.forecast_length


def itrans_model_lengths(state: PipelineState, dataset_lookback: int, dataset_horizon: int) -> Tuple[int, int]:
    """iTrans seq_len / pred_len decoupled from diffusion AR chunk canvas."""
    seq_len = int(state.itrans_lookback_length or dataset_lookback)
    pred_len = dataset_horizon
    return seq_len, pred_len


def wrap_itrans_guidance(
    model: nn.Module,
    state: PipelineState,
    *,
    seq_len: Optional[int] = None,
    pred_len: Optional[int] = None,
):
    """Attach iTransformer with explicit seq/pred lens (full LB, chunk forecast)."""
    from models.diffusion_tsf.guidance import iTransformerGuidance

    if seq_len is None:
        seq_len = int(state.itrans_lookback_length or getattr(model, "seq_len", state.lookback_length))
    if pred_len is None:
        pred_len = int(state.forecast_length)
    return iTransformerGuidance(model, seq_len=int(seq_len), pred_len=int(pred_len))


def _patch_guidance_out_len(state: PipelineState) -> int:
    """Native decoder forecast length (dataset horizon, not diffusion AR chunk)."""
    return int(state.forecast_length)


def _patch_guidance_pred_len(state: PipelineState) -> int:
    return int(state.forecast_length)


def _checkpoint_is_patch_guidance(ckpt: dict) -> bool:
    cfg = ckpt.get("config")
    if isinstance(cfg, dict) and cfg.get("in_len") and cfg.get("out_len") and cfg.get("patch_size"):
        return True
    sd = ckpt.get("model_state_dict")
    if not isinstance(sd, dict):
        return False
    return any(k.startswith("decoder.") or k.startswith("mixer.") for k in sd)


def create_patch_guidance_stack(
    state: PipelineState,
    num_vars: int,
    *,
    in_len: Optional[int] = None,
    out_len: Optional[int] = None,
    patch_size: Optional[int] = None,
) -> PatchGuidanceStack:
    cfg = PatchGuidanceStackConfig(
        in_len=int(in_len or state.lookback_length),
        out_len=int(out_len or _patch_guidance_out_len(state)),
        patch_size=int(patch_size or state.mmpd_patch_size),
        data_dim=int(num_vars),
    )
    stack = PatchGuidanceStack(cfg)
    stack.set_channel_dropout_drop_frac(float(getattr(state, "channel_dropout_drop_frac", 0.0)))
    return stack


def wrap_patch_guidance(state: PipelineState, stack: PatchGuidanceStack) -> PatchDecoderGuidance:
    ordinal_ladder = state.ordinal_ladder if state.use_ordinal_window_norm else None
    return PatchDecoderGuidance(
        stack,
        chunk_horizon=_patch_guidance_pred_len(state),
        ordinal_ladder=ordinal_ladder,
    )


def load_patch_guidance_from_checkpoint(
    state: PipelineState,
    path: str,
    num_vars: int,
    device: torch.device,
    ckpt: Optional[dict] = None,
) -> PatchGuidanceStack:
    if ckpt is None:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = dict(ckpt.get("config") or {})
    cfg_dict.setdefault("data_dim", num_vars)
    cfg_dict.setdefault("in_len", state.lookback_length)
    cfg_dict.setdefault("out_len", _patch_guidance_out_len(state))
    cfg_dict.setdefault("patch_size", state.mmpd_patch_size)
    cfg = PatchGuidanceStackConfig(**cfg_dict)
    stack = PatchGuidanceStack(cfg).to(device)
    stack.set_channel_dropout_drop_frac(float(getattr(state, "channel_dropout_drop_frac", 0.0)))
    stack.load_state_dict(ckpt["model_state_dict"], strict=True)
    stack.eval()
    return stack


def load_wrapped_guidance(
    state: PipelineState,
    ckpt_path: str,
    num_vars: int,
    device: torch.device,
    *,
    guidance_type: Optional[str] = None,
    dataset_lookback: Optional[int] = None,
    dataset_horizon: Optional[int] = None,
):
    """Load finetuned patch_decoder or iTransformer encoder tokens for DiT x-attn."""
    gtype = guidance_type or state.guidance_type
    if gtype == "itransformer":
        ds_lb = dataset_lookback
        ds_hz = dataset_horizon
        if ds_lb is None or ds_hz is None:
            resolved_lb, resolved_hz = dataset_window_lengths(state, state.dataset)
            ds_lb = ds_lb if ds_lb is not None else resolved_lb
            ds_hz = ds_hz if ds_hz is not None else resolved_hz
        seq_len, pred_len = itrans_model_lengths(state, int(ds_lb), int(ds_hz))
        model = load_itransformer_from_checkpoint(state, ckpt_path, num_vars, device)
        return wrap_itrans_guidance(model, state, seq_len=seq_len, pred_len=pred_len)
    if gtype != "patch_decoder":
        raise ValueError(f"Only patch_decoder or itransformer guidance is supported; got {gtype!r}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not _checkpoint_is_patch_guidance(ckpt):
        raise ValueError(
            f"guidance_type=patch_decoder but checkpoint is not patch guidance: {ckpt_path}"
        )
    if state.use_ordinal_window_norm:
        if state.ordinal_ladder is None:
            raise ValueError(
                "state.ordinal_ladder must be set before loading ordinal patch guidance"
            )
        if not bool(ckpt.get("ordinal_patch_guidance_unit_ranks", False)):
            raise ValueError(
                "Patch guidance checkpoint was not trained with unit-rank ordinal "
                "targets. Delete this patch guidance checkpoint and retrain it."
            )
    stack = load_patch_guidance_from_checkpoint(
        state, ckpt_path, num_vars, device, ckpt=ckpt,
    )
    return wrap_patch_guidance(state, stack)


def _window_norm_past_future(
    state: PipelineState,
    past: torch.Tensor,
    future: torch.Tensor,
    *,
    apply_ood_shift: bool = False,
    data_is_ranked: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if state.use_ordinal_window_norm:
        if state.ordinal_ladder is None:
            raise ValueError("state.ordinal_ladder must be set before ordinal encoding")
        if data_is_ranked:
            return past, future
        past_ord, future_ord, _ladder, _ood_shift = ordinal_encode(
            past,
            future,
            ladder=state.ordinal_ladder,
            apply_ood_shift=apply_ood_shift,
            causal_only=state.ordinal_ood_shift_causal_only,
        )
        return past_ord, future_ord
    if not state.use_window_normalization:
        return past, future
    if state.window_norm_center == "last":
        center = past[..., -1:]
    elif state.window_norm_center == "mean":
        center = past.mean(dim=-1, keepdim=True)
    else:
        raise ValueError(f"unknown window_norm_center {state.window_norm_center!r}")
    past_std = past.std(dim=-1, keepdim=True)
    if state.window_norm_low_var_threshold > 0.0:
        std_floor = past_std.clamp_min(state.window_norm_std_floor)
        unit = torch.full_like(past_std, state.window_norm_low_var_unit_std_by_dataset.get(state.dataset, state.window_norm_low_var_unit_std))
        low_var = past_std < state.window_norm_low_var_threshold
        flat = past_std <= state.window_norm_std_floor
        std = torch.where(flat | low_var, unit, std_floor)
    else:
        std = past_std.clamp_min(state.window_norm_std_floor)
    return (past - center) / std, (future - center) / std


def _set_ordinal_loader_mode(state: PipelineState, model, loader, *, eval_mode: bool = False) -> None:
    """Configure per-batch ordinal flags on the diffusion model."""
    if not state.use_ordinal_window_norm:
        return
    ranked = _dataset_yields_ordinal_ranks(loader.dataset)
    model._ordinal_input_is_ranked = ranked
    model._ordinal_apply_ood_shift = bool(eval_mode and not ranked)


def _dataset_yields_ordinal_ranks(dataset) -> bool:
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return bool(getattr(dataset, "yields_ordinal_ranks", False))


def _patch_guidance_batch(
    state: PipelineState,
    past: torch.Tensor,
    future: torch.Tensor,
    device: torch.device,
    *,
    apply_ood_shift: bool = False,
    data_is_ranked: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if _itrans_ar_enabled(state, future.shape[-1]):
        past, future = _sample_itrans_ar_chunk(state, past, future)
    past = past.to(device)
    future = future.to(device)
    # Univariate loaders can yield (B, T); decoder expects (B, V, T).
    if past.ndim == 2:
        past = past.unsqueeze(1)
        future = future.unsqueeze(1)
    if past.ndim != 3 or future.ndim != 3:
        raise ValueError(
            f"patch guidance expects (B,V,T), got past={tuple(past.shape)} "
            f"future={tuple(future.shape)}"
        )
    past_norm, future_norm = _window_norm_past_future(
        state, past,
        future,
        apply_ood_shift=apply_ood_shift,
        data_is_ranked=data_is_ranked,
    )
    avail = future_norm.shape[-1] - state.lookback_overlap if state.lookback_overlap > 0 else future_norm.shape[-1]
    pred_len = min(_patch_guidance_out_len(state), avail)
    if state.lookback_overlap > 0:
        target = future_norm[..., state.lookback_overlap : state.lookback_overlap + pred_len]
    else:
        target = future_norm[..., :pred_len]
    if state.use_ordinal_window_norm:
        if state.ordinal_ladder is None:
            raise ValueError("state.ordinal_ladder must be set before ordinal patch guidance")
        ladder_past = state.ordinal_ladder.expand_batch(past_norm.shape[0])
        ladder_target = state.ordinal_ladder.expand_batch(target.shape[0])
        past_norm = ranks_to_unit(past_norm, ladder_past)
        target = ranks_to_unit(target, ladder_target)
    return past_norm, target


def train_patch_guidance_epoch(state: PipelineState, stack, loader, optimizer, device, scheduler=None):
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
            state, past, future, device, data_is_ranked=data_is_ranked,
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


def validate_patch_guidance(state: PipelineState, stack, loader, device):
    stack.eval()
    total_loss = 0.0
    n_batches = 0
    data_is_ranked = _dataset_yields_ordinal_ranks(loader.dataset)
    with torch.no_grad():
        for past, future in loader:
            past_norm, y_true = _patch_guidance_batch(
                state, past,
                future,
                device,
                apply_ood_shift=state.use_ordinal_window_norm,
                data_is_ranked=data_is_ranked,
            )
            loss = stack.finetune_loss(past_norm, y_true)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def patch_guidance_hp_objective(
    state: PipelineState,
    trial,
    train_loader,
    val_loader,
    num_vars: int,
    device,
    smoke_test=False,
    fixed_batch_size: Optional[int] = None,
    max_epochs: Optional[int] = None,
    trial_ckpt_dir: Optional[str] = None,
):
    lr = trial.suggest_categorical("learning_rate", state.itrans_paper_lr_grid)
    batch_size = fixed_batch_size if fixed_batch_size is not None else state.itrans_paper_batch_size
    max_epochs = state.patch_guidance_hp_finetune_max_epochs if max_epochs is None else max_epochs

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    stack = create_patch_guidance_stack(state, num_vars).to(device)

    from models.diffusion_tsf.pipeline.config import training_value
    from models.diffusion_tsf.pipeline.train.diffusion_loop import (
        EpochGroupBatchSampler,
        log_epoch_shard_contract,
        make_grouped_train_loader,
    )

    n_groups_cfg = int(training_value(state, "train_epoch_groups", 1))
    if n_groups_cfg < 1:
        raise ValueError(
            f"training.train_epoch_groups must be >= 1, got {n_groups_cfg!r}"
        )
    raw_max_bytes = training_value(state, "train_epoch_max_bytes", None)
    max_bytes = None if raw_max_bytes is None else int(raw_max_bytes)
    ds_lb, ds_hz = dataset_window_lengths(state, state.dataset)
    train_loader_local, n_groups, window_nbytes, group_nbytes = make_grouped_train_loader(
        train_loader.dataset,
        batch_size=batch_size,
        n_groups=n_groups_cfg,
        seed=int(state.seed) + 31 * int(getattr(trial, "number", 0)),
        max_bytes=max_bytes,
        n_variates=int(num_vars),
        lookback=int(ds_lb),
        horizon=int(ds_hz),
        overlap=int(state.lookback_overlap),
        smoke_test=bool(smoke_test),
    )
    sampler = getattr(train_loader_local, "batch_sampler", None)
    if not isinstance(sampler, EpochGroupBatchSampler):
        raise TypeError(
            "patch-guidance training requires EpochGroupBatchSampler; "
            f"got {type(sampler)}"
        )
    val_bs = min(batch_size, 32)
    val_loader_local = DataLoader(
        val_loader.dataset, batch_size=val_bs, shuffle=False, num_workers=0,
    )
    logger.info(
        "[Patch guidance HP] groups=%d train_n=%d train_batches=%d "
        "window_bytes=%s group_bytes=%s bs=%d",
        n_groups, len(train_loader.dataset), len(train_loader_local),
        window_nbytes, group_nbytes, batch_size,
    )
    log_epoch_shard_contract(
        name="patch_guidance_hp",
        n_groups=n_groups,
        max_epochs=max_epochs if not smoke_test else 1,
        patience=None,
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
            train_loader_local.batch_sampler.set_epoch(epoch)
            train_patch_guidance_epoch(state, stack, train_loader_local, optimizer, device)
            val_loss = validate_patch_guidance(state, stack, val_loader_local, device)
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
                            "ordinal_patch_guidance_unit_ranks": bool(state.use_ordinal_window_norm),
                            "patch_guidance_target_space": (
                                "ordinal_unit_rank"
                                if state.use_ordinal_window_norm
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
    state: PipelineState,
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
        state, dataset_name, variate_indices,
        stride=train_stride or state.window_stride,
        test_stride=1 if test_stride is None else test_stride,
    )
    from models.diffusion_tsf.pipeline.data_subset import random_window_subset
    subset_meta = state.data_subset_resolved or {}
    train_ds = random_window_subset(
        train_ds,
        subset_meta.get("train_max_windows"),
        int(state.seed) + 17,
        label="patch_guidance/train",
    )
    val_ds = random_window_subset(
        val_ds,
        subset_meta.get("val_max_windows"),
        int(state.seed) + 29,
        label="patch_guidance/val",
    )
    from models.diffusion_tsf.train_window_aug import maybe_wrap_train_window_aug

    aug_cfg = state.train_window_aug
    train_ds = maybe_wrap_train_window_aug(
        train_ds,
        enabled=bool(aug_cfg.get("enabled", False)),
        apply_prob=float(aug_cfg.get("apply_prob", 0.5)),
        seed=state.seed,
        ladder=norm_stats.get("ordinal_ladder"),
        acf_threshold=float(aug_cfg.get("acf_threshold", 0.35)),
        excluded_names=aug_cfg.get("exclude_names", ()),
    )
    if smoke_test:
        train_ds = Subset(train_ds, list(range(min(2, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(2, len(val_ds)))))

    train_bs = state.itrans_paper_batch_size
    train_loader = DataLoader(
        train_ds, batch_size=train_bs, shuffle=True, num_workers=0, drop_last=not smoke_test,
    )
    val_loader = DataLoader(val_ds, batch_size=min(train_bs, 32), shuffle=False, num_workers=0)

    trial_dir = checkpoint_dir or state.checkpoint_dir
    os.makedirs(trial_dir, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from models.diffusion_tsf.pipeline.optuna_parallel import run_optuna_study

    def objective_builder(_worker_id: int):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def objective(trial):
            return patch_guidance_hp_objective(
                state, trial, train_loader, val_loader, n_vars, dev, smoke_test,
                fixed_batch_size=train_bs,
                max_epochs=state.patch_guidance_hp_finetune_max_epochs,
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


def _itrans_chunk_horizon(state: PipelineState) -> int:
    return int(state.forecast_length)


def _itrans_ar_enabled(state: PipelineState, future_len: int) -> bool:
    chunk = _itrans_chunk_horizon(state)
    if chunk <= 0:
        return False
    dataset_h = future_len - int(state.lookback_overlap)
    return dataset_h > chunk


def _itrans_ar_num_chunks(state: PipelineState, dataset_horizon: int) -> int:
    K = int(state.lookback_overlap)
    C = _itrans_chunk_horizon(state)
    if dataset_horizon <= C:
        return 1
    stride = max(1, C - K)
    return int(math.ceil((dataset_horizon - K) / stride))


def _sample_itrans_ar_chunk(
    state: PipelineState,
    past: torch.Tensor,
    future: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random AR chunk for iTrans: full-seq past window, 96-step target."""
    K = int(state.lookback_overlap)
    C = _itrans_chunk_horizon(state)
    dataset_h = future.shape[-1] - K
    n_chunks = _itrans_ar_num_chunks(state, dataset_h)
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


def get_synth_cache_dir(state: PipelineState, checkpoint_dir: Optional[str] = None, smoke_test: Optional[bool] = None) -> Optional[str]:
    """Resolve synthetic cache dir; prefer shared cache when configured."""
    if state.smoke_test if smoke_test is None else smoke_test:
        return None
    if state.synth_cache_dir:
        os.makedirs(state.synth_cache_dir, exist_ok=True)
        return state.synth_cache_dir
    path = os.path.join(project_root, 'synth_data')
    os.makedirs(path, exist_ok=True)
    return path


# ============================================================================
# Dimensionality Helpers
# ============================================================================

def get_dataset_n_cols(state: PipelineState, dataset_name: str) -> int:
    """Return the number of numeric columns in a dataset (excluding date)."""
    path, date_col = _resolve_registry_path(state, dataset_name)
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


def get_dataset_shape(state: PipelineState, dataset_name: str) -> Tuple[int, int]:
    """Return raw row/variate counts without materializing the full numeric array."""
    key = (state.datasets_dir, dataset_name)
    if key in state.dataset_shape_cache:
        return state.dataset_shape_cache[key]
    path, date_col = _resolve_registry_path(state, dataset_name)
    data = _load_dataset_array(path, date_col)
    shape = (int(data.shape[0]), int(data.shape[1]))
    state.dataset_shape_cache[key] = shape
    return shape


def resolve_pipeline_data_subset(state) -> Dict[str, Any]:
    """Resolve state.data_subset_by_dataset and write concrete variates/strides to state."""
    raw_rows, raw_variates = get_dataset_shape(state, state.dataset)
    base_indices = list(range(raw_variates))
    policy = {"data_subset_by_dataset": state.data_subset_by_dataset}
    resolved = resolve_data_subset(
        dataset_name=state.dataset,
        raw_rows=raw_rows,
        raw_variates=raw_variates,
        base_variate_indices=base_indices,
        default_subset_id=state.subset_id,
        default_window_stride=state.window_stride,
        seed=state.seed,
        policy=policy,
    )
    state.variate_indices = list(resolved["variate_indices"])
    state.n_variates = int(resolved["n_variates"])
    state.subset_id = str(resolved["subset_id"])
    state.data_subset_resolved = resolved
    print(
        f"[data_subset] {state.dataset}: subset_id={resolved['subset_id']} "
        f"n_variates={resolved['n_variates']} train_stride={resolved['train_stride']} "
        f"val_stride={resolved['val_stride']} test_stride={resolved['test_stride']} "
        f"train_max_windows={resolved.get('train_max_windows')} "
        f"val_max_windows={resolved.get('val_max_windows')} "
        f"reason={resolved.get('reason')}"
    )
    return resolved


def get_dim_for_dataset(state: PipelineState, dataset_name: str) -> int:
    """Return native dataset dimensionality (always full variates)."""
    return get_dataset_n_cols(state, dataset_name)


def pretrain_dir_for_dim(state: PipelineState, dim: int, base_dir: str = None) -> str:
    """Checkpoint subdirectory for a specific pretrain dimensionality."""
    base = base_dir or state.checkpoint_dir
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
    state: PipelineState,
    *,
    seq_len: Optional[int] = None,
    pred_len: Optional[int] = None,
    num_vars: Optional[int] = None,
    dropout: Optional[float] = None,
):
    """Create iTransformer config object from explicit pipeline state."""
    seq_len = state.itrans_lookback_length or state.lookback_length if seq_len is None else seq_len
    pred_len = state.forecast_length if pred_len is None else pred_len
    num_vars = state.n_variates if num_vars is None else num_vars
    dropout = state.itrans_paper_dropout if dropout is None else dropout
    class iTransConfig:
        def __init__(self):
            self.seq_len = seq_len
            self.pred_len = pred_len
            self.output_attention = False
            self.use_norm = True
            self.d_model = state.itrans_d_model
            self.d_ff = state.itrans_d_ff
            self.e_layers = state.itrans_e_layers
            self.n_heads = state.itrans_n_heads
            self.dropout = dropout
            self.activation = 'gelu'
            self.embed = 'fixed'
            self.freq = 'h'
            self.factor = 1
            self.enc_in = num_vars
            self.class_strategy = 'projection'
    return iTransConfig()


def create_itransformer(
    state: PipelineState,
    *,
    seq_len: Optional[int] = None,
    pred_len: Optional[int] = None,
    num_vars: Optional[int] = None,
    dropout: Optional[float] = None,
) -> nn.Module:
    """Create iTransformer model from explicit pipeline state."""
    iTransformerModel = get_itransformer_class()
    config = create_itransformer_config(
        state, seq_len=seq_len,
        pred_len=pred_len,
        num_vars=num_vars,
        dropout=dropout,
    )
    return iTransformerModel(config)


def load_itransformer_from_checkpoint(
    pipe_state: PipelineState,
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
    sd = ckpt['model_state_dict']
    weight_key = 'enc_embedding.value_embedding.weight'
    if weight_key not in sd:
        raise RuntimeError(
            f"iTransformer checkpoint {path} is missing key {weight_key!r}; "
            f"cannot infer seq_len."
        )
    ckpt_seq_len = int(sd[weight_key].shape[1])
    proj_key = 'projector.weight'
    if proj_key in sd:
        ckpt_pred_len = int(sd[proj_key].shape[0])
    else:
        ckpt_pred_len = pipe_state.forecast_length

    model = create_itransformer(
        pipe_state, seq_len=ckpt_seq_len,
        pred_len=ckpt_pred_len,
        num_vars=num_vars,
    ).to(device)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"Cannot load iTransformer checkpoint {path} "
            f"(inferred seq_len={ckpt_seq_len}): {e}"
        ) from e
    model.eval()
    expected_seq_len = pipe_state.itrans_lookback_length or pipe_state.lookback_length
    if ckpt_seq_len != expected_seq_len:
        logger.warning(
            f"iTransformer checkpoint {path} has seq_len={ckpt_seq_len}, "
            f"differs from current state seq_len={expected_seq_len}. "
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
    architecture_key = "conditioning_architecture_version"
    if architecture_key not in ckpt_state:
        raise RuntimeError(
            "Refusing legacy diffusion checkpoint without cross-attention-only architecture "
            "marker. Guided-channel checkpoints are intentionally incompatible."
        )
    model_state = model.state_dict()
    if not torch.equal(ckpt_state[architecture_key].cpu(), model_state[architecture_key].cpu()):
        raise RuntimeError("Diffusion checkpoint conditioning architecture version mismatch.")
    filtered = {}
    for k, v in ckpt_state.items():
        if k.startswith('guidance_model.'):
            continue
        if k in model_state and model_state[k].shape != v.shape:
            dst = model_state[k]
            # Floor is 512; datasets with V>512 grow nn.Embedding. Copy the
            # pretrained rows and leave the extra IDs at init.
            if (
                k.endswith("variate_embed.weight")
                and v.ndim == 2
                and dst.ndim == 2
                and dst.shape[1] == v.shape[1]
                and dst.shape[0] > v.shape[0]
            ):
                expanded = dst.clone()
                expanded[: v.shape[0]].copy_(v)
                filtered[k] = expanded
                logger.warning(
                    "Expanding %s %s -> %s (copy pretrained rows, extra IDs stay at init)",
                    k, tuple(v.shape), tuple(dst.shape),
                )
                continue
            raise RuntimeError(
                f"Diffusion checkpoint tensor mismatch for {k}: checkpoint {v.shape} "
                f"vs current {model_state[k].shape}. Do not partially load architecture changes."
            )
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

def _resolve_guidance_type(state: PipelineState, guidance_model, override: Optional[str] = None) -> str:
    """Match DiffusionTSF routing to the attached guidance, not YAML alone."""
    if override is not None:
        return str(override)
    if isinstance(guidance_model, PatchDecoderGuidance):
        return "patch_decoder"
    if guidance_model is not None:
        return "itransformer"
    return state.guidance_type


def create_diffusion_model(
    state: PipelineState,
    *,
    guidance_model=None,
    n_variates: Optional[int] = None,
    lookback: Optional[int] = None,
    horizon: Optional[int] = None,
    diffusion_stage: Optional[str] = None,
    **overrides: Any,
) -> DiffusionTSF:
    """Build DiffusionTSF from explicit PipelineState values.

    Pass explicit lookback/horizon/n_variates for per-dataset geometry, diffusion_stage
    for staged training, and overrides only for runtime exceptions (HP search, checkpoints).
    """
    def o(key: str, default: Any) -> Any:
        if key not in overrides:
            return default
        val = overrides[key]
        return default if val is None else val

    lb = state.lookback_length if lookback is None else lookback
    hz = state.forecast_length if horizon is None else horizon
    stage = state.diffusion_stage if diffusion_stage is None else diffusion_stage
    stitch = bool(state.horizon_stitch)
    inner = int(state.horizon_chunk_inner or 96)
    if stitch:
        model_hz = int(state.lookback_overlap) + inner
    else:
        model_hz = hz + state.lookback_overlap

    config = DiffusionTSFConfig(
        num_variables=state.n_variates if n_variates is None else n_variates,
        lookback_length=lb,
        forecast_length=model_hz,
        dataset_forecast_length=hz,
        lookback_overlap=state.lookback_overlap,
        diffusion_lookback_cap=int(state.diffusion_lookback_cap or 0),
        diffusion_chunk_horizon=0,
        horizon_stitch=stitch,
        horizon_chunk_inner=inner,
        representation_time_stride=int(state.representation_time_stride),
        past_cond_resize_to_horizon=bool(state.past_cond_resize_to_horizon),
        itrans_lookback_length=state.itrans_lookback_length,
        past_loss_weight=state.past_loss_weight,
        image_height=state.image_height,
        coarse_image_height=state.coarse_image_height,
        fine_image_height=state.fine_image_height,
        max_scale=o("max_scale", state.max_scale_by_dataset.get(state.dataset, state.max_scale)),
        staged_representation=o("staged_representation", state.staged_representation),
        binary_noise_schedule=o("binary_noise_schedule", state.binary_noise_schedule),
        binary_length_mode=o("binary_length_mode", state.binary_length_mode),
        binary_length_g=float(o("binary_length_g", state.binary_length_g_by_dataset.get(state.dataset, state.binary_length_g))),
        binary_length_scale=float(o("binary_length_scale", state.binary_length_scale)),
        prediction_target=o("prediction_target", state.prediction_target),
        loss_weighting=o("loss_weighting", state.loss_weighting),
        min_snr_gamma=o("min_snr_gamma", state.min_snr_gamma),
        use_coordinate_channel=state.use_coordinate_channel,
        use_raw_lookback_cond_channel=o(
            "use_raw_lookback_cond_channel", state.use_raw_lookback_cond_channel,
        ),
        guidance_penalty_weight=0.0,
        model_type=o("model_type", state.model_type),
        disable_cross_attention=state.disable_cross_attention,
        diffusion_stage=stage,
        dit_patch_size=state.dit_patch_size,
        dit_embed_dim=state.dit_embed_dim,
        dit_depth=state.dit_depth,
        dit_num_heads=state.dit_num_heads,
        dit_mlp_ratio=state.dit_mlp_ratio,
        dit_dropout=o("dit_dropout", state.dit_dropout),
        dit_cond_patch_size=state.dit_cond_patch_size,
        patch_refine_canvas_height=state.patch_refine_canvas_height,
        patch_refine_patch_height=state.patch_refine_patch_height,
        patch_refine_patch_width=state.patch_refine_patch_width,
        patch_refine_col_stride=state.patch_refine_col_stride,
        patch_refine_unique_segments=state.patch_refine_unique_segments,
        patch_refine_finetune_patch_fraction=float(
            getattr(state, "patch_refine_finetune_patch_fraction", 1.0)
        ),
        use_gradient_checkpointing=state.use_gradient_checkpointing,
        unet_max_chunk_size=state.unet_max_chunk_size,
        use_amp=state.use_amp,
        diffusion_type=o("diffusion_type", state.diffusion_type),
        use_ordinal_window_norm=o("use_ordinal_window_norm", state.use_ordinal_window_norm),
        ordinal_ood_shift_causal_only=o(
            "ordinal_ood_shift_causal_only", state.ordinal_ood_shift_causal_only,
        ),
        ordinal_tie_atol=o("ordinal_tie_atol", state.ordinal_tie_atol),
        ordinal_ladder=o("ordinal_ladder", state.ordinal_ladder),
        use_deterministic_anchor_loss=o("use_deterministic_anchor_loss", state.deterministic_anchor_loss),
        deterministic_anchor_lambda=o("deterministic_anchor_lambda", state.deterministic_anchor_lambda),
        deterministic_anchor_alpha=o("deterministic_anchor_alpha", state.deterministic_anchor_alpha),
        binary_anchor_input_mode=o("binary_anchor_input_mode", state.binary_anchor_input_mode),
        binary_use_boundary_weighted_bce=o(
            "binary_use_boundary_weighted_bce", state.binary_use_boundary_weighted_bce,
        ),
        binary_cdf_distance_alpha=float(
            o("binary_cdf_distance_alpha", state.binary_cdf_distance_alpha)
        ),
        cross_variate_context_bias=state.cross_variate_context_bias,
        channel_dropout_drop_frac=float(
            getattr(state, "channel_dropout_drop_frac", 0.0)
        ),
        cfg_dropout=state.cfg_dropout,
        binary_num_steps=o("binary_num_steps", state.binary_num_steps),
        binary_beta_start=o("binary_beta_start", state.binary_beta_start),
        binary_beta_end=o("binary_beta_end", state.binary_beta_end),
        use_window_normalization=state.use_window_normalization,
        window_norm_center=state.window_norm_center,
        window_norm_std_floor=state.window_norm_std_floor,
        window_norm_low_var_threshold=state.window_norm_low_var_threshold,
        window_norm_low_var_unit_std=state.window_norm_low_var_unit_std_by_dataset.get(state.dataset, state.window_norm_low_var_unit_std),
        window_norm_low_var_unit_std_per_variate=state.window_norm_low_var_unit_std_by_variate.get(state.dataset),
        skip_window_norm_variate_mask=list(state.skip_window_norm_variate_mask)
        if state.skip_window_norm_variate_mask is not None
        else None,
        hybrid_flat_dataset_norm=state.hybrid_flat_dataset_norm,
        hybrid_flat_frac_threshold=state.hybrid_flat_frac_threshold,
        hybrid_flat_oob_coverage=state.hybrid_flat_oob_coverage,
        lookback_overlap_center_shift=state.lookback_overlap_center_shift,
        itrans_d_model=state.itrans_d_model,
        guidance_type=_resolve_guidance_type(
            state, guidance_model, o("guidance_type", None),
        ),
        mmpd_patch_size=state.mmpd_patch_size,
    )
    model = DiffusionTSF(config, guidance_model=guidance_model)
    if bool(getattr(state, "torch_compile", False)) and not bool(state.smoke_test):
        if not torch.cuda.is_available():
            raise RuntimeError("training.torch_compile=true requires CUDA")
        logger.info(
            "torch.compile FactorizedDiT (inductor, fullgraph=False, dynamic=True)"
        )
        model.noise_predictor = torch.compile(
            model.noise_predictor,
            backend="inductor",
            fullgraph=False,
            dynamic=True,
        )
    return model


# ============================================================================
# Dataset Classes & Data Splitting Protocol
# ============================================================================

class TimeSeriesDataset(Dataset):
    """Dataset for multivariate time series forecasting.
    
    Generates sliding window samples of (past_lookback, future_horizon) from
    a continuous time-series 2D array of shape (time_steps, num_variates).
    
    Window Indexing:
    - start = idx * stride
    - past: time slice [start : start + lookback]
    - future: time slice [start + lookback - lookback_overlap : start + lookback + horizon]
      (includes lookback_overlap past steps to ensure smooth predictions at the boundary)
    """

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
        # Number of valid sliding windows fitting in data length given the stride
        self.n_samples = max(0, (len(data) - total_len) // stride + 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        source = self.rank_data if self.rank_data is not None else self.data
        # past lookback window: shape (num_variates, lookback)
        past = source[start:start + self.lookback].T
        # target future horizon window (with overlap): shape (num_variates, lookback_overlap + horizon)
        target_start = start + self.lookback - self.lookback_overlap
        target_end = start + self.lookback + self.horizon
        future = source[target_start:target_end].T
        return past, future


def _paper_split_borders(dataset_name: str, n: int, seq_len: int) -> Tuple[List[int], List[int]]:
    """Return (border1s, border2s) start/end index boundaries for train/val/test splits.

    Following the standard benchmark protocol from iTransformer / TimesNet:
    
    1. Border Ratios / Fixed Months:
       - ETTh1, ETTh2: Fixed 12-month train, 4-month val, 4-month test (24h resolution).
       - ETTm1, ETTm2: Fixed 12-month train, 4-month val, 4-month test (15-min resolution).
       - PeMS: Length-based ratio 60% train / 20% val / 20% test.
       - All other datasets (weather, electricity, traffic, etc.): 70% train / 10% val / 20% test.

    2. The Lookback Overlap Trick (b1 boundaries):
       - border2s = [end_train, end_val, end_test]
       - border1s = [0, end_train - seq_len, end_val - seq_len]
       - val/test split slice starts `seq_len` steps BEFORE the split boundary so that the
         very first validation/testing window has a full lookback history inside the dataset,
         preventing any unevaluated "dead zone" at split boundaries.
    """
    if dataset_name in ('ETTh1', 'ETTh2'):
        # 12 months, 4 months, 4 months at hourly resolution
        b2 = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    elif dataset_name in ('ETTm1', 'ETTm2'):
        # 12 months, 4 months, 4 months at 15-min resolution
        b2 = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    elif dataset_name in ('PeMS', 'PEMS03', 'PEMS07', 'PEMS08'):
        # 60% train, 20% val, 20% test
        n_train = int(n * 0.6)
        n_val = int(n * 0.2)
        n_test = n - n_train - n_val
        b2 = [n_train, n_train + n_val, n_train + n_val + n_test]
    elif dataset_name == 'illness':
        # iTransformer Dataset_Custom.__read_data__ and PatchTST_supervised
        # Dataset_Custom.__read_data__: num_train=int(n*0.7), num_test=int(n*0.2),
        # remainder val. Illness scripts set --data custom --data_path national_illness.csv
        # (not 60/20/20). Do not fold this into the generic else.
        n_train = int(n * 0.7)
        n_test = int(n * 0.2)
        n_val = n - n_train - n_test
        b2 = [n_train, n_train + n_val, n_train + n_val + n_test]
    else:
        # Standard benchmark ratio: 70% train, 10% val, 20% test
        n_train = int(n * 0.7)
        n_test = int(n * 0.2)
        n_val = n - n_train - n_test
        b2 = [n_train, n_train + n_val, n_train + n_val + n_test]
    # Apply lookback overlap to start indices of val (b1[1]) and test (b1[2])
    b1 = [0, b2[0] - seq_len, b2[1] - seq_len]
    return b1, b2


def load_dataset(
    state: PipelineState,
    dataset_name: str,
    variate_indices: List[int] = None,
    lookback: Optional[int] = None,
    horizon: Optional[int] = None,
    stride: Optional[int] = None,
    test_stride: Optional[int] = None,
    lookback_overlap: Optional[int] = None,
    ordinal_tie_atol: float = 1e-6,
    use_ordinal_window_norm: Optional[bool] = None,
    hybrid_flat_dataset_norm: Optional[bool] = None,
    hybrid_flat_frac_threshold: Optional[float] = None,
    hybrid_flat_oob_coverage: Optional[float] = None,
    max_scale: Optional[float] = None,
) -> Tuple[Dataset, Dataset, Dataset, Dict]:
    """Load raw dataset and construct PyTorch Dataset objects for train, val, and test splits.

    Data Flow & Leak Prevention:
    1. Load continuous raw array & filter requested variate columns.
    2. Partition into train/val/test boundary ranges using `_paper_split_borders`.
    3. Normalization: Compute mean and std strictly on the TRAINING SLICE (`data[:train_end]`),
       then normalize the ENTIRE sequence using these training statistics to prevent data leakage.
       When ``hybrid_flat_dataset_norm`` is on, flat variates (see
       ``utils/hybrid_flat_dataset_norm``) use a coverage scale instead of empirical std
       so >=oob_coverage of train lookbacks stay inside ``[-max_scale, max_scale]``.
    4. Optional Ordinal Ladder: When ordinal window norm is enabled, build a global rank
       ladder strictly from training data values (`build_global_ladder_from_training`).
    5. Construct TimeSeriesDataset instances:
       - train_ds: Uses `stride` (e.g., WINDOW_STRIDE)
       - val_ds: Uses `stride`
       - test_ds: Uses `test_stride` (allows dense eval test_stride=1 while train_stride is larger)
    """
    if stride is None:
        stride = state.window_stride
    if test_stride is None:
        test_stride = stride
    if lookback_overlap is None:
        lookback_overlap = state.lookback_overlap
    if lookback is None:
        lookback = state.lookback_length
    if horizon is None:
        horizon = state.forecast_length
    path, date_col = _resolve_registry_path(state, dataset_name)
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
    use_hybrid = (
        state.hybrid_flat_dataset_norm
        if hybrid_flat_dataset_norm is None
        else bool(hybrid_flat_dataset_norm)
    )
    use_ord = state.use_ordinal_window_norm if use_ordinal_window_norm is None else bool(use_ordinal_window_norm)
    if use_hybrid and use_ord:
        raise ValueError(
            "hybrid_flat_dataset_norm is incompatible with use_ordinal_window_norm"
        )
    frac_thr = (
        state.hybrid_flat_frac_threshold
        if hybrid_flat_frac_threshold is None
        else float(hybrid_flat_frac_threshold)
    )
    oob_cov = (
        state.hybrid_flat_oob_coverage
        if hybrid_flat_oob_coverage is None
        else float(hybrid_flat_oob_coverage)
    )
    ms = float(state.max_scale_by_dataset.get(dataset_name, state.max_scale) if max_scale is None else max_scale)

    if use_hybrid:
        from utils.hybrid_flat_dataset_norm import build_hybrid_affine_scales

        hybrid = build_hybrid_affine_scales(
            train_slice,
            lookback=int(lookback),
            max_scale=ms,
            frac_threshold=frac_thr,
            oob_coverage=oob_cov,
        )
        mean = hybrid["mean"].astype(np.float32)
        std = hybrid["std"].astype(np.float32)
        data = (data - mean) / std
        flat_mask = hybrid["flat_mask"]
        state.skip_window_norm_variate_mask = [bool(x) for x in flat_mask.tolist()]
    else:
        mean = train_slice.mean(axis=0, keepdims=True)
        std = train_slice.std(axis=0, keepdims=True) + 1e-8
        data = (data - mean) / std
        hybrid = None
        state.skip_window_norm_variate_mask = None

    ordinal_ladder = None
    rank_full = None
    if use_ord:
        ordinal_ladder = build_global_ladder_from_training(
            data[border1s[0]:border2s[0]],
            tie_atol=float(ordinal_tie_atol),
            precompute_ranks_for=data,
        )
        rank_full = ordinal_ladder.precomputed_ranks.numpy()
        state.ordinal_ladder = ordinal_ladder

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
    if hybrid is not None:
        stats['flat_variate_mask'] = hybrid['flat_mask']
        stats['flat_variate_frac'] = hybrid['flat_frac']
        stats['emp_std'] = hybrid['emp_std'].astype(np.float32)
        stats['hybrid_flat_details'] = hybrid['flat_details']
        stats['hybrid_flat_dataset_norm'] = True
        stats['hybrid_flat_frac_threshold'] = frac_thr
        stats['hybrid_flat_oob_coverage'] = oob_cov
        stats['hybrid_flat_max_scale'] = ms
        stats['hybrid_flat_lookback'] = int(lookback)
    return train_ds, val_ds, test_ds, stats


# ============================================================================
# Variate Subset Management
# ============================================================================

def generate_dataset_job(state: PipelineState, dataset_name: str, n_variates: int = None, seed: int = 42) -> Dict:
    """Return one full-dataset training job (no variate partitioning)."""
    path, date_col = _resolve_registry_path(state, dataset_name)
    n_cols = get_dataset_n_cols(state, dataset_name)
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

def _itrans_targets(state: PipelineState, future: torch.Tensor, model: nn.Module, device: torch.device) -> torch.Tensor:
    """Align supervised horizon with iTransformer pred_len (AR may use H>pred_len)."""
    y_true = future.permute(0, 2, 1).to(device)
    if state.lookback_overlap > 0:
        y_true = y_true[:, state.lookback_overlap:, :]
    pred_len = int(getattr(model, "pred_len", 0) or 0)
    if pred_len > 0:
        if y_true.shape[1] < pred_len:
            raise ValueError(
                f"iTrans target length {y_true.shape[1]} < model pred_len {pred_len}"
            )
        y_true = y_true[:, :pred_len, :]
    return y_true


def _itrans_batch(
    state: PipelineState,
    past: torch.Tensor,
    future: torch.Tensor,
    model: nn.Module,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare iTrans past/target: full seq_len lookback, 96-step chunk target."""
    if _itrans_ar_enabled(state, future.shape[-1]):
        past, future = _sample_itrans_ar_chunk(state, past, future)
    past = past.to(device)
    future = future.to(device)
    x_enc = past.permute(0, 2, 1)
    seq_sl = int(getattr(model, "seq_len", x_enc.shape[1]) or x_enc.shape[1])
    if x_enc.shape[1] > seq_sl:
        x_enc = x_enc[:, -seq_sl:, :]
    y_true = _itrans_targets(state, future, model, device)
    return x_enc, y_true


def train_itransformer_epoch(state: PipelineState, model, loader, optimizer, criterion, device, scheduler=None):
    """Train iTransformer for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for past, future in loader:
        x_enc, y_true = _itrans_batch(state, past, future, model, device)
        
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


def validate_itransformer(state: PipelineState, model, loader, criterion, device):
    """Validate iTransformer."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for past, future in loader:
            x_enc, y_true = _itrans_batch(state, past, future, model, device)
            y_pred = model(x_enc, None, None, None)
            loss = criterion(y_pred, y_true)
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / max(n_batches, 1)


def itrans_hp_objective(
    state: PipelineState,
    trial,
    synthetic_loader,
    val_loader,
    device,
    smoke_test=False,
    fixed_batch_size: Optional[int] = None,
    best_state: Optional[dict] = None,
    pretrained_ckpt: Optional[str] = None,
    max_epochs: Optional[int] = None,
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
    lr = trial.suggest_categorical('learning_rate', state.itrans_paper_lr_grid)
    batch_size = fixed_batch_size if fixed_batch_size is not None else state.itrans_paper_batch_size
    max_epochs = state.itrans_hp_pretrain_max_epochs if max_epochs is None else max_epochs

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = create_itransformer(state, seq_len=seq_len, pred_len=pred_len).to(device)
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
            train_itransformer_epoch(state, model, train_loader, optimizer, criterion, device)
            val_loss = validate_itransformer(state, model, val_loader_local, criterion, device)

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                tuned = {
                    'learning_rate': lr,
                    'batch_size': batch_size,
                    'dropout': state.itrans_paper_dropout,
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
    state: PipelineState,
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
        f"iTransformer seq_len={state.itrans_lookback_length or state.lookback_length} "
        f"(diffusion lookback={state.lookback_length})"
    )
    logger.info("=" * 60)

    requested_n = state.synthetic_samples_hp_tune
    requested_cap = synthetic_epoch_capacity_itrans_hp(state)
    n_samples, epoch_cap = resolve_synthetic_params(state, requested_n, requested_cap, smoke_test)

    n_val = 0 if smoke_test else min(n_samples // 10, 1000)
    synth_cache = get_synth_cache_dir(state, smoke_test=smoke_test)
    synthetic_loader = get_synthetic_dataloader(
        batch_size=64,
        lookback_length=state.lookback_length,
        forecast_length=state.forecast_length,
        num_variables=state.n_variates,
        num_samples=n_samples,
        num_workers=0,
        lookback_overlap=state.lookback_overlap,
        cache_dir=synth_cache,
        skip_cross_var_aug=(state.n_variates > 32),
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

    train_bs = state.itrans_paper_batch_size
    train_loader = DataLoader(train_subset, batch_size=train_bs, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_subset,   batch_size=min(train_bs, 32), shuffle=False, num_workers=0)

    trial_dir = checkpoint_dir or state.checkpoint_dir
    os.makedirs(trial_dir, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from models.diffusion_tsf.pipeline.optuna_parallel import run_optuna_study

    def objective_builder(_worker_id: int):
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        def objective(trial):
            return itrans_hp_objective(
                state, trial, train_loader, val_loader, dev, smoke_test,
                fixed_batch_size=train_bs,
                max_epochs=state.itrans_hp_pretrain_max_epochs,
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
    best_params['dropout'] = state.itrans_paper_dropout
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


def _ensure_synthetic_ordinal_ladder(state: PipelineState, dataset) -> None:
    """Build ordinal_ladder from RealTS pool when pretrain has no real-data ladder yet.

    Ordinal staged pretrain still trains patch guidance with unit-rank targets
    (`ordinal_patch_guidance_unit_ranks`). Real loaders set the ladder in
    ``load_normalized_splits``; synthetic guidance has only a RealTS pool, so
    build one here or fail fast.
    """
    if not state.use_ordinal_window_norm:
        return
    if state.ordinal_ladder is not None:
        return
    cache = getattr(dataset, "data_cache", None)
    if cache is None:
        raise ValueError(
            "ordinal synthetic patch guidance requires RealTS.data_cache to build "
            "state.ordinal_ladder (no real-data ladder at pretrain time)"
        )
    arr = np.asarray(cache)
    # Cap so mmap pools do not explode RAM; unique-value ladders converge quickly.
    max_seqs = 4096 if not getattr(state, "smoke_test", False) else min(arr.shape[0], 64)
    if arr.shape[0] > max_seqs:
        arr = np.asarray(arr[:max_seqs])
    if arr.ndim == 3:
        # RealTS multivariate cache: (N, V, T) -> (N*T, V)
        n, v, t = arr.shape
        train = np.transpose(arr, (0, 2, 1)).reshape(n * t, v)
    elif arr.ndim == 2:
        # Univariate: (N, T) -> (N*T, 1)
        train = arr.reshape(-1, 1)
    else:
        raise ValueError(f"unexpected RealTS data_cache shape {tuple(arr.shape)}")
    state.ordinal_ladder = build_global_ladder_from_training(
        train,
        tie_atol=float(state.ordinal_tie_atol),
    )
    logger.info(
        "Built synthetic ordinal_ladder for patch guidance (cells=%d, V=%d)",
        train.shape[0],
        train.shape[1],
    )


def run_patch_guidance_synthetic_tuning(
    state: PipelineState,
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

    requested_n = state.synthetic_samples_hp_tune
    requested_cap = synthetic_epoch_capacity_itrans_hp(state)
    n_samples, epoch_cap = resolve_synthetic_params(state, requested_n, requested_cap, smoke_test)

    n_val = 0 if smoke_test else min(n_samples // 10, 1000)
    synth_cache = get_synth_cache_dir(state, smoke_test=smoke_test)
    synthetic_loader = get_synthetic_dataloader(
        batch_size=64,
        lookback_length=state.lookback_length,
        forecast_length=state.forecast_length,
        num_variables=state.n_variates,
        num_samples=n_samples,
        num_workers=0,
        lookback_overlap=state.lookback_overlap,
        cache_dir=synth_cache,
        skip_cross_var_aug=(state.n_variates > 32),
        val_tail_n=n_val,
        synthetic_epoch_capacity=epoch_cap,
    )

    dataset = synthetic_loader.dataset
    _ensure_synthetic_ordinal_ladder(state, dataset)
    if smoke_test:
        n_val = max(1, min(len(dataset) // 4, len(dataset) - 1))
    else:
        n_val = min(len(dataset) // 10, 1000)
    train_subset = Subset(dataset, list(range(len(dataset) - n_val)))
    val_subset = Subset(dataset, list(range(len(dataset) - n_val, len(dataset))))

    train_bs = state.itrans_paper_batch_size
    train_loader = DataLoader(train_subset, batch_size=train_bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=min(train_bs, 32), shuffle=False, num_workers=0)

    trial_dir = checkpoint_dir or state.checkpoint_dir
    os.makedirs(trial_dir, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from models.diffusion_tsf.pipeline.optuna_parallel import run_optuna_study

    def objective_builder(_worker_id: int):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def objective(trial):
            return patch_guidance_hp_objective(
                state, trial, train_loader, val_loader, state.n_variates, dev, smoke_test,
                fixed_batch_size=train_bs,
                max_epochs=state.itrans_hp_pretrain_max_epochs,
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
    state: PipelineState,
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
    warm = (None if state.itrans_real_cold_start else pretrained_ckpt)
    logger.info("=" * 60)
    logger.info(f"iTrans Finetune HP Tuning: {label} ({n_trials} trials, {parallel_workers} workers)")
    logger.info(
        f"{state.itrans_hp_finetune_max_epochs} epochs per trial, "
        f"warm_start={'no (cold start)' if warm is None else os.path.basename(warm)}"
    )
    logger.info("=" * 60)

    train_ds, val_ds, _, _ = load_dataset(
        state, dataset_name, variate_indices,
        stride=train_stride or state.window_stride,
        test_stride=1 if test_stride is None else test_stride,
    )
    if smoke_test:
        train_ds = Subset(train_ds, list(range(min(2, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(2, len(val_ds)))))

    train_bs = state.itrans_paper_batch_size
    train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=min(train_bs, 32), shuffle=False, num_workers=0)

    trial_dir = checkpoint_dir or state.checkpoint_dir
    os.makedirs(trial_dir, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    ds_lb, ds_hz = dataset_window_lengths(state, dataset_name)
    itrans_seq, itrans_pred = itrans_model_lengths(state, ds_lb, ds_hz)

    from models.diffusion_tsf.pipeline.optuna_parallel import run_optuna_study

    def objective_builder(_worker_id: int):
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        def objective(trial):
            return itrans_hp_objective(
                state, trial, train_loader, val_loader, dev, smoke_test,
                fixed_batch_size=train_bs,
                pretrained_ckpt=warm,
                max_epochs=state.itrans_hp_finetune_max_epochs,
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
    best_params['dropout'] = state.itrans_paper_dropout
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
