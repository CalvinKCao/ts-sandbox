"""Train-split calibration for Fourier fine-band value range (fixed high-% cutoff)."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from models.diffusion_tsf.fourier_frequency import fourier_frequency_split_np
from models.diffusion_tsf.pipeline.globals_bridge import patch_globals
from models.diffusion_tsf.pipeline.state import PipelineState

logger = logging.getLogger(__name__)


def fourier_frequency_stats_path(state: PipelineState) -> str:
    return os.path.join(state.checkpoint_dir, "fourier_frequency_stats.json")


def _coarse_bin_value_range(state: PipelineState) -> float:
    max_scale = float(state.max_scale_by_dataset.get(state.dataset, state.max_scale))
    coarse_h = int(state.coarse_image_height)
    return 2.0 * max_scale / float(coarse_h)


def _apply_stats_to_state(state: PipelineState, stats: Dict[str, Any]) -> None:
    per_var = [float(x) for x in stats["fine_max_scale_per_variate"]]
    state.fourier_fine_max_scale_per_variate = per_var
    state.fourier_fine_max_scale = float(stats["fine_max_scale"])
    state.fourier_high_freq_cutoff_bins_per_variate = None
    state.extra["fourier_frequency_stats"] = stats


def _window_normalize(
    past: torch.Tensor,
    future: torch.Tensor,
    *,
    center_mode: str,
    std_floor: float,
) -> torch.Tensor:
    if center_mode == "last":
        center = past[..., -1:]
    elif center_mode == "mean":
        center = past.mean(dim=-1, keepdim=True)
    else:
        raise ValueError(f"unknown window_norm_center {center_mode!r}")
    std = past.std(dim=-1, keepdim=True).clamp_min(float(std_floor))
    return (future - center) / std


def ensure_fourier_frequency_calibration(state: PipelineState, *, force: bool = False) -> None:
    """Calibrate fine-band clip range from train |high| at fixed high_freq_percent."""
    if state.staged_representation != "fourier_frequency":
        return

    stats_path = fourier_frequency_stats_path(state)
    if os.path.exists(stats_path) and not force:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        _apply_stats_to_state(state, stats)
        logger.info(
            "fourier_frequency: loaded fine-scale calibration %s | fine_max_scale=%.6g per_var=%s",
            stats_path,
            state.fourier_fine_max_scale,
            state.fourier_fine_max_scale_per_variate,
        )
        return

    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod
    from models.diffusion_tsf.train_multivariate_pipeline import load_dataset

    patch_globals(pipeline_mod, state, honor_dataset_windows=True)
    variate_indices = state.variate_indices
    if variate_indices is None:
        from models.diffusion_tsf.train_multivariate_pipeline import generate_dataset_job

        variate_indices = generate_dataset_job(state.dataset)["variate_indices"]

    subset_meta = state.data_subset_resolved or {}
    train_stride = int(subset_meta.get("train_stride", state.window_stride))
    test_stride = int(subset_meta.get("test_stride", 1))
    train_ds, _, _, _ = load_dataset(
        state.dataset,
        variate_indices,
        stride=train_stride,
        test_stride=test_stride,
    )
    max_windows = int(state.extra.get("fourier_calibration_max_windows", 512))
    if state.smoke_test:
        max_windows = min(max_windows, 4)
    if len(train_ds) == 0:
        raise ValueError(f"fourier_frequency calibration found no train windows for {state.dataset}")
    if len(train_ds) > max_windows:
        train_ds = Subset(train_ds, list(range(max_windows)))

    loader = DataLoader(train_ds, batch_size=min(64, max(1, len(train_ds))), shuffle=False, num_workers=0)
    flatline_atol = float(state.fourier_flatline_atol)
    high_pct = float(state.fourier_high_freq_percent)
    quantile = float(getattr(state, "fourier_fine_scale_quantile", 0.95))
    floor = _coarse_bin_value_range(state)
    edge_mode = str(getattr(state, "fourier_fft_edge_mode", "mirror_pad"))
    mirror_pad_frac = float(getattr(state, "fourier_mirror_pad_frac", 0.25))

    n_vars = 0
    abs_by_var: List[List[np.ndarray]] = []

    for past, future in loader:
        future_norm = _window_normalize(
            past,
            future,
            center_mode=state.window_norm_center,
            std_floor=state.window_norm_std_floor,
        )
        if n_vars == 0:
            n_vars = int(future_norm.shape[1])
            abs_by_var = [[] for _ in range(n_vars)]
        for vi in range(n_vars):
            for bi in range(future_norm.shape[0]):
                series = future_norm[bi, vi].detach().cpu().numpy()
                _low, high = fourier_frequency_split_np(
                    series,
                    high_freq_percent=high_pct,
                    flatline_atol=flatline_atol,
                    edge_mode=edge_mode,
                    mirror_pad_frac=mirror_pad_frac,
                )
                abs_by_var[vi].append(np.abs(high))

    per_var_scales: List[float] = []
    per_var_meta: List[Dict[str, Any]] = []
    for vi, chunks in enumerate(abs_by_var):
        pooled = np.concatenate(chunks)
        q_scale = float(np.quantile(pooled, quantile))
        scale_v = max(q_scale, floor, 1e-6)
        per_var_scales.append(scale_v)
        per_var_meta.append({
            "variate_index": vi,
            "abs_high_quantile": q_scale,
            "coarse_bin_floor": floor,
            "fine_max_scale": scale_v,
        })

    fine_scale = float(max(per_var_scales))
    stats: Dict[str, Any] = {
        "dataset": state.dataset,
        "subset_id": state.subset_id,
        "variate_indices": list(variate_indices),
        "high_freq_percent": high_pct,
        "fine_scale_quantile": quantile,
        "coarse_bin_value_range_floor": floor,
        "fine_max_scale_per_variate": per_var_scales,
        "fine_max_scale": fine_scale,
        "n_train_windows_used": int(len(train_ds)),
        "per_variate": per_var_meta,
    }
    os.makedirs(state.checkpoint_dir, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    _apply_stats_to_state(state, stats)
    logger.info(
        "fourier_frequency: calibrated fine scales %s | floor=%.4f fine_max_scale=%.6g per_var=%s",
        stats_path,
        floor,
        fine_scale,
        per_var_scales,
    )
