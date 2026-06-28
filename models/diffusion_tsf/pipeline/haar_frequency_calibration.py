"""Train-split calibration for Haar low/high staged representations."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from models.diffusion_tsf.pipeline.globals_bridge import patch_globals
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.preprocessing import TimeSeriesTo2D

logger = logging.getLogger(__name__)


def haar_frequency_stats_path(state: PipelineState) -> str:
    return os.path.join(state.checkpoint_dir, "haar_frequency_stats.json")


def _apply_stats_to_state(state: PipelineState, stats: Dict[str, Any]) -> None:
    state.haar_high_freq_levels = int(stats["selected_high_freq_levels"])
    state.haar_high_freq_percent = float(stats["selected_high_freq_percent"])
    state.haar_fine_max_scale = float(stats["fine_max_scale"])
    state.extra["haar_frequency_stats"] = stats


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


def ensure_haar_frequency_calibration(state: PipelineState, *, force: bool = False) -> None:
    """Calibrate Haar cutoff and fine value range from the real train split."""
    if state.staged_representation != "haar_frequency":
        return

    stats_path = haar_frequency_stats_path(state)
    if os.path.exists(stats_path) and not force:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        _apply_stats_to_state(state, stats)
        logger.info(
            "haar_frequency: loaded calibration %s | high_levels=%s fine_max_scale=%.6g",
            stats_path,
            state.haar_high_freq_levels,
            state.haar_fine_max_scale,
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
    max_windows = int(state.extra.get("haar_calibration_max_windows", 512))
    if state.smoke_test:
        max_windows = min(max_windows, 4)
    if len(train_ds) == 0:
        raise ValueError(f"haar_frequency calibration found no train windows for {state.dataset}")
    if len(train_ds) > max_windows:
        train_ds = Subset(train_ds, list(range(max_windows)))

    loader = DataLoader(train_ds, batch_size=min(64, max(1, len(train_ds))), shuffle=False, num_workers=0)
    to_2d = TimeSeriesTo2D(
        height=int(state.image_height),
        max_scale=float(state.max_scale_by_dataset.get(state.dataset, state.max_scale)),
    )

    width = None
    candidate_vars: Dict[int, list[torch.Tensor]] = {}
    candidate_max_abs: Dict[int, list[torch.Tensor]] = {}
    for past, future in loader:
        future_norm = _window_normalize(
            past,
            future,
            center_mode=state.window_norm_center,
            std_floor=state.window_norm_std_floor,
        )
        if width is None:
            width = int(future_norm.shape[-1])
            levels = to_2d.haar_detail_levels(width)
            if levels <= 0:
                raise ValueError(f"haar_frequency calibration requires width >= 2, got {width}")
            candidate_vars = {k: [] for k in range(1, levels + 1)}
            candidate_max_abs = {k: [] for k in range(1, levels + 1)}
        for k in candidate_vars:
            _low, high = to_2d.haar_frequency_split_values(future_norm, high_freq_levels=k)
            candidate_vars[k].append(high.var(dim=-1, unbiased=False).reshape(-1).cpu())
            candidate_max_abs[k].append(high.abs().amax(dim=-1).reshape(-1).cpu())

    assert width is not None
    levels = to_2d.haar_detail_levels(width)
    prior_pct = float(state.haar_high_freq_percent or 0.38)
    prior_levels = max(1, min(levels, int(np.ceil(levels * prior_pct))))
    cv_threshold = float(state.haar_variance_cv_threshold)
    spread_threshold = float(state.haar_variance_log_spread_threshold)
    eps = 1e-12

    candidates = []
    valid_levels = []
    for k in range(1, levels + 1):
        vars_k = torch.cat(candidate_vars[k]).numpy()
        vars_k = np.maximum(vars_k, eps)
        mean_var = float(vars_k.mean())
        cv = float(vars_k.std() / max(mean_var, eps))
        log_vars = np.log(vars_k)
        log_spread = float(np.quantile(log_vars, 0.95) - np.quantile(log_vars, 0.05))
        max_abs_k = torch.cat(candidate_max_abs[k]).numpy()
        fine_scale_k = float(np.quantile(max_abs_k, float(state.haar_fine_scale_quantile)))
        is_valid = cv <= cv_threshold and log_spread <= spread_threshold
        if is_valid:
            valid_levels.append(k)
        candidates.append(
            {
                "high_freq_levels": k,
                "high_freq_percent": k / float(levels),
                "variance_mean": mean_var,
                "variance_cv": cv,
                "variance_log_q95_q05_spread": log_spread,
                "fine_scale_quantile": float(state.haar_fine_scale_quantile),
                "fine_scale_at_quantile": fine_scale_k,
                "valid_constantish_variance": bool(is_valid),
            }
        )

    if valid_levels:
        selected_levels = max(valid_levels)
        selection_reason = "max_valid_constantish_variance"
    else:
        selected_levels = prior_levels
        selection_reason = "fallback_to_prior_percent"

    selected_max_abs = torch.cat(candidate_max_abs[selected_levels]).numpy()
    fine_scale = float(np.quantile(selected_max_abs, float(state.haar_fine_scale_quantile)))
    fine_scale = max(fine_scale, 1e-6)

    stats: Dict[str, Any] = {
        "dataset": state.dataset,
        "subset_id": state.subset_id,
        "variate_indices": list(variate_indices),
        "data_subset": subset_meta,
        "window_norm_center": state.window_norm_center,
        "window_norm_std_floor": float(state.window_norm_std_floor),
        "target_width": int(width),
        "haar_detail_levels": int(levels),
        "prior_high_freq_percent": prior_pct,
        "prior_high_freq_levels": int(prior_levels),
        "selected_high_freq_levels": int(selected_levels),
        "selected_high_freq_percent": selected_levels / float(levels),
        "selection_reason": selection_reason,
        "variance_cv_threshold": cv_threshold,
        "variance_log_spread_threshold": spread_threshold,
        "fine_scale_quantile": float(state.haar_fine_scale_quantile),
        "fine_max_scale": fine_scale,
        "n_train_windows_used": int(len(train_ds)),
        "candidates": candidates,
    }
    os.makedirs(state.checkpoint_dir, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    _apply_stats_to_state(state, stats)
    logger.info(
        "haar_frequency: calibrated %s | levels=%d/%d (%.3f) fine_max_scale=%.6g quantile=%.2f",
        stats_path,
        selected_levels,
        levels,
        state.haar_high_freq_percent,
        state.haar_fine_max_scale,
        state.haar_fine_scale_quantile,
    )
