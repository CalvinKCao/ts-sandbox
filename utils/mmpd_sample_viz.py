"""Anchor + probabilistic sample panels for completed MMPD eval packs.

Operates on the raw npz packs written by ``utils/eval_mmpd_gaussian_anchor.py``
(``<output_dir>/raw/mmpd_{dataset}.npz`` and ``<output_dir>/raw/{variant}_anchor_{dataset}.npz``):
keys ``y_true`` (N, V, H), ``deterministic`` (N, V, H), optional ``samples`` (N, V, S, H),
optional ``indices`` (N,).

Used by:
- ``utils/eval_mmpd_gaussian_anchor.py`` to auto-generate + wandb-log panels after each
  dataset's MMPD eval phase.
- ``temp/scripts/mmpd_backfill_sample_viz.py`` to backfill panels for already-completed runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models.diffusion_tsf.pipeline.visualize_utils import (
    per_window_anchor_mse,
    per_window_crps,
    pick_sample_indices,
    save_figure_jpg,
)

MAX_VARIATES = 3
_GT_COLOR = "#2196F3"
_ANCHOR_COLOR = "#E91E63"
_SAMPLE_COLOR = "#FF9800"


def _variate_ids(n_vars: int, variate_indices: Optional[Sequence[int]]) -> List[int]:
    if variate_indices:
        return list(variate_indices)[:n_vars]
    return list(range(n_vars))


def plot_mmpd_anchor_panel(
    *,
    y_true: np.ndarray,
    deterministic: np.ndarray,
    dataset: str,
    model_label: str,
    window_index: int,
    score: float,
    out_dir: Path,
    variate_indices: Optional[Sequence[int]] = None,
    jpeg_dpi: int = 100,
) -> str:
    """GT vs deterministic ("anchor") point forecast, up to 3 variates."""
    n_vars = min(MAX_VARIATES, y_true.shape[0])
    var_ids = _variate_ids(n_vars, variate_indices)
    t_axis = np.arange(y_true.shape[-1])

    fig, axes = plt.subplots(1, n_vars, figsize=(4.2 * n_vars, 2.6), squeeze=False)
    for col in range(n_vars):
        ax = axes[0, col]
        ax.plot(t_axis, y_true[col], color=_GT_COLOR, linewidth=1.5, label="GT" if col == 0 else None)
        ax.plot(
            t_axis, deterministic[col], color=_ANCHOR_COLOR, linewidth=1.2, linestyle="--",
            label=f"{model_label} anchor" if col == 0 else None,
        )
        ax.grid(True, alpha=0.15)
        ax.set_title(f"var {var_ids[col]}", fontsize=8)
        if col == 0:
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(f"{dataset} | {model_label} anchor sample | window {window_index} | mse={score:.5f}", fontsize=9)
    fig.tight_layout()
    path = out_dir / f"anchor_win{window_index}.jpg"
    return save_figure_jpg(fig, str(path), dpi=jpeg_dpi)


def plot_mmpd_prob_panel(
    *,
    y_true: np.ndarray,
    samples: np.ndarray,
    dataset: str,
    model_label: str,
    window_index: int,
    score: float,
    out_dir: Path,
    variate_indices: Optional[Sequence[int]] = None,
    max_spaghetti: int = 20,
    jpeg_dpi: int = 100,
) -> str:
    """GT vs probabilistic sample fan (q10-q90 band + spaghetti + sample mean). samples: (V, S, H)."""
    n_vars = min(MAX_VARIATES, y_true.shape[0])
    var_ids = _variate_ids(n_vars, variate_indices)
    t_axis = np.arange(y_true.shape[-1])
    n_draw = min(int(max_spaghetti), int(samples.shape[1]))

    fig, axes = plt.subplots(1, n_vars, figsize=(4.2 * n_vars, 2.6), squeeze=False)
    for col in range(n_vars):
        ax = axes[0, col]
        draws = samples[col]
        q10 = np.percentile(draws, 10, axis=0)
        q90 = np.percentile(draws, 90, axis=0)
        ax.fill_between(t_axis, q10, q90, color=_SAMPLE_COLOR, alpha=0.22, label="q10-q90" if col == 0 else None)
        for s_i in range(n_draw):
            ax.plot(t_axis, draws[s_i], color=_SAMPLE_COLOR, linewidth=0.55, alpha=0.35)
        ax.plot(t_axis, y_true[col], color=_GT_COLOR, linewidth=1.5, label="GT" if col == 0 else None)
        ax.plot(t_axis, draws.mean(axis=0), color=_ANCHOR_COLOR, linewidth=1.2, label="sample mean" if col == 0 else None)
        ax.grid(True, alpha=0.15)
        ax.set_title(f"var {var_ids[col]} | {draws.shape[0]} samples", fontsize=8)
        if col == 0:
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(f"{dataset} | {model_label} probabilistic samples | window {window_index} | crps={score:.5f}", fontsize=9)
    fig.tight_layout()
    path = out_dir / f"prob_win{window_index}.jpg"
    return save_figure_jpg(fig, path=str(path), dpi=jpeg_dpi)


def generate_mmpd_sample_visualizations(
    pack: Dict[str, np.ndarray],
    *,
    dataset: str,
    out_dir: Path,
    model_label: str = "MMPD",
    n_windows: int = 4,
    seed: int = 2026,
    variate_indices: Optional[Sequence[int]] = None,
    jpeg_dpi: int = 100,
) -> List[str]:
    """Anchor + probabilistic sample panels for a handful of windows from an eval pack.

    ``pack`` is a raw npz dict (y_true, deterministic, samples optional, indices optional),
    the same shape produced for both the MMPD model and the binary/ordinal Gaussian-anchor
    models by ``utils/eval_mmpd_gaussian_anchor.py``. Returns local jpg paths; caller wandb-logs
    them (e.g. via ``models.diffusion_tsf.pipeline.wandb_utils.log_visualization_paths``).
    """
    y_true = pack.get("y_true")
    det = pack.get("deterministic")
    samples = pack.get("samples")
    if y_true is None or det is None:
        return []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_total = y_true.shape[0]
    window_rows = pick_sample_indices(n_total, n_windows, seed=seed)
    anchor_scores = per_window_anchor_mse(y_true, det)
    prob_scores = per_window_crps(y_true, samples) if samples is not None else None

    paths: List[str] = []
    for row in window_rows:
        window_index = int(pack["indices"][row]) if "indices" in pack else row
        paths.append(
            plot_mmpd_anchor_panel(
                y_true=y_true[row],
                deterministic=det[row],
                dataset=dataset,
                model_label=model_label,
                window_index=window_index,
                score=float(anchor_scores[row]),
                out_dir=out_dir,
                variate_indices=variate_indices,
                jpeg_dpi=jpeg_dpi,
            )
        )
        if samples is not None:
            paths.append(
                plot_mmpd_prob_panel(
                    y_true=y_true[row],
                    samples=samples[row],
                    dataset=dataset,
                    model_label=model_label,
                    window_index=window_index,
                    score=float(prob_scores[row]) if prob_scores is not None else float("nan"),
                    out_dir=out_dir,
                    variate_indices=variate_indices,
                    jpeg_dpi=jpeg_dpi,
                )
            )
    return paths
