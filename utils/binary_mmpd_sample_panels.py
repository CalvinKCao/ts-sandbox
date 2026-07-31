"""Side-by-side binary vs MMPD panels: GT + anchor + probabilistic mean/band.

Used by one-off Killarney scripts, ordinal disc eval, and (per-model) staged/MMPD
eval hooks. Operates on already-aligned ``(V, H)`` / ``(V, S, H)`` tensors for one
window — callers own index alignment (pack-test-stride / MMPD indices).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models.diffusion_tsf.pipeline.visualize_utils import save_figure_jpg

_GT = "#2196F3"
_BIN_ANCHOR = "#E91E63"
_BIN_PROB = "#FF9800"
_MMPD_ANCHOR = "#9C27B0"
_MMPD_PROB = "#4CAF50"


def _variate_ids(n_vars: int, variate_indices: Optional[Sequence[int]]) -> List[int]:
    if variate_indices:
        return list(variate_indices)[:n_vars]
    return list(range(n_vars))


def _prob_band(ax, t_axis, samples_vsh: np.ndarray, *, color: str, label_prefix: str) -> None:
    """samples_vsh: (S, H) for one variate."""
    q10 = np.percentile(samples_vsh, 10, axis=0)
    q90 = np.percentile(samples_vsh, 90, axis=0)
    mean = samples_vsh.mean(axis=0)
    ax.fill_between(t_axis, q10, q90, color=color, alpha=0.18, label=f"{label_prefix} q10-q90")
    ax.plot(t_axis, mean, color=color, linewidth=1.3, label=f"{label_prefix} mean")


def plot_binary_vs_mmpd_anchor_prob_window(
    *,
    y_true: np.ndarray,
    past: Optional[np.ndarray] = None,
    binary_anchor: Optional[np.ndarray] = None,
    binary_samples: Optional[np.ndarray] = None,
    mmpd_anchor: Optional[np.ndarray] = None,
    mmpd_samples: Optional[np.ndarray] = None,
    dataset: str,
    window_index: int,
    out_path: Path,
    variate_indices: Optional[Sequence[int]] = None,
    lookback_tail: int = 48,
    jpeg_dpi: int = 110,
) -> str:
    """One shared window: rows = models (binary / MMPD), cols = variates.

    ``y_true`` / anchors: ``(V, H)``. samples: ``(V, S, H)``.
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    n_vars = min(3, int(y_true.shape[0]))
    var_ids = _variate_ids(n_vars, variate_indices)
    horizon = int(y_true.shape[-1])
    t_h = np.arange(horizon)

    rows = []
    if binary_anchor is not None or binary_samples is not None:
        rows.append(("binary", binary_anchor, binary_samples, _BIN_ANCHOR, _BIN_PROB))
    if mmpd_anchor is not None or mmpd_samples is not None:
        rows.append(("mmpd", mmpd_anchor, mmpd_samples, _MMPD_ANCHOR, _MMPD_PROB))
    if not rows:
        raise ValueError("need at least one of binary_* / mmpd_* forecast arrays")

    fig, axes = plt.subplots(
        len(rows), n_vars, figsize=(4.0 * n_vars, 2.7 * len(rows)), squeeze=False,
    )
    for r, (name, anchor, samples, c_anchor, c_prob) in enumerate(rows):
        for c in range(n_vars):
            ax = axes[r, c]
            if past is not None:
                tail = np.asarray(past[c, -lookback_tail:], dtype=np.float32)
                t_past = np.arange(-len(tail), 0)
                ax.plot(t_past, tail, color="0.45", linewidth=1.0, label="lookback" if c == 0 else None)
                ax.axvline(0, color="black", linewidth=0.7, alpha=0.45)
            ax.plot(t_h, y_true[c], color=_GT, linewidth=1.5, label="GT" if c == 0 else None)
            if samples is not None and samples.shape[1] >= 1:
                _prob_band(ax, t_h, samples[c], color=c_prob, label_prefix=f"{name} prob")
            if anchor is not None:
                ax.plot(
                    t_h, anchor[c], color=c_anchor, linewidth=1.2, linestyle="--",
                    label=f"{name} anchor" if c == 0 else None,
                )
            ax.grid(True, alpha=0.15)
            ax.set_title(f"{name} | var {var_ids[c]}", fontsize=8)
            if c == 0:
                ax.legend(fontsize=6, loc="upper right")
    fig.suptitle(f"{dataset} | binary vs MMPD anchor+prob | pool index {window_index}", fontsize=9)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in {".png"}:
        fig.savefig(out_path, dpi=jpeg_dpi)
        plt.close(fig)
        return str(out_path)
    return save_figure_jpg(fig, str(out_path), dpi=jpeg_dpi)


def generate_binary_vs_mmpd_anchor_prob_panels(
    *,
    dataset: str,
    out_dir: Path,
    window_indices: Sequence[int],
    y_true: np.ndarray,
    past: Optional[np.ndarray] = None,
    binary_anchor: Optional[np.ndarray] = None,
    binary_samples: Optional[np.ndarray] = None,
    mmpd_anchor: Optional[np.ndarray] = None,
    mmpd_samples: Optional[np.ndarray] = None,
    pool_indices: Optional[Sequence[int]] = None,
    variate_indices: Optional[Sequence[int]] = None,
) -> List[str]:
    """``*_`` arrays are row-aligned to ``window_indices`` (row i = window_indices[i])."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    n = len(window_indices)
    for row, win in enumerate(window_indices):
        if row >= n:
            break
        pool_id = int(pool_indices[row]) if pool_indices is not None else int(win)
        path = out_dir / f"{dataset}_win{pool_id}_binary_vs_mmpd_anchor_prob.png"
        paths.append(
            plot_binary_vs_mmpd_anchor_prob_window(
                y_true=y_true[row],
                past=None if past is None else past[row],
                binary_anchor=None if binary_anchor is None else binary_anchor[row],
                binary_samples=None if binary_samples is None else binary_samples[row],
                mmpd_anchor=None if mmpd_anchor is None else mmpd_anchor[row],
                mmpd_samples=None if mmpd_samples is None else mmpd_samples[row],
                dataset=dataset,
                window_index=pool_id,
                out_path=path,
                variate_indices=variate_indices,
            )
        )
    return paths
