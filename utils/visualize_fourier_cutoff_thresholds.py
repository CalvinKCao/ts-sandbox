#!/usr/bin/env python3
"""Sweep Fourier auto-cutoff CV/spread thresholds and visualize selected high-band %.

Outputs under reports/fourier_cutoff_threshold_sweep/:
  {dataset}_threshold_heatmap.png   — % high freq for each (CV, spread) gate
  {dataset}_example_splits.png        — same window at several cutoffs
  summary.json

Example:
  python utils/visualize_fourier_cutoff_thresholds.py
  python utils/visualize_fourier_cutoff_thresholds.py --datasets ETTh1,weather
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.fourier_frequency import (
    fft_frequency_bins,
    fourier_frequency_split_np,
    prior_cutoff_bin,
    rle_compress_1d,
)
from models.diffusion_tsf.pipeline import load_experiment_config
from models.diffusion_tsf.pipeline.fourier_frequency_calibration import _window_normalize
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.train_multivariate_pipeline import load_dataset

DEFAULT_DATASETS = ["ETTh1", "exchange_rate", "weather"]
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "fourier_cutoff_threshold_sweep"

# Current defaults in config vs looser sweeps
DEFAULT_CV_THRESHOLDS = [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
DEFAULT_SPREAD_THRESHOLDS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0]


@dataclass
class CandidateStats:
    n_bins: int
    prior_cutoff: int
    prior_high_pct: float
    cutoff_metrics: Dict[int, Dict[str, float]]  # k -> {cv, spread, high_pct}


def _gather_candidate_stats(
    dataset: str,
    *,
    max_windows: int,
    center_mode: str,
    std_floor: float,
    flatline_atol: float,
    prior_pct: float,
) -> CandidateStats:
    train_ds, _, _, _ = load_dataset(dataset, stride=32)
    if len(train_ds) > max_windows:
        train_ds = Subset(train_ds, list(range(max_windows)))
    loader = DataLoader(train_ds, batch_size=min(64, max(1, len(train_ds))), shuffle=False, num_workers=0)

    candidate_vars: Dict[int, List[torch.Tensor]] = {}
    n_bins = 0
    for past, future in loader:
        future_norm = _window_normalize(
            past, future, center_mode=center_mode, std_floor=std_floor,
        )
        if n_bins == 0:
            sample = future_norm[0, 0].detach().cpu().numpy()
            comp, _ = rle_compress_1d(sample, flatline_atol)
            n_bins = fft_frequency_bins(int(comp.size))
            candidate_vars = {k: [] for k in range(1, n_bins)}

        for k in candidate_vars:
            highs = []
            for bi in range(future_norm.shape[0]):
                for vi in range(future_norm.shape[1]):
                    series = future_norm[bi, vi].detach().cpu().numpy()
                    _low, high = fourier_frequency_split_np(
                        series, cutoff_bin=k, flatline_atol=flatline_atol,
                    )
                    highs.append(torch.from_numpy(high))
            high_t = torch.stack(highs, dim=0)
            candidate_vars[k].append(high_t.var(dim=-1, unbiased=False).reshape(-1).cpu())

    eps = 1e-12
    metrics: Dict[int, Dict[str, float]] = {}
    for k in range(1, n_bins):
        vars_k = torch.cat(candidate_vars[k]).numpy()
        vars_k = np.maximum(vars_k, eps)
        log_vars = np.log(vars_k)
        metrics[k] = {
            "cv": float(vars_k.std() / max(float(vars_k.mean()), eps)),
            "spread": float(np.quantile(log_vars, 0.95) - np.quantile(log_vars, 0.05)),
            "high_pct": (n_bins - k) / float(n_bins),
        }

    prior_cutoff = prior_cutoff_bin(n_bins, prior_pct)
    return CandidateStats(
        n_bins=n_bins,
        prior_cutoff=prior_cutoff,
        prior_high_pct=(n_bins - prior_cutoff) / float(n_bins),
        cutoff_metrics=metrics,
    )


def _select_cutoff(
    stats: CandidateStats,
    cv_threshold: float,
    spread_threshold: float,
    *,
    pick: str,
) -> Tuple[int, str]:
    """pick='min' → most high-band (smallest k); pick='max' → most low-band among valid."""
    valid = [
        k for k, m in stats.cutoff_metrics.items()
        if m["cv"] <= cv_threshold and m["spread"] <= spread_threshold
    ]
    if not valid:
        return stats.prior_cutoff, "prior_fallback"
    if pick == "min":
        return min(valid), "min_valid"
    return max(valid), "max_valid"


def _plot_heatmaps(
    dataset: str,
    stats: CandidateStats,
    cv_thresholds: Sequence[float],
    spread_thresholds: Sequence[float],
    out_path: Path,
) -> Dict:
    n_cv, n_sp = len(cv_thresholds), len(spread_thresholds)
    pct_min = np.zeros((n_sp, n_cv))
    pct_max = np.zeros((n_sp, n_cv))
    rows = []

    for si, spread_th in enumerate(spread_thresholds):
        for ci, cv_th in enumerate(cv_thresholds):
            k_min, how_min = _select_cutoff(stats, cv_th, spread_th, pick="min")
            k_max, how_max = _select_cutoff(stats, cv_th, spread_th, pick="max")
            pct_min[si, ci] = stats.cutoff_metrics[k_min]["high_pct"] * 100
            pct_max[si, ci] = stats.cutoff_metrics[k_max]["high_pct"] * 100
            rows.append({
                "dataset": dataset,
                "cv_threshold": cv_th,
                "spread_threshold": spread_th,
                "min_valid_cutoff_bin": k_min,
                "min_valid_high_pct": pct_min[si, ci],
                "min_valid_selection": how_min,
                "max_valid_cutoff_bin": k_max,
                "max_valid_high_pct": pct_max[si, ci],
                "max_valid_selection": how_max,
            })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        f"{dataset} | n_bins={stats.n_bins} | prior={stats.prior_high_pct*100:.0f}% high "
        f"(cutoff {stats.prior_cutoff}) | current defaults CV≤1.0 spread≤2.0",
        fontsize=11,
    )
    for ax, data, title in (
        (axes[0], pct_min, "pick min(valid k) → MORE high/fine band"),
        (axes[1], pct_max, "pick max(valid k) → LESS high/fine band (current code)"),
    ):
        im = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            cmap="YlOrRd",
            vmin=0,
            vmax=100,
        )
        ax.set_xticks(range(n_cv))
        ax.set_xticklabels([f"{x:g}" for x in cv_thresholds], rotation=45, ha="right")
        ax.set_yticks(range(n_sp))
        ax.set_yticklabels([f"{x:g}" for x in spread_thresholds])
        ax.set_xlabel("CV threshold (higher = looser)")
        ax.set_ylabel("log-spread threshold (higher = looser)")
        ax.set_title(title, fontsize=10)
        # mark current default gate
        if 1.0 in cv_thresholds and 2.0 in spread_thresholds:
            ci = list(cv_thresholds).index(1.0)
            si = list(spread_thresholds).index(2.0)
            ax.plot(ci, si, "k*", ms=14, label="current default")
            ax.legend(loc="upper left", fontsize=8)
        for si in range(n_sp):
            for ci in range(n_cv):
                ax.text(ci, si, f"{data[si, ci]:.0f}", ha="center", va="center", fontsize=7, color="black")

    fig.colorbar(im, ax=axes.ravel().tolist(), label="% bins in high/fine band", shrink=0.85)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return {"heatmap": str(out_path), "grid_rows": rows}


def _plot_example_splits(
    dataset: str,
    future_norm: np.ndarray,
    cutoffs: Sequence[int],
    labels: Sequence[str],
    stats: CandidateStats,
    flatline_atol: float,
    out_path: Path,
) -> None:
    t = np.arange(future_norm.size)
    n_panels = len(cutoffs) + 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2.2 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    axes[0].plot(t, future_norm, color="black", lw=1.2)
    axes[0].set_ylabel("orig")
    axes[0].set_title(f"{dataset} | example window — low/high split at different cutoffs")
    axes[0].grid(alpha=0.25)

    for ax, k, label in zip(axes[1:], cutoffs, labels):
        low, high = fourier_frequency_split_np(future_norm, cutoff_bin=k, flatline_atol=flatline_atol)
        pct = stats.cutoff_metrics[k]["high_pct"] * 100
        ax.plot(t, low, color="tab:blue", lw=1.0, label="low")
        ax.plot(t, high, color="tab:red", lw=0.9, alpha=0.9, label="high")
        ax.set_ylabel(f"k={k}")
        ax.set_title(f"{label} | {pct:.0f}% high | CV={stats.cutoff_metrics[k]['cv']:.2f} spread={stats.cutoff_metrics[k]['spread']:.2f}")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("time step")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-windows", type=int, default=256)
    parser.add_argument("--cv-thresholds", default=",".join(str(x) for x in DEFAULT_CV_THRESHOLDS))
    parser.add_argument("--spread-thresholds", default=",".join(str(x) for x in DEFAULT_SPREAD_THRESHOLDS))
    args = parser.parse_args(argv)

    cv_thresholds = [float(x) for x in args.cv_thresholds.split(",") if x.strip()]
    spread_thresholds = [float(x) for x in args.spread_thresholds.split(",") if x.strip()]

    cfg_path = str(REPO_ROOT / "configs/binary_anchor_stationary_flat_subsets_grad_accum_150_fourier_freq.yaml")
    summary: List[Dict] = []

    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        cfg = load_experiment_config(cfg_path, {"dataset": ds})
        state = PipelineState.from_config(cfg)
        flatline_atol = float(state.fourier_flatline_atol)
        prior_pct = float(state.fourier_high_freq_percent)

        stats = _gather_candidate_stats(
            ds,
            max_windows=args.max_windows,
            center_mode=state.window_norm_center,
            std_floor=state.window_norm_std_floor,
            flatline_atol=flatline_atol,
            prior_pct=prior_pct,
        )

        hm_path = args.output_dir / f"{ds.lower()}_threshold_heatmap.png"
        hm_info = _plot_heatmaps(ds, stats, cv_thresholds, spread_thresholds, hm_path)
        summary.append(hm_info)

        # Example splits: prior, current default selection, and a high-freq option
        train_ds, _, _, _ = load_dataset(ds, stride=32)
        past_t, future_t = train_ds[min(50, len(train_ds) - 1)]
        future_norm = _window_normalize(
            past_t.unsqueeze(0),
            future_t.unsqueeze(0),
            center_mode=state.window_norm_center,
            std_floor=state.window_norm_std_floor,
        )[0, 0].detach().cpu().numpy()

        k_prior = stats.prior_cutoff
        k_default_min, _ = _select_cutoff(stats, 1.0, 2.0, pick="min")
        k_default_max, _ = _select_cutoff(stats, 1.0, 2.0, pick="max")
        k_loose_min, _ = _select_cutoff(stats, 4.0, 6.0, pick="min")
        k_loose_max, _ = _select_cutoff(stats, 4.0, 6.0, pick="max")

        example_cutoffs = []
        example_labels = []
        seen = set()
        for k, label in [
            (k_loose_min, "loose gate CV≤4 spread≤6, min(valid)"),
            (k_default_min, "default gate CV≤1 spread≤2, min(valid)"),
            (k_prior, f"prior {prior_pct*100:.0f}% high (fallback)"),
            (k_default_max, "default gate, max(valid) [current code]"),
            (k_loose_max, "loose gate, max(valid)"),
        ]:
            if k not in seen:
                example_cutoffs.append(k)
                example_labels.append(label)
                seen.add(k)

        ex_path = args.output_dir / f"{ds.lower()}_example_splits.png"
        _plot_example_splits(ds, future_norm, example_cutoffs, example_labels, stats, flatline_atol, ex_path)
        summary.append({"dataset": ds, "example_splits": str(ex_path)})

        # Per-k reference curve
        ks = sorted(stats.cutoff_metrics)
        summary.append({
            "dataset": ds,
            "n_bins": stats.n_bins,
            "prior_cutoff": stats.prior_cutoff,
            "per_cutoff": [
                {"k": k, **stats.cutoff_metrics[k]} for k in ks
            ],
        })

    out_json = args.output_dir / "summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote threshold sweep to {args.output_dir}")
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        print(f"  {args.output_dir}/{ds.lower()}_threshold_heatmap.png")
        print(f"  {args.output_dir}/{ds.lower()}_example_splits.png")


if __name__ == "__main__":
    main()
