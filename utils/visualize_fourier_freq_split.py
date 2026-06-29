#!/usr/bin/env python3
"""Visualize per-variate Fourier splits at the configured high-band %.

Outputs under reports/fourier_freq_split_viz/:
  {dataset}_per_variate_splits.png
  {dataset}_per_variate_splits_flatline.png  (when found)
  summary.json

Example:
  python utils/visualize_fourier_freq_split.py
  python utils/visualize_fourier_freq_split.py --datasets ETTh1,exchange_rate,weather
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
from models.diffusion_tsf.train_multivariate_pipeline import (
    _dataset_variate_names,
    _resolve_registry_path,
    get_dataset_n_cols,
    load_dataset,
)

DEFAULT_DATASETS = ["ETTh1", "exchange_rate", "weather"]
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "fourier_freq_split_viz"


@dataclass
class WindowSample:
    dataset: str
    window_idx: int
    variate_idx: int
    variate_name: str
    future_norm: np.ndarray
    compression_ratio: float


def _window_normalize_np(past: np.ndarray, future: np.ndarray, center_mode: str, std_floor: float) -> np.ndarray:
    p = torch.from_numpy(past[None, None])
    f = torch.from_numpy(future[None, None])
    out = _window_normalize(p, f, center_mode=center_mode, std_floor=std_floor)
    return out[0, 0].numpy()


def _collect_samples(
    dataset: str,
    *,
    max_windows: int,
    center_mode: str,
    std_floor: float,
    flatline_atol: float,
) -> Tuple[List[WindowSample], List[str]]:
    path, date_col = _resolve_registry_path(dataset)
    n_cols = get_dataset_n_cols(dataset)
    names = _dataset_variate_names(path, date_col, n_cols)
    train_ds, _, _, _ = load_dataset(dataset, stride=32)
    samples: List[WindowSample] = []
    for wi in range(min(max_windows, len(train_ds))):
        past_t, future_t = train_ds[wi]
        past = past_t.numpy()
        future = future_t.numpy()
        for vi in range(past.shape[0]):
            fut_norm = _window_normalize_np(past[vi], future[vi], center_mode, std_floor)
            comp, _ = rle_compress_1d(fut_norm, flatline_atol)
            ratio = len(fut_norm) / max(1, len(comp))
            samples.append(
                WindowSample(
                    dataset=dataset,
                    window_idx=wi,
                    variate_idx=vi,
                    variate_name=names[vi] if vi < len(names) else f"v{vi}",
                    future_norm=fut_norm,
                    compression_ratio=ratio,
                )
            )
    return samples, names


def _pick_flatline_sample(samples: Sequence[WindowSample]) -> Optional[WindowSample]:
    flat = [s for s in samples if s.compression_ratio >= 2.0]
    return max(flat, key=lambda s: s.compression_ratio) if flat else None


def _pick_dynamic_sample(samples: Sequence[WindowSample]) -> WindowSample:
    return min(samples, key=lambda s: s.compression_ratio)


def _load_state(dataset: str) -> PipelineState:
    cfg_path = str(REPO_ROOT / "configs/binary_anchor_stationary_flat_subsets_grad_accum_150_fourier_freq.yaml")
    cfg = load_experiment_config(cfg_path, {"dataset": dataset})
    return PipelineState.from_config(cfg)


def _split_info(state: PipelineState, width: int, n_vars: int) -> Dict:
    pct = float(state.fourier_high_freq_percent)
    n_bins_raw = fft_frequency_bins(width)
    k_raw = prior_cutoff_bin(n_bins_raw, pct)
    high_pct = 100.0 * (n_bins_raw - k_raw) / float(n_bins_raw)
    low_pct = 100.0 - high_pct
    return {
        "dataset": state.dataset,
        "fourier_flatline_atol": float(state.fourier_flatline_atol),
        "fourier_rfft_bins": n_bins_raw,
        "high_freq_percent": pct,
        "high_band_percent": high_pct,
        "low_band_percent": low_pct,
        "cutoffs_per_variate": "per_compressed_series",
    }


def _plot_dataset_panel(
    dataset: str,
    samples: Sequence[WindowSample],
    info: Dict,
    out_path: Path,
    *,
    title_suffix: str,
    variate_indices: Optional[Sequence[int]] = None,
) -> None:
    flatline_atol = float(info["fourier_flatline_atol"])
    high_pct = float(info["high_band_percent"])
    low_pct = float(info["low_band_percent"])
    pct = float(info["high_freq_percent"])

    if variate_indices is None:
        max_vi = max((s.variate_idx for s in samples), default=0)
        variate_indices = list(range(min(3, max_vi + 1)))

    n_rows = len(variate_indices)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.8 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]
    fig.suptitle(
        f"{dataset} | {high_pct:.0f}% high / {low_pct:.0f}% low (per variate) | {title_suffix}",
        fontsize=11,
    )

    for ax, vi in zip(axes, variate_indices):
        sample = next(s for s in samples if s.variate_idx == vi)
        x = sample.future_norm
        low, high = fourier_frequency_split_np(
            x, high_freq_percent=pct, flatline_atol=flatline_atol,
        )
        comp, _ = rle_compress_1d(x, flatline_atol)
        n_bins = fft_frequency_bins(comp.size)
        k = prior_cutoff_bin(n_bins, pct)
        recon = low + high
        t = np.arange(x.size)
        high_std = float(np.std(high))

        ax.plot(t, x, color="black", lw=1.0, alpha=0.55, label="orig")
        ax.plot(t, low, color="tab:blue", lw=1.1, label="low")
        ax.plot(t, high, color="tab:red", lw=0.9, label="high")
        ax.set_ylabel(sample.variate_name[:12])
        ax.set_title(
            f"win {sample.window_idx} | k={k}/{n_bins} ({high_pct:.0f}% high) | std(high)={high_std:.3f}",
            fontsize=9,
        )
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.25)
        err = float(np.max(np.abs(recon - x)))
        ax.text(0.01, 0.05, f"max recon err {err:.1e}", transform=ax.transAxes, fontsize=8)

    axes[-1].set_xlabel("time step")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-windows", type=int, default=128)
    args = parser.parse_args(argv)

    summary: List[Dict] = []
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        state = _load_state(ds)
        samples, _ = _collect_samples(
            ds,
            max_windows=args.max_windows,
            center_mode=state.window_norm_center,
            std_floor=state.window_norm_std_floor,
            flatline_atol=float(state.fourier_flatline_atol),
        )
        width = samples[0].future_norm.size if samples else state.forecast_length
        n_vars = max((s.variate_idx for s in samples), default=0) + 1
        info = _split_info(state, width, n_vars)
        flat = _pick_flatline_sample(samples)

        out_dyn = args.output_dir / f"{ds.lower()}_per_variate_splits.png"
        _plot_dataset_panel(
            ds, samples, info, out_dyn,
            title_suffix="dynamic-ish window (show first 3 variates)",
            variate_indices=[0, 1, 2] if n_vars >= 3 else [0],
        )

        entry = {
            "dataset": ds,
            "dynamic_plot": str(out_dyn),
            "cutoffs_per_variate": info["cutoffs_per_variate"],
        }
        if flat is not None:
            out_flat = args.output_dir / f"{ds.lower()}_per_variate_splits_flatline.png"
            _plot_dataset_panel(
                ds, samples, info, out_flat,
                title_suffix=f"flatline variate {flat.variate_name} win {flat.window_idx}",
                variate_indices=[flat.variate_idx],
            )
            entry["flatline_plot"] = str(out_flat)
        summary.append(entry)

    summary_path = args.output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote plots to {args.output_dir}")
    for item in summary:
        for k, v in item.items():
            if k.endswith("_plot") and v:
                print(f"  {v}")


if __name__ == "__main__":
    main()
