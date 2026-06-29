#!/usr/bin/env python3
"""Sweep high-frequency % cutoffs (50% down to 5%) on real dataset windows.

Also compares raw vs mirror-pad FFT edge handling (zoomed ends).

Outputs under reports/fourier_cutoff_percent_sweep/:
  {dataset}_{variate}_high_pct_sweep.png
  {dataset}_{variate}_edge_mode_compare.png  (with --edge-compare)

Example:
  python utils/visualize_fourier_cutoff_percent_sweep.py
  python utils/visualize_fourier_cutoff_percent_sweep.py --datasets ETTh2,ETTm1,illness --max-variates 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.fourier_frequency import (
    cutoff_bin_for_high_percent,
    fft_frequency_bins,
    fourier_frequency_split_np,
    rle_compress_1d,
)
from models.diffusion_tsf.pipeline import load_experiment_config
from models.diffusion_tsf.pipeline.fourier_frequency_calibration import _window_normalize
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.train_multivariate_pipeline import (
    _dataset_variate_names,
    get_dataset_n_cols,
    load_dataset,
    _resolve_registry_path,
)

DEFAULT_DATASETS = ["ETTh1", "exchange_rate", "weather"]
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "fourier_cutoff_percent_sweep"
DEFAULT_HIGH_PCTS = [50, 55, 60, 65, 70, 75, 80, 85, 90]
DEFAULT_MAX_VARIATES = 3
END_ZOOM = 18


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s).strip("_") or "var"


def _window_norm_future(past: np.ndarray, future: np.ndarray, state: PipelineState) -> np.ndarray:
    p = torch.from_numpy(past[None, None])
    f = torch.from_numpy(future[None, None])
    return _window_normalize(
        p, f,
        center_mode=state.window_norm_center,
        std_floor=state.window_norm_std_floor,
    )[0, 0].numpy()


def _pick_variate_indices(n_variates: int, *, max_variates: int, explicit: Optional[Sequence[int]]) -> List[int]:
    if explicit:
        return [vi for vi in explicit if 0 <= vi < n_variates]
    if n_variates <= max_variates:
        return list(range(n_variates))
    if max_variates == 1:
        return [n_variates // 2]
    return [int(round(i * (n_variates - 1) / (max_variates - 1))) for i in range(max_variates)]


def _load_example_window(dataset: str, state: PipelineState, window_idx: Optional[int] = None):
    path, date_col = _resolve_registry_path(dataset)
    names = _dataset_variate_names(path, date_col, get_dataset_n_cols(dataset))
    train_ds, _, _, _ = load_dataset(dataset, stride=32)
    wi = min(50 if window_idx is None else window_idx, len(train_ds) - 1)
    past_t, future_t = train_ds[wi]
    past, future = past_t.numpy(), future_t.numpy()
    return past, future, names, wi


def _example_series(
    past: np.ndarray,
    future: np.ndarray,
    variate_idx: int,
    names: Sequence[str],
    state: PipelineState,
) -> tuple[np.ndarray, str, int]:
    x = _window_norm_future(past[variate_idx], future[variate_idx], state)
    name = names[variate_idx] if variate_idx < len(names) else f"v{variate_idx}"
    return x, name, variate_idx


def _plot_pct_sweep(
    x: np.ndarray,
    *,
    dataset: str,
    variate_name: str,
    window_idx: int,
    n_bins: int,
    high_pcts: Sequence[int],
    flatline_atol: float,
    edge_mode: str,
    out_path: Path,
) -> None:
    n = len(high_pcts)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.4 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(
        f"{dataset} | {variate_name} win {window_idx} | high-band % sweep "
        f"(edge_mode={edge_mode}, n_bins={n_bins})",
        fontsize=11,
    )
    t = np.arange(x.size)
    for ax, pct in zip(axes, high_pcts):
        k = cutoff_bin_for_high_percent(n_bins, pct)
        low, high = fourier_frequency_split_np(
            x, cutoff_bin=k, flatline_atol=flatline_atol, edge_mode=edge_mode,
        )
        recon = low + high
        err = float(np.max(np.abs(recon - x)))
        ax.plot(t, x, color="black", lw=1.0, alpha=0.45, label="orig")
        ax.plot(t, low, color="tab:blue", lw=1.0, label="low")
        ax.plot(t, high, color="tab:red", lw=0.85, label="high")
        ax.set_ylabel(f"{pct}% hi")
        ax.set_title(f"k={k}/{n_bins} | std(high)={np.std(high):.3f} | max recon err {err:.1e}")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("time step")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_edge_compare(
    x: np.ndarray,
    *,
    dataset: str,
    variate_name: str,
    high_pct: int,
    n_bins: int,
    flatline_atol: float,
    out_path: Path,
) -> None:
    k = cutoff_bin_for_high_percent(n_bins, high_pct)
    modes = [("none", "raw periodic rFFT"), ("mirror_pad", "mirror pad 25%")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex="col")
    fig.suptitle(
        f"{dataset} | {variate_name} | {high_pct}% high (k={k}) | end zoom ({END_ZOOM} steps)",
        fontsize=11,
    )
    t = np.arange(x.size)
    left = slice(0, END_ZOOM)
    right = slice(-END_ZOOM, None)

    for col, (mode, label) in enumerate(modes):
        low, high = fourier_frequency_split_np(
            x, cutoff_bin=k, flatline_atol=flatline_atol, edge_mode=mode,
        )
        recon = low + high
        err = float(np.max(np.abs(recon - x)))

        ax_l = axes[0, col]
        ax_l.plot(t[left], x[left], "k-", lw=1.2, label="orig")
        ax_l.plot(t[left], low[left], color="tab:blue", lw=1.0, label="low")
        ax_l.plot(t[left], high[left], color="tab:red", lw=0.9, label="high")
        ax_l.set_title(f"{label} | start | err {err:.1e}")
        ax_l.legend(fontsize=7)
        ax_l.grid(alpha=0.25)

        ax_r = axes[1, col]
        ax_r.plot(t[right], x[right], "k-", lw=1.2, label="orig")
        ax_r.plot(t[right], low[right], color="tab:blue", lw=1.0, label="low")
        ax_r.plot(t[right], high[right], color="tab:red", lw=0.9, label="high")
        ax_r.set_title(f"{label} | end")
        ax_r.legend(fontsize=7)
        ax_r.grid(alpha=0.25)

    for ax in axes[1, :]:
        ax.set_xlabel("time step")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--high-pcts", default=",".join(str(p) for p in DEFAULT_HIGH_PCTS))
    parser.add_argument("--edge-mode", default="mirror_pad", choices=["none", "mirror_pad", "tukey"])
    parser.add_argument(
        "--max-variates",
        type=int,
        default=DEFAULT_MAX_VARIATES,
        help="Evenly spaced variate indices per dataset (ignored if --variate-indices set)",
    )
    parser.add_argument(
        "--variate-indices",
        default="",
        help="Comma-separated variate indices (overrides --max-variates)",
    )
    parser.add_argument("--window-idx", type=int, default=None, help="Train window index (default: 50)")
    parser.add_argument("--edge-compare", action="store_true", help="Also write edge-mode comparison PNGs")
    args = parser.parse_args(argv)

    high_pcts = [int(x) for x in args.high_pcts.split(",") if x.strip()]
    explicit_vi = [int(x) for x in args.variate_indices.split(",") if x.strip()]
    cfg_path = str(REPO_ROOT / "configs/binary_anchor_stationary_flat_subsets_grad_accum_150_fourier_freq.yaml")

    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        state = PipelineState.from_config(load_experiment_config(cfg_path, {"dataset": ds}))
        past, future, names, wi = _load_example_window(ds, state, args.window_idx)
        variate_indices = _pick_variate_indices(
            past.shape[0],
            max_variates=max(1, args.max_variates),
            explicit=explicit_vi or None,
        )

        for vi in variate_indices:
            x, vname, _ = _example_series(past, future, vi, names, state)
            comp, _ = rle_compress_1d(x, float(state.fourier_flatline_atol))
            n_bins = fft_frequency_bins(len(comp))
            tag = _safe_name(vname)

            sweep_path = args.output_dir / f"{ds.lower()}_{tag}_high_pct_sweep.png"
            _plot_pct_sweep(
                x,
                dataset=ds,
                variate_name=vname,
                window_idx=wi,
                n_bins=n_bins,
                high_pcts=high_pcts,
                flatline_atol=float(state.fourier_flatline_atol),
                edge_mode=args.edge_mode,
                out_path=sweep_path,
            )
            print(sweep_path)

            if args.edge_compare:
                edge_path = args.output_dir / f"{ds.lower()}_{tag}_edge_mode_compare.png"
                _plot_edge_compare(
                    x,
                    dataset=ds,
                    variate_name=vname,
                    high_pct=30,
                    n_bins=n_bins,
                    flatline_atol=float(state.fourier_flatline_atol),
                    out_path=edge_path,
                )
                print(edge_path)

    print(f"Done → {args.output_dir}")


if __name__ == "__main__":
    main()
