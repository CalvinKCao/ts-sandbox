#!/usr/bin/env python3
"""Visualize EMA-smoothed coarse targets for fine-stage residual design.

Training intent (not implemented here — viz only):
  - Fine diffusion **target**: residual vs EMA-smoothed decoded coarse (1D)
  - Fine diffusion **conditioning**: unsmoothed coarse 2D CDF map (unchanged)

EMA is causal with ``s[0]`` seeded from the last overlap past value to avoid a
jump at the lookback→forecast boundary.

Outputs under ``reports/coarse_ema_fine_residual/``:
  {dataset}_win{idx}_var{vi}_coarse_ema_sweep.png
  {dataset}_win{idx}_var{vi}_residual_compare.png
  {dataset}_win{idx}_overview.png

Example:
  python utils/visualize_coarse_ema_fine_residual.py
  python utils/visualize_coarse_ema_fine_residual.py --datasets ETTh1,exchange_rate --alphas 0.08,0.15,0.25,0.4
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

from models.diffusion_tsf.coarse_ema import (
    causal_ema_with_past_seed,
    fine_residual_vs_smoothed_coarse,
)
from models.diffusion_tsf.pipeline import load_experiment_config
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.preprocessing import TimeSeriesTo2D
from models.diffusion_tsf.train_multivariate_pipeline import get_dataset_n_cols, load_dataset

DEFAULT_DATASETS = ["ETTh1", "exchange_rate", "weather"]
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "coarse_ema_fine_residual"
DEFAULT_ALPHAS = [0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.70, 0.90]
DEFAULT_CONFIG = "binary_anchor_ar.yaml"


def _decode_coarse_1d(
    coarse_map: torch.Tensor,
    *,
    to_2d: TimeSeriesTo2D,
) -> torch.Tensor:
    if coarse_map.dim() == 3:
        coarse_map = coarse_map.unsqueeze(0)
    b, v, _, _ = coarse_map.shape
    flat = coarse_map.reshape(b * v, 1, coarse_map.shape[-2], coarse_map.shape[-1])
    out = to_2d._decode_occupancy_in_range(flat, value_range=to_2d.max_scale, cdf_decoder="mean")
    return out.reshape(b, v, -1)


def _core_slice(x: np.ndarray, k_overlap: int) -> np.ndarray:
    if k_overlap <= 0:
        return x
    if x.shape[-1] <= k_overlap:
        return x
    return x[..., k_overlap:]


def _load_window(
    dataset: str,
    state: PipelineState,
    window_idx: int,
):
    variate_indices = list(range(int(state.n_variates)))
    train_ds, _, _, _ = load_dataset(
        dataset,
        variate_indices,
        stride=1,
        lookback=int(state.lookback_length),
        horizon=int(state.forecast_length),
        lookback_overlap=int(state.lookback_overlap),
    )
    if window_idx < 0:
        window_idx = len(train_ds) // 2
    window_idx = int(window_idx) % len(train_ds)
    past, future = train_ds[window_idx]
    if past.dim() == 1:
        past = past.unsqueeze(0)
        future = future.unsqueeze(0)
    return window_idx, past, future


def _window_norm_pair(past: torch.Tensor, future: torch.Tensor, state: PipelineState):
    p = past.unsqueeze(0)
    f = future.unsqueeze(0)
    if state.window_norm_center == "last":
        center = p[..., -1:]
    elif state.window_norm_center == "mean":
        center = p.mean(dim=-1, keepdim=True)
    else:
        raise ValueError(f"unknown window_norm_center {state.window_norm_center!r}")
    std = p.std(dim=-1, keepdim=True).clamp_min(float(state.window_norm_std_floor))
    return ((p - center) / std)[0], ((f - center) / std)[0]


def _plot_coarse_ema_sweep(
    *,
    t: np.ndarray,
    gt_core: np.ndarray,
    coarse_core: np.ndarray,
    smooth_by_alpha: dict[float, np.ndarray],
    output_path: Path,
    dataset: str,
    var_idx: int,
    window_idx: int,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, constrained_layout=True)
    ax0, ax1 = axes
    ax0.plot(t, gt_core, color="#2196F3", linewidth=1.6, label="GT future (core)")
    ax0.plot(t, coarse_core, color="#FF9800", linewidth=1.2, linestyle="--", label="Decoded coarse (raw, jagged)")
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(smooth_by_alpha)))
    for (alpha, smooth), color in zip(sorted(smooth_by_alpha.items()), colors):
        ax0.plot(t, smooth, linewidth=1.0, color=color, label=f"EMA α={alpha:.2f}")
    ax0.set_ylabel("window-norm")
    ax0.set_title(
        f"{dataset} win={window_idx} var={var_idx}\n"
        "Fine target idea: residual vs smoothed coarse; conditioning stays raw coarse 2D"
    )
    ax0.legend(fontsize=7, ncol=2, loc="upper right")
    ax0.grid(True, alpha=0.15)

    raw_res = gt_core - coarse_core
    ax1.plot(t, raw_res, color="#9E9E9E", linewidth=1.1, linestyle=":", label="GT − raw coarse (old-ish)")
    for (alpha, smooth), color in zip(sorted(smooth_by_alpha.items()), colors):
        ax1.plot(t, gt_core - smooth, linewidth=1.0, color=color, label=f"GT − EMA α={alpha:.2f}")
    ax1.axhline(0.0, color="black", linewidth=0.6, alpha=0.35)
    ax1.set_xlabel("future step (core horizon)")
    ax1.set_ylabel("residual")
    ax1.legend(fontsize=7, ncol=2, loc="upper right")
    ax1.grid(True, alpha=0.15)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, format="jpg", bbox_inches="tight")
    plt.close(fig)


def _plot_residual_compare(
    *,
    t: np.ndarray,
    gt_core: np.ndarray,
    coarse_core: np.ndarray,
    alphas: Sequence[float],
    past_tail: np.ndarray,
    output_path: Path,
    dataset: str,
    var_idx: int,
) -> None:
    n = len(alphas) + 1
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.0 * nrows), squeeze=False, constrained_layout=True)
    panels = [("raw coarse", coarse_core, gt_core - coarse_core)]
    for alpha in alphas:
        smooth = causal_ema_with_past_seed(past_tail, coarse_core, alpha)
        panels.append((f"EMA α={alpha:.2f}", smooth, gt_core - smooth))

    for ax, (title, baseline, residual) in zip(axes.ravel(), panels):
        ax.plot(t, gt_core, color="#2196F3", linewidth=1.2, label="GT")
        ax.plot(t, baseline, color="#FF9800", linewidth=1.0, label="baseline")
        ax2 = ax.twinx()
        ax2.plot(t, residual, color="#E91E63", linewidth=0.9, alpha=0.85, label="residual")
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.12)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc="upper right")
    for ax in axes.ravel()[len(panels) :]:
        ax.axis("off")
    fig.suptitle(f"{dataset} var={var_idx}: GT / baseline / residual (fine target)", fontsize=11)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, format="jpg", bbox_inches="tight")
    plt.close(fig)


def _plot_overview(
    *,
    t: np.ndarray,
    gt_core: np.ndarray,
    coarse_core: np.ndarray,
    smooth_pick: np.ndarray,
    alpha_pick: float,
    coarse_map: np.ndarray,
    output_path: Path,
    dataset: str,
    window_idx: int,
) -> None:
    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])
    ax_ts = fig.add_subplot(gs[0, :])
    ax_ts.plot(t, gt_core, label="GT core", color="#2196F3", linewidth=1.5)
    ax_ts.plot(t, coarse_core, label="decoded coarse (conditioning source)", color="#FF9800", linewidth=1.1, linestyle="--")
    ax_ts.plot(t, smooth_pick, label=f"EMA coarse α={alpha_pick:.2f} (fine target baseline)", color="#4CAF50", linewidth=1.2)
    ax_ts.fill_between(t, coarse_core, smooth_pick, color="#4CAF50", alpha=0.12)
    ax_ts.set_title(f"{dataset} win={window_idx}: decomposition overview")
    ax_ts.set_ylabel("window-norm")
    ax_ts.legend(fontsize=8, loc="upper right")
    ax_ts.grid(True, alpha=0.15)

    ax_res_old = fig.add_subplot(gs[1, 0])
    ax_res_new = fig.add_subplot(gs[1, 1])
    ax_res_old.plot(t, gt_core - coarse_core, color="#757575", linewidth=1.1)
    ax_res_old.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax_res_old.set_title("Old fine residual: GT − raw coarse")
    ax_res_old.grid(True, alpha=0.12)
    ax_res_new.plot(t, gt_core - smooth_pick, color="#E91E63", linewidth=1.1)
    ax_res_new.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax_res_new.set_title(f"Proposed fine residual: GT − EMA coarse (α={alpha_pick:.2f})")
    ax_res_new.grid(True, alpha=0.12)
    for ax in (ax_res_old, ax_res_new):
        ax.set_xlabel("future step")

    ax_inset = ax_ts.inset_axes([0.02, 0.08, 0.22, 0.55])
    im = ax_inset.imshow(coarse_map, aspect="auto", origin="lower", cmap="viridis")
    ax_inset.set_title("cond: coarse 2D", fontsize=7)
    ax_inset.set_xlabel("t", fontsize=6)
    ax_inset.set_ylabel("bin", fontsize=6)
    fig.colorbar(im, ax=ax_inset, fraction=0.08, pad=0.02)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, format="jpg", bbox_inches="tight")
    plt.close(fig)


def run_for_dataset(
    dataset: str,
    *,
    state: PipelineState,
    output_dir: Path,
    alphas: Sequence[float],
    window_idx: Optional[int],
    max_variates: int,
    coarse_height: int,
    fine_height: int,
) -> List[Path]:
    policy = (state.data_subset or {}).get("max_variates_by_dataset") or {}
    cap = int(policy.get(dataset, state.n_variates))
    state.dataset = dataset
    state.n_variates = min(cap, int(get_dataset_n_cols(dataset)))
    window_idx, past, future = _load_window(dataset, state, window_idx if window_idx is not None else -1)
    past_norm, future_norm = _window_norm_pair(past, future, state)
    k = int(state.lookback_overlap)

    to_2d = TimeSeriesTo2D(height=coarse_height, max_scale=float(state.max_scale))
    future_b = future_norm.unsqueeze(0)
    coarse_map, fine_map = to_2d.encode_dual_heights(
        future_b,
        coarse_height=coarse_height,
        fine_height=fine_height,
    )
    coarse_1d = _decode_coarse_1d(coarse_map, to_2d=to_2d)[0].numpy()
    gt = future_norm.numpy()
    if k > 0:
        past_seed = past_norm[:, k - 1].numpy()
    else:
        past_seed = past_norm[:, -1].numpy()

    gt_core = _core_slice(gt, k)
    coarse_core = _core_slice(coarse_1d, k)

    t = np.arange(gt_core.shape[-1])
    saved: List[Path] = []
    n_vars = min(max_variates, gt_core.shape[0])

    for vi in range(n_vars):
        smooth_by_alpha = {}
        for alpha in alphas:
            _, smooth, _ = fine_residual_vs_smoothed_coarse(
                gt_core[vi],
                coarse_core[vi],
                past_tail=np.asarray(past_seed[vi]),
                alpha=alpha,
            )
            smooth_by_alpha[float(alpha)] = smooth

        sweep_path = output_dir / f"{dataset}_win{window_idx}_var{vi}_coarse_ema_sweep.jpg"
        _plot_coarse_ema_sweep(
            t=t,
            gt_core=gt_core[vi],
            coarse_core=coarse_core[vi],
            smooth_by_alpha=smooth_by_alpha,
            output_path=sweep_path,
            dataset=dataset,
            var_idx=vi,
            window_idx=window_idx,
        )
        saved.append(sweep_path)

        compare_path = output_dir / f"{dataset}_win{window_idx}_var{vi}_residual_compare.jpg"
        _plot_residual_compare(
            t=t,
            gt_core=gt_core[vi],
            coarse_core=coarse_core[vi],
            alphas=alphas,
            past_tail=np.asarray(past_seed[vi]),
            output_path=compare_path,
            dataset=dataset,
            var_idx=vi,
        )
        saved.append(compare_path)

        alpha_pick = float(alphas[len(alphas) // 2])
        overview_path = output_dir / f"{dataset}_win{window_idx}_var{vi}_overview.jpg"
        _plot_overview(
            t=t,
            gt_core=gt_core[vi],
            coarse_core=coarse_core[vi],
            smooth_pick=smooth_by_alpha[alpha_pick],
            alpha_pick=alpha_pick,
            coarse_map=coarse_map[0, vi].detach().cpu().numpy(),
            output_path=overview_path,
            dataset=dataset,
            window_idx=window_idx,
        )
        saved.append(overview_path)

    return saved


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--alphas", default=",".join(str(a) for a in DEFAULT_ALPHAS))
    parser.add_argument("--window-idx", type=int, default=None)
    parser.add_argument("--max-variates", type=int, default=3)
    args = parser.parse_args(argv)

    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]

    cfg = load_experiment_config(str(REPO_ROOT / "configs" / args.config))
    state = PipelineState.from_config(cfg)
    coarse_h = int(state.coarse_image_height)
    fine_h = int(state.fine_image_height)

    all_saved: List[Path] = []
    for dataset in datasets:
        paths = run_for_dataset(
            dataset,
            state=state,
            output_dir=args.output_dir,
            alphas=alphas,
            window_idx=args.window_idx,
            max_variates=args.max_variates,
            coarse_height=coarse_h,
            fine_height=fine_h,
        )
        all_saved.extend(paths)
        print(f"{dataset}: wrote {len(paths)} figures -> {args.output_dir}")

    print(f"done ({len(all_saved)} total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
