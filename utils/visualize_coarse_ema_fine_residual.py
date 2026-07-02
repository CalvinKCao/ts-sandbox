#!/usr/bin/env python3
"""Visualize flatline-preserving blur for fine-stage coarse baselines (viz only).

Proposed training (not wired yet):
  - Fine **target**: residual vs symmetric blur of collapsed coarse skeleton
  - Flatlines detected on **GT coarse+fine decode** (>=2 identical consecutive values);
    collapse uses mean coarse decode per run, blur on skeleton, restore plateaus.
  - Fine **conditioning**: unsmoothed coarse 2D CDF (unchanged)

Compared against causal EMA (shifts curve forward; smears plateaus).

Outputs under ``reports/coarse_flatline_blur_fine_residual/``:
  {dataset}_win{idx}_var{vi}_method_sweep.jpg
  {dataset}_win{idx}_var{vi}_skeleton.jpg
  {dataset}_win{idx}_var{vi}_overview.jpg
  {dataset}_win{idx}_pipeline_2d.jpg

Example:
  python utils/visualize_coarse_ema_fine_residual.py
  python utils/visualize_coarse_ema_fine_residual.py --blur-radii 4 --ema-alpha 0.25
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

from models.diffusion_tsf.coarse_ema import causal_ema_with_past_seed
from models.diffusion_tsf.coarse_flatline_blur import (
    ConstantRun,
    fine_residual_vs_flatline_blur_coarse,
    flatline_preserving_blur,
)
from models.diffusion_tsf.pipeline import load_experiment_config
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.preprocessing import TimeSeriesTo2D
from models.diffusion_tsf.train_multivariate_pipeline import get_dataset_n_cols, load_dataset

DEFAULT_DATASETS = ["ETTh1", "exchange_rate", "weather"]
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "coarse_flatline_blur_fine_residual"
DEFAULT_BLUR_RADII = [4]
DEFAULT_EMA_ALPHA = 0.25
DEFAULT_CONFIG = "binary_anchor_ar.yaml"


def _pipeline_max_scale(state: PipelineState, dataset: str) -> float:
    return float(state.max_scale_by_dataset.get(dataset, state.max_scale))


def _decode_gt_combined(
    coarse_map: torch.Tensor,
    fine_map: torch.Tensor,
    *,
    to_2d: TimeSeriesTo2D,
) -> np.ndarray:
    combined = to_2d.decode_dual(coarse_map, fine_map, squeeze_univariate=False)
    return combined[0].detach().cpu().numpy()


def _plot_pipeline_2d_maps(
    *,
    coarse_map: np.ndarray,
    fine_map: np.ndarray,
    output_path: Path,
    dataset: str,
    window_idx: int,
    max_scale: float,
    coarse_height: int,
    fine_height: int,
    n_vars: int,
) -> None:
    fine_range = max_scale / float(coarse_height)
    n_cols = min(n_vars, coarse_map.shape[0])
    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(4.0 * n_cols, 5.2),
        constrained_layout=True,
        squeeze=False,
    )
    row_labels = ("coarse 2D", "fine 2D")
    for col in range(n_cols):
        for row, data in enumerate((coarse_map, fine_map)):
            ax = axes[row, col]
            h, w = data[col].shape
            im = ax.imshow(
                data[col],
                aspect="auto",
                origin="lower",
                extent=[0, w, 0, h],
                cmap="plasma",
                vmin=0.0,
                vmax=1.0,
            )
            ax.set_title(f"{row_labels[row]} | var {col} ({h}x{w})", fontsize=8)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{dataset} win={window_idx} — pipeline 2D occupancy (config max_scale)\n"
        f"max_scale={max_scale:.2f} | coarse_H={coarse_height} fine_H={fine_height} | "
        f"coarse decode ±{max_scale:.2f} | fine residual ±{fine_range:.4f}",
        fontsize=10,
        fontweight="semibold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, format="jpg", bbox_inches="tight")
    plt.close(fig)


def _decode_coarse_1d(coarse_map: torch.Tensor, *, to_2d: TimeSeriesTo2D) -> torch.Tensor:
    if coarse_map.dim() == 3:
        coarse_map = coarse_map.unsqueeze(0)
    b, v, _, _ = coarse_map.shape
    flat = coarse_map.reshape(b * v, 1, coarse_map.shape[-2], coarse_map.shape[-1])
    out = to_2d._decode_occupancy_in_range(flat, value_range=to_2d.max_scale, cdf_decoder="mean")
    return out.reshape(b, v, -1)


def _core_slice(x: np.ndarray, k_overlap: int) -> np.ndarray:
    if k_overlap <= 0 or x.shape[-1] <= k_overlap:
        return x
    return x[..., k_overlap:]


def _load_window(dataset: str, state: PipelineState, window_idx: int):
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


def _shade_flatlines(ax, runs: Sequence[ConstantRun], *, min_flat_len: int, alpha: float = 0.18) -> None:
    for run in runs:
        if run.is_flatline(min_flat_len):
            ax.axvspan(run.start - 0.5, run.end - 0.5, color="#FFC107", alpha=alpha, lw=0)


def _plot_method_sweep(
    *,
    t: np.ndarray,
    gt_core: np.ndarray,
    coarse_core: np.ndarray,
    runs: Sequence[ConstantRun],
    ema_curve: np.ndarray,
    ema_alpha: float,
    blur_by_radius: dict[int, np.ndarray],
    output_path: Path,
    dataset: str,
    var_idx: int,
    window_idx: int,
    min_flat_len: int,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    ax0, ax1 = axes
    _shade_flatlines(ax0, runs, min_flat_len=min_flat_len)
    ax0.plot(t, gt_core, color="#2196F3", linewidth=1.6, label="GT future (core)")
    ax0.plot(
        t, coarse_core, color="#FF9800", linewidth=1.2, linestyle="--",
        label="Decoded coarse (raw, jagged)",
    )
    ax0.plot(
        t, ema_curve, color="#9C27B0", linewidth=1.1, linestyle="-.",
        label=f"EMA α={ema_alpha:.2f} (shifts / smears flats)",
    )
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(blur_by_radius)))
    for (radius, smooth), color in zip(sorted(blur_by_radius.items()), colors):
        ax0.plot(
            t, smooth, linewidth=1.1, color=color,
            label=f"flatline blur r={radius}",
        )
    ax0.set_ylabel("window-norm")
    ax0.set_title(
        f"{dataset} win={window_idx} var={var_idx}\n"
        "Yellow = flatlines from GT coarse+fine decode; blur on coarse skeleton"
    )
    ax0.legend(fontsize=7, ncol=2, loc="upper right")
    ax0.grid(True, alpha=0.15)

    ax1.plot(t, gt_core - coarse_core, color="#9E9E9E", linewidth=1.0, linestyle=":", label="GT − raw coarse")
    ax1.plot(t, gt_core - ema_curve, color="#9C27B0", linewidth=1.0, linestyle="-.", label=f"GT − EMA")
    for (radius, smooth), color in zip(sorted(blur_by_radius.items()), colors):
        ax1.plot(t, gt_core - smooth, linewidth=1.0, color=color, label=f"GT − blur r={radius}")
    ax1.axhline(0.0, color="black", linewidth=0.6, alpha=0.35)
    ax1.set_xlabel("future step (core horizon)")
    ax1.set_ylabel("fine target residual")
    ax1.legend(fontsize=7, ncol=2, loc="upper right")
    ax1.grid(True, alpha=0.15)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, format="jpg", bbox_inches="tight")
    plt.close(fig)


def _plot_skeleton(
    *,
    coarse_core: np.ndarray,
    runs: Sequence[ConstantRun],
    past_seed: float,
    blur_radius: int,
    output_path: Path,
    dataset: str,
    var_idx: int,
    min_flat_len: int,
) -> None:
    skeleton_x = np.arange(len(runs))
    skeleton_y = np.array(
        [float(np.mean(coarse_core[r.start : r.end])) for r in runs],
        dtype=np.float64,
    )
    padded = np.concatenate([[past_seed], skeleton_y])
    from models.diffusion_tsf.coarse_flatline_blur import symmetric_blur_1d

    blurred_padded = symmetric_blur_1d(padded, radius=blur_radius, kernel="gaussian")
    blurred = blurred_padded[1:]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    ax0, ax1 = axes
    t = np.arange(coarse_core.shape[-1])
    _shade_flatlines(ax0, runs, min_flat_len=min_flat_len)
    ax0.step(t, coarse_core, where="mid", color="#FF9800", linewidth=1.2, label="raw coarse decode")
    ax0.set_title(f"{dataset} var={var_idx}: GT flatlines → collapse coarse means → blur → expand")
    ax0.set_ylabel("window-norm")
    ax0.grid(True, alpha=0.12)
    ax0.legend(fontsize=8)

    ax1.plot(skeleton_x, skeleton_y, "o--", color="#FF9800", label="collapsed skeleton")
    ax1.plot(skeleton_x, blurred, "o-", color="#4CAF50", label=f"after symmetric blur (r={blur_radius})")
    for i, run in enumerate(runs):
        if run.is_flatline(min_flat_len):
            ax1.axvspan(i - 0.35, i + 0.35, color="#FFC107", alpha=0.25)
    ax1.set_xlabel("skeleton index (one point per constant run)")
    ax1.set_ylabel("representative value")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, format="jpg", bbox_inches="tight")
    plt.close(fig)


def _plot_overview(
    *,
    t: np.ndarray,
    gt_core: np.ndarray,
    coarse_core: np.ndarray,
    runs: Sequence[ConstantRun],
    blur_curve: np.ndarray,
    blur_radius: int,
    ema_curve: np.ndarray,
    ema_alpha: float,
    coarse_map: np.ndarray,
    output_path: Path,
    dataset: str,
    window_idx: int,
    min_flat_len: int,
) -> None:
    fig = plt.figure(figsize=(12, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])
    ax_ts = fig.add_subplot(gs[0, :])
    _shade_flatlines(ax_ts, runs, min_flat_len=min_flat_len)
    ax_ts.plot(t, gt_core, label="GT core", color="#2196F3", linewidth=1.5)
    ax_ts.plot(
        t, coarse_core, label="decoded coarse (conditioning)", color="#FF9800",
        linewidth=1.1, linestyle="--",
    )
    ax_ts.plot(
        t, blur_curve, label=f"flatline blur r={blur_radius} (fine target baseline)",
        color="#4CAF50", linewidth=1.2,
    )
    ax_ts.plot(
        t, ema_curve, label=f"EMA α={ema_alpha:.2f}", color="#9C27B0",
        linewidth=1.0, linestyle="-.", alpha=0.85,
    )
    ax_ts.set_title(f"{dataset} win={window_idx}: flatline-preserving blur vs EMA")
    ax_ts.set_ylabel("window-norm")
    ax_ts.legend(fontsize=8, loc="upper right")
    ax_ts.grid(True, alpha=0.15)

    ax_old = fig.add_subplot(gs[1, 0])
    ax_ema = fig.add_subplot(gs[1, 1])
    ax_old.plot(t, gt_core - coarse_core, color="#757575", linewidth=1.1)
    ax_old.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax_old.set_title("Residual: GT − raw coarse")
    ax_old.grid(True, alpha=0.12)
    ax_ema.plot(t, gt_core - blur_curve, color="#E91E63", linewidth=1.1, label="blur")
    ax_ema.plot(t, gt_core - ema_curve, color="#9C27B0", linewidth=1.0, linestyle="-.", label="EMA")
    ax_ema.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax_ema.set_title("Proposed vs EMA residual")
    ax_ema.legend(fontsize=8)
    ax_ema.grid(True, alpha=0.12)
    for ax in (ax_old, ax_ema):
        ax.set_xlabel("future step")

    ax_inset = ax_ts.inset_axes([0.02, 0.08, 0.22, 0.55])
    im = ax_inset.imshow(coarse_map, aspect="auto", origin="lower", cmap="viridis")
    ax_inset.set_title("cond: coarse 2D", fontsize=7)
    fig.colorbar(im, ax=ax_inset, fraction=0.08, pad=0.02)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, format="jpg", bbox_inches="tight")
    plt.close(fig)


def run_for_dataset(
    dataset: str,
    *,
    state: PipelineState,
    output_dir: Path,
    blur_radii: Sequence[int],
    ema_alpha: float,
    min_flat_len: int,
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
    max_scale = _pipeline_max_scale(state, dataset)

    to_2d = TimeSeriesTo2D(height=coarse_height, max_scale=max_scale)
    future_b = future_norm.unsqueeze(0)
    coarse_map, fine_map = to_2d.encode_dual_heights(
        future_b,
        coarse_height=coarse_height,
        fine_height=fine_height,
    )
    coarse_1d = _decode_coarse_1d(coarse_map, to_2d=to_2d)[0].numpy()
    gt_combined = _decode_gt_combined(coarse_map, fine_map, to_2d=to_2d)
    gt = future_norm.numpy()
    past_seed = past_norm[:, k - 1].numpy() if k > 0 else past_norm[:, -1].numpy()

    gt_core = _core_slice(gt, k)
    gt_combined_core = _core_slice(gt_combined, k)
    coarse_core = _core_slice(coarse_1d, k)
    t = np.arange(gt_core.shape[-1])
    saved: List[Path] = []
    n_vars = min(max_variates, gt_core.shape[0])

    map2d_path = output_dir / f"{dataset}_win{window_idx}_pipeline_2d.jpg"
    _plot_pipeline_2d_maps(
        coarse_map=coarse_map[0].detach().cpu().numpy(),
        fine_map=fine_map[0].detach().cpu().numpy(),
        output_path=map2d_path,
        dataset=dataset,
        window_idx=window_idx,
        max_scale=max_scale,
        coarse_height=coarse_height,
        fine_height=fine_height,
        n_vars=n_vars,
    )
    saved.append(map2d_path)

    for vi in range(n_vars):
        seed = float(past_seed[vi])
        ema_curve = causal_ema_with_past_seed(seed, coarse_core[vi], ema_alpha)
        blur_by_radius: dict[int, np.ndarray] = {}
        runs: Sequence[ConstantRun] = []
        for radius in blur_radii:
            _, smooth, _, runs = fine_residual_vs_flatline_blur_coarse(
                gt_core[vi],
                coarse_core[vi],
                gt_combined=gt_combined_core[vi],
                past_seed=seed,
                blur_radius=int(radius),
                min_flat_len=min_flat_len,
            )
            blur_by_radius[int(radius)] = smooth

        sweep_path = output_dir / f"{dataset}_win{window_idx}_var{vi}_method_sweep.jpg"
        _plot_method_sweep(
            t=t,
            gt_core=gt_core[vi],
            coarse_core=coarse_core[vi],
            runs=runs,
            ema_curve=ema_curve,
            ema_alpha=ema_alpha,
            blur_by_radius=blur_by_radius,
            output_path=sweep_path,
            dataset=dataset,
            var_idx=vi,
            window_idx=window_idx,
            min_flat_len=min_flat_len,
        )
        saved.append(sweep_path)

        pick_radius = int(blur_radii[-1])
        skeleton_path = output_dir / f"{dataset}_win{window_idx}_var{vi}_skeleton.jpg"
        _plot_skeleton(
            coarse_core=coarse_core[vi],
            runs=runs,
            past_seed=seed,
            blur_radius=pick_radius,
            output_path=skeleton_path,
            dataset=dataset,
            var_idx=vi,
            min_flat_len=min_flat_len,
        )
        saved.append(skeleton_path)

        overview_path = output_dir / f"{dataset}_win{window_idx}_var{vi}_overview.jpg"
        _plot_overview(
            t=t,
            gt_core=gt_core[vi],
            coarse_core=coarse_core[vi],
            runs=runs,
            blur_curve=blur_by_radius[pick_radius],
            blur_radius=pick_radius,
            ema_curve=ema_curve,
            ema_alpha=ema_alpha,
            coarse_map=coarse_map[0, vi].detach().cpu().numpy(),
            output_path=overview_path,
            dataset=dataset,
            window_idx=window_idx,
            min_flat_len=min_flat_len,
        )
        saved.append(overview_path)

    return saved


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--blur-radii", default=",".join(str(r) for r in DEFAULT_BLUR_RADII))
    parser.add_argument("--ema-alpha", type=float, default=DEFAULT_EMA_ALPHA)
    parser.add_argument("--min-flat-len", type=int, default=2)
    parser.add_argument("--window-idx", type=int, default=None)
    parser.add_argument("--max-variates", type=int, default=3)
    args = parser.parse_args(argv)

    blur_radii = [int(x.strip()) for x in args.blur_radii.split(",") if x.strip()]
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]

    cfg = load_experiment_config(str(REPO_ROOT / "configs" / args.config))
    state = PipelineState.from_config(cfg)

    all_saved: List[Path] = []
    for dataset in datasets:
        paths = run_for_dataset(
            dataset,
            state=state,
            output_dir=args.output_dir,
            blur_radii=blur_radii,
            ema_alpha=float(args.ema_alpha),
            min_flat_len=int(args.min_flat_len),
            window_idx=args.window_idx,
            max_variates=args.max_variates,
            coarse_height=int(state.coarse_image_height),
            fine_height=int(state.fine_image_height),
        )
        all_saved.extend(paths)
        print(f"{dataset}: wrote {len(paths)} figures -> {args.output_dir}")

    print(f"done ({len(all_saved)} total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
