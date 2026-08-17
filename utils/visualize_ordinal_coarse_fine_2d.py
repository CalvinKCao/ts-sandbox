#!/usr/bin/env python3
"""Full train-context viz: raw → z-score → ordinal → coarse/fine 2D CDF pixels."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.diffusion_tsf.config import DiffusionTSFConfig
from models.diffusion_tsf.diffusion_model import DiffusionTSF
from models.diffusion_tsf.ordinal_window_norm import (
    build_global_ladder_from_training,
    encode_with_ladder,
    ordinal_encode,
)
from models.diffusion_tsf.pipeline.config import load_experiment_config
from models.diffusion_tsf.preprocessing import TimeSeriesTo2D
from models.diffusion_tsf.train_multivariate_pipeline import (
    _load_dataset_array,
    _paper_split_borders,
    _resolve_registry_path,
    load_dataset,
)


def _plot_cdf_map(ax, cdf: np.ndarray, *, title: str, time_labels: np.ndarray | None = None):
    im = ax.imshow(
        cdf,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title(title)
    ax.set_ylabel("pixel row (value bin)")
    ax.set_xlabel("time column")
    if time_labels is not None and len(time_labels) == cdf.shape[1]:
        step = max(1, cdf.shape[1] // 8)
        ticks = np.arange(0, cdf.shape[1], step)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int(time_labels[i])) for i in ticks], fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _window_raw_span(
    *,
    window_idx: int,
    stride: int,
    lookback: int,
    horizon: int,
    overlap: int,
) -> tuple[int, int, int, int]:
    start = window_idx * stride
    past_end = start + lookback
    fut_start = start + lookback - overlap
    fut_end = start + lookback + horizon
    return start, past_end, fut_start, fut_end


def _subsample_1d(x: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return x
    return x[..., ::stride]


def _add_window_highlights(
    ax,
    *,
    hl_all_lo: float,
    hl_all_hi: float,
    hl_past_lo: float,
    hl_past_hi: float,
    hl_fut_lo: float,
    hl_fut_hi: float,
    start: int,
    past_end: int,
    fut_start: int,
    fut_end: int,
) -> None:
    ax.axvspan(hl_all_lo, hl_all_hi, color="gold", alpha=0.22, label=f"window [{start}, {fut_end})")
    ax.axvspan(hl_past_lo, hl_past_hi, color="C2", alpha=0.18, label=f"lookback [{start}, {past_end})")
    ax.axvspan(hl_fut_lo, hl_fut_hi, color="C3", alpha=0.18, label=f"target [{fut_start}, {fut_end})")
    ax.axvline(hl_all_lo, color="black", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.axvline(hl_all_hi, color="black", linestyle=":", linewidth=1.0, alpha=0.7)


def _plot_full_train_context(
    ax,
    *,
    t_axis: np.ndarray,
    series: np.ndarray,
    past_x: np.ndarray,
    past_y: np.ndarray,
    fut_x: np.ndarray,
    fut_y: np.ndarray,
    highlights: dict,
    ylabel: str,
    title: str,
    ylim: tuple[float, float] | None = None,
    line_label: str,
) -> None:
    ax.plot(t_axis, series, color="C0", linewidth=0.8, alpha=0.85, label=line_label)
    _add_window_highlights(
        ax,
        hl_all_lo=highlights["hl_all_lo"],
        hl_all_hi=highlights["hl_all_hi"],
        hl_past_lo=highlights["hl_past_lo"],
        hl_past_hi=highlights["hl_past_hi"],
        hl_fut_lo=highlights["hl_fut_lo"],
        hl_fut_hi=highlights["hl_fut_hi"],
        start=highlights["start"],
        past_end=highlights["past_end"],
        fut_start=highlights["fut_start"],
        fut_end=highlights["fut_end"],
    )
    ax.scatter(
        past_x, past_y, s=10, c="C2", edgecolors="black", linewidths=0.3, zorder=6,
        label="window lookback (repr pts)",
    )
    ax.scatter(
        fut_x, fut_y, s=10, c="C3", edgecolors="black", linewidths=0.3, zorder=6,
        label="window target (repr pts)",
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel(f"train time (÷ repr_stride={highlights['repr_stride']})")
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="upper right", fontsize=7, ncol=2)


def _window_z_scores(ds, window_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Past/future from z-scored ``ds.data``, not precomputed ordinal ranks."""
    start = window_idx * ds.stride
    past = ds.data[start : start + ds.lookback].T
    target_start = start + ds.lookback - ds.lookback_overlap
    target_end = start + ds.lookback + ds.horizon
    future = ds.data[target_start:target_end].T
    return past, future


def _count_ties(xs: np.ndarray, tie_atol: float) -> int:
    if xs.size == 0:
        return 0
    s = np.sort(xs.reshape(-1))
    n_unique = 1
    last = float(s[0])
    for v in s[1:]:
        if abs(float(v) - last) > tie_atol:
            n_unique += 1
        last = float(v)
    return int(xs.size - n_unique)


def pick_window_with_flatlines(
    train_ds,
    *,
    variate: int,
    tie_atol: float,
    max_scan: int = 512,
) -> int:
    best_idx, best_score = 0, -1
    for idx in range(min(len(train_ds), max_scan)):
        past, future = train_ds[idx]
        x = torch.cat([past[variate : variate + 1], future[variate : variate + 1]], dim=-1)
        score = _count_ties(x.numpy(), tie_atol)
        if score > best_score:
            best_score, best_idx = score, idx
    return best_idx


def pick_random_variate(n_vars: int, seed: int = 42) -> int:
    rng = random.Random(seed)
    return rng.randrange(n_vars)


def plot_ordinal_coarse_fine_2d(
    *,
    dataset: str,
    config_path: Path,
    out_dir: Path,
    window_idx: int,
    variate: int,
) -> Path:
    cfg = load_experiment_config(str(config_path))
    exp = cfg["experiment"]
    lookback = int(exp["lookback_length"])
    horizon = int(exp["forecast_length"])
    overlap = int(exp.get("lookback_overlap", 0))
    tie_atol = float(exp.get("ordinal_tie_atol", 1e-6))
    coarse_h = int(exp.get("coarse_image_height", exp.get("image_height", 16)))
    fine_h = int(exp.get("fine_image_height", exp.get("image_height", 16)))
    repr_stride = int(exp.get("representation_time_stride", 1))

    train_ds, _, _, norm_stats = load_dataset(
        dataset,
        lookback=lookback,
        horizon=horizon,
        lookback_overlap=overlap,
        stride=1,
        ordinal_tie_atol=tie_atol,
    )
    ladder = norm_stats.get("ordinal_ladder")
    if ladder is None:
        train_z = train_ds.data.numpy()
        ladder = build_global_ladder_from_training(
            train_z, tie_atol=tie_atol, precompute_ranks_for=train_z,
        )

    path, date_col = _resolve_registry_path(dataset)
    raw_all = _load_dataset_array(path, date_col)
    n = len(raw_all)
    border1s, border2s = _paper_split_borders(dataset, n, lookback)
    raw_train = raw_all[border1s[0]:border2s[0], variate].astype(np.float64)

    mean = norm_stats["mean"][0, variate]
    std = float(norm_stats["std"][0, variate])
    full_z_var = train_ds.data[:, variate].numpy().astype(np.float64)

    win_stride = int(train_ds.stride)
    start, past_end, fut_start, fut_end = _window_raw_span(
        window_idx=window_idx,
        stride=win_stride,
        lookback=lookback,
        horizon=horizon,
        overlap=overlap,
    )

    full_z = train_ds.data.T.unsqueeze(0)
    full_ord = encode_with_ladder(full_z, ladder.expand_batch(1))
    full_ord_var = full_ord[0, variate].detach().cpu().numpy()
    full_raw_sub = _subsample_1d(raw_train, repr_stride)
    full_z_sub = _subsample_1d(full_z_var, repr_stride)
    full_ord_sub = _subsample_1d(full_ord_var, repr_stride)
    t_full_sub = np.arange(len(full_ord_sub))

    def _raw_to_sub(lo: int, hi: int) -> tuple[float, float]:
        return lo / repr_stride, hi / repr_stride

    hl_past_lo, hl_past_hi = _raw_to_sub(start, past_end)
    hl_fut_lo, hl_fut_hi = _raw_to_sub(fut_start, fut_end)
    hl_all_lo, hl_all_hi = _raw_to_sub(start, fut_end)

    past, future = _window_z_scores(train_ds, window_idx)
    past_b = past.unsqueeze(0)
    fut_b = future.unsqueeze(0)
    past_ord, fut_ord, _ = ordinal_encode(past_b, fut_b, ladder=ladder)[:3]
    seq_ord = torch.cat([past_ord, fut_ord], dim=-1)

    vmax = ladder.rank_max_per_variate().reshape(-1).to(dtype=seq_ord.dtype)
    to2d = TimeSeriesTo2D(height=coarse_h, max_scale=1.0)

    model_cfg = DiffusionTSFConfig(
        num_variables=past.shape[0],
        lookback_length=lookback,
        forecast_length=horizon + overlap,
        dataset_forecast_length=horizon,
        lookback_overlap=overlap,
        representation_time_stride=repr_stride,
        image_height=coarse_h,
        coarse_image_height=coarse_h,
        fine_image_height=fine_h,
        use_ordinal_window_norm=True,
        ordinal_tie_atol=tie_atol,
        ordinal_ladder=ladder,
        use_window_normalization=False,
        staged_representation=str(exp.get("staged_representation", "value_precision")),
        disable_cross_attention=True,
    )
    model = DiffusionTSF(model_cfg)
    maps = model._encode_staged_maps(seq_ord)
    seq_sub = model._subsample_repr_time(seq_ord)
    coarse, fine = to2d.encode_dual_heights_bounded(
        seq_sub,
        coarse_height=coarse_h,
        fine_height=fine_h,
        value_min=0.0,
        value_max_per_variate=vmax,
    )
    assert torch.allclose(maps["coarse"], coarse, atol=1e-5)
    assert torch.allclose(maps["fine"], fine, atol=1e-5)

    k = int(ladder.n_unique[0, variate].item())
    rank_max = int(vmax[variate].item())

    coarse_np = coarse[0, variate].detach().cpu().numpy()
    fine_np = fine[0, variate].detach().cpu().numpy()
    ord_np = seq_sub[0, variate].detach().cpu().numpy()

    past_sub = model._subsample_repr_time(past_ord)[0, variate].detach().cpu().numpy()
    fut_sub = model._subsample_repr_time(fut_ord)[0, variate].detach().cpu().numpy()
    n_past_sub = len(past_sub)
    x_past_on_full = (start + np.arange(n_past_sub) * repr_stride) / repr_stride
    x_fut_on_full = (fut_start + np.arange(len(fut_sub)) * repr_stride) / repr_stride
    raw_past_idx = start + np.arange(n_past_sub) * repr_stride
    raw_fut_idx = fut_start + np.arange(len(fut_sub)) * repr_stride
    raw_past_on_full = raw_train[raw_past_idx]
    raw_fut_on_full = raw_train[raw_fut_idx]
    z_past_on_full = full_z_var[raw_past_idx]
    z_fut_on_full = full_z_var[raw_fut_idx]

    highlight_kw = dict(
        hl_all_lo=hl_all_lo,
        hl_all_hi=hl_all_hi,
        hl_past_lo=hl_past_lo,
        hl_past_hi=hl_past_hi,
        hl_fut_lo=hl_fut_lo,
        hl_fut_hi=hl_fut_hi,
        start=start,
        past_end=past_end,
        fut_start=fut_start,
        fut_end=fut_end,
        repr_stride=repr_stride,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}_v{variate}_win{window_idx}_ordinal_2d"
    out_path = out_dir / f"{stem}.png"

    fig = plt.figure(figsize=(18, 22))
    gs = fig.add_gridspec(6, 2, height_ratios=[1.1, 1.1, 1.1, 1.0, 2.0, 2.0], hspace=0.38, wspace=0.25)

    ax_raw = fig.add_subplot(gs[0, :])
    _plot_full_train_context(
        ax_raw,
        t_axis=t_full_sub,
        series=full_raw_sub,
        past_x=x_past_on_full,
        past_y=raw_past_on_full,
        fut_x=x_fut_on_full,
        fut_y=raw_fut_on_full,
        highlights=highlight_kw,
        ylabel="raw value",
        title=(
            f"{dataset} var={variate} | full train UNNORMALIZED | "
            f"window={window_idx} raw t={start}..{fut_end - 1}"
        ),
        line_label=f"full train raw (T={len(raw_train)})",
    )

    ax_z = fig.add_subplot(gs[1, :])
    _plot_full_train_context(
        ax_z,
        t_axis=t_full_sub,
        series=full_z_sub,
        past_x=x_past_on_full,
        past_y=z_past_on_full,
        fut_x=x_fut_on_full,
        fut_y=z_fut_on_full,
        highlights=highlight_kw,
        ylabel="global z-score",
        title=(
            f"{dataset} var={variate} | full train Z-SCORE | "
            f"μ={mean:.4g} σ={std:.4g}"
        ),
        line_label=f"full train z-score (repr_stride={repr_stride})",
    )

    ax_ord = fig.add_subplot(gs[2, :])
    _plot_full_train_context(
        ax_ord,
        t_axis=t_full_sub,
        series=full_ord_sub,
        past_x=x_past_on_full,
        past_y=past_sub,
        fut_x=x_fut_on_full,
        fut_y=fut_sub,
        highlights=highlight_kw,
        ylabel="ordinal rank",
        title=(
            f"{dataset} var={variate} | full train ORDINAL | "
            f"K={k} ranks [0,{rank_max}]"
        ),
        ylim=(-0.5, max(rank_max, 1) + 0.5),
        line_label="full train ordinal",
    )

    ax0 = fig.add_subplot(gs[3, :])
    z_concat = torch.cat([past[variate : variate + 1], future[variate : variate + 1]], dim=-1).unsqueeze(0)
    z_sub = model._subsample_repr_time(z_concat)[0, 0].detach().cpu().numpy()
    t_win = np.arange(len(ord_np))
    ax0.plot(t_win, z_sub, label="window z-score (repr)", alpha=0.5, linewidth=1)
    ax0.plot(t_win, ord_np, label="window ordinal (repr)", linewidth=1.5)
    ax0.axvline(n_past_sub - 0.5, color="gray", linestyle="--", linewidth=0.8, label="past | future")
    ax0.set_title(
        f"window zoom | win={window_idx} | model input cols={len(ord_np)} | repr_stride={repr_stride}"
    )
    ax0.legend(loc="upper right", fontsize=8)
    ax0.set_ylabel("value")
    ax0.set_xlabel("window column (repr subsampled)")

    ax1 = fig.add_subplot(gs[4, 0])
    _plot_cdf_map(ax1, coarse_np, title=f"coarse CDF ({coarse_h}×{coarse_np.shape[1]} px)", time_labels=t_win)
    ax2 = fig.add_subplot(gs[4, 1])
    _plot_cdf_map(ax2, fine_np, title=f"fine residual CDF ({fine_h}×{fine_np.shape[1]} px)", time_labels=t_win)

    ax3 = fig.add_subplot(gs[5, 0])
    ax3.imshow(coarse_np, aspect="auto", origin="lower", interpolation="nearest", cmap="gray", vmin=0, vmax=1)
    ax3.set_title("coarse pixels (exact 0/1 as model sees)")
    ax3.set_ylabel("row")
    ax3.set_xlabel("time")

    ax4 = fig.add_subplot(gs[5, 1])
    ax4.imshow(fine_np, aspect="auto", origin="lower", interpolation="nearest", cmap="gray", vmin=0, vmax=1)
    ax4.set_title("fine pixels (exact 0/1 as model sees)")
    ax4.set_ylabel("row")
    ax4.set_xlabel("time")

    fig.suptitle(
        "raw → z-score → ordinal ranks → coarse/fine 2D binary CDF (model view)",
        fontsize=12,
        y=0.995,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    meta_path = out_dir / f"{stem}.txt"
    meta_path.write_text(
        "\n".join(
            [
                f"dataset={dataset}",
                f"window_idx={window_idx}",
                f"variate={variate}",
                f"train_unique={k}",
                f"rank_max={rank_max}",
                f"raw_window_span=[{start},{fut_end})",
                f"raw_min={float(raw_train.min()):.6f}",
                f"raw_max={float(raw_train.max()):.6f}",
                f"z_min={float(full_z_var.min()):.4f}",
                f"z_max={float(full_z_var.max()):.4f}",
                f"repr_stride={repr_stride}",
                f"coarse_shape={tuple(coarse.shape)}",
                f"fine_shape={tuple(fine.shape)}",
                f"coarse_unique={np.unique(coarse_np).tolist()}",
                f"fine_unique={np.unique(fine_np).tolist()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm.yaml",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "reports/ordinal_coarse_fine_2d",
    )
    parser.add_argument("--window-idx", type=int, default=None)
    parser.add_argument("--variate", type=int, default=None)
    parser.add_argument("--prefer-flatlines", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_experiment_config(str(args.config))
    exp = cfg["experiment"]
    tie_atol = float(exp.get("ordinal_tie_atol", 1e-6))
    train_ds, _, _, _ = load_dataset(
        args.dataset,
        lookback=int(exp["lookback_length"]),
        horizon=int(exp["forecast_length"]),
        lookback_overlap=int(exp.get("lookback_overlap", 0)),
        stride=1,
        ordinal_tie_atol=tie_atol,
    )
    variate = args.variate
    if variate is None:
        variate = pick_random_variate(train_ds.data.shape[1], seed=args.seed)
    window_idx = args.window_idx
    if window_idx is None:
        window_idx = (
            pick_window_with_flatlines(train_ds, variate=variate, tie_atol=tie_atol)
            if args.prefer_flatlines
            else min(400, len(train_ds) - 1)
        )

    out = plot_ordinal_coarse_fine_2d(
        dataset=args.dataset,
        config_path=args.config,
        out_dir=args.out_dir,
        window_idx=window_idx,
        variate=variate,
    )
    print(f"wrote {out} (var={variate} win={window_idx})")


if __name__ == "__main__":
    main()
