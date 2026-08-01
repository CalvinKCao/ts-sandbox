#!/usr/bin/env python3
"""Per-slice (L=8/16/32) BEFORE/AFTER bin-center-shift viz for disc candidates.

Old full-horizon centering is wrong for the disc path: mean is over each
extracted L-slice only (same as UnivariateRealVsFakeDataset._norm_segment).

``--zoom``: short t-window (default 16 steps), ylim padded by ~2–3 ladder
rungs, all in-range legal levels as hlines, large step+marker markers so
discrete snaps are obvious. Default out: results/pulled/.../zoom/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from temp.scripts.viz_disc_input_gt_mmpd_lattice_snap import (  # noqa: E402
    DEFAULT_DISC_RAW,
    DEFAULT_MMPD,
    _pick_windows,
    _prepare_snapped,
)
from utils.disc_bin_center_shift import (  # noqa: E402
    bin_center_shift,
    center_bin_index,
    nearest_bin_indices,
)
from utils.eval_mmpd_gaussian_anchor import DEFAULT_MMPD_DATA  # noqa: E402
from utils.patch_refine_ordinal_ladder import snap_to_patch_refine_levels  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "results/pulled/h96-disc-bin-center-shift/per_slice"
DEFAULT_ZOOM_OUT = REPO_ROOT / "results/pulled/h96-disc-bin-center-shift/zoom"
DEFAULT_DATASETS = ("ETTh1", "traffic", "dynamic")
DEFAULT_ZOOM_DATASETS = ("ETTh1", "traffic")
DEFAULT_SLICE_LENGTHS = (8, 16, 32)
DEFAULT_ZOOM_SLICE_LENGTHS = (16, 32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--disc-raw-dir", type=Path, default=DEFAULT_DISC_RAW)
    p.add_argument("--mmpd-output-root", type=Path, default=DEFAULT_MMPD)
    p.add_argument("--mmpd-data-dir", type=Path, default=DEFAULT_MMPD_DATA)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--fake-agg", choices=["prob_mean", "sample0"], default="sample0")
    p.add_argument("--n-windows", type=int, default=2)
    p.add_argument("--variate", type=int, default=0)
    p.add_argument("--slice-lengths", type=int, nargs="+", default=None)
    p.add_argument(
        "--zoom",
        action="store_true",
        help="Heavy zoom: short t-window, y padded by ~2–3 ladder rungs, large markers.",
    )
    p.add_argument(
        "--zoom-steps",
        type=int,
        default=16,
        help="With --zoom, plot this many consecutive timesteps inside each L-slice (12–24).",
    )
    p.add_argument(
        "--zoom-t0",
        type=int,
        default=None,
        help="With --zoom, start index inside the L-slice (default: mid so window fits).",
    )
    p.add_argument(
        "--y-rung-pad",
        type=int,
        default=3,
        help="With --zoom, pad ylim by this many ladder rungs above/below the signal.",
    )
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--linewidth", type=float, default=None)
    p.add_argument("--marker-size", type=float, default=None)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    if args.datasets is None:
        args.datasets = list(DEFAULT_ZOOM_DATASETS if args.zoom else DEFAULT_DATASETS)
    else:
        args.datasets = [d for raw in args.datasets for d in str(raw).split(",") if d]
    if args.slice_lengths is None:
        args.slice_lengths = list(
            DEFAULT_ZOOM_SLICE_LENGTHS if args.zoom else DEFAULT_SLICE_LENGTHS
        )
    else:
        args.slice_lengths = [int(x) for x in args.slice_lengths]
    if args.output_dir is None:
        args.output_dir = DEFAULT_ZOOM_OUT if args.zoom else DEFAULT_OUT
    if args.linewidth is None:
        args.linewidth = 1.35 if args.zoom else 0.65
    if args.marker_size is None:
        args.marker_size = 7.0 if args.zoom else 3.0
    args.disc_raw_dir = args.disc_raw_dir.expanduser().resolve()
    args.mmpd_output_root = args.mmpd_output_root.expanduser().resolve()
    args.mmpd_data_dir = args.mmpd_data_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.zoom_steps = max(4, int(args.zoom_steps))
    args.y_rung_pad = max(1, int(args.y_rung_pad))
    if args.zoom:
        # One window × a couple L's is enough for heavily zoomed panels.
        args.n_windows = min(int(args.n_windows), 1)
    if args.smoke_test:
        args.datasets = args.datasets[:1]
        args.n_windows = 1
        args.slice_lengths = args.slice_lengths[:1]
    return args


def _maybe_binary(
    args: argparse.Namespace,
    dataset: str,
    tensors: Mapping[str, np.ndarray],
) -> Optional[np.ndarray]:
    path = args.disc_raw_dir / f"binary_ordinal_patch_refine_{dataset}.npz"
    if not path.is_file():
        return None
    with np.load(path) as data:
        if "samples" not in data.files:
            return None
        samples = data["samples"].astype(np.float32)
    if samples.ndim == 4:
        binary = samples[:, :, 0, :]
    elif samples.ndim == 3:
        binary = samples
    else:
        return None
    legal = tensors["legal_levels"]
    if binary.shape != tensors["gt"].shape:
        n = min(binary.shape[0], tensors["gt"].shape[0])
        binary = binary[:n]
        legal = legal[:n]
    snapped, _ = snap_to_patch_refine_levels(binary, legal)
    return snapped.astype(np.float32)


def _slice_offsets(horizon: int, slice_len: int) -> List[int]:
    """Offset 0 and mid (when distinct); mid = (H-L)//2."""
    max_off = horizon - slice_len
    if max_off < 0:
        raise ValueError(f"slice_len={slice_len} > horizon={horizon}")
    offs = [0]
    mid = max_off // 2
    if mid not in offs:
        offs.append(mid)
    return offs


def _draw_ladder_hlines(ax, levels_1d: np.ndarray, y_lo: float, y_hi: float, *, lw: float = 0.25) -> None:
    """All legal levels intersecting the visible y-range (no thinning)."""
    lev = np.asarray(levels_1d, dtype=np.float64)
    pad = 0.02 * max(1e-6, y_hi - y_lo)
    mask = (lev >= y_lo - pad) & (lev <= y_hi + pad)
    for y in lev[mask]:
        ax.axhline(float(y), color="0.78", lw=lw, zorder=0)


def _ylim_rung_pad(
    levels_1d: np.ndarray,
    y_vals: np.ndarray,
    *,
    rung_pad: int,
) -> Tuple[float, float]:
    """Tight ylim around signal, padded by ``rung_pad`` ladder rungs each side."""
    lev = np.asarray(levels_1d, dtype=np.float64)
    y = np.asarray(y_vals, dtype=np.float64)
    y_lo = float(np.min(y))
    y_hi = float(np.max(y))
    if y_hi <= y_lo:
        y_hi = y_lo + 1e-3
    # Rungs that touch the signal span, then expand by rung_pad.
    in_span = np.where((lev >= y_lo - 1e-9) & (lev <= y_hi + 1e-9))[0]
    if in_span.size == 0:
        # Fall back to nearest levels around the signal.
        mid = 0.5 * (y_lo + y_hi)
        nearest = int(np.argmin(np.abs(lev - mid)))
        lo_i = max(0, nearest - rung_pad)
        hi_i = min(len(lev) - 1, nearest + rung_pad)
    else:
        lo_i = max(0, int(in_span[0]) - rung_pad)
        hi_i = min(len(lev) - 1, int(in_span[-1]) + rung_pad)
    return float(lev[lo_i]), float(lev[hi_i])


def _zoom_t_window(slice_len: int, zoom_steps: int, zoom_t0: Optional[int]) -> Tuple[int, int]:
    """Return [t0, t1) inside a length-``slice_len`` slice."""
    steps = min(int(zoom_steps), int(slice_len))
    if zoom_t0 is None:
        t0 = max(0, (slice_len - steps) // 2)
    else:
        t0 = max(0, min(int(zoom_t0), slice_len - steps))
    return t0, t0 + steps


def _mean_centered_bin(values_1d: np.ndarray, levels_1d: np.ndarray) -> float:
    """Mean centered-bin index for one (L,) slice / (H,) ladder row."""
    vals = values_1d[None, None, :].astype(np.float32)
    levels = levels_1d[None, None, :].astype(np.float32)
    center = int(center_bin_index(levels)[0, 0])
    raw = nearest_bin_indices(vals, levels)[0, 0]
    return float((raw - center).astype(np.float64).mean())


def _shift_slice(
    values_1d: np.ndarray,
    levels_1d: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    vals = values_1d[None, None, :].astype(np.float32)
    levels = levels_1d[None, None, :].astype(np.float32)
    shifted, stats = bin_center_shift(vals, levels, reduce="per_variate")
    return shifted[0, 0].astype(np.float32), stats


def _plot_slice_panel(
    *,
    out_path: Path,
    dataset: str,
    local: int,
    pool_i: int,
    variate: int,
    slice_len: int,
    offset: int,
    gt_slice: np.ndarray,
    mmpd_slice: np.ndarray,
    binary_slice: Optional[np.ndarray],
    levels_1d: np.ndarray,
    lw: float,
    ms: float,
    dpi: int,
    zoom: bool = False,
    zoom_steps: int = 16,
    zoom_t0: Optional[int] = None,
    y_rung_pad: int = 3,
) -> Dict[str, float]:
    # Shift uses the full L-slice (disc protocol); zoom only crops the view.
    gt_s, gt_stats = _shift_slice(gt_slice, levels_1d)
    mmpd_s, mmpd_stats = _shift_slice(mmpd_slice, levels_1d)
    binary_s = None
    bin_stats = None
    if binary_slice is not None:
        binary_s, bin_stats = _shift_slice(binary_slice, levels_1d)

    mean_gt_b = _mean_centered_bin(gt_slice, levels_1d)
    mean_gt_a = _mean_centered_bin(gt_s, levels_1d)
    mean_m_b = _mean_centered_bin(mmpd_slice, levels_1d)
    mean_m_a = _mean_centered_bin(mmpd_s, levels_1d)

    center = int(center_bin_index(levels_1d[None, None, :])[0, 0])
    if zoom:
        t0, t1 = _zoom_t_window(slice_len, zoom_steps, zoom_t0)
        view = slice(t0, t1)
        x = np.arange(t0, t1)
        gt_b_v, gt_a_v = gt_slice[view], gt_s[view]
        mmpd_b_v, mmpd_a_v = mmpd_slice[view], mmpd_s[view]
        bin_b_v = None if binary_slice is None else binary_slice[view]
        bin_a_v = None if binary_s is None else binary_s[view]
        fig_w = max(8.5, 0.55 * (t1 - t0) + 3.0)
        fig, axes = plt.subplots(2, 1, figsize=(fig_w, 7.2), sharex=True)
        hline_lw = 0.55
    else:
        t0, t1 = 0, slice_len
        x = np.arange(slice_len)
        gt_b_v, gt_a_v = gt_slice, gt_s
        mmpd_b_v, mmpd_a_v = mmpd_slice, mmpd_s
        bin_b_v, bin_a_v = binary_slice, binary_s
        fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.4), sharex=True)
        hline_lw = 0.25
    ax0, ax1 = axes

    series_before = [
        ("GT", gt_b_v, "black"),
        ("MMPD", mmpd_b_v, "#d62728"),
    ]
    if bin_b_v is not None:
        series_before.append(("binary", bin_b_v, "#1f77b4"))
    series_after = [
        ("GT", gt_a_v, "black"),
        ("MMPD", mmpd_a_v, "#d62728"),
    ]
    if bin_a_v is not None:
        series_after.append(("binary", bin_a_v, "#1f77b4"))

    y_all = np.concatenate([gt_b_v, mmpd_b_v, gt_a_v, mmpd_a_v])
    if bin_b_v is not None:
        y_all = np.concatenate([y_all, bin_b_v, bin_a_v])
    if zoom:
        y_lo_p, y_hi_p = _ylim_rung_pad(levels_1d, y_all, rung_pad=y_rung_pad)
    else:
        y_lo = float(np.min(y_all))
        y_hi = float(np.max(y_all))
        if y_hi <= y_lo:
            y_hi = y_lo + 1.0
        pad = 0.08 * (y_hi - y_lo)
        y_lo_p, y_hi_p = y_lo - pad, y_hi + pad

    for ax, series, title_extra, mean_note in (
        (
            ax0,
            series_before,
            "BEFORE per-slice bin_center_shift",
            f"mean centered-bin GT={mean_gt_b:.2f} MMPD={mean_m_b:.2f}",
        ),
        (
            ax1,
            series_after,
            "AFTER per-slice bin_center_shift",
            f"mean centered-bin GT={mean_gt_a:.2f} MMPD={mean_m_a:.2f}",
        ),
    ):
        _draw_ladder_hlines(ax, levels_1d, y_lo_p, y_hi_p, lw=hline_lw)
        for label, y, color in series:
            ax.plot(
                x, y, color=color, lw=lw, label=label,
                drawstyle="steps-post", marker="o", markersize=ms,
                markeredgewidth=0.55 if zoom else 0.4,
                markerfacecolor=color, alpha=0.95,
            )
        ax.axhline(
            float(levels_1d[center]), color="0.35", ls="--",
            lw=0.9 if zoom else 0.55, label=f"center (idx={center})",
        )
        ax.set_ylim(y_lo_p, y_hi_p)
        ax.set_ylabel("dataset-z")
        ax.set_title(f"{title_extra}\n{mean_note}", fontsize=10)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        ax.grid(True, alpha=0.22)
        if zoom:
            ax.set_xticks(x)
            ax.tick_params(axis="x", labelsize=8)

    if zoom:
        ax1.set_xlabel(
            f"step t in L={slice_len} slice (offset={offset}; zoom t=[{t0},{t1}))"
        )
        fig.suptitle(
            f"{dataset} pool={pool_i} local={local} v={variate} | ZOOM L={slice_len} "
            f"off={offset} t=[{t0},{t1}) | rung_pad={y_rung_pad}",
            fontsize=11,
        )
    else:
        ax1.set_xlabel(f"step within L={slice_len} slice (offset={offset})")
        fig.suptitle(
            f"{dataset} pool={pool_i} local={local} v={variate} | L={slice_len} offset={offset} "
            f"(mean over L only, not full H)",
            fontsize=11,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    summary: Dict[str, float] = {
        "pool_index": float(pool_i),
        "local": float(local),
        "variate": float(variate),
        "slice_len": float(slice_len),
        "offset": float(offset),
        "center_index": float(center),
        "zoom": float(zoom),
        "zoom_t0": float(t0),
        "zoom_t1": float(t1),
        "gt_mean_centered_before": mean_gt_b,
        "gt_mean_centered_after": mean_gt_a,
        "mmpd_mean_centered_before": mean_m_b,
        "mmpd_mean_centered_after": mean_m_a,
        **{f"gt_{k}": float(val) for k, val in gt_stats.items() if isinstance(val, (int, float))},
        **{f"mmpd_{k}": float(val) for k, val in mmpd_stats.items() if isinstance(val, (int, float))},
    }
    if bin_stats is not None:
        summary["binary_mean_centered_before"] = float(bin_stats["mean_centered_before"])
        summary["binary_mean_centered_after"] = float(bin_stats["mean_centered_after"])
    return summary


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_paths: List[Path] = []
    meta_all: Dict[str, object] = {
        "mode": "zoom_per_slice_L" if args.zoom else "per_slice_L",
        "slice_lengths": list(args.slice_lengths),
        "fake_agg": args.fake_agg,
        "zoom": bool(args.zoom),
        "zoom_steps": int(args.zoom_steps) if args.zoom else None,
        "y_rung_pad": int(args.y_rung_pad) if args.zoom else None,
        "datasets": {},
    }

    for dataset in args.datasets:
        print(f"=== {dataset} (per-slice{'; zoom' if args.zoom else ''}) ===", flush=True)
        try:
            tensors, snap_meta, _indices = _prepare_snapped(args, dataset, device)
        except Exception as exc:
            print(f"[{dataset}] skip: {exc}", flush=True)
            meta_all["datasets"][dataset] = {"error": str(exc)}
            continue
        binary = _maybe_binary(args, dataset, tensors)
        gt = tensors["gt"]
        mmpd = tensors["mmpd"]
        levels = tensors["legal_levels"]
        pool_idx = tensors["indices"]
        horizon = int(gt.shape[-1])
        picks = _pick_windows(
            gt, mmpd, n=int(args.n_windows), variate=int(args.variate), seed=int(args.seed),
        )
        ds_dir = args.output_dir / dataset
        ds_dir.mkdir(parents=True, exist_ok=True)
        summaries: List[Dict[str, float]] = []

        for local in picks.tolist():
            pool_i = int(pool_idx[local])
            v = int(args.variate)
            levels_1d = levels[local, v]
            bin_win = None if binary is None else binary[local, v]
            for L in args.slice_lengths:
                if L > horizon:
                    continue
                # Zoom: offset 0 only (keep 2–4 PNGs). Full mode: 0 + mid.
                offsets = [0] if args.zoom else _slice_offsets(horizon, L)
                for offset in offsets:
                    sl = slice(offset, offset + L)
                    tag = "zoom_bin_center" if args.zoom else "bin_center"
                    out = (
                        ds_dir
                        / f"{dataset}_v{v}_local{local}_pool{pool_i}_L{L}_off{offset}_{tag}.png"
                    )
                    summary = _plot_slice_panel(
                        out_path=out,
                        dataset=dataset,
                        local=int(local),
                        pool_i=pool_i,
                        variate=v,
                        slice_len=int(L),
                        offset=int(offset),
                        gt_slice=gt[local, v, sl],
                        mmpd_slice=mmpd[local, v, sl],
                        binary_slice=None if bin_win is None else bin_win[sl],
                        levels_1d=levels_1d,
                        lw=float(args.linewidth),
                        ms=float(args.marker_size),
                        dpi=int(args.dpi),
                        zoom=bool(args.zoom),
                        zoom_steps=int(args.zoom_steps),
                        zoom_t0=args.zoom_t0,
                        y_rung_pad=int(args.y_rung_pad),
                    )
                    summaries.append(summary)
                    all_paths.append(out)
                    print(f"  wrote {out}", flush=True)

        meta_all["datasets"][dataset] = {
            "snap": {k: snap_meta[k] for k in snap_meta if k != "align"},
            "windows": summaries,
            "n_binary": None if binary is None else int(binary.shape[0]),
        }

    manifest = args.output_dir / "manifest.json"
    manifest.write_text(json.dumps(meta_all, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"[done] {len(all_paths)} PNGs → {args.output_dir}", flush=True)
    for path in all_paths:
        print(path)


if __name__ == "__main__":
    main()
