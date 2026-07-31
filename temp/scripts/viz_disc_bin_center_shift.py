#!/usr/bin/env python3
"""Per-slice (L=8/16/32) BEFORE/AFTER bin-center-shift viz for disc candidates.

Old full-horizon centering is wrong for the disc path: mean is over each
extracted L-slice only (same as UnivariateRealVsFakeDataset._norm_segment).
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
DEFAULT_DATASETS = ("ETTh1", "traffic", "dynamic")
DEFAULT_SLICE_LENGTHS = (8, 16, 32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    p.add_argument("--disc-raw-dir", type=Path, default=DEFAULT_DISC_RAW)
    p.add_argument("--mmpd-output-root", type=Path, default=DEFAULT_MMPD)
    p.add_argument("--mmpd-data-dir", type=Path, default=DEFAULT_MMPD_DATA)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--fake-agg", choices=["prob_mean", "sample0"], default="sample0")
    p.add_argument("--n-windows", type=int, default=2)
    p.add_argument("--variate", type=int, default=0)
    p.add_argument("--slice-lengths", type=int, nargs="+", default=list(DEFAULT_SLICE_LENGTHS))
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--linewidth", type=float, default=0.65)
    p.add_argument("--marker-size", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    args.datasets = [d for raw in args.datasets for d in str(raw).split(",") if d]
    args.disc_raw_dir = args.disc_raw_dir.expanduser().resolve()
    args.mmpd_output_root = args.mmpd_output_root.expanduser().resolve()
    args.mmpd_data_dir = args.mmpd_data_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.slice_lengths = [int(x) for x in args.slice_lengths]
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


def _draw_ladder_hlines(ax, levels_1d: np.ndarray, y_lo: float, y_hi: float) -> None:
    """All legal levels intersecting the visible y-range (no thinning)."""
    lev = np.asarray(levels_1d, dtype=np.float64)
    pad = 0.02 * max(1e-6, y_hi - y_lo)
    mask = (lev >= y_lo - pad) & (lev <= y_hi + pad)
    for y in lev[mask]:
        ax.axhline(float(y), color="0.82", lw=0.25, zorder=0)


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
) -> Dict[str, float]:
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
    x = np.arange(slice_len)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.4), sharex=True)
    ax0, ax1 = axes

    series_before = [
        ("GT", gt_slice, "black"),
        ("MMPD", mmpd_slice, "#d62728"),
    ]
    if binary_slice is not None:
        series_before.append(("binary", binary_slice, "#1f77b4"))
    series_after = [
        ("GT", gt_s, "black"),
        ("MMPD", mmpd_s, "#d62728"),
    ]
    if binary_s is not None:
        series_after.append(("binary", binary_s, "#1f77b4"))

    y_all = np.concatenate([gt_slice, mmpd_slice, gt_s, mmpd_s])
    if binary_slice is not None:
        y_all = np.concatenate([y_all, binary_slice, binary_s])
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
        _draw_ladder_hlines(ax, levels_1d, y_lo_p, y_hi_p)
        for label, y, color in series:
            ax.plot(
                x, y, color=color, lw=lw, label=label,
                drawstyle="steps-post", marker="o", markersize=ms,
                markeredgewidth=0.4, markerfacecolor=color, alpha=0.95,
            )
        ax.axhline(float(levels_1d[center]), color="0.35", ls="--", lw=0.55, label=f"center (idx={center})")
        ax.set_ylim(y_lo_p, y_hi_p)
        ax.set_ylabel("dataset-z")
        ax.set_title(f"{title_extra}\n{mean_note}", fontsize=10)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        ax.grid(True, alpha=0.22)

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
        "mode": "per_slice_L",
        "slice_lengths": list(args.slice_lengths),
        "fake_agg": args.fake_agg,
        "datasets": {},
    }

    for dataset in args.datasets:
        print(f"=== {dataset} (per-slice) ===", flush=True)
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
                for offset in _slice_offsets(horizon, L):
                    sl = slice(offset, offset + L)
                    out = (
                        ds_dir
                        / f"{dataset}_v{v}_local{local}_pool{pool_i}_L{L}_off{offset}_bin_center.png"
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
