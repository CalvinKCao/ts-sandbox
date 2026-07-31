#!/usr/bin/env python3
"""Before/after centered bin-index mean shift for ordinal disc candidates.

Uses the same post-snap binary-dataset-z path as
``temp/scripts/viz_disc_input_gt_mmpd_lattice_snap.py``, then applies
``utils.disc_bin_center_shift.bin_center_shift`` (per-variate, integer).

Panels: raw snapped vs shifted GT / MMPD / binary (if present); annotate mean
centered-bin before/after; show consecutive bin diffs unchanged (except clamps).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
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

DEFAULT_OUT = REPO_ROOT / "results/pulled/h96-disc-bin-center-shift"
DEFAULT_DATASETS = ("ETTh1", "traffic", "dynamic")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    p.add_argument("--disc-raw-dir", type=Path, default=DEFAULT_DISC_RAW)
    p.add_argument("--mmpd-output-root", type=Path, default=DEFAULT_MMPD)
    p.add_argument("--mmpd-data-dir", type=Path, default=DEFAULT_MMPD_DATA)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--fake-agg", choices=["prob_mean", "sample0"], default="sample0")
    p.add_argument("--n-windows", type=int, default=3)
    p.add_argument("--variate", type=int, default=0)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--linewidth", type=float, default=0.7)
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
    if args.smoke_test:
        args.datasets = args.datasets[:1]
        args.n_windows = 1
    return args


def _maybe_binary(
    args: argparse.Namespace,
    dataset: str,
    tensors: Mapping[str, np.ndarray],
) -> Optional[np.ndarray]:
    """Snap binary sample0 onto the same ladder when disc-raw samples exist."""
    path = args.disc_raw_dir / f"binary_ordinal_patch_refine_{dataset}.npz"
    if not path.is_file():
        return None
    with np.load(path) as data:
        if "samples" not in data.files:
            return None
        samples = data["samples"].astype(np.float32)
    # sample0 = first draw; shape (N,V,S,H) or already reduced.
    if samples.ndim == 4:
        binary = samples[:, :, 0, :]
    elif samples.ndim == 3:
        binary = samples
    else:
        return None
    legal = tensors["legal_levels"]
    if binary.shape != tensors["gt"].shape:
        # Align by shared indices length only.
        n = min(binary.shape[0], tensors["gt"].shape[0])
        binary = binary[:n]
        legal = legal[:n]
    snapped, _ = snap_to_patch_refine_levels(binary, legal)
    return snapped.astype(np.float32)


def _plot_window(
    *,
    out_path: Path,
    dataset: str,
    local: int,
    pool_i: int,
    variate: int,
    gt: np.ndarray,
    mmpd: np.ndarray,
    binary: Optional[np.ndarray],
    levels: np.ndarray,
    lw: float,
    dpi: int,
) -> Dict[str, float]:
    gt_b = gt[None, :, :]
    mmpd_b = mmpd[None, :, :]
    levels_b = levels[None, :, :]
    gt_s, gt_stats = bin_center_shift(gt_b, levels_b, reduce="per_variate")
    mmpd_s, mmpd_stats = bin_center_shift(mmpd_b, levels_b, reduce="per_variate")
    binary_s = None
    bin_stats = None
    if binary is not None:
        binary_s, bin_stats = bin_center_shift(binary[None], levels_b, reduce="per_variate")

    v = int(variate)
    center = int(center_bin_index(levels_b)[0, v])
    raw_gt = nearest_bin_indices(gt_b, levels_b)[0, v]
    raw_m = nearest_bin_indices(mmpd_b, levels_b)[0, v]
    cen_gt = raw_gt - center
    cen_m = raw_m - center
    raw_gt_s = nearest_bin_indices(gt_s, levels_b)[0, v]
    raw_m_s = nearest_bin_indices(mmpd_s, levels_b)[0, v]
    cen_gt_s = raw_gt_s - center
    cen_m_s = raw_m_s - center

    x = np.arange(gt.shape[-1])
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.2), sharex=True)
    ax0, ax1, ax2 = axes

    ax0.plot(x, gt[v], color="black", lw=lw, label="GT snapped", drawstyle="steps-post")
    ax0.plot(x, mmpd[v], color="#d62728", lw=lw, alpha=0.9, label="MMPD snapped", drawstyle="steps-post")
    if binary is not None:
        ax0.plot(x, binary[v], color="#1f77b4", lw=lw, alpha=0.85, label="binary snapped", drawstyle="steps-post")
    ax0.axhline(levels[v, center], color="0.4", ls="--", lw=0.6, label=f"center level (idx={center})")
    ax0.set_ylabel("dataset-z\n(before)")
    ax0.set_title(
        f"{dataset} pool={pool_i} local={local} v={v} | BEFORE bin-center shift\n"
        f"mean centered-bin GT={cen_gt.mean():.2f} MMPD={cen_m.mean():.2f}"
    )
    ax0.legend(loc="upper right", fontsize=8, frameon=False)
    ax0.grid(True, alpha=0.25)

    ax1.plot(x, gt_s[0, v], color="black", lw=lw, label="GT shifted", drawstyle="steps-post")
    ax1.plot(x, mmpd_s[0, v], color="#d62728", lw=lw, alpha=0.9, label="MMPD shifted", drawstyle="steps-post")
    if binary_s is not None:
        ax1.plot(x, binary_s[0, v], color="#1f77b4", lw=lw, alpha=0.85, label="binary shifted", drawstyle="steps-post")
    ax1.axhline(levels[v, center], color="0.4", ls="--", lw=0.6, label=f"center level (idx={center})")
    ax1.set_ylabel("dataset-z\n(after)")
    ax1.set_title(
        f"AFTER per-variate integer bin-center shift (no std scaling)\n"
        f"mean centered-bin GT={cen_gt_s.mean():.2f} MMPD={cen_m_s.mean():.2f} "
        f"| shift≈{gt_stats['mean_abs_shift']:.1f}"
    )
    ax1.legend(loc="upper right", fontsize=8, frameon=False)
    ax1.grid(True, alpha=0.25)

    d_gt = np.diff(cen_gt.astype(np.float64))
    d_gt_s = np.diff(cen_gt_s.astype(np.float64))
    d_m = np.diff(cen_m.astype(np.float64))
    d_m_s = np.diff(cen_m_s.astype(np.float64))
    ax2.plot(x[1:], d_gt, color="black", lw=lw, alpha=0.7, label="Δ centered-bin GT before")
    ax2.plot(x[1:], d_gt_s, color="0.45", lw=lw, ls="--", label="Δ GT after")
    ax2.plot(x[1:], d_m, color="#d62728", lw=lw, alpha=0.7, label="Δ MMPD before")
    ax2.plot(x[1:], d_m_s, color="#ff9896", lw=lw, ls="--", label="Δ MMPD after")
    ax2.set_ylabel("bin diffs")
    ax2.set_xlabel("horizon step")
    max_dd = float(np.max(np.abs(d_gt_s - d_gt))) if d_gt.size else 0.0
    ax2.set_title(
        f"Consecutive centered-bin diffs (spread unchanged if no clamp); "
        f"max|Δdiff|_GT={max_dd:.3g}  clamp_frac_gt={gt_stats['frac_clamped']:.3g}"
    )
    ax2.legend(loc="upper right", fontsize=8, frameon=False)
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    summary = {
        "pool_index": float(pool_i),
        "local": float(local),
        "variate": float(v),
        "center_index": float(center),
        "gt_mean_centered_before": float(cen_gt.mean()),
        "gt_mean_centered_after": float(cen_gt_s.mean()),
        "mmpd_mean_centered_before": float(cen_m.mean()),
        "mmpd_mean_centered_after": float(cen_m_s.mean()),
        "max_abs_diff_delta_gt": max_dd,
        **{f"gt_{k}": float(val) for k, val in gt_stats.items() if isinstance(val, (int, float))},
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
    meta_all: Dict[str, object] = {"fake_agg": args.fake_agg, "datasets": {}}

    for dataset in args.datasets:
        print(f"=== {dataset} ===", flush=True)
        tensors, snap_meta, _indices = _prepare_snapped(args, dataset, device)
        binary = _maybe_binary(args, dataset, tensors)
        gt = tensors["gt"]
        mmpd = tensors["mmpd"]
        levels = tensors["legal_levels"]
        pool_idx = tensors["indices"]
        picks = _pick_windows(gt, mmpd, n=int(args.n_windows), variate=int(args.variate), seed=int(args.seed))
        ds_dir = args.output_dir / dataset
        ds_dir.mkdir(parents=True, exist_ok=True)
        summaries = []
        for local in picks.tolist():
            pool_i = int(pool_idx[local])
            out = ds_dir / f"{dataset}_v{args.variate}_local{local}_pool{pool_i}_bin_center.png"
            bin_win = None if binary is None else binary[local]
            summary = _plot_window(
                out_path=out,
                dataset=dataset,
                local=int(local),
                pool_i=pool_i,
                variate=int(args.variate),
                gt=gt[local],
                mmpd=mmpd[local],
                binary=bin_win,
                levels=levels[local],
                lw=float(args.linewidth),
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
