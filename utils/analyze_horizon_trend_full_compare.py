#!/usr/bin/env python3
"""GT train/test + binary/MMPD test forecast horizon-trend compare (4-panel).

Instance-norm(last) - instance-norm(first) on horizon, same bins/metric as
`analyze_horizon_trend_distribution.py`. Writes one side-by-side figure per dataset.

Example:
  python utils/analyze_horizon_trend_full_compare.py
  python utils/analyze_horizon_trend_full_compare.py --datasets exchange_rate,weather
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
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.train_multivariate_pipeline import LOOKBACK_OVERLAP, load_dataset
from utils.analyze_gt_horizon_trend_baseline import _collate, gt_trends_for_loader
from utils.analyze_horizon_trend_distribution import (
    BINARY_STD_FLOOR,
    DEFAULT_OUTPUT,
    FLAT_THRESH,
    TREND_BINS,
    _load_past_batch,
    binary_instance_norm,
    horizon_trend,
    load_binary_deterministic,
    mmpd_instance_norm,
    trend_summary,
)
from utils.eval_mmpd_gaussian_anchor import (
    _load_data_subset_policy,
    resolve_subset_meta_for_dataset,
)
from utils.visualize_fair_mmpd_vs_binary_delta import (
    ALL_DATASETS,
    DEFAULT_MMPD_RUN,
    EVAL_TEST_STRIDE,
    TRAIN_STRIDE,
    discover_binary_run,
    load_mmpd_pack,
)

PANEL_SPECS = (
    ("GT train", "#43A047"),
    ("GT test", "#00897B"),
    ("Binary test", "#7E57C2"),
    ("MMPD test", "#26A69A"),
)


def forecast_trends_binary(
    dataset: str,
    pred: np.ndarray,
    window_indices: np.ndarray,
) -> np.ndarray:
    past = _load_past_batch(dataset, window_indices.tolist())
    norm_pred = binary_instance_norm(past, pred, BINARY_STD_FLOOR)
    return horizon_trend(norm_pred)


def forecast_trends_mmpd(
    dataset: str,
    pred: np.ndarray,
    window_indices: np.ndarray,
) -> np.ndarray:
    past = _load_past_batch(dataset, window_indices.tolist())
    norm_pred = mmpd_instance_norm(past, pred)
    return horizon_trend(norm_pred)


def load_forecast_trends(
    dataset: str,
    *,
    mmpd_run: Path,
    datasets_root: Path,
    ckpt_root: Path,
    allow_fallback_binary: bool,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, object]]:
    meta: Dict[str, object] = {}
    binary_trends: Optional[np.ndarray] = None
    mmpd_trends: Optional[np.ndarray] = None

    try:
        mmpd = load_mmpd_pack(mmpd_run, dataset)
    except FileNotFoundError as exc:
        print(f"[skip mmpd] {dataset}: {exc}")
        mmpd = None

    try:
        binary_run = discover_binary_run(
            datasets_root, ckpt_root, dataset, allow_fallback=allow_fallback_binary
        )
        binary_det = load_binary_deterministic(binary_run.results_dir, dataset)
        meta["binary_results_dir"] = str(binary_run.results_dir)
    except FileNotFoundError as exc:
        print(f"[skip binary] {dataset}: {exc}")
        binary_run = None
        binary_det = None

    if mmpd is not None and binary_det is not None:
        indices = np.asarray(mmpd["indices"], dtype=np.int64)
        n_bin = binary_det.shape[0]
        if int(indices.max(initial=0)) >= n_bin:
            raise RuntimeError(
                f"{dataset}: MMPD index {int(indices.max())} out of binary range {n_bin}"
            )
        mmpd_det = mmpd["deterministic"]
        binary_det = binary_det[indices]
    elif mmpd is not None:
        indices = np.asarray(mmpd["indices"], dtype=np.int64)
        mmpd_det = mmpd["deterministic"]
        if binary_det is not None:
            binary_det = binary_det[indices]
    elif binary_det is not None:
        indices = np.arange(binary_det.shape[0], dtype=np.int64)
        mmpd_det = None
    else:
        return None, None, meta

    if mmpd is not None and mmpd_det is not None:
        mmpd_trends = forecast_trends_mmpd(dataset, mmpd_det, indices)
        meta["mmpd"] = trend_summary(mmpd_trends)
    if binary_det is not None:
        binary_trends = forecast_trends_binary(dataset, binary_det, indices)
        meta["binary"] = trend_summary(binary_trends)

    return binary_trends, mmpd_trends, meta


def plot_four_panel_compare(
    *,
    dataset: str,
    panels: Sequence[Tuple[str, Optional[np.ndarray], str]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True)
    fig.suptitle(
        f"{dataset} · instance-norm horizon trend (norm[last] − norm[first])",
        fontsize=12,
        y=1.02,
    )

    for ax, (title, trends, color) in zip(axes, panels):
        if trends is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            ax.set_xlim(TREND_BINS[0], TREND_BINS[-1])
            continue

        flat = trends.ravel()
        summary = trend_summary(trends)
        ax.hist(
            flat,
            bins=TREND_BINS,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.3,
            density=True,
        )
        ax.axvline(0.0, color="#212121", lw=0.9, ls="--", alpha=0.6)
        ax.axvline(summary["mean"], color="#D84315", lw=1.2, label=f"μ={summary['mean']:.2f}")
        ax.set_title(
            f"{title}\n"
            f"n={summary['count']}  flat={summary['pct_flat']:.0f}%  "
            f"↑{summary['pct_up']:.0f}% ↓{summary['pct_down']:.0f}%",
            fontsize=9,
        )
        ax.set_xlim(TREND_BINS[0], TREND_BINS[-1])
        ax.grid(True, alpha=0.22)
        if title.startswith("GT"):
            ax.set_ylabel("density")
        ax.set_xlabel("trend")
        ax.legend(fontsize=7, loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def process_dataset(
    dataset: str,
    *,
    output_dir: Path,
    batch_size: int,
    mmpd_run: Path,
    datasets_root: Path,
    ckpt_root: Path,
    allow_fallback_binary: bool,
    write_gt_splits: bool,
) -> Dict[str, object]:
    policy = _load_data_subset_policy(REPO_ROOT / "configs" / "binary_anchor_stationary_flat_subsets.yaml")
    subset = resolve_subset_meta_for_dataset(dataset, policy, seed=2026)
    variate_indices = [int(i) for i in subset["variate_indices"]]

    train_ds, _, test_ds, _ = load_dataset(
        dataset,
        variate_indices,
        stride=TRAIN_STRIDE,
        test_stride=EVAL_TEST_STRIDE,
    )
    overlap = int(LOOKBACK_OVERLAP)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_collate
    )

    gt_train = gt_trends_for_loader(train_loader, overlap=overlap)
    gt_test = gt_trends_for_loader(test_loader, overlap=overlap)
    binary_trends, mmpd_trends, forecast_meta = load_forecast_trends(
        dataset,
        mmpd_run=mmpd_run,
        datasets_root=datasets_root,
        ckpt_root=ckpt_root,
        allow_fallback_binary=allow_fallback_binary,
    )

    ds_out = output_dir / dataset
    compare_path = ds_out / "horizon_trend_four_panel.png"
    panels = [
        (PANEL_SPECS[0][0], gt_train, PANEL_SPECS[0][1]),
        (PANEL_SPECS[1][0], gt_test, PANEL_SPECS[1][1]),
        (PANEL_SPECS[2][0], binary_trends, PANEL_SPECS[2][1]),
        (PANEL_SPECS[3][0], mmpd_trends, PANEL_SPECS[3][1]),
    ]
    plot_four_panel_compare(dataset=dataset, panels=panels, output_path=compare_path)

    ds_meta: Dict[str, object] = {
        "dataset": dataset,
        "gt_train": trend_summary(gt_train),
        "gt_test": trend_summary(gt_test),
        "forecasts": forecast_meta,
        "four_panel": str(compare_path.relative_to(REPO_ROOT)),
    }

    if write_gt_splits:
        from utils.analyze_gt_horizon_trend_baseline import plot_gt_histogram

        for split, trends in (("train", gt_train), ("test", gt_test)):
            summary = trend_summary(trends)
            plot_gt_histogram(
                dataset=dataset,
                split=split,
                trends=trends,
                summary=summary,
                output_path=ds_out / f"gt_{split}_horizon_trend_histogram.png",
            )

    bin_mu = "—" if binary_trends is None else f"{trend_summary(binary_trends)['mean']:+.3f}"
    mmpd_mu = "—" if mmpd_trends is None else f"{trend_summary(mmpd_trends)['mean']:+.3f}"
    print(
        f"[{dataset}] GT train μ={ds_meta['gt_train']['mean']:+.3f} "
        f"test μ={ds_meta['gt_test']['mean']:+.3f} | "
        f"binary={bin_mu} mmpd={mmpd_mu} "
        f"-> {compare_path.relative_to(REPO_ROOT)}"
    )
    return ds_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(ALL_DATASETS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mmpd-run", type=Path, default=DEFAULT_MMPD_RUN)
    parser.add_argument("--datasets-root", type=Path, default=REPO_ROOT / "results" / "datasets")
    parser.add_argument("--ckpt-root", type=Path, default=REPO_ROOT / "results" / "ckpts")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--allow-fallback-binary", action="store_true")
    parser.add_argument(
        "--write-gt-splits",
        action="store_true",
        help="Also write separate gt_train/gt_test histogram pngs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_meta: List[Dict[str, object]] = []
    skipped: List[str] = []
    for dataset in datasets:
        try:
            meta = process_dataset(
                dataset,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                mmpd_run=args.mmpd_run,
                datasets_root=args.datasets_root,
                ckpt_root=args.ckpt_root,
                allow_fallback_binary=args.allow_fallback_binary,
                write_gt_splits=args.write_gt_splits,
            )
            all_meta.append(meta)
        except Exception as exc:
            print(f"[skip] {dataset}: {exc}", file=sys.stderr)
            skipped.append(dataset)

    summary_path = args.output_dir / "horizon_trend_full_compare_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"datasets": all_meta, "skipped": skipped}, f, indent=2)

    print(f"Wrote {len(all_meta)} datasets -> {summary_path.relative_to(REPO_ROOT)}")
    if skipped:
        print(f"Skipped: {', '.join(skipped)}", file=sys.stderr)


if __name__ == "__main__":
    main()
