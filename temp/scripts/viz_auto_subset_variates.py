#!/usr/bin/env python3
"""List and plot auto-subset variates from configs/base/binary_staged.yaml.

For each dataset in experiment.data_subset_by_dataset, resolve subset_id /
variate_indices via resolve_data_subset, print a summary, and save a small
per-variate sample grid under ./temp/viz_auto_subset_variates/.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.pipeline.data_subset import resolve_data_subset  # noqa: E402
from models.diffusion_tsf.train_multivariate_pipeline import (  # noqa: E402
    DATASET_REGISTRY,
    _dataset_variate_names,
    _load_dataset_array,
    _resolve_registry_path,
    get_dataset_shape,
)

DEFAULT_CONFIG = REPO_ROOT / "configs/base/binary_staged.yaml"
DEFAULT_OUT = REPO_ROOT / "temp/viz_auto_subset_variates"
# Campaign-facing defaults; --all dumps every entry in the YAML table.
DEFAULT_DATASETS = ("ETTh1", "electricity", "traffic", "exchange_rate")
DEFAULT_LOOKBACK = 336
DEFAULT_HORIZON = 96


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML with experiment.data_subset_by_dataset (default: base/binary_staged)",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset names (default: ETTh1 electricity traffic exchange_rate)",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Include every dataset listed in data_subset_by_dataset",
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument(
        "--start",
        type=int,
        default=0,
        help="Window start index in the raw series (default: 0)",
    )
    return p.parse_args()


def _load_subset_table(config_path: Path) -> Dict[str, Any]:
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    exp = raw.get("experiment") or {}
    by_ds = exp.get("data_subset_by_dataset")
    if not isinstance(by_ds, dict) or not by_ds:
        raise ValueError(f"no experiment.data_subset_by_dataset in {config_path}")
    return by_ds


def _resolve_one(
    dataset: str,
    by_dataset: Dict[str, Any],
    window_stride: int = 1,
) -> Dict[str, Any]:
    if dataset not in DATASET_REGISTRY:
        raise KeyError(f"{dataset!r} not in DATASET_REGISTRY")
    raw_rows, raw_variates = get_dataset_shape(dataset)
    return resolve_data_subset(
        dataset_name=dataset,
        raw_rows=raw_rows,
        raw_variates=raw_variates,
        base_variate_indices=list(range(raw_variates)),
        default_window_stride=window_stride,
        policy={"data_subset_by_dataset": by_dataset},
    )


def _variate_labels(
    dataset: str,
    indices: Sequence[int],
) -> List[str]:
    path, date_col = _resolve_registry_path(dataset)
    names = _dataset_variate_names(path, date_col, get_dataset_shape(dataset)[1])
    labels = []
    for i in indices:
        name = names[i] if 0 <= i < len(names) else f"var_{i}"
        labels.append(f"[{i}] {name}")
    return labels


def _sample_window(
    data: np.ndarray,
    indices: Sequence[int],
    start: int,
    length: int,
) -> np.ndarray:
    """Return (n_variates, length) slice; fail if window does not fit."""
    if start < 0 or start + length > data.shape[0]:
        raise ValueError(
            f"window [{start}:{start + length}] out of range for T={data.shape[0]}"
        )
    cols = [int(i) for i in indices]
    bad = [c for c in cols if c < 0 or c >= data.shape[1]]
    if bad:
        raise ValueError(f"variate indices out of range V={data.shape[1]}: {bad}")
    return data[start : start + length, cols].T.copy()


def _plot_dataset(
    dataset: str,
    subset: Dict[str, Any],
    labels: Sequence[str],
    series: np.ndarray,
    lookback: int,
    out_path: Path,
) -> None:
    n = series.shape[0]
    ncols = min(2, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 2.4 * nrows), squeeze=False)
    t = np.arange(series.shape[1])
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.axis("off")
            continue
        ax.plot(t, series[i], lw=1.0, color="#1f4e79")
        if 0 < lookback < series.shape[1]:
            ax.axvline(lookback - 0.5, color="#888", ls="--", lw=0.8)
        ax.set_title(labels[i], fontsize=10)
        ax.set_xlabel("t")
        ax.grid(True, alpha=0.25)
    subset_id = subset["subset_id"]
    fig.suptitle(
        f"{dataset}  |  {subset_id}  |  n_variates={subset['n_variates']}  "
        f"| raw={subset['raw_variates']}v",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    by_dataset = _load_subset_table(args.config)
    if args.all:
        datasets = list(by_dataset.keys())
    elif args.datasets:
        datasets = list(args.datasets)
    else:
        datasets = [d for d in DEFAULT_DATASETS if d in by_dataset]
        missing = [d for d in DEFAULT_DATASETS if d not in by_dataset]
        if missing:
            raise KeyError(f"default datasets missing from YAML table: {missing}")

    win_len = int(args.lookback) + int(args.horizon)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    print(f"config: {args.config}")
    print(f"window: start={args.start} length={win_len} (lb={args.lookback}+hz={args.horizon})")
    print(f"out:    {out_dir}")
    print()
    print(f"{'dataset':<16} {'subset_id':<22} {'n':>3}  variates")
    print("-" * 90)

    for dataset in datasets:
        if dataset not in by_dataset:
            raise KeyError(f"{dataset!r} not in data_subset_by_dataset ({args.config})")
        subset = _resolve_one(dataset, by_dataset)
        indices = [int(i) for i in subset["variate_indices"]]
        raw_v = int(subset["raw_variates"])
        bad = [i for i in indices if i < 0 or i >= raw_v]
        if bad:
            # YAML stub (e.g. coverage_synth) can list more indices than raw cols.
            msg = (
                f"{dataset:<16} {subset['subset_id']:<22} "
                f"SKIP indices {bad} >= raw_variates={raw_v}"
            )
            print(msg)
            if not args.all:
                raise ValueError(msg.strip())
            continue
        labels = _variate_labels(dataset, indices)
        path, date_col = _resolve_registry_path(dataset)
        data = _load_dataset_array(path, date_col)
        series = _sample_window(data, indices, args.start, win_len)
        plot_path = out_dir / f"{dataset}_{subset['subset_id']}.png"
        _plot_dataset(dataset, subset, labels, series, args.lookback, plot_path)

        var_str = ", ".join(labels)
        print(
            f"{dataset:<16} {subset['subset_id']:<22} {subset['n_variates']:>3}  {var_str}"
        )
        rows.append(
            {
                "dataset": dataset,
                "subset_id": subset["subset_id"],
                "n_variates": subset["n_variates"],
                "variate_indices": indices,
                "variate_names": [lab.split("] ", 1)[-1] for lab in labels],
                "train_stride": subset["train_stride"],
                "sample_stride": subset["sample_stride"],
                "reason": subset.get("reason"),
                "plot": str(plot_path),
            }
        )

    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"config: {args.config}\n")
        f.write(f"window: start={args.start} length={win_len}\n\n")
        for r in rows:
            f.write(
                f"{r['dataset']}\t{r['subset_id']}\tn={r['n_variates']}\t"
                f"indices={r['variate_indices']}\tnames={r['variate_names']}\n"
                f"  plot: {r['plot']}\n"
            )
    print()
    print(f"wrote {len(rows)} plots + {summary_path}")


if __name__ == "__main__":
    main()
