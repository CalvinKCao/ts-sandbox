"""Train/val/test window protocol for ordinal patch refinement.

Train: overlapping forecast windows with source stride 2 (larger set).
Val/test: non-overlapping futures (timeline-disjoint eval).

Patch-level OOB filtering (in-bounds 32x8 only) is applied later in the
materializer — this module only selects source windows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.eval_mmpd_gaussian_anchor import load_tsf_pack_pool

HORIZON = 16
TRAIN_STRIDE = 2
LOOKBACK = 96
DATASET_N_VARIATES = {
    "ETTh1": 7,
    "exchange_rate": 8,
    "electricity": 8,
    "traffic": 8,
}


def nonoverlap_indices(starts, horizon: int = HORIZON):
    """Greedily retain windows whose future intervals do not overlap."""
    kept, end = [], -10**18
    for index, start in enumerate(starts):
        start = int(start)
        if start >= end:
            kept.append(index)
            end = start + horizon
    return kept


def build_protocol(dataset: str, n_variates: int, lookback: int = LOOKBACK):
    out = {
        "dataset": dataset,
        "lookback": lookback,
        "horizon": HORIZON,
        "train_stride": TRAIN_STRIDE,
        "splits": {},
    }
    # Train: stride-2 overlapping windows from the pack loader.
    train_pool, train_starts, *_ = load_tsf_pack_pool(
        dataset, list(range(n_variates)), lookback=lookback, horizon=HORIZON,
        train_stride=TRAIN_STRIDE, test_stride=4, pack_splits=["train"],
    )
    train_idx = list(range(len(train_pool)))
    out["splits"]["train"] = {
        "raw_windows": len(train_starts),
        "selected_windows": len(train_idx),
        "selection": f"pack train_stride={TRAIN_STRIDE} (overlapping)",
        "indices": train_idx,
    }
    # Val/test: non-overlapping for clean held-out scoring.
    for split in ("val", "test"):
        _pool, starts, *_ = load_tsf_pack_pool(
            dataset, list(range(n_variates)), lookback=lookback, horizon=HORIZON,
            train_stride=1, test_stride=4, pack_splits=[split],
        )
        kept = nonoverlap_indices(starts)
        out["splits"][split] = {
            "raw_windows": len(starts),
            "nonoverlap_windows": len(kept),
            "selected_windows": len(kept),
            "selection": "nonoverlap futures",
            "indices": kept,
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ETTh1", "exchange_rate", "electricity", "traffic"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ordinal_patch_refinement_killtest/window_protocol.json"),
    )
    args = parser.parse_args()
    result = {}
    for ds in args.datasets:
        if ds not in DATASET_N_VARIATES:
            raise ValueError(f"unsupported dataset {ds!r}; expected {sorted(DATASET_N_VARIATES)}")
        result[ds] = build_protocol(ds, DATASET_N_VARIATES[ds])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Indices make the JSON huge; write a summary sibling for humans.
    summary = {
        ds: {
            "train_stride": protocol["train_stride"],
            "splits": {
                name: {k: v for k, v in vals.items() if k != "indices"}
                for name, vals in protocol["splits"].items()
            },
        }
        for ds, protocol in result.items()
    }
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for ds, protocol in result.items():
        counts = {name: vals["selected_windows"] for name, vals in protocol["splits"].items()}
        print(ds, counts)


if __name__ == "__main__":
    main()
