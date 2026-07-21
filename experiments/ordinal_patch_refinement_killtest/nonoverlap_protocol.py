"""Split-safe, non-overlapping 16-step window protocol for refinement training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.eval_mmpd_gaussian_anchor import load_tsf_pack_pool

HORIZON = 16


def nonoverlap_indices(starts, horizon: int = HORIZON):
    """Greedily retain all windows whose future intervals do not overlap."""
    kept, end = [], -10**18
    for index, start in enumerate(starts):
        start = int(start)
        if start >= end:
            kept.append(index)
            end = start + horizon
    return kept


def build_protocol(dataset: str, n_variates: int, lookback: int = 96):
    out = {"dataset": dataset, "lookback": lookback, "horizon": HORIZON, "splits": {}}
    for split in ("train", "val", "test"):
        _pool, starts, *_ = load_tsf_pack_pool(
            dataset, list(range(n_variates)), lookback=lookback, horizon=HORIZON,
            train_stride=1, test_stride=4, pack_splits=[split],
        )
        kept = nonoverlap_indices(starts)
        # Deterministic every-other selection is exactly 50% up to one odd window.
        chosen = kept[::2] if split == "train" else kept
        out["splits"][split] = {
            "raw_windows": len(starts), "nonoverlap_windows": len(kept),
            "selected_windows": len(chosen), "unique_patches": len(chosen) * HORIZON,
            "indices": chosen,
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ETTh1", "exchange_rate"])
    parser.add_argument("--output", type=Path, default=Path("results/ordinal_patch_refinement_killtest/nonoverlap_protocol.json"))
    args = parser.parse_args()
    result = {ds: build_protocol(ds, 7 if ds == "ETTh1" else 8) for ds in args.datasets}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for ds, protocol in result.items():
        print(ds, {name: values["unique_patches"] for name, values in protocol["splits"].items()})


if __name__ == "__main__": main()
