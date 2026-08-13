#!/usr/bin/env python3
"""Export canvas128 table subsets as paper-format CSVs for iTransformer / PatchTST.

Uses the same variate_indices as configs/base/binary_staged.yaml data_subset_by_dataset
for the nine leaderboard datasets. Writes date + selected feature columns (OT = last).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from models.diffusion_tsf.train_multivariate_pipeline import (  # noqa: E402
    DATASET_REGISTRY,
    _load_dataset_array,
    _resolve_registry_path,
)

# Mirror configs/base/binary_staged.yaml for the table datasets.
SUBSETS = {
    "ETTh1": {"variate_indices": [0, 1, 2, 3, 4, 5, 6], "train_stride": 1, "val_stride": 1, "test_stride": 1, "freq": "h", "loader": "ETTh1"},
    "ETTh2": {"variate_indices": [0, 1, 2, 3, 4, 5, 6], "train_stride": 1, "val_stride": 1, "test_stride": 1, "freq": "h", "loader": "ETTh2"},
    "ETTm1": {"variate_indices": [0, 1, 2, 3], "train_stride": 3, "val_stride": 3, "test_stride": 1, "freq": "min", "loader": "ETTm1"},
    "ETTm2": {"variate_indices": [0, 1, 2, 3, 4, 5, 6], "train_stride": 4, "val_stride": 4, "test_stride": 1, "freq": "min", "loader": "ETTm2"},
    "electricity": {"variate_indices": [0, 1, 2, 3], "train_stride": 1, "val_stride": 1, "test_stride": 1, "freq": "h", "loader": "custom"},
    "traffic": {"variate_indices": [0, 1, 2, 3], "train_stride": 1, "val_stride": 1, "test_stride": 1, "freq": "h", "loader": "custom"},
    "exchange_rate": {"variate_indices": [0, 1, 2, 3, 4, 5, 6, 7], "train_stride": 1, "val_stride": 1, "test_stride": 1, "freq": "d", "loader": "custom"},
    # PEMS → Dataset_PEMS_CSV (60/20/20); not Dataset_Custom 70/10/20.
    "PeMS": {"variate_indices": [0, 1, 2, 3, 4, 5, 6], "train_stride": 1, "val_stride": 1, "test_stride": 1, "freq": "h", "loader": "PEMS"},
    "solar_Alabama": {"variate_indices": [0, 1], "train_stride": 1, "val_stride": 1, "test_stride": 1, "freq": "min", "loader": "custom"},
}


def _synthetic_dates(n: int, freq: str) -> pd.DatetimeIndex:
    # Paper loaders only need a parseable date column for stamps.
    start = "2016-07-01 00:00:00"
    if freq in ("t", "min", "15min"):
        return pd.date_range(start, periods=n, freq="15min")
    if freq == "d":
        return pd.date_range(start, periods=n, freq="D")
    return pd.date_range(start, periods=n, freq="h")


def export_one(name: str, out_dir: Path) -> dict:
    spec = SUBSETS[name]
    path, date_col = _resolve_registry_path(name)
    arr = _load_dataset_array(path, date_col)
    idx = list(spec["variate_indices"])
    if max(idx) >= arr.shape[1]:
        raise ValueError(f"{name}: variate_indices {idx} out of range for V={arr.shape[1]}")
    sub = arr[:, idx].astype(np.float64)

    # Prefer real dates when CSV has them.
    dates = None
    if path.endswith(".csv") and date_col:
        try:
            df_raw = pd.read_csv(path)
            if date_col in df_raw.columns and len(df_raw) == len(sub):
                dates = pd.to_datetime(df_raw[date_col])
        except Exception:
            dates = None
    if dates is None:
        dates = _synthetic_dates(len(sub), spec["freq"])

    n_v = sub.shape[1]
    cols = [f"V{i}" for i in range(n_v - 1)] + ["OT"]
    df = pd.DataFrame(sub, columns=cols)
    df.insert(0, "date", dates)
    out_path = out_dir / f"{name}.csv"
    df.to_csv(out_path, index=False)
    meta = {
        "dataset": name,
        "csv": str(out_path.name),
        "n_rows": int(len(df)),
        "n_variates": int(n_v),
        "variate_indices": idx,
        "train_stride": int(spec["train_stride"]),
        "val_stride": int(spec["val_stride"]),
        "test_stride": int(spec["test_stride"]),
        "freq": spec["freq"],
        "loader": spec["loader"],
        "source_path": path,
    }
    print(f"[ok] {name}: {out_path} shape={df.shape[0]}x{n_v}", flush=True)
    return meta


def main() -> int:
    out_dir = REPO / "temp" / "baselines_canvas128_subset" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    metas = [export_one(name, out_dir) for name in SUBSETS]
    (out_dir / "subset_meta.json").write_text(json.dumps(metas, indent=2) + "\n")
    print(f"[done] wrote {out_dir / 'subset_meta.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
