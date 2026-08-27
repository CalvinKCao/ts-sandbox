#!/usr/bin/env python3
"""Export canvas128 table subsets as paper-format CSVs for iTransformer / PatchTST.

Variate lists, strides, and window caps come from the lr10 binary campaign YAML
(``all_variates: true`` wins over leftover parent 4v index lists).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from models.diffusion_tsf.pipeline.config import load_experiment_config  # noqa: E402
from models.diffusion_tsf.train_multivariate_pipeline import (  # noqa: E402
    DATASET_REGISTRY,
    _load_dataset_array,
)

DEFAULT_SUBSET_YAML = (
    REPO / "configs" / "binary_window_norm_patch_refine_canvas128_p64x6_allv_randwin_lr10.yaml"
)

# Loader/freq only. Variate indices come from YAML.
LOADER_META = {
    "ETTh1": {"freq": "h", "loader": "ETTh1"},
    "ETTh2": {"freq": "h", "loader": "ETTh2"},
    "ETTm1": {"freq": "min", "loader": "ETTm1"},
    "ETTm2": {"freq": "min", "loader": "ETTm2"},
    "electricity": {"freq": "h", "loader": "custom"},
    "traffic": {"freq": "h", "loader": "custom"},
    "exchange_rate": {"freq": "d", "loader": "custom"},
    "weather": {"freq": "h", "loader": "custom"},
    "PeMS": {"freq": "h", "loader": "PEMS"},
    "solar_Alabama": {"freq": "min", "loader": "custom"},
    "illness": {"freq": "h", "loader": "custom"},
    "dynamic": {"freq": "h", "loader": "custom"},
}


def _synthetic_dates(n: int, freq: str) -> pd.DatetimeIndex:
    start = "2016-07-01 00:00:00"
    if freq in ("t", "min", "15min"):
        return pd.date_range(start, periods=n, freq="15min")
    if freq == "d":
        return pd.date_range(start, periods=n, freq="D")
    return pd.date_range(start, periods=n, freq="h")


def load_subset_specs(yaml_path: Path) -> dict:
    cfg = load_experiment_config(str(yaml_path))
    by_ds = (cfg.get("experiment") or {}).get("data_subset_by_dataset") or {}
    eval_caps = {}
    for phase in cfg.get("phases") or []:
        if phase.get("phase") == "staged_eval":
            eval_caps = dict(phase.get("eval_max_windows_by_dataset") or {})
            break
    specs = {}
    for name, spec in by_ds.items():
        if name not in LOADER_META:
            continue
        specs[name] = {
            **LOADER_META[name],
            "all_variates": bool(spec.get("all_variates", False)),
            "variate_indices": list(spec.get("variate_indices") or []),
            "train_stride": int(spec.get("train_stride", 1)),
            "val_stride": int(spec.get("val_stride", 1)),
            "test_stride": int(spec.get("test_stride", 1)),
            "train_max_windows": spec.get("train_max_windows"),
            "val_max_windows": spec.get("val_max_windows"),
            "eval_max_windows": eval_caps.get(name),
            "subset_id": spec.get("subset_id"),
        }
    if not specs:
        raise ValueError(f"{yaml_path}: no overlapping datasets with LOADER_META")
    return specs


def export_one(name: str, spec: dict, out_dir: Path) -> dict:
    rel, date_col, _ = DATASET_REGISTRY[name]
    path = str(REPO / "datasets" / rel)
    arr = _load_dataset_array(path, date_col)
    n_raw = int(arr.shape[1])
    if spec.get("all_variates"):
        idx = list(range(n_raw))
    else:
        idx = [int(i) for i in spec["variate_indices"]]
        if not idx:
            raise ValueError(f"{name}: empty variate_indices and all_variates is false")
    if max(idx) >= n_raw:
        raise ValueError(f"{name}: variate_indices {idx} out of range for V={n_raw}")
    sub = arr[:, idx].astype(np.float64)

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
        "all_variates": bool(spec.get("all_variates")),
        "variate_indices": idx,
        "train_stride": int(spec["train_stride"]),
        "val_stride": int(spec["val_stride"]),
        "test_stride": int(spec["test_stride"]),
        "train_max_windows": spec.get("train_max_windows"),
        "val_max_windows": spec.get("val_max_windows"),
        "eval_max_windows": spec.get("eval_max_windows"),
        "subset_id": spec.get("subset_id"),
        "freq": spec["freq"],
        "loader": spec["loader"],
        "source_path": path,
    }
    print(
        f"[ok] {name}: {out_path} shape={df.shape[0]}x{n_v} "
        f"all_variates={meta['all_variates']} "
        f"train_max_windows={meta['train_max_windows']}",
        flush=True,
    )
    return meta


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--subset-yaml",
        type=Path,
        default=DEFAULT_SUBSET_YAML,
        help="Binary campaign YAML whose data_subset_by_dataset to mirror",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated dataset names (default: all overlapping YAML keys)",
    )
    args = p.parse_args()
    specs = load_subset_specs(args.subset_yaml)
    if args.datasets.strip():
        names = [x.strip() for x in args.datasets.split(",") if x.strip()]
        missing = [n for n in names if n not in specs]
        if missing:
            raise KeyError(f"not in subset YAML/LOADER_META: {missing}")
    else:
        names = list(specs)
    out_dir = REPO / "temp" / "baselines_canvas128_subset" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "subset_meta.json"
    existing: dict = {}
    if meta_path.is_file():
        try:
            for row in json.loads(meta_path.read_text()):
                existing[row["dataset"]] = row
        except Exception:
            existing = {}
    for name in names:
        existing[name] = export_one(name, specs[name], out_dir)
    metas = list(existing.values())
    meta_path.write_text(json.dumps(metas, indent=2) + "\n")
    print(f"[done] wrote {meta_path} ({len(metas)} datasets)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
