#!/usr/bin/env python3
"""Calibrate per-dataset window_norm_low_var_unit_std for healthy window norm.

Scans train-split sliding windows (lb=336, hz=720, stride configurable) and binary-searches
the minimum unit_std with horizon clip rate below a budget (default 1%).

Example:
  python utils/calibrate_window_norm_unit_std.py --datasets ETTh1,electricity,exchange_rate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml
from numpy.lib.stride_tricks import sliding_window_view

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.analyze_window_norm_binning_diagnostics import (  # noqa: E402
    _extract_windows,
    _window_norm_center,
)

from models.diffusion_tsf.train_multivariate_pipeline import (  # noqa: E402
    _load_dataset_array,
    _paper_split_borders,
    _resolve_registry_path,
)

DEFAULT_OUTPUT = REPO_ROOT / "reports" / "window_norm_unit_std_calibration"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "base" / "binary_staged.yaml"
ZSCORE_EPS = 1e-8


def _dataset_zscore(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    std = np.maximum(std, ZSCORE_EPS)
    return (data - mean) / std, mean, std


def _healthy_window_std(
    past_std: np.ndarray,
    *,
    threshold: float,
    unit_std: float,
    std_floor: float,
) -> np.ndarray:
    std_floor_arr = np.maximum(past_std, std_floor)
    low_var = past_std < threshold
    flat = past_std <= std_floor
    return np.where(flat | low_var, unit_std, std_floor_arr)


def _horizon_clip_rate(
    past_z: np.ndarray,
    future_z: np.ndarray,
    *,
    threshold: float,
    unit_std: float,
    std_floor: float,
    center_mode: str,
    max_scale: float,
    image_height: int,
) -> float:
    center = _window_norm_center(past_z, center_mode)
    past_std = past_z.std(axis=-1, keepdims=True)
    std = _healthy_window_std(
        past_std, threshold=threshold, unit_std=unit_std, std_floor=std_floor,
    )
    future_norm = (future_z - center) / std
    return float((np.abs(future_norm) > max_scale).mean())


def _binary_search_unit_std(
    past_z: np.ndarray,
    future_z: np.ndarray,
    *,
    threshold: float,
    std_floor: float,
    center_mode: str,
    max_scale: float,
    image_height: int,
    clip_budget: float,
    lo: float = 0.05,
    hi: float = 2.0,
    steps: int = 40,
) -> Tuple[float, float]:
    best_unit = hi
    best_rate = _horizon_clip_rate(
        past_z, future_z,
        threshold=threshold, unit_std=hi, std_floor=std_floor,
        center_mode=center_mode, max_scale=max_scale, image_height=image_height,
    )
    if best_rate <= clip_budget:
        # search downward for minimum feasible unit_std
        left, right = lo, hi
        for _ in range(steps):
            mid = (left + right) / 2.0
            rate = _horizon_clip_rate(
                past_z, future_z,
                threshold=threshold, unit_std=mid, std_floor=std_floor,
                center_mode=center_mode, max_scale=max_scale, image_height=image_height,
            )
            if rate <= clip_budget:
                best_unit = mid
                best_rate = rate
                right = mid
            else:
                left = mid
        return round(best_unit, 4), best_rate

    # hi still clips too much — report hi anyway
    return round(hi, 4), best_rate


def calibrate_dataset(
    dataset: str,
    *,
    lookback: int,
    horizon: int,
    stride: int,
    threshold: float,
    std_floor: float,
    center_mode: str,
    max_scale: float,
    image_height: int,
    clip_budget: float,
) -> Dict[str, float]:
    path, date_col = _resolve_registry_path(dataset)
    data = _load_dataset_array(path, date_col)
    borders1, borders2 = _paper_split_borders(dataset, len(data), lookback + horizon)
    train = data[borders1[0] : borders2[0]]
    train_z, _, _ = _dataset_zscore(train.astype(np.float64))

    _, past_z, future_z = _extract_windows(
        train_z.astype(np.float32),
        lookback=lookback,
        horizon=horizon,
        stride=stride,
    )
    if past_z.shape[0] == 0:
        raise RuntimeError(f"{dataset}: no train windows for lb={lookback} hz={horizon}")

    unit_std, clip_rate = _binary_search_unit_std(
        past_z,
        future_z,
        threshold=threshold,
        std_floor=std_floor,
        center_mode=center_mode,
        max_scale=max_scale,
        image_height=image_height,
        clip_budget=clip_budget,
    )
    return {
        "dataset": dataset,
        "n_windows": int(past_z.shape[0]),
        "unit_std": unit_std,
        "horizon_frac_clipped_mean": clip_rate,
        "max_scale": max_scale,
        "lookback": lookback,
        "horizon": horizon,
        "stride": stride,
        "threshold": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="ETTh1,electricity,exchange_rate")
    parser.add_argument("--lookback", type=int, default=336)
    parser.add_argument("--horizon", type=int, default=720)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--clip-budget", type=float, default=0.01)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as f:
        exp = yaml.safe_load(f)["experiment"]
    std_floor = float(exp.get("window_norm_std_floor", 0.1))
    center_mode = str(exp.get("window_norm_center", "mean"))
    image_height = int(exp.get("image_height", 16))
    ms_map = dict(exp.get("max_scale_by_dataset") or {})

    datasets: List[str] = [d.strip() for d in args.datasets.split(",") if d.strip()]
    results = []
    by_dataset: Dict[str, float] = {}
    for ds in datasets:
        max_scale = float(ms_map.get(ds, exp.get("max_scale", 3.5)))
        row = calibrate_dataset(
            ds,
            lookback=args.lookback,
            horizon=args.horizon,
            stride=args.stride,
            threshold=args.threshold,
            std_floor=std_floor,
            center_mode=center_mode,
            max_scale=max_scale,
            image_height=image_height,
            clip_budget=args.clip_budget,
        )
        results.append(row)
        by_dataset[ds] = row["unit_std"]
        print(
            f"{ds}: unit_std={row['unit_std']:.4f} "
            f"clip={row['horizon_frac_clipped_mean']:.4f} "
            f"n={row['n_windows']}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    out_json = args.output / "calibration.json"
    out_yaml = args.output / "unit_std_by_dataset.yaml"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"results": results, "by_dataset": by_dataset}, f, indent=2)
    with out_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"window_norm_low_var_unit_std_by_dataset": by_dataset}, f)
    print(f"Wrote {out_json} and {out_yaml}")


if __name__ == "__main__":
    main()
