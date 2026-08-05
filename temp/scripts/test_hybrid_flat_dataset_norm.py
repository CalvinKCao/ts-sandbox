#!/usr/bin/env python3
"""Contract checks for ETTh2 hybrid flat dataset-norm."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from utils.hybrid_flat_dataset_norm import (
    apply_skip_window_norm_mask,
    build_hybrid_affine_scales,
    detect_flat_variate_mask,
)
from utils.patch_refine_value_grid import window_normalization_stats
from types import SimpleNamespace


def test_etth2_flat_detection():
    df = pd.read_csv(REPO / "datasets/ETT-small/ETTh2.csv")
    cols = [c for c in df.columns if c != "date"]
    train = df[cols].to_numpy(dtype=np.float64)[: 12 * 30 * 24]
    mask, frac = detect_flat_variate_mask(train, frac_threshold=0.5)
    assert mask.tolist() == [False, False, False, False, False, True, False], mask
    assert frac[5] > 0.5
    hybrid = build_hybrid_affine_scales(train, lookback=336, max_scale=5.2)
    assert hybrid["flat_mask"][5]
    # Coverage scale tighter than emp std for LULL.
    assert hybrid["std"][0, 5] < hybrid["emp_std"][0, 5]
    assert abs(hybrid["std"][0, 5] - 5.593996534511405) < 1e-6


def test_skip_mask_identity_stats():
    past = torch.randn(2, 7, 336)
    center = past.mean(-1, keepdim=True)
    std = past.std(-1, keepdim=True).clamp_min(0.1)
    mask = [False, False, False, False, False, True, False]
    c2, s2 = apply_skip_window_norm_mask(center, std, mask)
    assert torch.allclose(c2[:, 5, :], torch.zeros_like(c2[:, 5, :]))
    assert torch.allclose(s2[:, 5, :], torch.ones_like(s2[:, 5, :]))
    assert torch.allclose(c2[:, 0, :], center[:, 0, :])


def test_window_norm_stats_respects_mask():
    past = torch.randn(1, 7, 336)
    cfg = SimpleNamespace(
        use_ordinal_window_norm=False,
        use_window_normalization=True,
        window_norm_center="mean",
        window_norm_std_floor=0.1,
        window_norm_low_var_threshold=0.0,
        skip_window_norm_variate_mask=[False, False, False, False, False, True, False],
    )
    center, std = window_normalization_stats(past, cfg)
    assert float(center[0, 5, 0]) == 0.0
    assert float(std[0, 5, 0]) == 1.0


def test_yaml_loads():
    from models.diffusion_tsf.pipeline.config import load_experiment_config
    from models.diffusion_tsf.pipeline.state import PipelineState

    cfg = load_experiment_config(
        "configs/binary_window_norm_patch_refine_canvas128_p64x6_etth2_hybrid_flat_dsnorm.yaml"
    )
    state = PipelineState.from_config(cfg)
    assert state.hybrid_flat_dataset_norm is True
    assert float(state.max_scale_by_dataset["ETTh2"]) == 5.2


if __name__ == "__main__":
    test_etth2_flat_detection()
    test_skip_mask_identity_stats()
    test_window_norm_stats_respects_mask()
    test_yaml_loads()
    print("ok")
