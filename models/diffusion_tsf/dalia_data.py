"""DALIA Forecast100 tensors: multivariate ECG, PPG, and 3D accelerometer windows."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

DALIA_N_VARS: int = 5
DALIA_INPUT_STEPS: int = 100
DALIA_FORECAST_STEPS: int = 20
DALIA_CHANNEL_NAMES: List[str] = ["ecg", "ppg", "acc_x", "acc_y", "acc_z"]

# Forecast100 layout: X is (N, 5, 100) input, Y is (N, 5, 20) target (flattened on disk).
DALIA_DEFAULT_LOOKBACK: int = 80
DALIA_DEFAULT_FORECAST: int = 20


def dalia_window_lengths() -> Tuple[int, int]:
    return DALIA_DEFAULT_LOOKBACK, DALIA_DEFAULT_FORECAST


def resolve_dalia_pt_dir() -> str:
    env = os.environ.get("DALIA_PT_DIR")
    if env:
        return os.path.abspath(env)
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    return os.path.join(root, "DALIA")


@lru_cache(maxsize=1)
def load_dalia_tensors(pt_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``X`` (N, T_in, V) and ``Y`` (N, T_out, V) float32 arrays."""
    base = pt_dir or resolve_dalia_pt_dir()
    x_path = os.path.join(base, "Forecast100X.pt")
    y_path = os.path.join(base, "Forecast100Y.pt")
    if not os.path.isfile(x_path) or not os.path.isfile(y_path):
        raise FileNotFoundError(
            f"DALIA tensors not found under {base!r} "
            f"(expected Forecast100X.pt and Forecast100Y.pt; set DALIA_PT_DIR if needed)"
        )
    x_flat = torch.load(x_path, map_location="cpu", weights_only=False).numpy()
    y_flat = torch.load(y_path, map_location="cpu", weights_only=False).numpy()
    if x_flat.ndim != 2 or x_flat.shape[1] != DALIA_N_VARS * DALIA_INPUT_STEPS:
        raise ValueError(f"Unexpected Forecast100X shape {x_flat.shape}")
    if y_flat.ndim != 2 or y_flat.shape[1] != DALIA_N_VARS * DALIA_FORECAST_STEPS:
        raise ValueError(f"Unexpected Forecast100Y shape {y_flat.shape}")
    x = x_flat.reshape(-1, DALIA_N_VARS, DALIA_INPUT_STEPS).transpose(0, 2, 1)
    y = y_flat.reshape(-1, DALIA_N_VARS, DALIA_FORECAST_STEPS).transpose(0, 2, 1)
    return x.astype(np.float32), y.astype(np.float32)


class DaliaPrewindowedDataset(Dataset):
    """One Forecast100 row = one (lookback, horizon) multivariate window."""

    def __init__(
        self,
        sample_indices: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        variate_indices: List[int],
        lookback: int,
        horizon: int,
        lookback_overlap: int,
    ):
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)
        self.x = x
        self.y = y
        self.variate_indices = variate_indices
        self.lookback = lookback
        self.horizon = horizon
        self.lookback_overlap = lookback_overlap
        need = lookback + horizon
        total = DALIA_INPUT_STEPS + DALIA_FORECAST_STEPS
        if need > total:
            raise ValueError(
                f"DALIA window needs lookback+horizon={need} but only {total} steps "
                f"({DALIA_INPUT_STEPS} input + {DALIA_FORECAST_STEPS} forecast)"
            )

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, i: int):
        idx = int(self.sample_indices[i])
        data = np.concatenate([self.x[idx], self.y[idx]], axis=0)
        data = data[:, self.variate_indices]
        past = torch.from_numpy(data[: self.lookback].T)
        target_start = self.lookback - self.lookback_overlap
        target_end = self.lookback + self.horizon
        future = torch.from_numpy(data[target_start:target_end].T)
        return past, future


def _split_sample_indices(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_train = int(n * 0.7)
    n_test = int(n * 0.2)
    n_val = n - n_train - n_test
    train_idx = np.arange(0, n_train, dtype=np.int64)
    val_idx = np.arange(n_train, n_train + n_val, dtype=np.int64)
    test_idx = np.arange(n_train + n_val, n, dtype=np.int64)
    return train_idx, val_idx, test_idx


def load_dalia_dataset(
    variate_indices: Optional[List[int]] = None,
    lookback: int = DALIA_DEFAULT_LOOKBACK,
    horizon: int = DALIA_DEFAULT_FORECAST,
    stride: int = 1,
    test_stride: int = 1,
    lookback_overlap: int = 0,
    pt_dir: Optional[str] = None,
) -> Tuple[Dataset, Dataset, Dataset, Dict]:
    """Load DALIA Forecast100 with 70/10/20 sample splits and optional index stride."""
    if variate_indices is None:
        variate_indices = list(range(DALIA_N_VARS))
    x, y = load_dalia_tensors(pt_dir)
    n = len(x)
    train_idx, val_idx, test_idx = _split_sample_indices(n)

    train_flat = np.concatenate([x[train_idx].reshape(-1, DALIA_N_VARS), y[train_idx].reshape(-1, DALIA_N_VARS)])
    mean = train_flat.mean(axis=0, keepdims=True)
    std = train_flat.std(axis=0, keepdims=True) + 1e-8
    x = (x - mean) / std
    y = (y - mean) / std

    if stride > 1:
        train_idx = train_idx[::stride]
        val_idx = val_idx[::stride]
    if test_stride > 1:
        test_idx = test_idx[::test_stride]

    train_ds = DaliaPrewindowedDataset(
        train_idx, x, y, variate_indices, lookback, horizon, lookback_overlap,
    )
    val_ds = DaliaPrewindowedDataset(
        val_idx, x, y, variate_indices, lookback, horizon, lookback_overlap,
    )
    test_ds = DaliaPrewindowedDataset(
        test_idx, x, y, variate_indices, lookback, horizon, lookback_overlap,
    )
    mean_sel = mean[:, variate_indices]
    std_sel = std[:, variate_indices]
    return train_ds, val_ds, test_ds, {"mean": mean_sel, "std": std_sel}
