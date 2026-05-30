"""DALIA Forecast100: ECG, PPG, and 3D accelerometer windows under datasets/dalia/."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DALIA_N_VARS: int = 5
DALIA_INPUT_STEPS: int = 100
DALIA_FORECAST_STEPS: int = 20
DALIA_CHANNEL_NAMES: List[str] = ["ecg", "ppg", "acc_x", "acc_y", "acc_z"]
DALIA_CSV_NAME: str = "dalia.csv"
DALIA_PT_X: str = "Forecast100X.pt"
DALIA_PT_Y: str = "Forecast100Y.pt"

DALIA_DEFAULT_LOOKBACK: int = 80
DALIA_DEFAULT_FORECAST: int = 20

# Legacy repo-root folder (scratch layout); also datasets/DALIA from an earlier commit.
_LEGACY_DALIA_DIRS = ("DALIA", os.path.join("datasets", "DALIA"))


def dalia_window_lengths() -> Tuple[int, int]:
    return DALIA_DEFAULT_LOOKBACK, DALIA_DEFAULT_FORECAST


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))


def resolve_dalia_dir(datasets_dir: Optional[str] = None) -> str:
    """Canonical data dir: ``<datasets>/dalia`` (lowercase, like other benchmarks)."""
    env = os.environ.get("DALIA_DATA_DIR")
    if env:
        return os.path.abspath(env)
    if datasets_dir:
        return os.path.join(os.path.abspath(datasets_dir), "dalia")
    return os.path.join(_repo_root(), "datasets", "dalia")


def dalia_csv_path(datasets_dir: Optional[str] = None) -> str:
    return os.path.join(resolve_dalia_dir(datasets_dir), DALIA_CSV_NAME)


def _load_tensors_from_pt(x_path: str, y_path: str) -> Tuple[np.ndarray, np.ndarray]:
    x_flat = torch.load(x_path, map_location="cpu", weights_only=False).numpy()
    y_flat = torch.load(y_path, map_location="cpu", weights_only=False).numpy()
    if x_flat.ndim != 2 or x_flat.shape[1] != DALIA_N_VARS * DALIA_INPUT_STEPS:
        raise ValueError(f"Unexpected Forecast100X shape {x_flat.shape}")
    if y_flat.ndim != 2 or y_flat.shape[1] != DALIA_N_VARS * DALIA_FORECAST_STEPS:
        raise ValueError(f"Unexpected Forecast100Y shape {y_flat.shape}")
    x = x_flat.reshape(-1, DALIA_N_VARS, DALIA_INPUT_STEPS).transpose(0, 2, 1)
    y = y_flat.reshape(-1, DALIA_N_VARS, DALIA_FORECAST_STEPS).transpose(0, 2, 1)
    return x.astype(np.float32), y.astype(np.float32)


def _normalize_search_dir(path: str) -> str:
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(_repo_root(), path)
    return os.path.abspath(path)


def _dalia_pt_search_dirs(
    pt_dir: Optional[str] = None,
    datasets_dir: Optional[str] = None,
) -> List[str]:
    root = _repo_root()
    candidates = [resolve_dalia_dir(datasets_dir)]
    if pt_dir:
        candidates.append(_normalize_search_dir(pt_dir))
    for leg in _LEGACY_DALIA_DIRS:
        candidates.append(os.path.join(root, leg))
    seen: set = set()
    out: List[str] = []
    for d in candidates:
        d = os.path.abspath(d)
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _find_pt_pair(search_dirs: List[str]) -> Optional[Tuple[str, str]]:
    for base in search_dirs:
        x_path = os.path.join(base, DALIA_PT_X)
        y_path = os.path.join(base, DALIA_PT_Y)
        if os.path.isfile(x_path) and os.path.isfile(y_path):
            return x_path, y_path
    return None


def _pt_not_found_error(search_dirs: List[str]) -> FileNotFoundError:
    lines = [f"Could not find {DALIA_PT_X} and {DALIA_PT_Y}."]
    lines.append("Searched:")
    for base in search_dirs:
        xp = os.path.join(base, DALIA_PT_X)
        yp = os.path.join(base, DALIA_PT_Y)
        lines.append(
            f"  {base}: X={'yes' if os.path.isfile(xp) else 'no'} "
            f"Y={'yes' if os.path.isfile(yp) else 'no'}"
        )
    lines.append(
        f"Hint: find {_repo_root()} -name '{DALIA_PT_X}'  "
        "(or scp from a machine that still has DALIA/)"
    )
    return FileNotFoundError("\n".join(lines))


def convert_dalia_pt_to_csv(
    csv_path: str,
    pt_dir: Optional[str] = None,
    datasets_dir: Optional[str] = None,
    x_path: Optional[str] = None,
    y_path: Optional[str] = None,
) -> str:
    """Write ``dalia.csv`` from Forecast100 ``*.pt`` tensors. Returns ``csv_path``."""
    if x_path and y_path:
        pair = (os.path.abspath(x_path), os.path.abspath(y_path))
        if not (os.path.isfile(pair[0]) and os.path.isfile(pair[1])):
            raise FileNotFoundError(f"Missing tensor file(s): {pair[0]!r}, {pair[1]!r}")
    else:
        search_dirs = _dalia_pt_search_dirs(pt_dir, datasets_dir)
        pair = _find_pt_pair(search_dirs)
        if pair is None:
            raise _pt_not_found_error(search_dirs)
    x, y = _load_tensors_from_pt(*pair)
    n = len(x)
    steps = DALIA_INPUT_STEPS + DALIA_FORECAST_STEPS
    rows = np.zeros((n * steps, 2 + DALIA_N_VARS), dtype=np.float64)
    off = 0
    for wid in range(n):
        block = np.concatenate([x[wid], y[wid]], axis=0)
        for step in range(steps):
            rows[off] = [wid, step, *block[step]]
            off += 1
    cols = ["window_id", "step", *DALIA_CHANNEL_NAMES]
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    pd.DataFrame(rows, columns=cols).to_csv(csv_path, index=False)
    return csv_path


def ensure_dalia_csv(datasets_dir: Optional[str] = None) -> str:
    """Ensure ``datasets/dalia/dalia.csv`` exists; convert from ``*.pt`` if needed."""
    csv_path = dalia_csv_path(datasets_dir)
    if os.path.isfile(csv_path):
        return csv_path
    convert_dalia_pt_to_csv(csv_path, datasets_dir=datasets_dir)
    return csv_path


@lru_cache(maxsize=4)
def _load_dalia_csv_cached(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    required = {"window_id", "step", *DALIA_CHANNEL_NAMES}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
    n_windows = int(df["window_id"].max()) + 1
    x = np.zeros((n_windows, DALIA_INPUT_STEPS, DALIA_N_VARS), dtype=np.float32)
    y = np.zeros((n_windows, DALIA_FORECAST_STEPS, DALIA_N_VARS), dtype=np.float32)
    for wid in range(n_windows):
        w = df[df["window_id"] == wid].sort_values("step")
        data = w[DALIA_CHANNEL_NAMES].to_numpy(dtype=np.float32)
        if data.shape[0] != DALIA_INPUT_STEPS + DALIA_FORECAST_STEPS:
            raise ValueError(
                f"window {wid}: expected {DALIA_INPUT_STEPS + DALIA_FORECAST_STEPS} "
                f"steps, got {data.shape[0]}"
            )
        x[wid] = data[:DALIA_INPUT_STEPS]
        y[wid] = data[DALIA_INPUT_STEPS:]
    return x, y


def dalia_window_count(datasets_dir: Optional[str] = None) -> int:
    csv_path = ensure_dalia_csv(datasets_dir)
    df = pd.read_csv(csv_path, usecols=["window_id"])
    return int(df["window_id"].max()) + 1


@lru_cache(maxsize=1)
def load_dalia_tensors(datasets_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``X`` (N, T_in, V) and ``Y`` (N, T_out, V) float32 arrays from ``dalia.csv``."""
    csv_path = ensure_dalia_csv(datasets_dir)
    return _load_dalia_csv_cached(csv_path)


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
    datasets_dir: Optional[str] = None,
    pt_dir: Optional[str] = None,
) -> Tuple[Dataset, Dataset, Dataset, Dict]:
    """Load DALIA from ``datasets/dalia/dalia.csv`` (70/10/20 sample splits)."""
    del pt_dir  # legacy kwarg; tensors live in CSV now
    if variate_indices is None:
        variate_indices = list(range(DALIA_N_VARS))
    x, y = load_dalia_tensors(datasets_dir=datasets_dir)
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
