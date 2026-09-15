"""Dump RealTS PWB+linear windows as a stitched CSV for PatchTST / iTransformer / MMPD."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

PWB_LINEAR = ["PWB", "linear"]
SYNTH_SAMPLES_FULL = 4096
SYNTH_SAMPLES_SMOKE = 8


def pwb_linear_csv_name(dataset: str, seq_len: int, pred_len: int) -> str:
    return f"{dataset}_pwb_linear_lb{int(seq_len)}_hz{int(pred_len)}.csv"


def pwb_linear_n_windows(*, smoke: bool, train_frac: float = 0.7) -> int:
    """Stitched windows so a 70/10/20 split still has >= target train examples.

    Dataset_Custom / MMPD exchange splits are ~70% train. Stride L+H makes one
    stitched block one window, so val also needs >=1 block (smoke -> 12).
    """
    target = SYNTH_SAMPLES_SMOKE if smoke else SYNTH_SAMPLES_FULL
    if train_frac <= 0 or train_frac > 1:
        raise ValueError(f"train_frac={train_frac}")
    return max(int(target), int(math.ceil(int(target) / float(train_frac))))


def write_pwb_linear_csv(
    out_csv: Path,
    *,
    n_variates: int,
    seq_len: int,
    pred_len: int,
    n_windows: int,
    columns: Sequence[str],
    seed: int = 42,
) -> Path:
    """Concatenate independent RealTS windows of length L+H. Train with stride L+H."""
    from models.diffusion_tsf.realts import RealTS

    if len(columns) != n_variates:
        raise ValueError(f"expected {n_variates} columns, got {list(columns)}")
    if n_windows < 1:
        raise ValueError("n_windows must be >= 1")
    ds = RealTS(
        num_samples=int(n_windows),
        lookback_length=int(seq_len),
        forecast_length=int(pred_len),
        seed=int(seed),
        num_variables=int(n_variates),
        lookback_overlap=0,
        generator_names=list(PWB_LINEAR),
        skip_cross_var_aug=False,
    )
    chunks = []
    for i in range(int(n_windows)):
        past, future = ds[i]
        # (V, T) -> (T, V); overlap=0 so concat is the full L+H series.
        window = torch_cat_vt(past, future)
        chunks.append(window)
    values = np.concatenate(chunks, axis=0)
    # Dummy dates for Dataset_Custom. Daily pandas date_range overflows year 10000
    # at H>=192 (5852*(L+H) days). Hourly pandas date_range overflows int64 ns
    # at H=720. numpy datetime64[h] stays ~year 2705.
    origin = np.datetime64("2000-01-01T00", "h")
    stamp = origin + np.arange(values.shape[0], dtype=np.int64)
    date_col = np.char.replace(np.char.replace(stamp.astype("datetime64[m]").astype(str), "-", "/"), "T", " ")
    frame = pd.DataFrame(values, columns=list(columns))
    frame.insert(0, "date", date_col)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)
    return out_csv


def torch_cat_vt(past, future) -> np.ndarray:
    import torch

    p = past.detach().cpu() if isinstance(past, torch.Tensor) else np.asarray(past)
    f = future.detach().cpu() if isinstance(future, torch.Tensor) else np.asarray(future)
    if hasattr(p, "numpy"):
        p = p.numpy()
        f = f.numpy()
    # (V, Tp) + (V, Tf) -> (Tp+Tf, V)
    seq = np.concatenate([np.asarray(p, dtype=np.float32), np.asarray(f, dtype=np.float32)], axis=1)
    return seq.T


def columns_from_csv(src: Path) -> list[str]:
    header = src.read_text(encoding="utf-8").splitlines()[0].split(",")
    cols = [c.strip() for c in header if c.strip() and c.strip().lower() != "date"]
    if not cols:
        raise ValueError(f"no value columns in {src}")
    return cols
