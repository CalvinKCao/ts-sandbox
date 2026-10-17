#!/usr/bin/env python3
"""Shared H=720 in-horizon random crops for table texture + disc.

Old L=8 sliding on T=96 was 89 windows/series; L=16 was 81. New protocol: keep
that budget, but sample K starts from anywhere in a native H=720 forecast.
Same starts across models when window_index aligns (shared seed).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

HORIZON = 720
SEED = 20260812
LOOKBACK = 336
# 96 - 8 + 1 = 89; 96 - 16 + 1 = 81.
K_BY_LENGTH = {8: 89, 16: 81}
# Unique-seg U = V×T on paper val∪test only (train excluded). ETTm1 == ETTm2:
# n_val=10801, n_test=10801, V=7 → U_valtest=151214. Cap is half of that.
ETT_REF_DATASET = "ETTm1"
ETTM_VALTEST_WINDOWS = 21602
ETTM_VALTEST_V = 7
ETTM_VALTEST_U = ETTM_VALTEST_WINDOWS * ETTM_VALTEST_V  # 151214
ETTM_VALTEST_U_CAP = ETTM_VALTEST_U // 2  # 75607
DEFAULT_INDEX_PATH = (
    "results/table_texture_packs/kept_u_hz720_ettm1_valtest50_seed20260812.json"
)


def n_windows_for_univariate_budget(
    n_packed: int, n_variates: int, max_univariate_rows: int
) -> int:
    """Keep N = min(packed, round(U_cap / V)) unique-seg parent windows."""
    if n_packed < 1:
        raise ValueError(f"n_packed must be >= 1, got {n_packed}")
    if n_variates < 1:
        raise ValueError(f"n_variates must be >= 1, got {n_variates}")
    if max_univariate_rows < 1:
        raise ValueError(f"max_univariate_rows must be >= 1, got {max_univariate_rows}")
    n_keep = max(1, int(max_univariate_rows) // int(n_variates))
    return min(n_keep, n_packed)


def pick_origin_rows(n_packed: int, n_keep: int, seed: int) -> np.ndarray:
    """Same RNG as texture crop_origin_windows(..., sample='random')."""
    if n_keep < 1:
        raise ValueError(f"n_keep must be >= 1, got {n_keep}")
    if n_keep > n_packed:
        raise ValueError(f"n_keep={n_keep} exceeds packed={n_packed}")
    if n_keep == n_packed:
        return np.arange(n_packed, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return rng.choice(n_packed, size=int(n_keep), replace=False)


def crops_per_horizon(length: int, override: int | None = None) -> int:
    if override is not None:
        if override < 1:
            raise ValueError(f"crops_per_horizon must be >= 1, got {override}")
        return int(override)
    if int(length) not in K_BY_LENGTH:
        raise ValueError(
            f"no default K for L={length}; expected one of {sorted(K_BY_LENGTH)}"
        )
    return int(K_BY_LENGTH[int(length)])


def require_horizon(values: np.ndarray, *, source: str = "array", horizon: int = HORIZON) -> np.ndarray:
    """(N, V, H) with H exactly `horizon`. Never prefix-slice a shorter dump."""
    values = np.asarray(values)
    if values.ndim != 3:
        raise ValueError(f"{source}: expected (N,V,H), got {values.shape}")
    h = int(values.shape[-1])
    if h == 96 and int(horizon) != 96:
        raise ValueError(
            f"{source}: H=96 dump {tuple(values.shape)}; refuse to slice. Need H={horizon}."
        )
    if h != int(horizon):
        raise ValueError(
            f"{source}: horizon {h} != {horizon} in {tuple(values.shape)}"
        )
    return values


def crop_starts(
    window_index: int,
    *,
    horizon: int,
    length: int,
    k: int,
    seed: int,
) -> np.ndarray:
    """K unique start offsets in [0, horizon-length] for one origin window.

    SeedSequence mixes campaign seed + window_index + L + K + H so Binary /
    iTrans / MMPD draw the same starts when they share window_index.
    """
    horizon = int(horizon)
    length = int(length)
    k = int(k)
    if length < 1 or length > horizon:
        raise ValueError(f"length={length} outside 1..{horizon}")
    max_start = horizon - length + 1
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > max_start:
        raise ValueError(f"k={k} exceeds max_start={max_start} for L={length} H={horizon}")
    ss = np.random.SeedSequence(
        [int(seed), int(window_index), length, k, horizon]
    )
    rng = np.random.default_rng(ss)
    return np.sort(rng.choice(max_start, size=k, replace=False).astype(np.int64))


def crop_starts_batch(
    window_indices: np.ndarray,
    *,
    horizon: int,
    length: int,
    k: int,
    seed: int,
) -> np.ndarray:
    idx = np.asarray(window_indices, dtype=np.int64).reshape(-1)
    return np.stack(
        [
            crop_starts(int(wi), horizon=horizon, length=length, k=k, seed=seed)
            for wi in idx
        ],
        axis=0,
    )


def take_horizon_crops(values: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    """values (N,V,H) + starts (N,K) -> (N*K, V, L) in origin-major, crop-second order."""
    y = np.asarray(values)
    st = np.asarray(starts, dtype=np.int64)
    if y.ndim != 3:
        raise ValueError(f"expected (N,V,H), got {y.shape}")
    if st.ndim != 2 or st.shape[0] != y.shape[0]:
        raise ValueError(f"starts {st.shape} incompatible with values {y.shape}")
    n, n_var, horizon = y.shape
    k = int(st.shape[1])
    length = int(length)
    if int(st.min(initial=0)) < 0 or int(st.max(initial=0)) + length > horizon:
        raise ValueError(
            f"crop out of range: starts[{st.min()},{st.max()}] L={length} H={horizon}"
        )
    n_idx = np.arange(n, dtype=np.int64)[:, None, None, None]
    v_idx = np.arange(n_var, dtype=np.int64)[None, :, None, None]
    t_idx = st[:, None, :, None] + np.arange(length, dtype=np.int64)[None, None, None, :]
    gathered = y[n_idx, v_idx, t_idx]  # (N, V, K, L)
    return np.ascontiguousarray(
        np.transpose(gathered, (0, 2, 1, 3)).reshape(n * k, n_var, length)
    )


def load_unique_seg_index(path: str | Path) -> dict[str, Any]:
    """Load the texture sibling keep-index (cap_u at top level)."""
    index_path = Path(path)
    if not index_path.is_file():
        raise FileNotFoundError(f"unique-seg index missing: {index_path}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    cap = payload.get("cap_u", (payload.get("protocol") or {}).get("U_cap") if isinstance(payload.get("protocol"), dict) else None)
    if int(cap or 0) != int(ETTM_VALTEST_U_CAP):
        raise ValueError(
            f"{index_path}: cap_u={cap} != {ETTM_VALTEST_U_CAP} "
            "(50% of ETTm1 val∪test U; train excluded)"
        )
    proto = str(payload.get("protocol") or "")
    train_excl = int(payload.get("T_train_excluded") or 0) > 0 or "train excluded" in proto.lower()
    if not train_excl:
        raise ValueError(f"{index_path}: keep-index must exclude train from the ETTm1 cap")
    if "datasets" not in payload:
        raise ValueError(f"{index_path}: missing datasets")
    return payload


def rows_for_kept_window_indices(
    pack_indices: np.ndarray,
    kept_indices: np.ndarray,
    *,
    dataset: str,
) -> np.ndarray:
    """Map kept window_index values onto pack rows, preserving index-file order."""
    pack_indices = np.asarray(pack_indices, dtype=np.int64).reshape(-1)
    kept_indices = np.asarray(kept_indices, dtype=np.int64).reshape(-1)
    lookup = {int(i): r for r, i in enumerate(pack_indices.tolist())}
    missing = [int(i) for i in kept_indices.tolist() if int(i) not in lookup]
    if missing:
        raise KeyError(
            f"{dataset}: unique-seg index has {len(missing)} window_indices "
            f"absent from pack (e.g. {missing[:8]})"
        )
    return np.asarray([lookup[int(i)] for i in kept_indices.tolist()], dtype=np.int64)

