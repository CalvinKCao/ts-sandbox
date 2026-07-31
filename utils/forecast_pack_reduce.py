"""Reduce forecast packs to a single trajectory per window.

Packs from staged_eval / MMPD / ordinal disc materialize use
``samples`` with shape ``(N, V, S, H)``. Discriminator paths historically
took ``samples[:, :, 0, :]`` (one draw). Prefer ``prob_mean`` (mean over S)
for probabilistic comparisons; never silently fall back to ``deterministic``.
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

FAKE_AGG_CHOICES = ("prob_mean", "sample0", "deterministic")


def pack_index_key(pack: Mapping[str, np.ndarray]) -> str:
    if "indices" in pack:
        return "indices"
    if "window_indices" in pack:
        return "window_indices"
    raise KeyError("pack has neither 'indices' nor 'window_indices'")


def reduce_pack_forecast(
    pack: Mapping[str, np.ndarray],
    *,
    agg: str = "prob_mean",
) -> np.ndarray:
    """Return ``(N, V, H)`` forecast for the requested aggregation.

    ``prob_mean`` — mean over sample axis (requires ``samples`` with S>=1).
    ``sample0`` — first stochastic draw.
    ``deterministic`` — anchor / point forecast key.
    """
    mode = str(agg).strip().lower()
    if mode not in FAKE_AGG_CHOICES:
        raise ValueError(f"agg must be one of {FAKE_AGG_CHOICES}, got {agg!r}")

    if mode == "deterministic":
        for key in ("deterministic", "final_anchor", "anchor"):
            if key in pack:
                out = np.asarray(pack[key], dtype=np.float32)
                if out.ndim != 3:
                    raise ValueError(f"{key} must be (N,V,H), got {out.shape}")
                return out
        raise KeyError(
            "deterministic aggregation requested but pack has no "
            "deterministic/final_anchor/anchor array"
        )

    if "samples" not in pack:
        raise KeyError(f"{mode} aggregation requires pack['samples']")
    samples = np.asarray(pack["samples"], dtype=np.float32)
    if samples.ndim != 4 or samples.shape[2] < 1:
        raise ValueError(f"samples must be (N,V,S,H) with S>=1, got {samples.shape}")
    if mode == "sample0":
        return samples[:, :, 0, :].astype(np.float32, copy=False)
    # prob_mean
    if "sample_mean" in pack:
        mean = np.asarray(pack["sample_mean"], dtype=np.float32)
        if mean.shape == samples[:, :, 0, :].shape:
            return mean
    return samples.mean(axis=2).astype(np.float32)


def subset_pack_by_pool_indices(
    pack: Mapping[str, np.ndarray],
    pool_indices: np.ndarray,
    *,
    allow_missing: bool = False,
) -> dict:
    """Keep rows whose pool index is in ``pool_indices`` (order preserved)."""
    key = pack_index_key(pack)
    idx = np.asarray(pack[key], dtype=np.int64)
    want = np.asarray(pool_indices, dtype=np.int64)
    pos = {int(i): j for j, i in enumerate(idx.tolist())}
    rows = []
    missing = []
    for i in want.tolist():
        j = pos.get(int(i))
        if j is None:
            missing.append(int(i))
        else:
            rows.append(j)
    if missing and not allow_missing:
        raise KeyError(f"{len(missing)} pool indices missing from pack (e.g. {missing[:5]})")
    if not rows:
        raise ValueError("no overlapping pool indices between pack and request")
    rows_arr = np.asarray(rows, dtype=np.int64)
    n = int(idx.shape[0])
    out = {}
    for k, v in pack.items():
        if isinstance(v, np.ndarray) and v.shape[:1] == (n,):
            out[k] = v[rows_arr]
        else:
            out[k] = v
    return out


def assert_not_anchor_agg(agg: str) -> None:
    if str(agg).strip().lower() == "deterministic":
        raise ValueError(
            "refusing deterministic/anchor aggregation for a probabilistic disc path; "
            "use agg='prob_mean' (or 'sample0')"
        )
