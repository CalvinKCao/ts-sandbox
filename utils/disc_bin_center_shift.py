"""Centered bin-index mean shift for discriminator candidates (no variance scaling).

Applies to series **after** lattice snap onto the window-specific H-row
patch-refine ladder (binary dataset-z). Replaces per-slice ``zscore_time`` on
the ordinal disc path: additive bin offset only.

Mean is over the **disc candidate slice length L** (e.g. 8/16/32), not the
full forecast horizon. Call once per extracted L-slice in
``UnivariateRealVsFakeDataset`` / ``HorizonSliceDataset``; do not pre-shift
full-H packs before slicing.

Shapes: ``values`` ``(N, V, L)`` and ``legal_levels`` ``(N, V, H)`` — including
the univariate disc case ``(1, 1, L)`` + ``(1, 1, H)``.

Transform (“middle bin is zero”):
1. Map each snapped value to nearest legal-level **raw** bin index in ``[0, H)``.
2. ``center_index`` = ladder row whose level is closest to ``0.0`` dataset-z
   (fallback ``H // 2`` if the argmin is ambiguous / empty).
3. ``centered_idx = raw_idx - center_index`` (0 means the level nearest to 0.0).
4. Mean over the **slice length L** for that variate only
   (``reduce="per_variate"`` — required for univariate disc). Optional
   ``joint`` keeps one shift across variates for diagnostics.
5. ``shift = round(mean(centered_idx))`` (integer by default).
6. ``centered_idx' = centered_idx - shift``; ``raw' = clip(centered_idx' +
   center_index, 0, H-1)``; map ``raw'`` back to ladder levels.
7. Apply identically to GT, binary, and MMPD. Do **not** divide by std.

Univariate disc already feeds each variate as an independent sample; this
helper’s default per-variate mean matches that protocol.
"""

from __future__ import annotations

from typing import Dict, Literal, Tuple

import numpy as np

ReduceMode = Literal["per_variate", "joint"]


def nearest_bin_indices(values: np.ndarray, legal_levels: np.ndarray) -> np.ndarray:
    """Nearest ladder-row index for each value. Shapes: (N,V,L), (N,V,H) → (N,V,L)."""
    vals = np.asarray(values, dtype=np.float32)
    levels = np.asarray(legal_levels, dtype=np.float32)
    if vals.ndim != 3 or levels.ndim != 3:
        raise ValueError(f"expected (N,V,L)/(N,V,H), got {vals.shape}/{levels.shape}")
    if vals.shape[:2] != levels.shape[:2]:
        raise ValueError(f"N,V mismatch: {vals.shape} vs {levels.shape}")
    if not (np.isfinite(vals).all() and np.isfinite(levels).all()):
        raise ValueError("non-finite values/levels")
    delta = np.abs(vals[..., None] - levels[:, :, None, :])
    return np.argmin(delta, axis=-1).astype(np.int64)


def center_bin_index(legal_levels: np.ndarray) -> np.ndarray:
    """Per-(N,V) ladder row closest to dataset-z 0.0. Shape (N,V)."""
    levels = np.asarray(legal_levels, dtype=np.float32)
    if levels.ndim != 3:
        raise ValueError(f"expected (N,V,H), got {levels.shape}")
    n_rows = int(levels.shape[-1])
    if n_rows <= 0:
        raise ValueError("empty ladder")
    # Argmin |level - 0|; if all equal distance somehow, prefer H//2 via stable tie
    # by adding a tiny preference toward mid when distances tie.
    dist = np.abs(levels)
    mid = n_rows // 2
    # Break exact ties toward mid without changing unique argmins.
    tie_break = np.abs(np.arange(n_rows, dtype=np.float32) - float(mid)) * 1e-12
    dist = dist + tie_break[None, None, :]
    return np.argmin(dist, axis=-1).astype(np.int64)


def _mean_centered(
    centered: np.ndarray,
    *,
    reduce: ReduceMode,
) -> np.ndarray:
    """Broadcastable mean of centered indices, same shape as ``centered``."""
    if reduce == "per_variate":
        mean = centered.astype(np.float64).mean(axis=-1, keepdims=True)
        return np.broadcast_to(mean, centered.shape).astype(np.float64)
    if reduce == "joint":
        mean = centered.astype(np.float64).reshape(centered.shape[0], -1).mean(axis=-1)
        return mean[:, None, None].astype(np.float64)
    raise ValueError(f"unknown reduce={reduce!r}; expected 'per_variate' or 'joint'")


def bin_center_shift(
    values: np.ndarray,
    legal_levels: np.ndarray,
    *,
    reduce: ReduceMode = "per_variate",
    integer_shift: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Zero-mean in centered bin coords; remap to ladder levels. Additive only.

    If fakes are mostly “right shape, wrong level,” this + candidate_only L=8
    can erase the usable cue — deliberate for texture-focused protocol.
    """
    vals = np.asarray(values, dtype=np.float32)
    levels = np.asarray(legal_levels, dtype=np.float32)
    n_rows = int(levels.shape[-1])
    if n_rows <= 0:
        raise ValueError("empty ladder")

    raw = nearest_bin_indices(vals, levels)
    center = center_bin_index(levels)  # (N,V) — ladder row nearest dataset-z 0
    center_b = center[:, :, None]
    centered = raw.astype(np.int64) - center_b
    mean_c = _mean_centered(centered, reduce=reduce)
    if integer_shift:
        shift = np.rint(mean_c).astype(np.int64)
        centered_new = centered - shift
    else:
        shift = mean_c
        centered_new = np.rint(centered.astype(np.float64) - shift).astype(np.int64)

    raw_new = centered_new + center_b
    clamped = np.clip(raw_new, 0, n_rows - 1)
    n_clamped = int(np.sum(clamped != raw_new))
    # Remap shifted bin indices back onto the same window-specific ladder.
    shifted = np.take_along_axis(levels[:, :, None, :], clamped[..., None], axis=-1)[..., 0]

    raw_after = nearest_bin_indices(shifted, levels)
    centered_after = raw_after.astype(np.int64) - center_b
    diff_before = np.diff(centered.astype(np.float64), axis=-1)
    diff_after = np.diff(centered_after.astype(np.float64), axis=-1)
    return shifted.astype(np.float32), {
        "reduce": 0.0 if reduce == "per_variate" else 1.0,
        "integer_shift": 1.0 if integer_shift else 0.0,
        "n_rows": float(n_rows),
        "mean_center_index": float(center.mean()),
        "mean_centered_before": float(centered.mean()),
        "mean_centered_after": float(centered_after.mean()),
        "mean_abs_shift": float(np.abs(shift).mean()),
        "n_clamped": float(n_clamped),
        "frac_clamped": float(n_clamped) / float(raw.size),
        "diff_std_before": float(diff_before.std()) if diff_before.size else 0.0,
        "diff_std_after": float(diff_after.std()) if diff_after.size else 0.0,
        "max_abs_diff_delta": float(np.max(np.abs(diff_after - diff_before))) if diff_before.size else 0.0,
    }


def _self_check() -> None:
    """Deterministic check: ``python -m utils.disc_bin_center_shift``."""
    n, v, h, t = 2, 2, 256, 5
    # Level 128 ≈ 0 on a symmetric linspace over [-3, 3].
    levels_1d = np.linspace(-3.0, 3.0, h, dtype=np.float32)
    levels = np.broadcast_to(levels_1d[None, None, :], (n, v, h)).copy()
    center = int(np.argmin(np.abs(levels_1d)))
    # Offset +5 from center: centered mean should be ~5 → shift 5 → mean ~0.
    rows = np.asarray([center + 3, center + 4, center + 5, center + 6, center + 7], dtype=np.int64)
    idx_full = np.broadcast_to(rows[None, None, :], (n, v, t)).copy()
    values = np.take_along_axis(levels[:, :, None, :], idx_full[..., None], axis=-1)[..., 0]

    out, stats = bin_center_shift(values, levels, reduce="per_variate", integer_shift=True)
    assert abs(stats["mean_centered_after"]) < 1.0, stats
    assert stats["n_clamped"] == 0.0, stats
    assert stats["max_abs_diff_delta"] == 0.0, stats
    idx_out = nearest_bin_indices(out, levels)
    expected = rows - 5  # mean centered was 5
    assert np.array_equal(idx_out, np.broadcast_to(expected[None, None, :], (n, v, t))), (
        idx_out, expected, center
    )

    out_j, stats_j = bin_center_shift(values, levels, reduce="joint", integer_shift=True)
    assert abs(stats_j["mean_centered_after"]) < 1.0, stats_j
    assert np.isfinite(out_j).all()
    print(
        f"[disc_bin_center_shift] self-check ok center={center} "
        f"mean_c {stats['mean_centered_before']:.3f}->{stats['mean_centered_after']:.3f} "
        f"diff_std {stats['diff_std_before']:.4g}->{stats['diff_std_after']:.4g}",
        flush=True,
    )


if __name__ == "__main__":
    _self_check()
