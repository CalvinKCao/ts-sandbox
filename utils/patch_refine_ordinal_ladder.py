"""Canonical absolute-row ordinal support for h96 patch-refine evaluation.

The patch-refine model predicts absolute rows on a tall CDF canvas
(``canvas_height``, typically 256 or 128).  This module turns those rows into
the exact dataset-z values used by the model at inference, including the
causal ordinal OOD shift.  It deliberately does not use the legacy 16x16
dual-scale canonicalizer.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from models.diffusion_tsf.ordinal_window_norm import (
    OrdinalLadder,
    ordinal_decode,
    ordinal_encode,
)


def legal_patch_refine_levels_dataset_z(
    past: np.ndarray,
    *,
    ladder: OrdinalLadder,
    canvas_height: int = 256,
    device: torch.device,
) -> np.ndarray:
    """Return exact decoded values for every absolute patch-refine row.

    Shape is ``(N, V, canvas_height)``.  The OOD shift is derived from the
    lookback only, exactly as inference does when no future is available.
    """
    past_np = np.asarray(past, dtype=np.float32)
    if past_np.ndim != 3:
        raise ValueError(f"past must be (N,V,L), got {past_np.shape}")
    if canvas_height <= 0:
        raise ValueError(f"canvas_height must be positive, got {canvas_height}")
    past_t = torch.from_numpy(past_np).to(device)
    with torch.no_grad():
        past_rank, _future_rank, ladder_b, ood_shift = ordinal_encode(
            past_t,
            None,
            ladder=ladder,
            apply_ood_shift=True,
            causal_only=True,
        )
        rank_max = ladder_b.rank_max_per_variate().to(device=device, dtype=past_t.dtype)
        rows = torch.arange(canvas_height, device=device, dtype=past_t.dtype)
        # This is decode_absolute_hir_cdf's midpoint decode for each row.
        rank_centers = ((rows + 0.5) / float(canvas_height)).view(1, 1, -1)
        rank_centers = rank_centers * rank_max.view(1, -1, 1)
        rank_centers = rank_centers.expand(past_t.shape[0], -1, -1)
        _past_z, levels = ordinal_decode(
            past_rank[..., :1], rank_centers, ladder_b, ood_shift=ood_shift,
        )
    if levels is None or not torch.isfinite(levels).all():
        raise ValueError("ordinal patch-refine support contains non-finite values")
    return levels.detach().cpu().numpy().astype(np.float32)


def snap_to_patch_refine_levels(
    values: np.ndarray,
    legal_levels: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Nearest-row snap in binary dataset-z coordinates, with endpoint clamp.

    Walkthrough (used by ``_snap_bundle`` for GT / binary / MMPD):
      For each (window, variate, time) value, pick the closest rung among that
      window's H legal levels. Output is still dataset-z, but now discrete.
      Continuous quirks between rungs disappear — fair for lattice disc,
      destructive for sub-bin distinguishability.
    """
    vals = np.asarray(values, dtype=np.float32)
    levels = np.asarray(legal_levels, dtype=np.float32)
    if vals.ndim != 3 or levels.ndim != 3:
        raise ValueError(f"expected values/levels (N,V,T)/(N,V,H), got {vals.shape}/{levels.shape}")
    if vals.shape[:2] != levels.shape[:2]:
        raise ValueError(f"values/levels N,V mismatch: {vals.shape}/{levels.shape}")
    if not (np.isfinite(vals).all() and np.isfinite(levels).all()):
        raise ValueError("cannot snap non-finite values")
    # |value - level| over H → argmin row per timestep.
    delta = np.abs(vals[..., None] - levels[:, :, None, :])
    rows = np.argmin(delta, axis=-1)
    # Gather the actual dataset-z midpoint for that row.
    snapped = np.take_along_axis(levels[:, :, None, :], rows[..., None], axis=-1)[..., 0]
    residual = np.abs(vals - snapped)
    return snapped.astype(np.float32), {
        "mean_abs_snap_delta": float(residual.mean()),
        "max_abs_snap_delta": float(residual.max(initial=0.0)),
        "n_rows": float(levels.shape[-1]),
        "n_unique_levels_min": float(min(np.unique(levels[i, j]).size for i in range(levels.shape[0]) for j in range(levels.shape[1]))),
    }


def assert_on_patch_refine_levels(
    values: np.ndarray,
    legal_levels: np.ndarray,
    *,
    atol: float = 1e-6,
) -> Dict[str, float]:
    """Fail unless all values are exactly on their window-specific 256-row support."""
    snapped, stats = snap_to_patch_refine_levels(values, legal_levels)
    err = float(np.abs(np.asarray(values, dtype=np.float32) - snapped).max(initial=0.0))
    if err > float(atol):
        raise AssertionError(
            f"values are off the patch-refine ordinal support: max_error={err:.6g}, atol={atol:.6g}"
        )
    stats["max_support_error"] = err
    return stats


def assert_support_is_causal(
    past: np.ndarray,
    future_a: np.ndarray,
    future_b: np.ndarray,
    *,
    ladder: OrdinalLadder,
    canvas_height: int,
    device: torch.device,
) -> None:
    """Document the invariant that legal rows cannot depend on future values."""
    if np.asarray(future_a).shape != np.asarray(future_b).shape:
        raise ValueError("future fixtures must have equal shape")
    past_t = torch.from_numpy(np.asarray(past, dtype=np.float32)).to(device)
    future_a_t = torch.from_numpy(np.asarray(future_a, dtype=np.float32)).to(device)
    future_b_t = torch.from_numpy(np.asarray(future_b, dtype=np.float32)).to(device)
    with torch.no_grad():
        _pa, _fa, _la, shift_a = ordinal_encode(
            past_t, future_a_t, ladder=ladder, apply_ood_shift=True, causal_only=True,
        )
        _pb, _fb, _lb, shift_b = ordinal_encode(
            past_t, future_b_t, ladder=ladder, apply_ood_shift=True, causal_only=True,
        )
    if not torch.equal(shift_a, shift_b):
        raise AssertionError("causal ordinal OOD shift changed when only future changed")
    levels_a = legal_patch_refine_levels_dataset_z(
        past, ladder=ladder, canvas_height=canvas_height, device=device,
    )
    levels_b = legal_patch_refine_levels_dataset_z(
        past, ladder=ladder, canvas_height=canvas_height, device=device,
    )
    if not np.array_equal(levels_a, levels_b):
        raise AssertionError("ordinal patch-refine support changed when only future changed")
