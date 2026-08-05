"""The per-window value grid used by non-ordinal (window-norm) patch refinement.

Canvas128 leaves train with ``use_window_normalization=True`` /
``use_ordinal_window_norm=False`` and ``patch_refine_canvas_height=128``.
Absolute canvas rows map to dataset-z via past mean/std + ``max_scale`` —
same lattice as ``snap_to_unbounded_patch_refine_grid``, but clipped to the
finite ``[0, H)`` rows the model can emit.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch


def window_normalization_stats(
    past: torch.Tensor,
    config: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reproduce ``BinaryDiffusionForecast._normalize_sequence`` for value grids."""
    if bool(getattr(config, "use_ordinal_window_norm", False)):
        raise ValueError("patch-refine value grid is only defined for non-ordinal checkpoints")
    if not bool(getattr(config, "use_window_normalization", False)):
        return torch.zeros_like(past[..., :1]), torch.ones_like(past[..., :1])

    center_mode = str(getattr(config, "window_norm_center", "mean"))
    if center_mode == "mean":
        center = past.mean(dim=-1, keepdim=True)
    elif center_mode == "last":
        center = past[..., -1:]
    else:
        raise ValueError(f"unsupported window_norm_center={center_mode!r}")

    past_std = past.std(dim=-1, keepdim=True)
    std_floor = float(getattr(config, "window_norm_std_floor", 0.1))
    threshold = float(getattr(config, "window_norm_low_var_threshold", 0.0))
    if threshold <= 0.0:
        center_out, std_out = center, past_std.clamp_min(std_floor)
    else:
        default_unit = float(getattr(config, "window_norm_low_var_unit_std", 1.0))
        per_variate = getattr(config, "window_norm_low_var_unit_std_per_variate", None)
        if per_variate is None:
            unit = torch.full_like(past_std, default_unit)
        else:
            if len(per_variate) != past.shape[1]:
                raise ValueError("window_norm_low_var_unit_std_per_variate has wrong length")
            unit = torch.tensor(per_variate, device=past.device, dtype=past.dtype).view(1, -1, 1)
        std = past_std.clamp_min(std_floor)
        low_var = past_std < threshold
        flat = past_std <= std_floor
        center_out, std_out = center, torch.where(flat | low_var, unit, std)

    skip_mask = getattr(config, "skip_window_norm_variate_mask", None)
    if skip_mask is not None:
        from utils.hybrid_flat_dataset_norm import apply_skip_window_norm_mask

        center_out, std_out = apply_skip_window_norm_mask(center_out, std_out, skip_mask)
    return center_out, std_out


def normalized_grid_step(config: Any) -> float:
    height = int(getattr(config, "patch_refine_canvas_height", 256))
    max_scale = float(getattr(config, "max_scale"))
    if height <= 0 or max_scale <= 0.0:
        raise ValueError(f"invalid patch-refine grid: height={height}, max_scale={max_scale}")
    return 2.0 * max_scale / height


def legal_window_norm_patch_refine_levels_dataset_z(
    past: np.ndarray | torch.Tensor,
    config: Any,
) -> np.ndarray:
    """Finite H-row window-norm lattice midpoints in dataset-z. Shape ``(N, V, H)``.

    Row ``i`` → ``(-max_scale + (i + 0.5) * step) * std + center`` with
    ``center``/``std`` from the lookback only (same as training encode). Values
    returned are still in **global dataset-z** — this does not instance-norm
    the series; it only places the training canvas rungs into that space.
    """
    if bool(getattr(config, "use_ordinal_window_norm", False)):
        raise ValueError(
            "legal_window_norm_patch_refine_levels_dataset_z is for non-ordinal "
            "window-norm leaves; use legal_patch_refine_levels_dataset_z for ordinal"
        )
    past_np = np.asarray(past, dtype=np.float32)
    if past_np.ndim != 3:
        raise ValueError(f"past must be (N,V,L), got {past_np.shape}")
    height = int(getattr(config, "patch_refine_canvas_height", 0) or 0)
    if height <= 0:
        raise ValueError(f"patch_refine_canvas_height must be positive, got {height}")
    max_scale = float(getattr(config, "max_scale"))
    if max_scale <= 0.0:
        raise ValueError(f"max_scale must be positive, got {max_scale}")

    past_t = torch.from_numpy(past_np)
    center, std = window_normalization_stats(past_t, config)
    step = normalized_grid_step(config)
    rows = torch.arange(height, dtype=torch.float32)
    norm_levels = -max_scale + (rows + 0.5) * step  # (H,)
    levels = norm_levels.view(1, 1, -1) * std + center  # (N,V,H)
    if not torch.isfinite(levels).all():
        raise ValueError("window-norm patch-refine support contains non-finite values")
    return levels.detach().cpu().numpy().astype(np.float32)


def snap_to_unbounded_patch_refine_grid(
    values: torch.Tensor,
    past: torch.Tensor,
    config: Any,
) -> torch.Tensor:
    """Snap values to nearest patch-refine midpoint without clipping grid indices.

    The model can only emit rows 0..H-1.  The target is deliberately allowed
    to use any integer row, so extreme GT values are rounded by the same local
    step instead of being clipped to the model's finite value range.
    """
    center, std = window_normalization_stats(past, config)
    step = normalized_grid_step(config)
    normalized = (values - center) / std
    row = torch.round((normalized + float(config.max_scale)) / step - 0.5)
    snapped_normalized = -float(config.max_scale) + (row + 0.5) * step
    return snapped_normalized * std + center


def grid_coordinates(
    values: torch.Tensor,
    past: torch.Tensor,
    config: Any,
) -> torch.Tensor:
    """Return midpoint-grid row coordinates; integers mean values are snapped."""
    center, std = window_normalization_stats(past, config)
    step = normalized_grid_step(config)
    normalized = (values - center) / std
    return (normalized + float(config.max_scale)) / step - 0.5


def assert_on_patch_refine_grid(
    values: torch.Tensor,
    past: torch.Tensor,
    config: Any,
    *,
    atol: float = 2e-4,
) -> None:
    coordinates = grid_coordinates(values, past, config)
    error = (coordinates - coordinates.round()).abs().max().item()
    if error > atol:
        raise AssertionError(f"values are off the patch-refine grid: max row error={error:.6g}")


def snap_to_unbounded_patch_refine_grid(
    values: torch.Tensor,
    past: torch.Tensor,
    config: Any,
) -> torch.Tensor:
    """Snap values to nearest patch-refine midpoint without clipping grid indices.

    The model can only emit rows 0..255.  The target is deliberately allowed
    to use any integer row, so extreme GT values are rounded by the same local
    step instead of being clipped to the model's finite value range.
    """
    center, std = window_normalization_stats(past, config)
    step = normalized_grid_step(config)
    normalized = (values - center) / std
    row = torch.round((normalized + float(config.max_scale)) / step - 0.5)
    snapped_normalized = -float(config.max_scale) + (row + 0.5) * step
    return snapped_normalized * std + center


def grid_coordinates(
    values: torch.Tensor,
    past: torch.Tensor,
    config: Any,
) -> torch.Tensor:
    """Return midpoint-grid row coordinates; integers mean values are snapped."""
    center, std = window_normalization_stats(past, config)
    step = normalized_grid_step(config)
    normalized = (values - center) / std
    return (normalized + float(config.max_scale)) / step - 0.5


def assert_on_patch_refine_grid(
    values: torch.Tensor,
    past: torch.Tensor,
    config: Any,
    *,
    atol: float = 2e-4,
) -> None:
    coordinates = grid_coordinates(values, past, config)
    error = (coordinates - coordinates.round()).abs().max().item()
    if error > atol:
        raise AssertionError(f"values are off the patch-refine grid: max row error={error:.6g}")
