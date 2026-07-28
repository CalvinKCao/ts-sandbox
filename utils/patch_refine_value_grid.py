"""The per-window 256-value grid used by non-ordinal patch refinement."""

from __future__ import annotations

from typing import Any, Tuple

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
        return center, past_std.clamp_min(std_floor)

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
    return center, torch.where(flat | low_var, unit, std)


def normalized_grid_step(config: Any) -> float:
    height = int(getattr(config, "patch_refine_canvas_height", 256))
    max_scale = float(getattr(config, "max_scale"))
    if height <= 0 or max_scale <= 0.0:
        raise ValueError(f"invalid patch-refine grid: height={height}, max_scale={max_scale}")
    return 2.0 * max_scale / height


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
