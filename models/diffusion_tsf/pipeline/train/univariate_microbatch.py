"""Univariate DiT-row microbatches over multivariate windows.

Dataloader batches stay whole windows ``(B, V, L)``. GPU microbatches are at
most ``U`` flattened ``(window, variate)`` rows. When ``V > U`` one window is
split across microsteps; when ``U > V`` we pack ``ceil(U / V)`` windows and the
last window may contribute only some variates.

Optimizer steps accumulate by exact row count: each forward is scaled by
``slice_rows / target_rows`` so one step equals the target U-row budget, with
leftover rows carried across loader batches.
"""

from __future__ import annotations

import math
from typing import Any, Iterator, Tuple

_U_PLACEHOLDER_STRINGS = frozenset({"PROBE", "TODO", "PLACEHOLDER"})


def dataloader_windows_for_univariate_rows(max_rows: int, n_variates: int) -> int:
    """Smallest window batch that can supply ``max_rows`` univariate DiT rows."""
    u = max(1, int(max_rows))
    v = max(1, int(n_variates))
    return max(1, math.ceil(u / v))


def iter_flat_row_slices(
    n_windows: int,
    n_variates: int,
    max_rows: int,
) -> Iterator[Tuple[int, int]]:
    """Yield ``[start, end)`` slices into the flattened ``B * V`` row list."""
    n_windows = int(n_windows)
    n_variates = int(n_variates)
    max_rows = max(1, int(max_rows))
    if n_windows < 1 or n_variates < 1:
        raise ValueError(
            f"need n_windows>=1 and n_variates>=1, got {n_windows}, {n_variates}"
        )
    n_rows = n_windows * n_variates
    for start in range(0, n_rows, max_rows):
        yield start, min(start + max_rows, n_rows)


def next_row_take(
    n_remaining: int,
    *,
    gpu_u: int,
    remaining_budget: int,
    rows_in_step: int,
) -> int:
    """Rows to consume next, capped by GPU U, loader leftover, and opt budget.

    Also splits at the remainder of the current U-row micro so a slice never
    straddles two logical U-row batches.
    """
    n_remaining = int(n_remaining)
    gpu_u = max(1, int(gpu_u))
    remaining_budget = int(remaining_budget)
    rows_in_step = int(rows_in_step)
    if n_remaining < 1:
        raise ValueError(f"n_remaining must be >= 1, got {n_remaining}")
    if remaining_budget < 1:
        raise ValueError(f"remaining_budget must be >= 1, got {remaining_budget}")
    if rows_in_step < 0:
        raise ValueError(f"rows_in_step must be >= 0, got {rows_in_step}")
    remain_u = gpu_u - (rows_in_step % gpu_u)
    take = min(n_remaining, remain_u, remaining_budget)
    if take < 1:
        raise ValueError(
            f"row take is 0 (remaining={n_remaining} remain_u={remain_u} "
            f"budget={remaining_budget} rows_in_step={rows_in_step} gpu_u={gpu_u})"
        )
    return take


def loss_scale_for_rows(slice_rows: int, target_rows: int) -> float:
    """Mean-loss scale so one optimizer step equals exactly ``target_rows``."""
    slice_rows = int(slice_rows)
    target_rows = int(target_rows)
    if slice_rows < 1:
        raise ValueError(f"slice_rows must be >= 1, got {slice_rows}")
    if target_rows < 1:
        raise ValueError(f"target_rows must be >= 1, got {target_rows}")
    return slice_rows / target_rows


def is_univariate_u_placeholder(value: Any) -> bool:
    """True for unset / sentinel U values that must not be used for HP."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip().upper() in _U_PLACEHOLDER_STRINGS:
        return True
    try:
        return int(value) < 1
    except (TypeError, ValueError):
        return True


def require_probed_univariate_u(value: Any, *, dataset: str) -> int:
    """Return a probed U, or fail if YAML still has a placeholder/sentinel."""
    if is_univariate_u_placeholder(value):
        raise ValueError(
            f"max_univariate_micro_batch_by_dataset[{dataset!r}]={value!r} is a "
            "placeholder; run temp/scripts/probe_univariate_micro_batch.py on L40S "
            "and replace PROBE with the probed U before HP"
        )
    return int(value)
