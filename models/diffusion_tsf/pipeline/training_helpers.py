"""Pure helpers for synthetic pool sizing."""

from __future__ import annotations

from typing import Optional, Tuple


def resolve_synthetic_params(
    requested_n: int,
    requested_cap: int,
    smoke_test: bool,
    *,
    samples_cap: Optional[int],
    samples_min: int,
) -> Tuple[int, int]:
    if smoke_test:
        return 4, 1

    n = requested_n
    cap = requested_cap

    if samples_cap is not None:
        total = n * cap
        if total > samples_cap:
            n = max(samples_min, samples_cap // cap)
            if n * cap > samples_cap:
                cap = max(1, samples_cap // n)

    return int(n), int(cap)


def resolve_pretrain_virtual_dataset_size(
    smoke_test: bool,
    *,
    pretrain_epochs: int,
    pretrain_diffusion_max_epochs: int,
    pretrain_synthetic_override: Optional[int],
    samples_cap: Optional[int],
    samples_min: int,
) -> int:
    if smoke_test:
        return 4
    if pretrain_synthetic_override is not None:
        return max(4, int(pretrain_synthetic_override))

    steps = 32 + 48 * pretrain_epochs
    steps = max(64, steps)
    ref_bs = 8
    requested_n = steps * ref_bs
    max_cap = max(pretrain_epochs, pretrain_diffusion_max_epochs)
    n, _ = resolve_synthetic_params(
        requested_n, max_cap, smoke_test, samples_cap=samples_cap, samples_min=samples_min
    )
    return n
