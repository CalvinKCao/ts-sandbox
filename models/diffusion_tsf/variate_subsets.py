"""Generate variate-index subsets for multi-branch log-signature diffusion."""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple


def generate_variate_subsets(
    n_variates: int,
    *,
    scheme: str = "all",
    subset_size: Optional[int] = None,
    subset_stride: int = 1,
    n_subsets: Optional[int] = None,
    max_branches: int = 5,
    seed: int = 0,
) -> List[Tuple[int, ...]]:
    """Return a list of channel-index tuples (each is one branch)."""
    if n_variates < 1:
        raise ValueError("n_variates must be >= 1")

    cap = max(1, min(max_branches, n_subsets or max_branches))
    all_idx = tuple(range(n_variates))

    if scheme == "all":
        return [all_idx]

    k = subset_size or min(3, n_variates)
    k = max(1, min(k, n_variates))

    if scheme == "sliding":
        step = max(1, subset_stride)
        subsets: List[Tuple[int, ...]] = []
        for start in range(0, max(1, n_variates - k + 1), step):
            subsets.append(tuple(range(start, start + k)))
        if not subsets:
            subsets = [all_idx]
        return subsets[:cap]

    if scheme == "pairs":
        subsets = []
        for i in range(n_variates):
            for j in range(i + 1, n_variates):
                subsets.append((i, j))
        if not subsets:
            subsets = [all_idx]
        return subsets[:cap]

    if scheme == "random_k":
        rng = random.Random(seed)
        seen = set()
        subsets = []
        attempts = 0
        while len(subsets) < cap and attempts < cap * 20:
            attempts += 1
            pick = tuple(sorted(rng.sample(range(n_variates), k)))
            if pick in seen:
                continue
            seen.add(pick)
            subsets.append(pick)
        return subsets or [all_idx]

    raise ValueError(f"unknown subset_scheme: {scheme}")
