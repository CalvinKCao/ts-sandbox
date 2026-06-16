"""MMPD appendix D.3 hyperparameter helpers shared by eval and paper runners."""

from __future__ import annotations

# Repo dataset keys that always use the wide patch (paper ECL / Traffic).
WIDE_PATCH_DATASETS = frozenset({"electricity", "traffic", "ECL", "Traffic"})


def mmpd_patch_size(dataset: str, horizon: int) -> int:
    """Paper D.3: P=12 default; P=24 for tau in {336,720} or ECL/Traffic."""
    if dataset in WIDE_PATCH_DATASETS:
        return 24
    if horizon in (336, 720):
        return 24
    return 12


def resolve_mmpd_patch_size(
    dataset: str,
    horizon: int,
    override: int | None = None,
) -> int:
    if override is not None:
        return int(override)
    return mmpd_patch_size(dataset, horizon)
