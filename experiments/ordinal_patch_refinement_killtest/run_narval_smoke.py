"""Fixed Narval entry point for the gated oracle-coarse smoke test."""

from __future__ import annotations

from typing import Any

from experiments.ordinal_patch_refinement_killtest import smoke


class _HorizonView:
    def __init__(self, base: Any, horizon: int = 16) -> None:
        self.base, self.horizon = base, horizon

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        past, future = self.base[index]
        return past, future[..., : self.horizon]


_load_pool = smoke.load_tsf_pack_pool
_decode_ranks = smoke._decode_ranks


def _load_smoke_pool(*args: Any, **kwargs: Any):
    pool, *rest = _load_pool(*args, **kwargs)
    return _HorizonView(pool), *rest


def _decode_smoke_ranks(cdf, rank_max):
    """Decode 256 columns, then average each repeated 16-column time block."""
    rank_grid = _decode_ranks(cdf, rank_max)
    if rank_grid.shape[-1] != 256:
        raise ValueError(f"expected 256 high-res time columns, got {rank_grid.shape[-1]}")
    return rank_grid.reshape(*rank_grid.shape[:-1], 16, 16).mean(dim=-1)


smoke.load_tsf_pack_pool = _load_smoke_pool
smoke._decode_ranks = _decode_smoke_ranks


if __name__ == "__main__":
    smoke.main()
