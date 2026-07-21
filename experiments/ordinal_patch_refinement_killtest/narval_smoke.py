"""Narval entry point for the oracle-coarse smoke test.

Some repository datasets retain their configured horizon in pooled windows even
when a smaller horizon is requested.  The kill test is defined on exactly the
first sixteen future points, so crop at the loader boundary before delegating
to the shared smoke implementation.
"""

from __future__ import annotations

from typing import Any

from experiments.ordinal_patch_refinement_killtest import smoke


class _HorizonView:
    def __init__(self, base: Any, horizon: int = 16) -> None:
        self.base = base
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        past, future = self.base[index]
        return past, future[..., : self.horizon]


_load_pool = smoke.load_tsf_pack_pool


def _load_smoke_pool(*args: Any, **kwargs: Any):
    pool, *rest = _load_pool(*args, **kwargs)
    return _HorizonView(pool), *rest


smoke.load_tsf_pack_pool = _load_smoke_pool


if __name__ == "__main__":
    smoke.main()
