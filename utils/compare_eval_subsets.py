"""Campaign lookback/horizon grids and binary-matched 10% test-window indices.

10% eval uses the same ``random.Random(seed).sample`` as staged_eval:
``k = round(0.1 * n_campaign)`` drawn from the campaign-H test pool, then
those start indices are scored at the trained ``pred_len`` (clip if a
start no longer fits).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

DATASETS_ALL: Tuple[str, ...] = (
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "weather",
    "electricity",
    "exchange_rate",
    "solar_Alabama",
    "traffic",
    "illness",
    "PeMS",
    "dynamic",
)

# Binary campaign lookbacks so val/test border1 = end_prev - L matches hz720
# (illness lb104 leaf). Paper PEMS uses L=96; this repo's live PeMS leaf is L=336.
# No L=600 / H=300 dynamic leaf exists — binary hz720 uses L=336 H=720.
CAMPAIGN_LOOKBACK: Dict[str, int] = {
    "illness": 104,
}

CAMPAIGN_HORIZON: Dict[str, int] = {
    "illness": 60,
}

DEFAULT_LOOKBACK = 336
DEFAULT_CAMPAIGN_HORIZON = 720
EVAL_TEST_FRACTION = 0.1
EVAL_WINDOW_SEED = 42

# Informer/Autoformer/PatchTST/iTransformer long-term grids (PatchTST ILI;
# iTransformer Table 9 PEMS 12/24/48/96, not the PDF {12,24,36,48} typo).
# MMPD Dynamic paper uses L=600 and {60,120,180,300}; PatchTST/iTransformer
# have no Dynamic table, so paper_horizons(dynamic) stays the 96/192/336/720
# quartet (binary hz720 clip-prefix protocol).
PAPER_HORIZONS: Dict[str, Tuple[int, ...]] = {
    "illness": (24, 36, 48, 60),
    "PeMS": (12, 24, 48, 96),
}

DEFAULT_PAPER_HORIZONS: Tuple[int, ...] = (96, 192, 336, 720)

EVAL_SUBSET_NAMES = ("binary_10pct", "test_100pct")


def campaign_lookback(dataset: str) -> int:
    return int(CAMPAIGN_LOOKBACK.get(dataset, DEFAULT_LOOKBACK))


def campaign_horizon(dataset: str) -> int:
    return int(CAMPAIGN_HORIZON.get(dataset, DEFAULT_CAMPAIGN_HORIZON))


def paper_horizons(dataset: str) -> Tuple[int, ...]:
    return PAPER_HORIZONS.get(dataset, DEFAULT_PAPER_HORIZONS)


def parse_eval_subset_list(raw: str) -> List[str]:
    if not raw or not str(raw).strip():
        return []
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    bad = [p for p in parts if p not in EVAL_SUBSET_NAMES]
    if bad:
        raise ValueError(
            f"unknown eval subset(s) {bad}; allowed {EVAL_SUBSET_NAMES}"
        )
    return list(dict.fromkeys(parts))


def random_fraction_indices(n: int, fraction: float, seed: int) -> List[int]:
    """Same keep=round(n*frac) + ``random.Random(seed).sample`` as staged_eval r=0.

    Inlined so MMPD eval does not import pipeline.data_subset (Narval checkouts
    on other branches may not have that helper).
    """
    import random

    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be >= 1, got {n}")
    frac = float(fraction)
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction!r}")
    k = max(1, int(round(n * frac)))
    if k >= n:
        return list(range(n))
    rng = random.Random(int(seed))
    return sorted(rng.sample(range(n), k))


def binary_10pct_indices(
    n_campaign: int,
    n_pred: int,
    *,
    fraction: float = EVAL_TEST_FRACTION,
    seed: int = EVAL_WINDOW_SEED,
) -> List[int]:
    """Campaign-H 10% starts, clipped to the trained pred_len window pool."""
    n_campaign = int(n_campaign)
    n_pred = int(n_pred)
    if n_pred < 1:
        raise ValueError(f"n_pred must be >= 1, got {n_pred}")
    idxs = random_fraction_indices(n_campaign, fraction, seed)
    kept = [i for i in idxs if i < n_pred]
    if not kept:
        raise ValueError(
            f"binary_10pct empty after clip: n_campaign={n_campaign} n_pred={n_pred}"
        )
    return kept


def prefix_metrics(metrics: Dict[str, float], subset: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, val in metrics.items():
        if isinstance(val, (int, float)):
            out[f"{key}_{subset}"] = float(val)
    return out
