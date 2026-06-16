"""MMPD appendix D.3 hyperparameter helpers shared by eval and paper runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

# Repo dataset keys that always use the wide patch (paper ECL / Traffic).
WIDE_PATCH_DATASETS = frozenset({"electricity", "traffic", "ECL", "Traffic"})

DEFAULT_MMPD_HPARAMS: Dict[str, Any] = {
    "learning_rate": 1e-4,
    "point_weight": 0.01,
    "dropout": 0.2,
    "finetune_layers": 0,
    "neighbor_num": 0,
}


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


def tuning_result_path(output_dir: Path, dataset: str) -> Path:
    return output_dir / "tuning" / f"{dataset}_best.json"


def load_tuned_hparams(output_dir: Path, dataset: str) -> Optional[Dict[str, Any]]:
    path = tuning_result_path(output_dir, dataset)
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    hparams = payload.get("hparams")
    return dict(hparams) if isinstance(hparams, dict) else None


def resolved_mmpd_hparams(
    output_dir: Path,
    dataset: str,
    *,
    fallback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    tuned = load_tuned_hparams(output_dir, dataset)
    if tuned is not None:
        return {**DEFAULT_MMPD_HPARAMS, **tuned}
    if fallback is not None:
        return {**DEFAULT_MMPD_HPARAMS, **fallback}
    return dict(DEFAULT_MMPD_HPARAMS)
