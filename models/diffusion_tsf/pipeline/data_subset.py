"""Dataset reduction configuration policy.

YAML is the single source of truth for all per-dataset variate lists and per-split window strides.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from torch.utils.data import Subset

logger = logging.getLogger(__name__)


def random_window_subset(ds, max_windows, seed: int, *, label: str):
    """Keep a seeded random subset of windows. ``max_windows is None`` is a no-op."""
    if max_windows is None:
        return ds
    k = int(max_windows)
    if k < 1:
        raise ValueError(f"{label}: max_windows must be >= 1, got {max_windows!r}")
    n = len(ds)
    if n <= 0:
        raise ValueError(f"{label}: empty dataset")
    if k >= n:
        return ds
    rng = random.Random(int(seed))
    indices = sorted(rng.sample(range(n), k))
    logger.info("  [%s] random windows %d/%d (seed=%s)", label, k, n, seed)
    return Subset(ds, indices)


def resolve_data_subset(
    *,
    dataset_name: str,
    raw_rows: int,
    raw_variates: int,
    base_variate_indices: List[int],
    default_subset_id: Optional[str] = None,
    default_window_stride: int = 1,
    seed: int = 42,
    policy: Optional[Dict[str, Any]] = None,
    target_rows: Optional[int] = None,
    target_variates: Optional[int] = None,
) -> Dict[str, Any]:
    """Resolve data subset configuration for a dataset directly from YAML config.

    Reads explicit per-dataset subset settings (`data_subset_by_dataset` in YAML).
    Does NOT dynamically estimate sizes or auto-resize datasets in Python.
    """
    policy = dict(policy or {})
    by_dataset = policy.get("data_subset_by_dataset") or policy.get("by_dataset") or {}

    ds_spec = by_dataset.get(dataset_name)
    if ds_spec and isinstance(ds_spec, dict):
        all_variates = bool(ds_spec.get("all_variates", False))
        if all_variates:
            variate_indices = list(range(int(raw_variates)))
            n_variates = int(raw_variates)
        else:
            variate_indices = list(ds_spec.get("variate_indices", base_variate_indices))
            n_variates = int(ds_spec.get("n_variates", len(variate_indices)))
            if n_variates != len(variate_indices):
                raise ValueError(
                    f"{dataset_name}: n_variates={n_variates} != "
                    f"len(variate_indices)={len(variate_indices)}"
                )
        if any(int(i) < 0 or int(i) >= int(raw_variates) for i in variate_indices):
            raise ValueError(
                f"{dataset_name}: variate_indices out of range for raw_variates={raw_variates}"
            )
        train_stride = int(ds_spec.get("train_stride", default_window_stride))
        val_stride = int(ds_spec.get("val_stride", train_stride))
        test_stride = int(ds_spec.get("test_stride", 1))
        sample_stride = int(ds_spec.get("sample_stride", max(train_stride, 1)))
        train_max_windows = ds_spec.get("train_max_windows")
        val_max_windows = ds_spec.get("val_max_windows")
        if train_max_windows is not None:
            train_max_windows = int(train_max_windows)
            if train_max_windows < 1:
                raise ValueError(f"{dataset_name}: train_max_windows must be >= 1")
        if val_max_windows is not None:
            val_max_windows = int(val_max_windows)
            if val_max_windows < 1:
                raise ValueError(f"{dataset_name}: val_max_windows must be >= 1")
        subset_id = str(
            ds_spec.get(
                "subset_id",
                default_subset_id or f"{dataset_name}_{n_variates}v_s{sample_stride}",
            )
        )
        return {
            "enabled": True,
            "dataset": dataset_name,
            "subset_id": subset_id,
            "variate_indices": variate_indices,
            "n_variates": n_variates,
            "train_stride": train_stride,
            "val_stride": val_stride,
            "test_stride": test_stride,
            "sample_stride": sample_stride,
            "train_max_windows": train_max_windows,
            "val_max_windows": val_max_windows,
            "raw_rows": raw_rows,
            "raw_variates": raw_variates,
            "reason": "yaml_explicit",
            "policy": ds_spec,
        }

    # Fallback when dataset is not explicitly listed in data_subset_by_dataset
    variate_indices = list(base_variate_indices)
    n_variates = len(variate_indices)
    train_stride = max(1, int(default_window_stride))
    return {
        "enabled": False,
        "dataset": dataset_name,
        "subset_id": default_subset_id or dataset_name,
        "variate_indices": variate_indices,
        "n_variates": n_variates,
        "train_stride": train_stride,
        "val_stride": train_stride,
        "test_stride": 1,
        "sample_stride": 1,
        "train_max_windows": None,
        "val_max_windows": None,
        "raw_rows": raw_rows,
        "raw_variates": raw_variates,
        "reason": "yaml_fallback",
        "policy": policy,
    }
