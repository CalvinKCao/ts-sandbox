"""Deterministic dataset reduction policy for pipeline experiments."""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional


def estimate_dense_size_mb(n_rows: int, n_variates: int, bytes_per_value: int = 4) -> float:
    """Approximate normalized dense array size, independent of CSV text overhead."""
    return (int(n_rows) * int(n_variates) * int(bytes_per_value)) / (1024.0 * 1024.0)


def _pick_variate_indices(
    base_indices: List[int],
    n_variates: int,
    strategy: str,
    seed: int,
) -> List[int]:
    if n_variates >= len(base_indices):
        return list(base_indices)
    if strategy == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(list(base_indices), n_variates))
    if strategy != "first":
        raise ValueError(f"Unknown variate subset strategy: {strategy}")
    return list(base_indices[:n_variates])


def resolve_data_subset(
    *,
    dataset_name: str,
    raw_rows: int,
    raw_variates: int,
    base_variate_indices: List[int],
    default_subset_id: Optional[str],
    default_window_stride: int,
    seed: int,
    policy: Optional[Dict[str, Any]],
    target_rows: Optional[int] = None,
    target_variates: Optional[int] = None,
) -> Dict[str, Any]:
    """Resolve a YAML data-subset policy to concrete variates and strides.

    Reduction order is intentionally simple: cap variates first, then increase
    the sample/window stride if the reduced dense array is still above target.
    """
    policy = dict(policy or {})
    enabled = bool(policy.get("enabled", False))
    base_variate_indices = list(base_variate_indices)
    target_size_mb = policy.get("target_size_mb")
    if target_size_mb is None and target_rows is not None and target_variates is not None:
        target_size_mb = estimate_dense_size_mb(target_rows, target_variates)

    raw_size_mb = estimate_dense_size_mb(raw_rows, raw_variates)
    train_stride = max(1, int(default_window_stride))
    test_stride = 1

    if not enabled:
        return {
            "enabled": False,
            "dataset": dataset_name,
            "subset_id": default_subset_id or dataset_name,
            "variate_indices": base_variate_indices,
            "n_variates": len(base_variate_indices),
            "train_stride": train_stride,
            "val_stride": train_stride,
            "test_stride": test_stride,
            "sample_stride": 1,
            "raw_rows": raw_rows,
            "raw_variates": raw_variates,
            "raw_size_mb": raw_size_mb,
            "target_size_mb": target_size_mb,
            "reduced_size_mb": raw_size_mb,
            "reason": "disabled",
            "policy": policy,
        }

    reduce_only_if_larger = bool(policy.get("reduce_if_larger_than_target", True))
    should_reduce = target_size_mb is None or raw_size_mb > float(target_size_mb)
    if reduce_only_if_larger and not should_reduce:
        return {
            "enabled": True,
            "dataset": dataset_name,
            "subset_id": default_subset_id or dataset_name,
            "variate_indices": base_variate_indices,
            "n_variates": len(base_variate_indices),
            "train_stride": train_stride,
            "val_stride": train_stride,
            "test_stride": test_stride,
            "sample_stride": 1,
            "raw_rows": raw_rows,
            "raw_variates": raw_variates,
            "raw_size_mb": raw_size_mb,
            "target_size_mb": target_size_mb,
            "reduced_size_mb": raw_size_mb,
            "reason": "at_or_below_target",
            "policy": policy,
        }

    mv_by_dataset = policy.get("max_variates_by_dataset", {})
    if dataset_name in mv_by_dataset:
        max_variates = int(mv_by_dataset[dataset_name])
    else:
        max_variates = int(policy.get("max_variates", len(base_variate_indices)))
        
    n_variates = max(1, min(len(base_variate_indices), max_variates))
    strategy = str(policy.get("variate_strategy", "first"))
    variate_indices = _pick_variate_indices(base_variate_indices, n_variates, strategy, seed)

    size_after_variates = estimate_dense_size_mb(raw_rows, n_variates)
    sample_stride = 1
    if target_size_mb is not None and size_after_variates > float(target_size_mb):
        sample_stride = max(1, math.ceil(size_after_variates / float(target_size_mb)))
    if policy.get("sample_stride") not in (None, "auto"):
        sample_stride = max(sample_stride, int(policy["sample_stride"]))

    train_stride = max(train_stride, sample_stride)
    if bool(policy.get("apply_stride_to_test", True)):
        test_stride = sample_stride

    effective_rows = max(1, math.ceil(raw_rows / sample_stride))
    reduced_size_mb = estimate_dense_size_mb(effective_rows, n_variates)
    subset_id = default_subset_id
    if not subset_id:
        template = str(policy.get("subset_id_template", "{dataset}_{n_variates}v_s{sample_stride}"))
        subset_id = template.format(
            dataset=dataset_name,
            n_variates=n_variates,
            sample_stride=sample_stride,
            target_dataset=policy.get("target_dataset", "target"),
        )

    return {
        "enabled": True,
        "dataset": dataset_name,
        "subset_id": subset_id,
        "variate_indices": variate_indices,
        "n_variates": n_variates,
        "train_stride": train_stride,
        "val_stride": train_stride,
        "test_stride": test_stride,
        "sample_stride": sample_stride,
        "raw_rows": raw_rows,
        "raw_variates": raw_variates,
        "raw_size_mb": raw_size_mb,
        "target_size_mb": target_size_mb,
        "size_after_variates_mb": size_after_variates,
        "reduced_size_mb": reduced_size_mb,
        "reason": "reduced",
        "policy": policy,
    }
