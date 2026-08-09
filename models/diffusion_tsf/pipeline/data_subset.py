"""Dataset reduction configuration policy.

YAML is the single source of truth for all per-dataset variate lists and per-split window strides.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


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
        variate_indices = list(ds_spec.get("variate_indices", base_variate_indices))
        n_variates = int(ds_spec.get("n_variates", len(variate_indices)))
        train_stride = int(ds_spec.get("train_stride", default_window_stride))
        val_stride = int(ds_spec.get("val_stride", train_stride))
        test_stride = int(ds_spec.get("test_stride", 1))
        sample_stride = int(ds_spec.get("sample_stride", max(train_stride, 1)))
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
        "raw_rows": raw_rows,
        "raw_variates": raw_variates,
        "reason": "yaml_fallback",
        "policy": policy,
    }
