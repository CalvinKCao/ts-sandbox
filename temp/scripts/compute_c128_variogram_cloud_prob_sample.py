#!/usr/bin/env python3
"""Compute max-gap-16 variogram cloud from one random probabilistic path/window."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from compute_c128_variogram_cloud import REPO, RESULTS, SPECS, variogram_cloud


SEED = 20260810


def load_random_paths(path: Path, dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as pack:
        if "y_true" not in pack or "samples" not in pack:
            raise KeyError(f"{path} requires y_true and samples; found {pack.files}")
        y_true = np.asarray(pack["y_true"])
        samples = np.asarray(pack["samples"])
    if samples.ndim != 4 or samples.shape[:2] != y_true.shape[:2] or samples.shape[-1] != y_true.shape[-1]:
        raise ValueError(f"unexpected samples/y_true shapes in {path}: {samples.shape} vs {y_true.shape}")
    # A forecast draw is a full horizon path. Preserve its temporal and cross-variate
    # coherence by drawing one sample index per window, rather than per timestamp.
    rng = np.random.default_rng(np.random.SeedSequence([SEED, sum(map(ord, dataset))]))
    choice = rng.integers(samples.shape[2], size=samples.shape[0], endpoint=False)
    prediction = samples[np.arange(samples.shape[0]), :, choice, :]
    return y_true, prediction, choice, int(samples.shape[2])


def main() -> None:
    rows: list[dict[str, Any]] = []
    for dataset, spec in SPECS.items():
        binary_path = Path(spec["binary"])
        mmpd_path = Path(spec["mmpd"])
        binary_y, binary_pred, binary_choice, binary_sample_count = load_random_paths(binary_path, dataset)
        mmpd_y, mmpd_pred, mmpd_choice, mmpd_sample_count = load_random_paths(mmpd_path, dataset)
        binary_value, binary_per_gap = variogram_cloud(binary_y, binary_pred)
        mmpd_value, mmpd_per_gap = variogram_cloud(mmpd_y, mmpd_pred)
        rows.append(
            {
                "dataset": dataset,
                "max_gap": 16,
                "binary_quad_t": {
                    "value": binary_value,
                    "per_gap": binary_per_gap,
                    "n_windows": int(binary_y.shape[0]),
                    "n_variates": int(binary_y.shape[1]),
                    "n_saved_samples": binary_sample_count,
                    "sample_choices": binary_choice.tolist(),
                    "source": str(binary_path.relative_to(REPO)),
                },
                "mmpd_probabilistic": {
                    "value": mmpd_value,
                    "per_gap": mmpd_per_gap,
                    "n_windows": int(mmpd_y.shape[0]),
                    "n_variates": int(mmpd_y.shape[1]),
                    "n_saved_samples": mmpd_sample_count,
                    "sample_choices": mmpd_choice.tolist(),
                    "source": str(mmpd_path.relative_to(REPO)),
                },
                "delta_mmpd_minus_binary": mmpd_value - binary_value,
            }
        )

    payload = {
        "metric": "mean_{h=1..16} mean((error[t+h]-error[t])^2)",
        "max_gap": 16,
        "seed": SEED,
        "sampling": "One uniformly random saved sample trajectory per window, shared across variates.",
        "pool_note": "Binary and MMPD retain their own saved evaluation pools, matching the forecast table.",
        "rows": rows,
    }
    json_path = RESULTS / "variogram_cloud_prob_sample_gap16.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Canvas128 quad_t vs MMPD probabilistic variogram cloud (max gap 16)",
        "",
        "One uniformly random saved forecast trajectory per window, with seed 20260810. "
        "Each draw is kept intact through the 96-step horizon and across variates. Lower is better.",
        "",
        "| dataset | binary quad_t | MMPD sample | MMPD − binary | binary windows | MMPD windows |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['binary_quad_t']['value']:.6f} | "
            f"{row['mmpd_probabilistic']['value']:.6f} | {row['delta_mmpd_minus_binary']:+.6f} | "
            f"{row['binary_quad_t']['n_windows']} | {row['mmpd_probabilistic']['n_windows']} |"
        )
    (RESULTS / "variogram_cloud_prob_sample_gap16.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    for row in rows:
        print(
            f"{row['dataset']:12s} quad_t={row['binary_quad_t']['value']:.6f} "
            f"mmpd={row['mmpd_probabilistic']['value']:.6f} "
            f"delta={row['delta_mmpd_minus_binary']:+.6f}"
        )


if __name__ == "__main__":
    main()
