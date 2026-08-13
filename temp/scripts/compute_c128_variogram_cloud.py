#!/usr/bin/env python3
"""Add the gap-16 variogram-cloud metric to the canvas128 comparison table."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "temp" / "lean_disc_c128_results"
PULLED = REPO / "temp" / "variogram_cloud_packs" / "results" / "datasets"


SPECS: dict[str, dict[str, Path | str]] = {
    "ETTh1": {
        "binary": PULLED / "08-03-4571065-ETTh1-binary_window_norm_patch_refine_canvas128_p64x6/raw/staged_anchor_ETTh1.npz",
        "mmpd": REPO / "results/datasets/07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd/raw/mmpd_ETTh1.npz",
    },
    "ETTh2": {
        "binary": PULLED / "08-04-4601319-ETTh2-binary_window_norm_patch_refine_canvas128_p64x6_etth2/raw/staged_anchor_ETTh2.npz",
        "mmpd": REPO / "results/datasets/08-04-mmpd-decoder-paper-lb336-hz96-ETTh2/raw/mmpd_ETTh2.npz",
    },
    "electricity": {
        "binary": PULLED / "08-04-4597054-electricity-binary_window_norm_patch_refine_canvas128_p64x6_electricity/raw/staged_anchor_electricity.npz",
        "mmpd": REPO / "results/datasets/07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd/raw/mmpd_electricity.npz",
    },
    "traffic": {
        "binary": PULLED / "08-04-4597055-traffic-binary_window_norm_patch_refine_canvas128_p64x6_traffic/raw/staged_anchor_traffic.npz",
        "mmpd": REPO / "results/datasets/07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd/raw/mmpd_traffic.npz",
    },
    "exchange": {
        "binary": PULLED / "08-04-4597056-exchange_rate-binary_window_norm_patch_refine_canvas128_p64x6_exchange_rate/raw/staged_anchor_exchange_rate.npz",
        "mmpd": REPO / "results/datasets/07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd/raw/mmpd_exchange_rate.npz",
    },
    "PeMS": {
        "binary": PULLED / "08-05-4623005-PeMS-binary_window_norm_patch_refine_canvas128_p64x6_pems/raw/staged_anchor_PeMS.npz",
        "mmpd": REPO / "results/datasets/08-06-mmpd-decoder-paper-lb336-hz96-matched-c128-four/raw/mmpd_PeMS.npz",
    },
    "solar": {
        "binary": PULLED / "08-05-4623006-solar_Alabama-binary_window_norm_patch_refine_canvas128_p64x6_solar_alabama/raw/staged_anchor_solar_Alabama.npz",
        "mmpd": REPO / "results/datasets/08-06-mmpd-decoder-paper-lb336-hz96-matched-c128-four/raw/mmpd_solar_Alabama.npz",
    },
    "ETTm1": {
        "binary": PULLED / "08-05-4623007-ETTm1-binary_window_norm_patch_refine_canvas128_p64x6_ettm1/raw/staged_anchor_ETTm1.npz",
        "mmpd": REPO / "results/datasets/08-06-mmpd-decoder-paper-lb336-hz96-matched-c128-four/raw/mmpd_ETTm1.npz",
    },
    "ETTm2": {
        "binary": PULLED / "08-05-4623008-ETTm2-binary_window_norm_patch_refine_canvas128_p64x6_ettm2/raw/staged_anchor_ETTm2.npz",
        "mmpd": REPO / "results/datasets/08-06-mmpd-decoder-paper-lb336-hz96-matched-c128-four/raw/mmpd_ETTm2.npz",
    },
}


def variogram_cloud(y_true: np.ndarray, prediction: np.ndarray, max_gap: int = 16) -> tuple[float, list[float]]:
    """Mean lag-difference MSE, averaged uniformly over gaps 1..max_gap."""
    y_true = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if y_true.shape != prediction.shape or y_true.ndim < 1:
        raise ValueError(f"incompatible y/pred shapes: {y_true.shape} vs {prediction.shape}")
    horizon = y_true.shape[-1]
    if horizon < 2:
        raise ValueError(f"horizon must be at least two, got {horizon}")
    error = prediction - y_true
    per_gap = [
        float(np.mean((error[..., gap:] - error[..., :-gap]) ** 2))
        for gap in range(1, min(max_gap, horizon - 1) + 1)
    ]
    return float(np.mean(per_gap)), per_gap


def load_anchor(path: Path, prediction_key: str) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as pack:
        if "y_true" not in pack or prediction_key not in pack:
            raise KeyError(f"{path} requires y_true and {prediction_key}; found {pack.files}")
        return np.asarray(pack["y_true"]), np.asarray(pack[prediction_key])


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Canvas128 vs MMPD variogram cloud (max gap 16)",
        "",
        "Mean over gaps 1–16 of the lag-difference MSE, evaluated independently "
        "on each run's saved deterministic-anchor pack. Lower is better.",
        "",
        "| dataset | binary | MMPD | MMPD − binary | binary windows | MMPD windows |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['binary']['value']:.6f} | "
            f"{row['mmpd']['value']:.6f} | {row['delta_mmpd_minus_binary']:+.6f} | "
            f"{row['binary']['n_windows']} | {row['mmpd']['n_windows']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_summary_tables(rows: list[dict[str, Any]]) -> None:
    by_name = {row["dataset"]: row for row in rows}
    compact_path = RESULTS / "full_metrics_table_mlp.json"
    compact = json.loads(compact_path.read_text(encoding="utf-8"))
    for row in compact["rows"]:
        metric = by_name[row["dataset"]]
        row["bin_variogram_cloud_g16"] = metric["binary"]["value"]
        row["mmpd_variogram_cloud_g16"] = metric["mmpd"]["value"]
    compact["note"] = (
        "mlp AUROC only; forecast from full_metrics_table.json. "
        "Variogram cloud is deterministic-anchor lag-difference MSE, max gap 16."
    )
    compact_path.write_text(json.dumps(compact, indent=2) + "\n", encoding="utf-8")

    detailed_path = RESULTS / "full_metrics_table.json"
    detailed = json.loads(detailed_path.read_text(encoding="utf-8"))
    name_map = {"exchange_rate": "exchange", "solar_Alabama": "solar"}
    for row in detailed["datasets"]:
        metric = by_name[name_map.get(row["dataset"], row["dataset"])]
        row["binary_canvas128_hz96"]["variogram_cloud_g16"] = metric["binary"]["value"]
        row["mmpd"]["variogram_cloud_g16"] = metric["mmpd"]["value"]
    detailed["protocol"]["variogram_cloud_g16"] = (
        "Mean MSE of anchor-error increments for gaps 1..16; binary/MMPD remain "
        "their own saved evaluation pools, matching the existing forecast columns."
    )
    detailed["protocol"]["variogram_cloud_sources"] = str(RESULTS / "variogram_cloud_gap16.json")
    detailed["protocol"]["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    detailed_path.write_text(json.dumps(detailed, indent=2) + "\n", encoding="utf-8")

    markdown_path = RESULTS / "full_metrics_table_mlp.md"
    lines = [
        "# Canvas128 binary vs MMPD + MLP discriminator AUROC",
        "",
        "Forecast: anchor MAE/MSE + CRPS (dataset-global-z). Variogram cloud: deterministic-anchor lag-difference MSE through gap 16 (lower is better). Disc: **mlp** AUROC, unique_abs+bin-center protocol, L=8/16.",
        "",
        "| dataset | bin MAE | bin MSE | bin CRPS | bin vario16 | mmpd MAE | mmpd MSE | mmpd CRPS | mmpd vario16 | mlp L8 bin | mlp L8 mmpd | mlp L16 bin | mlp L16 mmpd |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in compact["rows"]:
        lines.append(
            f"| {row['dataset']} | {row['bin_mae']:.3f} | {row['bin_mse']:.3f} | "
            f"{row['bin_crps']:.3f} | {row['bin_variogram_cloud_g16']:.3f} | "
            f"{row['mmpd_mae']:.3f} | {row['mmpd_mse']:.3f} | {row['mmpd_crps']:.3f} | "
            f"{row['mmpd_variogram_cloud_g16']:.3f} | {row['mlp_L8_bin']:.3f} | "
            f"{row['mlp_L8_mmpd']:.3f} | {row['mlp_L16_bin']:.3f} | {row['mlp_L16_mmpd']:.3f} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows: list[dict[str, Any]] = []
    for dataset, spec in SPECS.items():
        binary_path = Path(spec["binary"])
        mmpd_path = Path(spec["mmpd"])
        binary_y, binary_pred = load_anchor(binary_path, "deterministic")
        mmpd_y, mmpd_pred = load_anchor(mmpd_path, "deterministic")
        binary_value, binary_per_gap = variogram_cloud(binary_y, binary_pred)
        mmpd_value, mmpd_per_gap = variogram_cloud(mmpd_y, mmpd_pred)
        rows.append(
            {
                "dataset": dataset,
                "max_gap": 16,
                "binary": {
                    "value": binary_value,
                    "per_gap": binary_per_gap,
                    "n_windows": int(binary_y.shape[0]),
                    "n_variates": int(binary_y.shape[1]),
                    "source": str(binary_path.relative_to(REPO)),
                },
                "mmpd": {
                    "value": mmpd_value,
                    "per_gap": mmpd_per_gap,
                    "n_windows": int(mmpd_y.shape[0]),
                    "n_variates": int(mmpd_y.shape[1]),
                    "source": str(mmpd_path.relative_to(REPO)),
                },
                "delta_mmpd_minus_binary": mmpd_value - binary_value,
            }
        )

    output = {
        "metric": "mean_{h=1..16} mean((pred[t+h]-gt[t+h] - (pred[t]-gt[t]))^2)",
        "equivalent": "mean_{h=1..16} mean((error[t+h]-error[t])^2)",
        "max_gap": 16,
        "pool_note": "The binary and MMPD runs retain their original saved pools, as in the forecast table.",
        "rows": rows,
    }
    (RESULTS / "variogram_cloud_gap16.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(rows, RESULTS / "variogram_cloud_gap16.md")
    update_summary_tables(rows)
    for row in rows:
        print(
            f"{row['dataset']:12s} binary={row['binary']['value']:.6f} "
            f"mmpd={row['mmpd']['value']:.6f} "
            f"delta={row['delta_mmpd_minus_binary']:+.6f}"
        )


if __name__ == "__main__":
    main()
