#!/usr/bin/env python3
"""Write a markdown summary for a completed mmpd-anchor-matrix eval run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS = ["mmpd", "gaussian_anchor", "binary_anchor"]
MODEL_LABELS = {
    "mmpd": "MMPD",
    "gaussian_anchor": "Gaussian anchor",
    "binary_anchor": "Binary anchor",
}

TEXTURE_METRICS = [
    ("ordinal_jsd", "Ordinal JSD"),
    ("rqa_distance", "RQA distance"),
    ("variogram_distance", "Variogram distance"),
    ("pathsig_distance", "Path signature distance"),
]

INFERENCE_MODES = [
    ("texture", "Deterministic (anchor / point path)"),
    ("sample_mean_texture", "Mean of probabilistic samples"),
    ("per_sample_mean_texture", "Per-sample mean (texture on each draw, averaged)"),
]


def load_rows(run_dir: Path) -> List[Dict[str, Any]]:
    csv_path = run_dir / "metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
    with csv_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_manifest(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "run_manifest.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def fnum(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def metric_key(prefix: str, suffix: str) -> str:
    return f"{prefix}_{suffix}"


def row_value(row: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if row is None or key not in row or row[key] in ("", None):
        return None
    try:
        return float(row[key])
    except (TypeError, ValueError):
        return None


def best_models(
    rows: List[Dict[str, Any]],
    dataset: str,
    key: str,
    lower: bool = True,
) -> Set[str]:
    scored: List[Tuple[float, str]] = []
    for model in MODELS:
        row = next((r for r in rows if r["dataset"] == dataset and r["model"] == model), None)
        value = row_value(row, key)
        if value is not None:
            scored.append((value, model))
    if not scored:
        return set()
    best_val = min(v for v, _ in scored) if lower else max(v for v, _ in scored)
    tol = max(1e-9, abs(best_val) * 1e-6)
    return {model for val, model in scored if abs(val - best_val) <= tol}


def fmt_cell(
    row: Optional[Dict[str, Any]],
    key: str,
    model: str,
    winners: Set[str],
) -> str:
    text = fnum(row_value(row, key))
    if model in winners and text != "—":
        return f"**{text}**"
    return text


def best_model_label(rows: List[Dict[str, Any]], dataset: str, key: str) -> str:
    winners = best_models(rows, dataset, key)
    if not winners:
        return "—"
    return ", ".join(MODEL_LABELS[m] for m in MODELS if m in winners)


def active_inference_modes(rows: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    modes: List[Tuple[str, str]] = []
    for prefix, title in INFERENCE_MODES:
        key = metric_key(prefix, TEXTURE_METRICS[0][0])
        if any(row_value(r, key) is not None for r in rows):
            modes.append((prefix, title))
    return modes


def build_report(run_dir: Path, report_path: Path) -> None:
    run_dir = run_dir.resolve()
    rows = load_rows(run_dir)
    manifest = load_manifest(run_dir)
    args = manifest.get("args", {})
    datasets = sorted({r["dataset"] for r in rows})
    texture_modes = active_inference_modes(rows)

    lines: List[str] = [
        f"# MMPD vs Gaussian vs Binary anchor — matrix eval ({run_dir.name})",
        "",
        "**Run directory (pulled results):**",
        f"`{run_dir.relative_to(REPO_ROOT)}`",
        "",
        "| Artifact | Path |",
        "|----------|------|",
        f"| Merged metrics | `{run_dir.name}/metrics.csv` |",
        f"| JSON | `{run_dir.name}/metrics.json` |",
        f"| Per-worker partials | `{run_dir.name}/partials/` |",
        f"| Raw predictions | `{run_dir.name}/raw/*.npz` |",
        f"| Slurm logs | `results/logs/{run_dir.name}/` |",
        "",
        "## Setup",
        "",
        f"- **Eval harness:** `utils/eval_mmpd_gaussian_anchor.py` (Slurm fan-out `slurm_mmpd_gaussian_anchor_eval.sh`)",
        f"- **Test subset:** {float(args.get('test_fraction', 0.5)):.0%} of test windows (seed {args.get('seed', '?')})",
        f"- **Probabilistic samples:** {int(float(args.get('sample_num', 9)))} draws, "
        f"{args.get('num_sampling_steps', 20)} diffusion steps, sampler `{args.get('anchor_prob_sampler', 'dpmpp')}`",
        f"- **Top-k modes:** GMM with up to {args.get('gmm_components', 9)} components",
        f"- **Datasets:** {', '.join(datasets)} (weather excluded)",
        f"- **MMPD:** trained {args.get('mmpd_train_epochs', '?')} epochs in-run unless reused ckpt",
        "",
        "**Metric definitions:**",
        "- `mse` / `mae`: deterministic path (anchor sampler for diffusion; MMPD point forecast)",
        "- `crps`: continuous ranked probability score from all samples",
        "- `top3_mse` / `top3_mae`: best of top-3 GMM modes vs ground truth",
        "- **Texture (4):** ordinal JSD, RQA distance, variogram distance, path signature distance",
        "- **Bold** = best (lowest) among the three models for that dataset and inference mode",
        "",
        "## Core metrics (lower is better; bold = best per column)",
        "",
    ]

    core_cols = [
        ("mse", "MSE (det)"),
        ("mae", "MAE (det)"),
        ("crps", "CRPS"),
        ("top3_mse", "top3 MSE"),
        ("top3_mae", "top3 MAE"),
    ]

    for dataset in datasets:
        lines.append(f"### {dataset}")
        lines.append("")
        header = "| Model | " + " | ".join(label for _, label in core_cols) + " | windows |"
        sep = "|-------|" + "|".join("---------:" for _ in core_cols) + "|--------:|"
        lines.append(header)
        lines.append(sep)
        winners_by_col = {key: best_models(rows, dataset, key) for key, _ in core_cols}
        for model in MODELS:
            row = next((r for r in rows if r["dataset"] == dataset and r["model"] == model), None)
            if row is None:
                continue
            cells = [
                fmt_cell(row, key, model, winners_by_col[key]) for key, _ in core_cols
            ]
            lines.append(
                f"| {MODEL_LABELS[model]} | {' | '.join(cells)} | "
                f"{int(float(row['n_windows']))} |"
            )
        lines.append("")

    lines.extend(["## Texture metrics (lower is better; bold = best per row)", ""])
    for suffix, metric_title in TEXTURE_METRICS:
        lines.append(f"### {metric_title}")
        lines.append("")
        for prefix, mode_title in texture_modes:
            lines.append(f"**{mode_title}**")
            lines.append("")
            lines.append("| Dataset | MMPD | Gaussian anchor | Binary anchor |")
            lines.append("|---------|-----:|----------------:|--------------:|")
            col_key = metric_key(prefix, suffix)
            for dataset in datasets:
                winners = best_models(rows, dataset, col_key)
                cells = []
                for model in MODELS:
                    row = next(
                        (r for r in rows if r["dataset"] == dataset and r["model"] == model),
                        None,
                    )
                    cells.append(fmt_cell(row, col_key, model, winners))
                lines.append(f"| {dataset} | {' | '.join(cells)} |")
            lines.append("")

    lines.extend(["## Notes", ""])
    lines.append(
        "1. **illness / MMPD:** Point and top-k MMPD metrics are much worse than anchors on this tiny "
        "test subset (49 windows). Treat illness MMPD numbers as suspect until spot-checked."
    )
    if not any(p == "per_sample_mean_texture" for p, _ in texture_modes):
        lines.append(
            "2. **No per-sample texture** in this run (`per_sample_mean_texture_*` absent). "
            "Re-run `slurm_mmpd_texture_per_sample.sh --reference-run ...` to add them from cached `raw/*.npz`."
        )
    lines.append(
        f"- **Regenerate:** `python utils/report_mmpd_anchor_matrix.py --run-dir {run_dir.relative_to(REPO_ROOT)}`"
    )
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Matrix output dir containing metrics.csv",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Output markdown (default: reports/<stem>_mmpd_anchor_matrix_report.md)",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    report_path = args.report_path or (
        REPO_ROOT / "reports" / f"{run_dir.name}_mmpd_anchor_matrix_report.md"
    )
    build_report(run_dir, report_path)


if __name__ == "__main__":
    main()
