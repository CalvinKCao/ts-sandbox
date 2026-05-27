#!/usr/bin/env python3
"""Write a markdown summary for a completed mmpd-anchor-matrix eval run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS = ["mmpd", "gaussian_anchor", "binary_anchor"]
MODEL_LABELS = {
    "mmpd": "MMPD",
    "gaussian_anchor": "Gaussian anchor",
    "binary_anchor": "Binary anchor",
}


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


def best_model(rows: List[Dict[str, Any]], dataset: str, key: str, lower: bool = True) -> str:
    subset = [r for r in rows if r["dataset"] == dataset]
    if not subset:
        return "—"
    scored: List[Tuple[float, str]] = []
    for row in subset:
        try:
            scored.append((float(row[key]), row["model"]))
        except (TypeError, ValueError):
            continue
    if not scored:
        return "—"
    scored.sort(key=lambda x: x[0], reverse=not lower)
    return MODEL_LABELS.get(scored[0][1], scored[0][1])


def pct_delta(a: float, b: float) -> str:
    if a == 0:
        return "—"
    return f"{100.0 * (b - a) / a:+.1f}%"


def build_report(run_dir: Path, report_path: Path) -> None:
    run_dir = run_dir.resolve()
    rows = load_rows(run_dir)
    manifest = load_manifest(run_dir)
    args = manifest.get("args", {})
    datasets = sorted({r["dataset"] for r in rows})

    stem = run_dir.name
    job_hint = stem.split("-")[2] if "-" in stem else stem

    lines: List[str] = [
        f"# MMPD vs Gaussian vs Binary anchor — matrix eval ({stem})",
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
        "- `texture_*`: shape metrics on deterministic forecast; `sample_mean_texture_*`: on mean of samples",
        "",
        "## Core metrics (lower is better)",
        "",
    ]

    for dataset in datasets:
        lines.append(f"### {dataset}")
        lines.append("")
        lines.append(
            "| Model | MSE (det) | MAE (det) | CRPS | top3 MSE | top3 MAE | windows |"
        )
        lines.append("|-------|----------:|----------:|-----:|---------:|---------:|--------:|")
        for model in MODELS:
            row = next((r for r in rows if r["dataset"] == dataset and r["model"] == model), None)
            if row is None:
                continue
            lines.append(
                f"| {MODEL_LABELS[model]} | {fnum(row['mse'])} | {fnum(row['mae'])} | "
                f"{fnum(row['crps'])} | {fnum(row['top3_mse'])} | {fnum(row['top3_mae'])} | "
                f"{int(float(row['n_windows']))} |"
            )
        lines.append("")
        lines.append(
            f"Best det MSE: **{best_model(rows, dataset, 'mse')}** · "
            f"Best CRPS: **{best_model(rows, dataset, 'crps')}** · "
            f"Best top3 MSE: **{best_model(rows, dataset, 'top3_mse')}**"
        )
        lines.append("")

    lines.extend(["## Cross-dataset winners (det MSE)", ""])
    lines.append("| Dataset | MMPD | Gauss anchor | Binary anchor | Best |")
    lines.append("|---------|-----:|-------------:|--------------:|------|")
    for dataset in datasets:
        vals = []
        for model in MODELS:
            row = next((r for r in rows if r["dataset"] == dataset and r["model"] == model), None)
            vals.append(fnum(row["mse"]) if row else "—")
        lines.append(
            f"| {dataset} | {vals[0]} | {vals[1]} | {vals[2]} | **{best_model(rows, dataset, 'mse')}** |"
        )

    lines.extend(["", "## Texture (path signature distance, lower is better)", ""])
    lines.append("| Dataset | MMPD det | Gauss det | Binary det | MMPD sample-mean | Gauss sm | Binary sm |")
    lines.append("|---------|--------:|----------:|-----------:|-----------------:|---------:|----------:|")
    for dataset in datasets:
        cells = []
        for model in MODELS:
            row = next((r for r in rows if r["dataset"] == dataset and r["model"] == model), None)
            cells.append(fnum(row.get("texture_pathsig_distance") if row else None))
        sm = []
        for model in MODELS:
            row = next((r for r in rows if r["dataset"] == dataset and r["model"] == model), None)
            sm.append(fnum(row.get("sample_mean_texture_pathsig_distance") if row else None))
        lines.append(f"| {dataset} | {cells[0]} | {cells[1]} | {cells[2]} | {sm[0]} | {sm[1]} | {sm[2]} |")

    lines.extend(["", "## Notes", ""])
    lines.append(
        "1. **illness / MMPD:** Point and top-k MMPD metrics are much worse than anchors on this tiny "
        "test subset (49 windows). Treat illness MMPD numbers as suspect until spot-checked."
    )
    lines.append(
        "2. **No per-sample texture** in this run (`per_sample_mean_texture_*` absent). "
        "Re-run `slurm_mmpd_texture_per_sample.sh --reference-run ...` to add them from cached `raw/*.npz`."
    )
    lines.append(
        "3. **Regenerate this report:** "
        f"`python utils/report_mmpd_anchor_matrix.py --run-dir {run_dir.relative_to(REPO_ROOT)}`"
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
