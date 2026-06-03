#!/usr/bin/env python3
"""Report: trend-robust texture eval, 2-stage binary vs MMPD."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_STEM = "06-03_trend_robust_texture_staged_vs_mmpd"
DEFAULT_METRICS = REPO_ROOT / "results/datasets/06-03-trend-robust-texture-staged-vs-mmpd/metrics.json"
DEFAULT_MANIFEST = REPO_ROOT / "results/datasets/06-03-trend-robust-texture-staged-vs-mmpd/run_manifest.json"
DEFAULT_ITRANS_METRICS = (
    REPO_ROOT / "results/datasets/06-03-trend-robust-texture-staged-itrans-guidance/metrics.json"
)
DEFAULT_ITRANS_MANIFEST = (
    REPO_ROOT / "results/datasets/06-03-trend-robust-texture-staged-itrans-guidance/run_manifest.json"
)
DEFAULT_DISC_METRICS = (
    REPO_ROOT / "results/datasets/06-03-discriminator-texture-staged-vs-mmpd/metrics.json"
)
DEFAULT_DISC_MANIFEST = (
    REPO_ROOT / "results/datasets/06-03-discriminator-texture-staged-vs-mmpd/run_manifest.json"
)
LOG2 = math.log(2.0)

MODEL_ORDER = ["binary_staged", "itrans_guidance", "mmpd"]
MODEL_LABELS = {
    "binary_staged": "2-stage binary",
    "itrans_guidance": "iTrans guidance",
    "mmpd": "MMPD",
}

CORE_METRICS = [
    ("mse", "MSE (anchor)"),
    ("mae", "MAE (anchor)"),
    ("crps", "CRPS (1 sample)"),
    ("top1_mse", "Top-1 mode MSE"),
    ("top1_mae", "Top-1 mode MAE"),
]

ROBUST_TEXTURE = [
    ("texture_increment_wasserstein", "Increment Wasserstein (det)"),
    ("texture_curvature_wasserstein", "Curvature Wasserstein (det)"),
    ("texture_haar_detail_jsd", "Haar detail JSD (det)"),
    ("texture_jump_plateau_distance", "Jump/plateau distance (det)"),
    ("texture_derivative_motif_jsd", "Derivative motif JSD (det)"),
]

ROBUST_TEXTURE_PROB = [
    ("prob_texture_increment_wasserstein", "Increment Wasserstein (prob)"),
    ("prob_texture_curvature_wasserstein", "Curvature Wasserstein (prob)"),
    ("prob_texture_haar_detail_jsd", "Haar detail JSD (prob)"),
    ("prob_texture_jump_plateau_distance", "Jump/plateau distance (prob)"),
    ("prob_texture_derivative_motif_jsd", "Derivative motif JSD (prob)"),
]

LEGACY_TEXTURE = [
    ("texture_ordinal_jsd", "Ordinal JSD (det)"),
    ("texture_rqa_distance", "RQA distance (det)"),
    ("texture_variogram_distance", "Variogram distance (det)"),
    ("texture_pathsig_distance", "Path signature distance (det)"),
]

LEGACY_TEXTURE_PROB = [
    ("prob_texture_ordinal_jsd", "Ordinal JSD (prob)"),
    ("prob_texture_rqa_distance", "RQA distance (prob)"),
    ("prob_texture_variogram_distance", "Variogram distance (prob)"),
    ("prob_texture_pathsig_distance", "Path signature distance (prob)"),
]

DISC_METRICS = [
    ("log2_bce_gap", "Discriminator |BCE−log2| (higher is better)"),
    ("disc_bce", "Discriminator BCE (0.6931 = chance)"),
    ("disc_acc", "Discriminator accuracy (0.5 = chance)"),
    ("disc_auroc", "Discriminator AUROC (0.5 = chance)"),
]


def fnum(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def best_models(
    table: Dict[str, Dict[str, Dict[str, float]]],
    dataset: str,
    metric: str,
) -> Set[str]:
    scored: List[Tuple[float, str]] = []
    for model in MODEL_ORDER:
        val = as_float(table.get(dataset, {}).get(model, {}).get(metric))
        if val is not None:
            scored.append((val, model))
    if not scored:
        return set()
    best = min(v for v, _ in scored)
    tol = max(1e-9, abs(best) * 1e-6)
    return {m for v, m in scored if abs(v - best) <= tol}


def fmt_val(
    table: Dict[str, Dict[str, Dict[str, float]]],
    dataset: str,
    model: str,
    metric: str,
    winners: Set[str],
) -> str:
    text = fnum(table.get(dataset, {}).get(model, {}).get(metric))
    if model in winners and text != "—":
        return f"**{text}**"
    return text


def metric_table(
    table: Dict[str, Dict[str, Dict[str, float]]],
    metric: str,
    title: str,
    datasets: List[str],
) -> List[str]:
    lines = [f"### {title}", ""]
    header = "| Dataset | " + " | ".join(MODEL_LABELS[m] for m in MODEL_ORDER) + " |"
    sep = "|---------|" + "|".join("---------:" for _ in MODEL_ORDER) + "|"
    lines.extend([header, sep])
    for ds in datasets:
        winners = best_models(table, ds, metric)
        cells = [fmt_val(table, ds, m, metric, winners) for m in MODEL_ORDER]
        if all(c == "—" for c in cells):
            continue
        lines.append(f"| {ds} | {' | '.join(cells)} |")
    lines.append("")
    return lines


def win_counts(
    table: Dict[str, Dict[str, Dict[str, float]]],
    datasets: List[str],
    metrics: List[Tuple[str, str]],
) -> Dict[str, int]:
    counts = {m: 0 for m in MODEL_ORDER}
    for key, _title in metrics:
        for ds in datasets:
            for model in best_models(table, ds, key):
                counts[model] += 1
    return counts


def merge_itrans_metrics(
    table: Dict[str, Dict[str, Dict[str, float]]],
    itrans: Dict[str, Dict[str, float]],
) -> None:
    for dataset, metrics in itrans.items():
        table.setdefault(dataset, {})["itrans_guidance"] = metrics


def disc_metric_value(
    disc: Mapping[str, Any],
    dataset: str,
    model: str,
    slice_len: int,
    metric: str,
) -> Optional[float]:
    if model == "itrans_guidance":
        return None
    row = disc.get(dataset, {}).get(model, {}).get(str(slice_len), {})
    if metric == "log2_bce_gap":
        gap = as_float(row.get("log2_bce_gap"))
        if gap is not None:
            return gap
        bce = as_float(row.get("disc_bce"))
        if bce is None:
            return None
        return abs(bce - LOG2)
    return as_float(row.get(metric))


def best_disc_models(
    disc: Mapping[str, Any],
    dataset: str,
    slice_len: int,
    metric: str,
) -> Set[str]:
    scored: List[Tuple[float, str]] = []
    for model in MODEL_ORDER:
        val = disc_metric_value(disc, dataset, model, slice_len, metric)
        if val is None:
            continue
        if metric == "log2_bce_gap":
            score = val
        elif metric == "disc_bce":
            score = -abs(val - LOG2)
        else:
            score = -abs(val - 0.5)
        scored.append((score, model))
    if not scored:
        return set()
    best = max(score for score, _model in scored)
    tol = max(1e-9, abs(best) * 1e-6)
    return {model for score, model in scored if abs(score - best) <= tol}


def discriminator_metric_table(
    disc: Mapping[str, Any],
    metric: str,
    title: str,
    datasets: List[str],
    slice_len: int,
) -> List[str]:
    lines = [f"### {title} (L={slice_len})", ""]
    header = "| Dataset | " + " | ".join(MODEL_LABELS[m] for m in MODEL_ORDER) + " |"
    sep = "|---------|" + "|".join("---------:" for _ in MODEL_ORDER) + "|"
    lines.extend([header, sep])
    for ds in datasets:
        winners = best_disc_models(disc, ds, slice_len, metric)
        cells = []
        for model in MODEL_ORDER:
            text = fnum(disc_metric_value(disc, ds, model, slice_len, metric))
            if model in winners and text != "—":
                text = f"**{text}**"
            cells.append(text)
        if all(c == "—" for c in cells):
            continue
        lines.append(f"| {ds} | {' | '.join(cells)} |")
    lines.append("")
    return lines


def disc_win_counts(
    disc: Mapping[str, Any],
    datasets: List[str],
    slice_lengths: List[int],
) -> Dict[str, int]:
    counts = {m: 0 for m in MODEL_ORDER}
    for slice_len in slice_lengths:
        for metric, _title in DISC_METRICS:
            for ds in datasets:
                for model in best_disc_models(disc, ds, slice_len, metric):
                    counts[model] += 1
    return counts


def build_report(
    table: Dict[str, Dict[str, Dict[str, float]]],
    manifest: Dict[str, Any],
    itrans_manifest: Dict[str, Any],
    disc: Optional[Mapping[str, Any]] = None,
    disc_manifest: Optional[Mapping[str, Any]] = None,
) -> str:
    datasets = list(manifest.get("datasets") or sorted(table.keys()))
    slice_lengths = [int(x) for x in (disc_manifest or {}).get("slice_lengths") or [8, 16, 32]]
    lines = [
        "# Trend-robust texture eval — 2-stage binary vs MMPD (Jun 3, 2026)",
        "",
        "Full test set (`test_fraction=1.0`), `test_stride=2`, 1× `dpmpp` sample per window "
        "(binary/MMPD). **iTrans guidance** = finetuned guidance ckpt only (deterministic). "
        "Legacy/robust metrics: **bold** = lowest. Discriminator metrics: **bold** = closest to chance "
        f"(BCE→`log(2)={LOG2:.4f}`, acc/AUROC→0.5).",
        "",
        "## Protocol",
        "",
        f"- **Metrics dir:** `results/datasets/06-03-trend-robust-texture-staged-vs-mmpd/`",
        f"- **iTrans metrics dir:** `results/datasets/06-03-trend-robust-texture-staged-itrans-guidance/`",
        f"- **Discriminator dir:** `results/datasets/06-03-discriminator-texture-staged-vs-mmpd/`",
        f"- **MMPD ckpts:** `{manifest.get('mmpd_output_root', '06-01-mmpd-binary-aligned')}`",
        f"- **Samples:** {manifest.get('sample_num', 1)} stochastic draw(s) (binary/MMPD)",
        "",
        "### 2-stage binary checkpoints",
        "",
        "| Dataset | Checkpoint |",
        "|---------|------------|",
    ]
    ckpts = manifest.get("staged_ckpts") or {}
    for ds in datasets:
        ckpt = ckpts.get(ds, "—")
        if isinstance(ckpt, dict):
            ckpt = ckpt.get("checkpoint_dir", "—")
        lines.append(f"| {ds} | `{ckpt}` |")
    lines.extend(
        [
            "",
            "### Guidance iTransformer checkpoints",
            "",
            "| Dataset | Guidance ckpt |",
            "|---------|---------------|",
        ]
    )
    itrans_ckpts = itrans_manifest.get("staged_ckpts") or {}
    for ds in datasets:
        entry = itrans_ckpts.get(ds, {})
        itrans_pt = entry.get("itrans_pt", "—") if isinstance(entry, dict) else "—"
        lines.append(f"| {ds} | `{itrans_pt}` |")
    lines.extend(["", "---", "", "## Core metrics (lower is better)", ""])
    for key, title in CORE_METRICS:
        lines.extend(metric_table(table, key, title, datasets))

    lines.extend(["---", "", "## Robust texture metrics (lower is better)", ""])
    for key, title in ROBUST_TEXTURE:
        lines.extend(metric_table(table, key, title, datasets))
    for key, title in ROBUST_TEXTURE_PROB:
        lines.extend(metric_table(table, key, title, datasets))

    lines.extend(["---", "", "## Legacy texture metrics (lower is better)", ""])
    for key, title in LEGACY_TEXTURE:
        lines.extend(metric_table(table, key, title, datasets))
    for key, title in LEGACY_TEXTURE_PROB:
        lines.extend(metric_table(table, key, title, datasets))

    if disc:
        lines.extend(
            [
                "---",
                "",
                "## Discriminator texture metrics",
                "",
                "Per-dataset classifiers on real vs stochastic horizon slices (separate disc per model). "
                "iTrans guidance has no discriminator eval. **Important:** most BCE values here are far "
                f"below chance (`log(2)={LOG2:.4f}`), meaning both generators are still easy to distinguish "
                "from GT — MMPD is usually *less* distinguishable, not indistinguishable. "
                "Use `|BCE−log2|` (higher is better) for ranking.",
                "",
            ]
        )
        for slice_len in slice_lengths:
            for key, title in DISC_METRICS:
                lines.extend(discriminator_metric_table(disc, key, title, datasets, slice_len))

    lines.extend(["---", "", "## Headlines", ""])
    all_metrics = CORE_METRICS + ROBUST_TEXTURE + ROBUST_TEXTURE_PROB + LEGACY_TEXTURE + LEGACY_TEXTURE_PROB
    robust_only = ROBUST_TEXTURE + ROBUST_TEXTURE_PROB
    for label, metric_group in [
        ("Core", CORE_METRICS),
        ("Robust texture", robust_only),
        ("All metrics", all_metrics),
    ]:
        counts = win_counts(table, datasets, metric_group)
        parts = [f"{MODEL_LABELS[m]}: {counts[m]}" for m in MODEL_ORDER]
        lines.append(f"- **{label}** row wins — " + ", ".join(parts))

    if disc:
        disc_counts = disc_win_counts(disc, datasets, slice_lengths)
        parts = [f"{MODEL_LABELS[m]}: {disc_counts[m]}" for m in MODEL_ORDER]
        lines.append(f"- **Discriminator texture** closest-to-chance wins — " + ", ".join(parts))
        lines.append(
            "- **Note:** MMPD discriminators must use MMPD-pack GT (not binary-pack GT) for the real class; "
            "re-run with `--force-train` after the y_true fix if numbers predate it."
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--itrans-metrics", type=Path, default=DEFAULT_ITRANS_METRICS)
    parser.add_argument("--itrans-manifest", type=Path, default=DEFAULT_ITRANS_MANIFEST)
    parser.add_argument("--disc-metrics", type=Path, default=DEFAULT_DISC_METRICS)
    parser.add_argument("--disc-manifest", type=Path, default=DEFAULT_DISC_MANIFEST)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "reports" / f"{REPORT_STEM}.md",
    )
    args = parser.parse_args()

    with args.metrics.open(encoding="utf-8") as f:
        table = json.load(f)
    manifest: Dict[str, Any] = {}
    if args.manifest.is_file():
        with args.manifest.open(encoding="utf-8") as f:
            manifest = json.load(f)

    itrans_manifest: Dict[str, Any] = {}
    if args.itrans_metrics.is_file():
        with args.itrans_metrics.open(encoding="utf-8") as f:
            merge_itrans_metrics(table, json.load(f))
    if args.itrans_manifest.is_file():
        with args.itrans_manifest.open(encoding="utf-8") as f:
            itrans_manifest = json.load(f)

    disc: Dict[str, Any] = {}
    disc_manifest: Dict[str, Any] = {}
    if args.disc_metrics.is_file():
        with args.disc_metrics.open(encoding="utf-8") as f:
            disc = json.load(f)
    if args.disc_manifest.is_file():
        with args.disc_manifest.open(encoding="utf-8") as f:
            disc_manifest = json.load(f)

    report = build_report(
        table,
        manifest,
        itrans_manifest,
        disc=disc or None,
        disc_manifest=disc_manifest or None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
