#!/usr/bin/env python3
"""Report learned discriminator texture eval for staged binary vs MMPD."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "results/datasets/06-03-discriminator-texture-staged-vs-mmpd/metrics.json"
DEFAULT_MANIFEST = REPO_ROOT / "results/datasets/06-03-discriminator-texture-staged-vs-mmpd/run_manifest.json"
DEFAULT_OUTPUT = REPO_ROOT / "reports/06-03_discriminator_texture_staged_vs_mmpd.md"
LOG2 = math.log(2.0)

MODEL_LABELS = {
    "binary_staged": "2-stage binary",
    "mmpd": "MMPD",
}
MODEL_ORDER = ["binary_staged", "mmpd"]


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 4) -> str:
    val = as_float(value)
    if val is None or math.isnan(val):
        return "-"
    return f"{val:.{digits}f}"


def metric_value(
    results: Mapping[str, Any],
    dataset: str,
    model: str,
    slice_len: int,
    metric: str,
) -> Optional[float]:
    return as_float(
        results.get(dataset, {})
        .get(model, {})
        .get(str(slice_len), {})
        .get(metric)
    )


def metric_score(metric: str, value: float) -> float:
    if metric == "disc_bce":
        return -abs(value - LOG2)
    if metric in {"disc_acc", "disc_auroc"}:
        return -abs(value - 0.5)
    return -value


def best_models(
    results: Mapping[str, Any],
    dataset: str,
    slice_len: int,
    metric: str,
) -> List[str]:
    scored: List[Tuple[float, str]] = []
    for model in MODEL_ORDER:
        val = metric_value(results, dataset, model, slice_len, metric)
        if val is not None:
            scored.append((metric_score(metric, val), model))
    if not scored:
        return []
    best = max(score for score, _model in scored)
    tol = max(1e-9, abs(best) * 1e-6)
    return [model for score, model in scored if abs(score - best) <= tol]


def metric_table(
    results: Mapping[str, Any],
    datasets: List[str],
    slice_len: int,
    metric: str,
    title: str,
) -> List[str]:
    lines = [f"### {title} (L={slice_len})", ""]
    lines.append("| Dataset | " + " | ".join(MODEL_LABELS[m] for m in MODEL_ORDER) + " |")
    lines.append("|---------|" + "|".join("---------:" for _ in MODEL_ORDER) + "|")
    for dataset in datasets:
        winners = set(best_models(results, dataset, slice_len, metric))
        cells = []
        for model in MODEL_ORDER:
            text = fmt(metric_value(results, dataset, model, slice_len, metric))
            if model in winners and text != "-":
                text = f"**{text}**"
            cells.append(text)
        lines.append(f"| {dataset} | {' | '.join(cells)} |")
    lines.append("")
    return lines


def win_counts(
    results: Mapping[str, Any],
    datasets: List[str],
    slice_lengths: List[int],
    metric: str,
) -> Dict[str, int]:
    counts = {model: 0 for model in MODEL_ORDER}
    for slice_len in slice_lengths:
        for dataset in datasets:
            for model in best_models(results, dataset, slice_len, metric):
                counts[model] += 1
    return counts


def build_report(results: Dict[str, Any], manifest: Dict[str, Any]) -> str:
    datasets = list(manifest.get("datasets") or results.keys())
    slice_lengths = [int(x) for x in manifest.get("slice_lengths") or [8, 16, 32]]
    lines = [
        "# Discriminator texture eval — 2-stage binary vs MMPD",
        "",
        "Per-dataset iTransformer-style discriminators classify real horizon slices vs stochastic generated slices. "
        "Harder-to-distinguish outputs are better: BCE closer to `log(2)=0.6931`, accuracy closer to `0.5`, "
        "and AUROC closer to `0.5`. **Bold** marks the value closest to chance.",
        "",
        "## Protocol",
        "",
        f"- **Results dir:** `{manifest.get('results_dir', DEFAULT_RESULTS.parent.relative_to(REPO_ROOT))}`",
        f"- **Raw eval dir:** `{manifest.get('raw_eval_dir', 'results/datasets/06-03-trend-robust-texture-staged-vs-mmpd')}`",
        f"- **Slice lengths:** {', '.join(str(x) for x in slice_lengths)}",
        f"- **Datasets:** {', '.join(datasets)}",
        "",
        "---",
        "",
        "## BCE Loss",
        "",
    ]
    for slice_len in slice_lengths:
        lines.extend(metric_table(results, datasets, slice_len, "disc_bce", "Held-out BCE"))

    lines.extend(["---", "", "## Accuracy", ""])
    for slice_len in slice_lengths:
        lines.extend(metric_table(results, datasets, slice_len, "disc_acc", "Held-out accuracy"))

    lines.extend(["---", "", "## AUROC", ""])
    for slice_len in slice_lengths:
        lines.extend(metric_table(results, datasets, slice_len, "disc_auroc", "Held-out AUROC"))

    lines.extend(["---", "", "## Headlines", ""])
    for metric, label in [
        ("disc_bce", "BCE closest to chance"),
        ("disc_acc", "Accuracy closest to 0.5"),
        ("disc_auroc", "AUROC closest to 0.5"),
    ]:
        counts = win_counts(results, datasets, slice_lengths, metric)
        parts = [f"{MODEL_LABELS[m]}: {counts[m]}" for m in MODEL_ORDER]
        lines.append(f"- **{label}** — " + ", ".join(parts))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    results = read_json(args.metrics)
    if not results:
        raise FileNotFoundError(f"No discriminator metrics found: {args.metrics}")
    manifest = read_json(args.manifest)
    manifest.setdefault("results_dir", str(args.metrics.parent.relative_to(REPO_ROOT)))
    report = build_report(results, manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
