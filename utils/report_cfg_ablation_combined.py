#!/usr/bin/env python3
"""Combine MMPD matrix eval + binary CFG-off + CFG inference ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

# (model_key, column label, default results dir under results/datasets/)
CFG_INFERENCE_COLUMNS: List[Tuple[str, str, str]] = [
    ("binary_w1_1", "Binary (CFG w=1.1)", "06-01-cfg-ablation-cfg1.1"),
    ("binary_w1_5", "Binary (CFG w=1.5)", "06-01-cfg-ablation-cfg1.5"),
    ("binary_w4", "Binary (CFG w=4)", "06-01-cfg-ablation-l40s-cfg4"),
    ("binary_w10", "Binary (CFG w=10)", "06-01-cfg-ablation-l40s-cfg10"),
]

MODEL_ORDER = ["mmpd", "binary_cfg_off"] + [k for k, _, _ in CFG_INFERENCE_COLUMNS]
MODEL_LABELS: Dict[str, str] = {
    "mmpd": "MMPD",
    "binary_cfg_off": "Binary (CFG off)",
    **{k: label for k, label, _ in CFG_INFERENCE_COLUMNS},
}

CORE_METRICS = [
    ("mse", "MSE (deterministic / anchor)"),
    ("mae", "MAE (deterministic / anchor)"),
    ("crps", "CRPS (100 stochastic draws)"),
    ("top1_mse", "Top-1 mode MSE"),
    ("top1_mae", "Top-1 mode MAE"),
    ("top3_mse", "Top-3 mode MSE"),
    ("top3_mae", "Top-3 mode MAE"),
]

TEXTURE_METRICS = [
    ("texture_ordinal_jsd", "Ordinal JSD (deterministic)"),
    ("texture_rqa_distance", "RQA distance (deterministic)"),
    ("texture_variogram_distance", "Variogram distance (deterministic)"),
    ("texture_pathsig_distance", "Path signature distance (deterministic)"),
    ("prob_texture_ordinal_jsd", "Ordinal JSD (prob., first 3 draws)"),
    ("prob_texture_rqa_distance", "RQA distance (prob., first 3 draws)"),
    ("prob_texture_variogram_distance", "Variogram distance (prob., first 3 draws)"),
    ("prob_texture_pathsig_distance", "Path signature distance (prob., first 3 draws)"),
]

CFG_ABLATION_JOBS: Dict[str, List[Tuple[str, str]]] = {
    "1.1": [
        ("3842420", "ETTh1"),
        ("3842421", "ETTh2"),
        ("3842422", "exchange_rate"),
        ("3842423", "weather"),
        ("3842424", "traffic"),
        ("3842425", "PeMS"),
        ("3842426", "dalia"),
    ],
    "1.5": [
        ("3841891", "ETTh1"),
        ("3841892", "ETTh2"),
        ("3841893", "exchange_rate"),
        ("3841894", "weather"),
        ("3841895", "traffic"),
        ("3841896", "PeMS"),
        ("3841897", "dalia"),
    ],
    "4": [
        ("3839781", "ETTh1"),
        ("3839782", "ETTh2"),
        ("3839783", "exchange_rate"),
        ("3839784", "weather"),
        ("3839785", "traffic"),
        ("3839786", "PeMS"),
        ("3839787", "dalia"),
    ],
    "10": [
        ("3839788", "ETTh1"),
        ("3839789", "ETTh2"),
        ("3839790", "exchange_rate"),
        ("3839791", "weather"),
        ("3839792", "traffic"),
        ("3839793", "PeMS"),
        ("3839794", "dalia"),
        ("3839918", "weather"),
        ("3839919", "traffic"),
        ("3839920", "PeMS"),
        ("3839921", "dalia"),
    ],
}


def load_partial(path: Path) -> Dict[str, float]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


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


def resolve_cfg_dirs(
    datasets_root: Path,
    overrides: Dict[str, Path],
) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for key, _, default_name in CFG_INFERENCE_COLUMNS:
        if key in overrides:
            out[key] = overrides[key].resolve()
        else:
            out[key] = (datasets_root / default_name).resolve()
    return out


def load_combined(
    matrix_dir: Path,
    cfg_dirs: Dict[str, Path],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    table: Dict[str, Dict[str, Dict[str, float]]] = {}
    partials = matrix_dir / "partials"

    for path in sorted(partials.glob("*_mmpd.json")):
        ds = path.name.replace("_mmpd.json", "")
        table.setdefault(ds, {})["mmpd"] = load_partial(path)

    for path in sorted(partials.glob("*_binary_anchor.json")):
        ds = path.name.replace("_binary_anchor.json", "")
        table.setdefault(ds, {})["binary_cfg_off"] = load_partial(path)

    for model_key, cfg_dir in cfg_dirs.items():
        cdir = cfg_dir / "partials"
        if not cdir.is_dir():
            continue
        for path in sorted(cdir.glob("*_binary_anchor.json")):
            ds = path.name.replace("_binary_anchor.json", "")
            table.setdefault(ds, {})[model_key] = load_partial(path)

    return table


def metric_table_markdown(
    table: Dict[str, Dict[str, Dict[str, float]]],
    metric: str,
    title: str,
    datasets: List[str],
) -> List[str]:
    active_models = [
        m for m in MODEL_ORDER
        if any(as_float(table.get(ds, {}).get(m, {}).get(metric)) is not None for ds in datasets)
    ]
    if not active_models:
        return []

    lines = [f"### {title}", ""]
    header = "| Dataset | " + " | ".join(MODEL_LABELS[m] for m in active_models) + " |"
    sep = "|---------|" + "|".join("---------:" for _ in active_models) + "|"
    lines.extend([header, sep])

    for ds in datasets:
        winners = best_models(table, ds, metric)
        cells = [fmt_val(table, ds, m, metric, winners) for m in active_models]
        if all(c == "—" for c in cells):
            continue
        lines.append(f"| {ds} | {' | '.join(cells)} |")
    lines.append("")
    return lines


def build_report(
    table: Dict[str, Dict[str, Dict[str, float]]],
    matrix_dir: Path,
    cfg_dirs: Dict[str, Path],
    report_path: Path,
) -> None:
    datasets = sorted(table.keys())
    lines: List[str] = [
        "# CFG ablation + MMPD matrix — combined eval (Jun 1, 2026)",
        "",
        "Aligned eval: 50% seeded test windows, 100× `dpmpp` (20 steps) for probabilistic metrics, "
        "1× anchor for deterministic MSE/MAE/texture. **Bold** = lowest value in that row.",
        "",
        "## Sources",
        "",
        "| Component | Path / jobs |",
        "|-----------|-------------|",
        f"| **Binary (CFG off)** — jobs 3828089–3828100, matrix re-eval 3838179+ | "
        f"`{matrix_dir.relative_to(REPO_ROOT)}` — `configs/binary_dual_scale.yaml` (no CFG train/infer); partials `*_binary_anchor.json` |",
        f"| MMPD (same matrix) | `{matrix_dir.relative_to(REPO_ROOT)}` — partials `*_mmpd.json` |",
    ]
    for model_key, label, default_name in CFG_INFERENCE_COLUMNS:
        cfg_dir = cfg_dirs[model_key]
        w = label.split("w=")[-1].rstrip(")")
        job_note = f"jobs in `results/logs/cfg_ablation/06-01-cfg-ablation-cfg{w}-*`"
        lines.append(
            f"| {label} | `{cfg_dir.relative_to(REPO_ROOT)}/partials/` — {job_note} |"
        )
    lines.extend(
        [
            "",
            "**Coverage notes:**",
            "- **CFG off:** 3828089 weights, no inference CFG. 9/12 datasets in matrix partials "
            "(missing ETTh1, ETTh2, PeMS binary at merge time).",
            "- **CFG w=1.1 / 1.5 / 4 / 10:** inference-only on 3828089 ckpts, 7 ablation datasets each.",
            "- **MMPD:** all 12 datasets in matrix partials.",
            "",
            "## Slurm — CFG ablation (completed)",
            "",
            "| Job | Dataset | CFG w |",
            "|-----|---------|------:|",
        ]
    )

    for scale_label, jobs in CFG_ABLATION_JOBS.items():
        seen = set()
        for job_id, ds in jobs:
            key = (scale_label, ds)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"| {job_id} | {ds} | {scale_label} |")

    lines.extend(
        [
            "",
            "## Metric glossary",
            "",
            "| Metric | Path |",
            "|--------|------|",
            "| MSE / MAE (det) | Single anchor decode per window |",
            "| CRPS, top1/top3 | 100 stochastic samples, GMM modes (top2 omitted) |",
            "| `texture_*` | Deterministic anchor vs ground truth |",
            "| `prob_texture_*` | Mean texture over first 3 probabilistic draws |",
            "",
            "---",
            "",
            "## Core metrics (lower is better)",
            "",
        ]
    )

    for key, title in CORE_METRICS:
        lines.extend(metric_table_markdown(table, key, title, datasets))

    lines.extend(["---", "", "## Texture metrics (lower is better)", ""])
    for key, title in TEXTURE_METRICS:
        lines.extend(metric_table_markdown(table, key, title, datasets))

    lines.extend(["## Headlines", ""])
    for key, title in CORE_METRICS[:3]:
        counts: Dict[str, int] = {m: 0 for m in MODEL_ORDER}
        for ds in datasets:
            winners = best_models(table, ds, key)
            for m in winners:
                counts[m] += 1
        parts = [f"{MODEL_LABELS[m]}: {counts[m]}" for m in MODEL_ORDER if counts[m]]
        lines.append(f"- **{title.split('(')[0].strip()}** — row wins: {', '.join(parts) or '—'}")

    regen_lines = [
        "python utils/report_cfg_ablation_combined.py \\",
        f"  --matrix-dir {matrix_dir.relative_to(REPO_ROOT)}",
    ]
    for model_key, _, default_name in CFG_INFERENCE_COLUMNS:
        regen_lines.append(
            f"  --{model_key.replace('binary_w', 'cfg')}-dir results/datasets/{default_name} \\"
        )
    regen_lines[-1] = regen_lines[-1].rstrip(" \\")

    lines.extend(
        [
            "",
            "## Visualizations",
            "",
            "CFG-off: `viz/`. CFG inference: `viz_cfg1.1/`, `viz_cfg1.5/`, `viz_cfg4/`, `viz_cfg10/` "
            "(see `utils/visualize_report_binary_dual_scale.py --cfg-ablation`).",
            "",
            "## Regenerate tables",
            "",
            "```bash",
            *regen_lines,
            "```",
            "",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        default=REPO_ROOT / "results/datasets/06-01-mmpd-binary-aligned",
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=REPO_ROOT / "results/datasets",
    )
    for model_key, _, default_name in CFG_INFERENCE_COLUMNS:
        arg_name = f"--{model_key.replace('binary_w', 'cfg')}-dir"
        parser.add_argument(
            arg_name,
            type=Path,
            default=None,
            help=f"Override default {default_name}",
        )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPO_ROOT / "reports/06-01_cfg_ablation_mmpd_matrix_combined.md",
    )
    args = parser.parse_args()

    overrides: Dict[str, Path] = {}
    for model_key, _, _ in CFG_INFERENCE_COLUMNS:
        arg_attr = model_key.replace("binary_w", "cfg") + "_dir"
        val = getattr(args, arg_attr, None)
        if val is not None:
            overrides[model_key] = val

    cfg_dirs = resolve_cfg_dirs(args.datasets_root, overrides)
    table = load_combined(args.matrix_dir.resolve(), cfg_dirs)
    build_report(table, args.matrix_dir.resolve(), cfg_dirs, args.report_path.resolve())


if __name__ == "__main__":
    main()
