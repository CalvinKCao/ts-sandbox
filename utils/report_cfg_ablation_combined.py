#!/usr/bin/env python3
"""Combine MMPD matrix eval + binary CFG-off + CFG inference ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_STEM = "06-01_cfg_ablation_mmpd_matrix_combined"

# (model_key, column label, default results dir under results/datasets/)
CFG_INFERENCE_COLUMNS: List[Tuple[str, str, str]] = [
    ("binary_w1_1", "Binary (CFG w=1.1)", "06-01-cfg-ablation-cfg1.1"),
    ("binary_w1_5", "Binary (CFG w=1.5)", "06-01-cfg-ablation-cfg1.5"),
    ("binary_w4", "Binary (CFG w=4)", "06-01-cfg-ablation-l40s-cfg4"),
    ("binary_w10", "Binary (CFG w=10)", "06-01-cfg-ablation-l40s-cfg10"),
]

PATCH48_KEY = "patch48_4x4"
DEFAULT_PATCH48_DIR = "06-02-patch48-redo-dpmpp"

STAGED_KEY = "binary_2stage"
STAGED_NORM_KEY = "binary_2stage_norm"
DEFAULT_STAGED_DIR = "06-02-staged-grid-dpmpp"

MODEL_ORDER = [
    "mmpd",
    "binary_cfg_off",
    PATCH48_KEY,
    STAGED_KEY,
    STAGED_NORM_KEY,
] + [k for k, _, _ in CFG_INFERENCE_COLUMNS]
MODEL_LABELS: Dict[str, str] = {
    "mmpd": "MMPD",
    "binary_cfg_off": "Binary (CFG off)",
    PATCH48_KEY: "4x4 patch",
    STAGED_KEY: "2-stage",
    STAGED_NORM_KEY: "2-stage (q99.5 MS)",
    **{k: label for k, label, _ in CFG_INFERENCE_COLUMNS},
}

CFG_OFF_REDO_JOBS: List[Tuple[str, str]] = [
    ("3848045", "ETTh1"),
    ("3848046", "ETTh2"),
    ("3848047", "PeMS"),
]

PATCH48_JOBS: List[Tuple[str, str]] = [
    ("3848019", "ETTm1"),
    ("3848020", "ETTm2"),
    ("3848021", "dalia"),
    ("3848022", "electricity"),
    ("3848023", "exchange_rate"),
    ("3848024", "solar_Alabama"),
    ("3848025", "traffic"),
    ("3848026", "weather"),
    ("3848027", "merge"),
]

STAGED_GRID_JOBS: List[Tuple[str, str]] = [
    ("3849018", "ETTh1"),
    ("3849019", "ETTh2"),
    ("3849020", "PeMS"),
    ("3849021", "dalia"),
    ("3849022", "exchange_rate"),
    ("3849023", "traffic"),
]

# q99.5 max_scale_by_dataset + window_norm_std_floor=0.1 retrain (Jun 2–3, 2026)
STAGED_NORM_GRID_JOBS: List[Tuple[str, str]] = [
    ("3852944", "ETTh1"),
    ("3852945", "ETTh2"),
    ("3852946", "ETTm1"),
    ("3852947", "ETTm2"),
    ("3852948", "illness"),
    ("3852949", "exchange_rate"),
    ("3852950", "weather"),
    ("3852951", "electricity"),
    ("3852952", "traffic"),
    ("3852953", "PeMS"),
    ("3852954", "solar_Alabama"),
    ("3852955", "dalia"),
]

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


def load_binary_partials(
    table: Dict[str, Dict[str, Dict[str, float]]],
    partials_dir: Path,
    model_key: str,
) -> None:
    if not partials_dir.is_dir():
        return
    for path in sorted(partials_dir.glob("*_binary_anchor.json")):
        ds = path.name.replace("_binary_anchor.json", "")
        table.setdefault(ds, {})[model_key] = load_partial(path)


def _staged_sources(
    datasets_root: Path,
    job_ids: List[Tuple[str, str]],
    staged_dir: Optional[Path],
) -> List[Path]:
    if staged_dir is not None and (staged_dir / "partials").is_dir():
        return [staged_dir]
    sources: List[Path] = []
    for job_id, _ in job_ids:
        sources.extend(sorted(datasets_root.glob(f"06-02-{job_id}-*")))
    return sources


def load_staged_partials(
    table: Dict[str, Dict[str, Dict[str, float]]],
    datasets_root: Path,
    staged_dir: Optional[Path] = None,
) -> None:
    """Load coarse→fine staged eval metrics (*_staged_anchor.json)."""
    for run_dir in _staged_sources(datasets_root, STAGED_GRID_JOBS, staged_dir):
        partials = run_dir / "partials"
        if not partials.is_dir():
            continue
        for path in sorted(partials.glob("*_staged_anchor.json")):
            ds = path.name.replace("_staged_anchor.json", "")
            table.setdefault(ds, {})[STAGED_KEY] = load_partial(path)


def load_staged_norm_partials(
    table: Dict[str, Dict[str, Dict[str, float]]],
    datasets_root: Path,
) -> None:
    """Load q99.5 max_scale + std-floor staged retrain (jobs 3852944–3852955)."""
    for run_dir in _staged_sources(datasets_root, STAGED_NORM_GRID_JOBS, None):
        partials = run_dir / "partials"
        if not partials.is_dir():
            continue
        for path in sorted(partials.glob("*_staged_anchor.json")):
            ds = path.name.replace("_staged_anchor.json", "")
            table.setdefault(ds, {})[STAGED_NORM_KEY] = load_partial(path)


def load_combined(
    matrix_dir: Path,
    cfg_dirs: Dict[str, Path],
    patch48_dir: Optional[Path] = None,
    staged_dir: Optional[Path] = None,
    datasets_root: Optional[Path] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    table: Dict[str, Dict[str, Dict[str, float]]] = {}
    partials = matrix_dir / "partials"

    for path in sorted(partials.glob("*_mmpd.json")):
        ds = path.name.replace("_mmpd.json", "")
        table.setdefault(ds, {})["mmpd"] = load_partial(path)

    load_binary_partials(table, partials, "binary_cfg_off")

    if patch48_dir is not None:
        load_binary_partials(table, patch48_dir / "partials", PATCH48_KEY)

    root = datasets_root if datasets_root is not None else matrix_dir.parent
    load_staged_partials(table, root, staged_dir)
    load_staged_norm_partials(table, root)

    for model_key, cfg_dir in cfg_dirs.items():
        load_binary_partials(table, cfg_dir / "partials", model_key)

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


def count_datasets_with_model(
    table: Dict[str, Dict[str, Dict[str, float]]],
    model_key: str,
) -> int:
    return sum(1 for ds in table if table[ds].get(model_key))


def build_report(
    table: Dict[str, Dict[str, Dict[str, float]]],
    matrix_dir: Path,
    cfg_dirs: Dict[str, Path],
    report_path: Path,
    patch48_dir: Optional[Path] = None,
    staged_dir: Optional[Path] = None,
    staged_pending: Optional[List[str]] = None,
) -> None:
    datasets = sorted(table.keys())
    n_cfg_off = count_datasets_with_model(table, "binary_cfg_off")
    n_patch48 = count_datasets_with_model(table, PATCH48_KEY)
    n_staged = count_datasets_with_model(table, STAGED_KEY)
    n_staged_norm = count_datasets_with_model(table, STAGED_NORM_KEY)
    patch48_rel = (
        patch48_dir.relative_to(REPO_ROOT)
        if patch48_dir is not None
        else Path("results/datasets") / DEFAULT_PATCH48_DIR
    )
    if staged_dir is not None:
        staged_note = (
            f"`{staged_dir.relative_to(REPO_ROOT)}` — merged partials `*_staged_anchor.json`"
        )
    else:
        staged_note = (
            "per-job `results/datasets/06-02-3849018-*-binary_dual_scale_staged/partials/` "
            "(auto-discovered by job id)"
        )

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
        f"| **Binary (CFG off)** — jobs 3828089–3828100, matrix re-eval 3838179+; ETTh1/ETTh2/PeMS redo 3848045–3848047 | "
        f"`{matrix_dir.relative_to(REPO_ROOT)}` — `configs/binary_dual_scale.yaml` (no CFG train/infer); partials `*_binary_anchor.json` |",
        f"| MMPD (same matrix) | `{matrix_dir.relative_to(REPO_ROOT)}` — partials `*_mmpd.json` |",
        f"| **4x4 patch** — jobs 3848019–3848026, merge 3848027 | "
        f"`{patch48_rel}` — patch-48 binary ckpts, aligned `dpmpp` eval; partials `*_binary_anchor.json` |",
        f"| **2-stage** — jobs 3849018–3849023 (coarse→fine grid) | {staged_note} |",
        "| **2-stage (q99.5 MS)** — jobs 3852944–3852955 | "
        "per-job `results/datasets/06-02-3852944-*-binary_dual_scale_staged/partials/` "
        "(`max_scale_by_dataset` + `window_norm_std_floor: 0.1`) |",
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
            f"- **CFG off:** 3828089 weights, no inference CFG. {n_cfg_off}/12 datasets in matrix partials "
            "(ETTh1, ETTh2, PeMS from jobs 3848045–3848047).",
            f"- **4x4 patch:** patch-48 binary weights; {n_patch48}/12 datasets "
            "(ETTm1/2, dalia, electricity, exchange_rate, solar_Alabama, traffic, weather).",
            f"- **2-stage:** coarse→fine staged binary; {n_staged}/6 grid jobs with eval partials "
            f"(ETTh1, ETTh2, dalia, exchange_rate, traffic"
            + ("" if not staged_pending else f"; pending: {', '.join(staged_pending)}")
            + ").",
            f"- **2-stage (q99.5 MS):** full 12-dataset retrain with calibrated per-dataset "
            f"`max_scale` and `window_norm_std_floor: 0.1`; {n_staged_norm}/12 eval partials.",
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
            "## Slurm — CFG off binary redo (completed)",
            "",
            "| Job | Dataset | CFG w |",
            "|-----|---------|------:|",
        ]
    )
    for job_id, ds in CFG_OFF_REDO_JOBS:
        lines.append(f"| {job_id} | {ds} | 1 (off) |")

    lines.extend(
        [
            "",
            "## Slurm — 4x4 patch sampler redo (completed)",
            "",
            "| Job | Dataset | Notes |",
            "|-----|---------|-------|",
        ]
    )
    for job_id, ds in PATCH48_JOBS:
        note = "merge" if ds == "merge" else "dpmpp eval"
        lines.append(f"| {job_id} | {ds} | {note} |")

    lines.extend(
        [
            "",
            "## Slurm — 2-stage grid (binary_dual_scale_staged)",
            "",
            "| Job | Dataset | Status |",
            "|-----|---------|--------|",
        ]
    )
    staged_done = {
        ds for ds in table if table[ds].get(STAGED_KEY)
    }
    for job_id, ds in STAGED_GRID_JOBS:
        if ds in staged_done:
            status = "completed"
        elif staged_pending and ds in staged_pending:
            status = "pending"
        else:
            status = "—"
        lines.append(f"| {job_id} | {ds} | {status} |")

    staged_norm_done = {ds for ds in table if table[ds].get(STAGED_NORM_KEY)}
    lines.extend(
        [
            "",
            "## Slurm — 2-stage norm-cal grid (binary_dual_scale_staged, jobs 3852944–3852955)",
            "",
            "| Job | Dataset | Status |",
            "|-----|---------|--------|",
        ]
    )
    for job_id, ds in STAGED_NORM_GRID_JOBS:
        status = "completed" if ds in staged_norm_done else "—"
        lines.append(f"| {job_id} | {ds} | {status} |")

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
        f"  --matrix-dir {matrix_dir.relative_to(REPO_ROOT)} \\",
        f"  --patch48-dir {patch48_rel} \\",
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
            "Forecast panels (GT, iTrans guidance/baseline, anchor, 5× dpmpp, extra lookbacks): "
            f"`{REPORT_STEM}/viz_cfg_off/`, `{REPORT_STEM}/viz_4x4_patch/`, `{REPORT_STEM}/viz_2stage/` "
            "(see `utils/visualize_report_cfg_ablation_combined.py`). "
            "CFG inference 2D denoise: `viz_cfg1.1/`, `viz_cfg1.5/`, `viz_cfg4/`, `viz_cfg10/` "
            "(see `utils/visualize_report_binary_dual_scale.py --cfg-ablation`).",
            "",
            "## Regenerate tables",
            "",
            "2-stage loads from `06-02-3849018-*` job dirs unless `--staged-dir` is set. "
            "2-stage (q99.5 MS) auto-discovers `06-02-3852944-*` / `06-02-385295*`.",
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
        "--patch48-dir",
        type=Path,
        default=None,
        help=f"4x4 patch eval partials (default: datasets/{DEFAULT_PATCH48_DIR})",
    )
    parser.add_argument(
        "--staged-dir",
        type=Path,
        default=None,
        help=(
            f"2-stage merged partials (default: datasets/{DEFAULT_STAGED_DIR} if present, "
            "else auto-discover 06-02-3849018-* job dirs)"
        ),
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

    patch48_dir = (
        args.patch48_dir.resolve()
        if args.patch48_dir is not None
        else (args.datasets_root / DEFAULT_PATCH48_DIR).resolve()
    )
    if not (patch48_dir / "partials").is_dir():
        patch48_dir = None

    staged_dir: Optional[Path] = None
    if args.staged_dir is not None:
        staged_dir = args.staged_dir.resolve()
    else:
        default_staged = (args.datasets_root / DEFAULT_STAGED_DIR).resolve()
        if (default_staged / "partials").is_dir():
            staged_dir = default_staged

    cfg_dirs = resolve_cfg_dirs(args.datasets_root, overrides)
    table = load_combined(
        args.matrix_dir.resolve(),
        cfg_dirs,
        patch48_dir,
        staged_dir,
        args.datasets_root.resolve(),
    )
    staged_pending = [
        ds for _, ds in STAGED_GRID_JOBS if STAGED_KEY not in table.get(ds, {})
    ]
    build_report(
        table,
        args.matrix_dir.resolve(),
        cfg_dirs,
        args.report_path.resolve(),
        patch48_dir,
        staged_dir,
        staged_pending,
    )


if __name__ == "__main__":
    main()
