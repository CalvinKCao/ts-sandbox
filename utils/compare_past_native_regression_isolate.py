#!/usr/bin/env python3
"""Compare regression isolation runs for traffic / electricity past_native vs ordinal_norm.

Aggregates:
  - Full runs (Jul-08 baseline, Jul-12 past_native_g*, fresh-guidance variants)
  - Short ablations (stride2_resize g*, past_native g*)

Example:
  python utils/compare_past_native_regression_isolate.py \\
    --dataset electricity \\
    --baseline-run 07-08-4122619-electricity-binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm \\
    --regressed-run 07-12-4208598-electricity-binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g4p0 \\
    --ckpt-root "$SCRATCH/ts-sandbox/results/ckpts" \\
    --out-dir reports/isolate_past_native_electricity
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.compare_noise_sched_ablation import (  # noqa: E402
    _find_subset_dir,
    _latest_run_dir,
    _load_eval_metrics,
    _load_history,
    _rel_diff,
)

DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "traffic": {
        "g": 1.5,
        "baseline_config": "binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm",
        "regressed_config": "binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g1p5",
        "past_native_g1": "binary_noise_sched_ablation_past_native_g1p0",
        "past_native_g_tuned": "binary_noise_sched_ablation_past_native_g1p5",
        "stride2_g1": "binary_noise_sched_ablation_stride2_resize_g1p0",
        "stride2_g_tuned": "binary_noise_sched_ablation_stride2_resize_g1p5",
        "fresh_guidance": "binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g1p5_fresh_guidance",
        "eval_only_past_native": "binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_eval_only",
    },
    "electricity": {
        "g": 4.0,
        "baseline_config": "binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm",
        "regressed_config": "binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g4p0",
        "past_native_g1": "binary_noise_sched_ablation_past_native_g1p0",
        "past_native_g_tuned": "binary_noise_sched_ablation_past_native_g4p0",
        "stride2_g1": "binary_noise_sched_ablation_stride2_resize_g1p0",
        "stride2_g_tuned": "binary_noise_sched_ablation_stride2_resize_g4p0",
        "fresh_guidance": "binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g4p0_fresh_guidance",
        "eval_only_past_native": "binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_eval_only",
    },
}

METRIC_KEYS = (
    "crps",
    "anchor_mse",
    "sample_mean_mse",
    "mse",
    "mae",
)


def _resolve_ckpt_root(raw: Optional[str]) -> Path:
    if raw:
        return Path(raw).expanduser().resolve()
    import os

    scratch = os.environ.get("SCRATCH", "")
    if scratch:
        p = Path(scratch) / "ts-sandbox" / "results" / "ckpts"
        if p.is_dir():
            return p.resolve()
    return (REPO / "results" / "ckpts").resolve()


def _run_dir(ckpt_root: Path, dataset: str, stem_or_run: str) -> Path:
    if re.match(r"^\d{2}-\d{2}-\d+-", stem_or_run):
        p = ckpt_root / stem_or_run
        if not p.is_dir():
            raise FileNotFoundError(p)
        return p
    return _latest_run_dir(dataset, stem_or_run)


def _row(
    label: str,
    run_dir: Path,
    dataset: str,
    baseline_crps: Optional[float],
) -> Dict[str, Any]:
    subset = _find_subset_dir(run_dir)
    coarse = _load_history(subset / "coarse")
    fine = _load_history(subset / "fine")
    ev = _load_eval_metrics(run_dir, dataset) or {}
    metrics = ev.get("metrics") or ev
    row: Dict[str, Any] = {
        "label": label,
        "run": run_dir.name,
        "coarse_best_val": (coarse or {}).get("best_val"),
        "fine_best_val": (fine or {}).get("best_val"),
    }
    for k in METRIC_KEYS:
        if k in metrics:
            row[k] = float(metrics[k])
    crps = row.get("crps")
    if baseline_crps is not None and crps is not None:
        row["crps_rel_vs_baseline"] = _rel_diff(baseline_crps, crps)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIGS))
    ap.add_argument("--baseline-run", required=True, help="Run dir or config stem for Jul-08 baseline")
    ap.add_argument("--regressed-run", required=True, help="Run dir or config stem for Jul-12 past_native_g*")
    ap.add_argument("--ckpt-root", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    spec = DATASET_CONFIGS[args.dataset]
    ckpt_root = _resolve_ckpt_root(args.ckpt_root)
    out_dir = Path(args.out_dir or f"reports/isolate_past_native_{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = _run_dir(ckpt_root, args.dataset, args.baseline_run)
    regressed_dir = _run_dir(ckpt_root, args.dataset, args.regressed_run)

    rows: List[Dict[str, Any]] = []
    baseline_row = _row("A_baseline_ordinal_norm", baseline_dir, args.dataset, None)
    baseline_crps = baseline_row.get("crps")
    rows.append(baseline_row)
    rows.append(_row("B_regressed_past_native_g", regressed_dir, args.dataset, baseline_crps))

    optional: List[tuple[str, str]] = [
        ("C_past_native_g1_ablation", spec["past_native_g1"]),
        ("D_past_native_g_tuned_ablation", spec["past_native_g_tuned"]),
        ("E_stride2_g1_ablation", spec["stride2_g1"]),
        ("F_stride2_g_tuned_ablation", spec["stride2_g_tuned"]),
        ("G_fresh_guidance_full", spec["fresh_guidance"]),
        ("H_baseline_eval_past_native_geom", spec["eval_only_past_native"]),
    ]
    for label, stem in optional:
        try:
            rd = _latest_run_dir(args.dataset, stem)
        except FileNotFoundError:
            rows.append({"label": label, "run": f"<missing:{stem}>", "note": "not submitted yet"})
            continue
        rows.append(_row(label, rd, args.dataset, baseline_crps))

    csv_path = out_dir / "isolation_metrics.csv"
    fieldnames = sorted({k for r in rows for k in r})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    summary_path = out_dir / "decision_tree.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# past_native regression isolation — {args.dataset}\n\n")
        f.write(f"- Baseline: `{baseline_dir.name}`\n")
        f.write(f"- Regressed: `{regressed_dir.name}`\n")
        f.write(f"- Tuned g: {spec['g']}\n\n")
        f.write("## How to read rows\n\n")
        f.write("| Row | Isolates |\n")
        f.write("|-----|----------|\n")
        f.write("| C vs A | past_native geometry only (g=1.0) |\n")
        f.write("| D vs C | g on past_native geometry |\n")
        f.write("| E vs A | g=1.0 on old resize=true geometry (short retrain) |\n")
        f.write("| F vs E | g alone on old geometry |\n")
        f.write("| G vs B | guidance reuse vs fresh in-run guidance |\n")
        f.write("| H vs A | eval-time geometry mismatch on baseline weights |\n\n")
        f.write("## Metrics\n\n")
        f.write("| label | run | crps | anchor_mse | crps_rel_vs_baseline |\n")
        f.write("|-------|-----|------|------------|----------------------|\n")
        for r in rows:
            f.write(
                f"| {r.get('label','')} | {r.get('run','')} | "
                f"{r.get('crps','')} | {r.get('anchor_mse','')} | "
                f"{r.get('crps_rel_vs_baseline','')} |\n"
            )

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
