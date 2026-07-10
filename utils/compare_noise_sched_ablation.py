#!/usr/bin/env python3
"""Compare noise-schedule ablation runs: val-loss curves + staged eval metrics.

Looks under results/ckpts/*-<dataset>-<config_stem>/ for:
  <subset>/coarse/val_loss_history.json
  <subset>/fine/val_loss_history.json
  and results/<same-run-stem>/partials/<dataset>_staged_anchor.json (if present)

Example:
  python utils/compare_noise_sched_ablation.py \\
    --dataset electricity \\
    --configs \\
      binary_noise_sched_ablation_elec_unc_g1p0 \\
      binary_noise_sched_ablation_elec_unc_g1p5 \\
      binary_noise_sched_ablation_elec_unc_g3p0 \\
    --out-dir reports/noise_sched_ablation_elec_unc
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
CKPT_ROOT = REPO / "results" / "ckpts"
RESULTS_ROOT = REPO / "results"

VARIANT_LABELS = {
    "binary_noise_sched_ablation_elec_unc_g1p0": "BASELINE g=1.0",
    "binary_noise_sched_ablation_elec_unc_g1p5": "GLOBAL-SHIFT g=1.5",
    "binary_noise_sched_ablation_elec_unc_g3p0": "PER-DATASET g=3.0",
}


def _latest_run_dir(dataset: str, config_stem: str) -> Path:
    pat = re.compile(
        rf"^\d{{2}}-\d{{2}}-\d+-{re.escape(dataset)}-{re.escape(config_stem)}$"
    )
    matches = [p for p in CKPT_ROOT.iterdir() if p.is_dir() and pat.match(p.name)]
    if not matches:
        raise FileNotFoundError(
            f"No ckpt dir matching *-{dataset}-{config_stem} under {CKPT_ROOT}"
        )
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def _find_subset_dir(run_dir: Path) -> Path:
    kids = [p for p in run_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not kids:
        raise FileNotFoundError(f"No subset dir under {run_dir}")
    # Prefer electricity_4v_s1-style names
    for p in kids:
        if "electricity" in p.name or p.name.endswith("_s1"):
            return p
    return kids[0]


def _load_history(stage_dir: Path) -> Optional[Dict[str, Any]]:
    path = stage_dir / "val_loss_history.json"
    if not path.is_file():
        meta = stage_dir / "metadata.json"
        if meta.is_file():
            with open(meta, encoding="utf-8") as f:
                m = json.load(f)
            return {
                "best_val": float(m.get("best_val_loss", float("nan"))),
                "best_epoch": int(m.get("best_epoch", 0)),
                "epochs": [],
                "source": "metadata_only",
            }
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data["source"] = "val_loss_history"
    return data


def _load_eval_metrics(run_dir: Path, dataset: str) -> Optional[Dict[str, Any]]:
    # results/ckpts/MM-DD-job-ds-cfg  <->  results/MM-DD-job-ds-cfg or results/.../partials
    stem = run_dir.name
    candidates = [
        RESULTS_ROOT / "datasets" / stem / "partials" / f"{dataset}_staged_anchor.json",
        RESULTS_ROOT / "datasets" / stem / dataset / "staged_results.json",
        RESULTS_ROOT / stem / "partials" / f"{dataset}_staged_anchor.json",
        RESULTS_ROOT / "partials" / f"{dataset}_staged_anchor.json",
        run_dir / "partials" / f"{dataset}_staged_anchor.json",
    ]
    # Also search sibling results dirs that share the job id prefix
    job_m = re.match(r"^(\d{2}-\d{2}-\d+)-", stem)
    if job_m:
        prefix = job_m.group(1)
        for root in (RESULTS_ROOT / "datasets", RESULTS_ROOT):
            if not root.is_dir():
                continue
            for p in root.iterdir():
                if p.is_dir() and p.name.startswith(prefix) and dataset in p.name:
                    candidates.append(p / "partials" / f"{dataset}_staged_anchor.json")
                    candidates.append(p / dataset / "staged_results.json")
    for c in candidates:
        if c.is_file():
            with open(c, encoding="utf-8") as f:
                return json.load(f)
    return None


def _rel_diff(a: float, b: float) -> float:
    if a == 0 or a != a or b != b:
        return float("nan")
    return (b - a) / abs(a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="electricity")
    ap.add_argument(
        "--configs",
        nargs="+",
        default=[
            "binary_noise_sched_ablation_elec_unc_g1p0",
            "binary_noise_sched_ablation_elec_unc_g1p5",
            "binary_noise_sched_ablation_elec_unc_g3p0",
        ],
    )
    ap.add_argument(
        "--out-dir",
        default="reports/noise_sched_ablation_elec_unc",
    )
    ap.add_argument("--stage", choices=["coarse", "fine", "both"], default="both")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stages = ["coarse", "fine"] if args.stage == "both" else [args.stage]
    rows: List[Dict[str, Any]] = []
    curves: Dict[str, Dict[str, List[Tuple[int, float]]]] = {
        s: {} for s in stages
    }

    for cfg in args.configs:
        label = VARIANT_LABELS.get(cfg, cfg)
        run_dir = _latest_run_dir(args.dataset, cfg)
        subset = _find_subset_dir(run_dir)
        eval_m = _load_eval_metrics(run_dir, args.dataset)
        row: Dict[str, Any] = {
            "variant": label,
            "config": cfg,
            "run_dir": str(run_dir),
            "subset": subset.name,
        }
        for stage in stages:
            hist = _load_history(subset / stage)
            if hist is None:
                row[f"{stage}_best_val"] = float("nan")
                row[f"{stage}_best_epoch"] = 0
                row[f"{stage}_final_val"] = float("nan")
                row[f"{stage}_n_epochs"] = 0
                continue
            epochs = hist.get("epochs") or []
            row[f"{stage}_best_val"] = float(hist.get("best_val", float("nan")))
            row[f"{stage}_best_epoch"] = int(hist.get("best_epoch", 0))
            row[f"{stage}_n_epochs"] = len(epochs)
            if epochs:
                row[f"{stage}_final_val"] = float(epochs[-1]["val_loss"])
                curves[stage][label] = [
                    (int(e["epoch"]), float(e["val_loss"])) for e in epochs
                ]
            else:
                row[f"{stage}_final_val"] = row[f"{stage}_best_val"]
            row[f"{stage}_length_mode"] = hist.get("length_mode")
            row[f"{stage}_length_g"] = hist.get("length_g")

        if eval_m:
            row["eval_crps"] = eval_m.get("crps")
            row["eval_anchor_mse"] = eval_m.get("anchor_mse")
            row["eval_prob_mse"] = eval_m.get("mse")
            row["eval_prob_mae"] = eval_m.get("mae")
        else:
            row["eval_crps"] = None
            row["eval_anchor_mse"] = None
            row["eval_prob_mse"] = None
            row["eval_prob_mae"] = None
            row["notes"] = "staged_eval metrics not found yet"
        rows.append(row)

    # Relative gaps vs baseline (first config)
    if rows:
        base = rows[0]
        for row in rows[1:]:
            notes = []
            for stage in stages:
                key = f"{stage}_final_val"
                rd = _rel_diff(float(base[key]), float(row[key]))
                row[f"{stage}_rel_vs_baseline"] = rd
                if rd == rd:
                    notes.append(f"{stage} final val {rd:+.1%} vs baseline")
            if row.get("eval_crps") is not None and base.get("eval_crps") is not None:
                rd = _rel_diff(float(base["eval_crps"]), float(row["eval_crps"]))
                row["eval_crps_rel_vs_baseline"] = rd
                if rd == rd:
                    notes.append(f"crps {rd:+.1%} vs baseline")
            row["notes"] = "; ".join(notes) if notes else row.get("notes", "")

    csv_path = out_dir / "comparison_table.csv"
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    md_path = out_dir / "comparison_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Noise schedule ablation — electricity 336/720_uncompressed\n\n")
        f.write(
            "Short coarse+fine retrain (4 epochs ≈ 20% of full 20-epoch budget), "
            "shared synthetic pretrain + patch guidance, seed/HP matched.\n\n"
        )
        f.write("| variant | coarse final val | fine final val | CRPS | anchor MSE | notes |\n")
        f.write("|---|---:|---:|---:|---:|---|\n")
        for r in rows:
            f.write(
                f"| {r['variant']} | "
                f"{r.get('coarse_final_val', float('nan')):.6f} | "
                f"{r.get('fine_final_val', float('nan')):.6f} | "
                f"{r.get('eval_crps') if r.get('eval_crps') is not None else 'n/a'} | "
                f"{r.get('eval_anchor_mse') if r.get('eval_anchor_mse') is not None else 'n/a'} | "
                f"{r.get('notes', '')} |\n"
            )
        f.write("\n## Interpretation guide\n\n")
        f.write(
            "- If fine final val / CRPS differ by **>5–10% relative**, schedule choice "
            "matters empirically → continue multi-knot schedule work.\n"
            "- If gaps are within typical seed noise, proxy MA-r gaps may not translate "
            "to trained quality → ship current g calibration / baseline.\n"
        )

    for stage in stages:
        if not curves[stage]:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, pts in curves[stage].items():
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker="o", label=label)
        ax.set_xlabel("epoch")
        ax.set_ylabel("val loss")
        ax.set_title(f"{args.dataset} {stage} — noise schedule ablation")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = out_dir / f"val_loss_{stage}.png"
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)

    summary_json = out_dir / "comparison_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "out_dir": str(out_dir)}, f, indent=2)

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {summary_json}")
    for stage in stages:
        p = out_dir / f"val_loss_{stage}.png"
        if p.is_file():
            print(f"Wrote {p}")


if __name__ == "__main__":
    main()
