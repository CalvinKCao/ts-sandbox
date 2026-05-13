#!/usr/bin/env python3
"""
Build a markdown report for completed window-norm ablation Slurm runs (*wn-a|b|c*).

Scans ./results for directories matching *-wn-[abc]-* and treats a run as complete when
the main .log contains both 'PIPELINE COMPLETE' and 'Job completed:'.

Usage (from repo root):
  python utils/window_norm_ablate/generate_completed_report.py
  python utils/window_norm_ablate/generate_completed_report.py --results-dir /path/to/results
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

RUN_DIR_RE = re.compile(r"^(\d{2}-\d{2})-(\d+)-(wn-[abc])-(.+)$")
STARTED_RE = re.compile(r"^Started:\s+(\d{2})-(\d{2}) (\d{2}:\d{2}:\d{2})")
JOB_DONE_RE = re.compile(r"^Job completed:\s+(\d{2})-(\d{2}) (\d{2}:\d{2}:\d{2})")
TS_LOG_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
ITRANS_RE = re.compile(
    r"\[([^\]]+)\] iTransformer baseline: MSE=([\d\.]+), MAE=([\d\.]+)"
)
DIFF_RE = re.compile(r"\[([^\]]+)\] Avg: MSE=([\d\.]+), MAE=([\d\.]+)")
WANDB_URL_RE = re.compile(r"https://wandb\.ai/[^\s)]+/runs/[^\s)]+")


def parse_log(log_path: Path, year: int = 2026) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    start_time = None
    for line in lines[:250]:
        m = STARTED_RE.search(line)
        if m:
            mo, da, hms = m.group(1), m.group(2), m.group(3)
            start_time = datetime.strptime(
                f"{year}-{mo}-{da} {hms}", "%Y-%m-%d %H:%M:%S"
            )
            break
        m2 = TS_LOG_RE.match(line)
        if m2:
            start_time = datetime.strptime(m2.group(1), "%Y-%m-%d %H:%M:%S")
            break

    end_time = None
    for line in reversed(lines[-400:]):
        m = JOB_DONE_RE.search(line)
        if m:
            mo, da, hms = m.group(1), m.group(2), m.group(3)
            end_time = datetime.strptime(
                f"{year}-{mo}-{da} {hms}", "%Y-%m-%d %H:%M:%S"
            )
            break

    duration_str = "Unknown"
    if start_time and end_time and end_time >= start_time:
        sec = (end_time - start_time).total_seconds()
        h, r = divmod(int(sec), 3600)
        m, s = divmod(r, 60)
        duration_str = f"{h}h {m}m {s}s"

    complete = "PIPELINE COMPLETE" in text and "Job completed:" in text
    failed_early = (
        "ModuleNotFoundError: No module named 'torch'" in text
        or ("Traceback (most recent call last)" in text and "PIPELINE COMPLETE" not in text)
    )

    # Prefer last eval lines in the log
    metrics: dict[str, dict[str, str]] = {}
    for line in reversed(lines):
        dm = DIFF_RE.search(line)
        if dm:
            ds = dm.group(1)
            metrics.setdefault(ds, {})["diff_mse"] = dm.group(2)
            metrics.setdefault(ds, {})["diff_mae"] = dm.group(3)
        im = ITRANS_RE.search(line)
        if im:
            ds = im.group(1)
            metrics.setdefault(ds, {})["itrans_mse"] = im.group(2)
            metrics.setdefault(ds, {})["itrans_mae"] = im.group(3)

    wandb_urls = list(dict.fromkeys(WANDB_URL_RE.findall(text)))[-3:]

    incomplete_reason = ""
    if not complete:
        if failed_early:
            incomplete_reason = "failed early (env/import or crash before pipeline end)"
        elif "exited unexpectedly" in text:
            incomplete_reason = "dataloader/worker crash (often /dev/shm) or similar"
        else:
            incomplete_reason = "no completion marker (timeout or still running)"

    return {
        "complete": complete and not failed_early,
        "failed_early": failed_early,
        "incomplete_reason": incomplete_reason,
        "duration": duration_str,
        "metrics": metrics,
        "wandb_urls": wandb_urls,
    }


def improvement(it_mse: float, df_mse: float) -> str:
    if it_mse == 0:
        return "N/A"
    pct = (it_mse - df_mse) / it_mse * 100.0
    return f"{pct:+.2f}%"


def metrics_row(metrics: dict[str, dict[str, str]], ds_key: str) -> dict[str, str]:
    if ds_key in metrics:
        return metrics[ds_key]
    for alt in (ds_key.replace("-", "_"), ds_key.replace("_", "-")):
        if alt in metrics:
            return metrics[alt]
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("reports/window_norm_ablate_completed.md"),
    )
    ap.add_argument(
        "--min-job-id",
        type=int,
        default=None,
        help="If set, only include run dirs whose Slurm job id is >= this (e.g. 3562485 for one batch).",
    )
    args = ap.parse_args()

    results_dir: Path = args.results_dir
    if not results_dir.is_dir():
        raise SystemExit(f"Missing results dir: {results_dir}")

    runs: list[dict] = []
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue
        m = RUN_DIR_RE.match(child.name)
        if not m or "wn-" not in child.name:
            continue
        date_prefix, job_id, arm, dataset = m.group(1), m.group(2), m.group(3), m.group(4)
        job_id_int = int(job_id)
        if args.min_job_id is not None and job_id_int < args.min_job_id:
            continue
        logs_dir = child / "logs"
        if not logs_dir.is_dir():
            continue
        log_files = sorted(logs_dir.glob("*.log"))
        if not log_files:
            continue
        log_path = log_files[0]
        info = parse_log(log_path)
        runs.append(
            {
                "dir": child.name,
                "date_prefix": date_prefix,
                "job_id": job_id,
                "arm": arm,
                "dataset": dataset,
                "log": str(log_path),
                **info,
            }
        )

    completed = [r for r in runs if r["complete"]]
    incomplete = [r for r in runs if not r["complete"]]

    args.out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Window norm ablation — completed runs\n\n")
    lines.append(
        "Frozen hyperparameters (from prior default Slurm runs), **20% random test subset** "
        "for eval, full pipeline otherwise. Arms: **wn-a** — per-window norm off, guidance "
        "penalty 0; **wn-b** — window norm on, uniform guidance penalty 0.03; **wn-c** — window "
        "norm on, spatial ramped guidance penalty (max 0.2).\n\n"
    )
    if args.min_job_id is not None:
        lines.append(
            f"**Filter:** job directories with Slurm id ≥ `{args.min_job_id}` only.\n\n"
        )
    lines.append(f"**Results directory scanned:** `{results_dir.resolve()}`  \n\n")
    lines.append(
        f"**Completed ({len(completed)}):** "
        + ", ".join(r["dir"] for r in sorted(completed, key=lambda x: x["dir"]))
        + "\n\n"
    )
    if incomplete:
        lines.append("## Not complete (failed, timed out, or still running)\n")
        for r in sorted(incomplete, key=lambda x: x["dir"]):
            reason = r.get("incomplete_reason") or "unknown"
            lines.append(f"- `{r['dir']}` — {reason}\n")
        lines.append("\n")

    lines.append("## Per-run metrics (eval)\n")
    for r in sorted(completed, key=lambda x: (x["dataset"], x["arm"], x["job_id"])):
        ds_key = r["dataset"]
        met = metrics_row(r["metrics"], ds_key)
        ds_label = ds_key.replace("-", "_")
        it_m = met.get("itrans_mse", "N/A")
        it_a = met.get("itrans_mae", "N/A")
        df_m = met.get("diff_mse", "N/A")
        df_a = met.get("diff_mae", "N/A")
        imp = "N/A"
        if it_m != "N/A" and df_m != "N/A":
            imp = improvement(float(it_m), float(df_m))
        lines.append(f"### `{r['dir']}` *(Duration: {r['duration']})*\n")
        lines.append("\n| Dataset | Arm | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Δ MSE (iT vs diff) |\n")
        lines.append("|---------|-----|------------|------------|---------------|---------------|--------------------|\n")
        lines.append(
            f"| {ds_label} | {r['arm']} | {it_m} | {it_a} | {df_m} | {df_a} | {imp} |\n"
        )
        if r["wandb_urls"]:
            lines.append("\nW&B: " + " · ".join(r["wandb_urls"][-1:]) + "\n")
        lines.append("\n")

    # ETTh1 three-way if all present
    etth1 = [r for r in completed if r["dataset"] == "ETTh1"]
    if len(etth1) >= 2:
        lines.append("## ETTh1 — compare arms (same dataset)\n")
        lines.append("\n| Arm | Diffusion MSE | Diffusion MAE | iTrans MSE | iTrans MAE |\n")
        lines.append("|-----|-----------------|---------------|------------|------------|\n")
        for arm in ("wn-a", "wn-b", "wn-c"):
            row = next((x for x in etth1 if x["arm"] == arm), None)
            if not row:
                continue
            m = metrics_row(row["metrics"], "ETTh1")
            lines.append(
                f"| {arm} | {m.get('diff_mse', 'N/A')} | {m.get('diff_mae', 'N/A')} | "
                f"{m.get('itrans_mse', 'N/A')} | {m.get('itrans_mae', 'N/A')} |\n"
            )
        lines.append("\n")

    lines.append("## Qualitative plots\n\n")
    lines.append(
        "Regenerate anytime from repo root (GPU recommended), with checkpoints under each run’s "
        "`ckpts/`:\n\n"
        "```bash\n"
        "MIN_JOB_ID=3562485 ./utils/window_norm_ablate/visualize_completed.sh\n"
        "```\n\n"
        "Plots land under `results/viz/window_norm_ablate/<run-dir>/` (ignored by git via `**/results/`).\n\n"
    )
    lines.append("### Comparison PNGs (this workspace)\n\n")
    for r in sorted(completed, key=lambda x: x["dir"]):
        slug = r["dataset"].replace("-", "_")
        lines.append(
            f"- `{r['dir']}` → `results/viz/window_norm_ablate/{r['dir']}/comparison_{slug}.png`\n"
        )
    lines.append("\n")
    args.out.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {args.out} ({len(completed)} completed, {len(incomplete)} incomplete)")


if __name__ == "__main__":
    main()
