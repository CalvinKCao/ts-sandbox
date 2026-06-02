#!/usr/bin/env python3
"""Batch forecast viz for 06-02 patch48 train grid (jobs 3844450–3844457).

One random test window per dataset: every variate on shared axes, GT vs
iTrans guidance (finetuned encoder) vs full-dataset iTrans baseline vs
deterministic anchor vs N probabilistic diffusion samples (default dpmpp×5),
plus random extra lookback-only panels.

Expects checkpoint layout:
  results/ckpts/06-02-{JOB}-{dataset}-binary_dual_scale_patch48/
    {subset_id}/best.pt
    {subset_id}_itransformer_finetuned.pt
    {subset_id}_itrans_full_dataset.pt

Example:
  python utils/visualize_report_patch48_train.py
  python utils/visualize_report_patch48_train.py --datasets exchange_rate,traffic --smoke-test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.visualize_binary_dual_scale_forecast import plot_forecast_panel

CKPT_SUFFIX = "binary_dual_scale_patch48"

PATCH48_DATASETS = [
    "ETTm1",
    "ETTm2",
    "dalia",
    "electricity",
    "exchange_rate",
    "solar_Alabama",
    "traffic",
    "weather",
]

JOB_DATASET_RE = re.compile(
    rf"06-02-384445[0-7]-(.+)-{re.escape(CKPT_SUFFIX)}$"
)


def pick_patch48_ckpt_dir(ckpt_root: Path, dataset: str) -> Path:
    """Newest 06-02-384445x-{dataset}-binary_dual_scale_patch48 with best.pt."""
    candidates: List[Path] = []
    for d in ckpt_root.iterdir():
        if not d.is_dir():
            continue
        m = JOB_DATASET_RE.match(d.name)
        if not m or m.group(1) != dataset:
            continue
        if any(d.glob("*/best.pt")):
            candidates.append(d)
    if not candidates:
        raise FileNotFoundError(
            f"No {CKPT_SUFFIX} ckpt with best.pt for {dataset} under {ckpt_root} "
            f"(expected 06-02-384445x-{dataset}-{CKPT_SUFFIX})"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def datasets_with_results(datasets_root: Path) -> List[str]:
    """Datasets that finished eval (results.json present)."""
    out: List[str] = []
    for run_dir in datasets_root.glob(f"06-02-384445*-{CKPT_SUFFIX}"):
        if not run_dir.is_dir():
            continue
        m2 = JOB_DATASET_RE.match(run_dir.name)
        ds_from_dir = m2.group(1) if m2 else None
        for results_path in run_dir.glob("*/results.json"):
            try:
                r = json.loads(results_path.read_text(encoding="utf-8"))
                out.append(str(r.get("dataset", ds_from_dir or "")))
            except (json.JSONDecodeError, OSError):
                if ds_from_dir:
                    out.append(ds_from_dir)
    return sorted({d for d in out if d})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt-root",
        type=Path,
        default=REPO_ROOT / "results" / "ckpts",
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=REPO_ROOT / "results" / "datasets",
        help="Used with --completed-only to skip datasets without results.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reports" / "3844450_binary_dual_scale_patch48_4x4_train",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(PATCH48_DATASETS),
    )
    parser.add_argument(
        "--completed-only",
        action="store_true",
        help="Only datasets with results/datasets/06-02-384445*/**/results.json",
    )
    parser.add_argument("--prob-samples", type=int, default=5)
    parser.add_argument("--num-extra-lookbacks", type=int, default=2)
    parser.add_argument("--anchor-sampler", type=str, default="anchor")
    parser.add_argument("--prob-sampler", type=str, default="dpmpp")
    parser.add_argument("--prob-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="exchange_rate only, 2 prob samples, 1 extra lookback",
    )
    args = parser.parse_args()

    if args.smoke_test:
        datasets = ["exchange_rate"]
        args.prob_samples = min(args.prob_samples, 2)
        args.num_extra_lookbacks = 1
    else:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    if args.completed_only:
        done = set(datasets_with_results(args.datasets_root.resolve()))
        datasets = [d for d in datasets if d in done]
        if not datasets:
            print("No completed patch48 results.json found.", flush=True)
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_base = args.output_dir.resolve()
    manifest: Dict[str, List[str]] = {}

    print(f"device={device}  out={out_base}  datasets={datasets}", flush=True)

    for dataset in datasets:
        try:
            ckpt_dir = pick_patch48_ckpt_dir(args.ckpt_root.resolve(), dataset)
        except FileNotFoundError as e:
            print(f"[skip] {dataset}: {e}", flush=True)
            continue

        print(f"[viz] {dataset} <- {ckpt_dir.name}", flush=True)
        out_path = plot_forecast_panel(
            ckpt_dir,
            dataset,
            out_base / "viz" / dataset,
            test_index=None,
            prob_samples=args.prob_samples,
            num_extra_lookbacks=args.num_extra_lookbacks,
            anchor_sampler=args.anchor_sampler,
            prob_sampler=args.prob_sampler,
            prob_steps=args.prob_steps,
            seed=args.seed + sum(ord(c) for c in dataset),
            device=device,
        )
        manifest[dataset] = [str(out_path.relative_to(out_base))]
        print(f"  wrote {out_path}", flush=True)

    manifest_path = out_base / "viz_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
