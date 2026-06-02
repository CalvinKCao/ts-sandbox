#!/usr/bin/env python3
"""Forecast panels for the combined CFG ablation + MMPD matrix report.

For each dataset, one random test window: every variate on shared axes with
ground truth, finetuned iTransformer guidance, full-dataset iTrans baseline,
deterministic diffusion anchor, and N probabilistic dpmpp samples (default 5).
Also plots a few extra lookback-only windows for context.

Output defaults to the report folder (same basename as the .md file):
  reports/06-01_cfg_ablation_mmpd_matrix_combined/viz_cfg_off/{dataset}/...
  reports/06-01_cfg_ablation_mmpd_matrix_combined/viz_4x4_patch/{dataset}/...
  reports/06-01_cfg_ablation_mmpd_matrix_combined/viz_2stage/{dataset}/...

Example:
  python utils/visualize_report_cfg_ablation_combined.py --smoke-test
  python utils/visualize_report_cfg_ablation_combined.py \\
    --variants cfg_off,4x4_patch,2stage
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.visualize_binary_dual_scale_forecast import plot_forecast_panel
from utils.visualize_report_binary_dual_scale import (
    REPORT_DATASETS,
    pick_ckpt_dir,
)
from utils.visualize_staged_forecast import pick_staged_ckpt_dir, plot_staged_forecast_panel

REPORT_STEM = "06-01_cfg_ablation_mmpd_matrix_combined"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / REPORT_STEM

PATCH48_CKPT_SUFFIX = "binary_dual_scale_patch48"
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
PATCH48_JOB_RE = re.compile(
    rf"06-02-384445[0-7]-(.+)-{re.escape(PATCH48_CKPT_SUFFIX)}$"
)


def pick_patch48_ckpt_dir(ckpt_root: Path, dataset: str) -> Path:
    candidates: List[Path] = []
    for d in ckpt_root.iterdir():
        if not d.is_dir():
            continue
        m = PATCH48_JOB_RE.match(d.name)
        if not m or m.group(1) != dataset:
            continue
        if any(d.glob("*/best.pt")):
            candidates.append(d)
    if not candidates:
        raise FileNotFoundError(
            f"No {PATCH48_CKPT_SUFFIX} ckpt for {dataset} under {ckpt_root}"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


VARIANTS: Dict[str, Dict[str, object]] = {
    "cfg_off": {
        "label": "Binary (CFG off)",
        "subdir": "viz_cfg_off",
        "pick_ckpt": pick_ckpt_dir,
        "datasets": REPORT_DATASETS,
        "slurm_note": "3828089 weights; eval jobs 3848045–3848047 for ETTh1/ETTh2/PeMS",
        "plot_fn": "standard",
    },
    "4x4_patch": {
        "label": "4x4 patch",
        "subdir": "viz_4x4_patch",
        "pick_ckpt": pick_patch48_ckpt_dir,
        "datasets": PATCH48_DATASETS,
        "slurm_note": "train 3844450–3844457; eval 3848019–3848026",
        "plot_fn": "standard",
    },
    "2stage": {
        "label": "2-stage",
        "subdir": "viz_2stage",
        "pick_ckpt": pick_staged_ckpt_dir,
        "datasets": [
            "ETTh1",
            "ETTh2",
            "PeMS",
            "dalia",
            "exchange_rate",
            "traffic",
        ],
        "slurm_note": "grid train/eval 3849018–3849023",
        "plot_fn": "staged",
    },
}


def parse_variants(spec: str) -> List[str]:
    keys = [v.strip() for v in spec.split(",") if v.strip()]
    unknown = [k for k in keys if k not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants {unknown}; choose from {list(VARIANTS)}")
    return keys


def run_variant(
    variant_key: str,
    ckpt_root: Path,
    output_base: Path,
    datasets: List[str],
    prob_samples: int,
    num_extra_lookbacks: int,
    prob_steps: int,
    seed: int,
    device: torch.device,
) -> Dict[str, List[str]]:
    meta = VARIANTS[variant_key]
    pick_ckpt: Callable[[Path, str], Path] = meta["pick_ckpt"]  # type: ignore[assignment]
    viz_root = output_base / str(meta["subdir"])
    manifest: Dict[str, List[str]] = {}

    print(
        f"[{variant_key}] {meta['label']} -> {viz_root}  "
        f"datasets={datasets}",
        flush=True,
    )

    for dataset in datasets:
        try:
            ckpt_dir = pick_ckpt(ckpt_root, dataset)
        except FileNotFoundError as e:
            print(f"  [skip] {dataset}: {e}", flush=True)
            continue

        ds_seed = seed + sum((i + 1) * ord(c) for i, c in enumerate(dataset))
        plot_kwargs = dict(
            checkpoint_dir=ckpt_dir,
            dataset=dataset,
            output_dir=viz_root / dataset,
            test_index=None,
            prob_samples=prob_samples,
            num_extra_lookbacks=num_extra_lookbacks,
            prob_steps=prob_steps,
            seed=ds_seed,
            device=device,
        )
        if meta.get("plot_fn") == "staged":
            out_path = plot_staged_forecast_panel(
                prob_sampler="dpmpp",
                **plot_kwargs,
            )
        else:
            out_path = plot_forecast_panel(
                anchor_sampler="anchor",
                prob_sampler="dpmpp",
                **plot_kwargs,
            )
        rel = str(out_path.relative_to(output_base))
        manifest[dataset] = [rel]
        print(f"  wrote {rel}", flush=True)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt-root",
        type=Path,
        default=REPO_ROOT / "results" / "ckpts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Report folder (default: reports/{REPORT_STEM})",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="cfg_off",
        help="Comma-separated: cfg_off, 4x4_patch, 2stage",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Override dataset list (comma-separated); default depends on variant",
    )
    parser.add_argument("--prob-samples", type=int, default=5)
    parser.add_argument("--num-extra-lookbacks", type=int, default=2)
    parser.add_argument("--prob-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="ETTh1 cfg_off only, 2 prob samples, 1 extra lookback",
    )
    args = parser.parse_args()

    if args.smoke_test:
        variant_keys = ["cfg_off"]
        datasets_override = ["ETTh1"]
        args.prob_samples = min(args.prob_samples, 2)
        args.num_extra_lookbacks = 1
    else:
        variant_keys = parse_variants(args.variants)
        datasets_override = (
            [d.strip() for d in args.datasets.split(",") if d.strip()]
            if args.datasets
            else None
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_base = args.output_dir.resolve()
    output_base.mkdir(parents=True, exist_ok=True)

    full_manifest: Dict[str, Dict[str, List[str]]] = {}

    for variant_key in variant_keys:
        meta = VARIANTS[variant_key]
        ds_list = (
            datasets_override
            if datasets_override is not None
            else list(meta["datasets"])  # type: ignore[arg-type]
        )
        full_manifest[variant_key] = run_variant(
            variant_key,
            args.ckpt_root.resolve(),
            output_base,
            ds_list,
            args.prob_samples,
            args.num_extra_lookbacks,
            args.prob_steps,
            args.seed,
            device,
        )

    manifest_path = output_base / "viz_manifest.json"
    manifest_path.write_text(json.dumps(full_manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
