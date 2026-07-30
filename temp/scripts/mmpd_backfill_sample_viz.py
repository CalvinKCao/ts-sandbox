#!/usr/bin/env python3
"""Backfill anchor + probabilistic sample panels for already-completed MMPD eval runs.

Completed MMPD jobs (utils/eval_mmpd_gaussian_anchor.py --phase mmpd) already have raw
prediction arrays saved as <output-dir>/raw/mmpd_{dataset}.npz (y_true, deterministic,
samples, mode_center, mode_prob, indices), but no forecast/sample visualizations get
generated or wandb-logged for them -- only ordinal-input-space diagnostics (roundtrip /
coarse-fine 2D) are wired for `use_ordinal_window_norm` runs. This script renders a
handful of "anchor" (GT vs deterministic point forecast) and "probabilistic" (GT vs
sample fan) panels per dataset from those raw npz packs and logs them to wandb, reusing
utils/mmpd_sample_viz.py (the same helper the eval pipeline now calls automatically for
future runs, see run_phase_mmpd in utils/eval_mmpd_gaussian_anchor.py).

If a completed dataset already has an `mmpd_eval` leaderboard run (marker file at
<output-dir>/partials/.leaderboard_{dataset}.json), this resumes that exact run and
appends the images so they land next to the dataset's anchor_mse/anchor_mae/crps. If no
marker exists (leaderboard logging was off, or --local-only), panels are still written to
<output-dir>/viz/mmpd_samples/{dataset}/ and wandb logging is skipped or a standalone
backfill run is created for it.

Usage:
    python temp/scripts/mmpd_backfill_sample_viz.py \\
        --output-dir results/datasets/07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd \\
        --datasets electricity ETTh1 dynamic traffic \\
        --n-windows 4

    # Local-only (no wandb), custom window count:
    python temp/scripts/mmpd_backfill_sample_viz.py \\
        --output-dir results/datasets/07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd \\
        --datasets electricity --n-windows 6 --local-only

    # Force a fresh standalone wandb run instead of resuming the mmpd_eval marker run:
    python temp/scripts/mmpd_backfill_sample_viz.py \\
        --output-dir results/datasets/07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd \\
        --datasets traffic --new-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.load_dotenv import load_repo_dotenv
from utils.log_mmpd_eval_leaderboard import ENTITY, PROJECT, load_leaderboard_marker
from utils.mmpd_sample_viz import generate_mmpd_sample_visualizations


def load_pack(output_dir: Path, dataset: str) -> Optional[dict]:
    npz_path = output_dir / "raw" / f"mmpd_{dataset}.npz"
    if not npz_path.is_file():
        print(f"[{dataset}] skip: no raw pack at {npz_path}")
        return None
    with np.load(npz_path) as data:
        return {key: data[key] for key in data.files}


def log_to_wandb(
    dataset: str,
    paths: List[str],
    *,
    output_dir: Path,
    project: str,
    entity: str,
    new_run: bool,
    dry_run: bool,
) -> None:
    marker = None if new_run else load_leaderboard_marker(output_dir, dataset)
    run_id = marker.get("run_id") if marker else None

    if dry_run:
        target = f"resume run_id={run_id}" if run_id else "new standalone backfill run"
        print(f"[{dataset}] would log {len(paths)} image(s) to {project} ({target})")
        return

    import wandb

    if run_id:
        run = wandb.init(project=project, entity=entity, id=run_id, resume="must")
    else:
        print(f"[{dataset}] no mmpd_eval marker found; creating standalone backfill run")
        run = wandb.init(
            project=project,
            entity=entity,
            group=f"{dataset}-mmpd-viz-backfill",
            job_type="mmpd_viz_backfill",
            name=f"{dataset}-mmpd-viz-backfill",
            tags=[dataset, "mmpd", "viz-backfill"],
        )
    try:
        images = [wandb.Image(p, caption=Path(p).name) for p in sorted(paths)]
        wandb.log({"eval/mmpd_visualizations": images})
        print(f"[{dataset}] logged {len(paths)} image(s) -> {run.url}")
    finally:
        wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True, help="MMPD campaign output dir (has raw/, partials/).")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--n-windows", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--jpeg-dpi", type=int, default=100)
    parser.add_argument("--local-only", action="store_true", help="Only write jpgs, skip wandb entirely.")
    parser.add_argument("--new-run", action="store_true", help="Create a standalone run instead of resuming the mmpd_eval marker run.")
    parser.add_argument("--wandb-project", default=PROJECT)
    parser.add_argument("--wandb-entity", default=ENTITY)
    parser.add_argument("--dry-run", action="store_true", help="Generate jpgs but only print what would be wandb-logged.")
    args = parser.parse_args()

    if not args.local_only:
        load_repo_dotenv(REPO)

    output_dir = args.output_dir.resolve()
    for dataset in args.datasets:
        pack = load_pack(output_dir, dataset)
        if pack is None:
            continue
        out_dir = output_dir / "viz" / "mmpd_samples" / dataset
        paths = generate_mmpd_sample_visualizations(
            pack,
            dataset=dataset,
            out_dir=out_dir,
            n_windows=args.n_windows,
            seed=args.seed,
            jpeg_dpi=args.jpeg_dpi,
        )
        if not paths:
            print(f"[{dataset}] no panels generated (empty pack?)")
            continue
        print(f"[{dataset}] wrote {len(paths)} panel(s) to {out_dir}")
        if args.local_only:
            continue
        log_to_wandb(
            dataset,
            paths,
            output_dir=output_dir,
            project=args.wandb_project,
            entity=args.wandb_entity,
            new_run=args.new_run,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
