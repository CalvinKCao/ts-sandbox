#!/usr/bin/env python3
"""H720 nostitch patch-refine redbox on 3 random test windows per ckpt.

Uses write_staged_sample_panels (same refine_boxes path as MMPD/gap redbox),
not the Combined-forecast 1d overlay. Random windows, seed-fixed.

Example (compute node, repo root):
  python -u temp/scripts/viz_h720_nostitch_redbox.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

JOBS = (
    {
        "job_id": "5160029",
        "dataset": "ETTh1",
        "ckpt": (
            "results/ckpts/09-01-5160029-ETTh1-binary_window_norm_patch_refine_"
            "canvas128_p64x6_allv_randwin_lr10_cap1x2x_hz720_nostitch_nopretrain"
        ),
        "config": (
            "configs/binary_window_norm_patch_refine_canvas128_p64x6_allv_"
            "randwin_lr10_cap1x2x_hz720_nostitch_nopretrain.yaml"
        ),
    },
    {
        "job_id": "5160030",
        "dataset": "exchange_rate",
        "ckpt": (
            "results/ckpts/09-01-5160030-exchange_rate-binary_window_norm_patch_refine_"
            "canvas128_p32x6_allv_randwin_lr10_cap1x2x_hz720_nostitch_nopretrain"
        ),
        "config": (
            "configs/binary_window_norm_patch_refine_canvas128_p32x6_allv_"
            "randwin_lr10_cap1x2x_hz720_nostitch_nopretrain.yaml"
        ),
    },
    {
        "job_id": "5160031",
        "dataset": "solar_Alabama",
        "ckpt": (
            "results/ckpts/09-01-5160031-solar_Alabama-binary_window_norm_patch_refine_"
            "canvas128_p32x6_allv_randwin_lr10_cap1x2x_hz720_nostitch_nopretrain"
        ),
        "config": (
            "configs/binary_window_norm_patch_refine_canvas128_p32x6_allv_"
            "randwin_lr10_cap1x2x_hz720_nostitch_nopretrain.yaml"
        ),
    },
)


def _load_patch_refine_models(
    *,
    dataset: str,
    ckpt_root: Path,
    config_path: str,
    lookback: int,
    horizon: int,
    device: torch.device,
) -> Tuple[Any, Any, Any, List[int]]:
    """Load coarse+refine via StagedEvalPhase._load_model (same as job eval)."""
    from models.diffusion_tsf.pipeline.phases.staged_eval import StagedEvalPhase
    from models.diffusion_tsf.train_multivariate_pipeline import (
        dataset_window_lengths,
        load_wrapped_guidance,
        resolve_pipeline_data_subset,
    )
    from utils.visualize_staged_eval_2d_preds import _build_state, _resolve_guidance_ckpt

    state = _build_state(ckpt_root, dataset, dataset, config_path)
    resolve_pipeline_data_subset(state)
    if state.use_ordinal_window_norm:
        raise RuntimeError(
            f"{dataset}: ordinal window-norm ckpt; nostitch redbox expects window-norm"
        )
    n_vars = list(state.variate_indices or [])
    if not n_vars:
        raise RuntimeError(f"{dataset}: empty variate_indices after subset resolve")
    ds_lb, ds_hz = dataset_window_lengths(state, dataset)
    if int(ds_lb) != int(lookback) or int(ds_hz) != int(horizon):
        raise RuntimeError(
            f"{dataset}: yaml windows lb={ds_lb} hz={ds_hz} != "
            f"requested lb={lookback} hz={horizon}"
        )
    n_iv = len(n_vars)
    guidance = None
    if state.needs_guidance:
        path, gtype = _resolve_guidance_ckpt(ckpt_root, str(state.subset_id), "auto")
        guidance = load_wrapped_guidance(
            state,
            str(path),
            n_iv,
            device,
            guidance_type=gtype,
            dataset_lookback=ds_lb,
            dataset_horizon=ds_hz,
        )
    phase = StagedEvalPhase()
    coarse = phase._load_model(state, "coarse", guidance, n_iv, device)
    refine = phase._load_model(state, "patch_refine", guidance, n_iv, device)
    for m in (coarse, refine):
        m._ordinal_input_is_ranked = False
        m._ordinal_apply_ood_shift = True
    return coarse, refine, state, n_vars


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--lookback", type=int, default=336)
    p.add_argument("--horizon", type=int, default=720)
    p.add_argument("--pack-test-stride", type=int, default=1)
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sampler", default="anchor")
    p.add_argument("--num-sampling-steps", type=int, default=1)
    p.add_argument(
        "--variables-to-plot",
        type=int,
        default=1,
        help="Variates per redbox panel. 1 → one refine_boxes jpg per window.",
    )
    p.add_argument("--jpeg-dpi", type=int, default=120)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--jobs",
        default="5160029,5160030,5160031",
        help="Comma-separated job ids to run.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    wanted = {s.strip() for s in str(args.jobs).split(",") if s.strip()}
    jobs = [j for j in JOBS if j["job_id"] in wanted]
    if not jobs:
        raise ValueError(f"no matching jobs in {wanted}")

    out_root = args.output_root
    if out_root is None:
        stamp = os.environ.get("SLURM_JOB_ID", "local")
        out_root = REPO_ROOT / "results" / "datasets" / f"h720-nostitch-redbox-{stamp}"
    out_root = out_root if out_root.is_absolute() else REPO_ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device} out_root={out_root}", flush=True)

    from utils.staged_eval_sample_viz import write_staged_sample_panels

    summary: List[Dict[str, Any]] = []
    for spec in jobs:
        ckpt = Path(spec["ckpt"])
        if not ckpt.is_absolute():
            ckpt = REPO_ROOT / ckpt
        cfg = Path(spec["config"])
        if not cfg.is_absolute():
            cfg = REPO_ROOT / cfg
        if not ckpt.is_dir():
            raise FileNotFoundError(f"missing ckpt: {ckpt}")
        if not cfg.is_file():
            raise FileNotFoundError(f"missing config: {cfg}")

        dataset = spec["dataset"]
        run_name = f"{spec['job_id']}-{dataset}"
        out_dir = out_root / run_name / "redbox"
        print(f"\n=== {run_name} ckpt={ckpt.name} ===", flush=True)

        coarse, refine, state, pool_vars = _load_patch_refine_models(
            dataset=dataset,
            ckpt_root=ckpt,
            config_path=str(cfg),
            lookback=args.lookback,
            horizon=args.horizon,
            device=device,
        )
        kind = "patch_refine"
        stitch = bool(getattr(coarse.config, "horizon_stitch", False))
        if stitch:
            raise RuntimeError(f"{run_name}: horizon_stitch=true; this util is nostitch-only")
        print(
            f"kind={kind} stitch={stitch} "
            f"dataset_H={getattr(coarse.config, 'dataset_forecast_length', None)} "
            f"canvas_W={getattr(coarse.config, 'forecast_length', None)}",
            flush=True,
        )
        from models.diffusion_tsf.train_multivariate_pipeline import load_dataset

        _, _, pool, _ = load_dataset(
            state,
            dataset,
            pool_vars,
            lookback=args.lookback,
            horizon=args.horizon,
            stride=1,
            test_stride=int(args.pack_test_stride),
            use_ordinal_window_norm=bool(state.use_ordinal_window_norm),
        )
        n_pool = len(pool)
        if n_pool < 1:
            raise RuntimeError(f"{run_name}: empty test pool")
        k = min(int(args.n_samples), n_pool)
        rng = np.random.default_rng(int(args.seed))
        picks = sorted(int(x) for x in rng.choice(n_pool, size=k, replace=False))
        print(f"n_pool={len(pool)} n_vars={len(pool_vars)} picks={picks}", flush=True)

        written = write_staged_sample_panels(
            out_dir=out_dir,
            run_name=run_name,
            dataset=dataset,
            kind=kind,
            coarse_model=coarse,
            fine_model=refine,
            pool=pool,
            picks=picks,
            device=device,
            sampler=str(args.sampler),
            num_sampling_steps=int(args.num_sampling_steps),
            seed=int(args.seed),
            variables_to_plot=int(args.variables_to_plot),
            jpeg_dpi=int(args.jpeg_dpi),
        )
        redbox_paths = [p for p in written if "refine_boxes" in p.name]
        if len(redbox_paths) != len(picks) * max(1, int(args.variables_to_plot)):
            raise RuntimeError(
                f"{run_name}: expected {len(picks)}×{args.variables_to_plot} "
                f"refine_boxes, got {len(redbox_paths)}"
            )
        row = {
            "job_id": spec["job_id"],
            "dataset": dataset,
            "ckpt": str(ckpt),
            "config": str(cfg),
            "n_pool": len(pool),
            "picks": picks,
            "sampler": args.sampler,
            "steps": args.num_sampling_steps,
            "redbox": [str(p) for p in redbox_paths],
        }
        summary.append(row)
        print(f"wrote {len(redbox_paths)} redbox panels -> {out_dir}", flush=True)
        for p in redbox_paths:
            print(p, flush=True)
        del coarse, refine, pool
        if device.type == "cuda":
            torch.cuda.empty_cache()

    manifest = out_root / "window_indices.json"
    manifest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\ndone: {manifest}", flush=True)


if __name__ == "__main__":
    main()
