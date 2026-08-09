#!/usr/bin/env python3
"""Offline lookback / GT / pred panels for a synthetic diffusion pretrain ckpt.

Writes per-sample panels with:
  - GT coarse / pred coarse / GT fine / pred fine occupancy 2D maps
  - 1D lookback + GT future + diffusion pred
  - stacked vertical canvas (GT vs pred) for vertical_dual

Uses RealTS windows (same generators as Phase-1).

Example (Killarney / local with ckpts present):
  source .venv/bin/activate
  python temp/scripts/viz_synth_pretrain.py \\
    --ckpt results/ckpts/<run>/pretrained_vertical_dual/pretrained_diffusion.pt \\
    --config configs/binary_noise_sched_ablation_vertical_dual_g1p0.yaml \\
    --n-samples 6
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_CONFIG = "configs/binary_noise_sched_ablation_vertical_dual_g1p0.yaml"
GUIDANCE_NAMES = (
    "pretrained_patch_guidance.pt",
    "patch_guidance_synthetic.pt",
    "patch_guidance_synthetic_hp_best.pt",
)
PRETRAIN_REL = {
    "vertical_dual": "pretrained_vertical_dual/pretrained_diffusion.pt",
    "channel_dual": "pretrained_channel_dual/pretrained_diffusion.pt",
    "coarse": "pretrained_coarse/pretrained_diffusion.pt",
    "fine": "pretrained_fine/pretrained_diffusion.pt",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("viz_synth_pretrain")


def _auto_run_dir(dataset: str, config: str) -> Path:
    """Newest results/ckpts/*-<dataset>-<config_stem> that has the stage ckpt."""
    stem = Path(config).stem
    root = REPO / "results" / "ckpts"
    token = f"-{dataset}-{stem}"
    if not root.is_dir():
        raise FileNotFoundError(f"missing ckpt root: {root}")
    candidates = sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.endswith(token)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"no run dirs ending with {token!r} under {root}; pass --ckpt or --run-dir"
        )
    return candidates[0]


def _find_guidance(run_dir: Path, explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file():
            return p.resolve()
        logger.warning("--guidance missing (%s); searching under %s", p, run_dir)
    for name in GUIDANCE_NAMES:
        cand = run_dir / name
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"no guidance under {run_dir} (tried {GUIDANCE_NAMES}); pass --guidance"
    )


def _resolve_ckpt(args: argparse.Namespace) -> tuple[Path, Path]:
    """Return (diffusion_ckpt, run_dir)."""
    rel = PRETRAIN_REL.get(args.stage)
    if rel is None:
        raise ValueError(f"unknown --stage {args.stage!r}")

    if args.ckpt:
        ckpt = Path(args.ckpt).expanduser()
        if ckpt.is_dir():
            run_dir = ckpt.resolve()
            ckpt = run_dir / rel
        else:
            ckpt = ckpt.resolve() if ckpt.exists() else ckpt
            run_dir = (
                ckpt.parent.parent
                if ckpt.parent.name.startswith("pretrained_")
                else ckpt.parent
            )
        if not ckpt.is_file():
            # Common miss: reused/ store never populated — fall back to results/ckpts.
            logger.warning(
                "missing diffusion ckpt %s; falling back to newest results/ckpts match",
                ckpt,
            )
            run_dir = _auto_run_dir(args.dataset, args.config)
            ckpt = run_dir / rel
        if not ckpt.is_file():
            raise FileNotFoundError(f"missing diffusion ckpt: {ckpt}")
        return ckpt.resolve(), run_dir.resolve()

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        ckpt = run_dir / rel
        if not ckpt.is_file():
            raise FileNotFoundError(f"missing {ckpt}")
        return ckpt, run_dir

    run_dir = _auto_run_dir(args.dataset, args.config)
    ckpt = run_dir / rel
    if not ckpt.is_file():
        raise FileNotFoundError(f"newest run {run_dir.name} missing {rel}")
    logger.info("auto-picked run_dir=%s", run_dir)
    return ckpt, run_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--dataset", default="ETTh1")
    p.add_argument("--ckpt", default="", help="pretrained_diffusion.pt or run dir")
    p.add_argument("--run-dir", default="", help="results/ckpts/<run> containing pretrained_*")
    p.add_argument("--guidance", default="", help="synthetic patch guidance .pt")
    p.add_argument(
        "--stage",
        default="vertical_dual",
        choices=tuple(PRETRAIN_REL),
    )
    p.add_argument("--n-samples", type=int, default=6)
    p.add_argument("--n-vars-plot", type=int, default=3)
    p.add_argument("--sampler", default="anchor", choices=("anchor", "dpmpp", "ddim"))
    p.add_argument("--steps", type=int, default=20, help="inference steps (ignored for anchor)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        default="",
        help="JPEG output dir (default: <results-dir>/viz/synth_pretrain_offline)",
    )
    p.add_argument("--device", default=None)
    args = p.parse_args()

    from models.diffusion_tsf.pipeline.config import load_experiment_config
    from models.diffusion_tsf.pipeline.globals_bridge import patch_globals
    from models.diffusion_tsf.pipeline.state import PipelineState
    from models.diffusion_tsf.pipeline.visualize_utils import (
        run_dual_concat_synthetic_pretrain_visualizations,
        run_pretrain_diffusion_visualizations,
    )
    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

    ckpt, run_dir = _resolve_ckpt(args)
    guidance = _find_guidance(run_dir, args.guidance or None)
    out_root = Path(args.out_dir).expanduser() if args.out_dir else (
        REPO / "results" / "viz" / "synth_pretrain_offline" / run_dir.name
    )
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = load_experiment_config(
        str(REPO / args.config),
        cli_overrides={
            "dataset": args.dataset,
            "checkpoint_dir": str(run_dir),
            "results_dir": str(out_root),
            "seed": args.seed,
        },
    )
    state = PipelineState.from_config(cfg)
    state.checkpoint_dir = str(run_dir)
    state.results_dir = str(out_root)
    if args.device:
        state.device = args.device
    state.merged_config = {
        **(state.merged_config or {}),
        "visualization": {
            **((state.merged_config or {}).get("visualization") or {}),
            "enabled": True,
            "n_samples": int(args.n_samples),
            "n_dual_scale_vars": int(args.n_vars_plot),
            "dual_scale_sampler": str(args.sampler),
            "dual_scale_inference_steps": int(args.steps),
            "jpeg_dpi": 100,
        },
    }

    patch_globals(pipeline_mod, state, honor_dataset_windows=True)
    logger.info("ckpt=%s", ckpt)
    logger.info("guidance=%s", guidance)
    logger.info("out=%s sampler=%s n=%d", out_root, args.sampler, args.n_samples)

    is_dual = bool(state.use_vertical_dual_concat) or bool(
        getattr(state, "use_channel_dual_concat", False)
    )
    if is_dual or args.stage in {"vertical_dual", "channel_dual"}:
        if args.stage == "channel_dual":
            state.use_channel_dual_concat = True
            state.use_vertical_dual_concat = False
            state.diffusion_channel_dual_pretrain_ckpt = str(ckpt)
        else:
            state.use_vertical_dual_concat = True
            state.diffusion_vertical_dual_pretrain_ckpt = str(ckpt)
        paths = run_dual_concat_synthetic_pretrain_visualizations(
            state,
            dual_ckpt_path=str(ckpt),
            guidance_ckpt_path=str(guidance),
            tuned_params=None,
            tag="synth_pretrain_offline",
        )
    else:
        if args.stage == "coarse":
            state.diffusion_coarse_pretrain_ckpt = str(ckpt)
        elif args.stage == "fine":
            state.diffusion_fine_pretrain_ckpt = str(ckpt)
        paths = run_pretrain_diffusion_visualizations(
            state,
            coarse_ckpt_path=state.diffusion_coarse_pretrain_ckpt,
            fine_ckpt_path=state.diffusion_fine_pretrain_ckpt or str(ckpt),
            itrans_ckpt_path=str(guidance),
            tuned_params=None,
            tag="synth_pretrain_offline",
        )

    if not paths:
        raise SystemExit("no panels written (viz disabled or empty dataset?)")
    for path in paths:
        logger.info("wrote %s", path)
    logger.info("done (%d panels) → %s", len(paths), out_root)


if __name__ == "__main__":
    main()
