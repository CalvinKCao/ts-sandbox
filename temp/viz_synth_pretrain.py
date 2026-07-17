#!/usr/bin/env python3
"""Offline lookback / GT / pred panels for a synthetic diffusion pretrain ckpt.

Uses RealTS windows (same generators as Phase-1) + anchor (default) or dpmpp
sampling so you can eyeball whether the reused g1 synth pretrain still looks
coherent before / after finetune.

Example (Killarney / local with ckpts present):
  source .venv/bin/activate
  python temp/viz_synth_pretrain.py \\
    --ckpt results/ckpts/<run>/pretrained_vertical_dual/pretrained_diffusion.pt \\
    --config configs/binary_noise_sched_ablation_vertical_dual_g1p0.yaml \\
    --n-samples 4

  # Or pass a run dir / config stem and let the script discover paths:
  python temp/viz_synth_pretrain.py \\
    --run-dir results/ckpts/07-15-4243853-ETTh2-binary_noise_sched_ablation_vertical_dual_g1p0 \\
    --config configs/binary_noise_sched_ablation_vertical_dual_g1p0.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_CONFIG = "configs/binary_noise_sched_ablation_vertical_dual_g1p0.yaml"
GUIDANCE_NAMES = ("pretrained_patch_guidance.pt", "patch_guidance_synthetic.pt")
PRETRAIN_REL = {
    "vertical_dual": "pretrained_vertical_dual/pretrained_diffusion.pt",
    "channel_dual": "pretrained_channel_dual/pretrained_diffusion.pt",
    "coarse": "pretrained_coarse/pretrained_diffusion.pt",
    "fine": "pretrained_fine/pretrained_diffusion.pt",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("viz_synth_pretrain")


def _find_guidance(run_dir: Path, explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"--guidance not a file: {p}")
        return p
    for name in GUIDANCE_NAMES:
        cand = run_dir / name
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"no guidance under {run_dir} (tried {GUIDANCE_NAMES}); pass --guidance"
    )


def _resolve_ckpt(args: argparse.Namespace) -> tuple[Path, Path]:
    """Return (diffusion_ckpt, run_dir)."""
    if args.ckpt:
        ckpt = Path(args.ckpt).expanduser().resolve()
        if ckpt.is_dir():
            # treat as run dir
            run_dir = ckpt
            rel = PRETRAIN_REL.get(args.stage)
            if rel is None:
                raise ValueError(f"unknown --stage {args.stage!r}")
            ckpt = run_dir / rel
        else:
            # .../pretrained_<stage>/pretrained_diffusion.pt → run dir two up
            run_dir = ckpt.parent.parent if ckpt.parent.name.startswith("pretrained_") else ckpt.parent
        if not ckpt.is_file():
            raise FileNotFoundError(f"missing diffusion ckpt: {ckpt}")
        return ckpt, run_dir

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        rel = PRETRAIN_REL.get(args.stage)
        if rel is None:
            raise ValueError(f"unknown --stage {args.stage!r}")
        ckpt = run_dir / rel
        if not ckpt.is_file():
            raise FileNotFoundError(f"missing {ckpt}")
        return ckpt, run_dir

    # Auto: newest *-<dataset>-<config_stem> under results/ckpts
    stem = Path(args.config).stem
    root = REPO / "results" / "ckpts"
    token = f"-{args.dataset}-{stem}"
    candidates = sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.endswith(token)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"no run dirs ending with {token!r} under {root}; pass --ckpt or --run-dir"
        )
    run_dir = candidates[0]
    rel = PRETRAIN_REL[args.stage]
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
    p.add_argument("--n-samples", type=int, default=4)
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
