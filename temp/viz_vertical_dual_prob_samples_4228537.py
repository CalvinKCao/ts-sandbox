#!/usr/bin/env python3
"""Offline probabilistic (dpmpp) sample viz for the ep20 g7 control ckpt.

Default target:
  results/ckpts/07-13-4228537-electricity-..._g7p0_ep20_fulleval

Writes JPEGs under:
  <results-dir>/viz/eval_prob_samples_offline/

Example (Killarney / local with ckpts pulled):
  source .venv/bin/activate
  python temp/viz_vertical_dual_prob_samples_4228537.py
  python temp/viz_vertical_dual_prob_samples_4228537.py --n-windows 4 --n-samples 8 --steps 20
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_RUN = (
    "07-13-4228537-electricity-binary_noise_sched_ablation_vertical_dual_g7p0_ep20_fulleval"
)
DEFAULT_CONFIG = (
    "configs/binary_noise_sched_ablation_vertical_dual_g7p0_ep20_fulleval.yaml"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("viz_prob_4228537")


def _resolve_run_dirs(run: str) -> tuple[Path, Path]:
    ckpt = REPO / "results" / "ckpts" / run
    res = REPO / "results" / "datasets" / run
    if not ckpt.is_dir():
        raise FileNotFoundError(f"missing ckpt dir: {ckpt}")
    res.mkdir(parents=True, exist_ok=True)
    return ckpt, res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default=DEFAULT_RUN, help="results/ckpts/<run> stem")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--dataset", default="electricity")
    p.add_argument("--n-windows", type=int, default=4)
    p.add_argument("--n-samples", type=int, default=8, help="dpmpp draws per window")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--sampler", default="dpmpp", choices=("dpmpp", "ddim"))
    p.add_argument(
        "--windows",
        default="",
        help="comma-separated test window indices (default: worst CRPS from json if present)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    from models.diffusion_tsf.pipeline.config import load_experiment_config
    from models.diffusion_tsf.pipeline.globals_bridge import patch_globals
    from models.diffusion_tsf.pipeline.phases.staged_eval import StagedEvalPhase
    from models.diffusion_tsf.pipeline.state import PipelineState
    from models.diffusion_tsf.pipeline.visualize_utils import (
        plot_probabilistic_sample_panel,
        _prob_window_pred_2d_maps,
        pick_sample_indices,
    )
    from models.diffusion_tsf.train_multivariate_pipeline import (
        generate_dataset_job,
        load_dataset,
        load_wrapped_guidance,
        dataset_window_lengths,
    )
    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

    ckpt_dir, results_dir = _resolve_run_dirs(args.run)
    cfg = load_experiment_config(
        str(REPO / args.config),
        cli_overrides={
            "dataset": args.dataset,
            "checkpoint_dir": str(ckpt_dir),
            "results_dir": str(results_dir),
            "seed": args.seed,
        },
    )
    state = PipelineState.from_config(cfg)
    state.checkpoint_dir = str(ckpt_dir)
    state.results_dir = str(results_dir)
    state.dataset = args.dataset
    state.seed = args.seed

    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    patch_globals(pipeline_mod, state, honor_dataset_windows=True)

    job = generate_dataset_job(state.dataset)
    variate_indices = state.variate_indices or job["variate_indices"]
    state.variate_indices = list(variate_indices)
    subset_meta = state.data_subset_resolved or {}
    train_stride = int(subset_meta.get("train_stride", state.window_stride))
    test_stride = 4
    _, _, test_ds, norm_stats = load_dataset(
        state.dataset,
        variate_indices,
        stride=train_stride,
        test_stride=test_stride,
        ordinal_tie_atol=float(state.ordinal_tie_atol),
        use_ordinal_window_norm=state.use_ordinal_window_norm,
    )
    if norm_stats.get("ordinal_ladder") is not None:
        state.extra["global_ordinal_ladder"] = norm_stats["ordinal_ladder"]
        pipeline_mod.GLOBAL_ORDINAL_LADDER = norm_stats["ordinal_ladder"]

    # Resolve subset_id from ckpt layout first (e.g. electricity_4v_s1/...).
    best_pt = None
    if state.subset_id:
        cand = ckpt_dir / state.subset_id / "vertical_dual" / "best.pt"
        if cand.is_file():
            best_pt = cand
    if best_pt is None:
        candidates = sorted(ckpt_dir.glob("*/vertical_dual/best.pt"))
        if not candidates:
            raise FileNotFoundError(
                f"missing vertical_dual best.pt under {ckpt_dir}\n"
                "Pull ckpts for this run (omit --no-npy-ckpt) or scp from scratch."
            )
        best_pt = candidates[0]
        state.subset_id = best_pt.parent.parent.name
    state.diffusion_vertical_dual_finetune_ckpt = str(best_pt)

    guidance_ckpt = state.default_guidance_finetune_ckpt_path()
    if not os.path.isfile(guidance_ckpt):
        alts = [
            ckpt_dir / f"{state.subset_id}_patch_guidance.pt",
            ckpt_dir / f"{state.subset_id}_patch_guidance_hp_best.pt",
            *sorted(ckpt_dir.glob("*_patch_guidance*.pt")),
        ]
        for alt in alts:
            if alt.is_file():
                guidance_ckpt = str(alt)
                break
        else:
            raise FileNotFoundError(
                f"missing patch guidance ckpt near {ckpt_dir}\n"
                "Pull ckpts for this run (omit --no-npy-ckpt) or scp from scratch."
            )
    logger.info("guidance=%s", guidance_ckpt)
    logger.info("diffusion=%s", best_pt)

    ds_lb, ds_hz = dataset_window_lengths(state.dataset)
    guidance = load_wrapped_guidance(
        guidance_ckpt,
        len(variate_indices),
        device,
        guidance_type=state.guidance_type,
        dataset_lookback=ds_lb,
        dataset_horizon=ds_hz,
    )
    phase = StagedEvalPhase(phase="staged_eval")
    model = phase._load_model(state, "vertical_dual", guidance, len(variate_indices), device)

    # Window selection: CLI > worst_windows.json CRPS > random
    windows: list[int] = []
    if args.windows.strip():
        windows = [int(x) for x in args.windows.split(",") if x.strip()]
    else:
        worst_path = results_dir / (state.subset_id or "") / "worst_windows.json"
        if worst_path.is_file():
            manifest = json.loads(worst_path.read_text())
            windows = [
                int(e["window_index"])
                for e in manifest
                if e.get("metric") == "crps"
            ][: args.n_windows]
            logger.info("using %d CRPS-worst windows from %s", len(windows), worst_path)
    if not windows:
        windows = pick_sample_indices(len(test_ds), args.n_windows, seed=args.seed)
        logger.info("using random windows: %s", windows)

    out_dir = results_dir / "viz" / "eval_prob_samples_offline"
    out_dir.mkdir(parents=True, exist_ok=True)
    k_overlap = int(getattr(model.config, "lookback_overlap", 0) or 0)
    ordinal_mode = bool(getattr(state, "use_ordinal_window_norm", False))

    paths: list[str] = []
    for rank, wi in enumerate(windows, start=1):
        logger.info(
            "window %d/%d idx=%s: generating 1 map + %d %s samples (%d steps)",
            rank, len(windows), wi, args.n_samples, args.sampler, args.steps,
        )
        past, future = test_ds[int(wi)]
        if not torch.is_tensor(past):
            past = torch.as_tensor(past, dtype=torch.float32)
        if not torch.is_tensor(future):
            future = torch.as_tensor(future, dtype=torch.float32)

        t0 = time.perf_counter()
        coarse_2d, fine_2d = _prob_window_pred_2d_maps(
            model,
            model,
            past,
            device=device,
            sampler=args.sampler,
            num_inference_steps=args.steps,
        )
        logger.info("  map done in %.1fs", time.perf_counter() - t0)

        draws = []
        for s_i in range(args.n_samples):
            t1 = time.perf_counter()
            torch.manual_seed(args.seed + int(wi) * 1009 + s_i * 17)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed + int(wi) * 1009 + s_i * 17)
            with torch.no_grad():
                out = model.generate(
                    past.unsqueeze(0).to(device),
                    sampler=args.sampler,
                    num_inference_steps=args.steps,
                )
            pred = out.get("prediction_global_norm", out.get("prediction"))
            draws.append(pred[0].detach().cpu().numpy())
            logger.info(
                "  sample %d/%d done in %.1fs",
                s_i + 1, args.n_samples, time.perf_counter() - t1,
            )
        samples = np.stack(draws, axis=1)  # (V, S, T)
        sample_mean = samples.mean(axis=1)

        path = plot_probabilistic_sample_panel(
            past=past,
            future=future,
            samples_vt_s=samples,
            sample_mean=sample_mean,
            coarse_2d=coarse_2d,
            fine_2d=fine_2d,
            metric="offline",
            rank=rank,
            window_index=int(wi),
            score=float("nan"),
            output_dir=str(out_dir),
            ordinal_mode=ordinal_mode,
            lookback_overlap=k_overlap,
            sampler_label=args.sampler,
            max_spaghetti=args.n_samples,
        )
        paths.append(path)
        logger.info("wrote %s (window total %.1fs)", path, time.perf_counter() - t0)

    print("\n".join(paths))


if __name__ == "__main__":
    main()
