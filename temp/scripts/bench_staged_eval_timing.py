#!/usr/bin/env python3
"""Time the staged eval generate path (canvas128 unique-seg patch refine).

Random weights, real geometry. Hits the same coarse → patch_refine generate
chain as staged_eval (anchor det + parallel MC quad_t). Viz is not run.

  source .venv/bin/activate
  python temp/scripts/bench_staged_eval_timing.py
  python temp/scripts/bench_staged_eval_timing.py --steps 4 --n-samp 2 --windows 1
  python temp/scripts/bench_staged_eval_timing.py --steps 20 --n-samp 2
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch

from models.diffusion_tsf.pipeline.config import apply_cli_state_overrides, load_experiment_config
from models.diffusion_tsf.pipeline.eval_bench import (
    configure as configure_eval_bench,
    dump as dump_eval_bench,
    reset as reset_eval_bench,
    snapshot,
    span as eval_bench_span,
)
from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import stage_state
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.train_multivariate_pipeline import (
    create_diffusion_model,
    create_patch_guidance_stack,
    dataset_window_lengths,
    wrap_patch_guidance,
)

logger = logging.getLogger("bench_staged_eval")

DEFAULT_CONFIG = "configs/binary_window_norm_patch_refine_canvas128_p64x6.yaml"


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_models(state: PipelineState, device: torch.device):
    state.torch_compile = False
    state.smoke_test = True
    n_var = int(state.n_variates)
    lb, hz = dataset_window_lengths(state, state.dataset)
    guidance = wrap_patch_guidance(
        state,
        create_patch_guidance_stack(state, n_var, in_len=lb, out_len=hz),
    ).to(device)
    guidance.eval()

    models = {}
    for stage in ("coarse", "patch_refine"):
        ms = stage_state(state, stage, honor_dataset_windows=True)
        ms.torch_compile = False
        ms.smoke_test = True
        model = create_diffusion_model(
            ms,
            n_variates=n_var,
            lookback=lb,
            horizon=hz,
            guidance_model=guidance,
            diffusion_stage=stage,
        ).to(device)
        model.eval()
        models[stage] = model
    return models["coarse"], models["patch_refine"], lb, hz, n_var


def _random_past(batch: int, n_var: int, lookback: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    return torch.randn(batch, n_var, lookback, generator=g).to(device)


@torch.no_grad()
def _staged_generate(coarse, fine, past, **gen_kwargs):
    with eval_bench_span("staged_generate"):
        coarse_out = coarse.generate(past, **gen_kwargs)
        fine_out = fine.generate(
            past,
            future_coarse_2d=coarse_out["future_2d_coarse"],
            **gen_kwargs,
        )
    return coarse_out, fine_out


def _run_case(name: str, coarse, fine, past, gen_kwargs, warmup: bool = False) -> dict:
    if warmup:
        logger.info("warmup %s ...", name)
        _staged_generate(coarse, fine, past, **gen_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return {}
    reset_eval_bench()
    t0 = time.perf_counter()
    coarse_out, fine_out = _staged_generate(coarse, fine, past, **gen_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    pred = fine_out["prediction_global_norm"]
    logger.info(
        "%s done wall=%.3fs pred=%s",
        name,
        wall,
        tuple(pred.shape),
    )
    text = dump_eval_bench(logger, title=name)
    snap = snapshot()
    snap["wall"] = wall
    snap["report"] = text
    return snap


def main() -> int:
    p = argparse.ArgumentParser(description="Staged eval generate-path timing")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--windows", type=int, default=1, help="Batch of windows (B)")
    p.add_argument("--n-samp", type=int, default=2, help="Parallel MC samples")
    p.add_argument("--steps", type=int, default=4, help="quad_t denoise steps for the short pass")
    p.add_argument("--full-steps", type=int, default=20, help="Realistic quad_t steps if the short pass is fast")
    p.add_argument("--skip-full", action="store_true", help="Skip the 20-step pass")
    p.add_argument("--n-variates", type=int, default=None)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    os.environ.setdefault("TS_EVAL_BENCH", "1")
    configure_eval_bench(True)

    cfg = load_experiment_config(os.path.abspath(args.config))
    state = PipelineState.from_config(cfg)
    apply_cli_state_overrides(state, cfg)
    if args.n_variates is not None:
        state.n_variates = int(args.n_variates)
    state.torch_compile = False
    state.smoke_test = True

    device = _device()
    logger.info(
        "device=%s dataset=%s V=%s unique_seg=%s canvas=%s patch=%sx%s stride=%s",
        device,
        state.dataset,
        state.n_variates,
        state.patch_refine_unique_segments,
        state.patch_refine_canvas_height,
        state.patch_refine_patch_height,
        state.patch_refine_patch_width,
        state.patch_refine_col_stride,
    )
    if device.type != "cuda":
        logger.warning("no GPU; timing the CPU generate path")

    coarse, fine, lb, hz, n_var = _build_models(state, device)
    W_fut = int(coarse.config.forecast_length)
    n_col0 = len(range(0, max(0, W_fut - int(state.patch_refine_patch_width)) + 1, int(state.patch_refine_col_stride)))
    logger.info(
        "lookback=%d horizon=%d model_W=%d n_ar_col0~%d unet_chunk=%s",
        lb, hz, W_fut, n_col0, coarse.config.unet_max_chunk_size,
    )

    logger.info("timing sklearn GMM (eval-path mode clustering)")
    reset_eval_bench()
    t_gmm = time.perf_counter()
    with eval_bench_span("sklearn_gmm"):
        from models.diffusion_tsf.metrics import empirical_modes_from_samples
        dummy = torch.randn(5, n_var, 10, hz).numpy()
        empirical_modes_from_samples(dummy, max_components=10, seed=0)
    logger.info(
        "sklearn GMM wall=%.3fs (B=5 V=%d n_samp=10 hz=%d)",
        time.perf_counter() - t_gmm, n_var, hz,
    )
    dump_eval_bench(logger, title="sklearn_gmm_killarney_shape")

    past = _random_past(args.windows, n_var, lb, device)
    det_kwargs = {"sampler": "anchor"}
    short_kwargs = {"sampler": "quad_t", "num_inference_steps": int(args.steps)}

    _run_case("warmup_det", coarse, fine, past, det_kwargs, warmup=True)
    det_snap = _run_case("det_anchor B=%d" % args.windows, coarse, fine, past, det_kwargs)

    past_exp = past.repeat_interleave(int(args.n_samp), dim=0)
    _run_case("warmup_prob", coarse, fine, past_exp, short_kwargs, warmup=True)
    short_name = "prob_quad_t steps=%d n_samp=%d B=%d parallel" % (
        args.steps, args.n_samp, args.windows,
    )
    short_snap = _run_case(short_name, coarse, fine, past_exp, short_kwargs)

    full_snap = None
    if not args.skip_full and short_snap.get("wall", 1e9) < 90.0:
        full_kwargs = {"sampler": "quad_t", "num_inference_steps": int(args.full_steps)}
        full_name = "prob_quad_t steps=%d n_samp=%d B=%d parallel" % (
            args.full_steps, args.n_samp, args.windows,
        )
        full_snap = _run_case(full_name, coarse, fine, past_exp, full_kwargs)
    elif not args.skip_full:
        logger.warning(
            "skipping %d-step pass; short pass took %.1fs",
            args.full_steps,
            short_snap.get("wall", -1),
        )

    print("\n======== summary ========")
    rows = [("det_anchor", det_snap)]
    rows.append((short_name, short_snap))
    if full_snap is not None:
        rows.append(("prob_full", full_snap))
    for label, snap in rows:
        wall = snap.get("wall", 0.0)
        notes = snap.get("notes") or {}
        print(f"\n{label}  wall={wall:.3f}s  notes={notes}")
        repeats = snap.get("repeats") or {}
        for k, st in repeats.items():
            pct = 100.0 * st["sum"] / wall if wall > 0 else 0.0
            print(
                f"  {k:22s} n={st['n']:<5d} mean={st['mean']*1000:7.1f}ms "
                f"p50={st['p50']*1000:7.1f}ms p95={st['p95']*1000:7.1f}ms "
                f"sum={st['sum']:7.3f}s ({pct:5.1f}%)"
            )
    print(
        "\nKillarney context: unique-seg AR is sequential over ~%d col0s; "
        "each col0 runs all denoise steps. Prob expands B by n_samp in one generate."
        % n_col0
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
