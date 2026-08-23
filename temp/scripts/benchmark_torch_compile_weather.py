#!/usr/bin/env python3
"""Compare eager and torch.compile on real guided Weather diffusion updates."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from models.diffusion_tsf import train_multivariate_pipeline as pipeline_mod
from models.diffusion_tsf.pipeline.config import load_experiment_config
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline.train.checkpointing import amp_context


def sync() -> None:
    torch.cuda.synchronize()


def phase_params(checkpoint_root: Path, checkpoint_run: str, stage: str) -> dict:
    """Use the actual micro-batch/accumulation plan from the completed run."""
    with (checkpoint_root / checkpoint_run / stage / "metadata.json").open() as handle:
        return dict(json.load(handle)["tuned_params"])


def run_update(model, loss_fn, optimizer, past, future, accum_steps: int) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss_value = 0.0
    for _ in range(accum_steps):
        with amp_context(bool(model.config.use_amp)):
            loss = loss_fn(past, future) / accum_steps
        loss.backward()
        loss_value += float(loss.detach())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss_value


def build_model(args, state, config, stage: str, device: torch.device):
    n_variates = len(state.variate_indices)
    lookback, horizon = pipeline_mod.dataset_window_lengths(state, state.dataset)
    root = Path(args.checkpoint_root)
    guidance = pipeline_mod.load_wrapped_guidance(
        state,
        str(root / f"{args.checkpoint_run}_patch_guidance.pt"),
        n_variates,
        device,
        guidance_type=state.guidance_type,
        dataset_lookback=lookback,
        dataset_horizon=horizon,
    )
    params = phase_params(root, args.checkpoint_run, stage)
    if stage == "patch_refine":
        # The campaign config keeps the coarse canvas height as its global
        # image_height. Patch refinement has a separately trained 64-row DiT.
        state.image_height = state.patch_refine_patch_height
    model = pipeline_mod.create_diffusion_model(
        state,
        n_variates=n_variates,
        lookback=lookback,
        horizon=horizon,
        guidance_model=guidance,
        diffusion_stage=stage,
        **params,
    ).to(device)
    ckpt = torch.load(
        root / args.checkpoint_run / stage / "best.pt",
        map_location=device,
        weights_only=False,
    )
    pipeline_mod.load_diffusion_state_keep_attached_guidance(model, ckpt["model_state_dict"])
    model.train()
    return model, int(params["batch_size"]), int(params["gradient_accumulation_steps"]), lookback, horizon


def benchmark(args, state, config, stage: str, compiled: bool) -> dict:
    device = torch.device("cuda")
    model, batch_size, accum_steps, lookback, horizon = build_model(args, state, config, stage, device)
    past = torch.randn(batch_size, len(state.variate_indices), lookback, device=device)
    future = torch.randn(
        batch_size,
        len(state.variate_indices),
        horizon + state.lookback_overlap,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = model.get_loss
    compile_seconds = 0.0
    if compiled:
        loss_fn = torch.compile(loss_fn, backend="inductor", fullgraph=False)
        sync()
        started = time.perf_counter()
        run_update(model, loss_fn, optimizer, past, future, accum_steps)
        sync()
        compile_seconds = time.perf_counter() - started

    for _ in range(args.warmup_updates):
        run_update(model, loss_fn, optimizer, past, future, accum_steps)
    sync()
    elapsed = []
    for _ in range(args.timed_updates):
        started = time.perf_counter()
        loss = run_update(model, loss_fn, optimizer, past, future, accum_steps)
        sync()
        elapsed.append(time.perf_counter() - started)
    result = {
        "stage": stage,
        "mode": "compiled" if compiled else "eager",
        "batch_size": batch_size,
        "gradient_accumulation_steps": accum_steps,
        "compile_first_update_seconds": compile_seconds,
        "warmup_updates": args.warmup_updates,
        "timed_updates": args.timed_updates,
        "mean_update_seconds": sum(elapsed) / len(elapsed),
        "median_update_seconds": sorted(elapsed)[len(elapsed) // 2],
        "last_loss": loss,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
    }
    del optimizer, model, past, future
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--checkpoint-run", default="weather_allv_s8")
    parser.add_argument("--warmup-updates", type=int, default=3)
    parser.add_argument("--timed-updates", type=int, default=10)
    parser.add_argument(
        "--modes",
        choices=("eager", "compiled"),
        nargs="+",
        default=("eager", "compiled"),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    config = load_experiment_config(args.config, {"dataset": "weather", "seed": 42})
    state = PipelineState.from_config(config)
    pipeline_mod.resolve_pipeline_data_subset(state)
    all_results = []
    for stage in ("coarse", "patch_refine"):
        for mode in args.modes:
            compiled = mode == "compiled"
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            result = benchmark(args, state, config, stage, compiled)
            all_results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    by_key = {(r["stage"], r["mode"]): r for r in all_results}
    if set(args.modes) == {"eager", "compiled"}:
        for stage in ("coarse", "patch_refine"):
            eager = by_key[(stage, "eager")]["mean_update_seconds"]
            compiled = by_key[(stage, "compiled")]["mean_update_seconds"]
            print(json.dumps({"stage": stage, "speedup": eager / compiled}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
