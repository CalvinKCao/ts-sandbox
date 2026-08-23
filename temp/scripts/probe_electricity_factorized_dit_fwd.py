#!/usr/bin/env python3
"""Forward-only FactorizedDiT timing: 321-var xattn vs two independent halves.

Random weights, random CUDA tensors, no dataset I/O. Hits the live canvas128
p64x6 create_diffusion_model path (bottleneck cross-attn, no 2D guidance
channel). Times compiled denoiser cold vs steady-state.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any, Optional

import torch
import torch.nn as nn

from models.diffusion_tsf import train_multivariate_pipeline as pipeline_mod
from models.diffusion_tsf.pipeline.config import load_experiment_config
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline.train.checkpointing import amp_context


class _TokenStub(nn.Module):
    """Satisfies create_diffusion_model; DiT xattn uses our random ctx tensors."""

    def get_encoder_tokens(self, past: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("stub is not used; probe feeds encoder_hidden_states directly")


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def n_past_patches(lookback: int, patch_size: int) -> int:
    pad = (patch_size - lookback % patch_size) % patch_size
    return (lookback + pad) // patch_size


def median_ms(samples: list[float]) -> float:
    return 1000.0 * statistics.median(samples)


def peak_mib(device: torch.device) -> float:
    return torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)


def build_model(state: PipelineState, n_variates: int, stage: str, device: torch.device):
    lookback, horizon = pipeline_mod.dataset_window_lengths(state, state.dataset)
    if stage == "patch_refine":
        state.image_height = int(state.patch_refine_patch_height)
    else:
        state.image_height = int(state.coarse_image_height)
    state.n_variates = int(n_variates)
    state.smoke_test = False
    state.torch_compile = True
    model = pipeline_mod.create_diffusion_model(
        state,
        n_variates=n_variates,
        lookback=lookback,
        horizon=horizon,
        guidance_model=_TokenStub(),
        diffusion_stage=stage,
    ).to(device)
    model.eval()
    return model, lookback, horizon


def make_inputs(
    model,
    *,
    batch: int,
    n_variates: int,
    n_past: int,
    device: torch.device,
):
    cfg = model.config
    stage = cfg.diffusion_stage
    B, V = batch, n_variates
    ctx_dim = int(cfg.context_embedding_dim)
    M = V * n_past
    token_variate_ids = torch.arange(V, device=device).repeat_interleave(n_past)

    if stage == "patch_refine":
        h = int(cfg.patch_refine_patch_height)
        w = int(cfg.patch_refine_patch_width)
        n_rows = B * V
        canvas = torch.randn(n_rows, cfg.backbone_in_channels, h, w, device=device)
        cond = torch.randn(n_rows, cfg.visual_cond_channels, h, w, device=device)
        t = torch.randint(0, cfg.binary_num_steps, (n_rows,), device=device)
        ctx = torch.randn(B, M, ctx_dim, device=device)
        context_window_indices = torch.arange(B, device=device).repeat_interleave(V)
        kwargs = {
            "encoder_hidden_states": ctx,
            "token_variate_ids": token_variate_ids,
            "context_window_indices": context_window_indices,
            "variate_indices": torch.arange(V, device=device).repeat(B),
            "patch_coarse_bin": torch.zeros(n_rows, dtype=torch.long, device=device),
            "patch_time0": torch.zeros(n_rows, dtype=torch.long, device=device),
        }
        return canvas, t, cond, kwargs

    h = int(cfg.image_height)
    w_canvas = int(cfg.lookback_overlap) + int(cfg.dataset_forecast_length or cfg.forecast_length)
    w_cond = int(cfg.diffusion_lookback_cap or 0) or w_canvas
    n_rows = B * V
    canvas = torch.randn(n_rows, cfg.backbone_in_channels, h, w_canvas, device=device)
    cond = torch.randn(n_rows, cfg.visual_cond_channels, h, w_cond, device=device)
    t = torch.randint(0, cfg.binary_num_steps, (n_rows,), device=device)
    ctx_win = torch.randn(B, M, ctx_dim, device=device)
    ctx = model._flatten_ctx_for_factorized_dit(ctx_win, B, V)
    kwargs = {
        "encoder_hidden_states": ctx,
        "token_variate_ids": token_variate_ids,
        "variate_indices": torch.arange(V, device=device).repeat(B),
    }
    return canvas, t, cond, kwargs


def run_fwd(model, canvas, t, cond, kwargs, use_amp: bool, chunk: int):
    extra = {k: v for k, v in kwargs.items() if k != "encoder_hidden_states"}
    ctx = kwargs.get("encoder_hidden_states")
    saved = int(model.config.unet_max_chunk_size)
    model.config.unet_max_chunk_size = int(chunk)
    try:
        with torch.no_grad(), amp_context(use_amp):
            if chunk > 0:
                return model._predict_noise_chunked(canvas, t, cond, ctx, **extra)
            return model.noise_predictor(canvas, t, cond, **kwargs)
    finally:
        model.config.unet_max_chunk_size = saved


def time_case(
    model,
    *,
    label: str,
    batch: int,
    n_variates: int,
    n_past: int,
    device: torch.device,
    warmup: int,
    timed: int,
    use_amp: bool,
    chunk: int,
) -> dict[str, Any]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    canvas, t, cond, kwargs = make_inputs(
        model, batch=batch, n_variates=n_variates, n_past=n_past, device=device,
    )
    shapes = {
        "canvas": list(canvas.shape),
        "cond": list(cond.shape),
        "ctx": list(kwargs["encoder_hidden_states"].shape),
        "n_rows": int(canvas.shape[0]),
        "n_ctx_tokens": int(kwargs["encoder_hidden_states"].shape[1]),
        "batch": batch,
        "n_variates": n_variates,
        "chunk": chunk,
    }
    print(f"  first-pass {label} {shapes}", flush=True)
    try:
        sync()
        t0 = time.perf_counter()
        out = run_fwd(model, canvas, t, cond, kwargs, use_amp, chunk)
        sync()
        cold_s = time.perf_counter() - t0
    except torch.cuda.OutOfMemoryError:
        if chunk > 0:
            raise
        print(f"  OOM unchunked {label}; retry chunk=128", flush=True)
        del canvas, t, cond, kwargs
        torch.cuda.empty_cache()
        return time_case(
            model,
            label=label + "_chunk128",
            batch=batch,
            n_variates=n_variates,
            n_past=n_past,
            device=device,
            warmup=warmup,
            timed=timed,
            use_amp=use_amp,
            chunk=128,
        )
    out_shape = list(out.shape)
    for _ in range(warmup):
        run_fwd(model, canvas, t, cond, kwargs, use_amp, chunk)
    sync()
    samples = []
    for _ in range(timed):
        t0 = time.perf_counter()
        run_fwd(model, canvas, t, cond, kwargs, use_amp, chunk)
        sync()
        samples.append(time.perf_counter() - t0)
    result = {
        "label": label,
        "cold_first_ms": 1000.0 * cold_s,
        "median_ms": median_ms(samples),
        "mean_ms": 1000.0 * (sum(samples) / len(samples)),
        "min_ms": 1000.0 * min(samples),
        "max_ms": 1000.0 * max(samples),
        "n_timed": timed,
        "warmup": warmup,
        "peak_mib": peak_mib(device),
        "out_shape": out_shape,
        **shapes,
    }
    del canvas, t, cond, kwargs, out
    torch.cuda.empty_cache()
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def summarize(rows: list[dict[str, Any]], gpu: str) -> dict[str, Any]:
    by = {r["label"]: r for r in rows}

    def ms(key: str) -> Optional[float]:
        row = by.get(key)
        return None if row is None else float(row["median_ms"])

    v321 = ms("patch_refine/V321/B1")
    v160 = ms("patch_refine/V160/B1")
    v161 = ms("patch_refine/V161/B1")
    seq = None
    if v160 is not None and v161 is not None:
        seq = v160 + v161
    elif v160 is not None:
        seq = 2.0 * v160
        v161 = v160
    summary = {
        "gpu": gpu,
        "patch_refine_ms_fwd_321": v321,
        "patch_refine_ms_fwd_160": v160,
        "patch_refine_ms_fwd_161": v161,
        "patch_refine_sequential_160_161": seq,
        "patch_refine_2x_one_half_parallel": None if v160 is None else 1.0 * v160,
        "patch_refine_speedup_split_sequential": None
        if v321 is None or seq in (None, 0) else v321 / seq,
        "patch_refine_speedup_split_2gpu": None
        if v321 is None or v160 in (None, 0) else v321 / v160,
        "coarse_ms_fwd_321": ms("coarse/V321/B1"),
        "coarse_ms_fwd_160": ms("coarse/V160/B1"),
        "rows": rows,
    }
    print("\n======== SUMMARY ========", flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2), flush=True)
    if v321 is not None:
        print(f"321-var xattn (patch_refine B=1): {v321:.2f} ms/fwd", flush=True)
    if v160 is not None:
        print(f"one 160-var half (patch_refine B=1): {v160:.2f} ms/fwd", flush=True)
    if seq is not None:
        print(f"sequential 160+161: {seq:.2f} ms  vs  single 321: {v321}", flush=True)
        print(f"2-GPU parallel (2x one-half): {v160:.2f} ms wall (each half on its GPU)", flush=True)
    mem321 = by.get("patch_refine/V321/B1", {}).get("peak_mib")
    mem160 = by.get("patch_refine/V160/B1", {}).get("peak_mib")
    print(f"peak mem 321: {mem321} MiB   160: {mem160} MiB", flush=True)
    print(f"GPU: {gpu}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/binary_window_norm_patch_refine_canvas128_p64x6_electricity.yaml",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timed", type=int, default=10)
    parser.add_argument("--stages", nargs="+", default=("patch_refine", "coarse"))
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--unchunked", action="store_true", help="Force one DiT call (default).")
    parser.add_argument(
        "--chunk",
        type=int,
        default=None,
        help="If set, chunk BV through FactorizedDiT like train (128).",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    gpu = torch.cuda.get_device_name(0)
    print(f"gpu={gpu}  torch={torch.__version__}", flush=True)

    cfg = load_experiment_config(args.config, {"dataset": "electricity", "seed": 42})
    state = PipelineState.from_config(cfg)
    lookback, _horizon = pipeline_mod.dataset_window_lengths(state, "electricity")
    patch_size = int(state.mmpd_patch_size)
    n_past = n_past_patches(lookback, patch_size)
    # Default: one denoiser call with all variates (fair 321 vs half).
    # Pass --chunk 128 to match train's unet_max_chunk_size.
    if args.chunk is not None:
        chunk_default = int(args.chunk)
    else:
        chunk_default = 0
    use_amp = (not args.no_amp) and bool(state.use_amp)
    print(
        json.dumps(
            {
                "lookback": lookback,
                "horizon": int(state.forecast_length),
                "overlap": int(state.lookback_overlap),
                "mmpd_patch_size": patch_size,
                "n_past_patches": n_past,
                "ctx_tokens_321": 321 * n_past,
                "ctx_tokens_160": 160 * n_past,
                "dit_patch": list(state.dit_patch_size),
                "canvas128_p64x6": {
                    "canvas_h": int(state.patch_refine_canvas_height),
                    "patch_h": int(state.patch_refine_patch_height),
                    "patch_w": int(state.patch_refine_patch_width),
                },
                "torch_compile": True,
                "compile_kwargs": {
                    "backend": "inductor",
                    "fullgraph": False,
                    "dynamic": True,
                },
                "use_amp": use_amp,
                "chunk": chunk_default,
                "disable_cross_attention": bool(state.disable_cross_attention),
                "guidance_type": state.guidance_type,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if bool(state.disable_cross_attention):
        raise RuntimeError("probe expects live xattn (disable_cross_attention=false)")

    rows: list[dict[str, Any]] = []
    # One compiled denoiser per stage (V=321). Halves reuse it with shorter ctx / fewer rows.
    for stage in args.stages:
        print(f"\n===== {stage}  build+compile V=321 =====", flush=True)
        t_build = time.perf_counter()
        model, _, _ = build_model(state, n_variates=321, stage=stage, device=device)
        print(f"  constructed in {time.perf_counter() - t_build:.1f}s", flush=True)
        in_ch = int(model.config.backbone_in_channels)
        cond_ch = int(model.config.visual_cond_channels)
        print(f"  in_channels={in_ch} cond_channels={cond_ch} (no 2D guidance map)", flush=True)
        cases = [
            ("V321/B1", 1, 321, chunk_default),
            ("V160/B1", 1, 160, chunk_default),
            ("V161/B1", 1, 161, chunk_default),
            ("V160/B2", 2, 160, chunk_default),
        ]
        if stage == "coarse":
            # target_univariate_batch=336 → B=1 for 321, B=2 for 160
            cases.append(("V321/B1_train", 1, 321, chunk_default))
        try:
            for name, batch, n_var, chunk in cases:
                rows.append(
                    time_case(
                        model,
                        label=f"{stage}/{name}",
                        batch=batch,
                        n_variates=n_var,
                        n_past=n_past,
                        device=device,
                        warmup=args.warmup,
                        timed=args.timed,
                        use_amp=use_amp,
                        chunk=chunk,
                    )
                )
        except torch.cuda.OutOfMemoryError as exc:
            print(f"OOM during {stage}: {exc}", flush=True)
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()

    summary = summarize(rows, gpu)
    out_path = "results/logs/probe-electricity-dit-fwd-summary.json"
    try:
        with open(out_path, "w") as handle:
            json.dump(summary, handle, indent=2)
        print(f"wrote {out_path}", flush=True)
    except OSError as exc:
        print(f"summary write skipped: {exc}", flush=True)


if __name__ == "__main__":
    main()
