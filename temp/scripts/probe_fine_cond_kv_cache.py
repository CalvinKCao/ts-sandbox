#!/usr/bin/env python3
"""Fine unique-seg eval generate: cache lookback-cond K/V vs baseline 228-token SDPA.

Traffic-shaped geometry (p32x6, L=336, H=720, ~145 col0s) at V=4. Full traffic
V=862 is not required to time the 145-pack and may OOM. Compile on both arms.
Cached path is NOT numerically identical (cond does not attend to crop).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")

from models.diffusion_tsf.patch_refine_geometry import primary_stride_col0s
from models.diffusion_tsf.pipeline.config import load_experiment_config
from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import stage_state
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline.train.checkpointing import amp_context
from models.diffusion_tsf.train_multivariate_pipeline import (
    create_diffusion_model,
    create_itransformer,
    wrap_itrans_guidance,
)

CONFIG = (
    "configs/binary_window_norm_patch_refine_canvas128_p32x6_allv_randwin_lr10_"
    "cap1x2x_hz720_nostitch_nopretrain.yaml"
)
TRAFFIC_V = 862


def _oom(exc: BaseException) -> bool:
    err = str(exc).lower()
    return "out of memory" in err or "cuda oom" in err


def _peak_mib(device: torch.device) -> float:
    return torch.cuda.max_memory_allocated(device) / 1024**2


def _build_refine(
    state: PipelineState,
    device: torch.device,
    n_variates: int,
    *,
    cache_cond_kv: bool,
    torch_compile: bool,
):
    st = state
    st.n_variates = int(n_variates)
    st.dataset = "traffic"
    st.wandb_enabled = False
    st.smoke_test = False
    st.torch_compile = bool(torch_compile)
    itrans = create_itransformer(st, num_vars=n_variates).to(device)
    for p in itrans.parameters():
        p.requires_grad = False
    itrans.eval()
    guidance = wrap_itrans_guidance(itrans, st)
    refine = create_diffusion_model(
        stage_state(st, "patch_refine"),
        n_variates=n_variates,
        guidance_model=guidance,
        cache_cond_kv=bool(cache_cond_kv),
    ).to(device)
    refine.eval()
    if bool(refine.config.cache_cond_kv) != bool(cache_cond_kv):
        raise RuntimeError(
            f"cache_cond_kv config {refine.config.cache_cond_kv} != {cache_cond_kv}"
        )
    pred = refine.noise_predictor
    inner = getattr(pred, "_orig_mod", pred)
    if bool(inner.cache_cond_kv) != bool(cache_cond_kv):
        raise RuntimeError(
            f"FactorizedDiT.cache_cond_kv={inner.cache_cond_kv} != {cache_cond_kv}"
        )
    if not bool(refine.config.patch_refine_unique_segments):
        raise RuntimeError("probe requires patch_refine_unique_segments=true")
    return refine, itrans


def _past(state: PipelineState, batch: int, n_variates: int, device: torch.device) -> torch.Tensor:
    return torch.randn(int(batch), n_variates, int(state.lookback_length), device=device)


def _dummy_coarse(
    state: PipelineState, batch: int, n_variates: int, device: torch.device,
) -> torch.Tensor:
    h = int(state.coarse_image_height)
    w = int(state.forecast_length) + int(state.lookback_overlap)
    return torch.rand(int(batch), n_variates, h, w, device=device)


def _generate_once(model, past, coarse_2d, sampler: str, steps: int) -> None:
    with torch.no_grad(), amp_context(True):
        model.generate(
            past,
            future_coarse_2d=coarse_2d,
            sampler=sampler,
            num_inference_steps=int(steps),
        )
    torch.cuda.synchronize()


def _time_arm(
    *,
    label: str,
    state: PipelineState,
    device: torch.device,
    n_variates: int,
    n_windows: int,
    n_samples: int,
    cache_cond_kv: bool,
    torch_compile: bool,
    chunk: int,
    sampler: str,
    steps: int,
    n_warm: int,
    n_meas: int,
) -> Dict[str, Any]:
    print(
        f"\n=== {label}: cache_cond_kv={cache_cond_kv} compile={torch_compile} ===",
        flush=True,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    refine, itrans = _build_refine(
        state, device, n_variates,
        cache_cond_kv=cache_cond_kv, torch_compile=torch_compile,
    )
    refine.config.unet_max_chunk_size = int(chunk)
    b_pack = int(n_windows) * int(n_samples)
    past = _past(state, b_pack, n_variates, device)
    coarse_2d = _dummy_coarse(state, b_pack, n_variates, device)
    canvas_w = int(state.forecast_length) + int(state.lookback_overlap)
    n_col0 = len(primary_stride_col0s(
        canvas_w, int(state.patch_refine_patch_width), int(state.patch_refine_col_stride),
    ))
    n_patches = n_col0 * b_pack * n_variates
    print(
        f"  packed B={b_pack} (windows={n_windows} x n_samples={n_samples}) "
        f"V={n_variates} n_col0={n_col0} n_patches={n_patches} chunk={chunk}",
        flush=True,
    )
    row: Dict[str, Any] = {
        "label": label,
        "cache_cond_kv": bool(cache_cond_kv),
        "compile": bool(torch_compile),
        "n_variates": int(n_variates),
        "n_windows": int(n_windows),
        "n_samples": int(n_samples),
        "n_col0": int(n_col0),
        "n_patches": int(n_patches),
        "chunk": int(chunk),
        "sampler": sampler,
        "steps": int(steps),
        "numerically_identical_to_baseline": False if cache_cond_kv else True,
    }
    t_compile = None
    try:
        t0 = time.perf_counter()
        for i in range(max(1, int(n_warm))):
            print(f"  warmup {i + 1}/{n_warm} (compile graph on first call)...", flush=True)
            _generate_once(refine, past, coarse_2d, sampler, steps)
        torch.cuda.synchronize()
        t_compile = time.perf_counter() - t0
        print(f"  warmup wall {t_compile:.1f}s  peak={_peak_mib(device):.0f} MiB", flush=True)
        torch.cuda.reset_peak_memory_stats(device)
        times = []
        for i in range(int(n_meas)):
            t0 = time.perf_counter()
            _generate_once(refine, past, coarse_2d, sampler, steps)
            dt = time.perf_counter() - t0
            times.append(dt)
            print(f"  meas {i + 1}/{n_meas}: {dt * 1e3:.1f} ms", flush=True)
        peak = _peak_mib(device)
        ms = [t * 1e3 for t in times]
        row.update({
            "ok": True,
            "warmup_s": round(t_compile, 3),
            "ms_per_generate": [round(x, 2) for x in ms],
            "ms_mean": round(statistics.mean(ms), 2),
            "ms_median": round(statistics.median(ms), 2),
            "peak_alloc_mib": round(peak, 1),
        })
        print(
            f"  mean {row['ms_mean']:.1f} ms/generate  peak {peak:.0f} MiB",
            flush=True,
        )
    except RuntimeError as exc:
        if not _oom(exc):
            raise
        row.update({"ok": False, "oom": True, "error": str(exc)[:400]})
        print(f"  OOM: {exc}", flush=True)
        torch.cuda.empty_cache()
    finally:
        del refine, itrans, past, coarse_2d
        gc.collect()
        torch.cuda.empty_cache()
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-variates", type=int, default=4)
    p.add_argument("--n-windows", type=int, default=1)
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--n-warm", type=int, default=1)
    p.add_argument("--n-meas", type=int, default=3)
    p.add_argument("--chunk", type=int, default=0, help="0 = unchunked 145-pack")
    p.add_argument("--sampler", default="quad_t")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--out-json", default="")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    print("NOTE: cached cond K/V path is NOT numerically identical to bidirectional "
          "228-token self-attn (cond tokens do not attend to crop tokens). "
          "This probe times eval generate only.", flush=True)
    print(f"torch {torch.__version__} gpu {torch.cuda.get_device_name(0)}", flush=True)

    cfg = load_experiment_config(str(REPO / CONFIG), {"dataset": "traffic", "seed": 42})
    state = PipelineState.from_config(cfg)
    state.unet_max_chunk_size = int(args.chunk)
    pw = int(state.patch_refine_patch_width)
    cs = int(state.patch_refine_col_stride)
    canvas_w = int(state.forecast_length) + int(state.lookback_overlap)
    n_col0 = len(primary_stride_col0s(canvas_w, pw, cs))
    print(
        f"geometry L={state.lookback_length} H={state.forecast_length} "
        f"patch={state.patch_refine_patch_height}x{pw} stride={cs} "
        f"canvas_w={canvas_w} n_col0={n_col0} unique={state.patch_refine_unique_segments}",
        flush=True,
    )
    if n_col0 < 100:
        raise RuntimeError(f"expected ~145 stride col0s, got {n_col0}")

    compile_on = not bool(args.no_compile)
    common = dict(
        state=state,
        device=device,
        n_variates=int(args.n_variates),
        n_windows=int(args.n_windows),
        n_samples=int(args.n_samples),
        torch_compile=compile_on,
        chunk=int(args.chunk),
        sampler=str(args.sampler),
        steps=int(args.steps),
        n_warm=int(args.n_warm),
        n_meas=int(args.n_meas),
    )
    off = _time_arm(label="baseline_bidirectional", cache_cond_kv=False, **common)
    on = _time_arm(label="cache_cond_kv", cache_cond_kv=True, **common)

    summary: Dict[str, Any] = {
        "config": CONFIG,
        "note": (
            "Cached path is not numerically identical (cond ↛ crop). "
            "AdaLN cache key is (window, variate, diffusion_t). "
            "Times fine unique-seg eval generate only (dummy coarse canvas)."
        ),
        "geometry": {
            "lookback": int(state.lookback_length),
            "horizon": int(state.forecast_length),
            "patch_h": int(state.patch_refine_patch_height),
            "patch_w": pw,
            "col_stride": cs,
            "n_col0": n_col0,
            "dit_patch": list(state.dit_patch_size),
            "n_cond_tokens_p32": 224,
            "n_crop_tokens_p32": 4,
        },
        "baseline": off,
        "cached": on,
        "traffic_v": TRAFFIC_V,
    }
    if off.get("ok") and on.get("ok"):
        speedup = float(off["ms_mean"]) / float(on["ms_mean"])
        v = int(args.n_variates)
        scale = TRAFFIC_V / float(v)
        summary["speedup_cache_vs_baseline"] = round(speedup, 3)
        summary["linear_v_scale"] = {
            "measured_v": v,
            "traffic_v": TRAFFIC_V,
            "factor": round(scale, 3),
            "baseline_traffic_s_per_window_if_linear": round(
                float(off["ms_mean"]) * 1e-3 * scale, 3,
            ),
            "cached_traffic_s_per_window_if_linear": round(
                float(on["ms_mean"]) * 1e-3 * scale, 3,
            ),
        }
        print(
            f"\nSPEEDUP cache/baseline: {speedup:.2f}x  "
            f"({off['ms_mean']:.1f} ms -> {on['ms_mean']:.1f} ms)",
            flush=True,
        )
        print(
            f"If wall scaled linearly with V: traffic V={TRAFFIC_V} ~ "
            f"{summary['linear_v_scale']['baseline_traffic_s_per_window_if_linear']:.1f}s "
            f"baseline vs "
            f"{summary['linear_v_scale']['cached_traffic_s_per_window_if_linear']:.1f}s cached "
            f"per packed generate (n_samples={args.n_samples}).",
            flush=True,
        )
    else:
        print("\nIncomplete: one or both arms failed (see JSON).", flush=True)

    text = json.dumps(summary, indent=2)
    print(text, flush=True)
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
