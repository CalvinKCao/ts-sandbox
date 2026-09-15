#!/usr/bin/env python3
"""A100 40GB max unet_max_chunk_size for eval-shaped unique-seg generate.

Binary-searches coarse and fine chunk size. Packs like staged_eval's
probabilistic path: past is repeat-interleaved to B=n_samples (default 10),
steps=4, cache_cond_kv=True, real V and fine crop counts.

Prints one line per trial:
  stage chunk n_items n_launches ok|OOM alloc_GiB reserved_GiB seconds
and a final JSON summary with separate coarse/fine maxima.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    dataset_window_lengths,
    wrap_itrans_guidance,
)

DATASET_V = {
    "electricity": 321,
    "traffic": 862,
    "solar_Alabama": 137,
    "PeMS": 307,
    "dynamic": 17,
}


def _oom(exc: BaseException) -> bool:
    err = str(exc).lower()
    return "out of memory" in err or "cuda oom" in err


def _gib(nbytes: int) -> float:
    return nbytes / (1024.0 ** 3)


def _log_mem(prefix: str, device: torch.device) -> None:
    alloc = _gib(torch.cuda.memory_allocated(device))
    reserved = _gib(torch.cuda.memory_reserved(device))
    peak_a = _gib(torch.cuda.max_memory_allocated(device))
    peak_r = _gib(torch.cuda.max_memory_reserved(device))
    print(
        f"{prefix} alloc={alloc:.2f}GiB reserved={reserved:.2f}GiB "
        f"peak_alloc={peak_a:.2f}GiB peak_reserved={peak_r:.2f}GiB",
        flush=True,
    )


def _build_state(config_path: str, dataset: str, n_variates: int) -> PipelineState:
    cfg = load_experiment_config(config_path, {"dataset": dataset})
    state = PipelineState.from_config(cfg)
    state.dataset = dataset
    state.n_variates = int(n_variates)
    state.wandb_enabled = False
    state.smoke_test = False
    if not bool(getattr(state, "patch_refine_unique_segments", False)):
        raise RuntimeError("probe requires patch_refine_unique_segments=true")
    return state


def _build_pair(
    state: PipelineState,
    device: torch.device,
    n_variates: int,
    *,
    torch_compile: bool,
):
    st = state
    st.n_variates = int(n_variates)
    st.torch_compile = bool(torch_compile)
    lookback, horizon = dataset_window_lengths(st, st.dataset)
    itrans = create_itransformer(st, num_vars=n_variates).to(device)
    for p in itrans.parameters():
        p.requires_grad = False
    itrans.eval()
    guidance = wrap_itrans_guidance(itrans, st)
    coarse = create_diffusion_model(
        stage_state(st, "coarse", honor_dataset_windows=True),
        n_variates=n_variates,
        lookback=lookback,
        horizon=horizon,
        guidance_model=guidance,
        diffusion_stage="coarse",
        cache_cond_kv=False,
    ).to(device)
    fine = create_diffusion_model(
        stage_state(st, "patch_refine", honor_dataset_windows=True),
        n_variates=n_variates,
        lookback=lookback,
        horizon=horizon,
        guidance_model=guidance,
        diffusion_stage="patch_refine",
        cache_cond_kv=True,
    ).to(device)
    coarse.eval()
    fine.eval()
    inner = getattr(fine.noise_predictor, "_orig_mod", fine.noise_predictor)
    if not bool(fine.config.cache_cond_kv) or not bool(inner.cache_cond_kv):
        raise RuntimeError("fine cache_cond_kv did not land on FactorizedDiT")
    if not bool(fine.config.patch_refine_unique_segments):
        raise RuntimeError("fine unique_segments is false")
    return coarse, fine, itrans, lookback, horizon


def _past(n_samples: int, lookback: int, n_variates: int, device: torch.device) -> torch.Tensor:
    return torch.randn(int(n_samples), n_variates, int(lookback), device=device)


def _dummy_coarse_2d(model, n_samples: int, n_variates: int, device: torch.device) -> torch.Tensor:
    h = int(model.config.coarse_image_height)
    w = int(model._repr_forecast_width(int(model.config.forecast_length)))
    return torch.rand(int(n_samples), n_variates, h, w, device=device)


@torch.no_grad()
def _run_generate(model, past, *, future_coarse_2d, sampler: str, steps: int) -> None:
    kwargs: Dict[str, Any] = {"sampler": sampler, "num_inference_steps": int(steps)}
    if future_coarse_2d is not None:
        kwargs["future_coarse_2d"] = future_coarse_2d
    with amp_context(True):
        model.generate(past, **kwargs)
    torch.cuda.synchronize()


def _n_launches(n_items: int, chunk: int) -> int:
    return int(math.ceil(n_items / float(chunk))) if chunk > 0 else 1


def _trial(
    *,
    stage: str,
    model,
    past,
    future_coarse_2d,
    chunk: int,
    sampler: str,
    steps: int,
    n_items: int,
    device: torch.device,
) -> Dict[str, Any]:
    saved = int(model.config.unet_max_chunk_size)
    model.config.unet_max_chunk_size = int(chunk)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    row: Dict[str, Any] = {
        "stage": stage,
        "chunk": int(chunk),
        "n_items": int(n_items),
        "n_launches": _n_launches(n_items, chunk),
    }
    try:
        _run_generate(
            model, past, future_coarse_2d=future_coarse_2d, sampler=sampler, steps=steps,
        )
        dt = time.perf_counter() - t0
        alloc = _gib(torch.cuda.max_memory_allocated(device))
        reserved = _gib(torch.cuda.max_memory_reserved(device))
        row.update({
            "ok": True,
            "oom": False,
            "seconds": round(dt, 2),
            "alloc_GiB": round(alloc, 2),
            "reserved_GiB": round(reserved, 2),
        })
        print(
            f"TRIAL {stage} chunk={chunk} n_items={n_items} "
            f"n_launches={row['n_launches']} OK "
            f"alloc={alloc:.2f}GiB reserved={reserved:.2f}GiB {dt:.1f}s",
            flush=True,
        )
    except RuntimeError as exc:
        dt = time.perf_counter() - t0
        if not _oom(exc):
            model.config.unet_max_chunk_size = saved
            raise
        row.update({
            "ok": False,
            "oom": True,
            "seconds": round(dt, 2),
            "alloc_GiB": None,
            "reserved_GiB": None,
            "error": str(exc).split("\n")[0][:240],
        })
        print(
            f"TRIAL {stage} chunk={chunk} n_items={n_items} "
            f"n_launches={row['n_launches']} OOM {dt:.1f}s {row['error']}",
            flush=True,
        )
        torch.cuda.empty_cache()
    finally:
        model.config.unet_max_chunk_size = saved
    return row


def _binary_search(
    *,
    stage: str,
    model,
    past,
    future_coarse_2d,
    sampler: str,
    steps: int,
    n_items: int,
    device: torch.device,
    lo: int,
    hi_excl: int,
    trials: List[Dict[str, Any]],
) -> int:
    first = _trial(
        stage=stage, model=model, past=past, future_coarse_2d=future_coarse_2d,
        chunk=lo, sampler=sampler, steps=steps, n_items=n_items, device=device,
    )
    trials.append(first)
    if not first["ok"]:
        raise RuntimeError(
            f"{stage} OOM at known-ok chunk {lo}; drop --lo and retry. "
            f"n_samples pack is larger than a 1-sample probe."
        )
    best = lo
    hi = min(int(hi_excl), int(n_items) + 1)
    if hi <= best + 1:
        return best
    cand = lo * 2
    while cand < hi:
        row = _trial(
            stage=stage, model=model, past=past, future_coarse_2d=future_coarse_2d,
            chunk=cand, sampler=sampler, steps=steps, n_items=n_items, device=device,
        )
        trials.append(row)
        if row["ok"]:
            best = cand
            cand *= 2
        else:
            hi = cand
            break
    while hi - best > 256:
        mid = (best + hi) // 2
        row = _trial(
            stage=stage, model=model, past=past, future_coarse_2d=future_coarse_2d,
            chunk=mid, sampler=sampler, steps=steps, n_items=n_items, device=device,
        )
        trials.append(row)
        if row["ok"]:
            best = mid
        else:
            hi = mid
    return best


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--n-variates", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=10,
                   help="Eval MC pack; matches staged_eval past.repeat_interleave")
    p.add_argument("--lo", type=int, default=1024)
    p.add_argument("--hi", type=int, default=None,
                   help="Exclusive upper bound. Default n_items+1. Pass a known-OOM size.")
    p.add_argument("--sampler", default="quad_t")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--no-compile", action="store_true")
    args = p.parse_args()

    n_variates = args.n_variates
    if n_variates is None:
        if args.dataset not in DATASET_V:
            raise SystemExit(f"--n-variates required for dataset {args.dataset}")
        n_variates = DATASET_V[args.dataset]
    n_samples = int(args.n_samples)
    if n_samples < 1:
        raise SystemExit("--n-samples must be >= 1")
    if int(args.lo) < 1:
        raise SystemExit("--lo must be >= 1")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    gpu = torch.cuda.get_device_name(0)
    total = _gib(torch.cuda.get_device_properties(0).total_memory)
    print(f"GPU={gpu} total={total:.2f}GiB", flush=True)
    print(
        f"weights=random_init (memory probe) dataset={args.dataset} V={n_variates} "
        f"n_samples={n_samples} cache_cond_kv=True unique_seg generate "
        f"lo={args.lo} hi_excl={args.hi} sampler={args.sampler} steps={args.steps}",
        flush=True,
    )

    state = _build_state(args.config, args.dataset, n_variates)
    compile_on = bool(state.torch_compile) and not bool(args.no_compile)
    print(
        f"config={args.config} lookback={state.lookback_length} "
        f"horizon={state.forecast_length} overlap={state.lookback_overlap} "
        f"patch={state.patch_refine_patch_width}x{state.patch_refine_patch_height} "
        f"stride={state.patch_refine_col_stride} torch_compile={compile_on}",
        flush=True,
    )

    t_build = time.perf_counter()
    coarse, fine, itrans, lookback, horizon = _build_pair(
        state, device, n_variates, torch_compile=compile_on,
    )
    print(f"built coarse+fine in {time.perf_counter() - t_build:.1f}s", flush=True)
    _log_mem("after_build", device)

    past = _past(n_samples, lookback, n_variates, device)
    coarse_2d = _dummy_coarse_2d(fine, n_samples, n_variates, device)
    w_fut = int(fine._repr_forecast_width(int(fine.config.forecast_length)))
    n_col0 = len(primary_stride_col0s(
        w_fut,
        int(fine.config.patch_refine_patch_width),
        int(fine.config.patch_refine_col_stride),
    ))
    n_fine = n_col0 * n_variates * n_samples
    n_coarse = n_variates * n_samples
    hi_excl: Optional[int] = int(args.hi) if args.hi is not None else None
    print(
        f"geometry W_fut={w_fut} n_col0={n_col0} "
        f"fine_n_items={n_fine} (B={n_samples} x V={n_variates} x n_col0) "
        f"coarse_n_items={n_coarse}",
        flush=True,
    )

    trials: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "gpu": gpu,
        "total_GiB": round(total, 2),
        "weights": "random_init",
        "dataset": args.dataset,
        "n_variates": int(n_variates),
        "n_samples": int(n_samples),
        "lookback": int(lookback),
        "horizon": int(horizon),
        "n_col0": int(n_col0),
        "fine_n_items": int(n_fine),
        "coarse_n_items": int(n_coarse),
        "cache_cond_kv": True,
        "torch_compile": compile_on,
        "sampler": args.sampler,
        "steps": int(args.steps),
        "lo": int(args.lo),
        "hi_excl": hi_excl,
    }

    print(
        f"\n=== COARSE unique-seg generate (B={n_samples}, V={n_variates}) ===",
        flush=True,
    )
    coarse_hi = n_coarse + 1 if hi_excl is None else hi_excl
    coarse_max = _binary_search(
        stage="coarse", model=coarse, past=past, future_coarse_2d=None,
        sampler=args.sampler, steps=args.steps, n_items=n_coarse,
        device=device, lo=args.lo, hi_excl=coarse_hi, trials=trials,
    )
    print(f"COARSE_MAX_CHUNK {coarse_max}", flush=True)

    print(
        f"\n=== FINE unique-seg generate (B={n_samples}, V={n_variates}, cache_cond_kv=True) ===",
        flush=True,
    )
    fine_hi = n_fine + 1 if hi_excl is None else hi_excl
    fine_max = _binary_search(
        stage="fine", model=fine, past=past, future_coarse_2d=coarse_2d,
        sampler=args.sampler, steps=args.steps, n_items=n_fine,
        device=device, lo=args.lo, hi_excl=fine_hi, trials=trials,
    )
    print(f"FINE_MAX_CHUNK {fine_max}", flush=True)

    shard_chunk = int(min(coarse_max, fine_max))
    summary.update({
        "coarse_max_chunk": int(coarse_max),
        "fine_max_chunk": int(fine_max),
        "shard_unet_max_chunk_size": shard_chunk,
        "trials": trials,
    })
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "trials"}, indent=2), flush=True)
    print(f"SHARD_UNET_MAX_CHUNK_SIZE {shard_chunk}", flush=True)

    del coarse, fine, itrans, past, coarse_2d
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
