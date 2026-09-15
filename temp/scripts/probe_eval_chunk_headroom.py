#!/usr/bin/env python3
"""Eval-shaped unique-seg chunk probe with reserved-memory headroom.

Loads real staged_eval weights, packs past like probabilistic eval
(n_samples=10, steps=4, cache_cond_kv on fine, compile if YAML says so),
and binary-searches unet_max_chunk_size. A trial only counts as a fit when
peak reserved GiB is under the cap. Does not treat "not OOM" or allocated
GiB as a pass.

--fast-chunk-fwd: memory search runs one/two DiT forwards on a batch of
exactly C items (same dtype/layout as eval), plus optional full-pack hold
tensors so reserved includes unique-seg xt/aux/canvas buffers. Do not
iterate all 1.25M crops during search. Only the winning C is timed as a
full window (or estimated from single-forward ms if --skip-full-window).

--det-only: pack B=n_samples (use 1 for anchor eval; no repeat_interleave(10)),
skip the probabilistic generate, and gate the reserved cap on a full det
window — not a fast-fwd scrape.

A winning chunk is accepted only after --confirm-repeats forwards/generates
on the same process (compile warm). Optionally step down one search notch
if the max-pass scrapes the cap.
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
from typing import Any, Dict, List, Optional, Set

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")

from models.diffusion_tsf.patch_refine import (
    build_patch_aux_channels_layout,
    naive_upscale_coarse_cdf,
)
from models.diffusion_tsf.patch_refine_geometry import (
    PatchLayout,
    coarse_edges_from_cdf,
    patch_layout_for_fixed_col0,
    primary_stride_col0s,
)
from models.diffusion_tsf.pipeline.config import load_experiment_config
from models.diffusion_tsf.pipeline.phases.staged_eval import (
    StagedEvalPhase,
    _staged_det_gen_kwargs,
)
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline.train.checkpointing import amp_context
from models.diffusion_tsf.train_multivariate_pipeline import (
    dataset_window_lengths,
    load_dataset,
    load_wrapped_guidance,
    resolve_pipeline_data_subset,
)

BANNED_CHUNKS = {21634, 26624, 39350}
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


def _n_launches(n_items: int, chunk: int) -> int:
    return int(math.ceil(n_items / float(chunk))) if chunk > 0 else 1


@torch.no_grad()
def _run_generate(model, past, *, future_coarse_2d, sampler: str, steps: int) -> Any:
    kwargs: Dict[str, Any] = {"sampler": sampler, "num_inference_steps": int(steps)}
    if future_coarse_2d is not None:
        kwargs["future_coarse_2d"] = future_coarse_2d
    with amp_context(True):
        out = model.generate(past, **kwargs)
    torch.cuda.synchronize()
    return out


@torch.no_grad()
def _time_det_window(
    *,
    coarse,
    fine,
    past,
    chunk: int,
    coarse_chunk: int,
    state: PipelineState,
    steps: int,
    device: torch.device,
) -> Dict[str, Any]:
    coarse.config.unet_max_chunk_size = int(coarse_chunk)
    fine.config.unet_max_chunk_size = int(chunk)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    det_kwargs = _staged_det_gen_kwargs(state, int(steps))
    t0 = time.perf_counter()
    try:
        coarse_det = coarse.generate(past, **det_kwargs)
        fine.generate(past, future_coarse_2d=coarse_det["future_2d_coarse"], **det_kwargs)
        torch.cuda.synchronize()
        det_s = time.perf_counter() - t0
        return {
            "ok": True,
            "oom": False,
            "seconds": det_s,
            "reserved_GiB": _gib(torch.cuda.max_memory_reserved(device)),
            "alloc_GiB": _gib(torch.cuda.max_memory_allocated(device)),
        }
    except RuntimeError as exc:
        det_s = time.perf_counter() - t0
        if not _oom(exc):
            raise
        torch.cuda.empty_cache()
        return {
            "ok": False,
            "oom": True,
            "seconds": det_s,
            "reserved_GiB": None,
            "alloc_GiB": None,
            "error": str(exc).split("\n")[0][:240],
        }


def _build_fine_pack(model, past, future_coarse_2d) -> Dict[str, Any]:
    """Cheap unique-seg tensors (parents + geometry), not the full crop pack."""
    device = past.device
    canvas_h, patch_h, patch_w, col_stride = model._patch_refine_geometry_knobs()
    coarse_h = int(model.config.coarse_image_height)
    W_fut = int(model._repr_forecast_width(int(model.config.forecast_length)))
    past_norm, _, _stats = model._normalize_sequence(past)
    coarse = future_coarse_2d.to(device)
    naive = naive_upscale_coarse_cdf(coarse, canvas_h)
    edges = coarse_edges_from_cdf(coarse, canvas_height=canvas_h)
    lookback_cond, _past_maps = model._patch_refine_lookback_cond(past_norm)
    ctx = None
    if not bool(getattr(model.config, "disable_cross_attention", False)):
        ctx = model._get_cross_variate_context(past, past_norm)
    col0s = primary_stride_col0s(int(edges.shape[-1]), patch_w, col_stride)
    return {
        "naive": naive,
        "edges": edges,
        "lookback_cond": lookback_cond,
        "ctx": ctx,
        "col0s": col0s,
        "canvas_h": int(canvas_h),
        "patch_h": int(patch_h),
        "patch_w": int(patch_w),
        "coarse_h": int(coarse_h),
        "W_fut": int(W_fut),
        "B": int(past.shape[0]),
        "device": device,
        "cache_kv": bool(getattr(model.config, "cache_cond_kv", False)),
        "in_ch": int(model.config.backbone_in_channels),
    }


def _alloc_full_pack_hold(pack: Dict[str, Any], n_items: int) -> Dict[str, torch.Tensor]:
    """Keep unique-seg-sized buffers alive so reserved includes generate packing."""
    device = pack["device"]
    dtype = pack["naive"].dtype
    pH, pW = pack["patch_h"], pack["patch_w"]
    hold = {
        "xt": torch.empty(n_items, 1, pH, pW, device=device, dtype=dtype),
        "aux": torch.empty(n_items, 3, pH, pW, device=device, dtype=dtype),
        "canvas": torch.empty(n_items, pack["in_ch"], pH, pW, device=device, dtype=dtype),
    }
    nbytes = sum(int(t.numel() * t.element_size()) for t in hold.values())
    print(
        f"full_pack_hold n_items={n_items} p={pH}x{pW} in_ch={pack['in_ch']} "
        f"{_gib(nbytes):.2f}GiB (xt+aux+canvas)",
        flush=True,
    )
    return hold


def _layout_of_size(pack: Dict[str, Any], chunk: int) -> PatchLayout:
    layouts: List[PatchLayout] = []
    n = 0
    device = pack["device"]
    for col0 in pack["col0s"]:
        lay = patch_layout_for_fixed_col0(
            pack["edges"],
            torch.full((pack["B"],), int(col0), device=device, dtype=torch.long),
            canvas_height=pack["canvas_h"],
            patch_height=pack["patch_h"],
            patch_width=pack["patch_w"],
        )
        layouts.append(lay)
        n += lay.n_patches
        if n >= chunk:
            break
    if not layouts:
        raise RuntimeError("no col0 layouts for fast chunk fwd")
    layout = PatchLayout.cat(layouts)
    if layout.n_patches > chunk:
        layout = layout.index_select(torch.arange(chunk, device=device, dtype=torch.long))
    if layout.n_patches < 1:
        raise RuntimeError("fast chunk layout is empty")
    return layout


@torch.no_grad()
def _one_fine_chunk_fwd(model, pack: Dict[str, Any], layout: PatchLayout) -> float:
    """One DiT / _predict_noise_chunked of size C. Returns milliseconds."""
    C = layout.n_patches
    aux_l, bins, t0 = build_patch_aux_channels_layout(
        pack["naive"], pack["edges"], layout,
        patch_height=pack["patch_h"], patch_width=pack["patch_w"],
        canvas_height=pack["canvas_h"], coarse_height=pack["coarse_h"],
        horizon_width=pack["W_fut"],
    )
    device = pack["device"]
    xt = torch.rand(C, 1, pack["patch_h"], pack["patch_w"], device=device, dtype=aux_l.dtype)
    canvas = model._inject_coordinate_channel(xt)
    canvas = model._inject_time_channels(canvas)
    canvas = torch.cat([canvas, aux_l], dim=1)
    t_batch = torch.full(
        (C,), int(model.config.binary_num_steps) - 1, device=device, dtype=torch.long,
    )
    extra: Dict[str, Any] = {}
    if pack["cache_kv"]:
        extra["cond_parent_index"] = layout.flat_index
    ctx = pack["ctx"]
    context_window_indices = layout.batch_index if ctx is not None else None
    torch.cuda.synchronize()
    t0s = time.perf_counter()
    with amp_context(True):
        model._predict_noise_chunked(
            canvas, t_batch, pack["lookback_cond"], ctx,
            context_window_indices=context_window_indices,
            variate_indices=layout.variate_index,
            token_variate_ids=model._ctx_token_variate_ids,
            patch_coarse_bin=bins,
            patch_time0=t0,
            **extra,
        )
    torch.cuda.synchronize()
    return (time.perf_counter() - t0s) * 1000.0


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
    max_reserved_gib: float,
    n_repeats: int = 1,
    fast_pack: Optional[Dict[str, Any]] = None,
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
        "n_repeats": int(n_repeats),
        "fast": bool(fast_pack is not None),
    }
    try:
        fwd_ms: List[float] = []
        if fast_pack is not None:
            layout = _layout_of_size(fast_pack, int(chunk))
            row["fwd_n"] = int(layout.n_patches)
            for _ in range(max(1, int(n_repeats))):
                fwd_ms.append(_one_fine_chunk_fwd(model, fast_pack, layout))
        else:
            out = None
            for _ in range(max(1, int(n_repeats))):
                out = _run_generate(
                    model, past, future_coarse_2d=future_coarse_2d,
                    sampler=sampler, steps=steps,
                )
            del out
        dt = time.perf_counter() - t0
        alloc = _gib(torch.cuda.max_memory_allocated(device))
        reserved = _gib(torch.cuda.max_memory_reserved(device))
        under_cap = reserved <= max_reserved_gib + 1e-6
        row.update({
            "ok": bool(under_cap),
            "oom": False,
            "high_reserved": not under_cap,
            "seconds": round(dt, 2),
            "alloc_GiB": round(alloc, 2),
            "reserved_GiB": round(reserved, 2),
        })
        if fwd_ms:
            row["fwd_ms"] = [round(x, 1) for x in fwd_ms]
            row["fwd_ms_last"] = round(fwd_ms[-1], 1)
        tag = "OK" if under_cap else "HIGH_RESERVED"
        extra = ""
        if fwd_ms:
            extra = f" fwd_ms={fwd_ms[-1]:.0f}"
        print(
            f"TRIAL {stage} chunk={chunk} n_items={n_items} "
            f"n_launches={row['n_launches']} repeats={n_repeats} "
            f"{'FAST ' if fast_pack is not None else ''}{tag} "
            f"alloc={alloc:.2f}GiB reserved={reserved:.2f}GiB "
            f"cap={max_reserved_gib:.2f}GiB {dt:.1f}s{extra}",
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
            "high_reserved": False,
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
    start: int,
    banned: Set[int],
    max_reserved_gib: float,
    trials: List[Dict[str, Any]],
    n_repeats: int = 1,
    fast_pack: Optional[Dict[str, Any]] = None,
) -> int:
    def run(chunk: int) -> Dict[str, Any]:
        if int(chunk) in banned:
            print(f"TRIAL {stage} chunk={chunk} SKIP banned", flush=True)
            row = {
                "stage": stage, "chunk": int(chunk), "ok": False, "oom": False,
                "high_reserved": True, "n_items": int(n_items),
                "n_launches": _n_launches(n_items, chunk), "banned": True,
            }
            trials.append(row)
            return row
        row = _trial(
            stage=stage, model=model, past=past, future_coarse_2d=future_coarse_2d,
            chunk=chunk, sampler=sampler, steps=steps, n_items=n_items,
            device=device, max_reserved_gib=max_reserved_gib,
            n_repeats=n_repeats, fast_pack=fast_pack,
        )
        trials.append(row)
        return row

    hi = min(int(hi_excl), int(n_items) + 1)
    lo = max(1, int(lo))
    if hi <= lo:
        raise RuntimeError(f"{stage}: empty search range lo={lo} hi_excl={hi}")
    start = min(max(lo, int(start)), hi - 1)

    first = run(start)
    if first["ok"]:
        best = start
        cand = start * 2
        while cand < hi:
            row = run(cand)
            if row["ok"]:
                best = cand
                nxt = cand * 2
                if nxt >= hi:
                    break
                cand = nxt
            else:
                hi = cand
                break
        while hi - best > 256:
            mid = (best + hi) // 2
            row = run(mid)
            if row["ok"]:
                best = mid
            else:
                hi = mid
        return best

    print(
        f"{stage} start={start} failed reserved/OOM; searching down toward lo={lo}",
        flush=True,
    )
    best = 0
    cand = start // 2
    floor_fail = lo
    while cand >= lo:
        row = run(cand)
        if row["ok"]:
            best = cand
            break
        floor_fail = cand
        cand //= 2
    if best == 0 and lo < start and lo not in banned:
        row = run(lo)
        if row["ok"]:
            best = lo
    if best == 0:
        raise RuntimeError(
            f"{stage} no chunk with reserved<={max_reserved_gib:.2f}GiB "
            f"at or above lo={lo} (start={start} and downward failed, "
            f"last_fail={floor_fail})"
        )
    fail_hi = start if not first["ok"] else hi
    while fail_hi - best > 256:
        mid = (best + fail_hi) // 2
        row = run(mid)
        if row["ok"]:
            best = mid
        else:
            fail_hi = mid
    return best


def _prev_ok_notch(trials: List[Dict[str, Any]], stage: str, best: int, lo: int) -> int:
    oks = [
        int(r["chunk"]) for r in trials
        if r.get("stage") == stage and r.get("ok") and int(r.get("chunk", 0)) < best
        and not r.get("banned")
    ]
    if oks:
        return max(oks)
    return max(lo, best // 2)


def _load_eval_pair(state: PipelineState, device: torch.device):
    subset_id = state.subset_id or state.dataset
    n_iv = len(state.variate_indices)
    source = str(state.extra.get("eval_source_checkpoint_dir") or state.checkpoint_dir)
    ft_guidance = os.path.join(source, f"{subset_id}_itransformer_finetuned.pt")
    if state.needs_guidance and not os.path.isfile(ft_guidance):
        raise FileNotFoundError(f"missing guidance ckpt: {ft_guidance}")
    ds_lb, ds_hz = dataset_window_lengths(state, state.dataset)
    guidance = None
    if state.needs_guidance:
        guidance = load_wrapped_guidance(
            state, ft_guidance, n_iv, device,
            guidance_type=state.guidance_type,
            dataset_lookback=ds_lb, dataset_horizon=ds_hz,
        )
    phase = StagedEvalPhase(phase="staged_eval")
    t0 = time.perf_counter()
    coarse = phase._load_model(state, "coarse", guidance, n_iv, device)
    fine = phase._load_model(state, "patch_refine", guidance, n_iv, device)
    print(f"loaded coarse+fine ckpts in {time.perf_counter() - t0:.1f}s from {source}", flush=True)
    inner = getattr(fine.noise_predictor, "_orig_mod", fine.noise_predictor)
    if not bool(fine.config.cache_cond_kv) or not bool(getattr(inner, "cache_cond_kv", False)):
        raise RuntimeError("fine cache_cond_kv did not land on FactorizedDiT")
    if not bool(fine.config.patch_refine_unique_segments):
        raise RuntimeError("fine unique_segments is false")
    return coarse, fine, ds_lb, ds_hz


def _load_real_past(state: PipelineState, device: torch.device) -> torch.Tensor:
    subset_meta = state.data_subset_resolved or {}
    train_stride = int(subset_meta.get("train_stride", state.window_stride))
    test_stride = int(subset_meta.get("test_stride", 1))
    _tr, _va, test_ds, _stats = load_dataset(
        state, state.dataset,
        state.variate_indices,
        stride=train_stride,
        test_stride=test_stride,
        ordinal_tie_atol=float(state.ordinal_tie_atol),
        use_ordinal_window_norm=state.use_ordinal_window_norm,
    )
    if len(test_ds) < 1:
        raise RuntimeError("empty test set; cannot time a real window")
    past, _future = test_ds[0]
    print(f"real test window 0/{len(test_ds)} past={tuple(past.shape)}", flush=True)
    return past.unsqueeze(0).to(device), len(test_ds)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--ckpt-dir", required=True,
                   help="Train run checkpoint root (eval-source-checkpoint-dir)")
    p.add_argument("--n-variates", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--lo", type=int, default=512)
    p.add_argument("--hi", type=int, default=None,
                   help="Exclusive upper bound. Default n_fine+1.")
    p.add_argument("--start", type=int, default=None,
                   help="First trial chunk (default max(lo, min(2048, hi/2)))")
    p.add_argument("--sampler", default="quad_t")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--headroom-gib", type=float, default=5.0)
    p.add_argument("--max-reserved-gib", type=float, default=None,
                   help="Override reserved cap. Default total - headroom.")
    p.add_argument("--min-ok-chunk", type=int, default=1024,
                   help="If the winning chunk is below this, exit 2 (do not shard).")
    p.add_argument("--confirm-repeats", type=int, default=2,
                   help="Generates/fwds on the winning chunk before accept (compile warm).")
    p.add_argument("--search-repeats", type=int, default=1,
                   help="Repeats during binary search (winner still uses --confirm-repeats).")
    p.add_argument("--step-down-scrape-gib", type=float, default=1.0,
                   help="If max-pass reserved is within this many GiB of the cap, "
                        "step down one search notch before confirm.")
    p.add_argument("--fast-chunk-fwd", action="store_true",
                   help="Fine search: one/two DiT fwds of size C, not full unique-seg generate.")
    p.add_argument("--no-full-pack-hold", action="store_true",
                   help="With --fast-chunk-fwd, do not keep n_fine xt/aux/canvas buffers.")
    p.add_argument("--skip-full-window", action="store_true",
                   help="Do not time a full det+prob window; estimate from fwd_ms.")
    p.add_argument(
        "--det-only",
        action="store_true",
        help="Anchor/det generate only (no prob). Reserved cap is applied to a "
        "full det window, not the fast-fwd scrape. Use --n-samples 1.",
    )
    p.add_argument("--no-compile", action="store_true")
    args = p.parse_args()

    ckpt_dir = Path(args.ckpt_dir).expanduser()
    if not ckpt_dir.is_absolute():
        ckpt_dir = (REPO / ckpt_dir).resolve()
    if not ckpt_dir.is_dir():
        raise SystemExit(f"ckpt dir missing: {ckpt_dir}")

    n_samples = int(args.n_samples)
    if n_samples < 1 or int(args.lo) < 1:
        raise SystemExit("n_samples and --lo must be >= 1")
    if args.det_only and n_samples != 1:
        raise SystemExit("--det-only requires --n-samples 1 (anchor pack is V x n_col0)")
    if args.det_only and args.skip_full_window:
        raise SystemExit("--det-only cannot combine with --skip-full-window")
    if int(args.confirm_repeats) < 1 or int(args.search_repeats) < 1:
        raise SystemExit("repeats must be >= 1")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    device = torch.device("cuda")
    gpu = torch.cuda.get_device_name(0)
    total = _gib(torch.cuda.get_device_properties(0).total_memory)
    max_reserved = (
        float(args.max_reserved_gib)
        if args.max_reserved_gib is not None
        else total - float(args.headroom_gib)
    )
    print(
        f"GPU={gpu} total={total:.2f}GiB reserved_cap={max_reserved:.2f}GiB "
        f"(headroom={args.headroom_gib:.2f}GiB)",
        flush=True,
    )
    print(
        f"ckpt={ckpt_dir} dataset={args.dataset} n_samples={n_samples} "
        f"cache_cond_kv=True unique_seg det_only={bool(args.det_only)} "
        f"fast_chunk_fwd={bool(args.fast_chunk_fwd)} "
        f"lo={args.lo} hi_excl={args.hi} sampler={args.sampler} steps={args.steps} "
        f"search_repeats={args.search_repeats} confirm_repeats={args.confirm_repeats} "
        f"banned={sorted(BANNED_CHUNKS)}",
        flush=True,
    )

    cfg = load_experiment_config(args.config, {"dataset": args.dataset})
    state = PipelineState.from_config(cfg)
    state.dataset = args.dataset
    state.wandb_enabled = False
    state.smoke_test = False
    state.extra["eval_source_checkpoint_dir"] = str(ckpt_dir)
    state.checkpoint_dir = str(ckpt_dir)
    if args.no_compile:
        state.torch_compile = False
    if not bool(getattr(state, "patch_refine_unique_segments", False)):
        raise RuntimeError("probe requires patch_refine_unique_segments=true")
    resolve_pipeline_data_subset(state)
    n_variates = int(state.n_variates)
    expect = DATASET_V.get(args.dataset)
    if args.n_variates is not None and int(args.n_variates) != n_variates:
        raise SystemExit(
            f"--n-variates {args.n_variates} != resolved {n_variates}"
        )
    if expect is not None and n_variates != expect:
        print(f"WARN resolved V={n_variates} vs table {expect}", flush=True)

    compile_on = bool(state.torch_compile)
    print(
        f"config={args.config} subset={state.subset_id} V={n_variates} "
        f"lookback={state.lookback_length} horizon={state.forecast_length} "
        f"patch={state.patch_refine_patch_width}x{state.patch_refine_patch_height} "
        f"stride={state.patch_refine_col_stride} torch_compile={compile_on}",
        flush=True,
    )

    coarse, fine, lookback, horizon = _load_eval_pair(state, device)
    _log_mem("after_ckpt_load", device)
    past, n_test = _load_real_past(state, device)
    past_exp = past.repeat_interleave(n_samples, dim=0)
    w_fut = int(fine._repr_forecast_width(int(fine.config.forecast_length)))
    n_col0 = len(primary_stride_col0s(
        w_fut,
        int(fine.config.patch_refine_patch_width),
        int(fine.config.patch_refine_col_stride),
    ))
    n_fine = n_col0 * n_variates * n_samples
    n_coarse = n_variates * n_samples
    hi_excl = int(args.hi) if args.hi is not None else n_fine + 1
    start = args.start
    if start is None:
        start = max(int(args.lo), min(2048, max(int(args.lo), hi_excl // 4)))
    print(
        f"geometry W_fut={w_fut} n_col0={n_col0} n_test={n_test} "
        f"fine_n_items={n_fine} (B={n_samples} x V={n_variates} x n_col0) "
        f"coarse_n_items={n_coarse} search_start={start}",
        flush=True,
    )

    trials: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "gpu": gpu,
        "total_GiB": round(total, 2),
        "reserved_cap_GiB": round(max_reserved, 2),
        "headroom_gib": float(args.headroom_gib),
        "weights": str(ckpt_dir),
        "dataset": args.dataset,
        "n_variates": int(n_variates),
        "n_samples": int(n_samples),
        "lookback": int(lookback),
        "horizon": int(horizon),
        "n_col0": int(n_col0),
        "n_test": int(n_test),
        "fine_n_items": int(n_fine),
        "coarse_n_items": int(n_coarse),
        "cache_cond_kv": True,
        "torch_compile": compile_on,
        "sampler": args.sampler,
        "steps": int(args.steps),
        "lo": int(args.lo),
        "hi_excl": int(hi_excl),
        "start": int(start),
        "fast_chunk_fwd": bool(args.fast_chunk_fwd),
        "det_only": bool(args.det_only),
        "confirm_repeats": int(args.confirm_repeats),
        "banned": sorted(BANNED_CHUNKS),
    }

    dummy_coarse = torch.rand(
        n_samples, n_variates,
        int(fine.config.coarse_image_height),
        w_fut, device=device,
    )

    print(
        f"\n=== COARSE unique-seg generate (B={n_samples}, V={n_variates}) ===",
        flush=True,
    )
    coarse_max = _binary_search(
        stage="coarse", model=coarse, past=past_exp, future_coarse_2d=None,
        sampler=args.sampler, steps=args.steps, n_items=n_coarse,
        device=device, lo=min(int(args.lo), n_coarse),
        hi_excl=max(hi_excl, n_coarse + 1),
        start=min(start, n_coarse), banned=BANNED_CHUNKS,
        max_reserved_gib=max_reserved, trials=trials,
        n_repeats=int(args.search_repeats),
    )
    print(f"COARSE_MAX_CHUNK {coarse_max}", flush=True)

    fine_pack = None
    hold = None
    if args.fast_chunk_fwd:
        print(
            "\n=== FINE FAST chunk fwd (C-item DiT, not full unique-seg generate) ===",
            flush=True,
        )
        fine_pack = _build_fine_pack(fine, past_exp, dummy_coarse)
        if not args.no_full_pack_hold:
            hold = _alloc_full_pack_hold(fine_pack, n_fine)
            _log_mem("after_full_pack_hold", device)
    else:
        print(
            f"\n=== FINE unique-seg generate (B={n_samples}, V={n_variates}, cache_cond_kv=True) ===",
            flush=True,
        )

    fine_max = _binary_search(
        stage="fine", model=fine, past=past_exp, future_coarse_2d=dummy_coarse,
        sampler=args.sampler, steps=args.steps, n_items=n_fine,
        device=device, lo=args.lo, hi_excl=hi_excl, start=start,
        banned=BANNED_CHUNKS, max_reserved_gib=max_reserved, trials=trials,
        n_repeats=int(args.search_repeats), fast_pack=fine_pack,
    )
    print(f"FINE_MAX_CHUNK {fine_max}", flush=True)

    fine_reserved = None
    for row in reversed(trials):
        if row.get("stage") == "fine" and int(row.get("chunk", -1)) == int(fine_max) and row.get("ok"):
            fine_reserved = row.get("reserved_GiB")
            break

    shard_chunk = int(fine_max)
    scrape = float(args.step_down_scrape_gib)
    if fine_reserved is not None and float(fine_reserved) > max_reserved - scrape:
        stepped = _prev_ok_notch(trials, "fine", shard_chunk, int(args.lo))
        if stepped < shard_chunk and stepped not in BANNED_CHUNKS:
            print(
                f"STEP_DOWN {shard_chunk} reserved={fine_reserved}GiB within "
                f"{scrape:.2f}GiB of cap {max_reserved:.2f}; trying {stepped}",
                flush=True,
            )
            shard_chunk = int(stepped)

    print(
        f"\n=== CONFIRM fine chunk={shard_chunk} repeats={args.confirm_repeats} "
        f"(compile warm, same process) ===",
        flush=True,
    )
    confirm_fast = fine_pack
    confirm = _trial(
        stage="fine_confirm", model=fine, past=past_exp,
        future_coarse_2d=dummy_coarse, chunk=shard_chunk,
        sampler=args.sampler, steps=args.steps, n_items=n_fine,
        device=device, max_reserved_gib=max_reserved,
        n_repeats=int(args.confirm_repeats),
        fast_pack=confirm_fast,
    )
    trials.append(confirm)
    if not confirm["ok"]:
        stepped = _prev_ok_notch(trials, "fine", shard_chunk, int(args.lo))
        if stepped >= int(args.min_ok_chunk) and stepped != shard_chunk and stepped not in BANNED_CHUNKS:
            print(f"CONFIRM failed; retrying stepped-down chunk={stepped}", flush=True)
            shard_chunk = int(stepped)
            confirm = _trial(
                stage="fine_confirm", model=fine, past=past_exp,
                future_coarse_2d=dummy_coarse, chunk=shard_chunk,
                sampler=args.sampler, steps=args.steps, n_items=n_fine,
                device=device, max_reserved_gib=max_reserved,
                n_repeats=int(args.confirm_repeats),
                fast_pack=confirm_fast,
            )
            trials.append(confirm)
    if not confirm["ok"]:
        print(
            f"STOP_NO_HEADROOM confirm failed for chunk={shard_chunk} "
            f"reserved={confirm.get('reserved_GiB')} cap={max_reserved:.2f}",
            flush=True,
        )
        del coarse, fine, past, past_exp, dummy_coarse, hold
        gc.collect()
        torch.cuda.empty_cache()
        raise SystemExit(2)

    confirm_reserved = confirm.get("reserved_GiB")
    fwd_ms_last = confirm.get("fwd_ms_last")
    n_launches_f = _n_launches(n_fine, shard_chunk)
    n_launches_c = _n_launches(n_coarse, max(int(coarse_max), n_coarse))
    n_launches_det_f = _n_launches(n_col0 * n_variates, shard_chunk)
    n_launches_det_c = _n_launches(n_variates, max(int(coarse_max), n_coarse))

    win_s = None
    det_s = None
    prob_s = None
    win_reserved = None
    win_alloc = None
    est_s = None
    if fwd_ms_last is not None:
        fwd_s = float(fwd_ms_last) / 1000.0
        est_prob = (n_launches_f + n_launches_c) * int(args.steps) * fwd_s
        est_det = (n_launches_det_f + n_launches_det_c) * int(args.steps) * fwd_s
        est_s = est_det + est_prob
        print(
            f"EST_WINDOW_SECONDS {est_s:.2f} det~{est_det:.2f}s prob~{est_prob:.2f}s "
            f"fwd_ms={fwd_ms_last} fine_launches={n_launches_f} steps={args.steps}",
            flush=True,
        )

    if not args.skip_full_window:
        if args.det_only:
            print(
                "\n=== TIME ONE REAL DET WINDOW (no prob) — reserved cap on this generate ===",
                flush=True,
            )
            if hold is not None:
                del hold
                hold = None
                gc.collect()
                torch.cuda.empty_cache()
                print("released full_pack_hold before det window", flush=True)
            # Fast-fwd scrape overestimates C; pick the largest fine OK with
            # >=8 GiB reserved slack vs the cap, else half the scrape max.
            slack = 8.0
            safe = 0
            for row in trials:
                if (
                    row.get("stage") == "fine"
                    and row.get("ok")
                    and not row.get("banned")
                    and row.get("reserved_GiB") is not None
                    and float(row["reserved_GiB"]) <= max_reserved - slack
                ):
                    safe = max(safe, int(row["chunk"]))
            if safe >= int(args.min_ok_chunk) and safe < shard_chunk:
                print(
                    f"DET_WINDOW_START {safe} (fast max {shard_chunk} had "
                    f"<{slack:.0f}GiB reserved slack)",
                    flush=True,
                )
                shard_chunk = int(safe)
            elif shard_chunk > int(args.lo) * 2:
                conservative = max(int(args.min_ok_chunk), shard_chunk // 2)
                if conservative < shard_chunk and conservative not in BANNED_CHUNKS:
                    print(
                        f"DET_WINDOW_START {conservative} (half of scrape max {shard_chunk})",
                        flush=True,
                    )
                    shard_chunk = int(conservative)
            coarse_chunk = max(int(coarse_max), n_coarse)
            while True:
                win = _time_det_window(
                    coarse=coarse, fine=fine, past=past, chunk=shard_chunk,
                    coarse_chunk=coarse_chunk, state=state, steps=args.steps,
                    device=device,
                )
                det_s = float(win["seconds"])
                win_s = det_s
                prob_s = 0.0
                if win.get("oom") or not win.get("ok"):
                    print(
                        f"WINDOW OOM/FAIL chunk={shard_chunk} {det_s:.1f}s "
                        f"{win.get('error', '')}",
                        flush=True,
                    )
                    win_reserved = float("inf")
                    win_alloc = None
                else:
                    win_reserved = float(win["reserved_GiB"])
                    win_alloc = float(win["alloc_GiB"])
                    print(
                        f"WINDOW_SECONDS {win_s:.2f} det={det_s:.2f}s prob=0.00s "
                        f"alloc={win_alloc:.2f}GiB reserved={win_reserved:.2f}GiB "
                        f"chunk={shard_chunk} cap={max_reserved:.2f}GiB",
                        flush=True,
                    )
                    print(f"WINDOW_DET_SECONDS {det_s:.2f}", flush=True)
                    if win_reserved <= max_reserved + 1e-6:
                        break
                stepped = _prev_ok_notch(trials, "fine", shard_chunk, int(args.lo))
                if stepped >= shard_chunk:
                    stepped = max(int(args.min_ok_chunk), shard_chunk // 2)
                if (
                    stepped < shard_chunk
                    and stepped >= int(args.min_ok_chunk)
                    and stepped not in BANNED_CHUNKS
                ):
                    print(
                        f"DET_WINDOW_OVER_CAP chunk={shard_chunk} -> {stepped}",
                        flush=True,
                    )
                    shard_chunk = int(stepped)
                    continue
                print(
                    f"STOP_NO_HEADROOM det window failed at chunk={shard_chunk} "
                    f"cap={max_reserved:.2f}GiB",
                    flush=True,
                )
                del coarse, fine, past, past_exp, dummy_coarse, hold
                gc.collect()
                torch.cuda.empty_cache()
                raise SystemExit(2)
        else:
            print("\n=== TIME ONE REAL WINDOW at fine chunk (det + prob n_samples) ===", flush=True)
            coarse.config.unet_max_chunk_size = max(int(coarse_max), n_coarse)
            fine.config.unet_max_chunk_size = shard_chunk
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            det_kwargs = _staged_det_gen_kwargs(state, args.steps)
            t_win = time.perf_counter()
            with torch.no_grad():
                coarse_det = coarse.generate(past, **det_kwargs)
                fine.generate(past, future_coarse_2d=coarse_det["future_2d_coarse"], **det_kwargs)
                torch.cuda.synchronize()
                det_s = time.perf_counter() - t_win
                t_prob = time.perf_counter()
                coarse_sample = _run_generate(
                    coarse, past_exp, future_coarse_2d=None,
                    sampler=args.sampler, steps=args.steps,
                )
                _run_generate(
                    fine, past_exp, future_coarse_2d=coarse_sample["future_2d_coarse"],
                    sampler=args.sampler, steps=args.steps,
                )
                prob_s = time.perf_counter() - t_prob
            win_s = time.perf_counter() - t_win
            win_reserved = _gib(torch.cuda.max_memory_reserved(device))
            win_alloc = _gib(torch.cuda.max_memory_allocated(device))
            print(
                f"WINDOW_SECONDS {win_s:.2f} det={det_s:.2f}s prob={prob_s:.2f}s "
                f"alloc={win_alloc:.2f}GiB reserved={win_reserved:.2f}GiB "
                f"chunk={shard_chunk}",
                flush=True,
            )
            if win_reserved > max_reserved + 1e-6:
                print(
                    f"WARN timed window reserved={win_reserved:.2f}GiB exceeds cap "
                    f"{max_reserved:.2f}GiB — shards still risky",
                    flush=True,
                )

    summary.update({
        "coarse_max_chunk": int(coarse_max),
        "fine_max_chunk": int(fine_max),
        "shard_unet_max_chunk_size": int(shard_chunk),
        "fine_reserved_GiB": fine_reserved,
        "confirm_reserved_GiB": confirm_reserved,
        "confirm_repeats": int(args.confirm_repeats),
        "window_seconds": None if win_s is None else round(win_s, 2),
        "window_det_seconds": None if det_s is None else round(det_s, 2),
        "window_prob_seconds": None if prob_s is None else round(prob_s, 2),
        "window_reserved_GiB": (
            None if win_reserved is None or not math.isfinite(float(win_reserved))
            else round(float(win_reserved), 2)
        ),
        "window_alloc_GiB": None if win_alloc is None else round(win_alloc, 2),
        "est_window_seconds": None if est_s is None else round(est_s, 2),
        "fwd_ms_last": fwd_ms_last,
        "trials": trials,
    })
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "trials"}, indent=2), flush=True)
    print(f"SHARD_UNET_MAX_CHUNK_SIZE {shard_chunk}", flush=True)
    print(f"FINE_RESERVED_GIB {fine_reserved}", flush=True)
    print(f"CONFIRM_RESERVED_GIB {confirm_reserved}", flush=True)
    if win_reserved is not None and math.isfinite(float(win_reserved)):
        print(f"WINDOW_RESERVED_GIB {win_reserved:.2f}", flush=True)
    if det_s is not None:
        print(f"WINDOW_DET_SECONDS {det_s:.2f}", flush=True)

    if shard_chunk < int(args.min_ok_chunk):
        print(
            f"STOP_NO_HEADROOM winning fine chunk {shard_chunk} < min-ok {args.min_ok_chunk}",
            flush=True,
        )
        del coarse, fine, past, past_exp, dummy_coarse, hold
        gc.collect()
        torch.cuda.empty_cache()
        raise SystemExit(2)

    del coarse, fine, past, past_exp, dummy_coarse, hold
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
