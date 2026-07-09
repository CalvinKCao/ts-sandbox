#!/usr/bin/env python3
"""Probe true L40S max micro-batch for staged diffusion train steps.

Unlike training AutoBS, this does **not** cap the search by ``512 // n_variates``.
It binary-searches until OOM for each dataset subset × geometry.

**Batch item = one variate** (univariate channel of one window). Reported sizes
are ``U = B_windows * C`` where ``C`` is the subset variate count. The train
step still uses real multivariate windows ``(B, C, T)`` so guidance /
cross-variate context match training; the UNet flattens to ``B*C`` tokens.

Example:
  python utils/probe_diffusion_max_batch.py \\
    --datasets ETTh1,weather,electricity,exchange_rate,traffic \\
    --geometries 96/96,336/720_uncompressed \\
    --stages coarse,fine \\
    --device cuda \\
    --output-csv reports/probe_diffusion_max_batch.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.pipeline.config import load_experiment_config
from models.diffusion_tsf.pipeline.globals_bridge import patch_globals
from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import patch_stage_globals
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.train_multivariate_pipeline import (
    amp_context,
    create_diffusion_model,
    create_patch_guidance_stack,
    load_dataset,
    resolve_pipeline_data_subset,
    wrap_patch_guidance,
    _even_batch_size,
    _probe_step_ok,
)
import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

GEOMETRY_CONFIGS = {
    "96/96": "configs/binary_anchor_ar_patch_decoder_ctx.yaml",
    "336/720_uncompressed": (
        "configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_uncompressed.yaml"
    ),
}

DEFAULT_DATASETS = "ETTh1,weather,electricity,exchange_rate,traffic"


def _cuda_mem_mb() -> Tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    free, total = torch.cuda.mem_get_info()
    return free / (1024**2), total / (1024**2)


def _build_state(config_path: str, dataset: str) -> PipelineState:
    cfg = load_experiment_config(config_path, cli_overrides={"dataset": dataset})
    state = PipelineState.from_config(cfg)
    state.dataset = dataset
    resolve_pipeline_data_subset(state)
    state.subset_id = state.subset_id or dataset
    return state


def _snap_univariate(u: int, n_vars: int, *, floor: int) -> int:
    """Snap u to a valid univariate count: multiple of n_vars, >= floor.

    Clamps up to floor when u < floor, then rounds down to a multiple of n_vars.
    When n_vars is odd, prefers an even U (step down by n_vars once if needed).
    """
    n_vars = max(1, int(n_vars))
    floor = max(n_vars, int(floor))
    u = max(floor, int(u))
    # Round down to multiple of n_vars
    u = (u // n_vars) * n_vars
    if u < floor:
        u = ((floor + n_vars - 1) // n_vars) * n_vars
    # Prefer even univariate counts when n_vars is odd (e.g. 7): step by 2*n_vars
    if u % 2 != 0 and n_vars % 2 == 1:
        alt = u - n_vars
        if alt >= floor and alt % n_vars == 0:
            u = alt
    return max(floor, u)


def _probe_max_univariate(
    *,
    try_window_fn,
    n_vars: int,
    max_candidate: int,
    min_candidate: int = 2,
) -> Tuple[int, int]:
    """Binary-search univariate count U=B*C. Returns (best_U, search_ceiling_U)."""
    n_vars = max(1, int(n_vars))
    min_u = _snap_univariate(max(n_vars, min_candidate), n_vars, floor=n_vars)
    ceiling = _snap_univariate(max_candidate, n_vars, floor=min_u)
    if ceiling < min_u:
        raise ValueError(
            f"max_candidate={max_candidate} too small for n_vars={n_vars} (need >= {min_u})"
        )

    lo = min_u
    hi = ceiling
    best = 0  # nothing confirmed until a probe succeeds
    # For even n_vars, multiples are already even; step by n_vars.
    # For odd n_vars, keep U even by stepping 2*n_vars when possible.
    step = n_vars if n_vars % 2 == 0 else 2 * n_vars

    while lo <= hi:
        mid = _snap_univariate((lo + hi) // 2, n_vars, floor=min_u)
        window_batch = mid // n_vars
        if window_batch < 1:
            hi = mid - step
            continue
        if _probe_step_ok(try_window_fn, window_batch):
            best = mid
            lo = mid + step
        else:
            hi = mid - step

    if best == 0:
        raise RuntimeError(
            f"OOM even at floor batch (window_batch=1, U={min_u}); "
            f"cannot fit on this device for n_vars={n_vars}"
        )
    return best, ceiling


def probe_one(
    *,
    config_path: str,
    geometry: str,
    dataset: str,
    stage: str,
    device: torch.device,
    max_candidate: int,
    safety_frac: float,
) -> Dict[str, Any]:
    state = _build_state(config_path, dataset)
    patch_globals(pipeline_mod, state, honor_dataset_windows=True)
    patch_stage_globals(pipeline_mod, state, stage, honor_dataset_windows=True)

    n_vars = len(state.variate_indices or [])
    if n_vars <= 0:
        raise RuntimeError(f"{dataset}: empty variate_indices after subset resolve")
    lookback = int(state.lookback_length)
    horizon = int(state.forecast_length)
    subset_id = str(state.subset_id)

    train_ds, _, _, _ = load_dataset(
        dataset,
        list(state.variate_indices),
        lookback=lookback,
        horizon=horizon,
        stride=int((state.data_subset_resolved or {}).get("train_stride", state.window_stride)),
        test_stride=1,
        ordinal_tie_atol=float(state.ordinal_tie_atol),
        use_ordinal_window_norm=state.use_ordinal_window_norm,
    )
    sample_past, sample_future = train_ds[0]
    # (C, L) / (C, H) — C is all subset variates; UNet sees B*C univariate tokens
    assert sample_past.shape[0] == n_vars, (sample_past.shape, n_vars)

    free0, total0 = _cuda_mem_mb()
    print(
        f"[probe] {geometry} {dataset}/{subset_id} stage={stage} "
        f"vars={n_vars} lb={lookback} hz={horizon} "
        f"sample=({tuple(sample_past.shape)},{tuple(sample_future.shape)}) "
        f"batch_item=1 variate (U=B*{n_vars}) "
        f"cuda_free={free0:.0f}/{total0:.0f} MiB max_candidate_U={max_candidate}",
        flush=True,
    )

    def _try_windows(window_batch: int) -> bool:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        stack = create_patch_guidance_stack(
            n_vars, in_len=lookback, out_len=horizon
        ).to(device)
        guidance = wrap_patch_guidance(stack)
        model = create_diffusion_model(
            guidance_model=guidance,
            n_variates=n_vars,
            lookback=lookback,
            horizon=horizon,
            diffusion_stage=stage,
        ).to(device)
        try:
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            past = sample_past.unsqueeze(0).repeat(window_batch, 1, 1).to(device)
            future = sample_future.unsqueeze(0).repeat(window_batch, 1, 1).to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context():
                loss = model.get_loss(past, future)
            loss.backward()
            optimizer.step()
            return True
        finally:
            del model, guidance, stack
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    t0 = time.perf_counter()
    best_u, ceiling_u = _probe_max_univariate(
        try_window_fn=_try_windows,
        n_vars=n_vars,
        max_candidate=max_candidate,
    )
    safe_u = _snap_univariate(int(best_u * safety_frac), n_vars, floor=n_vars)
    window_batch = best_u // n_vars
    safe_window_batch = safe_u // n_vars
    hit_cap = best_u >= ceiling_u
    elapsed = time.perf_counter() - t0
    free1, _ = _cuda_mem_mb()

    peak_mb = None
    if torch.cuda.is_available() and window_batch >= 1:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ok = _probe_step_ok(_try_windows, window_batch)
        if ok:
            peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

    row = {
        "geometry": geometry,
        "config": config_path,
        "dataset": dataset,
        "subset_id": subset_id,
        "n_variates": n_vars,
        "lookback": lookback,
        "horizon": horizon,
        "stage": stage,
        "search_ceiling": ceiling_u,
        "best_fit": best_u,
        "window_batch": window_batch,
        "safe_80pct": safe_u,
        "safe_window_batch": safe_window_batch,
        "hit_search_cap": int(hit_cap),
        "peak_alloc_mib_at_best": None if peak_mb is None else round(peak_mb, 1),
        "cuda_total_mib": round(total0, 1),
        "elapsed_s": round(elapsed, 1),
        "batch_item": f"1 variate; U=B*{n_vars} (train step uses (B,{n_vars},T))",
    }
    print(
        f"[result] {geometry} {subset_id} {stage}: "
        f"best_fit_U={best_u} window_batch={window_batch} "
        f"safe_U={safe_u} safe_windows={safe_window_batch} "
        f"ceiling_U={ceiling_u} hit_cap={hit_cap} "
        f"peak_mib={peak_mb and round(peak_mb, 1)} "
        f"elapsed={elapsed:.1f}s free_after={free1:.0f}MiB",
        flush=True,
    )
    return row


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", default=DEFAULT_DATASETS)
    p.add_argument(
        "--geometries",
        default="96/96,336/720_uncompressed",
        help=f"Comma list from {sorted(GEOMETRY_CONFIGS)}",
    )
    p.add_argument("--stages", default="coarse,fine")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--max-candidate",
        type=int,
        default=2048,
        help="Upper search bound in univariate units U=B*C. Raise if a run hits the cap.",
    )
    p.add_argument("--safety-frac", type=float, default=0.8)
    p.add_argument("--output-csv", type=Path, default=REPO_ROOT / "reports" / "probe_diffusion_max_batch.csv")
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="One dataset (ETTh1), one geometry (96/96), coarse only, max-candidate_U=32",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    geometries = [g.strip() for g in args.geometries.split(",") if g.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    max_candidate = int(args.max_candidate)

    if args.smoke_test:
        geometries = ["96/96"]
        datasets = ["ETTh1"]
        stages = ["coarse"]
        max_candidate = min(max_candidate, 32)

    for g in geometries:
        if g not in GEOMETRY_CONFIGS:
            raise ValueError(f"unknown geometry {g!r}; choose from {sorted(GEOMETRY_CONFIGS)}")

    out = args.output_csv.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "geometry",
        "config",
        "dataset",
        "subset_id",
        "n_variates",
        "lookback",
        "horizon",
        "stage",
        "search_ceiling",
        "best_fit",
        "window_batch",
        "safe_80pct",
        "safe_window_batch",
        "hit_search_cap",
        "peak_alloc_mib_at_best",
        "cuda_total_mib",
        "elapsed_s",
        "batch_item",
        "error",
    ]
    n_ok = 0
    n_fail = 0
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        f.flush()
        for geometry in geometries:
            config_path = GEOMETRY_CONFIGS[geometry]
            for dataset in datasets:
                for stage in stages:
                    try:
                        row = probe_one(
                            config_path=config_path,
                            geometry=geometry,
                            dataset=dataset,
                            stage=stage,
                            device=device,
                            max_candidate=max_candidate,
                            safety_frac=float(args.safety_frac),
                        )
                        row["error"] = ""
                        n_ok += 1
                    except Exception as exc:
                        n_fail += 1
                        print(
                            f"[error] {geometry} {dataset} {stage}: {exc}",
                            flush=True,
                        )
                        row = {
                            "geometry": geometry,
                            "config": config_path,
                            "dataset": dataset,
                            "subset_id": "",
                            "n_variates": "",
                            "lookback": "",
                            "horizon": "",
                            "stage": stage,
                            "search_ceiling": "",
                            "best_fit": "",
                            "window_batch": "",
                            "safe_80pct": "",
                            "safe_window_batch": "",
                            "hit_search_cap": "",
                            "peak_alloc_mib_at_best": "",
                            "cuda_total_mib": "",
                            "elapsed_s": "",
                            "batch_item": "1 variate; U=B*C",
                            "error": str(exc),
                        }
                    writer.writerow(row)
                    f.flush()
    print(
        f"[summary] wrote {n_ok} ok + {n_fail} fail rows -> {out}",
        flush=True,
    )
    if n_ok == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
