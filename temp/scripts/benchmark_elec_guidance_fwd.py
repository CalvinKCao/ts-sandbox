#!/usr/bin/env python3
"""Timed fwd+bwd train steps: patch_decoder vs iTransformer vs no cross-attn.

Electricity V=321, lb336/hz96 canvas128 p64x6. Random weights (no ckpt I/O).
Uses univariate micro-batch U (default 217) with include_anchor=True, matching
the lr10 L40S probe plan.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf import train_multivariate_pipeline as pipeline_mod
from models.diffusion_tsf.guidance import iTransformerGuidance
from models.diffusion_tsf.pipeline.config import load_experiment_config
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline.train.checkpointing import amp_context
from models.diffusion_tsf.pipeline.train.univariate_microbatch import (
    dataloader_windows_for_univariate_rows,
)

DEFAULT_CONFIG = (
    "configs/binary_window_norm_patch_refine_canvas128_p64x6_electricity_allv_msdefault_fixed.yaml"
)
MODES = ("patch_decoder", "itransformer", "no_xattn")
STAGES = ("coarse", "patch_refine")


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def git_head() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            )
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def n_past_patches(lookback: int, patch_size: int) -> int:
    pad = (patch_size - lookback % patch_size) % patch_size
    return (lookback + pad) // patch_size


def median_ms(samples: list[float]) -> float:
    return 1000.0 * statistics.median(samples)


def peak_mib(device: torch.device) -> float:
    return torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)


@dataclass
class BenchInputs:
    past: torch.Tensor
    future: torch.Tensor
    row_index: torch.Tensor
    patch_col0: Optional[torch.Tensor]
    n_windows: int
    n_rows: int


def make_inputs(
    *,
    u_rows: int,
    n_variates: int,
    lookback: int,
    horizon: int,
    overlap: int,
    stage: str,
    patch_w: int,
    unique_segments: bool,
    device: torch.device,
) -> BenchInputs:
    fut_w = int(horizon) + int(overlap)
    n_win = dataloader_windows_for_univariate_rows(u_rows, n_variates)
    past = torch.randn(n_win, n_variates, lookback, device=device)
    future = torch.randn(n_win, n_variates, fut_w, device=device)
    row_index = torch.arange(int(u_rows), device=device)
    patch_col0 = None
    if stage == "patch_refine" and unique_segments:
        max_c0 = max(0, fut_w - int(patch_w))
        patch_col0 = torch.zeros(n_win, device=device, dtype=torch.long)
        if max_c0 > 0:
            patch_col0 = torch.randint(0, max_c0 + 1, (n_win,), device=device)
    return BenchInputs(
        past=past,
        future=future,
        row_index=row_index,
        patch_col0=patch_col0,
        n_windows=n_win,
        n_rows=int(u_rows),
    )


def build_guidance(
    state: PipelineState,
    mode: str,
    n_variates: int,
    lookback: int,
    horizon: int,
    device: torch.device,
) -> Optional[nn.Module]:
    if mode == "no_xattn":
        return None
    if mode == "patch_decoder":
        stack = pipeline_mod.create_patch_guidance_stack(state, n_variates).to(device)
        stack.eval()
        return pipeline_mod.wrap_patch_guidance(state, stack)
    if mode == "itransformer":
        itrans = pipeline_mod.create_itransformer(
            state, seq_len=lookback, pred_len=horizon, num_vars=n_variates,
        ).to(device)
        itrans.eval()
        return iTransformerGuidance(itrans, seq_len=lookback, pred_len=horizon)
    raise ValueError(f"unknown mode {mode!r}")


def build_model(
    base_state: PipelineState,
    *,
    mode: str,
    n_variates: int,
    lookback: int,
    horizon: int,
    stage: str,
    device: torch.device,
    torch_compile: bool,
) -> nn.Module:
    state = copy.deepcopy(base_state)
    state.n_variates = int(n_variates)
    state.dataset = "electricity"
    state.smoke_test = False
    state.torch_compile = bool(torch_compile)
    if stage == "patch_refine":
        state.image_height = int(state.patch_refine_patch_height)
    else:
        state.image_height = int(state.coarse_image_height)
    if mode == "no_xattn":
        state.disable_cross_attention = True
    elif mode == "itransformer":
        state.disable_cross_attention = False
        state.guidance_type = "itransformer"
    else:
        state.disable_cross_attention = False
        state.guidance_type = "patch_decoder"

    guidance = build_guidance(state, mode, n_variates, lookback, horizon, device)
    model = pipeline_mod.create_diffusion_model(
        state,
        n_variates=n_variates,
        lookback=lookback,
        horizon=horizon,
        guidance_model=guidance,
        diffusion_stage=stage,
        guidance_type=state.guidance_type,
    ).to(device)
    model.train()
    return model


def expected_ctx_tokens(mode: str, n_variates: int, n_past: int) -> int:
    if mode == "no_xattn":
        return 0
    if mode == "itransformer":
        return int(n_variates)
    return int(n_variates) * int(n_past)


def run_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: BenchInputs,
    use_amp: bool,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    with amp_context(use_amp):
        loss = model.get_loss(
            inputs.past,
            inputs.future,
            patch_col0=inputs.patch_col0,
            include_anchor=True,
            univariate_row_index=inputs.row_index,
        )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


def time_mode(
    *,
    base_state: PipelineState,
    mode: str,
    stage: str,
    n_variates: int,
    lookback: int,
    horizon: int,
    overlap: int,
    u_rows: int,
    device: torch.device,
    warmup: int,
    timed: int,
    use_amp: bool,
    torch_compile: bool,
) -> dict[str, Any]:
    patch_w = int(base_state.patch_refine_patch_width)
    unique = bool(getattr(base_state, "patch_refine_unique_segments", False))
    n_past = n_past_patches(lookback, int(base_state.mmpd_patch_size))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    t_build = time.perf_counter()
    model = build_model(
        base_state,
        mode=mode,
        n_variates=n_variates,
        lookback=lookback,
        horizon=horizon,
        stage=stage,
        device=device,
        torch_compile=torch_compile,
    )
    build_s = time.perf_counter() - t_build
    inputs = make_inputs(
        u_rows=u_rows,
        n_variates=n_variates,
        lookback=lookback,
        horizon=horizon,
        overlap=overlap,
        stage=stage,
        patch_w=patch_w,
        unique_segments=unique,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # One eager step before warmup so compile finishes before timing.
    run_step(model, optimizer, inputs, use_amp)
    sync()

    for _ in range(warmup):
        run_step(model, optimizer, inputs, use_amp)
    sync()

    samples: list[float] = []
    last_loss = 0.0
    for _ in range(timed):
        t0 = time.perf_counter()
        last_loss = run_step(model, optimizer, inputs, use_amp)
        sync()
        samples.append(time.perf_counter() - t0)

    row = {
        "mode": mode,
        "stage": stage,
        "median_ms": median_ms(samples),
        "mean_ms": 1000.0 * (sum(samples) / len(samples)),
        "min_ms": 1000.0 * min(samples),
        "max_ms": 1000.0 * max(samples),
        "n_timed": timed,
        "warmup": warmup,
        "build_s": build_s,
        "last_loss": last_loss,
        "peak_mib": peak_mib(device),
        "u_rows": int(u_rows),
        "n_windows": inputs.n_windows,
        "n_variates": int(n_variates),
        "ctx_tokens": expected_ctx_tokens(mode, n_variates, n_past),
        "lookback": int(lookback),
        "horizon": int(horizon),
        "use_amp": bool(use_amp),
        "torch_compile": bool(torch_compile),
    }
    print(json.dumps(row, sort_keys=True), flush=True)

    del optimizer, model, inputs
    torch.cuda.empty_cache()
    return row


def summarize(rows: list[dict[str, Any]], gpu: str, git_commit: str) -> dict[str, Any]:
    by_key = {(r["stage"], r["mode"]): r for r in rows}
    summary: dict[str, Any] = {
        "gpu": gpu,
        "git_commit": git_commit,
        "per_stage": {},
        "combined": {},
    }
    for stage in STAGES:
        base = by_key.get((stage, "patch_decoder"))
        stage_out: dict[str, Any] = {"baseline_ms": None, "modes": {}}
        if base is None:
            summary["per_stage"][stage] = stage_out
            continue
        base_ms = float(base["median_ms"])
        stage_out["baseline_ms"] = base_ms
        for mode in MODES:
            row = by_key.get((stage, mode))
            if row is None:
                continue
            ms = float(row["median_ms"])
            speedup = base_ms / ms if ms > 0 else None
            saved = (1.0 - ms / base_ms) * 100.0 if base_ms > 0 else None
            stage_out["modes"][mode] = {
                "median_ms": ms,
                "speedup_vs_patch_decoder": speedup,
                "pct_train_time_saved_vs_patch_decoder": saved,
                "ctx_tokens": row["ctx_tokens"],
                "peak_mib": row["peak_mib"],
            }
        summary["per_stage"][stage] = stage_out

    patch_ms = []
    other_ms = {m: [] for m in MODES if m != "patch_decoder"}
    for stage in STAGES:
        b = by_key.get((stage, "patch_decoder"))
        if b is not None:
            patch_ms.append(float(b["median_ms"]))
        for mode in other_ms:
            r = by_key.get((stage, mode))
            if r is not None:
                other_ms[mode].append(float(r["median_ms"]))
    if patch_ms:
        avg_patch = sum(patch_ms) / len(patch_ms)
        summary["combined"]["patch_decoder_avg_ms"] = avg_patch
        for mode, vals in other_ms.items():
            if not vals:
                continue
            avg = sum(vals) / len(vals)
            summary["combined"][mode] = {
                "avg_ms": avg,
                "speedup_vs_patch_decoder": avg_patch / avg if avg > 0 else None,
                "pct_train_time_saved_vs_patch_decoder": (
                    (1.0 - avg / avg_patch) * 100.0 if avg_patch > 0 else None
                ),
            }
    summary["rows"] = rows
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--u-rows", type=int, default=217)
    parser.add_argument("--n-variates", type=int, default=321)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timed", type=int, default=100)
    parser.add_argument("--stages", nargs="+", default=STAGES)
    parser.add_argument("--modes", nargs="+", default=MODES)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--out",
        default="results/logs/benchmark-elec-guidance-fwd.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    gpu = torch.cuda.get_device_name(0)
    git_commit = git_head()
    print(
        json.dumps(
            {
                "gpu": gpu,
                "torch": torch.__version__,
                "git_commit": git_commit,
                "config": args.config,
                "u_rows": int(args.u_rows),
                "n_variates": int(args.n_variates),
                "warmup": int(args.warmup),
                "timed": int(args.timed),
                "stages": list(args.stages),
                "modes": list(args.modes),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    cfg = load_experiment_config(args.config, {"dataset": "electricity", "seed": 42})
    state = PipelineState.from_config(cfg)
    pipeline_mod.resolve_pipeline_data_subset(state)
    lookback, horizon = pipeline_mod.dataset_window_lengths(state, "electricity")
    overlap = int(state.lookback_overlap)
    n_past = n_past_patches(lookback, int(state.mmpd_patch_size))
    use_amp = (not args.no_amp) and bool(state.use_amp)
    torch_compile = (not args.no_compile) and bool(getattr(state, "torch_compile", True))

    print(
        json.dumps(
            {
                "lookback": lookback,
                "horizon": horizon,
                "overlap": overlap,
                "mmpd_patch_size": int(state.mmpd_patch_size),
                "n_past_patches": n_past,
                "patch_decoder_ctx_tokens": int(args.n_variates) * n_past,
                "itransformer_ctx_tokens": int(args.n_variates),
                "coarse_image_height": int(state.coarse_image_height),
                "patch_refine_patch_height": int(state.patch_refine_patch_height),
                "dit_patch_size": list(state.dit_patch_size),
                "dataloader_windows": dataloader_windows_for_univariate_rows(
                    int(args.u_rows), int(args.n_variates),
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for stage in args.stages:
        print(f"\n===== stage={stage} =====", flush=True)
        for mode in args.modes:
            print(f"\n--- mode={mode} ---", flush=True)
            try:
                rows.append(
                    time_mode(
                        base_state=state,
                        mode=mode,
                        stage=stage,
                        n_variates=int(args.n_variates),
                        lookback=int(lookback),
                        horizon=int(horizon),
                        overlap=overlap,
                        u_rows=int(args.u_rows),
                        device=device,
                        warmup=int(args.warmup),
                        timed=int(args.timed),
                        use_amp=use_amp,
                        torch_compile=torch_compile,
                    )
                )
            except torch.cuda.OutOfMemoryError as exc:
                print(json.dumps({"mode": mode, "stage": stage, "error": f"OOM: {exc}"}), flush=True)
                torch.cuda.empty_cache()
            except Exception as exc:
                print(json.dumps({"mode": mode, "stage": stage, "error": str(exc)}), flush=True)
                raise

    summary = summarize(rows, gpu, git_commit)
    print("\n======== SUMMARY ========", flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2), flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
