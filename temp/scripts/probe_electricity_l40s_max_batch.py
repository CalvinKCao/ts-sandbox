#!/usr/bin/env python3
"""L40S train-step max-B probe for electricity 321-var and 160-var halves.

Uses the live canvas128 p64x6 create_diffusion_model path (torch.compile on)
and the same get_loss + backward search as probe_train_batch_size.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch
import torch.nn as nn

from models.diffusion_tsf import train_multivariate_pipeline as pipeline_mod
from models.diffusion_tsf.pipeline.config import load_experiment_config
from models.diffusion_tsf.pipeline.phases.staged_diffusion_finetune_hp import (
    _probe_max_finetune_batch_size,
)
from models.diffusion_tsf.pipeline.state import PipelineState


class _TokenStub(nn.Module):
    """Patch-decoder-shaped context: (B, V * n_past, ctx_dim)."""

    def __init__(self, n_past: int, ctx_dim: int):
        super().__init__()
        self.n_past = int(n_past)
        self.ctx_dim = int(ctx_dim)
        self.token_variate_ids = None

    def get_encoder_tokens(self, past: torch.Tensor) -> torch.Tensor:
        if past.dim() == 2:
            past = past.unsqueeze(1)
        bsz, n_var, _ = past.shape
        self.token_variate_ids = torch.arange(n_var, device=past.device).repeat_interleave(
            self.n_past
        )
        return torch.zeros(
            bsz, n_var * self.n_past, self.ctx_dim, device=past.device, dtype=past.dtype,
        )


def n_past_patches(lookback: int, patch_size: int) -> int:
    pad = (patch_size - lookback % patch_size) % patch_size
    return (lookback + pad) // patch_size


def probe_one(
    state: PipelineState,
    *,
    n_variates: int,
    stage: str,
    device: torch.device,
    max_bs: int,
    headroom: float,
) -> dict:
    lookback, horizon = pipeline_mod.dataset_window_lengths(state, state.dataset)
    if stage == "patch_refine":
        state.image_height = int(state.patch_refine_patch_height)
    else:
        state.image_height = int(state.coarse_image_height)
    state.n_variates = int(n_variates)
    state.smoke_test = False
    state.torch_compile = True
    patch_size = int(state.mmpd_patch_size)
    n_past = n_past_patches(lookback, patch_size)
    ctx_dim = 256
    t0 = time.perf_counter()
    model = pipeline_mod.create_diffusion_model(
        state,
        n_variates=n_variates,
        lookback=lookback,
        horizon=horizon,
        guidance_model=_TokenStub(n_past, ctx_dim),
        diffusion_stage=stage,
    ).to(device)
    build_s = time.perf_counter() - t0
    print(
        f"  built {stage} V={n_variates} in {build_s:.1f}s  "
        f"compile={bool(state.torch_compile)}",
        flush=True,
    )
    try:
        t1 = time.perf_counter()
        usable = _probe_max_finetune_batch_size(
            model=model,
            lookback=int(lookback),
            horizon=int(horizon),
            overlap=int(state.lookback_overlap),
            n_variates=int(n_variates),
            device=device,
            stage=stage,
            min_bs=1,
            max_bs=int(max_bs),
            headroom=float(headroom),
        )
        probe_s = time.perf_counter() - t1
    finally:
        del model
        torch.cuda.empty_cache()
    row = {
        "stage": stage,
        "n_variates": int(n_variates),
        "batch_size": int(usable),
        "build_s": build_s,
        "probe_s": probe_s,
        "max_bs": int(max_bs),
        "headroom": float(headroom),
        "gpu": torch.cuda.get_device_name(0),
    }
    print(json.dumps(row, sort_keys=True), flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/binary_window_norm_patch_refine_canvas128_p64x6_electricity_allv_msdefault_fixed.yaml",
    )
    parser.add_argument("--max-bs", type=int, default=32)
    parser.add_argument("--headroom", type=float, default=0.85)
    parser.add_argument(
        "--variates",
        nargs="+",
        type=int,
        default=(321, 160),
    )
    parser.add_argument("--stages", nargs="+", default=("coarse", "patch_refine"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "config": args.config,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    cfg = load_experiment_config(args.config, {"dataset": "electricity", "seed": 42})
    state = PipelineState.from_config(cfg)
    if not bool(state.torch_compile):
        raise RuntimeError("probe expects training.torch_compile=true")
    rows = []
    for n_var in args.variates:
        for stage in args.stages:
            print(f"\n===== V={n_var} {stage} =====", flush=True)
            rows.append(
                probe_one(
                    state,
                    n_variates=int(n_var),
                    stage=str(stage),
                    device=device,
                    max_bs=args.max_bs,
                    headroom=args.headroom,
                )
            )
    summary = {
        "gpu": torch.cuda.get_device_name(0),
        "rows": rows,
        "by_key": {f"{r['stage']}/V{r['n_variates']}": r["batch_size"] for r in rows},
    }
    print("\n======== SUMMARY ========", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    out_path = "results/logs/probe-electricity-l40s-max-batch.json"
    with open(out_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise
    sys.exit(0)
