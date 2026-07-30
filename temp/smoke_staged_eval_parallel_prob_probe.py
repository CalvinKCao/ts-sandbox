#!/usr/bin/env python3
"""Smoke test for staged_eval's parallel-MC-sample + GPU batch probe path.

Builds tiny coarse/patch_refine models directly (no checkpoints, no full
dataset) so this runs in seconds, and drives the real staged_eval functions:
``_probe_max_staged_eval_batch_size``, ``_reshape_parallel_samples``, and
``StagedEvalPhase._run_eval`` (the repeat_interleave parallel-sample path).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.config import DiffusionTSFConfig
from models.diffusion_tsf.diffusion_model import DiffusionTSF
from models.diffusion_tsf.pipeline.phases.staged_eval import (
    StagedEvalPhase,
    _probe_max_staged_eval_batch_size,
    _reshape_parallel_samples,
)

N_VARIATES = 2
LOOKBACK = 32
LOOKBACK_OVERLAP = 4
DATASET_HZ = 16
FORECAST_LEN = DATASET_HZ + LOOKBACK_OVERLAP  # model horizon incl. overlap


def _tiny_common_kwargs() -> dict:
    return dict(
        num_variables=N_VARIATES,
        lookback_length=LOOKBACK,
        forecast_length=FORECAST_LEN,
        dataset_forecast_length=DATASET_HZ,
        lookback_overlap=LOOKBACK_OVERLAP,
        coarse_image_height=16,
        fine_image_height=16,
        finer_image_height=16,
        use_guidance_channel=False,
        disable_cross_attention=True,
        dit_embed_dim=32,
        dit_depth=2,
        dit_num_heads=2,
        dit_mlp_ratio=2.0,
    )


def _build_models(device: torch.device):
    coarse_cfg = DiffusionTSFConfig(
        image_height=16,
        diffusion_stage="coarse",
        **_tiny_common_kwargs(),
    )
    patch_cfg = DiffusionTSFConfig(
        image_height=32,
        diffusion_stage="patch_refine",
        patch_refine_canvas_height=256,
        patch_refine_patch_height=32,
        patch_refine_patch_width=8,
        patch_refine_col_stride=6,
        patch_refine_unique_segments=True,
        **_tiny_common_kwargs(),
    )
    coarse_model = DiffusionTSF(coarse_cfg).to(device).eval()
    fine_model = DiffusionTSF(patch_cfg).to(device).eval()
    return coarse_model, fine_model


class _TinyWindowDataset(Dataset):
    def __init__(self, n: int, v: int, lookback: int, future_w: int):
        self.past = torch.randn(n, v, lookback)
        self.future = torch.randn(n, v, future_w)

    def __len__(self) -> int:
        return self.past.shape[0]

    def __getitem__(self, idx: int):
        return self.past[idx], self.future[idx]


def _test_probe(coarse_model, fine_model, device: torch.device) -> None:
    max_fit = _probe_max_staged_eval_batch_size(
        coarse_model=coarse_model,
        fine_model=fine_model,
        lookback=LOOKBACK,
        n_variates=N_VARIATES,
        device=device,
        det_kwargs={"sampler": "anchor"},
        joint_dual=False,
        min_bs=1,
        max_bs=16,
    )
    assert max_fit >= 1, f"probe returned non-positive batch size: {max_fit}"
    print(f"[ok] batch probe: max_fit={max_fit} (device={device.type})")


def _test_reshape_parallel_samples() -> None:
    batch, n_samples, v, h = 3, 4, 2, 5
    flat = torch.arange(batch * n_samples * v * h, dtype=torch.float32).view(
        batch * n_samples, v, h
    )
    reshaped = _reshape_parallel_samples(flat, batch, n_samples)
    assert reshaped.shape == (batch, v, n_samples, h)
    # repeat_interleave order: flat[w*n_samples + s] belongs to (window=w, sample=s).
    for w in range(batch):
        for s in range(n_samples):
            assert torch.equal(reshaped[w, :, s, :], flat[w * n_samples + s])
    try:
        _reshape_parallel_samples(flat[:-1], batch, n_samples)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on shape mismatch")
    print("[ok] _reshape_parallel_samples matches repeat_interleave ordering")


def _test_run_eval(coarse_model, fine_model, device: torch.device) -> None:
    n_windows = 6
    batch_size = 2
    prob_samples = 4
    ds = _TinyWindowDataset(n_windows, N_VARIATES, LOOKBACK, FORECAST_LEN)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    state = SimpleNamespace(seed=42, eval_sampler="anchor", smoke_test=False)

    phase = StagedEvalPhase()
    metrics, pack = phase._run_eval(
        state=state,
        subset_id="smoke",
        loader=loader,
        device=device,
        coarse_model=coarse_model,
        fine_model=fine_model,
        finer_model=None,
        prob_sampler="ddim",
        prob_steps=3,
        prob_samples=prob_samples,
        gmm_components=1,
        topk_max=1,
        window_indices=list(range(n_windows)),
        test_stride=1,
    )
    assert pack["samples"].shape == (n_windows, N_VARIATES, prob_samples, DATASET_HZ)
    assert pack["y_true"].shape == (n_windows, N_VARIATES, DATASET_HZ)
    assert pack["deterministic"].shape == (n_windows, N_VARIATES, DATASET_HZ)
    for key in ("crps", "anchor_mse", "anchor_mae", "sample_mean_mse"):
        assert key in metrics, f"missing metric {key}"
    print(
        "[ok] _run_eval parallel-sample path: samples.shape="
        f"{pack['samples'].shape} crps={metrics['crps']:.4f} "
        f"anchor_mse={metrics['anchor_mse']:.4f}"
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coarse_model, fine_model = _build_models(device)
    _test_reshape_parallel_samples()
    _test_probe(coarse_model, fine_model, device)
    _test_run_eval(coarse_model, fine_model, device)
    print("staged_eval parallel-sample + probe smoke ok")


if __name__ == "__main__":
    main()
