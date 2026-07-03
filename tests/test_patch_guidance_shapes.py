"""Shape smoke tests for patch-decoder guidance + DiT cross-attention."""

from __future__ import annotations

import torch

from models.diffusion_tsf.config import DiffusionTSFConfig
from models.diffusion_tsf.diffusion_model import DiffusionTSF
from models.diffusion_tsf.guidance import PatchDecoderGuidance
from models.diffusion_tsf.patch_guidance_stack import PatchGuidanceStack, PatchGuidanceStackConfig


def test_patch_guidance_mixer_and_dit_shapes():
    B, V, L = 2, 7, 96
    patch_size = 12
    chunk_hz = 96
    overlap = 8

    stack_cfg = PatchGuidanceStackConfig(
        in_len=L,
        out_len=chunk_hz,
        patch_size=patch_size,
        data_dim=V,
    )
    stack = PatchGuidanceStack(stack_cfg)
    guidance = PatchDecoderGuidance(stack, chunk_horizon=chunk_hz)

    past_norm = torch.randn(B, V, L)
    mixed = guidance.get_encoder_tokens(past_norm)
    assert mixed.shape == (B, V * (L // patch_size), stack_cfg.context_dim)
    assert guidance.token_variate_ids is not None
    assert guidance.token_variate_ids.shape[0] == mixed.shape[1]

    cfg = DiffusionTSFConfig(
        num_variables=V,
        lookback_length=L,
        forecast_length=chunk_hz + overlap,
        dataset_forecast_length=chunk_hz,
        lookback_overlap=overlap,
        diffusion_chunk_horizon=chunk_hz,
        guidance_type="patch_decoder",
        mmpd_patch_size=patch_size,
        diffusion_stage="joint",
        use_guidance_channel=True,
    )
    model = DiffusionTSF(cfg, guidance_model=guidance)
    model.eval()

    past = torch.randn(B, V, L)
    future = torch.randn(B, V, overlap + chunk_hz)
    with torch.no_grad():
        out = model(past, future)
    assert "loss" in out
    assert torch.isfinite(out["loss"])


def test_patch_guidance_out_len_matches_short_horizon():
    from models.diffusion_tsf import train_multivariate_pipeline as pipe

    old_forecast = pipe.FORECAST_LENGTH
    old_chunk = pipe.DIFFUSION_CHUNK_HORIZON
    try:
        pipe.FORECAST_LENGTH = 20
        pipe.DIFFUSION_CHUNK_HORIZON = 96
        stack = pipe.create_patch_guidance_stack(5)
        assert stack.config.out_len == 20
    finally:
        pipe.FORECAST_LENGTH = old_forecast
        pipe.DIFFUSION_CHUNK_HORIZON = old_chunk
