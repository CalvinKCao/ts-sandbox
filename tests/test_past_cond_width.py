"""Past visual cond width: native lookback vs horizon resize."""

from __future__ import annotations

import torch

from models.diffusion_tsf.config import DiffusionTSFConfig
from models.diffusion_tsf.diffusion_model import DiffusionTSF


def _staged_model(*, resize_to_horizon: bool, stage: str) -> DiffusionTSF:
    lb, overlap, hz = 96, 8, 96
    cfg = DiffusionTSFConfig(
        num_variables=3,
        lookback_length=lb,
        forecast_length=hz + overlap,
        dataset_forecast_length=hz,
        lookback_overlap=overlap,
        diffusion_lookback_cap=lb,
        representation_time_stride=1,
        past_cond_resize_to_horizon=resize_to_horizon,
        diffusion_stage=stage,
        image_height=16,
        coarse_image_height=16,
        fine_image_height=16,
        use_guidance_channel=False,
        disable_cross_attention=True,
        dit_patch_size=(8, 8),
        dit_embed_dim=64,
        dit_depth=2,
        dit_num_heads=2,
    )
    return DiffusionTSF(cfg)


def test_coarse_past_cond_native_width_when_not_resized():
    model = _staged_model(resize_to_horizon=False, stage="coarse")
    past_norm = torch.randn(2, 3, 96)
    cond, _ = model._staged_past_condition(past_norm, target_width=104)
    assert cond.shape[-1] == 96


def test_fine_past_cond_padded_not_stretched_when_not_resized():
    model = _staged_model(resize_to_horizon=False, stage="fine")
    past_norm = torch.randn(1, 3, 96)
    future_norm = torch.randn(1, 3, 104)
    cond_past, _ = model._staged_past_condition(past_norm, target_width=104)
    assert cond_past.shape[-1] == 96
    future_maps = model._encode_staged_maps(future_norm)
    future_coarse = model._coarse_cdf_to_height(future_maps["coarse"], 16)
    horizon_flat = future_coarse.reshape(3, 1, 16, 104)
    cond = model._cat_past_and_horizon_cond(cond_past, horizon_flat)
    assert cond.shape[-1] == 104
    past_only = cond[:, :2]
    assert past_only[..., :96].abs().sum() > 0
    assert past_only[..., 96:].abs().sum() == 0
