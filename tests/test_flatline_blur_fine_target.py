import torch

from models.diffusion_tsf.config import DiffusionTSFConfig
from models.diffusion_tsf.diffusion_model import DiffusionTSF


def _tiny_blur_fourier_model() -> DiffusionTSF:
    cfg = DiffusionTSFConfig(
        num_variables=1,
        lookback_length=16,
        forecast_length=16,
        lookback_overlap=2,
        image_height=8,
        coarse_image_height=8,
        fine_image_height=8,
        max_scale=3.5,
        diffusion_stage="fine",
        staged_representation="fourier_frequency",
        fourier_fine_max_scale=0.5,
        coarse_flatline_blur_fine_target=True,
        coarse_flatline_blur_radius=4,
        use_guidance_channel=False,
        disable_cross_attention=True,
        binary_num_steps=4,
        dit_embed_dim=32,
        dit_depth=2,
        dit_num_heads=2,
        dit_patch_size=(4, 4),
    )
    return DiffusionTSF(cfg)


def test_fine_forward_with_flatline_blur_target():
    model = _tiny_blur_fourier_model()
    model.train()
    past = torch.randn(2, 1, 16)
    future = torch.randn(2, 1, 16)
    out = model(past, future)
    assert "loss" in out
    assert torch.isfinite(out["loss"])


def test_decode_blurred_coarse_plus_fine():
    model = _tiny_blur_fourier_model()
    model.eval()
    past = torch.randn(1, 1, 16)
    maps = model._encode_staged_maps(torch.randn(1, 1, 16))
    past_seed = past[..., -1]
    raw = model.decode_dual_from_2d(maps["coarse"], maps["fine"], past_seed=past_seed)
    assert raw.shape == (1, 16)
