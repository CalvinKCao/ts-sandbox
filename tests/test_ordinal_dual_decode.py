"""Ordinal staged dual decode must use bounded heights, not legacy max_scale."""

import torch

from models.diffusion_tsf.config import DiffusionTSFConfig
from models.diffusion_tsf.diffusion_model import DiffusionTSF
from models.diffusion_tsf.ordinal_window_norm import build_global_ladder_from_training


def _tiny_ordinal_fine_model() -> DiffusionTSF:
    train = torch.randn(80, 2)
    ladder = build_global_ladder_from_training(train.numpy(), tie_atol=1e-6)
    cfg = DiffusionTSFConfig(
        num_variables=2,
        lookback_length=16,
        forecast_length=16,
        image_height=8,
        coarse_image_height=8,
        fine_image_height=8,
        max_scale=5.4,
        diffusion_stage="fine",
        use_ordinal_window_norm=True,
        use_window_normalization=False,
        ordinal_ladder=ladder,
        use_guidance_channel=False,
        disable_cross_attention=True,
        binary_num_steps=4,
        dit_embed_dim=32,
        dit_depth=2,
        dit_num_heads=2,
        dit_patch_size=(4, 4),
    )
    return DiffusionTSF(cfg)


def test_ordinal_decode_dual_from_2d_tracks_encode():
    model = _tiny_ordinal_fine_model()
    model.eval()
    vmax = model._ordinal_rank_max_tensor(torch.device("cpu")).reshape(1, 2, 1)
    x = torch.rand(2, 2, 16) * vmax
    maps = model._encode_staged_maps(x)
    decoded = model.decode_dual_from_2d(maps["coarse"], maps["fine"], from_diffusion=False)
    assert decoded.shape == x.shape
    assert decoded.std() > 0.05
    assert torch.allclose(decoded, x, atol=1.0)


def test_ordinal_decode_dual_not_flat_on_spiky_signal():
    model = _tiny_ordinal_fine_model()
    model.eval()
    vmax = float(model._ordinal_rank_max_tensor(torch.device("cpu")).max().item())
    t = torch.linspace(0, vmax, steps=16)
    x = torch.stack([t, t.flip(0)], dim=0).unsqueeze(0)
    maps = model._encode_staged_maps(x)
    decoded = model.decode_dual_from_2d(maps["coarse"], maps["fine"], from_diffusion=False)
    assert decoded.shape == (1, 2, 16)
    assert decoded[0, 0].std() > 0.1
