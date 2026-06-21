import torch

from models.diffusion_tsf.config import DiffusionTSFConfig
from models.diffusion_tsf.diffusion_model import DiffusionTSF


def _model() -> DiffusionTSF:
    cfg = DiffusionTSFConfig(
        window_norm_center="last",
        num_variables=1,
        use_guidance_channel=False,
        disable_cross_attention=True,
    )
    return DiffusionTSF(cfg)


def test_window_norm_center_last_zeros_last_past_step():
    model = _model()
    past = torch.tensor([[[1.0, 2.0, 5.0]]])
    future = torch.tensor([[[6.0, 7.0]]])
    past_norm, future_norm, (center, std) = model._normalize_sequence(past, future)

    assert torch.allclose(center, past[..., -1:])
    assert torch.allclose(past_norm[..., -1], torch.zeros(1))
    assert torch.allclose(future_norm[..., 0], (future[..., 0] - past[..., -1]) / std)


def test_window_norm_center_mean_default():
    cfg = DiffusionTSFConfig(
        window_norm_center="mean",
        num_variables=1,
        use_guidance_channel=False,
        disable_cross_attention=True,
    )
    model = DiffusionTSF(cfg)
    past = torch.tensor([[[0.0, 2.0, 4.0]]])
    past_norm, _, (center, _) = model._normalize_sequence(past)
    assert torch.allclose(center, past.mean(dim=-1, keepdim=True))
    assert not torch.allclose(past_norm[..., -1], torch.zeros(1))
