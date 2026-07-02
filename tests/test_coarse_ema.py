import numpy as np

from models.diffusion_tsf.coarse_ema import (
    causal_ema_1d,
    causal_ema_with_past_seed,
    fine_residual_vs_smoothed_coarse,
)


def test_causal_ema_seeds_from_past():
    x = np.array([1.0, 1.0, 1.0, 5.0, 5.0])
    out = causal_ema_with_past_seed(0.0, x, alpha=0.5)
    assert out.shape == x.shape
    assert out[0] == 0.0
    assert out[1] == 0.5


def test_fine_residual_vs_smoothed_coarse():
    gt = np.linspace(0.0, 1.0, 8)
    coarse = np.array([0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0])
    raw, smooth, resid = fine_residual_vs_smoothed_coarse(
        gt, coarse, past_tail=-0.1, alpha=0.3,
    )
    assert raw.shape == gt.shape
    assert smooth.shape == gt.shape
    assert resid.shape == gt.shape
    assert np.allclose(resid, gt - smooth)
    assert not np.allclose(smooth, coarse)
