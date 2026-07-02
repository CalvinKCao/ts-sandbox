import numpy as np

from models.diffusion_tsf.coarse_flatline_blur import (
    flatline_preserving_blur,
    fine_residual_vs_flatline_blur_coarse,
    segment_constant_runs,
)
from models.diffusion_tsf.coarse_ema import causal_ema_with_past_seed


def test_flatline_plateau_unchanged_shape():
    x = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 0.5])
    smooth, runs = flatline_preserving_blur(x, blur_radius=1, past_seed=0.0)
    assert smooth.shape == x.shape
    assert np.all(smooth[0:3] == smooth[0])
    assert np.all(smooth[3:5] == smooth[3])
    assert np.all(smooth[5:8] == smooth[5])
    assert len(runs) == 3


def test_flatline_preserving_keeps_plateaus_flat():
    x = np.array([0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0])
    blur, _ = flatline_preserving_blur(x, blur_radius=1, past_seed=0.0)
    ema = causal_ema_with_past_seed(0.0, x, alpha=0.3)
    assert np.ptp(blur[0:4]) < 1e-9
    assert np.ptp(blur[4:8]) < 1e-9
    # EMA ramps through the step between plateaus (not flat within transition).
    assert abs(ema[4] - 2.0) > 0.1


def test_flatline_runs_from_gt_combined_source():
    gt_combined = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 1.0])
    coarse = np.array([0.2, 0.8, 0.2, 0.6, 1.0, 1.0])
    smooth, runs = flatline_preserving_blur(
        coarse, flatline_source=gt_combined, blur_radius=1, past_seed=0.0,
    )
    assert len(runs) == 2
    assert runs[0].length == 3
    assert np.ptp(smooth[0:3]) < 1e-9
    assert np.ptp(smooth[3:6]) < 1e-9
    assert not np.allclose(smooth[0:3], coarse[0:3])


def test_flatline_preserving_blur_torch_matches_numpy():
    series = np.array([0.2, 0.8, 0.2, 0.6, 1.0, 1.0], dtype=np.float64)
    gt_combined = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
    import torch
    from models.diffusion_tsf.coarse_flatline_blur import flatline_preserving_blur_torch

    np_out, _ = flatline_preserving_blur(
        series, flatline_source=gt_combined, blur_radius=4, past_seed=0.0,
    )
    torch_out = flatline_preserving_blur_torch(
        torch.tensor(series, dtype=torch.float64).unsqueeze(0),
        flatline_source=torch.tensor(gt_combined, dtype=torch.float64).unsqueeze(0),
        blur_radius=4,
        past_seed=torch.tensor([0.0], dtype=torch.float64),
    )
    assert np.allclose(np_out, torch_out.squeeze(0).numpy())
    gt = np.linspace(0.0, 1.0, 6)
    coarse = np.array([0.0, 0.0, 0.4, 0.4, 0.8, 0.8])
    gt_combined = np.array([0.0, 0.0, 0.2, 0.2, 0.8, 0.8])
    raw, smooth, resid, runs = fine_residual_vs_flatline_blur_coarse(
        gt, coarse, gt_combined=gt_combined, past_seed=-0.1, blur_radius=1,
    )
    assert np.allclose(resid, gt - smooth)
    assert len(runs) == 3
