"""Fast-path parity and OOD shift tests for ordinal encoding."""

import time

import torch

from models.diffusion_tsf.ordinal_window_norm import (
    _value_to_rank,
    _value_to_rank_slow,
    build_global_ladder_from_training,
    encode_with_ladder,
    ordinal_encode,
    shift_window_to_ordinal_envelope,
)


def test_value_to_rank_matches_slow_reference():
    torch.manual_seed(0)
    for k in (3, 7, 16):
        values = torch.sort(torch.randn(k) * 3).values
        probe = torch.linspace(values[0].item() - 1.0, values[-1].item() + 1.0, steps=200)
        slow = _value_to_rank_slow(values, probe, 1e-6)
        fast = _value_to_rank(values, probe, 1e-6)
        assert torch.equal(slow, fast)


def test_ood_shift_brings_window_inside_margin():
    train = torch.linspace(-2.0, 2.0, steps=100).unsqueeze(1)
    ladder = build_global_ladder_from_training(train, tie_atol=1e-6)
    past = torch.tensor([[[3.5, 3.6, 3.7, 3.8]]])
    future = torch.tensor([[[4.0, 4.1, 4.2]]])
    shifted_p, shifted_f = shift_window_to_ordinal_envelope(
        past, future, ladder, margin_frac=0.05,
    )
    tmin, tmax = ladder.z_envelope()
    margin = (tmax - tmin) * 0.05
    lo = (tmin + margin).item()
    hi = (tmax - margin).item()
    window = torch.cat([shifted_p, shifted_f], dim=-1)
    assert float(window.min()) >= lo - 1e-5
    assert float(window.max()) <= hi + 1e-5


def test_ood_shift_skipped_when_in_envelope():
    train = torch.linspace(-2.0, 2.0, steps=100).unsqueeze(1)
    ladder = build_global_ladder_from_training(train, tie_atol=1e-6)
    past = torch.tensor([[[-1.0, 0.0, 0.5, 1.0]]])
    future = torch.tensor([[[0.0, 0.2, 0.4]]])
    sp, sf = shift_window_to_ordinal_envelope(past, future, ladder)
    assert torch.allclose(sp, past)
    assert torch.allclose(sf, future)


def test_ood_shift_lb336_val_batch_with_forced_ood():
    """Regression: val OOD shift on lb336/hz720 must not shape-mismatch."""
    from models.diffusion_tsf.train_multivariate_pipeline import load_dataset, generate_dataset_job
    from torch.utils.data import DataLoader

    vi = generate_dataset_job("ETTh1")["variate_indices"]
    _, val_ds, _, stats = load_dataset(
        "ETTh1", vi, lookback=336, horizon=720, lookback_overlap=8,
        use_ordinal_window_norm=True,
    )
    ladder = stats["ordinal_ladder"]
    past, future = next(iter(DataLoader(val_ds, batch_size=8)))
    past = past + 50.0
    ordinal_encode(past, future, ladder=ladder, apply_ood_shift=True)


def test_ood_shift_btv_layout():
    train = torch.randn(100, 7)
    ladder = build_global_ladder_from_training(train.numpy(), tie_atol=1e-6)
    past = torch.randn(4, 336, 7)
    future = torch.randn(4, 728, 7)
    shift_window_to_ordinal_envelope(past, future, ladder, margin_frac=0.05)


def test_weather_sized_encode_is_fast():
    """Weather-scale batch encode should be ms, not seconds."""
    torch.manual_seed(1)
    t_len, v = 50_000, 4
    z = torch.randn(t_len, v).numpy()
    ladder = build_global_ladder_from_training(z[:35_000], precompute_ranks_for=z)
    x = torch.randn(32, v, 1056)
    t0 = time.perf_counter()
    for _ in range(3):
        encode_with_ladder(x, ladder)
    elapsed = (time.perf_counter() - t0) / 3.0
    assert elapsed < 0.5, f"encode_with_ladder took {elapsed:.3f}s (expected <0.5s)"


def test_precomputed_ranks_match_encode():
    z = torch.linspace(-3.0, 3.0, steps=500).unsqueeze(1).numpy()
    ladder = build_global_ladder_from_training(z[:350], precompute_ranks_for=z)
    x = torch.from_numpy(z[400:412]).T.unsqueeze(0)
    encoded = encode_with_ladder(x, ladder)
    sliced = ladder.precomputed_ranks[400:412, 0].unsqueeze(0)
    assert torch.allclose(encoded[0, 0], sliced, atol=1e-4)
