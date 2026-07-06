"""Tests for global training-set ordinal encoding."""

import torch

from models.diffusion_tsf.ordinal_window_norm import (
    build_global_ladder_from_training,
    encode_with_ladder,
    ordinal_decode,
    ordinal_encode,
)


def _ladder_from_values(values: torch.Tensor, tie_atol: float = 1e-6):
    return build_global_ladder_from_training(values, tie_atol=tie_atol)


def test_flatline_all_same_ordinal():
    train = torch.ones(20, 1)
    ladder = _ladder_from_values(train)
    past = torch.ones(2, 1, 10)
    future = torch.ones(2, 1, 5)
    past_ord, fut_ord, _ = ordinal_encode(past, future, ladder=ladder)
    assert torch.allclose(past_ord, torch.zeros_like(past_ord), atol=1e-5)
    assert torch.allclose(fut_ord, torch.zeros_like(fut_ord), atol=1e-5)
    assert int(ladder.n_unique.max()) == 1


def test_three_unique_with_ties():
    train = torch.tensor([[-1.0], [0.0], [0.0], [0.0], [1.0]])
    ladder = _ladder_from_values(train)
    past = torch.tensor([[[-1.0, 0.0, 0.0, 0.0, 1.0]]])
    future = torch.tensor([[[0.0, 1.0, -1.0]]])
    past_ord, fut_ord, _ = ordinal_encode(past, future, ladder=ladder)
    assert int(ladder.n_unique[0, 0]) == 3
    assert past_ord.max() <= 2.0
    past_back, fut_back = ordinal_decode(past_ord, fut_ord, ladder)
    assert torch.allclose(past_back, past, atol=1e-5)
    assert torch.allclose(fut_back, future, atol=1e-5)


def test_global_ladder_shared_across_windows():
    train = torch.linspace(-2.0, 2.0, steps=50).unsqueeze(1)
    ladder = _ladder_from_values(train)
    w1_past = torch.tensor([[[-2.0, -1.0, 0.0]]])
    w2_past = torch.tensor([[[0.0, 1.0, 2.0]]])
    w1_ord, _, _ = ordinal_encode(w1_past, None, ladder=ladder)
    w2_ord, _, _ = ordinal_encode(w2_past, None, ladder=ladder)
    assert float(w1_ord[0, 0, -1]) < float(w2_ord[0, 0, -1])


def test_tie_atol_merges_near_duplicates():
    train = torch.tensor([[1.0], [1.0 + 1e-8], [2.0]])
    ladder = _ladder_from_values(train, tie_atol=1e-6)
    assert int(ladder.n_unique[0, 0]) == 2


def test_no_rank_gaps_between_ladder_rungs():
    from models.diffusion_tsf.ordinal_window_norm import _value_to_rank

    past = torch.tensor([[[-1.0, 0.0, 0.5, 1.0, 3.0]]])
    uniq = past[0, 0]
    probe = torch.linspace(-1.0, 3.0, steps=81)
    ranks = _value_to_rank(uniq, probe, 1e-6)
    assert ranks.min() >= 0 and ranks.max() < len(uniq)
    assert ranks[0] == 0
    assert ranks[-1] == len(uniq) - 1


def test_no_semiinfinite_low_bin_mass():
    """Midpoint partitions used to dump ~all lows into rank 0; nearest rung avoids that."""
    train = torch.linspace(-2.0, 2.0, steps=200).unsqueeze(1)
    ladder = _ladder_from_values(train)
    full_ord = encode_with_ladder(train.T.unsqueeze(0), ladder.expand_batch(1))[0, 0]
    frac0 = float((full_ord == 0).float().mean())
    assert frac0 < 0.15, f"rank 0 absorbed {frac0:.1%} of evenly spaced training values"
