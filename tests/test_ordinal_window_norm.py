"""Tests for tie-aware ordinal window normalization."""

import torch

from models.diffusion_tsf.ordinal_window_norm import ordinal_decode, ordinal_encode


def test_flatline_all_same_ordinal():
    past = torch.ones(2, 1, 10)
    future = torch.ones(2, 1, 5)
    past_ord, fut_ord, ladder = ordinal_encode(past, future, max_scale=3.5, tie_atol=1e-6)
    assert torch.allclose(past_ord, torch.zeros_like(past_ord), atol=1e-5)
    assert torch.allclose(fut_ord, torch.zeros_like(fut_ord), atol=1e-5)
    assert int(ladder.n_unique.max()) == 1


def test_three_unique_with_ties():
    # unique at -1, 0, 1; middle value repeated
    past = torch.tensor([[[-1.0, 0.0, 0.0, 0.0, 1.0]]])
    future = torch.tensor([[[0.0, 1.0, -1.0]]])
    past_ord, fut_ord, ladder = ordinal_encode(past, future, max_scale=2.0, tie_atol=1e-6)
    assert int(ladder.n_unique[0, 0]) == 3
    assert len(torch.unique(past_ord)) <= 3
    past_back, fut_back = ordinal_decode(past_ord, fut_ord, ladder)
    assert torch.allclose(past_back, past, atol=1e-5)
    assert torch.allclose(fut_back, future, atol=1e-5)


def test_tie_atol_merges_near_duplicates():
    past = torch.tensor([[[1.0, 1.0 + 1e-8, 2.0]]])
    _, _, ladder = ordinal_encode(past, None, max_scale=1.0, tie_atol=1e-6)
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
