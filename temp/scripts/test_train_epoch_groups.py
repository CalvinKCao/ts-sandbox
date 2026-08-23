#!/usr/bin/env python3
"""Contract checks for train_epoch_groups batch packing."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from models.diffusion_tsf.pipeline.config import load_experiment_config
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline.train.diffusion_loop import (
    EpochGroupBatchSampler,
    fp32_window_nbytes,
    make_epoch_group_train_loader,
    make_grouped_train_loader,
    pack_constant_size_batches,
    resolve_train_epoch_groups,
    split_batches_into_groups,
)
from models.diffusion_tsf.train_multivariate_pipeline import patch_guidance_hp_objective


class _IndexDataset(Dataset):
    def __init__(self, n: int):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.tensor([idx], dtype=torch.float32), torch.tensor([idx], dtype=torch.float32)


def _unique_in_group(group):
    seen = []
    for batch in group:
        seen.extend(batch)
    return set(seen)


def test_default_n1_is_full_set():
    n, bsz = 20, 4
    sampler = EpochGroupBatchSampler(n, bsz, 1, seed=0)
    assert sampler.n_groups == 1
    sampler.set_epoch(0)
    batches = list(sampler)
    assert all(len(batch) == bsz for batch in batches)
    covered = {i for batch in batches for i in batch}
    assert set(range(n)) <= covered


def test_cycle_covers_all_indices():
    n, bsz, n_groups = 25, 4, 3
    sampler = EpochGroupBatchSampler(n, bsz, n_groups, seed=7)
    covered = set()
    for epoch in range(n_groups):
        sampler.set_epoch(epoch)
        for batch in sampler:
            assert len(batch) == bsz
            covered.update(batch)
    assert set(range(n)) <= covered
    union = set()
    for group in sampler.groups:
        union |= _unique_in_group(group)
    assert set(range(n)) <= union
    assert sampler.n_padded >= 0


def test_within_cycle_batch_order_shuffled():
    sampler = EpochGroupBatchSampler(80, 4, 2, seed=11)
    sampler.set_epoch(0)
    first = [tuple(b) for b in sampler]
    sampler.set_epoch(0)
    again = [tuple(b) for b in sampler]
    assert first == again
    sampler.set_epoch(1)
    other_group = [tuple(b) for b in sampler]
    first_idx = {i for b in first for i in b}
    other_idx = {i for b in other_group for i in b}
    assert first_idx.isdisjoint(other_idx)
    assert set(range(80)) <= (first_idx | other_idx)


def test_reshuffle_repacks_across_cycles():
    sampler = EpochGroupBatchSampler(80, 4, 2, seed=11)
    sampler.set_epoch(0)
    cycle0_g0 = {tuple(sorted(b)) for b in sampler}
    sampler.set_epoch(1)
    cycle0_g1 = {tuple(sorted(b)) for b in sampler}
    sampler.set_epoch(2)
    cycle1_g0 = {tuple(sorted(b)) for b in sampler}
    sampler.set_epoch(3)
    covered1 = set()
    for epoch in range(2, 4):
        sampler.set_epoch(epoch)
        for batch in sampler:
            covered1.update(batch)
    assert set(range(80)) <= covered1
    assert cycle0_g0 != cycle1_g0
    assert cycle0_g0 | cycle0_g1 != cycle1_g0


def test_compiled_b_constant_with_padding():
    batches, n_padded = pack_constant_size_batches(list(range(10)), 4)
    assert n_padded == 2
    assert all(len(b) == 4 for b in batches)
    flat = [i for b in batches for i in b]
    assert set(range(10)) <= set(flat)


def test_too_many_groups_fails():
    batches, _ = pack_constant_size_batches(list(range(8)), 4)
    try:
        split_batches_into_groups(batches, 5)
    except ValueError as exc:
        assert "zero batches" in str(exc)
    else:
        raise AssertionError("expected ValueError for empty group")


def test_smoke_clamps_groups():
    sampler = EpochGroupBatchSampler(5, 4, 20, seed=1, smoke_test=True)
    assert sampler.n_groups == 2
    assert sampler.n_groups_requested == 20
    sampler.set_epoch(0)
    assert all(len(b) == 4 for b in sampler)


def test_loader_constant_batch_size():
    ds = _IndexDataset(11)
    loader = make_epoch_group_train_loader(ds, batch_size=4, n_groups=2, seed=3)
    for epoch in range(2):
        loader.batch_sampler.set_epoch(epoch)
        n_batches = 0
        for past, future in loader:
            assert past.shape[0] == 4
            n_batches += 1
        assert n_batches == len(loader)


def test_n1_reshuffles_every_epoch():
    sampler = EpochGroupBatchSampler(20, 4, 1, seed=3)
    sampler.set_epoch(0)
    first = [tuple(sorted(b)) for b in sampler]
    sampler.set_epoch(1)
    second = [tuple(sorted(b)) for b in sampler]
    assert first != second
    assert {i for b in first for i in b} >= set(range(20))
    assert {i for b in second for i in b} >= set(range(20))


def test_byte_cap_matches_reviewer_mins():
    cap = 20 * 1024 * 1024
    lookback, horizon, overlap = 336, 96, 8
    cases = (
        ("traffic", 431, 11849, 439),
        ("dynamic", 17, 349569, 500),
        ("solar", 137, 36361, 423),
    )
    for name, v, n, expected_n in cases:
        nbytes = fp32_window_nbytes(v, lookback, horizon, overlap)
        n_groups, got_nbytes, group_bytes = resolve_train_epoch_groups(
            n_samples=n,
            batch_size=1,
            n_groups=1,
            max_bytes=cap,
            n_variates=v,
            lookback=lookback,
            horizon=horizon,
            overlap=overlap,
        )
        assert got_nbytes == nbytes, name
        assert n_groups == expected_n, (name, n_groups, expected_n)
        assert group_bytes <= cap, name
        max_windows = cap // nbytes
        assert (n + expected_n - 1) // expected_n <= max_windows, name


def test_byte_cap_accounts_for_batch_size():
    cap = 20 * 1024 * 1024
    n_groups, nbytes, group_bytes = resolve_train_epoch_groups(
        n_samples=11849,
        batch_size=16,
        n_groups=1,
        max_bytes=cap,
        n_variates=431,
        lookback=336,
        horizon=96,
        overlap=8,
    )
    assert nbytes == 431 * 440 * 4
    assert 16 * nbytes <= cap
    assert group_bytes <= cap
    assert n_groups >= 439


def test_byte_cap_wins_over_explicit_n():
    n_groups, _, _ = resolve_train_epoch_groups(
        n_samples=11849,
        batch_size=1,
        n_groups=429,
        max_bytes=20 * 1024 * 1024,
        n_variates=431,
        lookback=336,
        horizon=96,
        overlap=8,
    )
    assert n_groups == 439


def test_byte_cap_missing_keys_fail():
    try:
        resolve_train_epoch_groups(
            n_samples=10,
            batch_size=1,
            n_groups=1,
            max_bytes=1024,
            n_variates=None,
            lookback=336,
            horizon=96,
            overlap=8,
        )
    except ValueError as exc:
        assert "n_variates" in str(exc)
    else:
        raise AssertionError("expected fail-fast for missing n_variates")


def test_one_batch_over_cap_fails():
    try:
        resolve_train_epoch_groups(
            n_samples=100,
            batch_size=64,
            n_groups=1,
            max_bytes=1000,
            n_variates=431,
            lookback=336,
            horizon=96,
            overlap=8,
        )
    except ValueError as exc:
        assert "compile-constant batch" in str(exc)
    else:
        raise AssertionError("expected fail-fast when one batch exceeds cap")


def test_grouped_loader_uses_epoch_sampler():
    ds = _IndexDataset(11)
    loader, n_groups, nbytes, group_bytes = make_grouped_train_loader(
        ds,
        batch_size=4,
        n_groups=1,
        seed=3,
        max_bytes=64,
        n_variates=1,
        lookback=2,
        horizon=2,
        overlap=0,
    )
    assert isinstance(loader.batch_sampler, EpochGroupBatchSampler)
    assert n_groups >= 1
    assert nbytes == 1 * (2 + 0 + 2) * 4
    assert group_bytes <= 64


def test_guidance_objective_requires_grouped_sampler():
    src = Path(patch_guidance_hp_objective.__code__.co_filename).read_text()
    start = src.find("def patch_guidance_hp_objective")
    chunk = src[start:start + 4000]
    assert "make_grouped_train_loader" in chunk
    assert "EpochGroupBatchSampler" in chunk
    assert "shuffle=True" not in chunk.split("def run_patch_guidance_finetune_hp_tuning")[0]


def test_base_yaml_default_is_one():
    cfg = load_experiment_config(
        "configs/binary_window_norm_patch_refine_canvas128_p64x6.yaml"
    )
    assert int(cfg["training"]["train_epoch_groups"]) == 1
    assert cfg["training"]["train_epoch_max_bytes"] is None
    state = PipelineState.from_config(cfg)
    assert int(state.train_epoch_groups) == 1
    assert state.train_epoch_max_bytes is None


def test_leaf_yaml_override_and_load():
    stems = (
        "binary_window_norm_patch_refine_canvas128_p64x6_traffic_v000_430_s1_groups_msdefault_fixed",
        "binary_window_norm_patch_refine_canvas128_p64x6_traffic_v431_861_s1_groups_msdefault_fixed",
        "binary_window_norm_patch_refine_canvas128_p64x6_dynamic_allv_s1_groups_msdefault_fixed",
        "binary_window_norm_patch_refine_canvas128_p64x6_solar_alabama_allv_s1_groups_msdefault_fixed",
    )
    for stem in stems:
        cfg = load_experiment_config(f"configs/{stem}.yaml")
        assert int(cfg["training"]["train_epoch_groups"]) == 1, stem
        assert int(cfg["training"]["train_epoch_max_bytes"]) == 20971520, stem
        state = PipelineState.from_config(cfg)
        assert int(state.train_epoch_groups) == 1
        assert int(state.train_epoch_max_bytes) == 20971520
        assert float(state.patch_refine_finetune_window_fraction) == 1.0
        subset = state.data_subset_by_dataset[state.dataset]
        assert int(subset["train_stride"]) == 1
        assert int(subset["val_stride"]) == 1


if __name__ == "__main__":
    test_default_n1_is_full_set()
    test_cycle_covers_all_indices()
    test_within_cycle_batch_order_shuffled()
    test_reshuffle_repacks_across_cycles()
    test_compiled_b_constant_with_padding()
    test_too_many_groups_fails()
    test_smoke_clamps_groups()
    test_loader_constant_batch_size()
    test_n1_reshuffles_every_epoch()
    test_byte_cap_matches_reviewer_mins()
    test_byte_cap_accounts_for_batch_size()
    test_byte_cap_wins_over_explicit_n()
    test_byte_cap_missing_keys_fail()
    test_one_batch_over_cap_fails()
    test_grouped_loader_uses_epoch_sampler()
    test_guidance_objective_requires_grouped_sampler()
    test_base_yaml_default_is_one()
    test_leaf_yaml_override_and_load()
    print("sampler tests ok")
