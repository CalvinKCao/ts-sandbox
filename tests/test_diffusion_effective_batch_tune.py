"""Tests for effective-batch HP resolution in staged diffusion finetune."""

from __future__ import annotations

from models.diffusion_tsf.pipeline.phases.staged_diffusion_finetune_hp import (
    resolve_target_effective_batch,
)


def test_resolve_exact_effective_batch_with_accum():
    out = resolve_target_effective_batch(probed_max=32, target_effective=64)
    assert out["batch_size"] == 32
    assert out["gradient_accumulation_steps"] == 2
    assert out["effective_batch_size"] == 64


def test_resolve_low_effective_batch():
    out = resolve_target_effective_batch(probed_max=40, target_effective=10)
    assert out["batch_size"] <= 40
    assert out["effective_batch_size"] >= 1
    assert abs(out["effective_batch_size"] - 10) <= 2


def test_resolve_high_effective_batch_uses_accum():
    out = resolve_target_effective_batch(probed_max=42, target_effective=168)
    assert out["batch_size"] <= 42
    assert out["gradient_accumulation_steps"] >= 1
    assert out["effective_batch_size"] >= 150
