"""Reuse discovery must exact-match config stems (no _bs_* / _smoke siblings)."""

from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import (
    _run_dir_matches_config,
)


def test_exact_config_stem_matches():
    name = "07-08-4122622-ETTh1-binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_uncompressed"
    cfg = "binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_uncompressed"
    assert _run_dir_matches_config(name, "ETTh1", cfg)


def test_bs_and_smoke_siblings_do_not_match_donor():
    cfg = "binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_uncompressed"
    for suffix in ("_bs_small", "_bs_mid", "_bs_xlarge", "_bs_small_smoke", "_smoke"):
        name = f"07-09-4146308-ETTh1-{cfg}{suffix}"
        assert not _run_dir_matches_config(name, "ETTh1", cfg), name


def test_unrelated_dataset_rejected():
    name = "07-08-4122625-exchange_rate-binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_uncompressed"
    cfg = "binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_uncompressed"
    assert not _run_dir_matches_config(name, "ETTh1", cfg)
