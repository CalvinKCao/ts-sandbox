"""Focused checks for weather guid ablations: channel dropout, mixer mask, DiT pad, YAML."""

from __future__ import annotations

import inspect
import os
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def test_channel_dropout_helpers():
    from models.diffusion_tsf.channel_dropout import (
        n_keep_channels,
        sample_kept_channel_mask,
        token_drop_mask_from_channel_keep,
    )

    assert n_keep_channels(21, 0.7) == 6
    assert n_keep_channels(1, 0.7) == 1
    assert sample_kept_channel_mask(2, 8, 0.0, training=True, device=torch.device("cpu")) is None
    assert sample_kept_channel_mask(2, 8, 0.7, training=False, device=torch.device("cpu")) is None

    torch.manual_seed(0)
    keep = sample_kept_channel_mask(4, 10, 0.7, training=True, device=torch.device("cpu"))
    assert keep is not None
    assert keep.shape == (4, 10)
    assert int(keep.sum(dim=1).min()) == n_keep_channels(10, 0.7)

    var_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    keep_row = torch.tensor([[True, False, True]])
    dropped = token_drop_mask_from_channel_keep(keep_row, var_ids)
    assert dropped.tolist() == [[False, False, True, True, False, False]]
    print("ok helpers")


def _tiny_mixer(*, drop_frac: float = 0.0):
    from models.diffusion_tsf.patch_context_mixer import PatchContextMixer, PatchContextMixerConfig

    mixer = PatchContextMixer(
        PatchContextMixerConfig(
            d_in=8, d_model=8, d_out=8, n_layers=2, n_heads=2, d_ff=16, dropout=0.0,
            max_variates=8, max_past_patches=8,
        )
    )
    mixer.channel_dropout_drop_frac = drop_frac
    mixer.eval()
    return mixer


def test_mixer_mask_blocks_dropped_channels_in_eval():
    """Even if mixer is in eval, an explicit pad mask must block dropped channels."""
    mixer = _tiny_mixer(drop_frac=0.7)
    torch.manual_seed(1)
    tokens_a = torch.randn(2, 4, 3, 8)
    tokens_b = tokens_a.clone()
    tokens_b[:, 1, :, :] = torch.randn_like(tokens_b[:, 1, :, :])
    tokens_b[:, 3, :, :] = torch.randn_like(tokens_b[:, 3, :, :])

    var_ids = torch.arange(4).repeat_interleave(3)
    keep = torch.tensor([[True, False, True, False], [True, False, True, False]])
    from models.diffusion_tsf.channel_dropout import token_drop_mask_from_channel_keep
    pad = token_drop_mask_from_channel_keep(keep, var_ids)

    mixed_a, ids = mixer(tokens_a, src_key_padding_mask=pad)
    mixed_b, _ = mixer(tokens_b, src_key_padding_mask=pad)
    assert ids.tolist() == var_ids.tolist()
    assert mixed_a.shape == (2, 12, 8)

    kept = ~pad
    assert torch.allclose(mixed_a[kept], mixed_b[kept], atol=1e-5, rtol=1e-5)

    mixed_open_a, _ = mixer(tokens_a)
    mixed_open_b, _ = mixer(tokens_b)
    assert not torch.allclose(mixed_open_a[kept], mixed_open_b[kept], atol=1e-4, rtol=1e-4)
    print("ok mixer eval mask")


def test_drop_frac_zero_is_noop_in_eval():
    mixer = _tiny_mixer(drop_frac=0.0)
    tokens = torch.randn(1, 3, 2, 8)
    a, _ = mixer(tokens)
    b, _ = mixer(tokens)
    assert torch.equal(a, b)
    print("ok drop_frac=0")


def test_dit_pad_mask_ignores_dropped_keys():
    from models.diffusion_tsf.dit import _CrossAttention

    attn = _CrossAttention(dim=8, num_heads=2, drop=0.0)
    attn.eval()
    torch.manual_seed(2)
    q = torch.randn(2, 5, 8)
    ctx_a = torch.randn(2, 6, 8)
    ctx_b = ctx_a.clone()
    ctx_b[:, 1, :] = torch.randn_like(ctx_b[:, 1, :])
    ctx_b[:, 4, :] = torch.randn_like(ctx_b[:, 4, :])
    pad = torch.zeros(2, 6, dtype=torch.bool)
    pad[:, 1] = True
    pad[:, 4] = True
    out_a = attn(q, ctx_a, ctx_key_padding_mask=pad)
    out_b = attn(q, ctx_b, ctx_key_padding_mask=pad)
    assert torch.allclose(out_a, out_b, atol=1e-5, rtol=1e-5)
    out_open_a = attn(q, ctx_a)
    out_open_b = attn(q, ctx_b)
    assert not torch.allclose(out_open_a, out_open_b, atol=1e-4, rtol=1e-4)
    print("ok dit pad")


def test_post_mixer_cache_rejected_when_dropout_on():
    from models.diffusion_tsf.pipeline.train.cross_variate_cache import CrossVariateTokenCache

    class Cfg:
        disable_cross_attention = False
        channel_dropout_drop_frac = 0.7

    class Model:
        config = Cfg()

    try:
        CrossVariateTokenCache(
            model=Model(),
            device=torch.device("cpu"),
            storage="gpu",
            token_kind="mixed",
        )
    except RuntimeError as e:
        assert "post-mixer" in str(e)
        print("ok cache fail-fast")
        return
    raise AssertionError("expected post-mixer cache to fail when drop_frac>0")


def test_yaml_leaves():
    from models.diffusion_tsf.pipeline.config import load_experiment_config

    root = os.path.join(REPO, "configs")
    p24 = load_experiment_config(
        os.path.join(root, "binary_window_norm_patch_refine_canvas128_p32x6_weather_allv_p24_fixedhp.yaml")
    )
    chmask = load_experiment_config(
        os.path.join(root, "binary_window_norm_patch_refine_canvas128_p32x6_weather_allv_chmask70_fixedhp.yaml")
    )
    itrans = load_experiment_config(
        os.path.join(root, "binary_window_norm_patch_refine_canvas128_p32x6_weather_allv_itrans_guid_fixedhp.yaml")
    )
    assert p24["experiment"]["dataset"] == "weather"
    assert int(p24["experiment"]["mmpd_patch_size"]) == 24
    assert float(chmask["experiment"].get("channel_dropout_drop_frac", 0)) == 0.7
    assert itrans["experiment"]["guidance_type"] == "itransformer"
    by_name = {e["phase"]: e for e in p24["phases"]}
    assert by_name["patch_guidance_finetune_hp"]["n_trials"] == 1
    assert by_name["diffusion_coarse_finetune_hp"]["search_space"] == "fixed"
    assert "itrans_finetune_hp" in {e["phase"] for e in itrans["phases"]}
    assert "patch_guidance_finetune_hp" not in {e["phase"] for e in itrans["phases"]}
    weather = p24["experiment"]["data_subset_by_dataset"]["weather"]
    assert int(weather["train_max_windows"]) == 5051
    assert int(weather["val_max_windows"]) == 1435
    print("ok yaml")


def test_itrans_finetune_applies_window_caps():
    from models.diffusion_tsf.train_multivariate_pipeline import (
        run_itransformer_finetune_hp_tuning,
        run_patch_guidance_finetune_hp_tuning,
    )

    it_src = inspect.getsource(run_itransformer_finetune_hp_tuning)
    pg_src = inspect.getsource(run_patch_guidance_finetune_hp_tuning)
    assert "random_window_subset" in it_src
    assert "train_max_windows" in it_src
    assert "val_max_windows" in it_src
    assert "was ignored" in it_src
    assert "random_window_subset" in pg_src
    print("ok itrans window caps")


def main():
    test_channel_dropout_helpers()
    test_mixer_mask_blocks_dropped_channels_in_eval()
    test_drop_frac_zero_is_noop_in_eval()
    test_dit_pad_mask_ignores_dropped_keys()
    test_post_mixer_cache_rejected_when_dropout_on()
    test_yaml_leaves()
    test_itrans_finetune_applies_window_caps()
    print("all passed")


if __name__ == "__main__":
    main()
