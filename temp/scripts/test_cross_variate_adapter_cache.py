#!/usr/bin/env python3
"""Cached encoder tokens stay frozen; adapter grads stay live."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline.train.cross_variate_cache import CrossVariateTokenCache
from models.diffusion_tsf.train_multivariate_pipeline import (
    create_diffusion_model,
    create_itransformer,
    wrap_itrans_guidance,
)


def _tiny_state() -> PipelineState:
    s = PipelineState(experiment_name="adapter-cache-smoke", dataset="ETTh1", n_variates=3)
    s.lookback_length = 16
    s.forecast_length = 12
    s.lookback_overlap = 2
    s.itrans_lookback_length = 16
    s.itrans_d_model = 32
    s.itrans_d_ff = 64
    s.itrans_e_layers = 1
    s.itrans_n_heads = 4
    s.image_height = 8
    s.coarse_image_height = 8
    s.fine_image_height = 8
    s.dit_patch_size = (4, 2)
    s.dit_cond_patch_size = (4, 2)
    s.dit_embed_dim = 32
    s.dit_depth = 2
    s.dit_num_heads = 4
    s.disable_cross_attention = False
    s.use_gradient_checkpointing = False
    s.use_amp = False
    s.torch_compile = False
    s.use_patch_refine_stage = False
    s.diffusion_stage = "coarse"
    s.guidance_type = "itransformer"
    s.context_embedding_dim = 16
    return s


def _adapter_grad_norm(model) -> float:
    grads = [
        p.grad.detach().float().norm().item()
        for p in model.context_encoder.parameters()
        if p.grad is not None
    ]
    return float(sum(grads))


def test_removed_token_kinds_fail_fast():
    state = _tiny_state()
    device = torch.device("cpu")
    itrans = create_itransformer(state, num_vars=state.n_variates)
    guidance = wrap_itrans_guidance(itrans, state)
    model = create_diffusion_model(state, guidance_model=guidance, diffusion_stage="coarse")
    model.to(device)
    for kind in ("mixed", "pre_mixer"):
        try:
            CrossVariateTokenCache(model=model, device=device, storage="gpu", token_kind=kind)
        except ValueError as exc:
            assert "removed" in str(exc) or "raw" in str(exc)
        else:
            raise AssertionError(f"token_kind={kind!r} should fail fast")


def test_cache_matches_encoder_and_adapter_grads():
    torch.manual_seed(0)
    state = _tiny_state()
    device = torch.device("cpu")
    itrans = create_itransformer(state, num_vars=state.n_variates)
    guidance = wrap_itrans_guidance(itrans, state)
    model = create_diffusion_model(state, guidance_model=guidance, diffusion_stage="coarse")
    model.to(device)
    model.train()

    B, V, L = 2, state.n_variates, state.lookback_length
    past = torch.randn(B, V, L, device=device)
    past_norm = past

    live_enc = model._encode_frozen_encoder_tokens(past).detach()
    cache = CrossVariateTokenCache(
        model=model, device=device, storage="gpu", token_kind="raw",
    )
    cache.reserve(B)
    cache.add(past)
    cached = cache.get(past)
    assert cached.tokens.shape[-1] == state.itrans_d_model
    assert cached.tokens.shape[-1] != model.config.context_embedding_dim
    assert torch.allclose(cached.tokens, live_enc, atol=1e-5, rtol=1e-5)

    model.zero_grad(set_to_none=True)
    live_ctx = model._adapt_encoder_tokens(live_enc)
    live_ctx.pow(2).mean().backward()
    live_g = _adapter_grad_norm(model)
    assert live_g > 0.0, f"live adapter grad was {live_g}"

    model.zero_grad(set_to_none=True)
    cached_ctx = model._resolve_cross_variate_context(
        past, past_norm, cached.tokens, cached.token_variate_ids,
    )
    cached_ctx.pow(2).mean().backward()
    cached_g = _adapter_grad_norm(model)
    assert cached_g > 0.0, (
        f"cached adapter grad was {cached_g} (adapter was frozen into the cache)"
    )


if __name__ == "__main__":
    test_removed_token_kinds_fail_fast()
    test_cache_matches_encoder_and_adapter_grads()
    print("ok")
