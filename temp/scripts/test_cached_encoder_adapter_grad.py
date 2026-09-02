"""Adapter stays in-graph with cached frozen encoder tokens; encoder does not."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from models.diffusion_tsf.diffusion_model import DiffusionTSF
from models.diffusion_tsf.guidance import iTransformerTokenAdapter
from models.diffusion_tsf.pipeline.train.cross_variate_cache import CrossVariateTokenCache


class FrozenEncoderStub(nn.Module):
    def __init__(self, d_in: int = 4, d_model: int = 512):
        super().__init__()
        self.lin = nn.Linear(d_in, d_model)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def train(self, mode: bool = True):
        self.training = False
        return self

    @torch.no_grad()
    def get_encoder_tokens(self, past: torch.Tensor) -> torch.Tensor:
        feat = past.mean(dim=-1, keepdim=True).expand(-1, -1, 4)
        return self.lin(feat)


class CtxStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            disable_cross_attention=False,
            itrans_d_model=512,
            use_amp=False,
        )
        self.guidance_model = FrozenEncoderStub()
        self.context_encoder = iTransformerTokenAdapter(
            d_model=512, context_dim=32, max_variates=8, dropout=0.1,
        )
        self._ctx_token_variate_ids = None
        self._ctx_key_padding_mask = None


CtxStub._encode_frozen_encoder_tokens = DiffusionTSF._encode_frozen_encoder_tokens
CtxStub._adapt_encoder_tokens = DiffusionTSF._adapt_encoder_tokens
CtxStub._get_cross_variate_context = DiffusionTSF._get_cross_variate_context
CtxStub._resolve_cross_variate_context = DiffusionTSF._resolve_cross_variate_context


def main() -> None:
    torch.manual_seed(0)
    model = CtxStub()
    device = torch.device("cpu")
    past = torch.randn(2, 3, 16)

    cache = CrossVariateTokenCache(
        model=model, device=device, storage="gpu", token_kind="mixed",
    )
    cache.reserve(2)
    cache.add(past)
    cached = cache.get(past)
    assert cached.tokens.shape == (2, 3, 512), cached.tokens.shape

    model.train()
    ctx = model._resolve_cross_variate_context(
        past, past, cached.tokens, cached.token_variate_ids,
    )
    assert ctx.shape == (2, 3, 32), ctx.shape
    ctx.sum().backward()

    adapter_grad = model.context_encoder.proj.weight.grad
    if adapter_grad is None or float(adapter_grad.abs().sum()) == 0.0:
        raise AssertionError("adapter.proj.weight.grad must be nonzero with cached encoder tokens")
    if model.guidance_model.lin.weight.grad is not None:
        raise AssertionError("frozen encoder grads must stay None")
    if any(p.requires_grad for p in model.guidance_model.parameters()):
        raise AssertionError("encoder requires_grad must stay False")

    model.eval()
    with torch.no_grad():
        live = model._get_cross_variate_context(past, past)
    assert live.shape == (2, 3, 32), live.shape
    print("ok: adapter grad nonzero; encoder grad None; eval encode-once+adapt")


if __name__ == "__main__":
    main()
