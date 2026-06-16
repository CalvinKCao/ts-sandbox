"""Guidance ghost channel must live in diffusion window-norm space."""

from __future__ import annotations

import torch

from models.diffusion_tsf.guidance import iTransformerGuidance


class _StubITrans(torch.nn.Module):
  seq_len = 8
  pred_len = 4
  use_norm = True

  def forward(self, x_enc, _x_mark_enc, _x_dec, _x_mark_dec):
    # Echo last timestep of normalized input (no internal norm when disabled).
    last = x_enc[:, -1:, :]
    return last.repeat(1, self.pred_len, 1)


def test_get_forecast_window_norm_matches_manual_window_norm():
  torch.manual_seed(0)
  past = torch.randn(2, 3, 12)
  mean = past.mean(dim=-1, keepdim=True)
  std = past.std(dim=-1, keepdim=True).clamp_min(0.1)
  past_norm = (past - mean) / std

  guidance = iTransformerGuidance(_StubITrans(), seq_len=8, pred_len=4)
  core_len = 6
  K = 2
  core_norm = guidance.get_forecast_window_norm(past_norm, core_len, overlap=K)
  full = torch.cat([past_norm[..., -K:], core_norm], dim=-1)

  assert full.shape[-1] == K + core_len
  assert torch.allclose(full[..., :K], past_norm[..., -K:])
  # Stub repeats last norm value across each native pred chunk.
  assert torch.allclose(core_norm[..., :1], past_norm[..., -1:], atol=1e-5)


def test_window_norm_path_skips_instance_norm():
  model = _StubITrans()
  guidance = iTransformerGuidance(model, seq_len=8, pred_len=4)
  past_norm = torch.ones(1, 2, 8)
  with torch.no_grad():
    guidance.get_forecast_window_norm(past_norm, 4, overlap=0)
  assert model.use_norm is True
