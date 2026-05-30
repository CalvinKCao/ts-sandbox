"""Smoke test for FactorizedDiT + binary diffusion.

Run from repo root with the project venv active:
    python -m models.diffusion_tsf.smoke_test_dit
"""

from __future__ import annotations

import logging

import torch

from models.diffusion_tsf.config import DiffusionTSFConfig
from models.diffusion_tsf.diffusion_model import DiffusionTSF
from models.diffusion_tsf.guidance import iTransformerGuidance
from models.diffusion_tsf.train_multivariate_pipeline import create_itransformer


logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")


def build_model(
    num_variables: int,
    lookback: int,
    forecast: int,
    use_deterministic_anchor_loss: bool = False,
    use_dual_scale: bool = False,
) -> DiffusionTSF:
    cfg = DiffusionTSFConfig(
        lookback_length=lookback,
        forecast_length=forecast,
        lookback_overlap=0,
        num_variables=num_variables,
        image_height=16 if use_dual_scale else 32,
        cfg_dropout=0.0,
        use_coordinate_channel=True,
        use_guidance_channel=True,
        dit_patch_size=(8, 8),
        dit_embed_dim=64,
        dit_depth=4,
        dit_num_heads=4,
        dit_mlp_ratio=2.0,
        model_type="dit",
        diffusion_type="binary",
        binary_num_steps=32,
        binary_sample_steps=4,
        use_deterministic_anchor_loss=use_deterministic_anchor_loss,
        deterministic_anchor_lambda=0.9,
        use_dual_scale=use_dual_scale,
        dual_scale_fine_weight=0.5,
    )

    itrans = create_itransformer(seq_len=lookback, pred_len=forecast, num_vars=num_variables, dropout=0.0)
    guidance = iTransformerGuidance(itrans)
    return DiffusionTSF(cfg, guidance_model=guidance)


def smoke_forward_backward() -> None:
    print("\n--- smoke: dit binary forward/backward ---")
    torch.manual_seed(0)
    B, V, L, F_ = 2, 3, 48, 16
    model = build_model(V, L, F_).cpu()

    n_params = sum(p.numel() for p in model.noise_predictor.parameters())
    print(f"  backbone params: {n_params/1e6:.2f}M")

    past = torch.randn(B, V, L)
    future = torch.randn(B, V, F_)

    out = model(past, future)
    assert out["noise_pred"].shape == (B, V, model.config.image_height, F_), out["noise_pred"].shape
    assert out["x0_pred"].shape == (B, V, model.config.image_height, F_), out["x0_pred"].shape
    assert out["loss"].dim() == 0
    print(f"  forward OK: loss={out['loss'].item():.4f}, x0_pred shape={tuple(out['x0_pred'].shape)}")

    out["loss"].backward()
    grad_norms = [p.grad.norm().item() for p in model.noise_predictor.parameters() if p.grad is not None]
    assert grad_norms, "no grads flowed to backbone"
    print(f"  grads flowed through {len(grad_norms)} backbone tensors; max grad norm={max(grad_norms):.4f}")

    model.eval()
    with torch.no_grad():
        gen = model.generate(past, sampler="ddim", num_inference_steps=4, cfg_scale=1.0)
    assert gen["prediction"].shape == (B, V, F_), gen["prediction"].shape
    print(f"  generate OK: prediction shape={tuple(gen['prediction'].shape)}")


def smoke_guidance_required() -> None:
    print("\n--- smoke: guidance channel requires iTransformer ---")
    cfg = DiffusionTSFConfig(
        lookback_length=16,
        forecast_length=8,
        num_variables=1,
        use_guidance_channel=True,
    )
    try:
        DiffusionTSF(cfg, guidance_model=None)
    except ValueError as e:
        assert "guidance" in str(e).lower()
        print(f"  OK: {e}")
        return
    raise AssertionError("expected ValueError when guidance_model is None")


def smoke_binary_anchor() -> None:
    print("\n--- smoke: dit binary + anchor ---")
    torch.manual_seed(0)
    B, V, L, F_ = 2, 3, 48, 16
    model = build_model(V, L, F_, use_deterministic_anchor_loss=True).cpu()

    past = torch.randn(B, V, L)
    future = torch.randn(B, V, F_)

    out = model(past, future)
    assert out["anchor_loss"].dim() == 0
    out["loss"].backward()

    model.eval()
    with torch.no_grad():
        gen = model.generate(past, sampler="anchor", cfg_scale=1.0)
    assert gen["prediction"].shape == (B, V, F_), gen["prediction"].shape
    print(f"  anchor generate OK: prediction shape={tuple(gen['prediction'].shape)}")


def smoke_binary_dual_scale() -> None:
    print("\n--- smoke: dit binary dual scale ---")
    torch.manual_seed(0)
    B, V, L, F_ = 2, 3, 48, 16
    model = build_model(V, L, F_, use_deterministic_anchor_loss=True, use_dual_scale=True).cpu()
    H = model.config.image_height

    past = torch.randn(B, V, L)
    future = torch.randn(B, V, F_)

    out = model(past, future)
    assert out["x0_pred_coarse"].shape == (B, V, H, F_), out["x0_pred_coarse"].shape
    assert out["x0_pred_fine"].shape == (B, V, H, F_), out["x0_pred_fine"].shape
    out["loss"].backward()

    model.eval()
    with torch.no_grad():
        gen = model.generate(past, sampler="anchor", cfg_scale=1.0)
    assert gen["future_2d_coarse"].shape == (B, V, H, F_), gen["future_2d_coarse"].shape
    print(f"  dual anchor generate OK: prediction shape={tuple(gen['prediction'].shape)}")


if __name__ == "__main__":
    try:
        smoke_guidance_required()
        smoke_forward_backward()
        smoke_binary_anchor()
        smoke_binary_dual_scale()
    except Exception as e:
        print(f"SMOKE FAILED: {type(e).__name__}: {e}")
        raise SystemExit(1)
    print("\nall smoke checks passed.")
