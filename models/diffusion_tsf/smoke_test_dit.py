"""Smoke test for the FactorizedDiT backbone.

Builds DiffusionTSF with model_type='dit' (and 'unet' for parity), runs one
training step and one DDIM generation step on tiny tensors. Verifies output
shapes match the contract and that gradients flow through the DiT.

Run from repo root with the project venv active:
    python -m models.diffusion_tsf.smoke_test_dit
"""

from __future__ import annotations

import logging
import sys

import torch

from models.diffusion_tsf.config import DiffusionTSFConfig
from models.diffusion_tsf.diffusion_model import DiffusionTSF
from models.diffusion_tsf.guidance import iTransformerGuidance
from models.diffusion_tsf.train_multivariate_pipeline import create_itransformer


logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")


def build_model(
    model_type: str,
    num_variables: int,
    lookback: int,
    forecast: int,
    prediction_mode: str = "epsilon",
    diffusion_type: str = "gaussian",
    use_deterministic_anchor_loss: bool = False,
    use_dual_scale: bool = False,
) -> DiffusionTSF:
    cfg = DiffusionTSFConfig(
        lookback_length=lookback,
        forecast_length=forecast,
        lookback_overlap=0,
        num_variables=num_variables,
        image_height=16 if use_dual_scale else 32,
        blur_kernel_size=11,
        num_diffusion_steps=50,
        ddim_steps=4,
        cfg_dropout=0.0,
        use_coordinate_channel=True,
        use_guidance_channel=True,
        dit_patch_size=(8, 8),
        dit_embed_dim=64,
        dit_depth=4,
        dit_num_heads=4,
        dit_mlp_ratio=2.0,
        unet_channels=[32, 64],
        attention_levels=[1],
        model_type=model_type,
        prediction_mode=prediction_mode,
        diffusion_type=diffusion_type,
        binary_num_steps=32,
        binary_sample_steps=4,
        use_deterministic_anchor_loss=use_deterministic_anchor_loss,
        deterministic_anchor_lambda=0.9,
        deterministic_anchor_alpha=0.0 if diffusion_type == "binary" else 0.5,
        use_dual_scale=use_dual_scale,
        dual_scale_fine_weight=0.5,
    )

    itrans = create_itransformer(seq_len=lookback, pred_len=forecast, num_vars=num_variables, dropout=0.0)
    guidance = iTransformerGuidance(itrans)
    model = DiffusionTSF(cfg, guidance_model=guidance)
    return model


def smoke_one(model_type: str, prediction_mode: str = "epsilon") -> None:
    print(f"\n--- smoke: {model_type}, prediction_mode={prediction_mode} ---")
    torch.manual_seed(0)
    B, V, L, F_ = 2, 3, 48, 16
    model = build_model(model_type, V, L, F_, prediction_mode=prediction_mode).cpu()

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
        gen = model.generate(past, use_ddim=True, num_ddim_steps=4, cfg_scale=1.0)
    assert gen["prediction"].shape == (B, V, F_), gen["prediction"].shape
    print(f"  generate OK: prediction shape={tuple(gen['prediction'].shape)}")


def smoke_binary_anchor() -> None:
    print("\n--- smoke: dit, binary diffusion + anchor ---")
    torch.manual_seed(0)
    B, V, L, F_ = 2, 3, 48, 16
    model = build_model(
        "dit",
        V,
        L,
        F_,
        diffusion_type="binary",
        use_deterministic_anchor_loss=True,
    ).cpu()

    past = torch.randn(B, V, L)
    future = torch.randn(B, V, F_)

    out = model(past, future)
    assert out["noise_pred"].shape == (B, V, model.config.image_height, F_), out["noise_pred"].shape
    assert out["x0_pred"].shape == (B, V, model.config.image_height, F_), out["x0_pred"].shape
    assert out["anchor_loss"].dim() == 0
    assert out["loss"].dim() == 0
    print(
        "  forward OK: "
        f"loss={out['loss'].item():.4f}, anchor={out['anchor_loss'].item():.4f}"
    )

    out["loss"].backward()
    grad_norms = [p.grad.norm().item() for p in model.noise_predictor.parameters() if p.grad is not None]
    assert grad_norms, "no grads flowed to binary anchor backbone"
    print(f"  grads flowed through {len(grad_norms)} backbone tensors; max grad norm={max(grad_norms):.4f}")

    model.eval()
    with torch.no_grad():
        gen = model.generate(past, sampler="anchor", cfg_scale=1.0)
    assert gen["prediction"].shape == (B, V, F_), gen["prediction"].shape
    print(f"  anchor generate OK: prediction shape={tuple(gen['prediction'].shape)}")


def smoke_binary_dual_scale() -> None:
    print("\n--- smoke: dit, binary diffusion + dual scale ---")
    torch.manual_seed(0)
    B, V, L, F_ = 2, 3, 48, 16
    model = build_model(
        "dit",
        V,
        L,
        F_,
        diffusion_type="binary",
        use_deterministic_anchor_loss=True,
        use_dual_scale=True,
    ).cpu()

    past = torch.randn(B, V, L)
    future = torch.randn(B, V, F_)

    out = model(past, future)
    H = model.config.image_height
    assert out["x0_pred_coarse"].shape == (B, V, H, F_), out["x0_pred_coarse"].shape
    assert out["x0_pred_fine"].shape == (B, V, H, F_), out["x0_pred_fine"].shape
    assert out["anchor_loss"].dim() == 0
    assert out["loss"].dim() == 0
    print(
        "  forward OK: "
        f"loss={out['loss'].item():.4f}, coarse={out['coarse_loss'].item():.4f}, "
        f"fine={out['fine_loss'].item():.4f}"
    )

    out["loss"].backward()
    grad_norms = [p.grad.norm().item() for p in model.noise_predictor.parameters() if p.grad is not None]
    assert grad_norms, "no grads flowed to dual-scale backbone"
    print(f"  grads flowed through {len(grad_norms)} backbone tensors; max grad norm={max(grad_norms):.4f}")

    model.eval()
    with torch.no_grad():
        gen = model.generate(past, sampler="anchor", cfg_scale=1.0)
    assert gen["prediction"].shape == (B, V, F_), gen["prediction"].shape
    assert gen["future_2d_coarse"].shape == (B, V, H, F_), gen["future_2d_coarse"].shape
    assert gen["future_2d_fine"].shape == (B, V, H, F_), gen["future_2d_fine"].shape
    print(f"  dual anchor generate OK: prediction shape={tuple(gen['prediction'].shape)}")


if __name__ == "__main__":
    try:
        smoke_one("unet")
        smoke_one("dit")
        smoke_one("dit", prediction_mode="x0_cumsum")
        smoke_binary_anchor()
        smoke_binary_dual_scale()
    except Exception as e:
        print(f"SMOKE FAILED: {type(e).__name__}: {e}")
        raise SystemExit(1)
    print("\nall smoke checks passed.")
