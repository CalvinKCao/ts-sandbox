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
    *,
    e2e_joint_training: bool = False,
    joint_use_ghost_image: bool = True,
) -> DiffusionTSF:
    cfg = DiffusionTSFConfig(
        lookback_length=lookback,
        forecast_length=forecast,
        lookback_overlap=0,
        num_variables=num_variables,
        image_height=32,
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
        e2e_joint_training=e2e_joint_training,
        joint_use_ghost_image=joint_use_ghost_image,
    )

    itrans = create_itransformer(seq_len=lookback, pred_len=forecast, num_vars=num_variables, dropout=0.0)
    guidance = iTransformerGuidance(itrans, freeze=not e2e_joint_training)
    model = DiffusionTSF(cfg, guidance_model=guidance)
    return model


def smoke_one(
    model_type: str,
    *,
    e2e_joint_training: bool = False,
    joint_use_ghost_image: bool = True,
) -> None:
    tag = model_type
    if e2e_joint_training:
        tag += f" joint(ghost={'B' if joint_use_ghost_image else 'C'})"
    print(f"\n--- smoke: {tag} ---")
    torch.manual_seed(0)
    B, V, L, F_ = 2, 3, 48, 16
    model = build_model(
        model_type, V, L, F_,
        e2e_joint_training=e2e_joint_training,
        joint_use_ghost_image=joint_use_ghost_image,
    ).cpu()

    n_params = sum(p.numel() for p in model.noise_predictor.parameters())
    print(f"  backbone params: {n_params/1e6:.2f}M")

    past = torch.randn(B, V, L)
    future = torch.randn(B, V, F_)

    out = model(past, future)
    assert out["noise_pred"].shape == (B, V, model.config.image_height, F_), out["noise_pred"].shape
    assert out["loss"].dim() == 0
    print(f"  forward OK: loss={out['loss'].item():.4f}, noise_pred shape={tuple(out['noise_pred'].shape)}")

    if e2e_joint_training:
        assert "aux_forecast_loss" in out
        assert out["aux_forecast_loss"].item() > 0.0, "aux_forecast_loss should be nonzero"
        print(f"  aux_forecast_loss={out['aux_forecast_loss'].item():.4f}")

    out["loss"].backward()
    grad_norms = [p.grad.norm().item() for p in model.noise_predictor.parameters() if p.grad is not None]
    assert grad_norms, "no grads flowed to backbone"
    print(f"  grads flowed through {len(grad_norms)} backbone tensors; max grad norm={max(grad_norms):.4f}")

    itrans_params_with_grad = [p for p in model.guidance_model.model.parameters() if p.grad is not None]
    itrans_grad_norms = [p.grad.norm().item() for p in itrans_params_with_grad]
    if e2e_joint_training:
        assert itrans_grad_norms, "iTrans should receive gradient in joint mode"
        print(f"  iTrans grads: {len(itrans_grad_norms)} tensors, max norm={max(itrans_grad_norms):.4f}")
    else:
        assert not itrans_grad_norms, f"iTrans should be frozen, got {len(itrans_grad_norms)} grad tensors"
        print("  iTrans correctly frozen (no grads)")

    model.eval()
    with torch.no_grad():
        gen = model.generate(past, use_ddim=True, num_ddim_steps=4, cfg_scale=1.0)
    assert gen["prediction"].shape == (B, V, F_), gen["prediction"].shape
    print(f"  generate OK: prediction shape={tuple(gen['prediction'].shape)}")


if __name__ == "__main__":
    try:
        smoke_one("unet")
        smoke_one("dit")
        smoke_one("dit", e2e_joint_training=True, joint_use_ghost_image=True)   # Option B
        smoke_one("dit", e2e_joint_training=True, joint_use_ghost_image=False)  # Option C
        smoke_one("unet", e2e_joint_training=True, joint_use_ghost_image=True)
        smoke_one("unet", e2e_joint_training=True, joint_use_ghost_image=False)
    except Exception as e:
        print(f"SMOKE FAILED: {type(e).__name__}: {e}")
        raise SystemExit(1)
    print("\nall smoke checks passed.")
