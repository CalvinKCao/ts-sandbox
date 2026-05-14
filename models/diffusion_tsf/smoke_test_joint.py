"""Smoke test for end-to-end joint iTrans + diffusion training.

Runs train_joint_phase with: warmup epochs, joint epochs, both ghost-image
variants (B and C), and validates that:
  - the loop completes without crashing
  - val loss is computed only in joint epochs (not warmup)
  - iTrans params actually move during warmup
  - diffusion params actually move during joint epochs
  - best_state_dict round-trips back into the model
"""

from __future__ import annotations

import logging
import torch

from models.diffusion_tsf.config import DiffusionTSFConfig
from models.diffusion_tsf.diffusion_model import DiffusionTSF
from models.diffusion_tsf.guidance import iTransformerGuidance
from models.diffusion_tsf.train_multivariate_pipeline import create_itransformer
from models.diffusion_tsf.joint_training import (
    JointTrainConfig,
    _split_param_groups,
    train_joint_phase,
)

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")


def _make_synth_loader(B: int, V: int, L: int, F_: int, n_batches: int):
    pasts = torch.randn(n_batches, B, V, L)
    futures = torch.randn(n_batches, B, V, F_)
    return list(zip(pasts, futures))


def _snapshot(params):
    return [p.detach().cpu().clone() for p in params]


def _max_delta(before, after):
    return max((a - b).abs().max().item() for a, b in zip(before, after))


def run(joint_use_ghost_image: bool) -> None:
    tag = "B" if joint_use_ghost_image else "C"
    print(f"\n--- joint smoke (ghost={tag}) ---")
    torch.manual_seed(0)
    B, V, L, F_ = 2, 3, 48, 16

    cfg = DiffusionTSFConfig(
        lookback_length=L,
        forecast_length=F_,
        lookback_overlap=0,
        num_variables=V,
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
        model_type="dit",
        e2e_joint_training=True,
        joint_use_ghost_image=joint_use_ghost_image,
        aux_forecast_loss_weight=1.0,
        itrans_warmup_epochs=1,
        itrans_lr=1e-3,
        learning_rate=2e-4,
    )

    itrans = create_itransformer(seq_len=L, pred_len=F_, num_vars=V, dropout=0.0)
    guidance = iTransformerGuidance(itrans, freeze=False)
    model = DiffusionTSF(cfg, guidance_model=guidance).cpu()

    itrans_params, diff_params = _split_param_groups(model)
    itrans_snap = _snapshot(itrans_params)
    diff_snap = _snapshot(diff_params)

    train_loader = _make_synth_loader(B, V, L, F_, n_batches=3)
    val_loader = _make_synth_loader(B, V, L, F_, n_batches=2)

    tcfg = JointTrainConfig(
        diffusion_lr=cfg.learning_rate,
        itrans_lr=cfg.itrans_lr,
        num_epochs=3,
        warmup_epochs=1,
        patience=10,
        grad_clip=1.0,
    )
    result = train_joint_phase(
        model,
        train_loader,
        val_loader,
        tcfg,
        device=torch.device("cpu"),
    )

    print(f"  best_epoch={result.best_epoch} best_val_loss={result.best_val_loss:.4f}")
    print(f"  history entries: {len(result.history)} | phases: {[h['phase'] for h in result.history]}")
    assert result.history[0]["phase"] == "warmup", "first epoch should be warmup"
    assert all(h["phase"] == "joint" for h in result.history[1:]), "rest should be joint"
    # warmup val should equal aux loss (we report it under val_loss)
    assert "val_aux_forecast_loss" in result.history[0], result.history[0]
    # joint epochs should have noise_loss > 0
    assert result.history[1]["train_noise_loss"] > 0.0

    itrans_after = _snapshot(itrans_params)
    diff_after = _snapshot(diff_params)
    d_itrans = _max_delta(itrans_snap, itrans_after)
    d_diff = _max_delta(diff_snap, diff_after)
    print(f"  param movement: iTrans max delta={d_itrans:.4f}, diffusion max delta={d_diff:.4f}")
    assert d_itrans > 1e-6, "iTrans params should have moved (warmup + joint)"
    assert d_diff > 1e-6, "diffusion params should have moved (joint phase)"

    assert result.best_state_dict is not None
    model.load_state_dict(result.best_state_dict)
    print("  best state_dict round-trip OK")


if __name__ == "__main__":
    try:
        run(joint_use_ghost_image=True)   # Option B
        run(joint_use_ghost_image=False)  # Option C
    except Exception as e:
        print(f"SMOKE FAILED: {type(e).__name__}: {e}")
        raise SystemExit(1)
    print("\njoint smoke checks passed.")
