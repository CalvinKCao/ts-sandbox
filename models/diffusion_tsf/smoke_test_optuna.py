"""Smoke test for optuna_search_joint_phase end-to-end.

Builds a tiny DiffusionTSF, runs 2 Optuna trials with a fake DataLoader,
verifies that best_state_dict is captured and a sensible best_params dict
is returned.
"""

from __future__ import annotations

import logging
import torch

from models.diffusion_tsf.config import DiffusionTSFConfig
from models.diffusion_tsf.diffusion_model import DiffusionTSF
from models.diffusion_tsf.guidance import iTransformerGuidance
from models.diffusion_tsf.train_multivariate_pipeline import create_itransformer
from models.diffusion_tsf.joint_training import (
    JointSearchConfig,
    optuna_search_joint_phase,
)

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")


def _make_loader(B: int, V: int, L: int, F_: int, n_batches: int):
    pasts = torch.randn(n_batches, B, V, L)
    futures = torch.randn(n_batches, B, V, F_)
    return list(zip(pasts, futures))


def factory():
    cfg = DiffusionTSFConfig(
        lookback_length=48, forecast_length=16, lookback_overlap=0,
        num_variables=3, image_height=32, blur_kernel_size=11,
        num_diffusion_steps=50, ddim_steps=4, cfg_dropout=0.0,
        use_coordinate_channel=True, use_guidance_channel=True,
        dit_patch_size=(8, 8), dit_embed_dim=64, dit_depth=4,
        dit_num_heads=4, dit_mlp_ratio=2.0, model_type="dit",
        e2e_joint_training=True, joint_use_ghost_image=False,  # Option C
        aux_forecast_loss_weight=1.0, itrans_warmup_epochs=1,
    )
    itrans = create_itransformer(seq_len=48, pred_len=16, num_vars=3, dropout=0.0)
    guidance = iTransformerGuidance(itrans, freeze=False)
    return DiffusionTSF(cfg, guidance_model=guidance)


def main():
    torch.manual_seed(0)
    train_loader = _make_loader(2, 3, 48, 16, n_batches=2)
    val_loader = _make_loader(2, 3, 48, 16, n_batches=1)

    search_cfg = JointSearchConfig(
        n_trials=2,
        diffusion_lr_min=1e-5, diffusion_lr_max=1e-3,
        itrans_lr_min=1e-5, itrans_lr_max=1e-3,
        num_epochs=3, warmup_epochs=1,
        patience=10, grad_clip=1.0, use_amp=False,
        median_pruner_warmup=99,  # disable pruning for this tiny smoke
    )

    best_params, best_state, study = optuna_search_joint_phase(
        factory, train_loader, val_loader, search_cfg,
        device=torch.device("cpu"),
        study_name="smoke_optuna",
    )
    print(f"\nbest_params: {best_params}")
    print(f"best_state_dict tensors: {len(best_state) if best_state else 0}")
    print(f"study.best_value: {study.best_value:.4f}")
    print(f"trials run: {len(study.trials)}")
    assert len(study.trials) == 2
    assert best_state is not None
    assert "diffusion_lr" in best_params
    assert "itrans_lr" in best_params
    print("\noptuna smoke OK.")


if __name__ == "__main__":
    main()
