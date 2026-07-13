"""Staged synthetic diffusion pretrain."""

from __future__ import annotations

import logging
import os
import time
from typing import Dict

import torch
from torch.utils.data import DataLoader, Subset

from models.diffusion_tsf.pipeline.train.checkpointing import EarlyStopping, save_checkpoint
from models.diffusion_tsf.pipeline.train.diffusion_loop import (
    train_diffusion_epoch,
    validate_diffusion_epoch,
)
from models.diffusion_tsf.realts import get_synthetic_dataloader

logger = logging.getLogger(__name__)


def pretrain_diffusion(
    best_params: Dict,
    guidance_checkpoint: str,
    n_samples: int,
    epochs: int,
    patience: int,
    checkpoint_dir: str,
    smoke_test: bool = False,
) -> str:
    """Train one staged diffusion checkpoint on synthetic data (not post-HP retrain)."""
    # Lazy: train_multivariate_pipeline re-exports this module at import time.
    from models.diffusion_tsf import train_multivariate_pipeline as m

    logger.info("=" * 60)
    logger.info("Staged synthetic diffusion pretrain (with patch_decoder guidance)")
    logger.info("Samples: %s, Epochs: %s, Patience: %s", n_samples, epochs, patience)
    logger.info("Params: %s", best_params)
    logger.info("=" * 60)

    device = m.get_device()

    lr = m.require_tuned_param(best_params, "learning_rate", "Diffusion pretraining")
    tuned_batch_size = m.require_tuned_param(best_params, "batch_size", "Diffusion pretraining")
    batch_size = tuned_batch_size

    guidance = m.load_wrapped_guidance(
        guidance_checkpoint,
        m.N_VARIATES,
        device,
        guidance_type="patch_decoder",
    )

    synth_cache = m.get_synth_cache_dir(checkpoint_dir=checkpoint_dir, smoke_test=smoke_test)
    n_val = 0 if smoke_test else min(n_samples // 10, 5000)
    epoch_cap = 1 if smoke_test else m.synthetic_epoch_capacity_pretrain_diffusion()
    synthetic_loader = get_synthetic_dataloader(
        batch_size=min(16, max(2, tuned_batch_size)),
        lookback_length=m.LOOKBACK_LENGTH,
        forecast_length=m.FORECAST_LENGTH,
        num_variables=m.N_VARIATES,
        num_samples=n_samples,
        num_workers=0 if smoke_test else 4,
        lookback_overlap=m.LOOKBACK_OVERLAP,
        cache_dir=synth_cache,
        skip_cross_var_aug=(m.N_VARIATES > 32),
        val_tail_n=n_val,
        synthetic_epoch_capacity=epoch_cap,
    )

    dataset = synthetic_loader.dataset
    train_subset = Subset(dataset, list(range(len(dataset) - n_val)))
    val_subset = Subset(dataset, list(range(len(dataset) - n_val, len(dataset))))
    batch_size = tuned_batch_size or (
        min(4, m.DIFFUSION_BATCH_SIZE) if smoke_test else m.DIFFUSION_BATCH_SIZE
    )
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if smoke_test else 4,
    )
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    model_kwargs = m.anchor_kwargs_from_params(best_params)
    for key in (
        "max_scale",
        "dit_dropout",
        "prediction_target",
        "loss_weighting",
        "use_ordinal_window_norm",
        "ordinal_tie_atol",
    ):
        if key in best_params:
            model_kwargs[key] = best_params[key]
    model = m.create_diffusion_model(
        guidance_model=guidance,
        **model_kwargs,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01,
    )

    early_stop = EarlyStopping(patience=patience)
    best_val_loss = float("inf")
    ckpt_path = os.path.join(checkpoint_dir, "pretrained_diffusion.pt")

    for epoch in range(epochs):
        t0 = time.time()
        train_loss = train_diffusion_epoch(
            model,
            train_loader,
            device,
            optimizer,
            set_loader_mode=m._set_ordinal_loader_mode,
            set_training_epoch=m.set_realts_training_epoch,
            epoch=epoch,
        )
        val_loss = validate_diffusion_epoch(
            model,
            val_loader,
            device,
            set_loader_mode=m._set_ordinal_loader_mode,
        )

        scheduler.step()
        logger.info(
            "[Diffusion] Epoch %d/%d | Train: %.4f | Val: %.4f | LR: %.2e | Time: %.1fs",
            epoch + 1,
            epochs,
            train_loss,
            val_loss,
            scheduler.get_last_lr()[0],
            time.time() - t0,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                optimizer,
                epoch,
                train_loss,
                val_loss,
                {"diffusion_params": best_params, "guidance_checkpoint": guidance_checkpoint},
                ckpt_path,
            )
            logger.info("  -> New best! Saved to %s", ckpt_path)

        if early_stop(val_loss):
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    logger.info("Diffusion pretraining complete. Best val loss: %.4f", best_val_loss)
    return ckpt_path
