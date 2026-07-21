"""Binary bit-flip diffusion for ordinal patch refinement (mirrors binary pipeline).

Trains FactorizedDiT with XOR noise + dual-head BCE (x0 + flip mask), conditioned
on the vertical-only coarse upscale (and lookback hist). Sampling uses the same
BinaryDiffusionScheduler.sample iterative reverse process as the main pipeline.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from models.diffusion_tsf.diffusion import BinaryDiffusionScheduler
from models.diffusion_tsf.dit import FactorizedDiT

# Match vertical_dual / fine finetune defaults.
NUM_TRAIN_STEPS = 1000
NUM_SAMPLE_STEPS = 20
SCHEDULE = "linear"
MIN_SNR_GAMMA = 2.0
PRED_TARGET = "x0"


def make_scheduler(device: torch.device) -> BinaryDiffusionScheduler:
    return BinaryDiffusionScheduler(
        num_steps=NUM_TRAIN_STEPS,
        beta_start=1e-5,
        beta_end=0.5,
        schedule_type=SCHEDULE,
        device=str(device),
    ).to(str(device))


def make_refiner(
    patch_h: int,
    device: torch.device,
    *,
    patch_w: int | None = None,
) -> FactorizedDiT:
    """DiT: noisy CDF in; dual heads out; cond = [naive_upscale, past_hist].

    Accepts rectangular crops (H=patch_h, W=patch_w). Token patch defaults to
    4x4 when both axes are divisible by 4.
    """
    if patch_w is None:
        patch_w = patch_h
    tok = (4, 4) if (patch_h % 4 == 0 and patch_w % 4 == 0) else (patch_h, patch_w)
    return FactorizedDiT(
        in_channels=1,
        cond_channels=2,
        out_channels=2,
        image_height=patch_h,
        patch_size=tok,
        embed_dim=384,
        depth=8,
        num_heads=6,
        context_dim=1,
    ).to(device)


def _min_snr_weights(scheduler: BinaryDiffusionScheduler, t: torch.Tensor) -> torch.Tensor:
    beta_t = scheduler.betas[t].clamp(1e-5, 1.0 - 1e-5)
    snr = ((1.0 - beta_t) ** 2) / (beta_t ** 2)
    weight = torch.minimum(snr, torch.full_like(snr, MIN_SNR_GAMMA)) / snr
    return weight


def diffusion_loss(
    model: FactorizedDiT,
    scheduler: BinaryDiffusionScheduler,
    x0: torch.Tensor,
    cond: torch.Tensor,
    *,
    t: torch.Tensor | None = None,
) -> torch.Tensor:
    """One binary diffusion train step. x0/cond: (N, C, H, W) with x0 in {0,1}."""
    n = x0.shape[0]
    device = x0.device
    if t is None:
        t = torch.randint(0, scheduler.num_steps, (n,), device=device)
    xt, zt = scheduler.add_noise(x0, t)
    out = model(xt, t, cond)
    x0_logits, zt_logits = out[:, :1], out[:, 1:2]
    per_x0 = F.binary_cross_entropy_with_logits(x0_logits, x0, reduction="none")
    per_zt = F.binary_cross_entropy_with_logits(zt_logits, zt, reduction="none")
    w = _min_snr_weights(scheduler, t).view(n, 1, 1, 1)
    return (w * (per_x0 + per_zt)).mean()


@torch.no_grad()
def sample_patches(
    model: FactorizedDiT,
    scheduler: BinaryDiffusionScheduler,
    cond: torch.Tensor,
    *,
    num_steps: int = NUM_SAMPLE_STEPS,
    sampler: str = "quad_t",
) -> torch.Tensor:
    """Iterative binary reverse sample. Returns hard {0,1} CDF patches (N,1,H,W)."""
    model.eval()
    device = cond.device
    n, _, h, w = cond.shape
    shape = (n, 1, h, w)

    def model_fn(xt: torch.Tensor, t_batch: torch.Tensor):
        out = model(xt, t_batch, cond)
        return out[:, :1], out[:, 1:2]

    return scheduler.sample(
        model_fn,
        shape=shape,
        num_steps=num_steps,
        device=str(device),
        sampler=sampler,
    )


def build_cond(naive_patch: torch.Tensor, hist_patch: torch.Tensor) -> torch.Tensor:
    """Stack coarse-upscale + lookback hist as DiT visual cond (N,2,H,W)."""
    if naive_patch.dim() == 3:
        naive_patch = naive_patch.unsqueeze(1)
    if hist_patch.dim() == 3:
        hist_patch = hist_patch.unsqueeze(1)
    if naive_patch.shape[1] != 1:
        naive_patch = naive_patch[:, :1]
    if hist_patch.shape[1] != 1:
        hist_patch = hist_patch[:, :1]
    return torch.cat([naive_patch, hist_patch], dim=1)
