"""
Time one training epoch: pixel DiffusionTSF vs latent LatentDiffusionTSF on the same
synthetic RealTS loader (matched lookback / forecast / overlap / image height / U-Net
widths). Intended for Slurm or local GPU.

Univariate (num_variables=1) only: the VAE is 1-channel; multivariate CI code flattens
batches — this benchmark is a fair per-sample proxy for backbone cost.

Run from repo root:
  python -m models.diffusion_tsf.benchmark_pixel_vs_latent_epoch
  python -m models.diffusion_tsf.benchmark_pixel_vs_latent_epoch --num-samples 512 --batch-size 8
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.diffusion_tsf.config import DiffusionTSFConfig, LatentDiffusionConfig
from models.diffusion_tsf.dataset import get_synthetic_dataloader
from models.diffusion_tsf.diffusion_model import DiffusionTSF
from models.diffusion_tsf.guidance import LinearRegressionGuidance
from models.diffusion_tsf.latent_diffusion_model import LatentDiffusionTSF
from models.diffusion_tsf.vae import TimeSeriesVAE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _autocast_ctx(device: torch.device, amp: bool):
    if not amp or device.type != "cuda":
        return nullcontext()
    try:
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    except AttributeError:
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool,
) -> Tuple[int, float]:
    model.train()
    n_batches = 0
    _sync_cuda()
    t0 = time.perf_counter()
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    for past, future in loader:
        past = past.to(device, non_blocking=True)
        future = future.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_ctx(device, amp):
            out = model(past, future)
            loss = out["loss"]
        if amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        n_batches += 1
    _sync_cuda()
    elapsed = time.perf_counter() - t0
    return n_batches, elapsed


def _warmup(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool,
    n_batches: int,
) -> None:
    it = iter(loader)
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    for _ in range(n_batches):
        try:
            past, future = next(it)
        except StopIteration:
            it = iter(loader)
            past, future = next(it)
        past = past.to(device, non_blocking=True)
        future = future.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_ctx(device, amp):
            loss = model(past, future)["loss"]
        if amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
    _sync_cuda()


def _free_cuda(*objs):
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser(description="Benchmark one epoch pixel vs latent diffusion")
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Default 1: unified L+F at H=128×~1224 is VRAM-heavy on ~48GB GPUs.",
    )
    p.add_argument("--num-samples", type=int, default=512, help="Synthetic dataset size (one epoch)")
    p.add_argument("--warmup-batches", type=int, default=3)
    p.add_argument("--lookback-length", type=int, default=1024)
    p.add_argument("--forecast-length", type=int, default=192)
    p.add_argument("--lookback-overlap", type=int, default=8)
    p.add_argument("--image-height", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action="store_true", help="bf16 autocast (if CUDA)")
    p.add_argument("--lr", type=float, default=2e-4)
    args = p.parse_args()

    L = args.lookback_length
    F = args.forecast_length
    K = args.lookback_overlap
    H = args.image_height

    if K % 4 != 0:
        logger.warning("lookback_overlap should be divisible by 4 for latent VAE downsampling; got %s", K)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("No CUDA — timings will not reflect GPU training.")

    torch.manual_seed(args.seed)

    guidance = LinearRegressionGuidance()

    common_unet = dict(
        unet_channels=[64, 128, 256],
        attention_levels=[2],
        num_res_blocks=2,
        use_hybrid_condition=True,
        use_gradient_checkpointing=False,
        use_amp=False,
    )

    pixel_cfg = DiffusionTSFConfig(
        num_variables=1,
        lookback_length=L,
        forecast_length=F + K,
        lookback_overlap=K,
        past_loss_weight=0.3,
        image_height=H,
        unified_time_axis=True,
        use_coordinate_channel=True,
        use_time_ramp=False,
        use_time_sine=False,
        use_value_channel=False,
        use_guidance_channel=True,
        num_diffusion_steps=1000,
        model_type="unet",
        emd_lambda=0.0,
        use_monotonicity_loss=False,
        **common_unet,
    )
    latent_cfg = LatentDiffusionConfig(
        num_variables=1,
        lookback_length=L,
        forecast_length=F + K,
        lookback_overlap=K,
        past_loss_weight=0.3,
        image_height=H,
        unified_time_axis=True,
        use_coordinate_channel=True,
        use_time_ramp=False,
        use_time_sine=False,
        use_value_channel=False,
        use_guidance_channel=True,
        seasonal_period=24,
        latent_channels=4,
        num_diffusion_steps=1000,
        model_type="unet",
        emd_lambda=0.0,
        cfg_scale=1.0,
        **common_unet,
    )

    loader = get_synthetic_dataloader(
        num_samples=args.num_samples,
        lookback_length=L,
        forecast_length=F,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        seed=args.seed,
        num_variables=1,
        lookback_overlap=K,
        skip_cross_var_aug=True,
    )

    # Pixel model alone first — loading latent + VAE at the same time OOMs ~48GB GPUs.
    pixel_model = DiffusionTSF(pixel_cfg, guidance_model=guidance).to(device)
    opt_p = torch.optim.AdamW(pixel_model.parameters(), lr=args.lr)
    n_params_pixel = sum(x.numel() for x in pixel_model.parameters())

    _warmup(pixel_model, loader, opt_p, device, args.amp, args.warmup_batches)
    n_p, t_p = _run_epoch(pixel_model, loader, opt_p, device, args.amp)
    _free_cuda(pixel_model, opt_p)

    vae = TimeSeriesVAE(latent_channels=latent_cfg.latent_channels).to(device)
    latent_model = LatentDiffusionTSF(latent_cfg, vae, guidance_model=guidance).to(device)
    opt_l = torch.optim.AdamW(
        [x for x in latent_model.parameters() if x.requires_grad],
        lr=args.lr,
    )
    n_params_latent_train = sum(x.numel() for x in latent_model.parameters() if x.requires_grad)
    n_params_vae = sum(x.numel() for x in vae.parameters())

    _warmup(latent_model, loader, opt_l, device, args.amp, args.warmup_batches)
    n_l, t_l = _run_epoch(latent_model, loader, opt_l, device, args.amp)

    summary = {
        "device": str(device),
        "amp_bf16": bool(args.amp and device.type == "cuda"),
        "batch_size": args.batch_size,
        "num_samples_epoch": args.num_samples,
        "batches_per_epoch_pixel": n_p,
        "batches_per_epoch_latent": n_l,
        "lookback_length": L,
        "forecast_length_data": F,
        "lookback_overlap": K,
        "future_width_2d": F + K,
        "image_height": H,
        "latent_image_height": latent_cfg.latent_image_height,
        "seconds_pixel_epoch": round(t_p, 4),
        "seconds_latent_epoch": round(t_l, 4),
        "pixel_s_per_batch": round(t_p / max(n_p, 1), 6),
        "latent_s_per_batch": round(t_l / max(n_l, 1), 6),
        "speedup_pixel_over_latent": round(t_l / max(t_p, 1e-9), 4),
        "params_pixel_total": n_params_pixel,
        "params_latent_trainable": n_params_latent_train,
        "params_vae_frozen": n_params_vae,
        "note": "Univariate RealTS; unified L+F; LinearRegressionGuidance; matched U-Net widths. Pixel then latent loaded sequentially on GPU.",
    }

    print("\n" + "=" * 72)
    print("BENCHMARK SUMMARY (one epoch, same DataLoader)")
    print("=" * 72)
    print(json.dumps(summary, indent=2))
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
