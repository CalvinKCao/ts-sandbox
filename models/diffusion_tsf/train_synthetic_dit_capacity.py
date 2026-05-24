"""Synthetic DiT capacity probe: can a DiT learn simple 1-variate forecasts?

Trains occupancy-map diffusion on on-the-fly synthetic series only (no RealTS,
no real datasets). Two task families per epoch:
  - linear: y = slope * t + intercept
  - periodic: amplitude * sin(2*pi*f*t + phase) + offset (cos mix optional)

Four architecture variants (patch size fixed at 8x8 unless overridden):
  - dit_tiny_no_guidance
  - dit_default_no_guidance
  - dit_large_no_guidance
  - dit_default_with_guidance  (branch-default DiT + iTransformer guidance path)

Run from repo root:
    python -m models.diffusion_tsf.train_synthetic_dit_capacity --smoke-test
    python -m models.diffusion_tsf.train_synthetic_dit_capacity
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.diffusion_tsf.config import DiffusionTSFConfig
from models.diffusion_tsf.diffusion_model import DiffusionTSF
from models.diffusion_tsf.guidance import iTransformerGuidance

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PATCH_SIZE_DEFAULT: Tuple[int, int] = (8, 8)


@dataclass
class VariantSpec:
    name: str
    dit_embed_dim: int
    dit_depth: int
    dit_num_heads: int
    use_guidance: bool = False
    dit_patch_size: Tuple[int, int] = PATCH_SIZE_DEFAULT
    dit_mlp_ratio: float = 4.0


VARIANTS: Dict[str, VariantSpec] = {
    "dit_tiny_no_guidance": VariantSpec(
        name="dit_tiny_no_guidance",
        dit_embed_dim=192,
        dit_depth=4,
        dit_num_heads=3,
        use_guidance=False,
    ),
    "dit_default_no_guidance": VariantSpec(
        name="dit_default_no_guidance",
        dit_embed_dim=384,
        dit_depth=8,
        dit_num_heads=6,
        use_guidance=False,
    ),
    "dit_large_no_guidance": VariantSpec(
        name="dit_large_no_guidance",
        dit_embed_dim=512,
        dit_depth=12,
        dit_num_heads=8,
        use_guidance=False,
    ),
    "dit_default_with_guidance": VariantSpec(
        name="dit_default_with_guidance",
        dit_embed_dim=384,
        dit_depth=8,
        dit_num_heads=6,
        use_guidance=True,
    ),
}


@dataclass
class RunConfig:
    lookback: int = 96
    horizon: int = 96
    lookback_overlap: int = 8
    image_height: int = 64
    max_scale: float = 6.0
    num_diffusion_steps: int = 1000
    ddim_steps: int = 30
    batch_size: int = 16
    lr: float = 2e-4
    itrans_lr: float = 1e-3
    itrans_pretrain_epochs: int = 40
    max_epochs: int = 150
    patience: int = 12
    samples_per_epoch: int = 512
    val_samples: int = 128
    success_mse: float = 0.04
    steps_per_epoch: Optional[int] = None
    emd_lambda: float = 0.1
    cfg_dropout: float = 0.0
    cfg_scale: float = 1.0


@dataclass
class VariantResult:
    variant: str
    patch_size: Tuple[int, int]
    use_guidance: bool
    dit_embed_dim: int
    dit_depth: int
    dit_num_heads: int
    status: str
    epochs_trained: int
    best_combined_mse: float
    best_linear_mse: float
    best_periodic_mse: float
    final_linear_mse: float
    final_periodic_mse: float
    n_params_m: float
    stop_reason: str = ""


class SyntheticTaskDataset(Dataset):
    """On-the-fly univariate series; returns (past, future) for diffusion training."""

    TASK_LINEAR = "linear"
    TASK_PERIODIC = "periodic"

    def __init__(
        self,
        n_samples: int,
        lookback: int,
        horizon: int,
        lookback_overlap: int,
        task: str,
        seed: int = 0,
    ):
        self.n_samples = n_samples
        self.lookback = lookback
        self.horizon = horizon
        self.lookback_overlap = lookback_overlap
        self.task = task
        self.seed = seed
        self.total_len = lookback + horizon

    def __len__(self) -> int:
        return self.n_samples

    def _series(self, idx: int) -> torch.Tensor:
        g = torch.Generator()
        g.manual_seed(self.seed + idx * 7919 + (1 if self.task == self.TASK_PERIODIC else 0))
        t = torch.arange(self.total_len, dtype=torch.float32)

        if self.task == self.TASK_LINEAR:
            slope = torch.empty(1).uniform_(-2.0, 2.0, generator=g).item()
            intercept = torch.empty(1).uniform_(-3.0, 3.0, generator=g).item()
            y = slope * t + intercept
        else:
            amp = torch.empty(1).uniform_(0.5, 4.0, generator=g).item()
            offset = torch.empty(1).uniform_(-2.0, 2.0, generator=g).item()
            phase = torch.empty(1).uniform_(0.0, 2.0 * math.pi, generator=g).item()
            freq = torch.empty(1).uniform_(0.05, 0.35, generator=g).item()
            mix = torch.empty(1).uniform_(0.0, 1.0, generator=g).item()
            wave = math.sin(phase) * torch.sin(2 * math.pi * freq * t)
            wave = wave + mix * torch.cos(2 * math.pi * freq * t + phase)
            y = amp * wave + offset

        return y

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self._series(idx)
        past = y[: self.lookback].unsqueeze(0)
        k = self.lookback_overlap
        future = y[self.lookback - k : self.lookback + self.horizon].unsqueeze(0)
        return past, future


def smoke_run_config() -> RunConfig:
    return RunConfig(
        lookback=48,
        horizon=16,
        lookback_overlap=4,
        image_height=32,
        max_scale=6.0,
        num_diffusion_steps=50,
        ddim_steps=8,
        batch_size=4,
        itrans_pretrain_epochs=1,
        max_epochs=3,
        patience=2,
        samples_per_epoch=16,
        val_samples=8,
        success_mse=0.5,
        steps_per_epoch=2,
        emd_lambda=0.05,
    )


class ZeroGuidance(nn.Module):
    """Guidance object that preserves the default guidance channel but zeros it."""

    def get_forecast(self, past: torch.Tensor, forecast_length: int) -> torch.Tensor:
        return past.new_zeros(past.shape[0], past.shape[1], forecast_length)


def get_itransformer_class():
    itrans_root = os.path.join(project_root, "models", "iTransformer")
    if itrans_root not in sys.path:
        sys.path.insert(0, itrans_root)
    itrans_path = os.path.join(itrans_root, "model", "iTransformer.py")
    spec = importlib.util.spec_from_file_location("synthetic_iTransformer_module", itrans_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load iTransformer from {itrans_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model


def create_itransformer(seq_len: int, pred_len: int, num_vars: int, dropout: float = 0.1) -> nn.Module:
    cfg = SimpleNamespace(
        seq_len=seq_len,
        pred_len=pred_len,
        output_attention=False,
        use_norm=True,
        d_model=512,
        d_ff=512,
        e_layers=4,
        n_heads=8,
        dropout=dropout,
        activation="gelu",
        embed="fixed",
        freq="h",
        factor=1,
        enc_in=num_vars,
        class_strategy="projection",
    )
    return get_itransformer_class()(cfg)


def instance_normalize_target(past: torch.Tensor, future: torch.Tensor, overlap: int) -> torch.Tensor:
    mean = past.mean(dim=-1, keepdim=True)
    std = past.std(dim=-1, keepdim=True) + 1e-8
    return (future[..., overlap:] - mean) / std


def train_itransformer_guidance(run_cfg: RunConfig, device: torch.device, seed: int) -> iTransformerGuidance:
    """Train the branch-default iTransformer on the same synthetic task family."""
    itrans = create_itransformer(
        seq_len=run_cfg.lookback,
        pred_len=run_cfg.horizon,
        num_vars=1,
        dropout=0.1,
    ).to(device)
    itrans.train()
    opt = torch.optim.AdamW(itrans.parameters(), lr=run_cfg.itrans_lr)

    for epoch in range(1, run_cfg.itrans_pretrain_epochs + 1):
        train_linear = SyntheticTaskDataset(
            run_cfg.samples_per_epoch // 2,
            run_cfg.lookback,
            run_cfg.horizon,
            run_cfg.lookback_overlap,
            SyntheticTaskDataset.TASK_LINEAR,
            seed=seed + epoch * 13,
        )
        train_periodic = SyntheticTaskDataset(
            run_cfg.samples_per_epoch // 2,
            run_cfg.lookback,
            run_cfg.horizon,
            run_cfg.lookback_overlap,
            SyntheticTaskDataset.TASK_PERIODIC,
            seed=seed + epoch * 17,
        )
        loader_linear = DataLoader(train_linear, batch_size=run_cfg.batch_size, shuffle=True, drop_last=True)
        loader_periodic = DataLoader(train_periodic, batch_size=run_cfg.batch_size, shuffle=True, drop_last=True)
        steps = max(len(loader_linear), len(loader_periodic))
        it_lin = iter(loader_linear)
        it_per = iter(loader_periodic)
        loss_sum = 0.0
        n_steps = 0
        for _ in range(steps):
            for loader_name in ("linear", "periodic"):
                if loader_name == "linear":
                    try:
                        past, future = next(it_lin)
                    except StopIteration:
                        it_lin = iter(loader_linear)
                        past, future = next(it_lin)
                else:
                    try:
                        past, future = next(it_per)
                    except StopIteration:
                        it_per = iter(loader_periodic)
                        past, future = next(it_per)
                past, future = past.to(device), future.to(device)
                target = future[..., run_cfg.lookback_overlap :]
                pred = itrans(past.permute(0, 2, 1), None, None, None).permute(0, 2, 1)
                mean = past.mean(dim=-1, keepdim=True)
                std = past.std(dim=-1, keepdim=True) + 1e-8
                loss = torch.nn.functional.mse_loss((pred - mean) / std, (target - mean) / std)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                loss_sum += loss.item()
                n_steps += 1
        logger.info(
            "[iTransformer guidance] epoch %d/%d mse=%.5f",
            epoch,
            run_cfg.itrans_pretrain_epochs,
            loss_sum / max(n_steps, 1),
        )

    return iTransformerGuidance(itrans)


def build_model(spec: VariantSpec, run_cfg: RunConfig, device: torch.device) -> DiffusionTSF:
    fut_width = run_cfg.horizon + run_cfg.lookback_overlap
    cfg = DiffusionTSFConfig(
        num_variables=1,
        lookback_length=run_cfg.lookback,
        forecast_length=fut_width,
        lookback_overlap=run_cfg.lookback_overlap,
        past_loss_weight=0.3,
        image_height=run_cfg.image_height,
        max_scale=run_cfg.max_scale,
        blur_kernel_size=11 if run_cfg.image_height <= 32 else 31,
        num_diffusion_steps=run_cfg.num_diffusion_steps,
        ddim_steps=run_cfg.ddim_steps,
        cfg_dropout=run_cfg.cfg_dropout,
        cfg_scale=run_cfg.cfg_scale,
        emd_lambda=run_cfg.emd_lambda,
        use_coordinate_channel=True,
        use_guidance_channel=True,
        disable_cross_attention=not spec.use_guidance,
        model_type="dit",
        dit_patch_size=spec.dit_patch_size,
        dit_embed_dim=spec.dit_embed_dim,
        dit_depth=spec.dit_depth,
        dit_num_heads=spec.dit_num_heads,
        dit_mlp_ratio=spec.dit_mlp_ratio,
        use_gradient_checkpointing=False,
        use_amp=False,
        unet_max_chunk_size=0,
    )
    guidance = ZeroGuidance()
    model = DiffusionTSF(cfg, guidance_model=guidance)
    return model.to(device)


@torch.no_grad()
def eval_task_mse(
    model: DiffusionTSF,
    task: str,
    run_cfg: RunConfig,
    device: torch.device,
    n_samples: int,
    seed: int,
) -> float:
    model.eval()
    ds = SyntheticTaskDataset(
        n_samples=n_samples,
        lookback=run_cfg.lookback,
        horizon=run_cfg.horizon,
        lookback_overlap=run_cfg.lookback_overlap,
        task=task,
        seed=seed,
    )
    # batch_size=1: univariate generate() can mis-broadcast denorm when B>1
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    k = run_cfg.lookback_overlap
    h = run_cfg.horizon
    total = 0.0
    count = 0
    for past, future in loader:
        past = past.to(device)
        future = future.to(device)
        out = model.generate(
            past,
            use_ddim=True,
            num_ddim_steps=run_cfg.ddim_steps,
            cfg_scale=run_cfg.cfg_scale,
        )
        pred = out["prediction_norm"]
        if pred.dim() == 2:
            pred = pred.unsqueeze(1)
        target = instance_normalize_target(past, future, k)
        if target.shape[-1] != h:
            raise RuntimeError(f"target horizon mismatch target={target.shape} horizon={h}")
        if pred.shape != target.shape:
            raise RuntimeError(f"shape mismatch pred={pred.shape} target={target.shape}")
        total += torch.nn.functional.mse_loss(pred, target, reduction="sum").item()
        count += pred.numel()
    return total / max(count, 1)


def train_one_variant(
    spec: VariantSpec,
    run_cfg: RunConfig,
    device: torch.device,
    seed: int,
) -> VariantResult:
    torch.manual_seed(seed)
    model = build_model(spec, run_cfg, device)
    if spec.use_guidance:
        model.set_guidance_model(train_itransformer_guidance(run_cfg, device, seed + 50_000))
    n_params = sum(p.numel() for p in model.noise_predictor.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=run_cfg.lr)

    best_combined = float("inf")
    best_linear = float("inf")
    best_periodic = float("inf")
    stale = 0
    stop_reason = "max_epochs"
    status = "failed"

    val_seed = seed + 10_000

    for epoch in range(1, run_cfg.max_epochs + 1):
        train_linear = SyntheticTaskDataset(
            run_cfg.samples_per_epoch // 2,
            run_cfg.lookback,
            run_cfg.horizon,
            run_cfg.lookback_overlap,
            SyntheticTaskDataset.TASK_LINEAR,
            seed=seed + epoch * 101,
        )
        train_periodic = SyntheticTaskDataset(
            run_cfg.samples_per_epoch // 2,
            run_cfg.lookback,
            run_cfg.horizon,
            run_cfg.lookback_overlap,
            SyntheticTaskDataset.TASK_PERIODIC,
            seed=seed + epoch * 103,
        )
        loader_linear = DataLoader(train_linear, batch_size=run_cfg.batch_size, shuffle=True, drop_last=True)
        loader_periodic = DataLoader(train_periodic, batch_size=run_cfg.batch_size, shuffle=True, drop_last=True)
        model.train()
        steps = run_cfg.steps_per_epoch
        if steps is None:
            steps = max(len(loader_linear), len(loader_periodic))
        loss_sum = 0.0
        n_steps = 0
        it_lin = iter(loader_linear)
        it_per = iter(loader_periodic)
        for _ in range(steps):
            try:
                past, future = next(it_lin)
            except StopIteration:
                it_lin = iter(loader_linear)
                past, future = next(it_lin)
            past, future = past.to(device), future.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(past, future)
            out["loss"].backward()
            opt.step()
            loss_sum += out["loss"].item()
            n_steps += 1

            try:
                past, future = next(it_per)
            except StopIteration:
                it_per = iter(loader_periodic)
                past, future = next(it_per)
            past, future = past.to(device), future.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(past, future)
            out["loss"].backward()
            opt.step()
            loss_sum += out["loss"].item()
            n_steps += 1

        mse_lin = eval_task_mse(model, SyntheticTaskDataset.TASK_LINEAR, run_cfg, device, run_cfg.val_samples, val_seed)
        mse_per = eval_task_mse(
            model, SyntheticTaskDataset.TASK_PERIODIC, run_cfg, device, run_cfg.val_samples, val_seed + 1
        )
        combined = max(mse_lin, mse_per)
        logger.info(
            "[%s] epoch %d/%d train_loss=%.4f linear_mse=%.5f periodic_mse=%.5f combined=%.5f",
            spec.name,
            epoch,
            run_cfg.max_epochs,
            loss_sum / max(n_steps, 1),
            mse_lin,
            mse_per,
            combined,
        )

        if combined < best_combined - 1e-6:
            best_combined = combined
            best_linear = mse_lin
            best_periodic = mse_per
            stale = 0
        else:
            stale += 1

        if mse_lin < run_cfg.success_mse and mse_per < run_cfg.success_mse:
            status = "success"
            stop_reason = "threshold"
            final_lin, final_per = mse_lin, mse_per
            return VariantResult(
                variant=spec.name,
                patch_size=spec.dit_patch_size,
                use_guidance=spec.use_guidance,
                dit_embed_dim=spec.dit_embed_dim,
                dit_depth=spec.dit_depth,
                dit_num_heads=spec.dit_num_heads,
                status=status,
                epochs_trained=epoch,
                best_combined_mse=best_combined,
                best_linear_mse=best_linear,
                best_periodic_mse=best_periodic,
                final_linear_mse=final_lin,
                final_periodic_mse=final_per,
                n_params_m=n_params,
                stop_reason=stop_reason,
            )

        if stale >= run_cfg.patience:
            stop_reason = "patience"
            break

    final_lin = eval_task_mse(
        model, SyntheticTaskDataset.TASK_LINEAR, run_cfg, device, run_cfg.val_samples, val_seed
    )
    final_per = eval_task_mse(
        model, SyntheticTaskDataset.TASK_PERIODIC, run_cfg, device, run_cfg.val_samples, val_seed + 1
    )
    if final_lin < run_cfg.success_mse and final_per < run_cfg.success_mse:
        status = "success"
    elif best_combined < run_cfg.success_mse * 2:
        status = "partial"
    else:
        status = "failed"

    return VariantResult(
        variant=spec.name,
        patch_size=spec.dit_patch_size,
        use_guidance=spec.use_guidance,
        dit_embed_dim=spec.dit_embed_dim,
        dit_depth=spec.dit_depth,
        dit_num_heads=spec.dit_num_heads,
        status=status,
        epochs_trained=epoch,
        best_combined_mse=best_combined,
        best_linear_mse=best_linear,
        best_periodic_mse=best_periodic,
        final_linear_mse=final_lin,
        final_periodic_mse=final_per,
        n_params_m=n_params,
        stop_reason=stop_reason,
    )


def parse_variants(raw: Optional[str]) -> List[str]:
    if not raw:
        return list(VARIANTS.keys())
    names = [x.strip() for x in raw.split(",") if x.strip()]
    bad = [n for n in names if n not in VARIANTS]
    if bad:
        raise ValueError(f"Unknown variants: {bad}; choices={list(VARIANTS.keys())}")
    return names


def write_results(results: List[VariantResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"capacity_results_{stamp}.json"
    csv_path = out_dir / f"capacity_results_{stamp}.csv"
    payload = [asdict(r) for r in results]
    for row in payload:
        row["patch_size"] = list(row["patch_size"])
    json_path.write_text(json.dumps(payload, indent=2))
    if payload:
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=payload[0].keys())
            w.writeheader()
            for row in payload:
                w.writerow(row)
    logger.info("Wrote %s and %s", json_path, csv_path)


def print_summary(results: List[VariantResult], patch_size: Tuple[int, int]) -> None:
    print("\n=== Synthetic DiT capacity summary ===")
    print(f"DiT spatial patch size (all variants): {patch_size}")
    for r in results:
        print(
            f"  {r.variant}: status={r.status} epochs={r.epochs_trained} "
            f"params={r.n_params_m:.2f}M guidance={r.use_guidance} "
            f"best_mse lin/per/comb={r.best_linear_mse:.4f}/{r.best_periodic_mse:.4f}/{r.best_combined_mse:.4f} "
            f"({r.stop_reason})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic DiT capacity probe (no RealTS).")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--results-dir", type=str, default="results/synthetic_dit_capacity")
    parser.add_argument("--variants", type=str, default=None, help="Comma-separated variant names")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--itrans-pretrain-epochs", type=int, default=None)
    parser.add_argument("--max-scale", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--success-mse", type=float, default=None)
    args = parser.parse_args()

    run_cfg = smoke_run_config() if args.smoke_test else RunConfig()
    if args.max_epochs is not None:
        run_cfg.max_epochs = args.max_epochs
    if args.patience is not None:
        run_cfg.patience = args.patience
    if args.samples_per_epoch is not None:
        run_cfg.samples_per_epoch = args.samples_per_epoch
    if args.val_samples is not None:
        run_cfg.val_samples = args.val_samples
    if args.itrans_pretrain_epochs is not None:
        run_cfg.itrans_pretrain_epochs = args.itrans_pretrain_epochs
    if args.max_scale is not None:
        run_cfg.max_scale = args.max_scale
    if args.batch_size is not None:
        run_cfg.batch_size = args.batch_size
    if args.success_mse is not None:
        run_cfg.success_mse = args.success_mse

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("device=%s smoke=%s", device, args.smoke_test)

    variant_names = parse_variants(args.variants)
    if len(variant_names) != 1:
        logger.info("Training %d variant(s): %s", len(variant_names), ", ".join(variant_names))

    results: List[VariantResult] = []
    for i, vname in enumerate(variant_names):
        spec = VARIANTS[vname]
        logger.info("Training variant %s (patch=%s guidance=%s)", vname, spec.dit_patch_size, spec.use_guidance)
        res = train_one_variant(spec, run_cfg, device, seed=args.seed + i * 1000)
        results.append(res)

    out_dir = Path(args.results_dir)
    write_results(results, out_dir)
    print_summary(results, PATCH_SIZE_DEFAULT)


if __name__ == "__main__":
    main()
