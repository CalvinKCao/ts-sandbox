"""
Compare direct RealTS training vs wave-curriculum -> RealTS fine-tuning.

The curriculum arm first learns three simple forecasting families:
  1. linear
  2. sine
  3. linear + sine

Then both arms train on RealTS and are evaluated on a held-out RealTS test set.
One run corresponds to one `(arm, seed, lr, variant)` config; the Slurm wrapper
fans these out in parallel.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.diffusion_tsf.metrics import compute_metrics, log_metrics
from models.diffusion_tsf.realts import RealTS
from models.diffusion_tsf.train_synthetic_dit_capacity import (
    VARIANTS,
    VariantSpec,
    build_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


ARM_DIRECT = "direct_realts"
ARM_CURRICULUM = "wave_curriculum_realts"
WAVE_FAMILIES = ("linear", "sine", "linear_plus_sine")
NON_GUIDED_VARIANTS = sorted(name for name, spec in VARIANTS.items() if not spec.use_guidance)


@dataclass
class ExperimentConfig:
    variant_name: str = "dit_tiny_no_guidance"
    lookback: int = 96
    horizon: int = 96
    lookback_overlap: int = 8
    image_height: int = 64
    max_scale: float = 6.0
    num_diffusion_steps: int = 1000
    ddim_steps: int = 30
    batch_size: int = 16
    lr: float = 2e-4
    curriculum_stage_epochs: int = 30
    curriculum_min_epochs: int = 10
    curriculum_patience: int = 8
    curriculum_samples_per_epoch: int = 2048
    curriculum_val_samples: int = 256
    curriculum_success_mse: float = 0.08
    realts_train_samples: int = 4096
    realts_val_samples: int = 512
    realts_test_samples: int = 1024
    realts_epoch_capacity: int = 64
    realts_max_epochs: int = 160
    realts_min_epochs: int = 20
    realts_patience: int = 20
    realts_min_delta: float = 1e-5
    eval_ensemble: int = 3
    emd_lambda: float = 0.1
    cfg_dropout: float = 0.0
    cfg_scale: float = 1.0
    num_workers: int = 0


@dataclass
class StageSummary:
    stage: str
    epochs_trained: int
    best_val_mse: float
    stop_reason: str


class WaveCurriculumDataset(Dataset):
    def __init__(
        self,
        n_samples: int,
        lookback: int,
        horizon: int,
        lookback_overlap: int,
        family: str,
        seed: int,
    ):
        self.n_samples = n_samples
        self.lookback = lookback
        self.horizon = horizon
        self.lookback_overlap = lookback_overlap
        self.family = family
        self.seed = seed
        self.total_len = lookback + horizon

    def __len__(self) -> int:
        return self.n_samples

    def _make_series(self, idx: int) -> torch.Tensor:
        g = torch.Generator()
        g.manual_seed(self.seed + 1009 * idx)
        t = torch.linspace(0.0, 1.0, self.total_len, dtype=torch.float32)

        slope = torch.empty(1).uniform_(-2.5, 2.5, generator=g).item()
        intercept = torch.empty(1).uniform_(-2.0, 2.0, generator=g).item()
        amplitude = torch.empty(1).uniform_(0.3, 2.5, generator=g).item()
        frequency = torch.empty(1).uniform_(0.5, 6.0, generator=g).item()
        phase = torch.empty(1).uniform_(0.0, 2.0 * math.pi, generator=g).item()
        offset = torch.empty(1).uniform_(-1.5, 1.5, generator=g).item()

        linear = slope * t + intercept
        wave = amplitude * torch.sin(2.0 * math.pi * frequency * t + phase) + offset

        if self.family == "linear":
            y = linear
        elif self.family == "sine":
            y = wave
        elif self.family == "linear_plus_sine":
            y = linear + wave
        else:
            raise ValueError(f"Unknown wave family: {self.family}")

        return y

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self._make_series(idx)
        k = self.lookback_overlap
        past = y[: self.lookback].unsqueeze(0)
        future = y[self.lookback - k : self.lookback + self.horizon].unsqueeze(0)
        return past, future


def instance_normalize_target(past: torch.Tensor, future: torch.Tensor, overlap: int) -> torch.Tensor:
    mean = past.mean(dim=-1, keepdim=True)
    std = past.std(dim=-1, keepdim=True) + 1e-8
    return (future[..., overlap:] - mean) / std


def make_realts_loader(
    *,
    num_samples: int,
    cfg: ExperimentConfig,
    seed: int,
    shuffle: bool,
    batch_size: Optional[int] = None,
    cache_dir: Optional[str] = None,
    synthetic_epoch_capacity: int = 1,
) -> DataLoader:
    ds = RealTS(
        num_samples=num_samples,
        lookback_length=cfg.lookback,
        forecast_length=cfg.horizon,
        seed=seed,
        num_variables=1,
        cache_dir=cache_dir,
        lookback_overlap=cfg.lookback_overlap,
        synthetic_epoch_capacity=synthetic_epoch_capacity,
        val_tail_n=0,
    )
    return DataLoader(
        ds,
        batch_size=batch_size or cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle and (batch_size or cfg.batch_size) > 1,
    )


def make_wave_loader(
    *,
    family: str,
    num_samples: int,
    cfg: ExperimentConfig,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    ds = WaveCurriculumDataset(
        n_samples=num_samples,
        lookback=cfg.lookback,
        horizon=cfg.horizon,
        lookback_overlap=cfg.lookback_overlap,
        family=family,
        seed=seed,
    )
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )


def save_checkpoint(
    model,
    spec: VariantSpec,
    cfg: ExperimentConfig,
    checkpoint_path: Path,
    arm: str,
    seed: int,
    stage: str,
    epoch: int,
    best_val_mse: float,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "variant_name": spec.name,
            "variant_spec": asdict(spec),
            "experiment_config": asdict(cfg),
            "arm": arm,
            "seed": seed,
            "stage": stage,
            "epoch": epoch,
            "best_val_mse": best_val_mse,
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )


def train_one_epoch(model, loader: DataLoader, optimizer, device: torch.device) -> float:
    model.train()
    loss_sum = 0.0
    n_steps = 0
    for past, future in loader:
        if past.dim() == 2:
            past = past.unsqueeze(1)
        if future.dim() == 2:
            future = future.unsqueeze(1)
        past = past.to(device)
        future = future.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(past, future)
        out["loss"].backward()
        optimizer.step()
        loss_sum += out["loss"].item()
        n_steps += 1
    return loss_sum / max(n_steps, 1)


@torch.no_grad()
def eval_wave_loader(model, loader: DataLoader, cfg: ExperimentConfig, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for past, future in loader:
        if past.dim() == 2:
            past = past.unsqueeze(1)
        if future.dim() == 2:
            future = future.unsqueeze(1)
        past = past.to(device)
        future = future.to(device)
        pred = model.generate(
            past,
            use_ddim=True,
            num_ddim_steps=cfg.ddim_steps,
            cfg_scale=cfg.cfg_scale,
        )["prediction_norm"]
        if pred.dim() == 2:
            pred = pred.unsqueeze(1)
        target = instance_normalize_target(past, future, cfg.lookback_overlap)
        total += torch.nn.functional.mse_loss(pred, target, reduction="sum").item()
        count += pred.numel()
    return total / max(count, 1)


@torch.no_grad()
def evaluate_realts(model, loader: DataLoader, cfg: ExperimentConfig, device: torch.device, ensemble_size: int) -> Dict[str, Dict[str, float]]:
    model.eval()
    preds_single: List[torch.Tensor] = []
    preds_avg: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    K = cfg.lookback_overlap

    for batch_idx, (past, future) in enumerate(loader):
        if past.dim() == 2:
            past = past.unsqueeze(1)
        if future.dim() == 2:
            future = future.unsqueeze(1)
        past = past.to(device)
        result = model.generate(
            past,
            sampler="dpmpp",
            num_inference_steps=5 if ensemble_size <= 1 else 20,
        )
        single = result["prediction_norm"].cpu()
        if single.dim() == 2:
            single = single.unsqueeze(1)
        preds_single.append(single)

        if ensemble_size <= 1:
            preds_avg.append(single)
        else:
            samples = [single]
            for sample_idx in range(1, ensemble_size):
                torch.manual_seed(10_000 + batch_idx * 101 + sample_idx)
                sample = model.generate(
                    past,
                    sampler="dpmpp",
                    num_inference_steps=20,
                )["prediction_norm"].cpu()
                if sample.dim() == 2:
                    sample = sample.unsqueeze(1)
                samples.append(sample)
            preds_avg.append(torch.stack(samples, dim=0).mean(dim=0))

        targets.append(future[..., K:])

    pred_single = torch.cat(preds_single, dim=0).squeeze(1)
    pred_avg = torch.cat(preds_avg, dim=0).squeeze(1)
    target = torch.cat(targets, dim=0).squeeze(1)

    single_metrics = {k: float(v) for k, v in compute_metrics(pred_single, target).items()}
    avg_metrics = {k: float(v) for k, v in compute_metrics(pred_avg, target).items()}
    return {"single": single_metrics, "averaged": avg_metrics}


def maybe_set_epoch(loader: DataLoader, epoch: int) -> None:
    ds = loader.dataset
    if hasattr(ds, "set_synthetic_epoch"):
        ds.set_synthetic_epoch(epoch)


def run_curriculum_stage(
    *,
    model,
    optimizer,
    family: str,
    cfg: ExperimentConfig,
    device: torch.device,
    seed: int,
    checkpoint_dir: Path,
    arm: str,
    spec: VariantSpec,
) -> StageSummary:
    best_val = float("inf")
    stale = 0
    final_epoch = 0
    stop_reason = "max_epochs"

    for epoch in range(1, cfg.curriculum_stage_epochs + 1):
        train_loader = make_wave_loader(
            family=family,
            num_samples=cfg.curriculum_samples_per_epoch,
            cfg=cfg,
            seed=seed + epoch * 17,
            shuffle=True,
        )
        val_loader = make_wave_loader(
            family=family,
            num_samples=cfg.curriculum_val_samples,
            cfg=cfg,
            seed=seed + 50_000,
            shuffle=False,
        )
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_mse = eval_wave_loader(model, val_loader, cfg, device)
        final_epoch = epoch
        logger.info(
            "[%s:%s] epoch %d/%d train_loss=%.4f val_norm_mse=%.5f",
            arm,
            family,
            epoch,
            cfg.curriculum_stage_epochs,
            train_loss,
            val_mse,
        )

        if val_mse < best_val - 1e-5:
            best_val = val_mse
            stale = 0
            save_checkpoint(
                model,
                spec,
                cfg,
                checkpoint_dir / f"{arm}_{family}_best.pt",
                arm,
                seed,
                stage=family,
                epoch=epoch,
                best_val_mse=best_val,
            )
        else:
            stale += 1

        if epoch >= cfg.curriculum_min_epochs and val_mse <= cfg.curriculum_success_mse:
            stop_reason = "threshold"
            break
        if epoch >= cfg.curriculum_min_epochs and stale >= cfg.curriculum_patience:
            stop_reason = "patience"
            break

    return StageSummary(stage=family, epochs_trained=final_epoch, best_val_mse=best_val, stop_reason=stop_reason)


def train_realts_stage(
    *,
    model,
    optimizer,
    cfg: ExperimentConfig,
    device: torch.device,
    seed: int,
    cache_dir: Optional[str],
    checkpoint_path: Path,
    arm: str,
    spec: VariantSpec,
) -> Tuple[StageSummary, Dict[str, Dict[str, float]]]:
    train_loader = make_realts_loader(
        num_samples=cfg.realts_train_samples,
        cfg=cfg,
        seed=seed + 101,
        shuffle=True,
        cache_dir=cache_dir,
        synthetic_epoch_capacity=cfg.realts_epoch_capacity,
    )
    val_loader = make_realts_loader(
        num_samples=cfg.realts_val_samples,
        cfg=cfg,
        seed=seed + 202,
        shuffle=False,
        batch_size=1,
        cache_dir=None,
        synthetic_epoch_capacity=1,
    )
    test_loader = make_realts_loader(
        num_samples=cfg.realts_test_samples,
        cfg=cfg,
        seed=seed + 303,
        shuffle=False,
        batch_size=1,
        cache_dir=None,
        synthetic_epoch_capacity=1,
    )

    best_val = float("inf")
    best_epoch = 0
    stale = 0
    stop_reason = "max_epochs"

    for epoch in range(1, cfg.realts_max_epochs + 1):
        maybe_set_epoch(train_loader, epoch - 1)
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate_realts(model, val_loader, cfg, device, ensemble_size=1)
        val_mse = val_metrics["single"]["mse"]
        logger.info(
            "[%s:realts] epoch %d/%d train_loss=%.4f val_single_mse=%.5f val_single_mae=%.5f",
            arm,
            epoch,
            cfg.realts_max_epochs,
            train_loss,
            val_metrics["single"]["mse"],
            val_metrics["single"]["mae"],
        )

        if val_mse < best_val - cfg.realts_min_delta:
            best_val = val_mse
            best_epoch = epoch
            stale = 0
            save_checkpoint(
                model,
                spec,
                cfg,
                checkpoint_path,
                arm,
                seed,
                stage="realts",
                epoch=epoch,
                best_val_mse=best_val,
            )
        else:
            stale += 1

        if epoch >= cfg.realts_min_epochs and stale >= cfg.realts_patience:
            stop_reason = "patience"
            break

    best_payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model_state_dict"], strict=True)
    eval_metrics = evaluate_realts(model, test_loader, cfg, device, ensemble_size=cfg.eval_ensemble)
    logger.info(
        "[%s:test] single=%s | averaged=%s",
        arm,
        log_metrics(eval_metrics["single"]),
        log_metrics(eval_metrics["averaged"]),
    )
    return (
        StageSummary(stage="realts", epochs_trained=best_epoch, best_val_mse=best_val, stop_reason=stop_reason),
        eval_metrics,
    )


def run_experiment(
    *,
    arm: str,
    cfg: ExperimentConfig,
    seed: int,
    device: torch.device,
    checkpoint_dir: Path,
    results_dir: Path,
    cache_dir: Optional[str],
) -> Dict:
    spec = VARIANTS[cfg.variant_name]
    if spec.use_guidance:
        raise ValueError(
            f"Variant {cfg.variant_name} uses iTransformer guidance; "
            f"this experiment isolates curriculum effects, so choose one of {NON_GUIDED_VARIANTS}."
        )
    model = build_model(spec, cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    stage_summaries: List[StageSummary] = []
    if arm == ARM_CURRICULUM:
        for idx, family in enumerate(WAVE_FAMILIES):
            summary = run_curriculum_stage(
                model=model,
                optimizer=optimizer,
                family=family,
                cfg=cfg,
                device=device,
                seed=seed + idx * 1000,
                checkpoint_dir=checkpoint_dir,
                arm=arm,
                spec=spec,
            )
            stage_summaries.append(summary)

    realts_ckpt = checkpoint_dir / f"{arm}_seed{seed}_best.pt"
    realts_summary, eval_metrics = train_realts_stage(
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        device=device,
        seed=seed,
        cache_dir=cache_dir,
        checkpoint_path=realts_ckpt,
        arm=arm,
        spec=spec,
    )
    stage_summaries.append(realts_summary)

    result = {
        "arm": arm,
        "seed": seed,
        "variant_name": cfg.variant_name,
        "learning_rate": cfg.lr,
        "stage_summaries": [asdict(s) for s in stage_summaries],
        "eval_metrics": eval_metrics,
        "checkpoint_path": str(realts_ckpt),
        "config": asdict(cfg),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{arm}_seed{seed}.json"
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("Wrote %s", out_path)
    return result


def smoke_defaults(cfg: ExperimentConfig) -> ExperimentConfig:
    cfg.curriculum_stage_epochs = 2
    cfg.curriculum_min_epochs = 1
    cfg.curriculum_patience = 1
    cfg.curriculum_samples_per_epoch = 32
    cfg.curriculum_val_samples = 16
    cfg.realts_train_samples = 64
    cfg.realts_val_samples = 24
    cfg.realts_test_samples = 24
    cfg.realts_epoch_capacity = 2
    cfg.realts_max_epochs = 3
    cfg.realts_min_epochs = 1
    cfg.realts_patience = 1
    cfg.eval_ensemble = 1
    cfg.batch_size = 4
    cfg.num_diffusion_steps = 50
    cfg.ddim_steps = 8
    cfg.image_height = 32
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare direct RealTS vs wave-curriculum -> RealTS.")
    parser.add_argument("--arm", choices=[ARM_DIRECT, ARM_CURRICULUM], required=True)
    parser.add_argument("--variant", default="dit_tiny_no_guidance", choices=NON_GUIDED_VARIANTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--curriculum-stage-epochs", type=int, default=None)
    parser.add_argument("--realts-max-epochs", type=int, default=None)
    parser.add_argument("--realts-train-samples", type=int, default=None)
    parser.add_argument("--realts-val-samples", type=int, default=None)
    parser.add_argument("--realts-test-samples", type=int, default=None)
    parser.add_argument("--eval-ensemble", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = ExperimentConfig(variant_name=args.variant, lr=args.lr)
    if args.smoke_test:
        cfg = smoke_defaults(cfg)
    if args.curriculum_stage_epochs is not None:
        cfg.curriculum_stage_epochs = args.curriculum_stage_epochs
    if args.realts_max_epochs is not None:
        cfg.realts_max_epochs = args.realts_max_epochs
    if args.realts_train_samples is not None:
        cfg.realts_train_samples = args.realts_train_samples
    if args.realts_val_samples is not None:
        cfg.realts_val_samples = args.realts_val_samples
    if args.realts_test_samples is not None:
        cfg.realts_test_samples = args.realts_test_samples
    if args.eval_ensemble is not None:
        cfg.eval_ensemble = args.eval_ensemble
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "arm=%s variant=%s seed=%d lr=%.2e smoke=%s device=%s",
        args.arm,
        cfg.variant_name,
        args.seed,
        cfg.lr,
        args.smoke_test,
        device,
    )

    result = run_experiment(
        arm=args.arm,
        cfg=cfg,
        seed=args.seed,
        device=device,
        checkpoint_dir=Path(args.checkpoint_dir),
        results_dir=Path(args.results_dir),
        cache_dir=args.cache_dir,
    )

    print("\n=== RealTS curriculum experiment summary ===")
    print(
        f"{result['arm']} seed={result['seed']} lr={result['learning_rate']:.2e} "
        f"variant={result['variant_name']}"
    )
    print(f"single:   {result['eval_metrics']['single']}")
    print(f"averaged: {result['eval_metrics']['averaged']}")


if __name__ == "__main__":
    main()
