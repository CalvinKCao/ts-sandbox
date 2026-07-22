#!/usr/bin/env python3
"""MMPD Decoder oracle-coarse ordinal upscaling and discriminator kill test.

This is deliberately a 1D task.  Each input channel is the concatenation of
the full-resolution ordinal lookback (256-bin centres) and the oracle 16-bin
future.  MMPD predicts the matching 256-bin future centres.  No CDF canvas,
coarse/fine decomposition, guidance channel, or instance normalization is
used.

Before the discriminator sees either candidate, the MMPD output is quantized
to the same 256-bin centres as the target and snapped to the global ordinal
ladder used by the binary kill test.  This removes fractional-bin and
off-ladder cues from the discriminator comparison.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MMPD_ROOT = REPO_ROOT / "temp" / "MMPD"
if not MMPD_ROOT.is_dir():
    raise FileNotFoundError(
        f"Missing local MMPD checkout at {MMPD_ROOT}. Run through ./submit_mmpd.sh "
        "or clone the pinned MMPD checkout on the login node first."
    )
if str(MMPD_ROOT) not in sys.path:
    sys.path.insert(0, str(MMPD_ROOT))

# These are the unmodified upstream MMPD Decoder and Gaussian anchor loss.
from models.backbone_loss_model import BackboneLossModel  # type: ignore  # noqa: E402
from models.backbones.decoder_only_transformer import DecoderOnlyTransformer  # type: ignore  # noqa: E402
from models.loss_funcs.mmpd.mmpd_loss import MMPD_Loss  # type: ignore  # noqa: E402

from models.diffusion_tsf.ordinal_window_norm import ordinal_encode  # noqa: E402
from models.diffusion_tsf.train_multivariate_pipeline import load_dataset  # noqa: E402
from utils.eval_discriminator_texture_staged_vs_mmpd import (  # noqa: E402
    HorizonSliceDataset,
    InvertedSliceDiscriminator,
    evaluate_classifier,
)
from utils.eval_mmpd_gaussian_anchor import load_tsf_pack_pool  # noqa: E402

from experiments.ordinal_patch_refinement_killtest.nonoverlap_protocol import (  # noqa: E402
    DATASET_N_VARIATES,
    build_protocol,
)


@dataclass(frozen=True)
class RunConfig:
    lookback: int = 96
    horizon: int = 16
    coarse_bins: int = 16
    high_bins: int = 256
    backbone: str = "Decoder"
    patch_size: int = 12
    d_model: int = 256
    d_ff: int = 512
    n_heads: int = 4
    d_layers: int = 2
    dropout: float = 0.2
    d_diffusion: int = 256
    diffusion_layers: int = 1
    max_diffusion_steps: int = 1000
    beta_schedule: str = "linear"
    radius: int = 3
    num_sampling_steps: int = 20
    train_epochs: int = 20
    patience: int = 5
    learning_rate: float = 1.0e-4
    point_weight: float = 0.01
    batch_size: int = 32
    tune_trials: int = 7
    tune_epochs: int = 10
    tune_patience: int = 3
    tune_lr_min: float = 3.0e-5
    tune_lr_max: float = 3.0e-4
    tune_point_weight_min: float = 0.005
    tune_point_weight_max: float = 0.05
    tune_batch_min: int = 8
    tune_batch_max: int = 32
    train_stride: int = 2
    test_stride: int = 4
    disc_slice_len: int = 8
    disc_epochs: int = 8
    disc_batch_size: int = 64
    disc_lr: float = 1.0e-3
    disc_d_model: int = 128
    disc_d_ff: int = 256
    disc_n_heads: int = 4
    disc_depth: int = 2
    disc_dropout: float = 0.1


def _load_config(path: Path) -> RunConfig:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    block = raw.get("mmpd") or {}
    if block.get("task") != "ordinal_upscale":
        raise ValueError(f"{path} must set mmpd.task: ordinal_upscale")
    tune = block.get("tune_params") or {}
    fields = {
        "lookback": block.get("lookback", 96),
        "horizon": block.get("horizon", 16),
        "coarse_bins": block.get("coarse_bins", 16),
        "high_bins": block.get("high_bins", 256),
        "backbone": block.get("backbone", "Decoder"),
        "patch_size": block.get("patch_size_default", 12),
        "d_model": block.get("d_model", 256),
        "d_ff": block.get("d_ff", 512),
        "n_heads": block.get("n_heads", 4),
        "d_layers": block.get("d_layers", 2),
        "dropout": block.get("dropout", 0.2),
        "d_diffusion": block.get("d_diffusion", 256),
        "diffusion_layers": block.get("diffusion_layers", 1),
        "max_diffusion_steps": block.get("max_diffusion_steps", 1000),
        "beta_schedule": block.get("beta_schedule", "linear"),
        "radius": block.get("radius", 3),
        "num_sampling_steps": block.get("num_sampling_steps", 20),
        "train_epochs": block.get("train_epochs", 20),
        "patience": block.get("patience", 5),
        "learning_rate": block.get("learning_rate", 1.0e-4),
        "point_weight": block.get("point_weight", 0.01),
        "batch_size": block.get("batch_size", 32),
        "tune_trials": block.get("tune_trials", 7),
        "tune_epochs": block.get("tune_epochs", 10),
        "tune_patience": block.get("tune_patience", 3),
        "train_stride": block.get("train_stride", 2),
        "test_stride": block.get("test_stride", 4),
        "disc_slice_len": block.get("disc_slice_len", 8),
        "disc_epochs": block.get("disc_epochs", 8),
        "disc_batch_size": block.get("disc_batch_size", 64),
        "disc_lr": block.get("disc_lr", 1.0e-3),
        "disc_d_model": block.get("disc_d_model", 128),
        "disc_d_ff": block.get("disc_d_ff", 256),
        "disc_n_heads": block.get("disc_n_heads", 4),
        "disc_depth": block.get("disc_depth", 2),
        "disc_dropout": block.get("disc_dropout", 0.1),
    }
    for key, attr in (
        ("learning_rate", ("tune_lr_min", "tune_lr_max")),
        ("point_weight", ("tune_point_weight_min", "tune_point_weight_max")),
        ("batch_size", ("tune_batch_min", "tune_batch_max")),
    ):
        values = tune.get(key)
        if isinstance(values, list) and len(values) == 2:
            fields[attr[0]], fields[attr[1]] = values
    config = RunConfig(**fields)
    if config.backbone != "Decoder":
        raise ValueError(f"Only upstream MMPD Decoder is supported, got {config.backbone!r}")
    if config.horizon != 16 or config.coarse_bins != 16 or config.high_bins != 256:
        raise ValueError("This kill test is fixed at 16-bin -> 256-bin, horizon 16")
    return config


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rank_to_bin_centres(ranks: torch.Tensor, rank_max: torch.Tensor, bins: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Binary-CDF-equivalent ordinal bucket centres, returned normalized to [0, 1]."""
    safe_max = rank_max.to(device=ranks.device, dtype=ranks.dtype).clamp_min(1.0).view(1, -1, 1)
    raw_bins = torch.floor(ranks / safe_max * bins).long().clamp_(0, bins - 1)
    centres = (raw_bins.to(ranks.dtype) + 0.5) / float(bins)
    return centres, raw_bins


def _bin_centres_to_ranks(bin_ids: torch.Tensor, rank_max: torch.Tensor, bins: int) -> torch.Tensor:
    safe_max = rank_max.to(device=bin_ids.device, dtype=torch.float32).clamp_min(1.0).view(1, -1, 1)
    return (bin_ids.to(torch.float32) + 0.5) / float(bins) * safe_max


def _snap_ranks_to_ladder(ranks: torch.Tensor, ladder: Any) -> torch.Tensor:
    out = ranks.clone()
    for variate in range(int(ranks.shape[1])):
        n_unique = int(ladder.n_unique[0, variate].item())
        out[:, variate] = ranks[:, variate].round().clamp_(0, max(0, n_unique - 1))
    return out


def _encode_window(past: torch.Tensor, future: torch.Tensor, ladder: Any, config: RunConfig) -> dict[str, torch.Tensor]:
    """Reuse the binary path's global ordinal ladder and causal OOD handling."""
    past_ord, future_ord, batch_ladder, _ood_shift = ordinal_encode(
        past, future[..., : config.horizon], ladder=ladder, apply_ood_shift=True, causal_only=True,
    )
    assert future_ord is not None
    rank_max = batch_ladder.rank_max_per_variate().to(dtype=torch.float32)
    past_hi, _ = _rank_to_bin_centres(past_ord, rank_max, config.high_bins)
    coarse, coarse_bin = _rank_to_bin_centres(future_ord, rank_max, config.coarse_bins)
    target_hi, target_bin = _rank_to_bin_centres(future_ord, rank_max, config.high_bins)
    return {
        "condition": torch.cat([past_hi, coarse], dim=-1),
        "target_hi": target_hi,
        "past_rank": past_ord,
        "gt_rank": future_ord,
        "coarse_bin": coarse_bin,
        "target_bin": target_bin,
        "rank_max": rank_max,
        "ladder": batch_ladder,
    }


def _materialize(pool: Any, indices: Iterable[int], ladder: Any, config: RunConfig) -> dict[str, torch.Tensor]:
    fields: dict[str, list[torch.Tensor]] = {
        "condition": [], "target_hi": [], "past_rank": [], "gt_rank": [],
        "coarse_bin": [], "target_bin": [],
    }
    window_ids: list[int] = []
    for index in indices:
        past, future = pool[int(index)]
        encoded = _encode_window(past.unsqueeze(0), future.unsqueeze(0), ladder, config)
        for name in fields:
            fields[name].append(encoded[name][0].cpu())
        window_ids.append(int(index))
    if not window_ids:
        raise RuntimeError("split has no selected windows")
    result = {name: torch.stack(items) for name, items in fields.items()}
    result["window_ids"] = torch.tensor(window_ids, dtype=torch.long)
    return result


def _make_model(config: RunConfig, data_dim: int, *, dropout: float) -> BackboneLossModel:
    args = SimpleNamespace(
        in_len=config.lookback + config.horizon,
        out_len=config.horizon,
        patch_size=config.patch_size,
        data_dim=data_dim,
        d_model=config.d_model,
        d_ff=config.d_ff,
        n_heads=config.n_heads,
        d_layers=config.d_layers,
        dropout=dropout,
        d_diffusion=config.d_diffusion,
        diffusion_layers=config.diffusion_layers,
        max_diffusion_steps=config.max_diffusion_steps,
        beta_schedule=config.beta_schedule,
        radius=config.radius,
        num_sampling_steps=str(config.num_sampling_steps),
    )
    return BackboneLossModel(DecoderOnlyTransformer(args), MMPD_Loss(args))


def _epoch(
    model: BackboneLossModel,
    loader: DataLoader,
    device: torch.device,
    point_weight: float,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for condition, target in loader:
            condition = condition.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            loss = model.compute_loss(condition, target, point_weight=point_weight).mean()
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += float(loss.detach().item()) * len(condition)
            count += len(condition)
    return total / max(1, count)


def _fit(
    train: dict[str, torch.Tensor],
    val: dict[str, torch.Tensor],
    config: RunConfig,
    device: torch.device,
    *,
    lr: float,
    point_weight: float,
    batch_size: int,
    dropout: float,
    epochs: int,
    patience: int,
    seed: int,
) -> tuple[BackboneLossModel, float, int]:
    _set_seed(seed)
    model = _make_model(config, train["condition"].shape[1], dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(train["condition"], train["target_hi"]), batch_size=batch_size,
        shuffle=True, generator=generator, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        TensorDataset(val["condition"], val["target_hi"]), batch_size=batch_size,
        shuffle=False, pin_memory=device.type == "cuda",
    )
    best_state, best_loss, best_epoch, stale = None, float("inf"), 0, 0
    for epoch in range(1, epochs + 1):
        _epoch(model, train_loader, device, point_weight, optimizer)
        val_loss = _epoch(model, val_loader, device, point_weight, None)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, best_loss, best_epoch


def _sample_trial(config: RunConfig, rng: np.random.Generator) -> dict[str, float | int]:
    return {
        "learning_rate": float(math.exp(rng.uniform(math.log(config.tune_lr_min), math.log(config.tune_lr_max)))),
        "point_weight": float(math.exp(rng.uniform(math.log(config.tune_point_weight_min), math.log(config.tune_point_weight_max)))),
        "batch_size": int(rng.integers(config.tune_batch_min, config.tune_batch_max + 1)),
        "dropout": config.dropout,
    }


def _tune_and_refit(
    train: dict[str, torch.Tensor], val: dict[str, torch.Tensor], config: RunConfig, device: torch.device, seed: int, smoke: bool,
) -> tuple[BackboneLossModel, dict[str, Any], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    trials = [{
        "learning_rate": config.learning_rate,
        "point_weight": config.point_weight,
        "batch_size": config.batch_size,
        "dropout": config.dropout,
    }] if smoke else [_sample_trial(config, rng) for _ in range(config.tune_trials)]
    history: list[dict[str, Any]] = []
    winner: dict[str, Any] | None = None
    for trial_id, params in enumerate(trials):
        model, val_loss, best_epoch = _fit(
            train, val, config, device, seed=seed + trial_id,
            lr=float(params["learning_rate"]), point_weight=float(params["point_weight"]),
            batch_size=int(params["batch_size"]), dropout=float(params["dropout"]),
            epochs=1 if smoke else config.tune_epochs,
            patience=1 if smoke else config.tune_patience,
        )
        del model
        row = {"trial": trial_id, **params, "val_loss": val_loss, "best_epoch": best_epoch}
        history.append(row)
        if winner is None or val_loss < winner["val_loss"]:
            winner = row
    assert winner is not None
    model, refit_loss, refit_epoch = _fit(
        train, val, config, device, seed=seed + 1000,
        lr=float(winner["learning_rate"]), point_weight=float(winner["point_weight"]),
        batch_size=int(winner["batch_size"]), dropout=float(winner["dropout"]),
        epochs=1 if smoke else config.train_epochs,
        patience=1 if smoke else config.patience,
    )
    winner = {**winner, "refit_val_loss": refit_loss, "refit_best_epoch": refit_epoch}
    return model, winner, history


@torch.no_grad()
def _infer(model: BackboneLossModel, test: dict[str, torch.Tensor], ladder: Any, config: RunConfig, device: torch.device) -> dict[str, np.ndarray]:
    loader = DataLoader(test["condition"], batch_size=config.batch_size, shuffle=False)
    predictions = []
    model.eval()
    for condition in loader:
        pred, _modes, _samples = model.predict(condition.to(device), prob_pred=False)
        predictions.append(pred.cpu())
    raw_units = torch.cat(predictions)
    fake_bin = torch.floor(raw_units * config.high_bins).long().clamp_(0, config.high_bins - 1)
    rank_max = ladder.rank_max_per_variate().to(dtype=torch.float32)
    fake_raw_rank = _bin_centres_to_ranks(fake_bin, rank_max, config.high_bins)
    fake_rank = _snap_ranks_to_ladder(fake_raw_rank, ladder)
    gt_rank = test["gt_rank"].to(dtype=torch.float32)
    # The binary path exposes true ordinal ranks for GT and ladder-snapped decoded ranks for fakes.
    return {
        "past_rank": test["past_rank"].numpy(),
        "gt_rank": gt_rank.numpy(),
        "fake_rank": fake_rank.numpy(),
        "fake_rank_raw": fake_raw_rank.numpy(),
        "coarse_bin": test["coarse_bin"].numpy(),
        "gt_high_bin": test["target_bin"].numpy(),
        "fake_high_bin": fake_bin.numpy(),
        "window_ids": test["window_ids"].numpy(),
    }


def _train_discriminator(pack: dict[str, np.ndarray], config: RunConfig, device: torch.device, seed: int, smoke: bool):
    n_windows = len(pack["window_ids"])
    n_train = max(1, int(0.7 * n_windows))
    n_val = max(1, int(0.15 * n_windows))
    split = {
        "train": np.arange(0, n_train),
        "val": np.arange(n_train, min(n_windows, n_train + n_val)),
        "test": np.arange(min(n_windows, n_train + n_val), n_windows),
    }
    if len(split["val"]) == 0:
        split["val"] = split["train"].copy()
    if len(split["test"]) == 0:
        split["test"] = split["val"].copy()
    kwargs = dict(slice_len=config.disc_slice_len, include_past=True, offset_stride=1)
    ds_train = HorizonSliceDataset(pack["past_rank"], pack["gt_rank"], pack["fake_rank"], split["train"], seed=seed, **kwargs)
    ds_val = HorizonSliceDataset(pack["past_rank"], pack["gt_rank"], pack["fake_rank"], split["val"], seed=seed + 1, **kwargs)
    ds_test = HorizonSliceDataset(pack["past_rank"], pack["gt_rank"], pack["fake_rank"], split["test"], seed=seed + 2, **kwargs)
    model = InvertedSliceDiscriminator(
        seq_len=config.lookback + config.disc_slice_len,
        max_offset=config.horizon - config.disc_slice_len,
        d_model=config.disc_d_model,
        n_heads=config.disc_n_heads,
        depth=config.disc_depth,
        d_ff=config.disc_d_ff,
        dropout=config.disc_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.disc_lr)
    train_loader = DataLoader(ds_train, batch_size=config.disc_batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=config.disc_batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=config.disc_batch_size, shuffle=False)
    best_state, best_val = None, float("inf")
    for _ in range(1 if smoke else config.disc_epochs):
        model.train()
        for batch in train_loader:
            x, offsets, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(x, offsets), labels)
            loss.backward()
            optimizer.step()
        val_metrics = evaluate_classifier(model, val_loader, device)
        if val_metrics["disc_bce"] < best_val:
            best_val = val_metrics["disc_bce"]
            best_state = copy.deepcopy(model.state_dict())
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, evaluate_classifier(model, test_loader, device), ds_test, split


def _bucket(label: int, pred: int) -> str:
    return "TP" if label and pred else "TN" if not label and not pred else "FP" if pred else "FN"


@torch.no_grad()
def _write_confusions(model: InvertedSliceDiscriminator, dataset: HorizonSliceDataset, pack: dict[str, np.ndarray], out_dir: Path, *, per_bucket: int = 2) -> dict[str, int]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    records, cursor = [], 0
    for batch in DataLoader(dataset, batch_size=64, shuffle=False):
        logits = model(batch[0].to(device), batch[1].to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (logits >= 0).cpu().numpy().astype(np.int64)
        labels = batch[2].numpy().astype(np.int64)
        for i, label in enumerate(labels):
            window, offset, _ = dataset.items[cursor]
            records.append({"window": int(window), "offset": int(offset), "label": int(label), "pred": int(preds[i]), "prob": float(probs[i])})
            cursor += 1
    groups = {name: [] for name in ("TP", "TN", "FP", "FN")}
    for record in records:
        groups[_bucket(record["label"], record["pred"])].append(record)
    counts = {name: len(rows) for name, rows in groups.items()}
    for name, rows in groups.items():
        rows.sort(key=lambda row: row["prob"], reverse=name in {"TP", "FP"})
        for example, record in enumerate(rows[:per_bucket]):
            pos, offset = record["window"], record["offset"]
            fig, axes = plt.subplots(2, 2, figsize=(12, 7))
            past = pack["past_rank"][pos, 0]
            gt = pack["gt_rank"][pos, 0]
            fake = pack["fake_rank"][pos, 0]
            axes[0, 0].plot(np.arange(-len(past), 0), past, color="0.4", label="full-res lookback")
            axes[0, 0].plot(np.arange(config_h := len(gt)), gt, marker="o", label="GT ordinal rank")
            axes[0, 0].plot(np.arange(config_h), fake, marker="s", label="MMPD snapped rank")
            axes[0, 0].axvspan(offset, offset + dataset.slice_len, color="C3", alpha=0.15)
            axes[0, 0].legend(fontsize=8)
            axes[0, 0].set_title("1D discriminator input")
            axes[0, 1].plot(gt - fake, marker="d", color="C3")
            axes[0, 1].axhline(0, color="0.5", linewidth=0.8)
            axes[0, 1].set_title("GT − MMPD (ladder ranks)")
            coarse_as_high = (pack["coarse_bin"][pos, 0].astype(np.float32) + 0.5) / 16.0 * 256.0
            image = np.stack([coarse_as_high, pack["fake_high_bin"][pos, 0], pack["gt_high_bin"][pos, 0]])
            axes[1, 0].imshow(image, aspect="auto", interpolation="nearest", vmin=0, vmax=255, cmap="viridis")
            axes[1, 0].set_yticks([0, 1, 2], ["coarse 16→256", "MMPD 256", "GT 256"])
            axes[1, 0].set_title("2D bin-code representation (not CDF)")
            axes[1, 1].plot(coarse_as_high, marker="x", label="oracle coarse (scaled)")
            axes[1, 1].plot(pack["gt_high_bin"][pos, 0], marker="o", label="GT 256-bin")
            axes[1, 1].plot(pack["fake_high_bin"][pos, 0], marker="s", label="MMPD 256-bin")
            axes[1, 1].legend(fontsize=8)
            axes[1, 1].set_title("Ordinal-bin codes")
            fig.suptitle(f"{name}: p(fake)={record['prob']:.3f} window={pack['window_ids'][pos]} offset={offset}")
            fig.tight_layout()
            fig.savefig(out_dir / f"mmpd_vs_gt_{name}_{example}_w{pack['window_ids'][pos]}.png", dpi=150)
            plt.close(fig)
    (out_dir / "counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_N_VARIATES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    config = _load_config(args.config)
    _set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    n_variates = DATASET_N_VARIATES[args.dataset]
    protocol = build_protocol(args.dataset, n_variates, lookback=config.lookback)
    pool_by = {
        "train": load_tsf_pack_pool(args.dataset, list(range(n_variates)), lookback=config.lookback, horizon=config.horizon, train_stride=config.train_stride, test_stride=config.test_stride, pack_splits=["train"])[0],
        "val": load_tsf_pack_pool(args.dataset, list(range(n_variates)), lookback=config.lookback, horizon=config.horizon, train_stride=1, test_stride=config.test_stride, pack_splits=["val"])[0],
        "test": load_tsf_pack_pool(args.dataset, list(range(n_variates)), lookback=config.lookback, horizon=config.horizon, train_stride=1, test_stride=config.test_stride, pack_splits=["test"])[0],
    }
    _, _, _, stats = load_dataset(args.dataset, list(range(n_variates)), lookback=config.lookback, horizon=config.horizon, stride=1, test_stride=config.test_stride, use_ordinal_window_norm=True)
    ladder = stats["ordinal_ladder"]
    limit = 8 if args.smoke else None
    train = _materialize(pool_by["train"], protocol["splits"]["train"]["indices"][:limit], ladder, config)
    val = _materialize(pool_by["val"], protocol["splits"]["val"]["indices"][:limit], ladder, config)
    test = _materialize(pool_by["test"], protocol["splits"]["test"]["indices"][:limit], ladder, config)
    model, winner, trials = _tune_and_refit(train, val, config, device, args.seed, args.smoke)
    pack = _infer(model, test, ladder, config, device)
    discriminator, disc_metrics, disc_test, disc_split = _train_discriminator(pack, config, device, args.seed, args.smoke)

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "heldout_ordinal_upscale.npz", **pack)
    torch.save({"model_state_dict": model.state_dict(), "winner": winner, "config": config.__dict__}, args.output / "mmpd_ordinal_upscale.pt")
    torch.save({"model_state_dict": discriminator.state_dict(), "metrics": disc_metrics, "fake_source": "mmpd_ordinal_256_snapped"}, args.output / "disc_mmpd_vs_gt_ordinal.pt")
    counts = _write_confusions(discriminator, disc_test, pack, args.output / "disc_confusions")
    manifest = {
        "dataset": args.dataset,
        "device": str(device),
        "smoke": args.smoke,
        "normalization": "global ordinal ladder; causal OOD shift; no instance normalization",
        "representation": "1D normalized bin centres: lookback=256-bin, future condition=16-bin, target=256-bin",
        "mmpd": {"architecture": "upstream DecoderOnlyTransformer + MMPD_Loss", "config": config.__dict__, "tuning_winner": winner, "tuning_trials": trials},
        "protocol": {split: {key: value for key, value in data.items() if key != "indices"} for split, data in protocol["splits"].items()},
        "counts": {"train_windows": len(train["window_ids"]), "val_windows": len(val["window_ids"]), "test_windows": len(test["window_ids"])},
        "snapping": "MMPD continuous output -> 256-bin centre -> global ordinal ladder; GT is the binary path ordinal ladder rank",
        "discriminator": {**disc_metrics, "confusion_counts": counts, "split_window_positions": {name: values.tolist() for name, values in disc_split.items()}},
        "ordinal_rank_mae": float(np.mean(np.abs(pack["fake_rank"] - pack["gt_rank"]))),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
