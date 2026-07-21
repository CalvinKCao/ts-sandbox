#!/usr/bin/env python3
"""One-window oracle-coarse smoke: vertical-only upscale + in-bounds 8x8 cutouts.

Pipeline (W stays = horizon; no horizontal stretch):
  ordinal ranks -> hi-res CDF (H x W) + coarse CDF (16 x W)
  -> nearest upsample coarse on the vertical axis only (H x W)
  -> 8x8 crops centered on the mid-column coarse edge; skip canvas-OOB crops
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from models.diffusion_tsf.dit import FactorizedDiT
from models.diffusion_tsf.ordinal_window_norm import OrdinalLadder, ordinal_decode, ordinal_encode
from models.diffusion_tsf.preprocessing import TimeSeriesTo2D
from models.diffusion_tsf.train_multivariate_pipeline import load_dataset
from utils.eval_mmpd_gaussian_anchor import load_tsf_pack_pool


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "results" / "ordinal_patch_refinement_killtest" / "smoke"

HORIZON = 16
COARSE_H = 16
PATCH = 8
COL_STRIDE = 2  # overlapping 8-wide crops every 2 columns


@dataclass(frozen=True)
class SmokeMetadata:
    dataset: str
    variate: int
    source_window_index: int
    lookback: int
    horizon: int
    coarse_height: int
    fine_height: int
    patch_size: int
    normalization: str
    ordinal_snapping: str
    device: str
    parameters: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _subset_ladder(ladder: OrdinalLadder, variate: int) -> OrdinalLadder:
    return OrdinalLadder(
        values=ladder.values[:, variate : variate + 1].clone(),
        n_unique=ladder.n_unique[:, variate : variate + 1].clone(),
        tie_atol=ladder.tie_atol,
    )


def _cdf_from_values(
    values: torch.Tensor,
    rank_max: torch.Tensor,
    height: int,
) -> torch.Tensor:
    """Direct bounded CDF encoding (equal ordinal buckets). Shape (B, V, H, W)."""
    to_2d = TimeSeriesTo2D(height=height, max_scale=1.0).to(values.device)
    cdf, _unused = to_2d.encode_dual_heights_bounded(
        values,
        coarse_height=height,
        fine_height=1,
        value_min=0.0,
        value_max_per_variate=rank_max,
    )
    return cdf


def _decode_ranks(cdf: torch.Tensor, rank_max: torch.Tensor) -> torch.Tensor:
    height = int(cdf.shape[-2])
    bins = TimeSeriesTo2D.bin_indices_from_cdf(cdf)
    return (bins + 0.5) / float(height) * rank_max.view(1, -1, 1)


def _project_monotone(prob: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Project local probability maps to hard, bottom-up CDF staircases."""
    raw = (prob >= 0.5).to(prob.dtype)
    violation = float((raw[..., 1:, :] > raw[..., :-1, :]).float().mean().item())
    occupancy = raw.sum(dim=-2, keepdim=True).round().clamp(0, raw.shape[-2])
    rows = torch.arange(raw.shape[-2], device=raw.device).view(1, 1, -1, 1)
    return (rows < occupancy).to(raw.dtype), violation


def _vertical_upsample(coarse: torch.Tensor, height: int) -> torch.Tensor:
    """Nearest upsample on the value axis only; keep time width unchanged."""
    if coarse.shape[-1] < 1:
        raise ValueError("coarse canvas has empty time axis")
    return F.interpolate(coarse, size=(height, coarse.shape[-1]), mode="nearest")


def _coarse_edge_row(coarse_bin: int, high: int) -> int:
    """Last occupied hi-res row for coarse bin k after vertical NN upscale."""
    scale = high // COARSE_H
    return (int(coarse_bin) + 1) * scale - 1


def _crop_in_canvas(row0: int, col0: int, high: int, width: int, patch: int = PATCH) -> bool:
    """Skip crops that would require vertical/horizontal padding."""
    return 0 <= row0 <= high - patch and 0 <= col0 <= width - patch


def _extract_block(canvas: torch.Tensor, row0: int, col0: int, size: int) -> torch.Tensor:
    """Extract an in-bounds square block (caller must validate OOB first)."""
    return canvas[..., row0 : row0 + size, col0 : col0 + size].clone()


def _write_block(canvas: torch.Tensor, block: torch.Tensor, row0: int, col0: int) -> None:
    size = int(block.shape[-1])
    canvas[..., row0 : row0 + size, col0 : col0 + size] = block


def _blend_patches_into_canvas(
    canvas: torch.Tensor,
    patches: torch.Tensor,
    coords: list[tuple[int, int]],
) -> torch.Tensor:
    """Average overlapping patch writes (COL_STRIDE < PATCH)."""
    out = canvas.clone()
    acc = torch.zeros_like(out)
    weight = torch.zeros_like(out)
    size = int(patches.shape[-1])
    for i, (row0, col0) in enumerate(coords):
        acc[..., row0 : row0 + size, col0 : col0 + size] += patches[i : i + 1]
        weight[..., row0 : row0 + size, col0 : col0 + size] += 1.0
    mask = weight > 0
    out[mask] = acc[mask] / weight[mask]
    return out


def _patch_batch(
    upscaled: torch.Tensor,
    target: torch.Tensor,
    past_canvas: torch.Tensor,
    coarse_bins: torch.Tensor,
    *,
    patch_size: int = PATCH,
    col_stride: int = COL_STRIDE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int]], dict[str, int]]:
    """Build overlapping 8x8 crops; skip canvas-OOB placements.

    Vertical center is the mid-column coarse edge after vertical-only upscale.
    Neighboring columns may have different edges inside the same crop; that is
    expected. Only placements that stick out of the (H x W) canvas are skipped.

    upscaled/target/past_canvas: (C, H, W) with W = horizon.
    coarse_bins: (W,)
    """
    inputs, conds, targets, coords = [], [], [], []
    high = int(upscaled.shape[-2])
    width = int(coarse_bins.shape[-1])
    center = patch_size // 2
    rows = torch.arange(patch_size, device=upscaled.device).view(1, patch_size, 1)
    n_cand = 0
    n_skip = 0
    for col0 in range(0, width - patch_size + 1, col_stride):
        n_cand += 1
        c_mid = col0 + center
        edge = _coarse_edge_row(int(coarse_bins[c_mid].item()), high)
        row0 = edge - center
        if not _crop_in_canvas(row0, col0, high, width, patch_size):
            n_skip += 1
            continue
        inp = _extract_block(upscaled, row0, col0, patch_size)
        tgt = _extract_block(target, row0, col0, patch_size)
        hist = _extract_block(past_canvas, row0, past_canvas.shape[-1] - patch_size, patch_size)
        boundary = (rows == center).to(upscaled.dtype).expand(1, patch_size, patch_size)
        # Channel kept for DiT width compatibility; always 1 after OOB skip.
        valid = torch.ones_like(inp)
        time_pos = torch.full_like(inp, float(c_mid) / max(1, width - 1))
        vertical_pos = torch.linspace(
            row0 / high, (row0 + patch_size - 1) / high, patch_size, device=upscaled.device,
        ).view(1, patch_size, 1).expand_as(inp)
        inputs.append(torch.cat([inp, boundary, valid, time_pos, vertical_pos], dim=0))
        conds.append(hist)
        targets.append(tgt)
        coords.append((row0, col0))
    stats = {"candidates": n_cand, "skipped_oob": n_skip, "kept": len(coords)}
    if not coords:
        empty = torch.zeros(0, 5, patch_size, patch_size, device=upscaled.device)
        return empty, empty[:, :1], empty[:, :1], [], stats
    return torch.stack(inputs), torch.stack(conds), torch.stack(targets), coords, stats


def _plot(output: Path, arrays: dict[str, np.ndarray], meta: SmokeMetadata) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    title = f"{meta.dataset} v{meta.variate} window={meta.source_window_index}"
    paths: list[Path] = []
    patch = int(meta.patch_size)
    horizon = int(meta.horizon)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(np.arange(-len(arrays["past"]), 0), arrays["past"], label="history", color="0.4")
    ax.plot(np.arange(horizon), arrays["gt_snapped"], label="GT snapped", marker="o")
    ax.plot(np.arange(horizon), arrays["naive"], label="naive upscale", marker="x")
    ax.plot(np.arange(horizon), arrays["refined"], label="refined", marker="s")
    ax.set(title=f"Decoded normalized trace — {title}", xlabel="relative time", ylabel="train-z-score")
    ax.legend(ncol=2)
    fig.tight_layout()
    path = output / "decoded_trace.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    panels = [
        ("coarse_cdf_16xW", arrays["coarse_cdf_16"]),
        ("target_cdf_hires", arrays["target_cdf_256"]),
        ("upscaled_coarse_vert", arrays["upscaled_coarse_256"]),
        ("refined_cdf", arrays["refined_cdf_256"]),
        ("target_minus_upscaled", arrays["target_cdf_256"] - arrays["upscaled_coarse_256"]),
        ("refined_minus_target", arrays["refined_cdf_256"] - arrays["target_cdf_256"]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 10))
    for ax, (name, image) in zip(axes.flat, panels):
        ax.imshow(image, origin="lower", aspect="auto", cmap="coolwarm" if "minus" in name else "viridis")
        ax.set_title(name)
        ax.set_xlabel("time col (W=horizon)")
        ax.set_ylabel("value row")
    fig.suptitle(title)
    fig.tight_layout()
    path = output / "cdf_canvases.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(6, 10))
    ax.imshow(arrays["upscaled_coarse_256"], origin="lower", aspect="auto", cmap="viridis")
    for row0, col0 in arrays["patch_coords"]:
        ax.add_patch(plt.Rectangle((col0 - 0.5, row0 - 0.5), patch, patch, fill=False, edgecolor="white", linewidth=0.9))
    ax.set(
        title=f"In-bounds 8x8 crops (white) on vertical-only upscale — {title}",
        xlabel="time column",
        ylabel="value row",
    )
    fig.tight_layout()
    path = output / "patch_selection.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    n_show = min(8, len(arrays["input_patches"]))
    if n_show == 0:
        return paths
    fig, axes = plt.subplots(n_show, 4, figsize=(12, max(3.0, n_show * 2.0)), squeeze=False)
    for i in range(n_show):
        coarse = arrays["input_patches"][i, 0]
        gt = arrays["target_patches"][i, 0]
        refined = arrays["refined_patches"][i, 0]
        for ax, img, name, cmap in zip(
            axes[i],
            [coarse, gt, refined, refined - gt],
            ["naive coarse input", "direct hi-res GT", "refined", "refined − GT"],
            ["viridis", "viridis", "viridis", "coolwarm"],
        ):
            ax.imshow(img, origin="lower", aspect="auto", cmap=cmap)
            if name != "refined − GT":
                ax.axhline(patch // 2, color="white", linewidth=0.7, linestyle="--")
            ax.set_title(f"i={i}: {name}")
            ax.axis("off")
    fig.suptitle(f"8x8 patches (dashed = mid-col coarse edge) — {title}")
    fig.tight_layout()
    path = output / "patches_before.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, i in zip(axes.flat, range(n_show)):
        ax.imshow(arrays["refined_patches"][i, 0], origin="lower", cmap="viridis")
        ax.contour(arrays["target_patches"][i, 0], levels=[0.5], colors="white", linewidths=0.8)
        ax.axhline(patch // 2, color="cyan", linewidth=0.7, linestyle="--")
        ax.set_title(f"i={i}")
        ax.axis("off")
    fig.suptitle(f"Refined 8x8 with GT overlays — {title}")
    fig.tight_layout()
    path = output / "patches_after.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--variate", type=int, default=0)
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.horizon != HORIZON:
        raise ValueError(f"This gated smoke is fixed to horizon={HORIZON}.")
    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    n_variates = 7 if args.dataset == "ETTh1" else 1
    variates = list(range(n_variates))
    high = int(args.resolution)

    _train, _val, _test, norm_stats = load_dataset(
        args.dataset, variates, lookback=args.lookback, horizon=args.horizon,
        stride=1, test_stride=4, use_ordinal_window_norm=True,
    )
    ladder: OrdinalLadder = norm_stats["ordinal_ladder"]
    pool, _starts, _splits, _lens, _stats = load_tsf_pack_pool(
        args.dataset, variates, lookback=args.lookback, horizon=args.horizon,
        train_stride=1, test_stride=4, pack_splits=["test"],
    )
    past, future = pool[args.window_index]
    past = past.unsqueeze(0).to(device=device, dtype=torch.float32)
    future = future.unsqueeze(0).to(device=device, dtype=torch.float32)[..., : args.horizon]
    if future.shape[-1] != args.horizon:
        raise ValueError(f"future length {future.shape[-1]} shorter than horizon {args.horizon}")
    if not 0 <= args.variate < n_variates:
        raise ValueError(f"variate {args.variate} outside 0..{n_variates - 1}")

    past_ord, future_ord, ladder_b, ood_shift = ordinal_encode(
        past, future, ladder=ladder, apply_ood_shift=True, causal_only=True,
    )
    assert future_ord is not None
    rank_max = ladder_b.rank_max_per_variate().to(device=device, dtype=torch.float32)
    vi = args.variate
    target = _cdf_from_values(future_ord, rank_max, high)[:, vi : vi + 1]
    coarse = _cdf_from_values(future_ord, rank_max, COARSE_H)[:, vi : vi + 1]
    upscaled = _vertical_upsample(coarse, high)
    past_cdf = _cdf_from_values(past_ord[..., -PATCH:], rank_max, high)[:, vi : vi + 1]
    if target.shape[-1] != HORIZON or upscaled.shape[-1] != HORIZON:
        raise RuntimeError(
            f"expected W={HORIZON}, got target W={target.shape[-1]} upscaled W={upscaled.shape[-1]}"
        )
    coarse_bins = TimeSeriesTo2D.bin_indices_from_cdf(coarse)[0, 0].long()
    inputs, cond, target_patches, coords, patch_stats = _patch_batch(
        upscaled[0], target[0], past_cdf[0], coarse_bins,
    )
    if len(coords) == 0:
        raise RuntimeError(f"no in-bounds 8x8 crops for this window: {patch_stats}")

    model = FactorizedDiT(
        in_channels=5, cond_channels=1, out_channels=1, image_height=PATCH,
        patch_size=(4, 4), embed_dim=384, depth=8, num_heads=6, context_dim=1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    inputs, cond, target_patches = inputs.to(device), cond.to(device), target_patches.to(device)
    losses: list[float] = []
    for _step in range(args.steps):
        logits = model(inputs, torch.zeros(inputs.shape[0], device=device), cond)
        loss = F.binary_cross_entropy_with_logits(logits, target_patches)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    with torch.no_grad():
        prob = torch.sigmoid(model(inputs, torch.zeros(inputs.shape[0], device=device), cond))
        refined_patches, monotonic_violation = _project_monotone(prob)
        refined = _blend_patches_into_canvas(upscaled, refined_patches, coords)

    selected_rank_max = rank_max[vi : vi + 1]
    ladder_one = _subset_ladder(ladder_b, vi)
    selected_shift = ood_shift[:, vi : vi + 1]
    gt_rank = future_ord[:, vi : vi + 1]
    naive_rank = _decode_ranks(upscaled, selected_rank_max)
    refined_rank = _decode_ranks(refined, selected_rank_max)
    _unused, gt_snapped = ordinal_decode(past_ord[:, vi : vi + 1], gt_rank, ladder_one, ood_shift=selected_shift)
    _unused, naive = ordinal_decode(past_ord[:, vi : vi + 1], naive_rank, ladder_one, ood_shift=selected_shift)
    _unused, refined_value = ordinal_decode(
        past_ord[:, vi : vi + 1], refined_rank, ladder_one, ood_shift=selected_shift,
    )
    assert gt_snapped is not None and naive is not None and refined_value is not None
    metrics = {
        "initial_bce": losses[0],
        "final_bce": losses[-1],
        "steps": args.steps,
        "patch_monotonic_violation_pre_projection": monotonic_violation,
        "naive_rank_mae": float((naive_rank - gt_rank).abs().mean().item()),
        "refined_rank_mae": float((refined_rank - gt_rank).abs().mean().item()),
        "naive_snapped_z_mae": float((naive - gt_snapped).abs().mean().item()),
        "refined_snapped_z_mae": float((refined_value - gt_snapped).abs().mean().item()),
        "exact_snapped_bin_accuracy": float((refined_value == gt_snapped).float().mean().item()),
        "canvas_shape": [high, HORIZON],
        "patch_size": PATCH,
        "patch_stats": patch_stats,
        "boundary_local_row": PATCH // 2,
    }
    parameters = sum(p.numel() for p in model.parameters())
    meta = SmokeMetadata(
        args.dataset, vi, args.window_index, args.lookback, args.horizon, COARSE_H, high, PATCH,
        "train-split z-score; ordinal encode with causal OOD shift; no instance normalization",
        "decode_with_ladder rounds all GT/naive/refined ranks to the same global ladder",
        str(device), parameters,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = {
        "past": past[0, vi].detach().cpu().numpy(),
        "future_raw": future[0, vi].detach().cpu().numpy(),
        "gt_snapped": gt_snapped[0, 0].detach().cpu().numpy(),
        "naive": naive[0, 0].detach().cpu().numpy(),
        "refined": refined_value[0, 0].detach().cpu().numpy(),
        "coarse_cdf_16": coarse[0, 0].detach().cpu().numpy(),
        "target_cdf_256": target[0, 0].detach().cpu().numpy(),
        "upscaled_coarse_256": upscaled[0, 0].detach().cpu().numpy(),
        "refined_cdf_256": refined[0, 0].detach().cpu().numpy(),
        "input_patches": inputs.detach().cpu().numpy(),
        "target_patches": target_patches.detach().cpu().numpy(),
        "refined_patches": refined_patches.detach().cpu().numpy(),
        "patch_coords": np.asarray(coords, dtype=np.int64),
        "losses": np.asarray(losses, dtype=np.float32),
    }
    np.savez_compressed(args.output / "smoke_arrays.npz", **arrays)
    (args.output / "metrics.json").write_text(
        json.dumps({"metadata": asdict(meta), "metrics": metrics}, indent=2), encoding="utf-8",
    )
    figures = _plot(args.output, arrays, meta)
    print(json.dumps({"output": str(args.output), "metrics": metrics, "figures": [str(p) for p in figures]}, indent=2))


if __name__ == "__main__":
    main()
