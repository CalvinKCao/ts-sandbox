"""Full split-safe ordinal patch refinement + held-out discriminator kill test.

Geometry: hi-res CDF is (H x W=horizon); coarse is (16 x W); naive input is
vertical-only NN upsample to (H x W). Training uses in-bounds 32-tall x 8-wide
crops only (strict OOB skipped on train/val/test). Train windows: stride-2
overlapping.

Refiner: real binary XOR diffusion (FactorizedDiT dual-head, linear schedule,
min-SNR) conditioned on naive upscale + lookback hist; iterative quad_t sample.

Discriminator: 1D ordinal-rank refined vs GT after both classes pass through
the same rank -> 256-bin centre -> global-ladder canonicalization.
Datasets: ETTh1 (7), exchange_rate/electricity/traffic (first 8).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.ordinal_patch_refinement_killtest import diffusion_refine
from experiments.ordinal_patch_refinement_killtest import smoke
from experiments.ordinal_patch_refinement_killtest.nonoverlap_protocol import (
    DATASET_N_VARIATES,
    build_protocol,
)
from models.diffusion_tsf.ordinal_window_norm import ordinal_decode, ordinal_encode
from models.diffusion_tsf.preprocessing import TimeSeriesTo2D
from experiments.ordinal_patch_refinement_killtest.ordinal_grid import (
    canonicalize_ranks,
    snap_ranks_to_ladder,
)
from models.diffusion_tsf.train_multivariate_pipeline import load_dataset
from utils.eval_discriminator_texture_staged_vs_mmpd import (
    HorizonSliceDataset,
    InvertedSliceDiscriminator,
    evaluate_classifier,
)
from utils.eval_mmpd_gaussian_anchor import load_tsf_pack_pool

HORIZON = smoke.HORIZON
LOOKBACK = 96
PATCH_H = smoke.PATCH_H
PATCH_W = smoke.PATCH_W
COARSE_H = smoke.COARSE_H
LR_GRID = (5e-5, 2.41e-4, 1.5e-3)
BATCH_GRID = (512, 1024, 2048)


def _n_variates(dataset: str) -> int:
    if dataset not in DATASET_N_VARIATES:
        raise ValueError(
            f"unsupported dataset {dataset!r}; expected one of {sorted(DATASET_N_VARIATES)}"
        )
    return DATASET_N_VARIATES[dataset]




def _encode_window(
    past: torch.Tensor,
    future: torch.Tensor,
    ladder,
    device: torch.device,
    resolution: int,
) -> dict[str, Any]:
    past = past.to(device)
    future = future.to(device)[..., :HORIZON]
    past_ord, future_ord, ladder_b, ood_shift = ordinal_encode(
        past, future, ladder=ladder, apply_ood_shift=True, causal_only=True,
    )
    assert future_ord is not None
    rank_max = ladder_b.rank_max_per_variate().to(device=device, dtype=torch.float32)
    target = smoke._cdf_from_values(future_ord, rank_max, resolution)
    coarse = smoke._cdf_from_values(future_ord, rank_max, COARSE_H)
    upscaled = smoke._vertical_upsample(coarse, resolution)
    hist = smoke._cdf_from_values(past_ord[..., -PATCH_W:], rank_max, resolution)
    return {
        "past_ord": past_ord,
        "future_ord": future_ord,
        "ladder_b": ladder_b,
        "ood_shift": ood_shift,
        "rank_max": rank_max,
        "target": target,
        "coarse": coarse,
        "upscaled": upscaled,
        "hist": hist,
    }


def _patches_from_encoded(enc: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list, dict]:
    """Return naive, hist, target patch stacks (N,1,H,W) for diffusion."""
    naives, hists, tgts, meta = [], [], [], []
    total_stats = {
        "candidates": 0,
        "skipped_oob": 0,
        "skipped_oob_canvas": 0,
        "skipped_oob_column_edge": 0,
        "kept": 0,
    }
    v_count = enc["upscaled"].shape[1]
    for v in range(v_count):
        bins = TimeSeriesTo2D.bin_indices_from_cdf(enc["coarse"][:, v : v + 1])[0, 0].long()
        naive_p, hist_p, tgt_p, coords, stats = smoke._patch_batch(
            enc["upscaled"][0, v : v + 1],
            enc["target"][0, v : v + 1],
            enc["hist"][0, v : v + 1],
            bins,
        )
        for key in total_stats:
            total_stats[key] += int(stats[key])
        for i, (row0, col0) in enumerate(coords):
            naives.append(naive_p[i].cpu())
            hists.append(hist_p[i].cpu())
            tgts.append(tgt_p[i].cpu())
            meta.append({"variate": v, "row0": row0, "col0": col0})
    if not naives:
        empty = torch.zeros(0, 1, PATCH_H, PATCH_W)
        return empty, empty, empty, [], total_stats
    return torch.stack(naives), torch.stack(hists), torch.stack(tgts), meta, total_stats


def _materialize_split(
    pool,
    indices: list[int],
    ladder,
    device: torch.device,
    resolution: int,
    limit: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict], dict]:
    naives, hists, tgts, records = [], [], [], []
    agg = {
        "candidates": 0,
        "skipped_oob": 0,
        "skipped_oob_canvas": 0,
        "skipped_oob_column_edge": 0,
        "kept": 0,
        "windows_with_patches": 0,
    }
    chosen = indices if limit is None else indices[:limit]
    for wi in chosen:
        past, future = pool[wi]
        enc = _encode_window(past.unsqueeze(0), future.unsqueeze(0), ladder, device, resolution)
        pn, ph, pt, meta, stats = _patches_from_encoded(enc)
        for key in (
            "candidates",
            "skipped_oob",
            "skipped_oob_canvas",
            "skipped_oob_column_edge",
            "kept",
        ):
            agg[key] += stats[key]
        if stats["kept"] == 0:
            continue
        agg["windows_with_patches"] += 1
        for i, m in enumerate(meta):
            naives.append(pn[i])
            hists.append(ph[i])
            tgts.append(pt[i])
            records.append({"window": wi, **m})
    if not naives:
        empty = torch.zeros(0, 1, PATCH_H, PATCH_W)
        return empty, empty, empty, [], agg
    return torch.stack(naives), torch.stack(hists), torch.stack(tgts), records, agg


def _train_refiner(
    tn, th, ty, vn, vh, vy, device, epochs: int, smoke_mode: bool,
):
    """Binary XOR diffusion HP grid + refit (same LR/batch grid as vertical_dual)."""
    if len(tn) == 0:
        raise RuntimeError("no in-bounds train patches after OOB filter")
    grid = [(LR_GRID[0], BATCH_GRID[0])] if smoke_mode else [(lr, b) for lr in LR_GRID for b in BATCH_GRID]
    scheduler = diffusion_refine.make_scheduler(device)
    best = None
    trial_epochs = 1 if smoke_mode else min(4, epochs)
    for lr, eff in grid:
        model = diffusion_refine.make_refiner(PATCH_H, device, patch_w=PATCH_W)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        micro = min(64, len(tn))
        acc = max(1, eff // micro)
        model.train()
        for _ in range(trial_epochs):
            order = torch.randperm(len(tn))
            opt.zero_grad(set_to_none=True)
            for j, idx in enumerate(order.split(micro)):
                x0 = ty[idx].to(device)
                cond = diffusion_refine.build_cond(tn[idx].to(device), th[idx].to(device))
                loss = diffusion_refine.diffusion_loss(model, scheduler, x0, cond)
                (loss / acc).backward()
                if (j + 1) % acc == 0:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            if len(vn) == 0:
                val = float("inf")
            else:
                val = 0.0
                model.eval()
                for idx in torch.arange(len(vn)).split(64):
                    x0 = vy[idx].to(device)
                    cond = diffusion_refine.build_cond(vn[idx].to(device), vh[idx].to(device))
                    loss = diffusion_refine.diffusion_loss(model, scheduler, x0, cond)
                    val += float(loss.item()) * len(idx)
                val /= max(1, len(vn))
        if best is None or val < best[0]:
            best = (val, lr, eff, model)
    assert best is not None
    _, lr, eff, model = best
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(tn, th, ty), batch_size=64, shuffle=True)
    model.train()
    for _ in range(1 if smoke_mode else epochs):
        for nb, hb, yb in loader:
            x0 = yb.to(device)
            cond = diffusion_refine.build_cond(nb.to(device), hb.to(device))
            opt.zero_grad(set_to_none=True)
            loss = diffusion_refine.diffusion_loss(model, scheduler, x0, cond)
            loss.backward()
            opt.step()
    return model, scheduler, lr, eff


@torch.no_grad()
def _infer_windows(model, scheduler, pool, indices, ladder, device, resolution, *, smoke_mode=False):
    """Infer with iterative binary diffusion; disc uses 1D ordinal ranks."""
    past_ranks, gt_ranks, gt_ranks_raw, refined_ranks, refined_ranks_raw = [], [], [], [], []
    past_z, gt_z, naive_z, refined_z = [], [], [], []
    coarses, upscales, targets, refined_cdfs = [], [], [], []
    gt_high_bins, refined_high_bins = [], []
    kept_indices = []
    sample_steps = 5 if smoke_mode else diffusion_refine.NUM_SAMPLE_STEPS
    for wi in indices:
        past, future = pool[wi]
        enc = _encode_window(past.unsqueeze(0), future.unsqueeze(0), ladder, device, resolution)
        refined_canvas = enc["upscaled"].clone()
        v_count = enc["upscaled"].shape[1]
        any_patch = False
        for v in range(v_count):
            bins = TimeSeriesTo2D.bin_indices_from_cdf(enc["coarse"][:, v : v + 1])[0, 0].long()
            naive_p, hist_p, _tgt, coords, _stats = smoke._patch_batch(
                enc["upscaled"][0, v : v + 1],
                enc["target"][0, v : v + 1],
                enc["hist"][0, v : v + 1],
                bins,
            )
            if len(coords) == 0:
                continue
            any_patch = True
            cond = diffusion_refine.build_cond(naive_p.to(device), hist_p.to(device))
            patches = diffusion_refine.sample_patches(
                model, scheduler, cond, num_steps=sample_steps, sampler="quad_t",
            )
            refined_canvas[:, v : v + 1] = smoke._blend_patches_into_canvas(
                refined_canvas[:, v : v + 1], patches, coords,
            )
        if not any_patch:
            continue
        gt_rank_raw = enc["future_ord"]
        naive_rank_raw = smoke._decode_ranks(enc["upscaled"], enc["rank_max"])
        refined_rank_raw = smoke._decode_ranks(refined_canvas, enc["rank_max"])
        naive_rank = snap_ranks_to_ladder(naive_rank_raw, enc["ladder_b"])
        refined_ladder_rank = snap_ranks_to_ladder(refined_rank_raw, enc["ladder_b"])
        # Both classes use the exact shared rank -> bin centre -> ladder path.
        gt_rank, gt_high_bin = canonicalize_ranks(
            gt_rank_raw, enc["rank_max"], enc["ladder_b"], resolution,
        )
        refined_rank, refined_high_bin = canonicalize_ranks(
            refined_ladder_rank, enc["rank_max"], enc["ladder_b"], resolution,
        )
        _, gt = ordinal_decode(enc["past_ord"], gt_rank, enc["ladder_b"], ood_shift=enc["ood_shift"])
        _, naive = ordinal_decode(enc["past_ord"], naive_rank, enc["ladder_b"], ood_shift=enc["ood_shift"])
        _, refined = ordinal_decode(
            enc["past_ord"], refined_rank, enc["ladder_b"], ood_shift=enc["ood_shift"],
        )
        assert gt is not None and naive is not None and refined is not None
        past_ranks.append(enc["past_ord"][0].cpu().numpy())
        gt_ranks.append(gt_rank[0].cpu().numpy())
        gt_ranks_raw.append(gt_rank_raw[0].cpu().numpy())
        gt_high_bins.append(gt_high_bin[0].cpu().numpy())
        refined_high_bins.append(refined_high_bin[0].cpu().numpy())
        refined_ranks.append(refined_rank[0].cpu().numpy())
        refined_ranks_raw.append(refined_rank_raw[0].cpu().numpy())
        past_z.append(past.cpu().numpy())
        gt_z.append(gt[0].cpu().numpy())
        naive_z.append(naive[0].cpu().numpy())
        refined_z.append(refined[0].cpu().numpy())
        coarses.append(enc["coarse"][0].cpu().numpy())
        upscales.append(enc["upscaled"][0].cpu().numpy())
        targets.append(enc["target"][0].cpu().numpy())
        refined_cdfs.append(refined_canvas[0].cpu().numpy())
        kept_indices.append(wi)
    if not past_ranks:
        raise RuntimeError("no test windows retained after OOB patch filter")
    return {
        "past_rank": np.stack(past_ranks),
        "gt_rank": np.stack(gt_ranks),
        "gt_rank_raw": np.stack(gt_ranks_raw),
        "gt_high_bin": np.stack(gt_high_bins),
        "refined_high_bin": np.stack(refined_high_bins),
        "refined_rank": np.stack(refined_ranks),
        "refined_rank_raw": np.stack(refined_ranks_raw),
        "past": np.stack(past_z),
        "gt": np.stack(gt_z),
        "naive": np.stack(naive_z),
        "refined": np.stack(refined_z),
        "coarse_cdf": np.stack(coarses),
        "upscaled_cdf": np.stack(upscales),
        "target_cdf": np.stack(targets),
        "refined_cdf": np.stack(refined_cdfs),
        "window_ids": np.asarray(kept_indices, dtype=np.int64),
        "sample_steps": sample_steps,
        "disc_rank_space": f"shared_{resolution}_bin_centre_then_ladder",
    }


def _train_disc(past, real, fake, device, *, slice_len=8, epochs=8, seed=0):
    """Train texture disc after identical GT/fake bin-centre canonicalization."""
    n = past.shape[0]
    n_train = max(1, int(0.7 * n))
    n_val = max(1, int(0.15 * n))
    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, min(n, n_train + n_val))
    test_idx = np.arange(min(n, n_train + n_val), n)
    if len(test_idx) == 0:
        test_idx = val_idx.copy()
    if len(val_idx) == 0:
        val_idx = train_idx.copy()
    ds_train = HorizonSliceDataset(past, real, fake, train_idx, slice_len, seed=seed)
    ds_val = HorizonSliceDataset(past, real, fake, val_idx, slice_len, seed=seed + 1)
    ds_test = HorizonSliceDataset(past, real, fake, test_idx, slice_len, seed=seed + 2)
    model = InvertedSliceDiscriminator(
        seq_len=LOOKBACK + slice_len, max_offset=HORIZON - slice_len, d_model=128,
        n_heads=4, depth=2, d_ff=256, dropout=0.1,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_loader = DataLoader(ds_train, batch_size=64, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=64, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=64, shuffle=False)
    best_state, best_val = None, float("inf")
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            x, offsets, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(x, offsets), labels)
            loss.backward()
            opt.step()
        val_metrics = evaluate_classifier(model, val_loader, device)
        if val_metrics["disc_bce"] < best_val:
            best_val = val_metrics["disc_bce"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, evaluate_classifier(model, test_loader, device), ds_test


def _bucket(label: int, pred: int) -> str:
    if label == 1 and pred == 1:
        return "TP"
    if label == 0 and pred == 0:
        return "TN"
    if label == 0 and pred == 1:
        return "FP"
    return "FN"


@torch.no_grad()
def _confusion_plots(model, ds, pack, out_dir, *, per_bucket=2, variate=0):
    """TP/FP/TN/FN plots for 1D ordinal-rank GT vs refined (CDFs are context only)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    records = []
    cursor = 0
    model.eval()
    for batch in loader:
        x, offsets, labels = batch[0].to(device), batch[1].to(device), batch[2]
        logits = model(x, offsets)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (logits >= 0).cpu().numpy().astype(np.int64)
        labels_np = labels.numpy().astype(np.int64)
        for i in range(len(labels_np)):
            window, offset, label = ds.items[cursor]
            records.append({
                "window": int(window), "offset": int(offset), "label": int(label),
                "pred": int(preds[i]), "prob_fake": float(probs[i]),
                "bucket": _bucket(int(label), int(preds[i])),
            })
            cursor += 1
    counts = {k: 0 for k in ("TP", "TN", "FP", "FN")}
    by_bucket: dict[str, list] = {k: [] for k in counts}
    for rec in records:
        counts[rec["bucket"]] += 1
        by_bucket[rec["bucket"]].append(rec)
    for bucket, items in by_bucket.items():
        for j, rec in enumerate(items[:per_bucket]):
            pos = int(rec["window"])
            if not 0 <= pos < len(pack["past_rank"]):
                continue
            past_r = pack["past_rank"][pos, variate]
            gt_r = pack["gt_rank"][pos, variate]
            ref_r = pack["refined_rank"][pos, variate]
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            ax = axes[0, 0]
            ax.plot(np.arange(-len(past_r), 0), past_r, color="0.45", label="lookback ranks")
            ax.plot(np.arange(HORIZON), gt_r, marker="o", label="GT ranks")
            ax.plot(np.arange(HORIZON), ref_r, marker="s", label="refined ranks")
            ax.axvspan(rec["offset"], rec["offset"] + ds.slice_len, color="C3", alpha=0.15)
            ax.set_title(
                f"{bucket} p_fake={rec['prob_fake']:.3f} win={rec['window']} off={rec['offset']}"
            )
            ax.set_ylabel("ordinal rank")
            ax.legend(fontsize=8)
            ax = axes[0, 1]
            ax.plot(np.arange(HORIZON), ref_r - gt_r, marker="d", color="C3")
            ax.axhline(0.0, color="0.5", linewidth=0.8)
            ax.axvspan(rec["offset"], rec["offset"] + ds.slice_len, color="C3", alpha=0.15)
            ax.set_title("refined − GT (ordinal ranks)")
            ax.set_ylabel("rank delta")
            for ax, key, title in zip(
                axes[1],
                ("refined_cdf", "target_cdf"),
                ("refined CDF (context)", "GT hi-res CDF (context)"),
            ):
                ax.imshow(pack[key][pos, variate], origin="lower", aspect="auto", cmap="viridis")
                ax.set_title(title)
            fig.suptitle(f"1D ordinal disc: refined vs GT / {bucket} / v{variate}")
            fig.tight_layout()
            fig.savefig(out_dir / f"refined_vs_gt_{bucket}_{j}_w{rec['window']}.png", dpi=140)
            plt.close(fig)
    (out_dir / "refined_vs_gt_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    return counts


@torch.no_grad()
def _confusion_counts(model, ds) -> dict[str, int]:
    """Count discriminator outcomes without requiring CDF plotting artifacts."""
    counts = {key: 0 for key in ("TP", "TN", "FP", "FN")}
    device = next(model.parameters()).device
    model.eval()
    for batch in DataLoader(ds, batch_size=64, shuffle=False):
        logits = model(batch[0].to(device), batch[1].to(device))
        predictions = (logits >= 0).cpu().numpy().astype(np.int64)
        labels = batch[2].numpy().astype(np.int64)
        for label, prediction in zip(labels, predictions):
            counts[_bucket(int(label), int(prediction))] += 1
    return counts


def _load_disc_only_pack(path: Path, ladder, resolution: int) -> dict[str, np.ndarray | str]:
    """Load saved binary samples and canonicalize both classes identically."""
    with np.load(path, allow_pickle=False) as source:
        if "past_rank" not in source or "gt_rank" not in source:
            raise ValueError(f"{path} must contain past_rank and gt_rank")
        if "refined_rank_raw" in source:
            refined_raw = source["refined_rank_raw"]
        elif "refined_rank" in source:
            refined_raw = source["refined_rank"]
        else:
            raise ValueError(f"{path} must contain refined_rank_raw or refined_rank")
        gt_raw = source["gt_rank_raw"] if "gt_rank_raw" in source else source["gt_rank"]
        pack: dict[str, np.ndarray | str] = {key: source[key] for key in source.files}

    gt_tensor = torch.as_tensor(gt_raw, dtype=torch.float32)
    refined_raw_tensor = torch.as_tensor(refined_raw, dtype=torch.float32)
    rank_max = ladder.rank_max_per_variate().to(dtype=torch.float32)
    refined_ladder = snap_ranks_to_ladder(refined_raw_tensor, ladder)
    gt_rank, gt_high_bin = canonicalize_ranks(gt_tensor, rank_max, ladder, resolution)
    refined_rank, refined_high_bin = canonicalize_ranks(
        refined_ladder, rank_max, ladder, resolution,
    )
    pack.update({
        "gt_rank_raw": np.asarray(gt_raw),
        "refined_rank_raw": np.asarray(refined_raw),
        "gt_rank": gt_rank.numpy(),
        "refined_rank": refined_rank.numpy(),
        "gt_high_bin": gt_high_bin.numpy(),
        "refined_high_bin": refined_high_bin.numpy(),
        "disc_rank_space": f"shared_{resolution}_bin_centre_then_ladder",
    })

    cdf_path = path.with_name("heldout_cdfs.npz")
    if cdf_path.is_file():
        with np.load(cdf_path, allow_pickle=False) as cdfs:
            pack.update({key: cdfs[key] for key in cdfs.files})
    return pack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--disc-epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--disc-only-input", type=Path,
        help="existing heldout_windows.npz; skip refiner training/sampling",
    )
    args = parser.parse_args()

    smoke.set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    n_var = _n_variates(args.dataset)
    resolution = args.resolution
    protocol = build_protocol(args.dataset, n_var, lookback=LOOKBACK)
    limit = None
    if args.smoke:
        # Strict OOB can empty the first few windows; take enough to keep patches.
        limit = 64

    _, _, _, stats = load_dataset(
        args.dataset, list(range(n_var)), lookback=LOOKBACK, horizon=HORIZON,
        stride=1, test_stride=4, use_ordinal_window_norm=True,
    )
    ladder = stats["ordinal_ladder"]

    if args.disc_only_input is not None:
        pack = _load_disc_only_pack(args.disc_only_input, ladder, resolution)
        args.output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output / "heldout_windows_canonicalized.npz",
            **{key: value for key, value in pack.items() if isinstance(value, np.ndarray)},
        )
        disc, metrics, ds_test = _train_disc(
            pack["past_rank"], pack["gt_rank"], pack["refined_rank"], device,
            epochs=2 if args.smoke else args.disc_epochs, seed=args.seed,
        )
        if all(key in pack for key in ("refined_cdf", "target_cdf")):
            counts = _confusion_plots(disc, ds_test, pack, args.output / "disc_confusions")
        else:
            counts = _confusion_counts(disc, ds_test)
        metrics = {
            **metrics,
            "confusion_counts": counts,
            "input": f"1d_ordinal_ranks_shared_{resolution}_bin_canonicalized",
            "real": "gt_rank_bin_canonicalized",
            "fake": "refined_rank_bin_canonicalized",
            "rank_space": pack["disc_rank_space"],
        }
        torch.save(
            {
                "model_state_dict": disc.state_dict(),
                "metrics": metrics,
                "fake_source": "refined_rank_bin_canonicalized",
            },
            args.output / "disc_refined_vs_gt_ranks.pt",
        )
        manifest = {
            "dataset": args.dataset,
            "n_variates": n_var,
            "resolution": resolution,
            "seed": args.seed,
            "smoke": args.smoke,
            "disc_only": True,
            "disc_only_input": str(args.disc_only_input),
            "test_windows_scored": int(len(pack["past_rank"])),
            "canonicalization": (
                f"Both classes: ordinal rank -> {resolution}-bin centre -> global ordinal ladder"
            ),
            "discriminator": metrics,
        }
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8",
        )
        print(json.dumps(manifest, indent=2))
        return

    pool_train = load_tsf_pack_pool(
        args.dataset, list(range(n_var)), lookback=LOOKBACK, horizon=HORIZON,
        train_stride=2, test_stride=4, pack_splits=["train"],
    )[0]
    pool_by = {"train": pool_train}
    for split in ("val", "test"):
        pool_by[split] = load_tsf_pack_pool(
            args.dataset, list(range(n_var)), lookback=LOOKBACK, horizon=HORIZON,
            train_stride=1, test_stride=4, pack_splits=[split],
        )[0]
    tn, th, ty, _, train_patch_stats = _materialize_split(
        pool_by["train"], protocol["splits"]["train"]["indices"], ladder, device, resolution, limit,
    )
    vn, vh, vy, _, val_patch_stats = _materialize_split(
        pool_by["val"], protocol["splits"]["val"]["indices"], ladder, device, resolution, limit,
    )
    model, scheduler, best_lr, best_batch = _train_refiner(
        tn, th, ty, vn, vh, vy, device, args.epochs, args.smoke,
    )

    test_indices = protocol["splits"]["test"]["indices"]
    if args.smoke:
        test_indices = test_indices[: max(4, len(test_indices) // 20 or 4)]
    pack = _infer_windows(
        model, scheduler, pool_by["test"], test_indices, ladder, device, resolution,
        smoke_mode=args.smoke,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "heldout_windows.npz",
        past_rank=pack["past_rank"], gt_rank=pack["gt_rank"], gt_rank_raw=pack["gt_rank_raw"],
        gt_high_bin=pack["gt_high_bin"], refined_high_bin=pack["refined_high_bin"],
        refined_rank=pack["refined_rank"], refined_rank_raw=pack["refined_rank_raw"],
        past=pack["past"], gt=pack["gt"], naive=pack["naive"], refined=pack["refined"],
        window_ids=pack["window_ids"],
    )
    np.savez_compressed(
        args.output / "heldout_cdfs.npz",
        coarse_cdf=pack["coarse_cdf"], upscaled_cdf=pack["upscaled_cdf"],
        target_cdf=pack["target_cdf"], refined_cdf=pack["refined_cdf"],
        window_ids=pack["window_ids"],
    )

    # Discriminator: GT and refined both use the shared bin-centre canonicalizer.
    disc, metrics, ds_test = _train_disc(
        pack["past_rank"], pack["gt_rank"], pack["refined_rank"], device,
        epochs=2 if args.smoke else args.disc_epochs, seed=args.seed,
    )
    counts = _confusion_plots(disc, ds_test, pack, args.output / "disc_confusions")
    metrics = {
        **metrics,
        "confusion_counts": counts,
        "input": f"1d_ordinal_ranks_shared_{resolution}_bin_canonicalized",
        "real": "gt_rank_bin_canonicalized",
        "fake": "refined_rank_bin_canonicalized",
        "rank_space": pack["disc_rank_space"],
    }
    torch.save(
        {
            "model_state_dict": disc.state_dict(),
            "metrics": metrics,
            "fake_source": "refined_rank_bin_canonicalized",
        },
        args.output / "disc_refined_vs_gt_ranks.pt",
    )

    manifest = {
        "dataset": args.dataset,
        "n_variates": n_var,
        "resolution": resolution,
        "patch_h": PATCH_H,
        "patch_w": PATCH_W,
        "canvas": [resolution, HORIZON],
        "smoke": args.smoke,
        "canonicalization": f"Both classes: ordinal rank -> {resolution}-bin centre -> global ordinal ladder",
        "protocol": {
            split: {k: v for k, v in vals.items() if k != "indices"}
            for split, vals in protocol["splits"].items()
        },
        "patch_filter": {
            "train": train_patch_stats,
            "val": val_patch_stats,
            "rule": "strict: skip if canvas OOB or any coarse/GT column edge leaves the 32x8 crop",
        },
        "train_patches": len(tn),
        "val_patches": len(vn),
        "test_windows_scored": int(len(pack["window_ids"])),
        "best_lr": best_lr,
        "effective_batch": best_batch,
        "refiner": {
            "trainer": "binary_diffusion_xor",
            "noise_schedule": diffusion_refine.SCHEDULE,
            "train_T": diffusion_refine.NUM_TRAIN_STEPS,
            "sample_steps": pack["sample_steps"],
            "sampler": "quad_t",
            "prediction_target": diffusion_refine.PRED_TARGET,
            "min_snr_gamma": diffusion_refine.MIN_SNR_GAMMA,
            "cond": "naive_vertical_upscale + past_hist",
        },
        "discriminator": {"refined_vs_gt_ranks": metrics},
        "refine_mae_snapped_z": {
            "naive": float(np.mean(np.abs(pack["naive"] - pack["gt"]))),
            "refined": float(np.mean(np.abs(pack["refined"] - pack["gt"]))),
        },
        "refine_mae_ordinal_rank": {
            "refined_snapped": float(np.mean(np.abs(pack["refined_rank"] - pack["gt_rank"]))),
            "refined_raw_midbin": float(
                np.mean(np.abs(pack["refined_rank_raw"] - pack["gt_rank"]))
            ),
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
