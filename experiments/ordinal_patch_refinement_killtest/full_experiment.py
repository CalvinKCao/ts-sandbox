"""Full split-safe ordinal patch refinement + held-out discriminator kill test.

Geometry: hi-res CDF is (H x W=horizon); coarse is (16 x W); naive input is
vertical-only NN upsample to (H x W). Training uses in-bounds 8x8 crops only
(OOB skipped on train/val/test). Train windows: stride-2 overlapping.
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

from experiments.ordinal_patch_refinement_killtest import smoke
from experiments.ordinal_patch_refinement_killtest.nonoverlap_protocol import build_protocol
from models.diffusion_tsf.dit import FactorizedDiT
from models.diffusion_tsf.ordinal_window_norm import ordinal_decode, ordinal_encode
from models.diffusion_tsf.preprocessing import TimeSeriesTo2D
from models.diffusion_tsf.train_multivariate_pipeline import load_dataset
from utils.eval_discriminator_texture_staged_vs_mmpd import (
    HorizonSliceDataset,
    InvertedSliceDiscriminator,
    evaluate_classifier,
)
from utils.eval_mmpd_gaussian_anchor import load_tsf_pack_pool

HORIZON = smoke.HORIZON
LOOKBACK = 96
PATCH = smoke.PATCH
COARSE_H = smoke.COARSE_H
LR_GRID = (5e-5, 2.41e-4, 1.5e-3)
BATCH_GRID = (512, 1024, 2048)


def _n_variates(dataset: str) -> int:
    return 7 if dataset == "ETTh1" else 8


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
    hist = smoke._cdf_from_values(past_ord[..., -PATCH:], rank_max, resolution)
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
    xs, cs, ys, meta = [], [], [], []
    total_stats = {"candidates": 0, "skipped_oob": 0, "kept": 0}
    v_count = enc["upscaled"].shape[1]
    for v in range(v_count):
        bins = TimeSeriesTo2D.bin_indices_from_cdf(enc["coarse"][:, v : v + 1])[0, 0].long()
        x, c, y, coords, stats = smoke._patch_batch(
            enc["upscaled"][0, v : v + 1],
            enc["target"][0, v : v + 1],
            enc["hist"][0, v : v + 1],
            bins,
        )
        for key in total_stats:
            total_stats[key] += stats[key]
        for i, (row0, col0) in enumerate(coords):
            xs.append(x[i].cpu())
            cs.append(c[i].cpu())
            ys.append(y[i].cpu())
            meta.append({"variate": v, "row0": row0, "col0": col0})
    if not xs:
        empty = torch.zeros(0, 5, PATCH, PATCH)
        return empty, empty[:, :1], empty[:, :1], [], total_stats
    return torch.stack(xs), torch.stack(cs), torch.stack(ys), meta, total_stats


def _materialize_split(
    pool,
    indices: list[int],
    ladder,
    device: torch.device,
    resolution: int,
    limit: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict], dict]:
    xs, cs, ys, records = [], [], [], []
    agg = {"candidates": 0, "skipped_oob": 0, "kept": 0, "windows_with_patches": 0}
    chosen = indices if limit is None else indices[:limit]
    for wi in chosen:
        past, future = pool[wi]
        enc = _encode_window(past.unsqueeze(0), future.unsqueeze(0), ladder, device, resolution)
        px, pc, py, meta, stats = _patches_from_encoded(enc)
        for key in ("candidates", "skipped_oob", "kept"):
            agg[key] += stats[key]
        if stats["kept"] == 0:
            continue
        agg["windows_with_patches"] += 1
        for i, m in enumerate(meta):
            xs.append(px[i])
            cs.append(pc[i])
            ys.append(py[i])
            records.append({"window": wi, **m})
    if not xs:
        empty = torch.zeros(0, 5, PATCH, PATCH)
        return empty, empty[:, :1], empty[:, :1], [], agg
    return torch.stack(xs), torch.stack(cs), torch.stack(ys), records, agg


def _make_model(device: torch.device) -> FactorizedDiT:
    return FactorizedDiT(
        in_channels=5, cond_channels=1, out_channels=1, image_height=PATCH,
        patch_size=(4, 4), embed_dim=384, depth=8, num_heads=6, context_dim=1,
    ).to(device)


def _train_refiner(
    tx, tc, ty, vx, vc, vy, device, epochs: int, smoke_mode: bool,
) -> tuple[FactorizedDiT, float, int]:
    if len(tx) == 0:
        raise RuntimeError("no in-bounds train patches after OOB filter")
    grid = [(LR_GRID[0], BATCH_GRID[0])] if smoke_mode else [(lr, b) for lr in LR_GRID for b in BATCH_GRID]
    best: tuple[float, float, int, FactorizedDiT] | None = None
    trial_epochs = 1 if smoke_mode else min(4, epochs)
    for lr, eff in grid:
        model = _make_model(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        micro = min(64, len(tx))
        acc = max(1, eff // micro)
        for _ in range(trial_epochs):
            order = torch.randperm(len(tx))
            opt.zero_grad(set_to_none=True)
            for j, idx in enumerate(order.split(micro)):
                xb, cb, yb = tx[idx].to(device), tc[idx].to(device), ty[idx].to(device)
                loss = F.binary_cross_entropy_with_logits(
                    model(xb, torch.zeros(len(idx), device=device), cb), yb,
                )
                (loss / acc).backward()
                if (j + 1) % acc == 0:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            if len(vx) == 0:
                val = float("inf")
            else:
                val = 0.0
                for idx in torch.arange(len(vx)).split(64):
                    xb, cb, yb = vx[idx].to(device), vc[idx].to(device), vy[idx].to(device)
                    loss = F.binary_cross_entropy_with_logits(
                        model(xb, torch.zeros(len(idx), device=device), cb), yb,
                    )
                    val += float(loss.item()) * len(idx)
                val /= max(1, len(vx))
        if best is None or val < best[0]:
            best = (val, lr, eff, model)
    assert best is not None
    _, lr, eff, model = best
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(tx, tc, ty), batch_size=64, shuffle=True)
    for _ in range(1 if smoke_mode else epochs):
        for xb, cb, yb in loader:
            xb, cb, yb = xb.to(device), cb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(
                model(xb, torch.zeros(len(xb), device=device), cb), yb,
            )
            loss.backward()
            opt.step()
    return model, lr, eff


@torch.no_grad()
def _infer_windows(model, pool, indices, ladder, device, resolution):
    pasts, gts, naives, refineds = [], [], [], []
    coarses, upscales, targets, refined_cdfs = [], [], [], []
    kept_indices = []
    for wi in indices:
        past, future = pool[wi]
        enc = _encode_window(past.unsqueeze(0), future.unsqueeze(0), ladder, device, resolution)
        refined_canvas = enc["upscaled"].clone()
        v_count = enc["upscaled"].shape[1]
        any_patch = False
        for v in range(v_count):
            bins = TimeSeriesTo2D.bin_indices_from_cdf(enc["coarse"][:, v : v + 1])[0, 0].long()
            x, c, _y, coords, _stats = smoke._patch_batch(
                enc["upscaled"][0, v : v + 1],
                enc["target"][0, v : v + 1],
                enc["hist"][0, v : v + 1],
                bins,
            )
            if len(coords) == 0:
                continue
            any_patch = True
            logits = model(x.to(device), torch.zeros(len(x), device=device), c.to(device))
            patches, _ = smoke._project_monotone(torch.sigmoid(logits))
            refined_canvas[:, v : v + 1] = smoke._blend_patches_into_canvas(
                refined_canvas[:, v : v + 1], patches, coords,
            )
        if not any_patch:
            continue
        gt_rank = enc["future_ord"]
        naive_rank = smoke._decode_ranks(enc["upscaled"], enc["rank_max"])
        refined_rank = smoke._decode_ranks(refined_canvas, enc["rank_max"])
        _, gt = ordinal_decode(enc["past_ord"], gt_rank, enc["ladder_b"], ood_shift=enc["ood_shift"])
        _, naive = ordinal_decode(enc["past_ord"], naive_rank, enc["ladder_b"], ood_shift=enc["ood_shift"])
        _, refined = ordinal_decode(
            enc["past_ord"], refined_rank, enc["ladder_b"], ood_shift=enc["ood_shift"],
        )
        assert gt is not None and naive is not None and refined is not None
        pasts.append(past.cpu().numpy())
        gts.append(gt[0].cpu().numpy())
        naives.append(naive[0].cpu().numpy())
        refineds.append(refined[0].cpu().numpy())
        coarses.append(enc["coarse"][0].cpu().numpy())
        upscales.append(enc["upscaled"][0].cpu().numpy())
        targets.append(enc["target"][0].cpu().numpy())
        refined_cdfs.append(refined_canvas[0].cpu().numpy())
        kept_indices.append(wi)
    if not pasts:
        raise RuntimeError("no test windows retained after OOB patch filter")
    return {
        "past": np.stack(pasts),
        "gt": np.stack(gts),
        "naive": np.stack(naives),
        "refined": np.stack(refineds),
        "coarse_cdf": np.stack(coarses),
        "upscaled_cdf": np.stack(upscales),
        "target_cdf": np.stack(targets),
        "refined_cdf": np.stack(refined_cdfs),
        "window_ids": np.asarray(kept_indices, dtype=np.int64),
    }


def _train_disc(past, real, fake, device, *, slice_len=8, epochs=8, seed=0):
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
def _confusion_plots(model, ds, pack, out_dir, *, fake_name, per_bucket=2, variate=0):
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
            if not 0 <= pos < len(pack["past"]):
                continue
            past = pack["past"][pos, variate]
            gt = pack["gt"][pos, variate]
            naive = pack["naive"][pos, variate]
            refined = pack["refined"][pos, variate]
            fig, axes = plt.subplots(2, 3, figsize=(14, 8))
            ax = axes[0, 0]
            ax.plot(np.arange(-len(past), 0), past, color="0.45", label="lookback")
            ax.plot(np.arange(HORIZON), gt, marker="o", label="GT")
            ax.plot(np.arange(HORIZON), naive, marker="x", label="naive")
            ax.plot(np.arange(HORIZON), refined, marker="s", label="refined")
            ax.axvspan(rec["offset"], rec["offset"] + ds.slice_len, color="C3", alpha=0.15)
            ax.set_title(f"{bucket} p_fake={rec['prob_fake']:.3f} win={rec['window']} off={rec['offset']}")
            ax.legend(fontsize=8)
            for ax, key, title in zip(
                axes[0, 1:], ("coarse_cdf", "upscaled_cdf"), ("coarse 16xW", "naive vertical upscale"),
            ):
                ax.imshow(pack[key][pos, variate], origin="lower", aspect="auto", cmap="viridis")
                ax.set_title(title)
            for ax, key, title in zip(
                axes[1],
                ("refined_cdf", "target_cdf", "upscaled_cdf"),
                ("refined CDF", "GT hi-res CDF", "naive − GT"),
            ):
                if title.startswith("naive"):
                    ax.imshow(
                        pack["upscaled_cdf"][pos, variate] - pack["target_cdf"][pos, variate],
                        origin="lower", aspect="auto", cmap="coolwarm",
                    )
                else:
                    ax.imshow(pack[key][pos, variate], origin="lower", aspect="auto", cmap="viridis")
                ax.set_title(title)
            fig.suptitle(f"{fake_name} / {bucket} / variate {variate}")
            fig.tight_layout()
            fig.savefig(out_dir / f"{fake_name}_{bucket}_{j}_w{rec['window']}.png", dpi=140)
            plt.close(fig)
    (out_dir / f"{fake_name}_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    return counts


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
    args = parser.parse_args()

    smoke.set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    n_var = _n_variates(args.dataset)
    resolution = args.resolution
    protocol = build_protocol(args.dataset, n_var, lookback=LOOKBACK)
    limit = 2 if args.smoke else None

    # Train pack must use stride-2; val/test packs use their native strides.
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
    _, _, _, stats = load_dataset(
        args.dataset, list(range(n_var)), lookback=LOOKBACK, horizon=HORIZON,
        stride=1, test_stride=4, use_ordinal_window_norm=True,
    )
    ladder = stats["ordinal_ladder"]

    tx, tc, ty, _, train_patch_stats = _materialize_split(
        pool_by["train"], protocol["splits"]["train"]["indices"], ladder, device, resolution, limit,
    )
    vx, vc, vy, _, val_patch_stats = _materialize_split(
        pool_by["val"], protocol["splits"]["val"]["indices"], ladder, device, resolution, limit,
    )
    model, best_lr, best_batch = _train_refiner(
        tx, tc, ty, vx, vc, vy, device, args.epochs, args.smoke,
    )

    test_indices = protocol["splits"]["test"]["indices"]
    if args.smoke:
        test_indices = test_indices[: max(4, len(test_indices) // 20 or 4)]
    pack = _infer_windows(model, pool_by["test"], test_indices, ladder, device, resolution)

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "heldout_windows.npz",
        past=pack["past"], gt=pack["gt"], naive=pack["naive"], refined=pack["refined"],
        window_ids=pack["window_ids"],
    )
    np.savez_compressed(
        args.output / "heldout_cdfs.npz",
        coarse_cdf=pack["coarse_cdf"], upscaled_cdf=pack["upscaled_cdf"],
        target_cdf=pack["target_cdf"], refined_cdf=pack["refined_cdf"],
        window_ids=pack["window_ids"],
    )

    disc_metrics = {}
    for fake_name, fake in (("refined", pack["refined"]), ("naive", pack["naive"])):
        disc, metrics, ds_test = _train_disc(
            pack["past"], pack["gt"], fake, device,
            epochs=2 if args.smoke else args.disc_epochs, seed=args.seed,
        )
        counts = _confusion_plots(
            disc, ds_test, pack, args.output / "disc_confusions", fake_name=fake_name,
        )
        metrics = {**metrics, "confusion_counts": counts}
        disc_metrics[fake_name] = metrics
        torch.save(
            {"model_state_dict": disc.state_dict(), "metrics": metrics, "fake_source": fake_name},
            args.output / f"disc_{fake_name}.pt",
        )

    manifest = {
        "dataset": args.dataset,
        "resolution": resolution,
        "patch": PATCH,
        "canvas": [resolution, HORIZON],
        "smoke": args.smoke,
        "protocol": {
            split: {k: v for k, v in vals.items() if k != "indices"}
            for split, vals in protocol["splits"].items()
        },
        "patch_filter": {
            "train": train_patch_stats,
            "val": val_patch_stats,
            "rule": "skip 8x8 if crop would leave the (H x W) canvas",
        },
        "train_patches": len(tx),
        "val_patches": len(vx),
        "test_windows_scored": int(len(pack["window_ids"])),
        "best_lr": best_lr,
        "effective_batch": best_batch,
        "discriminator": disc_metrics,
        "refine_mae": {
            "naive": float(np.mean(np.abs(pack["naive"] - pack["gt"]))),
            "refined": float(np.mean(np.abs(pack["refined"] - pack["gt"]))),
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
