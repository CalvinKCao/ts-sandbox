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
    RawBundle,
    train_classifier,
)
from utils.eval_mmpd_gaussian_anchor import load_tsf_pack_pool
from utils.visualize_discriminator_texture_confusions import visualize_combo

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


def _write_standard_pack(
    path: Path,
    *,
    past: np.ndarray,
    y_true: np.ndarray,
    fake: np.ndarray,
    indices: np.ndarray,
) -> None:
    """Same npz schema as eval_discriminator_texture_staged_vs_mmpd packs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.asarray(fake, dtype=np.float32)[:, :, None, :]
    np.savez_compressed(
        path,
        y_true=np.asarray(y_true, dtype=np.float32),
        samples=samples,
        indices=np.asarray(indices, dtype=np.int64),
        past=np.asarray(past, dtype=np.float32),
    )


def _disc_namespace(output_dir: Path, *, smoke_mode: bool, seed: int) -> argparse.Namespace:
    """Defaults mirrored from utils/eval_discriminator_texture_staged_vs_mmpd.parse_args."""
    return argparse.Namespace(
        seed=seed,
        candidate_only=False,
        offset_stride=1,
        nonoverlapping_patches=False,
        no_offset_embedding=False,
        max_train_examples=128 if smoke_mode else None,
        max_eval_examples=128 if smoke_mode else None,
        max_batches_per_epoch=None,
        batch_size=64 if smoke_mode else 512,
        num_workers=0,
        d_model=128,
        n_heads=4,
        depth=2,
        d_ff=256,
        dropout=0.1,
        lr=1e-3,
        weight_decay=1e-4,
        epochs=2 if smoke_mode else 20,
        patience=5,
        grad_clip=1.0,
        train_fraction=0.7,
        val_fraction=0.15,
        save_checkpoints=True,
        output_dir=output_dir,
        visualize_confusions=True,
        viz_per_bucket=2,
        viz_variate=0,
        viz_lookback_tail=64,
        viz_plot_dir=output_dir / "disc_confusions",
        force_train=True,
        cpu=False,
        gpu=0,
        native_repr_stride=1,
    )


def _dense_splits(n: int, *, train_fraction: float, val_fraction: float) -> dict[str, np.ndarray]:
    """Chronological dense split for small/smoke packs (avoids temporal-purge edge cases)."""
    order = np.arange(n, dtype=np.int64)
    n_train = max(1, int(round(n * train_fraction)))
    n_val = max(1, int(round(n * val_fraction)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_val = max(1, n - n_train - n_test)
    return {
        "train": order[:n_train],
        "val": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }


def _run_original_discriminator(
    *,
    dataset: str,
    past: np.ndarray,
    y_true: np.ndarray,
    fake: np.ndarray,
    indices: np.ndarray,
    fake_source: str,
    output_dir: Path,
    device: torch.device,
    smoke_mode: bool,
    seed: int,
    slice_len: int = 8,
) -> dict[str, float]:
    """Train + TP/FP/TN/FN viz via the stock texture discriminator entrypoints."""
    disc_dir = output_dir / "disc"
    disc_dir.mkdir(parents=True, exist_ok=True)
    _write_standard_pack(
        disc_dir / "raw" / f"{fake_source}_{dataset}.npz",
        past=past, y_true=y_true, fake=fake, indices=indices,
    )
    args = _disc_namespace(disc_dir, smoke_mode=smoke_mode, seed=seed)
    bundle = RawBundle(
        run=None,
        sub={},
        indices=[int(i) for i in indices.tolist()],
        past=past.astype(np.float32),
        y_true_by_source={fake_source: y_true.astype(np.float32)},
        fakes={fake_source: fake.astype(np.float32)},
        series_starts=np.asarray(indices, dtype=np.int64),
        pack_splits=["killtest"],
    )
    splits = _dense_splits(
        past.shape[0], train_fraction=args.train_fraction, val_fraction=args.val_fraction,
    )
    metrics = train_classifier(args, dataset, fake_source, slice_len, bundle, splits, device)
    if args.visualize_confusions:
        visualize_combo(
            args, dataset, fake_source, slice_len, bundle, splits, device,
            per_bucket=args.viz_per_bucket,
            plot_dir=args.viz_plot_dir,
            variate=args.viz_variate,
            lookback_tail=args.viz_lookback_tail,
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--import-mmpd-packs-from",
        type=Path,
        default=None,
        help="Optional dir with existing mmpd_*.npz (same schema as the texture disc script). "
        "Only used when pack horizon matches this killtest (16).",
    )
    args = parser.parse_args()

    smoke.set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    n_var = _n_variates(args.dataset)
    resolution = args.resolution
    protocol = build_protocol(args.dataset, n_var, lookback=LOOKBACK)
    limit = 2 if args.smoke else None

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
        disc_metrics[fake_name] = _run_original_discriminator(
            dataset=args.dataset,
            past=pack["past"],
            y_true=pack["gt"],
            fake=fake,
            indices=pack["window_ids"],
            fake_source=fake_name,
            output_dir=args.output,
            device=device,
            smoke_mode=args.smoke,
            seed=args.seed,
        )

    mmpd_note = None
    if args.import_mmpd_packs_from is not None:
        # Reuse the stock pack loader path when a compatible hz=16 pack exists.
        from utils.eval_discriminator_texture_staged_vs_mmpd import load_npz

        candidates = [
            args.import_mmpd_packs_from / "raw" / f"mmpd_{args.dataset}.npz",
            args.import_mmpd_packs_from / f"mmpd_{args.dataset}.npz",
        ]
        mmpd_path = next((p for p in candidates if p.is_file()), None)
        if mmpd_path is None:
            mmpd_note = f"no mmpd_{args.dataset}.npz under {args.import_mmpd_packs_from}"
        else:
            mmpd_pack = load_npz(mmpd_path)
            hz = int(mmpd_pack["y_true"].shape[-1])
            if hz != HORIZON:
                mmpd_note = (
                    f"skipped MMPD pack {mmpd_path}: horizon={hz} != killtest {HORIZON}. "
                    "Point --import-mmpd-packs-from at an hz16 pack to run the stock MMPD disc side-by-side."
                )
            else:
                past_m = mmpd_pack.get("past")
                if past_m is None:
                    mmpd_note = f"{mmpd_path} missing past[]; cannot run disc without lookback"
                else:
                    disc_metrics["mmpd"] = _run_original_discriminator(
                        dataset=args.dataset,
                        past=past_m,
                        y_true=mmpd_pack["y_true"],
                        fake=mmpd_pack["samples"][:, :, 0, :],
                        indices=mmpd_pack["indices"],
                        fake_source="mmpd",
                        output_dir=args.output,
                        device=device,
                        smoke_mode=args.smoke,
                        seed=args.seed,
                    )
                    mmpd_note = f"ran stock disc on {mmpd_path}"

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
        "discriminator_backend": "utils.eval_discriminator_texture_staged_vs_mmpd.train_classifier",
        "mmpd_import": mmpd_note,
        "refine_mae": {
            "naive": float(np.mean(np.abs(pack["naive"] - pack["gt"]))),
            "refined": float(np.mean(np.abs(pack["refined"] - pack["gt"]))),
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
