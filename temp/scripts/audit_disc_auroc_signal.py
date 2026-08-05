#!/usr/bin/env python3
"""Audit whether ~0.50 disc AUROC is a pipeline bug or protocol/information loss.

Reproduces key checks on a finished ablation pack (raw binary + MMPD npz):
  1. Pre/post-snap horizon MSE/MAE/identity (real↔binary, real↔mmpd, binary↔mmpd)
  2. Label/pairing under unique_abs (0=real, 1=fake; same abs_t×variate)
  3. Post bin-center L-slice L2 / identical-rate
  4. Trivial baselines on the same unique_abs test slices (feature AUROC)
  5. Optional dense-protocol feature AUROC for contrast

Example:
  python temp/scripts/audit_disc_auroc_signal.py \\
    --pack-dir results/datasets/08-04-1843-ablation-disc-l8-l16-ETTh1-c128-valtest80-byvar \\
    --dataset ETTh1 --slice-len 8 --out temp/disc_auroc_audit_etth1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.train_multivariate_pipeline import load_dataset
from utils.disc_bin_center_shift import bin_center_shift
from utils.disc_shared import binary_auroc, split_windows
from utils.eval_discriminator_binary_vs_mmpd_univariate import (
    UnivariateRealVsFakeDataset,
    _unique_absolute_slice_items,
)
from utils.patch_refine_ordinal_ladder import (
    legal_patch_refine_levels_dataset_z,
    snap_to_patch_refine_levels,
)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _maxabs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def _frac_eq(a: np.ndarray, b: np.ndarray, atol: float = 1e-5) -> float:
    return float(np.mean(np.abs(a - b) <= atol))


def _dist_report(name: str, real: np.ndarray, fake: np.ndarray) -> Dict[str, float]:
    return {
        "pair": name,
        "mse": _mse(real, fake),
        "mae": _mae(real, fake),
        "max_abs": _maxabs(real, fake),
        "frac_eq": _frac_eq(real, fake),
        "identical_allclose": bool(np.allclose(real, fake)),
    }


def _find_npz(pack_dir: Path, prefix: str) -> Path:
    raw = pack_dir / "raw"
    hits = sorted(raw.glob(f"{prefix}*.npz"))
    if not hits:
        raise FileNotFoundError(f"no {prefix}*.npz under {raw}")
    if len(hits) > 1:
        print(f"[warn] multiple {prefix} packs; using {hits[0].name}", flush=True)
    return hits[0]


def _load_packs(pack_dir: Path) -> Dict[str, Any]:
    b_path = _find_npz(pack_dir, "binary_")
    m_path = _find_npz(pack_dir, "mmpd_")
    b = np.load(b_path, allow_pickle=True)
    m = np.load(m_path, allow_pickle=True)
    past = np.asarray(b["past"], dtype=np.float32)
    gt = np.asarray(b["y_true"], dtype=np.float32)
    samples_b = np.asarray(b["samples"], dtype=np.float32)
    samples_m = np.asarray(m["samples"], dtype=np.float32)
    fake_b = samples_b[:, :, 0, :] if samples_b.ndim == 4 else samples_b
    fake_m = samples_m[:, :, 0, :] if samples_m.ndim == 4 else samples_m
    gt_m = np.asarray(m["y_true"], dtype=np.float32)
    idx_b = np.asarray(b["indices"], dtype=np.int64)
    idx_m = np.asarray(m["indices"], dtype=np.int64)
    if not np.array_equal(idx_b, idx_m):
        raise RuntimeError("binary/MMPD indices differ")
    if "series_starts" not in b.files:
        raise RuntimeError(f"{b_path} missing series_starts")
    canvas_height = int(np.asarray(b["canvas_height"]).reshape(-1)[0]) if "canvas_height" in b.files else 128
    return {
        "binary_path": str(b_path),
        "mmpd_path": str(m_path),
        "past": past,
        "gt": gt,
        "gt_mmpd": gt_m,
        "fake_binary": fake_b.astype(np.float32),
        "fake_mmpd": fake_m.astype(np.float32),
        "indices": idx_b,
        "series_starts": np.asarray(b["series_starts"], dtype=np.int64),
        "canvas_height": canvas_height,
        "pack_splits": (
            [str(x) for x in np.asarray(b["pack_splits"]).tolist()]
            if "pack_splits" in b.files
            else None
        ),
    }


def _build_ladder(dataset: str, n_var: int, lookback: int, horizon: int):
    print(f"[ladder] load_dataset({dataset}) use_ordinal_window_norm=True …", flush=True)
    _, _, _, norm_stats = load_dataset(
        dataset,
        list(range(n_var)),
        stride=1,
        test_stride=4,
        lookback=lookback,
        horizon=horizon,
        ordinal_tie_atol=1e-6,
        use_ordinal_window_norm=True,
    )
    ladder = norm_stats.get("ordinal_ladder")
    if ladder is None:
        raise RuntimeError("ordinal_ladder missing from load_dataset norm_stats")
    return ladder


def _legal_levels_batched(
    past: np.ndarray,
    *,
    ladder,
    canvas_height: int,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    chunks = []
    n = past.shape[0]
    for i0 in range(0, n, batch_size):
        i1 = min(n, i0 + batch_size)
        chunks.append(
            legal_patch_refine_levels_dataset_z(
                past[i0:i1],
                ladder=ladder,
                canvas_height=canvas_height,
                device=device,
            )
        )
    return np.concatenate(chunks, axis=0).astype(np.float32)


def _split_args(
    *,
    lookback: int,
    horizon: int,
    pack_test_stride: int,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        seed=seed,
        lookback=lookback,
        horizon=horizon,
        pack_test_stride=pack_test_stride,
        max_windows=None,
    )


def _extract_slice_pairs(
    real: np.ndarray,
    fake: np.ndarray,
    items: Sequence[Tuple[int, int, int, int]],
    slice_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return paired real/fake (N_pair, L) plus variate/offset from unique_abs items."""
    by_key: Dict[Tuple[int, int, int], Dict[int, np.ndarray]] = {}
    for w, o, v, lab in items:
        key = (int(w), int(o), int(v))
        src = fake if int(lab) == 1 else real
        by_key.setdefault(key, {})[int(lab)] = src[w, v, o : o + slice_len].astype(np.float32)
    xs_r, xs_f, vars_, offs = [], [], [], []
    for (w, o, v), labs in by_key.items():
        if 0 not in labs or 1 not in labs:
            raise RuntimeError(f"unpaired unique_abs key {(w, o, v)} labs={list(labs)}")
        xs_r.append(labs[0])
        xs_f.append(labs[1])
        vars_.append(v)
        offs.append(o)
    return (
        np.stack(xs_r, axis=0),
        np.stack(xs_f, axis=0),
        np.asarray(vars_, dtype=np.int64),
        np.asarray(offs, dtype=np.int64),
    )


def _bin_center_batch(
    slices: np.ndarray,
    legal_levels: np.ndarray,
    windows: np.ndarray,
    variates: np.ndarray,
) -> np.ndarray:
    """Apply per-example bin_center_shift to (N, L) slices."""
    out = np.empty_like(slices, dtype=np.float32)
    for i in range(slices.shape[0]):
        w = int(windows[i])
        v = int(variates[i])
        seg = slices[i][None, None, :]  # (1,1,L)
        levels = legal_levels[w, v : v + 1, :][None, :, :]  # (1,1,H)
        shifted, _ = bin_center_shift(seg, levels, reduce="per_variate")
        out[i] = shifted[0, 0]
    return out


def _feature_auroc(real_x: np.ndarray, fake_x: np.ndarray) -> Dict[str, float]:
    """Trivial separable-signal probes on paired L-slices (higher = more separable)."""
    n = real_x.shape[0]
    labels = np.concatenate([np.zeros(n), np.ones(n)]).astype(np.float64)

    def _auc(score_real: np.ndarray, score_fake: np.ndarray) -> float:
        scores = np.concatenate([score_real, score_fake]).astype(np.float64)
        return binary_auroc(labels, scores)

    # Positive direction = "more fake-like". We take max(auc, 1-auc) only in summary
    # via abs_lift; raw AUROC kept as-is (may be <0.5 if polarity flipped).
    mean_r, mean_f = real_x.mean(axis=1), fake_x.mean(axis=1)
    std_r, std_f = real_x.std(axis=1), fake_x.std(axis=1)
    l2_pair = np.linalg.norm(real_x - fake_x, axis=1)
    # Score = own mean / std / range / first-diff energy (no oracle pairing needed at test)
    out = {
        "auroc_mean": _auc(mean_r, mean_f),
        "auroc_std": _auc(std_r, std_f),
        "auroc_abs_mean": _auc(np.abs(mean_r), np.abs(mean_f)),
        "auroc_range": _auc(real_x.max(1) - real_x.min(1), fake_x.max(1) - fake_x.min(1)),
        "auroc_diff_energy": _auc(
            np.mean(np.diff(real_x, axis=1) ** 2, axis=1),
            np.mean(np.diff(fake_x, axis=1) ** 2, axis=1),
        ),
        # Oracle-ish: how large is real↔fake L2? Not a classifier score, reported separately.
        "mean_l2_real_fake": float(l2_pair.mean()),
        "median_l2_real_fake": float(np.median(l2_pair)),
        "frac_l2_zero": float(np.mean(l2_pair <= 1e-6)),
        "frac_identical_slice": _frac_eq(real_x, fake_x, atol=1e-5),
    }
    # Tiny logistic on [mean, std, range, diff_energy] — sklearn-free
    feats_r = np.stack(
        [
            mean_r,
            std_r,
            real_x.max(1) - real_x.min(1),
            np.mean(np.diff(real_x, axis=1) ** 2, axis=1),
        ],
        axis=1,
    )
    feats_f = np.stack(
        [
            mean_f,
            std_f,
            fake_x.max(1) - fake_x.min(1),
            np.mean(np.diff(fake_x, axis=1) ** 2, axis=1),
        ],
        axis=1,
    )
    X = np.concatenate([feats_r, feats_f], axis=0).astype(np.float64)
    y = labels
    # Standardize
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xn = (X - mu) / sd
    # Closed-form ridge logistic via a few GD steps is enough for smoke separability
    rng = np.random.default_rng(0)
    # Train/test split chronological-ish: first 80% train
    n_tot = Xn.shape[0]
    order = rng.permutation(n_tot)
    n_tr = int(0.8 * n_tot)
    tr, te = order[:n_tr], order[n_tr:]
    w = np.zeros(Xn.shape[1], dtype=np.float64)
    b0 = 0.0
    for _ in range(400):
        z = Xn[tr] @ w + b0
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        err = p - y[tr]
        w -= 0.2 * (Xn[tr].T @ err / len(tr) + 1e-3 * w)
        b0 -= 0.2 * float(err.mean())
    z_te = Xn[te] @ w + b0
    p_te = 1.0 / (1.0 + np.exp(-np.clip(z_te, -30, 30)))
    out["auroc_logistic_4feat_holdout"] = binary_auroc(y[te], p_te)
    # Also full-set in-sample (upper bound on linear signal)
    z_all = Xn @ w + b0
    p_all = 1.0 / (1.0 + np.exp(-np.clip(z_all, -30, 30)))
    out["auroc_logistic_4feat_insample"] = binary_auroc(y, p_all)
    return out


def _check_labels(items: Sequence[Tuple[int, int, int, int]]) -> Dict[str, Any]:
    labs = [lab for *_, lab in items]
    n0 = sum(1 for x in labs if x == 0)
    n1 = sum(1 for x in labs if x == 1)
    # Pairing: every (w,o,v) with lab0 has lab1
    keys0 = {(w, o, v) for (w, o, v, lab) in items if lab == 0}
    keys1 = {(w, o, v) for (w, o, v, lab) in items if lab == 1}
    return {
        "n_items": len(items),
        "n_real_label0": n0,
        "n_fake_label1": n1,
        "balanced": n0 == n1,
        "paired_keys_equal": keys0 == keys1,
        "n_unique_pairs": len(keys0),
        "convention": "0=real/GT, 1=fake",
    }


def _dense_items(
    windows: np.ndarray,
    *,
    horizon: int,
    slice_len: int,
    n_var: int,
    seed: int,
    max_pairs: int,
) -> List[Tuple[int, int, int, int]]:
    offsets = list(range(0, horizon - slice_len + 1))
    real_items = [
        (int(w), int(o), int(v), 0)
        for w in windows
        for o in offsets
        for v in range(n_var)
    ]
    fake_items = [
        (int(w), int(o), int(v), 1)
        for w in windows
        for o in offsets
        for v in range(n_var)
    ]
    rng = np.random.default_rng(seed)
    n = min(len(real_items), max_pairs)
    # Paired dense sample (same keys) — fairer than independent draws
    keys = [(int(w), int(o), int(v)) for (w, o, v, _) in real_items]
    pick = rng.choice(len(keys), size=n, replace=False)
    items = []
    for i in pick:
        w, o, v = keys[int(i)]
        items.append((w, o, v, 0))
        items.append((w, o, v, 1))
    return items


def audit_one_source(
    *,
    name: str,
    real: np.ndarray,
    fake: np.ndarray,
    legal_levels: np.ndarray,
    series_starts: np.ndarray,
    splits: Dict[str, np.ndarray],
    lookback: int,
    slice_len: int,
    seed: int,
    do_dense: bool,
) -> Dict[str, Any]:
    horizon = int(real.shape[-1])
    n_var = int(real.shape[1])
    out: Dict[str, Any] = {"source": name, "slice_len": slice_len}

    # Horizon-level post-snap distances (full windows)
    out["post_snap_horizon"] = _dist_report(f"real↔{name}", real, fake)

    test_windows = np.asarray(splits["test"], dtype=np.int64)
    items = _unique_absolute_slice_items(
        test_windows,
        horizon=horizon,
        slice_len=slice_len,
        n_var=n_var,
        offset_stride=1,
        series_starts=series_starts,
        lookback=lookback,
        seed=seed,
    )
    out["label_check_unique_abs_test"] = _check_labels(items)

    # Map items → window ids for bin-center levels lookup
    pair_keys = [(w, o, v) for (w, o, v, lab) in items if lab == 0]
    windows_arr = np.asarray([k[0] for k in pair_keys], dtype=np.int64)
    real_x, fake_x, vars_, offs = _extract_slice_pairs(real, fake, items, slice_len)
    out["n_test_pairs_unique_abs"] = int(real_x.shape[0])
    out["pre_bincenter_slices"] = {
        **_dist_report(f"real↔{name}_L{slice_len}", real_x, fake_x),
        **_feature_auroc(real_x, fake_x),
    }

    # Post bin-center (protocol)
    real_c = _bin_center_batch(real_x, legal_levels, windows_arr, vars_)
    fake_c = _bin_center_batch(fake_x, legal_levels, windows_arr, vars_)
    out["post_bincenter_slices"] = {
        **_dist_report(f"real↔{name}_L{slice_len}_binc", real_c, fake_c),
        **_feature_auroc(real_c, fake_c),
    }

    # Sanity: dataset __getitem__ label wiring on a few examples
    ds = UnivariateRealVsFakeDataset(
        real,
        fake,
        real,  # past unused when candidate_only
        test_windows,
        slice_len,
        seed=seed,
        include_past=False,
        apply_zscore=False,
        apply_bin_center_shift=True,
        legal_levels=legal_levels,
        unique_absolute_slices=True,
        series_starts=series_starts,
        lookback=lookback,
    )
    # Verify a handful of items: label 0 pulls real, label 1 pulls fake
    mismatches = 0
    checked = 0
    for i in range(min(64, len(ds))):
        w, o, v, lab = ds.items[i]
        x, _off, ylab, _ww, _vv = ds[i]
        x_np = x.numpy()[0]
        src = fake if lab == 1 else real
        raw = src[w, v, o : o + slice_len]
        levels = legal_levels[w, v : v + 1][None, :, :]
        expected, _ = bin_center_shift(raw[None, None, :], levels, reduce="per_variate")
        if not np.allclose(x_np, expected[0, 0], atol=1e-5):
            mismatches += 1
        if float(ylab) != float(lab):
            mismatches += 1
        checked += 1
    out["getitem_bincenter_check"] = {
        "checked": checked,
        "mismatches": mismatches,
        "ok": mismatches == 0,
    }

    if do_dense:
        dense_items = _dense_items(
            test_windows,
            horizon=horizon,
            slice_len=slice_len,
            n_var=n_var,
            seed=seed + 99,
            max_pairs=min(8000, int(test_windows.size) * (horizon - slice_len + 1) * n_var),
        )
        d_keys = [(w, o, v) for (w, o, v, lab) in dense_items if lab == 0]
        d_windows = np.asarray([k[0] for k in d_keys], dtype=np.int64)
        d_real, d_fake, d_vars, _ = _extract_slice_pairs(real, fake, dense_items, slice_len)
        d_real_c = _bin_center_batch(d_real, legal_levels, d_windows, d_vars)
        d_fake_c = _bin_center_batch(d_fake, legal_levels, d_windows, d_vars)
        out["dense_paired_post_bincenter"] = {
            "n_pairs": int(d_real.shape[0]),
            **_dist_report(f"dense_real↔{name}", d_real_c, d_fake_c),
            **_feature_auroc(d_real_c, d_fake_c),
        }

    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack-dir", type=Path, required=True)
    p.add_argument("--dataset", default="ETTh1")
    p.add_argument("--slice-len", type=int, default=8)
    p.add_argument("--lookback", type=int, default=336)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--pack-test-stride", type=int, default=4)
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument("--val-fraction", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--canvas-height", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-dense", action="store_true")
    p.add_argument("--sources", nargs="+", default=["binary", "mmpd"])
    p.add_argument("--out", type=Path, default=Path("temp/disc_auroc_audit.json"))
    p.add_argument("--save-snapped", type=Path, default=None)
    args = p.parse_args()

    packs = _load_packs(args.pack_dir)
    past = packs["past"]
    gt = packs["gt"]
    N, V, H = gt.shape
    canvas_h = int(args.canvas_height or packs["canvas_height"])
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    report: Dict[str, Any] = {
        "pack_dir": str(args.pack_dir),
        "binary_path": packs["binary_path"],
        "mmpd_path": packs["mmpd_path"],
        "shapes": {
            "past": list(past.shape),
            "gt": list(gt.shape),
            "fake_binary": list(packs["fake_binary"].shape),
            "fake_mmpd": list(packs["fake_mmpd"].shape),
        },
        "canvas_height": canvas_h,
        "pack_splits": packs["pack_splits"],
        "gt_binary_vs_mmpd_identical": bool(np.allclose(gt, packs["gt_mmpd"])),
        "indices_aligned": True,
        "device": str(device),
    }

    print("=== PRE-SNAP distances ===", flush=True)
    pre = {
        "real↔binary": _dist_report("real↔binary", gt, packs["fake_binary"]),
        "real↔mmpd": _dist_report("real↔mmpd", packs["gt_mmpd"], packs["fake_mmpd"]),
        "binary↔mmpd": _dist_report("binary↔mmpd", packs["fake_binary"], packs["fake_mmpd"]),
        "gt_b↔gt_m": _dist_report("gt_b↔gt_m", gt, packs["gt_mmpd"]),
    }
    for k, v in pre.items():
        print(
            f"  {k}: MSE={v['mse']:.6f} MAE={v['mae']:.6f} "
            f"maxabs={v['max_abs']:.6f} frac_eq={v['frac_eq']:.4f}",
            flush=True,
        )
    report["pre_snap"] = pre

    ladder = _build_ladder(args.dataset, V, args.lookback, args.horizon)
    print(f"[snap] legal_levels canvas_height={canvas_h} on {device} …", flush=True)
    legal_levels = _legal_levels_batched(
        past, ladder=ladder, canvas_height=canvas_h, device=device,
    )
    gt_s, gt_st = snap_to_patch_refine_levels(gt, legal_levels)
    b_s, b_st = snap_to_patch_refine_levels(packs["fake_binary"], legal_levels)
    m_s, m_st = snap_to_patch_refine_levels(packs["fake_mmpd"], legal_levels)
    report["snap_stats"] = {"gt": gt_st, "binary": b_st, "mmpd": m_st}
    print("  snap_stats", report["snap_stats"], flush=True)

    if args.save_snapped:
        args.save_snapped.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save_snapped,
            gt=gt_s,
            binary=b_s,
            mmpd=m_s,
            past=past,
            legal_levels=legal_levels,
            series_starts=packs["series_starts"],
            indices=packs["indices"],
        )
        print(f"[save] snapped → {args.save_snapped}", flush=True)

    split_ns = _split_args(
        lookback=args.lookback,
        horizon=args.horizon,
        pack_test_stride=args.pack_test_stride,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    splits = split_windows(
        N,
        split_ns,
        args.dataset,
        indices=packs["indices"],
        lookback=args.lookback,
        horizon=args.horizon,
        test_stride=args.pack_test_stride,
        series_starts=packs["series_starts"],
    )
    report["splits"] = {k: int(np.asarray(v).size) for k, v in splits.items()}
    print(f"[split] {report['splits']}", flush=True)

    sources = {
        "binary": b_s,
        "mmpd": m_s,
    }
    report["sources"] = {}
    for src in args.sources:
        print(f"\n=== source={src} L={args.slice_len} ===", flush=True)
        one = audit_one_source(
            name=src,
            real=gt_s,
            fake=sources[src],
            legal_levels=legal_levels,
            series_starts=packs["series_starts"],
            splits=splits,
            lookback=args.lookback,
            slice_len=args.slice_len,
            seed=args.seed + (0 if src == "binary" else 17),
            do_dense=not args.no_dense,
        )
        report["sources"][src] = one
        pb = one["post_bincenter_slices"]
        print(
            f"  post-bincenter: MSE={pb['mse']:.6f} frac_eq={pb['frac_eq']:.4f} "
            f"mean_l2={pb['mean_l2_real_fake']:.4f} "
            f"logistic_holdout_auroc={pb['auroc_logistic_4feat_holdout']:.4f} "
            f"auroc_std={pb['auroc_std']:.4f}",
            flush=True,
        )
        print(f"  getitem_ok={one['getitem_bincenter_check']['ok']}", flush=True)

    # Load finished run AUROC if present
    auroc_table = args.pack_dir / "auroc_table.json"
    if auroc_table.exists():
        report["finished_run_auroc_table"] = json.loads(auroc_table.read_text())

    # Verdict heuristics
    verdict_bits = []
    any_identical = any(
        report["sources"][s]["post_bincenter_slices"]["identical_allclose"]
        for s in report["sources"]
    )
    max_log = max(
        report["sources"][s]["post_bincenter_slices"]["auroc_logistic_4feat_holdout"]
        for s in report["sources"]
    )
    max_pre_mse = max(pre["real↔binary"]["mse"], pre["real↔mmpd"]["mse"])
    if any_identical:
        verdict_bits.append("BUG_SUSPECT: post-bincenter real≡fake")
    if max_pre_mse < 1e-6:
        verdict_bits.append("BUG_SUSPECT: pre-snap fakes identical to GT")
    if max_log < 0.55:
        verdict_bits.append(
            "PROTOCOL_OR_HARD: trivial 4-feat logistic also ~chance after bin-center "
            f"(max holdout AUROC={max_log:.3f})"
        )
    elif max_log >= 0.65:
        verdict_bits.append(
            "MIXED: trivial features beat chance after protocol "
            f"(logistic holdout={max_log:.3f}) but transformer disc sat at ~0.5 — "
            "check train/capacity/early-stop, not empty tensors"
        )
    else:
        verdict_bits.append(
            f"WEAK_SIGNAL: trivial logistic holdout={max_log:.3f}; protocol largely strips cue"
        )
    report["verdict_hints"] = verdict_bits
    print("\nVERDICT HINTS:", *verdict_bits, sep="\n  - ", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=float))
    print(f"\n[write] {args.out}", flush=True)


if __name__ == "__main__":
    main()
