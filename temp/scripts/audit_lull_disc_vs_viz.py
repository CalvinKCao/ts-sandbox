#!/usr/bin/env python3
"""Empirical audit: ETTh2 LULL disc AUROC~0.5 vs patch/snap viz.

Answers: was disc trained on the same window_norm_grid instance-norm snapped
space the viz shows? Why AUROC~0.5 despite obvious H96 differences?

Outputs JSON to --out-json and prints a short summary.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from temp.scripts.eval_ablation_disc_l8_l16 import (  # noqa: E402
    _snap_bundle,
    load_ablation_run,
)
from utils.disc_bin_center_shift import bin_center_shift  # noqa: E402
from utils.disc_shared import binary_auroc, split_windows  # noqa: E402
from utils.dual_scale_bin_filter import align_mmpd_to_binary_dataset_norm  # noqa: E402
from utils.eval_discriminator_binary_vs_mmpd_univariate import (  # noqa: E402
    UnivariateRealVsFakeDataset,
    _unique_absolute_slice_items,
)
from utils.eval_discriminator_texture_staged_vs_mmpd import (  # noqa: E402
    binary_mmpd_train_scaler_map,
)
from utils.eval_mmpd_gaussian_anchor import DEFAULT_MMPD_DATA  # noqa: E402
from utils.forecast_pack_reduce import reduce_pack_forecast  # noqa: E402
from utils.patch_refine_ordinal_ladder import snap_to_patch_refine_levels  # noqa: E402

LULL = 5
VIZ_POOLS = (1169, 1393, 1310, 1340)
L8_OFF = 44
L16_OFF = 40


def _mse(a, b):
    return float(np.mean((a - b) ** 2))


def _mae(a, b):
    return float(np.mean(np.abs(a - b)))


def _frac_eq(a, b, atol=1e-5):
    return float(np.mean(np.abs(a - b) <= atol))


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _find_npz(raw: Path, prefix: str) -> Path:
    hits = sorted(p for p in raw.glob(f"{prefix}*.npz") if "indices" not in p.name)
    vt = [p for p in hits if "val-test" in p.name or "val_test" in p.name]
    if vt:
        return vt[0]
    if not hits:
        raise FileNotFoundError(f"no {prefix}*.npz under {raw}")
    return hits[0]


def _bin_center_1d(seg: np.ndarray, levels_row: np.ndarray) -> np.ndarray:
    shifted, _ = bin_center_shift(
        seg[None, None, :],
        levels_row[None, None, :],
        reduce="per_variate",
    )
    return shifted[0, 0].astype(np.float32)


def _train_probe(Xtr, ytr, Xte, yte, *, L: int, epochs: int = 25) -> Dict[str, float]:
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    Xtr_n = ((Xtr - mu) / sd).astype(np.float32)
    Xte_n = ((Xte - mu) / sd).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(model: nn.Module, name: str) -> float:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()
        loader = DataLoader(
            TensorDataset(torch.from_numpy(Xtr_n), torch.from_numpy(ytr)),
            batch_size=256,
            shuffle=True,
        )
        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(model(xb).squeeze(-1), yb)
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(Xte_n).to(device)).squeeze(-1).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        return float(binary_auroc(yte, probs))

    logistic = nn.Linear(L, 1).to(device)
    mlp = nn.Sequential(
        nn.Linear(L, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)
    return {
        "logistic": run(logistic, "logistic"),
        "mlp64x2": run(mlp, "mlp"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pack",
        type=Path,
        default=REPO
        / "results/datasets/08-04-2009-ablation-disc-l8-l16-ETTh2-c128-valtest80-byvar",
    )
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=REPO
        / "results/ckpts/08-04-4601319-ETTh2-binary_window_norm_patch_refine_canvas128_p64x6_etth2",
    )
    ap.add_argument(
        "--config",
        default="configs/binary_window_norm_patch_refine_canvas128_p64x6_etth2.yaml",
    )
    ap.add_argument("--out-json", type=Path, default=REPO / "temp/lull_disc_vs_viz_audit_numbers.json")
    ap.add_argument("--lookback", type=int, default=336)
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--mmpd-data-dir", type=Path, default=DEFAULT_MMPD_DATA)
    args = ap.parse_args()

    t0 = time.time()
    pack = args.pack.expanduser().resolve()
    ckpt = args.ckpt.expanduser().resolve()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    binary_path = _find_npz(pack / "raw", "binary_")
    mmpd_path = _find_npz(pack / "raw", "mmpd_")
    binary_pack = _load_npz(binary_path)
    mmpd_pack = _load_npz(mmpd_path)
    canvas_height = int(np.asarray(binary_pack["canvas_height"]).reshape(-1)[0])
    past = np.asarray(binary_pack["past"], dtype=np.float32)
    gt_pre = np.asarray(binary_pack["y_true"], dtype=np.float32)
    binary_pre = reduce_pack_forecast(binary_pack, agg="sample0")
    mmpd_pre_raw = reduce_pack_forecast(mmpd_pack, agg="sample0")
    indices = np.asarray(binary_pack["indices"], dtype=np.int64)
    series_starts = np.asarray(binary_pack["series_starts"], dtype=np.int64)
    n, v, h = gt_pre.shape
    assert v > LULL

    run, _stages, kind = load_ablation_run("ETTh2", ckpt)
    ns = SimpleNamespace(
        dataset="ETTh2",
        fake_agg="sample0",
        lookback=args.lookback,
        horizon=args.horizon,
        mmpd_data_dir=str(args.mmpd_data_dir),
    )
    print(f"[audit] snap_bundle kind={kind} canvas={canvas_height} device={device}", flush=True)
    snapped = _snap_bundle(
        binary_pack=binary_pack,
        mmpd_pack=mmpd_pack,
        run=run,
        ladder=None,
        args=ns,
        device=device,
        canvas_height=canvas_height,
        ckpt_root=ckpt,
        config_path=args.config,
    )
    snap_mode = str(snapped["snap_mode"])
    lattice = snapped.get("lattice") or {}
    snap_meta = dict(lattice.get("snap_meta") or snapped.get("snap_meta") or {})
    legal = np.asarray(snapped["legal_levels"], dtype=np.float32)
    gt = np.asarray(snapped["gt"], dtype=np.float32)
    binary = np.asarray(snapped["binary"], dtype=np.float32)
    mmpd = np.asarray(snapped["mmpd"], dtype=np.float32)

    # Align pre-snap MMPD into binary dataset-z for fair pre-snap MSE.
    scalers = binary_mmpd_train_scaler_map(ns, run)
    mmpd_pre, _align = align_mmpd_to_binary_dataset_norm(
        binary_y_true=gt_pre,
        mmpd_y_true=np.asarray(mmpd_pack["y_true"], dtype=np.float32),
        mmpd_fakes=mmpd_pre_raw,
        **scalers,
    )

    pool_to_local = {int(indices[i]): i for i in range(n)}
    viz_locals = []
    for pool in VIZ_POOLS:
        if pool not in pool_to_local:
            raise RuntimeError(f"viz pool {pool} missing from pack indices")
        viz_locals.append(pool_to_local[pool])

    # --- Check 2: MSE/MAE at stages for viz windows, LULL only ---
    stage_rows: List[Dict[str, Any]] = []
    for pool, local in zip(VIZ_POOLS, viz_locals):
        row: Dict[str, Any] = {"pool": int(pool), "local": int(local)}
        for stage, g, b, m in (
            ("pre_snap_H96", gt_pre[local, LULL], binary_pre[local, LULL], mmpd_pre[local, LULL]),
            ("post_snap_H96", gt[local, LULL], binary[local, LULL], mmpd[local, LULL]),
        ):
            row[stage] = {
                "gt_vs_mmpd_mse": _mse(g, m),
                "gt_vs_mmpd_mae": _mae(g, m),
                "gt_vs_binary_mse": _mse(g, b),
                "gt_vs_binary_mae": _mae(g, b),
                "binary_vs_mmpd_mse": _mse(b, m),
                "frac_gt_eq_mmpd": _frac_eq(g, m),
                "frac_gt_eq_binary": _frac_eq(g, b),
            }
        for L, off in ((8, L8_OFF), (16, L16_OFF)):
            g_s = gt[local, LULL, off : off + L]
            b_s = binary[local, LULL, off : off + L]
            m_s = mmpd[local, LULL, off : off + L]
            levels = legal[local, LULL]
            g_bc = _bin_center_1d(g_s, levels)
            b_bc = _bin_center_1d(b_s, levels)
            m_bc = _bin_center_1d(m_s, levels)
            row[f"post_snap_L{L}_off{off}"] = {
                "gt_vs_mmpd_mse": _mse(g_s, m_s),
                "gt_vs_binary_mse": _mse(g_s, b_s),
                "binary_vs_mmpd_mse": _mse(b_s, m_s),
            }
            row[f"post_binc_L{L}_off{off}"] = {
                "gt_vs_mmpd_mse": _mse(g_bc, m_bc),
                "gt_vs_binary_mse": _mse(g_bc, b_bc),
                "binary_vs_mmpd_mse": _mse(b_bc, m_bc),
                "gt_vs_mmpd_mae": _mae(g_bc, m_bc),
                "frac_gt_eq_mmpd": _frac_eq(g_bc, m_bc),
                "frac_gt_eq_binary": _frac_eq(g_bc, b_bc),
                "gt_series": g_bc.tolist(),
                "mmpd_series": m_bc.tolist(),
                "binary_series": b_bc.tolist(),
            }
        stage_rows.append(row)

    # Aggregate over viz windows
    def agg_key(stage: str, metric: str) -> float:
        return float(np.mean([r[stage][metric] for r in stage_rows]))

    viz_agg = {
        "pre_snap_H96_gt_mmpd_mse": agg_key("pre_snap_H96", "gt_vs_mmpd_mse"),
        "pre_snap_H96_gt_binary_mse": agg_key("pre_snap_H96", "gt_vs_binary_mse"),
        "post_snap_H96_gt_mmpd_mse": agg_key("post_snap_H96", "gt_vs_mmpd_mse"),
        "post_snap_H96_gt_binary_mse": agg_key("post_snap_H96", "gt_vs_binary_mse"),
        "post_binc_L8_gt_mmpd_mse": agg_key(f"post_binc_L8_off{L8_OFF}", "gt_vs_mmpd_mse"),
        "post_binc_L8_gt_binary_mse": agg_key(f"post_binc_L8_off{L8_OFF}", "gt_vs_binary_mse"),
        "post_binc_L8_frac_gt_eq_mmpd": agg_key(f"post_binc_L8_off{L8_OFF}", "frac_gt_eq_mmpd"),
        "post_binc_L16_gt_mmpd_mse": agg_key(f"post_binc_L16_off{L16_OFF}", "gt_vs_mmpd_mse"),
    }

    # --- Full-pack LULL H96 / L8 distances ---
    lull_pack = {
        "pre_snap_H96": {
            "gt_mmpd_mse": _mse(gt_pre[:, LULL], mmpd_pre[:, LULL]),
            "gt_binary_mse": _mse(gt_pre[:, LULL], binary_pre[:, LULL]),
            "frac_gt_eq_mmpd": _frac_eq(gt_pre[:, LULL], mmpd_pre[:, LULL]),
        },
        "post_snap_H96": {
            "gt_mmpd_mse": _mse(gt[:, LULL], mmpd[:, LULL]),
            "gt_binary_mse": _mse(gt[:, LULL], binary[:, LULL]),
            "frac_gt_eq_mmpd": _frac_eq(gt[:, LULL], mmpd[:, LULL]),
            "mean_abs_snap_delta_gt": float(lattice["gt_snap"]["mean_abs_snap_delta"]),
            "mean_abs_snap_delta_mmpd": float(lattice["mmpd_snap"]["mean_abs_snap_delta"]),
            "mean_abs_snap_delta_binary": float(lattice["binary_snap"]["mean_abs_snap_delta"]),
        },
    }

    # --- Check 5/4: unique_abs LULL-only test pairs + probe ---
    # Match campaign: --train-fraction 0.8 --val-fraction 0 on val+test pack
    # → 907/101/279 (see pack metrics).
    split_ns = SimpleNamespace(
        train_fraction=0.8,
        val_fraction=0.0,
        seed=args.seed,
        lookback=args.lookback,
        horizon=args.horizon,
        pack_test_stride=1,
        max_windows=None,
    )
    splits = split_windows(
        n,
        split_ns,
        "ETTh2",
        indices=indices,
        lookback=args.lookback,
        horizon=args.horizon,
        test_stride=1,
        series_starts=series_starts,
    )
    tr_idx = np.asarray(splits["train"], dtype=np.int64)
    va_idx = np.asarray(splits["val"], dtype=np.int64)
    te_idx = np.asarray(splits["test"], dtype=np.int64)
    print(
        f"[audit] split train/val/test = {len(tr_idx)}/{len(va_idx)}/{len(te_idx)}",
        flush=True,
    )

    items_te = _unique_absolute_slice_items(
        te_idx,
        horizon=args.horizon,
        slice_len=8,
        n_var=v,
        offset_stride=1,
        series_starts=series_starts,
        lookback=args.lookback,
        seed=args.seed,
    )
    # LULL-only keys
    lull_keys = [(w, o, vv) for (w, o, vv, lab) in items_te if lab == 0 and vv == LULL]
    print(f"[audit] LULL unique_abs test keys={len(lull_keys)}", flush=True)

    xs_r, xs_f = [], []
    identical = 0
    label_ok = True
    for w, o, vv in lull_keys:
        r = gt[w, vv, o : o + 8]
        f = mmpd[w, vv, o : o + 8]
        levels = legal[w, vv]
        r_bc = _bin_center_1d(r, levels)
        f_bc = _bin_center_1d(f, levels)
        xs_r.append(r_bc)
        xs_f.append(f_bc)
        if np.allclose(r_bc, f_bc, atol=1e-5):
            identical += 1
    xs_r = np.stack(xs_r).astype(np.float32)
    xs_f = np.stack(xs_f).astype(np.float32)
    pair_mse = float(np.mean((xs_r - xs_f) ** 2))
    pair_mae = float(np.mean(np.abs(xs_r - xs_f)))
    pair_l2 = float(np.mean(np.linalg.norm(xs_r - xs_f, axis=1)))

    # Also build via UnivariateRealVsFakeDataset __getitem__ for one viz window
    ds = UnivariateRealVsFakeDataset(
        real=gt,
        fake=mmpd,
        past=past,
        windows=np.asarray(viz_locals, dtype=np.int64),
        slice_len=8,
        seed=0,
        offset_stride=1,
        include_past=False,
        apply_zscore=False,
        apply_bin_center_shift=True,
        legal_levels=legal,
        bin_center_reduce="per_variate",
        unique_absolute_slices=False,
    )
    # Force items for pool1169 L8 off44 LULL real+fake
    local0 = viz_locals[0]
    ds.items = [
        (local0, L8_OFF, LULL, 0),
        (local0, L8_OFF, LULL, 1),
    ]
    x_real, _, lab0, w0, v0 = ds[0]
    x_fake, _, lab1, w1, v1 = ds[1]
    viz_bc = stage_rows[0][f"post_binc_L8_off{L8_OFF}"]
    getitem_vs_viz = {
        "label_real": float(lab0),
        "label_fake": float(lab1),
        "window": int(w0),
        "variate": int(v0),
        "getitem_real": x_real.numpy().reshape(-1).tolist(),
        "getitem_fake": x_fake.numpy().reshape(-1).tolist(),
        "viz_gt_series": viz_bc["gt_series"],
        "viz_mmpd_series": viz_bc["mmpd_series"],
        "max_abs_real_vs_viz_gt": float(
            np.max(np.abs(x_real.numpy().reshape(-1) - np.asarray(viz_bc["gt_series"])))
        ),
        "max_abs_fake_vs_viz_mmpd": float(
            np.max(np.abs(x_fake.numpy().reshape(-1) - np.asarray(viz_bc["mmpd_series"])))
        ),
        "real_identical_fake": bool(np.allclose(x_real.numpy(), x_fake.numpy())),
    }

    # Probe: holdout 20% of LULL pairs
    rng = np.random.default_rng(args.seed)
    n_pairs = xs_r.shape[0]
    X = np.concatenate([xs_r, xs_f], 0)
    y = np.concatenate([np.zeros(n_pairs), np.ones(n_pairs)]).astype(np.float32)
    # Keep pairs aligned in shuffle: permute pair indices
    order = rng.permutation(n_pairs)
    n_tr = int(0.8 * n_pairs)
    tr_p, te_p = order[:n_tr], order[n_tr:]
    Xtr = np.concatenate([xs_r[tr_p], xs_f[tr_p]], 0)
    ytr = np.concatenate([np.zeros(len(tr_p)), np.ones(len(tr_p))]).astype(np.float32)
    Xte = np.concatenate([xs_r[te_p], xs_f[te_p]], 0)
    yte = np.concatenate([np.zeros(len(te_p)), np.ones(len(te_p))]).astype(np.float32)
    # shuffle train
    sh = rng.permutation(len(ytr))
    Xtr, ytr = Xtr[sh], ytr[sh]
    print(f"[audit] probe train/test examples={len(ytr)}/{len(yte)}", flush=True)
    probe = _train_probe(Xtr, ytr, Xte, yte, L=8, epochs=30)

    # Pre-snap / post-snap probes (no bin-center) for contrast — same keys
    def slice_stack(src, keys, L=8):
        return np.stack([src[w, vv, o : o + L] for w, o, vv in keys]).astype(np.float32)

    for name, src_r, src_f, apply_bc in (
        ("presnap", gt_pre, mmpd_pre, False),
        ("postsnap", gt, mmpd, False),
        ("postsnap_binc", gt, mmpd, True),
    ):
        rr = slice_stack(src_r, [(w, o, vv) for w, o, vv in lull_keys])
        ff = slice_stack(src_f, [(w, o, vv) for w, o, vv in lull_keys])
        if apply_bc:
            rr2, ff2 = [], []
            for i, (w, o, vv) in enumerate(lull_keys):
                lv = legal[w, vv]
                rr2.append(_bin_center_1d(rr[i], lv))
                ff2.append(_bin_center_1d(ff[i], lv))
            rr, ff = np.stack(rr2), np.stack(ff2)
        Xtr = np.concatenate([rr[tr_p], ff[tr_p]], 0)[sh]
        # rebuild ytr same shuffle
        ytr_base = np.concatenate([np.zeros(len(tr_p)), np.ones(len(tr_p))]).astype(np.float32)
        Xtr = np.concatenate([rr[tr_p], ff[tr_p]], 0)
        ytr = ytr_base
        Xtr, ytr = Xtr[sh], ytr[sh]
        Xte = np.concatenate([rr[te_p], ff[te_p]], 0)
        yte = np.concatenate([np.zeros(len(te_p)), np.ones(len(te_p))]).astype(np.float32)
        probe[name] = _train_probe(Xtr, ytr, Xte, yte, L=8, epochs=30)
        probe[f"{name}_pair_mse"] = float(np.mean((rr - ff) ** 2))
        probe[f"{name}_frac_identical"] = _frac_eq(rr, ff)

    # Feature AUROCs post-binc
    mean_r, mean_f = xs_r.mean(1), xs_f.mean(1)
    std_r, std_f = xs_r.std(1), xs_f.std(1)
    labels = np.concatenate([np.zeros(n_pairs), np.ones(n_pairs)])
    feat = {
        "auroc_mean": binary_auroc(labels, np.concatenate([mean_r, mean_f])),
        "auroc_std": binary_auroc(labels, np.concatenate([std_r, std_f])),
        "auroc_diff_energy": binary_auroc(
            labels,
            np.concatenate(
                [
                    np.mean(np.diff(xs_r, axis=1) ** 2, axis=1),
                    np.mean(np.diff(xs_f, axis=1) ** 2, axis=1),
                ]
            ),
        ),
    }

    # Positive control: fake = real + 0.5 without bin-center should be easy;
    # with bin-center should collapse.
    ctrl_r = xs_r  # already binc'd
    ctrl_f_bias = xs_r + 0.5  # bias after binc — still separable if model sees mean
    # Better: apply bias before binc
    rr = slice_stack(gt, lull_keys)
    ff_bias = rr + 0.5
    rr_bc, ff_bc, ff_nobc = [], [], []
    for i, (w, o, vv) in enumerate(lull_keys):
        lv = legal[w, vv]
        rr_bc.append(_bin_center_1d(rr[i], lv))
        ff_bc.append(_bin_center_1d(ff_bias[i], lv))
        ff_nobc.append(ff_bias[i])
    rr_bc = np.stack(rr_bc)
    ff_bc = np.stack(ff_bc)
    ff_nobc = np.stack(ff_nobc)
    Xtr = np.concatenate([rr_bc[tr_p], ff_bc[tr_p]], 0)
    ytr = np.concatenate([np.zeros(len(tr_p)), np.ones(len(tr_p))]).astype(np.float32)
    Xtr, ytr = Xtr[sh], ytr[sh]
    Xte = np.concatenate([rr_bc[te_p], ff_bc[te_p]], 0)
    yte = np.concatenate([np.zeros(len(te_p)), np.ones(len(te_p))]).astype(np.float32)
    ctrl_binc = _train_probe(Xtr, ytr, Xte, yte, L=8, epochs=20)
    Xtr = np.concatenate([rr[tr_p], ff_nobc[tr_p]], 0)
    ytr = np.concatenate([np.zeros(len(tr_p)), np.ones(len(tr_p))]).astype(np.float32)
    Xtr, ytr = Xtr[sh], ytr[sh]
    Xte = np.concatenate([rr[te_p], ff_nobc[te_p]], 0)
    yte = np.concatenate([np.zeros(len(te_p)), np.ones(len(te_p))]).astype(np.float32)
    ctrl_nobinc = _train_probe(Xtr, ytr, Xte, yte, L=8, epochs=20)

    out = {
        "pack": str(pack),
        "ckpt": str(ckpt),
        "config": args.config,
        "elapsed_s": time.time() - t0,
        "protocol": {
            "snap_mode": snap_mode,
            "canvas_height": canvas_height,
            "snap_meta": snap_meta,
            "use_window_normalization": True,
            "bin_center": True,
            "unique_abs": True,
            "candidate_only": True,
            "L": [8, 16],
            "hybrid_flat": False,
            "disc_trained_on_lull_instance_norm_snapped": snap_mode == "window_norm_grid",
        },
        "viz_windows": stage_rows,
        "viz_agg": viz_agg,
        "lull_pack_distances": lull_pack,
        "lull_unique_abs_test_L8_binc_mmpd": {
            "n_pairs": n_pairs,
            "pair_mse": pair_mse,
            "pair_mae": pair_mae,
            "pair_mean_l2": pair_l2,
            "frac_identical": identical / max(n_pairs, 1),
            "feature_aurocs": feat,
        },
        "getitem_vs_viz_pool1169_L8_off44": getitem_vs_viz,
        "probe_lull_only": probe,
        "controls": {
            "bias_plus_0p5_then_binc": ctrl_binc,
            "bias_plus_0p5_no_binc": ctrl_nobinc,
        },
        "label_convention": "0=real/GT, 1=fake",
        "split": {"n_train": len(tr_idx), "n_val": len(va_idx), "n_test": len(te_idx)},
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "snap_mode": snap_mode,
        "disc_on_wn_grid": snap_mode == "window_norm_grid",
        "viz_agg": viz_agg,
        "probe": {k: probe[k] for k in probe if isinstance(probe[k], dict) or k.endswith("mse")},
        "getitem_match": {
            "real": getitem_vs_viz["max_abs_real_vs_viz_gt"],
            "fake": getitem_vs_viz["max_abs_fake_vs_viz_mmpd"],
            "identical": getitem_vs_viz["real_identical_fake"],
        },
        "pair_mse_binc": pair_mse,
        "frac_identical": identical / max(n_pairs, 1),
        "ctrl": {"binc": ctrl_binc, "nobinc": ctrl_nobinc},
        "out": str(args.out_json),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
