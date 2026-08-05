#!/usr/bin/env python3
"""MLP / logistic probe on disc L-slices — is there learnable signal?"""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from utils.disc_bin_center_shift import center_bin_index
from utils.disc_shared import binary_auroc, split_windows
from utils.eval_discriminator_binary_vs_mmpd_univariate import _unique_absolute_slice_items

torch.set_num_threads(4)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bin_center_many(slices, legal, windows, variates):
    levels = legal[windows, variates, :]
    out = np.empty_like(slices)
    bs = 2048
    for i0 in range(0, len(slices), bs):
        i1 = min(len(slices), i0 + bs)
        sl = slices[i0:i1]
        lv = levels[i0:i1]
        delta = np.abs(sl[:, :, None] - lv[:, None, :])
        raw = np.argmin(delta, axis=-1)
        center = center_bin_index(lv[:, None, :])[:, 0]
        centered = raw - center[:, None]
        shift = np.rint(centered.mean(axis=1, keepdims=True)).astype(np.int64)
        raw_new = np.clip(centered - shift + center[:, None], 0, lv.shape[-1] - 1)
        out[i0:i1] = np.take_along_axis(lv, raw_new, axis=1)
    return out.astype(np.float32)


def make_xy(windows, seed, real, fake, ss, legal, L, V, apply_binc=False, max_pairs=12000):
    t0 = time.time()
    items = _unique_absolute_slice_items(
        windows,
        horizon=96,
        slice_len=L,
        n_var=V,
        offset_stride=1,
        series_starts=ss,
        lookback=336,
        seed=seed,
    )
    keys = [(w, o, v) for (w, o, v, lab) in items if lab == 0]
    print(f"  unique_abs keys={len(keys)} in {time.time() - t0:.1f}s", flush=True)
    if len(keys) > max_pairs:
        rng = np.random.default_rng(seed)
        keys = [keys[i] for i in rng.choice(len(keys), size=max_pairs, replace=False)]
    ws = np.array([w for w, o, v in keys], np.int64)
    vs = np.array([v for w, o, v in keys], np.int64)
    xs_r = np.stack([real[w, v, o : o + L] for w, o, v in keys]).astype(np.float32)
    xs_f = np.stack([fake[w, v, o : o + L] for w, o, v in keys]).astype(np.float32)
    if apply_binc:
        t1 = time.time()
        xs_r = bin_center_many(xs_r, legal, ws, vs)
        xs_f = bin_center_many(xs_f, legal, ws, vs)
        print(f"  binc {time.time() - t1:.1f}s", flush=True)
    X = np.concatenate([xs_r, xs_f], 0)
    y = np.concatenate([np.zeros(len(keys)), np.ones(len(keys))]).astype(np.float32)
    return X, y, np.concatenate([vs, vs])


def train_eval(Xtr, ytr, Xte, yte, name, L, epochs=20):
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    Xtr_n = ((Xtr - mu) / sd).astype(np.float32)
    Xte_n = ((Xte - mu) / sd).astype(np.float32)
    Xtr_t = torch.from_numpy(Xtr_n).to(DEVICE)
    ytr_t = torch.from_numpy(ytr).to(DEVICE)
    Xte_t = torch.from_numpy(Xte_n).to(DEVICE)
    w = torch.zeros(L, device=DEVICE, requires_grad=True)
    b0 = torch.zeros(1, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([w, b0], lr=0.05)
    for _ in range(200):
        opt.zero_grad()
        nn.functional.binary_cross_entropy_with_logits(Xtr_t @ w + b0, ytr_t).backward()
        opt.step()
    with torch.no_grad():
        p = torch.sigmoid(Xte_t @ w + b0).detach().cpu().numpy()
    auc_log = binary_auroc(yte, p)

    model = nn.Sequential(
        nn.Linear(L, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=512, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(model(xb).squeeze(-1), yb).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(Xte_t).squeeze(-1)).detach().cpu().numpy()
    auc_mlp = binary_auroc(yte, p)

    mu0, mu1 = Xtr_n[ytr == 0].mean(0), Xtr_n[ytr == 1].mean(0)
    score = np.linalg.norm(Xte_n - mu0, axis=1) - np.linalg.norm(Xte_n - mu1, axis=1)
    auc_nc = binary_auroc(yte, score)
    print(
        f"{name}: logistic={auc_log:.4f} mlp={auc_mlp:.4f} centroid={auc_nc:.4f} "
        f"ntr={len(ytr)} nte={len(yte)}",
        flush=True,
    )
    return {"logistic": auc_log, "mlp": auc_mlp, "centroid": auc_nc}


def main():
    z = np.load("temp/disc_auroc_audit_etth1_snapped.npz")
    gt, binary, legal, ss = z["gt"], z["binary"], z["legal_levels"], z["series_starts"]
    b = np.load(
        "results/datasets/08-04-1843-ablation-disc-l8-l16-ETTh1-c128-valtest80-byvar/"
        "raw/binary_window_norm_c128_ETTh1_val-test.npz"
    )
    indices = np.asarray(b["indices"])
    gt0 = np.asarray(b["y_true"])
    fb = np.asarray(b["samples"])[:, :, 0, :]
    N, V, H = gt.shape
    L = 8
    args = SimpleNamespace(train_fraction=0.8, val_fraction=0.0, seed=42, max_windows=None)
    splits = split_windows(
        N,
        args,
        "ETTh1",
        indices=indices,
        lookback=336,
        horizon=96,
        test_stride=4,
        series_starts=ss,
    )
    print("splits", {k: len(v) for k, v in splits.items()}, flush=True)
    print(f"device={DEVICE}", flush=True)

    results = {}
    Xtr, ytr, _ = make_xy(splits["train"], 0, gt0, fb, ss, legal, L, V, False, 15000)
    Xte, yte, _ = make_xy(splits["test"], 2, gt0, fb, ss, legal, L, V, False, 15000)
    print("=== PRE-SNAP ===", flush=True)
    results["presnap"] = train_eval(Xtr, ytr, Xte, yte, "presnap", L)

    Xtr, ytr, _ = make_xy(splits["train"], 0, gt, binary, ss, legal, L, V, False, 15000)
    Xte, yte, _ = make_xy(splits["test"], 2, gt, binary, ss, legal, L, V, False, 15000)
    print("=== SNAP no binc ===", flush=True)
    results["snap"] = train_eval(Xtr, ytr, Xte, yte, "snap", L)

    Xtr, ytr, vtr = make_xy(splits["train"], 0, gt, binary, ss, legal, L, V, True, 15000)
    Xte, yte, vte = make_xy(splits["test"], 2, gt, binary, ss, legal, L, V, True, 15000)
    print("=== SNAP+binc ===", flush=True)
    results["binc"] = train_eval(Xtr, ytr, Xte, yte, "binc", L)

    print("=== per-var binc ===", flush=True)
    results["by_var"] = {}
    for v in range(V):
        tr, te = vtr == v, vte == v
        results["by_var"][str(v)] = train_eval(
            Xtr[tr], ytr[tr], Xte[te], yte[te], f"v{v}", L, epochs=30
        )

    # Controls
    print("=== CONTROL bias +0.5 (binc should erase) ===", flush=True)
    fake_easy = gt + 0.5
    Xtr, ytr, _ = make_xy(splits["train"], 0, gt, fake_easy, ss, legal, L, V, True, 8000)
    Xte, yte, _ = make_xy(splits["test"], 2, gt, fake_easy, ss, legal, L, V, True, 8000)
    results["control_bias_binc"] = train_eval(Xtr, ytr, Xte, yte, "control_bias_binc", L)

    Xtr, ytr, _ = make_xy(splits["train"], 0, gt, fake_easy, ss, legal, L, V, False, 8000)
    Xte, yte, _ = make_xy(splits["test"], 2, gt, fake_easy, ss, legal, L, V, False, 8000)
    results["control_bias_nobinc"] = train_eval(Xtr, ytr, Xte, yte, "control_bias_nobinc", L)

    print("=== CONTROL reverse texture + binc ===", flush=True)

    def make_rev(windows, seed, max_pairs=8000):
        items = _unique_absolute_slice_items(
            windows,
            horizon=96,
            slice_len=L,
            n_var=V,
            offset_stride=1,
            series_starts=ss,
            lookback=336,
            seed=seed,
        )
        keys = [(w, o, v) for (w, o, v, lab) in items if lab == 0]
        if len(keys) > max_pairs:
            rng = np.random.default_rng(seed)
            keys = [keys[i] for i in rng.choice(len(keys), max_pairs, replace=False)]
        ws = np.array([w for w, o, v in keys])
        vs = np.array([v for w, o, v in keys])
        xs_r = np.stack([gt[w, v, o : o + L] for w, o, v in keys]).astype(np.float32)
        xs_f = xs_r[:, ::-1].copy()
        xs_r = bin_center_many(xs_r, legal, ws, vs)
        xs_f = bin_center_many(xs_f, legal, ws, vs)
        return (
            np.concatenate([xs_r, xs_f]),
            np.concatenate([np.zeros(len(keys)), np.ones(len(keys))]).astype(np.float32),
        )

    Xtr, ytr = make_rev(splits["train"], 0)
    Xte, yte = make_rev(splits["test"], 2)
    results["control_reverse_binc"] = train_eval(Xtr, ytr, Xte, yte, "control_reverse_binc", L)

    out = Path("temp/disc_auroc_mlp_probe.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
