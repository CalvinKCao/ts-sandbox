#!/usr/bin/env python3
"""Viz for ETTh1 coarse-anchor → fine quad_t ×3 sample_mean probe (job 4651645).

Reads local npz (no model reload). Plots GT vs hybrid sample_mean vs pure
final_anchor, with faint individual fine samples, for all 7 ETTh1 variates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PROBE = REPO / "temp/lean_disc_c128_results/probe_etth1_4651645"
ETTH1_NAMES = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE)
    p.add_argument("--n-worst", type=int, default=4)
    p.add_argument("--n-best", type=int, default=2)
    p.add_argument("--n-mid", type=int, default=2)
    p.add_argument("--dpi", type=int, default=140)
    return p.parse_args()


def _load(probe: Path) -> tuple[dict[str, np.ndarray], dict]:
    npz_path = probe / "raw" / "hybrid_ETTh1.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"missing npz: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as z:
        pack = {k: z[k] for k in z.files}
    required = ("y_true", "sample_mean", "final_anchor", "samples")
    missing = [k for k in required if k not in pack]
    if missing:
        raise RuntimeError(f"npz lacks preds/keys {missing}; keys={list(pack)}")
    metrics = {}
    mpath = probe / "metrics.json"
    if mpath.is_file():
        metrics = json.loads(mpath.read_text(encoding="utf-8"))
    return pack, metrics


def _pick_locals(mse: np.ndarray, n_worst: int, n_best: int, n_mid: int) -> list[tuple[str, int]]:
    order = np.argsort(mse)
    n = int(mse.shape[0])
    picks: list[tuple[str, int]] = []
    seen: set[int] = set()

    def add(tag: str, idxs: list[int]) -> None:
        for i in idxs:
            i = int(i)
            if i in seen:
                continue
            seen.add(i)
            picks.append((tag, i))

    add("worst", list(order[::-1][:n_worst]))
    add("best", list(order[:n_best]))
    if n_mid > 0 and n > 0:
        mid = order[len(order) // 2]
        # neighborhood around median
        mid_pool = [mid]
        for d in range(1, n):
            if len(mid_pool) >= n_mid:
                break
            for j in (mid - d, mid + d):
                if 0 <= j < n and j not in mid_pool:
                    mid_pool.append(j)
                if len(mid_pool) >= n_mid:
                    break
        add("mid", mid_pool[:n_mid])
    return picks


def _plot_window(
    *,
    out_path: Path,
    local_i: int,
    tag: str,
    y_true: np.ndarray,
    sample_mean: np.ndarray,
    final_anchor: np.ndarray,
    samples: np.ndarray,
    win_idx: int,
    series_start: int,
    win_mse_h: float,
    win_mse_a: float,
    dpi: int,
) -> None:
    n_var, h = int(y_true.shape[0]), int(y_true.shape[1])
    names = list(ETTH1_NAMES[:n_var]) + [f"v{i}" for i in range(len(ETTH1_NAMES), n_var)]
    t = np.arange(h)
    fig, axes = plt.subplots(n_var, 1, figsize=(11, 1.55 * n_var + 0.8), sharex=True)
    if n_var == 1:
        axes = [axes]
    for v, ax in enumerate(axes):
        for s in range(samples.shape[1]):
            ax.plot(
                t,
                samples[v, s],
                color="tab:orange",
                alpha=0.22,
                lw=0.9,
                label="fine samples (×3)" if (v == 0 and s == 0) else None,
            )
        ax.plot(t, y_true[v], color="black", lw=1.4, label="GT" if v == 0 else None)
        ax.plot(
            t,
            final_anchor[v],
            color="tab:blue",
            lw=1.15,
            ls="--",
            label="pure anchor" if v == 0 else None,
        )
        ax.plot(
            t,
            sample_mean[v],
            color="tab:red",
            lw=1.25,
            label="hybrid sample_mean" if v == 0 else None,
        )
        ax.set_ylabel(names[v], fontsize=9)
        ax.grid(True, alpha=0.25)
        mse_v = float(np.mean((sample_mean[v] - y_true[v]) ** 2))
        ax.set_title(f"{names[v]}  hybrid MSE={mse_v:.3f}", fontsize=8, loc="right", pad=2)
    axes[0].legend(loc="upper right", fontsize=8, ncol=4, framealpha=0.9)
    axes[-1].set_xlabel("horizon step (H=96)")
    fig.suptitle(
        f"ETTh1 probe 4651645 · {tag} · local={local_i} win={win_idx} "
        f"start={series_start} · win MSE hybrid={win_mse_h:.3f} anchor={win_mse_a:.3f}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_overview(
    *,
    out_path: Path,
    mse_h: np.ndarray,
    mse_a: np.ndarray,
    per_var_h: np.ndarray,
    per_var_a: np.ndarray,
    metrics: dict,
    dpi: int,
) -> None:
    n_var = int(per_var_h.shape[0])
    names = list(ETTH1_NAMES[:n_var]) + [f"v{i}" for i in range(len(ETTH1_NAMES), n_var)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    x = np.arange(len(mse_h))
    axes[0].plot(x, mse_h, color="tab:red", lw=1.3, label="hybrid sample_mean")
    axes[0].plot(x, mse_a, color="tab:blue", lw=1.1, ls="--", label="pure anchor")
    axes[0].set_xlabel("local window index")
    axes[0].set_ylabel("window MSE (all vars)")
    axes[0].set_title("Per-window MSE")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)

    xpos = np.arange(n_var)
    w = 0.38
    axes[1].bar(xpos - w / 2, per_var_h, w, color="tab:red", label="hybrid")
    axes[1].bar(xpos + w / 2, per_var_a, w, color="tab:blue", label="anchor")
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels(names, rotation=30, ha="right")
    axes[1].set_ylabel("MSE")
    axes[1].set_title("Per-variate MSE")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.25)

    m = metrics.get("metrics", metrics)
    fig.suptitle(
        "ETTh1 coarse-anchor → fine quad_t×3 · "
        f"hybrid MSE={m.get('hybrid_sample_mean_mse', m.get('mse', float('nan'))):.4f}  "
        f"anchor MSE={m.get('anchor_mse', float('nan')):.4f}  "
        f"CRPS={m.get('crps', float('nan')):.4f}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    probe = args.probe_dir.expanduser().resolve()
    pack, metrics = _load(probe)
    yt = pack["y_true"]
    sm = pack["sample_mean"]
    fa = pack["final_anchor"]
    samples = pack["samples"]
    win_idx = pack.get("window_indices", np.arange(yt.shape[0]))
    starts = pack.get("series_starts", np.zeros(yt.shape[0], dtype=np.int64))

    if yt.ndim != 3 or sm.shape != yt.shape or fa.shape != yt.shape:
        raise RuntimeError(f"bad shapes yt={yt.shape} sm={sm.shape} fa={fa.shape}")
    if samples.ndim != 4 or samples.shape[0] != yt.shape[0] or samples.shape[1] != yt.shape[1]:
        raise RuntimeError(f"bad samples shape {samples.shape} vs yt {yt.shape}")

    mse_h = ((sm - yt) ** 2).mean(axis=(1, 2))
    mse_a = ((fa - yt) ** 2).mean(axis=(1, 2))
    per_var_h = ((sm - yt) ** 2).mean(axis=(0, 2))
    per_var_a = ((fa - yt) ** 2).mean(axis=(0, 2))

    out = probe / "viz"
    out.mkdir(parents=True, exist_ok=True)
    _plot_overview(
        out_path=out / "overview_mse.png",
        mse_h=mse_h,
        mse_a=mse_a,
        per_var_h=per_var_h,
        per_var_a=per_var_a,
        metrics=metrics,
        dpi=args.dpi,
    )

    picks = _pick_locals(mse_h, args.n_worst, args.n_best, args.n_mid)
    written: list[str] = []
    for tag, li in picks:
        path = out / f"{tag}_local{li:03d}_win{int(win_idx[li])}_allvars.png"
        _plot_window(
            out_path=path,
            local_i=li,
            tag=tag,
            y_true=yt[li],
            sample_mean=sm[li],
            final_anchor=fa[li],
            samples=samples[li],
            win_idx=int(win_idx[li]),
            series_start=int(starts[li]),
            win_mse_h=float(mse_h[li]),
            win_mse_a=float(mse_a[li]),
            dpi=args.dpi,
        )
        written.append(str(path))

    manifest = {
        "probe_dir": str(probe),
        "npz_keys": list(pack.keys()),
        "n_windows": int(yt.shape[0]),
        "n_variates": int(yt.shape[1]),
        "horizon": int(yt.shape[2]),
        "n_samples": int(samples.shape[2]),
        "overview": str(out / "overview_mse.png"),
        "panels": written,
        "picks": [{"tag": t, "local": i, "win": int(win_idx[i]), "mse_hybrid": float(mse_h[i])} for t, i in picks],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(written)} panels + overview → {out}", flush=True)
    for p in [out / "overview_mse.png", *written]:
        print(f"  {p}", flush=True)


if __name__ == "__main__":
    main()
