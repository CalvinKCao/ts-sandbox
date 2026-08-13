#!/usr/bin/env python3
"""DTW alignment viz: binary canvas128 vs MMPD anchors, L∈{8,16}.

For each table dataset: pick ~5 random (window, variate, start) slices, plot GT/pred
with lines along the DTW warp path, and report mean demeaned L1 DTW (Sakoe-Chiba r=3).

Packs taken from temp/lean_disc_c128_results/variogram_cloud_gap16.json sources.
Binary pred = final_anchor; MMPD pred = deterministic (gaussian-anchor).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
VARIOGRAM_INDEX = REPO / "temp/lean_disc_c128_results/variogram_cloud_gap16.json"
REPORT_STEM = "dtw_binary_mmpd_l8_l16"
OUT_DIR = REPO / "reports" / REPORT_STEM
REPORT_MD = REPO / "reports" / f"{REPORT_STEM}.md"
SEED = 20260812
DEFAULT_N_VIZ = 5
LENGTHS = (8, 16)
RADIUS = 3
AGG_MAX_SLICES = 20_000


def _load_index() -> list[dict[str, Any]]:
    return json.loads(VARIOGRAM_INDEX.read_text())["rows"]


def _load_pair(path: Path, kind: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as pack:
        y_true = np.asarray(pack["y_true"], dtype=np.float64)
        if kind == "binary":
            if "final_anchor" in pack.files:
                pred = np.asarray(pack["final_anchor"], dtype=np.float64)
            else:
                pred = np.asarray(pack["deterministic"], dtype=np.float64)
        else:
            pred = np.asarray(pack["deterministic"], dtype=np.float64)
    if y_true.shape != pred.shape:
        raise ValueError(f"{path}: y_true {y_true.shape} vs pred {pred.shape}")
    return y_true, pred


def dtw_l1_band(
    target: np.ndarray,
    pred: np.ndarray,
    radius: int = RADIUS,
) -> tuple[float, list[tuple[int, int]]]:
    """Sakoe-Chiba banded L1 DTW; returns (total_cost, warp_path as (i,j) 0-based)."""
    n = int(len(target))
    m = int(len(pred))
    assert n == m
    inf = 1e30
    cost = np.full((n + 1, m + 1), inf, dtype=np.float64)
    back = np.full((n + 1, m + 1, 2), -1, dtype=np.int16)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        j0 = max(1, i - radius)
        j1 = min(m, i + radius)
        ti = float(target[i - 1])
        for j in range(j0, j1 + 1):
            local = abs(ti - float(pred[j - 1]))
            cands = (
                (cost[i - 1, j - 1], i - 1, j - 1),
                (cost[i - 1, j], i - 1, j),
                (cost[i, j - 1], i, j - 1),
            )
            best = min(cands, key=lambda t: t[0])
            cost[i, j] = local + best[0]
            back[i, j] = (best[1], best[2])
    if not np.isfinite(cost[n, m]):
        # widen band if needed (short series edge)
        return dtw_l1_band(target, pred, radius=max(radius, n))
    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            path.append((i - 1, j - 1))
        pi, pj = int(back[i, j, 0]), int(back[i, j, 1])
        if pi < 0:
            break
        i, j = pi, pj
    path.reverse()
    return float(cost[n, m]), path


def batched_dtw_l1_band3(target: np.ndarray, pred: np.ndarray, radius: int = RADIUS) -> np.ndarray:
    n_paths, horizon = target.shape
    prev = np.full((n_paths, horizon + 1), np.inf, dtype=np.float64)
    prev[:, 0] = 0.0
    for i in range(1, horizon + 1):
        curr = np.full_like(prev, np.inf)
        for j in range(max(1, i - radius), min(horizon, i + radius) + 1):
            curr[:, j] = np.abs(target[:, i - 1] - pred[:, j - 1]) + np.minimum(
                prev[:, j - 1], np.minimum(prev[:, j], curr[:, j - 1]),
            )
        prev = curr
    return prev[:, -1]


def mean_dtw_over_slices(
    y_true: np.ndarray,
    pred: np.ndarray,
    length: int,
    rng: np.random.Generator,
    max_slices: int = AGG_MAX_SLICES,
) -> dict[str, float]:
    """Mean demeaned banded L1 DTW over random sliding subwindows."""
    n_w, n_v, h = y_true.shape
    n_starts = h - length + 1
    total_possible = n_w * n_v * n_starts
    flat_idx = rng.choice(total_possible, size=min(max_slices, total_possible), replace=False)
    targets = np.empty((len(flat_idx), length), dtype=np.float64)
    forecasts = np.empty_like(targets)
    for k, idx in enumerate(flat_idx):
        w = int(idx // (n_v * n_starts))
        rem = int(idx % (n_v * n_starts))
        v = rem // n_starts
        s = rem % n_starts
        t = y_true[w, v, s : s + length]
        p = pred[w, v, s : s + length]
        targets[k] = t - t.mean()
        forecasts[k] = p - p.mean()
    costs = []
    for start in range(0, len(targets), 4096):
        stop = min(start + 4096, len(targets))
        costs.append(batched_dtw_l1_band3(targets[start:stop], forecasts[start:stop]))
    vals = np.concatenate(costs)
    return {
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "p90": float(np.quantile(vals, 0.9)),
        "n_slices": int(len(vals)),
        "n_possible": int(total_possible),
    }


def pick_viz_slices(
    y_true: np.ndarray,
    length: int,
    n: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    n_w, n_v, h = y_true.shape
    n_starts = h - length + 1
    picks = set()
    out: list[tuple[int, int, int]] = []
    while len(out) < n:
        w = int(rng.integers(n_w))
        v = int(rng.integers(n_v))
        s = int(rng.integers(n_starts))
        key = (w, v, s)
        if key in picks:
            continue
        picks.add(key)
        out.append(key)
    return out


def plot_dataset_length(
    dataset: str,
    length: int,
    binary: tuple[np.ndarray, np.ndarray],
    mmpd: tuple[np.ndarray, np.ndarray],
    out_path: Path,
    seed: int,
    n_viz: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, length, sum(map(ord, dataset))]))
    fig, axes = plt.subplots(n_viz, 2, figsize=(11, 2.2 * n_viz), squeeze=False)
    records: list[dict[str, Any]] = []
    for col, (kind, pair) in enumerate((("binary", binary), ("mmpd", mmpd))):
        y_true, pred = pair
        picks = pick_viz_slices(y_true, length, n_viz, rng)
        for row_i, (w, v, s) in enumerate(picks):
            ax = axes[row_i, col]
            gt = y_true[w, v, s : s + length].astype(np.float64)
            pr = pred[w, v, s : s + length].astype(np.float64)
            gt_d = gt - gt.mean()
            pr_d = pr - pr.mean()
            cost, path = dtw_l1_band(gt_d, pr_d, radius=RADIUS)
            t = np.arange(length)
            ax.plot(t, gt_d, color="#1f77b4", lw=1.6, label="GT", marker="o", ms=3)
            ax.plot(t, pr_d, color="#d62728", lw=1.6, label="pred", marker="o", ms=3)
            for i, j in path:
                ax.plot([i, j], [gt_d[i], pr_d[j]], color="0.55", lw=0.7, alpha=0.75, zorder=0)
            ax.set_title(
                f"{kind}  w={w} v={v} t0={s}  DTW={cost:.3f}  |path|={len(path)}",
                fontsize=9,
            )
            ax.grid(True, alpha=0.25)
            if row_i == 0:
                ax.legend(loc="upper right", fontsize=8)
            if row_i == n_viz - 1:
                ax.set_xlabel("t within slice")
            records.append({
                "dataset": dataset,
                "length": length,
                "kind": kind,
                "window": w,
                "variate": v,
                "start": s,
                "dtw_l1_demeaned_band3": cost,
                "path_len": len(path),
            })
    fig.suptitle(
        f"{dataset}  L={length}  demeaned L1 DTW (Sakoe-Chiba r={RADIUS})  "
        f"GT↔pred warp lines",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return records


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", type=str, default="", help="comma list; default all")
    p.add_argument("--n-viz", type=int, default=DEFAULT_N_VIZ)
    args = p.parse_args()
    n_viz = int(args.n_viz)

    rows = _load_index()
    if args.datasets.strip():
        want = {x.strip() for x in args.datasets.split(",") if x.strip()}
        rows = [r for r in rows if r["dataset"] in want]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "seed": SEED,
        "n_viz": n_viz,
        "lengths": list(LENGTHS),
        "radius": RADIUS,
        "metric": (
            "demeaned L1 DTW with Sakoe-Chiba band radius=3; "
            "aggregate = mean over up to 20k random sliding subwindows; "
            "binary pred=final_anchor, mmpd pred=deterministic"
        ),
        "datasets": [],
        "viz_records": [],
    }

    md = [
        "# DTW binary vs MMPD (L=8 / L=16)",
        "",
        summary["metric"],
        "",
        f"Seed `{SEED}`, ~{n_viz} random slices per panel. "
        "Figures: GT (blue) / pred (red) with gray warp-path connectors.",
        "",
        "| dataset | binary DTW L8 | MMPD DTW L8 | binary DTW L16 | MMPD DTW L16 |",
        "|---|---:|---:|---:|---:|",
    ]

    for row in rows:
        ds = row["dataset"]
        print(f"[ds] {ds}", flush=True)
        bin_path = REPO / row["binary"]["source"]
        mmpd_path = REPO / row["mmpd"]["source"]
        yb, pb = _load_pair(bin_path, "binary")
        ym, pm = _load_pair(mmpd_path, "mmpd")
        ds_entry: dict[str, Any] = {
            "dataset": ds,
            "binary_source": str(bin_path.relative_to(REPO)),
            "mmpd_source": str(mmpd_path.relative_to(REPO)),
            "binary_shape": list(yb.shape),
            "mmpd_shape": list(ym.shape),
            "aggregate": {},
        }
        cell = {"binary": {}, "mmpd": {}}
        for length in LENGTHS:
            for kind, (yt, pr) in (("binary", (yb, pb)), ("mmpd", (ym, pm))):
                rng = np.random.default_rng(
                    np.random.SeedSequence([SEED, length, sum(map(ord, ds + kind))])
                )
                stats = mean_dtw_over_slices(yt, pr, length, rng)
                ds_entry["aggregate"][f"{kind}_L{length}"] = stats
                cell[kind][length] = stats["mean"]
                print(
                    f"  {kind} L{length}: mean={stats['mean']:.4f} "
                    f"n={stats['n_slices']}/{stats['n_possible']}",
                    flush=True,
                )
            fig_name = f"{ds}_L{length}.png"
            recs = plot_dataset_length(
                ds, length, (yb, pb), (ym, pm), OUT_DIR / fig_name, SEED, n_viz,
            )
            summary["viz_records"].extend(recs)
            ds_entry[f"figure_L{length}"] = str((OUT_DIR / fig_name).relative_to(REPO))
        summary["datasets"].append(ds_entry)
        md.append(
            f"| {ds} | {cell['binary'][8]:.4f} | {cell['mmpd'][8]:.4f} | "
            f"{cell['binary'][16]:.4f} | {cell['mmpd'][16]:.4f} |"
        )

    md.extend(["", "## Figures", ""])
    for ds_entry in summary["datasets"]:
        ds = ds_entry["dataset"]
        md.append(f"### {ds}")
        md.append("")
        for length in LENGTHS:
            rel = ds_entry[f"figure_L{length}"]
            md.append(f"![L={length}]({REPORT_STEM}/{Path(rel).name})")
            md.append("")

    REPORT_MD.write_text("\n".join(md) + "\n")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[done] {REPORT_MD}", flush=True)
    print(f"[done] {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
