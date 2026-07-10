#!/usr/bin/env python3
"""CRPS-targeted noise-schedule g calibration report.

Aggregates short ablation runs under results/{ckpts,datasets}/*binary_noise_sched_ablation*,
applies the post-hoc stopping rule, estimates seed noise floor from g=1.0 replicates,
and writes per-dataset tables + CRPS/anchor plots + a cross-dataset recommendation.

Example:
  python utils/analyze_noise_sched_crps_grid.py \\
    --out-dir reports/noise_sched_crps_grid
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
CKPT_ROOT = REPO / "results" / "ckpts"
RES_ROOT = REPO / "results" / "datasets"

# Config stem → g. Seed replicates keep g=1.0.
G_FROM_STEM = {
    "binary_noise_sched_ablation_elec_unc_g1p0": 1.0,
    "binary_noise_sched_ablation_elec_unc_g1p0_s43": 1.0,
    "binary_noise_sched_ablation_elec_unc_g1p0_s44": 1.0,
    "binary_noise_sched_ablation_elec_unc_g1p5": 1.5,
    "binary_noise_sched_ablation_elec_unc_g3p0": 3.0,
    "binary_noise_sched_ablation_elec_unc_g4p0": 4.0,
    "binary_noise_sched_ablation_elec_unc_g5p0": 5.0,
    "binary_noise_sched_ablation_elec_unc_g7p0": 7.0,
    "binary_noise_sched_ablation_elec_unc_g10p0": 10.0,
}
SEED_FROM_STEM = {
    "binary_noise_sched_ablation_elec_unc_g1p0_s43": 43,
    "binary_noise_sched_ablation_elec_unc_g1p0_s44": 44,
}
DATASETS_DEFAULT = ("ETTh1", "traffic", "exchange_rate", "electricity")
RUN_RE = re.compile(
    r"^(\d{2}-\d{2})-(\d+)-(ETTh1|traffic|exchange_rate|electricity)-"
    r"(binary_noise_sched_ablation_elec_unc_.+)$"
)

# Stopping rule (post-hoc on sorted g grid)
CRPS_IMPROVE_FLOOR = 0.02  # relative CRPS improvement between consecutive g
ANCHOR_DEGRADE_CAP = 0.05  # relative anchor MSE vs g=1.0 baseline


@dataclass
class RunRow:
    dataset: str
    g: float
    seed: int
    job: int
    stem: str
    fine_val: Optional[float] = None
    fine_epochs: int = 0
    crps: Optional[float] = None
    anchor_mse: Optional[float] = None
    is_smoke: bool = False
    notes: str = ""


def _parse_runs() -> List[Tuple[str, str, int, Path]]:
    out = []
    if not CKPT_ROOT.is_dir():
        return out
    for p in CKPT_ROOT.iterdir():
        m = RUN_RE.match(p.name)
        if not m:
            continue
        ds, stem, jid = m.group(3), m.group(4), int(m.group(2))
        if stem not in G_FROM_STEM:
            continue
        out.append((ds, stem, jid, p))
    return out


def _subset_dir(run_dir: Path) -> Optional[Path]:
    kids = [c for c in run_dir.iterdir() if c.is_dir()]
    for c in kids:
        if (c / "fine").is_dir() or (c / "coarse").is_dir():
            return c
    return kids[0] if kids else None


def _load_fine(run_dir: Path) -> Tuple[Optional[float], int, Optional[int], Optional[float]]:
    subset = _subset_dir(run_dir)
    if subset is None:
        return None, 0, None, None
    hp = subset / "fine" / "val_loss_history.json"
    if not hp.is_file():
        meta = subset / "fine" / "metadata.json"
        if meta.is_file():
            d = json.loads(meta.read_text())
            return float(d.get("best_val_loss", float("nan"))), int(d.get("max_epochs") or 0), None, None
        return None, 0, None, None
    d = json.loads(hp.read_text())
    epochs = d.get("epochs") or []
    seed = d.get("seed")
    g = d.get("length_g")
    return (
        float(d.get("best_val", float("nan"))),
        len(epochs),
        int(seed) if seed is not None else None,
        float(g) if g is not None else None,
    )


def _load_eval(stem: str, dataset: str) -> Optional[Dict[str, Any]]:
    partial = RES_ROOT / stem / "partials" / f"{dataset}_staged_anchor.json"
    if partial.is_file():
        return json.loads(partial.read_text())
    # nested staged_results
    for p in (RES_ROOT / stem).glob("*/staged_results.json"):
        d = json.loads(p.read_text())
        m = (d.get("eval_metrics") or {}).get("staged_anchor") or d
        if "crps" in m:
            out = dict(m)
            if "seed" in d:
                out.setdefault("seed", d["seed"])
            if "binary_length_g" in d:
                out.setdefault("binary_length_g", d["binary_length_g"])
            return out
    return None


def collect_rows(min_epochs: int = 4) -> List[RunRow]:
    # Prefer newest job per (dataset, stem) — stem encodes g and seed-replicate identity.
    best: Dict[Tuple[str, str], Tuple[int, Path]] = {}
    for ds, stem, jid, path in _parse_runs():
        k = (ds, stem)
        if k not in best or jid > best[k][0]:
            best[k] = (jid, path)

    rows: List[RunRow] = []
    for (ds, stem), (jid, path) in sorted(best.items()):
        fine_val, n_ep, hist_seed, hist_g = _load_fine(path)
        ev = _load_eval(path.name, ds)
        g = float(hist_g) if hist_g is not None else G_FROM_STEM[stem]
        seed = SEED_FROM_STEM.get(stem)
        if seed is None and hist_seed is not None:
            seed = hist_seed
        if seed is None and ev and ev.get("seed") is not None:
            seed = int(ev["seed"])
        if seed is None:
            seed = 42  # default for original g1p0/g1p5/g3p0 runs
        smoke = n_ep > 0 and n_ep < min_epochs
        if smoke:
            continue  # drop 1-epoch smokes from the report
        if fine_val is None and ev is None:
            continue
        rows.append(
            RunRow(
                dataset=ds,
                g=g,
                seed=seed,
                job=jid,
                stem=stem,
                fine_val=fine_val,
                fine_epochs=n_ep,
                crps=float(ev["crps"]) if ev and ev.get("crps") is not None else None,
                anchor_mse=float(ev["anchor_mse"]) if ev and ev.get("anchor_mse") is not None else None,
                is_smoke=False,
            )
        )
    return rows


def _rel(a: float, b: float) -> float:
    if a == 0 or math.isnan(a) or math.isnan(b):
        return float("nan")
    return (b - a) / abs(a)


def seed_noise_floor(rows: List[RunRow], dataset: str) -> Tuple[Optional[float], List[float]]:
    """Half-range (max-min)/2 of g=1.0 CRPS across seeds; also return raw CRPS list."""
    vals = [
        r.crps
        for r in rows
        if r.dataset == dataset and abs(r.g - 1.0) < 1e-9 and r.crps is not None
    ]
    if len(vals) < 2:
        return None, vals
    return 0.5 * (max(vals) - min(vals)), vals


def apply_stopping_rule(
    points: List[RunRow],
    baseline_anchor: float,
) -> Tuple[Optional[float], str]:
    """Return (stop_after_g, reason) scanning ascending g. None = no stop (hit ceiling)."""
    pts = sorted([p for p in points if p.crps is not None], key=lambda r: r.g)
    # Prefer seed=42 for the main curve when duplicates exist
    by_g: Dict[float, RunRow] = {}
    for p in pts:
        if p.g not in by_g or p.seed == 42:
            by_g[p.g] = p
    ordered = [by_g[g] for g in sorted(by_g)]
    for i in range(1, len(ordered)):
        prev, cur = ordered[i - 1], ordered[i]
        crps_improve = -_rel(prev.crps, cur.crps)  # positive = better
        anchor_deg = _rel(baseline_anchor, cur.anchor_mse) if cur.anchor_mse is not None else 0.0
        if crps_improve < CRPS_IMPROVE_FLOOR and anchor_deg > ANCHOR_DEGRADE_CAP:
            return cur.g, (
                f"stop at g={cur.g}: CRPS improve vs prev={crps_improve:.1%} < {CRPS_IMPROVE_FLOOR:.0%} "
                f"AND anchor degrade vs g=1={anchor_deg:.1%} > {ANCHOR_DEGRADE_CAP:.0%}"
            )
        if crps_improve < CRPS_IMPROVE_FLOOR and i == len(ordered) - 1:
            # plateau at end without necessarily hitting anchor cap
            return cur.g, (
                f"plateau near g={cur.g}: CRPS improve vs prev={crps_improve:.1%} < {CRPS_IMPROVE_FLOOR:.0%}"
            )
    last = ordered[-1].g if ordered else None
    return None, f"no stop: reached grid ceiling g={last}"


def recommend_g(
    points: List[RunRow],
    baseline: RunRow,
    noise: Optional[float],
) -> Tuple[float, str]:
    """Best CRPS among points not clearly within noise of a cheaper g; soft-penalize anchor."""
    by_g: Dict[float, RunRow] = {}
    for p in points:
        if p.crps is None:
            continue
        if p.g not in by_g or p.seed == 42:
            by_g[p.g] = p
    if not by_g:
        return 1.0, "no CRPS data"
    # raw best CRPS
    best = min(by_g.values(), key=lambda r: r.crps)
    # if anchor degrades >10% vs baseline, prefer next-best with milder anchor cost
    candidates = sorted(by_g.values(), key=lambda r: r.crps)
    chosen = best
    for c in candidates:
        if c.anchor_mse is None or baseline.anchor_mse is None:
            chosen = c
            break
        deg = _rel(baseline.anchor_mse, c.anchor_mse)
        if deg <= 0.10:
            chosen = c
            break
    else:
        chosen = candidates[0]
    note = f"raw best CRPS g={best.g}"
    if chosen.g != best.g:
        note += f"; recommend g={chosen.g} (anchor soft-cap ≤10% degrade)"
    else:
        note += f"; recommend g={chosen.g}"
    if noise is not None and baseline.crps is not None:
        if abs(chosen.crps - baseline.crps) < noise:
            note += f"; ΔCRPS within seed noise (±{noise:.4f})"
    return chosen.g, note


def plot_dataset(
    dataset: str,
    points: List[RunRow],
    noise: Optional[float],
    baseline_crps: Optional[float],
    out_dir: Path,
) -> None:
    by_g: Dict[float, RunRow] = {}
    for p in points:
        if p.crps is None:
            continue
        if p.g not in by_g or p.seed == 42:
            by_g[p.g] = p
    if not by_g:
        return
    gs = sorted(by_g)
    crps = [by_g[g].crps for g in gs]
    anchors = [by_g[g].anchor_mse for g in gs]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(gs, crps, marker="o", color="#1f4e79", label="CRPS")
    if noise is not None and baseline_crps is not None:
        ax.axhspan(
            baseline_crps - noise,
            baseline_crps + noise,
            color="#1f4e79",
            alpha=0.15,
            label=f"seed noise floor (±{noise:.3f} around g=1 CRPS)",
        )
    ax.set_xlabel("g (power length shift)")
    ax.set_ylabel("CRPS (↓ better)")
    ax.set_title(f"{dataset}: CRPS vs g")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plot_dir = out_dir / "comparison_summary"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / f"crps_vs_g_{dataset}.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(gs, anchors, marker="s", color="#8b4513", label="anchor MSE")
    if baseline_crps is not None and by_g.get(1.0) and by_g[1.0].anchor_mse is not None:
        base_a = by_g[1.0].anchor_mse
        ax.axhline(base_a * 1.05, color="#8b4513", ls="--", alpha=0.5, label="+5% vs g=1.0")
    ax.set_xlabel("g (power length shift)")
    ax.set_ylabel("anchor MSE (↓ better)")
    ax.set_title(f"{dataset}: anchor MSE vs g")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / f"anchor_mse_vs_g_{dataset}.png", dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="reports/noise_sched_crps_grid")
    ap.add_argument("--min-epochs", type=int, default=4)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS_DEFAULT))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison_summary").mkdir(parents=True, exist_ok=True)

    rows = collect_rows(min_epochs=args.min_epochs)
    if not rows:
        print("No completed (≥4 epoch) ablation runs found.")
        return

    # Per-dataset tables
    summary_rows = []
    md: List[str] = [
        "# CRPS-targeted noise schedule g calibration\n\n",
        "Fine val/BCE is reported for completeness only — **not comparable across g** "
        "(different forward process).\n\n",
        f"Stopping rule (post-hoc): consecutive CRPS improve < {CRPS_IMPROVE_FLOOR:.0%} "
        f"AND anchor MSE degrade vs g=1.0 > {ANCHOR_DEGRADE_CAP:.0%}.\n\n",
    ]

    curve_shapes: Dict[str, List[Tuple[float, float]]] = {}

    for ds in args.datasets:
        ds_rows = [r for r in rows if r.dataset == ds]
        if not ds_rows:
            md.append(f"## {ds}\n\n_No runs yet._\n\n")
            continue
        noise, seed_crps = seed_noise_floor(rows, ds)
        # main curve: prefer seed 42 per g
        by_g: Dict[float, RunRow] = {}
        for r in ds_rows:
            if r.crps is None:
                continue
            if r.g not in by_g or r.seed == 42:
                by_g[r.g] = r
        if 1.0 not in by_g:
            md.append(f"## {ds}\n\n_Missing g=1.0 baseline._\n\n")
            continue
        base = by_g[1.0]
        stop_g, stop_reason = apply_stopping_rule(list(by_g.values()), base.anchor_mse or float("nan"))
        rec_g, rec_note = recommend_g(list(by_g.values()), base, noise)

        md.append(f"## {ds}\n\n")
        if noise is not None:
            md.append(
                f"Seed noise floor (half-range of g=1.0 CRPS across seeds "
                f"{seed_crps}): **±{noise:.4f}**\n\n"
            )
        else:
            md.append(
                f"Seed noise floor: _insufficient replicates_ "
                f"(have {len(seed_crps)} g=1.0 CRPS value(s); need ≥2).\n\n"
            )
        md.append(f"Stopping: {stop_reason}\n\n")
        md.append(
            "| g | seed | fine val* | CRPS | ΔCRPS vs g=1 | anchor MSE | Δanchor | within seed noise? | notes |\n"
            "|---:|---:|---:|---:|---:|---:|---:|---|---|\n"
        )
        table_csv = []
        for g in sorted(by_g):
            r = by_g[g]
            dcrps = _rel(base.crps, r.crps)
            danc = _rel(base.anchor_mse, r.anchor_mse) if r.anchor_mse is not None else float("nan")
            within = ""
            if noise is not None and r.crps is not None and base.crps is not None:
                within = "yes" if abs(r.crps - base.crps) < noise and g != 1.0 else "no"
            note = ""
            if stop_g is not None and g == stop_g:
                note = "stop rule"
            if abs(g - rec_g) < 1e-9:
                note = (note + "; " if note else "") + "recommended"
            md.append(
                f"| {g:g} | {r.seed} | {r.fine_val:.4f} | {r.crps:.4f} | {dcrps:+.1%} | "
                f"{r.anchor_mse:.4f} | {danc:+.1%} | {within} | {note} |\n"
            )
            table_csv.append({
                "dataset": ds,
                "g": g,
                "seed": r.seed,
                "fine_val_not_comparable": r.fine_val,
                "crps": r.crps,
                "delta_crps_vs_g1": dcrps,
                "anchor_mse": r.anchor_mse,
                "delta_anchor_vs_g1": danc,
                "within_seed_noise": within,
                "notes": note,
                "job": r.job,
            })
        md.append("\n\\*fine val not comparable across schedules\n\n")
        md.append(f"Recommendation: {rec_note}\n\n")

        csv_path = out_dir / f"table_{ds}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(table_csv[0].keys()))
            w.writeheader()
            w.writerows(table_csv)

        plot_dataset(ds, ds_rows, noise, base.crps, out_dir)
        curve_shapes[ds] = [(g, by_g[g].crps) for g in sorted(by_g)]

        best_g = min(by_g.values(), key=lambda r: r.crps).g
        best = by_g[best_g]
        hit_ceiling = max(by_g) >= 10.0 - 1e-9 and (
            stop_g is None or (best_g >= 10.0 - 1e-9)
        )
        # plateauing: last consecutive improve < 2%
        ordered = [by_g[g] for g in sorted(by_g)]
        plateau = False
        if len(ordered) >= 2:
            plateau = -_rel(ordered[-2].crps, ordered[-1].crps) < CRPS_IMPROVE_FLOOR
        summary_rows.append({
            "dataset": ds,
            "best_crps_g": best_g,
            "crps_improvement": _rel(base.crps, best.crps),
            "hit_g10_without_plateau": bool(hit_ceiling and not plateau),
            "anchor_cost_at_best": _rel(base.anchor_mse, best.anchor_mse),
            "recommended_g": rec_g,
            "stop_reason": stop_reason,
            "seed_noise": noise,
        })

    # Cross-dataset summary
    md.append("## Cross-dataset summary\n\n")
    md.append(
        "| dataset | best-CRPS g | CRPS Δ at best | hit g=10 w/o plateau? | "
        "anchor cost at best | recommended g |\n"
        "|---|---:|---:|---|---:|---:|\n"
    )
    for s in summary_rows:
        md.append(
            f"| {s['dataset']} | {s['best_crps_g']:g} | {s['crps_improvement']:+.1%} | "
            f"{'yes' if s['hit_g10_without_plateau'] else 'no'} | "
            f"{s['anchor_cost_at_best']:+.1%} | {s['recommended_g']:g} |\n"
        )

    # Shape comparison
    md.append("\n## Curve-shape comparison (5e)\n\n")
    shapes_mono_improve = []
    shapes_nonmono = []
    for ds, pts in curve_shapes.items():
        crps_vals = [c for _, c in pts]
        # strictly improving (decreasing CRPS) until last, or has a clear upturn
        upturn = False
        for i in range(1, len(crps_vals)):
            if crps_vals[i] > crps_vals[i - 1] * 1.02:  # >2% worse
                upturn = True
                break
        if upturn:
            shapes_nonmono.append(ds)
        else:
            shapes_mono_improve.append(ds)
    md.append(
        f"- Monotone-improving / flat CRPS vs g: {', '.join(shapes_mono_improve) or 'none'}\n"
        f"- Non-monotone (CRPS worsens >2% at some step): {', '.join(shapes_nonmono) or 'none'}\n\n"
    )

    # Final recommendation
    md.append("## Final recommendation\n\n")
    if not summary_rows:
        md.append("_Insufficient data._\n")
        rec_text = "insufficient data"
    elif shapes_nonmono and len(shapes_nonmono) >= 2:
        rec_text = (
            "Multi-knot / free-form schedule is justified: ≥2 datasets show "
            "genuinely different non-monotone CRPS-vs-g shapes, so a single "
            "scalar g per dataset may be insufficient."
        )
        md.append(rec_text + "\n")
    elif any(s["hit_g10_without_plateau"] for s in summary_rows):
        rec_text = (
            "Per-dataset scalar g is still the right first tool, but the grid "
            "ceiling (g=10) was hit without plateau on at least one dataset — "
            "extend the scalar search further before investing in multi-knot."
        )
        md.append(rec_text + "\n")
    else:
        rec_text = (
            "Per-dataset scalar g is sufficient: CRPS-vs-g curves share a "
            "similar diminishing-returns shape across datasets (different "
            "magnitudes / optimal g, no conflicting non-monotone patterns that "
            "would require multi-knot schedules)."
        )
        md.append(rec_text + "\n")
        if shapes_nonmono:
            md.append(
                f"Note: mild non-monotone signal on {', '.join(shapes_nonmono)} — "
                "re-check against seed noise before escalating to multi-knot.\n"
            )

    summary_path = out_dir / "comparison_summary.md"
    summary_path.write_text("".join(md), encoding="utf-8")
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {"summary_rows": summary_rows, "recommendation": rec_text, "curve_shapes": curve_shapes},
            f,
            indent=2,
        )
    print(f"Wrote {summary_path}")
    print(f"Recommendation: {rec_text}")
    for s in summary_rows:
        print(
            f"  {s['dataset']}: best_g={s['best_crps_g']} rec_g={s['recommended_g']} "
            f"CRPSΔ={s['crps_improvement']:+.1%} anchorΔ={s['anchor_cost_at_best']:+.1%}"
        )


if __name__ == "__main__":
    main()
