#!/usr/bin/env python3
"""CRPS-targeted noise-schedule g calibration report.

Aggregates short ablation runs under results/{ckpts,datasets}/*binary_noise_sched_ablation*,
applies the post-hoc stopping rule, estimates seed noise floor from g=1.0 replicates,
compares confirmation seeds at recommended g, and writes per-dataset tables + plots.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
CKPT_ROOT = REPO / "results" / "ckpts"
RES_ROOT = REPO / "results" / "datasets"

# All repo datasets from test_submit.sh except dalia (explicitly excluded).
ALL_DATASETS = (
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "illness",
    "exchange_rate",
    "weather",
    "electricity",
    "traffic",
    "PeMS",
    "solar_Alabama",
)
DATASETS_DEFAULT = ALL_DATASETS

# Stem tag → g (seed replicates keep the parent's g).
_G_TAG = {
    "g1p0": 1.0,
    "g1p5": 1.5,
    "g3p0": 3.0,
    "g4p0": 4.0,
    "g5p0": 5.0,
    "g6p0": 6.0,
    "g7p0": 7.0,
    "g8p0": 8.0,
    "g9p0": 9.0,
    "g10p0": 10.0,
}

# Supports uncompressed elec_unc ablations and past-native stride-2 ablations.
STEM_RE = re.compile(
    r"^binary_noise_sched_ablation_(?:elec_unc|past_native)_(g\d+p\d+)(?:_s(\d+))?$"
)
DS_ALT = "|".join(re.escape(d) for d in ALL_DATASETS)
RUN_RE = re.compile(
    rf"^(\d{{2}}-\d{{2}})-(\d+)-({DS_ALT})-"
    r"(binary_noise_sched_ablation_(?:elec_unc|past_native)_.+)$"
)

# Coarse grid used before fine refinement
COARSE_G = {1.0, 1.5, 3.0, 4.0, 5.0, 7.0, 10.0}
FINE_G = {6.0, 8.0, 9.0}

CRPS_IMPROVE_FLOOR = 0.02
ANCHOR_DEGRADE_CAP = 0.05


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


def _g_seed_from_stem(stem: str) -> Tuple[Optional[float], Optional[int]]:
    m = STEM_RE.match(stem)
    if not m:
        return None, None
    g = _G_TAG.get(m.group(1))
    seed = int(m.group(2)) if m.group(2) else None
    return g, seed


def _parse_runs() -> List[Tuple[str, str, int, Path]]:
    out = []
    if not CKPT_ROOT.is_dir():
        return out
    for p in CKPT_ROOT.iterdir():
        m = RUN_RE.match(p.name)
        if not m:
            continue
        ds, stem, jid = m.group(3), m.group(4), int(m.group(2))
        g, _ = _g_seed_from_stem(stem)
        if g is None:
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


def _load_eval(run_name: str, dataset: str) -> Optional[Dict[str, Any]]:
    # Prefer results keyed by full run stem (MM-DD-jid-ds-cfg)
    stem_only = run_name
    # Also try config-only stem under results/datasets/<cfg>/
    cfg_stem = None
    m = RUN_RE.match(run_name)
    if m:
        cfg_stem = m.group(4)

    candidates = []
    if cfg_stem:
        candidates += [
            RES_ROOT / cfg_stem / "partials" / f"{dataset}_staged_anchor.json",
            RES_ROOT / "datasets" / cfg_stem / "partials" / f"{dataset}_staged_anchor.json",
        ]
    candidates += [
        RES_ROOT / stem_only / "partials" / f"{dataset}_staged_anchor.json",
        RES_ROOT / "datasets" / stem_only / "partials" / f"{dataset}_staged_anchor.json",
    ]
    for partial in candidates:
        if partial.is_file():
            return json.loads(partial.read_text())

    search_roots = []
    if cfg_stem:
        search_roots += [RES_ROOT / cfg_stem, RES_ROOT / "datasets" / cfg_stem]
    search_roots += [RES_ROOT / stem_only, RES_ROOT / "datasets" / stem_only]
    for root in search_roots:
        if not root.is_dir():
            continue
        for p in root.glob("*/staged_results.json"):
            d = json.loads(p.read_text())
            m = (d.get("eval_metrics") or {}).get("staged_anchor") or d
            if "crps" in m:
                out = dict(m)
                if "seed" in d:
                    out.setdefault("seed", d["seed"])
                if "binary_length_g" in d:
                    out.setdefault("binary_length_g", d["binary_length_g"])
                return out
        # nested under dataset name
        nested = root / dataset / "staged_results.json"
        if nested.is_file():
            d = json.loads(nested.read_text())
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
        stem_g, stem_seed = _g_seed_from_stem(stem)
        g = float(hist_g) if hist_g is not None else float(stem_g or 1.0)
        if ev and ev.get("binary_length_g") is not None and hist_g is None:
            g = float(ev["binary_length_g"])
        seed = stem_seed
        if seed is None and hist_seed is not None:
            seed = hist_seed
        if seed is None and ev and ev.get("seed") is not None:
            seed = int(ev["seed"])
        if seed is None:
            seed = 42
        smoke = n_ep > 0 and n_ep < min_epochs
        if smoke:
            continue
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


def _curve_by_g(points: List[RunRow], prefer_seed: int = 42) -> Dict[float, RunRow]:
    by_g: Dict[float, RunRow] = {}
    for p in points:
        if p.crps is None:
            continue
        if p.g not in by_g or p.seed == prefer_seed:
            by_g[p.g] = p
    return by_g


def apply_stopping_rule(
    points: List[RunRow],
    baseline_anchor: float,
) -> Tuple[Optional[float], str]:
    by_g = _curve_by_g(points)
    ordered = [by_g[g] for g in sorted(by_g)]
    for i in range(1, len(ordered)):
        prev, cur = ordered[i - 1], ordered[i]
        crps_improve = -_rel(prev.crps, cur.crps)
        anchor_deg = _rel(baseline_anchor, cur.anchor_mse) if cur.anchor_mse is not None else 0.0
        if crps_improve < CRPS_IMPROVE_FLOOR and anchor_deg > ANCHOR_DEGRADE_CAP:
            return cur.g, (
                f"stop at g={cur.g}: CRPS improve vs prev={crps_improve:.1%} < {CRPS_IMPROVE_FLOOR:.0%} "
                f"AND anchor degrade vs g=1={anchor_deg:.1%} > {ANCHOR_DEGRADE_CAP:.0%}"
            )
        if crps_improve < CRPS_IMPROVE_FLOOR and i == len(ordered) - 1:
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
    by_g = _curve_by_g(points)
    if not by_g:
        return 1.0, "no CRPS data"
    best = min(by_g.values(), key=lambda r: r.crps)
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
    # Near-tie on CRPS (≤0.5% relative): prefer clearly better anchor (e.g. elec g=3 vs g=10).
    for c in candidates:
        if c.g == chosen.g or c.crps is None or chosen.crps is None:
            continue
        if c.crps > chosen.crps * 1.005:
            continue
        if c.anchor_mse is not None and chosen.anchor_mse is not None:
            if c.anchor_mse < chosen.anchor_mse * 0.97:
                chosen = c
                break
    note = f"raw best CRPS g={best.g}"
    if chosen.g != best.g:
        note += f"; recommend g={chosen.g} (anchor soft-cap / near-tie)"
    else:
        note += f"; recommend g={chosen.g}"
    if noise is not None and baseline.crps is not None:
        if abs(chosen.crps - baseline.crps) < noise:
            note += f"; ΔCRPS within seed noise (±{noise:.4f})"
    return chosen.g, note


def refinement_notes(by_g: Dict[float, RunRow], rec_g: float) -> str:
    """Describe fine-grid behavior around the recommended peak."""
    has_fine = bool(set(by_g) & FINE_G)
    if not has_fine:
        # If peak is interior to coarse grid with neighbors present, note that
        neighbors = [g for g in sorted(by_g) if abs(g - rec_g) > 1e-9]
        if not neighbors:
            return "no neighbors yet"
        return "coarse grid only (fine neighbors not required or not yet run)"

    # Smoothness through 6→7→8→9 if those exist
    fine_path = [g for g in (6.0, 7.0, 8.0, 9.0, 10.0) if g in by_g]
    if len(fine_path) >= 3:
        crps = [by_g[g].crps for g in fine_path]
        # count sign changes in first differences
        diffs = [crps[i] - crps[i - 1] for i in range(1, len(crps))]
        sign_changes = sum(
            1 for i in range(1, len(diffs)) if diffs[i] * diffs[i - 1] < 0
        )
        peak_local = min(fine_path, key=lambda g: by_g[g].crps)
        if sign_changes <= 1:
            return (
                f"fine grid smooth; local min at g={peak_local:g} "
                f"(path {', '.join(f'{g:g}' for g in fine_path)})"
            )
        return (
            f"fine grid jagged (sign changes={sign_changes}); "
            f"local min at g={peak_local:g} — treat as provisional"
        )
    return f"partial fine grid present; recommended g={rec_g:g}"


def confirmation_status(
    rows: List[RunRow],
    dataset: str,
    rec_g: float,
    noise: Optional[float],
) -> Tuple[str, Optional[float], Optional[float]]:
    """Return (yes|flagged|missing, crps_42, crps_43) at recommended g."""
    at_g = [
        r for r in rows
        if r.dataset == dataset and abs(r.g - rec_g) < 1e-9 and r.crps is not None
    ]
    by_seed = {r.seed: r for r in at_g}
    r42 = by_seed.get(42)
    r43 = by_seed.get(43)
    c42 = r42.crps if r42 else None
    c43 = r43.crps if r43 else None
    if r43 is None:
        return "missing", c42, c43
    if r42 is None:
        return "flagged (no seed=42 at rec g)", c42, c43
    delta = abs(c43 - c42)
    # Disagree if outside g=1.0 seed noise floor (when available), else >5% relative
    if noise is not None:
        if delta > max(noise, 1e-6):
            return "flagged", c42, c43
        return "yes", c42, c43
    if r42.crps and delta / abs(r42.crps) > 0.05:
        return "flagged", c42, c43
    return "yes", c42, c43


def plot_dataset(
    dataset: str,
    points: List[RunRow],
    noise: Optional[float],
    baseline_crps: Optional[float],
    out_dir: Path,
) -> None:
    by_g = _curve_by_g(points)
    if not by_g:
        return
    gs = sorted(by_g)
    crps = [by_g[g].crps for g in gs]
    anchors = [by_g[g].anchor_mse for g in gs]

    plot_dir = out_dir / "comparison_summary"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Dual-axis: CRPS + anchor MSE
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
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
    ax.set_ylabel("CRPS (↓ better)", color="#1f4e79")
    ax.tick_params(axis="y", labelcolor="#1f4e79")
    ax.set_title(f"{dataset}: CRPS + anchor MSE vs g")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(gs, anchors, marker="s", color="#8b4513", ls="--", label="anchor MSE")
    ax2.set_ylabel("anchor MSE (↓ better)", color="#8b4513")
    ax2.tick_params(axis="y", labelcolor="#8b4513")

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(plot_dir / f"crps_anchor_vs_g_{dataset}.png", dpi=140)
    # Keep legacy single-metric filenames too
    fig.savefig(plot_dir / f"crps_vs_g_{dataset}.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(gs, anchors, marker="s", color="#8b4513", label="anchor MSE")
    if by_g.get(1.0) and by_g[1.0].anchor_mse is not None:
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


def _curve_shape_label(pts: List[Tuple[float, float]]) -> str:
    """Classify CRPS-vs-g shape: unimodal / monotone / multi-modal / flat.

    Secondary local minima only count as multi-modal if they are competitive
    (within 5% of the global best CRPS); shallow mid-grid dips do not.
    """
    if len(pts) < 3:
        return "too_few_points"
    crps_vals = [c for _, c in pts]
    global_best = min(crps_vals)
    local_mins = []
    for i in range(1, len(crps_vals) - 1):
        if crps_vals[i] < crps_vals[i - 1] and crps_vals[i] < crps_vals[i + 1]:
            local_mins.append((pts[i][0], crps_vals[i]))
    significant = [g for g, c in local_mins if c <= global_best * 1.05]
    best_i = min(range(len(crps_vals)), key=lambda i: crps_vals[i])
    upturn = any(
        crps_vals[i] > crps_vals[i - 1] * 1.02 for i in range(1, len(crps_vals))
    )
    if len(significant) >= 2:
        return "multi_modal"
    if upturn and best_i not in (0, len(crps_vals) - 1):
        return "unimodal"
    if upturn:
        return "non_monotone"
    return "monotone_improve"


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

    summary_rows = []
    md: List[str] = [
        "# CRPS-targeted noise schedule g calibration\n\n",
        "Fine val/BCE is reported for completeness only — **not comparable across g** "
        "(different forward process).\n\n",
        f"Stopping rule (post-hoc): consecutive CRPS improve < {CRPS_IMPROVE_FLOOR:.0%} "
        f"AND anchor MSE degrade vs g=1.0 > {ANCHOR_DEGRADE_CAP:.0%}.\n\n",
        "Confirmation seed: seed=43 at recommended g vs seed=42; flagged if |ΔCRPS| exceeds "
        "that dataset's g=1.0 seed noise floor.\n\n",
    ]

    curve_shapes: Dict[str, List[Tuple[float, float]]] = {}
    shape_labels: Dict[str, str] = {}

    for ds in args.datasets:
        ds_rows = [r for r in rows if r.dataset == ds]
        if not ds_rows:
            md.append(f"## {ds}\n\n_No runs yet._\n\n")
            continue
        noise, seed_crps = seed_noise_floor(rows, ds)
        by_g = _curve_by_g(ds_rows)
        if 1.0 not in by_g:
            md.append(f"## {ds}\n\n_Missing g=1.0 baseline._\n\n")
            continue
        base = by_g[1.0]
        stop_g, stop_reason = apply_stopping_rule(list(by_g.values()), base.anchor_mse or float("nan"))
        rec_g, rec_note = recommend_g(list(by_g.values()), base, noise)
        ref_note = refinement_notes(by_g, rec_g)
        conf_status, c42, c43 = confirmation_status(rows, ds, rec_g, noise)

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
        md.append(f"Grid refinement: {ref_note}\n\n")
        conf_line = f"Confirmation seed @ g={rec_g:g}: **{conf_status}**"
        if c42 is not None and c43 is not None:
            conf_line += f" (seed42 CRPS={c42:.4f}, seed43 CRPS={c43:.4f}, Δ={c43 - c42:+.4f})"
        elif c42 is not None:
            conf_line += f" (seed42 CRPS={c42:.4f}; seed43 pending)"
        md.append(conf_line + "\n\n")

        md.append(
            "| g | seed | fine val* | CRPS | ΔCRPS vs g=1 | anchor MSE | Δanchor | within seed noise? | notes |\n"
            "|---:|---:|---:|---:|---:|---:|---:|---|---|\n"
        )
        # Show all seeds at each g in the table (main curve seed=42 first)
        table_csv = []
        for g in sorted(by_g):
            r = by_g[g]
            dcrps = _rel(base.crps, r.crps)
            danc = _rel(base.anchor_mse, r.anchor_mse) if r.anchor_mse is not None else float("nan")
            within = ""
            if noise is not None and r.crps is not None and base.crps is not None:
                within = "yes" if abs(r.crps - base.crps) < noise and g != 1.0 else "no"
            note = ""
            if g in FINE_G:
                note = "fine grid"
            if stop_g is not None and g == stop_g:
                note = (note + "; " if note else "") + "stop rule"
            if abs(g - rec_g) < 1e-9:
                note = (note + "; " if note else "") + "recommended"
            fv = f"{r.fine_val:.4f}" if r.fine_val is not None else "nan"
            md.append(
                f"| {g:g} | {r.seed} | {fv} | "
                f"{r.crps:.4f} | {dcrps:+.1%} | "
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
        # Extra: list confirmation / other seeds at recommended g
        extra = [
            r for r in ds_rows
            if abs(r.g - rec_g) < 1e-9 and r.seed != 42 and r.crps is not None
        ]
        if extra:
            md.append("\nConfirmation / extra seeds at recommended g:\n\n")
            md.append("| g | seed | CRPS | anchor MSE |\n|---:|---:|---:|---:|\n")
            for r in sorted(extra, key=lambda x: x.seed):
                md.append(
                    f"| {r.g:g} | {r.seed} | {r.crps:.4f} | "
                    f"{r.anchor_mse:.4f if r.anchor_mse is not None else float('nan')} |\n"
                )

        md.append("\n\\*fine val not comparable across schedules\n\n")
        md.append(f"Recommendation: {rec_note}\n\n")

        csv_path = out_dir / f"table_{ds}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(table_csv[0].keys()))
            w.writeheader()
            w.writerows(table_csv)

        plot_dataset(ds, ds_rows, noise, base.crps, out_dir)
        curve_shapes[ds] = [(g, by_g[g].crps) for g in sorted(by_g)]
        shape_labels[ds] = _curve_shape_label(curve_shapes[ds])

        best_g = min(by_g.values(), key=lambda r: r.crps).g
        best = by_g[best_g]
        # Ceiling flag: recommended g still at the top of the grid with no plateau
        ordered = [by_g[g] for g in sorted(by_g)]
        plateau = False
        if len(ordered) >= 2:
            plateau = -_rel(ordered[-2].crps, ordered[-1].crps) < CRPS_IMPROVE_FLOOR
        hit_ceiling = abs(rec_g - 10.0) < 1e-9 and not plateau
        summary_rows.append({
            "dataset": ds,
            "best_crps_g": best_g,
            "crps_improvement": _rel(base.crps, best.crps),
            "crps_improvement_at_rec": _rel(base.crps, by_g[rec_g].crps),
            "hit_g10_without_plateau": bool(hit_ceiling),
            "anchor_cost_at_best": _rel(base.anchor_mse, best.anchor_mse),
            "anchor_cost_at_rec": _rel(base.anchor_mse, by_g[rec_g].anchor_mse),
            "recommended_g": rec_g,
            "stop_reason": stop_reason,
            "seed_noise": noise,
            "confirmation": conf_status,
            "confirm_crps_42": c42,
            "confirm_crps_43": c43,
            "refinement_notes": ref_note,
            "curve_shape": shape_labels[ds],
        })

    # Cross-dataset summary
    md.append("## Cross-dataset summary\n\n")
    md.append(
        "| dataset | recommended g | CRPS Δ vs g=1 | anchor Δ vs g=1 | "
        "confirm seed | grid refinement | curve shape |\n"
        "|---|---:|---:|---:|---|---|---|\n"
    )
    for s in summary_rows:
        md.append(
            f"| {s['dataset']} | {s['recommended_g']:g} | "
            f"{s['crps_improvement_at_rec']:+.1%} | {s['anchor_cost_at_rec']:+.1%} | "
            f"{s['confirmation']} | {s['refinement_notes']} | {s['curve_shape']} |\n"
        )

    md.append("\n## Curve-shape comparison\n\n")
    by_label: Dict[str, List[str]] = {}
    for ds, lab in shape_labels.items():
        by_label.setdefault(lab, []).append(ds)
    for lab, dslist in sorted(by_label.items()):
        md.append(f"- **{lab}**: {', '.join(dslist)}\n")
    md.append("\n")

    multi = by_label.get("multi_modal") or []
    conflict = [
        s["dataset"] for s in summary_rows
        if "provisional" in (s.get("refinement_notes") or "")
        or str(s.get("confirmation", "")).startswith("flagged")
    ]

    md.append("## Final recommendation\n\n")
    if not summary_rows:
        md.append("_Insufficient data._\n")
        rec_text = "insufficient data"
    elif multi:
        rec_text = (
            f"Multi-knot / free-form schedule is justified: multi-modal CRPS-vs-g on "
            f"{', '.join(multi)} — a single scalar g cannot resolve multiple local optima."
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
            "Per-dataset scalar g is sufficient: CRPS-vs-g curves are unimodal or "
            "monotone with dataset-specific peak locations; no new dataset shows a "
            "shape that a scalar g cannot resolve."
        )
        md.append(rec_text + "\n")
    if conflict:
        md.append(
            f"\nFlagged for closer look (jagged fine grid and/or confirmation mismatch): "
            f"{', '.join(conflict)}.\n"
        )

    summary_path = out_dir / "comparison_summary.md"
    summary_path.write_text("".join(md), encoding="utf-8")
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary_rows": summary_rows,
                "recommendation": rec_text,
                "curve_shapes": {k: v for k, v in curve_shapes.items()},
                "shape_labels": shape_labels,
            },
            f,
            indent=2,
        )
    print(f"Wrote {summary_path}")
    print(f"Recommendation: {rec_text}")
    for s in summary_rows:
        print(
            f"  {s['dataset']}: rec_g={s['recommended_g']} "
            f"CRPSΔ={s['crps_improvement_at_rec']:+.1%} "
            f"anchorΔ={s['anchor_cost_at_rec']:+.1%} "
            f"confirm={s['confirmation']} shape={s['curve_shape']}"
        )


if __name__ == "__main__":
    main()
