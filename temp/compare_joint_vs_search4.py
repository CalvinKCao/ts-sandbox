#!/usr/bin/env python3
"""Compare joint_g_lr_batch_s30r20 vs search4_refit20 via anchor_mse only."""
from __future__ import annotations

import json
import re
from pathlib import Path

LOG_DIR = Path("results/logs")
CKPT_DIR = Path("results/ckpts")
DATASETS = [
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "weather",
    "PeMS",
    "solar_Alabama",
    "dynamic",
]

RE_STAGED = re.compile(
    r"staged eval done:.*?anchor_mse=([0-9.]+)"
)
RE_TRIAL_START = re.compile(
    r"trial=(\d+)\s+START\b.*?lr=([0-9.eE+-]+).*?\bg=([0-9.]+)"
)
RE_TRIAL_DONE = re.compile(
    r"trial=(\d+)\s+DONE\s+best_val=([0-9.]+).*?selection=(\S+)"
)
RE_STUDY_DONE = re.compile(
    r"Optuna study done.*?best_trial=(\d+)\s+best_val=([0-9.]+).*?lr=([0-9.eE+-]+)"
)
RE_REFIT = re.compile(
    r"refit_best:.*?refit_epochs=(\d+).*?lr=([0-9.eE+-]+).*?\bg=([0-9.]+)"
)
RE_G_FROM_NAME = re.compile(r"_g(\d+)p0\.log$")


def newest_log(dataset: str, suffix: str) -> Path | None:
    """Newest log whose name contains -<dataset>-...<suffix>.log (mtime)."""
    pat = f"*-{dataset}-*{suffix}.log"
    matches = list(LOG_DIR.glob(pat))
    if not matches:
        # solar_Alabama / PeMS naming is exact; also try without extra hyphens
        matches = [p for p in LOG_DIR.glob(f"*{suffix}.log") if f"-{dataset}-" in p.name]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def newest_g_grid_logs(dataset: str) -> dict[int, Path]:
    """Newest ablation log per g in 1..10 for dataset."""
    out: dict[int, Path] = {}
    for g in range(1, 11):
        matches = list(LOG_DIR.glob(f"*-{dataset}-*binary_noise_sched_ablation_vertical_dual_g{g}p0.log"))
        if not matches:
            continue
        out[g] = max(matches, key=lambda p: p.stat().st_mtime)
    return out


def read_text(path: Path) -> str:
    return path.read_text(errors="replace")


def extract_last_staged_anchor(text: str) -> float | None:
    vals = RE_STAGED.findall(text)
    return float(vals[-1]) if vals else None


def parse_joint_trials(text: str) -> list[dict]:
    starts: dict[int, dict] = {}
    for m in RE_TRIAL_START.finditer(text):
        tid = int(m.group(1))
        starts[tid] = {"trial": tid, "lr": float(m.group(2)), "g": float(m.group(3))}

    trials = []
    for m in RE_TRIAL_DONE.finditer(text):
        tid = int(m.group(1))
        sel = m.group(3)
        row = {
            "trial": tid,
            "best_val_anchor_mse": float(m.group(2)),
            "selection": sel,
            "lr": starts.get(tid, {}).get("lr"),
            "g": starts.get(tid, {}).get("g"),
        }
        trials.append(row)
    return trials


def parse_joint_status(text: str) -> dict:
    study = RE_STUDY_DONE.search(text)
    refit = RE_REFIT.search(text)
    return {
        "study_done": study is not None,
        "study_best_trial": int(study.group(1)) if study else None,
        "study_best_val": float(study.group(2)) if study else None,
        "study_best_lr": float(study.group(3)) if study else None,
        "refit_started": refit is not None or "refit_best:" in text,
        "refit_line": refit.group(0) if refit else None,
        "pipeline_complete": "PIPELINE COMPLETE" in text,
        "n_done_trials": len(RE_TRIAL_DONE.findall(text)),
        "n_start_trials": len(RE_TRIAL_START.findall(text)),
    }


def find_metadata(run_stem: str, dataset: str) -> Path | None:
    """Locate metadata.json under ckpts for this run stem."""
    base = CKPT_DIR / run_stem
    if not base.exists():
        # try glob
        cands = list(CKPT_DIR.glob(f"*{run_stem.split('-', 2)[-1]}*")) if False else []
        cands = [p for p in CKPT_DIR.iterdir() if p.name.startswith(run_stem) or run_stem in p.name]
        if not cands:
            # stem is full log stem without .log
            cands = [CKPT_DIR / run_stem] if (CKPT_DIR / run_stem).exists() else []
        if not cands:
            return None
        base = cands[0]
    metas = list(base.rglob("metadata.json"))
    # prefer vertical_dual
    vd = [m for m in metas if "vertical_dual" in str(m)]
    if vd:
        return vd[0]
    return metas[0] if metas else None


def fmt(x, nd=4, sci=False):
    if x is None:
        return "—"
    if isinstance(x, float):
        if sci or (0 < abs(x) < 1e-2) or abs(x) >= 1e3:
            return f"{x:.2e}"
        return f"{x:.{nd}f}"
    return str(x)


def main():
    print("=" * 100)
    print("COMPARISON: joint_g_lr_batch_s30r20  vs  search4_refit20  vs  g-grid ablation")
    print("Metric note: joint selection = decoded VAL anchor_mse (NOT diffusion val loss).")
    print("search4 staged eval = TEST anchor_mse. search4 meta best_val_loss = diffusion BCE (ignore).")
    print("g-grid = TEST anchor_mse from binary_noise_sched_ablation_vertical_dual_g{1..10}p0.")
    print("=" * 100)

    rows = []
    joint_details = {}

    for ds in DATASETS:
        s4 = newest_log(ds, "search4_refit20")
        jt = newest_log(ds, "joint_g_lr_batch_s30r20")
        g_logs = newest_g_grid_logs(ds)

        s4_test = extract_last_staged_anchor(read_text(s4)) if s4 else None
        jt_text = read_text(jt) if jt else ""
        trials = parse_joint_trials(jt_text) if jt else []
        status = parse_joint_status(jt_text) if jt else {}

        # best joint VAL among DONE trials
        best_jt = None
        if trials:
            best_jt = min(trials, key=lambda r: r["best_val_anchor_mse"])

        # g-grid best TEST
        g_results = {}
        for g, path in sorted(g_logs.items()):
            am = extract_last_staged_anchor(read_text(path))
            if am is not None:
                g_results[g] = am
        best_g = None
        best_g_mse = None
        if g_results:
            best_g, best_g_mse = min(g_results.items(), key=lambda kv: kv[1])

        rows.append(
            {
                "dataset": ds,
                "search4_log": s4.name if s4 else None,
                "joint_log": jt.name if jt else None,
                "search4_test_anchor": s4_test,
                "joint_best_val_anchor": best_jt["best_val_anchor_mse"] if best_jt else None,
                "joint_best_trial": best_jt["trial"] if best_jt else None,
                "joint_best_lr": best_jt["lr"] if best_jt else None,
                "joint_best_g": best_jt["g"] if best_jt else None,
                "ggrid_best_g": best_g,
                "ggrid_best_test_anchor": best_g_mse,
                "ggrid_n": len(g_results),
                "joint_n_done": len(trials),
                "status": status,
                "trials": trials,
                "g_results": g_results,
                "joint_path": jt,
                "search4_path": s4,
            }
        )
        joint_details[ds] = {"trials": trials, "status": status, "path": jt}

    # ---- summary table ----
    print("\n## SUMMARY TABLE (lower anchor_mse is better)")
    print(
        f"{'dataset':<16} {'s4_TEST':>10} {'jt_VAL':>10} {'jt_trial':>8} "
        f"{'jt_lr':>10} {'jt_g':>8} {'ggrid_TEST':>10} {'gg_g':>6} {'Δ jt-s4':>10} {'Δ jt-gg':>10}"
    )
    print("-" * 110)
    for r in rows:
        s4 = r["search4_test_anchor"]
        jt = r["joint_best_val_anchor"]
        gg = r["ggrid_best_test_anchor"]
        d_s4 = (jt - s4) if (jt is not None and s4 is not None) else None
        d_gg = (jt - gg) if (jt is not None and gg is not None) else None
        print(
            f"{r['dataset']:<16} {fmt(s4):>10} {fmt(jt):>10} "
            f"{fmt(r['joint_best_trial'],0):>8} {fmt(r['joint_best_lr'],2):>10} "
            f"{fmt(r['joint_best_g'],3):>8} {fmt(gg):>10} {fmt(r['ggrid_best_g'],0):>6} "
            f"{fmt(d_s4):>10} {fmt(d_gg):>10}"
        )
    print()
    print("Δ jt-s4 / Δ jt-gg: joint VAL − search4 TEST / joint VAL − g-grid TEST (negative ⇒ joint lower).")
    print("WARNING: VAL vs TEST splits differ — directional only, not a fair head-to-head on the same split.")

    # ---- log paths ----
    print("\n## LOG FILES USED (newest by mtime)")
    for r in rows:
        print(f"\n{r['dataset']}:")
        print(f"  search4: {r['search4_log'] or 'MISSING'}")
        print(f"  joint:   {r['joint_log'] or 'MISSING'}")
        print(f"  g-grid:  {r['ggrid_n']}/10 logs with staged eval")

    # ---- g-grid detail ----
    print("\n## G-GRID TEST anchor_mse (best highlighted)")
    for r in rows:
        if not r["g_results"]:
            print(f"\n{r['dataset']}: no g-grid results")
            continue
        parts = []
        for g in range(1, 11):
            v = r["g_results"].get(g)
            if v is None:
                parts.append(f"g{g}=—")
            elif g == r["ggrid_best_g"]:
                parts.append(f"g{g}={v:.4f}*")
            else:
                parts.append(f"g{g}={v:.4f}")
        print(f"\n{r['dataset']}: " + "  ".join(parts))

    # ---- joint trial details ----
    print("\n## JOINT TRIAL DETAILS (DONE trials; selection=anchor_mse VAL)")
    for r in rows:
        print(f"\n### {r['dataset']}  (n_done={r['joint_n_done']})")
        if not r["trials"]:
            print("  (no DONE trials yet)")
            st = r["status"]
            if st:
                print(f"  starts={st.get('n_start_trials')} study_done={st.get('study_done')} "
                      f"refit={st.get('refit_started')} PIPELINE_COMPLETE={st.get('pipeline_complete')}")
            continue
        print(f"  {'trial':>5} {'val_anchor':>12} {'lr':>12} {'g':>10} {'sel':>12}")
        for t in sorted(r["trials"], key=lambda x: x["best_val_anchor_mse"]):
            mark = " <-- best" if t["trial"] == r["joint_best_trial"] else ""
            print(
                f"  {t['trial']:>5} {fmt(t['best_val_anchor_mse']):>12} "
                f"{fmt(t['lr'], sci=True):>12} {fmt(t['g'],4):>10} {t['selection']:>12}{mark}"
            )
        st = r["status"]
        print(
            f"  status: study_done={st.get('study_done')} "
            f"study_best_trial={st.get('study_best_trial')} "
            f"study_best_val={fmt(st.get('study_best_val'))} "
            f"refit_started={st.get('refit_started')} "
            f"PIPELINE_COMPLETE={st.get('pipeline_complete')}"
        )

    # ---- dynamic special ----
    print("\n" + "=" * 100)
    print("## DYNAMIC JOINT — completion + metadata.json tuned params")
    print("=" * 100)
    dyn = next(r for r in rows if r["dataset"] == "dynamic")
    st = dyn["status"]
    print(f"log: {dyn['joint_log']}")
    print(f"  Optuna study done:     {st.get('study_done')}")
    if st.get("study_done"):
        print(
            f"    best_trial={st.get('study_best_trial')} "
            f"best_val_anchor={fmt(st.get('study_best_val'))} "
            f"lr={fmt(st.get('study_best_lr'), sci=True)}"
        )
    print(f"  refit started:         {st.get('refit_started')}")
    if st.get("refit_line"):
        print(f"    {st['refit_line']}")
    print(f"  PIPELINE COMPLETE:     {st.get('pipeline_complete')}")
    print(f"  DONE trials / STARTs:  {st.get('n_done_trials')} / {st.get('n_start_trials')}")

    # staged eval progress for dynamic
    if dyn["joint_path"]:
        text = read_text(dyn["joint_path"])
        staged = RE_STAGED.findall(text)
        print(f"  staged eval done lines: {len(staged)}")
        if staged:
            print(f"  last staged TEST anchor_mse: {staged[-1]}")
        if "staged eval start" in text and not staged:
            m = re.search(r"staged eval batch (\d+)/(\d+)", text)
            # last batch progress
            batches = re.findall(r"staged eval batch (\d+)/(\d+)", text)
            if batches:
                print(f"  staged eval in progress: batch {batches[-1][0]}/{batches[-1][1]}")

    run_stem = dyn["joint_log"].replace(".log", "") if dyn["joint_log"] else None
    meta_path = None
    if run_stem:
        meta_path = find_metadata(run_stem, "dynamic")
        # also try direct
        if meta_path is None:
            cand = list((CKPT_DIR / run_stem).rglob("metadata.json")) if (CKPT_DIR / run_stem).exists() else []
            meta_path = cand[0] if cand else None

    if meta_path and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"\nmetadata.json: {meta_path}")
        print(f"  selection_metric:       {meta.get('selection_metric')}")
        print(f"  best_trial:             {meta.get('best_trial')}")
        print(f"  best_selection_score:   {meta.get('best_selection_score')}  (refit VAL anchor)")
        print(f"  hp_best_val_loss:       {meta.get('hp_best_val_loss')}  (search VAL anchor)")
        print(f"  best_val_loss:          {meta.get('best_val_loss')}  (same as best_selection_score here)")
        print(f"  refit_completed:        {meta.get('refit_completed')}")
        print(f"  best_epoch:             {meta.get('best_epoch')}")
        tp = meta.get("tuned_params") or {}
        print("  tuned_params:")
        for k in sorted(tp.keys()):
            print(f"    {k}: {tp[k]}")
    else:
        print(f"\nmetadata.json: NOT FOUND for stem={run_stem}")

    print("\nDone.")


if __name__ == "__main__":
    main()
