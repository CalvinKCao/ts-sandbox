#!/usr/bin/env python3
"""Canvas128 binary vs hz96 MMPD: top-10 |anchor MSE| gap plots + redbox on same windows.

Ranking / gap panels use gaussian ANCHOR path only:
  - binary: final_anchor (StagedEvalPhase / sampler=anchor)
  - MMPD: deterministic det pack
  - score: |binary_anchor_mse - mmpd_anchor_mse|  (compare abs_diff)

Redbox uses the same top-10 window_index list with sampler=anchor (not sample_mean).

Example:
  python temp/scripts/viz_c128_point_gap_and_redbox.py --datasets ETTh2
  python temp/scripts/viz_c128_point_gap_and_redbox.py --datasets ETTh2 --skip-redbox
  python temp/scripts/viz_c128_point_gap_and_redbox.py --all
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_ROOT_DEFAULT = REPO_ROOT / "temp" / "lean_disc_c128_results" / "viz_point_gap"
COMPARE_PY = REPO_ROOT / "utils" / "compare_binary_mmpd_staged_diag.py"
REDBOX_PY = REPO_ROOT / "temp" / "scripts" / "viz_ablation_staged_eval_samples.py"
MMPD_CFG = "configs/mmpd_decoder_flat_subsets_paper_lb336_hz96_matched_binary.yaml"

# dataset -> (binary_cfg, ckpt_stem, mmpd_campaign_relpath)
DATASET_SPEC: Dict[str, Dict[str, str]] = {
    "ETTh1": {
        "binary_cfg": "configs/binary_window_norm_patch_refine_canvas128_p64x6.yaml",
        "ckpt_stem": "binary_window_norm_patch_refine_canvas128_p64x6",
        "mmpd_root": "results/datasets/07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd",
    },
    "ETTh2": {
        "binary_cfg": "configs/binary_window_norm_patch_refine_canvas128_p64x6_etth2.yaml",
        "ckpt_stem": "binary_window_norm_patch_refine_canvas128_p64x6_etth2",
        "mmpd_root": "results/datasets/08-04-mmpd-decoder-paper-lb336-hz96-ETTh2",
    },
    "electricity": {
        "binary_cfg": "configs/binary_window_norm_patch_refine_canvas128_p64x6_electricity.yaml",
        "ckpt_stem": "binary_window_norm_patch_refine_canvas128_p64x6_electricity",
        "mmpd_root": "results/datasets/07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd",
    },
    "traffic": {
        "binary_cfg": "configs/binary_window_norm_patch_refine_canvas128_p64x6_traffic.yaml",
        "ckpt_stem": "binary_window_norm_patch_refine_canvas128_p64x6_traffic",
        "mmpd_root": "results/datasets/07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd",
    },
    "exchange_rate": {
        "binary_cfg": "configs/binary_window_norm_patch_refine_canvas128_p64x6_exchange_rate.yaml",
        "ckpt_stem": "binary_window_norm_patch_refine_canvas128_p64x6_exchange_rate",
        "mmpd_root": "results/datasets/07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd",
    },
    "PeMS": {
        "binary_cfg": "configs/binary_window_norm_patch_refine_canvas128_p64x6_pems.yaml",
        "ckpt_stem": "binary_window_norm_patch_refine_canvas128_p64x6_pems",
        "mmpd_root": "results/datasets/08-06-mmpd-decoder-paper-lb336-hz96-matched-c128-four",
    },
    "solar_Alabama": {
        "binary_cfg": "configs/binary_window_norm_patch_refine_canvas128_p64x6_solar_alabama.yaml",
        "ckpt_stem": "binary_window_norm_patch_refine_canvas128_p64x6_solar_alabama",
        # Prefer binary-aligned fix campaign; fall back handled at runtime.
        "mmpd_root": "results/datasets/08-06-mmpd-solar-binary-aligned-fix",
        "mmpd_root_fallback": "results/datasets/08-06-mmpd-decoder-paper-lb336-hz96-matched-c128-four",
    },
    "ETTm1": {
        "binary_cfg": "configs/binary_window_norm_patch_refine_canvas128_p64x6_ettm1.yaml",
        "ckpt_stem": "binary_window_norm_patch_refine_canvas128_p64x6_ettm1",
        "mmpd_root": "results/datasets/08-06-mmpd-decoder-paper-lb336-hz96-matched-c128-four",
    },
    "ETTm2": {
        "binary_cfg": "configs/binary_window_norm_patch_refine_canvas128_p64x6_ettm2.yaml",
        "ckpt_stem": "binary_window_norm_patch_refine_canvas128_p64x6_ettm2",
        "mmpd_root": "results/datasets/08-06-mmpd-decoder-paper-lb336-hz96-matched-c128-four",
    },
}

ALL_DATASETS = list(DATASET_SPEC.keys())


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated dataset list (default: none; use --all).",
    )
    p.add_argument("--all", action="store_true", help="Run all known canvas128 datasets.")
    p.add_argument("--out-root", type=Path, default=OUT_ROOT_DEFAULT)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--device", default=None)
    p.add_argument("--skip-gap", action="store_true")
    p.add_argument("--skip-redbox", action="store_true")
    p.add_argument("--plots-only", action="store_true", help="Pass through to compare util.")
    p.add_argument("--force-eval", action="store_true")
    p.add_argument(
        "--test-max-items",
        type=int,
        default=None,
        help="Cap windows for a fast plumbing run (still ranks by anchor MSE).",
    )
    p.add_argument("--variables-to-plot", type=int, default=99,
                   help="Max variates per window for gap jpgs (capped by V; default all).")
    p.add_argument("--redbox-variables-to-plot", type=int, default=0,
                   help="Max variates for redbox (0 = all).")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _resolve_mmpd_root(spec: Dict[str, str], dataset: str) -> Path:
    primary = REPO_ROOT / spec["mmpd_root"]
    pack = primary / "raw" / f"mmpd_{dataset}.npz"
    if pack.is_file():
        return primary
    fb = spec.get("mmpd_root_fallback")
    if fb:
        alt = REPO_ROOT / fb
        alt_pack = alt / "raw" / f"mmpd_{dataset}.npz"
        if alt_pack.is_file():
            print(f"[mmpd] {dataset}: primary pack missing; using fallback {alt}", flush=True)
            return alt
    if not primary.is_dir():
        raise FileNotFoundError(f"{dataset}: missing MMPD campaign {primary}")
    if not pack.is_file():
        raise FileNotFoundError(
            f"{dataset}: missing MMPD pack {pack} (align/packs broken — fail-fast)"
        )
    return primary


# MMPD hz96 matched-binary packs were evaluated on the stride-4 lattice
# (same as base staged_eval). Canvas128 leaves often override staged_eval
# test_stride to 16 — do NOT use that for pack-aligned gap/redbox ranking.
EVAL_TEST_STRIDE = 4


def _ensure_guidance_ckpt(ckpt: Path, dataset: str, subset_id: str) -> Path:
    """Prefer `{subset_id}_patch_guidance.pt`; symlink legacy `{dataset}_…` names."""
    wanted = ckpt / f"{subset_id}_patch_guidance.pt"
    if wanted.is_file():
        return wanted
    legacy_candidates = [
        ckpt / f"{dataset}_patch_guidance.pt",
        ckpt / "patch_guidance.pt",
    ]
    for legacy in legacy_candidates:
        if legacy.is_file():
            wanted.symlink_to(legacy.name)
            print(f"[guidance] {dataset}: linked {wanted.name} -> {legacy.name}", flush=True)
            return wanted
    raise FileNotFoundError(
        f"{dataset}: missing guidance ckpt {wanted.name} under {ckpt} "
        "(rsync from Killarney $SCRATCH/ts-sandbox-ordinal-fine/results/ckpts/...)"
    )


def _ensure_subset_stage_dir(ckpt: Path, dataset: str, subset_id: str) -> Path:
    """Prefer `{subset_id}/coarse|patch_refine`; symlink legacy `{dataset}/` dirs."""
    wanted = ckpt / subset_id
    if wanted.is_dir():
        return wanted
    legacy = ckpt / dataset
    if legacy.is_dir():
        wanted.symlink_to(legacy.name)
        print(f"[ckpt-dir] {dataset}: linked {wanted.name} -> {legacy.name}", flush=True)
        return wanted
    raise FileNotFoundError(
        f"{dataset}: missing staged subset dir {wanted} (or legacy {legacy})"
    )


def _preflight(dataset: str, spec: Dict[str, str]) -> Path:
    cfg = REPO_ROOT / spec["binary_cfg"]
    if not cfg.is_file():
        raise FileNotFoundError(f"{dataset}: missing binary config {cfg}")
    mmpd_root = _resolve_mmpd_root(spec, dataset)
    # discover_binary_ckpt will validate weights; just ensure stem dir likely exists
    stem = spec["ckpt_stem"]
    ckpt_base = REPO_ROOT / "results" / "ckpts"
    matches = list(ckpt_base.glob(f"*-{dataset}-{stem}"))
    if not matches:
        raise FileNotFoundError(
            f"{dataset}: no binary ckpt *-{dataset}-{stem} under {ckpt_base}"
        )
    ckpt = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    # Fail-fast if patch guidance weights missing (needed for redbox overlay).
    from utils.eval_mmpd_gaussian_anchor import (
        build_anchor_runs_from_subset_config,
        run_subset_id,
    )

    runs = build_anchor_runs_from_subset_config(Path(spec["binary_cfg"]), [dataset], 2026)
    sid = run_subset_id(runs[dataset])
    _ensure_guidance_ckpt(ckpt, dataset, sid)
    subset_dir = _ensure_subset_stage_dir(ckpt, dataset, sid)
    for stage in ("coarse", "patch_refine"):
        best = subset_dir / stage / "best.pt"
        if not best.is_file() or best.stat().st_size < 1024:
            raise FileNotFoundError(
                f"{dataset}: missing/empty {best} "
                "(rsync stage best.pt from Killarney ckpts/...)"
            )
    return mmpd_root


def _run(cmd: List[str], *, dry_run: bool) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def run_gap(
    dataset: str,
    spec: Dict[str, str],
    mmpd_root: Path,
    *,
    out_root: Path,
    work_dir: Path,
    args: argparse.Namespace,
) -> Path:
    """Run compare util; copy top_diff jpgs + top_windows json into out_root/<ds>/gap/."""
    gap_dir = out_root / dataset / "gap"
    gap_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        str(COMPARE_PY),
        "--mmpd-config",
        str(REPO_ROOT / MMPD_CFG),
        "--mmpd-config-suffix",
        "mmpd_decoder_flat_subsets_paper_lb336_hz96_matched_binary",
        "--mmpd-dir",
        str(mmpd_root),
        "--mmpd-output-root",
        str(mmpd_root),
        "--no-auto-mmpd-ckpt",
        "--binary-config",
        spec["binary_cfg"],
        "--binary-ckpt-stem",
        spec["ckpt_stem"],
        "--datasets",
        dataset,
        "--output-dir",
        str(work_dir),
        "--top-k",
        str(int(args.top_k)),
        "--random-k",
        "0",
        "--diff-mode",
        "abs_diff",
        "--test-fraction",
        "1.0",
        "--variables-to-plot",
        str(int(args.variables_to_plot)),
        "--min-spacing",
        "48",
        "--eval-test-stride",
        str(EVAL_TEST_STRIDE),
    ]
    if args.device:
        cmd.extend(["--device", str(args.device)])
    if args.plots_only:
        cmd.append("--plots-only")
    if args.force_eval:
        cmd.append("--force-eval")
    if args.test_max_items is not None:
        cmd.extend(["--test-max-items", str(int(args.test_max_items))])

    _run(cmd, dry_run=bool(args.dry_run))
    if args.dry_run:
        return gap_dir

    top_src = work_dir / "top_windows" / f"{dataset}.json"
    if not top_src.is_file():
        raise FileNotFoundError(f"missing top windows manifest: {top_src}")
    shutil.copy2(top_src, gap_dir / "top_windows.json")

    plot_src = work_dir / "plots" / dataset
    n_copied = 0
    if plot_src.is_dir():
        for jpg in sorted(plot_src.glob("top_diff_*.jpg")):
            shutil.copy2(jpg, gap_dir / jpg.name)
            n_copied += 1
    meta = {
        "dataset": dataset,
        "rank_metric": "abs(binary_anchor_mse - mmpd_anchor_mse)",
        "binary_pred": "final_anchor / sampler=anchor",
        "mmpd_pred": "deterministic (gaussian anchor det)",
        "eval_test_stride": EVAL_TEST_STRIDE,
        "top_k": int(args.top_k),
        "gap_jpg_count": n_copied,
        "mmpd_root": str(mmpd_root),
        "binary_cfg": spec["binary_cfg"],
        "ckpt_stem": spec["ckpt_stem"],
        "compare_work_dir": str(work_dir),
    }
    (gap_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[gap] {dataset}: copied {n_copied} top_diff jpgs -> {gap_dir}", flush=True)
    return gap_dir


def _top_diff_window_indices(gap_dir: Path, top_k: int) -> List[int]:
    path = gap_dir / "top_windows.json"
    with path.open(encoding="utf-8") as f:
        rows: List[Dict[str, Any]] = json.load(f)
    picks = [
        int(r["window_index"])
        for r in rows
        if str(r.get("pick_kind", "top_diff")) == "top_diff"
    ]
    picks = picks[: int(top_k)]
    if not picks:
        raise RuntimeError(f"{path}: no top_diff entries")
    return picks


def run_redbox(
    dataset: str,
    spec: Dict[str, str],
    *,
    out_root: Path,
    gap_dir: Path,
    args: argparse.Namespace,
) -> Path:
    redbox_dir = out_root / dataset / "redbox"
    redbox_dir.mkdir(parents=True, exist_ok=True)
    windows = _top_diff_window_indices(gap_dir, int(args.top_k))
    (redbox_dir / "top_windows_used.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "window_indices": windows,
                "sampler": "anchor",
                "guidance_overlay": "guidance_prediction_global_norm on 1d panels",
                "note": "same top-10 as gap/ (anchor MSE ranking); 1d shows GT+refine+guidance",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # pack_test_stride must match gap ranking lattice (MMPD pack / eval_test_stride).
    ckpt_base = REPO_ROOT / "results" / "ckpts"
    matches = sorted(
        ckpt_base.glob(f"*-{dataset}-{spec['ckpt_stem']}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"{dataset}: no ckpt for redbox")
    ckpt = matches[0]
    run_spec = f"canvas128:{ckpt}:{spec['binary_cfg']}"

    cmd = [
        sys.executable,
        "-u",
        str(REDBOX_PY),
        "--dataset",
        dataset,
        "--output-root",
        str(redbox_dir),
        "--lookback",
        "336",
        "--horizon",
        "96",
        "--pack-test-stride",
        str(EVAL_TEST_STRIDE),
        "--pack-splits",
        "test",
        "--n-samples",
        str(len(windows)),
        "--pool-indices",
        *[str(i) for i in windows],
        "--variables-to-plot",
        str(int(args.redbox_variables_to_plot)),
        "--sampler",
        "anchor",
        "--num-sampling-steps",
        "1",
        "--runs",
        run_spec,
    ]
    if args.device:
        cmd.extend(["--device", str(args.device)])
    _run(cmd, dry_run=bool(args.dry_run))
    if args.dry_run:
        return redbox_dir

    # Count refine_boxes jpgs
    n_rb = len(list(redbox_dir.rglob("*_refine_boxes.jpg")))
    print(f"[redbox] {dataset}: {n_rb} refine_boxes jpgs under {redbox_dir}", flush=True)
    return redbox_dir


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.all:
        datasets = list(ALL_DATASETS)
    elif args.datasets:
        datasets = [d.strip() for d in str(args.datasets).split(",") if d.strip()]
    else:
        raise SystemExit("pass --datasets ... or --all")

    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    work_root = out_root / "_compare_work"
    summary: List[Dict[str, Any]] = []

    print(
        "[policy] rank+plot = gaussian ANCHOR only "
        "(|binary_anchor_mse - mmpd_anchor_mse|; MMPD deterministic; "
        "binary final_anchor / sampler=anchor). Not sample_mean. "
        f"eval/pack test_stride={EVAL_TEST_STRIDE} (MMPD matched-binary lattice). "
        "Gap + redbox 1d overlay guidance_prediction_global_norm.",
        flush=True,
    )

    for dataset in datasets:
        if dataset not in DATASET_SPEC:
            raise KeyError(f"unknown dataset {dataset!r}; known={list(DATASET_SPEC)}")
        spec = DATASET_SPEC[dataset]
        row: Dict[str, Any] = {"dataset": dataset, "ok": False}
        try:
            mmpd_root = _preflight(dataset, spec)
            gap_dir = out_root / dataset / "gap"
            if not args.skip_gap:
                gap_dir = run_gap(
                    dataset,
                    spec,
                    mmpd_root,
                    out_root=out_root,
                    work_dir=work_root / dataset,
                    args=args,
                )
            elif not (gap_dir / "top_windows.json").is_file():
                raise FileNotFoundError(
                    f"--skip-gap but missing {gap_dir / 'top_windows.json'}"
                )

            if args.dry_run:
                row["ok"] = True
                row["dry_run"] = True
                summary.append(row)
                continue

            windows = _top_diff_window_indices(gap_dir, int(args.top_k))
            row["top10_window_indices"] = windows
            row["gap_jpg_count"] = len(list(gap_dir.glob("top_diff_*.jpg")))

            if not args.skip_redbox:
                redbox_dir = run_redbox(
                    dataset, spec, out_root=out_root, gap_dir=gap_dir, args=args
                )
                row["redbox_refine_count"] = len(
                    list(redbox_dir.rglob("*_refine_boxes.jpg"))
                )
                row["redbox_dir"] = str(redbox_dir)
            row["gap_dir"] = str(gap_dir)
            row["ok"] = True
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[FAIL] {dataset}: {row['error']}", flush=True)
            if dataset == "solar_Alabama":
                print(
                    "[note] solar_Alabama skipped/failed (align/packs) — fail-fast as requested",
                    flush=True,
                )
        summary.append(row)

    summary_path = out_root / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[summary] {summary_path}", flush=True)
    for row in summary:
        status = "OK" if row.get("ok") else "FAIL"
        print(
            f"  {status} {row['dataset']}: gap={row.get('gap_jpg_count')} "
            f"redbox={row.get('redbox_refine_count')} "
            f"windows={row.get('top10_window_indices')} "
            f"{row.get('error', '')}",
            flush=True,
        )
    if any(not r.get("ok") for r in summary):
        sys.exit(1)


if __name__ == "__main__":
    main()
