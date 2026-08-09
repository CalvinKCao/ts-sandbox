#!/usr/bin/env python3
"""Run the ordinal patch-refine → MMPD → assert → disc phase graph under coverage.

Fresh dirs only: never reuse shared synth caches, donor pretains, or prior ckpts.
Intended for Killarney via temp/submit_pipeline_coverage_deadcode.sh.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BINARY_CONFIG = "configs/coverage_deadcode_binary_patch_refine.yaml"
MMPD_CONFIG = "configs/coverage_deadcode_mmpd.yaml"
DATASET = "coverage_synth"
DISC_EVALUATOR = "temp/scripts/eval_univariate_patch_refine_ordinal_vs_mmpd.py"
COVERAGERC_NAME = ".coveragerc_deadcode"


def _ts() -> str:
    return time.strftime("%d-%H:%M:%S")


def _run_name(explicit: str | None) -> str:
    if explicit:
        return explicit
    job = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    return f"{time.strftime('%m-%d-%H%M')}-{job[-3:]}-coverage-deadcode"


def _write_coveragerc(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "[run]",
                "branch = True",
                "parallel = True",
                "patch = subprocess",
                "source =",
                "    models",
                "    utils",
                "    temp",
                "omit =",
                "    */temp/MMPD/*",
                "    */.venv/*",
                "    */site-packages/*",
                "    */tests/*",
                "",
                "[report]",
                "skip_empty = True",
                "show_missing = True",
                "precision = 1",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _coverage_env(coveragerc: Path, data_file: Path) -> dict:
    env = os.environ.copy()
    env["COVERAGE_PROCESS_START"] = str(coveragerc)
    env["COVERAGE_FILE"] = str(data_file)
    env["WANDB_MODE"] = "disabled"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return env


def _run_covered(
    label: str,
    argv: Sequence[str],
    *,
    env: dict,
    log_dir: Path,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{label}.log"
    print(f"[{_ts()}] === {label} ===", flush=True)
    print(f"[{_ts()}] cmd: {' '.join(argv)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"cmd: {' '.join(argv)}\n\n")
        log.flush()
        proc = subprocess.run(
            list(argv),
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(
            f"{label} failed with exit {proc.returncode}. Tail of {log_path}:\n{tail}"
        )
    print(f"[{_ts()}] {label} ok → {log_path}", flush=True)


def _cov_python(coveragerc: Path) -> List[str]:
    return [
        sys.executable,
        "-m",
        "coverage",
        "run",
        "--rcfile",
        str(coveragerc),
        "--parallel-mode",
    ]


def _find_binary_ckpt_root(ckpt_dir: Path) -> Path:
    coarse = sorted(ckpt_dir.glob("**/coarse/best.pt"))
    refine = sorted(ckpt_dir.glob("**/patch_refine/best.pt"))
    if len(coarse) != 1 or len(refine) != 1:
        raise FileNotFoundError(
            f"expected one coarse and one patch_refine best.pt under {ckpt_dir}; "
            f"got coarse={len(coarse)} refine={len(refine)}"
        )
    # Disc evaluator expects the run root that contains <subset>/{coarse,patch_refine}/.
    return ckpt_dir.resolve()


def _never_executed_summary(cov_json: dict, roots: Sequence[str]) -> List[dict]:
    files = cov_json.get("files") or {}
    rows: List[dict] = []
    for rel in roots:
        root = REPO_ROOT / rel
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if "MMPD" in path.parts:
                continue
            key = str(path.resolve())
            # coverage json keys are usually absolute or relative; try both
            info = files.get(key) or files.get(str(path)) or files.get(str(path.relative_to(REPO_ROOT)))
            if info is None:
                # unmatched path → treat as never imported
                rows.append(
                    {
                        "file": str(path.relative_to(REPO_ROOT)),
                        "executed_lines": 0,
                        "missing_lines": None,
                        "coverage_pct": 0.0,
                        "status": "never_imported",
                    }
                )
                continue
            summary = info.get("summary") or {}
            covered = int(summary.get("covered_lines") or 0)
            n_stmt = int(summary.get("num_statements") or 0)
            pct = float(summary.get("percent_covered") or 0.0)
            if covered == 0 or pct < 1.0:
                rows.append(
                    {
                        "file": str(path.relative_to(REPO_ROOT)),
                        "executed_lines": covered,
                        "missing_lines": int(summary.get("missing_lines") or max(0, n_stmt - covered)),
                        "coverage_pct": pct,
                        "status": "zero_or_near_zero" if covered == 0 else "low",
                    }
                )
    rows.sort(key=lambda r: (r["coverage_pct"], r["file"]))
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", default=None)
    p.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    p.add_argument("--skip-inventory", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print planned cmds and exit")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run = _run_name(args.run_name)
    results_root = args.results_root.expanduser().resolve()
    run_root = results_root / run
    ckpt_dir = results_root / "ckpts" / run
    data_dir = results_root / "datasets" / run
    mmpd_dir = results_root / "datasets" / f"{run}-mmpd"
    disc_dir = results_root / "datasets" / f"{run}-disc"
    raw_dir = results_root / "datasets" / f"{run}-disc-raw"
    cov_dir = run_root / "coverage"
    log_dir = run_root / "logs"
    synth_cache = run_root / "synth_cache"
    for d in (run_root, ckpt_dir, data_dir, mmpd_dir, disc_dir, raw_dir, cov_dir, log_dir, synth_cache):
        d.mkdir(parents=True, exist_ok=True)

    coveragerc = run_root / COVERAGERC_NAME
    cov_data = cov_dir / ".coverage"
    _write_coveragerc(coveragerc)
    env = _coverage_env(coveragerc, cov_data)
    py = _cov_python(coveragerc)

    binary_argv = py + [
        "-m",
        "models.diffusion_tsf.train_multivariate_pipeline",
        "--config",
        BINARY_CONFIG,
        "--dataset",
        DATASET,
        "--n-variates",
        "2",
        "--fresh",
        "--checkpoint-dir",
        str(ckpt_dir),
        "--results-dir",
        str(data_dir),
        "--datasets-dir",
        str(REPO_ROOT / "datasets"),
        "--synth-cache-dir",
        str(synth_cache),
    ]

    mmpd_common = [
        "-m",
        "utils.eval_mmpd_gaussian_anchor",
        "--mmpd-run-config",
        MMPD_CONFIG,
        "--datasets",
        DATASET,
        "--output-dir",
        str(mmpd_dir),
        "--mmpd-repo",
        str(REPO_ROOT / "temp" / "MMPD"),
        "--mmpd-data-dir",
        str(run_root / "mmpd_data"),
        "--force-mmpd-train",
        "--force-mmpd-eval",
        "--force-indices",
        "--mmpd-instance-norm",
        "--smoke-test",
        # Disc hard-temporal split needs span ≳ lb+hz; 1-window MMPD packs always fail.
        "--test-max-items",
        "48",
        "--no-update-mmpd",
        "--skip-mmpd-sample-viz",
    ]

    mmpd_data = run_root / "mmpd_data"

    disc_common = [
        DISC_EVALUATOR,
        "--datasets",
        DATASET,
        "--binary-config",
        BINARY_CONFIG,
        "--mmpd-output-root",
        str(mmpd_dir),
        "--mmpd-data-dir",
        str(mmpd_data),
        "--mmpd-repo",
        str(REPO_ROOT / "temp" / "MMPD"),
        "--pack-test-stride",
        "4",
        "--test-stride",
        "4",
        "--test-fraction",
        "1.0",
        "--disc-index-stride",
        "1",
        "--raw-binary-batch-size",
        "1",
        "--slice-lengths",
        "8",
        "--epochs",
        "2",
        "--max-batches-per-epoch",
        "1",
        "--batch-size",
        "8",
        "--force-raw-eval",
        "--force-train",
        "--no-visualize-confusions",
        "--no-viz-anchor-prob-panels",
        # No --smoke-test: that caps max_windows at 8 and breaks lb336 temporal purge.
        "--max-windows",
        "48",
        "--num-sampling-steps",
        "2",
        "--max-train-examples",
        "128",
        "--max-eval-examples",
        "64",
    ]

    plan = {
        "run": run,
        "ckpt_dir": str(ckpt_dir),
        "mmpd_dir": str(mmpd_dir),
        "disc_dir": str(disc_dir),
        "coverage_dir": str(cov_dir),
    }
    (run_root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        print("dry-run: binary", " ".join(binary_argv))
        return 0

    mmpd_repo = REPO_ROOT / "temp" / "MMPD"
    if not (mmpd_repo / ".git").is_dir() and not (mmpd_repo / "main_mmpd.py").is_file():
        raise FileNotFoundError(
            f"MMPD checkout missing at {mmpd_repo}. Clone on the login node before submit."
        )
    metrics_py = mmpd_repo / "metrics" / "prob_metrics.py"
    decoder_py = mmpd_repo / "models" / "backbones" / "decoder_only_transformer.py"
    if not metrics_py.is_file() or not decoder_py.is_file():
        raise FileNotFoundError(
            f"MMPD checkout incomplete (need {metrics_py.name} and {decoder_py.name}). "
            "Re-clone temp/MMPD on the login node."
        )
    from utils.eval_mmpd_gaussian_anchor import ensure_mmpd_repo

    print(f"[{_ts()}] ensuring MMPD patches at {mmpd_repo}", flush=True)
    ensure_mmpd_repo(mmpd_repo, update=False)

    from temp.make_coverage_synth_dataset import write_coverage_synth

    synth_csv = REPO_ROOT / "datasets" / "coverage_synth" / "coverage_synth.csv"
    if not synth_csv.is_file():
        write_coverage_synth(synth_csv)
        print(f"[{_ts()}] wrote {synth_csv}", flush=True)

    t0 = time.time()
    _run_covered("01_binary_pipeline", binary_argv, env=env, log_dir=log_dir)

    for phase, label in (("init", "02_mmpd_init"), ("mmpd", "03_mmpd_train_eval"), ("merge", "04_mmpd_merge")):
        argv = py + mmpd_common + ["--phase", phase]
        if phase == "merge":
            argv += ["--cpu"]
        _run_covered(label, argv, env=env, log_dir=log_dir)

    ckpt_root = _find_binary_ckpt_root(ckpt_dir)
    assert_argv = py + disc_common + [
        "--checkpoint-dir",
        str(ckpt_root),
        "--output-dir",
        str(disc_dir / "assert"),
        "--raw-eval-dir",
        str(raw_dir / "assert"),
        "--assert-only",
        "--assert-max-windows",
        "1",
    ]
    _run_covered("05_ordinal_assert", assert_argv, env=env, log_dir=log_dir)

    disc_argv = py + disc_common + [
        "--checkpoint-dir",
        str(ckpt_root),
        "--output-dir",
        str(disc_dir),
        "--raw-eval-dir",
        str(raw_dir),
    ]
    _run_covered("06_ordinal_disc", disc_argv, env=env, log_dir=log_dir)

    print(f"[{_ts()}] combining coverage…", flush=True)
    subprocess.run(
        [sys.executable, "-m", "coverage", "combine", "--rcfile", str(coveragerc)],
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
    )
    html_dir = cov_dir / "html"
    json_path = cov_dir / "coverage.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "html",
            "--rcfile",
            str(coveragerc),
            "-d",
            str(html_dir),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "json",
            "--rcfile",
            str(coveragerc),
            "-o",
            str(json_path),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
    )
    report_txt = cov_dir / "coverage_report.txt"
    with report_txt.open("w", encoding="utf-8") as fh:
        subprocess.run(
            [sys.executable, "-m", "coverage", "report", "--rcfile", str(coveragerc), "-m"],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        )

    cov_payload = json.loads(json_path.read_text(encoding="utf-8"))
    never_rows = _never_executed_summary(cov_payload, ["models/diffusion_tsf", "utils", "temp"])
    never_path = cov_dir / "never_or_low_coverage.json"
    never_path.write_text(json.dumps(never_rows, indent=2) + "\n", encoding="utf-8")
    print(f"[{_ts()}] low/never coverage files: {len(never_rows)} → {never_path}", flush=True)

    if not args.skip_inventory:
        inv_argv = [
            sys.executable,
            str(REPO_ROOT / "temp" / "inventory_redundant_artifacts.py"),
            "--output",
            str(run_root / "redundant_candidates.md"),
            "--coverage-json",
            str(json_path),
        ]
        print(f"[{_ts()}] running inventory…", flush=True)
        subprocess.run(inv_argv, cwd=str(REPO_ROOT), check=True)

    elapsed = time.time() - t0
    summary = {
        "run": run,
        "elapsed_sec": round(elapsed, 1),
        "ckpt_root": str(ckpt_root),
        "mmpd_dir": str(mmpd_dir),
        "coverage_html": str(html_dir / "index.html"),
        "coverage_json": str(json_path),
        "never_or_low": str(never_path),
        "redundant_candidates": str(run_root / "redundant_candidates.md"),
        "n_low_coverage_files": len(never_rows),
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[{_ts()}] done in {elapsed / 60.0:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
