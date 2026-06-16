#!/usr/bin/env python3
"""MMPD paper reproduction: MaskAE backbone, full datasets, appendix D.3 hyperparams.

Trains and tests upstream MMPD directly (no binary-anchor subsets). One worker
runs all horizons {96,192,336,720} for a single MMPD dataset name.

Paper defaults (D.3):
  T=336, P=12 (P=24 for tau in {336,720} or ECL/Traffic)
  d_model=256, r=3, K_train=1000, lambda=0.99 -> point_weight=0.01
  Adam 1e-4, 20 epochs, patience 5
  Inference: N=100, M=10, K_infer=20, EM=10, rho=0.5, u=100
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.eval_mmpd_gaussian_anchor import (  # noqa: E402
    DATASET_FILES,
    DEFAULT_MMPD_REPO,
    ensure_mmpd_repo,
    pipeline_python,
    run_cmd,
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "datasets" / "mmpd-paper-maskae"
DEFAULT_MMPD_DATA = REPO_ROOT / "temp" / "mmpd_paper_datasets"

PAPER_HORIZONS = (96, 192, 336, 720)
IN_LEN = 336
BACKBONE = "MaskAE"
LOSS = "MMPD"

# MMPD --data name -> repo dataset key + staged CSV filename
PAPER_DATASETS: Dict[str, Dict[str, str]] = {
    "ETTh1": {"source": "ETTh1", "csv": "ETTh1.csv"},
    "ETTh2": {"source": "ETTh2", "csv": "ETTh2.csv"},
    "ETTm1": {"source": "ETTm1", "csv": "ETTm1.csv"},
    "ETTm2": {"source": "ETTm2", "csv": "ETTm2.csv"},
    "weather": {"source": "weather", "csv": "weather.csv"},
    "ECL": {"source": "electricity", "csv": "electricity.csv"},
    "Traffic": {"source": "traffic", "csv": "traffic.csv"},
}
DEFAULT_DATASET_ORDER = tuple(PAPER_DATASETS.keys())


from utils.mmpd_paper_hparams import mmpd_patch_size as paper_patch_size


def paper_train_batch_size(mmpd_data: str, requested: int) -> int:
    if mmpd_data == "Traffic":
        return min(requested, 4)
    if mmpd_data == "ECL":
        return min(requested, 8)
    return requested


def paper_test_batch_size(mmpd_data: str) -> int:
    if mmpd_data in ("ECL", "Traffic"):
        return 1
    return 32


def mmpd_setting(
    mmpd_data: str,
    horizon: int,
    patch_size: int,
    *,
    backbone: str = BACKBONE,
    loss_func: str = LOSS,
    point_weight: float = 0.01,
) -> str:
    return (
        f"data{mmpd_data}_il{IN_LEN}_ol{horizon}_backbone{backbone}_loss{loss_func}"
        f"_weightedTrue_patch{patch_size}_pointW{point_weight}"
        f"_diffH256_diffLayer1_radius3_diffStep1000_betalinear"
    )


def stage_paper_datasets(data_dir: Path, datasets: Sequence[str]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in datasets:
        spec = PAPER_DATASETS[name]
        src = DATASET_FILES[spec["source"]]
        if not src.is_file():
            raise FileNotFoundError(f"Missing dataset source for {name}: {src}")
        dst = data_dir / spec["csv"]
        if dst.exists():
            try:
                if dst.samefile(src):
                    continue
            except OSError:
                pass
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)
        print(f"[stage] {name}: {dst} <- {src}")


def checkpoint_path(output_root: Path, setting: str) -> Path:
    return (
        output_root
        / "checkpoints"
        / f"{BACKBONE}-{LOSS}"
        / setting
        / "model_checkpoint.pth"
    )


def metrics_path(output_root: Path, setting: str) -> Path:
    return (
        output_root
        / "results"
        / f"{BACKBONE}-{LOSS}"
        / setting
        / "metrics.txt"
    )


def parse_metrics_txt(path: Path) -> Dict[str, float]:
    text = path.read_text(encoding="utf-8")
    out: Dict[str, float] = {}
    for key in ("CRPS", "MSE", "MAE"):
        m = re.search(rf"^{key}:\s*([0-9.eE+-]+)", text, re.MULTILINE)
        if m:
            out[key.lower()] = float(m.group(1))
    for k in range(1, 6):
        for metric in ("mse", "mae"):
            m = re.search(
                rf"^top{k}\s+{metric.upper()}:\s*([0-9.eE+-]+)",
                text,
                re.MULTILINE | re.IGNORECASE,
            )
            if m:
                out[f"top{k}_{metric}"] = float(m.group(1))
    return out


def build_main_cmd(
    args: argparse.Namespace,
    mmpd_data: str,
    horizon: int,
    *,
    training: bool,
    testing: bool,
) -> List[str]:
    patch_size = paper_patch_size(mmpd_data, horizon)
    train_bs = paper_train_batch_size(mmpd_data, args.batch_size)
    test_bs = paper_test_batch_size(mmpd_data) if testing else train_bs
    batch_size = test_bs if testing and not training else train_bs

    cmd = [
        pipeline_python(),
        "-u",
        "main_mmpd.py",
        "--data",
        mmpd_data,
        "--root_path",
        str(args.mmpd_data_dir),
        "--output_root",
        str(args.output_dir / "mmpd_out"),
        "--backbone",
        BACKBONE,
        "--loss_func",
        LOSS,
        "--in_len",
        str(IN_LEN),
        "--out_len",
        str(horizon),
        "--patch_size",
        str(patch_size),
        "--d_layers",
        "2",
        "--d_model",
        "256",
        "--d_ff",
        "512",
        "--n_heads",
        "4",
        "--weighted",
        "True",
        "--point_weight",
        "0.01",
        "--d_diffusion",
        "256",
        "--diffusion_layers",
        "1",
        "--radius",
        "3",
        "--max_diffusion_steps",
        "1000",
        "--beta_schedule",
        "linear",
        "--batch_size",
        str(batch_size),
        "--learning_rate",
        "1e-4",
        "--lradj",
        "cosine",
        "--train_epochs",
        str(args.train_epochs),
        "--patience",
        str(args.patience),
        "--training",
        "True" if training else "False",
        "--testing",
        "True" if testing else "False",
        "--num_workers",
        str(args.num_workers),
        "--gpu",
        str(args.gpu),
    ]
    if testing:
        cmd.extend(
            [
                "--prob_pred",
                "True",
                "--test_batch_num",
                str(args.test_batch_num),
                "--sample_num",
                str(args.sample_num),
                "--num_sampling_steps",
                str(args.num_sampling_steps),
                "--temperature",
                "1.0",
                "--gmm_components",
                str(args.gmm_components),
                "--prior_pi_decay",
                "0.5",
                "--prior_precision_shape",
                "100",
                "--gmm_iterations",
                str(args.gmm_iterations),
            ]
        )
    if args.cpu or not __import__("torch").cuda.is_available():
        cmd.extend(["--use_gpu", "False"])
    return cmd


def run_horizon(
    args: argparse.Namespace,
    mmpd_data: str,
    horizon: int,
    log_dir: Path,
) -> Dict[str, Any]:
    patch_size = paper_patch_size(mmpd_data, horizon)
    setting = mmpd_setting(mmpd_data, horizon, patch_size)
    out_root = args.output_dir / "mmpd_out"
    ckpt = checkpoint_path(out_root, setting)
    metrics_file = metrics_path(out_root, setting)
    record: Dict[str, Any] = {
        "mmpd_data": mmpd_data,
        "horizon": horizon,
        "patch_size": patch_size,
        "setting": setting,
        "checkpoint": str(ckpt),
        "metrics_file": str(metrics_file),
    }

    if not ckpt.is_file() or args.force_train:
        log = log_dir / f"train_{mmpd_data}_hz{horizon}.log"
        run_cmd(
            build_main_cmd(args, mmpd_data, horizon, training=True, testing=False),
            cwd=args.mmpd_repo,
            log_path=log,
        )
    else:
        print(f"[train] reuse {ckpt}")

    if not metrics_file.is_file() or args.force_test:
        log = log_dir / f"test_{mmpd_data}_hz{horizon}.log"
        run_cmd(
            build_main_cmd(args, mmpd_data, horizon, training=False, testing=True),
            cwd=args.mmpd_repo,
            log_path=log,
        )
    else:
        print(f"[test] reuse {metrics_file}")

    if metrics_file.is_file():
        record["metrics"] = parse_metrics_txt(metrics_file)
    return record


def run_dataset(args: argparse.Namespace, mmpd_data: str) -> Dict[str, Any]:
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    horizons = list(args.horizons)
    if args.smoke_test:
        horizons = horizons[:1]
    results = []
    for horizon in horizons:
        print(f"[run] {mmpd_data} horizon={horizon}", flush=True)
        results.append(run_horizon(args, mmpd_data, horizon, log_dir))
    partial = {
        "mmpd_data": mmpd_data,
        "horizons": results,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    partial_path = args.output_dir / "partials" / f"{mmpd_data}_maskae_paper.json"
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    with partial_path.open("w", encoding="utf-8") as f:
        json.dump(partial, f, indent=2, sort_keys=True)
    print(f"[run] wrote {partial_path}")
    return partial


def run_phase_init(args: argparse.Namespace, commit: str) -> None:
    stage_paper_datasets(args.mmpd_data_dir, args.datasets)
    manifest = {
        "args": vars(args),
        "mmpd_commit": commit,
        "datasets": list(args.datasets),
        "horizons": list(args.horizons),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"[init] {args.output_dir / 'run_manifest.json'}")


def run_phase_merge(args: argparse.Namespace, commit: str) -> None:
    table: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name in args.datasets:
        partial_path = args.output_dir / "partials" / f"{name}_maskae_paper.json"
        if not partial_path.is_file():
            print(f"[merge] skip missing {partial_path}")
            continue
        with partial_path.open(encoding="utf-8") as f:
            partial = json.load(f)
        table[name] = {}
        for entry in partial.get("horizons", []):
            hz = str(entry["horizon"])
            table[name][hz] = entry.get("metrics", {})

    metrics_path_out = args.output_dir / "metrics.json"
    with metrics_path_out.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "mmpd_commit": commit,
                "backbone": BACKBONE,
                "metrics_by_dataset_horizon": table,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"[merge] wrote {metrics_path_out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase",
        choices=("init", "run", "merge"),
        default="run",
    )
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASET_ORDER))
    p.add_argument("--horizons", nargs="+", type=int, default=list(PAPER_HORIZONS))
    p.add_argument("--mmpd-repo", type=Path, default=DEFAULT_MMPD_REPO)
    p.add_argument("--mmpd-data-dir", type=Path, default=DEFAULT_MMPD_DATA)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--train-epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--sample-num", type=int, default=100)
    p.add_argument("--num-sampling-steps", type=int, default=20)
    p.add_argument("--gmm-components", type=int, default=10)
    p.add_argument("--gmm-iterations", type=int, default=10)
    p.add_argument("--test-batch-num", type=int, default=-1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-update-mmpd", action="store_true")
    p.add_argument("--force-train", action="store_true")
    p.add_argument("--force-test", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()
    args.datasets = list(dict.fromkeys(args.datasets))
    unknown = sorted(set(args.datasets) - set(PAPER_DATASETS))
    if unknown:
        raise ValueError(f"Unsupported paper dataset(s): {unknown}")
    if args.phase == "run" and len(args.datasets) != 1:
        raise ValueError("--phase run requires exactly one --datasets entry")
    if args.smoke_test:
        args.train_epochs = 1
        args.patience = 1
        args.sample_num = 5
        args.num_sampling_steps = 5
        args.gmm_components = 5
        args.gmm_iterations = 3
        args.test_batch_num = 2
        args.datasets = [args.datasets[0]]
        args.horizons = [args.horizons[0]]
    args.mmpd_repo = args.mmpd_repo.resolve()
    args.mmpd_data_dir = args.mmpd_data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    return args


def main() -> None:
    args = parse_args()
    commit = ensure_mmpd_repo(args.mmpd_repo, update=not args.no_update_mmpd)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "init":
        run_phase_init(args, commit)
        return
    if args.phase == "merge":
        run_phase_merge(args, commit)
        return
    run_dataset(args, args.datasets[0])


if __name__ == "__main__":
    main()
