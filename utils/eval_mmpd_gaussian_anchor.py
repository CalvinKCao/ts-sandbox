#!/usr/bin/env python3
"""One-off MMPD vs binary/Gaussian-anchor evaluation.

The script keeps the upstream MMPD checkout and generated artifacts in ignored
scratch/output folders. It trains upstream MMPD on the same datasets as the
latest Gaussian-anchor checkpoints, then evaluates both models on a shared
random subset of the test windows.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from utils.mmpd_eval_progress import EvalProgress, fmt_duration
from utils.mmpd_paper_hparams import resolve_mmpd_patch_size


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pipeline_python() -> str:
    """Venv interpreter; module load after activate can leave sys.executable on CVMFS."""
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        py = Path(venv) / "bin" / "python"
        if py.is_file():
            return str(py)
    py_env = os.environ.get("PYTHON")
    if py_env and Path(py_env).is_file():
        return py_env
    return sys.executable
DEFAULT_MMPD_REPO = REPO_ROOT / "temp" / "MMPD"
DEFAULT_MMPD_DATA = REPO_ROOT / "temp" / "mmpd_datasets"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "mmpd_anchor_eval"
PATCHED_DATASET_MTS = REPO_ROOT / "utils" / "mmpd_patches" / "dataset_mts.py"
PATCHED_MASKAE_BACKBONE = (
    REPO_ROOT / "utils" / "mmpd_patches" / "models" / "backbones" / "mask_ae_transformer.py"
)
PATCHED_EXP_FORECAST = REPO_ROOT / "utils" / "mmpd_patches" / "exp" / "exp_forecast.py"
MMPD_BACKBONES = ("Decoder", "EncoderDecoder", "MaskAE")
MMPD_URL = "https://github.com/Thinklab-SJTU/MMPD.git"

DATASET_FILES = {
    "ETTh1": REPO_ROOT / "datasets" / "ETT-small" / "ETTh1.csv",
    "ETTh2": REPO_ROOT / "datasets" / "ETT-small" / "ETTh2.csv",
    "ETTm1": REPO_ROOT / "datasets" / "ETT-small" / "ETTm1.csv",
    "ETTm2": REPO_ROOT / "datasets" / "ETT-small" / "ETTm2.csv",
    "illness": REPO_ROOT / "datasets" / "illness" / "national_illness.csv",
    "exchange_rate": REPO_ROOT / "datasets" / "exchange_rate" / "exchange_rate.csv",
    "weather": REPO_ROOT / "datasets" / "weather" / "weather.csv",
    "electricity": REPO_ROOT / "datasets" / "electricity" / "electricity.csv",
    "traffic": REPO_ROOT / "datasets" / "traffic" / "traffic.csv",
    "PeMS": REPO_ROOT / "datasets" / "PeMS" / "PEMS04.npz",
    "solar_Alabama": REPO_ROOT / "datasets" / "solar_Alabama" / "solar_Alabama.csv",
    "dalia": REPO_ROOT / "datasets" / "dalia" / "dalia.csv",
    "dynamic": REPO_ROOT / "datasets" / "dynamic" / "dynamic_500K.csv",
}
DATASET_DIMS = {
    "ETTh1": 7,
    "ETTh2": 7,
    "ETTm1": 7,
    "ETTm2": 7,
    "illness": 7,
    "exchange_rate": 8,
    "weather": 21,
    "electricity": 321,
    "traffic": 862,
    "PeMS": 307,
    "solar_Alabama": 137,
    "dalia": 5,
    "dynamic": 17,
}
DATASET_SPLITS = {
    "ETTh1": "8640,2880,2880",
    "ETTh2": "8640,2880,2880",
    "ETTm1": "34560,11520,11520",
    "ETTm2": "34560,11520,11520",
    "illness": "0.7,0.1,0.2",
    "exchange_rate": "0.7,0.1,0.2",
    "weather": "0.7,0.1,0.2",
    "electricity": "0.7,0.1,0.2",
    "traffic": "0.7,0.1,0.2",
    "PeMS": "0.7,0.1,0.2",
    "solar_Alabama": "0.7,0.1,0.2",
    "dalia": "0.7,0.1,0.2",
    "dynamic": "0.7,0.1,0.2",
}
DEFAULT_DATASETS = [
    "ETTh1", "ETTh2", "ETTm1", "ETTm2", "illness", "exchange_rate",
    "weather", "electricity", "traffic", "PeMS", "solar_Alabama", "dalia", "dynamic",
]
ANCHOR_VARIANTS = {
    "gaussian": {"slug": "gauss-anchor", "model_name": "gaussian_anchor"},
    "binary": {"slug": "binary", "model_name": "binary_anchor"},
}
@dataclass
class AnchorRun:
    variant: str
    dataset: str
    root: Path
    subset_dir: Path
    best_pt: Optional[Path]
    itrans_pt: Optional[Path]
    metadata: Dict[str, Any]


def run_cmd(
    cmd: Sequence[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    log_path: Optional[Path] = None,
) -> None:
    printable = " ".join(str(x) for x in cmd)
    print(f"[cmd] {printable}")
    if log_path is None:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {printable}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        rc = proc.wait()
        if rc:
            raise subprocess.CalledProcessError(rc, cmd)


def ensure_mmpd_repo(path: Path, update: bool = True) -> str:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        run_cmd(["git", "clone", MMPD_URL, str(path)])
    elif not (path / ".git").exists():
        raise RuntimeError(f"{path} exists but is not a git checkout")
    elif update:
        try:
            run_cmd(["git", "pull", "--ff-only"], cwd=path)
        except subprocess.CalledProcessError:
            print(f"[warn] git pull failed for {path}; using existing checkout")

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        text=True,
    ).strip()
    apply_mmpd_compatibility_patches(path)
    return commit


def apply_mmpd_compatibility_patches(path: Path) -> None:
    """Patch ignored upstream checkout for current local dependencies."""
    tools_py = path / "utils" / "tools.py"
    if tools_py.exists():
        text = tools_py.read_text(encoding="utf-8")
        patched = text.replace("np.Inf", "np.inf")
        if patched != text:
            tools_py.write_text(patched, encoding="utf-8")

    main_py = path / "main_mmpd.py"
    if main_py.exists():
        text = main_py.read_text(encoding="utf-8")
        patched = text.replace(
            "if args.data in data_parser.keys():",
            "if args.data in data_parser.keys() and os.environ.get('MMPD_KEEP_CLI_DATA_ARGS') != '1':",
        )
        dropout_line = (
            "parser.add_argument('--dropout', type=float, default=0.2, help='dropout')"
        )
        if dropout_line in patched and "finetune_layers" not in patched:
            patched = patched.replace(
                dropout_line,
                dropout_line
                + "\nparser.add_argument('--finetune_layers', type=int, default=0, "
                + "help='MaskAE TC depth; 0 uses d_layers')"
                + "\nparser.add_argument('--neighbor_num', type=int, default=0, "
                + "help='MaskAE kNN neighbors; 0 uses min(10, data_dim)')",
            )
        if patched != text:
            main_py.write_text(patched, encoding="utf-8")

    dataset_py = path / "data_provider" / "dataset_mts.py"
    if PATCHED_DATASET_MTS.is_file():
        dataset_py.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PATCHED_DATASET_MTS, dataset_py)

    mask_ae_py = path / "models" / "backbones" / "mask_ae_transformer.py"
    if PATCHED_MASKAE_BACKBONE.is_file():
        mask_ae_py.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PATCHED_MASKAE_BACKBONE, mask_ae_py)

    exp_forecast_py = path / "exp" / "exp_forecast.py"
    if PATCHED_EXP_FORECAST.is_file():
        shutil.copy2(PATCHED_EXP_FORECAST, exp_forecast_py)


def mmpd_staged_filename(dataset: str) -> str:
    return DATASET_FILES[dataset].name


def safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def run_subset_id(run: AnchorRun) -> str:
    return str(run.metadata.get("subset_id") or run.dataset)


def run_variate_indices(run: AnchorRun) -> List[int]:
    return [int(i) for i in run.metadata["variate_indices"]]


def run_data_subset(run: AnchorRun) -> Dict[str, Any]:
    return dict(run.metadata.get("data_subset") or {})


def run_train_stride(run: AnchorRun) -> int:
    subset = run_data_subset(run)
    return max(1, int(subset.get("train_stride", 1)))


def run_test_stride(run: AnchorRun) -> int:
    subset = run_data_subset(run)
    return max(1, int(subset.get("test_stride", run_train_stride(run))))


def mmpd_dataset_name(run: AnchorRun) -> str:
    # Use the binary subset id as the MMPD data name so checkpoints cannot be
    # accidentally reused across different variate/stride subsets.
    return safe_stem(run_subset_id(run))


def mmpd_staged_filename_for_run(run: AnchorRun) -> str:
    return f"{mmpd_dataset_name(run)}.csv"


def _write_mmpd_csv(path: Path, values: np.ndarray, columns: Sequence[str]) -> None:
    """Write MTS-style CSV: date column + one column per variate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_rows, n_cols = values.shape
    if len(columns) != n_cols:
        raise ValueError(f"Expected {n_cols} columns, got {len(columns)}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", *columns])
        for i in range(n_rows):
            writer.writerow([i, *values[i].tolist()])


def _load_pems_npz_array(path: Path) -> np.ndarray:
    raw = np.load(path, allow_pickle=True)
    data = raw["data"]
    if data.ndim == 3:
        data = data[:, :, 0]
    return np.asarray(data, dtype=np.float32)


def _export_csv_variate_subset(src_csv: Path, dst_csv: Path, variate_indices: Sequence[int]) -> None:
    """Stage only the selected value columns, preserving the first date/index column."""
    variate_indices = [int(i) for i in variate_indices]
    with src_csv.open(encoding="utf-8", newline="") as src, dst_csv.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst)
        header = next(reader)
        selected_cols = [i + 1 for i in variate_indices]
        writer.writerow([header[0], *[header[i] for i in selected_cols]])
        for row in reader:
            writer.writerow([row[0], *[row[i] for i in selected_cols]])
    print(
        f"[mmpd-data] {src_csv.name}: wrote {dst_csv} "
        f"({len(variate_indices)} selected variates)"
    )


def _export_pems_mmpd_csv(src_npz: Path, dst_csv: Path, variate_indices: Sequence[int]) -> None:
    values = _load_pems_npz_array(src_npz)
    values = values[:, [int(i) for i in variate_indices]]
    columns = [f"var_{i}" for i in range(values.shape[1])]
    _write_mmpd_csv(dst_csv, values, columns)
    print(f"[mmpd-data] PeMS: wrote {dst_csv} ({values.shape[0]} steps, {values.shape[1]} vars)")


def _dalia_mmpd_meta_path(data_dir: Path) -> Path:
    return data_dir / "dalia_mmpd.meta.json"


def _export_dalia_mmpd_csv(dst_csv: Path, data_dir: Path, variate_indices: Sequence[int]) -> None:
    """Stage DALIA as train|val|test row blocks for block-strided MMPD loading."""
    from models.diffusion_tsf.dalia_data import (
        DALIA_CHANNEL_NAMES,
        load_dalia_tensors,
    )
    from models.diffusion_tsf.dalia_data import _split_sample_indices

    x, y = load_dalia_tensors()
    n = len(x)
    train_idx, val_idx, test_idx = _split_sample_indices(n)
    variate_indices = [int(i) for i in variate_indices]
    names = [DALIA_CHANNEL_NAMES[i] for i in variate_indices]
    x = x[:, :, variate_indices]
    y = y[:, :, variate_indices]

    train_blocks = [np.concatenate([x[i], y[i]], axis=0) for i in train_idx]
    val_blocks = [np.concatenate([x[i], y[i]], axis=0) for i in val_idx]
    test_blocks = [np.concatenate([x[i], y[i]], axis=0) for i in test_idx]
    train_rows = sum(b.shape[0] for b in train_blocks)
    val_rows = sum(b.shape[0] for b in val_blocks)
    test_rows = sum(b.shape[0] for b in test_blocks)
    values = np.concatenate(train_blocks + val_blocks + test_blocks, axis=0)
    _write_mmpd_csv(dst_csv, values, names)
    meta = {
        "data_split": [int(train_rows), int(val_rows), int(test_rows)],
        "n_windows": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "block_len": int(x.shape[1] + y.shape[1]),
    }
    with _dalia_mmpd_meta_path(data_dir).open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(
        f"[mmpd-data] DALIA: wrote {dst_csv} "
        f"({meta['n_windows']}, rows={meta['data_split']}, {values.shape[1]} vars)"
    )


def mmpd_data_split(dataset: str, data_dir: Path) -> str:
    if dataset != "dalia":
        return DATASET_SPLITS[dataset]
    meta_path = _dalia_mmpd_meta_path(data_dir)
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {meta_path}; re-run staging (delete stale dalia_mmpd.csv first)."
        )
    with meta_path.open(encoding="utf-8") as f:
        parts = json.load(f)["data_split"]
    return ",".join(str(int(x)) for x in parts)


def stage_mmpd_dataset_for_run(data_dir: Path, run: AnchorRun) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset = run.dataset
    src = DATASET_FILES[dataset]
    if not src.exists():
        raise FileNotFoundError(f"Missing dataset file for {dataset}: {src}")

    dst = data_dir / mmpd_staged_filename_for_run(run)
    meta_path = dst.with_suffix(dst.suffix + ".meta.json")
    expected_meta = {
        "dataset": dataset,
        "subset_id": run_subset_id(run),
        "variate_indices": run_variate_indices(run),
        "train_stride": run_train_stride(run),
        "test_stride": run_test_stride(run),
    }
    if dst.is_symlink():
        dst.unlink()
    if dst.exists() and meta_path.exists():
        try:
            with meta_path.open(encoding="utf-8") as f:
                if json.load(f) == expected_meta:
                    return
        except Exception:
            pass
    if dst.exists():
        dst.unlink()
    if meta_path.exists():
        meta_path.unlink()

    if dataset == "PeMS":
        _export_pems_mmpd_csv(src, dst, expected_meta["variate_indices"])
    elif dataset == "dalia":
        _export_dalia_mmpd_csv(dst, data_dir, expected_meta["variate_indices"])
        expected_meta["dalia_data_split"] = mmpd_data_split(dataset, data_dir)
    else:
        _export_csv_variate_subset(src, dst, expected_meta["variate_indices"])
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(expected_meta, f, indent=2, sort_keys=True)


def mmpd_train_batch_size(args: argparse.Namespace, dataset: str, data_dim: Optional[int] = None) -> int:
    """Cap batch size for wide datasets to avoid L40S OOM during MMPD training."""
    dim = int(data_dim if data_dim is not None else DATASET_DIMS[dataset])
    cap = args.mmpd_batch_size
    if dim >= 800:
        cap = min(cap, 4)
    elif dim >= 300:
        cap = min(cap, 8)
    elif dim >= 150:
        cap = min(cap, 16)
    if cap < args.mmpd_batch_size:
        print(f"[mmpd] {dataset}: batch_size {args.mmpd_batch_size} -> {cap} (data_dim={dim})")
    return cap


def mmpd_eval_batch_size(args: argparse.Namespace, dataset: str, data_dim: Optional[int] = None) -> int:
    """Cap eval batch; 100-sample diffusion predict is much heavier than train."""
    dim = int(data_dim if data_dim is not None else DATASET_DIMS[dataset])
    cap = args.mmpd_eval_batch_size
    if dim >= 800:
        cap = min(cap, 1)
    elif dim >= 300:
        cap = min(cap, 2)
    elif dim >= 130:
        cap = min(cap, 4)
    elif dim <= 21:
        cap = max(cap, min(32, args.mmpd_eval_batch_size * 2))
    if cap != args.mmpd_eval_batch_size:
        print(f"[mmpd-eval] {dataset}: batch_size {args.mmpd_eval_batch_size} -> {cap} (data_dim={dim})")
    return cap


STAGED_ANCHOR_STAGES = ("finer", "fine", "coarse")


def _anchor_run_from_metadata(
    root: Path,
    meta_path: Path,
    meta: Dict[str, Any],
    variant: str,
) -> Optional[AnchorRun]:
    dataset = meta.get("dataset_name") or meta.get("dataset")
    if not dataset:
        return None
    subset_id = meta.get("subset_id", dataset)
    subset_dir = meta_path.parent
    best_pt = subset_dir / "best.pt"
    itrans_pt = root / f"{subset_id}_itransformer_finetuned.pt"
    if not best_pt.exists() or not itrans_pt.exists():
        return None
    return AnchorRun(
        variant=variant,
        dataset=dataset,
        root=root,
        subset_dir=subset_dir,
        best_pt=best_pt,
        itrans_pt=itrans_pt,
        metadata=meta,
    )


def _anchor_metadata_rank(meta_path: Path) -> int:
    stage = meta_path.parent.name
    if stage not in STAGED_ANCHOR_STAGES:
        return len(STAGED_ANCHOR_STAGES)
    return STAGED_ANCHOR_STAGES.index(stage)


def _load_data_subset_policy(config_path: Path) -> Dict[str, Any]:
    from models.diffusion_tsf.pipeline.config import load_experiment_config

    cfg = load_experiment_config(str(config_path.resolve()))
    policy = cfg.get("experiment", {}).get("data_subset")
    if not policy:
        raise ValueError(f"{config_path} missing experiment.data_subset")
    return dict(policy)


def resolve_subset_meta_for_dataset(
    dataset: str,
    policy: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    from models.diffusion_tsf.pipeline.data_subset import resolve_data_subset
    from models.diffusion_tsf.train_multivariate_pipeline import get_dataset_shape

    raw_rows, raw_variates = get_dataset_shape(dataset)
    target_dataset = policy.get("target_dataset")
    target_rows = target_variates = None
    if target_dataset:
        target_rows, target_variates = get_dataset_shape(str(target_dataset))
    return resolve_data_subset(
        dataset_name=dataset,
        raw_rows=raw_rows,
        raw_variates=raw_variates,
        base_variate_indices=list(range(raw_variates)),
        default_subset_id=None,
        default_window_stride=1,
        seed=seed,
        policy=policy,
        target_rows=target_rows,
        target_variates=target_variates,
    )


def build_anchor_run_from_subset_meta(
    dataset: str,
    subset_meta: Dict[str, Any],
    *,
    variant: str = "binary",
) -> AnchorRun:
    subset_id = str(subset_meta["subset_id"])
    metadata = {
        "dataset_name": dataset,
        "dataset": dataset,
        "subset_id": subset_id,
        "variate_indices": [int(i) for i in subset_meta["variate_indices"]],
        "data_subset": subset_meta,
    }
    stub = REPO_ROOT / "results" / "subset_spec" / dataset
    return AnchorRun(
        variant=variant,
        dataset=dataset,
        root=stub,
        subset_dir=stub / subset_id,
        best_pt=None,
        itrans_pt=None,
        metadata=metadata,
    )


def build_anchor_runs_from_subset_config(
    config_path: Path,
    datasets: Sequence[str],
    seed: int,
) -> Dict[str, AnchorRun]:
    policy = _load_data_subset_policy(config_path)
    runs: Dict[str, AnchorRun] = {}
    for dataset in datasets:
        subset_meta = resolve_subset_meta_for_dataset(dataset, policy, seed)
        runs[dataset] = build_anchor_run_from_subset_meta(dataset, subset_meta)
        print(
            f"[subset-config] {dataset}: {subset_meta['subset_id']} "
            f"variates={subset_meta['n_variates']} "
            f"stride={subset_meta['sample_stride']}",
            flush=True,
        )
    return runs


def _roots_for_anchor_config(
    ckpt_base: Path,
    datasets: Sequence[str],
    anchor_config: str,
) -> List[Path]:
    roots: List[Path] = []
    for ds in datasets:
        matches = sorted(
            [
                p
                for p in ckpt_base.iterdir()
                if p.is_dir() and p.name.endswith(f"-{ds}-{anchor_config}")
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if matches:
            roots.append(matches[0])
    return roots


def find_anchor_runs(
    datasets: Sequence[str],
    explicit_roots: Sequence[Path],
    ckpt_base: Path,
    variant: str,
    anchor_config: Optional[str] = None,
) -> Dict[str, AnchorRun]:
    slug = ANCHOR_VARIANTS[variant]["slug"]
    roots: List[Path]
    if explicit_roots:
        roots = [p.resolve() for p in explicit_roots]
    elif anchor_config:
        roots = _roots_for_anchor_config(ckpt_base, datasets, anchor_config)
    else:
        roots = sorted(
            [p for p in ckpt_base.glob(f"*{slug}*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    found: Dict[str, AnchorRun] = {}
    best_rank: Dict[str, int] = {}
    for root in roots:
        meta_paths = list(root.glob("*/metadata.json"))
        for stage in STAGED_ANCHOR_STAGES:
            meta_paths.extend(root.glob(f"*/{stage}/metadata.json"))
        for meta_path in meta_paths:
            with meta_path.open(encoding="utf-8") as f:
                meta = json.load(f)
            dataset = meta.get("dataset_name") or meta.get("dataset")
            if dataset not in datasets:
                continue
            rank = _anchor_metadata_rank(meta_path)
            if dataset in found and rank <= best_rank[dataset]:
                continue
            run = _anchor_run_from_metadata(root, meta_path, meta, variant)
            if run is None:
                continue
            found[dataset] = run
            best_rank[dataset] = rank
    missing = [d for d in datasets if d not in found]
    if missing:
        searched = ", ".join(str(p) for p in roots) if roots else str(ckpt_base)
        hint = ""
        if anchor_config:
            hint = f" (expected ckpt dirs like *-<dataset>-{anchor_config})"
        raise RuntimeError(
            f"Could not find completed {variant} anchor runs for: "
            + ", ".join(missing)
            + f" under {searched}{hint}"
        )
    return found


def mmpd_setting(
    dataset: str,
    lookback: int,
    horizon: int,
    patch_size: int,
    backbone: str = "Decoder",
    loss_func: str = "MMPD",
    weighted: bool = True,
    point_weight: float = 0.01,
    d_diffusion: int = 256,
    diffusion_layers: int = 1,
    radius: int = 3,
    max_diffusion_steps: int = 1000,
    beta_schedule: str = "linear",
) -> str:
    return (
        f"data{dataset}_il{lookback}_ol{horizon}_backbone{backbone}_loss{loss_func}"
        f"_weighted{weighted}_patch{patch_size}_pointW{point_weight}"
        f"_diffH{d_diffusion}_diffLayer{diffusion_layers}_radius{radius}"
        f"_diffStep{max_diffusion_steps}_beta{beta_schedule}"
    )


def dataset_window_lengths(args: argparse.Namespace, dataset: str) -> Tuple[int, int]:
    """Use repo dataset-specific lengths for prewindowed datasets like DALIA."""
    if dataset == "dalia":
        pipeline = load_tsf_pipeline()
        return pipeline.dataset_window_lengths(dataset)
    return args.lookback, args.horizon


def mmpd_env_for_run(run: AnchorRun) -> Dict[str, str]:
    env = os.environ.copy()
    env["MMPD_KEEP_CLI_DATA_ARGS"] = "1"
    env["MMPD_WINDOW_STRIDE"] = str(run_train_stride(run))
    env["MMPD_TEST_STRIDE"] = str(run_test_stride(run))
    if run.dataset == "dalia":
        env["MMPD_BLOCK_LEN"] = "120"
    else:
        env.pop("MMPD_BLOCK_LEN", None)
    repo = str(REPO_ROOT)
    env["TS_SANDBOX_REPO"] = repo
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo if not prev else f"{repo}:{prev}"
    return env


def dataset_mmpd_patch_size(args: argparse.Namespace, dataset: str) -> int:
    lookback, horizon = dataset_window_lengths(args, dataset)
    patch = resolve_mmpd_patch_size(dataset, horizon, args.patch_size)
    if args.patch_size is None:
        print(f"[mmpd] {dataset}: patch_size={patch} (horizon={horizon})", flush=True)
    return patch


def build_mmpd_train_cmd(
    args: argparse.Namespace,
    run: AnchorRun,
    *,
    hparams: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    train_epochs: Optional[int] = None,
    patience: Optional[int] = None,
) -> List[str]:
    from utils.mmpd_paper_hparams import resolved_mmpd_hparams

    dataset = run.dataset
    data_path = mmpd_staged_filename_for_run(run)
    lookback, horizon = dataset_window_lengths(args, dataset)
    patch_size = dataset_mmpd_patch_size(args, dataset)
    data_dim = len(run_variate_indices(run))
    batch_size = mmpd_train_batch_size(args, dataset, data_dim=data_dim)
    hp = resolved_mmpd_hparams(args.output_dir, dataset, fallback=hparams)
    epochs = int(train_epochs if train_epochs is not None else args.mmpd_train_epochs)
    stop_patience = int(patience if patience is not None else args.mmpd_patience)
    mmpd_out = output_root if output_root is not None else args.output_dir / "mmpd_out"
    cmd = [
        pipeline_python(),
        "-u",
        "main_mmpd.py",
        "--data",
        mmpd_dataset_name(run),
        "--root_path",
        str(args.mmpd_data_dir),
        "--data_path",
        data_path,
        "--data_split",
        mmpd_data_split(dataset, args.mmpd_data_dir),
        "--output_root",
        str(mmpd_out),
        "--backbone",
        args.mmpd_backbone,
        "--loss_func",
        "MMPD",
        "--in_len",
        str(lookback),
        "--out_len",
        str(horizon),
        "--patch_size",
        str(patch_size),
        "--data_dim",
        str(data_dim),
        "--d_layers",
        "2",
        "--d_model",
        "256",
        "--d_ff",
        "512",
        "--n_heads",
        "4",
        "--dropout",
        str(hp["dropout"]),
        "--weighted",
        "True",
        "--point_weight",
        str(hp["point_weight"]),
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
        str(hp["learning_rate"]),
        "--lradj",
        "cosine",
        "--train_epochs",
        str(epochs),
        "--patience",
        str(stop_patience),
        "--training",
        "True",
        "--testing",
        "False",
        "--num_workers",
        str(args.num_workers),
        "--gpu",
        str(args.gpu),
    ]
    if args.mmpd_backbone == "MaskAE":
        cmd.extend(
            [
                "--finetune_layers",
                str(int(hp.get("finetune_layers", 0))),
                "--neighbor_num",
                str(int(hp.get("neighbor_num", 0))),
            ]
        )
    if not torch.cuda.is_available() or args.cpu:
        cmd.extend(["--use_gpu", "False"])
    return cmd


def mmpd_checkpoint_data_names(run: AnchorRun) -> List[str]:
    """MMPD setting prefixes to try (subset id first, then legacy dataset name)."""
    names: List[str] = []
    for name in (mmpd_dataset_name(run), run.dataset):
        if name not in names:
            names.append(name)
    return names


def mmpd_checkpoint_path(
    args: argparse.Namespace,
    run: AnchorRun,
    *,
    data_name: Optional[str] = None,
) -> Path:
    lookback, horizon = dataset_window_lengths(args, run.dataset)
    name = data_name or mmpd_dataset_name(run)
    patch_size = dataset_mmpd_patch_size(args, run.dataset)
    setting = mmpd_setting(name, lookback, horizon, patch_size, backbone=args.mmpd_backbone)
    return (
        mmpd_output_root(args)
        / "mmpd_out"
        / "checkpoints"
        / f"{args.mmpd_backbone}-MMPD"
        / setting
        / "model_checkpoint.pth"
    )


def resolve_mmpd_checkpoint(
    args: argparse.Namespace, run: AnchorRun
) -> Tuple[Path, str]:
    """Return existing checkpoint path and the MMPD `data` name used in its setting dir."""
    for name in mmpd_checkpoint_data_names(run):
        ckpt = mmpd_checkpoint_path(args, run, data_name=name)
        if ckpt.exists():
            if name != mmpd_dataset_name(run):
                print(
                    f"[mmpd] {run.dataset}: using legacy checkpoint data name {name!r} "
                    f"(subset id {mmpd_dataset_name(run)!r})",
                    flush=True,
                )
            return ckpt, name
    return mmpd_checkpoint_path(args, run), mmpd_dataset_name(run)


def train_mmpd(args: argparse.Namespace, runs: Sequence[AnchorRun]) -> None:
    from utils.mmpd_paper_hparams import load_tuned_hparams, tuning_result_path
    from utils.mmpd_subset_tune import tune_mmpd_subset

    for run in runs:
        dataset = run.dataset
        stage_mmpd_dataset_for_run(args.mmpd_data_dir, run)
        if args.mmpd_tune_trials > 0:
            tuned = load_tuned_hparams(args.output_dir, dataset)
            if tuned is None or args.force_mmpd_tune:
                tune_mmpd_subset(args, run)
            else:
                print(f"[mmpd-tune] {dataset}: reusing {tuning_result_path(args.output_dir, dataset)}")
        ckpt, _ = resolve_mmpd_checkpoint(args, run)
        if ckpt.exists() and not args.force_mmpd_train:
            print(f"[mmpd] Reusing checkpoint for {dataset}: {ckpt}")
            continue
        if args.skip_mmpd_train:
            if args.skip_mmpd_eval:
                print(f"[mmpd] Skipping train/eval for {dataset}; checkpoint not required.")
                continue
            raise FileNotFoundError(f"--skip-mmpd-train set but missing {ckpt}")
        log_path = args.output_dir / "logs" / f"mmpd_train_{dataset}.log"
        run_cmd(
            build_mmpd_train_cmd(args, run),
            cwd=args.mmpd_repo,
            env=mmpd_env_for_run(run),
            log_path=log_path,
        )


def write_mmpd_eval_helper(mmpd_repo: Path) -> Path:
    helper = mmpd_repo / "_ts_sandbox_eval_helper.py"
    helper.write_text(
        textwrap.dedent(
            r'''
            import argparse
            import json
            import os
            import pickle
            import random
            import sys
            import time
            from types import SimpleNamespace

            import numpy as np
            import torch
            from einops import rearrange
            from torch.utils.data import DataLoader, Subset

            import importlib.util
            from pathlib import Path as _Path

            _repo = os.environ.get("TS_SANDBOX_REPO") or str(
                _Path(__file__).resolve().parents[2]
            )
            _mmpd = str(_Path(__file__).resolve().parent)
            # MMPD has utils/tools.py; repo must win over script-dir utils.
            for _p in (_repo, _mmpd):
                while _p in sys.path:
                    sys.path.remove(_p)
            sys.path.insert(0, _repo)
            sys.path.insert(1, _mmpd)

            def _load_eval_progress():
                _path = _Path(_repo) / "utils" / "mmpd_eval_progress.py"
                _spec = importlib.util.spec_from_file_location(
                    "_ts_mmpd_eval_progress", _path
                )
                if _spec is None or _spec.loader is None:
                    raise ImportError(f"cannot load {_path}")
                _mod = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                return _mod.EvalProgress, _mod.fmt_duration

            EvalProgress, fmt_duration = _load_eval_progress()

            from data_provider.dataset_mts import Dataset_MTS
            from exp.exp_forecast import Exp_Forecast
            from exp.normalization import get_statistics, normalize, denormalize


            def str2bool(v):
                if isinstance(v, bool):
                    return v
                return str(v).lower() in ("true", "1", "yes", "y")


            def parse_split(value):
                parts = [x.strip() for x in str(value).split(",") if x.strip()]
                parsed = [float(x) for x in parts]
                if parsed and all(x > 1 for x in parsed):
                    return [int(x) for x in parsed]
                return parsed


            def make_args(ns):
                return SimpleNamespace(
                    data=ns.dataset,
                    root_path=ns.root_path,
                    data_path=ns.data_path,
                    data_split=parse_split(ns.data_split),
                    output_root=ns.output_root,
                    backbone=ns.mmpd_backbone,
                    in_len=ns.lookback,
                    out_len=ns.horizon,
                    patch_size=ns.patch_size,
                    data_dim=ns.data_dim,
                    d_model=256,
                    d_ff=512,
                    n_heads=4,
                    e_layers=2,
                    d_layers=2,
                    dropout=0.2,
                    loss_func="MMPD",
                    point_weight=0.01,
                    weighted=True,
                    d_diffusion=256,
                    diffusion_layers=1,
                    max_diffusion_steps=1000,
                    beta_schedule="linear",
                    radius=3,
                    training=False,
                    num_workers=ns.num_workers,
                    batch_size=ns.batch_size,
                    train_epochs=20,
                    patience=5,
                    learning_rate=1e-4,
                    lradj="cosine",
                    test_batch_num=-1,
                    testing=True,
                    prob_pred=True,
                    sample_num=ns.sample_num,
                    num_sampling_steps=str(ns.num_sampling_steps),
                    temperature=1.0,
                    gmm_components=ns.gmm_components,
                    prior_pi_decay=0.5,
                    prior_precision_shape=1e2,
                    gmm_iterations=ns.gmm_iterations,
                    use_gpu=(torch.cuda.is_available() and not ns.cpu),
                    gpu=ns.gpu,
                    use_multi_gpu=False,
                    devices="0,1,2,3",
                )


            def setting(args):
                return (
                    f"data{args.data}_il{args.in_len}_ol{args.out_len}_backbone{args.backbone}"
                    f"_loss{args.loss_func}_weighted{args.weighted}_patch{args.patch_size}"
                    f"_pointW{args.point_weight}_diffH{args.d_diffusion}"
                    f"_diffLayer{args.diffusion_layers}_radius{args.radius}"
                    f"_diffStep{args.max_diffusion_steps}_beta{args.beta_schedule}"
                )


            def load_model(args):
                exp = Exp_Forecast(args)
                ckpt_path = os.path.join(
                    args.output_root,
                    "checkpoints",
                    f"{args.backbone}-{args.loss_func}",
                    setting(args),
                    "model_checkpoint.pth",
                )
                state = torch.load(ckpt_path, map_location="cpu")
                model_state = exp.model.state_dict()
                for k, v in state.items():
                    if "gen_diffusion" not in k:
                        model_state[k] = v
                exp.model.load_state_dict(model_state)
                exp.model.eval()
                return exp


            def main():
                parser = argparse.ArgumentParser()
                parser.add_argument("--dataset", required=True)
                parser.add_argument("--root-path", required=True)
                parser.add_argument("--data-path", required=True)
                parser.add_argument("--data-split", required=True)
                parser.add_argument("--output-root", required=True)
                parser.add_argument("--out-npz", required=True)
                parser.add_argument("--indices-json", required=True)
                parser.add_argument("--lookback", type=int, required=True)
                parser.add_argument("--horizon", type=int, required=True)
                parser.add_argument("--patch-size", type=int, required=True)
                parser.add_argument("--data-dim", type=int, required=True)
                parser.add_argument("--mmpd-backbone", type=str, default="Decoder")
                parser.add_argument("--sample-num", type=int, required=True)
                parser.add_argument("--num-sampling-steps", type=int, required=True)
                parser.add_argument("--gmm-components", type=int, required=True)
                parser.add_argument("--gmm-iterations", type=int, required=True)
                parser.add_argument("--batch-size", type=int, default=16)
                parser.add_argument("--num-workers", type=int, default=0)
                parser.add_argument("--gpu", type=int, default=0)
                parser.add_argument("--cpu", action="store_true")
                ns = parser.parse_args()

                seed = 2024
                torch.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                args = make_args(ns)
                exp = load_model(args)
                device = exp.device

                test_data = Dataset_MTS(
                    root_path=args.root_path,
                    data_path=args.data_path,
                    flag="test",
                    size=[args.in_len, args.out_len],
                    data_split=args.data_split,
                )
                with open(ns.indices_json, "r", encoding="utf-8") as f:
                    indices = json.load(f)
                subset = Subset(test_data, indices)
                loader = DataLoader(
                    subset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    drop_last=False,
                )

                y_true_all = []
                det_all = []
                samples_all = []
                mode_center_all = []
                mode_prob_all = []

                n_batches = len(loader)
                n_windows = len(indices)
                print(
                    f"[mmpd-eval] {ns.dataset}: start "
                    f"windows={n_windows} batches={n_batches} batch_size={args.batch_size} "
                    f"samples={args.sample_num} steps={args.num_sampling_steps}",
                    flush=True,
                )
                progress = EvalProgress(f"mmpd-eval/{ns.dataset}", n_batches)

                with torch.no_grad():
                    for batch_i, (batch_x, batch_y) in enumerate(loader):
                        t_batch = time.time()
                        batch_x = batch_x.float().to(device)
                        batch_y = batch_y.float().to(device)
                        batch_x = rearrange(batch_x, "b l d -> b d l")
                        batch_y = rearrange(batch_y, "b l d -> b d l")

                        x_shift, x_scale = get_statistics(batch_x)
                        normed_x = normalize(batch_x, x_shift, x_scale)
                        det, modes, samples = exp.model.predict(
                            normed_x,
                            prob_pred=True,
                            sample_num=args.sample_num,
                            temperature=args.temperature,
                            gmm=True,
                            gmm_components=args.gmm_components,
                            prior_pi_decay=args.prior_pi_decay,
                            prior_precision_shape=args.prior_precision_shape,
                            gmm_iterations=args.gmm_iterations,
                        )

                        y_true_all.append(batch_y.detach().cpu().numpy())
                        det_all.append(denormalize(det, x_shift, x_scale).detach().cpu().numpy())
                        samples_all.append(denormalize(samples, x_shift, x_scale).detach().cpu().numpy())
                        mode_center_all.append(
                            denormalize(modes["mode_center"], x_shift, x_scale).detach().cpu().numpy()
                        )
                        mode_prob_all.append(modes["mode_prob"].detach().cpu().numpy())
                        done = batch_i + 1
                        progress.maybe_log(
                            done,
                            extra=(
                                f"last_batch={fmt_duration(time.time() - t_batch)} "
                                f"windows~{min(done * args.batch_size, n_windows)}/{n_windows}"
                            ),
                        )

                progress.done(extra=f"writing {ns.out_npz}")
                os.makedirs(os.path.dirname(ns.out_npz), exist_ok=True)
                np.savez_compressed(
                    ns.out_npz,
                    y_true=np.concatenate(y_true_all, axis=0),
                    deterministic=np.concatenate(det_all, axis=0),
                    samples=np.concatenate(samples_all, axis=0),
                    mode_center=np.concatenate(mode_center_all, axis=0),
                    mode_prob=np.concatenate(mode_prob_all, axis=0),
                    indices=np.array(indices, dtype=np.int64),
                )
                print(f"[mmpd-eval] {ns.dataset}: saved {ns.out_npz}", flush=True)


            if __name__ == "__main__":
                main()
            '''
        ).lstrip(),
        encoding="utf-8",
    )
    return helper


def load_tsf_pipeline():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module("models.diffusion_tsf.train_multivariate_pipeline")


def get_ckpt_config_value(ckpt: Dict[str, Any], key: str, default: Any = None) -> Any:
    cfg = ckpt.get("config")
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if isinstance(cfg, dict) and key in cfg:
        return cfg[key]
    return default


def infer_diffusion_type(ckpt: Dict[str, Any], variant: Optional[str] = None) -> str:
    if variant == "binary":
        return "binary"
    if variant == "gaussian":
        return "gaussian"
    value = get_ckpt_config_value(ckpt, "diffusion_type")
    if value:
        return value
    state = ckpt.get("model_state_dict", {})
    for key, tensor in state.items():
        if key.endswith("noise_predictor.head.weight"):
            # DiT heads have out_features = diffusion_channels * patch_h * patch_w.
            # The current checkpoints use 8x8 patches, so Gaussian=64 and binary=128.
            out_features = int(getattr(tensor, "shape", [0])[0])
            if out_features % 128 == 0:
                return "binary"
            if out_features % 64 == 0:
                return "gaussian"
        if key.endswith("noise_predictor.final_conv.weight") and getattr(tensor, "shape", [0])[0] == 2:
            return "binary"
    return "gaussian"


def infer_model_type(ckpt: Dict[str, Any]) -> str:
    value = get_ckpt_config_value(ckpt, "model_type")
    if value:
        return value
    for key in ckpt.get("model_state_dict", {}):
        if "noise_predictor.blocks." in key:
            return "dit"
        if "noise_predictor.down_blocks." in key:
            return "unet"
    return "unet"


def infer_prediction_mode(ckpt: Dict[str, Any]) -> str:
    return get_ckpt_config_value(ckpt, "prediction_mode", "epsilon")


def infer_image_height_from_ckpt(ckpt: Dict[str, Any]) -> Optional[int]:
    """Config may omit image_height; bin_centers length matches occupancy height."""
    value = get_ckpt_config_value(ckpt, "image_height")
    if value is not None:
        return int(value)
    bin_centers = ckpt.get("model_state_dict", {}).get("to_2d.bin_centers")
    if bin_centers is not None:
        return int(bin_centers.shape[0])
    return None


def apply_ckpt_architecture_globals(pipeline: Any, ckpt: Dict[str, Any], diffusion_type: str) -> None:
    """Set pipeline globals before create_diffusion_model so checkpoint shapes match."""
    image_height = infer_image_height_from_ckpt(ckpt)
    if image_height is not None:
        pipeline.IMAGE_HEIGHT = image_height
    pipeline.DISABLE_CROSS_ATTENTION = bool(get_ckpt_config_value(ckpt, "disable_cross_attention", True))
    pipeline.USE_DUAL_SCALE = bool(get_ckpt_config_value(ckpt, "use_dual_scale", False))
    pipeline.DUAL_SCALE_FINE_WEIGHT = float(get_ckpt_config_value(ckpt, "dual_scale_fine_weight", 0.5))
    pipeline.DUAL_SCALE_INDEPENDENT_TIMESTEPS = bool(
        get_ckpt_config_value(ckpt, "dual_scale_independent_timesteps", True)
    )
    pipeline.CROSS_VARIATE_CONTEXT_BIAS = float(get_ckpt_config_value(ckpt, "cross_variate_context_bias", 0.0))
    pipeline.CFG_DROPOUT = float(get_ckpt_config_value(ckpt, "cfg_dropout", 0.1))
    pipeline.CFG_SCALE = float(get_ckpt_config_value(ckpt, "cfg_scale", 1.0))
    pipeline.USE_CFG_INFERENCE = bool(get_ckpt_config_value(ckpt, "use_cfg_inference", False))
    if get_ckpt_config_value(ckpt, "dit_patch_size") is not None:
        pipeline.DIT_PATCH_SIZE = tuple(get_ckpt_config_value(ckpt, "dit_patch_size"))
    if get_ckpt_config_value(ckpt, "dit_embed_dim") is not None:
        pipeline.DIT_EMBED_DIM = int(get_ckpt_config_value(ckpt, "dit_embed_dim"))
    if get_ckpt_config_value(ckpt, "dit_depth") is not None:
        pipeline.DIT_DEPTH = int(get_ckpt_config_value(ckpt, "dit_depth"))
    if get_ckpt_config_value(ckpt, "dit_num_heads") is not None:
        pipeline.DIT_NUM_HEADS = int(get_ckpt_config_value(ckpt, "dit_num_heads"))
    if get_ckpt_config_value(ckpt, "dit_mlp_ratio") is not None:
        pipeline.DIT_MLP_RATIO = float(get_ckpt_config_value(ckpt, "dit_mlp_ratio"))
    if get_ckpt_config_value(ckpt, "dit_dropout") is not None:
        pipeline.DIT_DROPOUT = float(get_ckpt_config_value(ckpt, "dit_dropout"))

    state = ckpt.get("model_state_dict", {})
    head_weight = state.get("noise_predictor.head.weight")
    if head_weight is not None:
        out_features, embed_dim = map(int, head_weight.shape[:2])
        pipeline.DIT_EMBED_DIM = embed_dim
        out_channels = 2 if diffusion_type == "binary" else 1
        patch_area = out_features // out_channels
        patch_side = int(round(math.sqrt(patch_area)))
        if patch_side * patch_side == patch_area:
            pipeline.DIT_PATCH_SIZE = (patch_side, patch_side)


def resolve_inference_cfg(
    ckpt: Dict[str, Any],
    args: argparse.Namespace,
) -> tuple[float, bool]:
    """CFG scale / inference flag: CLI overrides checkpoint when set."""
    from models.diffusion_tsf.cfg_inference import cfg_mix_applies

    cfg_scale = float(get_ckpt_config_value(ckpt, "cfg_scale", 1.0))
    use_cfg = bool(get_ckpt_config_value(ckpt, "use_cfg_inference", False))
    if getattr(args, "cfg_scale", None) is not None:
        cfg_scale = float(args.cfg_scale)
        if getattr(args, "use_cfg_inference", None) is None and cfg_mix_applies(cfg_scale):
            use_cfg = True
    if getattr(args, "use_cfg_inference", None) is not None:
        use_cfg = bool(args.use_cfg_inference)
    return cfg_scale, use_cfg


def load_anchor_model(run: AnchorRun, args: argparse.Namespace, device: torch.device):
    if run.best_pt is None or run.itrans_pt is None:
        raise ValueError(
            f"anchor eval for {run.dataset} requires binary ckpts; "
            "use --anchor-config instead of --subset-config."
        )
    pipeline = load_tsf_pipeline()
    ckpt = torch.load(run.best_pt, map_location=device, weights_only=False)
    n_vars = len(run.metadata["variate_indices"])
    itrans = pipeline.load_itransformer_from_checkpoint(str(run.itrans_pt), n_vars, device)

    diffusion_type = infer_diffusion_type(ckpt, run.variant)
    apply_ckpt_architecture_globals(pipeline, ckpt, diffusion_type)
    cfg_scale, use_cfg = resolve_inference_cfg(ckpt, args)
    pipeline.CFG_SCALE = cfg_scale
    pipeline.USE_CFG_INFERENCE = use_cfg
    tuned = run.metadata.get("tuned_params", {})
    lookback, horizon = dataset_window_lengths(args, run.dataset)
    guidance_mod = importlib.import_module("models.diffusion_tsf.guidance")
    itrans_guidance = guidance_mod.iTransformerGuidance(itrans)
    model = pipeline.create_diffusion_model(
        n_variates=n_vars,
        lookback=lookback,
        horizon=horizon,
        diffusion_type=diffusion_type,
        model_type=infer_model_type(ckpt),
        use_deterministic_anchor_loss=True,
        deterministic_anchor_lambda=float(tuned.get("deterministic_anchor_lambda", 0.99)),
        deterministic_anchor_alpha=float(tuned.get(
            "deterministic_anchor_alpha",
            0.0 if diffusion_type == "binary" else 0.5,
        )),
        cross_variate_context_bias=float(get_ckpt_config_value(ckpt, "cross_variate_context_bias", 0.0)),
        cfg_dropout=float(get_ckpt_config_value(ckpt, "cfg_dropout", 0.1)),
        cfg_scale=cfg_scale,
        use_cfg_inference=use_cfg,
        guidance_model=itrans_guidance,
    ).to(device)
    pipeline.load_diffusion_state_keep_attached_guidance(model, ckpt["model_state_dict"])
    model.eval()
    return model


def make_eval_indices(n: int, fraction: float, seed: int, max_items: Optional[int]) -> List[int]:
    rng = np.random.default_rng(seed)
    count = max(1, int(round(n * fraction)))
    if max_items is not None:
        count = min(count, max_items)
    count = min(count, n)
    return sorted(rng.choice(n, size=count, replace=False).tolist())


def load_tsf_test_subset(
    dataset: str,
    variate_indices: Sequence[int],
    indices: Sequence[int],
    lookback: Optional[int],
    horizon: Optional[int],
    train_stride: int,
    test_stride: int,
):
    pipeline = load_tsf_pipeline()
    _, _, test_ds, _ = pipeline.load_dataset(
        dataset,
        list(variate_indices),
        lookback=lookback,
        horizon=horizon,
        stride=train_stride,
        test_stride=test_stride,
    )
    return Subset(test_ds, list(indices))


def anchor_prob_generate_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    if args.anchor_prob_sampler == "ddpm":
        return {"sampler": "ddpm", "use_ddim": False}
    return {
        "sampler": args.anchor_prob_sampler,
        "num_inference_steps": args.num_sampling_steps,
    }


def evaluate_anchor(
    args: argparse.Namespace,
    run: AnchorRun,
    indices: Sequence[int],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    lookback, horizon = dataset_window_lengths(args, run.dataset)
    subset = load_tsf_test_subset(
        run.dataset,
        run.metadata["variate_indices"],
        indices,
        lookback,
        horizon,
        run_train_stride(run),
        run_test_stride(run),
    )
    loader = DataLoader(
        subset,
        batch_size=args.anchor_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    model = load_anchor_model(run, args, device)
    y_true: List[np.ndarray] = []
    det: List[np.ndarray] = []
    samples_all: List[np.ndarray] = []
    collect_prob_samples = run.variant == "binary" and args.sample_num > 0
    prob_kwargs = anchor_prob_generate_kwargs(args)

    n_batches = len(loader)
    n_windows = len(indices)
    print(
        f"[anchor-eval] {run.dataset}: start "
        f"windows={n_windows} batches={n_batches} batch_size={args.anchor_batch_size} "
        f"mode=deterministic-anchor"
        + (
            f" + probabilistic-{prob_kwargs.get('sampler')} samples={args.sample_num}"
            if collect_prob_samples else ""
        ),
        flush=True,
    )
    batch_progress = EvalProgress(f"anchor-eval/{run.dataset}", n_batches)
    with torch.no_grad():
        for batch_idx, (past, future) in enumerate(loader):
            t_batch = time.time()
            past = past.to(device)
            future = future.to(device)
            K = getattr(model.config, "lookback_overlap", 0)
            if K > 0:
                future = future[..., K:]
            y_true.append(future.cpu().numpy())

            anchor = model.generate(past, sampler="anchor")["prediction"]
            det.append(anchor.cpu().numpy())
            if collect_prob_samples:
                batch_samples = []
                for sample_idx in range(args.sample_num):
                    torch.manual_seed(args.seed + batch_idx * 1009 + sample_idx * 17)
                    sample = model.generate(past, **prob_kwargs)["prediction"]
                    batch_samples.append(sample.detach().cpu())
                samples_all.append(torch.stack(batch_samples, dim=2).numpy())

            done = batch_idx + 1
            batch_progress.maybe_log(
                done,
                extra=(
                    f"last_batch={fmt_duration(time.time() - t_batch)} "
                    f"windows~{min(done * args.anchor_batch_size, n_windows)}/{n_windows}"
                ),
            )

    batch_progress.done()
    out = {
        "y_true": np.concatenate(y_true, axis=0),
        "deterministic": np.concatenate(det, axis=0),
        "indices": np.array(indices, dtype=np.int64),
    }
    if samples_all:
        out["samples"] = np.concatenate(samples_all, axis=0)
    return out


def run_mmpd_eval(
    args: argparse.Namespace,
    run: AnchorRun,
    indices: Sequence[int],
) -> Dict[str, np.ndarray]:
    dataset = run.dataset
    out_npz = args.output_dir / "raw" / f"mmpd_{dataset}.npz"
    indices_json = args.output_dir / "raw" / f"indices_{dataset}_mmpd_eval.json"
    indices_json.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(indices_json, list(indices))

    if not out_npz.exists() or args.force_mmpd_eval:
        stage_mmpd_dataset_for_run(args.mmpd_data_dir, run)
        ckpt_path, mmpd_data = resolve_mmpd_checkpoint(args, run)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"MMPD checkpoint missing for {dataset}: {ckpt_path}")
        helper = write_mmpd_eval_helper(args.mmpd_repo)
        lookback, horizon = dataset_window_lengths(args, dataset)
        data_dim = len(run_variate_indices(run))
        batch_size = mmpd_eval_batch_size(args, dataset, data_dim=data_dim)
        patch_size = dataset_mmpd_patch_size(args, dataset)
        cmd = [
            pipeline_python(),
            "-u",
            str(helper),
            "--dataset",
            mmpd_data,
            "--root-path",
            str(args.mmpd_data_dir),
            "--data-path",
            mmpd_staged_filename_for_run(run),
            "--data-split",
            mmpd_data_split(dataset, args.mmpd_data_dir),
            "--output-root",
            str(mmpd_output_root(args) / "mmpd_out"),
            "--out-npz",
            str(out_npz),
            "--indices-json",
            str(indices_json),
            "--lookback",
            str(lookback),
            "--horizon",
            str(horizon),
            "--patch-size",
            str(patch_size),
            "--data-dim",
            str(data_dim),
            "--mmpd-backbone",
            args.mmpd_backbone,
            "--sample-num",
            str(args.sample_num),
            "--num-sampling-steps",
            str(args.num_sampling_steps),
            "--gmm-components",
            str(args.gmm_components),
            "--gmm-iterations",
            str(args.gmm_iterations),
            "--batch-size",
            str(batch_size),
            "--num-workers",
            str(args.num_workers),
            "--gpu",
            str(args.gpu),
        ]
        if args.cpu:
            cmd.append("--cpu")
        env = mmpd_env_for_run(run)
        print(
            f"[mmpd-eval] {dataset}: launching helper "
            f"(windows={len(indices)}, batch={batch_size}, variates={data_dim}, "
            f"stride={run_test_stride(run)})",
            flush=True,
        )
        run_cmd(
            cmd,
            cwd=args.mmpd_repo,
            env=env,
            log_path=args.output_dir / "logs" / f"mmpd_eval_{dataset}.log",
        )
        print(f"[mmpd-eval] {dataset}: helper finished -> {out_npz}", flush=True)

    with np.load(out_npz) as data:
        return {key: data[key] for key in data.files}


def _as_float(x: np.ndarray) -> float:
    return float(np.asarray(x, dtype=np.float64).mean())


def deterministic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mse": _as_float((y_true - y_pred) ** 2),
        "mae": _as_float(np.abs(y_true - y_pred)),
    }


def crps_gr(y_true: np.ndarray, samples: np.ndarray) -> float:
    # y_true: [B, V, L], samples: [B, V, S, L]
    expected_abs = np.abs(samples - y_true[:, :, None, :]).mean(axis=2)
    sample_count = samples.shape[2]
    total = np.zeros_like(y_true, dtype=np.float64)
    chunk = max(1, 256 // max(1, sample_count))
    for start in range(0, samples.shape[0], chunk):
        end = min(samples.shape[0], start + chunk)
        s = samples[start:end].astype(np.float64)
        total[start:end] = np.abs(s[:, :, :, None, :] - s[:, :, None, :, :]).mean(axis=(2, 3))
    return _as_float(expected_abs - 0.5 * total)


def topk_from_modes(
    y_true: np.ndarray,
    mode_center: np.ndarray,
    mode_prob: np.ndarray,
    max_k: int = 5,
) -> Dict[str, float]:
    # y_true [B,V,L], centers [B,V,M,L], probs [B,V,M]
    order = np.argsort(-mode_prob, axis=2)
    out: Dict[str, float] = {}
    max_k = min(max_k, mode_center.shape[2])
    for k in sorted({1, max_k}):
        if k < 1 or k > max_k:
            continue
        gathered = np.take_along_axis(mode_center, order[:, :, :k, None], axis=2)
        mse = ((gathered - y_true[:, :, None, :]) ** 2).mean(axis=-1).min(axis=2)
        mae = np.abs(gathered - y_true[:, :, None, :]).mean(axis=-1).min(axis=2)
        out[f"top{k}_mse"] = _as_float(mse)
        out[f"top{k}_mae"] = _as_float(mae)
    return out


def empirical_modes_from_samples(
    samples: np.ndarray,
    max_components: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster probabilistic samples into modes for top-k metrics."""
    from sklearn.mixture import GaussianMixture

    batch_size, n_variates, sample_count, horizon = samples.shape
    mode_count = min(max_components, sample_count)
    centers = np.zeros((batch_size, n_variates, mode_count, horizon), dtype=np.float64)
    probs = np.zeros((batch_size, n_variates, mode_count), dtype=np.float64)

    for b in range(batch_size):
        for v in range(n_variates):
            trajectories = samples[b, v]
            if sample_count == 1:
                centers[b, v, 0] = trajectories[0]
                probs[b, v, 0] = 1.0
                continue
            n_comp = min(mode_count, sample_count)
            gmm = GaussianMixture(
                n_components=n_comp,
                random_state=seed + b * 131 + v,
                covariance_type="diag",
                reg_covar=1e-4,
                max_iter=50,
            )
            try:
                gmm.fit(trajectories)
                centers[b, v, :n_comp] = gmm.means_
                weights = gmm.weights_
                probs[b, v, :n_comp] = weights / weights.sum()
            except ValueError:
                centers[b, v, :n_comp] = trajectories[:n_comp]
                probs[b, v, :n_comp] = 1.0 / n_comp
    return centers, probs


def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + 1e-8)


def ordinal_jsd(a: np.ndarray, b: np.ndarray, order: int = 4) -> float:
    from itertools import permutations

    perms = list(permutations(range(order)))
    lookup = {p: i for i, p in enumerate(perms)}

    def dist(x: np.ndarray) -> np.ndarray:
        counts = np.zeros(len(perms), dtype=np.float64)
        if len(x) < order:
            counts += 1.0
        else:
            for i in range(len(x) - order + 1):
                ranks = tuple(np.argsort(np.argsort(x[i : i + order], kind="mergesort"), kind="mergesort"))
                counts[lookup[ranks]] += 1.0
        counts += 1e-12
        return counts / counts.sum()

    p = dist(a)
    q = dist(b)
    m = 0.5 * (p + q)
    jsd = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    return float(jsd)


def _line_lengths_bool(arr: np.ndarray) -> List[int]:
    lengths: List[int] = []
    run = 0
    for value in arr:
        if value:
            run += 1
        elif run:
            lengths.append(run)
            run = 0
    if run:
        lengths.append(run)
    return lengths


def rqa_features(x: np.ndarray, eps: float = 0.2, min_len: int = 2) -> np.ndarray:
    x = zscore_1d(x)
    R = np.abs(x[:, None] - x[None, :]) < eps
    np.fill_diagonal(R, False)
    recurrence = R.sum() + 1e-8

    diag_points = 0
    for offset in range(-len(x) + 1, len(x)):
        if offset == 0:
            continue
        for length in _line_lengths_bool(np.diagonal(R, offset=offset)):
            if length >= min_len:
                diag_points += length

    vert_points = 0
    for col in range(R.shape[1]):
        for length in _line_lengths_bool(R[:, col]):
            if length >= min_len:
                vert_points += length

    det = diag_points / recurrence
    lam = vert_points / recurrence
    return np.array([lam, det], dtype=np.float64)


def variogram(x: np.ndarray, max_lag: int = 24) -> np.ndarray:
    x = zscore_1d(x)
    lags = []
    for lag in range(1, min(max_lag, len(x) - 1) + 1):
        diff = x[lag:] - x[:-lag]
        lags.append(0.5 * np.mean(diff * diff))
    return np.asarray(lags, dtype=np.float64)


def _fallback_signature_features(path: np.ndarray) -> np.ndarray:
    t = path[:, 0]
    x = path[:, 1]
    dx = np.diff(x)
    dt = np.diff(t)
    if hasattr(np, "trapezoid"):
        area = np.trapezoid(x, t)
    else:
        area = np.sum((x[1:] + x[:-1]) * 0.5 * dt) if len(dt) else 0.0
    return np.array(
        [
            x[-1] - x[0],
            np.sum(np.abs(dx)),
            np.mean(dx) if len(dx) else 0.0,
            np.std(dx) if len(dx) else 0.0,
            area,
            np.sum(dt * dx) if len(dx) else 0.0,
        ],
        dtype=np.float64,
    )


def path_signature_distance(a: np.ndarray, b: np.ndarray, window: int = 12, depth: int = 3) -> float:
    try:
        import iisignature  # type: ignore
    except Exception:
        iisignature = None

    a = zscore_1d(a)
    b = zscore_1d(b)
    distances = []
    for start in range(0, len(a) - window + 1, window):
        aa = a[start : start + window]
        bb = b[start : start + window]
        t = np.linspace(0.0, 1.0, window)
        pa = np.column_stack([t, aa])
        pb = np.column_stack([t, bb])
        if iisignature is not None:
            fa = np.asarray(iisignature.sig(pa, depth), dtype=np.float64)
            fb = np.asarray(iisignature.sig(pb, depth), dtype=np.float64)
        else:
            fa = _fallback_signature_features(pa)
            fb = _fallback_signature_features(pb)
        distances.append(np.linalg.norm(fa - fb) / math.sqrt(max(1, fa.size)))
    if not distances:
        return 0.0
    return float(np.mean(distances))


def texture_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from models.diffusion_tsf.metrics import texture_metrics as shared_texture_metrics

    return shared_texture_metrics(y_true, y_pred)


def texture_metrics_per_sample(
    y_true: np.ndarray,
    samples: np.ndarray,
    max_draws: int = 3,
) -> Dict[str, float]:
    """Texture metrics on the first few probabilistic draws, then mean."""
    sample_count = min(samples.shape[2], max_draws)
    if sample_count == 0:
        return {}
    per_draw: Dict[str, List[float]] = {}
    for draw_idx in range(sample_count):
        draw_metrics = texture_metrics(y_true, samples[:, :, draw_idx, :])
        for key, value in draw_metrics.items():
            per_draw.setdefault(key, []).append(value)
    return {
        f"prob_{key}": float(np.mean(values))
        for key, values in per_draw.items()
    }


def summarize_prob_core_metrics(
    pack: Dict[str, np.ndarray],
    gmm_components: int = 10,
    seed: int = 0,
    topk_max: int = 3,
) -> Dict[str, float]:
    """Probabilistic metrics only: MSE/MAE of mean-of-samples, CRPS, top-k modes."""
    y_true = pack["y_true"]
    samples = pack["samples"]
    sample_mean = samples.mean(axis=2)
    metrics: Dict[str, float] = {
        "mse": deterministic_metrics(y_true, sample_mean)["mse"],
        "mae": deterministic_metrics(y_true, sample_mean)["mae"],
        "crps": crps_gr(y_true, samples),
        "n_windows": float(y_true.shape[0]),
        "n_variates": float(y_true.shape[1]),
        "n_samples": float(samples.shape[2]),
    }
    mode_center = pack.get("mode_center")
    mode_prob = pack.get("mode_prob")
    if mode_center is None or mode_prob is None:
        mode_center, mode_prob = empirical_modes_from_samples(
            samples,
            max_components=gmm_components,
            seed=seed,
        )
    metrics.update(topk_from_modes(y_true, mode_center, mode_prob, max_k=topk_max))
    return metrics


def summarize_deterministic_pack(pack: Dict[str, np.ndarray], include_texture: bool = True) -> Dict[str, float]:
    y_true = pack["y_true"]
    det = pack["deterministic"]
    metrics: Dict[str, float] = deterministic_metrics(y_true, det)
    metrics["n_windows"] = float(y_true.shape[0])
    metrics["n_variates"] = float(y_true.shape[1])
    metrics["n_samples"] = 1.0
    if include_texture:
        metrics.update(texture_metrics(y_true, det))
    return metrics


def summarize_prediction_pack(
    pack: Dict[str, np.ndarray],
    mode_center: Optional[np.ndarray] = None,
    mode_prob: Optional[np.ndarray] = None,
    gmm_components: int = 10,
    seed: int = 0,
    topk_max: int = 5,
    include_texture: bool = True,
    texture_per_sample: bool = False,
) -> Dict[str, float]:
    y_true = pack["y_true"]
    det = pack["deterministic"]
    samples = pack["samples"]

    metrics: Dict[str, float] = {}
    metrics.update(deterministic_metrics(y_true, det))
    metrics["crps"] = crps_gr(y_true, samples)
    if mode_center is None or mode_prob is None:
        mode_center, mode_prob = pack.get("mode_center"), pack.get("mode_prob")
    if mode_center is None or mode_prob is None:
        mode_center, mode_prob = empirical_modes_from_samples(
            samples,
            max_components=gmm_components,
            seed=seed,
        )
    metrics.update(topk_from_modes(y_true, mode_center, mode_prob, max_k=topk_max))
    if include_texture:
        metrics.update(texture_metrics(y_true, det))
    if include_texture and texture_per_sample:
        metrics.update(texture_metrics_per_sample(y_true, samples))
    metrics["n_windows"] = float(y_true.shape[0])
    metrics["n_variates"] = float(y_true.shape[1])
    metrics["det_n_samples"] = 1.0
    metrics["prob_n_samples"] = float(samples.shape[2])
    metrics["n_samples"] = metrics["prob_n_samples"]
    return metrics


def mmpd_output_root(args: argparse.Namespace) -> Path:
    return (args.mmpd_output_root or args.output_dir).resolve()


def indices_root(args: argparse.Namespace) -> Path:
    return (args.indices_dir or args.output_dir).resolve()


def indices_path(indices_root_dir: Path, dataset: str) -> Path:
    return indices_root_dir / "raw" / f"indices_{dataset}.json"


def summarize_for_profile(
    pack: Dict[str, np.ndarray],
    args: argparse.Namespace,
    dataset: str,
) -> Dict[str, float]:
    seed = stable_dataset_seed(args.seed, dataset)
    if "samples" not in pack:
        return summarize_deterministic_pack(
            pack,
            include_texture=args.metrics_profile != "prob-core",
        )
    if args.metrics_profile == "prob-core":
        return summarize_prob_core_metrics(
            pack,
            gmm_components=args.gmm_components,
            seed=seed,
            topk_max=args.topk_max,
        )
    return summarize_prediction_pack(
        pack,
        gmm_components=args.gmm_components,
        seed=seed,
        topk_max=args.topk_max,
        texture_per_sample=args.texture_per_sample,
    )


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def save_indices(indices_root_dir: Path, dataset: str, indices: Sequence[int]) -> None:
    path = indices_path(indices_root_dir, dataset)
    write_json_atomic(path, list(indices))


def load_indices(indices_root_dir: Path, dataset: str) -> List[int]:
    path = indices_path(indices_root_dir, dataset)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing indices file {path}; run --phase init or a worker job first."
        )
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_indices_for_dataset(
    args: argparse.Namespace,
    run: AnchorRun,
) -> List[int]:
    dataset = run.dataset
    pipeline = load_tsf_pipeline()
    lookback, horizon = dataset_window_lengths(args, dataset)
    _, _, test_ds, _ = pipeline.load_dataset(
        dataset,
        run_variate_indices(run),
        lookback=lookback,
        horizon=horizon,
        stride=run_train_stride(run),
        test_stride=run_test_stride(run),
    )
    indices = make_eval_indices(
        len(test_ds),
        args.test_fraction,
        stable_dataset_seed(args.seed, dataset),
        args.test_max_items,
    )
    print(
        f"[subset] {dataset}: {len(indices)}/{len(test_ds)} test windows "
        f"(variates={len(run_variate_indices(run))}, test_stride={run_test_stride(run)})"
    )
    return indices


def get_or_create_indices(
    args: argparse.Namespace,
    run: AnchorRun,
) -> List[int]:
    dataset = run.dataset
    root = indices_root(args)
    path = indices_path(root, dataset)
    if args.indices_dir and not path.exists():
        raise FileNotFoundError(
            f"--indices-dir {args.indices_dir} missing {path}; run matrix init first."
        )
    if path.exists() and not args.force_indices:
        indices = load_indices(root, dataset)
        print(f"[subset] {dataset}: reusing {len(indices)} indices from {path}")
        return indices
    indices = build_indices_for_dataset(args, run)
    save_indices(root, dataset, indices)
    return indices


def partial_metrics_path(output_dir: Path, dataset: str, model: str) -> Path:
    return output_dir / "partials" / f"{dataset}_{model}.json"


def write_partial_metrics(
    output_dir: Path,
    dataset: str,
    model: str,
    metrics: Dict[str, float],
) -> None:
    path = partial_metrics_path(output_dir, dataset, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(f"[partial] {dataset}/{model} -> {path}")


def load_partial_metrics(output_dir: Path, dataset: str, model: str) -> Optional[Dict[str, float]]:
    path = partial_metrics_path(output_dir, dataset, model)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def collect_results_from_partials(
    output_dir: Path,
    datasets: Sequence[str],
    *,
    mmpd_only: bool = False,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    model_names = ["mmpd"] if mmpd_only else ["mmpd", "binary_anchor"]
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    missing: List[str] = []
    for dataset in datasets:
        results[dataset] = {}
        for model in model_names:
            metrics = load_partial_metrics(output_dir, dataset, model)
            if metrics is None:
                missing.append(f"{dataset}/{model}")
            else:
                results[dataset][model] = metrics
    if missing:
        raise FileNotFoundError(
            "Missing partial metrics (workers may still be running): "
            + ", ".join(missing)
        )
    return results


def discover_anchors_by_variant(
    args: argparse.Namespace,
    datasets: Sequence[str],
) -> Dict[str, Dict[str, AnchorRun]]:
    if args.subset_config is not None:
        return {
            "binary": build_anchor_runs_from_subset_config(
                args.subset_config,
                datasets,
                args.seed,
            ),
        }
    if not args.anchor_config and not args.binary_anchor_root:
        raise ValueError(
            "Set --subset-config (recommended) or --anchor-config / --binary-anchor-root."
        )
    return {
        "binary": find_anchor_runs(
            datasets,
            args.binary_anchor_root,
            args.ckpt_base,
            "binary",
            anchor_config=args.anchor_config,
        ),
    }


def anchors_to_manifest(anchors_by_variant: Dict[str, Dict[str, AnchorRun]]) -> Dict[str, Any]:
    return {
        variant: {
            d: {
                "root": str(r.root),
                "best_pt": str(r.best_pt) if r.best_pt else None,
                "itrans_pt": str(r.itrans_pt) if r.itrans_pt else None,
                "metadata": r.metadata,
            }
            for d, r in anchors.items()
        }
        for variant, anchors in anchors_by_variant.items()
    }


def run_phase_init(args: argparse.Namespace, commit: str) -> None:
    anchors = discover_anchors_by_variant(args, args.datasets)
    indices_by_dataset: Dict[str, List[int]] = {}
    for dataset in args.datasets:
        run = anchors["binary"][dataset]
        stage_mmpd_dataset_for_run(args.mmpd_data_dir, run)
        indices_by_dataset[dataset] = get_or_create_indices(args, run)

    manifest = {
        "args": jsonable_args(args),
        "mmpd_commit": commit,
        "anchor_runs": anchors_to_manifest(anchors),
        "indices_by_dataset": indices_by_dataset,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"[init] Wrote {args.output_dir / 'run_manifest.json'}")


def run_phase_mmpd(
    args: argparse.Namespace,
    dataset: str,
    anchors_by_variant: Dict[str, Dict[str, AnchorRun]],
) -> None:
    binary_run = anchors_by_variant["binary"][dataset]
    indices = get_or_create_indices(args, binary_run)
    if not args.skip_mmpd_train:
        train_mmpd(args, [binary_run])
    elif not args.skip_mmpd_eval:
        stage_mmpd_dataset_for_run(args.mmpd_data_dir, binary_run)
        ckpt, _ = resolve_mmpd_checkpoint(args, binary_run)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"--skip-mmpd-train but missing MMPD checkpoint: {ckpt}"
            )
    if args.skip_mmpd_eval:
        return
    print(f"[mmpd] {dataset}: eval phase ({len(indices)} windows)", flush=True)
    mmpd_pack = run_mmpd_eval(args, binary_run, indices)
    print(f"[mmpd] {dataset}: summarizing metrics", flush=True)
    metrics = summarize_for_profile(mmpd_pack, args, dataset)
    write_partial_metrics(args.output_dir, dataset, "mmpd", metrics)


def run_phase_anchor(
    args: argparse.Namespace,
    dataset: str,
    variant: str,
    anchors_by_variant: Dict[str, Dict[str, AnchorRun]],
    device: torch.device,
) -> None:
    if variant not in ANCHOR_VARIANTS:
        raise ValueError(f"Unknown anchor variant: {variant}")
    run = anchors_by_variant[variant][dataset]
    model_name = ANCHOR_VARIANTS[variant]["model_name"]
    indices = get_or_create_indices(args, run)
    print(f"[anchor] {dataset}: eval phase ({len(indices)} windows)", flush=True)
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    anchor_raw_path = raw_dir / f"{variant}_anchor_{dataset}.npz"
    if anchor_raw_path.exists() and not args.force_anchor_eval:
        with np.load(anchor_raw_path) as data:
            anchor_pack = {key: data[key] for key in data.files}
    else:
        anchor_pack = evaluate_anchor(args, run, indices, device)
        np.savez_compressed(anchor_raw_path, **anchor_pack)
    metrics = summarize_for_profile(anchor_pack, args, dataset)
    write_partial_metrics(args.output_dir, dataset, model_name, metrics)


def run_phase_merge(args: argparse.Namespace, commit: str) -> None:
    manifest_path = args.output_dir / "run_manifest.json"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        anchors = discover_anchors_by_variant(args, args.datasets)
        manifest = {
            "args": jsonable_args(args),
            "mmpd_commit": commit,
            "anchor_runs": anchors_to_manifest(anchors),
            "indices_by_dataset": {},
        }

    results = collect_results_from_partials(
        args.output_dir,
        args.datasets,
        mmpd_only=args.mmpd_only,
    )
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_outputs(args, manifest, results)
    print_summary(results, profile=args.metrics_profile)


def write_outputs(
    args: argparse.Namespace,
    manifest: Dict[str, Any],
    results: Dict[str, Dict[str, Dict[str, float]]],
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    rows = []
    keys = set()
    for dataset, by_model in results.items():
        for model, metrics in by_model.items():
            row = {"dataset": dataset, "model": model}
            row.update(metrics)
            rows.append(row)
            keys.update(metrics)
    fieldnames = ["dataset", "model"] + sorted(keys)
    with (args.output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def jsonable_args(args: argparse.Namespace) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, list):
            out[key] = [str(v) if isinstance(v, Path) else v for v in value]
        else:
            out[key] = value
    return out


def stable_dataset_seed(base_seed: int, dataset: str) -> int:
    return base_seed + sum((i + 1) * ord(ch) for i, ch in enumerate(dataset))


def print_summary(results: Dict[str, Dict[str, Dict[str, float]]], profile: str = "full") -> None:
    print("\nSummary")
    if profile == "prob-core":
        print("dataset,model,mse,mae,crps,top1_mse,top1_mae,top3_mse,top3_mae,n_samples")
        for dataset in sorted(results):
            for model in sorted(results[dataset]):
                m = results[dataset][model]
                print(
                    f"{dataset},{model},"
                    f"{m.get('mse', float('nan')):.6f},"
                    f"{m.get('mae', float('nan')):.6f},"
                    f"{m.get('crps', float('nan')):.6f},"
                    f"{m.get('top1_mse', float('nan')):.6f},"
                    f"{m.get('top1_mae', float('nan')):.6f},"
                    f"{m.get('top3_mse', float('nan')):.6f},"
                    f"{m.get('top3_mae', float('nan')):.6f},"
                    f"{m.get('n_samples', float('nan')):.0f}"
                )
        return
    print(
        "dataset,model,mse,mae,crps,top1_mse,top1_mae,top3_mse,top3_mae,"
        "texture_increment_wasserstein,texture_curvature_wasserstein,"
        "texture_haar_detail_jsd,texture_jump_plateau_distance,"
        "texture_derivative_motif_jsd"
    )
    for dataset in sorted(results):
        for model in sorted(results[dataset]):
            m = results[dataset][model]
            print(
                f"{dataset},{model},"
                f"{m.get('mse', float('nan')):.6f},"
                f"{m.get('mae', float('nan')):.6f},"
                f"{m.get('crps', float('nan')):.6f},"
                f"{m.get('top1_mse', float('nan')):.6f},"
                f"{m.get('top1_mae', float('nan')):.6f},"
                f"{m.get('top3_mse', float('nan')):.6f},"
                f"{m.get('top3_mae', float('nan')):.6f},"
                f"{m.get('texture_increment_wasserstein', float('nan')):.6f},"
                f"{m.get('texture_curvature_wasserstein', float('nan')):.6f},"
                f"{m.get('texture_haar_detail_jsd', float('nan')):.6f},"
                f"{m.get('texture_jump_plateau_distance', float('nan')):.6f},"
                f"{m.get('texture_derivative_motif_jsd', float('nan')):.6f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--anchor-root", action="append", type=Path, default=[],
                        help="Legacy alias for --gaussian-anchor-root")
    parser.add_argument("--gaussian-anchor-root", action="append", type=Path, default=[])
    parser.add_argument("--binary-anchor-root", action="append", type=Path, default=[])
    parser.add_argument(
        "--subset-config",
        type=Path,
        default=None,
        help="YAML with experiment.data_subset; resolve variates/stride per dataset (no binary ckpts).",
    )
    parser.add_argument(
        "--anchor-config",
        type=str,
        default=None,
        help="Legacy: resolve binary ckpts as <ckpt-base>/*-<dataset>-<anchor-config>.",
    )
    parser.add_argument(
        "--mmpd-only",
        action="store_true",
        help="Train/eval MMPD only; merge expects mmpd partials (no binary_anchor).",
    )
    parser.add_argument("--ckpt-base", type=Path, default=REPO_ROOT / "results" / "ckpts")
    parser.add_argument("--mmpd-repo", type=Path, default=DEFAULT_MMPD_REPO)
    parser.add_argument("--mmpd-data-dir", type=Path, default=DEFAULT_MMPD_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--indices-dir",
        type=Path,
        default=None,
        help="Reuse raw/indices_{dataset}.json from a prior matrix run.",
    )
    parser.add_argument(
        "--mmpd-output-root",
        type=Path,
        default=None,
        help="Directory containing mmpd_out/ (default: --output-dir).",
    )
    parser.add_argument(
        "--metrics-profile",
        choices=["full", "prob-core"],
        default="full",
        help="prob-core: mean-of-samples MSE/MAE, CRPS, top-k only (no texture/det).",
    )
    parser.add_argument(
        "--texture-per-sample",
        action="store_true",
        help="Full profile only: texture on the first 3 probabilistic draws, averaged (prob_texture_*).",
    )
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument(
        "--patch-size",
        type=int,
        default=None,
        help="Override MMPD patch size; default picks per dataset/horizon (paper D.3).",
    )
    parser.add_argument(
        "--mmpd-backbone",
        choices=MMPD_BACKBONES,
        default="Decoder",
        help="Upstream MMPD backbone (MaskAE = UP2ME-style masked autoencoder).",
    )
    parser.add_argument("--test-fraction", type=float, default=0.5)
    parser.add_argument("--test-max-items", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sample-num", type=int, default=9)
    parser.add_argument("--num-sampling-steps", type=int, default=20)
    parser.add_argument("--gmm-components", type=int, default=10)
    parser.add_argument("--gmm-iterations", type=int, default=10)
    parser.add_argument("--mmpd-train-epochs", type=int, default=20)
    parser.add_argument("--mmpd-patience", type=int, default=5)
    parser.add_argument("--mmpd-tune-trials", type=int, default=0,
                        help="Optuna trials per dataset before final MMPD train (0=off).")
    parser.add_argument("--mmpd-tune-epochs", type=int, default=10,
                        help="Max epochs per tune trial (shorter than final train).")
    parser.add_argument("--mmpd-tune-patience", type=int, default=3,
                        help="Early-stop patience during tune trials.")
    parser.add_argument("--force-mmpd-tune", action="store_true",
                        help="Re-run Optuna even if tuning/<dataset>_best.json exists.")
    parser.add_argument("--mmpd-batch-size", type=int, default=32)
    parser.add_argument("--mmpd-eval-batch-size", type=int, default=16)
    parser.add_argument("--anchor-batch-size", type=int, default=16)
    parser.add_argument("--anchor-prob-sampler", choices=["dpmpp", "ddim", "ddpm"], default="dpmpp")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip-mmpd-train", action="store_true")
    parser.add_argument("--skip-mmpd-eval", action="store_true")
    parser.add_argument("--force-mmpd-train", action="store_true")
    parser.add_argument("--force-mmpd-eval", action="store_true")
    parser.add_argument("--force-anchor-eval", action="store_true")
    parser.add_argument("--force-indices", action="store_true",
                        help="Recompute shared test indices even if cached on disk.")
    parser.add_argument(
        "--phase",
        choices=["all", "init", "mmpd", "anchor", "merge"],
        default="all",
        help="all=serial end-to-end; init=indices+manifest; mmpd/anchor=one worker; merge=aggregate partials.",
    )
    parser.add_argument(
        "--anchor-variant",
        choices=sorted(ANCHOR_VARIANTS),
        default=None,
        help="Required when --phase anchor.",
    )
    parser.add_argument("--topk-max", type=int, default=3,
                        help="Report top1 and topK only; default K=3 (top2 is intentionally omitted).")
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=None,
        help="Override checkpoint cfg_scale at inference (CFG ablation sweeps).",
    )
    parser.add_argument(
        "--use-cfg-inference",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override checkpoint use_cfg_inference (default: on when cfg_scale != 1).",
    )
    parser.add_argument("--no-update-mmpd", action="store_true")
    return parser.parse_args()


def validate_phase_args(args: argparse.Namespace) -> None:
    if args.phase in ("mmpd", "anchor") and len(args.datasets) != 1:
        raise ValueError(f"--phase {args.phase} requires exactly one --datasets entry")
    if args.phase == "anchor" and args.anchor_variant is None:
        raise ValueError("--phase anchor requires --anchor-variant binary")


def run_phase_all(args: argparse.Namespace, commit: str) -> None:
    anchors_by_variant = discover_anchors_by_variant(args, args.datasets)
    binary_runs = [anchors_by_variant["binary"][dataset] for dataset in args.datasets]
    train_mmpd(args, binary_runs)

    indices_by_dataset: Dict[str, List[int]] = {}
    for dataset in args.datasets:
        run = anchors_by_variant["binary"][dataset]
        indices_by_dataset[dataset] = get_or_create_indices(args, run)

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}"
    )
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for dataset in args.datasets:
        results[dataset] = {}
        indices = indices_by_dataset[dataset]

        if not args.skip_mmpd_eval:
            mmpd_pack = run_mmpd_eval(args, anchors_by_variant["binary"][dataset], indices)
            results[dataset]["mmpd"] = summarize_for_profile(mmpd_pack, args, dataset)

        for variant, anchors in anchors_by_variant.items():
            raw_dir = args.output_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            anchor_raw_path = raw_dir / f"{variant}_anchor_{dataset}.npz"
            if anchor_raw_path.exists() and not args.force_anchor_eval:
                with np.load(anchor_raw_path) as data:
                    anchor_pack = {key: data[key] for key in data.files}
            else:
                anchor_pack = evaluate_anchor(args, anchors[dataset], indices, device)
                np.savez_compressed(anchor_raw_path, **anchor_pack)
            results[dataset][ANCHOR_VARIANTS[variant]["model_name"]] = summarize_for_profile(
                anchor_pack, args, dataset
            )

    manifest = {
        "args": jsonable_args(args),
        "mmpd_commit": commit,
        "anchor_runs": anchors_to_manifest(anchors_by_variant),
        "indices_by_dataset": indices_by_dataset,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_outputs(args, manifest, results)
    print_summary(results, profile=args.metrics_profile)
    print(f"\nWrote metrics to {args.output_dir / 'metrics.json'}")
    print(f"Wrote CSV to {args.output_dir / 'metrics.csv'}")


def main() -> None:
    args = parse_args()
    args.datasets = list(dict.fromkeys(args.datasets))
    unknown = sorted(set(args.datasets) - set(DATASET_FILES))
    if unknown:
        raise ValueError(f"Unsupported dataset(s): {unknown}")
    validate_phase_args(args)
    args.output_dir = args.output_dir.resolve()
    args.mmpd_repo = args.mmpd_repo.resolve()
    args.mmpd_data_dir = args.mmpd_data_dir.resolve()
    args.ckpt_base = args.ckpt_base.resolve()
    if args.indices_dir is not None:
        args.indices_dir = args.indices_dir.resolve()
    if args.mmpd_output_root is not None:
        args.mmpd_output_root = args.mmpd_output_root.resolve()
    if args.subset_config is not None:
        args.subset_config = args.subset_config.resolve()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit = ensure_mmpd_repo(args.mmpd_repo, update=not args.no_update_mmpd)

    if args.phase == "all":
        run_phase_all(args, commit)
        return

    if args.phase == "init":
        run_phase_init(args, commit)
        return

    if args.phase == "merge":
        run_phase_merge(args, commit)
        return

    anchors_by_variant = discover_anchors_by_variant(args, args.datasets)
    dataset = args.datasets[0]

    if args.phase == "mmpd":
        run_phase_mmpd(args, dataset, anchors_by_variant)
        return

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}"
    )
    run_phase_anchor(args, dataset, args.anchor_variant, anchors_by_variant, device)


if __name__ == "__main__":
    main()
