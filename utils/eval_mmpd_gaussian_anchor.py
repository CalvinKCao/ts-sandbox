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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_MMPD_REPO = REPO_ROOT / "temp" / "MMPD"
DEFAULT_MMPD_DATA = REPO_ROOT / "temp" / "mmpd_datasets"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "mmpd_anchor_eval"
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
}
DEFAULT_DATASETS = [
    "ETTh1", "ETTh2", "ETTm1", "ETTm2", "illness", "exchange_rate",
    "weather", "electricity", "traffic", "PeMS", "solar_Alabama", "dalia",
]
ANCHOR_VARIANTS = {
    "gaussian": {"slug": "gauss-anchor", "model_name": "gaussian_anchor"},
    "binary": {"slug": "binary", "model_name": "binary_anchor"},
}
# Upstream MMPD only reads CSV; NPZ / prewindowed DALIA need conversion when staged.
MMPD_STAGED_FILENAMES = {
    "PeMS": "PeMS04.csv",
    "dalia": "dalia_mmpd.csv",
}


@dataclass
class AnchorRun:
    variant: str
    dataset: str
    root: Path
    subset_dir: Path
    best_pt: Path
    itrans_pt: Path
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


def mmpd_staged_filename(dataset: str) -> str:
    return MMPD_STAGED_FILENAMES.get(dataset, DATASET_FILES[dataset].name)


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


def _export_pems_mmpd_csv(src_npz: Path, dst_csv: Path) -> None:
    values = _load_pems_npz_array(src_npz)
    columns = [f"var_{i}" for i in range(values.shape[1])]
    _write_mmpd_csv(dst_csv, values, columns)
    print(f"[mmpd-data] PeMS: wrote {dst_csv} ({values.shape[0]} steps, {values.shape[1]} vars)")


def _export_dalia_mmpd_csv(dst_csv: Path) -> None:
    from models.diffusion_tsf.dalia_data import (
        DALIA_CHANNEL_NAMES,
        load_dalia_tensors,
    )

    x, y = load_dalia_tensors()
    windows = np.concatenate([x, y], axis=1)
    values = windows.reshape(-1, windows.shape[-1])
    _write_mmpd_csv(dst_csv, values, DALIA_CHANNEL_NAMES)
    print(
        f"[mmpd-data] DALIA: wrote {dst_csv} "
        f"({len(x)} windows x {windows.shape[1]} steps, {values.shape[1]} vars)"
    )


def stage_mmpd_datasets(data_dir: Path, datasets: Sequence[str]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        src = DATASET_FILES[dataset]
        if not src.exists():
            raise FileNotFoundError(f"Missing dataset file for {dataset}: {src}")
        dst = data_dir / mmpd_staged_filename(dataset)

        if dataset == "PeMS":
            if dst.is_symlink():
                dst.unlink()
            if not dst.exists() or dst.stat().st_size == 0:
                _export_pems_mmpd_csv(src, dst)
            continue

        if dataset == "dalia":
            if dst.is_symlink():
                dst.unlink()
            if not dst.exists() or dst.stat().st_size == 0:
                _export_dalia_mmpd_csv(dst)
            continue

        if dst.exists() or dst.is_symlink():
            continue
        try:
            dst.symlink_to(src)
        except OSError:
            shutil.copy2(src, dst)


def mmpd_train_batch_size(args: argparse.Namespace, dataset: str) -> int:
    """Cap batch size for wide datasets to avoid L40S OOM during MMPD training."""
    dim = DATASET_DIMS[dataset]
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


def find_anchor_runs(
    datasets: Sequence[str],
    explicit_roots: Sequence[Path],
    ckpt_base: Path,
    variant: str,
) -> Dict[str, AnchorRun]:
    slug = ANCHOR_VARIANTS[variant]["slug"]
    roots: List[Path]
    if explicit_roots:
        roots = [p.resolve() for p in explicit_roots]
    else:
        roots = sorted(
            [p for p in ckpt_base.glob(f"*{slug}*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    found: Dict[str, AnchorRun] = {}
    for root in roots:
        for meta_path in root.glob("*/metadata.json"):
            with meta_path.open(encoding="utf-8") as f:
                meta = json.load(f)
            dataset = meta.get("dataset_name") or meta.get("dataset")
            if dataset not in datasets or dataset in found:
                continue
            subset_id = meta.get("subset_id", dataset)
            subset_dir = meta_path.parent
            best_pt = subset_dir / "best.pt"
            itrans_pt = root / f"{subset_id}_itransformer_finetuned.pt"
            if best_pt.exists() and itrans_pt.exists():
                found[dataset] = AnchorRun(
                    variant=variant,
                    dataset=dataset,
                    root=root,
                    subset_dir=subset_dir,
                    best_pt=best_pt,
                    itrans_pt=itrans_pt,
                    metadata=meta,
                )
    missing = [d for d in datasets if d not in found]
    if missing:
        raise RuntimeError(
            f"Could not find completed {variant} anchor runs for: "
            + ", ".join(missing)
            + f" under {ckpt_base}"
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


def build_mmpd_train_cmd(args: argparse.Namespace, dataset: str) -> List[str]:
    data_path = mmpd_staged_filename(dataset)
    lookback, horizon = dataset_window_lengths(args, dataset)
    batch_size = mmpd_train_batch_size(args, dataset)
    cmd = [
        sys.executable,
        "-u",
        "main_mmpd.py",
        "--data",
        dataset,
        "--root_path",
        str(args.mmpd_data_dir),
        "--data_path",
        data_path,
        "--data_split",
        DATASET_SPLITS[dataset],
        "--output_root",
        str(args.output_dir / "mmpd_out"),
        "--backbone",
        "Decoder",
        "--loss_func",
        "MMPD",
        "--in_len",
        str(lookback),
        "--out_len",
        str(horizon),
        "--patch_size",
        str(args.patch_size),
        "--data_dim",
        str(DATASET_DIMS[dataset]),
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
        str(args.mmpd_train_epochs),
        "--patience",
        str(args.mmpd_patience),
        "--training",
        "True",
        "--testing",
        "False",
        "--num_workers",
        str(args.num_workers),
        "--gpu",
        str(args.gpu),
    ]
    if not torch.cuda.is_available() or args.cpu:
        cmd.extend(["--use_gpu", "False"])
    return cmd


def mmpd_checkpoint_path(args: argparse.Namespace, dataset: str) -> Path:
    lookback, horizon = dataset_window_lengths(args, dataset)
    setting = mmpd_setting(dataset, lookback, horizon, args.patch_size)
    return (
        mmpd_output_root(args)
        / "mmpd_out"
        / "checkpoints"
        / "Decoder-MMPD"
        / setting
        / "model_checkpoint.pth"
    )


def train_mmpd(args: argparse.Namespace, datasets: Sequence[str]) -> None:
    for dataset in datasets:
        ckpt = mmpd_checkpoint_path(args, dataset)
        if ckpt.exists() and not args.force_mmpd_train:
            print(f"[mmpd] Reusing checkpoint for {dataset}: {ckpt}")
            continue
        if args.skip_mmpd_train:
            if args.skip_mmpd_eval:
                print(f"[mmpd] Skipping train/eval for {dataset}; checkpoint not required.")
                continue
            raise FileNotFoundError(f"--skip-mmpd-train set but missing {ckpt}")
        log_path = args.output_dir / "logs" / f"mmpd_train_{dataset}.log"
        run_cmd(build_mmpd_train_cmd(args, dataset), cwd=args.mmpd_repo, log_path=log_path)


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
            from types import SimpleNamespace

            import numpy as np
            import torch
            from einops import rearrange
            from torch.utils.data import DataLoader, Subset

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
                    backbone="Decoder",
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

                with torch.no_grad():
                    for batch_x, batch_y in loader:
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


def load_anchor_model(run: AnchorRun, args: argparse.Namespace, device: torch.device):
    pipeline = load_tsf_pipeline()
    ckpt = torch.load(run.best_pt, map_location=device, weights_only=False)
    n_vars = len(run.metadata["variate_indices"])
    itrans = pipeline.load_itransformer_from_checkpoint(str(run.itrans_pt), n_vars, device)

    diffusion_type = infer_diffusion_type(ckpt, run.variant)
    apply_ckpt_architecture_globals(pipeline, ckpt, diffusion_type)
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
        cfg_scale=float(get_ckpt_config_value(ckpt, "cfg_scale", 1.0)),
        use_cfg_inference=bool(get_ckpt_config_value(ckpt, "use_cfg_inference", False)),
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
):
    pipeline = load_tsf_pipeline()
    _, _, test_ds, _ = pipeline.load_dataset(
        dataset,
        list(variate_indices),
        lookback=lookback,
        horizon=horizon,
        stride=1,
    )
    return Subset(test_ds, list(indices))


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
    samples: List[np.ndarray] = []
    sample_kwargs = {
        "sampler": args.anchor_prob_sampler,
        "num_inference_steps": args.num_sampling_steps,
    }
    if args.anchor_prob_sampler == "ddpm":
        sample_kwargs = {"sampler": "ddpm", "use_ddim": False}

    with torch.no_grad():
        for batch_idx, (past, future) in enumerate(loader):
            past = past.to(device)
            future = future.to(device)
            K = getattr(model.config, "lookback_overlap", 0)
            if K > 0:
                future = future[..., K:]
            y_true.append(future.cpu().numpy())

            anchor = model.generate(past, sampler="anchor")["prediction"]
            det.append(anchor.cpu().numpy())

            batch_samples = []
            for sample_idx in range(args.sample_num):
                torch.manual_seed(args.seed + batch_idx * 1009 + sample_idx * 17)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(args.seed + batch_idx * 1009 + sample_idx * 17)
                pred = model.generate(past, **sample_kwargs)["prediction"]
                batch_samples.append(pred.cpu().numpy())
            samples.append(np.stack(batch_samples, axis=2))

    return {
        "y_true": np.concatenate(y_true, axis=0),
        "deterministic": np.concatenate(det, axis=0),
        "samples": np.concatenate(samples, axis=0),
        "indices": np.array(indices, dtype=np.int64),
    }


def run_mmpd_eval(
    args: argparse.Namespace,
    dataset: str,
    indices: Sequence[int],
) -> Dict[str, np.ndarray]:
    out_npz = args.output_dir / "raw" / f"mmpd_{dataset}.npz"
    indices_json = args.output_dir / "raw" / f"indices_{dataset}.json"
    indices_json.parent.mkdir(parents=True, exist_ok=True)
    with indices_json.open("w", encoding="utf-8") as f:
        json.dump(list(indices), f)

    if not out_npz.exists() or args.force_mmpd_eval:
        helper = write_mmpd_eval_helper(args.mmpd_repo)
        lookback, horizon = dataset_window_lengths(args, dataset)
        cmd = [
            sys.executable,
            "-u",
            str(helper),
            "--dataset",
            dataset,
            "--root-path",
            str(args.mmpd_data_dir),
            "--data-path",
            mmpd_staged_filename(dataset),
            "--data-split",
            DATASET_SPLITS[dataset],
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
            str(args.patch_size),
            "--data-dim",
            str(DATASET_DIMS[dataset]),
            "--sample-num",
            str(args.sample_num),
            "--num-sampling-steps",
            str(args.num_sampling_steps),
            "--gmm-components",
            str(args.gmm_components),
            "--gmm-iterations",
            str(args.gmm_iterations),
            "--batch-size",
            str(args.mmpd_eval_batch_size),
            "--num-workers",
            str(args.num_workers),
            "--gpu",
            str(args.gpu),
        ]
        if args.cpu:
            cmd.append("--cpu")
        run_cmd(cmd, cwd=args.mmpd_repo, log_path=args.output_dir / "logs" / f"mmpd_eval_{dataset}.log")

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
    for k in range(1, max_k + 1):
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
    vals = {
        "texture_ordinal_jsd": [],
        "texture_rqa_distance": [],
        "texture_variogram_distance": [],
        "texture_pathsig_distance": [],
    }
    flat_true = y_true.reshape(-1, y_true.shape[-1])
    flat_pred = y_pred.reshape(-1, y_pred.shape[-1])
    for gt, pred in zip(flat_true, flat_pred):
        gt_z = zscore_1d(gt)
        pred_z = zscore_1d(pred)
        vals["texture_ordinal_jsd"].append(ordinal_jsd(gt_z, pred_z))
        vals["texture_rqa_distance"].append(float(np.linalg.norm(rqa_features(gt_z) - rqa_features(pred_z))))
        va = variogram(gt_z)
        vb = variogram(pred_z)
        vals["texture_variogram_distance"].append(float(np.linalg.norm(va - vb) / math.sqrt(max(1, va.size))))
        vals["texture_pathsig_distance"].append(path_signature_distance(gt_z, pred_z))
    return {key: float(np.mean(value)) for key, value in vals.items()}


def texture_metrics_per_sample(
    y_true: np.ndarray,
    samples: np.ndarray,
) -> Dict[str, float]:
    """Texture metrics on each probabilistic draw, then mean over draws."""
    sample_count = samples.shape[2]
    if sample_count == 0:
        return {}
    per_draw: Dict[str, List[float]] = {}
    for draw_idx in range(sample_count):
        draw_metrics = texture_metrics(y_true, samples[:, :, draw_idx, :])
        for key, value in draw_metrics.items():
            per_draw.setdefault(key, []).append(value)
    return {
        f"per_sample_mean_{key}": float(np.mean(values))
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


def summarize_prediction_pack(
    pack: Dict[str, np.ndarray],
    mode_center: Optional[np.ndarray] = None,
    mode_prob: Optional[np.ndarray] = None,
    gmm_components: int = 10,
    seed: int = 0,
    topk_max: int = 5,
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
    metrics.update(texture_metrics(y_true, det))
    sample_mean = samples.mean(axis=2)
    for key, value in texture_metrics(y_true, sample_mean).items():
        metrics[f"sample_mean_{key}"] = value
    if texture_per_sample:
        metrics.update(texture_metrics_per_sample(y_true, samples))
    metrics["n_windows"] = float(y_true.shape[0])
    metrics["n_variates"] = float(y_true.shape[1])
    metrics["n_samples"] = float(samples.shape[2])
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


def save_indices(indices_root_dir: Path, dataset: str, indices: Sequence[int]) -> None:
    path = indices_path(indices_root_dir, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(indices), f)


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
    dataset: str,
    variate_indices: Sequence[int],
) -> List[int]:
    pipeline = load_tsf_pipeline()
    lookback, horizon = dataset_window_lengths(args, dataset)
    _, _, test_ds, _ = pipeline.load_dataset(
        dataset,
        list(variate_indices),
        lookback=lookback,
        horizon=horizon,
        stride=1,
    )
    indices = make_eval_indices(
        len(test_ds),
        args.test_fraction,
        stable_dataset_seed(args.seed, dataset),
        args.test_max_items,
    )
    print(
        f"[subset] {dataset}: {len(indices)}/{len(test_ds)} test windows"
    )
    return indices


def get_or_create_indices(
    args: argparse.Namespace,
    dataset: str,
    variate_indices: Sequence[int],
) -> List[int]:
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
    indices = build_indices_for_dataset(args, dataset, variate_indices)
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
) -> Dict[str, Dict[str, Dict[str, float]]]:
    model_names = ["mmpd", "binary_anchor"]
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
    return {
        "binary": find_anchor_runs(datasets, args.binary_anchor_root, args.ckpt_base, "binary"),
    }


def anchors_to_manifest(anchors_by_variant: Dict[str, Dict[str, AnchorRun]]) -> Dict[str, Any]:
    return {
        variant: {
            d: {
                "root": str(r.root),
                "best_pt": str(r.best_pt),
                "itrans_pt": str(r.itrans_pt),
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
        variates = run.metadata["variate_indices"]
        indices_by_dataset[dataset] = get_or_create_indices(args, dataset, variates)

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
    indices = get_or_create_indices(args, dataset, binary_run.metadata["variate_indices"])
    if not args.skip_mmpd_train:
        train_mmpd(args, [dataset])
    elif not args.skip_mmpd_eval:
        ckpt = mmpd_checkpoint_path(args, dataset)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"--skip-mmpd-train but missing MMPD checkpoint: {ckpt}"
            )
    if args.skip_mmpd_eval:
        return
    mmpd_pack = run_mmpd_eval(args, dataset, indices)
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
    indices = get_or_create_indices(args, dataset, run.metadata["variate_indices"])
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

    results = collect_results_from_partials(args.output_dir, args.datasets)
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
        print("dataset,model,mse,mae,crps,top3_mse,top3_mae,n_samples")
        for dataset in sorted(results):
            for model in sorted(results[dataset]):
                m = results[dataset][model]
                print(
                    f"{dataset},{model},"
                    f"{m.get('mse', float('nan')):.6f},"
                    f"{m.get('mae', float('nan')):.6f},"
                    f"{m.get('crps', float('nan')):.6f},"
                    f"{m.get('top3_mse', float('nan')):.6f},"
                    f"{m.get('top3_mae', float('nan')):.6f},"
                    f"{m.get('n_samples', float('nan')):.0f}"
                )
        return
    print("dataset,model,mse,mae,crps,top3_mse,top3_mae,texture_pathsig_distance")
    for dataset in sorted(results):
        for model in sorted(results[dataset]):
            m = results[dataset][model]
            print(
                f"{dataset},{model},"
                f"{m.get('mse', float('nan')):.6f},"
                f"{m.get('mae', float('nan')):.6f},"
                f"{m.get('crps', float('nan')):.6f},"
                f"{m.get('top3_mse', float('nan')):.6f},"
                f"{m.get('top3_mae', float('nan')):.6f},"
                f"{m.get('texture_pathsig_distance', float('nan')):.6f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--anchor-root", action="append", type=Path, default=[],
                        help="Legacy alias for --gaussian-anchor-root")
    parser.add_argument("--gaussian-anchor-root", action="append", type=Path, default=[])
    parser.add_argument("--binary-anchor-root", action="append", type=Path, default=[])
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
        help="Full profile only: texture metrics on each draw, averaged (per_sample_mean_*).",
    )
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--patch-size", type=int, default=12)
    parser.add_argument("--test-fraction", type=float, default=0.5)
    parser.add_argument("--test-max-items", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sample-num", type=int, default=9)
    parser.add_argument("--num-sampling-steps", type=int, default=20)
    parser.add_argument("--gmm-components", type=int, default=10)
    parser.add_argument("--gmm-iterations", type=int, default=10)
    parser.add_argument("--mmpd-train-epochs", type=int, default=20)
    parser.add_argument("--mmpd-patience", type=int, default=5)
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
    parser.add_argument("--topk-max", type=int, default=3)
    parser.add_argument("--no-update-mmpd", action="store_true")
    return parser.parse_args()


def validate_phase_args(args: argparse.Namespace) -> None:
    if args.phase in ("mmpd", "anchor") and len(args.datasets) != 1:
        raise ValueError(f"--phase {args.phase} requires exactly one --datasets entry")
    if args.phase == "anchor" and args.anchor_variant is None:
        raise ValueError("--phase anchor requires --anchor-variant binary")


def run_phase_all(args: argparse.Namespace, commit: str) -> None:
    anchors_by_variant = discover_anchors_by_variant(args, args.datasets)
    train_mmpd(args, args.datasets)

    indices_by_dataset: Dict[str, List[int]] = {}
    for dataset in args.datasets:
        run = anchors_by_variant["binary"][dataset]
        indices_by_dataset[dataset] = get_or_create_indices(
            args, dataset, run.metadata["variate_indices"]
        )

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}"
    )
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for dataset in args.datasets:
        results[dataset] = {}
        indices = indices_by_dataset[dataset]

        if not args.skip_mmpd_eval:
            mmpd_pack = run_mmpd_eval(args, dataset, indices)
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit = ensure_mmpd_repo(args.mmpd_repo, update=not args.no_update_mmpd)
    stage_mmpd_datasets(args.mmpd_data_dir, args.datasets)

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
