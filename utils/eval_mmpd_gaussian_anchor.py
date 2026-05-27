#!/usr/bin/env python3
"""MMPD vs ts-sandbox anchor evaluation (Gaussian and binary+anchor).

Trains upstream MMPD per dataset, then scores MMPD and anchor checkpoints on a
shared random test subset. Supports phased runs for parallel Slurm fan-out,
shared MMPD caches (e.g. reuse ETTh1/2/exchange from a prior job), and binary
diffusion anchor runs from the 92d3 matrix.
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
DEFAULT_MMPD_REPO = REPO_ROOT / "temp" / "MMPD"
DEFAULT_MMPD_DATA = REPO_ROOT / "temp" / "mmpd_datasets"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "mmpd_anchor_eval"
MMPD_URL = "https://github.com/Thinklab-SJTU/MMPD.git"

_ETTM_TRAIN = 12 * 30 * 24 * 4
_ETTM_VAL = 4 * 30 * 24 * 4
_ETTM_TEST = 4 * 30 * 24 * 4

DATASET_FILES = {
    "ETTh1": REPO_ROOT / "datasets" / "ETT-small" / "ETTh1.csv",
    "ETTh2": REPO_ROOT / "datasets" / "ETT-small" / "ETTh2.csv",
    "ETTm1": REPO_ROOT / "datasets" / "ETT-small" / "ETTm1.csv",
    "ETTm2": REPO_ROOT / "datasets" / "ETT-small" / "ETTm2.csv",
    "exchange_rate": REPO_ROOT / "datasets" / "exchange_rate" / "exchange_rate.csv",
    "illness": REPO_ROOT / "datasets" / "illness" / "national_illness.csv",
}
DATASET_DIMS = {
    "ETTh1": 7,
    "ETTh2": 7,
    "ETTm1": 7,
    "ETTm2": 7,
    "exchange_rate": 8,
    "illness": 7,
}
DATASET_SPLITS = {
    "ETTh1": "8640,2880,2880",
    "ETTh2": "8640,2880,2880",
    "ETTm1": f"{_ETTM_TRAIN},{_ETTM_VAL},{_ETTM_TEST}",
    "ETTm2": f"{_ETTM_TRAIN},{_ETTM_VAL},{_ETTM_TEST}",
    "exchange_rate": "0.7,0.1,0.2",
    "illness": "0.7,0.1,0.2",
}

ANCHOR_VARIANTS = ("gaussian", "binary")
ANCHOR_CKPT_GLOBS = {
    "gaussian": ("*gauss-anchor*", "*gaussian-anchor*"),
    "binary": ("*binary-anchor*",),
}
MODEL_KEYS = {
    "gaussian": "gaussian_anchor",
    "binary": "binary_anchor",
}


@dataclass
class AnchorRun:
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


def stage_mmpd_datasets(data_dir: Path, datasets: Sequence[str]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        src = DATASET_FILES[dataset]
        if not src.exists():
            raise FileNotFoundError(f"Missing dataset CSV for {dataset}: {src}")
        dst = data_dir / src.name
        if dst.exists() or dst.is_symlink():
            continue
        try:
            dst.symlink_to(src)
        except OSError:
            shutil.copy2(src, dst)


def discover_anchor_roots(
    variant: str,
    explicit_roots: Sequence[Path],
    ckpt_base: Path,
) -> List[Path]:
    if explicit_roots:
        return [p.resolve() for p in explicit_roots]
    roots: List[Path] = []
    for pattern in ANCHOR_CKPT_GLOBS[variant]:
        roots.extend(p for p in ckpt_base.glob(pattern) if p.is_dir())
    return sorted(roots, key=lambda p: p.stat().st_mtime, reverse=True)


def find_anchor_runs(
    datasets: Sequence[str],
    explicit_roots: Sequence[Path],
    ckpt_base: Path,
    variant: str = "gaussian",
) -> Dict[str, AnchorRun]:
    roots = discover_anchor_roots(variant, explicit_roots, ckpt_base)
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
            + f" under {ckpt_base} (explicit roots: {len(explicit_roots)})"
        )
    return found


def resolve_storage_paths(args: argparse.Namespace) -> None:
    if args.mmpd_output_root is None:
        args.mmpd_output_root = args.output_dir / "mmpd_out"
    else:
        args.mmpd_output_root = args.mmpd_output_root.resolve()
    if args.mmpd_raw_dir is None:
        args.mmpd_raw_dir = args.output_dir / "raw"
    else:
        args.mmpd_raw_dir = args.mmpd_raw_dir.resolve()
    if args.reuse_anchor_raw_from is not None:
        args.reuse_anchor_raw_from = args.reuse_anchor_raw_from.resolve()
    if args.mmpd_raw_fallback is not None:
        args.mmpd_raw_fallback = args.mmpd_raw_fallback.resolve()


def mmpd_raw_path(args: argparse.Namespace, dataset: str) -> Path:
    return args.mmpd_raw_dir / f"mmpd_{dataset}.npz"


def resolve_mmpd_raw_read_path(args: argparse.Namespace, dataset: str) -> Path:
    primary = mmpd_raw_path(args, dataset)
    if primary.exists():
        return primary
    if args.mmpd_raw_fallback is not None:
        fallback = args.mmpd_raw_fallback / f"mmpd_{dataset}.npz"
        if fallback.exists():
            return fallback
    return primary


def indices_path(args: argparse.Namespace, dataset: str) -> Path:
    return args.mmpd_raw_dir / f"indices_{dataset}.json"


def anchor_raw_path(
    args: argparse.Namespace,
    variant: str,
    dataset: str,
) -> Path:
    primary = args.mmpd_raw_dir / f"anchor_{variant}_{dataset}.npz"
    if primary.exists():
        return primary
    if variant != "gaussian":
        return primary
    legacy = args.mmpd_raw_dir / f"anchor_{dataset}.npz"
    if legacy.exists():
        return legacy
    if args.reuse_anchor_raw_from is not None:
        for name in (f"anchor_{variant}_{dataset}.npz", f"anchor_{dataset}.npz"):
            candidate = args.reuse_anchor_raw_from / name
            if candidate.exists():
                return candidate
    return primary


def partial_metrics_path(args: argparse.Namespace, dataset: str, model_key: str) -> Path:
    return args.output_dir / "metrics_partial" / f"{dataset}__{model_key}.json"


def write_partial_metrics(
    args: argparse.Namespace,
    dataset: str,
    model_key: str,
    metrics: Dict[str, float],
) -> None:
    path = partial_metrics_path(args, dataset, model_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(f"[partial] {path}")


def load_partial_metrics(args: argparse.Namespace) -> Dict[str, Dict[str, Dict[str, float]]]:
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    partial_dir = args.output_dir / "metrics_partial"
    if not partial_dir.exists():
        return results
    for path in sorted(partial_dir.glob("*.json")):
        stem = path.stem
        if "__" not in stem:
            continue
        dataset, model_key = stem.split("__", 1)
        with path.open(encoding="utf-8") as f:
            metrics = json.load(f)
        results.setdefault(dataset, {})[model_key] = metrics
    return results


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


def build_mmpd_train_cmd(args: argparse.Namespace, dataset: str) -> List[str]:
    data_path = DATASET_FILES[dataset].name
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
        str(args.mmpd_output_root),
        "--backbone",
        "Decoder",
        "--loss_func",
        "MMPD",
        "--in_len",
        str(args.lookback),
        "--out_len",
        str(args.horizon),
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
        str(args.mmpd_batch_size),
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
    setting = mmpd_setting(dataset, args.lookback, args.horizon, args.patch_size)
    return (
        args.mmpd_output_root
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


def infer_diffusion_type(ckpt: Dict[str, Any]) -> str:
    value = get_ckpt_config_value(ckpt, "diffusion_type")
    if value:
        return value
    for key, tensor in ckpt.get("model_state_dict", {}).items():
        if key.endswith("noise_predictor.head.bias") and getattr(tensor, "shape", [0])[0] == 2:
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


def load_anchor_model(run: AnchorRun, args: argparse.Namespace, device: torch.device):
    pipeline = load_tsf_pipeline()
    ckpt = torch.load(run.best_pt, map_location=device, weights_only=False)
    n_vars = len(run.metadata["variate_indices"])
    itrans = pipeline.load_itransformer_from_checkpoint(str(run.itrans_pt), n_vars, device)

    cfg_disable = get_ckpt_config_value(ckpt, "disable_cross_attention", True)
    pipeline.DISABLE_CROSS_ATTENTION = bool(cfg_disable)
    tuned = run.metadata.get("tuned_params", {})
    model = pipeline.create_diffusion_model(
        n_variates=n_vars,
        lookback=args.lookback,
        horizon=args.horizon,
        diffusion_type=infer_diffusion_type(ckpt),
        prediction_mode=infer_prediction_mode(ckpt),
        model_type=infer_model_type(ckpt),
        use_deterministic_anchor_loss=True,
        deterministic_anchor_lambda=float(tuned.get("deterministic_anchor_lambda", 0.99)),
        deterministic_anchor_alpha=float(tuned.get("deterministic_anchor_alpha", 0.5)),
    ).to(device)
    guidance_mod = importlib.import_module("models.diffusion_tsf.guidance")
    model.set_guidance_model(guidance_mod.iTransformerGuidance(itrans))
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
    lookback: int,
    horizon: int,
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


def anchor_prob_sample_kwargs(model: Any, args: argparse.Namespace) -> Dict[str, Any]:
    """Match training eval: binary+anchor uses one-step anchor; Gaussian uses DPM++."""
    if getattr(model.config, "diffusion_type", None) == "binary":
        return {"sampler": "anchor"}
    kwargs: Dict[str, Any] = {"sampler": args.anchor_prob_sampler}
    if args.anchor_prob_sampler != "ddpm":
        kwargs["num_inference_steps"] = args.num_sampling_steps
    return kwargs


def evaluate_anchor(
    args: argparse.Namespace,
    run: AnchorRun,
    indices: Sequence[int],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    subset = load_tsf_test_subset(
        run.dataset,
        run.metadata["variate_indices"],
        indices,
        args.lookback,
        args.horizon,
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
    sample_kwargs = anchor_prob_sample_kwargs(model, args)
    if getattr(model.config, "diffusion_type", None) == "binary":
        n_draws = args.sample_num
    elif sample_kwargs.get("sampler") == "anchor":
        n_draws = 1
    else:
        n_draws = args.sample_num

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
            for sample_idx in range(n_draws):
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
    out_npz = mmpd_raw_path(args, dataset)
    indices_json = indices_path(args, dataset)
    indices_json.parent.mkdir(parents=True, exist_ok=True)
    with indices_json.open("w", encoding="utf-8") as f:
        json.dump(list(indices), f)

    if not out_npz.exists() or args.force_mmpd_eval:
        helper = write_mmpd_eval_helper(args.mmpd_repo)
        cmd = [
            sys.executable,
            "-u",
            str(helper),
            "--dataset",
            dataset,
            "--root-path",
            str(args.mmpd_data_dir),
            "--data-path",
            DATASET_FILES[dataset].name,
            "--data-split",
            DATASET_SPLITS[dataset],
            "--output-root",
            str(args.mmpd_output_root),
            "--out-npz",
            str(out_npz),
            "--indices-json",
            str(indices_json),
            "--lookback",
            str(args.lookback),
            "--horizon",
            str(args.horizon),
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


def _fit_trajectory_gmm(
    trajectories: np.ndarray,
    max_components: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a small GMM on diffusion samples; fall back when samples collapse."""
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.mixture import GaussianMixture

    trajectories = np.asarray(trajectories, dtype=np.float64)
    sample_count, _horizon = trajectories.shape
    if sample_count == 1:
        return trajectories[:1], np.array([1.0], dtype=np.float64)

    rounded = np.round(trajectories, decimals=5)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    unique_traj = trajectories[unique_idx]
    n_unique = len(unique_traj)
    if n_unique == 1:
        return unique_traj, np.array([1.0], dtype=np.float64)

    n_comp = min(max_components, sample_count, n_unique)
    fit_data = unique_traj if n_unique < sample_count else trajectories

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for attempt in range(5):
            reg = 1e-6 * (10.0**attempt)
            try:
                gmm = GaussianMixture(
                    n_components=n_comp,
                    covariance_type="full",
                    reg_covar=reg,
                    random_state=seed,
                    max_iter=80,
                    n_init=1,
                )
                gmm.fit(fit_data)
                means = gmm.means_
                weights = gmm.weights_
                weights = weights / weights.sum()
                return means, weights
            except ValueError:
                n_comp = max(1, n_comp // 2)
                if n_comp == 1:
                    break

    mean_traj = trajectories.mean(axis=0, keepdims=True)
    return mean_traj, np.array([1.0], dtype=np.float64)


def empirical_modes_from_samples(
    samples: np.ndarray,
    max_components: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster probabilistic samples into modes for top-k metrics."""
    batch_size, n_variates, sample_count, horizon = samples.shape
    mode_count = min(max_components, sample_count)
    centers = np.zeros((batch_size, n_variates, mode_count, horizon), dtype=np.float64)
    probs = np.zeros((batch_size, n_variates, mode_count), dtype=np.float64)

    for b in range(batch_size):
        for v in range(n_variates):
            means, weights = _fit_trajectory_gmm(
                samples[b, v],
                max_components=mode_count,
                seed=seed + b * 131 + v,
            )
            n_fit = means.shape[0]
            centers[b, v, :n_fit] = means
            probs[b, v, :n_fit] = weights
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


def summarize_prediction_pack(
    pack: Dict[str, np.ndarray],
    mode_center: Optional[np.ndarray] = None,
    mode_prob: Optional[np.ndarray] = None,
    gmm_components: int = 10,
    seed: int = 0,
    topk_max: int = 5,
) -> Dict[str, float]:
    y_true = pack["y_true"]
    det = pack["deterministic"]
    samples = pack["samples"]

    metrics: Dict[str, float] = {}
    metrics.update(deterministic_metrics(y_true, det))
    metrics["crps"] = crps_gr(y_true, samples)
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
    metrics["n_windows"] = float(y_true.shape[0])
    metrics["n_variates"] = float(y_true.shape[1])
    metrics["n_samples"] = float(samples.shape[2])
    return metrics


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


def print_summary(results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    print("\nSummary")
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
    parser.add_argument(
        "--phase",
        choices=("all", "indices", "mmpd-train", "mmpd-eval", "anchor", "merge"),
        default="all",
        help="Run one stage (for parallel Slurm) or the full pipeline.",
    )
    parser.add_argument("--datasets", nargs="+", default=["ETTh1", "ETTh2", "exchange_rate"])
    parser.add_argument(
        "--anchor-variants",
        nargs="+",
        choices=ANCHOR_VARIANTS,
        default=["gaussian"],
        help="Anchor arms to evaluate in anchor/all phases.",
    )
    parser.add_argument(
        "--anchor-variant",
        choices=ANCHOR_VARIANTS,
        default=None,
        help="Single anchor arm for --phase anchor (overrides --anchor-variants).",
    )
    parser.add_argument("--anchor-root", action="append", type=Path, default=[])
    parser.add_argument("--binary-anchor-root", action="append", type=Path, default=[])
    parser.add_argument("--ckpt-base", type=Path, default=REPO_ROOT / "results" / "ckpts")
    parser.add_argument("--mmpd-repo", type=Path, default=DEFAULT_MMPD_REPO)
    parser.add_argument("--mmpd-data-dir", type=Path, default=DEFAULT_MMPD_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--mmpd-output-root",
        type=Path,
        default=None,
        help="Shared MMPD checkpoints (default: <output-dir>/mmpd_out).",
    )
    parser.add_argument(
        "--mmpd-raw-dir",
        type=Path,
        default=None,
        help="Directory for mmpd_*.npz and indices_*.json (default: <output-dir>/raw).",
    )
    parser.add_argument(
        "--reuse-anchor-raw-from",
        type=Path,
        default=None,
        help="Optional second raw/ dir for cached anchor_gaussian_* or legacy anchor_*.npz.",
    )
    parser.add_argument(
        "--mmpd-raw-fallback",
        type=Path,
        default=None,
        help="Optional second raw/ dir to read existing mmpd_*.npz (e.g. prior eval job).",
    )
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--patch-size", type=int, default=12)
    parser.add_argument("--test-fraction", type=float, default=0.5)
    parser.add_argument("--test-max-items", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sample-num", type=int, default=100)
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an existing output dir: skip MMPD training and load raw/mmpd_*.npz "
        "when present (still runs missing MMPD eval). Anchor loads raw/anchor_*.npz unless "
        "--force-anchor-eval.",
    )
    parser.add_argument("--force-mmpd-train", action="store_true")
    parser.add_argument("--force-mmpd-eval", action="store_true")
    parser.add_argument("--force-anchor-eval", action="store_true")
    parser.add_argument("--topk-max", type=int, default=5)
    parser.add_argument("--no-update-mmpd", action="store_true")
    return parser.parse_args()


def load_raw_pack(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def explicit_roots_for_variant(args: argparse.Namespace, variant: str) -> List[Path]:
    if variant == "gaussian":
        return list(args.anchor_root)
    return list(args.binary_anchor_root)


def load_or_create_indices(
    args: argparse.Namespace,
    dataset: str,
    variate_indices: Sequence[int],
) -> List[int]:
    path = indices_path(args, dataset)
    if path.exists() and args.phase != "indices":
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    pipeline = load_tsf_pipeline()
    _, _, test_ds, _ = pipeline.load_dataset(
        dataset,
        list(variate_indices),
        lookback=args.lookback,
        horizon=args.horizon,
        stride=1,
    )
    indices = make_eval_indices(
        len(test_ds),
        args.test_fraction,
        stable_dataset_seed(args.seed, dataset),
        args.test_max_items,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(indices, f)
    print(f"[subset] {dataset}: {len(indices)}/{len(test_ds)} test windows -> {path}")
    return indices


def run_indices_phase(
    args: argparse.Namespace,
    anchors_by_variant: Dict[str, Dict[str, AnchorRun]],
) -> Dict[str, List[int]]:
    indices_by_dataset: Dict[str, List[int]] = {}
    for dataset in args.datasets:
        run = anchors_by_variant["gaussian"].get(dataset) or next(
            (anchors_by_variant[v][dataset] for v in ANCHOR_VARIANTS if dataset in anchors_by_variant[v]),
            None,
        )
        if run is None:
            raise RuntimeError(f"No anchor checkpoint to infer variates for {dataset}")
        indices_by_dataset[dataset] = load_or_create_indices(
            args,
            dataset,
            run.metadata["variate_indices"],
        )
    return indices_by_dataset


def run_mmpd_eval_phase(
    args: argparse.Namespace,
    indices_by_dataset: Dict[str, List[int]],
) -> None:
    for dataset in args.datasets:
        indices = indices_by_dataset[dataset]
        cached = resolve_mmpd_raw_read_path(args, dataset)
        if cached.exists() and not args.force_mmpd_eval:
            print(f"[mmpd] Reusing cached eval pack: {cached}")
            mmpd_pack = load_raw_pack(cached)
        else:
            mmpd_pack = run_mmpd_eval(args, dataset, indices)
        metrics = summarize_prediction_pack(
            mmpd_pack,
            mode_center=mmpd_pack.get("mode_center"),
            mode_prob=mmpd_pack.get("mode_prob"),
            gmm_components=args.gmm_components,
            seed=stable_dataset_seed(args.seed, dataset),
            topk_max=args.topk_max,
        )
        write_partial_metrics(args, dataset, "mmpd", metrics)


def run_anchor_phase(
    args: argparse.Namespace,
    variant: str,
    anchors: Dict[str, AnchorRun],
    indices_by_dataset: Dict[str, List[int]],
    device: torch.device,
) -> None:
    model_key = MODEL_KEYS[variant]
    for dataset in args.datasets:
        indices = indices_by_dataset[dataset]
        cache_path = anchor_raw_path(args, variant, dataset)
        save_path = args.mmpd_raw_dir / f"anchor_{variant}_{dataset}.npz"
        if cache_path.exists() and not args.force_anchor_eval:
            print(f"[resume] loading cached anchor pack: {cache_path}")
            anchor_pack = load_raw_pack(cache_path)
        else:
            anchor_pack = evaluate_anchor(args, anchors[dataset], indices, device)
            np.savez_compressed(save_path, **anchor_pack)
        metrics = summarize_prediction_pack(
            anchor_pack,
            gmm_components=args.gmm_components,
            seed=stable_dataset_seed(args.seed, dataset),
            topk_max=args.topk_max,
        )
        write_partial_metrics(args, dataset, model_key, metrics)


def build_manifest(
    args: argparse.Namespace,
    commit: str,
    anchors_by_variant: Dict[str, Dict[str, AnchorRun]],
    indices_by_dataset: Dict[str, List[int]],
) -> Dict[str, Any]:
    anchor_runs: Dict[str, Any] = {}
    for variant, anchors in anchors_by_variant.items():
        for dataset, run in anchors.items():
            anchor_runs[f"{variant}:{dataset}"] = {
                "root": str(run.root),
                "best_pt": str(run.best_pt),
                "itrans_pt": str(run.itrans_pt),
                "metadata": run.metadata,
            }
    return {
        "args": jsonable_args(args),
        "mmpd_commit": commit,
        "anchor_runs": anchor_runs,
        "indices_by_dataset": indices_by_dataset,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def load_indices_from_disk(
    args: argparse.Namespace,
    datasets: Sequence[str],
) -> Dict[str, List[int]]:
    indices_by_dataset: Dict[str, List[int]] = {}
    for dataset in datasets:
        path = indices_path(args, dataset)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}; run --phase indices first (or a full all-phase job)."
            )
        with path.open(encoding="utf-8") as f:
            indices_by_dataset[dataset] = json.load(f)
        print(f"[subset] {dataset}: {len(indices_by_dataset[dataset])} cached windows")
    return indices_by_dataset


def run_merge_phase(args: argparse.Namespace, manifest: Optional[Dict[str, Any]]) -> None:
    results = load_partial_metrics(args)
    if not results:
        raise FileNotFoundError(
            f"No metrics_partial/*.json under {args.output_dir}; run eval phases first."
        )
    if manifest is None:
        manifest_path = args.output_dir / "run_manifest.json"
        if manifest_path.exists():
            with manifest_path.open(encoding="utf-8") as f:
                manifest = json.load(f)
        else:
            manifest = {"updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_outputs(args, manifest, results)
    print_summary(results)
    print(f"\nWrote metrics to {args.output_dir / 'metrics.json'}")
    print(f"Wrote CSV to {args.output_dir / 'metrics.csv'}")


def main() -> None:
    args = parse_args()
    if args.resume:
        args.skip_mmpd_train = True
    if args.anchor_variant is not None:
        args.anchor_variants = [args.anchor_variant]
    args.datasets = list(dict.fromkeys(args.datasets))
    unknown = sorted(set(args.datasets) - set(DATASET_FILES))
    if unknown:
        raise ValueError(f"Unsupported dataset(s): {unknown}")
    args.output_dir = args.output_dir.resolve()
    args.mmpd_repo = args.mmpd_repo.resolve()
    args.mmpd_data_dir = args.mmpd_data_dir.resolve()
    args.ckpt_base = args.ckpt_base.resolve()
    resolve_storage_paths(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.mmpd_raw_dir.mkdir(parents=True, exist_ok=True)
    args.mmpd_output_root.mkdir(parents=True, exist_ok=True)

    commit = ""
    if args.phase != "merge":
        commit = ensure_mmpd_repo(args.mmpd_repo, update=not args.no_update_mmpd)
        stage_mmpd_datasets(args.mmpd_data_dir, args.datasets)

    anchors_by_variant: Dict[str, Dict[str, AnchorRun]] = {}
    need_anchors = args.phase in ("all", "indices", "anchor")
    if need_anchors:
        for variant in ANCHOR_VARIANTS:
            try:
                anchors_by_variant[variant] = find_anchor_runs(
                    args.datasets,
                    explicit_roots_for_variant(args, variant),
                    args.ckpt_base,
                    variant=variant,
                )
            except RuntimeError as exc:
                if variant in args.anchor_variants:
                    raise
                print(f"[warn] {exc}")

    manifest: Optional[Dict[str, Any]] = None
    indices_by_dataset: Dict[str, List[int]] = {}

    if args.phase == "indices":
        if not anchors_by_variant:
            raise RuntimeError("Need at least one anchor checkpoint tree for --phase indices.")
        indices_by_dataset = run_indices_phase(args, anchors_by_variant)
        manifest = build_manifest(args, commit, anchors_by_variant, indices_by_dataset)
        with (args.output_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        return

    if args.phase in ("all", "mmpd-eval", "anchor"):
        if args.phase == "all":
            if not anchors_by_variant:
                raise RuntimeError("Need anchor checkpoints for a full run.")
            indices_by_dataset = run_indices_phase(args, anchors_by_variant)
        else:
            indices_by_dataset = load_indices_from_disk(args, args.datasets)

    if args.phase in ("all", "mmpd-train"):
        train_mmpd(args, args.datasets)

    if args.phase in ("all", "mmpd-eval"):
        run_mmpd_eval_phase(args, indices_by_dataset)

    if args.phase in ("all", "anchor"):
        device = torch.device(
            "cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}"
        )
        for variant in args.anchor_variants:
            if variant not in anchors_by_variant:
                anchors_by_variant[variant] = find_anchor_runs(
                    args.datasets,
                    explicit_roots_for_variant(args, variant),
                    args.ckpt_base,
                    variant=variant,
                )
            run_anchor_phase(args, variant, anchors_by_variant[variant], indices_by_dataset, device)

    if args.phase == "merge":
        run_merge_phase(args, manifest=None)
        return

    if args.phase == "mmpd-train":
        print("[mmpd] train phase complete.")
        return

    manifest = build_manifest(args, commit, anchors_by_variant, indices_by_dataset)
    with (args.output_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    if args.phase == "all":
        run_merge_phase(args, manifest=manifest)


if __name__ == "__main__":
    main()
