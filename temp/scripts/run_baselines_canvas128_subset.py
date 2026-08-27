#!/usr/bin/env python3
"""Train iTransformer / PatchTST on canvas128 table subsets (paper protocol + early stop).

Lookback/horizon match the leaderboard table (seq_len=336, pred_len=96).
Variate subsets + window strides match configs/base/binary_staged.yaml.
iTransformer / PatchTST: published per-dataset script HPs (one train each), early stop.

Usage:
  python temp/scripts/run_baselines_canvas128_subset.py --model itransformer --dataset ETTh1
  python temp/scripts/run_baselines_canvas128_subset.py --model patchtst --dataset ETTh1 --smoke-test
  python temp/scripts/run_baselines_canvas128_subset.py --model both --all
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "temp" / "baselines_canvas128_subset" / "data"
OUT_ROOT = REPO / "temp" / "baselines_canvas128_subset" / "results"
ITRANS_DIR = REPO / "temp" / "iTransformer"
PATCH_DIR = REPO / "temp" / "PatchTST" / "PatchTST_supervised"
APPLY_PATCHES = REPO / "temp" / "scripts" / "apply_baseline_canvas128_patches.py"


def _out_dir_for(model: str, dataset: str, pred_len: int) -> Path:
    """Isolate H=96 vs H=720 campaigns so summaries do not collide."""
    return OUT_ROOT / f"hz{int(pred_len)}" / model / dataset


DATASETS_ALL = [
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "electricity",
    "traffic",
    "exchange_rate",
    "weather",
    "solar_Alabama",
    "PeMS",
    "illness",
    "dynamic",
]


def _load_meta() -> Dict[str, dict]:
    path = DATA_DIR / "subset_meta.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run export_canvas128_subset_csvs.py first")
    rows = json.loads(path.read_text())
    return {r["dataset"]: r for r in rows}


def _data_name(loader: str) -> str:
    if loader.startswith("ETT") or loader == "PEMS":
        return loader
    return "custom"


def _itrans_arch(dataset: str) -> Dict[str, Any]:
    """Official scripts/multivariate_forecasting/*/iTransformer*.sh for pred_len=96.

    Defaults from run.py when scripts omit a flag: lr=1e-4, batch=32, epochs=10, patience=3.
    """
    base = dict(
        e_layers=2, d_model=256, d_ff=256, n_heads=8,
        batch_size=32, learning_rate=1e-4,
        train_epochs=10, patience=3, lradj="type1",
    )
    if dataset == "ETTh1":
        return {**base, "e_layers": 2, "d_model": 256, "d_ff": 256}
    if dataset in ("ETTh2", "ETTm1", "ETTm2", "exchange_rate"):
        return {**base, "e_layers": 2, "d_model": 128, "d_ff": 128}
    if dataset == "electricity":
        return {**base, "e_layers": 3, "d_model": 512, "d_ff": 512,
                "batch_size": 16, "learning_rate": 5e-4}
    if dataset == "traffic":
        return {**base, "e_layers": 4, "d_model": 512, "d_ff": 512,
                "batch_size": 16, "learning_rate": 1e-3}
    if dataset == "solar_Alabama":
        return {**base, "e_layers": 2, "d_model": 512, "d_ff": 512,
                "learning_rate": 5e-4}
    if dataset == "PeMS":
        return {**base, "e_layers": 4, "d_model": 512, "d_ff": 512,
                "learning_rate": 1e-3}
    if dataset == "weather":
        return {**base, "e_layers": 3, "d_model": 512, "d_ff": 512}
    if dataset == "illness":
        return {**base, "e_layers": 2, "d_model": 128, "d_ff": 128}
    if dataset == "dynamic":
        return {**base, "e_layers": 3, "d_model": 512, "d_ff": 512}
    raise KeyError(f"no iTransformer script HP map for {dataset}")


def _patchtst_arch(dataset: str) -> Dict[str, Any]:
    """Published scripts/PatchTST/*.sh settings (enc_in overridden by subset)."""
    small = dict(
        e_layers=3, n_heads=4, d_model=16, d_ff=128,
        dropout=0.3, fc_dropout=0.3, batch_size=128,
        learning_rate=1e-4, patience=20, train_epochs=100,
        lradj="type3", pct_start=0.3,
    )
    large = dict(
        e_layers=3, n_heads=16, d_model=128, d_ff=256,
        dropout=0.2, fc_dropout=0.2, batch_size=128,
        learning_rate=1e-4, patience=20, train_epochs=100,
        lradj="type3", pct_start=0.3,
    )
    if dataset in ("ETTh1", "ETTh2", "illness"):
        return small
    if dataset in ("ETTm1", "ETTm2"):
        return {**large, "lradj": "TST", "pct_start": 0.4, "patience": 20}
    if dataset == "electricity":
        return {**large, "lradj": "TST", "pct_start": 0.2, "patience": 10, "batch_size": 32}
    if dataset == "traffic":
        return {**large, "lradj": "TST", "pct_start": 0.2, "patience": 10, "batch_size": 24}
    if dataset == "solar_Alabama":
        return small
    return large


def _cap_args(meta: dict) -> List[str]:
    out: List[str] = []
    mapping = (
        ("train_max_windows", "--train_max_windows"),
        ("val_max_windows", "--val_max_windows"),
        ("eval_max_windows", "--test_max_windows"),
    )
    for key, flag in mapping:
        val = meta.get(key)
        if val is None:
            continue
        out.extend([flag, str(int(val))])
    return out


def _parse_mse_mae(text: str) -> Tuple[Optional[float], Optional[float]]:
    mse = mae = None
    for m in re.finditer(r"mse[:\s]+([0-9.eE+-]+).*?mae[:\s]+([0-9.eE+-]+)", text, re.I | re.S):
        mse, mae = float(m.group(1)), float(m.group(2))
    if mse is None:
        ms = re.findall(r"\bmse[=:\s]+([0-9.eE+-]+)", text, re.I)
        ma = re.findall(r"\bmae[=:\s]+([0-9.eE+-]+)", text, re.I)
        if ms:
            mse = float(ms[-1])
        if ma:
            mae = float(ma[-1])
    return mse, mae


def _run(cmd: List[str], cwd: Path, log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] cwd={cwd}\n      {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}; see {log_path}")
    return text


def run_itransformer(
    dataset: str,
    meta: dict,
    *,
    smoke: bool,
    seq_len: int,
    pred_len: int,
    out_dir: Path,
) -> dict:
    n_v = int(meta["n_variates"])
    arch = _itrans_arch(dataset)
    if smoke:
        arch = {**arch, "train_epochs": 2, "patience": 1}

    data_name = _data_name(meta["loader"])
    tag = f"L{arch['e_layers']}_D{arch['d_model']}_lr{arch['learning_rate']}"
    model_id = f"{dataset}_{seq_len}_{pred_len}_{tag}"
    log = out_dir / f"itrans_{tag}.log"
    cmd = [
        sys.executable, "-u", "run.py",
        "--is_training", "1",
        "--model_id", model_id,
        "--model", "iTransformer",
        "--data", data_name,
        "--root_path", str(DATA_DIR) + "/",
        "--data_path", meta["csv"],
        "--features", "M",
        "--seq_len", str(seq_len),
        "--label_len", "48",
        "--pred_len", str(pred_len),
        "--e_layers", str(arch["e_layers"]),
        "--enc_in", str(n_v),
        "--dec_in", str(n_v),
        "--c_out", str(n_v),
        "--d_model", str(arch["d_model"]),
        "--d_ff", str(arch["d_ff"]),
        "--n_heads", str(arch["n_heads"]),
        "--des", "canvas128_subset",
        "--itr", "1",
        "--batch_size", str(arch["batch_size"]),
        "--learning_rate", str(arch["learning_rate"]),
        "--train_epochs", str(arch["train_epochs"]),
        "--patience", str(arch["patience"]),
        "--lradj", arch["lradj"],
        "--use_norm", "1",
        "--freq", meta["freq"],
        "--checkpoints", str(out_dir / "ckpts") + "/",
        "--num_workers", "4",
        "--train_window_stride", str(meta["train_stride"]),
        "--val_window_stride", str(meta["val_stride"]),
        "--test_window_stride", str(meta["test_stride"]),
        "--window_subset_seed", "42",
        * _cap_args(meta),
    ]
    text = _run(cmd, ITRANS_DIR, log)
    mse, mae = _parse_mse_mae(text)
    vals = [float(x) for x in re.findall(r"Vali Loss:\s*([0-9.eE+-]+)", text)]
    vali = min(vals) if vals else None
    row = {
        "tag": tag,
        "arch": arch,
        "vali_loss": vali,
        "mse": mse,
        "mae": mae,
        "log": str(log),
    }
    print(f"[itrans] {dataset} {tag}: vali={vali} mse={mse} mae={mae}", flush=True)
    result = {
        "model": "iTransformer",
        "dataset": dataset,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "n_variates": n_v,
        "subset": meta,
        "selection": "published_script_hp",
        "best": row,
        "git_sha": _git_sha(ITRANS_DIR),
    }
    (out_dir / "itransformer_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_patchtst(
    dataset: str,
    meta: dict,
    *,
    smoke: bool,
    seq_len: int,
    pred_len: int,
    out_dir: Path,
) -> dict:
    arch = _patchtst_arch(dataset)
    if smoke:
        arch = {**arch, "train_epochs": 2, "patience": 1, "batch_size": min(32, arch["batch_size"])}

    n_v = int(meta["n_variates"])
    data_name = _data_name(meta["loader"])
    model_id = f"{dataset}_{seq_len}_{pred_len}_patchtst42"
    log = out_dir / "patchtst.log"
    cmd = [
        sys.executable, "-u", "run_longExp.py",
        "--random_seed", "2021",
        "--is_training", "1",
        "--model_id", model_id,
        "--model", "PatchTST",
        "--data", data_name,
        "--root_path", str(DATA_DIR) + "/",
        "--data_path", meta["csv"],
        "--features", "M",
        "--seq_len", str(seq_len),
        "--label_len", "48",
        "--pred_len", str(pred_len),
        "--enc_in", str(n_v),
        "--dec_in", str(n_v),
        "--c_out", str(n_v),
        "--e_layers", str(arch["e_layers"]),
        "--n_heads", str(arch["n_heads"]),
        "--d_model", str(arch["d_model"]),
        "--d_ff", str(arch["d_ff"]),
        "--dropout", str(arch["dropout"]),
        "--fc_dropout", str(arch["fc_dropout"]),
        "--head_dropout", "0",
        "--patch_len", "16",
        "--stride", "8",
        "--padding_patch", "end",
        "--revin", "1",
        "--affine", "0",
        "--subtract_last", "0",
        "--decomposition", "0",
        "--individual", "0",
        "--train_epochs", str(arch["train_epochs"]),
        "--patience", str(arch["patience"]),
        "--batch_size", str(arch["batch_size"]),
        "--learning_rate", str(arch["learning_rate"]),
        "--lradj", arch["lradj"],
        "--pct_start", str(arch["pct_start"]),
        "--itr", "1",
        "--des", "canvas128_subset",
        "--freq", meta["freq"],
        "--checkpoints", str(out_dir / "ckpts") + "/",
        "--num_workers", "4",
        "--train_window_stride", str(meta["train_stride"]),
        "--val_window_stride", str(meta["val_stride"]),
        "--test_window_stride", str(meta["test_stride"]),
        "--window_subset_seed", "42",
        * _cap_args(meta),
    ]
    text = _run(cmd, PATCH_DIR, log)
    mse, mae = _parse_mse_mae(text)
    result = {
        "model": "PatchTST",
        "dataset": dataset,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "n_variates": n_v,
        "subset": meta,
        "arch": arch,
        "mse": mse,
        "mae": mae,
        "log": str(log),
        "git_sha": _git_sha(PATCH_DIR.parent),
    }
    print(f"[patchtst] {dataset}: mse={mse} mae={mae}", flush=True)
    (out_dir / "patchtst_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _git_sha(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(path), text=True
        ).strip()
    except Exception:
        return "unknown"


def _ensure_patches() -> None:
    subprocess.check_call([sys.executable, str(APPLY_PATCHES)], cwd=str(REPO))
    sys.path.insert(0, str(APPLY_PATCHES.parent))
    from apply_baseline_canvas128_patches import assert_stride_wrap_present  # noqa: E402
    assert_stride_wrap_present()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["itransformer", "patchtst", "both"], required=True)
    p.add_argument("--dataset", type=str, default="")
    p.add_argument("--all", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--force", action="store_true", help="re-run even if summary.json exists")
    p.add_argument("--seq-len", type=int, default=336, help="Match canvas128 table lookback")
    p.add_argument("--pred-len", type=int, default=96, help="Match canvas128 table horizon")
    args = p.parse_args()

    if not ITRANS_DIR.is_dir() or not PATCH_DIR.is_dir():
        raise FileNotFoundError("clone temp/iTransformer and temp/PatchTST first")

    if not (DATA_DIR / "subset_meta.json").is_file():
        subprocess.check_call(
            [sys.executable, str(REPO / "temp/scripts/export_canvas128_subset_csvs.py")],
            cwd=str(REPO),
        )
    _ensure_patches()
    meta_by = _load_meta()

    if args.all:
        datasets = DATASETS_ALL
    elif args.dataset.strip():
        datasets = [x.strip() for x in args.dataset.split(",") if x.strip()]
    else:
        raise SystemExit("pass --dataset NAME or --all")

    models = []
    if args.model in ("itransformer", "both"):
        models.append("itransformer")
    if args.model in ("patchtst", "both"):
        models.append("patchtst")

    summaries = []
    errors = []
    for ds in datasets:
        if ds not in meta_by:
            raise KeyError(f"no subset meta for {ds}")
        meta = meta_by[ds]
        for model in models:
            out_dir = _out_dir_for(model, ds, args.pred_len)
            out_dir.mkdir(parents=True, exist_ok=True)
            summary_name = (
                "itransformer_summary.json" if model == "itransformer" else "patchtst_summary.json"
            )
            summary_path = out_dir / summary_name
            if summary_path.is_file() and not args.smoke_test and not args.force:
                print(f"[skip] {summary_path} exists", flush=True)
                try:
                    summaries.append(json.loads(summary_path.read_text()))
                except Exception:
                    pass
                continue
            try:
                if model == "itransformer":
                    summaries.append(
                        run_itransformer(
                            ds, meta, smoke=args.smoke_test,
                            seq_len=args.seq_len, pred_len=args.pred_len, out_dir=out_dir,
                        )
                    )
                else:
                    summaries.append(
                        run_patchtst(
                            ds, meta, smoke=args.smoke_test,
                            seq_len=args.seq_len, pred_len=args.pred_len, out_dir=out_dir,
                        )
                    )
            except Exception as e:
                errors.append({"dataset": ds, "model": model, "error": str(e)})
                print(f"[error] {model} {ds}: {e}", flush=True)
                traceback.print_exc()

    stamp_dir = OUT_ROOT / f"hz{int(args.pred_len)}"
    stamp_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp_dir / "campaign_summary.json"
    stamp.write_text(json.dumps({"summaries": summaries, "errors": errors}, indent=2) + "\n")
    print(f"[done] {stamp} errors={len(errors)}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
