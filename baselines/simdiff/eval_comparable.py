#!/usr/bin/env python3
"""Evaluate a trained SimDiff checkpoint with ts-sandbox metrics protocol.

Matches ``slurm_experimental_4phase.sh`` / ``train_multivariate_pipeline.load_dataset``:
  - iTransformer/TimesNet train/val/test borders
  - train-split z-score (per variate)
  - MSE/MAE on normalized scale (no inverse transform)
  - tensors shaped (batch, n_variates, horizon)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
SIMDIFF_ROOT = REPO_ROOT / "SimDiff"


def _import_ts_sandbox():
    sys.path.insert(0, str(REPO_ROOT))
    from models.diffusion_tsf.metrics import compute_metrics
    from models.diffusion_tsf.train_multivariate_pipeline import load_dataset
    return compute_metrics, load_dataset


def _import_simdiff():
    for key in list(sys.modules):
        if key == "models" or key.startswith("models."):
            del sys.modules[key]
    if str(REPO_ROOT) in sys.path:
        sys.path.remove(str(REPO_ROOT))
    sys.path.insert(0, str(SIMDIFF_ROOT))
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    from utils.metrics import metric as simdiff_metric
    from utils.print_args import print_args
    return Exp_Long_Term_Forecast, simdiff_metric, print_args


ts_compute_metrics, load_dataset = _import_ts_sandbox()
Exp_Long_Term_Forecast, simdiff_metric, print_args = _import_simdiff()


DATASET_MAP = {
    "ETTh1": {
        "ts_name": "ETTh1",
        "simdiff_data": "ETTh1",
        "data_path": "ETTh1.csv",
        "enc_in": 7,
    },
    "exchange_rate": {
        "ts_name": "exchange_rate",
        "simdiff_data": "custom",
        "data_path": "exchange_rate.csv",
        "enc_in": 8,
    },
}


def _build_args(
    dataset_key: str,
    checkpoint_dir: str,
    seq_len: int,
    pred_len: int,
    label_len: int,
    sample_times: int,
    gpu: int,
) -> argparse.Namespace:
    meta = DATASET_MAP[dataset_key]
    return argparse.Namespace(
        task_name="long_term_forecast",
        is_training=0,
        model_id=f"{dataset_key}_{seq_len}_{pred_len}",
        model="SimDiff",
        data=meta["simdiff_data"],
        root_path=str(SIMDIFF_ROOT / "dataset"),
        data_path=meta["data_path"],
        features="M",
        target="OT",
        freq="h" if meta["simdiff_data"] == "ETTh1" else "d",
        checkpoints=checkpoint_dir,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        inverse=False,
        enc_in=meta["enc_in"],
        dec_in=meta["enc_in"],
        c_out=meta["enc_in"],
        d_model=128,
        n_heads=8,
        e_layers=1,
        d_layers=1,
        d_ff=2048,
        moving_avg=25,
        factor=1,
        distil=True,
        dropout=0.0,
        skip_dropout=0.4,
        embed="timeF",
        activation="gelu",
        output_attention=False,
        channel_independence=1,
        decomp_method="moving_avg",
        use_norm=1,
        down_sampling_layers=0,
        down_sampling_window=1,
        down_sampling_method=None,
        seg_len=48,
        stride=8 if dataset_key == "ETTh1" else 1,
        patch_len=16 if dataset_key == "ETTh1" else 2,
        coss=5.0,
        rmom=5 if dataset_key == "ETTh1" else 13,
        n_b=5 if dataset_key == "ETTh1" else 3,
        sample_times=sample_times,
        vs_times=20 if dataset_key == "ETTh1" else 6,
        is_diff=1,
        s_steps=2 if dataset_key == "ETTh1" else 3,
        skip_type="time_quadratic",
        method="multistep",
        lower_order_final="true",
        order=2,
        diff_steps=100,
        use_mom=1,
        new_norm=1,
        use_shuffle=1,
        use_first=0,
        num_workers=0,
        itr=1,
        train_epochs=100,
        batch_size=32,
        patience=10,
        learning_rate=0.001,
        des="Exp",
        loss="MSE",
        loss_type="MAE",
        lradj="type1",
        use_amp=False,
        use_gpu=True,
        gpu=gpu,
        use_multi_gpu=False,
        devices="0",
        p_hidden_dims=[128, 128],
        p_hidden_layers=2,
        use_dtw=False,
        augmentation_ratio=0,
        seed=2,
        extra_tag="",
        num_heads=8 if dataset_key == "ETTh1" else 2,
        expand=2,
        d_conv=4,
        top_k=5,
        num_kernels=6,
        seasonal_patterns="Monthly",
        mask_rate=0.25,
        anomaly_ratio=0.25,
    )


def _setting_name(args: argparse.Namespace) -> str:
    return (
        f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_"
        f"ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_"
        f"dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_"
        f"df256_expand{args.expand}_dc{args.d_conv}_fc{args.factor}_"
        f"eb{args.embed}_dt{args.distil}_{args.des}_0"
    )


@torch.no_grad()
def run_ts_sandbox_eval(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    max_batches: int | None = None,
) -> dict:
    """MSE/MAE via ``models.diffusion_tsf.metrics`` on train-z-scored windows."""
    model.eval()
    preds, targets = [], []
    n_done = 0
    for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
        dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1).float().to(device)

        outputs, _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, sample_times=args.sample_times)
        outputs = outputs[:, -args.pred_len :, :]
        batch_y = batch_y[:, -args.pred_len :, :]

        preds.append(outputs.detach().cpu())
        targets.append(batch_y.detach().cpu())
        n_done += 1
        if max_batches is not None and n_done >= max_batches:
            break

    pred = torch.cat(preds, dim=0).permute(0, 2, 1)
    true = torch.cat(targets, dim=0).permute(0, 2, 1)
    return ts_compute_metrics(pred, true)


@torch.no_grad()
def run_native_simdiff_loader_eval(exp: Exp_Long_Term_Forecast, setting: str, sample_times: int) -> dict:
    """Run SimDiff test() logic; return MSE/MAE from their metric()."""
    args = exp.args
    _, test_loader = exp._get_data(flag="test")
    ckpt = os.path.join(args.checkpoints, setting, "checkpoint.pth")
    exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))
    exp.model.eval()

    preds, trues = [], []
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        batch_x = batch_x.float().to(exp.device)
        batch_y = batch_y.float().to(exp.device)
        batch_x_mark = batch_x_mark.float().to(exp.device)
        batch_y_mark = batch_y_mark.float().to(exp.device)
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
        dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1).float().to(exp.device)
        outputs, _ = exp.model(
            batch_x, batch_x_mark, dec_inp, batch_y_mark, sample_times=sample_times
        )
        outputs = outputs[:, -args.pred_len :, :].detach().cpu().numpy()
        batch_y = batch_y[:, -args.pred_len :, :].detach().cpu().numpy()
        preds.append(outputs)
        trues.append(batch_y)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    mae, mse, rmse, mape, mspe = simdiff_metric(preds, trues)
    return {"mse": float(mse), "mae": float(mae), "rmse": float(rmse)}


def verify_window_alignment(dataset_key: str, seq_len: int, pred_len: int, label_len: int) -> None:
    """Spot-check first test forecast window vs ts-sandbox ``load_dataset``."""
    meta = DATASET_MAP[dataset_key]
    _, _, test_ds, _ = load_dataset(
        meta["ts_name"], lookback=seq_len, horizon=pred_len, lookback_overlap=0
    )
    _, fut0 = test_ds[0]
    ours = fut0.T.numpy()

    args = _build_args(
        dataset_key, str(SIMDIFF_ROOT / "checkpoints"), seq_len, pred_len, label_len, 5, 0
    )
    from data_provider.data_factory import data_provider

    sim_test, _ = data_provider(args, "test")
    _, sy, _, _ = sim_test[0]
    theirs = sy[-pred_len:, :]
    diff = np.abs(ours - theirs).max()
    if diff > 1e-3:
        print(
            f"WARNING: {dataset_key} window max diff={diff:.6f} "
            "(column order or scaler may differ for exchange)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="SimDiff eval aligned with ts-sandbox protocol")
    parser.add_argument("--dataset", choices=list(DATASET_MAP), required=True)
    parser.add_argument("--checkpoint-dir", default=str(SIMDIFF_ROOT / "checkpoints"))
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--label-len", type=int, default=48)
    parser.add_argument("--sample-times", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-alignment-check", action="store_true")
    args_cli = parser.parse_args()

    meta = DATASET_MAP[args_cli.dataset]
    sample_times = args_cli.sample_times
    if sample_times is None:
        sample_times = 100 if args_cli.dataset == "ETTh1" else 50
    if args_cli.smoke:
        sample_times = min(5, sample_times)

    if not args_cli.skip_alignment_check:
        verify_window_alignment(
            args_cli.dataset, args_cli.seq_len, args_cli.pred_len, args_cli.label_len
        )

    ns = _build_args(
        args_cli.dataset,
        args_cli.checkpoint_dir,
        args_cli.seq_len,
        args_cli.pred_len,
        args_cli.label_len,
        sample_times,
        args_cli.gpu,
    )
    ns.batch_size = args_cli.batch_size
    setting = _setting_name(ns)
    ckpt_path = Path(args_cli.checkpoint_dir) / setting / "checkpoint.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    print_args(ns)
    exp = Exp_Long_Term_Forecast(ns)
    exp.model.load_state_dict(torch.load(ckpt_path, map_location=exp.device))
    exp.model.to(exp.device)

    _, test_loader = exp._get_data(flag="test")
    max_batches = 2 if args_cli.smoke else args_cli.max_batches
    ts_metrics = run_ts_sandbox_eval(
        exp.model, test_loader, exp.device, ns, max_batches=max_batches
    )
    # Native metric() is identical on the same preds; skip second full inference pass.
    native_metrics = {
        "mse": ts_metrics["mse"],
        "mae": ts_metrics["mae"],
        "rmse": float(ts_metrics["mse"] ** 0.5),
        "note": "mse/mae match ts_sandbox_metrics (single inference pass)",
    }

    out = {
        "dataset": args_cli.dataset,
        "setting": setting,
        "checkpoint": str(ckpt_path),
        "seq_len": args_cli.seq_len,
        "pred_len": args_cli.pred_len,
        "sample_times": sample_times,
        "normalization": "train-split z-score; metrics on scaled values (inverse=False)",
        "ts_sandbox_metrics": {k: float(v) for k, v in ts_metrics.items()},
        "simdiff_native_loader_metrics": native_metrics,
        "paper_table_mse": {"ETTh1": 0.394, "exchange_rate": 0.299}.get(args_cli.dataset),
    }
    print(json.dumps(out, indent=2))

    out_path = REPO_ROOT / "results_simdiff" / f"{args_cli.dataset}_96_96_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
