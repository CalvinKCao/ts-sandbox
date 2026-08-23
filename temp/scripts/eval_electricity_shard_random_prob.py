#!/usr/bin/env python3
"""Time-boxed random-window probabilistic eval for electricity 160/161 shards.

Draws n_samples MC draws per randomly chosen test window (quad_t / 20 steps,
same as the s32_prob staged eval) and logs each sample's MSE. Seeded shuffle
of the test set; optional --n-windows cap. Stops ~drain-seconds before the
wall budget instead of walking windows in order.

Example:
  python -u temp/scripts/eval_electricity_shard_random_prob.py \\
    --config configs/binary_window_norm_patch_refine_canvas128_p64x6_electricity_v000_159_s2_every4.yaml \\
    --ckpt-dir results/ckpts/08-17-4854714-electricity-binary_window_norm_patch_refine_canvas128_p64x6_electricity_v000_159_s2_every4 \\
    --out-jsonl results/logs/elec-randprob40.jsonl \\
    --n-windows 40 --n-samples 10 --max-seconds 27900
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

logger = logging.getLogger("elec_randprob")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--dataset", default="electricity")
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--n-windows", type=int, default=0,
                   help="cap on seeded-shuffle test windows (0 = all, still time-boxed)")
    p.add_argument("--test-stride", type=int, default=32,
                   help="test window stride; 1 = every valid test start (not the s32 eval grid)")
    p.add_argument("--sampler", default="quad_t")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-seconds", type=float, default=1500.0)
    p.add_argument("--drain-seconds", type=float, default=90.0)
    p.add_argument("--stop-unix", type=float, default=0.0,
                   help="unix timestamp; if set, stop by min(this, start+max-seconds)")
    p.add_argument("--sequential-samples", action="store_true",
                   help="one generate() per sample instead of repeat_interleave")
    return p.parse_args()


def _deadline(args: argparse.Namespace, t0: float) -> float:
    end = t0 + float(args.max_seconds)
    if args.stop_unix > 0:
        end = min(end, float(args.stop_unix) - float(args.drain_seconds))
    return end


def _write(fh, rec: dict) -> None:
    fh.write(json.dumps(rec) + "\n")
    fh.flush()


def _mse_mae(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    err = pred - y
    return float(np.mean(err ** 2)), float(np.mean(np.abs(err)))


def _generate_samples(coarse, fine, past, n_samples: int, sampler: str, steps: int, parallel: bool):
    kwargs = {"sampler": sampler, "num_inference_steps": int(steps)}
    if parallel:
        past_exp = past.repeat_interleave(n_samples, dim=0)
        coarse_out = coarse.generate(past_exp, **kwargs)
        fine_out = fine.generate(
            past_exp,
            future_coarse_2d=coarse_out["future_2d_coarse"],
            **kwargs,
        )
        return fine_out["prediction_global_norm"].detach()
    preds = []
    for _ in range(n_samples):
        coarse_out = coarse.generate(past, **kwargs)
        fine_out = fine.generate(
            past,
            future_coarse_2d=coarse_out["future_2d_coarse"],
            **kwargs,
        )
        preds.append(fine_out["prediction_global_norm"].detach())
    return torch.cat(preds, dim=0)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    os.environ.setdefault("WANDB_MODE", "disabled")
    t0 = time.time()
    deadline = _deadline(args, t0)

    from models.diffusion_tsf.pipeline.config import apply_cli_state_overrides, load_experiment_config
    from models.diffusion_tsf.pipeline.phases.staged_eval import StagedEvalPhase
    from models.diffusion_tsf.pipeline.state import PipelineState
    from models.diffusion_tsf.train_multivariate_pipeline import (
        dataset_window_lengths,
        load_dataset,
        load_wrapped_guidance,
        resolve_pipeline_data_subset,
    )

    ckpt_dir = str(Path(args.ckpt_dir).expanduser().resolve())
    if not Path(ckpt_dir).is_dir():
        raise FileNotFoundError(f"ckpt dir missing: {ckpt_dir}")

    cfg = load_experiment_config(
        str(Path(args.config).resolve()),
        cli_overrides={
            "dataset": args.dataset,
            "checkpoint_dir": ckpt_dir,
            "seed": args.seed,
        },
    )
    state = PipelineState.from_config(cfg)
    apply_cli_state_overrides(state, cfg)
    state.dataset = args.dataset
    state.checkpoint_dir = ckpt_dir
    state.extra["eval_source_checkpoint_dir"] = ckpt_dir
    state.seed = int(args.seed)
    resolve_pipeline_data_subset(state)

    device = state.resolve_device()
    if device.type != "cuda":
        raise RuntimeError("this probe expects a GPU")
    logger.info("device=%s ckpt=%s subset=%s n_var=%d", device, ckpt_dir, state.subset_id, state.n_variates)

    subset_meta = state.data_subset_resolved or {}
    train_stride = int(subset_meta.get("train_stride", state.window_stride))
    # CLI stride wins. Do not clamp up to yaml subset test_stride (often 8);
    # --test-stride 1 samples every valid test start, not the s32/s8 grid.
    test_stride = int(args.test_stride)
    if test_stride < 1:
        raise ValueError(f"--test-stride must be >= 1, got {test_stride}")
    n_iv = len(state.variate_indices)
    _train, _val, test_ds, norm_stats = load_dataset(
        state, state.dataset,
        state.variate_indices,
        stride=train_stride,
        test_stride=test_stride,
        ordinal_tie_atol=float(state.ordinal_tie_atol),
        use_ordinal_window_norm=state.use_ordinal_window_norm,
    )
    if norm_stats.get("ordinal_ladder") is not None:
        state.ordinal_ladder = norm_stats["ordinal_ladder"]
    logger.info("test windows=%d stride=%d (yaml subset stride=%s)",
                len(test_ds), test_stride, subset_meta.get("test_stride"))

    ds_lb, ds_hz = dataset_window_lengths(state, state.dataset)
    guide_path = os.path.join(ckpt_dir, f"{state.subset_id}_patch_guidance.pt")
    if not os.path.isfile(guide_path):
        raise FileNotFoundError(f"missing guidance: {guide_path}")
    guidance = load_wrapped_guidance(
        state, guide_path, n_iv, device,
        guidance_type=state.guidance_type,
        dataset_lookback=ds_lb,
        dataset_horizon=ds_hz,
    )
    phase = StagedEvalPhase(phase="staged_eval")
    coarse = phase._load_model(state, "coarse", guidance, n_iv, device)
    fine = phase._load_model(state, "patch_refine", guidance, n_iv, device)
    ranked = getattr(test_ds, "yields_ordinal_ranks", False)
    for m in (coarse, fine):
        m._ordinal_input_is_ranked = ranked
        m._ordinal_apply_ood_shift = not ranked

    rng = np.random.default_rng(int(args.seed))
    order = rng.permutation(len(test_ds)).tolist()
    n_want = int(args.n_windows)
    if n_want > 0:
        order = order[:n_want]
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out_path.with_suffix(".summary.json")

    n_samples = int(args.n_samples)
    parallel = not bool(args.sequential_samples)
    K = int(getattr(coarse.config, "lookback_overlap", 0) or 0)
    window_mean_mses = []
    n_done = 0
    last_window_s = 0.0

    header = {
        "type": "header",
        "ckpt_dir": ckpt_dir,
        "config": args.config,
        "subset_id": state.subset_id,
        "n_variates": n_iv,
        "n_test_windows": len(test_ds),
        "n_windows": len(order),
        "test_stride": test_stride,
        "n_samples": n_samples,
        "sampler": args.sampler,
        "steps": int(args.steps),
        "seed": int(args.seed),
        "parallel": parallel,
        "deadline_unix": deadline,
    }
    logger.info("header %s", json.dumps(header))

    with out_path.open("w") as fh:
        _write(fh, header)
        for wi in order:
            remaining = deadline - time.time()
            if remaining < args.drain_seconds:
                logger.info("time budget gone (%.0fs left), stopping after %d windows", remaining, n_done)
                break
            if last_window_s > 0 and remaining < last_window_s + 15:
                logger.info("not enough time for another window (need ~%.0fs, have %.0fs)",
                            last_window_s, remaining)
                break

            past, future = test_ds[int(wi)]
            past = past.unsqueeze(0).to(device)
            future = future.unsqueeze(0).to(device)
            if K > 0:
                future = future[..., K:]
            y = future.detach().cpu().numpy()

            w_t0 = time.time()
            try:
                with torch.no_grad():
                    torch.manual_seed(int(args.seed) + 1009 * int(wi))
                    pred = _generate_samples(
                        coarse, fine, past, n_samples, args.sampler, args.steps, parallel,
                    )
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or not parallel:
                    raise
                logger.warning("parallel n_samples=%d OOM; retrying sequential", n_samples)
                torch.cuda.empty_cache()
                parallel = False
                with torch.no_grad():
                    torch.manual_seed(int(args.seed) + 1009 * int(wi))
                    pred = _generate_samples(
                        coarse, fine, past, n_samples, args.sampler, args.steps, False,
                    )
            pred_np = pred.detach().cpu().numpy()
            last_window_s = time.time() - w_t0
            if pred_np.shape[0] != n_samples:
                raise ValueError(f"expected {n_samples} samples, got {pred_np.shape}")

            sample_mses = []
            for s in range(n_samples):
                mse, mae = _mse_mae(pred_np[s], y[0])
                sample_mses.append(mse)
                rec = {
                    "type": "sample",
                    "window_index": int(wi),
                    "sample_idx": s,
                    "mse": mse,
                    "mae": mae,
                    "elapsed_s": time.time() - t0,
                }
                _write(fh, rec)
                logger.info("window=%d sample=%d mse=%.6f mae=%.6f", wi, s, mse, mae)

            mean_pred = pred_np.mean(axis=0, keepdims=True)
            prob_mse, prob_mae = _mse_mae(mean_pred[0], y[0])
            window_mean_mses.append(prob_mse)
            n_done += 1
            running = float(np.mean(window_mean_mses))
            wrec = {
                "type": "window",
                "window_index": int(wi),
                "sample_mean_mse": prob_mse,
                "sample_mean_mae": prob_mae,
                "mean_of_sample_mses": float(np.mean(sample_mses)),
                "n_windows_done": n_done,
                "running_prob_mse": running,
                "window_s": last_window_s,
                "elapsed_s": time.time() - t0,
            }
            _write(fh, wrec)
            logger.info(
                "window=%d/%d sample_mean_mse=%.6f running_prob_mse=%.6f (n=%d) window_s=%.1f",
                n_done, len(order), prob_mse, running, n_done, last_window_s,
            )
            summary = {
                "type": "summary",
                "n_windows": n_done,
                "n_samples": n_samples,
                "running_prob_mse": running,
                "per_window_sample_mean_mse": window_mean_mses,
                "elapsed_s": time.time() - t0,
                "parallel": parallel,
            }
            summary_path.write_text(json.dumps(summary, indent=2))
            torch.cuda.empty_cache()

        footer = {
            "type": "done",
            "n_windows": n_done,
            "running_prob_mse": float(np.mean(window_mean_mses)) if window_mean_mses else None,
            "elapsed_s": time.time() - t0,
            "hit_deadline": time.time() >= deadline - args.drain_seconds,
        }
        _write(fh, footer)
        logger.info("done %s", json.dumps(footer))


if __name__ == "__main__":
    main()
