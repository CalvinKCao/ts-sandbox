"""Evaluate joint finetuned checkpoints on the held-out test split.

Loads ``{subset}_joint_finetuned_gB.pt`` / ``_gC.pt`` (or legacy ``*_joint_finetuned.pt``),
runs the same ``evaluate_model`` path as legacy finetune (DPM-Solver++, averaged
samples), optionally the same seeded random **half** of test windows, and writes
``{results_dir}/{subset_id}/results.json`` via ``save_eval_results``. Also runs
``evaluate_itransformer_baseline`` using the iTransformer weights extracted from
the joint checkpoint (temp file).

Examples::

    # All joint finetuned checkpoints in a directory
    python -m models.diffusion_tsf.eval_joint_testset \\
        --results-dir /path/to/results \\
        --checkpoint-dir /path/to/checkpoints_7var

    # Single checkpoint
    python -m models.diffusion_tsf.eval_joint_testset \\
        --results-dir /path/to/results \\
        --checkpoint /path/to/ETTh1_joint_finetuned_gB.pt

    # Subset id does not match registry dataset name (e.g. ETTh1_smoke)
    python -m models.diffusion_tsf.eval_joint_testset \\
        --results-dir /path/to/results \\
        --checkpoint /path/to/ETTh1_smoke_joint_finetuned_gB.pt \\
        --dataset ETTh1

    # Full test set (no random half-subset)
    python -m models.diffusion_tsf.eval_joint_testset \\
        --results-dir /path/to/results --checkpoint-dir /path/ckpt --full-test
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import tempfile
from dataclasses import fields
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Repo root on path (same pattern as train_multivariate_pipeline)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.diffusion_tsf.config import DiffusionTSFConfig
from models.diffusion_tsf.diffusion_model import DiffusionTSF
from models.diffusion_tsf.guidance import iTransformerGuidance
from models.diffusion_tsf.pipeline_config import EVAL_NUM_SAMPLES
from models.diffusion_tsf.train_multivariate_pipeline import (
    DATASET_REGISTRY,
    FORECAST_LENGTH,
    ITRANSFORMER_SEQ_LEN,
    create_itransformer,
    evaluate_itransformer_baseline,
    evaluate_model,
    generate_dataset_job,
    load_dataset,
    save_eval_results,
)

logger = logging.getLogger(__name__)

_JOINT_FT_RE = re.compile(
    r"^(?P<subset>.+)_joint_finetuned(?P<variant>_gB|_gC)?\.pt$",
    re.IGNORECASE,
)


def _parse_joint_finetune_filename(path: str) -> Tuple[str, Optional[bool]]:
    """Return (subset_id, joint_use_ghost_image or None if pattern does not match)."""
    base = os.path.basename(path)
    m = _JOINT_FT_RE.match(base)
    if not m:
        return base.replace(".pt", ""), None
    subset = m.group("subset")
    suf = m.group("variant")
    if suf is None or suf.lower() == "_gb":
        return subset, True
    if suf.lower() == "_gc":
        return subset, False
    return subset, True


def _config_from_checkpoint(raw: dict) -> DiffusionTSFConfig:
    names = {f.name for f in fields(DiffusionTSFConfig)}
    return DiffusionTSFConfig(**{k: v for k, v in raw.items() if k in names})


def _itrans_state_dict_from_joint(full_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefix = "guidance_model.model."
    out = {k[len(prefix) :]: v for k, v in full_sd.items() if k.startswith(prefix)}
    if not out:
        raise RuntimeError("No keys starting with 'guidance_model.model.' in checkpoint")
    return out


def build_joint_model_from_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[DiffusionTSF, dict]:
    """Load full joint DiffusionTSF from a ``*_joint_finetuned*.pt`` / joint bundle."""
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" not in blob or "config" not in blob:
        raise RuntimeError(f"Checkpoint {ckpt_path} missing model_state_dict or config")
    raw_cfg = blob["config"]
    cfg = _config_from_checkpoint(raw_cfg)

    itrans = create_itransformer(
        seq_len=ITRANSFORMER_SEQ_LEN,
        pred_len=FORECAST_LENGTH,
        num_vars=cfg.num_variables,
        dropout=0.1,
    ).to(device)
    guidance = iTransformerGuidance(itrans, freeze=False)
    model = DiffusionTSF(cfg, guidance_model=guidance).to(device)
    missing, unexpected = model.load_state_dict(blob["model_state_dict"], strict=False)
    if missing:
        logger.warning("load_state_dict missing (non-fatal): %s", missing[:8])
    if unexpected:
        logger.warning("load_state_dict unexpected: %s", unexpected[:8])
    model.eval()
    return model, blob


def _write_temp_itrans_ckpt(full_sd: Dict[str, torch.Tensor], device: torch.device) -> str:
    inner = _itrans_state_dict_from_joint(full_sd)
    fd, path = tempfile.mkstemp(suffix="_itrans_from_joint.pt", dir=None)
    os.close(fd)
    torch.save({"model_state_dict": inner}, path)
    return path


def discover_joint_finetune_checkpoints(checkpoint_dir: str) -> List[str]:
    out: List[str] = []
    for fn in sorted(os.listdir(checkpoint_dir)):
        if not fn.endswith(".pt"):
            continue
        if _JOINT_FT_RE.match(fn):
            out.append(os.path.join(checkpoint_dir, fn))
    return out


def run_one(
    ckpt_path: str,
    results_dir: str,
    device: torch.device,
    *,
    dataset_name: Optional[str],
    smoke_test: bool,
    full_test: bool,
    n_samples: int,
) -> None:
    subset_id, ghost_from_name = _parse_joint_finetune_filename(ckpt_path)
    ds_name = dataset_name or subset_id
    if ds_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset {ds_name!r} not in DATASET_REGISTRY. "
            f"Pass --dataset with the registry name (subset_id from filename is {subset_id!r})."
        )

    variate_indices = generate_dataset_job(ds_name)["variate_indices"]

    model, blob = build_joint_model_from_checkpoint(ckpt_path, device)

    _, _, test_ds, _ = load_dataset(ds_name, variate_indices, stride=1)
    if smoke_test:
        test_ds = Subset(test_ds, list(range(min(2, len(test_ds)))))
        logger.info("[%s] smoke test: %d test windows", subset_id, len(test_ds))
    elif full_test:
        logger.info("[%s] full test set: %d windows", subset_id, len(test_ds))
    else:
        n_full = len(test_ds)
        n_eval = max(1, n_full // 2)
        rng = np.random.default_rng(42)
        eval_idx = sorted(rng.choice(n_full, size=n_eval, replace=False).tolist())
        test_ds = Subset(test_ds, eval_idx)
        logger.info("[%s] eval subset: %d/%d test windows (seeded random half)", subset_id, n_eval, n_full)

    test_loader = DataLoader(test_ds, batch_size=8 if not smoke_test else 2, shuffle=False)

    eval_results = evaluate_model(
        model, test_loader, device, n_samples=n_samples, smoke_test=smoke_test
    )
    logger.info(
        "[%s] diffusion test | single MSE=%.6f MAE=%.6f | avg MSE=%.6f MAE=%.6f",
        subset_id,
        eval_results["single"]["mse"],
        eval_results["single"]["mae"],
        eval_results["averaged"]["mse"],
        eval_results["averaged"]["mae"],
    )

    tuned = blob.get("tuned_params") or {}
    if ghost_from_name is not None:
        tuned = {**tuned, "ghost_variant_inferred_from_filename": "B" if ghost_from_name else "C"}
    train_metrics = {
        "checkpoint": os.path.abspath(ckpt_path),
        "best_val_loss": blob.get("best_val_loss"),
        "best_epoch": blob.get("best_epoch"),
        "tuned_params": tuned,
        "eval_protocol": "full_test" if full_test or smoke_test else "half_test_seeded_42",
    }

    save_eval_results(subset_id, ds_name, variate_indices, train_metrics, eval_results, results_dir)

    tmp_itrans = None
    try:
        tmp_itrans = _write_temp_itrans_ckpt(blob["model_state_dict"], device)
        evaluate_itransformer_baseline(
            subset_id,
            ds_name,
            variate_indices,
            tmp_itrans,
            results_dir,
            device,
            smoke_test=smoke_test,
            test_loader=test_loader,
        )
    except Exception as e:
        logger.warning("[%s] iTransformer baseline eval failed: %s", subset_id, e)
    finally:
        if tmp_itrans and os.path.isfile(tmp_itrans):
            try:
                os.remove(tmp_itrans)
            except OSError:
                pass


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    p = argparse.ArgumentParser(description="Test-set eval for joint finetuned checkpoints")
    p.add_argument("--results-dir", type=str, required=True, help="Where to write per-subset results.json")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint-dir", type=str, help="Directory containing *_joint_finetuned*.pt")
    g.add_argument("--checkpoint", type=str, help="Single joint finetuned .pt path")
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Registry dataset name if subset_id (from filename) differs, e.g. ETTh1 for ETTh1_smoke",
    )
    p.add_argument("--smoke-test", action="store_true", help="Tiny eval (2 windows, fast sampler)")
    p.add_argument(
        "--full-test",
        action="store_true",
        help="Use entire test split (default: random half of test windows, same as legacy)",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=EVAL_NUM_SAMPLES,
        help=f"Averaged diffusion samples per window (default {EVAL_NUM_SAMPLES})",
    )
    args = p.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint:
        paths = [args.checkpoint]
    else:
        paths = discover_joint_finetune_checkpoints(args.checkpoint_dir)
        if not paths:
            logger.error("No *_joint_finetuned*.pt files in %s", args.checkpoint_dir)
            sys.exit(1)
        logger.info("Found %d checkpoint(s) under %s", len(paths), args.checkpoint_dir)

    for ck in paths:
        if not os.path.isfile(ck):
            logger.error("Missing file: %s", ck)
            sys.exit(1)
        logger.info("=== Evaluating %s ===", ck)
        run_one(
            ck,
            args.results_dir,
            device,
            dataset_name=args.dataset,
            smoke_test=args.smoke_test,
            full_test=args.full_test,
            n_samples=args.n_samples,
        )

    # Rebuild summary.csv from all results.json under results_dir
    from models.diffusion_tsf.train_multivariate_pipeline import update_summary_csv

    update_summary_csv(args.results_dir)
    logger.info("Wrote summary: %s", os.path.join(args.results_dir, "summary.csv"))


if __name__ == "__main__":
    main()
