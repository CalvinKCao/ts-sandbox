#!/usr/bin/env python3
"""One-off: shared-window binary vs MMPD panels with anchor + probabilistic samples.

Uses MMPD packs (deterministic + samples) and regenerates binary anchor + a few
prob draws for the same pool indices (disc-raw / pack-test-stride 4 alignment).
Staged_eval window indices are a different pool — do not mix them blindly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.pipeline.globals_bridge import patch_globals  # noqa: E402
from models.diffusion_tsf.train_multivariate_pipeline import (  # noqa: E402
    load_dataset,
    load_wrapped_guidance,
    resolve_pipeline_data_subset,
)
from temp.eval_univariate_patch_refine_ordinal_vs_mmpd import _mmpd_pack, _pack_test_stride  # noqa: E402
from temp.eval_univariate_patch_refine_vs_gt import load_patch_refine_run  # noqa: E402
from utils.binary_mmpd_sample_panels import generate_binary_vs_mmpd_anchor_prob_panels  # noqa: E402
from utils.dual_scale_bin_filter import align_mmpd_to_binary_dataset_norm  # noqa: E402
from utils.eval_discriminator_texture_staged_vs_mmpd import binary_mmpd_train_scaler_map  # noqa: E402
from utils.eval_mmpd_gaussian_anchor import (  # noqa: E402
    load_tsf_pack_pool,
    parse_pack_splits,
    run_subset_id,
    run_test_stride,
    run_train_stride,
    run_variate_indices,
)
from utils.eval_trend_robust_texture_staged_vs_mmpd import generate_staged_forecast  # noqa: E402
from utils.forecast_pack_reduce import subset_pack_by_pool_indices  # noqa: E402
from utils.visualize_staged_eval_2d_preds import (  # noqa: E402
    _build_state,
    _load_stage_model,
    _resolve_guidance_ckpt,
)

DEFAULT_DATASETS = ("electricity", "ETTh1", "dynamic", "traffic")
DEFAULT_BINARY = {
    "electricity": "results/ckpts/07-29-4462979-electricity-binary_patch_refine_lb336_hz96_ordinal_tuned_synth_fallback",
    "ETTh1": "results/ckpts/07-29-4462980-ETTh1-binary_patch_refine_lb336_hz96_ordinal_tuned_synth_fallback",
    "dynamic": "results/ckpts/07-29-4462981-dynamic-binary_patch_refine_lb336_hz96_ordinal_tuned_synth_fallback",
    "traffic": "results/ckpts/07-29-4462982-traffic-binary_patch_refine_lb336_hz96_ordinal_tuned_synth_fallback",
}
DEFAULT_DISC_RAW = REPO_ROOT / "results/datasets/07-31-0925-h96-ordinal-disc-raw"
DEFAULT_MMPD = REPO_ROOT / "results/datasets/07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd"
DEFAULT_OUT = REPO_ROOT / "results/datasets/07-31-h96-binary-vs-mmpd-anchor-prob"
BINARY_CONFIG = "configs/binary_patch_refine_lb336_hz96_ordinal_tuned_synth_fallback.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    p.add_argument("--disc-raw-dir", type=Path, default=DEFAULT_DISC_RAW)
    p.add_argument("--mmpd-output-root", type=Path, default=DEFAULT_MMPD)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n-windows", type=int, default=2)
    p.add_argument("--n-prob-samples", type=int, default=4)
    p.add_argument("--num-inference-steps", type=int, default=20)
    p.add_argument("--sampler", default="quad_t")
    p.add_argument("--pack-test-stride", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    args.datasets = [d for raw in args.datasets for d in str(raw).split(",") if d]
    args.disc_raw_dir = args.disc_raw_dir.expanduser().resolve()
    args.mmpd_output_root = args.mmpd_output_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.smoke_test:
        args.datasets = args.datasets[:1]
        args.n_windows = 1
        args.n_prob_samples = 1
        args.num_inference_steps = min(int(args.num_inference_steps), 2)
    return args


def _load_models(dataset: str, ckpt_root: Path, device: torch.device):
    run, stages = load_patch_refine_run(dataset, ckpt_root, test_stride=None)
    state = _build_state(ckpt_root, dataset, run_subset_id(run), str(REPO_ROOT / BINARY_CONFIG))
    resolve_pipeline_data_subset(state)
    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

    _, _, _, norm_stats = load_dataset(
        dataset,
        run_variate_indices(run),
        stride=run_train_stride(run),
        test_stride=run_test_stride(run),
        lookback=336,
        horizon=96,
        ordinal_tie_atol=float(state.ordinal_tie_atol),
        use_ordinal_window_norm=True,
    )
    ladder = norm_stats["ordinal_ladder"]
    state.extra["global_ordinal_ladder"] = ladder
    pipeline_mod.GLOBAL_ORDINAL_LADDER = ladder
    patch_globals(pipeline_mod, state, honor_dataset_windows=True)

    guidance = None
    if bool(state.use_guidance_channel) or not bool(state.disable_cross_attention):
        path, guidance_type = _resolve_guidance_ckpt(ckpt_root, run_subset_id(run), "auto")
        guidance = load_wrapped_guidance(
            str(path), len(run_variate_indices(run)), device,
            guidance_type=guidance_type, dataset_lookback=336, dataset_horizon=96,
        )
        if hasattr(guidance, "ordinal_ladder"):
            guidance.ordinal_ladder = ladder
    coarse = _load_stage_model(
        state, "coarse", stages["coarse_pt"], guidance, len(run_variate_indices(run)), device,
        strict_non_guidance_shapes=True,
    )
    refine = _load_stage_model(
        state, "patch_refine", stages["refine_pt"], guidance, len(run_variate_indices(run)), device,
        strict_non_guidance_shapes=True,
    )
    return run, coarse, refine


@torch.no_grad()
def _binary_anchor_and_samples(
    *,
    coarse,
    refine,
    past: torch.Tensor,
    n_prob: int,
    sampler: str,
    steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return staged anchor (deterministic) + S probabilistic draws."""
    torch.manual_seed(seed)
    try:
        anchor_out = generate_staged_forecast(
            coarse, refine, past, vertical_dual=False, sampler="anchor",
        )
    except Exception:
        # Some patch-refine builds only expose stochastic samplers.
        anchor_out = generate_staged_forecast(
            coarse, refine, past, vertical_dual=False,
            sampler=sampler, num_inference_steps=max(1, min(5, steps)),
        )
    anchor = anchor_out["prediction_global_norm"].detach().cpu().numpy().astype(np.float32)
    draws = []
    for s_i in range(n_prob):
        torch.manual_seed(seed + 1009 * (s_i + 1))
        out = generate_staged_forecast(
            coarse, refine, past, vertical_dual=False,
            sampler=sampler, num_inference_steps=steps,
        )
        draws.append(out["prediction_global_norm"].detach().cpu().numpy().astype(np.float32))
    samples = np.stack(draws, axis=2)  # (B,V,S,H)
    return anchor, samples


def run_dataset(args: argparse.Namespace, dataset: str, device: torch.device) -> List[str]:
    disc_pack = np.load(args.disc_raw_dir / f"binary_ordinal_patch_refine_{dataset}.npz")
    all_indices = [int(i) for i in disc_pack["indices"].tolist()]
    rng = np.random.default_rng(args.seed + sum(ord(c) for c in dataset))
    n_pick = min(int(args.n_windows), len(all_indices))
    pick_pos = np.sort(rng.choice(len(all_indices), size=n_pick, replace=False))
    indices = [all_indices[int(i)] for i in pick_pos.tolist()]
    print(f"[{dataset}] panels for pool indices {indices}", flush=True)

    mmpd_full = _mmpd_pack(args.mmpd_output_root, dataset)
    mmpd = subset_pack_by_pool_indices(mmpd_full, np.asarray(indices, dtype=np.int64))

    ckpt_root = (REPO_ROOT / DEFAULT_BINARY[dataset]).resolve()
    run, coarse, refine = _load_models(dataset, ckpt_root, device)
    pool, _, _, _, _ = load_tsf_pack_pool(
        dataset,
        run_variate_indices(run),
        lookback=336,
        horizon=96,
        train_stride=run_train_stride(run),
        test_stride=max(1, int(args.pack_test_stride)),
        pack_splits=parse_pack_splits("test"),
        use_ordinal_window_norm=False,
    )
    loader = DataLoader(Subset(pool, indices), batch_size=len(indices), shuffle=False)
    past_t, future_t = next(iter(loader))
    past_t = past_t.to(device)
    overlap = int(refine.config.lookback_overlap)
    y_true_bin = future_t[..., overlap:].cpu().numpy().astype(np.float32) if overlap else future_t.cpu().numpy().astype(np.float32)

    binary_anchor, binary_samples = _binary_anchor_and_samples(
        coarse=coarse, refine=refine, past=past_t,
        n_prob=int(args.n_prob_samples), sampler=args.sampler,
        steps=int(args.num_inference_steps), seed=int(args.seed),
    )

    # Align MMPD into binary dataset-z for the shared panel.
    args_ns = argparse.Namespace(
        mmpd_to_binary_dataset_norm=True,
        mmpd_instance_norm=True,
        mmpd_ordinal_norm=False,
    )
    # binary_mmpd_train_scaler_map expects pipeline-ish args; reuse disc defaults.
    from utils.eval_discriminator_texture_staged_vs_mmpd import parse_args as disc_parse

    saved = sys.argv
    sys.argv = [saved[0], "--datasets", dataset, "--lookback", "336", "--horizon", "96"]
    try:
        disc_args = disc_parse()
    finally:
        sys.argv = saved
    disc_args.mmpd_instance_norm = True
    disc_args.mmpd_ordinal_norm = False
    scalers = binary_mmpd_train_scaler_map(disc_args, run)

    mmpd_anchor_z, _ = align_mmpd_to_binary_dataset_norm(
        binary_y_true=y_true_bin,
        mmpd_y_true=mmpd["y_true"].astype(np.float32),
        mmpd_fakes=mmpd["deterministic"].astype(np.float32),
        **scalers,
    )
    mmpd_draws = []
    for s_i in range(mmpd["samples"].shape[2]):
        aligned, _ = align_mmpd_to_binary_dataset_norm(
            binary_y_true=y_true_bin,
            mmpd_y_true=mmpd["y_true"].astype(np.float32),
            mmpd_fakes=mmpd["samples"][:, :, s_i, :].astype(np.float32),
            **scalers,
        )
        mmpd_draws.append(aligned)
    mmpd_samples_z = np.stack(mmpd_draws, axis=2).astype(np.float32)

    past_np = past_t.detach().cpu().numpy().astype(np.float32)
    paths = generate_binary_vs_mmpd_anchor_prob_panels(
        dataset=dataset,
        out_dir=args.output_dir / "panels",
        window_indices=list(range(len(indices))),
        y_true=y_true_bin,
        past=past_np,
        binary_anchor=binary_anchor,
        binary_samples=binary_samples,
        mmpd_anchor=mmpd_anchor_z,
        mmpd_samples=mmpd_samples_z,
        pool_indices=indices,
    )
    return paths


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}")
    all_paths: List[str] = []
    for dataset in args.datasets:
        paths = run_dataset(args, dataset, device)
        all_paths.extend(paths)
        print(f"[{dataset}] wrote {len(paths)} panels", flush=True)
    manifest = {
        "datasets": list(args.datasets),
        "n_windows": args.n_windows,
        "n_prob_samples": args.n_prob_samples,
        "output_dir": str(args.output_dir),
        "panels": all_paths,
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
