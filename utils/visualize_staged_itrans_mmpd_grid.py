#!/usr/bin/env python3
"""Side-by-side forecast grid: binary 2-stage, iTrans guidance, MMPD.

For each dataset, one test window, all 7 variates. Layout: rows = model family,
columns = variate. Each cell shows context, GT, 1 deterministic anchor, and
N probabilistic samples (default 3) on shared axes.

Auto-picks the newest staged checkpoint with a successful eval partial, the
matching finetuned iTrans guidance from the same run, and MMPD from the aligned
matrix (06-01-mmpd-binary-aligned by default).

Example:
  python utils/visualize_staged_itrans_mmpd_grid.py
  python utils/visualize_staged_itrans_mmpd_grid.py --datasets ETTh1 traffic --test-index 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.guidance import iTransformerGuidance
from models.diffusion_tsf.train_multivariate_pipeline import (
    load_dataset,
    load_itransformer_from_checkpoint,
)
from models.diffusion_tsf.visualize_comparison import denorm
from utils.eval_mmpd_gaussian_anchor import (
    AnchorRun,
    DEFAULT_MMPD_DATA,
    DEFAULT_MMPD_REPO,
    ensure_mmpd_repo,
    mmpd_data_split,
    mmpd_env_for_run,
    mmpd_staged_filename_for_run,
    resolve_mmpd_checkpoint,
    run_test_stride,
    run_train_stride,
    run_variate_indices,
    stage_mmpd_dataset_for_run,
)
from utils.visualize_staged_forecast import (
    _build_pipeline_state,
    _itrans_forward,
    _load_staged_bundle,
    _load_staged_diffusion,
    _resolve_itrans_paths,
    _staged_anchor_and_samples,
    _window_lengths,
)

DEFAULT_DATASETS = ["ETTh1", "ETTm2", "solar_Alabama", "traffic"]
DEFAULT_MMPD_OUTPUT = REPO_ROOT / "results" / "datasets" / "06-01-mmpd-binary-aligned"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "staged_itrans_mmpd_grid"
PROB_COLORS = ["#E91E63", "#FF9800", "#4CAF50"]


@dataclass
class StagedRun:
    ckpt_dir: Path
    bundle: Dict[str, Any]
    eval_partial: Optional[Path]


def _staged_run_basename(ckpt_dir: Path) -> str:
    return ckpt_dir.name


def _successful_staged_partials(datasets_root: Path, dataset: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for path in datasets_root.glob(f"*-{dataset}-binary_dual_scale_staged/partials/{dataset}_staged_anchor.json"):
        run_name = path.parent.parent.name
        if "smoke" in run_name:
            continue
        out[run_name] = path
    return out


def discover_staged_run(
    dataset: str,
    ckpt_root: Path,
    datasets_root: Path,
) -> StagedRun:
    partials = _successful_staged_partials(datasets_root, dataset)
    candidates: List[Tuple[float, Path, bool]] = []

    for ckpt_dir in ckpt_root.iterdir():
        if not ckpt_dir.is_dir():
            continue
        name = ckpt_dir.name
        if dataset not in name or "binary_dual_scale_staged" not in name:
            continue
        if any(x in name for x in ("smoke", "patch48", "best_scale")):
            continue
        try:
            _load_staged_bundle(ckpt_dir, dataset)
        except FileNotFoundError:
            continue
        has_partial = _staged_run_basename(ckpt_dir) in partials
        candidates.append((ckpt_dir.stat().st_mtime, ckpt_dir, has_partial))

    if not candidates:
        raise FileNotFoundError(
            f"No eval-ready binary_dual_scale_staged checkpoint for {dataset} under {ckpt_root}"
        )

    with_partial = [c for c in candidates if c[2]]
    pool = with_partial or candidates
    _, ckpt_dir, _ = max(pool, key=lambda x: x[0])
    partial = partials.get(_staged_run_basename(ckpt_dir))
    return StagedRun(ckpt_dir=ckpt_dir, bundle=_load_staged_bundle(ckpt_dir, dataset), eval_partial=partial)


def anchor_run_from_staged(run: StagedRun) -> AnchorRun:
    meta = run.bundle["fine_metadata"]
    subset_id = meta["subset_id"]
    itrans_pt = run.ckpt_dir / f"{subset_id}_itransformer_finetuned.pt"
    if not itrans_pt.is_file():
        alt = run.ckpt_dir / f"{subset_id}_itrans_ft_hp_best.pt"
        if alt.is_file():
            itrans_pt = alt
    return AnchorRun(
        variant="binary",
        dataset=meta["dataset_name"],
        root=run.ckpt_dir,
        subset_dir=run.bundle["fine_pt"].parent,
        best_pt=run.bundle["fine_pt"],
        itrans_pt=itrans_pt,
        metadata=meta,
    )


def _parse_mmpd_split(value: str) -> List[float]:
    parts = [float(x) for x in str(value).split(",") if x.strip()]
    if parts and all(x > 1 for x in parts):
        return [int(x) for x in parts]
    return parts


def _to_channel_first(x: torch.Tensor, n_vars: int) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError(f"expected 2D forecast tensor, got {tuple(x.shape)}")
    if x.shape[0] == n_vars:
        return x
    if x.shape[1] == n_vars:
        return x.transpose(0, 1)
    raise ValueError(f"cannot map forecast shape {tuple(x.shape)} to {n_vars} channels")


def _mmpd_args(
    *,
    mmpd_output_root: Path,
    mmpd_data_dir: Path,
    lookback: int,
    horizon: int,
    patch_size: int,
    sample_num: int,
    num_sampling_steps: int,
    gmm_components: int,
    gmm_iterations: int,
    gpu: int,
    cpu: bool,
) -> argparse.Namespace:
    return argparse.Namespace(
        mmpd_output_root=mmpd_output_root,
        mmpd_data_dir=mmpd_data_dir,
        lookback=lookback,
        horizon=horizon,
        patch_size=patch_size,
        sample_num=sample_num,
        num_sampling_steps=num_sampling_steps,
        gmm_components=gmm_components,
        gmm_iterations=gmm_iterations,
        gpu=gpu,
        cpu=cpu,
        num_workers=0,
    )


def _ensure_mmpd_imports(mmpd_repo: Path) -> None:
    repo = str(REPO_ROOT)
    mmpd = str(mmpd_repo.resolve())
    for p in (repo, mmpd):
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, mmpd)
    sys.path.insert(1, repo)

    utils_mod = sys.modules.get("utils")
    if utils_mod is not None:
        mod_path = str(getattr(utils_mod, "__file__", "") or "")
        if mod_path and REPO_ROOT.as_posix() in mod_path.replace("\\", "/"):
            for key in list(sys.modules):
                if key == "utils" or key.startswith("utils."):
                    del sys.modules[key]


def mmpd_predict_single(
    run: AnchorRun,
    test_index: int,
    *,
    mmpd_repo: Path,
    mmpd_data_dir: Path,
    mmpd_output_root: Path,
    lookback: int,
    horizon: int,
    patch_size: int,
    sample_num: int,
    num_sampling_steps: int,
    gmm_components: int,
    gmm_iterations: int,
    gpu: int,
    cpu: bool,
    seed: int,
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, Any]:
    """Return (det, prob_samples, context, scaler) in MMPD global-z space."""
    stage_mmpd_dataset_for_run(mmpd_data_dir, run)
    args = _mmpd_args(
        mmpd_output_root=mmpd_output_root,
        mmpd_data_dir=mmpd_data_dir,
        lookback=lookback,
        horizon=horizon,
        patch_size=patch_size,
        sample_num=sample_num,
        num_sampling_steps=num_sampling_steps,
        gmm_components=gmm_components,
        gmm_iterations=gmm_iterations,
        gpu=gpu,
        cpu=cpu,
    )
    ckpt_path, data_name = resolve_mmpd_checkpoint(args, run)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"MMPD checkpoint missing: {ckpt_path}")

    ensure_mmpd_repo(mmpd_repo, update=False)
    old_cwd = os.getcwd()
    old_env = os.environ.copy()
    os.environ.update(mmpd_env_for_run(run))
    os.chdir(mmpd_repo)
    _ensure_mmpd_imports(mmpd_repo)
    try:
        from data_provider.dataset_mts import Dataset_MTS
        from exp.exp_forecast import Exp_Forecast
        from exp.normalization import denormalize, get_statistics, normalize

        dataset = run.dataset
        data_path = mmpd_staged_filename_for_run(run)
        data_dim = len(run_variate_indices(run))
        ns = SimpleNamespace(
            dataset=data_name,
            root_path=str(mmpd_data_dir),
            data_path=data_path,
            data_split=mmpd_data_split(dataset, mmpd_data_dir),
            output_root=str(mmpd_output_root / "mmpd_out"),
            lookback=lookback,
            horizon=horizon,
            patch_size=patch_size,
            data_dim=data_dim,
            sample_num=sample_num,
            num_sampling_steps=num_sampling_steps,
            gmm_components=gmm_components,
            gmm_iterations=gmm_iterations,
            batch_size=1,
            num_workers=0,
            gpu=gpu,
            cpu=cpu,
        )

        exp_args = SimpleNamespace(
            data=ns.dataset,
            root_path=ns.root_path,
            data_path=ns.data_path,
            data_split=_parse_mmpd_split(str(ns.data_split)),
            output_root=ns.output_root,
            backbone="Decoder",
            in_len=lookback,
            out_len=horizon,
            patch_size=patch_size,
            data_dim=data_dim,
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
            num_workers=0,
            batch_size=1,
            train_epochs=20,
            patience=5,
            learning_rate=1e-4,
            lradj="cosine",
            test_batch_num=-1,
            testing=True,
            prob_pred=True,
            sample_num=sample_num,
            num_sampling_steps=str(num_sampling_steps),
            temperature=1.0,
            gmm_components=gmm_components,
            prior_pi_decay=0.5,
            prior_precision_shape=1e2,
            gmm_iterations=gmm_iterations,
            use_gpu=(torch.cuda.is_available() and not cpu),
            gpu=gpu,
            use_multi_gpu=False,
            devices="0,1,2,3",
        )

        exp = Exp_Forecast(exp_args)
        state = torch.load(ckpt_path, map_location="cpu")
        model_state = exp.model.state_dict()
        for k, v in state.items():
            if "gen_diffusion" not in k:
                model_state[k] = v
        exp.model.load_state_dict(model_state)
        exp.model.eval()
        device = exp.device

        test_data = Dataset_MTS(
            root_path=ns.root_path,
            data_path=ns.data_path,
            flag="test",
            size=[lookback, horizon],
            data_split=exp_args.data_split,
        )
        if test_index < 0 or test_index >= len(test_data):
            raise IndexError(f"MMPD test_index {test_index} out of range [0, {len(test_data)})")

        scaler = test_data.scaler

        torch.manual_seed(seed + test_index)
        batch_x, batch_y = test_data[test_index]
        batch_x = torch.as_tensor(batch_x).unsqueeze(0).float().to(device)
        batch_y = torch.as_tensor(batch_y).unsqueeze(0).float().to(device)
        batch_x = rearrange(batch_x, "b l d -> b d l")
        batch_y = rearrange(batch_y, "b l d -> b d l")

        x_shift, x_scale = get_statistics(batch_x)
        normed_x = normalize(batch_x, x_shift, x_scale)
        with torch.no_grad():
            det, _modes, samples = exp.model.predict(
                normed_x,
                prob_pred=True,
                sample_num=sample_num,
                temperature=1.0,
                gmm=True,
                gmm_components=gmm_components,
                prior_pi_decay=0.5,
                prior_precision_shape=1e2,
                gmm_iterations=gmm_iterations,
            )

        det_dn = _to_channel_first(
            denormalize(det, x_shift, x_scale).detach().cpu().squeeze(0),
            data_dim,
        )
        samples_dn = denormalize(samples, x_shift, x_scale).detach().cpu()
        if samples_dn.dim() == 4:
            # (B, D, N, L)
            samples_ch = samples_dn[0]
            prob_list = [
                _to_channel_first(samples_ch[:, i, :], data_dim)
                for i in range(samples_ch.shape[1])
            ]
        elif samples_dn.dim() == 3:
            # (D, N, L)
            prob_list = [
                _to_channel_first(samples_dn[:, i, :], data_dim)
                for i in range(samples_dn.shape[1])
            ]
        else:
            raise ValueError(f"unexpected MMPD samples shape {tuple(samples_dn.shape)}")
        context_z = _to_channel_first(
            batch_x.detach().cpu().squeeze(0),
            data_dim,
        )
        return det_dn, prob_list, context_z, scaler
    finally:
        os.chdir(old_cwd)
        os.environ.clear()
        os.environ.update(old_env)


def _future_horizon_slice(future: torch.Tensor, horizon: int) -> torch.Tensor:
    """Pure forecast tail (drops overlap prefix when future is (K+horizon))."""
    return future[:, -horizon:]


def _align_forecast(pred: torch.Tensor, horizon: int, overlap_k: int) -> torch.Tensor:
    """Match model output length to the plotted horizon (no double overlap trim)."""
    if pred.shape[-1] == horizon:
        return pred
    if overlap_k > 0 and pred.shape[-1] == horizon + overlap_k:
        return pred[..., overlap_k:]
    if pred.shape[-1] > horizon:
        return pred[..., -horizon:]
    return pred


def _mmpd_global_z_to_physical(
    z: torch.Tensor,
    scaler,
) -> torch.Tensor:
    """MMPD sklearn-scaled space (C, T) -> physical units."""
    arr = z.detach().cpu().numpy().T
    phys = scaler.inverse_transform(arr).T
    return torch.tensor(phys, dtype=torch.float32)


def plot_model_grid(
    dataset: str,
    run: StagedRun,
    output_dir: Path,
    test_index: Optional[int],
    prob_samples: int,
    prob_sampler: str,
    prob_steps: int,
    seed: int,
    device: torch.device,
    *,
    mmpd_repo: Path,
    mmpd_data_dir: Path,
    mmpd_output_root: Path,
    gmm_components: int,
    gmm_iterations: int,
    gpu: int,
    cpu: bool,
) -> Path:
    sub = run.bundle
    meta = sub["fine_metadata"]
    subset_id = meta["subset_id"]
    variate_indices = sub["variate_indices"]
    n_vars = len(variate_indices)
    state = _build_pipeline_state(run.ckpt_dir, dataset, subset_id)
    lookback, horizon = _window_lengths(dataset, state)

    train_stride = run_train_stride(anchor_run_from_staged(run))
    test_stride = run_test_stride(anchor_run_from_staged(run))
    _, _, test_ds, norm_stats = load_dataset(
        dataset,
        variate_indices,
        stride=train_stride,
        test_stride=test_stride,
        lookback=lookback,
        horizon=horizon,
    )
    n_test = len(test_ds)
    if n_test == 0:
        raise ValueError(f"Empty test set for {dataset}")

    rng = random.Random(seed)
    if test_index is None:
        test_index = rng.randrange(n_test)

    mean = torch.tensor(norm_stats["mean"], dtype=torch.float32)
    std = torch.tensor(norm_stats["std"], dtype=torch.float32)

    guidance_path, _ = _resolve_itrans_paths(run.ckpt_dir, subset_id)
    if guidance_path is None:
        raise FileNotFoundError(f"Missing iTrans guidance for {subset_id} under {run.ckpt_dir}")

    guidance_model = load_itransformer_from_checkpoint(str(guidance_path), n_vars, device)
    itrans_guidance = iTransformerGuidance(guidance_model)
    coarse_model = _load_staged_diffusion(
        state, "coarse", sub["coarse_pt"], itrans_guidance, n_vars, device
    )
    fine_model = _load_staged_diffusion(
        state, "fine", sub["fine_pt"], itrans_guidance, n_vars, device
    )

    past, future = test_ds[test_index]
    past_t = past.unsqueeze(0).to(device)

    with torch.no_grad():
        guidance_pred = _itrans_forward(guidance_model, past_t, horizon, device)

    staged_anchor, staged_probs = _staged_anchor_and_samples(
        coarse_model,
        fine_model,
        past_t,
        prob_samples=prob_samples,
        prob_sampler=prob_sampler,
        prob_steps=prob_steps,
        seed=seed,
        test_index=test_index,
    )

    anchor_run = anchor_run_from_staged(run)
    mmpd_det, mmpd_probs, _mmpd_ctx_z, mmpd_scaler = mmpd_predict_single(
        anchor_run,
        test_index,
        mmpd_repo=mmpd_repo,
        mmpd_data_dir=mmpd_data_dir,
        mmpd_output_root=mmpd_output_root,
        lookback=lookback,
        horizon=horizon,
        patch_size=12,
        sample_num=prob_samples,
        num_sampling_steps=prob_steps,
        gmm_components=gmm_components,
        gmm_iterations=gmm_iterations,
        gpu=gpu,
        cpu=cpu,
        seed=seed,
    )

    overlap_k = int(getattr(coarse_model.config, "lookback_overlap", 0) or 0)
    context_len = min(horizon * 2, lookback)
    t_past = np.arange(-context_len, 0)
    future_slice = _future_horizon_slice(future, horizon)
    t_fut_len = int(future_slice.shape[-1])
    t_future = np.arange(0, t_fut_len)

    past_dn = denorm(past, mean, std)
    gt_dn = denorm(future_slice, mean, std)
    guidance_dn = denorm(_align_forecast(guidance_pred, horizon, overlap_k), mean, std)
    staged_anchor_dn = denorm(_align_forecast(staged_anchor, horizon, overlap_k), mean, std)
    staged_prob_dns = [
        denorm(_align_forecast(p, horizon, overlap_k), mean, std) for p in staged_probs
    ]

    mmpd_det_phys = _mmpd_global_z_to_physical(
        _align_forecast(mmpd_det, horizon, 0), mmpd_scaler
    )
    mmpd_prob_phys = [
        _mmpd_global_z_to_physical(_align_forecast(p, horizon, 0), mmpd_scaler)
        for p in mmpd_probs
    ]

    var_names = sub["variate_names"] or [f"v{i}" for i in range(n_vars)]
    row_labels = [
        f"Binary 2-stage ({run.ckpt_dir.name})",
        f"iTrans guidance ({guidance_path.name})",
        f"MMPD ({mmpd_output_root.name})",
    ]

    fig, axes = plt.subplots(
        3,
        n_vars,
        figsize=(5.5 * n_vars, 2.8 * 3),
        squeeze=False,
        constrained_layout=True,
    )

    for row, (label, anchor_line, prob_lines, show_prob) in enumerate(
        [
            (row_labels[0], staged_anchor_dn, staged_prob_dns, True),
            (row_labels[1], guidance_dn, [], False),
            (row_labels[2], mmpd_det_phys, mmpd_prob_phys, True),
        ]
    ):
        for col in range(n_vars):
            ax = axes[row, col]
            ax.plot(
                t_past,
                past_dn[col, -context_len:].numpy(),
                color="#424242",
                lw=1.2,
                alpha=0.85,
                label="Context" if row == 0 and col == 0 else "",
            )
            ax.plot(
                t_future,
                gt_dn[col].numpy(),
                color="#2196F3",
                lw=1.8,
                label="GT" if row == 0 and col == 0 else "",
            )
            ax.plot(
                t_future,
                anchor_line[col].numpy(),
                color="#6A1B9A",
                lw=1.5,
                label="Anchor" if row == 0 and col == 0 else "",
            )
            if show_prob:
                for i, prob in enumerate(prob_lines):
                    color = PROB_COLORS[i % len(PROB_COLORS)]
                    ax.plot(
                        t_future,
                        prob[col].numpy(),
                        color=color,
                        lw=1.1,
                        alpha=0.85,
                        label=f"Prob {i + 1}" if row == 0 and col == 0 else "",
                    )
            ax.axvline(0, color="k", ls=":", alpha=0.25)
            ax.grid(alpha=0.2)
            if row == 0:
                ax.set_title(var_names[col], fontsize=9, fontweight="semibold")
            if col == 0:
                ax.set_ylabel(label.split(" (")[0], fontsize=8)
            ax.tick_params(labelsize=7)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, 1.02))

    partial_note = ""
    if run.eval_partial is not None:
        partial_note = f" | eval={run.eval_partial.parent.parent.name}"
    fig.suptitle(
        f"{dataset} / {subset_id} — test idx {test_index} | "
        f"anchor + {prob_sampler}×{prob_samples} (steps={prob_steps}){partial_note}",
        fontsize=11,
        fontweight="bold",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"compare_grid_{dataset}_{subset_id}_idx{test_index}_{prob_sampler}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--ckpt-root", type=Path, default=REPO_ROOT / "results" / "ckpts")
    parser.add_argument("--datasets-root", type=Path, default=REPO_ROOT / "results" / "datasets")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mmpd-repo", type=Path, default=DEFAULT_MMPD_REPO)
    parser.add_argument("--mmpd-data-dir", type=Path, default=DEFAULT_MMPD_DATA)
    parser.add_argument("--mmpd-output-root", type=Path, default=DEFAULT_MMPD_OUTPUT)
    parser.add_argument("--test-index", type=int, default=None)
    parser.add_argument("--prob-samples", type=int, default=3)
    parser.add_argument("--prob-sampler", type=str, default="dpmpp")
    parser.add_argument("--prob-steps", type=int, default=20)
    parser.add_argument("--gmm-components", type=int, default=10)
    parser.add_argument("--gmm-iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="ETTh1 only, CPU ok")
    args = parser.parse_args()

    if args.smoke_test:
        args.datasets = ["ETTh1"]
        args.cpu = True
        args.prob_samples = 1
        args.prob_steps = 5

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[viz] device={device}", flush=True)

    manifest: Dict[str, Any] = {"plots": {}, "runs": {}}
    for dataset in args.datasets:
        run = discover_staged_run(dataset, args.ckpt_root.resolve(), args.datasets_root.resolve())
        print(
            f"[{dataset}] staged={run.ckpt_dir.name} "
            f"partial={'yes' if run.eval_partial else 'no'}",
            flush=True,
        )
        out = plot_model_grid(
            dataset,
            run,
            args.output_dir.resolve(),
            args.test_index,
            args.prob_samples,
            args.prob_sampler,
            args.prob_steps,
            args.seed,
            device,
            mmpd_repo=args.mmpd_repo.resolve(),
            mmpd_data_dir=args.mmpd_data_dir.resolve(),
            mmpd_output_root=args.mmpd_output_root.resolve(),
            gmm_components=args.gmm_components,
            gmm_iterations=args.gmm_iterations,
            gpu=args.gpu,
            cpu=args.cpu or not torch.cuda.is_available(),
        )
        manifest["plots"][dataset] = str(out)
        manifest["runs"][dataset] = {
            "staged_ckpt": str(run.ckpt_dir),
            "eval_partial": str(run.eval_partial) if run.eval_partial else None,
            "subset_id": run.bundle["fine_metadata"]["subset_id"],
        }
        print(f"[{dataset}] saved {out}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[viz] manifest -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
