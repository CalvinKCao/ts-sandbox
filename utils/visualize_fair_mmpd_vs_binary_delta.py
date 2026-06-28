#!/usr/bin/env python3
"""Top test windows by per-window anchor_mse / CRPS delta (binary − fair MMPD).

Uses saved eval NPZ when available; optional checkpoint inference for missing
binary eval artifacts and for coarse/fine 2D anchor decomposition plots.

Example (grad-accum 1.5× lr-hi vs fair MMPD, flat subsets):
  python utils/visualize_fair_mmpd_vs_binary_delta.py \\
    --binary-config binary_anchor_stationary_flat_subsets_grad_accum_150_lr_hi \\
    --mmpd-run results/datasets/06-16-mmpd-maskae-fair-13d \\
    --datasets weather,traffic,exchange_rate,solar_Alabama,electricity,ETTh2,ETTh1 \\
    --infer-binary \\
    --output-dir reports/fair_mmpd_vs_grad_accum_150_lr_hi

Explicit paths (skip auto-discovery):
  python utils/visualize_fair_mmpd_vs_binary_delta.py \\
    --binary-results-dir results/datasets/06-19-...-ETTh1-..._lr_lo \\
    --binary-ckpt-dir results/ckpts/06-14-...-ETTh1-..._lr_hi \\
    --mmpd-run results/datasets/06-16-mmpd-maskae-fair-13d \\
    --dataset ETTh1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.guidance import iTransformerGuidance
from models.diffusion_tsf.pipeline.config import load_experiment_config
from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import patch_stage_globals
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.train_multivariate_pipeline import (
    generate_dataset_job,
    load_dataset,
    load_diffusion_state_keep_attached_guidance,
    load_itransformer_from_checkpoint,
    create_diffusion_model,
    anchor_kwargs_from_params,
)
from models.diffusion_tsf.visualize_comparison import denorm
from utils.eval_mmpd_gaussian_anchor import (
    _load_data_subset_policy,
    crps_gr,
    resolve_subset_meta_for_dataset,
)

DEFAULT_MMPD_RUN = REPO_ROOT / "results" / "datasets" / "06-16-mmpd-maskae-fair-13d"
DEFAULT_SUBSET_CONFIG = REPO_ROOT / "configs" / "binary_anchor_stationary_flat_subsets.yaml"
DEFAULT_BINARY_CONFIG = "binary_anchor_stationary_flat_subsets_grad_accum_150_lr_hi"
FALLBACK_BINARY_CONFIG = "binary_anchor_stationary_flat_subsets_grad_accum_150_lr_lo"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "fair_mmpd_vs_grad_accum_150_lr_hi"
EVAL_TEST_STRIDE = 4
TRAIN_STRIDE = 1
PROB_COLORS = ["#E91E63", "#FF9800", "#4CAF50"]
DEFAULT_DATASETS = (
    "weather",
    "traffic",
    "exchange_rate",
    "solar_Alabama",
    "electricity",
    "ETTh2",
    "ETTh1",
)


@dataclass
class BinaryRun:
    results_dir: Path
    ckpt_dir: Path
    metrics: Dict[str, float]
    config_suffix: str


@dataclass
class AlignedPack:
    indices: np.ndarray
    y_true: np.ndarray
    binary_det: np.ndarray
    binary_samples: np.ndarray
    mmpd_det: np.ndarray
    mmpd_samples: np.ndarray


@dataclass
class Anchor2DMaps:
    """Native coarse/fine CDF maps (V, H, W) plus per-column signed fine decode (V, W)."""

    coarse: np.ndarray
    fine: np.ndarray
    fine_1d: np.ndarray


@dataclass
class BinaryStagedInference:
    ckpt_dir: Path
    dataset: str
    config_name: str
    device: torch.device
    subset_config: Path
    seed: int = 2026
    _bundle: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)
    _coarse_model: Any = field(default=None, init=False, repr=False)
    _fine_model: Any = field(default=None, init=False, repr=False)
    _test_ds: Any = field(default=None, init=False, repr=False)
    _norm_stats: Optional[Dict[str, torch.Tensor]] = field(default=None, init=False, repr=False)
    _horizon: int = field(default=96, init=False, repr=False)

    def _ensure_loaded(self) -> None:
        if self._coarse_model is not None:
            return
        self._bundle = _load_staged_bundle(self.ckpt_dir, self.dataset)
        subset_id = self._bundle["subset_id"]
        variate_indices = self._bundle["variate_indices"]
        n_vars = len(variate_indices)
        state = _build_pipeline_state(
            self.ckpt_dir, self.dataset, subset_id, self.config_name,
        )
        lookback, horizon = _window_lengths(self.dataset, state)
        self._horizon = horizon

        data_subset = self._bundle["fine_metadata"].get("data_subset") or {}
        test_stride = int(data_subset.get("test_stride", EVAL_TEST_STRIDE))
        _, _, test_ds, norm_stats = load_dataset(
            self.dataset,
            variate_indices,
            stride=TRAIN_STRIDE,
            test_stride=test_stride,
            lookback=lookback,
            horizon=horizon,
        )
        self._test_ds = test_ds
        self._norm_stats = {
            "mean": torch.tensor(norm_stats["mean"], dtype=torch.float32),
            "std": torch.tensor(norm_stats["std"], dtype=torch.float32),
        }

        guidance_path = self.ckpt_dir / f"{subset_id}_itransformer_finetuned.pt"
        if not guidance_path.is_file():
            raise FileNotFoundError(
                f"Missing {guidance_path.name} under {self.ckpt_dir}"
            )
        guidance_model = load_itransformer_from_checkpoint(
            str(guidance_path), n_vars, self.device,
        )
        itrans_guidance = iTransformerGuidance(guidance_model)
        self._coarse_model = _load_staged_diffusion(
            state, "coarse", self._bundle["coarse_pt"], itrans_guidance, n_vars, self.device,
        )
        self._fine_model = _load_staged_diffusion(
            state, "fine", self._bundle["fine_pt"], itrans_guidance, n_vars, self.device,
        )

    def norm_stats(self) -> Dict[str, torch.Tensor]:
        self._ensure_loaded()
        assert self._norm_stats is not None
        return self._norm_stats

    def infer_on_indices(
        self,
        window_indices: Sequence[int],
        *,
        prob_draws: int,
        batch_size: int = 4,
        prob_steps: int = 20,
        prob_sampler: str = "dpmpp",
    ) -> Dict[str, np.ndarray]:
        """Run staged anchor + probabilistic sampling on test windows."""
        self._ensure_loaded()
        assert self._test_ds is not None
        assert self._coarse_model is not None and self._fine_model is not None

        idx_list = [int(i) for i in window_indices]
        subset = Subset(self._test_ds, idx_list)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

        y_true_all: List[np.ndarray] = []
        det_all: List[np.ndarray] = []
        sample_all: List[np.ndarray] = []
        prob_kwargs = {"sampler": prob_sampler, "num_inference_steps": prob_steps}

        with torch.no_grad():
            for batch_idx, (past, future) in enumerate(loader):
                past = past.to(self.device)
                future = future.to(self.device)
                K = int(getattr(self._coarse_model.config, "lookback_overlap", 0) or 0)
                if K > 0:
                    future = future[..., K:]
                y_true_all.append(future.cpu().numpy())

                torch.manual_seed(self.seed + batch_idx)
                coarse_det = self._coarse_model.generate(past, sampler="anchor")
                fine_det = self._fine_model.generate(
                    past,
                    sampler="anchor",
                    future_coarse_2d=coarse_det["future_2d_coarse"],
                )
                det_t = fine_det.get("prediction_global_norm", fine_det["prediction"])
                det_all.append(det_t.detach().cpu().numpy())

                batch_samples: List[np.ndarray] = []
                for sample_idx in range(prob_draws):
                    sample_seed = self.seed + batch_idx * 1009 + sample_idx * 17
                    torch.manual_seed(sample_seed)
                    coarse_s = self._coarse_model.generate(past, **prob_kwargs)
                    torch.manual_seed(sample_seed)
                    fine_s = self._fine_model.generate(
                        past,
                        future_coarse_2d=coarse_s["future_2d_coarse"],
                        **prob_kwargs,
                    )
                    pred = fine_s.get("prediction_global_norm", fine_s["prediction"])
                    batch_samples.append(pred.cpu().numpy())
                sample_all.append(np.stack(batch_samples, axis=2))

        return {
            "y_true": np.concatenate(y_true_all, axis=0),
            "deterministic": np.concatenate(det_all, axis=0),
            "samples": np.concatenate(sample_all, axis=0),
        }

    def infer_anchor_2d(self, window_index: int) -> Anchor2DMaps:
        """Return future coarse/fine CDF maps for one test window (model-native [0,1])."""
        self._ensure_loaded()
        assert self._test_ds is not None
        assert self._coarse_model is not None and self._fine_model is not None

        past, _ = self._test_ds[int(window_index)]
        past_t = past.unsqueeze(0).to(self.device)
        with torch.no_grad():
            torch.manual_seed(self.seed + int(window_index))
            coarse_det = self._coarse_model.generate(past_t, sampler="anchor")
            fine_det = self._fine_model.generate(
                past_t,
                sampler="anchor",
                future_coarse_2d=coarse_det["future_2d_coarse"],
            )
            coarse_2d = coarse_det["future_2d_coarse"][0].detach().cpu().numpy()
            fine_2d = fine_det["future_2d_fine"][0].detach().cpu().numpy()
            fine_1d = self._decode_fine_1d(fine_2d)
        return Anchor2DMaps(coarse=coarse_2d, fine=fine_2d, fine_1d=fine_1d)

    def _decode_fine_1d(self, fine_2d: np.ndarray) -> np.ndarray:
        """Signed within-bin residual decode per column, shape (V, W)."""
        assert self._fine_model is not None
        to_2d = self._fine_model.to_2d
        h = fine_2d.shape[-2]
        fine_t = torch.from_numpy(fine_2d).to(
            device=self.device, dtype=torch.float32,
        )
        if fine_t.dim() == 2:
            fine_t = fine_t.unsqueeze(0)
        v = fine_t.shape[0]
        fine_flat = fine_t.reshape(v, h, fine_t.shape[-1])
        residual_range = float(to_2d.max_scale) / float(h)
        fine_1d = to_2d._decode_occupancy_in_range(
            fine_flat,
            value_range=residual_range,
            cdf_decoder="mean",
        )
        return fine_1d.detach().cpu().numpy()


def _read_json(path: Path) -> Dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _newest_match(root: Path, pattern: str) -> Optional[Path]:
    matches = sorted(
        root.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _finished_binary_run(
    datasets_root: Path,
    ckpt_root: Path,
    dataset: str,
    config_suffix: str,
    *,
    results_dir_override: Optional[Path] = None,
    ckpt_dir_override: Optional[Path] = None,
) -> Optional[BinaryRun]:
    results_dir = results_dir_override
    if results_dir is None:
        results_dir = _newest_match(datasets_root, f"*-{dataset}-{config_suffix}")
    if results_dir is None or not results_dir.is_dir():
        return None

    partial = results_dir / "partials" / f"{dataset}_staged_anchor.json"
    anchor_npz = results_dir / "raw" / f"staged_anchor_{dataset}.npz"
    samples_npz = results_dir / "raw" / f"staged_dpmpp_samples_{dataset}.npz"
    has_npz = partial.is_file() and anchor_npz.is_file() and samples_npz.is_file()
    metrics: Dict[str, float] = {}
    if partial.is_file():
        metrics = _read_json(partial)
    elif not has_npz:
        return None

    ckpt_dir = ckpt_dir_override
    if ckpt_dir is None:
        ckpt_dir = _newest_match(ckpt_root, f"*-{dataset}-{config_suffix}")
    if ckpt_dir is None:
        ckpt_dir = results_dir

    return BinaryRun(
        results_dir=results_dir,
        ckpt_dir=ckpt_dir,
        metrics=metrics,
        config_suffix=config_suffix,
    )


def discover_binary_run(
    datasets_root: Path,
    ckpt_root: Path,
    dataset: str,
    *,
    config_suffix: str,
    allow_fallback: bool,
    fallback_config: str,
    results_dir_override: Optional[Path] = None,
    ckpt_dir_override: Optional[Path] = None,
) -> BinaryRun:
    run = _finished_binary_run(
        datasets_root,
        ckpt_root,
        dataset,
        config_suffix,
        results_dir_override=results_dir_override,
        ckpt_dir_override=ckpt_dir_override,
    )
    if run is not None:
        return run
    if allow_fallback and fallback_config != config_suffix:
        fb = _finished_binary_run(
            datasets_root,
            ckpt_root,
            dataset,
            fallback_config,
            results_dir_override=results_dir_override,
            ckpt_dir_override=ckpt_dir_override,
        )
        if fb is not None:
            return fb
    raise FileNotFoundError(
        f"No binary run dir for {dataset} ({config_suffix}); "
        f"set --binary-results-dir / --binary-ckpt-dir or use --allow-fallback-binary."
    )


def discover_binary_ckpt(
    ckpt_root: Path,
    dataset: str,
    config_suffix: str,
    *,
    ckpt_dir_override: Optional[Path] = None,
) -> Path:
    if ckpt_dir_override is not None:
        if not ckpt_dir_override.is_dir():
            raise FileNotFoundError(f"Missing --binary-ckpt-dir: {ckpt_dir_override}")
        return ckpt_dir_override
    match = _newest_match(ckpt_root, f"*-{dataset}-{config_suffix}")
    if match is None:
        raise FileNotFoundError(
            f"No checkpoint dir matching *-{dataset}-{config_suffix} under {ckpt_root}"
        )
    return match


def _load_staged_bundle(checkpoint_dir: Path, dataset: str) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for sub_dir in sorted(checkpoint_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        coarse_pt = sub_dir / "coarse" / "best.pt"
        fine_pt = sub_dir / "fine" / "best.pt"
        fine_meta_path = sub_dir / "fine" / "metadata.json"
        if not (coarse_pt.is_file() and fine_pt.is_file() and fine_meta_path.is_file()):
            continue
        with fine_meta_path.open(encoding="utf-8") as f:
            fine_meta = json.load(f)
        if fine_meta.get("dataset_name") != dataset:
            continue
        coarse_meta: Dict[str, Any] = {}
        coarse_meta_path = sub_dir / "coarse" / "metadata.json"
        if coarse_meta_path.is_file():
            with coarse_meta_path.open(encoding="utf-8") as f:
                coarse_meta = json.load(f)
        candidates.append(
            {
                "subset_id": fine_meta["subset_id"],
                "variate_indices": fine_meta["variate_indices"],
                "variate_names": fine_meta.get("variate_names", []),
                "coarse_pt": coarse_pt,
                "fine_pt": fine_pt,
                "fine_metadata": fine_meta,
                "coarse_metadata": coarse_meta,
                "root": checkpoint_dir,
            }
        )
    if not candidates:
        raise FileNotFoundError(
            f"No staged coarse/fine best.pt for dataset={dataset} under {checkpoint_dir}"
        )
    return candidates[0]


def _build_pipeline_state(
    checkpoint_dir: Path,
    dataset: str,
    subset_id: str,
    config_name: str,
) -> PipelineState:
    cfg_path = REPO_ROOT / "configs" / f"{config_name}.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing experiment config: {cfg_path}")
    cfg = load_experiment_config(str(cfg_path), cli_overrides={"dataset": dataset})
    state = PipelineState.from_config(cfg)
    state.checkpoint_dir = str(checkpoint_dir.resolve())
    state.dataset = dataset
    state.subset_id = subset_id
    return state


def _window_lengths(dataset: str, state: PipelineState) -> Tuple[int, int]:
    if dataset == "dalia":
        from models.diffusion_tsf.dalia_data import dalia_window_lengths

        return dalia_window_lengths()
    return state.lookback_length, state.forecast_length


def _load_staged_diffusion(
    state: PipelineState,
    stage: str,
    ckpt_path: Path,
    itrans_guidance: iTransformerGuidance,
    n_vars: int,
    device: torch.device,
) -> torch.nn.Module:
    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

    patch_stage_globals(pipeline_mod, state, stage, honor_dataset_windows=True)
    lookback, horizon = _window_lengths(state.dataset, state)
    meta_path = ckpt_path.parent / "metadata.json"
    tuned: Dict[str, Any] = {}
    if meta_path.is_file():
        with meta_path.open(encoding="utf-8") as f:
            tuned = json.load(f).get("tuned_params") or {}

    model = create_diffusion_model(
        n_variates=n_vars,
        lookback=lookback,
        horizon=horizon,
        guidance_model=itrans_guidance,
        diffusion_stage=stage,
        use_guidance_channel=state.use_guidance_channel,
        **anchor_kwargs_from_params(tuned),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    load_diffusion_state_keep_attached_guidance(model, ckpt["model_state_dict"])
    model.eval()
    return model


def load_mmpd_pack(mmpd_run: Path, dataset: str) -> Dict[str, np.ndarray]:
    path = mmpd_run / "raw" / f"mmpd_{dataset}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing MMPD eval npz: {path}")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def load_binary_pack(binary_run: BinaryRun, dataset: str) -> Optional[Dict[str, np.ndarray]]:
    anchor_path = binary_run.results_dir / "raw" / f"staged_anchor_{dataset}.npz"
    samples_path = binary_run.results_dir / "raw" / f"staged_dpmpp_samples_{dataset}.npz"
    if not (anchor_path.is_file() and samples_path.is_file()):
        return None
    with np.load(anchor_path) as anchor:
        det = anchor["deterministic"]
        y_true_anchor = anchor["y_true"]
    with np.load(samples_path) as samples:
        y_true = samples["y_true"]
        sample_arr = samples["samples"]
    if not np.allclose(y_true_anchor, y_true, rtol=1e-4, atol=1e-5):
        raise RuntimeError(f"{dataset}: staged anchor vs dpmpp y_true mismatch")
    return {
        "y_true": y_true,
        "deterministic": det,
        "samples": sample_arr,
    }


def align_packs(
    mmpd: Dict[str, np.ndarray],
    binary: Dict[str, np.ndarray],
    dataset: str,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> AlignedPack:
    indices = np.asarray(mmpd["indices"], dtype=np.int64)
    n_bin = binary["y_true"].shape[0]
    if indices.max(initial=0) >= n_bin:
        raise RuntimeError(
            f"{dataset}: MMPD index {int(indices.max())} out of binary range {n_bin}"
        )
    bin_y = binary["y_true"][indices]
    bin_det = binary["deterministic"][indices]
    bin_samples = binary["samples"][indices]
    m_y = mmpd["y_true"]
    if not np.allclose(bin_y, m_y, rtol=rtol, atol=atol):
        bad = int(np.argmax(np.abs(bin_y - m_y).reshape(len(indices), -1).mean(axis=1)))
        raise RuntimeError(
            f"{dataset}: y_true mismatch at window idx={int(indices[bad])} "
            f"(row {bad}); check eval_test_stride alignment."
        )
    return AlignedPack(
        indices=indices,
        y_true=m_y,
        binary_det=bin_det,
        binary_samples=bin_samples,
        mmpd_det=mmpd["deterministic"],
        mmpd_samples=mmpd["samples"],
    )


def align_mmpd_with_inferred_binary(
    mmpd: Dict[str, np.ndarray],
    inferred: Dict[str, np.ndarray],
    dataset: str,
) -> AlignedPack:
    indices = np.asarray(mmpd["indices"], dtype=np.int64)
    m_y = mmpd["y_true"]
    if not np.allclose(inferred["y_true"], m_y, rtol=1e-4, atol=1e-4):
        bad = int(np.argmax(np.abs(inferred["y_true"] - m_y).reshape(len(indices), -1).mean(axis=1)))
        raise RuntimeError(
            f"{dataset}: inferred y_true mismatch at row {bad} (window {int(indices[bad])})"
        )
    return AlignedPack(
        indices=indices,
        y_true=m_y,
        binary_det=inferred["deterministic"],
        binary_samples=inferred["samples"],
        mmpd_det=mmpd["deterministic"],
        mmpd_samples=mmpd["samples"],
    )


def per_window_anchor_mse(y_true: np.ndarray, det: np.ndarray) -> np.ndarray:
    return ((y_true - det) ** 2).mean(axis=(1, 2))


def per_window_crps(y_true: np.ndarray, samples: np.ndarray, *, chunk: int = 32) -> np.ndarray:
    batch = y_true.shape[0]
    out = np.empty(batch, dtype=np.float64)
    for start in range(0, batch, chunk):
        end = min(start + chunk, batch)
        yt = y_true[start:end]
        ss = samples[start:end].astype(np.float64)
        term1 = np.abs(ss - yt[:, :, None, :]).mean(axis=2)
        term2 = np.abs(ss[:, :, :, None, :] - ss[:, :, None, :, :]).mean(axis=(2, 3))
        out[start:end] = (term1 - 0.5 * term2).mean(axis=(1, 2))
    return out


def rank_top_k(delta: np.ndarray, top_k: int) -> np.ndarray:
    k = min(top_k, delta.size)
    if k <= 0:
        return np.array([], dtype=np.int64)
    order = np.argsort(-delta)
    return order[:k]


def _variate_names(dataset: str, n_vars: int, subset_config: Path) -> List[str]:
    job = generate_dataset_job(dataset)
    policy = _load_data_subset_policy(subset_config)
    subset = resolve_subset_meta_for_dataset(dataset, policy, seed=2026)
    indices = [int(i) for i in subset["variate_indices"][:n_vars]]
    all_names = job.get("variate_names") or []
    if all_names and max(indices, default=0) < len(all_names):
        return [str(all_names[i]) for i in indices]
    return [f"v{i}" for i in range(n_vars)]


def _load_test_context(
    dataset: str,
    window_indices: Sequence[int],
    subset_config: Path,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    policy = _load_data_subset_policy(subset_config)
    subset = resolve_subset_meta_for_dataset(dataset, policy, seed=2026)
    variate_indices = [int(i) for i in subset["variate_indices"]]
    _, _, test_ds, norm_stats = load_dataset(
        dataset,
        variate_indices,
        stride=TRAIN_STRIDE,
        test_stride=EVAL_TEST_STRIDE,
    )
    past_list = []
    future_list = []
    for idx in window_indices:
        past, future = test_ds[int(idx)]
        past_list.append(past)
        future_list.append(future)
    past_batch = torch.stack(past_list, dim=0)
    future_batch = torch.stack(future_list, dim=0)
    mean = torch.tensor(norm_stats["mean"], dtype=torch.float32)
    std = torch.tensor(norm_stats["std"], dtype=torch.float32)
    return past_batch, future_batch, {"mean": mean, "std": std}


def _plot_prob_lines(ax, t_future: np.ndarray, prob_lines: Sequence[torch.Tensor], col: int) -> None:
    for pi, prob in enumerate(prob_lines):
        color = PROB_COLORS[pi % len(PROB_COLORS)]
        ax.plot(
            t_future,
            prob[col].numpy(),
            color=color,
            lw=1.0,
            alpha=0.75,
            label=f"sample {pi + 1}" if col == 0 else "",
        )



def _rgb_coarse_fine_column_overlay(
    coarse_hw: np.ndarray,
    fine_signed: np.ndarray,
    *,
    residual_range: float,
) -> np.ndarray:
    """Grayscale coarse (H,W) with green/red column overlays from signed fine decode."""
    h, w = coarse_hw.shape
    gray = np.clip(coarse_hw, 0.0, 1.0)
    rgb = np.stack([gray, gray, gray], axis=-1)
    scale = max(residual_range, 1e-8)
    for col in range(w):
        fv = float(fine_signed[col])
        strength = min(abs(fv) / scale, 1.0)
        if strength < 0.02:
            continue
        if fv > 0:
            rgb[:, col, 0] *= 1.0 - 0.65 * strength
            rgb[:, col, 1] = np.clip(rgb[:, col, 1] + 0.55 * strength, 0, 1)
            rgb[:, col, 2] *= 1.0 - 0.65 * strength
        else:
            rgb[:, col, 0] = np.clip(rgb[:, col, 0] + 0.65 * strength, 0, 1)
            rgb[:, col, 1] *= 1.0 - 0.55 * strength
            rgb[:, col, 2] *= 1.0 - 0.55 * strength
    return np.clip(rgb, 0.0, 1.0)


def _plot_coarse_fine_summary(
    ax,
    *,
    coarse_map: np.ndarray,
    fine_1d: np.ndarray,
    gt_dn: torch.Tensor,
    mmpd_dn: torch.Tensor,
    var_idx: int,
    residual_range: float,
    title: str,
    ylabel: str = "",
) -> None:
    w_map = coarse_map.shape[-1]
    horizon = gt_dn.shape[-1]
    if w_map > horizon:
        coarse_map = coarse_map[..., -horizon:]
        fine_col = fine_1d[var_idx, -horizon:]
        w = horizon
    elif w_map < horizon:
        fine_col = fine_1d[var_idx]
        w = w_map
    else:
        fine_col = fine_1d[var_idx]
        w = w_map
    h, w = coarse_map.shape
    rgb = _rgb_coarse_fine_column_overlay(
        coarse_map, fine_col, residual_range=residual_range,
    )
    ax.imshow(
        rgb,
        aspect="auto",
        origin="lower",
        extent=[0, w, 0, h],
        interpolation="nearest",
    )
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_title(title, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel("forecast column", fontsize=7)

    ax_line = ax.twinx()
    x_cols = np.arange(w) + 0.5
    gt = gt_dn[var_idx].numpy()
    mmpd = mmpd_dn[var_idx].numpy()
    if len(gt) != w:
        x_line = np.linspace(0.5, w - 0.5, len(gt))
    else:
        x_line = x_cols
    ax_line.plot(x_line, gt, color="#2196F3", lw=1.8, label="GT", zorder=5)
    ax_line.plot(x_line, mmpd, color="#FF9800", lw=1.5, ls="--", label="MMPD", zorder=5)
    ax_line.set_ylabel("denorm value", fontsize=7)
    if var_idx == 0:
        ax_line.legend(fontsize=6, loc="upper right")


def _plot_2d_cdf_map(ax, data: np.ndarray, *, title: str, ylabel: str = "") -> None:
    h, w = data.shape
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        extent=[0, w, 0, h],
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel("forecast column", fontsize=7)
    fig = ax.figure
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_window_panel(
    *,
    dataset: str,
    window_idx: int,
    rank: int,
    metric: str,
    delta: float,
    binary_mse: float,
    mmpd_mse: float,
    binary_crps: float,
    mmpd_crps: float,
    past: torch.Tensor,
    future: torch.Tensor,
    norm: Dict[str, torch.Tensor],
    pack_row: int,
    aligned: AlignedPack,
    output_path: Path,
    prob_draws: int,
    context_len: int,
    binary_label: str,
    mmpd_label: str,
    subset_config: Path,
    anchor_2d: Optional[Anchor2DMaps] = None,
    residual_range: float = 1.0,
) -> None:
    mean, std = norm["mean"], norm["std"]
    n_vars = aligned.y_true.shape[1]
    horizon = aligned.y_true.shape[2]
    var_names = _variate_names(dataset, n_vars, subset_config)
    crps_panel = metric == "crps"

    past_dn = denorm(past, mean, std)
    gt_t = torch.from_numpy(aligned.y_true[pack_row]).to(dtype=torch.float32)
    gt_dn = denorm(gt_t, mean, std)

    bin_det_dn = denorm(torch.from_numpy(aligned.binary_det[pack_row]), mean, std)
    mmpd_det_dn = denorm(torch.from_numpy(aligned.mmpd_det[pack_row]), mean, std)

    n_s = min(prob_draws, aligned.binary_samples.shape[2])
    bin_probs = [
        denorm(torch.from_numpy(aligned.binary_samples[pack_row, :, si, :]), mean, std)
        for si in range(n_s)
    ]
    mmpd_probs = [
        denorm(torch.from_numpy(aligned.mmpd_samples[pack_row, :, si, :]), mean, std)
        for si in range(min(n_s, aligned.mmpd_samples.shape[2]))
    ]

    t_past = np.arange(-context_len, 0)
    t_future = np.arange(horizon)
    line_labels = [binary_label, mmpd_label]
    show_decomp = anchor_2d is not None
    n_2d_rows = 3 if show_decomp else 0  # coarse, fine, summary
    n_rows = 2 + n_2d_rows

    fig, axes = plt.subplots(
        n_rows,
        n_vars,
        figsize=(4.8 * n_vars, 2.5 * n_rows + 1.2),
        squeeze=False,
        constrained_layout=True,
    )
    metric_note = "prob samples only" if crps_panel else "anchor + prob samples"
    fig.suptitle(
        f"{dataset} | test window {window_idx} | rank {rank} by {metric} Δ ({metric_note})\n"
        f"Δ={delta:+.5f}  anchor_mse: bin={binary_mse:.5f} mmpd={mmpd_mse:.5f}  "
        f"crps: bin={binary_crps:.5f} mmpd={mmpd_crps:.5f}",
        fontsize=11,
    )

    for row, (label, det_dn, prob_dns) in enumerate(
        [
            (line_labels[0], bin_det_dn, bin_probs),
            (line_labels[1], mmpd_det_dn, mmpd_probs),
        ]
    ):
        for col in range(n_vars):
            ax = axes[row, col]
            ax.plot(
                t_past,
                past_dn[col, -context_len:].numpy(),
                color="#424242",
                lw=1.1,
                alpha=0.85,
            )
            ax.plot(t_future, gt_dn[col].numpy(), color="#2196F3", lw=1.8, label="GT")
            if not crps_panel:
                ax.plot(t_future, det_dn[col].numpy(), color="#6A1B9A", lw=1.6, label="anchor")
            _plot_prob_lines(ax, t_future, prob_dns, col)
            if row == 0:
                ax.set_title(var_names[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(label, fontsize=9)
            ax.grid(True, alpha=0.25)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper left")

    if show_decomp:
        for col in range(n_vars):
            _plot_2d_cdf_map(
                axes[2, col],
                anchor_2d.coarse[col],
                title=f"{var_names[col]} coarse CDF",
                ylabel="coarse bins" if col == 0 else "",
            )
            _plot_2d_cdf_map(
                axes[3, col],
                anchor_2d.fine[col],
                title=f"{var_names[col]} fine CDF",
                ylabel="fine bins" if col == 0 else "",
            )
            _plot_coarse_fine_summary(
                axes[4, col],
                coarse_map=anchor_2d.coarse[col],
                fine_1d=anchor_2d.fine_1d,
                gt_dn=gt_dn,
                mmpd_dn=mmpd_det_dn,
                var_idx=col,
                residual_range=residual_range,
                title=f"{var_names[col]} coarse + fine Δ",
                ylabel="summary" if col == 0 else "",
            )
        fig.text(
            0.5,
            0.01,
            "Summary row: B/W coarse occupancy; green column = fine adds, red = fine subtracts; "
            "lines = GT (blue) + MMPD (orange)",
            ha="center",
            fontsize=8,
            color="#555555",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def process_dataset(
    dataset: str,
    *,
    mmpd_run: Path,
    datasets_root: Path,
    ckpt_root: Path,
    output_dir: Path,
    top_k: int,
    prob_draws: int,
    allow_fallback: bool,
    skip_plots: bool,
    binary_config: str,
    fallback_config: str,
    subset_config: Path,
    binary_label: str,
    mmpd_label: str,
    infer_binary: bool,
    skip_decomposition: bool,
    device: torch.device,
    results_dir_override: Optional[Path],
    ckpt_dir_override: Optional[Path],
    infer_batch_size: int,
) -> Dict[str, object]:
    binary_run = discover_binary_run(
        datasets_root,
        ckpt_root,
        dataset,
        config_suffix=binary_config,
        allow_fallback=allow_fallback,
        fallback_config=fallback_config,
        results_dir_override=results_dir_override,
        ckpt_dir_override=ckpt_dir_override,
    )
    if ckpt_dir_override is not None:
        ckpt_dir = ckpt_dir_override
    elif binary_run.ckpt_dir.is_dir():
        ckpt_dir = binary_run.ckpt_dir
    else:
        ckpt_dir = discover_binary_ckpt(ckpt_root, dataset, binary_run.config_suffix)

    mmpd = load_mmpd_pack(mmpd_run, dataset)
    binary = load_binary_pack(binary_run, dataset)

    runner: Optional[BinaryStagedInference] = None
    if binary is None:
        if not infer_binary:
            raise FileNotFoundError(
                f"{dataset}: no binary eval NPZ under {binary_run.results_dir}/raw; "
                "pass --infer-binary to run staged inference from checkpoints."
            )
        runner = BinaryStagedInference(
            ckpt_dir=ckpt_dir,
            dataset=dataset,
            config_name=binary_run.config_suffix,
            device=device,
            subset_config=subset_config,
        )
        print(f"[{dataset}] inferring binary preds on {len(mmpd['indices'])} MMPD-aligned windows...")
        inferred = runner.infer_on_indices(
            mmpd["indices"],
            prob_draws=prob_draws,
            batch_size=infer_batch_size,
        )
        aligned = align_mmpd_with_inferred_binary(mmpd, inferred, dataset)
    else:
        aligned = align_packs(mmpd, binary, dataset)
        if not skip_decomposition:
            runner = BinaryStagedInference(
                ckpt_dir=ckpt_dir,
                dataset=dataset,
                config_name=binary_run.config_suffix,
                device=device,
                subset_config=subset_config,
            )

    mse_bin = per_window_anchor_mse(aligned.y_true, aligned.binary_det)
    mse_mmpd = per_window_anchor_mse(aligned.y_true, aligned.mmpd_det)
    crps_bin = per_window_crps(aligned.y_true, aligned.binary_samples)
    crps_mmpd = per_window_crps(aligned.y_true, aligned.mmpd_samples)

    mse_delta = mse_bin - mse_mmpd
    crps_delta = crps_bin - crps_mmpd

    agg_mse_delta = float(mse_delta.mean())
    agg_crps_delta = float(crps_delta.mean())
    print(
        f"[{dataset}] windows={len(aligned.indices)} "
        f"binary={binary_run.results_dir.name} ckpt={ckpt_dir.name} "
        f"mean Δmse={agg_mse_delta:+.6f} mean Δcrps={agg_crps_delta:+.6f} "
        f"(partial anchor_mse bin={binary_run.metrics.get('anchor_mse')} "
        f"crps={binary_run.metrics.get('crps')})"
    )

    mmpd_partial = mmpd_run / "partials" / f"{dataset}_mmpd.json"
    if mmpd_partial.is_file():
        mmpd_metrics = _read_json(mmpd_partial)
        print(
            f"  mmpd partial anchor_mse={mmpd_metrics.get('anchor_mse')} "
            f"crps={mmpd_metrics.get('crps')}"
        )

    rankings: Dict[str, List[Dict[str, float]]] = {}
    ds_out = output_dir / dataset
    ds_out.mkdir(parents=True, exist_ok=True)

    for metric_name, delta, b_vals, m_vals in (
        ("anchor_mse", mse_delta, mse_bin, mse_mmpd),
        ("crps", crps_delta, crps_bin, crps_mmpd),
    ):
        top_rows = rank_top_k(delta, top_k)
        rows_meta: List[Dict[str, float]] = []
        for rank, pack_row in enumerate(top_rows, start=1):
            win_idx = int(aligned.indices[pack_row])
            row = {
                "rank": rank,
                "test_window_index": win_idx,
                "pack_row": int(pack_row),
                "delta": float(delta[pack_row]),
                "binary": float(b_vals[pack_row]),
                "mmpd": float(m_vals[pack_row]),
            }
            rows_meta.append(row)
            if skip_plots:
                continue
            past_batch, future_batch, norm = _load_test_context(
                dataset, [win_idx], subset_config,
            )
            anchor_2d = None
            residual_range = 1.0
            if runner is not None and not skip_decomposition:
                anchor_2d = runner.infer_anchor_2d(win_idx)
                runner._ensure_loaded()
                assert runner._fine_model is not None
                h = int(anchor_2d.coarse.shape[-2])
                residual_range = float(runner._fine_model.to_2d.max_scale) / float(h)
            plot_window_panel(
                dataset=dataset,
                window_idx=win_idx,
                rank=rank,
                metric=metric_name,
                delta=float(delta[pack_row]),
                binary_mse=float(mse_bin[pack_row]),
                mmpd_mse=float(mse_mmpd[pack_row]),
                binary_crps=float(crps_bin[pack_row]),
                mmpd_crps=float(crps_mmpd[pack_row]),
                past=past_batch[0],
                future=future_batch[0],
                norm=norm,
                pack_row=int(pack_row),
                aligned=aligned,
                output_path=ds_out / f"{metric_name}_delta_rank{rank:02d}_win{win_idx}.png",
                prob_draws=prob_draws,
                context_len=min(aligned.y_true.shape[2], 96 * 2),
                binary_label=binary_label,
                mmpd_label=mmpd_label,
                subset_config=subset_config,
                anchor_2d=anchor_2d,
                residual_range=residual_range,
            )
        rankings[metric_name] = rows_meta

    meta = {
        "dataset": dataset,
        "binary_results_dir": str(binary_run.results_dir),
        "binary_ckpt_dir": str(ckpt_dir),
        "binary_config": binary_run.config_suffix,
        "mmpd_run": str(mmpd_run),
        "inferred_binary": binary is None,
        "n_windows": int(len(aligned.indices)),
        "mean_delta_anchor_mse": agg_mse_delta,
        "mean_delta_crps": agg_crps_delta,
        "aggregate_check": {
            "binary_anchor_mse": float(mse_bin.mean()),
            "mmpd_anchor_mse": float(mse_mmpd.mean()),
            "binary_crps": float(crps_gr(aligned.y_true, aligned.binary_samples)),
            "mmpd_crps": float(crps_gr(aligned.y_true, aligned.mmpd_samples)),
        },
        "rankings": rankings,
    }
    with (ds_out / "delta_rankings.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def write_summary_csv(output_dir: Path, metas: Sequence[Dict[str, object]]) -> Path:
    path = output_dir / "delta_top_summary.csv"
    fields = [
        "dataset",
        "metric",
        "rank",
        "test_window_index",
        "delta",
        "binary",
        "mmpd",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for meta in metas:
            ds = meta["dataset"]
            for metric, rows in meta["rankings"].items():
                for row in rows:
                    writer.writerow(
                        {
                            "dataset": ds,
                            "metric": metric,
                            "rank": row["rank"],
                            "test_window_index": row["test_window_index"],
                            "delta": f"{row['delta']:.8f}",
                            "binary": f"{row['binary']:.8f}",
                            "mmpd": f"{row['mmpd']:.8f}",
                        }
                    )
    return path


def _config_label(config_suffix: str) -> str:
    return config_suffix.replace("binary_anchor_stationary_flat_subsets_", "").replace("_", " ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated datasets.",
    )
    parser.add_argument(
        "--binary-config",
        default=DEFAULT_BINARY_CONFIG,
        help="Experiment suffix for auto-discovery (results/datasets + results/ckpts).",
    )
    parser.add_argument(
        "--binary-results-dir",
        type=Path,
        default=None,
        help="Explicit binary results root (one dataset per invocation).",
    )
    parser.add_argument(
        "--binary-ckpt-dir",
        type=Path,
        default=None,
        help="Explicit binary checkpoint root (coarse/fine best.pt).",
    )
    parser.add_argument(
        "--binary-label",
        default=None,
        help="Row label for binary line plots (default: derived from --binary-config).",
    )
    parser.add_argument(
        "--mmpd-run",
        type=Path,
        default=DEFAULT_MMPD_RUN,
        help="Fair MMPD results root (partials + raw/mmpd_*.npz).",
    )
    parser.add_argument(
        "--mmpd-label",
        default="Fair MMPD",
        help="Row label for MMPD line plots.",
    )
    parser.add_argument(
        "--subset-config",
        type=Path,
        default=DEFAULT_SUBSET_CONFIG,
        help="YAML with flat-subset variate policy.",
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=REPO_ROOT / "results" / "datasets",
    )
    parser.add_argument(
        "--ckpt-root",
        type=Path,
        default=REPO_ROOT / "results" / "ckpts",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--prob-draws",
        type=int,
        default=3,
        help="Probabilistic sample lines per panel (also used when --infer-binary).",
    )
    parser.add_argument(
        "--allow-fallback-binary",
        action="store_true",
        help=f"Fall back to {FALLBACK_BINARY_CONFIG} when primary config missing.",
    )
    parser.add_argument(
        "--fallback-binary-config",
        default=FALLBACK_BINARY_CONFIG,
        help="Secondary binary config for --allow-fallback-binary.",
    )
    parser.add_argument(
        "--infer-binary",
        action="store_true",
        help="Run staged checkpoint inference when binary eval NPZ is missing.",
    )
    parser.add_argument(
        "--skip-decomposition",
        action="store_true",
        help="Skip coarse/fine 2D anchor map rows (line plots only).",
    )
    parser.add_argument(
        "--infer-batch-size",
        type=int,
        default=4,
        help="Batch size for --infer-binary.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for inference / 2D decomposition.",
    )
    parser.add_argument("--skip-plots", action="store_true", help="Only write rankings JSON/CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    binary_label = args.binary_label or _config_label(args.binary_config)

    metas: List[Dict[str, object]] = []
    skipped: List[str] = []
    for dataset in datasets:
        try:
            meta = process_dataset(
                dataset,
                mmpd_run=args.mmpd_run,
                datasets_root=args.datasets_root,
                ckpt_root=args.ckpt_root,
                output_dir=args.output_dir,
                top_k=args.top_k,
                prob_draws=args.prob_draws,
                allow_fallback=args.allow_fallback_binary,
                skip_plots=args.skip_plots,
                binary_config=args.binary_config,
                fallback_config=args.fallback_binary_config,
                subset_config=args.subset_config,
                binary_label=binary_label,
                mmpd_label=args.mmpd_label,
                infer_binary=args.infer_binary,
                skip_decomposition=args.skip_decomposition,
                device=device,
                results_dir_override=args.binary_results_dir,
                ckpt_dir_override=args.binary_ckpt_dir,
                infer_batch_size=args.infer_batch_size,
            )
            metas.append(meta)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"[skip] {dataset}: {exc}")
            skipped.append(dataset)

    if metas:
        summary = write_summary_csv(args.output_dir, metas)
        print(f"Wrote {len(metas)} datasets -> {args.output_dir}")
        print(f"Summary CSV: {summary}")
    else:
        print("No datasets processed.", file=sys.stderr)
        sys.exit(1)
    if skipped:
        print(f"Skipped ({len(skipped)}): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
