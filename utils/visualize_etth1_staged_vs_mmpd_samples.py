#!/usr/bin/env python3
"""ETTh1: probabilistic forecasts vs GT — staged 2-stage vs MMPD (separate panels).

Default: anchor test window (manifest idx 1153) plus 5 more random test windows,
same indices for both models. Each window: 5 prob samples, distinct colors.

Example:
  python utils/visualize_etth1_staged_vs_mmpd_samples.py
  python utils/visualize_etth1_staged_vs_mmpd_samples.py --anchor-index 1153 --extra-windows 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = (
    REPO_ROOT
    / "reports"
    / "06-01_cfg_ablation_mmpd_matrix_combined"
    / "viz_etth1_staged_vs_mmpd_samples"
)
MANIFEST_PATH = (
    REPO_ROOT / "reports" / "06-01_cfg_ablation_mmpd_matrix_combined" / "viz_manifest.json"
)

SAMPLE_COLORS = ["#e41a1c", "#ff7f00", "#4daf4a", "#984ea3", "#a65628"]
GT_COLOR = "#2196F3"
PAST_COLOR = "#9E9E9E"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.guidance import iTransformerGuidance
from models.diffusion_tsf.visualize_comparison import denorm
from models.diffusion_tsf.train_multivariate_pipeline import load_dataset, load_itransformer_from_checkpoint
from utils.eval_mmpd_gaussian_anchor import (
    DEFAULT_MMPD_DATA,
    DEFAULT_MMPD_REPO,
    AnchorRun,
    ensure_mmpd_repo,
    find_anchor_runs,
    mmpd_data_split,
    mmpd_staged_filename_for_run,
    stage_mmpd_dataset_for_run,
)
from utils.visualize_staged_forecast import (
    _build_pipeline_state,
    _load_staged_bundle,
    _load_staged_diffusion,
    _resolve_itrans_paths,
    _staged_anchor_and_samples,
    _window_lengths,
    pick_staged_ckpt_dir,
)


def pick_shared_test_indices(
    n_test: int,
    anchor: int,
    extra: int,
    seed: int,
) -> List[int]:
    """Anchor window plus *extra* distinct random test indices (shared across models)."""
    if anchor < 0 or anchor >= n_test:
        raise IndexError(f"anchor index {anchor} out of range [0, {n_test})")
    indices = [anchor]
    rng = random.Random(seed)
    while len(indices) < extra + 1:
        j = rng.randrange(n_test)
        if j not in indices:
            indices.append(j)
    return indices


def _test_index_from_manifest(dataset: str = "ETTh1") -> Optional[int]:
    if not MANIFEST_PATH.is_file():
        return None
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for section in ("cfg_off", "coarse_fine_multiphase", "coarse_fine_components"):
        paths = (data.get(section) or {}).get(dataset) or []
        if not paths:
            continue
        stem = Path(paths[0]).stem
        if "_idx" in stem:
            try:
                return int(stem.rsplit("_idx", 1)[-1])
            except ValueError:
                pass
    return None


def _find_recent_mmpd_matrix_roots(limit: int = 5) -> List[Path]:
    """Newest matrix output dirs that contain a working ETTh1 MMPD checkpoint."""
    hits: List[Tuple[float, Path]] = []
    for base in (REPO_ROOT / "results" / "datasets", REPO_ROOT / "results" / "archive" / "datasets"):
        if not base.is_dir():
            continue
        for d in base.iterdir():
            if not d.is_dir():
                continue
            ckpt = (
                d
                / "mmpd_out"
                / "checkpoints"
                / "Decoder-MMPD"
                / "dataETTh1_il96_ol96_backboneDecoder_lossMMPD_weightedTrue_patch12_pointW0.01_diffH256_diffLayer1_radius3_diffStep1000_betalinear"
                / "model_checkpoint.pth"
            )
            if ckpt.is_file():
                hits.append((ckpt.stat().st_mtime, d))
    hits.sort(key=lambda x: x[0], reverse=True)
    seen: List[Path] = []
    for _, d in hits:
        if d not in seen:
            seen.append(d)
        if len(seen) >= limit:
            break
    if not seen:
        raise FileNotFoundError("No ETTh1 MMPD matrix checkpoint under results/datasets")
    return seen


def _load_mmpd_exp(
    mmpd_repo: Path,
    output_root: Path,
    *,
    lookback: int,
    horizon: int,
    patch_size: int,
    data_dim: int,
    gpu: int,
    cpu: bool,
) -> Any:
    _ensure_mmpd_on_path(mmpd_repo)
    from exp.exp_forecast import Exp_Forecast

    def setting(args: SimpleNamespace) -> str:
        return (
            f"data{args.data}_il{args.in_len}_ol{args.out_len}_backbone{args.backbone}"
            f"_loss{args.loss_func}_weighted{args.weighted}_patch{args.patch_size}"
            f"_pointW{args.point_weight}_diffH{args.d_diffusion}"
            f"_diffLayer{args.diffusion_layers}_radius{args.radius}"
            f"_diffStep{args.max_diffusion_steps}_beta{args.beta_schedule}"
        )

    args = SimpleNamespace(
        data="ETTh1",
        root_path=str(DEFAULT_MMPD_DATA.resolve()),
        data_path="ETTh1.csv",
        data_split=[8640, 2880, 2880],
        output_root=str(output_root.resolve()),
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
        sample_num=5,
        num_sampling_steps="20",
        temperature=1.0,
        gmm_components=10,
        prior_pi_decay=0.5,
        prior_precision_shape=1e2,
        gmm_iterations=3,
        use_gpu=(torch.cuda.is_available() and not cpu),
        gpu=gpu,
        use_multi_gpu=False,
        devices="0,1,2,3",
    )
    exp = Exp_Forecast(args)
    ckpt_path = os.path.join(
        args.output_root,
        "checkpoints",
        f"{args.backbone}-{args.loss_func}",
        setting(args),
        "model_checkpoint.pth",
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_state = exp.model.state_dict()
    for k, v in state.items():
        if "gen_diffusion" not in k:
            model_state[k] = v
    exp.model.load_state_dict(model_state)
    exp.model.eval()
    return exp


def _ensure_mmpd_on_path(mmpd_repo: Path) -> None:
    """MMPD imports utils.tools; drop pipeline-hijacked utils modules first."""
    repo_str = str(REPO_ROOT)
    mmpd_str = str(mmpd_repo.resolve())
    for name in list(sys.modules):
        if name == "utils" or name.startswith("utils."):
            sys.modules.pop(name, None)
    for p in (repo_str, mmpd_str):
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, repo_str)
    sys.path.insert(1, mmpd_str)
    os.environ["TS_SANDBOX_REPO"] = repo_str


class MmpdSession:
    """Loaded MMPD model + test set for repeated window inference."""

    def __init__(
        self,
        matrix_root: Path,
        mmpd_repo: Path,
        *,
        device_id: int,
        cpu: bool,
    ) -> None:
        _ensure_mmpd_on_path(mmpd_repo)
        from data_provider.dataset_mts import Dataset_MTS

        self.mmpd_repo = mmpd_repo
        self.dev = torch.device(
            f"cuda:{device_id}" if torch.cuda.is_available() and not cpu else "cpu"
        )
        lookback, horizon = 96, 96
        os.environ["MMPD_TEST_STRIDE"] = "1"
        data_dir = DEFAULT_MMPD_DATA.resolve()
        run = find_anchor_runs(
            ["ETTh1"],
            [REPO_ROOT / "results" / "ckpts" / "05-31-3828089-ETTh1-binary_dual_scale"],
            REPO_ROOT / "results" / "ckpts",
            "binary",
        )["ETTh1"]
        stage_mmpd_dataset_for_run(data_dir, run)
        parts = [int(x) for x in mmpd_data_split("ETTh1", data_dir).split(",")]
        self.test_data = Dataset_MTS(
            root_path=str(data_dir),
            data_path=mmpd_staged_filename_for_run(run),
            flag="test",
            size=[lookback, horizon],
            data_split=parts,
        )
        self.exp = _load_mmpd_exp(
            mmpd_repo,
            matrix_root / "mmpd_out",
            lookback=lookback,
            horizon=horizon,
            patch_size=12,
            data_dim=7,
            gpu=device_id,
            cpu=cpu,
        )
        self.exp.model = self.exp.model.to(self.dev)

    @torch.no_grad()
    def prob_samples_at_index(
        self,
        test_index: int,
        n_samples: int,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        from exp.normalization import denormalize as mmpd_denorm, get_statistics, normalize

        if test_index < 0 or test_index >= len(self.test_data):
            raise IndexError(f"test_index {test_index} out of range [0, {len(self.test_data)})")

        batch_x, batch_y = self.test_data[test_index]
        batch_x = torch.tensor(batch_x, dtype=torch.float32).unsqueeze(0).to(self.dev)
        batch_x = rearrange(batch_x, "b l d -> b d l")

        torch.manual_seed(seed + test_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + test_index)

        x_shift, x_scale = get_statistics(batch_x)
        normed_x = normalize(batch_x, x_shift, x_scale)
        _, _, samples = self.exp.model.predict(
            normed_x,
            prob_pred=True,
            sample_num=n_samples,
            temperature=1.0,
            gmm=True,
            gmm_components=10,
            prior_pi_decay=0.5,
            prior_precision_shape=1e2,
            gmm_iterations=3,
        )
        samples = mmpd_denorm(samples, x_shift, x_scale).detach().cpu().numpy()[0]
        past_phys = (
            rearrange(mmpd_denorm(batch_x, x_shift, x_scale).detach().cpu(), "b d l -> b l d")
            [0]
            .numpy()
            .T
        )
        gt_phys = batch_y.T  # (C, H)
        return past_phys, gt_phys, samples


class StagedSession:
    """Loaded 2-stage models + test set."""

    def __init__(self, ckpt_dir: Path, device: torch.device) -> None:
        self.ckpt_dir = ckpt_dir
        self.device = device
        sub = _load_staged_bundle(ckpt_dir, "ETTh1")
        self.subset_id = sub["subset_id"]
        self.variate_names = sub.get("variate_names") or []
        state = _build_pipeline_state(ckpt_dir, "ETTh1", self.subset_id)
        self.lookback, self.horizon = _window_lengths("ETTh1", state)
        self.k = 8
        n_vars = len(sub["variate_indices"])
        _, _, self.test_ds, self.norm_stats = load_dataset(
            "ETTh1",
            sub["variate_indices"],
            stride=1,
            test_stride=1,
            lookback=self.lookback,
            horizon=self.horizon,
        )
        guidance_path, _ = _resolve_itrans_paths(ckpt_dir, self.subset_id)
        guidance_model = load_itransformer_from_checkpoint(
            str(guidance_path), n_vars, device
        )
        ig = iTransformerGuidance(guidance_model)
        self.coarse = _load_staged_diffusion(
            state, "coarse", sub["coarse_pt"], ig, n_vars, device
        )
        self.fine = _load_staged_diffusion(
            state, "fine", sub["fine_pt"], ig, n_vars, device
        )
        self.mean = torch.tensor(self.norm_stats["mean"], dtype=torch.float32)
        self.std = torch.tensor(self.norm_stats["std"], dtype=torch.float32)

    @torch.no_grad()
    def prob_samples_at_index(
        self,
        test_index: int,
        n_samples: int,
        seed: int,
        prob_steps: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        past, future = self.test_ds[test_index]
        past_t = past.unsqueeze(0).to(self.device)
        _, prob_preds = _staged_anchor_and_samples(
            self.coarse,
            self.fine,
            past_t,
            prob_samples=n_samples,
            prob_sampler="dpmpp",
            prob_steps=prob_steps,
            seed=seed,
            test_index=test_index,
        )
        future_slice = future[:, -self.horizon :]
        if self.k > 0:
            future_slice = future_slice[..., self.k :]
        gt = denorm(future_slice, self.mean, self.std).numpy()
        past_dn = denorm(past, self.mean, self.std).numpy()
        samples = [
            denorm(_forecast_tail(p, self.horizon, self.k), self.mean, self.std).numpy()
            for p in prob_preds
        ]
        return past_dn, gt, samples


def _forecast_tail(pred: torch.Tensor, horizon: int, k: int) -> torch.Tensor:
    tail = pred[:, -horizon:]
    return tail[..., k:] if k > 0 else tail


def plot_multi_window_panel(
    *,
    windows: Sequence[Dict[str, Any]],
    sample_labels: Sequence[str],
    title: str,
    out_path: Path,
    variate_names: Optional[List[str]] = None,
) -> None:
    """One block of variate rows per test window (same indices across models)."""
    n_windows = len(windows)
    n_vars = windows[0]["gt"].shape[0]
    t_fut = windows[0]["gt"].shape[1]
    context_len = min(t_fut * 2, windows[0]["past"].shape[-1])
    t_past = np.arange(-context_len, 0)
    t_future = np.arange(0, t_fut)
    names = variate_names or [f"v{i}" for i in range(n_vars)]

    fig, axes = plt.subplots(
        n_windows * n_vars,
        1,
        figsize=(12, 2.0 * n_windows * n_vars),
        squeeze=False,
        constrained_layout=True,
    )
    legend_done = False

    for w_i, win in enumerate(windows):
        past_phys = win["past"]
        gt_phys = win["gt"]
        sample_phys = win["samples"]
        idx = win["test_index"]
        for v in range(n_vars):
            ax = axes[w_i * n_vars + v, 0]
            ax.plot(
                t_past,
                past_phys[v, -context_len:],
                color=PAST_COLOR,
                lw=0.9,
                alpha=0.55,
            )
            ax.plot(t_future, gt_phys[v], color=GT_COLOR, lw=2.0, label="GT" if not legend_done else "")
            for k, (sp, lab) in enumerate(zip(sample_phys, sample_labels)):
                ax.plot(
                    t_future,
                    sp[v],
                    color=SAMPLE_COLORS[k % len(SAMPLE_COLORS)],
                    lw=1.2,
                    alpha=0.9,
                    label=lab if (not legend_done and v == 0 and w_i == 0) else "",
                )
            ax.axvline(0, color="k", ls=":", alpha=0.25)
            ylabel = names[v] if v < len(names) else f"v{v}"
            ax.set_ylabel(f"idx {idx}\n{ylabel}", fontsize=8)
            ax.grid(alpha=0.2)
        legend_done = True

    axes[0, 0].legend(loc="upper right", fontsize=7, ncol=2)
    fig.suptitle(title, fontsize=11, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-index", type=int, default=None, help="Primary test window")
    parser.add_argument("--test-index", type=int, default=None, help="Alias for --anchor-index")
    parser.add_argument("--extra-windows", type=int, default=5, help="Additional random test windows")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prob-steps", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    anchor = args.anchor_index if args.anchor_index is not None else args.test_index
    if anchor is None:
        anchor = _test_index_from_manifest() or 1153

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = args.output_dir.resolve()
    n = args.n_samples
    labels = [f"sample {i + 1}" for i in range(n)]

    staged_ckpt = pick_staged_ckpt_dir(REPO_ROOT / "results" / "ckpts", "ETTh1")
    staged_sess = StagedSession(staged_ckpt, device)
    test_indices = pick_shared_test_indices(
        len(staged_sess.test_ds),
        anchor,
        args.extra_windows,
        args.seed,
    )
    print(f"Shared test indices ({len(test_indices)}): {test_indices}", flush=True)

    staged_windows: List[Dict[str, Any]] = []
    for idx in test_indices:
        past_dn, gt, samples = staged_sess.prob_samples_at_index(
            idx, n, args.seed, args.prob_steps
        )
        staged_windows.append(
            {"test_index": idx, "past": past_dn, "gt": gt, "samples": samples}
        )

    idx_tag = "_".join(str(i) for i in test_indices)
    staged_out = out_dir / f"etth1_staged_5samples_{len(test_indices)}windows.png"
    plot_multi_window_panel(
        windows=staged_windows,
        sample_labels=labels,
        variate_names=staged_sess.variate_names,
        title=(
            f"ETTh1 — 2-stage coarse→fine — test idx {test_indices} | "
            f"{n}× dpmpp (steps={args.prob_steps})"
        ),
        out_path=staged_out,
    )
    print(f"Wrote {staged_out}")

    mmpd_roots = _find_recent_mmpd_matrix_roots(1)
    mmpd_repo = DEFAULT_MMPD_REPO.resolve()
    ensure_mmpd_repo(mmpd_repo, update=False)
    mmpd_sess = MmpdSession(mmpd_roots[0], mmpd_repo, device_id=0, cpu=args.cpu)

    mmpd_windows: List[Dict[str, Any]] = []
    for idx in test_indices:
        past_m, gt_m, samples_m = mmpd_sess.prob_samples_at_index(idx, n, args.seed)
        mmpd_windows.append(
            {
                "test_index": idx,
                "past": past_m,
                "gt": gt_m,
                "samples": [samples_m[:, i, :] for i in range(n)],
            }
        )

    mmpd_out = out_dir / f"etth1_mmpd_5samples_{len(test_indices)}windows.png"
    plot_multi_window_panel(
        windows=mmpd_windows,
        sample_labels=labels,
        title=(
            f"ETTh1 — MMPD ({mmpd_roots[0].name}) — same test idx {test_indices} | "
            f"{n}× prob samples (steps={args.prob_steps})"
        ),
        out_path=mmpd_out,
    )
    print(f"Wrote {mmpd_out}")

    (out_dir / "shared_test_indices.json").write_text(
        json.dumps({"ETTh1": test_indices}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
