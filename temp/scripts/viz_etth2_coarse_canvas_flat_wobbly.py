#!/usr/bin/env python3
"""ETTh2 canvas128: coarse H=16 2D canvases + multistep bitflip maps.

Test-set examples of:
  1) flat coarse runs that are wiggles (≥3 identical coarse bins, continuous
     window-z range > ε)
  2) true GT flatlines (≥3 identical bins AND window-z range ≤ ε)

Flat/wobbly defs match temp/scripts/etth2_coarse_flat_pred_acc.py
(ε = 0.25 × coarse_bin_width, window-norm mean/std).

Sampling note
-------------
Staged *point* eval uses sampler=anchor (one forward at t=T−1) — that path has
no multi-step reverse schedule. This script uses the staged *probabilistic*
sampler from the leaf YAML: quad_t with N=20 steps.

Captures bitflip / ε logits + P(flip) at several reverse-schedule inference
indices (default: early / mid-early / mid-late / penultimate). Fail-fast if
N<2, schedule final ≠ 0, or requested snapshot indices are missing.

prediction_target=epsilon on this ckpt → primary DiT head is the bitflip / ε
logits. zt head is also recorded in meta/npz but not plotted.

Examples:
  source .venv/bin/activate
  python temp/scripts/viz_etth2_coarse_canvas_flat_wobbly.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.pipeline.globals_bridge import patch_globals  # noqa: E402
from models.diffusion_tsf.train_multivariate_pipeline import (  # noqa: E402
    load_wrapped_guidance,
    resolve_pipeline_data_subset,
)
import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod  # noqa: E402
from temp.scripts.eval_ablation_disc_l8_l16 import (  # noqa: E402
    _max_scale_from_ckpt_metadata,
    load_ablation_run,
)
from temp.scripts.etth2_coarse_flat_pred_acc import (  # noqa: E402
    FLAT_EPS_FRAC,
    MIN_RUN,
    TRAIN_END,
    VAL_END,
    _find_runs,
    _window_norm_z_and_bins,
)
from utils.eval_mmpd_gaussian_anchor import (  # noqa: E402
    run_subset_id,
    run_variate_indices,
)
from utils.visualize_staged_eval_2d_preds import (  # noqa: E402
    _build_state,
    _load_stage_model,
    _resolve_guidance_ckpt,
)

DATASET = "ETTh2"
ETTH2_NAMES = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")
DEFAULT_PACK = (
    REPO_ROOT
    / "results/datasets/08-04-2009-ablation-disc-l8-l16-ETTh2-c128-valtest80-byvar"
    / "raw"
    / "binary_window_norm_c128_ETTh2_val-test.npz"
)
DEFAULT_CKPT = (
    REPO_ROOT
    / "results/ckpts/08-04-4601319-ETTh2-binary_window_norm_patch_refine_canvas128_p64x6_etth2"
)
DEFAULT_CFG = "configs/binary_window_norm_patch_refine_canvas128_p64x6_etth2.yaml"
DEFAULT_OUT = REPO_ROOT / "temp/lean_disc_c128_results/etth2_coarse_canvas_viz_multistep"
DEFAULT_SAMPLER = "quad_t"
DEFAULT_STEPS = 20
# Relative positions along reverse schedule (0=start/noisy … 1=penultimate).
DEFAULT_SNAPSHOT_FRACS = (0.0, 0.33, 0.66, 1.0)


@dataclass
class RunExample:
    kind: str  # flat | wobbly
    local_idx: int
    pool_idx: int
    series_start: int
    variate: int
    run_a: int
    run_b: int
    bin_id: int
    z_range: float
    run_len: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--config", type=str, default=DEFAULT_CFG)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--flat-eps-frac", type=float, default=FLAT_EPS_FRAC)
    p.add_argument("--min-run", type=int, default=MIN_RUN)
    p.add_argument("--n-per-class", type=int, default=4)
    p.add_argument("--sampler", type=str, default=DEFAULT_SAMPLER)
    p.add_argument("--num-inference-steps", type=int, default=DEFAULT_STEPS)
    p.add_argument(
        "--snapshot-indices",
        type=str,
        default=None,
        help=(
            "Comma-separated reverse-schedule inference indices to capture "
            "(0 … N-2). Default: early/mid-early/mid-late/penultimate via fracs."
        ),
    )
    p.add_argument(
        "--snapshot-fracs",
        type=str,
        default=",".join(str(x) for x in DEFAULT_SNAPSHOT_FRACS),
        help="Frac positions along [0, N-2] when --snapshot-indices unset.",
    )
    p.add_argument("--lookback", type=int, default=None)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--seed", type=int, default=20260808)
    p.add_argument("--dpi", type=int, default=130)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    args.pack = args.pack.expanduser().resolve()
    args.ckpt = args.ckpt.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def _parse_int_list(s: str) -> List[int]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if not parts:
        raise ValueError("empty int list")
    return [int(p) for p in parts]


def _parse_float_list(s: str) -> List[float]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if not parts:
        raise ValueError("empty float list")
    return [float(p) for p in parts]


def _resolve_snapshot_indices(
    *,
    n_steps: int,
    snapshot_indices: Optional[Sequence[int]],
    snapshot_fracs: Sequence[float],
) -> List[int]:
    """Pick unique inference indices in [0, N-2] (exclude final t→0 step)."""
    last_capture = int(n_steps) - 2
    if last_capture < 0:
        raise RuntimeError(f"need ≥2 inference steps; got {n_steps}")
    if snapshot_indices is not None:
        idxs = [int(i) for i in snapshot_indices]
    else:
        idxs = []
        for f in snapshot_fracs:
            if f < 0.0 or f > 1.0:
                raise ValueError(f"snapshot frac {f} outside [0,1]")
            idxs.append(int(round(float(f) * last_capture)))
    # Dedup preserve order; clamp + validate
    out: List[int] = []
    seen: set[int] = set()
    for i in idxs:
        if i < 0 or i > last_capture:
            raise RuntimeError(
                f"snapshot inference index {i} outside [0, {last_capture}] "
                f"(N={n_steps}; final t=0 step is not captured)"
            )
        if i in seen:
            continue
        seen.add(i)
        out.append(i)
    if not out:
        raise RuntimeError("no snapshot indices resolved")
    return out


def _reverse_step_indices(
    *,
    sampler: str,
    num_steps: int,
    binary_num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    name = str(sampler).lower()
    if name in {"anchor", "deterministic_anchor"}:
        raise RuntimeError(
            "sampler='anchor' is one-shot (t=T-1 only) and has no multi-step "
            "denoising schedule; use quad_t/ddim (probabilistic eval path)."
        )
    if int(num_steps) < 2:
        raise RuntimeError(
            f"need ≥2 inference steps for reverse schedule; got {num_steps}"
        )
    T = int(binary_num_steps)
    if name in {"quad_t", "ddim_quad"}:
        ramp = torch.linspace(1.0, 0.0, num_steps, device=device)
        return torch.round((ramp**2) * (T - 1)).long()
    if name == "ddim":
        return torch.linspace(T - 1, 0, num_steps, device=device, dtype=torch.long)
    raise ValueError(f"unsupported sampler {sampler!r}; expected ddim or quad_t")


@torch.no_grad()
def _coarse_generate_capture_snapshots(
    model: torch.nn.Module,
    past: torch.Tensor,
    *,
    sampler: str,
    num_steps: int,
    seed: int,
    snapshot_step_is: Sequence[int],
) -> Dict[str, Any]:
    """Coarse generate; capture ε/bitflip logits at selected reverse steps."""
    if str(getattr(model.config, "diffusion_stage", "")) != "coarse":
        raise RuntimeError(
            f"expected coarse stage, got {model.config.diffusion_stage!r}"
        )
    if model.binary_scheduler is None:
        raise RuntimeError("binary_scheduler missing")

    device = past.device
    step_indices = _reverse_step_indices(
        sampler=sampler,
        num_steps=num_steps,
        binary_num_steps=int(model.config.binary_num_steps),
        device=device,
    )
    if int(step_indices[-1].item()) != 0:
        raise RuntimeError(
            f"reverse schedule must end at t=0; got {int(step_indices[-1].item())} "
            f"(indices={step_indices.tolist()})"
        )
    want = sorted({int(i) for i in snapshot_step_is})
    last_ok = int(num_steps) - 2
    for i in want:
        if i < 0 or i > last_ok:
            raise RuntimeError(
                f"snapshot i={i} outside [0, {last_ok}]; schedule={step_indices.tolist()}"
            )
    want_set = set(want)
    pred_target = str(getattr(model.config, "prediction_target", "x0"))

    captured: Dict[int, Dict[str, Any]] = {}
    orig_sample = model.binary_scheduler.sample

    def hooked_sample(
        model_fn,
        shape,
        num_steps: int = 20,
        device: str = "cpu",
        verbose: bool = False,
        sampler: str = "ddim",
        yield_intermediates: bool = False,
        reverse_step_indices: Optional[torch.Tensor] = None,
        snapshot_timesteps: Optional[Tuple[int, ...]] = None,
    ):
        del verbose, sampler, yield_intermediates, snapshot_timesteps
        if reverse_step_indices is not None:
            idxs = reverse_step_indices.to(device=device, dtype=torch.long)
        else:
            idxs = step_indices
        if idxs.numel() != num_steps:
            raise RuntimeError(
                f"step count mismatch: schedule len={idxs.numel()} vs num_steps={num_steps}"
            )
        if int(idxs[-1].item()) != 0:
            raise RuntimeError(f"hooked schedule final t={int(idxs[-1].item())} != 0")
        xt = torch.bernoulli(torch.full(shape, 0.5, device=device))
        for i, t_val in enumerate(idxs):
            t_idx = int(t_val.item())
            t_batch = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
            x0_logits, zt_logits = model_fn(xt, t_batch)
            if i in want_set:
                captured[i] = {
                    "step_i": i,
                    "t_idx": t_idx,
                    "xt": xt.detach().clone(),
                    "x0_logits": x0_logits.detach().clone(),
                    "zt_logits": zt_logits.detach().clone(),
                }
            x0_hat = torch.bernoulli(torch.sigmoid(x0_logits))
            if i < len(idxs) - 1:
                t_next = int(idxs[i + 1].item())
                beta_next = float(model.binary_scheduler.betas[t_next].item())
                zt_new = torch.bernoulli(torch.full_like(x0_hat, beta_next))
                xt = (x0_hat.bool() ^ zt_new.bool()).float()
            else:
                xt = x0_hat
        missing = sorted(want_set - set(captured.keys()))
        if missing:
            raise RuntimeError(
                f"failed to capture logits at inference indices {missing}; "
                f"schedule={idxs.tolist()}"
            )
        return xt

    model.binary_scheduler.sample = hooked_sample
    try:
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        out = model.generate(
            past,
            sampler=sampler,
            num_inference_steps=int(num_steps),
        )
    finally:
        model.binary_scheduler.sample = orig_sample

    B = past.shape[0]
    V = int(model.config.num_variables)
    H = int(model.config.image_height)
    snaps: List[Dict[str, Any]] = []
    # Preserve caller order (not sorted)
    for i in snapshot_step_is:
        c = captured[int(i)]
        xt_pen = c["xt"]
        t_here = int(c["t_idx"])
        if pred_target == "epsilon":
            x0_l = c["x0_logits"]
            primary = torch.where(xt_pen > 0.5, -x0_l, x0_l)
        else:
            primary = c["x0_logits"]
        W = int(c["x0_logits"].shape[-1])
        snaps.append(
            {
                "step_i": int(i),
                "t_idx": t_here,
                "bitflip_logits": primary.reshape(B, V, H, W),
                "zt_logits": c["zt_logits"].reshape(B, V, H, W),
                "x0_logits": c["x0_logits"].reshape(B, V, H, W),
                "xt": xt_pen.reshape(B, V, H, W),
            }
        )

    return {
        "generate_out": out,
        "step_indices": [int(x) for x in step_indices.detach().cpu().tolist()],
        "snapshot_step_is": [int(i) for i in snapshot_step_is],
        "snapshot_ts": [int(s["t_idx"]) for s in snaps],
        "snapshots": snaps,
        "n_steps": int(num_steps),
        "sampler": sampler,
        "prediction_target": pred_target,
        "penultimate_step_i": int(num_steps) - 2,
        "penultimate_t": int(step_indices[int(num_steps) - 2].item()),
    }


def _pick_examples(
    *,
    bins_gt: np.ndarray,
    z_gt: np.ndarray,
    indices: np.ndarray,
    series_starts: np.ndarray,
    flat_eps: float,
    min_run: int,
    n_per_class: int,
) -> Tuple[List[RunExample], List[RunExample]]:
    """Greedy diversify-by-variate picks for flat / wobbly runs."""
    n_win, n_vars, _h = bins_gt.shape
    flat_cands: List[RunExample] = []
    wob_cands: List[RunExample] = []
    for n in range(n_win):
        for v in range(n_vars):
            for a, b, bin_id in _find_runs(bins_gt[n, v], min_run):
                z_range = float(z_gt[n, v, a:b].max() - z_gt[n, v, a:b].min())
                ex = RunExample(
                    kind="flat" if z_range <= flat_eps else "wobbly",
                    local_idx=int(n),
                    pool_idx=int(indices[n]),
                    series_start=int(series_starts[n]),
                    variate=int(v),
                    run_a=int(a),
                    run_b=int(b),
                    bin_id=int(bin_id),
                    z_range=z_range,
                    run_len=int(b - a),
                )
                (flat_cands if ex.kind == "flat" else wob_cands).append(ex)

    def select(cands: List[RunExample], prefer_large_z: bool) -> List[RunExample]:
        cands = sorted(
            cands,
            key=lambda e: (
                -e.run_len,
                -e.z_range if prefer_large_z else e.z_range,
                e.local_idx,
            ),
        )
        picked: List[RunExample] = []
        used_vars: set[int] = set()
        used_windows: set[int] = set()
        for e in cands:
            if e.variate in used_vars or e.local_idx in used_windows:
                continue
            picked.append(e)
            used_vars.add(e.variate)
            used_windows.add(e.local_idx)
            if len(picked) >= n_per_class:
                return picked
        for e in cands:
            if e.local_idx in used_windows:
                continue
            if any(
                p.variate == e.variate and p.run_a == e.run_a and p.run_b == e.run_b
                for p in picked
            ):
                continue
            picked.append(e)
            used_windows.add(e.local_idx)
            if len(picked) >= n_per_class:
                break
        if len(picked) < n_per_class:
            raise RuntimeError(
                f"only found {len(picked)}/{n_per_class} examples "
                f"(pool size={len(cands)}, prefer_large_z={prefer_large_z})"
            )
        return picked

    return select(flat_cands, prefer_large_z=False), select(wob_cands, prefer_large_z=True)


def _trim_hz(arr: np.ndarray, hz: int, k_ov: int) -> np.ndarray:
    if arr.shape[-1] == hz + k_ov:
        return arr[..., k_ov:]
    if arr.shape[-1] == hz:
        return arr
    raise RuntimeError(
        f"width {arr.shape[-1]} not in {{{hz}, {hz + k_ov}}} (lookback_overlap={k_ov})"
    )


def _plot_example(
    *,
    out_path: Path,
    title: str,
    gt_canvas: np.ndarray,
    snaps_np: List[Dict[str, Any]],
    z_1d: np.ndarray,
    bins_1d: np.ndarray,
    ex: RunExample,
    n_steps: int,
    sampler: str,
    flat_eps: float,
    dpi: int,
) -> None:
    h, w = gt_canvas.shape
    n_snap = len(snaps_np)
    # Shared clim across timesteps for logits so colors are comparable
    abs_stack = np.concatenate(
        [np.abs(s["bitflip_logits"]).ravel() for s in snaps_np]
    )
    vmax = float(np.percentile(abs_stack, 99.0))
    vmax = max(vmax, 1e-3)

    fig_h = 2.0 + 2.1 + 2.0 * n_snap
    fig = plt.figure(figsize=(12.8, fig_h), layout="constrained")
    # row0: z | row1: GT canvas + bin index | then n_snap rows of logits|P(flip)
    height_ratios = [1.05, 1.7] + [1.55] * n_snap
    gs = fig.add_gridspec(2 + n_snap, 2, height_ratios=height_ratios, hspace=0.28, wspace=0.22)

    ax_z = fig.add_subplot(gs[0, :])
    t = np.arange(w)
    ax_z.plot(t, z_1d, color="#212121", lw=1.4, label="window-z GT")
    ax_z.axvspan(ex.run_a - 0.5, ex.run_b - 0.5, color="#FF8A65", alpha=0.25, label="same-bin run")
    ax_z.axhline(0.0, color="#9E9E9E", lw=0.6)
    ax_z.set_xlim(-0.5, w - 0.5)
    ax_z.set_ylabel("z")
    snap_label = ", ".join(
        f"i={s['step_i']}→t={s['t_idx']}" for s in snaps_np
    )
    ax_z.set_title(
        f"{title}\n"
        f"run [{ex.run_a},{ex.run_b}) bin={ex.bin_id} len={ex.run_len} "
        f"z_range={ex.z_range:.4f} (ε={flat_eps:.4f}) | "
        f"snapshots [{snap_label}] | bins={bins_1d[ex.run_a:ex.run_b].tolist()}",
        fontsize=9.5,
    )
    ax_z.legend(loc="upper right", fontsize=8, frameon=False)

    ax_c = fig.add_subplot(gs[1, 0])
    im0 = ax_c.imshow(
        gt_canvas, aspect="auto", origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0,
        extent=[-0.5, w - 0.5, -0.5, h - 0.5],
    )
    ax_c.add_patch(
        Rectangle(
            (ex.run_a - 0.5, -0.5),
            ex.run_b - ex.run_a,
            h,
            fill=False,
            ec="#E65100",
            lw=1.6,
        )
    )
    ax_c.set_title(f"GT coarse bit canvas (H={h})")
    ax_c.set_xlabel("horizon t")
    ax_c.set_ylabel("row (bin)")
    fig.colorbar(im0, ax=ax_c, fraction=0.046, pad=0.04)

    ax_x = fig.add_subplot(gs[1, 1])
    occ = (gt_canvas > 0.5).astype(np.float32)
    top = np.where(occ.any(axis=0), occ.shape[0] - 1 - np.argmax(occ[::-1], axis=0), -1)
    ax_x.step(t, top, where="mid", color="#1565C0", lw=1.5, label="GT top occupied row")
    ax_x.axvspan(ex.run_a - 0.5, ex.run_b - 0.5, color="#FF8A65", alpha=0.25)
    ax_x.set_ylim(-1.5, h - 0.5)
    ax_x.set_xlim(-0.5, w - 0.5)
    ax_x.set_ylabel("row")
    ax_x.set_xlabel("horizon t")
    ax_x.set_title("GT coarse bin index (from canvas)")
    ax_x.legend(loc="upper right", fontsize=8, frameon=False)

    for si, snap in enumerate(snaps_np):
        row = 2 + si
        logits = snap["bitflip_logits"]
        step_i = int(snap["step_i"])
        t_idx = int(snap["t_idx"])
        tag = f"i={step_i}/{n_steps - 1} t={t_idx}"
        if step_i == n_steps - 2:
            tag += " (penultimate)"
        elif step_i == 0:
            tag += " (early)"

        ax_l = fig.add_subplot(gs[row, 0])
        im1 = ax_l.imshow(
            logits,
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            extent=[-0.5, w - 0.5, -0.5, h - 0.5],
        )
        ax_l.add_patch(
            Rectangle(
                (ex.run_a - 0.5, -0.5),
                ex.run_b - ex.run_a,
                h,
                fill=False,
                ec="#E65100",
                lw=1.4,
            )
        )
        ax_l.set_title(f"bitflip / ε logits @ {tag}\nsampler={sampler}", fontsize=9)
        ax_l.set_xlabel("horizon t")
        ax_l.set_ylabel("row (bin)")
        fig.colorbar(im1, ax=ax_l, fraction=0.046, pad=0.04, label="logit")

        ax_p = fig.add_subplot(gs[row, 1])
        p_flip = 1.0 / (1.0 + np.exp(-logits))
        im2 = ax_p.imshow(
            p_flip,
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            extent=[-0.5, w - 0.5, -0.5, h - 0.5],
        )
        ax_p.add_patch(
            Rectangle(
                (ex.run_a - 0.5, -0.5),
                ex.run_b - ex.run_a,
                h,
                fill=False,
                ec="#80DEEA",
                lw=1.4,
            )
        )
        ax_p.set_title(f"P(flip)=sigmoid(logits) @ {tag}", fontsize=9)
        ax_p.set_xlabel("horizon t")
        ax_p.set_ylabel("row (bin)")
        fig.colorbar(im2, ax=ax_p, fraction=0.046, pad=0.04)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.pack.is_file():
        raise FileNotFoundError(args.pack)
    if not args.ckpt.is_dir():
        raise FileNotFoundError(args.ckpt)

    sampler = str(args.sampler).lower()
    if sampler in {"anchor", "deterministic_anchor"}:
        raise RuntimeError(
            "Refusing sampler=anchor: no multi-step reverse schedule. "
            "Use quad_t (probabilistic eval default) or ddim."
        )
    if int(args.num_inference_steps) < 2:
        raise RuntimeError("num_inference_steps must be ≥ 2")

    snap_idx_arg = (
        _parse_int_list(args.snapshot_indices) if args.snapshot_indices else None
    )
    snap_fracs = _parse_float_list(args.snapshot_fracs)
    snapshot_step_is = _resolve_snapshot_indices(
        n_steps=int(args.num_inference_steps),
        snapshot_indices=snap_idx_arg,
        snapshot_fracs=snap_fracs,
    )

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}")
    max_scale = float(_max_scale_from_ckpt_metadata(args.ckpt, DATASET))
    if max_scale <= 0.0:
        raise RuntimeError(f"invalid max_scale={max_scale}")

    from models.diffusion_tsf.pipeline.config import load_experiment_config
    from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import (
        patch_stage_globals,
    )
    from models.diffusion_tsf.pipeline.state import PipelineState

    cfg = load_experiment_config(args.config, cli_overrides={"dataset": DATASET})
    state0 = PipelineState.from_config(cfg)
    state0.dataset = DATASET
    state0.subset_id = DATASET
    patch_globals(pipeline_mod, state0, honor_dataset_windows=True)
    patch_stage_globals(pipeline_mod, state0, "coarse", honor_dataset_windows=True)

    if bool(getattr(pipeline_mod, "USE_ORDINAL_WINDOW_NORM", False)):
        raise RuntimeError("expected use_ordinal_window_norm=False")
    if not bool(getattr(pipeline_mod, "USE_WINDOW_NORMALIZATION", False)):
        raise RuntimeError("expected use_window_normalization=True")

    coarse_h = int(getattr(pipeline_mod, "COARSE_IMAGE_HEIGHT", 0) or state0.coarse_image_height)
    canvas_h = int(getattr(state0, "patch_refine_canvas_height", 0) or 0)
    if coarse_h != 16:
        raise RuntimeError(f"expected coarse H=16, got {coarse_h}")
    if canvas_h != 128:
        raise RuntimeError(f"expected canvas128, got {canvas_h}")

    lookback = int(args.lookback or pipeline_mod.LOOKBACK_LENGTH)
    horizon = int(args.horizon or pipeline_mod.FORECAST_LENGTH)
    std_floor = float(getattr(pipeline_mod, "WINDOW_NORM_STD_FLOOR", 0.1))
    bin_width = 2.0 * max_scale / float(coarse_h)
    flat_eps = float(args.flat_eps_frac) * bin_width

    pack = dict(np.load(args.pack, allow_pickle=True))
    for k in ("past", "y_true", "series_starts", "indices", "kind"):
        if k not in pack:
            raise KeyError(f"pack missing {k}")
    kind = str(np.asarray(pack["kind"]).reshape(-1)[0])
    if kind != "patch_refine":
        raise RuntimeError(f"expected kind=patch_refine, got {kind!r}")

    past = np.asarray(pack["past"], dtype=np.float32)
    y_true = np.asarray(pack["y_true"], dtype=np.float32)
    series_starts = np.asarray(pack["series_starts"], dtype=np.int64)
    indices = np.asarray(pack["indices"], dtype=np.int64)

    test_start_min = VAL_END - lookback
    test_mask = series_starts >= test_start_min
    if not bool(np.any(test_mask)):
        raise RuntimeError(f"no test windows (series_starts >= {test_start_min})")

    past_t = past[test_mask]
    y_t = y_true[test_mask]
    ss_t = series_starts[test_mask]
    idx_t = indices[test_mask]

    z_gt, bins_gt = _window_norm_z_and_bins(
        past_t, y_t, max_scale=max_scale, coarse_h=coarse_h, std_floor=std_floor
    )
    flats, wobbles = _pick_examples(
        bins_gt=bins_gt,
        z_gt=z_gt,
        indices=idx_t,
        series_starts=ss_t,
        flat_eps=flat_eps,
        min_run=int(args.min_run),
        n_per_class=int(args.n_per_class),
    )
    examples = flats + wobbles
    print(
        f"[coarse_viz] device={device} test_windows={past_t.shape[0]} "
        f"picked flat={len(flats)} wobbly={len(wobbles)} "
        f"sampler={sampler} steps={args.num_inference_steps} "
        f"snapshot_i={snapshot_step_is} flat_eps={flat_eps:.6g}",
        flush=True,
    )
    for e in examples:
        print(
            f"  {e.kind}: v={e.variate}({ETTH2_NAMES[e.variate]}) "
            f"local={e.local_idx} pool={e.pool_idx} "
            f"run=[{e.run_a},{e.run_b}) z_range={e.z_range:.4f}",
            flush=True,
        )

    run, stages, _kind = load_ablation_run(DATASET, args.ckpt)
    state = _build_state(args.ckpt, DATASET, run_subset_id(run), args.config)
    resolve_pipeline_data_subset(state)
    pipeline_mod.GLOBAL_ORDINAL_LADDER = None
    state.extra.pop("global_ordinal_ladder", None)
    patch_globals(pipeline_mod, state, honor_dataset_windows=True)

    guidance = None
    if bool(state.use_guidance_channel) or not bool(state.disable_cross_attention):
        gpath, gtype = _resolve_guidance_ckpt(args.ckpt, run_subset_id(run), "auto")
        guidance = load_wrapped_guidance(
            str(gpath),
            len(run_variate_indices(run)),
            device,
            guidance_type=gtype,
            dataset_lookback=lookback,
            dataset_horizon=horizon,
        )
    coarse = _load_stage_model(
        state,
        "coarse",
        stages["coarse_pt"],
        guidance,
        len(run_variate_indices(run)),
        device,
        strict_non_guidance_shapes=True,
    )
    if int(coarse.config.image_height) != coarse_h:
        raise RuntimeError(
            f"coarse image_height={coarse.config.image_height} != {coarse_h}"
        )
    if str(coarse.config.prediction_target) != "epsilon":
        raise RuntimeError(
            f"expected prediction_target=epsilon on this leaf, got "
            f"{coarse.config.prediction_target!r}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Dict[str, Any]] = []
    schedule_meta: Optional[Dict[str, Any]] = None

    for ex in examples:
        past_b = torch.from_numpy(past_t[ex.local_idx : ex.local_idx + 1]).to(device)
        fut_b = torch.from_numpy(y_t[ex.local_idx : ex.local_idx + 1]).to(device)

        past_norm, future_norm, _stats = coarse._normalize_sequence(past_b, fut_b)
        assert future_norm is not None
        maps = coarse._encode_staged_maps(future_norm)
        gt_canvas = maps["coarse"][0, ex.variate].detach().cpu().numpy().astype(np.float32)
        if gt_canvas.shape[0] != coarse_h:
            raise RuntimeError(f"GT coarse H={gt_canvas.shape[0]} != {coarse_h}")

        cap = _coarse_generate_capture_snapshots(
            coarse,
            past_b,
            sampler=sampler,
            num_steps=int(args.num_inference_steps),
            seed=int(args.seed) + 1009 * ex.local_idx + 17 * ex.variate,
            snapshot_step_is=snapshot_step_is,
        )
        if schedule_meta is None:
            schedule_meta = {
                "sampler": cap["sampler"],
                "n_steps": cap["n_steps"],
                "step_indices": cap["step_indices"],
                "snapshot_inference_indices": cap["snapshot_step_is"],
                "snapshot_diffusion_t": cap["snapshot_ts"],
                "penultimate_step_i": cap["penultimate_step_i"],
                "penultimate_t": cap["penultimate_t"],
                "prediction_target": cap["prediction_target"],
                "note": (
                    "Snapshots are model forwards at reverse-schedule inference "
                    f"indices {cap['snapshot_step_is']} → diffusion t={cap['snapshot_ts']} "
                    f"(N={cap['n_steps']}, final schedule entry t=0 not plotted). "
                    "Eval point forecast uses anchor (1 step); this viz uses "
                    "probabilistic quad_t/N from the leaf YAML."
                ),
            }
        elif cap["step_indices"] != schedule_meta["step_indices"]:
            raise RuntimeError("inconsistent reverse schedules across examples")
        elif cap["snapshot_step_is"] != schedule_meta["snapshot_inference_indices"]:
            raise RuntimeError("inconsistent snapshot indices across examples")

        k_ov = int(getattr(coarse.config, "lookback_overlap", 0) or 0)
        hz = int(horizon)
        snaps_np: List[Dict[str, Any]] = []
        npz_kw: Dict[str, Any] = {
            "gt_canvas": gt_canvas,
            "z_1d": z_gt[ex.local_idx, ex.variate],
            "bins_1d": bins_gt[ex.local_idx, ex.variate],
            "lookback_overlap_trimmed": np.int32(k_ov),
            "snapshot_step_is": np.asarray(cap["snapshot_step_is"], dtype=np.int32),
            "snapshot_ts": np.asarray(cap["snapshot_ts"], dtype=np.int32),
        }
        for snap in cap["snapshots"]:
            flip_full = snap["bitflip_logits"][0, ex.variate].detach().cpu().numpy().astype(np.float32)
            zt_full = snap["zt_logits"][0, ex.variate].detach().cpu().numpy().astype(np.float32)
            xt_full = snap["xt"][0, ex.variate].detach().cpu().numpy().astype(np.float32)
            flip_logits = _trim_hz(flip_full, hz, k_ov)
            zt_np = _trim_hz(zt_full, hz, k_ov)
            xt_np = _trim_hz(xt_full, hz, k_ov)
            if gt_canvas.shape[-1] != hz or flip_logits.shape[-1] != hz:
                raise RuntimeError(
                    f"width mismatch gt={gt_canvas.shape} logits={flip_logits.shape} hz={hz}"
                )
            si = int(snap["step_i"])
            ti = int(snap["t_idx"])
            snaps_np.append(
                {
                    "step_i": si,
                    "t_idx": ti,
                    "bitflip_logits": flip_logits,
                }
            )
            npz_kw[f"bitflip_logits_i{si}_t{ti}"] = flip_logits
            npz_kw[f"zt_logits_i{si}_t{ti}"] = zt_np
            npz_kw[f"xt_i{si}_t{ti}"] = xt_np

        vname = ETTH2_NAMES[ex.variate]
        fname = (
            f"{ex.kind}_v{ex.variate}_{vname}_local{ex.local_idx}_pool{ex.pool_idx}_"
            f"run{ex.run_a}-{ex.run_b}_bin{ex.bin_id}.png"
        )
        out_path = args.output_dir / fname
        _plot_example(
            out_path=out_path,
            title=(
                f"ETTh2 coarse H=16 | {ex.kind} | v={ex.variate} ({vname}) | "
                f"local={ex.local_idx} pool={ex.pool_idx} ss={ex.series_start}"
            ),
            gt_canvas=gt_canvas,
            snaps_np=snaps_np,
            z_1d=z_gt[ex.local_idx, ex.variate],
            bins_1d=bins_gt[ex.local_idx, ex.variate],
            ex=ex,
            n_steps=int(cap["n_steps"]),
            sampler=sampler,
            flat_eps=flat_eps,
            dpi=int(args.dpi),
        )
        np.savez_compressed(args.output_dir / fname.replace(".png", ".npz"), **npz_kw)
        written.append(
            {
                **asdict(ex),
                "variate_name": vname,
                "png": str(out_path),
                "snapshot_inference_indices": list(cap["snapshot_step_is"]),
                "snapshot_diffusion_t": list(cap["snapshot_ts"]),
                "pred_canvas_shape": list(cap["generate_out"]["future_2d_coarse"].shape),
            }
        )
        print(f"[coarse_viz] wrote {out_path}", flush=True)

    meta = {
        "dataset": DATASET,
        "pack": str(args.pack),
        "ckpt": str(args.ckpt),
        "config": args.config,
        "coarse_image_height": coarse_h,
        "patch_refine_canvas_height": canvas_h,
        "max_scale": max_scale,
        "coarse_bin_width": bin_width,
        "flat_eps_frac": float(args.flat_eps_frac),
        "flat_eps_abs": flat_eps,
        "min_run": int(args.min_run),
        "lookback": lookback,
        "horizon": horizon,
        "train_end": TRAIN_END,
        "val_end": VAL_END,
        "test_series_start_min": test_start_min,
        "n_test_windows": int(past_t.shape[0]),
        "device": str(device),
        "seed": int(args.seed),
        "plotted_inference_indices": list(snapshot_step_is),
        "plotted_diffusion_t": (
            None
            if schedule_meta is None
            else list(schedule_meta["snapshot_diffusion_t"])
        ),
        "schedule": schedule_meta,
        "legend": {
            "gt_canvas": "Coarse-stage H=16 binary CDF occupancy (GT encode, window-norm).",
            "orange_box": "Maximal ≥3 same-bin run used for flat/wobbly class.",
            "bitflip_logits": (
                "Primary DiT head logits at selected reverse-schedule inference "
                "indices (prediction_target=epsilon → bitflip / ε head). "
                "Shared |logit| clim across timesteps within each figure."
            ),
            "p_flip": "sigmoid(bitflip_logits).",
            "flat_vs_wobbly": (
                f"Same ≥{args.min_run} identical coarse bins; "
                f"flat iff max(z)-min(z) ≤ {args.flat_eps_frac}×bin_width."
            ),
        },
        "examples": written,
    }
    meta_path = args.output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[coarse_viz] wrote {meta_path}", flush=True)
    if schedule_meta is not None:
        print(
            f"[coarse_viz] snapshots: i={schedule_meta['snapshot_inference_indices']} "
            f"t={schedule_meta['snapshot_diffusion_t']} "
            f"schedule={schedule_meta['step_indices']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
