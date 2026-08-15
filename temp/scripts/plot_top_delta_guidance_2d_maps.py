#!/usr/bin/env python3
"""2D occupancy maps (lookback | GT horizon | guidance) for top-delta windows.

Reads ``temp/viz_anchor_mse_delta_wn_vs_guided_p8/summary.json`` (or ``--summary``),
loads ordinal guided_p8 patch guidance + a coarse-stage DiffusionModel (encode only;
weights optional), and writes one PNG per selected window:

  rows = variates
  cols = lookback | GT horizon | guidance ghost
  (coarse CDF occupancy; optional second file for fine)

Example:

  python temp/scripts/plot_top_delta_guidance_2d_maps.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.visualize_staged_eval_2d_preds import (  # noqa: E402
    _build_state,
    _load_stage_model,
    _resolve_guidance_ckpt,
)

DEFAULT_SUMMARY = "temp/viz_anchor_mse_delta_wn_vs_guided_p8/summary.json"
DEFAULT_CKPT = (
    "results/ckpts/08-01-4519745-ETTh1-binary_patch_refine_lb336_hz96_ordinal_tuned_guided_p8"
)
DEFAULT_CFG = "configs/binary_patch_refine_lb336_hz96_ordinal_tuned_guided_p8.yaml"
DEFAULT_OUT = "temp/viz_anchor_mse_delta_wn_vs_guided_p8/guidance_2d"


def _resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p


def _plot_occupancy_triple(
    *,
    out_path: Path,
    look_maps: torch.Tensor,
    gt_maps: torch.Tensor,
    guid_maps: torch.Tensor,
    title: str,
    scale_name: str,
    max_variates: int,
) -> None:
    """look/gt/guid: (V, H, W) occupancy in [0, 1]."""
    V = min(int(look_maps.shape[0]), int(max_variates))
    fig, axes = plt.subplots(
        V, 3,
        figsize=(11.0, 1.55 * V + 0.8),
        squeeze=False,
        constrained_layout=True,
    )
    col_titles = (
        f"lookback {scale_name}",
        f"GT horizon {scale_name}",
        f"guidance {scale_name}",
    )
    panels = (look_maps, gt_maps, guid_maps)
    for vi in range(V):
        for col, (cname, panel) in enumerate(zip(col_titles, panels)):
            ax = axes[vi, col]
            img = panel[vi].detach().float().cpu().numpy()
            ax.imshow(img, aspect="auto", origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0)
            if vi == 0:
                ax.set_title(cname, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"v{vi}\n{img.shape[0]}x{img.shape[1]}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(title, fontsize=10, fontweight="semibold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _build_encode_model(
    *,
    ckpt_dir: Path,
    config_path: Path,
    dataset: str,
    device: torch.device,
    load_coarse_weights: bool,
) -> Tuple[Any, Any]:
    """Return (model, test_ds). Model has guidance attached for ghost encode."""
    from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import (
        stage_state,
    )
    from models.diffusion_tsf.train_multivariate_pipeline import (
        create_diffusion_model,
        generate_dataset_job,
        load_dataset,
        load_wrapped_guidance,
        resolve_pipeline_data_subset,
    )
    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod
    from utils.visualize_staged_forecast import _window_lengths
    from models.diffusion_tsf.pipeline.phases.staged_diffusion_finetune_hp import (
        _model_kwargs_from_tuned,
    )
    from models.diffusion_tsf.train_multivariate_pipeline import anchor_kwargs_from_params

    state = _build_state(ckpt_dir, dataset, dataset, str(config_path))
    job = generate_dataset_job(dataset)
    state.variate_indices = list(state.variate_indices or job["variate_indices"])
    resolve_pipeline_data_subset(state)

    train_stride = int((state.data_subset_resolved or {}).get("train_stride", state.window_stride))
    # eval packs used test_stride=16
    _, _, test_ds, norm_stats = load_dataset(
        dataset,
        list(state.variate_indices),
        stride=train_stride,
        test_stride=16,
        lookback=int(state.lookback_length),
        horizon=int(state.forecast_length),
        ordinal_tie_atol=float(state.ordinal_tie_atol),
        use_ordinal_window_norm=state.use_ordinal_window_norm,
    )
    if norm_stats.get("ordinal_ladder") is not None:
        state.extra["global_ordinal_ladder"] = norm_stats["ordinal_ladder"]
        pipeline_mod.GLOBAL_ORDINAL_LADDER = norm_stats["ordinal_ladder"]

    n_vars = len(state.variate_indices)
    gpath, _gtype = _resolve_guidance_ckpt(ckpt_dir, dataset, "patch_decoder")
    guidance = load_wrapped_guidance(str(gpath), n_vars, device)

    coarse_pt = ckpt_dir / dataset / "coarse" / "best.pt"
    if load_coarse_weights and coarse_pt.is_file():
        model = _load_stage_model(
            state, "coarse", coarse_pt, guidance, n_vars, device,
            strict_non_guidance_shapes=True,
        )
    else:
        # Encode + guidance forecast only — no need for trained coarse weights.
        state = stage_state(state, "coarse", honor_dataset_windows=True)
        lookback, horizon = _window_lengths(state.dataset, state)
        meta_path = ckpt_dir / dataset / "coarse" / "metadata.json"
        tuned: Dict[str, Any] = {}
        if meta_path.is_file():
            tuned = json.loads(meta_path.read_text(encoding="utf-8")).get("tuned_params") or {}
        model_kwargs = anchor_kwargs_from_params(tuned)
        model_kwargs.update(_model_kwargs_from_tuned(tuned))
        model = create_diffusion_model(
            n_variates=n_vars,
            lookback=lookback,
            horizon=horizon,
            guidance_model=guidance,
            diffusion_stage="coarse",
            use_guidance_channel=state.use_guidance_channel,
            ordinal_ladder=pipeline_mod.GLOBAL_ORDINAL_LADDER,
            **model_kwargs,
        ).to(device)
        model.eval()

    ranked = bool(getattr(test_ds, "yields_ordinal_ranks", False))
    model._ordinal_input_is_ranked = ranked
    model._ordinal_apply_ood_shift = not ranked
    return model, test_ds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", default=DEFAULT_SUMMARY)
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT)
    p.add_argument("--config", default=DEFAULT_CFG)
    p.add_argument("--dataset", default="ETTh1")
    p.add_argument("--output-dir", default=DEFAULT_OUT)
    p.add_argument("--max-variates", type=int, default=7)
    p.add_argument("--also-fine", action="store_true", default=True)
    p.add_argument("--no-fine", action="store_true")
    p.add_argument("--load-coarse-weights", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = _resolve(args.summary)
    ckpt_dir = _resolve(args.ckpt_dir)
    config_path = _resolve(args.config)
    out_dir = _resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    also_fine = bool(args.also_fine) and not bool(args.no_fine)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected: List[Dict[str, Any]] = list(summary["selected"])
    if not selected:
        raise RuntimeError(f"no selected windows in {summary_path}")

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device={device} windows={len(selected)} ckpt={ckpt_dir}", flush=True)
    model, test_ds = _build_encode_model(
        ckpt_dir=ckpt_dir,
        config_path=config_path,
        dataset=args.dataset,
        device=device,
        load_coarse_weights=bool(args.load_coarse_weights),
    )

    manifest = []
    with torch.no_grad():
        for row in selected:
            rank = int(row["rank"])
            wi = int(row["window_index"])
            ss = int(row["series_start"])
            past, future = test_ds[wi]
            past_t = past.unsqueeze(0).to(device) if not torch.is_tensor(past) else past.unsqueeze(0).to(device)
            fut_t = future.unsqueeze(0).to(device) if not torch.is_tensor(future) else future.unsqueeze(0).to(device)
            if past_t.dim() == 2:
                past_t = past_t.unsqueeze(0)
            if fut_t.dim() == 2:
                fut_t = fut_t.unsqueeze(0)

            past_norm, future_norm, stats = model._normalize_sequence(past_t, fut_t)
            guidance_norm = model._get_guidance_forecast_norm(
                past_t, past_norm, stats, int(future_norm.shape[-1]),
            )
            past_maps = model._encode_staged_maps(past_norm)
            future_maps = model._encode_staged_maps(future_norm)
            guidance_maps = model._encode_staged_maps(guidance_norm)

            title = (
                f"#{rank:02d} wi={wi} start={ss}  "
                f"mse_w={row['mse_window']:.3f} mse_o={row['mse_ordinal']:.3f} "
                f"Δ(o−w)={row['delta']:+.3f}  |  abs ordinal → CDF occupancy"
            )
            coarse_path = out_dir / f"rank{rank:02d}_wi{wi}_start{ss}_coarse.png"
            _plot_occupancy_triple(
                out_path=coarse_path,
                look_maps=past_maps["coarse"][0],
                gt_maps=future_maps["coarse"][0],
                guid_maps=guidance_maps["coarse"][0],
                title=title,
                scale_name="coarse",
                max_variates=args.max_variates,
            )
            entry = {
                "rank": rank,
                "window_index": wi,
                "series_start": ss,
                "coarse": str(coarse_path),
            }
            print(f"  wrote {coarse_path.name}", flush=True)

            if also_fine and "fine" in past_maps:
                fine_path = out_dir / f"rank{rank:02d}_wi{wi}_start{ss}_fine.png"
                _plot_occupancy_triple(
                    out_path=fine_path,
                    look_maps=past_maps["fine"][0],
                    gt_maps=future_maps["fine"][0],
                    guid_maps=guidance_maps["fine"][0],
                    title=title,
                    scale_name="fine",
                    max_variates=args.max_variates,
                )
                entry["fine"] = str(fine_path)
                print(f"  wrote {fine_path.name}", flush=True)

            # Quick numeric check: guidance vs GT MAE in absolute ordinal space.
            K = int(guidance_norm.shape[-1] - future_norm.shape[-1])
            if K < 0:
                K = 0
            g_core = guidance_norm[..., K:]
            n = min(g_core.shape[-1], future_norm.shape[-1])
            mae = float((g_core[..., :n] - future_norm[..., :n]).abs().mean().item())
            entry["guidance_vs_gt_mae_abs_rank"] = mae
            print(f"  guidance vs GT MAE (abs rank)={mae:.4f}", flush=True)
            manifest.append(entry)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
