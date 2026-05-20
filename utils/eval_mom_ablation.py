#!/usr/bin/env python3
"""Compare diffusion test metrics: sample mean vs median vs SimDiff median-of-means.

Evaluates checkpoints from the residual A/B/AB report and the May-12 default
full-pipeline runs (3539360–3539365). Uses a seeded random fraction of test
windows (default 5%) for speed.

MoM follows simdiff.md / SimDiff._rob_median_of_means:
  shuffle n samples → split into K blocks → mean each block → median → repeat R → average.

Example:
  python utils/eval_mom_ablation.py --test-fraction 0.05 --n-samples 30
  python utils/eval_mom_ablation.py --only default,ETTh1 --max-runs 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Run registry (canonical successful runs from reports/)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunSpec:
    label: str  # human tag, e.g. "default" or "exp_A"
    run_dir: Path  # results/... or results/runs/...
    experiment: str  # baseline | A | B | A+B
    dataset_key: str  # registry name: ETTh1, exchange_rate, ...


def _default_runs() -> List[RunSpec]:
    base = REPO_ROOT / "results"
    mapping = [
        ("3539360", "ETTh1"),
        ("3539361", "ETTh2"),
        ("3539362", "ETTm1"),
        ("3539363", "ETTm2"),
        ("3539364", "weather"),
        ("3539365", "exchange_rate"),
    ]
    out = []
    for job, ds in mapping:
        out.append(
            RunSpec(
                label="default",
                run_dir=base / f"05-12-{job}-default-{ds.replace('_', '-')}",
                experiment="baseline",
                dataset_key=ds,
            )
        )
    return out


def _exp_runs() -> List[RunSpec]:
    base = REPO_ROOT / "results" / "runs"
    # (run_stem, experiment, dataset registry name)
    rows = [
        ("05-19-3662565-exp_A_ETTh1", "A", "ETTh1"),
        ("05-19-3662566-exp_B_ETTh1", "B", "ETTh1"),
        ("05-19-3662567-exp_A_B_ETTh1", "A+B", "ETTh1"),
        ("05-19-3662568-exp_A_ETTh2", "A", "ETTh2"),
        ("05-19-3662569-exp_B_ETTh2", "B", "ETTh2"),
        ("05-19-3662570-exp_A_B_ETTh2", "A+B", "ETTh2"),
        ("05-18-3650646-exp_A_ETTm1", "A", "ETTm1"),
        ("05-18-3650647-exp_B_ETTm1", "B", "ETTm1"),
        ("05-18-3650648-exp_A_B_ETTm1", "A+B", "ETTm1"),
        ("05-18-3650649-exp_A_ETTm2", "A", "ETTm2"),
        ("05-18-3650650-exp_B_ETTm2", "B", "ETTm2"),
        ("05-18-3650651-exp_A_B_ETTm2", "A+B", "ETTm2"),
        ("05-19-3662571-exp_A_exchange-rate", "A", "exchange_rate"),
        ("05-19-3662572-exp_B_exchange-rate", "B", "exchange_rate"),
        ("05-19-3662573-exp_A_B_exchange-rate", "A+B", "exchange_rate"),
        ("05-18-3650652-exp_A_weather", "A", "weather"),
        ("05-18-3650653-exp_B_weather", "B", "weather"),
        ("05-18-3650654-exp_A_B_weather", "A+B", "weather"),
    ]
    return [
        RunSpec(label=f"exp_{exp}", run_dir=base / stem, experiment=exp, dataset_key=ds)
        for stem, exp, ds in rows
    ]


# ---------------------------------------------------------------------------
# MoM (SimDiff-style)
# ---------------------------------------------------------------------------


def median_of_means_once(samples: torch.Tensor, n_blocks: int) -> torch.Tensor:
    """samples: (S, C, T) → (C, T)."""
    s = samples.size(0)
    if s < 1:
        raise ValueError("empty sample stack")
    k = min(n_blocks, s)
    if k < 1:
        k = 1
    block_size = max(1, s // k)
    means: List[torch.Tensor] = []
    for i in range(k):
        start = i * block_size
        end = start + block_size if (i + 1) < k else s
        means.append(samples[start:end].mean(dim=0))
    return torch.stack(means).median(dim=0).values


def robust_median_of_means(
    samples: torch.Tensor,
    n_blocks: int,
    n_repeats: int,
    seed: int,
) -> torch.Tensor:
    """samples: (S, C, T). Average of R median-of-means draws with reshuffles."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    outs: List[torch.Tensor] = []
    for _ in range(n_repeats):
        perm = torch.randperm(samples.size(0), generator=gen)
        outs.append(median_of_means_once(samples[perm], n_blocks))
    return torch.stack(outs).mean(dim=0)


def aggregate_forecasts(
    sample_stack: torch.Tensor,
    mom_blocks: int,
    mom_repeats: int,
    seed: int,
) -> Dict[str, torch.Tensor]:
    """sample_stack: (S, B, C, T) for one batch window."""
    mean_pred = sample_stack.mean(dim=0)
    median_pred = sample_stack.median(dim=0).values
    mom_preds = []
    for b in range(sample_stack.size(1)):
        mom_preds.append(
            robust_median_of_means(
                sample_stack[:, b],
                n_blocks=mom_blocks,
                n_repeats=mom_repeats,
                seed=seed + b,
            )
        )
    mom_pred = torch.stack(mom_preds, dim=0)
    return {"mean": mean_pred, "median": median_pred, "mom": mom_pred}


def compute_batch_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    mse = F.mse_loss(pred, target).item()
    mae = F.l1_loss(pred, target).item()
    return {"mse": mse, "mae": mae}


# ---------------------------------------------------------------------------
# Model load
# ---------------------------------------------------------------------------


def infer_image_height(state_dict: Dict[str, torch.Tensor]) -> int:
    w = state_dict.get("to_2d.bin_centers")
    if w is None:
        raise KeyError("checkpoint missing to_2d.bin_centers — cannot infer image_height")
    return int(w.shape[0])


def resolve_paths(spec: RunSpec) -> Tuple[Path, Path, Path]:
    """Return (metadata.json, best.pt, itrans finetuned.pt)."""
    if spec.experiment == "baseline":
        subset = spec.dataset_key
    else:
        subset = f"exp_{spec.experiment}"
    ckpt_dir = spec.run_dir / "ckpts" / subset
    meta = ckpt_dir / "metadata.json"
    best = ckpt_dir / "best.pt"
    itrans = ckpt_dir.parent / f"{subset}_itransformer_finetuned.pt"
    return meta, best, itrans


def load_model_for_run(
    spec: RunSpec,
    device: torch.device,
) -> Tuple[object, dict]:
    import models.diffusion_tsf.train_multivariate_pipeline as pipeline
    from models.diffusion_tsf.guidance import iTransformerGuidance
    from models.diffusion_tsf.train_multivariate_pipeline import (
        FORECAST_LENGTH,
        LOOKBACK_LENGTH,
        LOOKBACK_OVERLAP,
        load_diffusion_state_keep_attached_guidance,
        load_itransformer_from_checkpoint,
        create_diffusion_model,
    )

    meta_path, ckpt_path, itrans_path = resolve_paths(spec)
    if not meta_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(f"missing ckpt/meta under {spec.run_dir}")

    with open(meta_path) as f:
        meta = json.load(f)

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    image_h = infer_image_height(blob["model_state_dict"])

    pipeline.EXPERIMENT = spec.experiment
    pipeline.IMAGE_HEIGHT = image_h
    pipeline.LOOKBACK_LENGTH = LOOKBACK_LENGTH
    pipeline.FORECAST_LENGTH = FORECAST_LENGTH
    pipeline.LOOKBACK_OVERLAP = LOOKBACK_OVERLAP
    pipeline.GUIDANCE_PENALTY_WEIGHT = 0.0

    n_vars = len(meta["variate_indices"])
    itrans = load_itransformer_from_checkpoint(str(itrans_path), n_vars, device)
    guidance = iTransformerGuidance(itrans)

    model = create_diffusion_model(n_variates=n_vars).to(device)
    model.set_guidance_model(guidance)
    load_diffusion_state_keep_attached_guidance(model, blob["model_state_dict"])
    model.eval()

    return model, meta


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_run(
    spec: RunSpec,
    device: torch.device,
    test_fraction: float,
    seed: int,
    n_samples: int,
    mom_blocks: int,
    mom_repeats: int,
    batch_size: int,
    num_inference_steps: int,
) -> Dict:
    from models.diffusion_tsf.train_multivariate_pipeline import load_dataset

    model, meta = load_model_for_run(spec, device)
    dataset_name = meta["dataset_name"]
    variate_indices = meta["variate_indices"]
    _, _, test_ds, _ = load_dataset(dataset_name, variate_indices, stride=1)

    n_full = len(test_ds)
    n_eval = max(1, int(math.ceil(n_full * test_fraction)))
    rng = np.random.default_rng(seed)
    eval_idx = sorted(rng.choice(n_full, size=min(n_eval, n_full), replace=False).tolist())
    test_ds = Subset(test_ds, eval_idx)

    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    K = getattr(model.config, "lookback_overlap", 0)
    gen_kwargs = {"sampler": "dpmpp", "num_inference_steps": num_inference_steps}

    sums = {
        "mean": {"mse": 0.0, "mae": 0.0, "n": 0},
        "median": {"mse": 0.0, "mae": 0.0, "n": 0},
        "mom": {"mse": 0.0, "mae": 0.0, "n": 0},
    }

    t0 = time.perf_counter()
    for batch_idx, (past, future) in enumerate(loader):
        past = past.to(device)
        if K > 0:
            future = future[..., K:]

        stack = []
        for s_idx in range(n_samples):
            torch.manual_seed(1000 + s_idx * 17 + batch_idx + seed)
            out = model.generate(past, **gen_kwargs)
            pred = out.get("prediction", out.get("forecast"))
            stack.append(pred.cpu())
        stack_t = torch.stack(stack, dim=0)  # (S, B, C, T)

        agg = aggregate_forecasts(
            stack_t,
            mom_blocks=mom_blocks,
            mom_repeats=mom_repeats,
            seed=seed + batch_idx * 10007,
        )
        target = future.cpu()
        for name, pred in agg.items():
            m = compute_batch_metrics(pred, target)
            bsz = pred.size(0)
            sums[name]["mse"] += m["mse"] * bsz
            sums[name]["mae"] += m["mae"] * bsz
            sums[name]["n"] += bsz

    wall = time.perf_counter() - t0
    metrics = {}
    for name, acc in sums.items():
        n = max(acc["n"], 1)
        metrics[name] = {
            "mse": acc["mse"] / n,
            "mae": acc["mae"] / n,
        }

    return {
        "spec": {
            "label": spec.label,
            "experiment": spec.experiment,
            "dataset": dataset_name,
            "run_dir": str(spec.run_dir),
        },
        "eval": {
            "test_fraction": test_fraction,
            "n_eval_windows": n_eval,
            "n_full_windows": n_full,
            "n_samples": n_samples,
            "mom_blocks": mom_blocks,
            "mom_repeats": mom_repeats,
            "seed": seed,
            "image_height_inferred": int(model.config.image_height),
        },
        "metrics": metrics,
        "wall_sec": wall,
    }


def delta_pct(base: float, other: float) -> float:
    if base == 0:
        return float("nan")
    return (base - other) / base * 100.0


def format_report_row(r: Dict) -> str:
    m = r["metrics"]
    mean_mse = m["mean"]["mse"]
    med_mse = m["median"]["mse"]
    mom_mse = m["mom"]["mse"]
    d_med = delta_pct(mean_mse, med_mse)
    d_mom = delta_pct(mean_mse, mom_mse)
    sp = r["spec"]
    return (
        f"| {sp['dataset']} | {sp['label']} | {sp['experiment']} | "
        f"{mean_mse:.4f} | {med_mse:.4f} ({d_med:+.1f}%) | "
        f"{mom_mse:.4f} ({d_mom:+.1f}%) | {r['eval']['n_eval_windows']}/{r['eval']['n_full_windows']} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MoM vs mean diffusion eval ablation")
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=30, help="Stochastic traces per window")
    parser.add_argument("--mom-blocks", type=int, default=5, help="K blocks for MoM (SimDiff n_b)")
    parser.add_argument("--mom-repeats", type=int, default=10, help="R shuffle repeats (SimDiff rmom)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--only", type=str, default="", help="Comma filter: default,exp_A,ETTh1,...")
    parser.add_argument("--max-runs", type=int, default=0, help="Limit runs (0 = all)")
    parser.add_argument("--out-json", type=Path, default=REPO_ROOT / "reports/mom_ablation_results.json")
    parser.add_argument("--out-md", type=Path, default=REPO_ROOT / "reports/mom_ablation_report.md")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filters = {x.strip() for x in args.only.split(",") if x.strip()}

    specs = _default_runs() + _exp_runs()
    if filters:
        specs = [
            s
            for s in specs
            if s.label in filters
            or s.experiment in filters
            or s.dataset_key in filters
            or any(f in str(s.run_dir) for f in filters)
        ]

    if args.max_runs > 0:
        specs = specs[: args.max_runs]

    if not specs:
        print("No runs matched filters.")
        sys.exit(1)

    print(f"Device={device} | {len(specs)} runs | test_fraction={args.test_fraction}")
    print(f"n_samples={args.n_samples} mom_blocks={args.mom_blocks} mom_repeats={args.mom_repeats}")

    results: List[Dict] = []
    for i, spec in enumerate(specs):
        print(f"\n[{i+1}/{len(specs)}] {spec.run_dir.name} ({spec.experiment})")
        try:
            r = evaluate_run(
                spec,
                device=device,
                test_fraction=args.test_fraction,
                seed=args.seed,
                n_samples=args.n_samples,
                mom_blocks=args.mom_blocks,
                mom_repeats=args.mom_repeats,
                batch_size=args.batch_size,
                num_inference_steps=args.inference_steps,
            )
            results.append(r)
            print(
                f"  mean MSE={r['metrics']['mean']['mse']:.4f} "
                f"median={r['metrics']['median']['mse']:.4f} "
                f"mom={r['metrics']['mom']['mse']:.4f} "
                f"({r['eval']['n_eval_windows']} windows, {r['wall_sec']:.0f}s)"
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append(
                {
                    "spec": {
                        "label": spec.label,
                        "experiment": spec.experiment,
                        "dataset": spec.dataset_key,
                        "run_dir": str(spec.run_dir),
                    },
                    "error": str(e),
                }
            )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    with open(args.out_json, "w") as f:
        json.dump({"config": cfg, "results": results}, f, indent=2)

    lines = [
        "# Median-of-means ensemble ablation (5% test subset)\n\n",
        "Compares **sample mean** (current pipeline default with `n_samples` averaged), "
        "**per-window median** over samples, and **SimDiff-style robust MoM** "
        f"(`n_samples={args.n_samples}`, `K={args.mom_blocks}` blocks, `R={args.mom_repeats}` repeats).\n\n",
        f"**Test subset:** `{args.test_fraction*100:.0g}%` of windows, seed `{args.seed}`.\n\n",
        "**Δ MSE %** = improvement vs mean (positive ⇒ lower MSE than mean).\n\n",
        "| Dataset | Tag | Exp | Mean MSE | Median MSE | MoM MSE | Windows |\n",
        "|---------|-----|-----|----------|------------|---------|----------|\n",
    ]
    for r in results:
        if "error" in r:
            sp = r["spec"]
            lines.append(
                f"| {sp.get('dataset','?')} | {sp.get('label','?')} | {sp.get('experiment','?')} "
                f"| — | — | — | ERROR: {r['error'][:40]} |\n"
            )
        else:
            lines.append(format_report_row(r) + "\n")

    lines.append(f"\nRaw JSON: `{args.out_json}`\n")
    with open(args.out_md, "w") as f:
        f.writelines(lines)

    print(f"\nWrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
