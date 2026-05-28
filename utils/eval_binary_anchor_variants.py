#!/usr/bin/env python3
"""Evaluate binary-anchor ablation variants with the matrix metric stack."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import eval_mmpd_gaussian_anchor as base


def resolve_eval_window_lengths(
    metadata: Dict,
    dataset: str,
    default_lookback: int,
    default_horizon: int,
) -> Tuple[int, int]:
    """Match training windows from checkpoint metadata (DALIA uses 80/20, not 96/96)."""
    tuned = metadata.get("tuned_params") or {}
    lb = tuned.get("lookback_length", metadata.get("lookback_length"))
    hz = tuned.get("forecast_length", metadata.get("forecast_length"))
    if lb is None or hz is None:
        if dataset == "dalia":
            from models.diffusion_tsf.dalia_data import dalia_window_lengths
            d_lb, d_hz = dalia_window_lengths()
            lb = lb if lb is not None else d_lb
            hz = hz if hz is not None else d_hz
        else:
            lb = lb if lb is not None else default_lookback
            hz = hz if hz is not None else default_horizon
    return int(lb), int(hz)


def parse_variant_root(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("variant roots must be LABEL=PATH")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("variant label cannot be empty")
    return label, Path(raw_path)


def _resolve_itrans_ckpt(root: Path, subset_id: str, dataset: str, role: str) -> Optional[Path]:
    """Find guidance or full-dataset baseline iTrans checkpoint under ``root``."""
    names = {
        "guidance": (f"{subset_id}_itransformer_finetuned.pt", f"{dataset}_itransformer_finetuned.pt"),
        "full_baseline": (
            f"{subset_id}_itrans_full_dataset.pt",
            f"{dataset}_itrans_full_dataset.pt",
        ),
    }
    for name in names[role]:
        path = root / name
        if path.exists():
            return path
    return None


def find_binary_run(root: Path, dataset: str) -> base.AnchorRun:
    root = root.resolve()
    for meta_path in root.glob("*/metadata.json"):
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)
        meta_dataset = meta.get("dataset_name") or meta.get("dataset")
        if meta_dataset != dataset:
            continue
        subset_id = meta.get("subset_id", dataset)
        best_pt = meta_path.parent / "best.pt"
        itrans_pt = _resolve_itrans_ckpt(root, subset_id, dataset, "guidance")
        if best_pt.exists() and itrans_pt is not None:
            return base.AnchorRun(
                variant="binary",
                dataset=dataset,
                root=root,
                subset_dir=meta_path.parent,
                best_pt=best_pt,
                itrans_pt=itrans_pt,
                metadata=meta,
            )
    raise FileNotFoundError(f"No completed binary run for {dataset} under {root}")


def safe_model_id(label: str) -> str:
    return "binary_" + "".join(c if c.isalnum() else "_" for c in label.lower()).strip("_")


def expected_partial_paths(
    output_dir: Path,
    datasets: Sequence[str],
    variant_labels: Sequence[str],
) -> List[Path]:
    paths: List[Path] = []
    for dataset in datasets:
        for label in variant_labels:
            paths.append(
                base.partial_metrics_path(output_dir, dataset, safe_model_id(label))
            )
    return paths


def missing_partial_paths(
    output_dir: Path,
    datasets: Sequence[str],
    variant_labels: Sequence[str],
) -> List[Path]:
    return [p for p in expected_partial_paths(output_dir, datasets, variant_labels) if not p.exists()]


def write_outputs(args: argparse.Namespace, results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_args = base.jsonable_args(args)
    manifest_args["variant_root"] = [(label, str(path)) for label, path in args.variant_root]
    manifest = {
        "args": manifest_args,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with (args.output_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    rows: List[Dict[str, float]] = []
    keys = set()
    for dataset, by_model in results.items():
        for model, metrics in by_model.items():
            row = {"dataset": dataset, "model": model}
            row.update(metrics)
            rows.append(row)
            keys.update(metrics)
    fieldnames = ["dataset", "model"] + sorted(keys)
    with (args.output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_results(args: argparse.Namespace) -> Dict[str, Dict[str, Dict[str, float]]]:
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dataset in args.datasets:
        results[dataset] = {}
        for label, _root in args.variant_root:
            model_id = safe_model_id(label)
            metrics = base.load_partial_metrics(args.output_dir, dataset, model_id)
            if metrics is not None:
                results[dataset][model_id] = metrics
    return results


def fmt(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4g}"


def build_report(args: argparse.Namespace, results: Dict[str, Dict[str, Dict[str, float]]]) -> Path:
    report_path = args.report_path or REPO_ROOT / "reports" / f"{args.output_dir.name}_binary_variant_report.md"
    labels = [(safe_model_id(label), label) for label, _root in args.variant_root]
    lines = [
        f"# Binary h128 cross-var ablation report ({args.output_dir.name})",
        "",
        f"- **Run dir:** `{args.output_dir.relative_to(REPO_ROOT)}`",
        f"- **Samples:** {args.sample_num} draws, sampler `{args.anchor_prob_sampler}`, steps {args.num_sampling_steps}",
        f"- **Test subset:** {args.test_fraction:.0%} of windows, seed {args.seed}",
        "- **Metrics:** deterministic anchor MSE/MAE, CRPS, top-k modes, and texture/per-sample texture from `utils/eval_mmpd_gaussian_anchor.py`",
        "",
        "## Core Metrics",
        "",
    ]
    core = ["mse", "mae", "crps", "top3_mse", "top3_mae"]
    for dataset in args.datasets:
        lines.append(f"### {dataset}")
        lines.append("")
        lines.append("| Model | MSE | MAE | CRPS | top3 MSE | top3 MAE | windows | vars |")
        lines.append("|-------|----:|----:|-----:|---------:|---------:|--------:|-----:|")
        for model_id, label in labels:
            row = results.get(dataset, {}).get(model_id, {})
            lines.append(
                f"| {label} | "
                f"{fmt(row.get('mse'))} | {fmt(row.get('mae'))} | {fmt(row.get('crps'))} | "
                f"{fmt(row.get('top3_mse'))} | {fmt(row.get('top3_mae'))} | "
                f"{fmt(row.get('n_windows'))} | {fmt(row.get('n_variates'))} |"
            )
        itrans_row = results.get(dataset, {}).get("itrans_full_dataset")
        if itrans_row:
            lines.append(
                f"| iTrans full-train baseline | "
                f"{fmt(itrans_row.get('mse'))} | {fmt(itrans_row.get('mae'))} | "
                f"- | - | - | "
                f"{fmt(itrans_row.get('n_windows'))} | {fmt(itrans_row.get('n_variates'))} |"
            )
        lines.append("")

    texture_keys = [
        ("ordinal_jsd", "Ordinal JSD"),
        ("rqa_dist", "RQA distance"),
        ("variogram_dist", "Variogram distance"),
        ("path_signature_dist", "Path signature distance"),
        ("per_sample_mean_ordinal_jsd", "Per-sample ordinal JSD"),
        ("per_sample_mean_rqa_dist", "Per-sample RQA distance"),
        ("per_sample_mean_variogram_dist", "Per-sample variogram distance"),
        ("per_sample_mean_path_signature_dist", "Per-sample path signature distance"),
    ]
    lines.extend(["## Texture Metrics", ""])
    for key, title in texture_keys:
        if not any(key in row for by_model in results.values() for row in by_model.values()):
            continue
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| Dataset | " + " | ".join(label for _model_id, label in labels) + " |")
        lines.append("|---------|" + "|".join("---:" for _ in labels) + "|")
        for dataset in args.datasets:
            cells = [fmt(results.get(dataset, {}).get(model_id, {}).get(key)) for model_id, _label in labels]
            lines.append(f"| {dataset} | " + " | ".join(cells) + " |")
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def eval_itrans_full_dataset_baseline(
    args: argparse.Namespace,
    run: base.AnchorRun,
    indices: Sequence[int],
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate the iTrans-only model trained on the full train split (same test windows)."""
    from models.diffusion_tsf import train_multivariate_pipeline as pipeline

    subset_id = run.metadata.get("subset_id", run.dataset)
    variate_indices = run.metadata["variate_indices"]
    lb, hz = resolve_eval_window_lengths(
        run.metadata, run.dataset, args.lookback, args.horizon,
    )

    pipeline.CHECKPOINT_DIR = str(run.root.resolve())
    full_ckpt = _resolve_itrans_ckpt(run.root, subset_id, run.dataset, "full_baseline")
    if full_ckpt is None:
        print(f"[itrans] training full-dataset baseline for {subset_id}...")
        full_ckpt = Path(
            pipeline.train_subset_itransformer_full_baseline(
                run.dataset,
                list(variate_indices),
                subset_id,
                device,
                smoke_test=False,
            )
        )
    else:
        full_ckpt = Path(full_ckpt)

    subset = base.load_tsf_test_subset(
        run.dataset, variate_indices, indices, lb, hz,
    )
    n_iv = len(variate_indices)
    model = pipeline.load_itransformer_from_checkpoint(str(full_ckpt), n_iv, device)
    preds, targets = [], []
    with torch.no_grad():
        for past, future in torch.utils.data.DataLoader(
            subset, batch_size=args.anchor_batch_size, shuffle=False,
        ):
            past = past.to(device)
            B, C, _ = past.shape
            x_enc = past.permute(0, 2, 1)
            seq_sl = getattr(model, "seq_len", x_enc.shape[1])
            if x_enc.shape[1] > seq_sl:
                x_enc = x_enc[:, -seq_sl:, :]
            x_dec = torch.zeros(B, hz, C, device=device, dtype=past.dtype)
            out = model(x_enc, None, x_dec, None)
            if isinstance(out, tuple):
                out = out[0]
            preds.append(out.permute(0, 2, 1).cpu())
            overlap = pipeline.LOOKBACK_OVERLAP
            if overlap > 0:
                future = future[..., overlap:]
            targets.append(future)
    pred_t = torch.cat(preds, dim=0)
    tgt_t = torch.cat(targets, dim=0)
    mse = torch.nn.functional.mse_loss(pred_t, tgt_t).item()
    mae = torch.nn.functional.l1_loss(pred_t, tgt_t).item()
    print(f"[itrans] full-train baseline {subset_id}: MSE={mse:.4f} MAE={mae:.4f} ({len(indices)} windows)")
    return {
        "mse": mse,
        "mae": mae,
        "n_windows": float(len(indices)),
        "n_variates": float(n_iv),
    }


def run_eval(args: argparse.Namespace) -> None:
    if len(args.datasets) != 1 or len(args.variant_root) != 1:
        raise ValueError("--phase eval expects exactly one --datasets value and one --variant-root")
    label, root = args.variant_root[0]
    dataset = args.datasets[0]
    run = find_binary_run(root, dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lb, hz = resolve_eval_window_lengths(
        run.metadata, dataset, args.lookback, args.horizon,
    )
    if (lb, hz) != (args.lookback, args.horizon):
        print(f"[eval] {dataset}: using lookback={lb} horizon={hz} from run metadata")
    args.lookback = lb
    args.horizon = hz
    indices = base.get_or_create_indices(args, dataset, run.metadata["variate_indices"])
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{safe_model_id(label)}_{dataset}.npz"
    if raw_path.exists() and not args.force_anchor_eval:
        with np.load(raw_path) as data:
            pack = {key: data[key] for key in data.files}
    else:
        device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}")
        pack = base.evaluate_anchor(args, run, indices, device)
        np.savez_compressed(raw_path, **pack)
    metrics = base.summarize_for_profile(pack, args, dataset)
    base.write_partial_metrics(args.output_dir, dataset, safe_model_id(label), metrics)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}")
    itrans_metrics = eval_itrans_full_dataset_baseline(args, run, indices, device)
    base.write_partial_metrics(args.output_dir, dataset, "itrans_full_dataset", itrans_metrics)


def run_merge(args: argparse.Namespace) -> None:
    results = collect_results(args)
    write_outputs(args, results)
    report_path = build_report(args, results)
    print(f"Wrote metrics to {args.output_dir / 'metrics.json'}")
    print(f"Wrote report to {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["eval", "merge", "check-partials"], default="eval")
    parser.add_argument("--variant-root", action="append", type=parse_variant_root, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--indices-dir", type=Path, default=None)
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--test-fraction", type=float, default=0.5)
    parser.add_argument("--test-max-items", type=int, default=None)
    parser.add_argument("--force-indices", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sample-num", type=int, default=9)
    parser.add_argument("--num-sampling-steps", type=int, default=20)
    parser.add_argument("--gmm-components", type=int, default=9)
    parser.add_argument("--topk-max", type=int, default=5)
    parser.add_argument("--metrics-profile", choices=["full", "prob-core"], default="full")
    parser.add_argument("--texture-per-sample", action="store_true")
    parser.add_argument("--anchor-batch-size", type=int, default=16)
    parser.add_argument("--anchor-prob-sampler", choices=["dpmpp", "ddim", "ddpm"], default="dpmpp")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--force-anchor-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.report_path is not None:
        args.report_path = args.report_path.resolve()
    if args.phase == "eval":
        run_eval(args)
    elif args.phase == "check-partials":
        variant_labels = [label for label, _ in args.variant_root]
        missing = missing_partial_paths(args.output_dir, args.datasets, variant_labels)
        for path in missing:
            print(path.relative_to(args.output_dir))
        if missing:
            sys.exit(1)
        print(f"ok: {len(args.datasets) * len(variant_labels)} partials present")
    else:
        run_merge(args)


if __name__ == "__main__":
    main()
