"""TP/TN/FP/FN sample plots for the univariate real-vs-fake discriminator.

Uses the univariate dataset / checkpoint naming from
``utils/eval_discriminator_binary_vs_mmpd_univariate.py`` and the h96 ordinal
evaluator. When a wandb run is active, PNGs are logged automatically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.eval_discriminator_binary_vs_mmpd_univariate import UnivariateRealVsFakeDataset
from utils.disc_shared import (
    InvertedSliceDiscriminator,
    stable_hash,
)


def bucket_name(label: int, pred: int) -> str:
    if label == 1 and pred == 1:
        return "TP"
    if label == 0 and pred == 0:
        return "TN"
    if label == 0 and pred == 1:
        return "FP"
    return "FN"


def summarize_buckets(records: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for row in records:
        counts[bucket_name(row["label"], row["pred"])] += 1
    return counts


def pick_examples(
    records: List[Dict[str, Any]],
    per_bucket: int,
    rng: np.random.Generator,
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {key: [] for key in ("TP", "TN", "FP", "FN")}
    for bucket in out:
        pool = [row for row in records if bucket_name(row["label"], row["pred"]) == bucket]
        if not pool:
            continue
        pool.sort(key=lambda row: row["prob_fake"], reverse=bucket in ("TP", "FP"))
        n = min(per_bucket, len(pool))
        if n == len(pool):
            out[bucket] = pool
            continue
        top = pool[: max(1, n // 2)]
        rest = pool[max(1, n // 2) :]
        extra = min(n - len(top), len(rest))
        if extra:
            idx = rng.choice(len(rest), size=extra, replace=False)
            top.extend(rest[i] for i in idx)
        out[bucket] = top[:n]
    return out


def plot_example(
    row: Dict[str, Any],
    *,
    variate: int,
    lookback_tail: int,
    title: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    past = row["past"][variate, -lookback_tail:]
    gt = row.get("gt", row["candidate"])[variate]
    fake = row.get("fake", row["candidate"])[variate]
    t_past = np.arange(-len(past), 0)
    t_horizon = np.arange(len(gt))
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(t_past, past, color="#444444", lw=1.4, label="lookback")
    ax.plot(t_horizon, gt, color="#1f77b4", lw=2.0, label="GT")
    ax.plot(t_horizon, fake, color="#d62728", lw=2.0, alpha=0.9, label="model pred")
    ax.axvline(0, color="black", ls="--", lw=0.8, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("time (steps; 0 = horizon start)")
    ax.set_ylabel("value")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def univariate_checkpoint_path(
    output_dir: Path, dataset: str, fake_source: str, slice_len: int
) -> Path:
    return (
        Path(output_dir)
        / "checkpoints"
        / f"{dataset}_{fake_source}_L{slice_len}_univariate_discriminator.pt"
    )


def load_univariate_checkpoint(
    path: Path,
    device: torch.device,
    *,
    lookback: int,
    horizon: int,
    slice_len: int,
) -> Tuple[InvertedSliceDiscriminator, Dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    ckpt_args = payload.get("args") or {}
    candidate_only = bool(ckpt_args.get("candidate_only", False))
    use_offset_embedding = not bool(ckpt_args.get("no_offset_embedding", False))
    seq_len = int(slice_len if candidate_only else lookback + slice_len)
    model = InvertedSliceDiscriminator(
        seq_len=seq_len,
        max_offset=horizon - slice_len,
        d_model=int(ckpt_args.get("d_model", 128)),
        n_heads=int(ckpt_args.get("n_heads", 4)),
        depth=int(ckpt_args.get("depth", 2)),
        d_ff=int(ckpt_args.get("d_ff", 256)),
        dropout=float(ckpt_args.get("dropout", 0.1)),
        use_offset_embedding=use_offset_embedding,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, {
        "dataset": payload.get("dataset"),
        "fake_source": payload.get("fake_source"),
        "slice_len": slice_len,
        "candidate_only": candidate_only,
        "seq_len": seq_len,
    }


@torch.no_grad()
def collect_univariate_test_records(
    model: InvertedSliceDiscriminator,
    ds: UnivariateRealVsFakeDataset,
    device: torch.device,
    batch_size: int,
) -> List[Dict[str, Any]]:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    records: List[Dict[str, Any]] = []
    cursor = 0
    for batch in loader:
        x, offsets, labels = batch[0], batch[1], batch[2]
        logits = model(x.to(device), offsets.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (logits >= 0.0).cpu().numpy().astype(np.int64)
        labels_np = labels.cpu().numpy().astype(np.int64)
        for i in range(len(labels_np)):
            window, offset, variate, label = ds.items[cursor]
            past = ds.past[window, variate : variate + 1]
            gt = ds.real[window, variate : variate + 1, offset : offset + ds.slice_len]
            fake = ds.fake[window, variate : variate + 1, offset : offset + ds.slice_len]
            records.append(
                {
                    "window": int(window),
                    "offset": int(offset),
                    "variate": int(variate),
                    "label": int(label),
                    "pred": int(preds[i]),
                    "prob_fake": float(probs[i]),
                    "past": past.copy(),
                    "gt": gt.copy(),
                    "fake": fake.copy(),
                    "candidate": (fake if label == 1 else gt).copy(),
                }
            )
            cursor += 1
    return records


def maybe_log_confusion_paths(paths: Sequence[Path | str], *, wandb_key: str) -> None:
    """Log PNGs to wandb when a run is already active; no-op otherwise."""
    try:
        from models.diffusion_tsf.pipeline import wandb_utils
    except Exception:
        return
    wandb_utils.log_visualization_paths(
        [str(p) for p in paths],
        wandb_key=wandb_key,
    )


def visualize_univariate_combo(
    *,
    output_dir: Path,
    dataset: str,
    fake_source: str,
    slice_len: int,
    past: np.ndarray,
    y_true: np.ndarray,
    fake: np.ndarray,
    test_windows: np.ndarray,
    device: torch.device,
    seed: int = 2026,
    batch_size: int = 512,
    per_bucket: int = 2,
    lookback_tail: int = 32,
    plot_dir: Optional[Path] = None,
    max_eval_examples: Optional[int] = None,
    candidate_only: bool = True,
    offset_stride: int = 1,
    apply_zscore: bool = True,
    wandb_key: Optional[str] = None,
) -> Path:
    ckpt = univariate_checkpoint_path(output_dir, dataset, fake_source, slice_len)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing univariate discriminator checkpoint: {ckpt}")

    lookback = int(past.shape[-1])
    horizon = int(y_true.shape[-1])
    model, _meta = load_univariate_checkpoint(
        ckpt, device, lookback=lookback, horizon=horizon, slice_len=slice_len,
    )
    seed_base = int(seed) + stable_hash(f"{dataset}:{fake_source}:uni:{slice_len}")
    ds = UnivariateRealVsFakeDataset(
        y_true,
        fake,
        past,
        test_windows,
        slice_len,
        seed=seed_base + 2,
        offset_stride=offset_stride,
        max_examples=max_eval_examples,
        include_past=not candidate_only,
        apply_zscore=apply_zscore,
    )
    records = collect_univariate_test_records(model, ds, device, batch_size)
    counts = summarize_buckets(records)
    acc = sum(counts[k] for k in ("TP", "TN")) / max(1, len(records))
    print(
        f"[viz-uni] {dataset}/{fake_source}/L{slice_len} test n={len(records)} "
        f"acc={acc:.4f} buckets={counts}",
        flush=True,
    )

    out_root = Path(plot_dir) if plot_dir is not None else Path(output_dir) / "disc_confusions"
    stem = f"{dataset}_{fake_source}_L{slice_len}"
    out_dir = out_root / stem
    rng = np.random.default_rng(seed_base + 17)
    examples = pick_examples(records, per_bucket, rng)
    written: List[Path] = []
    for bucket, rows in examples.items():
        for i, row in enumerate(rows):
            title = (
                f"{bucket} | shown={'fake' if row['label'] == 1 else 'real'} "
                f"pred={'fake' if row['pred'] == 1 else 'real'} "
                f"p(fake)={row['prob_fake']:.3f} win={row['window']} "
                f"var={row['variate']} off={row['offset']}"
            )
            path = out_dir / f"{stem}_{bucket}_{i:02d}.png"
            plot_example(
                row,
                variate=0,
                lookback_tail=lookback_tail,
                title=title,
                out_path=path,
            )
            written.append(path)

    summary = {
        "checkpoint": str(ckpt),
        "dataset": dataset,
        "fake_source": fake_source,
        "slice_len": slice_len,
        "n_test": len(records),
        "acc": acc,
        "buckets": counts,
        "plots": str(out_dir),
        "forecast_agg": "probabilistic (caller-supplied fake tensor)",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[viz-uni] wrote plots under {out_dir}", flush=True)

    key = wandb_key or f"disc/confusions/{dataset}/{fake_source}/L{slice_len}"
    maybe_log_confusion_paths(written, wandb_key=key)
    return out_dir
