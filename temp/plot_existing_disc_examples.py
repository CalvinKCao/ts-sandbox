#!/usr/bin/env python3
"""Plot a few forecast/GT examples, grouped by discriminator TP/TN/FP/FN.

This is deliberately a *raw-pack* utility.  It does not call a binary or MMPD
forecaster: provide the `.npz` produced by a completed discriminator job.  If
the original L8 discriminator checkpoint is available, it is loaded verbatim.
For h96 patch-refine jobs that did not persist one, `--quick-train` fits a
small diagnostic discriminator on at most ``--max-windows`` materialized
forecasts.

Examples:
  # Existing h720 MMPD raw pack + its saved L8 discriminator.
  python temp/plot_existing_disc_examples.py \
    --raw-pack results/datasets/07-28-2021-mmpd-h720-existing-disc-raw/raw/mmpd_ETTh1.npz \
    --past-pack results/datasets/07-28-2021-mmpd-h720-existing-disc-raw/raw/binary_staged_ETTh1.npz \
    --checkpoint results/datasets/07-28-2021-mmpd-h720-existing-disc/checkpoints/ETTh1_mmpd_L8_univariate_discriminator.pt \
    --dataset ETTh1 --source mmpd --output-dir temp/disc_examples/h720_mmpd_ETTh1

  # Existing h96 patch-refine pack (no saved disc checkpoint): quick local
  # diagnostic classifier only; it never reruns coarse/refine sampling.
  python temp/plot_existing_disc_examples.py \
    --raw-pack results/datasets/07-28-2021-patch-refine-h96-existing-disc/ETTh1/raw/binary_patch_refine_ETTh1.npz \
    --dataset ETTh1 --source binary_patch_refine --quick-train \
    --output-dir temp/disc_examples/h96_patch_refine_ETTh1
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.eval_discriminator_binary_vs_mmpd_univariate import (  # noqa: E402
    UnivariateRealVsFakeDataset,
)
from utils.eval_discriminator_texture_staged_vs_mmpd import (  # noqa: E402
    InvertedSliceDiscriminator,
    split_windows,
)
from utils.eval_mmpd_gaussian_anchor import load_tsf_pack_pool  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-pack", type=Path, required=True)
    parser.add_argument("--past-pack", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slice-len", type=int, default=8)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-windows", type=int, default=16)
    parser.add_argument("--per-corner", type=int, default=1)
    parser.add_argument("--lookback-tail", type=int, default=96)
    parser.add_argument("--test-stride", type=int, default=4)
    parser.add_argument("--pack-splits", default=None, help="Canonical pool splits when raw-pack metadata is absent.")
    parser.add_argument("--representation", choices=("auto", "overlap_blended", "unblended_nonoverlap"), default="auto")
    parser.add_argument("--window-split", choices=("auto", "all"), default="auto")
    parser.add_argument("--quick-train", action="store_true")
    parser.add_argument("--quick-epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.max_windows < 6:
        parser.error("--max-windows must be >=6 so quick diagnostic splits remain non-empty")
    if args.per_corner < 1:
        parser.error("--per-corner must be positive")
    if args.checkpoint is None and not args.quick_train:
        parser.error("supply --checkpoint, or explicitly opt into --quick-train")
    return args


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _infer_past_pack(raw_pack: Path) -> Path | None:
    name = raw_pack.name
    if name.startswith("mmpd_"):
        candidate = raw_pack.with_name("binary_staged_" + name[len("mmpd_") :])
        if candidate.is_file():
            return candidate
    return None


def reconstruct_past_from_dataset(
    *,
    dataset: str,
    indices: np.ndarray,
    y_true: np.ndarray,
    lookback: int,
    horizon: int,
    test_stride: int,
    pack_splits: Sequence[str],
) -> np.ndarray:
    """Recover omitted history from the same indexed canonical test pool.

    Generic h720 raw packs intentionally omit past.  The returned pool target
    can contain a short context prefix, so only its final ``horizon`` values
    are used for the provenance equality check.
    """
    n_variates = int(y_true.shape[1])
    pool, _starts, _pool_splits, _lengths, _stats = load_tsf_pack_pool(
        dataset,
        list(range(n_variates)),
        lookback=lookback,
        horizon=horizon,
        train_stride=1,
        test_stride=test_stride,
        pack_splits=tuple(pack_splits),
    )
    if indices.size == 0 or int(indices.min()) < 0 or int(indices.max()) >= len(pool):
        raise IndexError(f"raw-pack indices [{indices.min()},{indices.max()}] outside reconstructed pool of {len(pool)}")
    past_rows: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    for index in indices.tolist():
        past, future = pool[int(index)]
        past_rows.append(past.detach().cpu().numpy())
        targets.append(future.detach().cpu().numpy()[..., -horizon:])
    past_np = np.stack(past_rows).astype(np.float32)
    target_np = np.stack(targets).astype(np.float32)
    max_abs = float(np.max(np.abs(target_np - y_true)))
    if max_abs > 2e-5:
        raise ValueError(
            "reconstructed dataset target does not match raw-pack y_true "
            f"(max abs {max_abs:.3g}); use the exact --test-stride/--pack-splits of the completed job"
        )
    return past_np


def _pack_splits(pack: Mapping[str, np.ndarray], requested: str | None) -> Tuple[str, ...]:
    if "pack_splits" in pack:
        values = tuple(str(value) for value in np.asarray(pack["pack_splits"]).tolist())
        if values:
            return values
    if requested:
        values = tuple(piece.strip() for piece in requested.split(",") if piece.strip())
        if values:
            return values
    raise ValueError("raw pack has no pack_splits metadata; provide --pack-splits exactly as the completed job used")


def load_forecast_pack(args: argparse.Namespace) -> Dict[str, Any]:
    pack = _load_npz(args.raw_pack)
    missing = {"y_true", "samples", "indices"} - set(pack)
    if missing:
        raise KeyError(f"{args.raw_pack} missing {sorted(missing)}")
    full_y_true = np.asarray(pack["y_true"], dtype=np.float32)
    samples = np.asarray(pack["samples"], dtype=np.float32)
    if samples.ndim != 4:
        raise ValueError(f"samples must be (N,V,S,H), got {samples.shape}")
    if not 0 <= args.sample_index < samples.shape[2]:
        raise ValueError(f"sample index {args.sample_index} outside [0,{samples.shape[2]})")
    full_fake = samples[:, :, args.sample_index, :]
    if full_fake.shape != full_y_true.shape:
        raise ValueError(f"fake/y_true mismatch: {full_fake.shape} vs {full_y_true.shape}")
    full_indices = np.asarray(pack["indices"], dtype=np.int64)
    pack_splits = _pack_splits(pack, args.pack_splits)
    past = pack.get("past")
    if past is None:
        past_path = args.past_pack or _infer_past_pack(args.raw_pack)
        if past_path is not None:
            past_pack = _load_npz(past_path)
            if "past" in past_pack and "indices" in past_pack:
                if not np.array_equal(pack["indices"], past_pack["indices"]):
                    raise ValueError("raw-pack and past-pack indices differ; refusing to misalign forecasts")
                past = past_pack["past"]
        if past is None:
            past = reconstruct_past_from_dataset(
                dataset=args.dataset,
                indices=np.asarray(pack["indices"], dtype=np.int64),
                y_true=full_y_true,
                lookback=336,
                horizon=int(full_y_true.shape[-1]),
                test_stride=args.test_stride,
                pack_splits=pack_splits,
            )
    past = np.asarray(past, dtype=np.float32)
    if past.shape[:2] != full_y_true.shape[:2]:
        raise ValueError(f"past/y_true mismatch: {past.shape} vs {full_y_true.shape}")

    fields = {
        "unblended_nonoverlap_patch_pred", "unblended_nonoverlap_patch_gt",
        "unblended_nonoverlap_patch_past", "unblended_nonoverlap_patch_parent",
        "unblended_nonoverlap_patch_start", "unblended_nonoverlap_patch_variate",
    }
    use_unblended = args.representation == "unblended_nonoverlap" or (
        args.representation == "auto" and fields <= set(pack)
    )
    if args.representation == "unblended_nonoverlap" and not fields <= set(pack):
        raise KeyError(f"{args.raw_pack} lacks raw coherent patch fields: {sorted(fields - set(pack))}")
    if use_unblended:
        parent = np.asarray(pack["unblended_nonoverlap_patch_parent"], dtype=np.int64)
        if parent.size == 0 or parent.min() < 0 or parent.max() >= len(full_indices):
            raise ValueError("unblended patch parent rows are invalid")
        return {
            "y_true": np.asarray(pack["unblended_nonoverlap_patch_gt"], dtype=np.float32),
            "fake": np.asarray(pack["unblended_nonoverlap_patch_pred"], dtype=np.float32),
            "past": np.asarray(pack["unblended_nonoverlap_patch_past"], dtype=np.float32),
            "indices": full_indices[parent], "full_indices": full_indices,
            "series_starts": np.asarray(pack.get("series_starts", []), dtype=np.int64),
            "parent_rows": parent,
            "patch_starts": np.asarray(pack["unblended_nonoverlap_patch_start"], dtype=np.int64),
            "source_variates": np.asarray(pack["unblended_nonoverlap_patch_variate"], dtype=np.int64),
            "pack_splits": pack_splits, "representation": "unblended_nonoverlap",
        }
    return {
        "y_true": full_y_true, "fake": full_fake, "past": past,
        "indices": full_indices, "full_indices": full_indices,
        "series_starts": np.asarray(pack.get("series_starts", []), dtype=np.int64),
        "parent_rows": None, "patch_starts": None, "source_variates": None,
        "pack_splits": pack_splits, "representation": "overlap_blended",
    }


def select_windows(n_windows: int, limit: int) -> np.ndarray:
    count = min(n_windows, limit)
    selected = np.linspace(0, n_windows - 1, num=count, dtype=np.int64)
    return np.unique(selected)


def select_test_rows(
    view: Mapping[str, Any], *, saved: Mapping[str, Any], args: argparse.Namespace, lookback: int, horizon: int,
) -> Tuple[np.ndarray, str]:
    if args.window_split == "all":
        return np.arange(len(view["y_true"]), dtype=np.int64), "all_pool_diagnostic"
    starts = np.asarray(view["series_starts"], dtype=np.int64)
    full_indices = np.asarray(view["full_indices"], dtype=np.int64)
    if starts.shape != full_indices.shape:
        raise ValueError(
            "cannot reconstruct the original holdout without series_starts; use --window-split all "
            "to request explicitly all-pool diagnostic plots"
        )
    split_args = SimpleNamespace(
        max_windows=None,
        train_fraction=float(saved.get("train_fraction", 0.7)),
        val_fraction=float(saved.get("val_fraction", 0.15)),
    )
    splits = split_windows(
        len(full_indices), split_args, args.dataset,
        indices=full_indices, lookback=lookback, horizon=horizon,
        test_stride=int(saved.get("test_stride", args.test_stride)), series_starts=starts,
    )
    test_parent = splits["test"]
    if view["parent_rows"] is None:
        return test_parent, "temporally_purged_test"
    return np.flatnonzero(np.isin(np.asarray(view["parent_rows"]), test_parent)).astype(np.int64), "temporally_purged_test_raw_unblended_patch"


def _saved_args(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = payload.get("args", {})
    if isinstance(raw, Mapping):
        return raw
    return vars(raw)


def build_saved_model(path: Path, *, horizon: int, lookback: int, slice_len: int, device: torch.device) -> Tuple[InvertedSliceDiscriminator, Dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in payload:
        raise KeyError(f"{path} is not a saved univariate discriminator")
    saved = dict(_saved_args(payload))
    if int(payload.get("slice_len", slice_len)) != slice_len:
        raise ValueError(f"{path} was saved for L{payload.get('slice_len')}, not L{slice_len}")
    include_past = not bool(saved.get("candidate_only", False))
    model = InvertedSliceDiscriminator(
        seq_len=(lookback + slice_len) if include_past else slice_len,
        max_offset=horizon - slice_len,
        d_model=int(saved.get("d_model", 128)),
        n_heads=int(saved.get("n_heads", 4)),
        depth=int(saved.get("depth", 2)),
        d_ff=int(saved.get("d_ff", 256)),
        dropout=float(saved.get("dropout", 0.1)),
        use_offset_embedding=not bool(saved.get("no_offset_embedding", False)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, saved


def build_quick_model(*, horizon: int, lookback: int, slice_len: int, include_past: bool, device: torch.device) -> Tuple[InvertedSliceDiscriminator, Dict[str, Any]]:
    saved = {
        "candidate_only": not include_past,
        "nonoverlapping_patches": False,
        "d_model": 64,
        "n_heads": 4,
        "depth": 1,
        "d_ff": 128,
        "dropout": 0.1,
        "no_offset_embedding": False,
    }
    model = InvertedSliceDiscriminator(
        seq_len=(lookback + slice_len) if include_past else slice_len,
        max_offset=horizon - slice_len,
        d_model=64,
        n_heads=4,
        depth=1,
        d_ff=128,
        dropout=0.1,
        use_offset_embedding=True,
    ).to(device)
    return model, saved


def loader_for(
    y_true: np.ndarray,
    fake: np.ndarray,
    past: np.ndarray,
    windows: Sequence[int],
    *,
    slice_len: int,
    saved: Mapping[str, Any],
    seed: int,
    batch_size: int = 512,
) -> DataLoader:
    offset_stride = slice_len if bool(saved.get("nonoverlapping_patches", False)) else int(saved.get("offset_stride", 1))
    ds = UnivariateRealVsFakeDataset(
        y_true,
        fake,
        past,
        np.asarray(windows, dtype=np.int64),
        slice_len,
        seed=seed,
        offset_stride=offset_stride,
        include_past=not bool(saved.get("candidate_only", False)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def quick_train(
    model: InvertedSliceDiscriminator,
    loader: DataLoader,
    *,
    epochs: int,
    device: torch.device,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for _epoch in range(epochs):
        for x, offsets, labels, _windows, _variates in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(device), offsets.to(device))
            loss = F.binary_cross_entropy_with_logits(logits, labels.to(device))
            loss.backward()
            optimizer.step()
    model.eval()


@torch.no_grad()
def score(model: InvertedSliceDiscriminator, loader: DataLoader, device: torch.device) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    model.eval()
    for x, offsets, labels, windows, variates in loader:
        probs = torch.sigmoid(model(x.to(device), offsets.to(device))).cpu().numpy()
        for probability, label, window, variate, offset in zip(
            probs, labels.numpy(), windows.numpy(), variates.numpy(), offsets.numpy()
        ):
            rows.append(
                {
                    "prob_fake": float(probability),
                    "label": int(label),
                    "window": int(window),
                    "variate": int(variate),
                    "offset": int(offset),
                }
            )
    return rows


def aggregate_windows(rows: Iterable[Mapping[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[Tuple[int, int, int], List[float]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["window"]), int(row["variate"]), int(row["label"]))].append(float(row["prob_fake"]))
    out: List[Dict[str, float]] = []
    for (window, variate, label), probs in grouped.items():
        prob = float(np.mean(probs))
        pred = int(prob >= 0.5)
        corner = ("T" if pred == label else "F") + ("P" if pred else "N")
        out.append({"window": window, "variate": variate, "label": label, "prob_fake": prob, "pred": pred, "corner": corner})
    return out


def choose_examples(rows: Sequence[Mapping[str, float]], per_corner: int) -> List[Mapping[str, float]]:
    chosen: List[Mapping[str, float]] = []
    for corner in ("TP", "TN", "FP", "FN"):
        pool = [row for row in rows if row["corner"] == corner]
        # Most confident examples make the classifier call legible.
        pool.sort(key=lambda row: abs(float(row["prob_fake"]) - 0.5), reverse=True)
        chosen.extend(pool[:per_corner])
    return chosen


def plot_example(
    *,
    out_path: Path,
    dataset: str,
    source: str,
    y_true: np.ndarray,
    fake: np.ndarray,
    past: np.ndarray,
    row: Mapping[str, float],
    raw_scores: Sequence[Mapping[str, float]],
    tail: int,
) -> None:
    window, variate, label = int(row["window"]), int(row["variate"]), int(row["label"])
    future = y_true[window, variate]
    prediction = fake[window, variate]
    history = past[window, variate, -min(tail, past.shape[-1]) :]
    score_rows = [
        score for score in raw_scores
        if int(score["window"]) == window and int(score["variate"]) == variate and int(score["label"]) == label
    ]
    score_rows.sort(key=lambda score: int(score["offset"]))

    fig, (ax, score_ax) = plt.subplots(2, 1, figsize=(12, 6), sharex=False, gridspec_kw={"height_ratios": [3.2, 1]})
    x_hist = np.arange(-len(history), 0)
    x_future = np.arange(len(future))
    ax.plot(x_hist, history, color="0.4", lw=1.1, label="lookback")
    if label:
        ax.plot(x_future, future, color="0.35", lw=1.0, ls="--", label="GT reference")
        ax.plot(x_future, prediction, color="tab:orange", lw=1.8, label=f"classified {source} forecast/fake")
    else:
        ax.plot(x_future, prediction, color="0.55", lw=1.0, ls="--", label=f"{source} forecast reference")
        ax.plot(x_future, future, color="black", lw=1.8, label="classified GT/real")
    ax.axvline(0, color="0.2", lw=0.8, ls="--")
    shown = "forecast/fake" if label else "GT/real"
    predicted = "fake" if int(row["pred"]) else "real"
    ax.set_title(
        f"{dataset} | {row['corner']} | shown={shown}, discriminator={predicted}, "
        f"mean P(fake)={float(row['prob_fake']):.3f} | window={window}, variate={variate}"
    )
    ax.set_ylabel("forecast coordinate")
    ax.legend(loc="best")

    if score_rows:
        score_ax.plot(
            [score["offset"] for score in score_rows],
            [score["prob_fake"] for score in score_rows],
            marker="o",
            markersize=2.5,
            lw=1.0,
            color="tab:purple",
        )
    score_ax.axhline(0.5, color="0.25", lw=0.8, ls="--")
    score_ax.set(xlabel="forecast timestep", ylabel="P(fake)", ylim=(-0.03, 1.03))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    view = load_forecast_pack(args)
    y_true = view["y_true"]
    fake = view["fake"]
    past = view["past"]
    indices = view["indices"]
    horizon, lookback = int(y_true.shape[-1]), int(past.shape[-1])

    if args.checkpoint is not None:
        model, saved = build_saved_model(
            args.checkpoint, horizon=horizon, lookback=lookback, slice_len=args.slice_len, device=device
        )
        mode = "saved_checkpoint"
    else:
        model, saved = build_quick_model(
            horizon=horizon, lookback=lookback, slice_len=args.slice_len,
            include_past=True, device=device,
        )
        mode = "quick_train_heldout_diagnostic_only"

    eligible, split_name = select_test_rows(view, saved=saved, args=args, lookback=lookback, horizon=horizon)
    if not len(eligible):
        raise ValueError("no eligible rows after representation/split filtering")
    selected = eligible[select_windows(len(eligible), args.max_windows)]
    if args.checkpoint is None:
        cut_train = max(3, int(0.6 * len(selected)))
        if len(selected) - cut_train < 2:
            raise ValueError("quick diagnostic needs at least two held-out selected rows")
        train_loader = loader_for(y_true, fake, past, selected[:cut_train], slice_len=args.slice_len, saved=saved, seed=args.seed)
        quick_train(model, train_loader, epochs=args.quick_epochs, device=device)
        selected = selected[cut_train:]
        split_name = f"{split_name}_quick_train_holdout"

    all_loader = loader_for(y_true, fake, past, selected, slice_len=args.slice_len, saved=saved, seed=args.seed + 1)
    raw_scores = score(model, all_loader, device)
    summary_rows = aggregate_windows(raw_scores)
    examples = choose_examples(summary_rows, args.per_corner)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ordinal, row in enumerate(examples):
        name = f"{ordinal:02d}_{row['corner']}_w{int(row['window'])}_v{int(row['variate'])}.png"
        plot_example(
            out_path=args.output_dir / name,
            dataset=args.dataset,
            source=args.source,
            y_true=y_true,
            fake=fake,
            past=past,
            row=row,
            raw_scores=raw_scores,
            tail=args.lookback_tail,
        )
    manifest = {
        "dataset": args.dataset,
        "source": args.source,
        "mode": mode,
        "raw_pack": str(args.raw_pack),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "representation": view["representation"],
        "score_split": split_name,
        "selected_windows": selected.tolist(),
        "selected_series_indices": indices[selected].tolist(),
        "slice_len": args.slice_len,
        "n_patch_scores": len(raw_scores),
        "window_classification_counts": {
            corner: sum(1 for row in summary_rows if row["corner"] == corner)
            for corner in ("TP", "TN", "FP", "FN")
        },
        "examples": examples,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
