#!/usr/bin/env python3
"""Plot GT / binary / MMPD patches with disc TP/TN/FP/FN corners.

Needs pulled raw packs + disc checkpoints, e.g.:

  results/datasets/disc-lb336-hz720-ordinal-four-raw-trainval25/raw/{binary_staged,mmpd}_ETTh1.npz
  results/datasets/disc-lb336-hz720-ordinal-four-native-stride2/checkpoints/ETTh1_{binary_staged,mmpd}_L8_discriminator.pt

Example (Phase-2 native stride):

  source .venv/bin/activate
  python temp/scripts/plot_disc_mmpd_vs_binary_confusion_corners.py \\
    --dataset ETTh1 --slice-len 8 \\
    --disc-run results/datasets/disc-lb336-hz720-ordinal-four-native-stride2 \\
    --raw-run results/datasets/disc-lb336-hz720-ordinal-four-raw-trainval25 \\
    --native-repr-stride 2 --n-total 24 --per-corner 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.eval_discriminator_texture_staged_vs_mmpd import (  # noqa: E402
    HorizonSliceDataset,
    build_raw_bundle,
    parse_args as disc_parse_args,
    split_windows,
    stable_hash,
)
from utils.eval_mmpd_gaussian_anchor import run_test_stride  # noqa: E402
from utils.visualize_discriminator_texture_confusions import (  # noqa: E402
    bucket_name,
    collect_test_records,
    load_checkpoint,
    summarize_buckets,
)


SOURCES = ("binary_staged", "mmpd")
CORNERS = ("TP", "TN", "FP", "FN")


def _disc_defaults() -> argparse.Namespace:
    """Fair-campaign defaults matching the Killarney wrapper."""
    argv_backup = sys.argv
    # disc_parse_args uses nargs="+"/choices — pass sources as separate tokens, not CSV.
    sys.argv = [
        argv_backup[0],
        "--datasets",
        "ETTh1",
        "--fake-sources",
        "binary_staged",
        "mmpd",
        "--slice-lengths",
        "8",
        "--candidate-only",
        "--nonoverlapping-patches",
        "--no-offset-embedding",
        "--ordinal-ladder-quantize",
        "--pack-splits",
        "train,val",
        "--pack-fraction",
        "0.25",
        "--lookback",
        "336",
        "--horizon",
        "720",
        "--test-stride",
        "1",
        "--cpu",
    ]
    try:
        args = disc_parse_args()
    finally:
        sys.argv = argv_backup
    return args


def checkpoint_path(disc_run: Path, dataset: str, source: str, slice_len: int) -> Path:
    return disc_run / "checkpoints" / f"{dataset}_{source}_L{slice_len}_discriminator.pt"


def index_records(
    records: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[int, int, int], Mapping[str, Any]]:
    return {(int(r["window"]), int(r["offset"]), int(r["label"])): r for r in records}


def pick_corner_keys(
    records: Sequence[Mapping[str, Any]],
    *,
    per_corner: int,
    rng: np.random.Generator,
) -> List[Tuple[str, int, int, int]]:
    """Return (corner, window, offset, label) covering confusion corners."""
    by_corner: Dict[str, List[Mapping[str, Any]]] = {c: [] for c in CORNERS}
    for row in records:
        by_corner[bucket_name(int(row["label"]), int(row["pred"]))].append(row)

    picked: List[Tuple[str, int, int, int]] = []
    for corner in CORNERS:
        pool = list(by_corner[corner])
        if not pool:
            print(f"[warn] empty corner {corner}", flush=True)
            continue
        # prefer confident mistakes / correct calls, then random fill
        if corner in ("TP", "FP"):
            pool.sort(key=lambda r: float(r["prob_fake"]), reverse=True)
        else:
            pool.sort(key=lambda r: float(r["prob_fake"]))
        n = min(per_corner, len(pool))
        head_n = max(1, n // 2)
        chosen = pool[:head_n]
        rest = pool[head_n:]
        need = n - len(chosen)
        if need > 0 and rest:
            idx = rng.choice(len(rest), size=min(need, len(rest)), replace=False)
            chosen.extend(rest[i] for i in idx)
        for row in chosen[:n]:
            picked.append((corner, int(row["window"]), int(row["offset"]), int(row["label"])))
    return picked


def pick_random_keys(
    records: Sequence[Mapping[str, Any]],
    *,
    n: int,
    rng: np.random.Generator,
    exclude: set,
) -> List[Tuple[str, int, int, int]]:
    pool = [
        (bucket_name(int(r["label"]), int(r["pred"])), int(r["window"]), int(r["offset"]), int(r["label"]))
        for r in records
        if (int(r["window"]), int(r["offset"]), int(r["label"])) not in exclude
    ]
    if not pool or n <= 0:
        return []
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[i] for i in idx]


def fmt_call(row: Optional[Mapping[str, Any]]) -> str:
    if row is None:
        return "n/a"
    corner = bucket_name(int(row["label"]), int(row["pred"]))
    shown = "fake" if int(row["label"]) == 1 else "real"
    pred = "fake" if int(row["pred"]) == 1 else "real"
    return f"{corner} shown={shown} pred={pred} p(fake)={float(row['prob_fake']):.3f}"


def plot_comparison(
    *,
    past: np.ndarray,
    gt: np.ndarray,
    binary_fake: np.ndarray,
    mmpd_fake: np.ndarray,
    variate: int,
    lookback_tail: int,
    title: str,
    subtitle: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    v = int(variate)
    past_v = past[v, -lookback_tail:]
    gt_v = gt[v]
    bin_v = binary_fake[v]
    mmpd_v = mmpd_fake[v]
    t_past = np.arange(-len(past_v), 0)
    t_h = np.arange(0, len(gt_v))

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    ax = axes[0]
    ax.plot(t_past, past_v, color="#666666", lw=1.2, label="lookback")
    ax.plot(t_h, gt_v, color="#1f77b4", lw=2.0, label="GT")
    ax.plot(t_h, bin_v, color="#d62728", lw=1.8, alpha=0.9, label="binary")
    ax.plot(t_h, mmpd_v, color="#2ca02c", lw=1.8, alpha=0.9, label="mmpd")
    ax.axvline(0, color="black", ls="--", lw=0.8, alpha=0.45)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("value")
    ax.legend(loc="upper right", fontsize=8, ncol=4)

    axes[1].axis("off")
    axes[1].text(
        0.01,
        0.95,
        subtitle,
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def score_source(
    args: argparse.Namespace,
    *,
    dataset: str,
    source: str,
    slice_len: int,
    bundle: Any,
    splits: Mapping[str, np.ndarray],
    device: torch.device,
    disc_run: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    ckpt = checkpoint_path(disc_run, dataset, source, slice_len)
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint: {ckpt}\n"
            "Pull disc-run checkpoints/ from Killarney, or run this on $SCRATCH."
        )
    y_true = bundle.y_true_by_source[source]
    model, _ = load_checkpoint(
        ckpt,
        device,
        lookback=int(bundle.past.shape[-1]),
        horizon=int(y_true.shape[-1]),
        slice_len=slice_len,
    )
    seed_base = args.seed + stable_hash(f"{dataset}:{source}:{slice_len}")
    ds = HorizonSliceDataset(
        bundle.past,
        y_true,
        bundle.fakes[source],
        splits["test"],
        slice_len,
        seed=seed_base + 2,
        offset_stride=(
            int(slice_len)
            if bool(getattr(args, "nonoverlapping_patches", False))
            else args.offset_stride
        ),
        max_examples=args.max_eval_examples,
        include_past=not bool(getattr(args, "candidate_only", False)),
    )
    records = collect_test_records(model, ds, device, args.batch_size)
    return records, summarize_buckets(records)


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="ETTh1")
    p.add_argument("--slice-len", type=int, default=8)
    p.add_argument(
        "--disc-run",
        type=Path,
        default=REPO_ROOT / "results/datasets/disc-lb336-hz720-ordinal-four-native-stride2",
    )
    p.add_argument(
        "--raw-run",
        type=Path,
        default=REPO_ROOT / "results/datasets/disc-lb336-hz720-ordinal-four-raw-trainval25",
    )
    p.add_argument("--native-repr-stride", type=int, default=2)
    p.add_argument("--per-corner", type=int, default=2, help="Min samples per TP/TN/FP/FN per source.")
    p.add_argument("--n-total", type=int, default=24, help="Target plot count after corner picks + random.")
    p.add_argument("--variate", type=int, default=0)
    p.add_argument("--lookback-tail", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: temp/disc_mmpd_vs_binary_corners/<dataset>_L<L>/",
    )
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--anchor-config",
        default="binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native",
    )
    p.add_argument(
        "--binary-config",
        default="configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native.yaml",
    )
    p.add_argument("--mmpd-run", default="07-10-mmpd-decoder-paper-lb336-hz720-subset")
    return p.parse_args()


def main() -> None:
    cli = parse_cli()
    args = _disc_defaults()
    args.datasets = [cli.dataset]
    args.fake_sources = list(SOURCES)
    args.slice_lengths = [cli.slice_len]
    args.raw_eval_dir = cli.raw_run.resolve()
    args.output_dir = cli.disc_run.resolve()
    args.native_repr_stride = int(cli.native_repr_stride)
    args.seed = int(cli.seed)
    args.cpu = bool(cli.cpu) or not torch.cuda.is_available()
    args.gpu = int(cli.gpu)
    args.anchor_config = cli.anchor_config
    args.binary_config = cli.binary_config
    args.mmpd_run = cli.mmpd_run
    # Don't regenerate packs / retrain.
    args.force_raw_eval = False
    args.force_train = False

    out_dir = cli.out_dir
    if out_dir is None:
        out_dir = (
            REPO_ROOT
            / "temp"
            / "disc_mmpd_vs_binary_corners"
            / f"{cli.dataset}_L{cli.slice_len}_stride{cli.native_repr_stride}"
        )
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if args.cpu else f"cuda:{args.gpu}")
    print(f"[device] {device}", flush=True)
    print(f"[raw] {args.raw_eval_dir}", flush=True)
    print(f"[disc] {args.output_dir}", flush=True)

    for src in SOURCES:
        npz = args.raw_eval_dir / "raw" / f"{src}_{cli.dataset}.npz"
        if not npz.is_file():
            raise FileNotFoundError(
                f"Missing raw pack {npz}. Pull raw-run packs from Killarney first."
            )

    bundle = build_raw_bundle(args, cli.dataset, device)
    y_ref = next(iter(bundle.y_true_by_source.values()))
    splits = split_windows(
        int(y_ref.shape[0]),
        args,
        cli.dataset,
        indices=bundle.indices,
        lookback=int(bundle.past.shape[-1]),
        horizon=int(y_ref.shape[-1]),
        test_stride=run_test_stride(bundle.run),
        series_starts=bundle.series_starts,
    )
    print(
        f"[split] train/val/test="
        f"{len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])} "
        f"horizon={y_ref.shape[-1]} native_stride={args.native_repr_stride}",
        flush=True,
    )

    scored: Dict[str, List[Dict[str, Any]]] = {}
    indexes: Dict[str, Dict[Tuple[int, int, int], Mapping[str, Any]]] = {}
    bucket_summary: Dict[str, Dict[str, int]] = {}
    for src in SOURCES:
        recs, buckets = score_source(
            args,
            dataset=cli.dataset,
            source=src,
            slice_len=cli.slice_len,
            bundle=bundle,
            splits=splits,
            device=device,
            disc_run=args.output_dir,
        )
        scored[src] = recs
        indexes[src] = index_records(recs)
        bucket_summary[src] = buckets
        print(f"[score] {src} n={len(recs)} buckets={buckets}", flush=True)

    rng = np.random.default_rng(cli.seed + stable_hash(f"corners:{cli.dataset}:{cli.slice_len}"))
    selections: List[Tuple[str, str, int, int, int]] = []  # focus_src, corner, win, off, label
    seen_keys: set = set()
    for src in SOURCES:
        for corner, win, off, lab in pick_corner_keys(
            scored[src], per_corner=cli.per_corner, rng=rng
        ):
            key = (win, off, lab)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            selections.append((src, corner, win, off, lab))

    need = max(0, int(cli.n_total) - len(selections))
    # fill from binary pool primarily (same test offsets exist for both)
    for corner, win, off, lab in pick_random_keys(
        scored["binary_staged"], n=need, rng=rng, exclude=seen_keys
    ):
        seen_keys.add((win, off, lab))
        selections.append(("binary_staged", corner, win, off, lab))

    manifest_rows = []
    for i, (focus_src, corner, win, off, lab) in enumerate(selections):
        gt = bundle.y_true_by_source["binary_staged"][win, :, off : off + cli.slice_len]
        # MMPD GT should match binary GT under ordinal snap; prefer binary pack GT.
        bin_f = bundle.fakes["binary_staged"][win, :, off : off + cli.slice_len]
        mmpd_f = bundle.fakes["mmpd"][win, :, off : off + cli.slice_len]
        past = bundle.past[win]

        bin_row = indexes["binary_staged"].get((win, off, lab))
        mmpd_row = indexes["mmpd"].get((win, off, lab))
        # also report the paired shown=fake / shown=real calls at this offset
        bin_fake = indexes["binary_staged"].get((win, off, 1))
        bin_real = indexes["binary_staged"].get((win, off, 0))
        mmpd_fake = indexes["mmpd"].get((win, off, 1))
        mmpd_real = indexes["mmpd"].get((win, off, 0))

        title = (
            f"{cli.dataset} L{cli.slice_len} | focus={focus_src} {corner} | "
            f"win={win} off={off} shown={'fake' if lab else 'real'} v={cli.variate}"
        )
        focus_row = bin_row if focus_src == "binary_staged" else mmpd_row
        subtitle = "\n".join(
            [
                f"focus record: {focus_src} {fmt_call(focus_row)}",
                f"binary @fake: {fmt_call(bin_fake)}",
                f"binary @real: {fmt_call(bin_real)}",
                f"mmpd   @fake: {fmt_call(mmpd_fake)}",
                f"mmpd   @real: {fmt_call(mmpd_real)}",
            ]
        )
        out_path = out_dir / f"{i:02d}_{focus_src}_{corner}_win{win}_off{off}_lab{lab}.png"
        plot_comparison(
            past=past,
            gt=gt,
            binary_fake=bin_f,
            mmpd_fake=mmpd_f,
            variate=cli.variate,
            lookback_tail=cli.lookback_tail,
            title=title,
            subtitle=subtitle,
            out_path=out_path,
        )
        manifest_rows.append(
            {
                "file": out_path.name,
                "focus_source": focus_src,
                "corner": corner,
                "window": win,
                "offset": off,
                "label": lab,
                "binary_fake": fmt_call(bin_fake),
                "binary_real": fmt_call(bin_real),
                "mmpd_fake": fmt_call(mmpd_fake),
                "mmpd_real": fmt_call(mmpd_real),
            }
        )

    summary = {
        "dataset": cli.dataset,
        "slice_len": cli.slice_len,
        "native_repr_stride": cli.native_repr_stride,
        "disc_run": str(args.output_dir),
        "raw_run": str(args.raw_eval_dir),
        "n_plots": len(manifest_rows),
        "bucket_summary": bucket_summary,
        "corner_counts": {
            src: {
                c: sum(1 for r in manifest_rows if r["focus_source"] == src and r["corner"] == c)
                for c in CORNERS
            }
            for src in SOURCES
        },
        "plots": manifest_rows,
    }
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] wrote {len(manifest_rows)} plots -> {out_dir}", flush=True)
    print(json.dumps(summary["corner_counts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
