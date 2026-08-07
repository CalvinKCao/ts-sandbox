#!/usr/bin/env python3
"""Train lean disc arches on cached canvas128 packs (same snap as transformer runs).

Reuses binary/MMPD packs under an ablation dir, snaps with ``_snap_bundle``
(``window_norm_grid`` / hybrid_flat), then trains transformer / mlp / cnn1d /
flatness under unique_abs + bin-center + candidate_only. Writes overall +
per-variate accuracy/AUROC tables.

Viz defaults on: ``--viz-sanity all`` (snap+pre_post) and binary↔MMPD
disagreement panels. Disable with ``--viz-sanity none`` / ``--no-viz`` /
``--no-disc-disagreement``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from temp.scripts.eval_ablation_disc_l8_l16 import (  # noqa: E402
    _load_scores,
    _run_viz_sanity_hooks,
    _snap_bundle,
    apply_viz_master_switch,
    load_ablation_run,
)
from utils.disc_shared import DISC_ARCH_CHOICES, apply_disc_pack_protocol, write_json  # noqa: E402
from utils.disc_snap_viz import (  # noqa: E402
    DEFAULT_VIZ_SANITY,
    parse_viz_sanity,
    write_disc_disagreement_viz,
)
from utils.eval_discriminator_binary_vs_mmpd_univariate import train_classifier  # noqa: E402
from utils.eval_discriminator_texture_staged_vs_mmpd import split_windows  # noqa: E402
from utils.eval_mmpd_gaussian_anchor import DEFAULT_MMPD_DATA  # noqa: E402


DEFAULT_PACK = (
    REPO / "results/datasets/08-04-2009-ablation-disc-l8-l16-ETTh2-c128-valtest80-byvar"
)
DEFAULT_CKPT = (
    REPO
    / "results/ckpts/08-04-4601319-ETTh2-binary_window_norm_patch_refine_canvas128_p64x6_etth2"
)
DEFAULT_CONFIG = "configs/binary_window_norm_patch_refine_canvas128_p64x6_etth2.yaml"
LULL = 5
ALLOWED_SNAP_MODES = ("window_norm_grid", "window_norm_grid_hybrid_flat")


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _find_npz(raw: Path, prefix: str) -> Path:
    hits = sorted(p for p in raw.glob(f"{prefix}*.npz") if "indices" not in p.name)
    vt = [p for p in hits if "val-test" in p.name or "val_test" in p.name]
    if vt:
        return vt[0]
    if not hits:
        raise FileNotFoundError(f"no {prefix}*.npz under {raw}")
    return hits[0]


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _make_args(
    *,
    dataset: str,
    out_dir: Path,
    arch: str,
    disc_variates: Optional[Sequence[int]],
    lookback: int,
    horizon: int,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    smoke: bool,
) -> SimpleNamespace:
    ns = SimpleNamespace(
        dataset=str(dataset),
        output_dir=out_dir,
        lookback=lookback,
        horizon=horizon,
        pack_test_stride=4,
        pack_splits="val,test",
        train_fraction=0.8,
        val_fraction=0.0,
        fake_agg="sample0",
        slice_lengths=[8, 16],
        candidate_only=True,
        disc_bin_center_shift=True,
        disc_apply_zscore=False,
        disc_bin_center_reduce="per_variate",
        unique_absolute_slices=True,
        nonoverlapping_patches=False,
        no_offset_embedding=False,
        offset_stride=1,
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-3,
        d_model=128,
        n_heads=4,
        depth=2,
        d_ff=256,
        dropout=0.1 if arch == "transformer" else 0.0,
        weight_decay=0.0,
        grad_clip=1.0,
        patience=patience,
        max_batches_per_epoch=None,
        max_train_examples=None,
        max_eval_examples=None,
        max_windows=None,
        num_workers=0,
        seed=seed,
        disc_arch=arch,
        disc_variates=list(disc_variates) if disc_variates else None,
        disc_mlp_hidden=64,
        save_classification_scores=True,
        return_test_scores=True,
        mmpd_data_dir=str(DEFAULT_MMPD_DATA),
    )
    if smoke:
        ns.epochs = min(int(ns.epochs), 3)
        ns.patience = min(int(ns.patience), 2)
        ns.max_train_examples = 4000
        ns.max_eval_examples = 2000
    apply_disc_pack_protocol(ns)
    return ns


def main() -> None:
    """Walkthrough: load cached packs → snap → loop arches × sources × L → train.

    This is the lean multi-arch runner. It does NOT regenerate forecasts; it
    reuses an ablation pack and the same ``_snap_bundle`` path as the transformer
    ablation so the only variable is ``--disc-arch``.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ETTh2")
    ap.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--output-dir", type=Path, default=REPO / "temp/lean_disc_etth2")
    ap.add_argument("--arches", nargs="+", default=list(DISC_ARCH_CHOICES))
    ap.add_argument("--sources", nargs="+", default=["binary_staged", "mmpd"])
    ap.add_argument("--slice-lengths", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--lookback", type=int, default=336)
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument(
        "--viz-sanity",
        default=DEFAULT_VIZ_SANITY,
        help="After _snap_bundle: snap and/or pre_post (comma list or all/true). "
        "Default: all. Pass none/off to disable.",
    )
    ap.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable sanity + disagreement viz.",
    )
    ap.add_argument("--viz-dir", type=Path, default=None)
    ap.add_argument("--viz-n-windows", type=int, default=2)
    ap.add_argument("--viz-variates", type=int, nargs="+", default=None)
    ap.add_argument(
        "--disc-disagreement",
        action="store_true",
        default=True,
        help="After both sources train for an arch×L, write disagreement panels "
        "(default on).",
    )
    ap.add_argument(
        "--no-disc-disagreement",
        action="store_false",
        dest="disc_disagreement",
        help="Skip binary↔MMPD disagreement panels.",
    )
    ap.add_argument(
        "--disc-disagreement-max",
        type=int,
        default=12,
        help="Max panels per disagreement direction (default 12; smoke clamps to 2).",
    )
    ap.add_argument(
        "--also-lull-only",
        action="store_true",
        default=False,
        help="Also train each arch on LULL-only (v=5) for L=8 MMPD (ETTh2).",
    )
    ap.add_argument("--no-also-lull-only", action="store_false", dest="also_lull_only")
    args = ap.parse_args()
    # Lean has no redbox/encode-bins flags; master switch still clears disagreement.
    if not hasattr(args, "viz_encode_bins"):
        args.viz_encode_bins = False
    if not hasattr(args, "redbox_viz"):
        args.redbox_viz = False
    apply_viz_master_switch(args)
    if args.smoke_test:
        args.viz_n_windows = min(int(args.viz_n_windows), 1)
        args.disc_disagreement_max = min(int(args.disc_disagreement_max), 2)

    t0 = time.time()
    dataset = str(args.dataset)
    pack = args.pack.expanduser().resolve()
    ckpt = args.ckpt.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    # --- Load cached binary + MMPD forecast packs (npz under pack/raw/). ---
    binary_path = _find_npz(pack / "raw", "binary_")
    mmpd_path = _find_npz(pack / "raw", "mmpd_")
    binary_pack = _load_npz(binary_path)
    mmpd_pack = _load_npz(mmpd_path)
    if "series_starts" not in mmpd_pack:
        mmpd_pack = dict(mmpd_pack)
        mmpd_pack["series_starts"] = np.asarray(binary_pack["series_starts"], dtype=np.int64)
    canvas_height = int(np.asarray(binary_pack["canvas_height"]).reshape(-1)[0])
    print(
        f"[lean] dataset={dataset} pack={pack.name} binary={binary_path.name} "
        f"mmpd={mmpd_path.name} canvas={canvas_height} device={device} "
        f"viz_sanity={args.viz_sanity!r} disc_disagreement={args.disc_disagreement}",
        flush=True,
    )

    # --- Snap onto the training lattice (must be window_norm_grid for canvas128). ---
    run, _stages, kind = load_ablation_run(dataset, ckpt)
    snap_args = SimpleNamespace(
        dataset=dataset,
        fake_agg="sample0",
        lookback=args.lookback,
        horizon=args.horizon,
        mmpd_data_dir=str(DEFAULT_MMPD_DATA),
    )
    snapped = _snap_bundle(
        binary_pack=binary_pack,
        mmpd_pack=mmpd_pack,
        run=run,
        ladder=None,
        args=snap_args,
        device=device,
        canvas_height=canvas_height,
        ckpt_root=ckpt,
        config_path=args.config,
    )
    snap_mode = str(snapped["snap_mode"])
    if snap_mode not in ALLOWED_SNAP_MODES:
        raise RuntimeError(
            f"expected snap_mode in {ALLOWED_SNAP_MODES} for canvas128 leaf, "
            f"got {snap_mode!r}"
        )
    print(f"[lean] snap_mode={snap_mode} kind={kind}", flush=True)
    write_json(
        out_dir / "snap_meta.json",
        {
            "dataset": dataset,
            "snap_mode": snap_mode,
            "kind": kind,
            "canvas_height": canvas_height,
            "snap_meta": (snapped.get("lattice") or {}).get("snap_meta"),
            "binary_path": str(binary_path),
            "mmpd_path": str(mmpd_path),
            "ckpt": str(ckpt),
            "config": args.config,
        },
    )

    # Render-only snap sanity (default: all = snap+pre_post).
    if parse_viz_sanity(getattr(args, "viz_sanity", "") or ""):
        # Default viz panel: LULL (v=5) only on ETTh2; else v0.
        # Electricity/traffic canvas packs are 4-var subsets — LULL is OOB.
        n_snap_vars = int(np.asarray(snapped["gt"]).shape[1])
        if args.viz_variates:
            viz_vars = [int(v) for v in args.viz_variates]
        elif dataset == "ETTh2" and LULL < n_snap_vars:
            viz_vars = [LULL]
        else:
            viz_vars = [0]
        for v in viz_vars:
            if v < 0 or v >= n_snap_vars:
                raise ValueError(
                    f"viz variate={v} out of range for dataset={dataset} V={n_snap_vars}"
                )
        viz_ns = SimpleNamespace(
            viz_sanity=args.viz_sanity,
            viz_encode_bins=False,
            viz_dir=args.viz_dir or (out_dir / "viz"),
            viz_n_windows=min(int(args.viz_n_windows), 1) if args.smoke_test else int(args.viz_n_windows),
            viz_variate=viz_vars[0],
            viz_variates=viz_vars,
            viz_zoom_steps=12,
            slice_lengths=list(args.slice_lengths),
            seed=int(args.seed),
            lookback=int(args.lookback),
            horizon=int(args.horizon),
            output_dir=out_dir,
        )
        sanity = _run_viz_sanity_hooks(
            args=viz_ns,
            run_name="lean",
            dataset=dataset,
            snapped=snapped,
            ckpt_root=ckpt,
            config_path=args.config,
            device=device,
        )
        write_json(out_dir / "viz_sanity.json", sanity)

    # Bundle shape expected by train_classifier (fakes + GT + legal_levels).
    bundle = SimpleNamespace(
        fakes={"binary_staged": snapped["binary"], "mmpd": snapped["mmpd"]},
        y_true_by_source={
            "binary_staged": snapped["gt"],
            "mmpd": snapped["gt"].copy(),
        },
        past=snapped["past"],
        legal_levels=snapped["legal_levels"],
        indices=snapped["indices"],
        series_starts=snapped["series_starts"],
        run=run,
    )

    # Chronological 80/20 split shared across all arch jobs (fair compare).
    base_args = _make_args(
        dataset=dataset,
        out_dir=out_dir,
        arch="transformer",
        disc_variates=None,
        lookback=args.lookback,
        horizon=args.horizon,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        smoke=bool(args.smoke_test),
    )
    splits = split_windows(
        len(snapped["gt"]),
        base_args,
        dataset,
        indices=bundle.indices,
        lookback=args.lookback,
        horizon=args.horizon,
        test_stride=4,
        series_starts=bundle.series_starts,
    )
    write_json(
        out_dir / "splits.json",
        {k: np.asarray(v).tolist() for k, v in splits.items()},
    )

    # Job grid: every arch × fake source × L, optionally + LULL-only extras.
    jobs: List[Dict[str, Any]] = []
    for arch in args.arches:
        for source in args.sources:
            for L in args.slice_lengths:
                jobs.append(
                    {
                        "arch": arch,
                        "source": source,
                        "slice_len": int(L),
                        "disc_variates": None,
                        "tag": "all_variates",
                    }
                )
    if args.also_lull_only:
        if dataset != "ETTh2":
            print(
                f"[lean] warn: --also-lull-only ignored for dataset={dataset} "
                f"(LULL=v5 is ETTh2-specific)",
                flush=True,
            )
        else:
            for arch in args.arches:
                jobs.append(
                    {
                        "arch": arch,
                        "source": "mmpd",
                        "slice_len": 8,
                        "disc_variates": [LULL],
                        "tag": "lull_only",
                    }
                )

    summary: Dict[str, Any] = {}
    table_rows: List[Dict[str, Any]] = []
    by_var_rows: List[Dict[str, Any]] = []
    # arch → L → source → test score arrays (for disagreement viz).
    scores_cache: Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]] = {}

    for i, job in enumerate(jobs, 1):
        arch = str(job["arch"])
        source = str(job["source"])
        L = int(job["slice_len"])
        tag = str(job["tag"])
        disc_vars = job["disc_variates"]
        key = f"{arch}__{source}__L{L}__{tag}"
        print(f"\n=== [{i}/{len(jobs)}] {dataset} {key} ===", flush=True)
        # Per-job args: same protocol flags, different --disc-arch / variate filter.
        train_args = _make_args(
            dataset=dataset,
            out_dir=out_dir / key,
            arch=arch,
            disc_variates=disc_vars,
            lookback=args.lookback,
            horizon=args.horizon,
            seed=args.seed,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            smoke=bool(args.smoke_test),
        )
        train_args.slice_lengths = [L]
        # train_classifier builds dataset + model + train loop + test metrics.
        # Signature: (args, dataset, fake_source, slice_len, bundle, splits, device).
        mets_raw = train_classifier(
            train_args, dataset, source, L, bundle, splits, device,
        )
        scores = mets_raw.pop("_test_scores", None)
        if scores is None and mets_raw.get("score_path"):
            scores = _load_scores(Path(str(mets_raw["score_path"])))
        if scores is not None and tag == "all_variates":
            scores_cache.setdefault(arch, {}).setdefault(str(L), {})[source] = scores
        mets = _jsonable(mets_raw)
        summary[key] = mets
        table_rows.append(
            {
                "dataset": dataset,
                "arch": arch,
                "source": source,
                "slice_len": L,
                "tag": tag,
                "disc_auroc": mets.get("disc_auroc"),
                "disc_acc": mets.get("disc_acc"),
                "disc_bce": mets.get("disc_bce"),
                "prob_std": mets.get("prob_std"),
                "disc_collapsed": mets.get("disc_collapsed"),
                "n_params": mets.get("n_params"),
            }
        )
        # Flatten per-variate AUROC/acc for the side table (LULL = v5 on ETTh2).
        by_var = mets.get("by_variate") or {}
        for vk, vm in by_var.items():
            by_var_rows.append(
                {
                    "dataset": dataset,
                    "arch": arch,
                    "source": source,
                    "slice_len": L,
                    "tag": tag,
                    "variate": int(vk),
                    "is_lull": int(dataset == "ETTh2" and int(vk) == LULL),
                    "disc_acc": float(vm["disc_acc"]),
                    "disc_auroc": float(vm["disc_auroc"]),
                    "n_examples": float(vm["n_examples"]),
                }
            )
        write_json(out_dir / "auroc_table.json", table_rows)
        write_json(out_dir / "auroc_by_variate.json", by_var_rows)
        write_json(out_dir / "summary.json", summary)

    # Binary↔MMPD disagreement panels per arch×L (same keys as ablation).
    disagree_manifests: Dict[str, Any] = {}
    if bool(args.disc_disagreement):
        disagree_root = out_dir / "viz" / "disc_disagreement"
        for arch, by_L in scores_cache.items():
            for L_key, by_src in by_L.items():
                if "binary_staged" not in by_src or "mmpd" not in by_src:
                    continue
                run_name = f"lean_{arch}"
                sub = disagree_root / run_name
                disagree_manifests[f"{arch}__L{L_key}"] = write_disc_disagreement_viz(
                    out_dir=sub,
                    run_name=run_name,
                    dataset=dataset,
                    slice_len=int(L_key),
                    snapped=snapped,
                    binary_scores=by_src["binary_staged"],
                    mmpd_scores=by_src["mmpd"],
                    include_past=False,  # lean is candidate_only
                    max_panels=int(args.disc_disagreement_max),
                    seed=int(args.seed),
                )
        write_json(disagree_root / "summary.json", disagree_manifests)
    else:
        print("[lean] skipped disc disagreement viz", flush=True)

    # Console summary so you can eyeball collapse vs real signal without opening JSON.
    print("\n========== OVERALL (disc_acc / disc_auroc / collapsed) ==========", flush=True)
    for row in table_rows:
        print(
            f"  {row['arch']:12s} {row['source']:14s} L{row['slice_len']:<3} "
            f"{row['tag']:12s}  acc={float(row.get('disc_acc') or float('nan')):.4f}  "
            f"auc={float(row.get('disc_auroc') or float('nan')):.4f}  "
            f"collapsed={int(row.get('disc_collapsed') or 0)}  "
            f"n_params={int(row.get('n_params') or 0)}",
            flush=True,
        )

    print("\n========== PER-VARIATE ACC (all_variates, L=8) ==========", flush=True)
    for arch in args.arches:
        for source in args.sources:
            subset = [
                r
                for r in by_var_rows
                if r["arch"] == arch
                and r["source"] == source
                and r["slice_len"] == 8
                and r["tag"] == "all_variates"
            ]
            if not subset:
                continue
            parts = [
                f"v{r['variate']}={float(r['disc_acc']):.3f}"
                + ("*" if r.get("is_lull") else "")
                for r in sorted(subset, key=lambda x: int(x["variate"]))
            ]
            print(f"  {arch:12s} {source:14s}  " + "  ".join(parts), flush=True)

    print(f"\n[lean] done in {time.time() - t0:.1f}s → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
