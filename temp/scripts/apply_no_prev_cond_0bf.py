#!/usr/bin/env python3
"""Apply no-prev-cond patches onto an ef30f27e / pre-xattn tree. Idempotent."""
from __future__ import annotations

from pathlib import Path
import sys


def _sub(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new.strip() and new in text and old not in text:
        print(f"already applied: {path}")
        return
    if old not in text:
        raise SystemExit(f"missing snippet in {path}:\n{old[:180]!r}")
    path.write_text(text.replace(old, new, 1))
    print(f"patched {path}")


def main() -> None:
    root = Path(sys.argv[1]).resolve()

    cfg = root / "models/diffusion_tsf/config.py"
    _sub(
        cfg,
        """    patch_refine_unique_segments: bool = False
    patch_refine_prev_cond_dropout: float = 0.5
""",
        """    patch_refine_unique_segments: bool = False
    patch_refine_use_prev_cond: bool = True
    patch_refine_prev_cond_dropout: float = 0.5
""",
    )

    pconf = root / "models/diffusion_tsf/pipeline/config.py"
    _sub(
        pconf,
        """    "patch_refine_unique_segments",
    "patch_refine_prev_cond_dropout",
""",
        """    "patch_refine_unique_segments",
    "patch_refine_use_prev_cond",
    "patch_refine_prev_cond_dropout",
""",
    )

    state = root / "models/diffusion_tsf/pipeline/state.py"
    _sub(
        state,
        """    patch_refine_unique_segments: bool = False
    patch_refine_prev_cond_dropout: float = 0.5
""",
        """    patch_refine_unique_segments: bool = False
    patch_refine_use_prev_cond: bool = True
    patch_refine_prev_cond_dropout: float = 0.5
""",
    )
    _sub(
        state,
        """        if "patch_refine_unique_segments" in init_kwargs:
            init_kwargs["patch_refine_unique_segments"] = bool(
                init_kwargs["patch_refine_unique_segments"]
            )
        if "patch_refine_prev_cond_dropout" in init_kwargs:
            init_kwargs["patch_refine_prev_cond_dropout"] = float(
                init_kwargs["patch_refine_prev_cond_dropout"]
            )
""",
        """        if "patch_refine_unique_segments" in init_kwargs:
            init_kwargs["patch_refine_unique_segments"] = bool(
                init_kwargs["patch_refine_unique_segments"]
            )
        if "patch_refine_use_prev_cond" in init_kwargs:
            init_kwargs["patch_refine_use_prev_cond"] = bool(
                init_kwargs["patch_refine_use_prev_cond"]
            )
        if "patch_refine_prev_cond_dropout" in init_kwargs:
            init_kwargs["patch_refine_prev_cond_dropout"] = float(
                init_kwargs["patch_refine_prev_cond_dropout"]
            )
        if (
            not bool(init_kwargs.get("patch_refine_use_prev_cond", True))
            and float(init_kwargs.get("patch_refine_prev_cond_dropout", 0.0)) != 0.0
        ):
            raise ValueError(
                "patch_refine_prev_cond_dropout must be 0 when "
                "patch_refine_use_prev_cond is false"
            )
""",
    )

    tm = root / "models/diffusion_tsf/train_multivariate_pipeline.py"
    _sub(
        tm,
        """        patch_refine_unique_segments=state.patch_refine_unique_segments,
        patch_refine_prev_cond_dropout=state.patch_refine_prev_cond_dropout,
""",
        """        patch_refine_unique_segments=state.patch_refine_unique_segments,
        patch_refine_use_prev_cond=bool(
            getattr(state, "patch_refine_use_prev_cond", True)
        ),
        patch_refine_prev_cond_dropout=state.patch_refine_prev_cond_dropout,
""",
    )

    bs = root / "configs/base/binary_staged.yaml"
    _sub(
        bs,
        """  patch_refine_unique_segments: false
  patch_refine_prev_cond_dropout: 0.5
""",
        """  patch_refine_unique_segments: false
  patch_refine_use_prev_cond: true
  patch_refine_prev_cond_dropout: 0.5
""",
    )

    dm = root / "models/diffusion_tsf/diffusion_model.py"
    _sub(
        dm,
        """        prev_refine_16 = None
        if unique:
            # Prev GT in the previous primary's row frame (matches AR infer).
            prev_32 = extract_prev_refine_crops(
""",
        """        prev_refine_16 = None
        use_prev = bool(getattr(self.config, "patch_refine_use_prev_cond", True))
        if unique and use_prev:
            # Prev GT in the previous primary's row frame (matches AR infer).
            prev_32 = extract_prev_refine_crops(
""",
    )
    _sub(
        dm,
        """        unique = bool(getattr(self.config, "patch_refine_unique_segments", False))
        lookback_cond, past_maps = self._patch_refine_lookback_cond(past_norm)
        ctx = None if getattr(self.config, "disable_cross_attention", False) else self._get_cross_variate_context(past, past_norm)
        ctx_flat = self._flatten_ctx_for_factorized_dit(ctx, B, V)

        def _sample_locations(
            locs: List,
            prev_refine_16: Optional[torch.Tensor],
        ) -> torch.Tensor:
            if not locs:
                return torch.zeros(0, 1, patch_h, patch_w, device=device)
""",
        """        unique = bool(getattr(self.config, "patch_refine_unique_segments", False))
        use_prev = bool(getattr(self.config, "patch_refine_use_prev_cond", True))
        lookback_cond, past_maps = self._patch_refine_lookback_cond(past_norm)
        ctx = None if getattr(self.config, "disable_cross_attention", False) else self._get_cross_variate_context(past, past_norm)
        ctx_flat = self._flatten_ctx_for_factorized_dit(ctx, B, V)

        def _sample_locations(
            locs: List,
            prev_refine_16: Optional[torch.Tensor],
        ) -> torch.Tensor:
            if not use_prev and prev_refine_16 is not None:
                raise ValueError(
                    "prev_refine_16 must be None when patch_refine_use_prev_cond is false"
                )
            if not locs:
                return torch.zeros(0, 1, patch_h, patch_w, device=device)
""",
    )
    _sub(
        dm,
        """            last_pred: Dict[Tuple[int, int], torch.Tensor] = {}
            primary_pred_by_key: Dict[Tuple[int, int, int, int], torch.Tensor] = {}
            for _col0, col_locs in group_locations_by_col0(primary_locs):
                # Batch all (B,V) at this col0 together.
                prev_chunks = []
                for loc in col_locs:
                    key = (loc.batch_index, loc.variate_index)
                    if key in last_pred:
                        prev_chunks.append(
                            compress_prev_refine_32_to_16(last_pred[key].unsqueeze(0))
                        )
                    else:
                        prev_chunks.append(
                            torch.zeros(1, 1, 16, patch_w, device=device)
                        )
                prev_16 = torch.cat(prev_chunks, dim=0)
                pred = _sample_locations(col_locs, prev_16)
                for j, loc in enumerate(col_locs):
                    last_pred[(loc.batch_index, loc.variate_index)] = pred[j]
                    primary_pred_by_key[
                        (loc.batch_index, loc.variate_index, loc.col0, loc.row0)
                    ] = pred[j]

            gap_locs = select_coverage_gap_locations(
                edges,
                primary_locs,
                canvas_height=canvas_h,
                patch_height=patch_h,
                patch_width=patch_w,
            )
            locations = list(primary_locs) + list(gap_locs)
            n_patches = len(locations)
            patch_cdf = torch.zeros(n_patches, 1, patch_h, patch_w, device=device)
            for i, loc in enumerate(primary_locs):
                patch_cdf[i] = primary_pred_by_key[
                    (loc.batch_index, loc.variate_index, loc.col0, loc.row0)
                ]
            if gap_locs:
                gap_prev = torch.zeros(len(gap_locs), 1, 16, patch_w, device=device)
                gap_pred = _sample_locations(gap_locs, gap_prev)
                patch_cdf[len(primary_locs) :] = gap_pred
""",
        """            last_pred: Dict[Tuple[int, int], torch.Tensor] = {}
            primary_pred_by_key: Dict[Tuple[int, int, int, int], torch.Tensor] = {}
            if use_prev:
                for _col0, col_locs in group_locations_by_col0(primary_locs):
                    prev_chunks = []
                    for loc in col_locs:
                        key = (loc.batch_index, loc.variate_index)
                        if key in last_pred:
                            prev_chunks.append(
                                compress_prev_refine_32_to_16(last_pred[key].unsqueeze(0))
                            )
                        else:
                            prev_chunks.append(
                                torch.zeros(1, 1, 16, patch_w, device=device)
                            )
                    prev_16 = torch.cat(prev_chunks, dim=0)
                    pred = _sample_locations(col_locs, prev_16)
                    for j, loc in enumerate(col_locs):
                        last_pred[(loc.batch_index, loc.variate_index)] = pred[j]
                        primary_pred_by_key[
                            (loc.batch_index, loc.variate_index, loc.col0, loc.row0)
                        ] = pred[j]
            else:
                pred = _sample_locations(primary_locs, None)
                for j, loc in enumerate(primary_locs):
                    primary_pred_by_key[
                        (loc.batch_index, loc.variate_index, loc.col0, loc.row0)
                    ] = pred[j]

            gap_locs = select_coverage_gap_locations(
                edges,
                primary_locs,
                canvas_height=canvas_h,
                patch_height=patch_h,
                patch_width=patch_w,
            )
            locations = list(primary_locs) + list(gap_locs)
            n_patches = len(locations)
            patch_cdf = torch.zeros(n_patches, 1, patch_h, patch_w, device=device)
            for i, loc in enumerate(primary_locs):
                patch_cdf[i] = primary_pred_by_key[
                    (loc.batch_index, loc.variate_index, loc.col0, loc.row0)
                ]
            if gap_locs:
                gap_prev = None if not use_prev else torch.zeros(
                    len(gap_locs), 1, 16, patch_w, device=device,
                )
                gap_pred = _sample_locations(gap_locs, gap_prev)
                patch_cdf[len(primary_locs) :] = gap_pred
""",
    )

    sf = root / "models/diffusion_tsf/pipeline/phases/staged_diffusion_finetune_hp.py"
    _sub(
        sf,
        """class _BaseStagedDiffusionFinetuneHPPhase(PipelinePhase):
    stage = ""

    def should_skip(self, state: PipelineState) -> bool:
        # retrain=true forces a fresh train on a new run, but --resume must still
        # honor local best.pt+metadata so we can finish eval after quota crashes.
        if self.get("retrain", False) and not bool(getattr(state, "resume", False)):
            return False
        best_pt = _stage_best_ckpt(state, self.stage)
""",
        """class _BaseStagedDiffusionFinetuneHPPhase(PipelinePhase):
    stage = ""

    def _copy_reused_stage_checkpoint(self, state: PipelineState) -> None:
        reuse_dir = self.get("reuse_checkpoint_dir")
        if not reuse_dir:
            return
        reuse_dir = os.path.abspath(str(reuse_dir))
        src_best = _stage_best_ckpt(state, self.stage, checkpoint_dir=reuse_dir)
        src_meta = os.path.join(
            _stage_subset_dir(state, self.stage, checkpoint_dir=reuse_dir),
            "metadata.json",
        )
        if not os.path.isfile(src_best) or not os.path.isfile(src_meta):
            raise FileNotFoundError(
                f"{self.name} reuse_checkpoint_dir={reuse_dir!r} missing "
                f"{self.stage} best.pt/metadata.json (looked for {src_best})"
            )
        dest_dir = _stage_subset_dir(state, self.stage)
        dest_best = _stage_best_ckpt(state, self.stage)
        dest_meta = os.path.join(dest_dir, "metadata.json")
        if os.path.isfile(dest_best) and os.path.isfile(dest_meta):
            return
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(src_best, dest_best)
        shutil.copy2(src_meta, dest_meta)
        logger.info("  [%s] copied %s checkpoint from %s", self.name, self.stage, reuse_dir)

    def should_skip(self, state: PipelineState) -> bool:
        # retrain=true forces a fresh train on a new run, but --resume must still
        # honor local best.pt+metadata so we can finish eval after quota crashes.
        if self.get("retrain", False) and not bool(getattr(state, "resume", False)):
            return False
        self._copy_reused_stage_checkpoint(state)
        best_pt = _stage_best_ckpt(state, self.stage)
""",
    )

    pg = root / "models/diffusion_tsf/pipeline/phases/patch_guidance_finetune_hp.py"
    _sub(
        pg,
        """        reuse_from = self.get("reuse_checkpoint_from_config")
        if not reuse_from:
            return False
""",
        """        reuse_from = self.get("reuse_checkpoint_from_config")
        reuse_dir = self.get("reuse_checkpoint_dir")
        if reuse_dir:
            src_ckpt = os.path.join(
                os.path.abspath(str(reuse_dir)), f"{subset_id}_patch_guidance.pt",
            )
            if not self._patch_guidance_ckpt_usable(state, src_ckpt):
                raise FileNotFoundError(
                    f"{self.name} reuse_checkpoint_dir={reuse_dir!r} missing "
                    f"usable patch guidance at {src_ckpt}"
                )
            os.makedirs(os.path.dirname(ft_ckpt), exist_ok=True)
            shutil.copy2(src_ckpt, ft_ckpt)
            state.patch_guidance_finetune_ckpt = ft_ckpt
            logger.info("  [%s] reused finetuned patch guidance from %s", self.name, src_ckpt)
            return True

        if not reuse_from:
            return False
""",
    )
    print("done")


if __name__ == "__main__":
    main()
