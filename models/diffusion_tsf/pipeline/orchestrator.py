"""Pipeline orchestrator — runs an ordered list of phases."""

from __future__ import annotations

import logging
from typing import List, Optional

from models.diffusion_tsf.pipeline.phase import PipelinePhase
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline.config import build_wandb_config
from models.diffusion_tsf.pipeline import wandb_utils

logger = logging.getLogger(__name__)


class Pipeline:
    """Runs phases sequentially, managing wandb groups and error handling."""

    def __init__(
        self,
        phases: List[PipelinePhase],
        state: PipelineState,
        merged_config: Optional[dict] = None,
    ):
        self.phases = phases
        self.state = state
        self.merged_config = merged_config or state.merged_config or {}

    def run(self) -> PipelineState:
        self.state.seed_everything()
        self.state.resolve_device()
        self.state.ensure_dirs()

        wandb_manifest: dict = {}
        if self.state.wandb_enabled:
            wandb_manifest = wandb_utils.resolve_wandb_settings(
                self.state, self.merged_config,
            )

        logger.info("=" * 60)
        logger.info(f"Pipeline: {self.state.experiment_name}")
        logger.info(f"Dataset: {self.state.dataset} | Variates: {self.state.n_variates}")
        logger.info(f"Phases: {[p.name for p in self.phases]}")
        logger.info(f"Device: {self.state.device}")
        if self.state.wandb_group:
            logger.info(f"wandb group: {self.state.wandb_group}")
        if self.state.wandb_enabled:
            logger.info(f"wandb project: {self.state.wandb_project}")
        logger.info("=" * 60)

        for i, phase in enumerate(self.phases):
            phase_label = f"[{i+1}/{len(self.phases)}] {phase.name}"

            if phase.should_skip(self.state):
                logger.info(f"{phase_label}: SKIPPED (cached)")
                continue

            logger.info(f"{phase_label}: STARTING")
            logger.info(f"  overrides: {phase.overrides}")

            # Per-phase wandb run
            run = None
            if self.state.wandb_enabled:
                phase_config = build_wandb_config(
                    self.merged_config,
                    self.state,
                    phase_name=phase.name,
                    phase_overrides=phase.overrides,
                )
                run_id = self.state.wandb_phase_run_ids.get(phase.name)
                run = wandb_utils.init_phase_run(
                    phase_slug=phase.wandb_run_name,
                    group=self.state.wandb_group or "",
                    project=self.state.wandb_project,
                    job_type=phase.wandb_job_type,
                    config=phase_config,
                    tags=wandb_utils.build_run_tags(
                        dataset=self.state.dataset,
                        phase_name=phase.name,
                        extra_tags=self.state.wandb_tags,
                    ),
                    yaml_path=self.merged_config.get("_yaml_path"),
                    run_id=run_id,
                )
                if run is not None and getattr(run, "id", None):
                    wandb_utils.record_phase_run_id(
                        self.state.checkpoint_dir,
                        phase.name,
                        run.id,
                        wandb_manifest,
                    )

            try:
                self.state = phase.execute(self.state)
                logger.info(f"{phase_label}: DONE")
            except KeyboardInterrupt:
                logger.info(f"\nInterrupted during {phase.name}")
                raise
            except Exception:
                logger.exception(f"{phase_label}: FAILED")
                raise
            finally:
                if run is not None:
                    wandb_utils.finish_phase_run()

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        return self.state
