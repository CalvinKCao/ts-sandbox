"""Pipeline orchestrator — runs an ordered list of phases."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Dict, List

from models.diffusion_tsf.pipeline.phase import PipelinePhase
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline import wandb_utils

logger = logging.getLogger(__name__)


class Pipeline:
    """Runs phases sequentially, managing wandb groups and error handling."""

    def __init__(self, phases: List[PipelinePhase], state: PipelineState):
        self.phases = phases
        self.state = state

    def run(self) -> PipelineState:
        self.state.seed_everything()
        self.state.resolve_device()
        self.state.ensure_dirs()

        # Generate wandb group if enabled
        if self.state.wandb_enabled and not self.state.wandb_group:
            self.state.wandb_group = wandb_utils.make_group_name(
                self.state.experiment_name,
                self.state.dataset,
                self.state.seed,
            )

        logger.info("=" * 60)
        logger.info(f"Pipeline: {self.state.experiment_name}")
        logger.info(f"Dataset: {self.state.dataset} | Variates: {self.state.n_variates}")
        logger.info(f"Phases: {[p.name for p in self.phases]}")
        logger.info(f"Device: {self.state.device}")
        if self.state.wandb_group:
            logger.info(f"wandb group: {self.state.wandb_group}")
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
                # Build a flat config dict for this phase
                phase_config = _state_as_config(self.state)
                phase_config.update(phase.overrides)
                run = wandb_utils.init_phase_run(
                    phase_name=phase.wandb_run_name,
                    group=self.state.wandb_group or "",
                    project=self.state.wandb_project,
                    job_type=phase.wandb_job_type,
                    config=phase_config,
                    tags=[self.state.dataset, self.state.diffusion_type],
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


def _state_as_config(state: PipelineState) -> Dict:
    """Flatten state into a serializable dict for wandb config."""
    d = {}
    for k, v in asdict(state).items():
        # Skip non-serializable / large fields
        if k in ("device", "phase_configs", "extra"):
            continue
        if v is not None:
            d[k] = v
    return d
