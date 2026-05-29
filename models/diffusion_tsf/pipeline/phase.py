"""Abstract base class for pipeline phases."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from models.diffusion_tsf.pipeline.state import PipelineState

logger = logging.getLogger(__name__)


class PipelinePhase(ABC):
    """One discrete step in the training pipeline.

    Subclasses implement ``execute`` to do their work (HP search, training,
    eval, etc.), reading inputs from ``state`` and writing outputs back.
    """

    name: str = "base_phase"

    def __init__(self, **overrides: Any):
        self.overrides = overrides

    def get(self, key: str, default: Any = None) -> Any:
        """Read a per-phase override, falling back to *default*."""
        return self.overrides.get(key, default)

    # -- hooks --

    def should_skip(self, state: PipelineState) -> bool:
        """Return True to skip execution (e.g. cached artifacts exist).

        Default: never skip.
        """
        return False

    @abstractmethod
    def execute(self, state: PipelineState) -> PipelineState:
        """Run the phase, mutating *state* with produced artifacts.

        Must return the (possibly mutated) state object.
        """
        ...

    @property
    def wandb_job_type(self) -> str:
        """wandb ``job_type`` tag for runs created by this phase."""
        return self.name

    @property
    def wandb_run_name(self) -> str:
        """Human-readable name for the wandb run."""
        return self.name.replace("_", "-")

    def __repr__(self) -> str:
        overrides_str = ", ".join(f"{k}={v!r}" for k, v in self.overrides.items() if k != "phase")
        return f"{self.__class__.__name__}({overrides_str})"
