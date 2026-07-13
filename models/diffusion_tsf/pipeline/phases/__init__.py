"""Phase registry — maps YAML phase names to concrete classes."""

from models.diffusion_tsf.pipeline.phases.patch_guidance_finetune_hp import PatchGuidanceFinetuneHPPhase
from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import StagedDiffusionPretrainPhase
from models.diffusion_tsf.pipeline.phases.staged_diffusion_finetune_hp import (
    CoarseDiffusionFinetuneHPPhase,
    FineDiffusionFinetuneHPPhase,
    FinerDiffusionFinetuneHPPhase,
    VerticalDualDiffusionFinetuneHPPhase,
)
from models.diffusion_tsf.pipeline.phases.staged_eval import StagedEvalPhase

PHASE_REGISTRY = {
    "patch_guidance_finetune_hp": PatchGuidanceFinetuneHPPhase,
    "staged_diffusion_pretrain": StagedDiffusionPretrainPhase,
    "diffusion_coarse_finetune_hp": CoarseDiffusionFinetuneHPPhase,
    "diffusion_fine_finetune_hp": FineDiffusionFinetuneHPPhase,
    "diffusion_finer_finetune_hp": FinerDiffusionFinetuneHPPhase,
    "diffusion_vertical_dual_finetune_hp": VerticalDualDiffusionFinetuneHPPhase,
    "staged_eval": StagedEvalPhase,
}

__all__ = [
    "PHASE_REGISTRY",
    "PatchGuidanceFinetuneHPPhase",
    "StagedDiffusionPretrainPhase",
    "CoarseDiffusionFinetuneHPPhase",
    "FineDiffusionFinetuneHPPhase",
    "FinerDiffusionFinetuneHPPhase",
    "VerticalDualDiffusionFinetuneHPPhase",
    "StagedEvalPhase",
]
