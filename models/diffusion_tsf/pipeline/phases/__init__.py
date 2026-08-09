"""Phase registry — maps YAML phase names to concrete classes."""

from models.diffusion_tsf.pipeline.phases.patch_guidance_finetune_hp import PatchGuidanceFinetuneHPPhase
from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import StagedDiffusionPretrainPhase
from models.diffusion_tsf.pipeline.phases.staged_diffusion_finetune_hp import (
    CoarseDiffusionFinetuneHPPhase,
    PatchRefineDiffusionFinetuneHPPhase,
)
from models.diffusion_tsf.pipeline.phases.staged_eval import StagedEvalPhase

PHASE_REGISTRY = {
    "patch_guidance_finetune_hp": PatchGuidanceFinetuneHPPhase,
    "staged_diffusion_pretrain": StagedDiffusionPretrainPhase,
    "diffusion_coarse_finetune_hp": CoarseDiffusionFinetuneHPPhase,
    "diffusion_patch_refine_finetune_hp": PatchRefineDiffusionFinetuneHPPhase,
    "staged_eval": StagedEvalPhase,
}

__all__ = [
    "PHASE_REGISTRY",
    "PatchGuidanceFinetuneHPPhase",
    "StagedDiffusionPretrainPhase",
    "CoarseDiffusionFinetuneHPPhase",
    "PatchRefineDiffusionFinetuneHPPhase",
    "StagedEvalPhase",
]
