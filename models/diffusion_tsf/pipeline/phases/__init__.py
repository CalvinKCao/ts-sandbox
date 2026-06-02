"""Phase registry — maps YAML phase names to concrete classes."""

from models.diffusion_tsf.pipeline.phases.itrans_hp_pretrain import ITransHPPretrainPhase
from models.diffusion_tsf.pipeline.phases.diffusion_hp_pretrain import DiffusionHPPretrainPhase
from models.diffusion_tsf.pipeline.phases.itrans_finetune_hp import ITransFinetuneHPPhase
from models.diffusion_tsf.pipeline.phases.diffusion_finetune_hp import DiffusionFinetuneHPPhase
from models.diffusion_tsf.pipeline.phases.eval import EvalPhase
from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import StagedDiffusionPretrainPhase
from models.diffusion_tsf.pipeline.phases.staged_diffusion_finetune_hp import (
    CoarseDiffusionFinetuneHPPhase,
    FineDiffusionFinetuneHPPhase,
)
from models.diffusion_tsf.pipeline.phases.staged_eval import StagedEvalPhase

PHASE_REGISTRY = {
    "itrans_hp_pretrain": ITransHPPretrainPhase,
    "diffusion_hp_pretrain": DiffusionHPPretrainPhase,
    "itrans_finetune_hp": ITransFinetuneHPPhase,
    "diffusion_finetune_hp": DiffusionFinetuneHPPhase,
    "staged_diffusion_pretrain": StagedDiffusionPretrainPhase,
    "diffusion_coarse_finetune_hp": CoarseDiffusionFinetuneHPPhase,
    "diffusion_fine_finetune_hp": FineDiffusionFinetuneHPPhase,
    "staged_eval": StagedEvalPhase,
    "eval": EvalPhase,
}

__all__ = [
    "PHASE_REGISTRY",
    "ITransHPPretrainPhase",
    "DiffusionHPPretrainPhase",
    "ITransFinetuneHPPhase",
    "DiffusionFinetuneHPPhase",
    "StagedDiffusionPretrainPhase",
    "CoarseDiffusionFinetuneHPPhase",
    "FineDiffusionFinetuneHPPhase",
    "StagedEvalPhase",
    "EvalPhase",
]
