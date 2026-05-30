"""Phase registry — maps YAML phase names to concrete classes."""

from models.diffusion_tsf.pipeline.phases.itrans_hp_pretrain import ITransHPPretrainPhase
from models.diffusion_tsf.pipeline.phases.diffusion_hp_pretrain import DiffusionHPPretrainPhase
from models.diffusion_tsf.pipeline.phases.itrans_finetune_hp import ITransFinetuneHPPhase
from models.diffusion_tsf.pipeline.phases.diffusion_finetune_hp import DiffusionFinetuneHPPhase
from models.diffusion_tsf.pipeline.phases.eval import EvalPhase

PHASE_REGISTRY = {
    "itrans_hp_pretrain": ITransHPPretrainPhase,
    "diffusion_hp_pretrain": DiffusionHPPretrainPhase,
    "itrans_finetune_hp": ITransFinetuneHPPhase,
    "diffusion_finetune_hp": DiffusionFinetuneHPPhase,
    "eval": EvalPhase,
}

__all__ = [
    "PHASE_REGISTRY",
    "ITransHPPretrainPhase",
    "DiffusionHPPretrainPhase",
    "ITransFinetuneHPPhase",
    "DiffusionFinetuneHPPhase",
    "EvalPhase",
]
