"""
Guidance models for Hybrid "Visual Guide" Forecasting.

Stage 1 predictor generates a coarse forecast → converted to 2D ghost image →
fed to the diffusion model as additional conditioning. iTransformerGuidance
is the only supported type; passing no guidance model raises an error.
"""

import torch
import torch.nn as nn
from typing import Optional, Protocol, runtime_checkable
from abc import ABC, abstractmethod


@runtime_checkable
class GuidanceModel(Protocol):
    """Protocol for Stage 1 guidance models."""

    def get_forecast(
        self,
        past: torch.Tensor,
        forecast_length: int
    ) -> torch.Tensor:
        """Generate coarse forecast from past context.

        Args:
            past: (batch, [num_vars,] past_len)
            forecast_length: steps to predict

        Returns:
            (batch, [num_vars,] forecast_length)
        """
        ...


class BaseGuidance(nn.Module, ABC):
    """Base class for guidance models."""

    @abstractmethod
    def get_forecast(
        self,
        past: torch.Tensor,
        forecast_length: int
    ) -> torch.Tensor:
        pass

    def forward(self, past: torch.Tensor, forecast_length: int) -> torch.Tensor:
        return self.get_forecast(past, forecast_length)


class iTransformerGuidance(BaseGuidance):
    """Wrapper for pre-trained iTransformer model.

    Wraps an iTransformer checkpoint to use as Stage 1 guidance.
    The iTransformer is kept frozen in eval mode at all times — even
    when model.train() is called on the parent DiffusionTSF.

    Usage:
        itrans_model = load_itransformer_checkpoint(...)
        guidance = iTransformerGuidance(itrans_model)
        forecast = guidance.get_forecast(past, forecast_length=96)
    """

    def __init__(
        self,
        model: nn.Module,
        use_norm: bool = True,
        seq_len: Optional[int] = None,
        pred_len: Optional[int] = None
    ):
        super().__init__()
        self.model = model
        self.use_norm = use_norm
        self.seq_len = seq_len
        self.pred_len = pred_len

        for param in self.model.parameters():
            param.requires_grad = False

        self.training = False
        self.model.eval()

    def train(self, mode: bool = True):
        # always stay in eval; guidance is frozen
        self.training = False
        self.model.eval()
        return self

    def eval(self):
        self.training = False
        self.model.eval()
        return self

    @torch.no_grad()
    def get_forecast(
        self,
        past: torch.Tensor,
        forecast_length: int
    ) -> torch.Tensor:
        if self.pred_len is not None and forecast_length != self.pred_len:
            raise ValueError(
                f"iTransformer was trained for pred_len={self.pred_len}, "
                f"but got forecast_length={forecast_length}"
            )

        is_univariate = (past.dim() == 2)
        if is_univariate:
            past = past.unsqueeze(-1)
        else:
            past = past.permute(0, 2, 1)

        batch_size, seq_len, num_vars = past.shape

        if self.seq_len is not None and seq_len != self.seq_len:
            raise ValueError(
                f"iTransformer was trained for seq_len={self.seq_len}, "
                f"but got past_len={seq_len}"
            )

        x_enc = past
        x_dec = torch.zeros(
            batch_size, forecast_length, num_vars,
            device=past.device, dtype=past.dtype
        )

        output = self.model(x_enc, None, x_dec, None)
        if isinstance(output, tuple):
            output = output[0]

        if is_univariate:
            output = output.squeeze(-1)
        else:
            output = output.permute(0, 2, 1)

        return output
