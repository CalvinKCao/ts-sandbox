"""
Guidance models for hybrid visual-guide forecasting.

Stage 1 predictors produce coarse forecasts converted to 2D ghost images for
the diffusion model. Only iTransformer-backed guidance is supported.
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
        forecast_length: int,
    ) -> torch.Tensor:
        ...


class BaseGuidance(nn.Module, ABC):
    @abstractmethod
    def get_forecast(
        self,
        past: torch.Tensor,
        forecast_length: int,
    ) -> torch.Tensor:
        pass

    def forward(
        self,
        past: torch.Tensor,
        forecast_length: int,
    ) -> torch.Tensor:
        return self.get_forecast(past, forecast_length)


class iTransformerGuidance(BaseGuidance):
    """Wrapper for a pre-trained iTransformer used as Stage 1 guidance."""

    def __init__(
        self,
        model: nn.Module,
        use_norm: bool = True,
        seq_len: Optional[int] = None,
        pred_len: Optional[int] = None,
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
        self.training = False
        self.model.eval()
        return self

    def eval(self):
        self.training = False
        self.model.eval()
        return self

    def _past_seq_len(self) -> Optional[int]:
        if self.seq_len is not None:
            return self.seq_len
        return getattr(self.model, "seq_len", None)

    def _past_pred_len(self) -> Optional[int]:
        if self.pred_len is not None:
            return self.pred_len
        return getattr(self.model, "pred_len", None)

    def _slice_past_to_model_len(self, past: torch.Tensor) -> torch.Tensor:
        Lwant = self._past_seq_len()
        if Lwant is None:
            return past
        if past.dim() == 2:
            if past.shape[-1] < Lwant:
                raise ValueError(
                    f"past length {past.shape[-1]} < iTransformer seq_len {Lwant}"
                )
            if past.shape[-1] > Lwant:
                return past[:, -Lwant:]
            return past
        if past.dim() != 3:
            raise ValueError(f"past must be (B,L) or (B,V,L), got shape {tuple(past.shape)}")
        if past.shape[-1] < Lwant:
            raise ValueError(
                f"past length {past.shape[-1]} < iTransformer seq_len {Lwant}"
            )
        if past.shape[-1] > Lwant:
            return past[..., -Lwant:]
        return past

    @torch.no_grad()
    def get_encoder_tokens(self, past: torch.Tensor) -> torch.Tensor:
        """Return iTransformer encoder output before the linear projector."""
        past = self._slice_past_to_model_len(past)
        is_univariate = past.dim() == 2
        if is_univariate:
            x_enc = past.unsqueeze(-1)
        else:
            x_enc = past.permute(0, 2, 1)

        if self.model.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        enc_out = self.model.enc_embedding(x_enc, None)
        enc_out, _ = self.model.encoder(enc_out, attn_mask=None)
        return enc_out

    @torch.no_grad()
    def get_forecast(
        self,
        past: torch.Tensor,
        forecast_length: int,
    ) -> torch.Tensor:
        pred_expect = self._past_pred_len()
        if pred_expect is not None and forecast_length != pred_expect:
            raise ValueError(
                f"iTransformer was trained for pred_len={pred_expect}, "
                f"but got forecast_length={forecast_length}"
            )

        past = self._slice_past_to_model_len(past)

        is_univariate = past.dim() == 2
        if is_univariate:
            past = past.unsqueeze(-1)
        else:
            past = past.permute(0, 2, 1)

        batch_size, seq_len, num_vars = past.shape
        x_enc = past
        x_mark_enc = None
        x_dec = torch.zeros(
            batch_size, forecast_length, num_vars,
            device=past.device, dtype=past.dtype,
        )
        x_mark_dec = None

        output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if isinstance(output, tuple):
            output = output[0]

        if is_univariate:
            output = output.squeeze(-1)
        else:
            output = output.permute(0, 2, 1)

        return output


class iTransformerTokenAdapter(nn.Module):
    """Projects frozen iTransformer encoder tokens to context_dim for DiT cross-attention."""

    def __init__(
        self,
        d_model: int,
        context_dim: int,
        max_variates: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(d_model, context_dim)
        self.variate_embed = nn.Embedding(max_variates, context_dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(context_dim)

    def forward(self, enc_tokens: torch.Tensor) -> torch.Tensor:
        """enc_tokens: (B, V, d_model) -> (B, V, context_dim)"""
        B, V, _ = enc_tokens.shape
        x = self.proj(enc_tokens)
        ids = torch.arange(V, device=enc_tokens.device)
        x = x + self.variate_embed(ids)
        return self.norm(self.drop(x))
