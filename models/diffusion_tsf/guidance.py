"""
Guidance models for hybrid visual-guide forecasting.

Stage 1 predictors produce coarse forecasts converted to 2D ghost images for
the diffusion model and cross-variate context tokens for DiT.
"""

import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Callable, Optional, Protocol, runtime_checkable
from abc import ABC, abstractmethod

from models.diffusion_tsf.patch_guidance_stack import PatchGuidanceStack
from models.diffusion_tsf.ordinal_window_norm import ranks_from_unit, ranks_to_unit


@runtime_checkable
class GuidanceModel(Protocol):
    """Protocol for Stage 1 guidance models."""

    def get_forecast(
        self,
        past: torch.Tensor,
        forecast_length: int,
        overlap: int = 0,
    ) -> torch.Tensor:
        ...


class BaseGuidance(nn.Module, ABC):
    @abstractmethod
    def get_forecast(
        self,
        past: torch.Tensor,
        forecast_length: int,
        overlap: int = 0,
    ) -> torch.Tensor:
        pass

    def forward(
        self,
        past: torch.Tensor,
        forecast_length: int,
        overlap: int = 0,
    ) -> torch.Tensor:
        return self.get_forecast(past, forecast_length, overlap=overlap)


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

    def _model_pred_len(self) -> int:
        pred_expect = self._past_pred_len()
        if pred_expect is not None:
            return int(pred_expect)
        return int(getattr(self.model, "pred_len", 1))

    def _forward_raw(self, past: torch.Tensor) -> torch.Tensor:
        """One iTransformer forward at native pred_len."""
        past = self._slice_past_to_model_len(past)
        pred_step = self._model_pred_len()

        is_univariate = past.dim() == 2
        if is_univariate:
            x_enc = past.unsqueeze(-1)
        else:
            x_enc = past.permute(0, 2, 1)

        batch_size, _seq_len, num_vars = x_enc.shape
        x_mark_enc = None
        x_dec = torch.zeros(
            batch_size, pred_step, num_vars,
            device=x_enc.device, dtype=x_enc.dtype,
        )
        x_mark_dec = None

        output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if isinstance(output, tuple):
            output = output[0]

        if is_univariate:
            output = output.squeeze(-1)
        else:
            output = output.permute(0, 2, 1)

        return output[..., :pred_step]

    @contextmanager
    def _instance_norm_disabled(self):
        if not hasattr(self.model, "use_norm"):
            yield
            return
        saved = bool(self.model.use_norm)
        self.model.use_norm = False
        try:
            yield
        finally:
            self.model.use_norm = saved

    def _autoregressive_rollout(
        self,
        past: torch.Tensor,
        forecast_length: int,
        overlap: int,
        forward_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        if forecast_length <= 0:
            raise ValueError(f"forecast_length must be positive, got {forecast_length}")

        pred_step = self._model_pred_len()
        if forecast_length <= pred_step:
            return forward_fn(past)[..., :forecast_length]

        K = max(0, int(overlap))
        chunks = []
        remaining = forecast_length
        cur_past = past

        while remaining > 0:
            step_out = forward_fn(cur_past)
            take = min(remaining, pred_step)
            chunks.append(step_out[..., :take])
            remaining -= take
            if remaining <= 0:
                break

            if K > 0:
                roll = step_out[..., K:pred_step]
                cur_past = torch.cat([cur_past[..., K:], step_out[..., :K], roll], dim=-1)
            else:
                cur_past = torch.cat([cur_past, step_out], dim=-1)
            cur_past = self._slice_past_to_model_len(cur_past)

        return torch.cat(chunks, dim=-1)

    @torch.no_grad()
    def get_forecast(
        self,
        past: torch.Tensor,
        forecast_length: int,
        overlap: int = 0,
    ) -> torch.Tensor:
        return self._autoregressive_rollout(
            past, forecast_length, overlap, self._forward_raw,
        )

    @torch.no_grad()
    def get_forecast_window_norm(
        self,
        past_norm: torch.Tensor,
        forecast_length: int,
        overlap: int = 0,
    ) -> torch.Tensor:
        """Forecast in diffusion per-window z-score space (same scale as future_norm).

        Disables iTransformer instance normalization and autoregresses entirely in the
        window-normalized domain so the 2D guidance ghost matches the diffused horizon.
        """
        def _forward_norm_space(past: torch.Tensor) -> torch.Tensor:
            return self._forward_raw(past)

        with self._instance_norm_disabled():
            return self._autoregressive_rollout(
                past_norm, forecast_length, overlap, _forward_norm_space,
            )


class PatchDecoderGuidance(BaseGuidance):
    """MMPD decoder patch tokens mixed across variates for DiT cross-attention."""

    def __init__(
        self,
        stack: PatchGuidanceStack,
        *,
        chunk_horizon: int,
        ordinal_ladder=None,
    ):
        super().__init__()
        self.stack = stack
        self.chunk_horizon = int(chunk_horizon)
        self.ordinal_ladder = ordinal_ladder
        self._token_variate_ids: Optional[torch.Tensor] = None

        for param in self.stack.parameters():
            param.requires_grad = False
        self.training = False
        self.stack.eval()

    def train(self, mode: bool = True):
        self.training = False
        self.stack.eval()
        return self

    def eval(self):
        self.training = False
        self.stack.eval()
        return self

    @property
    def token_variate_ids(self) -> Optional[torch.Tensor]:
        return self._token_variate_ids

    def _ladder_for_batch(self, x: torch.Tensor):
        if self.ordinal_ladder is None:
            return None
        batch_size = x.shape[0] if x.dim() >= 3 else 1
        return self.ordinal_ladder.expand_batch(batch_size)

    def _to_model_space(self, x: torch.Tensor) -> torch.Tensor:
        ladder = self._ladder_for_batch(x)
        if ladder is None:
            return x
        return ranks_to_unit(x, ladder)

    def _from_model_space(self, x: torch.Tensor) -> torch.Tensor:
        ladder = self._ladder_for_batch(x)
        if ladder is None:
            return x
        return ranks_from_unit(x, ladder)

    def _prepare_past(self, past: torch.Tensor) -> torch.Tensor:
        if past.dim() == 2:
            past = past.unsqueeze(1)
        return self._to_model_space(past)

    @torch.no_grad()
    def encode_past_tokens(self, past: torch.Tensor) -> torch.Tensor:
        """Per-channel past tokens (B, V, N_past, d) before mixer mixing."""
        return self.stack.decoder.encode_past_tokens(self._prepare_past(past))

    @torch.no_grad()
    def mix_past_tokens(
        self,
        past_tokens: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mix full-M tokens. Optional pad mask is True=drop (eval-safe)."""
        mixed, var_ids = self.stack.mixer(
            past_tokens, src_key_padding_mask=src_key_padding_mask,
        )
        self._token_variate_ids = var_ids
        return mixed, var_ids

    @torch.no_grad()
    def get_encoder_tokens(self, past: torch.Tensor) -> torch.Tensor:
        """Return mixed patch context tokens (B, M, context_dim)."""
        mixed, var_ids = self.stack.encode_context(self._prepare_past(past))
        self._token_variate_ids = var_ids
        return mixed

    def _forward_chunk(self, past: torch.Tensor) -> torch.Tensor:
        out = self.stack.forecast(past)
        if self.ordinal_ladder is not None:
            out = out.clamp(0.0, 1.0)
        return out

    def _autoregressive_rollout(
        self,
        past: torch.Tensor,
        forecast_length: int,
        overlap: int,
    ) -> torch.Tensor:
        if past.dim() == 2:
            past = past.unsqueeze(1)
        pred_step = self.chunk_horizon
        if forecast_length <= pred_step:
            return self._forward_chunk(past)[..., :forecast_length]

        K = max(0, int(overlap))
        chunks = []
        remaining = forecast_length
        cur_past = past
        in_len = self.stack.config.in_len

        while remaining > 0:
            step_out = self._forward_chunk(cur_past)
            take = min(remaining, pred_step)
            chunks.append(step_out[..., :take])
            remaining -= take
            if remaining <= 0:
                break
            if K > 0:
                roll = step_out[..., K:pred_step]
                cur_past = torch.cat([cur_past[..., K:], step_out[..., :K], roll], dim=-1)
            else:
                cur_past = torch.cat([cur_past, step_out], dim=-1)
            if cur_past.shape[-1] > in_len:
                cur_past = cur_past[..., -in_len:]
        return torch.cat(chunks, dim=-1)

    @torch.no_grad()
    def get_forecast(
        self,
        past: torch.Tensor,
        forecast_length: int,
        overlap: int = 0,
    ) -> torch.Tensor:
        past_model = self._to_model_space(past)
        forecast_model = self._autoregressive_rollout(past_model, forecast_length, overlap)
        return self._from_model_space(forecast_model)

    @torch.no_grad()
    def get_forecast_window_norm(
        self,
        past_norm: torch.Tensor,
        forecast_length: int,
        overlap: int = 0,
    ) -> torch.Tensor:
        past_model = self._to_model_space(past_norm)
        forecast_model = self._autoregressive_rollout(past_model, forecast_length, overlap)
        return self._from_model_space(forecast_model)


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
