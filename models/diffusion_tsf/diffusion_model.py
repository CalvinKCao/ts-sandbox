"""
Complete Diffusion-based Time Series Forecasting Model.

Binary CDF images, FactorizedDiT denoiser, iTransformer guidance channel.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from .config import DiffusionTSFConfig
from .preprocessing import TimeSeriesTo2D
from .diffusion import BinaryDiffusionScheduler
from .ordinal_window_norm import OrdinalLadder, ordinal_decode, ordinal_encode
from .guidance import GuidanceModel, iTransformerTokenAdapter
from .dit import FactorizedDiT

logger = logging.getLogger(__name__)


def beam_search_decoder(
    cdf_map: torch.Tensor,
    bin_centers: torch.Tensor,
    beam_width: int = 5,
    jump_penalty_scale: float = 1.0,
    search_radius: int = 10,
    eps: float = 1e-8
) -> torch.Tensor:
    """beam search for decoding the CDF/occupancy maps.
    
    finds a path through the prob map. tries to maximize likelihood while 
    punishing big jumps between time steps so it stays smoothish.
    """
    batch_size, height, seq_len = cdf_map.shape
    device = cdf_map.device
    
    # 1. cdf to pdf conversion: drop[y] = cdf[y] - cdf[y+1]
    # occupancy is high at bottom, low at top. 
    # stick some zeros at the top so shape matches
    pdf = torch.zeros_like(cdf_map)
    pdf[:, :-1, :] = cdf_map[:, :-1, :] - cdf_map[:, 1:, :]
    pdf = torch.clamp(pdf, min=0.0)
    
    # norm the pdf per column
    pdf_sum = pdf.sum(dim=1, keepdim=True).clamp(min=eps)
    pdf = pdf / pdf_sum
    
    # log probs (clamped to avoid -inf explosion)
    log_pdf = torch.log(pdf.clamp(min=eps))  # (batch, height, seq_len)
    
    results = []
    
    for b in range(batch_size):
        log_pdf_b = log_pdf[b]  # (height, seq_len)
        
        # start with top beam_width positions at t=0
        init_scores = log_pdf_b[:, 0]  # (height,)
        topk_scores, topk_indices = init_scores.topk(min(beam_width, height))
        
        # beams: (score, [path indices])
        beam_scores = topk_scores  # (beam_width,)
        beam_paths = topk_indices.unsqueeze(1)  # (beam_width, 1)
        
        # walk through time
        for t in range(1, seq_len):
            num_beams = beam_scores.shape[0]
            
            # current ends for each beam
            prev_positions = beam_paths[:, -1]  # (num_beams,)
            
            # For each beam, compute scores for all possible next positions
            # within search_radius
            all_candidates_scores = []
            all_candidates_paths = []
            
            for beam_idx in range(num_beams):
                prev_pos = prev_positions[beam_idx].item()
                prev_score = beam_scores[beam_idx]
                
                # Define search window
                lo = max(0, prev_pos - search_radius)
                hi = min(height, prev_pos + search_radius + 1)
                
                # Candidate positions and their scores
                candidates = torch.arange(lo, hi, device=device)
                candidate_log_probs = log_pdf_b[lo:hi, t]
                
                # Jump penalties
                jumps = (candidates - prev_pos).abs().float()
                penalties = jump_penalty_scale * jumps
                
                # Total scores
                candidate_scores = prev_score + candidate_log_probs - penalties
                
                # Store candidates
                for i, pos in enumerate(candidates):
                    all_candidates_scores.append(candidate_scores[i])
                    new_path = torch.cat([beam_paths[beam_idx], pos.unsqueeze(0)])
                    all_candidates_paths.append(new_path)
            
            if len(all_candidates_scores) == 0:
                # Edge case: no valid candidates, keep current beams
                continue
            
            # Stack and prune to top beam_width
            all_scores = torch.stack(all_candidates_scores)
            topk_count = min(beam_width, len(all_scores))
            topk_scores, topk_idx = all_scores.topk(topk_count)
            
            beam_scores = topk_scores
            beam_paths = torch.stack([all_candidates_paths[i] for i in topk_idx.tolist()])
        
        # Select best path
        best_idx = beam_scores.argmax()
        best_path = beam_paths[best_idx]  # (seq_len,) indices
        
        # Convert indices to values using bin_centers
        path_values = bin_centers[best_path.long()]
        results.append(path_values)
    
    return torch.stack(results)  # (batch, seq_len)


class DiffusionTSF(nn.Module):
    """Binary diffusion TSF with FactorizedDiT and optional iTransformer guidance."""

    def __init__(
        self,
        config: DiffusionTSFConfig,
        guidance_model: Optional[Union[GuidanceModel, nn.Module]] = None,
    ):
        super().__init__()
        self.config = config

        needs_guidance_model = config.use_guidance_channel or not config.disable_cross_attention
        if needs_guidance_model and guidance_model is None:
            raise ValueError(
                "A guidance model is required for forecast channels or cross-variate "
                "encoder tokens; none was provided."
            )

        self.to_2d = TimeSeriesTo2D(
            height=config.image_height,
            max_scale=config.max_scale,
        )
        self.guidance_model = guidance_model if needs_guidance_model else None

        backbone_in_channels = config.backbone_in_channels
        self.noise_predictor = FactorizedDiT(
            in_channels=backbone_in_channels,
            cond_channels=config.visual_cond_channels,
            out_channels=config.dit_out_channels,
            image_height=config.image_height,
            patch_size=config.dit_patch_size,
            embed_dim=config.dit_embed_dim,
            depth=config.dit_depth,
            num_heads=config.dit_num_heads,
            mlp_ratio=config.dit_mlp_ratio,
            dropout=config.dit_dropout,
            context_dim=config.context_embedding_dim,
            gradient_checkpointing=config.use_gradient_checkpointing,
            use_scale_embedding=False,
            enable_cross_scale_attention=False,
            use_variate_embedding=(
                config.use_variate_embedding
                and config.variate_factorized
                and config.num_variables > 1
            ),
            max_variates=max(config.num_variables, 512),
            cross_variate_context_bias=config.cross_variate_context_bias,
        )

        self._ctx_token_variate_ids: Optional[torch.Tensor] = None
        if config.guidance_type == "patch_decoder":
            self.context_encoder = None
        else:
            self.context_encoder = iTransformerTokenAdapter(
                d_model=config.itrans_d_model,
                context_dim=config.context_embedding_dim,
                max_variates=max(config.num_variables, 512),
                dropout=0.1,
            )

        self.binary_scheduler = BinaryDiffusionScheduler(
            num_steps=config.binary_num_steps,
            beta_start=config.binary_beta_start,
            beta_end=config.binary_beta_end,
            schedule_type=config.binary_noise_schedule,
            length_mode=getattr(config, "binary_length_mode", "none"),
            length_g=float(getattr(config, "binary_length_g", 1.0)),
            length_scale=float(getattr(config, "binary_length_scale", 1.0)),
        )

        logger.debug("DiffusionTSF initialized:")
        logger.debug(
            "  Variables: %d (%s)",
            config.num_variables,
            "multivariate" if config.num_variables > 1 else "univariate",
        )
        logger.debug(
            "  Lookback: %d, Forecast: %d",
            config.lookback_length,
            config.forecast_length,
        )
        logger.debug(
            "  Image size: %d x %d (H x W)",
            config.image_height,
            config.forecast_length,
        )

    def to(self, device):
        """Move model and scheduler to device."""
        super().to(device)
        if self.binary_scheduler is not None:
            self.binary_scheduler = self.binary_scheduler.to(device)
        return self

    def _get_coordinate_grid(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Create a vertical coordinate gradient map."""
        y_coords = torch.linspace(1.0, -1.0, height, device=device, dtype=dtype)
        coord_grid = y_coords.view(1, 1, height, 1).expand(batch_size, 1, height, width)
        return coord_grid
    
    def _inject_coordinate_channel(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate vertical coordinate channel to input tensor."""
        if not self.config.use_coordinate_channel:
            return x
        batch_size, _, height, width = x.shape
        coord_grid = self._get_coordinate_grid(
            batch_size, height, width, x.device, x.dtype
        )
        return torch.cat([x, coord_grid], dim=1)
    
    def _get_time_features(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create horizontal time-aware coordinate channels."""
        ramp = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        ramp = ramp.view(1, 1, 1, width).expand(batch_size, 1, height, width)
        t_idx = torch.arange(width, device=device, dtype=dtype)
        sine = torch.sin(2 * math.pi * t_idx / self.config.seasonal_period)
        sine = sine.view(1, 1, 1, width).expand(batch_size, 1, height, width)
        return ramp, sine
    
    def _inject_time_channels(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate horizontal time coordinate channels to input tensor."""
        if not self.config.use_time_ramp and not self.config.use_time_sine:
            return x
        batch_size, _, height, width = x.shape
        ramp, sine = self._get_time_features(
            batch_size, height, width, x.device, x.dtype
        )
        channels_to_add = [x]
        if self.config.use_time_ramp:
            channels_to_add.append(ramp)
        if self.config.use_time_sine:
            channels_to_add.append(sine)
        return torch.cat(channels_to_add, dim=1)

    def _resample_1d_time_series(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """Linearly resample the trailing time axis to exactly target_len."""
        if target_len <= 0:
            raise ValueError(f"target_len must be positive, got {target_len}")
        if x.shape[-1] == target_len:
            return x
        if x.shape[-1] == 1:
            return x.expand(*x.shape[:-1], target_len)
        flat = x.reshape(-1, 1, x.shape[-1])
        out = F.interpolate(flat, size=target_len, mode="linear", align_corners=False)
        return out.reshape(*x.shape[:-1], target_len)

    def _representation_time_stride(self) -> int:
        return max(1, int(getattr(self.config, "representation_time_stride", 1)))

    def _subsample_repr_time(self, x: torch.Tensor) -> torch.Tensor:
        stride = self._representation_time_stride()
        if stride <= 1:
            return x
        return x[..., ::stride]

    def _repr_time_len(self, raw_len: int) -> int:
        stride = self._representation_time_stride()
        if stride <= 1:
            return raw_len
        return (int(raw_len) + stride - 1) // stride

    def _repr_forecast_width(self, raw_horizon_width: Optional[int] = None) -> int:
        raw = int(raw_horizon_width if raw_horizon_width is not None else self.config.forecast_length)
        return self._repr_time_len(raw)

    def _overlap_repr_cols(self) -> int:
        k = int(self.config.lookback_overlap)
        if k <= 0:
            return 0
        stride = self._representation_time_stride()
        return k // stride if stride > 1 else k

    def _raw_dataset_horizon(self) -> int:
        dhz = int(self.config.dataset_forecast_length or 0)
        if dhz > 0:
            return dhz
        return max(1, int(self.config.forecast_length) - int(self.config.lookback_overlap))

    def _upsample_repr_to_raw_horizon(self, x_repr: torch.Tensor) -> torch.Tensor:
        if self._representation_time_stride() <= 1:
            return x_repr
        return self._resample_1d_time_series(x_repr, self._raw_dataset_horizon())

    def _raw_canvas_length(self) -> int:
        return int(self.config.lookback_overlap) + self._raw_dataset_horizon()

    def _strip_overlap_and_upsample_repr(self, future_norm: torch.Tensor) -> torch.Tensor:
        """Window-norm decode path: drop overlap prefix then upsample forecast to raw hz."""
        k = self._overlap_repr_cols()
        if k > 0:
            future_norm = future_norm[..., k:]
        return self._upsample_repr_to_raw_horizon(future_norm)

    def _get_guidance_forecast_norm(
        self,
        past: torch.Tensor,
        past_norm: torch.Tensor,
        stats: Tuple[torch.Tensor, torch.Tensor],
        horizon_width: int,
    ) -> torch.Tensor:
        """Build a window-normalized 1D series for the 2D guidance ghost channel.

        Output shape is (B, V, horizon_width): overlap prefix from past_norm plus a core
        forecast whose length is horizon_width - lookback_overlap, matching future_norm.
        iTransformer instance norm is disabled; rollout stays in diffusion window-norm space.
        """
        if self.guidance_model is None:
            raise ValueError("guidance model is None but guidance channel requested")
        K = int(self.config.lookback_overlap)
        core_len = int(horizon_width) - K
        if core_len <= 0:
            raise ValueError(
                f"horizon_width={horizon_width} must exceed lookback_overlap={K}"
            )

        if self.config.zero_guidance_forecast:
            core_norm = torch.zeros(
                past.shape[0],
                self.config.num_variables,
                core_len,
                device=past.device,
                dtype=past.dtype,
            )
            if K > 0:
                out = torch.cat([torch.zeros_like(past_norm[..., -K:]), core_norm], dim=-1)
            else:
                out = core_norm
        else:
            if not hasattr(self.guidance_model, "get_forecast_window_norm"):
                raise RuntimeError(
                    "guidance model must implement get_forecast_window_norm() "
                    "when use_guidance_channel is enabled"
                )
            with torch.no_grad():
                core_norm = self.guidance_model.get_forecast_window_norm(
                    past_norm, core_len, overlap=K,
                )
            if core_norm.shape[-1] != core_len:
                core_norm = self._resample_1d_time_series(core_norm, core_len)
            if K > 0:
                out = torch.cat([past_norm[..., -K:], core_norm], dim=-1)
            else:
                out = core_norm

        if out.shape[-1] != horizon_width:
            raise RuntimeError(
                f"guidance width {out.shape[-1]} != horizon canvas width {horizon_width}"
            )
        return out

    def _flatten_ctx_for_factorized_dit(
        self,
        ctx: Optional[torch.Tensor],
        B: int,
        V: int,
    ) -> Optional[torch.Tensor]:
        if ctx is None:
            return None
        return ctx.unsqueeze(1).expand(-1, V, -1, -1).reshape(B * V, ctx.shape[1], -1)

    def _get_cross_variate_context(
        self,
        past_raw: torch.Tensor,
        past_norm: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Produce (B, M, ctx_dim) context tokens for DiT bottleneck cross-attention."""
        if self.guidance_model is None or not hasattr(self.guidance_model, "get_encoder_tokens"):
            raise RuntimeError(
                "Cross-attention requires a guidance model with get_encoder_tokens()."
            )
        self._ctx_token_variate_ids = None
        if getattr(self.config, "guidance_type", "itransformer") == "patch_decoder":
            if past_norm is None:
                past_norm, _, _ = self._normalize_sequence(past_raw, None)
            enc_tokens = self.guidance_model.get_encoder_tokens(past_norm)
            self._ctx_token_variate_ids = getattr(
                self.guidance_model, "token_variate_ids", None
            )
            return enc_tokens
        enc_tokens = self.guidance_model.get_encoder_tokens(past_raw)
        if self.context_encoder is None:
            raise RuntimeError("itransformer guidance requires context_encoder.")
        return self.context_encoder(enc_tokens)

    def _window_norm_center(self, past: torch.Tensor) -> torch.Tensor:
        if self.config.window_norm_center == "last":
            return past[..., -1:]
        if self.config.window_norm_center != "mean":
            raise ValueError(
                f"unknown window_norm_center {self.config.window_norm_center!r}"
            )
        return past.mean(dim=-1, keepdim=True)

    def _normalize_sequence(
        self,
        past: torch.Tensor,
        future: Optional[torch.Tensor] = None,
        *,
        apply_ood_shift: Optional[bool] = None,
        data_is_ranked: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Optional[OrdinalLadder]]]:
        """Prepare sequences for diffusion: ordinal-only or legacy window norm."""
        if apply_ood_shift is None:
            apply_ood_shift = bool(getattr(self, "_ordinal_apply_ood_shift", False))
        if data_is_ranked is None:
            data_is_ranked = bool(getattr(self, "_ordinal_input_is_ranked", False))
        if self.config.use_ordinal_window_norm:
            ladder = self.config.ordinal_ladder
            if ladder is None:
                raise ValueError("ordinal_ladder is required when use_ordinal_window_norm=True")
            if data_is_ranked:
                batch_size = past.shape[0] if past.dim() == 3 else 1
                ladder_b = ladder.expand_batch(batch_size)
                center = torch.zeros_like(past[..., :1])
                std = torch.ones_like(past[..., :1])
                return past, future, (center, std, ladder_b)
            past_ord, future_ord, ladder_b, ood_shift = ordinal_encode(
                past,
                future,
                ladder=ladder,
                apply_ood_shift=apply_ood_shift,
            )
            center = torch.zeros_like(past[..., :1])
            std = torch.ones_like(past[..., :1])
            return past_ord, future_ord, (center, std, ladder_b, ood_shift)

        if not self.config.use_window_normalization:
            mean = torch.zeros_like(past[..., :1])
            std = torch.ones_like(past[..., :1])
            return past, future, (mean, std, None)
        center = self._window_norm_center(past)
        past_std = past.std(dim=-1, keepdim=True)
        threshold = float(self.config.window_norm_low_var_threshold)
        if threshold > 0.0:
            std_floor = past_std.clamp_min(self.config.window_norm_std_floor)
            per_v = self.config.window_norm_low_var_unit_std_per_variate
            default_unit = float(self.config.window_norm_low_var_unit_std)
            if per_v is not None:
                if len(per_v) != past.shape[1]:
                    raise ValueError(
                        "window_norm_low_var_unit_std_per_variate length "
                        f"{len(per_v)} != num_variables {past.shape[1]}"
                    )
                unit = torch.tensor(
                    per_v, device=past.device, dtype=past.dtype,
                ).view(1, -1, 1).expand_as(past_std)
            else:
                unit = torch.full_like(past_std, default_unit)
            low_var = past_std < threshold
            flat = past_std <= self.config.window_norm_std_floor
            std = torch.where(flat | low_var, unit, std_floor)
        else:
            std = past_std.clamp_min(self.config.window_norm_std_floor)
        past_norm = (past - center) / std
        if future is not None:
            future_norm = (future - center) / std
        else:
            future_norm = None
        return past_norm, future_norm, (center, std, None)
    
    def _denormalize(
        self,
        x: torch.Tensor,
        stats: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Denormalize using stored statistics."""
        mean, std = stats
        return x * std + mean

    def _denormalize_future(
        self,
        future_norm: torch.Tensor,
        past: torch.Tensor,
        stats: Tuple[torch.Tensor, torch.Tensor, Optional[OrdinalLadder], ...],
    ) -> torch.Tensor:
        """Map decoded future back to global z-score (ordinal inverse or window denorm)."""
        mean, std, ladder, *rest = stats
        ood_shift = rest[0] if rest else None
        if ladder is not None:
            _, future_val = ordinal_decode(
                torch.zeros_like(past[..., :1]).expand_as(past),
                future_norm,
                ladder,
                ood_shift=ood_shift,
            )
            future_norm = future_val
            mean = torch.zeros_like(mean)
            std = torch.ones_like(std)

        K_raw = int(self.config.lookback_overlap)
        K_repr = self._overlap_repr_cols()
        center_shift = (
            ladder is None
            and K_raw > 0
            and getattr(self.config, "lookback_overlap_center_shift", False)
            and future_norm.shape[-1] >= max(K_repr, 1)
        )
        if center_shift:
            overlap_repr = future_norm[..., :K_repr]
            if self._representation_time_stride() > 1 and K_repr > 0:
                overlap_norm = self._resample_1d_time_series(overlap_repr, K_raw)
            else:
                overlap_norm = future_norm[..., :K_raw]
            past_tail = past[..., -K_raw:]
            overlap_raw = overlap_norm * std + mean
            shift = (past_tail - overlap_raw).mean(dim=-1, keepdim=True)
            future = future_norm * std + mean + shift
        else:
            future = self._denormalize(future_norm, (mean, std))

        if self._representation_time_stride() > 1:
            future = self._resample_1d_time_series(future, self._raw_canvas_length())
        if K_raw > 0:
            future = future[..., K_raw:]
        elif self._representation_time_stride() > 1:
            future = self._upsample_repr_to_raw_horizon(future)
        return future
    
    def _predict_noise_chunked(
        self,
        canvas: torch.Tensor,
        t_flat: torch.Tensor,
        cond_for_unet: Optional[torch.Tensor],
        ctx_flat: Optional[torch.Tensor],
        scale_indices: Optional[torch.Tensor] = None,
        variate_indices: Optional[torch.Tensor] = None,
        token_variate_ids: Optional[torch.Tensor] = None,
        return_cross_attn_weights: bool = False,
    ) -> torch.Tensor:
        """Run the denoiser with the same chunking rule used by training/eval."""
        chunk_size = self.config.unet_max_chunk_size
        n_items = canvas.shape[0]
        if chunk_size > 0 and n_items > chunk_size:
            outs = []
            for i in range(0, n_items, chunk_size):
                end = min(i + chunk_size, n_items)
                c_canvas = canvas[i:end]
                c_t = t_flat[i:end] if t_flat.shape[0] == n_items else t_flat
                c_cond = cond_for_unet[i:end] if cond_for_unet is not None else None
                c_ctx = ctx_flat[i:end] if ctx_flat is not None else None
                c_scale = scale_indices[i:end] if scale_indices is not None else None
                c_var = variate_indices[i:end] if variate_indices is not None else None
                kwargs = {
                    "encoder_hidden_states": c_ctx,
                    "token_variate_ids": token_variate_ids,
                    "return_cross_attn_weights": return_cross_attn_weights and i == 0,
                }
                if c_scale is not None:
                    kwargs["scale_indices"] = c_scale
                if c_var is not None:
                    kwargs["variate_indices"] = c_var
                outs.append(self.noise_predictor(c_canvas, c_t, c_cond, **kwargs))
            return torch.cat(outs, dim=0)
        kwargs = {
            "encoder_hidden_states": ctx_flat,
            "token_variate_ids": token_variate_ids,
            "return_cross_attn_weights": return_cross_attn_weights,
        }
        if scale_indices is not None:
            kwargs["scale_indices"] = scale_indices
        if variate_indices is not None:
            kwargs["variate_indices"] = variate_indices
        return self.noise_predictor(canvas, t_flat, cond_for_unet, **kwargs)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Filter out legacy decode_smoothing_kernel from old checkpoints."""
        key = prefix + "decode_smoothing_kernel"
        if key in state_dict:
            del state_dict[key]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def encode_to_2d_binary(self, x: torch.Tensor) -> torch.Tensor:
        """Encode 1D series to a hard binary CDF image without blur."""
        return self.to_2d(self._subsample_repr_time(x))

    def _staged_image_heights(self) -> Tuple[int, int, int]:
        return (
            int(getattr(self.config, "coarse_image_height", self.config.image_height)),
            int(getattr(self.config, "fine_image_height", self.config.image_height)),
            int(getattr(self.config, "finer_image_height", self.config.image_height)),
        )

    def _uses_global_ordinal_encoding(self) -> bool:
        return bool(
            self.config.use_ordinal_window_norm
            and self.config.ordinal_ladder is not None
        )

    def _ordinal_rank_max_tensor(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        ladder = self.config.ordinal_ladder
        if ladder is None:
            raise ValueError("ordinal_ladder is required for global ordinal encoding")
        return ladder.rank_max_per_variate().reshape(-1).to(device=device, dtype=dtype)

    def _staged_fine_residual_value_range(self) -> float:
        if self._uses_global_ordinal_encoding():
            coarse_h = int(getattr(self.config, "coarse_image_height", self.config.image_height))
            vmax = float(self._ordinal_rank_max_tensor(self.to_2d.bin_centers.device).max().item())
            span = max(vmax, 0.0)
            return span / float(coarse_h) * 0.5 if span > 0.0 else 0.0
        coarse_h = int(getattr(self.config, "coarse_image_height", self.config.image_height))
        return float(self.config.max_scale) / float(coarse_h)

    def _decode_staged_combined_1d(
        self,
        coarse_map: torch.Tensor,
        fine_map: torch.Tensor,
        *,
        cdf_decoder: str = "mean",
        expectation_sharpen_temp: Optional[float] = None,
    ) -> torch.Tensor:
        if self._uses_global_ordinal_encoding():
            vmax = self._ordinal_rank_max_tensor(coarse_map.device, dtype=coarse_map.dtype)
            return self.to_2d.decode_dual_heights_bounded(
                coarse_map,
                fine_map,
                value_min=0.0,
                value_max_per_variate=vmax,
                cdf_decoder=cdf_decoder,
                expectation_sharpen_temp=expectation_sharpen_temp,
                squeeze_univariate=False,
            )
        return self.to_2d.decode_dual(
            coarse_map,
            fine_map,
            cdf_decoder=cdf_decoder,
            expectation_sharpen_temp=expectation_sharpen_temp,
            squeeze_univariate=False,
        )

    def _decode_coarse_1d_from_map(
        self,
        coarse_map: torch.Tensor,
        *,
        cdf_decoder: str = "mean",
        expectation_sharpen_temp: Optional[float] = None,
    ) -> torch.Tensor:
        if self._uses_global_ordinal_encoding():
            vmax = self._ordinal_rank_max_tensor(coarse_map.device, dtype=coarse_map.dtype)
            vals = []
            for vi in range(coarse_map.shape[1]):
                span = float(vmax[vi].item())
                vals.append(
                    self.to_2d._decode_occupancy_bounded(
                        coarse_map[:, vi : vi + 1],
                        value_min=0.0,
                        value_max=span,
                        cdf_decoder=cdf_decoder,
                        expectation_sharpen_temp=expectation_sharpen_temp,
                    )
                )
            out = torch.cat(vals, dim=1)
            if coarse_map.shape[1] == 1:
                return out.squeeze(1)
            return out
        return self.to_2d._decode_occupancy_in_range(
            coarse_map,
            value_range=self.config.max_scale,
            cdf_decoder=cdf_decoder,
            expectation_sharpen_temp=expectation_sharpen_temp,
        )

    def _decode_fine_1d_from_map(
        self,
        fine_map: torch.Tensor,
        *,
        coarse_height: int,
        cdf_decoder: str = "mean",
        expectation_sharpen_temp: Optional[float] = None,
    ) -> torch.Tensor:
        fine_range = self._staged_fine_residual_value_range()
        if self._uses_global_ordinal_encoding():
            vals = []
            for vi in range(fine_map.shape[1]):
                vals.append(
                    self.to_2d._decode_occupancy_bounded(
                        fine_map[:, vi : vi + 1],
                        value_min=-fine_range,
                        value_max=fine_range,
                        cdf_decoder=cdf_decoder,
                        expectation_sharpen_temp=expectation_sharpen_temp,
                    )
                )
            out = torch.cat(vals, dim=1)
            if fine_map.shape[1] == 1:
                return out.squeeze(1)
            return out
        return self.to_2d._decode_occupancy_in_range(
            fine_map,
            value_range=fine_range,
            cdf_decoder=cdf_decoder,
            expectation_sharpen_temp=expectation_sharpen_temp,
        )

    def _encode_staged_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self._subsample_repr_time(x)
        coarse_h, fine_h, finer_h = self._staged_image_heights()
        if self._uses_global_ordinal_encoding():
            vmax = self._ordinal_rank_max_tensor(x.device, dtype=x.dtype)
            coarse, fine = self.to_2d.encode_dual_heights_bounded(
                x,
                coarse_height=coarse_h,
                fine_height=fine_h,
                value_min=0.0,
                value_max_per_variate=vmax,
            )
            return {"coarse": coarse, "fine": fine}
        if getattr(self.config, "use_triple_scale", False):
            coarse, fine, finer = self.to_2d.encode_triple_heights(
                x,
                coarse_height=coarse_h,
                fine_height=fine_h,
                finer_height=finer_h,
            )
            return {"coarse": coarse, "fine": fine, "finer": finer}
        coarse, fine = self.to_2d.encode_dual_heights(
            x,
            coarse_height=coarse_h,
            fine_height=fine_h,
        )
        return {"coarse": coarse, "fine": fine}








    def _binary_anchor_canvas_like(self, like: torch.Tensor) -> torch.Tensor:
        """Bernoulli(0.5) stationary anchor: flat 0.5 mean or random bits."""
        if self.config.binary_anchor_input_mode == "random_bits":
            return torch.bernoulli(torch.full_like(like, 0.5))
        return torch.full_like(like, 0.5)

    def _binary_anchor_canvas_shape(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        template = torch.empty(shape, device=device, dtype=dtype)
        return self._binary_anchor_canvas_like(template)






    def _resize_cdf_height(self, image: torch.Tensor, target_height: int) -> torch.Tensor:
        if image.shape[2] == target_height:
            return image
        flat = image.reshape(-1, 1, image.shape[2], image.shape[3])
        resized = F.interpolate(flat, size=(target_height, image.shape[3]), mode="bilinear", align_corners=False)
        return resized.reshape(image.shape[0], image.shape[1], target_height, image.shape[3])

    def _coarse_cdf_to_height(self, coarse_map: torch.Tensor, target_height: int) -> torch.Tensor:
        if coarse_map.shape[2] == target_height:
            return coarse_map
        coarse_value = self.to_2d._decode_occupancy_in_range(
            coarse_map,
            value_range=self.config.max_scale,
            cdf_decoder="mean",
        )
        return self.to_2d._encode_values_in_range(
            coarse_value,
            value_range=self.config.max_scale,
            height=target_height,
        )

    def decode_dual_from_2d(
        self,
        coarse_map: torch.Tensor,
        fine_map: torch.Tensor,
        from_diffusion: bool = False,
        decoder_method: str = "mean",
        past_seed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode dual-scale CDF maps to normalized 1D values."""
        if from_diffusion:
            coarse_map = (coarse_map + 1.0) / 2.0
            fine_map = (fine_map + 1.0) / 2.0
        cdf_decoder = "pdf_expectation" if decoder_method == "pdf_expectation" else decoder_method
        temperature = self.config.decode_temperature if cdf_decoder == "pdf_expectation" else None
        if self._uses_global_ordinal_encoding():
            return self._decode_staged_combined_1d(
                coarse_map,
                fine_map,
                cdf_decoder=cdf_decoder,
                expectation_sharpen_temp=temperature,
            )
        return self.to_2d.decode_dual(
            coarse_map,
            fine_map,
            cdf_decoder=cdf_decoder,
            expectation_sharpen_temp=temperature,
            squeeze_univariate=(coarse_map.shape[1] == 1),
        )

    def decode_triple_from_2d(
        self,
        coarse_map: torch.Tensor,
        fine_map: torch.Tensor,
        finer_map: torch.Tensor,
        from_diffusion: bool = False,
        decoder_method: str = "mean",
    ) -> torch.Tensor:
        """Decode triple-scale CDF maps to normalized 1D values."""
        if from_diffusion:
            coarse_map = (coarse_map + 1.0) / 2.0
            fine_map = (fine_map + 1.0) / 2.0
            finer_map = (finer_map + 1.0) / 2.0
        cdf_decoder = "pdf_expectation" if decoder_method == "pdf_expectation" else decoder_method
        temperature = self.config.decode_temperature if cdf_decoder == "pdf_expectation" else None
        return self.to_2d.decode_triple(
            coarse_map,
            fine_map,
            finer_map,
            cdf_decoder=cdf_decoder,
            expectation_sharpen_temp=temperature,
            squeeze_univariate=(coarse_map.shape[1] == 1),
        )
    
    def decode_from_2d(
        self,
        image: torch.Tensor,
        from_diffusion: bool = True,
        decoder_method: str = "mean",
        beam_width: int = 5,
        jump_penalty_scale: float = 1.0,
        search_radius: int = 10
    ) -> torch.Tensor:
        """Decode 2D representation to 1D time series."""
        batch_size, num_vars, height, seq_len = image.shape
        squeeze_output = (num_vars == 1)
        
        if self.config.representation_mode == "pdf":
            temperature = self.config.decode_temperature if from_diffusion else None
            x = self.to_2d.inverse(
                image,
                cdf_decoder="expectation",
                expectation_sharpen_temp=temperature,
                squeeze_univariate=squeeze_output,
            )
            return x
        else:
            if from_diffusion:
                cdf_map = (image + 1.0) / 2.0
            else:
                cdf_map = image
            cdf_map = torch.clamp(cdf_map, min=0.0, max=1.0)
            if decoder_method in ("mean", "pdf_expectation"):
                temperature = getattr(self.config, "decode_temperature", None) if decoder_method == "pdf_expectation" else None
                cdf_decoder = "expectation" if decoder_method == "pdf_expectation" else "mean"
                x = self.to_2d.inverse(
                    cdf_map,
                    cdf_decoder=cdf_decoder,
                    expectation_sharpen_temp=temperature,
                    squeeze_univariate=squeeze_output,
                )
                return x
            if num_vars > 1:
                raise NotImplementedError(f"decoder_method='{decoder_method}' not yet supported for multivariate.")
            cdf_map_squeezed = cdf_map.squeeze(1)
            centers = self.to_2d.bin_centers.view(1, -1, 1).to(cdf_map_squeezed.device)
            if decoder_method == "median":
                below_half_mask = cdf_map_squeezed < 0.5
                has_below = below_half_mask.any(dim=1)
                first_below = below_half_mask.float().argmax(dim=1)
                median_idx = (first_below - 1).clamp(min=0)
                median_idx = torch.where(has_below, median_idx, torch.full_like(median_idx, self.config.image_height - 1))
                all_below = (first_below == 0) & has_below
                median_idx = torch.where(all_below, torch.zeros_like(median_idx), median_idx)
                x = torch.gather(centers.expand(cdf_map_squeezed.shape[0], -1, cdf_map_squeezed.shape[2]), 1, median_idx.unsqueeze(1)).squeeze(1)
            elif decoder_method == "mode":
                drop = -torch.diff(cdf_map_squeezed, dim=1, prepend=cdf_map_squeezed[:, :1, :])
                drop = torch.relu(drop)
                peak_idx = drop.argmax(dim=1)
                x = torch.gather(centers.expand(cdf_map_squeezed.shape[0], -1, cdf_map_squeezed.shape[2]), 1, peak_idx.unsqueeze(1)).squeeze(1)
            elif decoder_method == "beam":
                x = beam_search_decoder(
                    cdf_map_squeezed,
                    bin_centers=self.to_2d.bin_centers.to(cdf_map_squeezed.device),
                    beam_width=beam_width,
                    jump_penalty_scale=jump_penalty_scale,
                    search_radius=search_radius
                )
            else:
                raise ValueError(f"Unknown decoder_method '{decoder_method}'")
        return x
    
    def _apply_coarse_dropout(self, image: torch.Tensor) -> torch.Tensor:
        """Randomly zero rectangular regions to encourage continuity learning."""
        if not self.training or self.config.cutout_prob <= 0:
            return image
        if torch.rand(1, device=image.device).item() >= self.config.cutout_prob:
            return image
        b, c, h, w = image.shape
        num_masks = torch.randint(
            self.config.cutout_min_masks,
            self.config.cutout_max_masks + 1,
            (1,),
            device=image.device
        ).item()
        for _ in range(num_masks):
            shape_idx = torch.randint(0, len(self.config.cutout_shapes), (1,), device=image.device).item()
            mask_h, mask_w = self.config.cutout_shapes[shape_idx]
            mask_h = min(mask_h, h)
            mask_w = min(mask_w, w)
            if mask_h <= 0 or mask_w <= 0: continue
            top_max = max(1, h - mask_h + 1)
            left_max = max(1, w - mask_w + 1)
            top = torch.randint(0, top_max, (1,), device=image.device).item()
            left = torch.randint(0, left_max, (1,), device=image.device).item()
            image[:, :, top:top + mask_h, left:left + mask_w] = -1.0
        return image

    def forward(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass (binary factorized DiT path)."""
        if self.config.diffusion_stage in {
            "coarse", "fine", "finer", "vertical_dual", "channel_dual",
        }:
            return self._forward_binary_staged(past, future, t)
        return self._forward_binary_factorized(past, future, t)

    @torch.no_grad()
    def generate(
        self,
        past: torch.Tensor,
        use_ddim: bool = True,
        num_ddim_steps: int = 50,
        eta: float = 0.0,
        verbose: bool = False,
        decoder_method: str = "mean",
        beam_width: int = 5,
        jump_penalty_scale: float = 1.0,
        search_radius: int = 10,
        sampler: str = "ddim",
        num_inference_steps: Optional[int] = None,
        yield_intermediates: bool = False,
        reverse_step_indices: Optional[torch.Tensor] = None,
        snapshot_timesteps: Optional[Tuple[int, ...]] = None,
        future_coarse_2d: Optional[torch.Tensor] = None,
        future_fine_2d: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate future predictions via binary reverse sampling.

        sampler: 'ddim' (default), 'anchor' / 'deterministic_anchor' for one-shot anchor decode.
        num_inference_steps overrides binary_sample_steps when set.
        """
        steps = num_inference_steps if num_inference_steps is not None else self.config.binary_sample_steps
        gen_common = dict(
            num_steps=steps,
            verbose=verbose,
            decoder_method=decoder_method,
            beam_width=beam_width,
            jump_penalty_scale=jump_penalty_scale,
            search_radius=search_radius,
            sampler=sampler,
            yield_intermediates=yield_intermediates,
            reverse_step_indices=reverse_step_indices,
            snapshot_timesteps=snapshot_timesteps,
            future_coarse_2d=future_coarse_2d,
            future_fine_2d=future_fine_2d,
        )
        if self.config.diffusion_stage in {
            "coarse", "fine", "finer", "vertical_dual", "channel_dual",
        }:
            return self._generate_binary_staged(past, **gen_common)
        return self._generate_binary_factorized(past, **gen_common)


    def _cdf_distance_weight_tensor(
        self,
        target: torch.Tensor,
        *,
        weight_source: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Distance weights matching target map shape (BV,1,H,W) or (B,V,H,W).

        Always derive |r−k| from a clean CDF map (weight_source or target). Never
        from a random flip mask — that would scramble the distance geometry.
        """
        alpha = float(getattr(self.config, "binary_cdf_distance_alpha", 1.0))
        coarse_h = None
        per_ch = False
        if self.config.diffusion_stage == "vertical_dual":
            coarse_h = int(self.config.coarse_image_height)
        elif self.config.diffusion_stage == "channel_dual":
            per_ch = True
        src = target if weight_source is None else weight_source
        return self.to_2d.cdf_distance_weights(
            src, alpha, coarse_height=coarse_h, per_occupancy_channel=per_ch,
        )

    def _binary_plain_bce_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        *,
        weight_source: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Unweighted BCE for binary CDF images (optional distance weights)."""
        per_elem = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
        if self.config.binary_use_boundary_weighted_bce:
            per_elem = per_elem * self._cdf_distance_weight_tensor(
                target, weight_source=weight_source,
            )
        return per_elem.mean()

    def _binary_weighted_bce_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        t_flat: Optional[torch.Tensor] = None,
        *,
        weight_source: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """BCE with optional CDF-distance + min-SNR timestep weighting."""
        per_elem = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
        if self.config.binary_use_boundary_weighted_bce:
            per_elem = per_elem * self._cdf_distance_weight_tensor(
                target, weight_source=weight_source,
            )
        if t_flat is None or self.config.loss_weighting == "none":
            return per_elem.mean()
        beta_t = self.binary_scheduler.betas[t_flat].clamp(1e-5, 1.0 - 1e-5)
        snr = ((1.0 - beta_t) ** 2) / (beta_t ** 2)
        weight = torch.minimum(snr, torch.full_like(snr, self.config.min_snr_gamma)) / snr
        view_shape = (-1,) + (1,) * (per_elem.dim() - 1)
        return (per_elem * weight.view(view_shape)).mean()

    def _soft_decode_vertical_dual_1d(
        self,
        soft_canvas: torch.Tensor,
        *,
        B: int,
        V: int,
    ) -> torch.Tensor:
        """Soft (pre-threshold) stacked canvas → normalized 1D via decode_dual."""
        Hc = int(self.config.coarse_image_height)
        # soft_canvas: (BV, 1, H, W) or (B, V, H, W)
        if soft_canvas.dim() == 4 and soft_canvas.shape[1] == 1 and soft_canvas.shape[0] == B * V:
            canvas = soft_canvas.reshape(B, V, soft_canvas.shape[2], soft_canvas.shape[3])
        else:
            canvas = soft_canvas
        coarse, fine = self.to_2d.split_vertical_dual(canvas, Hc)
        return self._decode_staged_combined_1d(coarse, fine, cdf_decoder="mean")

    def _soft_decode_channel_dual_1d(
        self,
        soft_canvas: torch.Tensor,
        *,
        B: int,
        V: int,
    ) -> torch.Tensor:
        """Soft (BV, 2, H, W) channel-stacked canvas → normalized 1D."""
        if soft_canvas.dim() != 4 or soft_canvas.shape[1] != 2:
            raise ValueError(
                f"channel_dual soft canvas expected (BV, 2, H, W), got {tuple(soft_canvas.shape)}"
            )
        coarse, fine = self.to_2d.split_channel_dual_flat(soft_canvas, B=B, V=V)
        return self._decode_staged_combined_1d(coarse, fine, cdf_decoder="mean")

    def _occupancy_channels(self) -> int:
        return int(getattr(self.config, "data_occupancy_channels", 1))

    def _split_binary_heads(self, out_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split DiT binary outputs into x0 and zt heads (C occupancy channels each)."""
        c = self._occupancy_channels()
        if out_flat.shape[1] < 2 * c:
            raise ValueError(
                f"binary head expected >= {2 * c} channels, got {out_flat.shape[1]}"
            )
        return out_flat[:, :c], out_flat[:, c : 2 * c]
    def _x0_logits_from_prediction(
        self,
        primary_logits: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.prediction_target == "epsilon":
            return torch.where(xt > 0.5, -primary_logits, primary_logits)
        return primary_logits

    def _flat_variate_indices(self, bv: int, num_variates: int, device: torch.device) -> torch.Tensor:
        if bv % num_variates != 0:
            raise ValueError(f"bv={bv} not divisible by num_variates={num_variates}")
        batch_size = bv // num_variates
        return torch.arange(num_variates, device=device).unsqueeze(0).expand(batch_size, -1).reshape(bv)

    def _past_cond_resize_to_horizon(self) -> bool:
        return bool(getattr(self.config, "past_cond_resize_to_horizon", True))

    def _past_cond_tail_len(self, past_len: int, target_width: int) -> int:
        cap = int(self.config.diffusion_lookback_cap or 0)
        if cap > 0:
            return min(past_len, cap)
        if not self._past_cond_resize_to_horizon():
            return past_len
        return min(past_len, target_width)

    def _resize_past_cond_to_width(self, cond: torch.Tensor, width: int) -> torch.Tensor:
        H = cond.shape[-2]
        cur = cond.shape[-1]
        if cur == width:
            return cond
        if self._past_cond_resize_to_horizon():
            return F.interpolate(
                cond, size=(H, width), mode="bilinear", align_corners=False,
            )
        if cur > width:
            raise ValueError(
                f"past visual cond width {cur} exceeds target {width} "
                "with past_cond_resize_to_horizon=False"
            )
        return F.pad(cond, (0, width - cur))

    def _cat_past_and_horizon_cond(
        self,
        past_cond: torch.Tensor,
        horizon_cond: torch.Tensor,
    ) -> torch.Tensor:
        past_cond = self._resize_past_cond_to_width(past_cond, horizon_cond.shape[-1])
        return torch.cat((past_cond, horizon_cond), dim=1)

    def _chunk_horizon(self) -> int:
        chunk = int(self.config.diffusion_chunk_horizon or 0)
        if chunk > 0:
            return chunk
        return max(1, int(self.config.dataset_forecast_length or 0) or (self.config.forecast_length - self.config.lookback_overlap))

    def _ar_stride(self) -> int:
        return self._chunk_horizon() - int(self.config.lookback_overlap)

    def _ar_num_chunks(self, dataset_horizon: int) -> int:
        K = int(self.config.lookback_overlap)
        C = self._chunk_horizon()
        if dataset_horizon <= C:
            return 1
        return int(math.ceil((dataset_horizon - K) / max(1, self._ar_stride())))

    def _ar_training_enabled(self, future_len: int) -> bool:
        chunk = int(self.config.diffusion_chunk_horizon or 0)
        if chunk <= 0:
            return False
        dataset_h = future_len - int(self.config.lookback_overlap)
        return dataset_h > chunk

    def _sample_ar_training_chunk(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pick one random AR chunk per batch row (teacher-forced history)."""
        K = int(self.config.lookback_overlap)
        C = self._chunk_horizon()
        dataset_h = future.shape[-1] - K
        n_chunks = self._ar_num_chunks(dataset_h)
        if n_chunks <= 1:
            return past, future

        device = past.device
        B = past.shape[0]
        past_out = []
        future_out = []
        for b in range(B):
            c = int(torch.randint(0, n_chunks, (1,), device=device).item())
            offset = c * self._ar_stride()
            end = min(offset + K + C, future.shape[-1])
            fut_b = future[b : b + 1, ..., offset:end]
            if c == 0:
                past_b = past[b : b + 1]
            else:
                hist = future[b : b + 1, ..., K : K + offset]
                past_b = torch.cat([past[b : b + 1], hist], dim=-1)
            past_out.append(past_b)
            future_out.append(fut_b)
        max_past = max(p.shape[-1] for p in past_out)
        max_fut = max(f.shape[-1] for f in future_out)
        past_pad = []
        future_pad = []
        for p, f in zip(past_out, future_out):
            if p.shape[-1] < max_past:
                pad = max_past - p.shape[-1]
                p = torch.cat([p[..., :1].expand(*p.shape[:-1], pad), p], dim=-1)
            if f.shape[-1] < max_fut:
                f = F.pad(f, (0, max_fut - f.shape[-1]))
            past_pad.append(p)
            future_pad.append(f)
        return torch.cat(past_pad, dim=0), torch.cat(future_pad, dim=0)

    def _append_raw_lookback_cond_channel(
        self,
        cond: torch.Tensor,
        past_raw: torch.Tensor,
        past_tail_len: int,
        target_width: int,
    ) -> torch.Tensor:
        if not self.config.use_raw_lookback_cond_channel:
            return cond
        if past_raw is None:
            raise ValueError("past_raw is required when use_raw_lookback_cond_channel=True")
        B, V = past_raw.shape[:2]
        H = self.config.image_height
        BV = B * V
        past_tail_raw = past_raw[..., -past_tail_len:]
        raw_maps = self._encode_staged_maps(past_tail_raw)
        if self.config.diffusion_stage == "vertical_dual":
            stacked = self.to_2d.stack_vertical_dual(raw_maps["coarse"], raw_maps["fine"])
            past_repr_w = stacked.shape[-1]
            raw_cond = stacked.reshape(BV, 1, H, past_repr_w)
        elif self.config.diffusion_stage == "channel_dual":
            raw_cond = self.to_2d.stack_channel_dual_flat(
                raw_maps["coarse"], raw_maps["fine"],
            )
        else:
            if self.config.diffusion_stage == "coarse":
                raw_coarse = self._resize_cdf_height(raw_maps["coarse"], H)
            else:
                raw_coarse = self._coarse_cdf_to_height(raw_maps["coarse"], H)
            past_repr_w = raw_maps["coarse"].shape[-1]
            raw_cond = raw_coarse.reshape(BV, 1, H, past_repr_w)
        if self._past_cond_resize_to_horizon():
            raw_cond = F.interpolate(
                raw_cond,
                size=(H, target_width),
                mode="bilinear",
                align_corners=True,
            )
        elif raw_cond.shape[-1] != cond.shape[-1]:
            raw_cond = self._resize_past_cond_to_width(raw_cond, cond.shape[-1])
        return torch.cat([cond, raw_cond], dim=1)

    def _staged_past_condition(
        self,
        past_norm: torch.Tensor,
        target_width: int,
        past_raw: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Build GT lookback conditioning maps for staged denoisers.

        Uses only the trailing diffusion_lookback_cap (default 96) timesteps for 2D
        past CDFs; iTrans cross-attn / guidance ghost use the full past separately.
        """
        B, V = past_norm.shape[:2]
        H = self.config.image_height
        BV = B * V
        past_tail_len = self._past_cond_tail_len(past_norm.shape[-1], target_width)
        past_tail_norm = past_norm[..., -past_tail_len:]
        past_maps = self._encode_staged_maps(past_tail_norm)
        if self.config.diffusion_stage == "vertical_dual":
            stacked = self.to_2d.stack_vertical_dual(past_maps["coarse"], past_maps["fine"])
            past_repr_w = stacked.shape[-1]
            cond = stacked.reshape(BV, 1, H, past_repr_w)
        elif self.config.diffusion_stage == "channel_dual":
            cond = self.to_2d.stack_channel_dual_flat(
                past_maps["coarse"], past_maps["fine"],
            )
            past_repr_w = past_maps["coarse"].shape[-1]
            if cond.shape[-1] != past_repr_w:
                raise ValueError(
                    f"channel_dual past cond W={cond.shape[-1]} != {past_repr_w}"
                )
        else:
            cond_maps = []
            if self.config.diffusion_stage == "coarse":
                cond_maps.append(self._resize_cdf_height(past_maps["coarse"], H))
            else:
                cond_maps.append(self._coarse_cdf_to_height(past_maps["coarse"], H))
            cond_maps.append(self._resize_cdf_height(past_maps["fine"], H))
            if getattr(self.config, "use_triple_scale", False):
                cond_maps.append(self._resize_cdf_height(past_maps["finer"], H))
            past_repr_w = past_maps["coarse"].shape[-1]
            cond = torch.cat(
                [m.reshape(BV, 1, H, past_repr_w) for m in cond_maps],
                dim=1,
            )
        if self._past_cond_resize_to_horizon():
            cond = F.interpolate(cond, size=(H, target_width), mode='bilinear', align_corners=False)
        cond = self._append_raw_lookback_cond_channel(
            cond, past_raw, past_tail_len, target_width,
        )
        return cond, past_maps


    def _forward_binary_staged(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Train one staged denoiser: coarse, fine, finer, vertical_dual, or channel_dual."""
        assert self.binary_scheduler is not None, "binary scheduler is not initialized"
        stage = self.config.diffusion_stage
        if stage not in {"coarse", "fine", "finer", "vertical_dual", "channel_dual"}:
            raise ValueError(f"_forward_binary_staged called for stage={stage!r}")

        B = past.shape[0]
        V = self.config.num_variables
        H = self.config.image_height
        device = past.device
        BV = B * V
        C_occ = self._occupancy_channels()

        past_norm, future_norm, _stats = self._normalize_sequence(past, future)
        future_maps = self._encode_staged_maps(future_norm)
        if stage == "vertical_dual":
            target_2d = self.to_2d.stack_vertical_dual(future_maps["coarse"], future_maps["fine"])
            W_fut = target_2d.shape[3]
            H = target_2d.shape[2]
            target_flat = target_2d.reshape(BV, 1, H, W_fut)
        elif stage == "channel_dual":
            target_flat = self.to_2d.stack_channel_dual_flat(
                future_maps["coarse"], future_maps["fine"],
            )
            H = target_flat.shape[2]
            W_fut = target_flat.shape[3]
        else:
            target_2d = future_maps[stage]
            W_fut = target_2d.shape[3]
            H = target_2d.shape[2]
            target_flat = target_2d.reshape(BV, 1, H, W_fut)

        if t is None:
            t = torch.randint(0, self.config.binary_num_steps, (B,), device=device)
        t_flat = t.unsqueeze(1).expand(-1, V).reshape(BV)
        variate_indices = None
        if self.config.use_variate_embedding and self.config.variate_factorized and V > 1:
            variate_indices = self._flat_variate_indices(BV, V, device)

        xt_flat, zt_flat = self.binary_scheduler.add_noise(target_flat, t_flat)

        ctx = None if getattr(self.config, 'disable_cross_attention', False) else self._get_cross_variate_context(past, past_norm)
        ctx_flat = self._flatten_ctx_for_factorized_dit(ctx, B, V)

        cond_for_unet, past_maps = self._staged_past_condition(past_norm, W_fut, past_raw=past)
        if stage in {"fine", "finer"}:
            future_coarse_cond = self._coarse_cdf_to_height(future_maps["coarse"], H)
            future_coarse_flat = future_coarse_cond.reshape(BV, 1, H, W_fut)
            cond_for_unet = self._cat_past_and_horizon_cond(cond_for_unet, future_coarse_flat)
        if stage == "finer":
            future_fine_cond = self._resize_cdf_height(future_maps["fine"], H)
            future_fine_flat = future_fine_cond.reshape(BV, 1, H, W_fut)
            cond_for_unet = torch.cat((cond_for_unet, future_fine_flat), dim=1)
        base_cond_for_unet = cond_for_unet

        guidance_flat = None
        if self.config.use_guidance_channel:
            raw_hz_w = int(future_norm.shape[-1])
            guidance_forecast_norm = self._get_guidance_forecast_norm(past, past_norm, _stats, raw_hz_w)
            guidance_maps = self._encode_staged_maps(guidance_forecast_norm)
            if stage == "vertical_dual":
                g_stack = self.to_2d.stack_vertical_dual(guidance_maps["coarse"], guidance_maps["fine"])
                guidance_flat = g_stack.reshape(BV, 1, H, W_fut)
            elif stage == "channel_dual":
                guidance_flat = self.to_2d.stack_channel_dual_flat(
                    guidance_maps["coarse"], guidance_maps["fine"],
                )
            elif stage == "coarse":
                guidance_flat = self._resize_cdf_height(guidance_maps["coarse"], H).reshape(BV, 1, H, W_fut)
            elif stage == "fine":
                guidance_flat = self._resize_cdf_height(guidance_maps["fine"], H).reshape(BV, 1, H, W_fut)
            elif stage == "finer":
                guidance_flat = self._resize_cdf_height(guidance_maps["finer"], H).reshape(BV, 1, H, W_fut)

        canvas = self._inject_coordinate_channel(xt_flat.float())
        canvas = self._inject_time_channels(canvas)

        # Staged visual conditioning is always GT during training. CFG dropout is
        # restricted to context tokens so the fine stage never sees predicted coarse.
        ctx_anchor = ctx_flat
        if self.training and self.config.cfg_dropout > 0.0:
            drop_mask = torch.rand(B, device=device) < self.config.cfg_dropout
            drop_mask_flat = drop_mask.unsqueeze(1).expand(-1, V).reshape(BV)
            if ctx_flat is not None:
                ctx_flat = torch.where(
                    drop_mask_flat.view(BV, 1, 1),
                    torch.zeros_like(ctx_flat),
                    ctx_flat,
                )
            if guidance_flat is not None:
                guidance_for_unet = torch.where(
                    drop_mask_flat.view(BV, 1, 1, 1),
                    torch.zeros_like(guidance_flat),
                    guidance_flat,
                )
                canvas = torch.cat([canvas, guidance_for_unet], dim=1)
        elif guidance_flat is not None:
            canvas = torch.cat([canvas, guidance_flat], dim=1)

        out_flat = self._predict_noise_chunked(
            canvas, t_flat, cond_for_unet, ctx_flat, variate_indices=variate_indices, token_variate_ids=self._ctx_token_variate_ids,
        )
        primary_logits, zt_logits = self._split_binary_heads(out_flat)
        x0_logits = self._x0_logits_from_prediction(primary_logits, xt_flat)
        if self.config.prediction_target == "epsilon":
            loss_x0 = self._binary_weighted_bce_loss(
                primary_logits, zt_flat, t_flat, weight_source=target_flat,
            )
            loss_zt = self._binary_weighted_bce_loss(
                zt_logits, target_flat, t_flat, weight_source=target_flat,
            )
        else:
            loss_x0 = self._binary_weighted_bce_loss(
                primary_logits, target_flat, t_flat, weight_source=target_flat,
            )
            loss_zt = self._binary_weighted_bce_loss(
                zt_logits, zt_flat, t_flat, weight_source=target_flat,
            )
        regular_loss = loss_x0 + loss_zt

        anchor_loss = torch.tensor(0.0, device=device)
        combined_loss = regular_loss
        if self.config.use_deterministic_anchor_loss:
            anchor_t_flat = torch.full(
                (BV,),
                self.config.binary_num_steps - 1,
                device=device,
                dtype=t_flat.dtype,
            )
            neutral_future_flat = self._binary_anchor_canvas_like(target_flat)
            anchor_canvas = self._inject_coordinate_channel(neutral_future_flat)
            anchor_canvas = self._inject_time_channels(anchor_canvas)
            if guidance_flat is not None:
                anchor_canvas = torch.cat([anchor_canvas, guidance_flat], dim=1)
            anchor_out_flat = self._predict_noise_chunked(
                anchor_canvas,
                anchor_t_flat,
                base_cond_for_unet,
                ctx_anchor,
                variate_indices=variate_indices, token_variate_ids=self._ctx_token_variate_ids,
            )
            anchor_primary, _ = self._split_binary_heads(anchor_out_flat)
            anchor_x0_logits = self._x0_logits_from_prediction(anchor_primary, neutral_future_flat)
            anchor_bce = self._binary_plain_bce_loss(
                anchor_x0_logits, target_flat, weight_source=target_flat,
            )
            if stage == "vertical_dual":
                soft_1d = self._soft_decode_vertical_dual_1d(
                    torch.sigmoid(anchor_x0_logits), B=B, V=V,
                )
                future_tgt = self._subsample_repr_time(future_norm)
                if soft_1d.shape[-1] != future_tgt.shape[-1]:
                    raise ValueError(
                        f"soft decode width {soft_1d.shape[-1]} != target {future_tgt.shape[-1]}"
                    )
                anchor_mse = F.mse_loss(soft_1d, future_tgt)
                lam_mse = float(getattr(self.config, "anchor_mse_proxy_lambda", 0.5))
                anchor_loss = lam_mse * anchor_bce + (1.0 - lam_mse) * anchor_mse
            elif stage == "channel_dual":
                soft_1d = self._soft_decode_channel_dual_1d(
                    torch.sigmoid(anchor_x0_logits), B=B, V=V,
                )
                future_tgt = self._subsample_repr_time(future_norm)
                if soft_1d.shape[-1] != future_tgt.shape[-1]:
                    raise ValueError(
                        f"soft decode width {soft_1d.shape[-1]} != target {future_tgt.shape[-1]}"
                    )
                anchor_mse = F.mse_loss(soft_1d, future_tgt)
                lam_mse = float(getattr(self.config, "anchor_mse_proxy_lambda", 0.5))
                anchor_loss = lam_mse * anchor_bce + (1.0 - lam_mse) * anchor_mse
            else:
                anchor_loss = anchor_bce
            lam = self.config.deterministic_anchor_lambda
            combined_loss = lam * regular_loss + (1.0 - lam) * anchor_loss

        if stage == "channel_dual":
            x0_pred = torch.sigmoid(x0_logits)  # (BV, 2, H, W)
            x0_coarse, x0_fine = self.to_2d.split_channel_dual_flat(x0_pred, B=B, V=V)
            result = {
                'loss': combined_loss,
                'noise_loss': regular_loss,
                'combined_mse_loss': combined_loss,
                'anchor_loss': anchor_loss,
                'loss_x0': loss_x0,
                'loss_zt': loss_zt,
                'emd_loss': torch.tensor(0.0, device=device),
                'guidance_loss': torch.tensor(0.0, device=device),
                'noise_pred': x0_coarse,
                'x0_pred': x0_pred,
                'x0_pred_channel_dual': x0_pred,
                'x0_pred_coarse': x0_coarse,
                'x0_pred_fine': x0_fine,
                'future_2d': target_flat.reshape(B, V, C_occ, H, W_fut),
                'future_2d_coarse': future_maps["coarse"],
                'future_2d_fine': future_maps["fine"],
                'past_2d_coarse': past_maps["coarse"],
                'past_2d_fine': past_maps["fine"],
                't': t,
                'diffusion_stage': stage,
            }
            if "finer" in future_maps:
                result['future_2d_finer'] = future_maps["finer"]
                result['past_2d_finer'] = past_maps["finer"]
            return result

        x0_pred = torch.sigmoid(x0_logits).reshape(B, V, H, W_fut)
        result = {
            'loss': combined_loss,
            'noise_loss': regular_loss,
            'combined_mse_loss': combined_loss,
            'anchor_loss': anchor_loss,
            'loss_x0': loss_x0,
            'loss_zt': loss_zt,
            'emd_loss': torch.tensor(0.0, device=device),
            'guidance_loss': torch.tensor(0.0, device=device),
            'noise_pred': x0_pred,
            'x0_pred': x0_pred,
            'future_2d': target_2d,
            'future_2d_coarse': future_maps["coarse"],
            'future_2d_fine': future_maps["fine"],
            'past_2d_coarse': past_maps["coarse"],
            'past_2d_fine': past_maps["fine"],
            't': t,
            'diffusion_stage': stage,
        }
        if "finer" in future_maps:
            result['future_2d_finer'] = future_maps["finer"]
            result['past_2d_finer'] = past_maps["finer"]
        if stage == "coarse":
            result['x0_pred_coarse'] = x0_pred
        elif stage == "fine":
            result['x0_pred_fine'] = x0_pred
        elif stage == "vertical_dual":
            result['x0_pred_vertical_dual'] = x0_pred
            Hc = int(self.config.coarse_image_height)
            result['x0_pred_coarse'] = x0_pred[:, :, :Hc]
            result['x0_pred_fine'] = x0_pred[:, :, Hc:]
        else:
            result['x0_pred_finer'] = x0_pred
        return result

    @torch.no_grad()
    def diagnostic_capture_staged(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        *,
        capture_cross_attn: bool = True,
    ) -> Dict[str, Any]:
        """One diagnostic forward: conditioning tensors + optional cross-attn weights."""
        B = past.shape[0]
        V = self.config.num_variables
        device = past.device
        BV = B * V
        stage = self.config.diffusion_stage

        past_norm, future_norm, norm_stats = self._normalize_sequence(past, future)
        future_maps = self._encode_staged_maps(future_norm)
        if stage == "vertical_dual":
            target_2d = self.to_2d.stack_vertical_dual(future_maps["coarse"], future_maps["fine"])
            W_fut = target_2d.shape[3]
            H = target_2d.shape[2]
            target_flat = target_2d.reshape(BV, 1, H, W_fut)
        elif stage == "channel_dual":
            target_flat = self.to_2d.stack_channel_dual_flat(
                future_maps["coarse"], future_maps["fine"],
            )
            H = target_flat.shape[2]
            W_fut = target_flat.shape[3]
            target_2d = future_maps["coarse"]
        else:
            target_2d = future_maps[stage]
            W_fut = target_2d.shape[3]
            H = target_2d.shape[2]
            target_flat = target_2d.reshape(BV, 1, H, W_fut)

        cond_for_unet, past_maps = self._staged_past_condition(past_norm, W_fut, past_raw=past)
        guidance_norm = self._get_guidance_forecast_norm(
            past, past_norm, norm_stats, int(future_norm.shape[-1]),
        )
        guidance_maps = self._encode_staged_maps(guidance_norm) if guidance_norm is not None else None

        t = torch.zeros(B, device=device, dtype=torch.long)
        t_flat = t.unsqueeze(1).expand(-1, V).reshape(BV)
        variate_indices = None
        if self.config.use_variate_embedding and self.config.variate_factorized and V > 1:
            variate_indices = self._flat_variate_indices(BV, V, device)

        ctx = None if getattr(self.config, "disable_cross_attention", False) else self._get_cross_variate_context(past, past_norm)
        ctx_flat = self._flatten_ctx_for_factorized_dit(ctx, B, V)

        xt_flat, _ = self.binary_scheduler.add_noise(target_flat, t_flat)
        canvas = self._inject_coordinate_channel(xt_flat)
        canvas = self._inject_time_channels(canvas)

        if stage in {"fine", "finer"}:
            future_coarse_cond = self._coarse_cdf_to_height(future_maps["coarse"], H)
            future_coarse_flat = future_coarse_cond.reshape(BV, 1, H, W_fut)
            cond_for_unet = self._cat_past_and_horizon_cond(cond_for_unet, future_coarse_flat)

        if self.config.use_guidance_channel and guidance_maps is not None:
            if stage == "vertical_dual":
                g_stack = self.to_2d.stack_vertical_dual(
                    guidance_maps["coarse"], guidance_maps["fine"],
                )
                guidance_flat = g_stack.reshape(BV, 1, H, W_fut)
            elif stage == "channel_dual":
                guidance_flat = self.to_2d.stack_channel_dual_flat(
                    guidance_maps["coarse"], guidance_maps["fine"],
                )
            else:
                guidance_flat = guidance_maps[stage].reshape(BV, 1, H, W_fut)
            canvas = torch.cat([canvas, guidance_flat], dim=1)

        base_cond = cond_for_unet
        self._predict_noise_chunked(
            canvas, t_flat, base_cond, ctx_flat,
            variate_indices=variate_indices, token_variate_ids=self._ctx_token_variate_ids,
            return_cross_attn_weights=capture_cross_attn,
        )
        cross_attn_weights = getattr(self.noise_predictor, "_diag_cross_attn_weights", None)

        return {
            "past_norm": past_norm,
            "future_norm": future_norm,
            "norm_stats": norm_stats,
            "cond_for_unet": cond_for_unet,
            "past_maps": past_maps,
            "guidance_norm": guidance_norm,
            "guidance_maps": guidance_maps,
            "cross_attn_weights": cross_attn_weights,
            "future_maps": future_maps,
        }



    @torch.no_grad()
    def _generate_binary_staged(
        self,
        past: torch.Tensor,
        num_steps: int = 20,
        verbose: bool = False,
        decoder_method: str = "mean",
        sampler: str = "ddim",
        yield_intermediates: bool = False,
        reverse_step_indices: Optional[torch.Tensor] = None,
        snapshot_timesteps: Optional[Tuple[int, ...]] = None,
        future_coarse_2d: Optional[torch.Tensor] = None,
        future_fine_2d: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Generate one staged output, chaining coarse/fine maps into later stages."""
        assert self.binary_scheduler is not None, "binary scheduler is not initialized"
        stage = self.config.diffusion_stage
        if stage not in {"coarse", "fine", "finer", "vertical_dual", "channel_dual"}:
            raise ValueError(f"_generate_binary_staged called for stage={stage!r}")

        B = past.shape[0]
        V = self.config.num_variables
        H = self.config.image_height
        device = past.device
        BV = B * V
        C_occ = self._occupancy_channels()
        raw_hz_w = int(self.config.forecast_length)
        W_fut = self._repr_forecast_width(raw_hz_w)

        past_norm, _, stats = self._normalize_sequence(past)
        cond_for_unet, past_maps = self._staged_past_condition(past_norm, W_fut, past_raw=past)
        coarse_for_decode = future_coarse_2d
        fine_for_decode = future_fine_2d
        if stage in {"fine", "finer"}:
            if future_coarse_2d is None:
                raise ValueError(f"{stage}-stage generation requires future_coarse_2d from the coarse model.")
            if future_coarse_2d.shape[:2] != (B, V) or future_coarse_2d.shape[3] != W_fut:
                raise ValueError(
                    "future_coarse_2d must have shape "
                    f"(B={B}, V={V}, Hc, W={W_fut}), got {tuple(future_coarse_2d.shape)}"
                )
            future_coarse_cond = self._coarse_cdf_to_height(future_coarse_2d.to(device), H)
            future_coarse_flat = future_coarse_cond.reshape(BV, 1, H, W_fut)
            cond_for_unet = self._cat_past_and_horizon_cond(cond_for_unet, future_coarse_flat)
        if stage == "finer":
            if future_fine_2d is None:
                raise ValueError("finer-stage generation requires future_fine_2d from the fine model.")
            if future_fine_2d.shape[:2] != (B, V) or future_fine_2d.shape[3] != W_fut:
                raise ValueError(
                    "future_fine_2d must have shape "
                    f"(B={B}, V={V}, Hf, W={W_fut}), got {tuple(future_fine_2d.shape)}"
                )
            future_fine_cond = self._resize_cdf_height(future_fine_2d.to(device), H)
            cond_for_unet = torch.cat(
                (cond_for_unet, future_fine_cond.reshape(BV, 1, H, W_fut)),
                dim=1,
            )

        ctx = None if getattr(self.config, 'disable_cross_attention', False) else self._get_cross_variate_context(past, past_norm)
        ctx_flat = self._flatten_ctx_for_factorized_dit(ctx, B, V)
        variate_indices = None
        if self.config.use_variate_embedding and self.config.variate_factorized and V > 1:
            variate_indices = self._flat_variate_indices(BV, V, device)

        guidance_flat = None
        if self.config.use_guidance_channel:
            guidance_forecast_norm = self._get_guidance_forecast_norm(past, past_norm, stats, raw_hz_w)
            guidance_maps = self._encode_staged_maps(guidance_forecast_norm)
            if stage == "vertical_dual":
                g_stack = self.to_2d.stack_vertical_dual(guidance_maps["coarse"], guidance_maps["fine"])
                guidance_flat = g_stack.reshape(BV, 1, H, W_fut)
            elif stage == "channel_dual":
                guidance_flat = self.to_2d.stack_channel_dual_flat(
                    guidance_maps["coarse"], guidance_maps["fine"],
                )
            elif stage == "coarse":
                guidance_flat = self._resize_cdf_height(guidance_maps["coarse"], H).reshape(BV, 1, H, W_fut)
            elif stage == "fine":
                guidance_flat = self._resize_cdf_height(guidance_maps["fine"], H).reshape(BV, 1, H, W_fut)
            elif stage == "finer":
                guidance_flat = self._resize_cdf_height(guidance_maps["finer"], H).reshape(BV, 1, H, W_fut)

        def _build_canvas(xt: torch.Tensor) -> torch.Tensor:
            canvas = self._inject_coordinate_channel(xt)
            canvas = self._inject_time_channels(canvas)
            if guidance_flat is not None:
                canvas = torch.cat([canvas, guidance_flat], dim=1)
            return canvas

        def _chunked_model_fn(xt: torch.Tensor, t_batch: torch.Tensor):
            out = self._predict_noise_chunked(
                _build_canvas(xt), t_batch, cond_for_unet, ctx_flat,
                variate_indices=variate_indices, token_variate_ids=self._ctx_token_variate_ids,
            )
            primary, zt = self._split_binary_heads(out)
            x0_logits = self._x0_logits_from_prediction(primary, xt)
            return x0_logits, zt

        intermediates = None
        sample_shape = (BV, C_occ, H, W_fut)
        if sampler in ("anchor", "deterministic_anchor"):
            t_batch = torch.full(
                (BV,),
                self.config.binary_num_steps - 1,
                device=device,
                dtype=torch.long,
            )
            neutral_future_flat = self._binary_anchor_canvas_shape(
                sample_shape, device=device,
            )
            x0_logits, _zt_logits = _chunked_model_fn(neutral_future_flat, t_batch)
            future_2d_flat = (torch.sigmoid(x0_logits) > 0.5).float()
            if yield_intermediates:
                intermediates = [(999, neutral_future_flat.clone()), (0, future_2d_flat.clone())]
        else:
            sample_kwargs = dict(
                model_fn=_chunked_model_fn,
                shape=sample_shape,
                num_steps=num_steps,
                device=device,
                verbose=verbose,
                sampler=sampler,
                reverse_step_indices=reverse_step_indices,
                snapshot_timesteps=snapshot_timesteps,
            )
            if yield_intermediates:
                future_2d_flat, intermediates = self.binary_scheduler.sample(
                    yield_intermediates=True,
                    **sample_kwargs,
                )
            else:
                future_2d_flat = self.binary_scheduler.sample(**sample_kwargs)

        if stage == "channel_dual":
            future_2d_coarse, future_2d_fine = self.to_2d.split_channel_dual_flat(
                future_2d_flat, B=B, V=V,
            )
            future_2d_finer = None
            cdf_decoder = "pdf_expectation" if decoder_method == "pdf_expectation" else decoder_method
            temperature = self.config.decode_temperature if cdf_decoder == "pdf_expectation" else None
            future_norm = self._decode_staged_combined_1d(
                future_2d_coarse,
                future_2d_fine,
                cdf_decoder=cdf_decoder,
                expectation_sharpen_temp=temperature,
            )
            generated_2d = future_2d_flat.reshape(B, V, C_occ, H, W_fut)
        else:
            generated_2d = future_2d_flat.reshape(B, V, H, W_fut)
            if stage == "vertical_dual":
                Hc = int(self.config.coarse_image_height)
                future_2d_coarse, future_2d_fine = self.to_2d.split_vertical_dual(generated_2d, Hc)
                future_2d_finer = None
                cdf_decoder = "pdf_expectation" if decoder_method == "pdf_expectation" else decoder_method
                temperature = self.config.decode_temperature if cdf_decoder == "pdf_expectation" else None
                future_norm = self._decode_staged_combined_1d(
                    future_2d_coarse,
                    future_2d_fine,
                    cdf_decoder=cdf_decoder,
                    expectation_sharpen_temp=temperature,
                )
            elif stage == "coarse":
                future_2d_coarse = generated_2d
                # Must use staged coarse decode: ordinal maps live in [0, rank_max], not
                # legacy [-max_scale, max_scale] (to_2d.inverse → garbage after ordinal_decode).
                cdf_decoder = "pdf_expectation" if decoder_method == "pdf_expectation" else decoder_method
                temperature = self.config.decode_temperature if cdf_decoder == "pdf_expectation" else None
                future_norm = self._decode_coarse_1d_from_map(
                    future_2d_coarse,
                    cdf_decoder=cdf_decoder,
                    expectation_sharpen_temp=temperature,
                )
                future_2d_fine = None
                future_2d_finer = None
            elif stage == "fine":
                future_2d_coarse = coarse_for_decode.to(device)
                future_2d_fine = generated_2d
                future_2d_finer = None
                k = int(self.config.lookback_overlap)
                if k > 0:
                    past_seed = past_norm[..., k - 1]
                else:
                    past_seed = past_norm[..., -1]
                future_norm = self.decode_dual_from_2d(
                    future_2d_coarse,
                    future_2d_fine,
                    from_diffusion=False,
                    decoder_method=decoder_method,
                    past_seed=past_seed,
                )
            else:
                future_2d_coarse = coarse_for_decode.to(device)
                future_2d_fine = fine_for_decode.to(device)
                future_2d_finer = generated_2d
                future_norm = self.decode_triple_from_2d(
                    future_2d_coarse,
                    future_2d_fine,
                    future_2d_finer,
                    from_diffusion=False,
                    decoder_method=decoder_method,
                )
        future = self._denormalize_future(future_norm, past, stats)

        result = {
            'prediction': future,
            'prediction_norm': future_norm,
            'prediction_global_norm': future,
            'future_2d': generated_2d,
            'future_2d_coarse': future_2d_coarse,
            'past_2d_coarse': past_maps["coarse"],
            'past_2d_fine': past_maps["fine"],
            'diffusion_stage': stage,
        }
        if future_2d_fine is not None:
            result['future_2d_fine'] = future_2d_fine
        if future_2d_finer is not None:
            result['future_2d_finer'] = future_2d_finer
        if "finer" in past_maps:
            result['past_2d_finer'] = past_maps["finer"]
        if intermediates is not None:
            reshaped_intermediates = []
            for (t_idx, i_tensor) in intermediates:
                if stage == "channel_dual":
                    reshaped_intermediates.append(
                        (t_idx, i_tensor.reshape(B, V, C_occ, H, W_fut))
                    )
                else:
                    reshaped_intermediates.append((t_idx, i_tensor.reshape(B, V, H, W_fut)))
            result['intermediates'] = reshaped_intermediates
        return result

    def _forward_binary_factorized(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Binary training: XOR-noise hard CDF images and predict clean x0 + noise mask."""
        assert self.binary_scheduler is not None, "binary scheduler is not initialized"

        B = past.shape[0]
        V = self.config.num_variables
        H = self.config.image_height
        device = past.device
        BV = B * V

        past_norm, future_norm, stats = self._normalize_sequence(past, future)
        future_2d = self.encode_to_2d_binary(future_norm)
        W_fut = future_2d.shape[3]

        if t is None:
            t = torch.randint(0, self.config.binary_num_steps, (B,), device=device)
        t_flat = t.unsqueeze(1).expand(-1, V).reshape(BV)
        variate_indices = None
        if self.config.use_variate_embedding and self.config.variate_factorized and V > 1:
            variate_indices = self._flat_variate_indices(BV, V, device)

        future_flat = future_2d.reshape(BV, 1, H, W_fut)
        xt_flat, zt_flat = self.binary_scheduler.add_noise(future_flat, t_flat)

        guidance_2d = None
        if self.config.use_guidance_channel:
            guidance_forecast_norm = self._get_guidance_forecast_norm(
                past, past_norm, stats, int(future_norm.shape[-1]),
            )
            guidance_2d = self.encode_to_2d_binary(guidance_forecast_norm)

        ctx = None if getattr(self.config, 'disable_cross_attention', False) else self._get_cross_variate_context(past, past_norm)
        ctx_flat = self._flatten_ctx_for_factorized_dit(ctx, B, V)
        ctx_anchor = ctx_flat

        canvas = self._inject_coordinate_channel(xt_flat.float())
        canvas = self._inject_time_channels(canvas)

        past_2d = self.encode_to_2d_binary(past_norm)
        W_past = past_2d.shape[3]
        past_flat = past_2d.reshape(BV, 1, H, W_past)
        if self._past_cond_resize_to_horizon():
            cond_for_unet = F.interpolate(past_flat, size=(H, W_fut), mode='bilinear', align_corners=False)
        else:
            cond_for_unet = past_flat
        cond_for_unet = self._apply_coarse_dropout(cond_for_unet)
        base_cond_for_unet = cond_for_unet
        guidance_2d_flat = guidance_2d.reshape(BV, 1, H, W_fut) if guidance_2d is not None else None

        if self.training and self.config.cfg_dropout > 0.0:
            drop_mask = torch.rand(B, device=device) < self.config.cfg_dropout
            drop_mask_flat = drop_mask.unsqueeze(1).expand(-1, V).reshape(BV)
            cond_for_unet = torch.where(
                drop_mask_flat.view(BV, 1, 1, 1),
                torch.zeros_like(cond_for_unet),
                cond_for_unet,
            )
            if ctx_flat is not None:
                ctx_flat = torch.where(
                    drop_mask_flat.view(BV, 1, 1),
                    torch.zeros_like(ctx_flat),
                    ctx_flat,
                )

            if guidance_2d is not None:
                guide_flat = torch.where(
                    drop_mask_flat.view(BV, 1, 1, 1),
                    torch.zeros_like(guidance_2d_flat),
                    guidance_2d_flat,
                )
                canvas = torch.cat([canvas, guide_flat], dim=1)
        elif guidance_2d_flat is not None:
            canvas = torch.cat([canvas, guidance_2d_flat], dim=1)

        out_flat = self._predict_noise_chunked(
            canvas, t_flat, cond_for_unet, ctx_flat, variate_indices=variate_indices, token_variate_ids=self._ctx_token_variate_ids,
        )

        x0_logits = out_flat[:, 0:1, :, :]
        zt_logits = out_flat[:, 1:2, :, :]
        loss_x0 = self._binary_plain_bce_loss(x0_logits, future_flat)
        loss_zt = self._binary_plain_bce_loss(zt_logits, zt_flat)
        regular_loss = loss_x0 + loss_zt

        anchor_loss = torch.tensor(0.0, device=device)
        combined_mse_loss = regular_loss
        if self.config.use_deterministic_anchor_loss:
            anchor_t_flat = torch.full(
                (BV,),
                self.config.binary_num_steps - 1,
                device=device,
                dtype=t_flat.dtype,
            )
            neutral_future_flat = self._binary_anchor_canvas_like(future_flat)
            anchor_canvas = self._inject_coordinate_channel(neutral_future_flat)
            anchor_canvas = self._inject_time_channels(anchor_canvas)
            if guidance_2d_flat is not None:
                anchor_canvas = torch.cat([anchor_canvas, guidance_2d_flat], dim=1)
            anchor_out_flat = self._predict_noise_chunked(
                anchor_canvas,
                anchor_t_flat,
                base_cond_for_unet,
                ctx_anchor,
                variate_indices=variate_indices, token_variate_ids=self._ctx_token_variate_ids,
            )
            anchor_x0_logits = anchor_out_flat[:, 0:1, :, :]
            anchor_loss = self._binary_plain_bce_loss(anchor_x0_logits, future_flat)
            lam = self.config.deterministic_anchor_lambda
            combined_mse_loss = lam * regular_loss + (1.0 - lam) * anchor_loss

        loss = combined_mse_loss
        x0_pred = torch.sigmoid(x0_logits).reshape(B, V, H, W_fut)

        return {
            'loss': loss,
            'noise_loss': regular_loss,
            'combined_mse_loss': combined_mse_loss,
            'anchor_loss': anchor_loss,
            'loss_x0': loss_x0,
            'loss_zt': loss_zt,
            'emd_loss': torch.tensor(0.0, device=device),
            'guidance_loss': torch.tensor(0.0, device=device),
            'noise_pred': x0_pred,
            'x0_pred': x0_pred,
            't': t,
        }

    @torch.no_grad()
    def _generate_binary_factorized(
        self,
        past: torch.Tensor,
        num_steps: int = 20,
        verbose: bool = False,
        decoder_method: str = "mean",
        sampler: str = "ddim",
        yield_intermediates: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Binary reverse sampling from random bits to a clean hard CDF image."""
        assert self.binary_scheduler is not None, "binary scheduler is not initialized"

        B = past.shape[0]
        V = self.config.num_variables
        H = self.config.image_height
        device = past.device
        BV = B * V
        raw_hz_w = int(self.config.forecast_length)
        W_fut = self._repr_forecast_width(raw_hz_w)

        past_norm, _, stats = self._normalize_sequence(past)
        past_2d = self.encode_to_2d_binary(past_norm)
        W_past = past_2d.shape[3]
        past_flat = past_2d.reshape(BV, 1, H, W_past)
        if self._past_cond_resize_to_horizon():
            cond_for_unet = F.interpolate(past_flat, size=(H, W_fut), mode='bilinear', align_corners=False)
        else:
            cond_for_unet = past_flat

        guidance_2d = None
        guide_flat = None
        if self.config.use_guidance_channel:
            guidance_forecast_norm = self._get_guidance_forecast_norm(past, past_norm, stats, raw_hz_w)
            guidance_2d = self.encode_to_2d_binary(guidance_forecast_norm)
            guide_flat = guidance_2d.reshape(BV, 1, H, W_fut)

        ctx = None if getattr(self.config, 'disable_cross_attention', False) else self._get_cross_variate_context(past, past_norm)
        ctx_flat = self._flatten_ctx_for_factorized_dit(ctx, B, V)
        variate_indices = None
        if self.config.use_variate_embedding and self.config.variate_factorized and V > 1:
            variate_indices = self._flat_variate_indices(BV, V, device)

        def _build_canvas(xt: torch.Tensor, guide: Optional[torch.Tensor] = None) -> torch.Tensor:
            canvas = self._inject_coordinate_channel(xt)
            canvas = self._inject_time_channels(canvas)
            if guide is not None:
                canvas = torch.cat([canvas, guide], dim=1)
            return canvas

        def _chunked_model_fn(xt: torch.Tensor, t_batch: torch.Tensor):
            canvas = _build_canvas(xt, guide_flat)
            out = self._predict_noise_chunked(
                canvas, t_batch, cond_for_unet, ctx_flat, variate_indices=variate_indices, token_variate_ids=self._ctx_token_variate_ids,
            )
            return out[:, 0:1], out[:, 1:2]

        intermediates = None
        if sampler in ("anchor", "deterministic_anchor"):
            t_batch = torch.full(
                (BV,),
                self.config.binary_num_steps - 1,
                device=device,
                dtype=torch.long,
            )
            neutral_future_flat = self._binary_anchor_canvas_shape(
                (BV, 1, H, W_fut), device=device,
            )
            x0_logits, _zt_logits = _chunked_model_fn(neutral_future_flat, t_batch)
            future_2d_flat = (torch.sigmoid(x0_logits) > 0.5).float()
            if yield_intermediates:
                intermediates = [(999, neutral_future_flat.clone()), (0, future_2d_flat.clone())]
        else:
            if yield_intermediates:
                future_2d_flat, intermediates = self.binary_scheduler.sample(
                    model_fn=_chunked_model_fn,
                    shape=(BV, 1, H, W_fut),
                    num_steps=num_steps,
                    device=device,
                    verbose=verbose,
                    yield_intermediates=True,
                )
            else:
                future_2d_flat = self.binary_scheduler.sample(
                    model_fn=_chunked_model_fn,
                    shape=(BV, 1, H, W_fut),
                    num_steps=num_steps,
                    device=device,
                    verbose=verbose,
                )
        future_2d = future_2d_flat.reshape(B, V, H, W_fut)
        future_norm = self.decode_from_2d(
            future_2d, from_diffusion=False, decoder_method=decoder_method, **kwargs
        )
        future = self._denormalize_future(future_norm, past, stats)

        result = {
            'prediction': future,
            'prediction_norm': future_norm,
            'prediction_global_norm': future,
            'future_2d': future_2d,
            'past_2d': past_2d,
        }
        if guidance_2d is not None:
            result['guidance_2d'] = guidance_2d
        if intermediates is not None:
            # Reshape intermediate lists from (BV, ...) to (B, V, ...)
            reshaped_intermediates = []
            for (t_idx, i_tensor) in intermediates:
                reshaped_intermediates.append((t_idx, i_tensor.reshape(B, V, H, W_fut)))
            result['intermediates'] = reshaped_intermediates
            
        return result

    def get_loss(
        self,
        past: torch.Tensor,
        future: torch.Tensor
    ) -> torch.Tensor:
        """Convenience method to get just the loss for training."""
        if self._ar_training_enabled(future.shape[-1]):
            past, future = self._sample_ar_training_chunk(past, future)
        outputs = self.forward(past, future)
        return outputs['loss']
