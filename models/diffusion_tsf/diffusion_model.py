"""
Complete Diffusion-based Time Series Forecasting Model.

Binary CDF images, FactorizedDiT denoiser, iTransformer guidance channel.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Optional, Tuple, Union

from .config import DiffusionTSFConfig
from .preprocessing import TimeSeriesTo2D
from .diffusion import BinaryDiffusionScheduler, OrdinalD3PMScheduler
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
            use_scale_embedding=config.use_dual_scale,
            enable_cross_scale_attention=config.use_dual_scale,
            use_variate_embedding=(
                config.use_variate_embedding
                and config.variate_factorized
                and config.num_variables > 1
            ),
            max_variates=max(config.num_variables, 512),
            cross_variate_context_bias=config.cross_variate_context_bias,
        )

        self.context_encoder = iTransformerTokenAdapter(
            d_model=config.itrans_d_model,
            context_dim=config.context_embedding_dim,
            max_variates=max(config.num_variables, 512),
            dropout=0.1,
        )

        self.binary_scheduler = None
        self.ordinal_scheduler = None
        if config.diffusion_type == "binary":
            self.binary_scheduler = BinaryDiffusionScheduler(
                num_steps=config.binary_num_steps,
                beta_start=config.binary_beta_start,
                beta_end=config.binary_beta_end,
                schedule_type=config.binary_noise_schedule,
            )
        elif config.diffusion_type == "ordinal_d3pm":
            self.ordinal_scheduler = OrdinalD3PMScheduler(
                num_steps=config.binary_num_steps,
                num_classes=config.image_height,
                transition_min=config.d3pm_transition_min,
                transition_max=config.d3pm_transition_max,
                schedule_type=config.d3pm_noise_schedule,
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
        if self.ordinal_scheduler is not None:
            self.ordinal_scheduler = self.ordinal_scheduler.to(device)
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

    def _get_guidance_forecast_norm(
        self,
        past: torch.Tensor,
        past_norm: torch.Tensor,
        stats: Tuple[torch.Tensor, torch.Tensor],
        forecast_length: int,
    ) -> torch.Tensor:
        """Run the guidance model and return normalized forecast (B, V, forecast_length)."""
        if self.guidance_model is None:
            raise ValueError("guidance model is None but guidance channel requested")
        mean, std = stats
        K = self.config.lookback_overlap
        H = forecast_length - K
        if self.config.zero_guidance_forecast:
            coarse_norm = torch.zeros(
                past.shape[0],
                self.config.num_variables,
                H,
                device=past.device,
                dtype=past.dtype,
            )
            if K > 0:
                coarse_norm = torch.cat([torch.zeros_like(past_norm[..., -K:]), coarse_norm], dim=-1)
            return coarse_norm
        with torch.no_grad():
            coarse = self.guidance_model.get_forecast(past, H)
        coarse_norm = (coarse - mean) / std
        if K > 0:
            coarse_norm = torch.cat([past_norm[..., -K:], coarse_norm], dim=-1)
        return coarse_norm  # (B, V, forecast_length) normalized

    def _get_cross_variate_context(self, past_raw: torch.Tensor) -> Optional[torch.Tensor]:
        """produce (B, V, ctx_dim) encoder_hidden_states for the bottleneck.

        feeds raw past through the frozen iTransformer encoder and projects to context_dim.
        """
        if self.guidance_model is None or not hasattr(self.guidance_model, 'get_encoder_tokens'):
            raise RuntimeError(
                "iTransformerTokenAdapter requires a guidance model with get_encoder_tokens(). "
                "Ensure guidance model is iTransformerGuidance."
            )
        enc_tokens = self.guidance_model.get_encoder_tokens(past_raw)   # (B, V, d_model)
        return self.context_encoder(enc_tokens)                          # (B, V, ctx_dim)

    def _normalize_sequence(
        self,
        past: torch.Tensor,
        future: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Per-window z-score using past mean/std; future uses the same stats."""
        if not self.config.use_window_normalization:
            mean = torch.zeros_like(past[..., :1])
            std = torch.ones_like(past[..., :1])
            return past, future, (mean, std)
        mean = past.mean(dim=-1, keepdim=True)
        std = past.std(dim=-1, keepdim=True).clamp_min(self.config.window_norm_std_floor)
        past_norm = (past - mean) / std
        if future is not None:
            future_norm = (future - mean) / std
        else:
            future_norm = None
        return past_norm, future_norm, (mean, std)
    
    def _denormalize(
        self,
        x: torch.Tensor,
        stats: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Denormalize using stored statistics."""
        mean, std = stats
        return x * std + mean
    
    def _predict_noise_chunked(
        self,
        canvas: torch.Tensor,
        t_flat: torch.Tensor,
        cond_for_unet: Optional[torch.Tensor],
        ctx_flat: Optional[torch.Tensor],
        scale_indices: Optional[torch.Tensor] = None,
        variate_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the denoiser with the same chunking rule used by training/eval."""
        chunk_size = self.config.unet_max_chunk_size
        n_items = canvas.shape[0]
        if self.config.use_dual_scale and chunk_size > 0:
            chunk_size = max(2, (chunk_size // 2) * 2)
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
                kwargs = {"encoder_hidden_states": c_ctx}
                if c_scale is not None:
                    kwargs["scale_indices"] = c_scale
                if c_var is not None:
                    kwargs["variate_indices"] = c_var
                outs.append(self.noise_predictor(c_canvas, c_t, c_cond, **kwargs))
            return torch.cat(outs, dim=0)
        kwargs = {"encoder_hidden_states": ctx_flat}
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
        return self.to_2d(x)

    def encode_dual_to_2d_binary(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode 1D series to coarse and residual hard binary CDF images."""
        if self.config.diffusion_stage in {"coarse", "fine", "finer"}:
            return self._encode_staged_dual_to_2d_binary(x)
        return self.to_2d.encode_dual(x)

    def _staged_image_heights(self) -> Tuple[int, int, int]:
        return (
            int(getattr(self.config, "coarse_image_height", self.config.image_height)),
            int(getattr(self.config, "fine_image_height", self.config.image_height)),
            int(getattr(self.config, "finer_image_height", self.config.image_height)),
        )

    def _encode_staged_dual_to_2d_binary(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        coarse_h, fine_h, _finer_h = self._staged_image_heights()
        return self.to_2d.encode_dual_heights(
            x,
            coarse_height=coarse_h,
            fine_height=fine_h,
        )

    def _encode_staged_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        coarse_h, fine_h, finer_h = self._staged_image_heights()
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

    def _encode_staged_maps_skyline(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        coarse_h, fine_h, _finer_h = self._staged_image_heights()
        coarse, fine, coarse_bins, fine_bins = self.to_2d.encode_dual_skyline_heights(
            x,
            coarse_height=coarse_h,
            fine_height=fine_h,
        )
        return {
            "coarse": coarse,
            "fine": fine,
            "coarse_bins": coarse_bins,
            "fine_bins": fine_bins,
        }

    def _resize_skyline_height(self, image: torch.Tensor, target_height: int) -> torch.Tensor:
        if image.shape[2] == target_height:
            return image
        flat = image.reshape(-1, 1, image.shape[2], image.shape[3])
        resized = F.interpolate(
            flat,
            size=(target_height, image.shape[3]),
            mode="nearest",
        )
        return resized.reshape(image.shape[0], image.shape[1], target_height, image.shape[3])

    def _coarse_skyline_to_height(self, coarse_map: torch.Tensor, target_height: int) -> torch.Tensor:
        if coarse_map.shape[2] == target_height:
            return coarse_map
        coarse_value = self.to_2d.decode_skyline(
            coarse_map,
            value_range=self.config.max_scale,
            squeeze_univariate=False,
        )
        skyline, _ = self.to_2d.encode_skyline(
            coarse_value,
            height=target_height,
            value_range=self.config.max_scale,
        )
        return skyline

    def decode_dual_from_skyline(
        self,
        coarse_map: torch.Tensor,
        fine_map: torch.Tensor,
        squeeze_univariate: bool = True,
    ) -> torch.Tensor:
        return self.to_2d.decode_dual_skyline(
            coarse_map,
            fine_map,
            squeeze_univariate=squeeze_univariate,
        )

    def _random_uniform_skyline(
        self,
        shape: Tuple[int, int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        n, _c, h, w = shape
        bins = torch.randint(0, h, (n, w), device=device)
        return self.ordinal_scheduler._skyline_from_bins(bins)

    def _staged_past_condition_skyline(
        self,
        past_norm: torch.Tensor,
        target_width: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, V = past_norm.shape[:2]
        H = self.config.image_height
        BV = B * V
        past_tail_len = min(past_norm.shape[-1], target_width)
        past_tail_norm = past_norm[..., -past_tail_len:]
        past_maps = self._encode_staged_maps_skyline(past_tail_norm)
        cond_maps = []
        if self.config.diffusion_stage == "coarse":
            cond_maps.append(self._resize_skyline_height(past_maps["coarse"], H))
        else:
            cond_maps.append(self._coarse_skyline_to_height(past_maps["coarse"], H))
        cond_maps.append(self._resize_skyline_height(past_maps["fine"], H))
        cond = torch.cat(
            [m.reshape(BV, 1, H, past_tail_len) for m in cond_maps],
            dim=1,
        )
        cond = F.interpolate(cond, size=(H, target_width), mode="nearest")
        return cond, past_maps

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
    ) -> torch.Tensor:
        """Decode dual-scale CDF maps to normalized 1D values."""
        if from_diffusion:
            coarse_map = (coarse_map + 1.0) / 2.0
            fine_map = (fine_map + 1.0) / 2.0
        cdf_decoder = "pdf_expectation" if decoder_method == "pdf_expectation" else decoder_method
        temperature = self.config.decode_temperature if cdf_decoder == "pdf_expectation" else None
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
        if self.config.diffusion_type == "ordinal_d3pm":
            if self.config.diffusion_stage in {"coarse", "fine", "finer"}:
                return self._forward_ordinal_d3pm_staged(past, future, t)
            raise ValueError("ordinal_d3pm requires staged diffusion_stage 'coarse' or 'fine'.")
        if self.config.diffusion_stage in {"coarse", "fine", "finer"}:
            return self._forward_binary_staged(past, future, t)
        if self.config.use_dual_scale:
            return self._forward_binary_dual_scale(past, future, t)
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
        if self.config.diffusion_type == "ordinal_d3pm":
            if self.config.diffusion_stage in {"coarse", "fine", "finer"}:
                return self._generate_ordinal_d3pm_staged(past, **gen_common)
            raise ValueError("ordinal_d3pm requires staged diffusion_stage 'coarse' or 'fine'.")
        if self.config.diffusion_stage in {"coarse", "fine", "finer"}:
            return self._generate_binary_staged(past, **gen_common)
        if self.config.use_dual_scale:
            return self._generate_binary_dual_scale(past, **gen_common)
        return self._generate_binary_factorized(past, **gen_common)


    def _binary_plain_bce_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Unweighted BCE for binary CDF images."""
        return F.binary_cross_entropy_with_logits(logits, target.float())

    def _binary_weighted_bce_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        t_flat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """BCE with optional min-SNR timestep weighting."""
        per_elem = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
        if t_flat is None or self.config.loss_weighting == "none":
            return per_elem.mean()
        beta_t = self.binary_scheduler.betas[t_flat].clamp(1e-5, 1.0 - 1e-5)
        snr = ((1.0 - beta_t) ** 2) / (beta_t ** 2)
        weight = torch.minimum(snr, torch.full_like(snr, self.config.min_snr_gamma)) / snr
        view_shape = (-1,) + (1,) * (per_elem.dim() - 1)
        return (per_elem * weight.view(view_shape)).mean()

    def _x0_logits_from_prediction(
        self,
        primary_logits: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.prediction_target == "epsilon":
            return torch.where(xt > 0.5, -primary_logits, primary_logits)
        return primary_logits

    def _stack_dual_scale_flat(self, coarse: torch.Tensor, fine: torch.Tensor) -> torch.Tensor:
        """Interleave coarse/fine tensors so each (B,V) pair is adjacent in batch."""
        if coarse.shape != fine.shape:
            raise ValueError(f"coarse/fine shapes differ: {coarse.shape} vs {fine.shape}")
        return torch.stack((coarse, fine), dim=1).reshape(coarse.shape[0] * 2, *coarse.shape[1:])

    def _merge_dual_scale_channels(self, coarse: torch.Tensor, fine: torch.Tensor) -> torch.Tensor:
        """Channel-stack coarse+fine, then repeat for each denoise scale row (BV*2)."""
        if coarse.shape != fine.shape:
            raise ValueError(f"coarse/fine shapes differ: {coarse.shape} vs {fine.shape}")
        merged = torch.cat((coarse, fine), dim=1)
        return merged.unsqueeze(1).expand(-1, 2, -1, -1, -1).reshape(merged.shape[0] * 2, *merged.shape[1:])

    def _flat_variate_indices(self, bv: int, num_variates: int, device: torch.device) -> torch.Tensor:
        if bv % num_variates != 0:
            raise ValueError(f"bv={bv} not divisible by num_variates={num_variates}")
        batch_size = bv // num_variates
        return torch.arange(num_variates, device=device).unsqueeze(0).expand(batch_size, -1).reshape(bv)

    def _dual_scale_variate_indices(self, bv: int, num_variates: int, device: torch.device) -> torch.Tensor:
        base = self._flat_variate_indices(bv, num_variates, device)
        return base.unsqueeze(1).expand(-1, 2).reshape(bv * 2)

    def _dual_scale_indices(self, bv: int, device: torch.device) -> torch.Tensor:
        return torch.arange(2, device=device, dtype=torch.long).view(1, 2).expand(bv, -1).reshape(bv * 2)

    def _expand_ctx_to_dual_scale(
        self,
        ctx: Optional[torch.Tensor],
        B: int,
        V: int,
    ) -> Optional[torch.Tensor]:
        if ctx is None:
            return None
        return ctx.unsqueeze(1).unsqueeze(2).expand(-1, V, 2, -1, -1).reshape(B * V * 2, V, -1)

    def _staged_past_condition(
        self,
        past_norm: torch.Tensor,
        target_width: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Build GT lookback conditioning maps for staged denoisers."""
        B, V = past_norm.shape[:2]
        H = self.config.image_height
        BV = B * V
        past_tail_len = min(past_norm.shape[-1], target_width)
        past_tail_norm = past_norm[..., -past_tail_len:]
        past_maps = self._encode_staged_maps(past_tail_norm)
        cond_maps = []
        if self.config.diffusion_stage == "coarse":
            cond_maps.append(self._resize_cdf_height(past_maps["coarse"], H))
        else:
            cond_maps.append(self._coarse_cdf_to_height(past_maps["coarse"], H))
        cond_maps.append(self._resize_cdf_height(past_maps["fine"], H))
        if getattr(self.config, "use_triple_scale", False):
            cond_maps.append(self._resize_cdf_height(past_maps["finer"], H))
        cond = torch.cat(
            [m.reshape(BV, 1, H, past_tail_len) for m in cond_maps],
            dim=1,
        )
        cond = F.interpolate(cond, size=(H, target_width), mode='bilinear', align_corners=False)
        return cond, past_maps

    def _ordinal_ce_loss(self, logits: torch.Tensor, target_bins: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target_bins.long())

    def _forward_binary_staged(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Train one staged denoiser: future coarse, fine residual, or finer residual."""
        assert self.binary_scheduler is not None, "binary scheduler is not initialized"
        stage = self.config.diffusion_stage
        if stage not in {"coarse", "fine", "finer"}:
            raise ValueError(f"_forward_binary_staged called for stage={stage!r}")

        B = past.shape[0]
        V = self.config.num_variables
        H = self.config.image_height
        device = past.device
        BV = B * V

        past_norm, future_norm, _stats = self._normalize_sequence(past, future)
        future_maps = self._encode_staged_maps(future_norm)
        target_2d = future_maps[stage]
        W_fut = target_2d.shape[3]
        H = target_2d.shape[2]

        if t is None:
            t = torch.randint(0, self.config.binary_num_steps, (B,), device=device)
        t_flat = t.unsqueeze(1).expand(-1, V).reshape(BV)
        variate_indices = None
        if self.config.use_variate_embedding and self.config.variate_factorized and V > 1:
            variate_indices = self._flat_variate_indices(BV, V, device)

        target_flat = target_2d.reshape(BV, 1, H, W_fut)
        xt_flat, zt_flat = self.binary_scheduler.add_noise(target_flat, t_flat)

        ctx = None if getattr(self.config, 'disable_cross_attention', False) else self._get_cross_variate_context(past)
        ctx_flat = ctx.unsqueeze(1).expand(-1, V, -1, -1).reshape(BV, V, -1) if ctx is not None else None

        cond_for_unet, past_maps = self._staged_past_condition(past_norm, W_fut)
        if stage in {"fine", "finer"}:
            future_coarse_cond = self._coarse_cdf_to_height(future_maps["coarse"], H)
            future_coarse_flat = future_coarse_cond.reshape(BV, 1, H, W_fut)
            cond_for_unet = torch.cat((cond_for_unet, future_coarse_flat), dim=1)
        if stage == "finer":
            future_fine_cond = self._resize_cdf_height(future_maps["fine"], H)
            future_fine_flat = future_fine_cond.reshape(BV, 1, H, W_fut)
            cond_for_unet = torch.cat((cond_for_unet, future_fine_flat), dim=1)
        base_cond_for_unet = cond_for_unet

        guidance_flat = None
        if self.config.use_guidance_channel:
            guidance_forecast_norm = self._get_guidance_forecast_norm(past, past_norm, _stats, W_fut)
            guidance_maps = self._encode_staged_maps(guidance_forecast_norm)
            if stage == "coarse":
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
            canvas, t_flat, cond_for_unet, ctx_flat, variate_indices=variate_indices,
        )
        primary_logits = out_flat[:, 0:1, :, :]
        x0_logits = self._x0_logits_from_prediction(primary_logits, xt_flat)
        zt_logits = out_flat[:, 1:2, :, :]
        if self.config.prediction_target == "epsilon":
            loss_x0 = self._binary_weighted_bce_loss(primary_logits, zt_flat, t_flat)
            loss_zt = self._binary_weighted_bce_loss(zt_logits, target_flat, t_flat)
        else:
            loss_x0 = self._binary_weighted_bce_loss(primary_logits, target_flat, t_flat)
            loss_zt = self._binary_weighted_bce_loss(zt_logits, zt_flat, t_flat)
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
            neutral_future_flat = torch.full_like(target_flat, 0.5)
            anchor_canvas = self._inject_coordinate_channel(neutral_future_flat)
            anchor_canvas = self._inject_time_channels(anchor_canvas)
            if guidance_flat is not None:
                anchor_canvas = torch.cat([anchor_canvas, guidance_flat], dim=1)
            anchor_out_flat = self._predict_noise_chunked(
                anchor_canvas,
                anchor_t_flat,
                base_cond_for_unet,
                ctx_anchor,
                variate_indices=variate_indices,
            )
            anchor_primary = anchor_out_flat[:, 0:1]
            anchor_x0_logits = self._x0_logits_from_prediction(anchor_primary, neutral_future_flat)
            anchor_loss = self._binary_plain_bce_loss(anchor_x0_logits, target_flat)
            lam = self.config.deterministic_anchor_lambda
            combined_loss = lam * regular_loss + (1.0 - lam) * anchor_loss

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
        else:
            result['x0_pred_finer'] = x0_pred
        return result

    def _forward_ordinal_d3pm_staged(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Staged ordinal D3PM training with skyline maps and cross-entropy loss."""
        assert self.ordinal_scheduler is not None, "ordinal scheduler is not initialized"
        stage = self.config.diffusion_stage
        if stage not in {"coarse", "fine"}:
            raise ValueError(f"ordinal_d3pm staged forward does not support stage={stage!r}")

        B = past.shape[0]
        V = self.config.num_variables
        device = past.device
        BV = B * V

        past_norm, future_norm, _stats = self._normalize_sequence(past, future)
        future_maps = self._encode_staged_maps_skyline(future_norm)
        target_skyline = future_maps[stage]
        target_bins = future_maps[f"{stage}_bins"]
        W_fut = target_skyline.shape[3]
        H = target_skyline.shape[2]

        if t is None:
            t = torch.randint(0, self.config.binary_num_steps, (B,), device=device)
        t_flat = t.unsqueeze(1).expand(-1, V).reshape(BV)
        variate_indices = None
        if self.config.use_variate_embedding and self.config.variate_factorized and V > 1:
            variate_indices = self._flat_variate_indices(BV, V, device)

        target_flat = target_skyline.reshape(BV, 1, H, W_fut)
        target_bins_flat = target_bins.reshape(BV, W_fut)
        xt_flat, _xt_bins = self.ordinal_scheduler.add_noise(target_flat, t_flat)

        ctx = None if getattr(self.config, "disable_cross_attention", False) else self._get_cross_variate_context(past)
        ctx_flat = ctx.unsqueeze(1).expand(-1, V, -1, -1).reshape(BV, V, -1) if ctx is not None else None

        cond_for_unet, past_maps = self._staged_past_condition_skyline(past_norm, W_fut)
        if stage == "fine":
            future_coarse_cond = self._coarse_skyline_to_height(future_maps["coarse"], H)
            cond_for_unet = torch.cat(
                (cond_for_unet, future_coarse_cond.reshape(BV, 1, H, W_fut)),
                dim=1,
            )
        base_cond_for_unet = cond_for_unet

        guidance_flat = None
        if self.config.use_guidance_channel:
            guidance_forecast_norm = self._get_guidance_forecast_norm(past, past_norm, _stats, W_fut)
            guidance_maps = self._encode_staged_maps_skyline(guidance_forecast_norm)
            guidance_flat = self._resize_skyline_height(guidance_maps[stage], H).reshape(BV, 1, H, W_fut)

        canvas = self._inject_coordinate_channel(xt_flat.float())
        canvas = self._inject_time_channels(canvas)
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
            canvas, t_flat, cond_for_unet, ctx_flat, variate_indices=variate_indices,
        )
        logits = out_flat[:, 0]
        regular_loss = self._ordinal_ce_loss(logits, target_bins_flat)

        anchor_loss = torch.tensor(0.0, device=device)
        combined_loss = regular_loss
        if self.config.use_deterministic_anchor_loss:
            anchor_t_flat = torch.full(
                (BV,),
                self.config.binary_num_steps - 1,
                device=device,
                dtype=t_flat.dtype,
            )
            neutral_future_flat = self._random_uniform_skyline(
                (BV, 1, H, W_fut),
                device,
            )
            anchor_canvas = self._inject_coordinate_channel(neutral_future_flat)
            anchor_canvas = self._inject_time_channels(anchor_canvas)
            if guidance_flat is not None:
                anchor_canvas = torch.cat([anchor_canvas, guidance_flat], dim=1)
            anchor_out_flat = self._predict_noise_chunked(
                anchor_canvas,
                anchor_t_flat,
                base_cond_for_unet,
                ctx_anchor,
                variate_indices=variate_indices,
            )
            anchor_logits = anchor_out_flat[:, 0]
            anchor_loss = self._ordinal_ce_loss(anchor_logits, target_bins_flat)
            lam = self.config.deterministic_anchor_lambda
            combined_loss = lam * regular_loss + (1.0 - lam) * anchor_loss

        x0_pred = F.softmax(logits, dim=1).reshape(B, V, H, W_fut)

        result = {
            "loss": combined_loss,
            "noise_loss": regular_loss,
            "combined_mse_loss": combined_loss,
            "anchor_loss": anchor_loss,
            "loss_x0": regular_loss,
            "loss_zt": torch.tensor(0.0, device=device),
            "emd_loss": torch.tensor(0.0, device=device),
            "guidance_loss": torch.tensor(0.0, device=device),
            "noise_pred": x0_pred,
            "x0_pred": x0_pred,
            "future_2d": target_skyline,
            "future_2d_coarse": future_maps["coarse"],
            "future_2d_fine": future_maps["fine"],
            "past_2d_coarse": past_maps["coarse"],
            "past_2d_fine": past_maps["fine"],
            "t": t,
            "diffusion_stage": stage,
        }
        if stage == "coarse":
            result["x0_pred_coarse"] = x0_pred
        else:
            result["x0_pred_fine"] = x0_pred
        return result

    @torch.no_grad()
    def _generate_ordinal_d3pm_staged(
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
        """Generate staged ordinal skylines and decode to 1D forecasts."""
        assert self.ordinal_scheduler is not None, "ordinal scheduler is not initialized"
        stage = self.config.diffusion_stage
        if stage not in {"coarse", "fine"}:
            raise ValueError(f"ordinal_d3pm staged generate does not support stage={stage!r}")

        B = past.shape[0]
        V = self.config.num_variables
        H = self.config.image_height
        device = past.device
        BV = B * V
        W_fut = self.config.forecast_length

        past_norm, _, stats = self._normalize_sequence(past)
        cond_for_unet, past_maps = self._staged_past_condition_skyline(past_norm, W_fut)
        coarse_for_decode = future_coarse_2d
        if stage == "fine":
            if future_coarse_2d is None:
                raise ValueError("fine-stage generation requires future_coarse_2d from the coarse model.")
            future_coarse_cond = self._coarse_skyline_to_height(future_coarse_2d.to(device), H)
            cond_for_unet = torch.cat(
                (cond_for_unet, future_coarse_cond.reshape(BV, 1, H, W_fut)),
                dim=1,
            )

        ctx = None if getattr(self.config, "disable_cross_attention", False) else self._get_cross_variate_context(past)
        ctx_flat = ctx.unsqueeze(1).expand(-1, V, -1, -1).reshape(BV, V, -1) if ctx is not None else None
        variate_indices = None
        if self.config.use_variate_embedding and self.config.variate_factorized and V > 1:
            variate_indices = self._flat_variate_indices(BV, V, device)

        guidance_flat = None
        if self.config.use_guidance_channel:
            guidance_forecast_norm = self._get_guidance_forecast_norm(past, past_norm, stats, W_fut)
            guidance_maps = self._encode_staged_maps_skyline(guidance_forecast_norm)
            guidance_flat = self._resize_skyline_height(guidance_maps[stage], H).reshape(BV, 1, H, W_fut)

        def _build_canvas(xt: torch.Tensor) -> torch.Tensor:
            canvas = self._inject_coordinate_channel(xt)
            canvas = self._inject_time_channels(canvas)
            if guidance_flat is not None:
                canvas = torch.cat([canvas, guidance_flat], dim=1)
            return canvas

        def _chunked_model_fn(xt: torch.Tensor, t_batch: torch.Tensor):
            out = self._predict_noise_chunked(
                _build_canvas(xt), t_batch, cond_for_unet, ctx_flat,
                variate_indices=variate_indices,
            )
            return out[:, 0:1]

        intermediates = None
        if sampler in ("anchor", "deterministic_anchor"):
            t_batch = torch.full(
                (BV,),
                self.config.binary_num_steps - 1,
                device=device,
                dtype=torch.long,
            )
            neutral_future_flat = self._random_uniform_skyline((BV, 1, H, W_fut), device)
            logits = _chunked_model_fn(neutral_future_flat, t_batch)[:, 0]
            future_2d_flat = self.ordinal_scheduler._skyline_from_bins(logits.argmax(dim=1))
            if yield_intermediates:
                intermediates = [(999, neutral_future_flat.clone()), (0, future_2d_flat.clone())]
        else:
            sample_kwargs = dict(
                model_fn=_chunked_model_fn,
                shape=(BV, 1, H, W_fut),
                num_steps=num_steps,
                device=device,
                verbose=verbose,
                sampler=sampler,
                reverse_step_indices=reverse_step_indices,
                snapshot_timesteps=snapshot_timesteps,
            )
            if yield_intermediates:
                future_2d_flat, intermediates = self.ordinal_scheduler.sample(
                    yield_intermediates=True,
                    **sample_kwargs,
                )
            else:
                future_2d_flat = self.ordinal_scheduler.sample(**sample_kwargs)

        generated_2d = future_2d_flat.reshape(B, V, H, W_fut)
        if stage == "coarse":
            future_2d_coarse = generated_2d
            future_norm = self.to_2d.decode_skyline(
                future_2d_coarse,
                value_range=self.config.max_scale,
                squeeze_univariate=(V == 1),
            )
            future_2d_fine = None
        else:
            future_2d_coarse = coarse_for_decode.to(device)
            future_2d_fine = generated_2d
            future_norm = self.decode_dual_from_skyline(
                future_2d_coarse,
                future_2d_fine,
                squeeze_univariate=(V == 1),
            )
        future = self._denormalize(future_norm, stats)

        K = self.config.lookback_overlap
        if K > 0:
            future = future[..., K:]
            future_norm = future_norm[..., K:]

        result = {
            "prediction": future,
            "prediction_norm": future_norm,
            "prediction_global_norm": future,
            "future_2d": generated_2d,
            "future_2d_coarse": future_2d_coarse,
            "past_2d_coarse": past_maps["coarse"],
            "past_2d_fine": past_maps["fine"],
            "diffusion_stage": stage,
        }
        if future_2d_fine is not None:
            result["future_2d_fine"] = future_2d_fine
        if intermediates is not None:
            reshaped_intermediates = []
            for (t_idx, i_tensor) in intermediates:
                reshaped_intermediates.append((t_idx, i_tensor.reshape(B, V, H, W_fut)))
            result["intermediates"] = reshaped_intermediates
        return result

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
        if stage not in {"coarse", "fine", "finer"}:
            raise ValueError(f"_generate_binary_staged called for stage={stage!r}")

        B = past.shape[0]
        V = self.config.num_variables
        H = self.config.image_height
        device = past.device
        BV = B * V
        W_fut = self.config.forecast_length

        past_norm, _, stats = self._normalize_sequence(past)
        cond_for_unet, past_maps = self._staged_past_condition(past_norm, W_fut)
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
            cond_for_unet = torch.cat(
                (cond_for_unet, future_coarse_cond.reshape(BV, 1, H, W_fut)),
                dim=1,
            )
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

        ctx = None if getattr(self.config, 'disable_cross_attention', False) else self._get_cross_variate_context(past)
        ctx_flat = ctx.unsqueeze(1).expand(-1, V, -1, -1).reshape(BV, V, -1) if ctx is not None else None
        variate_indices = None
        if self.config.use_variate_embedding and self.config.variate_factorized and V > 1:
            variate_indices = self._flat_variate_indices(BV, V, device)

        guidance_flat = None
        if self.config.use_guidance_channel:
            guidance_forecast_norm = self._get_guidance_forecast_norm(past, past_norm, stats, W_fut)
            guidance_maps = self._encode_staged_maps(guidance_forecast_norm)
            if stage == "coarse":
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
                variate_indices=variate_indices,
            )
            x0_logits = self._x0_logits_from_prediction(out[:, 0:1], xt)
            return x0_logits, out[:, 1:2]

        intermediates = None
        if sampler in ("anchor", "deterministic_anchor"):
            t_batch = torch.full(
                (BV,),
                self.config.binary_num_steps - 1,
                device=device,
                dtype=torch.long,
            )
            neutral_future_flat = torch.full((BV, 1, H, W_fut), 0.5, device=device)
            x0_logits, _zt_logits = _chunked_model_fn(neutral_future_flat, t_batch)
            future_2d_flat = (torch.sigmoid(x0_logits) > 0.5).float()
            if yield_intermediates:
                intermediates = [(999, neutral_future_flat.clone()), (0, future_2d_flat.clone())]
        else:
            sample_kwargs = dict(
                model_fn=_chunked_model_fn,
                shape=(BV, 1, H, W_fut),
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

        generated_2d = future_2d_flat.reshape(B, V, H, W_fut)
        if stage == "coarse":
            future_2d_coarse = generated_2d
            future_norm = self.decode_from_2d(
                future_2d_coarse, from_diffusion=False, decoder_method=decoder_method, **kwargs
            )
            future_2d_fine = None
            future_2d_finer = None
        elif stage == "fine":
            future_2d_coarse = coarse_for_decode.to(device)
            future_2d_fine = generated_2d
            future_2d_finer = None
            future_norm = self.decode_dual_from_2d(
                future_2d_coarse,
                future_2d_fine,
                from_diffusion=False,
                decoder_method=decoder_method,
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
        future = self._denormalize(future_norm, stats)

        K = self.config.lookback_overlap
        if K > 0:
            future = future[..., K:]
            future_norm = future_norm[..., K:]

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
                reshaped_intermediates.append((t_idx, i_tensor.reshape(B, V, H, W_fut)))
            result['intermediates'] = reshaped_intermediates
        return result

    def _forward_binary_dual_scale(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Binary training with full-range coarse CDF plus residual fine CDF."""
        assert self.binary_scheduler is not None, "binary scheduler is not initialized"

        B = past.shape[0]
        V = self.config.num_variables
        device = past.device
        BV = B * V

        past_norm, future_norm, stats = self._normalize_sequence(past, future)
        future_coarse, future_fine = self.encode_dual_to_2d_binary(future_norm)
        H = future_coarse.shape[2]
        W_fut = future_coarse.shape[3]

        if t is None:
            t = torch.randint(0, self.config.binary_num_steps, (B,), device=device)
        t_bv = t.unsqueeze(1).expand(-1, V).reshape(BV)
        if self.config.dual_scale_independent_timesteps:
            # Independent timestep for fine scale so each scale's denoising
            # difficulty varies independently during training
            t_fine = torch.randint(0, self.config.binary_num_steps, (B,), device=device)
            t_fine_bv = t_fine.unsqueeze(1).expand(-1, V).reshape(BV)
            t_bvs = torch.stack([t_bv, t_fine_bv], dim=1).reshape(BV * 2)
        else:
            t_fine_bv = t_bv
            t_bvs = t_bv.unsqueeze(1).expand(-1, 2).reshape(BV * 2)
        scale_indices = self._dual_scale_indices(BV, device)
        variate_indices = None
        if self.config.use_variate_embedding and self.config.variate_factorized and V > 1:
            variate_indices = self._dual_scale_variate_indices(BV, V, device)

        future_coarse_flat = future_coarse.reshape(BV, 1, H, W_fut)
        future_fine_flat = future_fine.reshape(BV, 1, H, W_fut)
        xt_coarse, zt_coarse = self.binary_scheduler.add_noise(future_coarse_flat, t_bv)
        xt_fine, zt_fine = self.binary_scheduler.add_noise(future_fine_flat, t_fine_bv)

        xt_flat = self._stack_dual_scale_flat(xt_coarse, xt_fine)
        future_flat = self._stack_dual_scale_flat(future_coarse_flat, future_fine_flat)

        guidance_flat = None
        guidance_coarse = None
        guidance_fine = None
        if self.config.use_guidance_channel:
            guidance_forecast_norm = self._get_guidance_forecast_norm(past, past_norm, stats, W_fut)
            guidance_coarse, guidance_fine = self.encode_dual_to_2d_binary(guidance_forecast_norm)
            guidance_flat = self._merge_dual_scale_channels(
                guidance_coarse.reshape(BV, 1, H, W_fut),
                guidance_fine.reshape(BV, 1, H, W_fut),
            )

        ctx = None if getattr(self.config, 'disable_cross_attention', False) else self._get_cross_variate_context(past)
        ctx_flat = self._expand_ctx_to_dual_scale(ctx, B, V)
        ctx_anchor = ctx_flat

        canvas = self._inject_coordinate_channel(xt_flat.float())
        canvas = self._inject_time_channels(canvas)

        past_tail_len = min(past_norm.shape[-1], W_fut)
        past_tail_norm = past_norm[..., -past_tail_len:]
        past_coarse, past_fine = self.encode_dual_to_2d_binary(past_tail_norm)
        past_merged = self._merge_dual_scale_channels(
            past_coarse.reshape(BV, 1, H, past_tail_len),
            past_fine.reshape(BV, 1, H, past_tail_len),
        )
        cond_for_unet = F.interpolate(past_merged, size=(H, W_fut), mode='bilinear', align_corners=False)
        cond_for_unet = self._apply_coarse_dropout(cond_for_unet)
        base_cond_for_unet = cond_for_unet

        if self.training and self.config.cfg_dropout > 0.0:
            drop_mask = torch.rand(B, device=device) < self.config.cfg_dropout
            drop_mask_flat = drop_mask.view(B, 1, 1).expand(-1, V, 2).reshape(BV * 2)
            cond_for_unet = torch.where(
                drop_mask_flat.view(BV * 2, 1, 1, 1),
                torch.zeros_like(cond_for_unet),
                cond_for_unet,
            )
            if ctx_flat is not None:
                ctx_flat = torch.where(
                    drop_mask_flat.view(BV * 2, 1, 1),
                    torch.zeros_like(ctx_flat),
                    ctx_flat,
                )
            if guidance_flat is not None:
                guidance_for_unet = torch.where(
                    drop_mask_flat.view(BV * 2, 1, 1, 1),
                    torch.zeros_like(guidance_flat),
                    guidance_flat,
                )
                canvas = torch.cat([canvas, guidance_for_unet], dim=1)
        elif guidance_flat is not None:
            canvas = torch.cat([canvas, guidance_flat], dim=1)

        out_flat = self._predict_noise_chunked(
            canvas,
            t_bvs,
            cond_for_unet,
            ctx_flat,
            scale_indices=scale_indices,
            variate_indices=variate_indices,
        )
        out = out_flat.reshape(BV, 2, 2, H, W_fut)
        coarse_out = out[:, 0]
        fine_out = out[:, 1]
        x0_logits_coarse, zt_logits_coarse = coarse_out[:, 0:1], coarse_out[:, 1:2]
        x0_logits_fine, zt_logits_fine = fine_out[:, 0:1], fine_out[:, 1:2]

        loss_x0_coarse = self._binary_plain_bce_loss(x0_logits_coarse, future_coarse_flat)
        loss_zt_coarse = self._binary_plain_bce_loss(zt_logits_coarse, zt_coarse)
        loss_x0_fine = self._binary_plain_bce_loss(x0_logits_fine, future_fine_flat)
        loss_zt_fine = self._binary_plain_bce_loss(zt_logits_fine, zt_fine)
        coarse_loss = loss_x0_coarse + loss_zt_coarse
        fine_loss = loss_x0_fine + loss_zt_fine
        fine_weight = self.config.dual_scale_fine_weight
        regular_loss = (1.0 - fine_weight) * coarse_loss + fine_weight * fine_loss

        anchor_loss = torch.tensor(0.0, device=device)
        combined_mse_loss = regular_loss
        if self.config.use_deterministic_anchor_loss:
            anchor_t_flat = torch.full(
                (BV * 2,),
                self.config.binary_num_steps - 1,
                device=device,
                dtype=t_bvs.dtype,
            )
            neutral_future_flat = torch.full_like(future_flat, 0.5)
            anchor_canvas = self._inject_coordinate_channel(neutral_future_flat)
            anchor_canvas = self._inject_time_channels(anchor_canvas)
            if guidance_flat is not None:
                anchor_canvas = torch.cat([anchor_canvas, guidance_flat], dim=1)
            anchor_out_flat = self._predict_noise_chunked(
                anchor_canvas,
                anchor_t_flat,
                base_cond_for_unet,
                ctx_anchor,
                scale_indices=scale_indices,
                variate_indices=variate_indices,
            )
            anchor_out = anchor_out_flat.reshape(BV, 2, 2, H, W_fut)
            anchor_loss = (
                self._binary_plain_bce_loss(anchor_out[:, 0, 0:1], future_coarse_flat)
                + self._binary_plain_bce_loss(anchor_out[:, 1, 0:1], future_fine_flat)
            )
            lam = self.config.deterministic_anchor_lambda
            combined_mse_loss = lam * regular_loss + (1.0 - lam) * anchor_loss

        x0_pred_coarse = torch.sigmoid(x0_logits_coarse).reshape(B, V, H, W_fut)
        x0_pred_fine = torch.sigmoid(x0_logits_fine).reshape(B, V, H, W_fut)

        result = {
            'loss': combined_mse_loss,
            'noise_loss': regular_loss,
            'combined_mse_loss': combined_mse_loss,
            'anchor_loss': anchor_loss,
            'loss_x0': (1.0 - fine_weight) * loss_x0_coarse + fine_weight * loss_x0_fine,
            'loss_zt': (1.0 - fine_weight) * loss_zt_coarse + fine_weight * loss_zt_fine,
            'loss_x0_coarse': loss_x0_coarse,
            'loss_zt_coarse': loss_zt_coarse,
            'loss_x0_fine': loss_x0_fine,
            'loss_zt_fine': loss_zt_fine,
            'coarse_loss': coarse_loss,
            'fine_loss': fine_loss,
            'emd_loss': torch.tensor(0.0, device=device),
            'guidance_loss': torch.tensor(0.0, device=device),
            'noise_pred': x0_pred_coarse,
            'x0_pred': x0_pred_coarse,
            'x0_pred_coarse': x0_pred_coarse,
            'x0_pred_fine': x0_pred_fine,
            'future_2d': future_coarse,
            'future_2d_coarse': future_coarse,
            'future_2d_fine': future_fine,
            't': t,
        }
        if guidance_coarse is not None and guidance_fine is not None:
            result['guidance_2d_coarse'] = guidance_coarse
            result['guidance_2d_fine'] = guidance_fine
        return result

    @torch.no_grad()
    def _generate_binary_dual_scale(
        self,
        past: torch.Tensor,
        num_steps: int = 20,
        verbose: bool = False,
        decoder_method: str = "mean",
        sampler: str = "ddim",
        yield_intermediates: bool = False,
        reverse_step_indices: Optional[torch.Tensor] = None,
        snapshot_timesteps: Optional[Tuple[int, ...]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Binary reverse sampling in lock-step over coarse and residual scales."""
        assert self.binary_scheduler is not None, "binary scheduler is not initialized"

        B = past.shape[0]
        V = self.config.num_variables
        H = self.config.image_height
        device = past.device
        BV = B * V
        BVS = BV * 2
        W_fut = self.config.forecast_length
        scale_indices = self._dual_scale_indices(BV, device)
        variate_indices = None
        if self.config.use_variate_embedding and self.config.variate_factorized and V > 1:
            variate_indices = self._dual_scale_variate_indices(BV, V, device)

        past_norm, _, stats = self._normalize_sequence(past)
        past_tail_len = min(past_norm.shape[-1], W_fut)
        past_tail_norm = past_norm[..., -past_tail_len:]
        past_coarse, past_fine = self.encode_dual_to_2d_binary(past_tail_norm)
        past_merged = self._merge_dual_scale_channels(
            past_coarse.reshape(BV, 1, H, past_tail_len),
            past_fine.reshape(BV, 1, H, past_tail_len),
        )
        cond_for_unet = F.interpolate(past_merged, size=(H, W_fut), mode='bilinear', align_corners=False)

        guidance_coarse = None
        guidance_fine = None
        guide_flat = None
        if self.config.use_guidance_channel:
            guidance_forecast_norm = self._get_guidance_forecast_norm(past, past_norm, stats, W_fut)
            guidance_coarse, guidance_fine = self.encode_dual_to_2d_binary(guidance_forecast_norm)
            guide_flat = self._merge_dual_scale_channels(
                guidance_coarse.reshape(BV, 1, H, W_fut),
                guidance_fine.reshape(BV, 1, H, W_fut),
            )

        ctx = None if getattr(self.config, 'disable_cross_attention', False) else self._get_cross_variate_context(past)
        ctx_flat = self._expand_ctx_to_dual_scale(ctx, B, V)

        def _build_canvas(xt: torch.Tensor, guide: Optional[torch.Tensor] = None) -> torch.Tensor:
            canvas = self._inject_coordinate_channel(xt)
            canvas = self._inject_time_channels(canvas)
            if guide is not None:
                canvas = torch.cat([canvas, guide], dim=1)
            return canvas

        def _chunked_model_fn(xt: torch.Tensor, t_batch: torch.Tensor):
            canvas = _build_canvas(xt, guide_flat)
            out = self._predict_noise_chunked(
                canvas, t_batch, cond_for_unet, ctx_flat,
                scale_indices=scale_indices, variate_indices=variate_indices,
            )
            return out[:, 0:1], out[:, 1:2]

        intermediates = None
        if sampler in ("anchor", "deterministic_anchor"):
            t_batch = torch.full(
                (BVS,),
                self.config.binary_num_steps - 1,
                device=device,
                dtype=torch.long,
            )
            neutral_future_flat = torch.full((BVS, 1, H, W_fut), 0.5, device=device)
            x0_logits, _zt_logits = _chunked_model_fn(neutral_future_flat, t_batch)
            future_2d_flat = (torch.sigmoid(x0_logits) > 0.5).float()
            if yield_intermediates:
                intermediates = [(999, neutral_future_flat.clone()), (0, future_2d_flat.clone())]
        else:
            sample_kwargs = dict(
                model_fn=_chunked_model_fn,
                shape=(BVS, 1, H, W_fut),
                num_steps=num_steps,
                device=device,
                verbose=verbose,
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

        future_by_scale = future_2d_flat.reshape(BV, 2, 1, H, W_fut)
        future_2d_coarse = future_by_scale[:, 0, 0].reshape(B, V, H, W_fut)
        future_2d_fine = future_by_scale[:, 1, 0].reshape(B, V, H, W_fut)
        future_norm = self.decode_dual_from_2d(
            future_2d_coarse,
            future_2d_fine,
            from_diffusion=False,
            decoder_method=decoder_method,
        )
        future = self._denormalize(future_norm, stats)

        K = self.config.lookback_overlap
        if K > 0:
            future = future[..., K:]
            future_norm = future_norm[..., K:]

        result = {
            'prediction': future,
            'prediction_norm': future_norm,
            'prediction_global_norm': future,
            'future_2d': future_2d_coarse,
            'future_2d_coarse': future_2d_coarse,
            'future_2d_fine': future_2d_fine,
            'past_2d': past_coarse,
            'past_2d_coarse': past_coarse,
            'past_2d_fine': past_fine,
        }
        if guidance_coarse is not None and guidance_fine is not None:
            result['guidance_2d_coarse'] = guidance_coarse
            result['guidance_2d_fine'] = guidance_fine
        if intermediates is not None:
            reshaped_intermediates = []
            for (t_idx, i_tensor) in intermediates:
                reshaped_intermediates.append((t_idx, i_tensor.reshape(B, V, 2, H, W_fut)))
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
            guidance_forecast_norm = self._get_guidance_forecast_norm(past, past_norm, stats, W_fut)
            guidance_2d = self.encode_to_2d_binary(guidance_forecast_norm)

        ctx = None if getattr(self.config, 'disable_cross_attention', False) else self._get_cross_variate_context(past)
        ctx_flat = None
        if ctx is not None:
            ctx_flat = ctx.unsqueeze(1).expand(-1, V, -1, -1).reshape(BV, V, -1)
        ctx_anchor = ctx_flat

        canvas = self._inject_coordinate_channel(xt_flat.float())
        canvas = self._inject_time_channels(canvas)

        past_2d = self.encode_to_2d_binary(past_norm)
        W_past = past_2d.shape[3]
        past_flat = past_2d.reshape(BV, 1, H, W_past)
        cond_for_unet = F.interpolate(past_flat, size=(H, W_fut), mode='bilinear', align_corners=False)
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
            canvas, t_flat, cond_for_unet, ctx_flat, variate_indices=variate_indices,
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
            neutral_future_flat = torch.full_like(future_flat, 0.5)
            anchor_canvas = self._inject_coordinate_channel(neutral_future_flat)
            anchor_canvas = self._inject_time_channels(anchor_canvas)
            if guidance_2d_flat is not None:
                anchor_canvas = torch.cat([anchor_canvas, guidance_2d_flat], dim=1)
            anchor_out_flat = self._predict_noise_chunked(
                anchor_canvas,
                anchor_t_flat,
                base_cond_for_unet,
                ctx_anchor,
                variate_indices=variate_indices,
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
        W_fut = self.config.forecast_length

        past_norm, _, stats = self._normalize_sequence(past)
        past_2d = self.encode_to_2d_binary(past_norm)
        W_past = past_2d.shape[3]
        past_flat = past_2d.reshape(BV, 1, H, W_past)
        cond_for_unet = F.interpolate(past_flat, size=(H, W_fut), mode='bilinear', align_corners=False)

        guidance_2d = None
        guide_flat = None
        if self.config.use_guidance_channel:
            guidance_forecast_norm = self._get_guidance_forecast_norm(past, past_norm, stats, W_fut)
            guidance_2d = self.encode_to_2d_binary(guidance_forecast_norm)
            guide_flat = guidance_2d.reshape(BV, 1, H, W_fut)

        ctx = None if getattr(self.config, 'disable_cross_attention', False) else self._get_cross_variate_context(past)
        ctx_flat = ctx.unsqueeze(1).expand(-1, V, -1, -1).reshape(BV, V, -1) if ctx is not None else None
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
                canvas, t_batch, cond_for_unet, ctx_flat, variate_indices=variate_indices,
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
            neutral_future_flat = torch.full((BV, 1, H, W_fut), 0.5, device=device)
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
        future = self._denormalize(future_norm, stats)

        K = self.config.lookback_overlap
        if K > 0:
            future = future[..., K:]
            future_norm = future_norm[..., K:]

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
        outputs = self.forward(past, future)
        return outputs['loss']
