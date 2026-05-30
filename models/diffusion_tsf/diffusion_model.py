"""
Complete Diffusion-based Time Series Forecasting Model.

stuff in here:
- Preprocessing (norm, 2D encoding, blur)
- Conditional U-Net
- DDPM/DDIM diffusion
- 2D to 1D decoding
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Optional, Tuple, Union

from .config import DiffusionTSFConfig
from .preprocessing import TimeSeriesTo2D, VerticalGaussianBlur
from .unet import ConditionalUNet2D, iTransformerTokenAdapter
from .diffusion import BinaryDiffusionScheduler, DiffusionScheduler
from .guidance import GuidanceModel, LinearRegressionGuidance
from .metrics import monotonicity_loss
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
    """Diffusion-based Time Series Forecasting Model.
    
    Pipeline:
    1. Normalize input (past + future) using local mean/std
    2. Convert to 2D "stripe" representation
    3. Apply vertical Gaussian blur
    4. Train U-Net to denoise future conditioned on past
    5. At inference: generate future via DDPM/DDIM
    6. Decode 2D representation back to 1D
    
    Optional Hybrid "Visual Guide" mode (use_guidance_channel=True):
    - A Stage 1 predictor (e.g., iTransformer) generates a coarse forecast
    - The coarse forecast is converted to a 2D "ghost image"
    - This ghost image is concatenated to the U-Net input
    - The diffusion model focuses on refining texture/residuals
    """
    
    def __init__(
        self,
        config: DiffusionTSFConfig,
        guidance_model: Optional[Union[GuidanceModel, nn.Module]] = None
    ):
        """
        Args:
            config: Model configuration
            guidance_model: Optional Stage 1 predictor for hybrid forecasting.
                           If config.use_guidance_channel is True but no model
                           is provided, a LinearRegressionGuidance is used as default.
        """
        super().__init__()
        self.config = config
        
        # Preprocessing modules
        self.to_2d = TimeSeriesTo2D(
            height=config.image_height,
            max_scale=config.max_scale,
        )
        self.blur = VerticalGaussianBlur(
            kernel_size=config.blur_kernel_size,
            sigma=config.blur_sigma
        )
        
        # Guidance model for hybrid "visual guide" forecasting
        if config.use_guidance_channel:
            if guidance_model is not None:
                self.guidance_model = guidance_model
            else:
                # Default to linear regression if no model provided
                self.guidance_model = LinearRegressionGuidance()
                logger.info("Using default LinearRegressionGuidance for guidance channel")
        else:
            self.guidance_model = None
        
        # Noise prediction backbone (U-Net or Transformer)
        # Input channels: num_variables (data) + aux channels (coord, time_ramp, time_sine)
        # Use the config property for consistent calculation
        backbone_in_channels = config.backbone_in_channels
        
        denoiser_out_channels = 2 if config.diffusion_type == "binary" else 1

        if config.model_type == "dit":
            self.noise_predictor = FactorizedDiT(
                in_channels=backbone_in_channels,
                cond_channels=config.visual_cond_channels,
                out_channels=denoiser_out_channels,
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
            )
        else:
            self.noise_predictor = ConditionalUNet2D(
                in_channels=backbone_in_channels,
                out_channels=denoiser_out_channels,
                channels=config.unet_channels,
                num_res_blocks=config.num_res_blocks,
                attention_levels=config.attention_levels,
                image_height=config.image_height,
                kernel_size=config.unet_kernel_size,
                use_dilated_middle=config.use_dilated_middle,
                context_dim=config.context_embedding_dim,
                visual_cond_channels=config.visual_cond_channels,
                use_gradient_checkpointing=config.use_gradient_checkpointing,
            )

        # iTransformer token adapter is shared by both backbones; it produces the
        # (B, V, ctx_dim) cross-attention memory consumed at the bottleneck.
        # Use a large enough max_variates (default 512) to stay compatible with
        # pretrained checkpoints even when finetuning on fewer variables.
        self.context_encoder = iTransformerTokenAdapter(
            d_model=config.itrans_d_model,
            context_dim=config.context_embedding_dim,
            max_variates=max(config.num_variables, 512),
            dropout=0.1,
        )

        # Diffusion scheduler (not a nn.Module, managed separately)
        self.scheduler = DiffusionScheduler(
            num_steps=config.num_diffusion_steps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            schedule=config.noise_schedule
        )
        self.binary_scheduler = None
        if config.diffusion_type == "binary":
            self.binary_scheduler = BinaryDiffusionScheduler(
                num_steps=config.binary_num_steps,
                beta_start=config.binary_beta_start,
                beta_end=config.binary_beta_end,
            )

        logger.info(f"DiffusionTSF initialized:")
        logger.info(f"  Variables: {config.num_variables} ({'multivariate' if config.num_variables > 1 else 'univariate'})")
        logger.info(f"  Lookback: {config.lookback_length}, Forecast: {config.forecast_length}")
        logger.info(
            f"  Image size: {config.image_height} x {config.forecast_length} "
            f"(H x W; denoised future canvas)"
        )
        logger.info(f"  Diffusion type: {config.diffusion_type}")
        logger.info(f"  Prediction mode: {config.prediction_mode}")

    
    def to(self, device):
        """Move model and scheduler to device."""
        super().to(device)
        self.scheduler = self.scheduler.to(device)
        if self.binary_scheduler is not None:
            self.binary_scheduler = self.binary_scheduler.to(device)
        return self
    
    def set_guidance_model(self, guidance_model: Optional[Union[GuidanceModel, nn.Module]]) -> None:
        """Set or replace the guidance model for hybrid forecasting.
        
        This allows swapping the Stage 1 predictor after model initialization,
        e.g., to plug in a pre-trained iTransformer checkpoint.
        
        Args:
            guidance_model: Stage 1 predictor model. Set to None to disable
                           guidance (requires config.use_guidance_channel=False).
        """
        if guidance_model is None and self.config.use_guidance_channel:
            raise ValueError(
                "Cannot set guidance_model to None when use_guidance_channel=True. "
                "Either provide a guidance model or set config.use_guidance_channel=False."
            )
        self.guidance_model = guidance_model
        if guidance_model is not None:
            logger.info(f"Guidance model set: {type(guidance_model).__name__}")

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
    
    def _get_value_channel(
        self,
        values_norm: torch.Tensor,
        height: int
    ) -> torch.Tensor:
        """Create a 2D channel containing the normalized values broadcast across height."""
        if values_norm.dim() == 3:
            batch_size, num_vars, seq_len = values_norm.shape
            value_channel = values_norm.unsqueeze(2).expand(-1, -1, height, -1)
        else:
            batch_size, seq_len = values_norm.shape
            value_channel = values_norm.unsqueeze(1).unsqueeze(2).expand(-1, -1, height, -1)
        value_channel = value_channel.clamp(-self.config.max_scale, self.config.max_scale)
        value_channel = value_channel / self.config.max_scale
        return value_channel
    
    def _inject_value_channel(
        self,
        x: torch.Tensor,
        values_norm: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate value channel to input tensor."""
        if not self.config.use_value_channel:
            return x
        _, _, height, _ = x.shape
        value_channel = self._get_value_channel(values_norm, height)
        if value_channel.shape[1] > 1:
            value_channel = value_channel[:, 0:1, :, :]
        return torch.cat([x, value_channel], dim=1)
    
    def _generate_guidance_2d(
        self,
        past: torch.Tensor,
        past_norm: torch.Tensor,
        stats: Tuple[torch.Tensor, torch.Tensor],
        forecast_length: int
    ) -> torch.Tensor:
        """Generate 2D "ghost image" from Stage 1 guidance model."""
        coarse_norm = self._get_guidance_forecast_norm(past, past_norm, stats, forecast_length)
        return self.encode_to_2d(coarse_norm, scale_for_diffusion=True)
    
    def _inject_guidance_channel(
        self,
        x: torch.Tensor,
        guidance_2d: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Concatenate guidance 2D image to input tensor."""
        if not self.config.use_guidance_channel or guidance_2d is None:
            return x
        return torch.cat([x, guidance_2d], dim=1)
    
    def _prepare_visual_conditioning(
        self,
        past_2d: torch.Tensor,
        target_width: int
    ) -> torch.Tensor:
        """Prepare past 2D image for visual concatenation conditioning."""
        _, _, height, past_len = past_2d.shape
        if past_len >= target_width:
            visual_cond = past_2d[:, :, :, -target_width:]
        else:
            visual_cond = F.interpolate(
                past_2d, 
                size=(height, target_width), 
                mode='bilinear', 
                align_corners=False
            )
        return visual_cond
    
    def _get_guidance_forecast_norm(
        self,
        past: torch.Tensor,
        past_norm: torch.Tensor,
        stats: Tuple[torch.Tensor, torch.Tensor],
        forecast_length: int,
    ) -> torch.Tensor:
        """run the guidance model and return normalized forecast (B, V, forecast_length).

        separating this from _generate_guidance_2d so we can reuse the raw forecast
        as cross-variate context without calling get_forecast() twice.
        """
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
        std = past.std(dim=-1, keepdim=True) + 1e-8
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
    
    def _compute_emd_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute column-wise Wasserstein-1 distance via CDF trick."""
        if self.config.representation_mode == "pdf":
            temperature = self.config.decode_temperature
            prob_pred = F.softmax(pred / temperature, dim=2)
            prob_target = F.softmax(target / temperature, dim=2)
            cdf_pred = prob_pred.cumsum(dim=2)
            cdf_target = prob_target.cumsum(dim=2)
        else:
            cdf_pred = (pred + 1.0) / 2.0
            cdf_target = (target + 1.0) / 2.0
        emd = (cdf_pred - cdf_target).abs().mean()
        return emd

    def _logits_to_cdf_x0(self, logits: torch.Tensor) -> torch.Tensor:
        """Mold unconstrained height logits into the repo's bottom-filled CDF x0."""
        pdf = F.softmax(logits, dim=2)
        cdf = torch.flip(torch.cumsum(torch.flip(pdf, dims=(2,)), dim=2), dims=(2,))
        return cdf * 2.0 - 1.0

    def _deterministic_anchor_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return timestep and scale for alpha_bar closest to the configured anchor."""
        alphas = self.scheduler.alphas_cumprod
        target = torch.tensor(
            self.config.deterministic_anchor_alpha,
            device=alphas.device,
            dtype=alphas.dtype,
        )
        t_anchor = torch.argmin((alphas - target).abs()).long()
        alpha_bar = alphas[t_anchor].clamp(min=1e-8, max=1.0 - 1e-8)
        scale = -torch.sqrt(alpha_bar) / torch.sqrt(1.0 - alpha_bar)
        return t_anchor, scale

    def _predict_noise_chunked(
        self,
        canvas: torch.Tensor,
        t_flat: torch.Tensor,
        cond_for_unet: Optional[torch.Tensor],
        ctx_flat: Optional[torch.Tensor],
        scale_indices: Optional[torch.Tensor] = None,
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
                kwargs = {"encoder_hidden_states": c_ctx}
                if c_scale is not None:
                    kwargs["scale_indices"] = c_scale
                outs.append(self.noise_predictor(c_canvas, c_t, c_cond, **kwargs))
            return torch.cat(outs, dim=0)
        kwargs = {"encoder_hidden_states": ctx_flat}
        if scale_indices is not None:
            kwargs["scale_indices"] = scale_indices
        return self.noise_predictor(canvas, t_flat, cond_for_unet, **kwargs)

    def _build_anchor_canvas(
        self,
        zero_future_flat: torch.Tensor,
        guidance_2d_flat: Optional[torch.Tensor],
        use_null_guidance: bool,
        null_guide: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Zero-noise canvas for the deterministic anchor forward pass."""
        canvas = self._inject_coordinate_channel(zero_future_flat)
        canvas = self._inject_time_channels(canvas)
        if guidance_2d_flat is not None:
            if use_null_guidance:
                guide = null_guide if null_guide is not None else torch.zeros_like(guidance_2d_flat)
            else:
                guide = guidance_2d_flat
            canvas = torch.cat([canvas, guide], dim=1)
        return canvas

    def _predict_anchor_noise(
        self,
        zero_future_flat: torch.Tensor,
        t_flat: torch.Tensor,
        cond: torch.Tensor,
        ctx: Optional[torch.Tensor],
        guidance_2d_flat: Optional[torch.Tensor],
        cfg_scale: float,
        null_cond: Optional[torch.Tensor] = None,
        null_ctx: Optional[torch.Tensor] = None,
        null_guide: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Anchor noise prediction with optional CFG (cond vs null cond/ctx/guidance)."""
        if cfg_scale <= 1.0:
            canvas = self._build_anchor_canvas(
                zero_future_flat, guidance_2d_flat, use_null_guidance=False, null_guide=null_guide,
            )
            return self._predict_noise_chunked(canvas, t_flat, cond, ctx)
        out_c = self._predict_noise_chunked(
            self._build_anchor_canvas(
                zero_future_flat, guidance_2d_flat, use_null_guidance=False, null_guide=null_guide,
            ),
            t_flat,
            cond,
            ctx,
        )
        out_u = self._predict_noise_chunked(
            self._build_anchor_canvas(
                zero_future_flat, guidance_2d_flat, use_null_guidance=True, null_guide=null_guide,
            ),
            t_flat,
            null_cond,
            null_ctx,
        )
        return out_u + cfg_scale * (out_c - out_u)

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

    def encode_to_2d(self, x: torch.Tensor, scale_for_diffusion: bool = True) -> torch.Tensor:
        """Encode 1D time series to blurred 2D representation."""
        image = self.to_2d(x)
        blurred = self.blur(image)
        if scale_for_diffusion:
            if self.config.representation_mode == "pdf":
                scaled = blurred * 30.0
                scaled = scaled * 2.0 - 1.0
            else:
                scaled = blurred.clamp(min=0.0, max=1.0) * 2.0 - 1.0
            return scaled
        return blurred

    def encode_to_2d_binary(self, x: torch.Tensor) -> torch.Tensor:
        """Encode 1D series to a hard binary CDF image without blur."""
        return self.to_2d(x)

    def encode_dual_to_2d_binary(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode 1D series to coarse and residual hard binary CDF images."""
        return self.to_2d.encode_dual(x)

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
    
    def _pad_to_window(
        self,
        tensor: torch.Tensor,
        mode: str,
        total_length: int
    ) -> torch.Tensor:
        """Pad tensor to total window length (Lookback + Forecast)."""
        batch, channels, height, length = tensor.shape
        if length >= total_length:
            return tensor[..., :total_length]
        padding_len = total_length - length
        if mode == 'past':
            return F.pad(tensor, (0, padding_len, 0, 0))
        elif mode == 'future':
            return F.pad(tensor, (padding_len, 0, 0, 0))
        else:
            raise ValueError(f"Unknown padding mode: {mode}")

    def forward(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        t: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass (shared factorized path for unet and dit backbones)."""
        if self.config.diffusion_type == "binary":
            if self.config.use_dual_scale:
                return self._forward_binary_dual_scale(past, future, t)
            return self._forward_binary_factorized(past, future, t)
        return self._forward_factorized(past, future, t)

    @torch.no_grad()
    def generate(
        self,
        past: torch.Tensor,
        use_ddim: bool = True,
        num_ddim_steps: int = 50,
        eta: float = 0.0,
        cfg_scale: Optional[float] = None,
        verbose: bool = False,
        decoder_method: str = "mean",
        beam_width: int = 5,
        jump_penalty_scale: float = 1.0,
        search_radius: int = 10,
        sampler: str = "ddim",
        num_inference_steps: Optional[int] = None,
        yield_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Generate future predictions (shared factorized path for unet and dit backbones).

        sampler:
            'ddim' (default) — existing DDIM(+CFG) path, ``num_ddim_steps`` steps
            'ddpm'           — full T-step DDPM(+CFG)
            'dpmpp'          — DPM-Solver++(2M); CFG not supported (uses cond only)
            'anchor'         — one deterministic anchor pass at alpha_bar ~= 0.5
        num_inference_steps overrides num_ddim_steps when set (used by dpmpp/ddim).
        """
        if self.config.diffusion_type == "binary":
            steps = num_inference_steps if num_inference_steps is not None else self.config.binary_sample_steps
            if self.config.use_dual_scale:
                return self._generate_binary_dual_scale(
                    past, num_steps=steps, verbose=verbose,
                    decoder_method=decoder_method, beam_width=beam_width,
                    jump_penalty_scale=jump_penalty_scale,
                    search_radius=search_radius,
                    sampler=sampler,
                    yield_intermediates=yield_intermediates,
                )
            return self._generate_binary_factorized(
                past, num_steps=steps, verbose=verbose,
                decoder_method=decoder_method, beam_width=beam_width,
                jump_penalty_scale=jump_penalty_scale,
                search_radius=search_radius,
                sampler=sampler,
                yield_intermediates=yield_intermediates,
            )
        return self._generate_factorized(
            past, use_ddim=use_ddim, num_ddim_steps=num_ddim_steps,
            eta=eta, cfg_scale=cfg_scale, verbose=verbose,
            decoder_method=decoder_method, beam_width=beam_width,
            jump_penalty_scale=jump_penalty_scale, search_radius=search_radius,
            sampler=sampler, num_inference_steps=num_inference_steps,
        )
    
    # ====================================================================
    # Factorized U-Net forward/generate — per-variate shared-weight U-Net
    # with cross-variate context at the bottleneck via iTransformerTokenAdapter
    # ====================================================================

    def _forward_factorized(self, past: torch.Tensor, future: torch.Tensor, t: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """training forward: each variate's occupancy map denoised independently.

        the U-Net weights are shared across all V variates. cross-variate info
        is injected at the bottleneck via cross-attention on V context tokens
        from iTransformerTokenAdapter (iTransformer enc_out projected + variate identity).
        """
        B = past.shape[0]
        V = self.config.num_variables
        H = self.config.image_height
        device = past.device
        BV = B * V

        past_norm, future_norm, stats = self._normalize_sequence(past, future)
        past_2d   = self.encode_to_2d(past_norm)    # (B, V, H, W_past)
        future_2d = self.encode_to_2d(future_norm)   # (B, V, H, W_fut)
        past_2d   = self._apply_coarse_dropout(past_2d)

        W_past = past_2d.shape[3]
        W_fut  = future_2d.shape[3]

        if t is None:
            t = torch.randint(0, self.config.num_diffusion_steps, (B,), device=device)

        noisy_future, noise = self.scheduler.add_noise(future_2d, t)  # (B, V, H, W_fut)

        # compute guidance once, reuse for both the 2D ghost image and the context encoder
        guidance_forecast_norm = None
        guidance_2d = None
        if self.config.use_guidance_channel:
            guidance_forecast_norm = self._get_guidance_forecast_norm(past, past_norm, stats, W_fut)
            guidance_2d = self.encode_to_2d(guidance_forecast_norm, scale_for_diffusion=True)

        ctx = None if getattr(self.config, 'disable_cross_attention', False) else self._get_cross_variate_context(past)
        # ctx: (B, V, ctx_dim) or None

        # flatten variates into batch dim for shared-weight U-Net
        # t: same timestep for all V variates of each batch element
        t_flat = t.unsqueeze(1).expand(-1, V).reshape(BV)  # (BV,)

        base_canvas = noisy_future.reshape(BV, 1, H, W_fut)
        base_canvas = self._inject_coordinate_channel(base_canvas)
        base_canvas = self._inject_time_channels(base_canvas)

        # visual cond: per-variate past bilinearly resized to match future width
        past_flat     = past_2d.reshape(BV, 1, H, W_past)
        base_cond_for_unet = F.interpolate(past_flat, size=(H, W_fut), mode='bilinear', align_corners=False)

        # broadcast context: every one of the BV U-Net forward passes sees ALL V tokens
        ctx_anchor = None
        if ctx is not None:
            # (B, V, ctx_dim) → (BV, V, ctx_dim); kept full-strength for the anchor pass
            ctx_anchor = ctx.unsqueeze(1).expand(-1, V, -1, -1).reshape(BV, V, -1)

        canvas = base_canvas
        cond_for_unet = base_cond_for_unet
        ctx_cfg_dropout = ctx_anchor
        guidance_2d_flat = guidance_2d.reshape(BV, 1, H, W_fut) if guidance_2d is not None else None

        # --- Apply Classifier-Free Guidance Dropout ---
        if self.training and self.config.cfg_dropout > 0.0:
            drop_mask = torch.rand(B, device=device) < self.config.cfg_dropout
            drop_mask_flat = drop_mask.unsqueeze(1).expand(-1, V).reshape(BV)
            
            cond_for_unet = torch.where(drop_mask_flat.view(BV, 1, 1, 1), torch.zeros_like(cond_for_unet), cond_for_unet)
            
            if ctx_cfg_dropout is not None:
                ctx_cfg_dropout = torch.where(
                    drop_mask_flat.view(BV, 1, 1), torch.zeros_like(ctx_cfg_dropout), ctx_cfg_dropout,
                )
                
            if guidance_2d_flat is not None:
                guide_for_unet = torch.where(
                    drop_mask_flat.view(BV, 1, 1, 1),
                    torch.zeros_like(guidance_2d_flat),
                    guidance_2d_flat,
                )
                canvas = torch.cat([canvas, guide_for_unet], dim=1)
        else:
            if guidance_2d_flat is not None:
                canvas = torch.cat([canvas, guidance_2d_flat], dim=1)

        model_out_flat = self._predict_noise_chunked(canvas, t_flat, cond_for_unet, ctx_cfg_dropout)
            
        model_out = model_out_flat.reshape(B, V, H, W_fut)

        K = self.config.lookback_overlap
        if self.config.prediction_mode == "x0_cumsum":
            x0_pred = self._logits_to_cdf_x0(model_out)
            if K > 0:
                nl_past = F.mse_loss(x0_pred[..., :K], future_2d[..., :K])
                nl_fut = F.mse_loss(x0_pred[..., K:], future_2d[..., K:])
                noise_loss = self.config.past_loss_weight * nl_past + nl_fut
            else:
                noise_loss = F.mse_loss(x0_pred, future_2d)
            noise_pred = torch.zeros_like(noise)
        else:
            noise_pred = model_out
            if K > 0:
                nl_past = F.mse_loss(noise_pred[..., :K], noise[..., :K])
                nl_fut  = F.mse_loss(noise_pred[..., K:],  noise[..., K:])
                noise_loss = self.config.past_loss_weight * nl_past + nl_fut
            else:
                noise_loss = F.mse_loss(noise_pred, noise)
            x0_pred = self.scheduler.predict_x0_from_noise(noisy_future, t, noise_pred)
            
            # Clamp x0_pred for numerical stability at high t
            # Matches ddim_step logic: range [-2, 2] provides slack beyond [-1, 1]
            x0_pred = torch.clamp(x0_pred, -2.0, 2.0)

        anchor_loss = torch.tensor(0.0, device=device)
        combined_mse_loss = noise_loss
        anchor_t = None
        anchor_scale = None
        if self.config.use_deterministic_anchor_loss:
            anchor_t, anchor_scale = self._deterministic_anchor_params()
            anchor_t_flat = torch.full(
                (BV,),
                int(anchor_t.item()),
                device=device,
                dtype=t_flat.dtype,
            )
            cfg_scale = self.config.cfg_scale
            zero_future_flat = torch.zeros_like(noisy_future.reshape(BV, 1, H, W_fut))
            null_cond_anchor = (
                torch.zeros_like(base_cond_for_unet) if cfg_scale > 1.0 else None
            )
            null_ctx_anchor = (
                torch.zeros_like(ctx_anchor) if (ctx_anchor is not None and cfg_scale > 1.0) else None
            )
            null_guide_anchor = (
                torch.zeros_like(guidance_2d_flat)
                if (guidance_2d_flat is not None and cfg_scale > 1.0)
                else None
            )
            anchor_pred_flat = self._predict_anchor_noise(
                zero_future_flat,
                anchor_t_flat,
                base_cond_for_unet,
                ctx_anchor,
                guidance_2d_flat,
                cfg_scale,
                null_cond_anchor,
                null_ctx_anchor,
                null_guide_anchor,
            )
            anchor_pred = anchor_pred_flat.reshape(B, V, H, W_fut)
            anchor_target = anchor_scale.to(device=device, dtype=future_2d.dtype) * future_2d
            anchor_loss = F.mse_loss(anchor_pred, anchor_target)
            lam = self.config.deterministic_anchor_lambda
            combined_mse_loss = lam * noise_loss + (1.0 - lam) * anchor_loss
        
        emd_loss = self._compute_emd_loss(x0_pred, future_2d)

        mono_loss = torch.tensor(0.0, device=device)
        if self.config.use_monotonicity_loss and self.config.representation_mode == "cdf":
            cdf_pred  = torch.clamp((x0_pred + 1.0) / 2.0, 0.0, 1.0)
            mono_loss = monotonicity_loss(cdf_pred)

        guidance_loss = torch.tensor(0.0, device=device)
        if guidance_2d is not None and self.config.guidance_penalty_weight > 0:
            guidance_loss = F.mse_loss(x0_pred, guidance_2d)

        loss = (
            combined_mse_loss +
            self.config.emd_lambda * emd_loss + 
            self.config.monotonicity_weight * mono_loss +
            self.config.guidance_penalty_weight * guidance_loss
        )

        result = {
            'loss': loss, 'noise_loss': noise_loss, 'emd_loss': emd_loss,
            'combined_mse_loss': combined_mse_loss, 'anchor_loss': anchor_loss,
            'guidance_loss': guidance_loss,
            'noise_pred': noise_pred, 'model_out': model_out, 'x0_pred': x0_pred, 't': t,
        }
        if anchor_t is not None:
            result['anchor_timestep'] = anchor_t
            result['anchor_scale'] = anchor_scale
        if guidance_2d is not None:
            result['guidance_2d'] = guidance_2d
        return result

    @torch.no_grad()
    def _generate_factorized(self, past: torch.Tensor, use_ddim: bool = True,
                              num_ddim_steps: int = 50, eta: float = 0.0,
                              cfg_scale: Optional[float] = None, verbose: bool = False,
                              decoder_method: str = "mean",
                              sampler: str = "ddim",
                              num_inference_steps: Optional[int] = None,
                              **kwargs) -> Dict[str, torch.Tensor]:
        """inference: per-variate DDIM/DDPM sampling with cross-variate bottleneck context."""
        B = past.shape[0]
        V = self.config.num_variables
        H = self.config.image_height
        device = past.device
        BV = B * V
        if cfg_scale is None:
            cfg_scale = self.config.cfg_scale

        past_norm, _, stats = self._normalize_sequence(past)
        past_2d = self.encode_to_2d(past_norm)
        W_past  = past_2d.shape[3]
        W_fut   = self.config.forecast_length

        # per-variate past visual cond, interpolated to future width
        past_flat     = past_2d.reshape(BV, 1, H, W_past)
        cond_flat     = F.interpolate(past_flat, size=(H, W_fut), mode='bilinear', align_corners=False)
        null_cond     = torch.zeros_like(cond_flat) if cfg_scale > 1.0 else None

        # guidance: compute once before loop
        guidance_forecast_norm = None
        guidance_2d = None
        guide_flat  = None
        if self.config.use_guidance_channel:
            guidance_forecast_norm = self._get_guidance_forecast_norm(past, past_norm, stats, W_fut)
            guidance_2d = self.encode_to_2d(guidance_forecast_norm, scale_for_diffusion=True)
            guide_flat  = guidance_2d.reshape(BV, 1, H, W_fut)
        null_guide = torch.zeros_like(guide_flat) if (guide_flat is not None and cfg_scale > 1.0) else None

        # cross-variate context tokens — fixed for entire sampling trajectory
        ctx = None if getattr(self.config, 'disable_cross_attention', False) else self._get_cross_variate_context(past)
        ctx_flat      = ctx.unsqueeze(1).expand(-1, V, -1, -1).reshape(BV, V, -1) if ctx is not None else None
        null_ctx_flat = torch.zeros_like(ctx_flat) if (ctx_flat is not None and cfg_scale > 1.0) else None

        def _build_canvas(x_noisy, use_null=False):
            c = self._inject_coordinate_channel(x_noisy)
            c = self._inject_time_channels(c)
            if guide_flat is not None:
                c = torch.cat([c, null_guide if use_null else guide_flat], dim=1)
            return c

        def _chunked_model_fn(x_chunk, t_batch_chunk, cond_arg_chunk, ctx_flat_chunk):
            chunk_size = self.config.unet_max_chunk_size
            BV_curr = x_chunk.shape[0]
            if chunk_size > 0 and BV_curr > chunk_size:
                outs = []
                for i in range(0, BV_curr, chunk_size):
                    end = min(i + chunk_size, BV_curr)
                    c_x = x_chunk[i:end]
                    c_t = t_batch_chunk[i:end] if t_batch_chunk.shape[0] == BV_curr else t_batch_chunk
                    c_cond = cond_arg_chunk[i:end] if cond_arg_chunk is not None else None
                    c_ctx = ctx_flat_chunk[i:end] if ctx_flat_chunk is not None else None
                    outs.append(self.noise_predictor(c_x, c_t, c_cond, encoder_hidden_states=c_ctx))
                return torch.cat(outs, dim=0)
            else:
                return self.noise_predictor(x_chunk, t_batch_chunk, cond_arg_chunk, encoder_hidden_states=ctx_flat_chunk)

        def model_fn(x, t_batch, cond_arg):
            if cfg_scale <= 1.0:
                out = _chunked_model_fn(_build_canvas(x), t_batch, cond_arg, ctx_flat)
                return self._logits_to_cdf_x0(out) if self.config.prediction_mode == "x0_cumsum" else out
            # CFG: cond vs uncond pass
            out_c = _chunked_model_fn(_build_canvas(x, use_null=False), t_batch, cond_flat, ctx_flat)
            out_u = _chunked_model_fn(_build_canvas(x, use_null=True),  t_batch, null_cond, null_ctx_flat)
            out = out_u + cfg_scale * (out_c - out_u)
            return self._logits_to_cdf_x0(out) if self.config.prediction_mode == "x0_cumsum" else out

        noise_shape = (BV, 1, H, W_fut)
        scheduler_prediction_mode = "x0" if self.config.prediction_mode == "x0_cumsum" else "epsilon"

        if sampler in ("anchor", "deterministic_anchor"):
            if self.config.prediction_mode == "x0_cumsum":
                raise ValueError("The anchor sampler is not supported with prediction_mode='x0_cumsum'.")
            t_anchor, anchor_scale = self._deterministic_anchor_params()
            t_batch = torch.full(
                (BV,),
                int(t_anchor.item()),
                device=device,
                dtype=torch.long,
            )
            zero_future_flat = torch.zeros(noise_shape, device=device)
            anchor_pred = self._predict_anchor_noise(
                zero_future_flat,
                t_batch,
                cond_flat,
                ctx_flat,
                guide_flat,
                cfg_scale,
                null_cond,
                null_ctx_flat,
                null_guide,
            )
            future_2d_flat = anchor_pred / anchor_scale.to(device=device, dtype=anchor_pred.dtype)
        elif sampler == "dpmpp":
            steps = num_inference_steps if num_inference_steps is not None else 20
            future_2d_flat = self.scheduler.sample_dpmpp(
                model=model_fn, shape=noise_shape, cond=cond_flat,
                num_steps=steps, prediction_mode=scheduler_prediction_mode,
                device=device, verbose=verbose,
            )
        elif use_ddim:
            steps = num_inference_steps if num_inference_steps is not None else num_ddim_steps
            future_2d_flat = self.scheduler.sample_ddim_cfg(
                model=model_fn, shape=noise_shape, cond=cond_flat,
                null_cond=null_cond, cfg_scale=1.0,
                num_steps=steps, eta=eta, prediction_mode=scheduler_prediction_mode,
                device=device, verbose=verbose,
            )
        else:
            future_2d_flat = self.scheduler.sample_ddpm_cfg(
                model=model_fn, shape=noise_shape, cond=cond_flat,
                null_cond=null_cond, cfg_scale=1.0,
                prediction_mode=scheduler_prediction_mode, device=device, verbose=verbose,
            )

        future_2d  = future_2d_flat.reshape(B, V, H, W_fut)
        future_norm = self.decode_from_2d(future_2d, decoder_method=decoder_method, **kwargs)
        future      = self._denormalize(future_norm, stats)

        K = self.config.lookback_overlap
        if K > 0:
            future      = future[..., K:]
            future_norm = future_norm[..., K:]

        result = {
            'prediction': future, 'prediction_norm': future_norm,
            'prediction_global_norm': future,
            'future_2d': future_2d, 'past_2d': past_2d,
        }
        if guidance_2d is not None:
            result['guidance_2d'] = guidance_2d
        return result

    # ====================================================================
    # Binary diffusion — hard CDF images and XOR bit-flip noise
    # ====================================================================

    def _binary_plain_bce_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Unweighted BCE for binary CDF images."""
        return F.binary_cross_entropy_with_logits(logits, target.float())

    def _boundary_bce_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """BCE weighted toward the CDF boundary, where forecast geometry lives."""
        if self.config.diffusion_type == "binary":
            raise ValueError(
                "Edge CDF boundary-weighted BCE is not supported for binary diffusion yet."
            )
        H = target.shape[2]
        bw = self.config.binary_boundary_width
        high_w = self.config.binary_boundary_weight
        low_w = self.config.binary_background_weight

        filled_count = target.sum(dim=2, keepdim=True).long().clamp(0, H - 1)
        row_idx = torch.arange(H, device=target.device).view(1, 1, H, 1)
        dist = (row_idx - filled_count).abs()
        weight = torch.where(
            dist <= bw,
            torch.full_like(dist, high_w, dtype=torch.float),
            torch.full_like(dist, low_w, dtype=torch.float),
        )
        bce = F.binary_cross_entropy_with_logits(logits, target.float(), reduction='none')
        return (bce * weight).mean()

    def _stack_dual_scale_flat(self, coarse: torch.Tensor, fine: torch.Tensor) -> torch.Tensor:
        """Interleave coarse/fine tensors so each (B,V) pair is adjacent in batch."""
        if coarse.shape != fine.shape:
            raise ValueError(f"coarse/fine shapes differ: {coarse.shape} vs {fine.shape}")
        return torch.stack((coarse, fine), dim=1).reshape(coarse.shape[0] * 2, *coarse.shape[1:])

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
        t_bvs = t_bv.unsqueeze(1).expand(-1, 2).reshape(BV * 2)
        scale_indices = self._dual_scale_indices(BV, device)

        future_coarse_flat = future_coarse.reshape(BV, 1, H, W_fut)
        future_fine_flat = future_fine.reshape(BV, 1, H, W_fut)
        xt_coarse, zt_coarse = self.binary_scheduler.add_noise(future_coarse_flat, t_bv)
        xt_fine, zt_fine = self.binary_scheduler.add_noise(future_fine_flat, t_bv)
        xt_flat = self._stack_dual_scale_flat(xt_coarse, xt_fine)
        future_flat = self._stack_dual_scale_flat(future_coarse_flat, future_fine_flat)

        guidance_flat = None
        guidance_coarse = None
        guidance_fine = None
        if self.config.use_guidance_channel:
            guidance_forecast_norm = self._get_guidance_forecast_norm(past, past_norm, stats, W_fut)
            guidance_coarse, guidance_fine = self.encode_dual_to_2d_binary(guidance_forecast_norm)
            guidance_flat = self._stack_dual_scale_flat(
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
        past_flat = self._stack_dual_scale_flat(
            past_coarse.reshape(BV, 1, H, past_tail_len),
            past_fine.reshape(BV, 1, H, past_tail_len),
        )
        cond_for_unet = F.interpolate(past_flat, size=(H, W_fut), mode='bilinear', align_corners=False)
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
            neutral_future_flat = torch.bernoulli(torch.full_like(future_flat, 0.5))
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

        past_norm, _, stats = self._normalize_sequence(past)
        past_tail_len = min(past_norm.shape[-1], W_fut)
        past_tail_norm = past_norm[..., -past_tail_len:]
        past_coarse, past_fine = self.encode_dual_to_2d_binary(past_tail_norm)
        past_flat = self._stack_dual_scale_flat(
            past_coarse.reshape(BV, 1, H, past_tail_len),
            past_fine.reshape(BV, 1, H, past_tail_len),
        )
        cond_for_unet = F.interpolate(past_flat, size=(H, W_fut), mode='bilinear', align_corners=False)

        guidance_coarse = None
        guidance_fine = None
        guide_flat = None
        if self.config.use_guidance_channel:
            guidance_forecast_norm = self._get_guidance_forecast_norm(past, past_norm, stats, W_fut)
            guidance_coarse, guidance_fine = self.encode_dual_to_2d_binary(guidance_forecast_norm)
            guide_flat = self._stack_dual_scale_flat(
                guidance_coarse.reshape(BV, 1, H, W_fut),
                guidance_fine.reshape(BV, 1, H, W_fut),
            )

        ctx = None if getattr(self.config, 'disable_cross_attention', False) else self._get_cross_variate_context(past)
        ctx_flat = self._expand_ctx_to_dual_scale(ctx, B, V)

        def _build_canvas(xt: torch.Tensor) -> torch.Tensor:
            canvas = self._inject_coordinate_channel(xt)
            canvas = self._inject_time_channels(canvas)
            if guide_flat is not None:
                canvas = torch.cat([canvas, guide_flat], dim=1)
            return canvas

        def _chunked_model_fn(xt: torch.Tensor, t_batch: torch.Tensor):
            canvas = _build_canvas(xt)
            out = self._predict_noise_chunked(
                canvas,
                t_batch,
                cond_for_unet,
                ctx_flat,
                scale_indices=scale_indices,
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
            neutral_future_flat = torch.bernoulli(
                torch.full((BVS, 1, H, W_fut), 0.5, device=device)
            )
            x0_logits, _zt_logits = _chunked_model_fn(neutral_future_flat, t_batch)
            future_2d_flat = (torch.sigmoid(x0_logits) > 0.5).float()
            if yield_intermediates:
                intermediates = [(999, neutral_future_flat.clone()), (0, future_2d_flat.clone())]
        else:
            if yield_intermediates:
                future_2d_flat, intermediates = self.binary_scheduler.sample(
                    model_fn=_chunked_model_fn,
                    shape=(BVS, 1, H, W_fut),
                    num_steps=num_steps,
                    device=device,
                    verbose=verbose,
                    yield_intermediates=True,
                )
            else:
                future_2d_flat = self.binary_scheduler.sample(
                    model_fn=_chunked_model_fn,
                    shape=(BVS, 1, H, W_fut),
                    num_steps=num_steps,
                    device=device,
                    verbose=verbose,
                )

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

        out_flat = self._predict_noise_chunked(canvas, t_flat, cond_for_unet, ctx_flat)

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
            neutral_future_flat = torch.bernoulli(torch.full_like(future_flat, 0.5))
            anchor_canvas = self._inject_coordinate_channel(neutral_future_flat)
            anchor_canvas = self._inject_time_channels(anchor_canvas)
            if guidance_2d_flat is not None:
                anchor_canvas = torch.cat([anchor_canvas, guidance_2d_flat], dim=1)
            anchor_out_flat = self._predict_noise_chunked(
                anchor_canvas,
                anchor_t_flat,
                base_cond_for_unet,
                ctx_anchor,
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

        def _build_canvas(xt: torch.Tensor) -> torch.Tensor:
            canvas = self._inject_coordinate_channel(xt)
            canvas = self._inject_time_channels(canvas)
            if guide_flat is not None:
                canvas = torch.cat([canvas, guide_flat], dim=1)
            return canvas

        def _chunked_model_fn(xt: torch.Tensor, t_batch: torch.Tensor):
            canvas = _build_canvas(xt)
            chunk_size = self.config.unet_max_chunk_size
            if chunk_size > 0 and xt.shape[0] > chunk_size:
                outs = []
                for i in range(0, xt.shape[0], chunk_size):
                    end = min(i + chunk_size, xt.shape[0])
                    c_ctx = ctx_flat[i:end] if ctx_flat is not None else None
                    outs.append(
                        self.noise_predictor(
                            canvas[i:end],
                            t_batch[i:end],
                            cond_for_unet[i:end],
                            encoder_hidden_states=c_ctx,
                        )
                    )
                out = torch.cat(outs, dim=0)
            else:
                out = self.noise_predictor(
                    canvas, t_batch, cond_for_unet, encoder_hidden_states=ctx_flat
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
            neutral_future_flat = torch.bernoulli(
                torch.full((BV, 1, H, W_fut), 0.5, device=device)
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
