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

# Handle imports for both module and script execution
try:
    from .config import DiffusionTSFConfig
    from .preprocessing import TimeSeriesTo2D, VerticalGaussianBlur
    from .unet import ConditionalUNet2D, TimeSeriesContextEncoder, VariateCrossEncoder
    from .transformer import DiffusionTransformer
    from .ci_dit import ChannelIndependentDiT
    from .diffusion import DiffusionScheduler, BinaryDiffusionScheduler
    from .guidance import GuidanceModel, LinearRegressionGuidance
    from .metrics import monotonicity_loss
except ImportError:
    from config import DiffusionTSFConfig
    from preprocessing import TimeSeriesTo2D, VerticalGaussianBlur
    from unet import ConditionalUNet2D, TimeSeriesContextEncoder, VariateCrossEncoder
    from transformer import DiffusionTransformer
    from ci_dit import ChannelIndependentDiT
    from diffusion import DiffusionScheduler, BinaryDiffusionScheduler
    from guidance import GuidanceModel, LinearRegressionGuidance
    from metrics import monotonicity_loss

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
            representation_mode=config.representation_mode
        )
        self.blur = VerticalGaussianBlur(
            kernel_size=config.blur_kernel_size,
            sigma=config.blur_sigma
        )
        self.register_buffer(
            "decode_smoothing_kernel",
            self._build_decode_smoothing_kernel(sigma_x=3.0, sigma_y=1.0)
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
        
        if config.model_type == "ci_dit":
            self.noise_predictor = ChannelIndependentDiT(
                image_height=config.image_height,
                patch_size=config.ci_dit_patch_size,
                embed_dim=config.ci_dit_embed_dim,
                depth=config.ci_dit_depth,
                num_heads=config.ci_dit_num_heads,
                mlp_ratio=config.ci_dit_mlp_ratio,
                in_channels=config.ci_dit_in_channels,
                cond_channels=config.ci_dit_cond_channels,
                out_channels=1,
                n_variates=config.num_variables,
                cross_variate_every=config.ci_dit_cross_variate_every,
                dropout=config.ci_dit_dropout,
                gradient_checkpointing=config.use_gradient_checkpointing,
            )
            # CI-DiT does cross-variate attn internally, no separate context encoder
            self.context_encoder = None
        elif config.model_type == "transformer":
            self.noise_predictor = DiffusionTransformer(
                image_height=config.image_height,
                patch_height=config.transformer_patch_height,
                patch_width=config.transformer_patch_width,
                embed_dim=config.transformer_embed_dim,
                depth=config.transformer_depth,
                num_heads=config.transformer_num_heads,
                dropout=config.transformer_dropout,
                in_channels=backbone_in_channels,
                out_channels=config.num_variables,  # Output one channel per variable
            )
            # Note: Transformer backbone does not yet support hybrid conditioning
            self.context_encoder = None
        else:
            # guidance channels only added to the noisy future canvas, not the past cond
            cond_in_channels = backbone_in_channels - config.guidance_channels

            # in factorized mode: unet sees 1 variate at a time → out_channels=1 (gaussian)
            # or out_channels=2 (binary: x0_hat + zt_hat concatenated along channel dim)
            if config.variate_factorized:
                unet_out_channels = 2 if config.diffusion_type == "binary" else 1
            else:
                unet_out_channels = config.num_variables

            self.noise_predictor = ConditionalUNet2D(
                in_channels=backbone_in_channels,
                out_channels=unet_out_channels,
                channels=config.unet_channels,
                num_res_blocks=config.num_res_blocks,
                attention_levels=config.attention_levels,
                image_height=config.image_height,
                kernel_size=config.unet_kernel_size,
                use_dilated_middle=config.use_dilated_middle,
                use_hybrid_condition=config.use_hybrid_condition,
                context_dim=config.context_embedding_dim,
                conditioning_mode=config.conditioning_mode,
                visual_cond_channels=config.visual_cond_channels,
                cond_in_channels=cond_in_channels
            )

            if config.use_hybrid_condition:
                if config.variate_factorized:
                    # VariateCrossEncoder: takes (B, V, T) → (B, V, ctx_dim).
                    # each variate gets a summary token; a small transformer mixes cross-variate
                    # info so the bottleneck actually receives meaningful joint context.
                    self.context_encoder = VariateCrossEncoder(
                        context_dim=config.context_embedding_dim,
                        num_layers=config.context_encoder_layers,
                        num_heads=4,
                        dropout=0.1,
                    )
                else:
                    self.context_encoder = TimeSeriesContextEncoder(
                        input_channels=config.context_input_channels,
                        embedding_dim=config.context_embedding_dim,
                        num_layers=config.context_encoder_layers,
                        num_heads=4,
                        dropout=0.1,
                        max_seq_len=max(config.lookback_length, config.forecast_length) + 256
                    )
            else:
                self.context_encoder = None

        # Diffusion scheduler (not a nn.Module, managed separately)
        self.scheduler = DiffusionScheduler(
            num_steps=config.num_diffusion_steps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            schedule=config.noise_schedule
        )

        # Binary diffusion scheduler — only created when diffusion_type=="binary"
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
        logger.info(f"  Image size: {config.image_height} x W")
        logger.info(f"  Input channels: {config.backbone_in_channels} (data: {config.num_variables}, aux: {config.num_aux_channels}, guidance: {config.guidance_channels})")
        logger.info(f"  Conditioning mode: {config.conditioning_mode}")
        logger.info(f"  Diffusion steps: {config.num_diffusion_steps}")
        logger.info(f"  Coordinate channel: {config.use_coordinate_channel}")
        logger.info(f"  Time ramp channel: {config.use_time_ramp}")
        logger.info(f"  Time sine channel: {config.use_time_sine}")
        logger.info(f"  Value channel: {config.use_value_channel}")
        if config.use_guidance_channel:
            logger.info(f"  Guidance channel: enabled (Stage 1 → 2D ghost image)")
        if config.use_time_sine:
            logger.info(f"  Seasonal period: {config.seasonal_period}")
        if config.model_type == "unet":
            logger.info(f"  U-Net kernel size: {config.unet_kernel_size}")
            if config.variate_factorized and config.num_variables > 1:
                logger.info(f"  Variate-factorized: enabled (B*V={config.num_variables} per forward, shared weights)")
            if config.use_dilated_middle:
                logger.info(f"  Dilated middle block: enabled (dilations=1,2,4,8)")
            if config.use_hybrid_condition:
                enc_type = "VariateCrossEncoder" if config.variate_factorized else "TimeSeriesContextEncoder"
                logger.info(f"  Hybrid conditioning: {enc_type} (context_dim={config.context_embedding_dim}, layers={config.context_encoder_layers})")
    
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
    
    def _prepare_1d_context(
        self,
        past_norm: torch.Tensor
    ) -> torch.Tensor:
        """Prepare 1D context input for the TimeSeriesContextEncoder."""
        if past_norm.dim() == 3:
            past_1d = past_norm[:, 0, :]
        else:
            past_1d = past_norm
        batch_size, seq_len = past_1d.shape
        device = past_1d.device
        dtype = past_1d.dtype
        time_idx = torch.linspace(0.0, 1.0, seq_len, device=device, dtype=dtype)
        time_idx = time_idx.unsqueeze(0).expand(batch_size, -1)
        context_input = torch.stack([past_1d, time_idx], dim=-1)
        return context_input
    
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
        with torch.no_grad():
            coarse = self.guidance_model.get_forecast(past, H)
        coarse_norm = (coarse - mean) / std
        if K > 0:
            coarse_norm = torch.cat([past_norm[..., -K:], coarse_norm], dim=-1)
        return coarse_norm  # (B, V, forecast_length) normalized

    def _get_cross_variate_context(
        self,
        past_norm: torch.Tensor,
        guidance_forecast_norm: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """produce (B, V, ctx_dim) encoder_hidden_states for the bottleneck.

        prefers the iTransformer forecast when available — it already has cross-variate
        structure baked in. falls back to all-variates normalized past otherwise.
        """
        if self.context_encoder is None:
            return None

        if guidance_forecast_norm is not None:
            src = guidance_forecast_norm                    # (B, V, H_forecast)
        elif past_norm.dim() == 3:
            src = past_norm                                 # (B, V, L)
        else:
            src = past_norm.unsqueeze(1)                    # (B, 1, L) univariate edge case

        return self.context_encoder(src)                    # (B, V, ctx_dim)

    def _normalize_sequence(
        self,
        past: torch.Tensor,
        future: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Normalize sequences using past statistics."""
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
    
    def _build_decode_smoothing_kernel(
        self,
        sigma_x: float,
        sigma_y: float
    ) -> torch.Tensor:
        """Create anisotropic Gaussian kernel for decode-time smoothing."""
        size_x = int(6 * sigma_x + 1)
        size_y = int(6 * sigma_y + 1)
        if size_x % 2 == 0: size_x += 1
        if size_y % 2 == 0: size_y += 1
        x = torch.arange(size_x, dtype=torch.float32) - size_x // 2
        y = torch.arange(size_y, dtype=torch.float32) - size_y // 2
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        kernel = torch.exp(-(xx**2 / (2 * sigma_x**2) + yy**2 / (2 * sigma_y**2)))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size_y, size_x)
    
    def _apply_decode_smoothing(self, prob: torch.Tensor) -> torch.Tensor:
        """Apply horizontal-heavy Gaussian smoothing to probability map."""
        kernel = self.decode_smoothing_kernel.to(device=prob.device, dtype=prob.dtype)
        pad_y = kernel.shape[2] // 2
        pad_x = kernel.shape[3] // 2
        prob_2d = prob.unsqueeze(1)
        prob_padded = F.pad(prob_2d, (pad_x, pad_x, pad_y, pad_y), mode="reflect")
        smoothed = F.conv2d(prob_padded, kernel)
        return smoothed.squeeze(1)
    
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
        """Backward compatibility for checkpoints without decode_smoothing_kernel."""
        key = prefix + "decode_smoothing_kernel"
        if key not in state_dict:
            state_dict[key] = self.decode_smoothing_kernel
            if key in missing_keys:
                missing_keys.remove(key)
            logger.warning("decode_smoothing_kernel missing in checkpoint; using default anisotropic kernel.")
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
        """encode 1D series to hard binary CDF image — no blur, stays in {0,1}.

        this is the binary diffusion version of encode_to_2d. skipping the gaussian blur
        is the whole point: binary diffusion provides proper gradients for hard boundaries
        so we don't need to manufacture soft ones.
        """
        return self.to_2d(x)  # already {0,1} float from TimeSeriesTo2D in CDF mode

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
                pdf_temperature=temperature,
                squeeze_univariate=squeeze_output
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
                x = self.to_2d.inverse(
                    cdf_map,
                    cdf_decoder=decoder_method,
                    pdf_temperature=temperature,
                    squeeze_univariate=squeeze_output
                )
                return x
            if num_vars > 1:
                raise NotImplementedError(f"decoder_method='{decoder_method}' not yet supported for multivariate.")
            cdf_map_squeezed = cdf_map.squeeze(1)
            if not self.training and self.config.decode_smoothing and self.decode_smoothing_kernel is not None:
                cdf_map_squeezed = self._apply_decode_smoothing(cdf_map_squeezed)
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
        """Training forward pass using either unified L+F or optimized Future-Only scheme."""
        if self.config.model_type == "ci_dit":
            return self._forward_ci_dit(past, future, t)
        if self.config.variate_factorized and self.config.num_variables > 1:
            if self.config.diffusion_type == "binary":
                return self._forward_binary_factorized(past, future, t)
            return self._forward_factorized(past, future, t)
        
        batch_size = past.shape[0]
        device = past.device
        past_norm, future_norm, stats = self._normalize_sequence(past, future)
        past_2d = self.encode_to_2d(past_norm)
        future_2d = self.encode_to_2d(future_norm)
        past_2d = self._apply_coarse_dropout(past_2d)
        
        past_len = past_2d.shape[-1]
        future_len = future_2d.shape[-1]
        total_len = past_len + future_len
        
        if t is None:
            t = torch.randint(0, self.config.num_diffusion_steps, (batch_size,), device=device)
        
        noisy_future, noise = self.scheduler.add_noise(future_2d, t)
        
        encoder_hidden_states = None
        if self.context_encoder is not None:
            context_input = self._prepare_1d_context(past_norm)
            encoder_hidden_states = self.context_encoder(context_input)
            
        # Guidance Generation (Common)
        guidance_2d = None
        if self.config.use_guidance_channel:
            guidance_2d = self._generate_guidance_2d(past, past_norm, stats, future_len)

        if self.config.unified_time_axis:
            # === UNIFIED MODE (Slower, Better Continuity) ===
            # Diffuse on full L+F width
            
            past_padded = self._pad_to_window(past_2d, 'past', total_len)
            noisy_future_padded = self._pad_to_window(noisy_future, 'future', total_len)
            canvas = past_padded + noisy_future_padded
            
            canvas = self._inject_coordinate_channel(canvas)
            canvas = self._inject_time_channels(canvas)
            
            val_channel = None
            if self.config.use_value_channel:
                if past_norm.dim() == 3: past_vals = past_norm
                else: past_vals = past_norm.unsqueeze(1)
                vals_padded = F.pad(past_vals, (0, future_len))
                val_channel = self._get_value_channel(vals_padded, self.config.image_height)
                if val_channel.shape[1] > 1: val_channel = val_channel[:, 0:1, :, :]
                canvas = torch.cat([canvas, val_channel], dim=1)
                
            if guidance_2d is not None:
                guidance_padded = self._pad_to_window(guidance_2d, 'future', total_len)
                canvas = self._inject_guidance_channel(canvas, guidance_padded)
            
            # Conditioning: past_padded
            cond_for_unet = past_padded
            if val_channel is not None:
                 cond_for_unet = torch.cat([cond_for_unet, val_channel], dim=1)

            noise_pred_full = self.noise_predictor(
                canvas, t, cond_for_unet,
                encoder_hidden_states=encoder_hidden_states
            )
            noise_pred = noise_pred_full[..., past_len:]
            
        else:
            # === OPTIMIZED MODE (Faster) ===
            # Diffuse on Future width only
            
            canvas = noisy_future
            # print(f"DEBUG: Canvas start: {canvas.shape}")
            canvas = self._inject_coordinate_channel(canvas)
            # print(f"DEBUG: Canvas after coord: {canvas.shape}")
            canvas = self._inject_time_channels(canvas)
            # print(f"DEBUG: Canvas after time: {canvas.shape}")
            
            val_channel = None
            if self.config.use_value_channel:
                # Broadcast last past value as reference
                if past_norm.dim() == 3: 
                    last_val = past_norm[:, :, -1:]
                    last_val_expanded = last_val.expand(-1, -1, future_len)
                else: 
                    last_val = past_norm[:, -1:]
                    last_val_expanded = last_val.expand(-1, future_len)
                
                val_channel = self._get_value_channel(last_val_expanded, self.config.image_height)
                if val_channel.shape[1] > 1: val_channel = val_channel[:, 0:1, :, :]
                canvas = torch.cat([canvas, val_channel], dim=1)
                # print(f"DEBUG: Canvas after value: {canvas.shape}")
            
            if guidance_2d is not None:
                canvas = self._inject_guidance_channel(canvas, guidance_2d)
                
            # Conditioning: Visual part of past matched to future width
            cond_for_unet = self._prepare_visual_conditioning(past_2d, target_width=future_len)
            if val_channel is not None:
                 cond_for_unet = torch.cat([cond_for_unet, val_channel], dim=1)
            
            # print(f"DEBUG: Final canvas to UNet: {canvas.shape}, expected channels: {self.config.backbone_in_channels}")
            noise_pred = self.noise_predictor(
                canvas, t, cond_for_unet,
                encoder_hidden_states=encoder_hidden_states
            )
        
        K = self.config.lookback_overlap
        if K > 0:
            noise_loss_past = F.mse_loss(noise_pred[..., :K], noise[..., :K])
            noise_loss_future = F.mse_loss(noise_pred[..., K:], noise[..., K:])
            noise_loss = self.config.past_loss_weight * noise_loss_past + noise_loss_future
        else:
            noise_loss = F.mse_loss(noise_pred, noise)

        x0_pred = self.scheduler.predict_x0_from_noise(noisy_future, t, noise_pred)
        emd_loss = self._compute_emd_loss(x0_pred, future_2d)
        
        mono_loss = torch.tensor(0.0, device=device)
        if self.config.use_monotonicity_loss and self.config.representation_mode == "cdf":
            cdf_pred = torch.clamp((x0_pred + 1.0) / 2.0, 0.0, 1.0)
            mono_loss = monotonicity_loss(cdf_pred)
            
        loss = noise_loss + self.config.emd_lambda * emd_loss + self.config.monotonicity_weight * mono_loss
        
        result = {
            'loss': loss,
            'noise_loss': noise_loss,
            'emd_loss': emd_loss,
            'noise_pred': noise_pred,
            't': t
        }
        if guidance_2d is not None:
            result['guidance_2d'] = guidance_2d
        return result

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
        search_radius: int = 10
    ) -> Dict[str, torch.Tensor]:
        """Generate future predictions using unified (L+F) or optimized Future-Only scheme."""
        if self.config.model_type == "ci_dit":
            return self._generate_ci_dit(
                past, use_ddim=use_ddim, num_ddim_steps=num_ddim_steps,
                eta=eta, cfg_scale=cfg_scale, verbose=verbose,
                decoder_method=decoder_method, beam_width=beam_width,
                jump_penalty_scale=jump_penalty_scale, search_radius=search_radius,
            )
        if self.config.variate_factorized and self.config.num_variables > 1:
            if self.config.diffusion_type == "binary":
                return self._generate_binary_factorized(
                    past, num_steps=self.config.binary_sample_steps,
                    verbose=verbose, decoder_method=decoder_method,
                    beam_width=beam_width, jump_penalty_scale=jump_penalty_scale,
                    search_radius=search_radius,
                )
            return self._generate_factorized(
                past, use_ddim=use_ddim, num_ddim_steps=num_ddim_steps,
                eta=eta, cfg_scale=cfg_scale, verbose=verbose,
                decoder_method=decoder_method, beam_width=beam_width,
                jump_penalty_scale=jump_penalty_scale, search_radius=search_radius,
            )
        
        batch_size = past.shape[0]
        device = past.device
        if cfg_scale is None: cfg_scale = self.config.cfg_scale
        past_norm, _, stats = self._normalize_sequence(past)
        past_2d = self.encode_to_2d(past_norm)
        past_len = past_2d.shape[-1]
        future_len = self.config.forecast_length
        total_len = past_len + future_len
        
        # Prepare Conditioning (Common)
        encoder_hidden_states = None
        null_encoder_hidden_states = None
        if self.context_encoder is not None:
            context_input = self._prepare_1d_context(past_norm)
            encoder_hidden_states = self.context_encoder(context_input)
            if cfg_scale > 1.0:
                null_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)

        guidance_2d = None
        if self.config.use_guidance_channel:
            guidance_2d = self._generate_guidance_2d(past, past_norm, stats, future_len)

        # === Mode-Specific Setup ===
        if self.config.unified_time_axis:
            # UNIFIED MODE SETUP
            past_padded = self._pad_to_window(past_2d, 'past', total_len)
            
            val_channel = None
            if self.config.use_value_channel:
                if past_norm.dim() == 3: past_vals = past_norm
                else: past_vals = past_norm.unsqueeze(1)
                vals_padded = F.pad(past_vals, (0, future_len))
                val_channel = self._get_value_channel(vals_padded, self.config.image_height)
                if val_channel.shape[1] > 1: val_channel = val_channel[:, 0:1, :, :]
            
            cond_for_unet = past_padded
            if val_channel is not None:
                 cond_for_unet = torch.cat([cond_for_unet, val_channel], dim=1)
            
            guidance_padded = None
            null_guidance_padded = None
            if guidance_2d is not None:
                guidance_padded = self._pad_to_window(guidance_2d, 'future', total_len)
                if cfg_scale > 1.0:
                    null_guidance_padded = torch.zeros_like(guidance_padded)
        else:
            # OPTIMIZED MODE SETUP
            # Value Channel: Broadcast last value
            val_channel = None
            if self.config.use_value_channel:
                if past_norm.dim() == 3: last_val = past_norm[:, :, -1:]
                else: last_val = past_norm[:, -1:]
                
                if past_norm.dim() == 3:
                    last_val_expanded = last_val.expand(-1, -1, future_len)
                else:
                    last_val_expanded = last_val.expand(-1, future_len)
                    
                val_channel = self._get_value_channel(last_val_expanded, self.config.image_height)
                if val_channel.shape[1] > 1: val_channel = val_channel[:, 0:1, :, :]

            # Conditioning: Visual part matched to future width
            cond_for_unet = self._prepare_visual_conditioning(past_2d, target_width=future_len)
            if val_channel is not None:
                 cond_for_unet = torch.cat([cond_for_unet, val_channel], dim=1)
            
            guidance_padded = guidance_2d # Reuse variable name for simplicity
            null_guidance_padded = None
            if guidance_2d is not None and cfg_scale > 1.0:
                 null_guidance_padded = torch.zeros_like(guidance_2d)

        # Null cond for visual concatenation (zeros)
        null_cond_for_unet = None
        if cfg_scale > 1.0:
            null_cond = torch.zeros_like(cond_for_unet)
            null_cond_for_unet = null_cond
        
        noise_shape = (batch_size, self.config.num_variables, self.config.image_height, future_len)
        
        def model_fn(x_future, t, cond, use_null_context=False, use_null_guidance=False):
            if self.config.unified_time_axis:
                # Unified Mode Construction
                x_future_padded = self._pad_to_window(x_future, 'future', total_len)
                canvas = past_padded + x_future_padded
                canvas_full = self._inject_coordinate_channel(canvas)
                canvas_full = self._inject_time_channels(canvas_full)
                if val_channel is not None:
                    canvas_full = torch.cat([canvas_full, val_channel], dim=1)
                if self.config.use_guidance_channel:
                    guide = null_guidance_padded if use_null_guidance else guidance_padded
                    canvas_full = self._inject_guidance_channel(canvas_full, guide)
                
                ctx = null_encoder_hidden_states if use_null_context else encoder_hidden_states
                out_full = self.noise_predictor(canvas_full, t, cond, encoder_hidden_states=ctx)
                return out_full[..., past_len:]
            else:
                # Optimized Mode Construction
                canvas = x_future
                canvas_full = self._inject_coordinate_channel(canvas)
                canvas_full = self._inject_time_channels(canvas_full)
                if val_channel is not None:
                    canvas_full = torch.cat([canvas_full, val_channel], dim=1)
                if self.config.use_guidance_channel:
                    guide = null_guidance_padded if use_null_guidance else guidance_padded
                    canvas_full = self._inject_guidance_channel(canvas_full, guide)
                
                ctx = null_encoder_hidden_states if use_null_context else encoder_hidden_states
                out = self.noise_predictor(canvas_full, t, cond, encoder_hidden_states=ctx)
                return out
            
        def model_cfg(x, t, cond, null_cond, cfg_scale):
            if cfg_scale <= 1.0: return model_fn(x, t, cond)
            
            # For efficiency in optimized mode, we can batch; 
            # for unified mode, batching is also possible but consumes more memory.
            # Here we just execute twice for simplicity and OOM safety.
            cond_out = model_fn(x, t, cond, use_null_context=False, use_null_guidance=False)
            uncond_out = model_fn(x, t, null_cond, use_null_context=True, use_null_guidance=True)
            return uncond_out + cfg_scale * (cond_out - uncond_out)

        if use_ddim:
            future_2d = self.scheduler.sample_ddim_cfg(
                model=lambda x, t, c: model_cfg(x, t, c, null_cond_for_unet, cfg_scale),
                shape=noise_shape,
                cond=cond_for_unet,
                null_cond=null_cond_for_unet,
                cfg_scale=1.0, # Handled inside model_cfg
                num_steps=num_ddim_steps,
                eta=eta,
                device=device,
                verbose=verbose
            )
        else:
             future_2d = self.scheduler.sample_ddpm_cfg(
                model=lambda x, t, c: model_cfg(x, t, c, null_cond_for_unet, cfg_scale),
                shape=noise_shape,
                cond=cond_for_unet,
                null_cond=null_cond_for_unet,
                cfg_scale=1.0, # Handled inside model_cfg
                device=device,
                verbose=verbose
            )

        future_norm = self.decode_from_2d(
            future_2d,
            decoder_method=decoder_method,
            beam_width=beam_width,
            jump_penalty_scale=jump_penalty_scale,
            search_radius=search_radius
        )
        future = self._denormalize(future_norm, stats)

        # Discard the reconstructed overlap, keep only the real forecast
        K = self.config.lookback_overlap
        if K > 0:
            future = future[..., K:]
            future_norm = future_norm[..., K:]

        result = {
            'prediction': future,
            'prediction_norm': future_norm,
            'future_2d': future_2d,
            'past_2d': past_2d
        }
        if guidance_2d is not None: result['guidance_2d'] = guidance_2d
        return result
    
    # ====================================================================
    # Factorized U-Net forward/generate — per-variate shared-weight U-Net
    # with cross-variate context at the bottleneck via VariateCrossEncoder
    # ====================================================================

    def _forward_factorized(self, past: torch.Tensor, future: torch.Tensor, t: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """training forward: each variate's occupancy map denoised independently.

        the U-Net weights are shared across all V variates. cross-variate info
        is injected at the bottleneck via cross-attention on V context tokens
        produced by VariateCrossEncoder (from iTransformer output or mean past).
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

        ctx = self._get_cross_variate_context(past_norm, guidance_forecast_norm)
        # ctx: (B, V, ctx_dim) or None

        # flatten variates into batch dim for shared-weight U-Net
        # t: same timestep for all V variates of each batch element
        t_flat = t.unsqueeze(1).expand(-1, V).reshape(BV)  # (BV,)

        canvas = noisy_future.reshape(BV, 1, H, W_fut)
        canvas = self._inject_coordinate_channel(canvas)
        canvas = self._inject_time_channels(canvas)

        if guidance_2d is not None:
            canvas = torch.cat([canvas, guidance_2d.reshape(BV, 1, H, W_fut)], dim=1)

        # visual cond: per-variate past bilinearly resized to match future width
        past_flat     = past_2d.reshape(BV, 1, H, W_past)
        cond_for_unet = F.interpolate(past_flat, size=(H, W_fut), mode='bilinear', align_corners=False)

        # broadcast context: every one of the BV U-Net forward passes sees ALL V tokens
        ctx_flat = None
        if ctx is not None:
            # (B, V, ctx_dim) → (BV, V, ctx_dim)
            ctx_flat = ctx.unsqueeze(1).expand(-1, V, -1, -1).reshape(BV, V, -1)

        noise_pred_flat = self.noise_predictor(canvas, t_flat, cond_for_unet, encoder_hidden_states=ctx_flat)
        noise_pred = noise_pred_flat.reshape(B, V, H, W_fut)

        K = self.config.lookback_overlap
        if K > 0:
            nl_past = F.mse_loss(noise_pred[..., :K], noise[..., :K])
            nl_fut  = F.mse_loss(noise_pred[..., K:],  noise[..., K:])
            noise_loss = self.config.past_loss_weight * nl_past + nl_fut
        else:
            noise_loss = F.mse_loss(noise_pred, noise)

        x0_pred  = self.scheduler.predict_x0_from_noise(noisy_future, t, noise_pred)
        emd_loss = self._compute_emd_loss(x0_pred, future_2d)

        mono_loss = torch.tensor(0.0, device=device)
        if self.config.use_monotonicity_loss and self.config.representation_mode == "cdf":
            cdf_pred  = torch.clamp((x0_pred + 1.0) / 2.0, 0.0, 1.0)
            mono_loss = monotonicity_loss(cdf_pred)

        loss = noise_loss + self.config.emd_lambda * emd_loss + self.config.monotonicity_weight * mono_loss

        result = {
            'loss': loss, 'noise_loss': noise_loss, 'emd_loss': emd_loss,
            'noise_pred': noise_pred, 't': t,
        }
        if guidance_2d is not None:
            result['guidance_2d'] = guidance_2d
        return result

    @torch.no_grad()
    def _generate_factorized(self, past: torch.Tensor, use_ddim: bool = True,
                              num_ddim_steps: int = 50, eta: float = 0.0,
                              cfg_scale: Optional[float] = None, verbose: bool = False,
                              decoder_method: str = "mean", **kwargs) -> Dict[str, torch.Tensor]:
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

        # shared coord channel (constant across denoising steps)
        coord = None
        if self.config.use_coordinate_channel:
            coord = self._get_coordinate_grid(BV, H, W_fut, device, dtype=cond_flat.dtype)

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
        ctx = self._get_cross_variate_context(past_norm, guidance_forecast_norm)
        ctx_flat      = ctx.unsqueeze(1).expand(-1, V, -1, -1).reshape(BV, V, -1) if ctx is not None else None
        null_ctx_flat = torch.zeros_like(ctx_flat) if (ctx_flat is not None and cfg_scale > 1.0) else None

        def _build_canvas(x_noisy, use_null=False):
            parts = [x_noisy]
            if coord is not None:
                parts.append(coord)
            if guide_flat is not None:
                parts.append(null_guide if use_null else guide_flat)
            return torch.cat(parts, dim=1)

        def model_fn(x, t_batch, cond_arg):
            if cfg_scale <= 1.0:
                return self.noise_predictor(_build_canvas(x), t_batch, cond_arg, encoder_hidden_states=ctx_flat)
            # CFG: cond vs uncond pass
            out_c = self.noise_predictor(_build_canvas(x, use_null=False), t_batch, cond_flat,  encoder_hidden_states=ctx_flat)
            out_u = self.noise_predictor(_build_canvas(x, use_null=True),  t_batch, null_cond,  encoder_hidden_states=null_ctx_flat)
            return out_u + cfg_scale * (out_c - out_u)

        noise_shape = (BV, 1, H, W_fut)

        if use_ddim:
            future_2d_flat = self.scheduler.sample_ddim_cfg(
                model=model_fn, shape=noise_shape, cond=cond_flat,
                null_cond=null_cond, cfg_scale=1.0,
                num_steps=num_ddim_steps, eta=eta, device=device, verbose=verbose,
            )
        else:
            future_2d_flat = self.scheduler.sample_ddpm_cfg(
                model=model_fn, shape=noise_shape, cond=cond_flat,
                null_cond=null_cond, cfg_scale=1.0, device=device, verbose=verbose,
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
            'future_2d': future_2d, 'past_2d': past_2d,
        }
        if guidance_2d is not None:
            result['guidance_2d'] = guidance_2d
        return result

    # ====================================================================
    # Binary diffusion (BDPM-inspired) — per-variate, no gaussian blur
    # ====================================================================

    def _boundary_bce_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """spatially-weighted BCE that upweights pixels near the CDF boundary.

        the CDF image is bottom-filled (row 0 = 1, rows above boundary = 0).
        the boundary is where the column transitions from 1→0.
        pixels within `binary_boundary_width` rows of that boundary get
        weight 1.0; everything else gets 0.1. analogous to BDPM bit-plane weighting.

        Args:
            logits: (BV, 1, H, W) raw model output (pre-sigmoid)
            target: (BV, 1, H, W) binary ground truth {0,1}
        """
        H = target.shape[2]
        bw = self.config.binary_boundary_width
        high_w = self.config.binary_boundary_weight
        low_w  = self.config.binary_background_weight

        # number of filled rows per column = position of boundary
        # filled_count shape: (BV, 1, W)
        filled_count = target.sum(dim=2, keepdim=True).long().clamp(0, H - 1)

        # row indices: (1, 1, H, 1) for broadcasting
        row_idx = torch.arange(H, device=target.device).view(1, 1, H, 1)

        # distance of each row from the boundary row
        dist = (row_idx - filled_count.unsqueeze(2)).abs()  # (BV, 1, H, W)

        weight = torch.where(dist <= bw,
                             torch.full_like(dist, high_w, dtype=torch.float),
                             torch.full_like(dist, low_w,  dtype=torch.float))

        bce = F.binary_cross_entropy_with_logits(logits, target.float(), reduction='none')
        return (bce * weight).mean()

    def _forward_binary_factorized(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """binary diffusion training: XOR noise, predict x0 + zt, boundary-weighted BCE.

        no gaussian blur on the occupancy images. the U-Net now outputs 2 channels
        (x0_logits | zt_logits) instead of 1 (noise estimate).
        cross-variate context at the bottleneck is the same as the gaussian path.
        """
        assert self.binary_scheduler is not None, "binary_scheduler not initialized"

        B = past.shape[0]
        V = self.config.num_variables
        H = self.config.image_height
        device = past.device
        BV = B * V

        past_norm, future_norm, stats = self._normalize_sequence(past, future)

        # hard binary images — no blur at all
        past_2d   = self.encode_to_2d_binary(past_norm)    # (B, V, H, W_past) {0,1}
        future_2d = self.encode_to_2d_binary(future_norm)   # (B, V, H, W_fut) {0,1}
        past_2d   = self._apply_coarse_dropout(past_2d)

        W_past = past_2d.shape[3]
        W_fut  = future_2d.shape[3]

        if t is None:
            t = torch.randint(0, self.config.binary_num_steps, (B,), device=device)

        # broadcast t across variates for per-variate noise (same timestep per batch el)
        t_flat = t.unsqueeze(1).expand(-1, V).reshape(BV)  # (BV,)

        # per-variate XOR noise
        future_flat = future_2d.reshape(BV, 1, H, W_fut)
        xt_flat, zt_flat = self.binary_scheduler.add_noise(future_flat, t_flat)
        # xt_flat, zt_flat: (BV, 1, H, W_fut) {0,1}

        # guidance: binary ghost image from iTransformer (no blur)
        guidance_forecast_norm = None
        guidance_2d = None
        if self.config.use_guidance_channel:
            guidance_forecast_norm = self._get_guidance_forecast_norm(past, past_norm, stats, W_fut)
            guidance_2d = self.encode_to_2d_binary(guidance_forecast_norm)  # {0,1}, no blur

        ctx = self._get_cross_variate_context(past_norm, guidance_forecast_norm)

        # build canvas — same channel layout as gaussian factorized path
        canvas = xt_flat.float()
        canvas = self._inject_coordinate_channel(canvas)
        canvas = self._inject_time_channels(canvas)

        if guidance_2d is not None:
            canvas = torch.cat([canvas, guidance_2d.reshape(BV, 1, H, W_fut)], dim=1)

        past_flat     = past_2d.reshape(BV, 1, H, W_past)
        cond_for_unet = F.interpolate(past_flat, size=(H, W_fut), mode='bilinear', align_corners=False)

        ctx_flat = None
        if ctx is not None:
            ctx_flat = ctx.unsqueeze(1).expand(-1, V, -1, -1).reshape(BV, V, -1)

        # forward — output is (BV, 2, H, W_fut): first ch=x0_logits, second=zt_logits
        out_flat = self.noise_predictor(canvas, t_flat, cond_for_unet, encoder_hidden_states=ctx_flat)
        x0_logits = out_flat[:, 0:1, :, :]   # (BV, 1, H, W_fut)
        zt_logits  = out_flat[:, 1:2, :, :]   # (BV, 1, H, W_fut)

        # boundary-weighted BCE for both predictions
        loss_x0 = self._boundary_bce_loss(x0_logits, future_flat)
        loss_zt = self._boundary_bce_loss(zt_logits,  zt_flat)
        loss    = loss_x0 + loss_zt

        # x0_hat for emd/mono (optional — reshape back for consistency)
        x0_hat = torch.sigmoid(x0_logits).reshape(B, V, H, W_fut)

        return {
            'loss': loss,
            'loss_x0': loss_x0,
            'loss_zt': loss_zt,
            'noise_loss': loss,  # keep key name compatible with training loop
            'emd_loss': torch.tensor(0.0, device=device),
            'noise_pred': x0_hat,  # expose x0 hat for logging convenience
            't': t,
        }

    @torch.no_grad()
    def _generate_binary_factorized(
        self,
        past: torch.Tensor,
        num_steps: int = 20,
        verbose: bool = False,
        decoder_method: str = "mean",
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """binary diffusion inference: bit-flip reverse from random binary → clean CDF image."""
        assert self.binary_scheduler is not None, "binary_scheduler not initialized"

        B = past.shape[0]
        V = self.config.num_variables
        H = self.config.image_height
        device = past.device
        BV = B * V
        W_fut = self.config.forecast_length

        past_norm, _, stats = self._normalize_sequence(past)
        past_2d = self.encode_to_2d_binary(past_norm)
        W_past  = past_2d.shape[3]

        past_flat     = past_2d.reshape(BV, 1, H, W_past)
        cond_for_unet = F.interpolate(past_flat, size=(H, W_fut), mode='bilinear', align_corners=False)

        coord = None
        if self.config.use_coordinate_channel:
            coord = self._get_coordinate_grid(BV, H, W_fut, device, dtype=cond_for_unet.dtype)

        guidance_forecast_norm = None
        guidance_2d = None
        guide_flat  = None
        if self.config.use_guidance_channel:
            guidance_forecast_norm = self._get_guidance_forecast_norm(past, past_norm, stats, W_fut)
            guidance_2d = self.encode_to_2d_binary(guidance_forecast_norm)
            guide_flat  = guidance_2d.reshape(BV, 1, H, W_fut)

        ctx = self._get_cross_variate_context(past_norm, guidance_forecast_norm)
        ctx_flat = ctx.unsqueeze(1).expand(-1, V, -1, -1).reshape(BV, V, -1) if ctx is not None else None

        def model_fn(xt, t_batch):
            parts = [xt]
            if coord is not None:
                parts.append(coord)
            if guide_flat is not None:
                parts.append(guide_flat)
            canvas = torch.cat(parts, dim=1)
            out = self.noise_predictor(canvas, t_batch, cond_for_unet, encoder_hidden_states=ctx_flat)
            return out[:, 0:1], out[:, 1:2]  # (x0_logits, zt_logits)

        noise_shape  = (BV, 1, H, W_fut)
        future_2d_flat = self.binary_scheduler.sample(
            model_fn=model_fn,
            shape=noise_shape,
            num_steps=num_steps,
            device=device,
            verbose=verbose,
        )

        # decode: binary {0,1} image — use from_diffusion=False so we don't rescale
        future_2d   = future_2d_flat.reshape(B, V, H, W_fut)
        future_norm = self.decode_from_2d(future_2d, from_diffusion=False, decoder_method=decoder_method, **kwargs)
        future      = self._denormalize(future_norm, stats)

        K = self.config.lookback_overlap
        if K > 0:
            future      = future[..., K:]
            future_norm = future_norm[..., K:]

        result = {
            'prediction': future, 'prediction_norm': future_norm,
            'future_2d': future_2d, 'past_2d': past_2d,
        }
        if guidance_2d is not None:
            result['guidance_2d'] = guidance_2d
        return result

    # ====================================================================
    # CI-DiT specific forward/generate — channel-independent processing
    # ====================================================================

    def _forward_ci_dit(self, past, future, t=None):
        """CI-DiT training forward: process each variate independently."""
        B = past.shape[0]
        V = self.config.num_variables
        device = past.device
        H = self.config.image_height

        past_norm, future_norm, stats = self._normalize_sequence(past, future)
        past_2d = self.encode_to_2d(past_norm)      # (B, V, H, W_past)
        future_2d = self.encode_to_2d(future_norm)   # (B, V, H, W_fut)
        past_2d = self._apply_coarse_dropout(past_2d)

        W_past = past_2d.shape[-1]
        W_fut = future_2d.shape[-1]

        if t is None:
            t = torch.randint(0, self.config.num_diffusion_steps, (B,), device=device)

        noisy_future, noise = self.scheduler.add_noise(future_2d, t)

        # --- build per-variate input: (B*V, C_per_var, H, W_fut) ---
        noisy_flat = noisy_future.reshape(B * V, 1, H, W_fut)

        channels = [noisy_flat]
        if self.config.use_coordinate_channel:
            coord = self._get_coordinate_grid(1, H, W_fut, device, dtype=noisy_flat.dtype)
            channels.append(coord.expand(B * V, -1, -1, -1))

        guidance_2d = None
        if self.config.use_guidance_channel:
            guidance_2d = self._generate_guidance_2d(past, past_norm, stats, W_fut)
            channels.append(guidance_2d.reshape(B * V, 1, H, W_fut))

        x_flat = torch.cat(channels, dim=1)  # (BV, ci_dit_in_channels, H, W_fut)

        # --- conditioning: resize past 2D per variate to match future width ---
        past_flat = past_2d.reshape(B * V, 1, H, W_past)
        cond_flat = F.interpolate(past_flat, size=(H, W_fut), mode='bilinear', align_corners=False)

        # --- run CI-DiT backbone ---
        noise_pred_flat = self.noise_predictor(x_flat, t, cond_flat)
        noise_pred = noise_pred_flat.reshape(B, V, H, W_fut)

        # --- loss (same as standard path) ---
        K = self.config.lookback_overlap
        if K > 0:
            noise_loss_past = F.mse_loss(noise_pred[..., :K], noise[..., :K])
            noise_loss_future = F.mse_loss(noise_pred[..., K:], noise[..., K:])
            noise_loss = self.config.past_loss_weight * noise_loss_past + noise_loss_future
        else:
            noise_loss = F.mse_loss(noise_pred, noise)

        x0_pred = self.scheduler.predict_x0_from_noise(noisy_future, t, noise_pred)
        emd_loss = self._compute_emd_loss(x0_pred, future_2d)

        mono_loss = torch.tensor(0.0, device=device)
        if self.config.use_monotonicity_loss and self.config.representation_mode == "cdf":
            cdf_pred = torch.clamp((x0_pred + 1.0) / 2.0, 0.0, 1.0)
            mono_loss = monotonicity_loss(cdf_pred)

        loss = noise_loss + self.config.emd_lambda * emd_loss + self.config.monotonicity_weight * mono_loss

        result = {
            'loss': loss, 'noise_loss': noise_loss, 'emd_loss': emd_loss,
            'noise_pred': noise_pred, 't': t,
        }
        if guidance_2d is not None:
            result['guidance_2d'] = guidance_2d
        return result

    @torch.no_grad()
    def _generate_ci_dit(self, past, use_ddim=True, num_ddim_steps=50, eta=0.0,
                         cfg_scale=None, verbose=False, decoder_method="mean", **kwargs):
        """CI-DiT generation path."""
        B = past.shape[0]
        V = self.config.num_variables
        device = past.device
        H = self.config.image_height
        if cfg_scale is None:
            cfg_scale = self.config.cfg_scale

        past_norm, _, stats = self._normalize_sequence(past)
        past_2d = self.encode_to_2d(past_norm)
        W_past = past_2d.shape[-1]
        W_fut = self.config.forecast_length

        # conditioning: per-variate past resized to future width
        past_flat = past_2d.reshape(B * V, 1, H, W_past)
        cond_flat = F.interpolate(past_flat, size=(H, W_fut), mode='bilinear', align_corners=False)

        # shared coordinate channel
        coord = None
        if self.config.use_coordinate_channel:
            coord = self._get_coordinate_grid(1, H, W_fut, device, dtype=cond_flat.dtype)
            coord = coord.expand(B * V, -1, -1, -1)

        # per-variate guidance
        guidance_2d = None
        guide_flat = None
        if self.config.use_guidance_channel:
            guidance_2d = self._generate_guidance_2d(past, past_norm, stats, W_fut)
            guide_flat = guidance_2d.reshape(B * V, 1, H, W_fut)

        null_cond = torch.zeros_like(cond_flat) if cfg_scale > 1.0 else None
        null_guide = torch.zeros_like(guide_flat) if (guide_flat is not None and cfg_scale > 1.0) else None

        def _build_x(x_noisy, use_null=False):
            parts = [x_noisy]
            if coord is not None:
                parts.append(coord)
            if guide_flat is not None:
                parts.append(null_guide if use_null else guide_flat)
            return torch.cat(parts, dim=1)

        def model_fn(x, t_batch, cond_arg):
            if cfg_scale <= 1.0:
                inp = _build_x(x)
                return self.noise_predictor(inp, t_batch, cond_arg)
            # CFG: two passes
            out_c = self.noise_predictor(_build_x(x, use_null=False), t_batch, cond_flat)
            out_u = self.noise_predictor(_build_x(x, use_null=True), t_batch, null_cond)
            return out_u + cfg_scale * (out_c - out_u)

        noise_shape = (B * V, 1, H, W_fut)

        if use_ddim:
            future_2d_flat = self.scheduler.sample_ddim_cfg(
                model=model_fn, shape=noise_shape, cond=cond_flat,
                null_cond=null_cond, cfg_scale=1.0,
                num_steps=num_ddim_steps, eta=eta, device=device, verbose=verbose,
            )
        else:
            future_2d_flat = self.scheduler.sample_ddpm_cfg(
                model=model_fn, shape=noise_shape, cond=cond_flat,
                null_cond=null_cond, cfg_scale=1.0, device=device, verbose=verbose,
            )

        future_2d = future_2d_flat.reshape(B, V, H, W_fut)
        future_norm = self.decode_from_2d(future_2d, decoder_method=decoder_method, **kwargs)
        future = self._denormalize(future_norm, stats)

        K = self.config.lookback_overlap
        if K > 0:
            future = future[..., K:]
            future_norm = future_norm[..., K:]

        result = {
            'prediction': future, 'prediction_norm': future_norm,
            'future_2d': future_2d, 'past_2d': past_2d,
        }
        if guidance_2d is not None:
            result['guidance_2d'] = guidance_2d
        return result

    def get_loss(
        self,
        past: torch.Tensor,
        future: torch.Tensor
    ) -> torch.Tensor:
        """Convenience method to get just the loss for training."""
        outputs = self.forward(past, future)
        return outputs['loss']
