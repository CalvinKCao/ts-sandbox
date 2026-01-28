"""
Complete Diffusion-based Time Series Forecasting Model.

This module integrates:
- Preprocessing (normalization, 2D encoding, blur)
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
    from .unet import ConditionalUNet2D, TimeSeriesContextEncoder
    from .transformer import DiffusionTransformer
    from .diffusion import DiffusionScheduler
    from .guidance import GuidanceModel, LinearRegressionGuidance
    from .metrics import monotonicity_loss
except ImportError:
    from config import DiffusionTSFConfig
    from preprocessing import TimeSeriesTo2D, VerticalGaussianBlur
    from unet import ConditionalUNet2D, TimeSeriesContextEncoder
    from transformer import DiffusionTransformer
    from diffusion import DiffusionScheduler
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
    """Beam search decoder for CDF/occupancy maps with temporal continuity.
    
    Finds a continuous path through the probability map that maximizes
    likelihood while penalizing large vertical jumps between time steps.
    
    Args:
        cdf_map: Occupancy map of shape (batch, height, seq_len), values in [0, 1].
                 High at bottom, low at top.
        bin_centers: Tensor of shape (height,) mapping bin indices to values.
        beam_width: Number of candidate paths to keep at each step.
        jump_penalty_scale: Multiplier for jump penalty (higher = smoother paths).
        search_radius: Max vertical pixels to search from previous position.
        eps: Small constant for numerical stability in log.
        
    Returns:
        Decoded values of shape (batch, seq_len).
    """
    batch_size, height, seq_len = cdf_map.shape
    device = cdf_map.device
    
    # 1. Convert CDF to PDF: drop[y] = cdf[y] - cdf[y+1] (value drops going up)
    # For occupancy: high at bottom, low at top, so cdf[y] - cdf[y+1] > 0 at transition
    # Append zeros at top for shape consistency
    pdf = torch.zeros_like(cdf_map)
    pdf[:, :-1, :] = cdf_map[:, :-1, :] - cdf_map[:, 1:, :]
    pdf = torch.clamp(pdf, min=0.0)
    
    # Normalize PDF per column to get valid probabilities
    pdf_sum = pdf.sum(dim=1, keepdim=True).clamp(min=eps)
    pdf = pdf / pdf_sum
    
    # Log probabilities (with floor to avoid -inf)
    log_pdf = torch.log(pdf.clamp(min=eps))  # (batch, height, seq_len)
    
    results = []
    
    for b in range(batch_size):
        log_pdf_b = log_pdf[b]  # (height, seq_len)
        
        # Initialize: top beam_width starting positions at t=0
        init_scores = log_pdf_b[:, 0]  # (height,)
        topk_scores, topk_indices = init_scores.topk(min(beam_width, height))
        
        # Each beam: (score, [path indices])
        # Store as tensors for efficiency
        beam_scores = topk_scores  # (beam_width,)
        beam_paths = topk_indices.unsqueeze(1)  # (beam_width, 1)
        
        # Step through time
        for t in range(1, seq_len):
            num_beams = beam_scores.shape[0]
            
            # Current ending positions for each beam
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
        
        if config.model_type == "transformer":
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
            # Calculate input channels for conditioning encoder (past context)
            # Past context has num_vars + aux channels, but does NOT have guidance channels
            # (Guidance is added to the noisy future input, not the past context)
            cond_in_channels = backbone_in_channels - config.guidance_channels
            
            self.noise_predictor = ConditionalUNet2D(
                in_channels=backbone_in_channels,
                out_channels=config.num_variables,  # Output one channel per variable
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
            
            # Create TimeSeriesContextEncoder for hybrid conditioning
            if config.use_hybrid_condition:
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
            if config.use_dilated_middle:
                logger.info(f"  Dilated middle block: enabled (dilations=1,2,4,8)")
            if config.use_hybrid_condition:
                logger.info(f"  Hybrid 1D conditioning: enabled (context_dim={config.context_embedding_dim}, layers={config.context_encoder_layers})")
    
    def to(self, device):
        """Move model and scheduler to device."""
        super().to(device)
        self.scheduler = self.scheduler.to(device)
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
        if self.guidance_model is None:
            raise ValueError("Guidance model is None but guidance channel is requested")
        with torch.no_grad():
            coarse_forecast = self.guidance_model.get_forecast(past, forecast_length)
        mean, std = stats
        coarse_norm = (coarse_forecast - mean) / std
        guidance_2d = self.encode_to_2d(coarse_norm, scale_for_diffusion=True)
        return guidance_2d
    
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
        """Training forward pass using unified L+F channel scheme."""
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
        
        past_padded = self._pad_to_window(past_2d, 'past', total_len)
        noisy_future_padded = self._pad_to_window(noisy_future, 'future', total_len)
        canvas = past_padded + noisy_future_padded
        
        canvas_with_coords = self._inject_coordinate_channel(canvas)
        canvas_full = self._inject_time_channels(canvas_with_coords)
        
        if self.config.use_value_channel:
            if past_norm.dim() == 3: past_vals = past_norm
            else: past_vals = past_norm.unsqueeze(1)
            vals_padded = F.pad(past_vals, (0, future_len))
            val_channel = self._get_value_channel(vals_padded, self.config.image_height)
            if val_channel.shape[1] > 1: val_channel = val_channel[:, 0:1, :, :]
            canvas_full = torch.cat([canvas_full, val_channel], dim=1)
            
        guidance_2d = None
        if self.config.use_guidance_channel:
            guidance_2d = self._generate_guidance_2d(past, past_norm, stats, future_len)
            guidance_padded = self._pad_to_window(guidance_2d, 'future', total_len)
            canvas_full = self._inject_guidance_channel(canvas_full, guidance_padded)
        
        # Conditioning: Must match canvas length (608) for visual_concat
        # We do NOT inject aux channels (coords, time) here because they are already in the main canvas
        # and shared across the spatial dimensions.
        cond_for_unet = past_padded
        
        # Value channel for conditioning (if used)
        if self.config.use_value_channel:
             # Use same value channel as input
             cond_for_unet = torch.cat([cond_for_unet, val_channel], dim=1)
        
        encoder_hidden_states = None
        if self.context_encoder is not None:
            context_input = self._prepare_1d_context(past_norm)
            encoder_hidden_states = self.context_encoder(context_input)

        noise_pred_full = self.noise_predictor(
            canvas_full, t, cond_for_unet,
            encoder_hidden_states=encoder_hidden_states
        )
        noise_pred = noise_pred_full[..., past_len:]
        
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
        """Generate future predictions using the unified (L+F) channel scheme."""
        batch_size = past.shape[0]
        device = past.device
        if cfg_scale is None: cfg_scale = self.config.cfg_scale
        past_norm, _, stats = self._normalize_sequence(past)
        past_2d = self.encode_to_2d(past_norm)
        past_len = past_2d.shape[-1]
        future_len = self.config.forecast_length
        total_len = past_len + future_len
        
        past_padded = self._pad_to_window(past_2d, 'past', total_len)
        
        # Value Channel (Full Window)
        val_channel = None
        if self.config.use_value_channel:
            if past_norm.dim() == 3: past_vals = past_norm
            else: past_vals = past_norm.unsqueeze(1)
            vals_padded = F.pad(past_vals, (0, future_len))
            val_channel = self._get_value_channel(vals_padded, self.config.image_height)
            if val_channel.shape[1] > 1: val_channel = val_channel[:, 0:1, :, :]

        # Conditioning
        cond_for_unet = past_padded
        if self.config.use_value_channel:
             cond_for_unet = torch.cat([cond_for_unet, val_channel], dim=1)
        
        null_cond_for_unet = None
        if cfg_scale > 1.0:
            null_cond = torch.zeros_like(past_padded)
            # No aux channels for null cond either
            if self.config.use_value_channel:
                 _, _, h, w = null_cond.shape
                 zvc = torch.zeros(batch_size, 1, h, w, device=device, dtype=past.dtype)
                 null_cond = torch.cat([null_cond, zvc], dim=1)
            null_cond_for_unet = null_cond

        encoder_hidden_states = None
        null_encoder_hidden_states = None
        if self.context_encoder is not None:
            context_input = self._prepare_1d_context(past_norm)
            encoder_hidden_states = self.context_encoder(context_input)
            if cfg_scale > 1.0:
                null_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)

        guidance_2d = None
        guidance_padded = None
        null_guidance_padded = None
        if self.config.use_guidance_channel:
            guidance_2d = self._generate_guidance_2d(past, past_norm, stats, future_len)
            guidance_padded = self._pad_to_window(guidance_2d, 'future', total_len)
            if cfg_scale > 1.0:
                null_guidance_padded = torch.zeros_like(guidance_padded)
        
        noise_shape = (batch_size, self.config.num_variables, self.config.image_height, future_len)
        
        def model_fn(x_future, t, cond, use_null_context=False, use_null_guidance=False):
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
            
        def model_cfg(x, t, cond, null_cond, cfg_scale):
            if cfg_scale <= 1.0: return model_fn(x, t, cond)
            x_dbl = torch.cat([x, x], dim=0)
            t_dbl = torch.cat([t, t], dim=0)
            cond_dbl = torch.cat([cond, null_cond], dim=0)
            
            # Double fixed inputs
            # Note: We must construct canvas inside to respect batching
            # But we can also double the fixed parts here passed to model_fn?
            # model_fn constructs canvas. 
            # We can't use model_fn directly for batching unless we modify it to take full canvas.
            # Let's expand logic here for correctness.
            
            # This logic is duplicated, but safe.
            # Actually, `cond` passed here IS `cond_dbl`.
            # We need `past_padded` doubled, `val_channel` doubled, `guidance` doubled.
            
            past_padded_dbl = torch.cat([past_padded, past_padded], dim=0)
            x_future_padded_dbl = self._pad_to_window(x_dbl, 'future', total_len)
            canvas_dbl = past_padded_dbl + x_future_padded_dbl
            
            canvas_full = self._inject_coordinate_channel(canvas_dbl)
            canvas_full = self._inject_time_channels(canvas_full)
            
            if val_channel is not None:
                val_dbl = torch.cat([val_channel, val_channel], dim=0)
                canvas_full = torch.cat([canvas_full, val_dbl], dim=1)
                
            if self.config.use_guidance_channel:
                guide_dbl = torch.cat([guidance_padded, null_guidance_padded], dim=0)
                canvas_full = self._inject_guidance_channel(canvas_full, guide_dbl)
                
            ctx_dbl = None
            if encoder_hidden_states is not None:
                ctx_dbl = torch.cat([encoder_hidden_states, null_encoder_hidden_states], dim=0)
            
            out_dbl = self.noise_predictor(canvas_full, t_dbl, cond_dbl, encoder_hidden_states=ctx_dbl)
            out_future = out_dbl[..., past_len:]
            cond_out, uncond_out = out_future.chunk(2, dim=0)
            return uncond_out + cfg_scale * (cond_out - uncond_out)

        if use_ddim:
            future_2d = self.scheduler.sample_ddim_cfg(
                model=lambda x, t, c: model_cfg(x, t, c, null_cond_for_unet, cfg_scale),
                shape=noise_shape,
                cond=cond_for_unet,
                null_cond=null_cond_for_unet,
                cfg_scale=1.0,
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
                cfg_scale=1.0,
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
        result = {
            'prediction': future,
            'prediction_norm': future_norm,
            'future_2d': future_2d,
            'past_2d': past_2d
        }
        if guidance_2d is not None: result['guidance_2d'] = guidance_2d
        return result
    
    def get_loss(
        self,
        past: torch.Tensor,
        future: torch.Tensor
    ) -> torch.Tensor:
        """Convenience method to get just the loss for training."""
        outputs = self.forward(past, future)
        return outputs['loss']
