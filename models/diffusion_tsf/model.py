"""
Complete Diffusion-based Time Series Forecasting Model.

This module integrates:
- Preprocessing (normalization, 2D encoding, blur)
- Conditional U-Net
- DDPM/DDIM diffusion
- 2D to 1D decoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Optional, Tuple

# Handle imports for both module and script execution
try:
    from .config import DiffusionTSFConfig
    from .preprocessing import Standardizer, TimeSeriesTo2D, VerticalGaussianBlur
    from .unet import ConditionalUNet2D
    from .transformer import DiffusionTransformer
    from .diffusion import DiffusionScheduler
except ImportError:
    from config import DiffusionTSFConfig
    from preprocessing import Standardizer, TimeSeriesTo2D, VerticalGaussianBlur
    from unet import ConditionalUNet2D
    from transformer import DiffusionTransformer
    from diffusion import DiffusionScheduler

logger = logging.getLogger(__name__)


class DiffusionTSF(nn.Module):
    """Diffusion-based Time Series Forecasting Model.
    
    Pipeline:
    1. Normalize input (past + future) using local mean/std
    2. Convert to 2D "stripe" representation
    3. Apply vertical Gaussian blur
    4. Train U-Net to denoise future conditioned on past
    5. At inference: generate future via DDPM/DDIM
    6. Decode 2D representation back to 1D
    """
    
    def __init__(self, config: DiffusionTSFConfig):
        """
        Args:
            config: Model configuration
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
        
        # Noise prediction backbone (U-Net or Transformer)
        if config.model_type == "transformer":
            self.noise_predictor = DiffusionTransformer(
                image_height=config.image_height,
                patch_size=config.transformer_patch_size,
                embed_dim=config.transformer_embed_dim,
                depth=config.transformer_depth,
                num_heads=config.transformer_num_heads,
                dropout=config.transformer_dropout,
            )
        else:
            self.noise_predictor = ConditionalUNet2D(
                in_channels=1,
                out_channels=1,
                channels=config.unet_channels,
                num_res_blocks=config.num_res_blocks,
                attention_levels=config.attention_levels,
                image_height=config.image_height
            )
        
        # Diffusion scheduler (not a nn.Module, managed separately)
        self.scheduler = DiffusionScheduler(
            num_steps=config.num_diffusion_steps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            schedule=config.noise_schedule
        )
        
        logger.info(f"DiffusionTSF initialized:")
        logger.info(f"  Lookback: {config.lookback_length}, Forecast: {config.forecast_length}")
        logger.info(f"  Image size: {config.image_height} x W")
        logger.info(f"  Diffusion steps: {config.num_diffusion_steps}")
    
    def to(self, device):
        """Move model and scheduler to device."""
        super().to(device)
        self.scheduler = self.scheduler.to(device)
        return self
    
    def _normalize_sequence(
        self,
        past: torch.Tensor,
        future: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Normalize sequences using past statistics.
        
        Args:
            past: Past sequence of shape (batch, past_len)
            future: Optional future sequence of shape (batch, future_len)
            
        Returns:
            (past_norm, future_norm, (mean, std))
        """
        # Compute statistics from past only
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
        if size_x % 2 == 0:
            size_x += 1
        if size_y % 2 == 0:
            size_y += 1
        
        x = torch.arange(size_x, dtype=torch.float32) - size_x // 2
        y = torch.arange(size_y, dtype=torch.float32) - size_y // 2
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        kernel = torch.exp(-(xx**2 / (2 * sigma_x**2) + yy**2 / (2 * sigma_y**2)))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size_y, size_x)
    
    def _apply_decode_smoothing(self, prob: torch.Tensor) -> torch.Tensor:
        """Apply horizontal-heavy Gaussian smoothing to probability map.
        
        This is only used at inference to connect vertical streaks along time.
        """
        kernel = self.decode_smoothing_kernel.to(device=prob.device, dtype=prob.dtype)
        pad_y = kernel.shape[2] // 2
        pad_x = kernel.shape[3] // 2
        prob_2d = prob.unsqueeze(1)  # (batch, 1, height, width)
        prob_padded = F.pad(prob_2d, (pad_x, pad_x, pad_y, pad_y), mode="reflect")
        smoothed = F.conv2d(prob_padded, kernel)
        return smoothed.squeeze(1)
    
    def _compute_emd_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute column-wise Wasserstein-1 distance via CDF trick.
        
        pred/target are logits over height; we softmax to get probabilities,
        build CDFs along height, then take mean L1 difference.
        """
        if pred.dim() == 4:
            pred = pred.squeeze(1)
            target = target.squeeze(1)
        
        temperature = self.config.decode_temperature
        prob_pred = F.softmax(pred / temperature, dim=1)
        prob_target = F.softmax(target / temperature, dim=1)
        
        cdf_pred = prob_pred.cumsum(dim=1)
        cdf_target = prob_target.cumsum(dim=1)
        
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
            # Insert the default kernel so strict loading succeeds.
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
        """Encode 1D time series to blurred 2D representation.
        
        Args:
            x: Normalized time series of shape (batch, seq_len)
            scale_for_diffusion: If True, scale output to [-1, 1] range for diffusion
            
        Returns:
            Blurred 2D image of shape (batch, 1, height, seq_len)
        """
        image = self.to_2d(x)
        blurred = self.blur(image)
        
        if scale_for_diffusion:
            if self.config.representation_mode == "pdf":
                # Scale from [0, ~0.03] to [-1, 1] for proper diffusion SNR
                # The blur already creates a pseudo-probability, but values are small
                scaled = blurred * 30.0  # Now roughly [0, 1]
                scaled = scaled * 2.0 - 1.0  # Now [-1, 1]
            else:
                # Occupancy map already dense in [0, 1]; just shift to [-1, 1]
                scaled = blurred.clamp(min=0.0, max=1.0) * 2.0 - 1.0
            return scaled
        
        return blurred
    
    def decode_from_2d(self, image: torch.Tensor, from_diffusion: bool = True) -> torch.Tensor:
        """Decode 2D representation to 1D time series.
        
        Uses expectation over the probability distribution at each time step.
        
        Args:
            image: 2D image of shape (batch, 1, height, seq_len)
            from_diffusion: If True, image is in [-1, 1] range from diffusion
            
        Returns:
            Time series of shape (batch, seq_len)
        """
        # Remove channel dimension: (batch, height, seq_len)
        if image.dim() == 4:
            image = image.squeeze(1)
        
        if self.config.representation_mode == "pdf":
            if from_diffusion:
                # Image is in [-1, 1] range. Higher values = higher probability.
                # We need to convert to proper probabilities.
                temperature = self.config.decode_temperature
                prob = F.softmax(image / temperature, dim=1)
            else:
                prob = F.softmax(image, dim=1)
            
            # Inference-only anisotropic smoothing to encourage temporal continuity
            if not self.training and self.config.decode_smoothing and self.decode_smoothing_kernel is not None:
                prob = self._apply_decode_smoothing(prob)
            
            # Compute expectation: sum_j P(j) * center(j)
            centers = self.to_2d.bin_centers.view(1, -1, 1).to(image.device)
            x = (prob * centers).sum(dim=1)
        else:
            if from_diffusion:
                cdf_map = (image + 1.0) / 2.0
            else:
                cdf_map = image
            
            cdf_map = torch.clamp(cdf_map, min=0.0)
            
            if not self.training and self.config.decode_smoothing and self.decode_smoothing_kernel is not None:
                cdf_map = self._apply_decode_smoothing(cdf_map)
            
            column_sum = cdf_map.sum(dim=1)
            column_sum = torch.clamp(column_sum, 0.0, float(self.config.image_height))
            normalized = column_sum / float(self.config.image_height)
            x = normalized * (2 * self.config.max_scale) - self.config.max_scale
        
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
            if mask_h <= 0 or mask_w <= 0:
                continue
            
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
        t: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass.
        
        Computes the diffusion loss:
        1. Normalize and encode to 2D
        2. Add noise to future image
        3. Predict noise with U-Net conditioned on past
        4. Return loss between true and predicted noise
        
        Args:
            past: Past sequence of shape (batch, past_len)
            future: Future sequence of shape (batch, future_len)
            t: Optional diffusion timesteps (sampled randomly if None)
            
        Returns:
            Dictionary with 'loss' and intermediate values for debugging
        """
        batch_size = past.shape[0]
        device = past.device
        
        # Normalize using past statistics
        past_norm, future_norm, stats = self._normalize_sequence(past, future)
        
        # Encode to 2D
        past_2d = self.encode_to_2d(past_norm)  # (batch, 1, H, past_len)
        future_2d = self.encode_to_2d(future_norm)  # (batch, 1, H, future_len)
        
        # Coarse dropout / cutout augmentation on CONDITIONING ONLY
        # Never apply to future_2d - that's the ground truth target!
        past_2d = self._apply_coarse_dropout(past_2d)
        
        logger.debug(f"past_2d shape: {past_2d.shape}, future_2d shape: {future_2d.shape}")
        
        # Classifier-Free Guidance: randomly drop conditioning during training
        # This teaches the model to work both with and without conditioning
        if self.training and self.config.cfg_dropout > 0:
            # Create a mask for which samples should have conditioning dropped
            drop_mask = torch.rand(batch_size, device=device) < self.config.cfg_dropout
            if drop_mask.any():
                # Replace dropped conditions with zeros (null conditioning)
                null_cond = torch.zeros_like(past_2d)
                past_2d = torch.where(
                    drop_mask.view(-1, 1, 1, 1).expand_as(past_2d),
                    null_cond,
                    past_2d
                )
        
        # Sample random timesteps if not provided
        if t is None:
            t = torch.randint(0, self.config.num_diffusion_steps, (batch_size,), device=device)
        
        # Add noise to future
        noisy_future, noise = self.scheduler.add_noise(future_2d, t)
        
        # Predict noise
        noise_pred = self.noise_predictor(noisy_future, t, past_2d)
        
        # L2 loss on noise
        noise_loss = F.mse_loss(noise_pred, noise)
        
        # Estimate clean image x0_hat from predicted noise
        x0_pred = self.scheduler.predict_x0_from_noise(noisy_future, t, noise_pred)
        
        # EMD term between estimated x0 and ground truth x0 (future_2d)
        emd_loss = self._compute_emd_loss(x0_pred, future_2d)
        
        loss = noise_loss + self.config.emd_lambda * emd_loss
        
        return {
            'loss': loss,
            'noise_loss': noise_loss,
            'emd_loss': emd_loss,
            'noise': noise,
            'noise_pred': noise_pred,
            'past_2d': past_2d,
            'future_2d': future_2d,
            'noisy_future': noisy_future,
            't': t
        }
    
    @torch.no_grad()
    def generate(
        self,
        past: torch.Tensor,
        use_ddim: bool = True,
        num_ddim_steps: int = 50,
        eta: float = 0.0,
        cfg_scale: Optional[float] = None,
        verbose: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Generate future predictions given past context.
        
        Args:
            past: Past sequence of shape (batch, past_len)
            use_ddim: Whether to use accelerated DDIM sampling
            num_ddim_steps: Number of DDIM steps (ignored if use_ddim=False)
            eta: DDIM stochasticity parameter
            cfg_scale: Classifier-free guidance scale. If None, uses config value.
                       1.0 = no guidance, >1 = stronger conditioning adherence
            verbose: Whether to log progress
            
        Returns:
            Dictionary with predictions and intermediate values
        """
        batch_size = past.shape[0]
        device = past.device
        
        # Use config cfg_scale if not specified
        if cfg_scale is None:
            cfg_scale = self.config.cfg_scale
        
        # Normalize past
        past_norm, _, stats = self._normalize_sequence(past)
        
        # Encode past to 2D
        past_2d = self.encode_to_2d(past_norm)
        
        # Create null conditioning for CFG (zeros)
        null_cond = torch.zeros_like(past_2d) if cfg_scale > 1.0 else None
        
        # Shape for future image
        future_shape = (
            batch_size,
            1,
            self.config.image_height,
            self.config.forecast_length
        )
        
        # Generate via diffusion with CFG
        if use_ddim:
            future_2d = self.scheduler.sample_ddim_cfg(
                model=self.noise_predictor,
                shape=future_shape,
                cond=past_2d,
                null_cond=null_cond,
                cfg_scale=cfg_scale,
                num_steps=num_ddim_steps,
                eta=eta,
                device=device,
                verbose=verbose
            )
        else:
            future_2d = self.scheduler.sample_ddpm_cfg(
                model=self.noise_predictor,
                shape=future_shape,
                cond=past_2d,
                null_cond=null_cond,
                cfg_scale=cfg_scale,
                device=device,
                verbose=verbose
            )
        
        # Decode to 1D (normalized)
        future_norm = self.decode_from_2d(future_2d)
        
        # Denormalize
        future = self._denormalize(future_norm, stats)
        
        return {
            'prediction': future,
            'prediction_norm': future_norm,
            'future_2d': future_2d,
            'past_2d': past_2d
        }
    
    def get_loss(
        self,
        past: torch.Tensor,
        future: torch.Tensor
    ) -> torch.Tensor:
        """Convenience method to get just the loss for training.
        
        Args:
            past: Past sequence of shape (batch, past_len)
            future: Future sequence of shape (batch, future_len)
            
        Returns:
            Scalar loss value
        """
        outputs = self.forward(past, future)
        return outputs['loss']

