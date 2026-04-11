"""
DDPM and DDIM Diffusion Scheduler for Time Series Forecasting.

Implements:
- Forward diffusion process (adding noise)
- Reverse denoising process (DDPM)
- Accelerated sampling (DDIM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class DiffusionScheduler:
    """Diffusion noise scheduler supporting DDPM and DDIM.
    
    Handles the forward process q(x_t | x_0) and reverse process p(x_{t-1} | x_t).
    
    Forward process:
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        
    DDPM reverse process:
        x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * epsilon_theta)
                  + sigma_t * z
                  
    DDIM reverse process (deterministic when eta=0):
        x_{t-1} = sqrt(alpha_bar_{t-1}) * predicted_x0 
                  + sqrt(1 - alpha_bar_{t-1} - sigma^2) * epsilon_theta
                  + sigma * z
    """
    
    def __init__(
        self,
        num_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear",
        device: str = "cpu"
    ):
        """
        Args:
            num_steps: Total number of diffusion steps T
            beta_start: Starting value of beta
            beta_end: Ending value of beta
            schedule: Noise schedule type ("linear", "cosine", "sigmoid", "quadratic")
            device: Device to place tensors on
        """
        self.num_steps = num_steps
        self.device = device
        
        # Create noise schedule
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        elif schedule == "cosine":
            betas = self._cosine_schedule(num_steps, device)
        elif schedule == "sigmoid":
            betas = self._sigmoid_schedule(num_steps, beta_start, beta_end, device)
        elif schedule == "quadratic":
            betas = self._quadratic_schedule(num_steps, beta_start, beta_end, device)
        else:
            raise ValueError(f"Unknown schedule: {schedule}. Supported: linear, cosine, sigmoid, quadratic")
        
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Pre-compute useful quantities
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # For posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        logger.info(f"DiffusionScheduler initialized: T={num_steps}, schedule={schedule}")
        logger.debug(f"  beta range: [{betas[0]:.6f}, {betas[-1]:.6f}]")
        logger.debug(f"  alpha_bar range: [{self.alphas_cumprod[-1]:.6f}, {self.alphas_cumprod[0]:.6f}]")
    
    def _cosine_schedule(self, num_steps: int, device: str) -> torch.Tensor:
        """Cosine noise schedule from 'Improved DDPM' paper."""
        s = 0.008  # Small offset to prevent beta from being too small
        steps = torch.linspace(0, num_steps, num_steps + 1, device=device)
        alpha_bar = torch.cos((steps / num_steps + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        # Note: High betas at the end of schedule can cause DDPM to explode numerically.
        # Use DDIM for inference which is more stable. The clamp to 0.9999 matches
        # common implementations but may cause issues with DDPM at extreme timesteps.
        return torch.clamp(betas, 0.0001, 0.9999)

    def _sigmoid_schedule(self, num_steps: int, beta_start: float, beta_end: float, device: str) -> torch.Tensor:
        """Sigmoid noise schedule: smooth S-curve from beta_start to beta_end."""
        steps = torch.linspace(0, 1, num_steps, device=device)
        # Sigmoid function: maps [0,1] to [beta_start, beta_end] with smooth transition
        betas = beta_start + (beta_end - beta_start) * torch.sigmoid((steps - 0.5) * 6.0)
        return torch.clamp(betas, 0.0001, 0.9999)

    def _quadratic_schedule(self, num_steps: int, beta_start: float, beta_end: float, device: str) -> torch.Tensor:
        """Quadratic noise schedule: quadratic interpolation from beta_start to beta_end."""
        steps = torch.linspace(0, 1, num_steps, device=device)
        # Quadratic: starts slow, accelerates in middle
        betas = beta_start + (beta_end - beta_start) * (steps ** 2)
        return torch.clamp(betas, 0.0001, 0.9999)
    
    def to(self, device: str) -> "DiffusionScheduler":
        """Move all tensors to specified device."""
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self
    
    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward process: add noise to x_0 to get x_t.
        
        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        
        Args:
            x_0: Clean data of shape (batch, ...)
            t: Timesteps of shape (batch,)
            noise: Optional pre-generated noise
            
        Returns:
            (x_t, noise): Noisy data and the noise added
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Get coefficients for this timestep (broadcast to batch)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting: (batch,) -> (batch, 1, 1, 1)
        while sqrt_alpha_bar.dim() < x_0.dim():
            sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.unsqueeze(-1)
        
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        
        return x_t, noise
    
    def predict_x0_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise_pred: torch.Tensor
    ) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise.
        
        x_0 = (x_t - sqrt(1 - alpha_bar_t) * epsilon) / sqrt(alpha_bar_t)
        """
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t]
        
        while sqrt_alpha_bar.dim() < x_t.dim():
            sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.unsqueeze(-1)
        
        x_0 = (x_t - sqrt_one_minus_alpha_bar * noise_pred) / sqrt_alpha_bar
        return x_0
    
    @torch.no_grad()
    def ddpm_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise_pred: torch.Tensor
    ) -> torch.Tensor:
        """Single DDPM reverse step: x_t -> x_{t-1}.
        
        Args:
            x_t: Current noisy sample
            t: Current timestep (scalar or batch)
            noise_pred: Predicted noise from U-Net
            
        Returns:
            x_{t-1}: Denoised sample at previous timestep
        """
        # Ensure t is a tensor
        if isinstance(t, int):
            t = torch.tensor([t], device=x_t.device)
        
        # Get coefficients
        beta_t = self.betas[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alpha = self.sqrt_recip_alphas[t]
        
        # Reshape for broadcasting
        while beta_t.dim() < x_t.dim():
            beta_t = beta_t.unsqueeze(-1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.unsqueeze(-1)
            sqrt_recip_alpha = sqrt_recip_alpha.unsqueeze(-1)
        
        # Compute mean
        # mu = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1-alpha_bar_t)) * epsilon)
        model_mean = sqrt_recip_alpha * (
            x_t - (beta_t / sqrt_one_minus_alpha_bar) * noise_pred
        )
        
        # Add noise (except for t=0)
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            posterior_var = self.posterior_variance[t]
            while posterior_var.dim() < x_t.dim():
                posterior_var = posterior_var.unsqueeze(-1)
            x_prev = model_mean + torch.sqrt(posterior_var) * noise
        else:
            x_prev = model_mean
        
        return x_prev
    
    @torch.no_grad()
    def ddim_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        noise_pred: torch.Tensor,
        eta: float = 0.0,
        is_final_step: bool = False
    ) -> torch.Tensor:
        """Single DDIM reverse step with accelerated sampling.
        
        DDIM allows skipping steps and is deterministic when eta=0.
        
        Args:
            x_t: Current noisy sample
            t: Current timestep
            t_prev: Previous timestep (can skip steps)
            noise_pred: Predicted noise from U-Net
            eta: Stochasticity parameter (0 = deterministic)
            is_final_step: Whether this is the final step (t_prev would be -1)
            
        Returns:
            x_{t_prev}: Denoised sample at previous timestep
        """
        # Get alpha_bar values
        alpha_bar_t = self.alphas_cumprod[t]
        if is_final_step:
            alpha_bar_t_prev = torch.ones_like(alpha_bar_t)
        else:
            alpha_bar_t_prev = self.alphas_cumprod[t_prev]
        
        # Reshape for broadcasting
        while alpha_bar_t.dim() < x_t.dim():
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
            alpha_bar_t_prev = alpha_bar_t_prev.unsqueeze(-1)
        
        # Predict x_0
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
        pred_x0 = (x_t - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
        
        # Clip predicted x_0 for stability
        # Use dynamic clipping: wider range at high noise (early steps), tighter at low noise
        # This prevents destroying signal at high noise levels while still constraining final output
        # alpha_bar goes from ~1 (t=0) to ~0 (t=T), so we use it to scale clamp range
        # At t=0: clamp to [-1, 1]; at high t: clamp to [-2, 2]
        clamp_scale = 1.0 + (1.0 - alpha_bar_t.mean().item())  # 1.0 to 2.0
        pred_x0 = torch.clamp(pred_x0, -clamp_scale, clamp_scale)
        
        # Compute variance
        sigma = eta * torch.sqrt(
            (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev)
        )
        
        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma ** 2) * noise_pred
        
        # Compute x_{t_prev}
        x_prev = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt
        
        if eta > 0:
            noise = torch.randn_like(x_t)
            x_prev = x_prev + sigma * noise
        
        return x_prev
    
    @torch.no_grad()
    def sample_ddpm(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        cond: torch.Tensor,
        device: str = "cpu",
        verbose: bool = True
    ) -> torch.Tensor:
        """Full DDPM sampling: generate samples from noise.
        
        Args:
            model: U-Net model that predicts noise
            shape: Shape of samples to generate (batch, channels, height, width)
            cond: Conditioning tensor (past context image)
            device: Device to generate on
            verbose: Whether to log progress
            
        Returns:
            Generated samples
        """
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        if verbose:
            logger.info(f"Starting DDPM sampling with {self.num_steps} steps")
        
        for t in reversed(range(self.num_steps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = model(x, t_batch, cond)
            
            # Reverse step
            x = self.ddpm_step(x, t_batch, noise_pred)
            
            if verbose and t % 100 == 0:
                logger.debug(f"  Step {self.num_steps - t}/{self.num_steps}")
        
        return x
    
    @torch.no_grad()
    def sample_ddim(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        cond: torch.Tensor,
        num_steps: int = 50,
        eta: float = 0.0,
        device: str = "cpu",
        verbose: bool = True
    ) -> torch.Tensor:
        """Accelerated DDIM sampling.
        
        Args:
            model: U-Net model that predicts noise
            shape: Shape of samples to generate
            cond: Conditioning tensor (past context image)
            num_steps: Number of DDIM steps (can be much smaller than T)
            eta: Stochasticity (0 = deterministic)
            device: Device to generate on
            verbose: Whether to log progress
            
        Returns:
            Generated samples
        """
        # Create timestep schedule (evenly spaced, including the max timestep)
        # We want num_steps timesteps from T-1 down to 0, evenly spaced
        # E.g., for T=1000 and num_steps=50: [999, 979, 959, ..., 19]
        timesteps = torch.linspace(self.num_steps - 1, 0, num_steps, dtype=torch.long).tolist()
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        if verbose:
            logger.info(f"Starting DDIM sampling with {num_steps} steps (eta={eta})")
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            is_final_step = (i == len(timesteps) - 1)
            t_prev = timesteps[i + 1] if not is_final_step else 0
            t_prev_batch = torch.full((shape[0],), t_prev, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = model(x, t_batch, cond)
            
            # DDIM step
            x = self.ddim_step(x, t_batch, t_prev_batch, noise_pred, eta, is_final_step)
            
            if verbose and i % 10 == 0:
                logger.debug(f"  Step {i + 1}/{num_steps} (t={t})")
        
        return x
    
    @torch.no_grad()
    def sample_ddpm_cfg(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        cond: torch.Tensor,
        null_cond: Optional[torch.Tensor],
        cfg_scale: float = 1.0,
        device: str = "cpu",
        verbose: bool = True
    ) -> torch.Tensor:
        """DDPM sampling with Classifier-Free Guidance.
        
        CFG formula: noise_pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
        
        Args:
            model: U-Net model that predicts noise
            shape: Shape of samples to generate
            cond: Conditioning tensor (past context image)
            null_cond: Null conditioning (zeros) for unconditional prediction
            cfg_scale: Guidance scale (1.0 = no guidance, >1 = stronger conditioning)
            device: Device to generate on
            verbose: Whether to log progress
            
        Returns:
            Generated samples
        """
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        if verbose:
            logger.info(f"Starting DDPM+CFG sampling with {self.num_steps} steps (scale={cfg_scale})")
        
        use_cfg = cfg_scale > 1.0 and null_cond is not None
        
        for t in reversed(range(self.num_steps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            if use_cfg:
                # Classifier-Free Guidance: compute both conditional and unconditional
                noise_pred_cond = model(x, t_batch, cond)
                noise_pred_uncond = model(x, t_batch, null_cond)
                # Interpolate: uncond + scale * (cond - uncond)
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = model(x, t_batch, cond)
            
            # Reverse step
            x = self.ddpm_step(x, t_batch, noise_pred)
            
            if verbose and t % 100 == 0:
                logger.debug(f"  Step {self.num_steps - t}/{self.num_steps}")
        
        return x
    
    @torch.no_grad()
    def sample_ddim_cfg(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        cond: torch.Tensor,
        null_cond: Optional[torch.Tensor],
        cfg_scale: float = 1.0,
        num_steps: int = 50,
        eta: float = 0.0,
        device: str = "cpu",
        verbose: bool = True
    ) -> torch.Tensor:
        """Accelerated DDIM sampling with Classifier-Free Guidance.
        
        CFG formula: noise_pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
        
        Args:
            model: U-Net model that predicts noise
            shape: Shape of samples to generate
            cond: Conditioning tensor (past context image)
            null_cond: Null conditioning (zeros) for unconditional prediction
            cfg_scale: Guidance scale (1.0 = no guidance, >1 = stronger conditioning)
            num_steps: Number of DDIM steps
            eta: Stochasticity (0 = deterministic)
            device: Device to generate on
            verbose: Whether to log progress
            
        Returns:
            Generated samples
        """
        # Create timestep schedule (evenly spaced, including the max timestep)
        # We want num_steps timesteps from T-1 down to 0, evenly spaced
        # E.g., for T=1000 and num_steps=50: [999, 979, 959, ..., 19]
        timesteps = torch.linspace(self.num_steps - 1, 0, num_steps, dtype=torch.long).tolist()
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        if verbose:
            logger.info(f"Starting DDIM+CFG sampling with {num_steps} steps (eta={eta}, scale={cfg_scale})")
        
        use_cfg = cfg_scale > 1.0 and null_cond is not None
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            is_final_step = (i == len(timesteps) - 1)
            t_prev = timesteps[i + 1] if not is_final_step else 0
            t_prev_batch = torch.full((shape[0],), t_prev, device=device, dtype=torch.long)
            
            if use_cfg:
                # Classifier-Free Guidance: compute both conditional and unconditional
                noise_pred_cond = model(x, t_batch, cond)
                noise_pred_uncond = model(x, t_batch, null_cond)
                # Interpolate: uncond + scale * (cond - uncond)
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = model(x, t_batch, cond)
            
            # DDIM step
            x = self.ddim_step(x, t_batch, t_prev_batch, noise_pred, eta, is_final_step)
            
            if verbose and i % 10 == 0:
                logger.debug(f"  Step {i + 1}/{num_steps} (t={t})")
        
        return x


class BinaryDiffusionScheduler:
    """bit-flip diffusion scheduler (BDPM-inspired).

    forward process: xt = x0 XOR Bernoulli(beta_t)
        beta_t is the per-pixel bit-flip probability, following a quadratic schedule
        from ~0 at t=0 to 0.5 at t=T (0.5 = max entropy random binary image).

    reverse process: predict x0_hat from xt, then re-add fresh noise at level t-1.
        no need for fancy DDIM — the binary reverse step is just one XOR.

    quadratic schedule: beta_t = (sqrt(s) + (t/T) * (sqrt(e) - sqrt(s)))^2
    """

    def __init__(
        self,
        num_steps: int = 1000,
        beta_start: float = 1e-5,
        beta_end: float = 0.5,
        device: str = "cpu",
    ):
        self.num_steps = num_steps
        self.device = device

        t = torch.linspace(0.0, 1.0, num_steps, device=device)
        sq_s = math.sqrt(beta_start)
        sq_e = math.sqrt(beta_end)
        self.betas = (sq_s + t * (sq_e - sq_s)) ** 2  # (T,) flip probabilities

        logger.info(
            f"BinaryDiffusionScheduler: T={num_steps}, beta range=[{self.betas[0]:.2e}, {self.betas[-1]:.3f}]"
        )

    def to(self, device):
        self.device = device
        self.betas = self.betas.to(device)
        return self

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor):
        """forward process: add bit-flip noise to binary image x0.

        Args:
            x0: binary tensor (...) with values in {0, 1}
            t:  (batch,) integer timesteps

        Returns:
            xt: noisy binary image, same shape as x0
            zt: the noise mask that was XOR'd in (also binary)
        """
        beta_t = self.betas[t]                                # (batch,)
        shape  = (-1,) + (1,) * (x0.dim() - 1)               # (-1, 1, 1, 1) for 4d input
        beta_t = beta_t.view(shape).expand_as(x0)
        zt = torch.bernoulli(beta_t)                          # {0,1}
        xt = (x0.bool() ^ zt.bool()).float()
        return xt, zt

    @torch.no_grad()
    def sample(
        self,
        model_fn,
        shape: tuple,
        num_steps: int = 20,
        device: str = "cpu",
        verbose: bool = False,
    ) -> torch.Tensor:
        """binary reverse sampling using a subset of T timesteps.

        model_fn signature: (xt, t_batch) -> (x0_logits, zt_logits)
            each output is the same shape as xt

        Args:
            model_fn: denoiser that returns (x0_logits, zt_logits)
            shape: output shape, typically (BV, 1, H, W)
            num_steps: how many denoising steps (20 is fine, vs 1000 for ddpm)
        """
        # evenly spaced timesteps from T-1 down to 0
        step_indices = torch.linspace(self.num_steps - 1, 0, num_steps, dtype=torch.long)

        xt = torch.bernoulli(torch.full(shape, 0.5, device=device))

        for i, t_val in enumerate(step_indices):
            t_idx  = t_val.item()
            t_batch = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)

            x0_logits, _zt_logits = model_fn(xt, t_batch)
            x0_hat = (torch.sigmoid(x0_logits) > 0.5).float()

            if i < len(step_indices) - 1:
                # re-add noise at the level of the next (lower) timestep
                t_next = step_indices[i + 1].item()
                beta_next = self.betas[int(t_next)].item()
                zt_new = torch.bernoulli(torch.full_like(x0_hat, beta_next))
                xt = (x0_hat.bool() ^ zt_new.bool()).float()
            else:
                xt = x0_hat

            if verbose and i % 5 == 0:
                logger.debug(f"  binary step {i+1}/{num_steps} (t={t_idx})")

        return xt  # final clean binary image


