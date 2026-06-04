"""Binary bit-flip diffusion scheduler for hard CDF images."""

import math
import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class BinaryDiffusionScheduler:
    """Bit-flip diffusion scheduler for hard binary CDF images.

    The forward process flips each bit with probability beta_t. The reverse
    sampler predicts a clean x0 image, then re-noises it at the next lower
    timestep until reaching a clean binary map.
    """

    def __init__(
        self,
        num_steps: int = 1000,
        beta_start: float = 1e-5,
        beta_end: float = 0.5,
        schedule_type: str = "sqrt_linear",
        device: str = "cpu",
    ):
        self.num_steps = num_steps
        self.device = device

        t = torch.linspace(0.0, 1.0, num_steps, device=device)
        if schedule_type == "linear":
            self.betas = beta_start + t * (beta_end - beta_start)
        elif schedule_type == "cosine":
            eased = 0.5 * (1.0 - torch.cos(math.pi * t))
            self.betas = beta_start + eased * (beta_end - beta_start)
        elif schedule_type == "sqrt_linear":
            sq_s = math.sqrt(beta_start)
            sq_e = math.sqrt(beta_end)
            self.betas = (sq_s + t * (sq_e - sq_s)) ** 2
        else:
            raise ValueError(f"Unknown binary noise schedule: {schedule_type!r}")
        self.schedule_type = schedule_type
        logger.info(
            f"BinaryDiffusionScheduler initialized: T={num_steps}, "
            f"schedule={schedule_type}, beta=[{self.betas[0]:.2e}, {self.betas[-1]:.3f}]"
        )

    def to(self, device: str) -> "BinaryDiffusionScheduler":
        self.device = device
        self.betas = self.betas.to(device)
        return self

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add XOR bit-flip noise to a binary tensor in {0, 1}."""
        beta_t = self.betas[t]
        shape = (-1,) + (1,) * (x0.dim() - 1)
        beta_t = beta_t.view(shape).expand_as(x0)
        zt = torch.bernoulli(beta_t)
        xt = (x0.bool() ^ zt.bool()).float()
        return xt, zt

    @torch.no_grad()
    def sample(
        self,
        model_fn,
        shape: Tuple[int, ...],
        num_steps: int = 20,
        device: str = "cpu",
        verbose: bool = False,
        sampler: str = "ddim",
        yield_intermediates: bool = False,
        reverse_step_indices: Optional[torch.Tensor] = None,
        snapshot_timesteps: Optional[Tuple[int, ...]] = None,
    ):
        """Sample a clean binary image with a reduced set of reverse steps."""
        if reverse_step_indices is not None:
            step_indices = reverse_step_indices.to(device=device, dtype=torch.long)
        else:
            if sampler == "dpmpp":
                # Karras-style spacing: more denoise evaluations near low-noise timesteps.
                ramp = torch.linspace(1.0, 0.0, num_steps, device=device)
                step_indices = torch.round((ramp ** 2) * (self.num_steps - 1)).long()
            else:
                step_indices = torch.linspace(
                    self.num_steps - 1,
                    0,
                    num_steps,
                    device=device,
                    dtype=torch.long,
                )
        snapshot_set = None
        if snapshot_timesteps is not None:
            snapshot_set = {int(min(max(0, t), self.num_steps - 1)) for t in snapshot_timesteps}
        xt = torch.bernoulli(torch.full(shape, 0.5, device=device))

        intermediates = []

        for i, t_val in enumerate(step_indices):
            t_idx = int(t_val.item())
            if yield_intermediates and (
                snapshot_set is None or t_idx in snapshot_set
            ):
                intermediates.append((t_idx, xt.clone()))

            t_batch = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
            x0_logits, _zt_logits = model_fn(xt, t_batch)
            x0_hat = (torch.sigmoid(x0_logits) > 0.5).float()

            if i < len(step_indices) - 1:
                t_next = int(step_indices[i + 1].item())
                beta_next = self.betas[t_next].item()
                zt_new = torch.bernoulli(torch.full_like(x0_hat, beta_next))
                xt = (x0_hat.bool() ^ zt_new.bool()).float()
            else:
                xt = x0_hat

            if verbose and i % 5 == 0:
                logger.debug(f"  binary step {i + 1}/{num_steps} (t={t_idx})")

        if yield_intermediates:
            if snapshot_set is None or 0 in snapshot_set:
                intermediates.append((0, xt.clone()))
            return xt, intermediates
        return xt
