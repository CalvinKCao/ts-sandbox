"""Binary bit-flip scheduler for hard CDF images."""

import math
import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _build_transition_schedule(
    num_steps: int,
    transition_min: float,
    transition_max: float,
    schedule_type: str,
    device: str,
    *,
    length_mode: str = "none",
    length_g: float = 1.0,
    length_scale: float = 1.0,
) -> torch.Tensor:
    """Build β_t in [transition_min, transition_max].

    length_mode remaps the base schedule for longer sequences (diag / optional train):
      none  — identity
      power — β(u) = β0 + (β1-β0) * u^(1/g); g>1 front-loads high β (more time near 0.5)
      scale — β(u) = clip( (β0+(β1-β0)*u) * scale, max=β1 ); scale>1 hits β1 earlier
    """
    t = torch.linspace(0.0, 1.0, num_steps, device=device)
    if schedule_type == "linear":
        betas = transition_min + t * (transition_max - transition_min)
    elif schedule_type == "cosine":
        eased = 0.5 * (1.0 - torch.cos(math.pi * t))
        betas = transition_min + eased * (transition_max - transition_min)
    elif schedule_type == "sqrt_linear":
        sq_s = math.sqrt(transition_min)
        sq_e = math.sqrt(transition_max)
        betas = (sq_s + t * (sq_e - sq_s)) ** 2
    else:
        raise ValueError(f"Unknown noise schedule: {schedule_type!r}")

    mode = (length_mode or "none").lower()
    if mode not in {"none", "power", "scale"}:
        raise ValueError(f"Unknown length_mode: {length_mode!r}")
    if mode == "power":
        g = float(length_g)
        if g <= 0:
            raise ValueError(f"length_g must be > 0, got {g}")
        # Rebuild from normalized progress so power applies cleanly to linear/cosine too.
        betas = transition_min + (transition_max - transition_min) * (t ** (1.0 / g))
    elif mode == "scale":
        scale = float(length_scale)
        if scale <= 0:
            raise ValueError(f"length_scale must be > 0, got {scale}")
        if schedule_type != "sqrt_linear":
            betas = (transition_min + t * (transition_max - transition_min)) * scale
        else:
            betas = betas * scale
        betas = betas.clamp(max=transition_max)

    return betas.clamp(1e-8, 1.0 - 1e-8)


def length_schedule_g(
    seq_len: int,
    *,
    l_ref: float = 104.0,
    g_ref: float = 1.0,
    l_cal: float = 728.0,
    g_cal: float = 1.5,
) -> float:
    """Log-linear g(L): identity at l_ref, g_cal at l_cal, interpolate in log-L."""
    L = float(max(1, seq_len))
    if L <= l_ref or abs(g_cal - g_ref) < 1e-12:
        return float(g_ref)
    if L >= l_cal:
        return float(g_cal)
    t = (math.log(L) - math.log(l_ref)) / (math.log(l_cal) - math.log(l_ref))
    return float(g_ref + t * (g_cal - g_ref))


def length_schedule_scale(
    seq_len: int,
    *,
    l_ref: float = 104.0,
    scale_ref: float = 1.0,
    l_cal: float = 728.0,
    scale_cal: float = 1.5,
) -> float:
    """Log-linear scale(L); same interpolation as length_schedule_g."""
    return length_schedule_g(
        seq_len, l_ref=l_ref, g_ref=scale_ref, l_cal=l_cal, g_cal=scale_cal
    )


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
        length_mode: str = "none",
        length_g: float = 1.0,
        length_scale: float = 1.0,
    ):
        self.num_steps = num_steps
        self.device = device
        self.length_mode = (length_mode or "none").lower()
        self.length_g = float(length_g)
        self.length_scale = float(length_scale)

        self.betas = _build_transition_schedule(
            num_steps,
            beta_start,
            beta_end,
            schedule_type,
            device,
            length_mode=self.length_mode,
            length_g=self.length_g,
            length_scale=self.length_scale,
        )
        self.schedule_type = schedule_type
        logger.debug(
            "BinaryDiffusionScheduler initialized: T=%d, schedule=%s, "
            "length_mode=%s g=%.3f scale=%.3f, beta=[%.2e, %.3f]",
            num_steps,
            schedule_type,
            self.length_mode,
            self.length_g,
            self.length_scale,
            self.betas[0].item(),
            self.betas[-1].item(),
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
