"""Binary bit-flip and ordinal D3PM schedulers for hard CDF / skyline images."""

import math
import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _build_transition_schedule(
    num_steps: int,
    transition_min: float,
    transition_max: float,
    schedule_type: str,
    device: str,
) -> torch.Tensor:
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
    return betas.clamp(1e-8, 1.0 - 1e-8)


def _build_banded_transition_matrix(
    beta: float,
    num_classes: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Row-stochastic ordinal kernel: stay mass 1-beta, spread beta to neighbors."""
    h = num_classes
    idx = torch.arange(h, device=device, dtype=dtype)
    dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
    weights = torch.exp(-0.5 * dist.pow(2))
    weights.fill_diagonal_(0.0)
    off = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    q = off * float(beta)
    q.diagonal().add_(1.0 - float(beta))
    return q


class OrdinalD3PMScheduler:
    """Ordinal categorical diffusion with banded transition matrices."""

    def __init__(
        self,
        num_steps: int = 1000,
        num_classes: int = 16,
        transition_min: float = 1e-5,
        transition_max: float = 0.3,
        schedule_type: str = "sqrt_linear",
        device: str = "cpu",
    ):
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.device = device
        self.betas = _build_transition_schedule(
            num_steps, transition_min, transition_max, schedule_type, device
        )
        self.schedule_type = schedule_type

        q_list = []
        for beta in self.betas.tolist():
            q_list.append(
                _build_banded_transition_matrix(
                    beta,
                    num_classes,
                    device=torch.device(device),
                    dtype=torch.float32,
                )
            )
        q_stack = torch.stack(q_list, dim=0)  # (T, H, H)
        q_bar = torch.empty_like(q_stack)
        q_bar[0] = q_stack[0]
        for step in range(1, num_steps):
            q_bar[step] = q_bar[step - 1] @ q_stack[step]
        self.Q = q_stack
        self.Q_bar = q_bar

        logger.debug(
            "OrdinalD3PMScheduler: T=%d, H=%d, schedule=%s, beta=[%.2e, %.3f]",
            num_steps,
            num_classes,
            schedule_type,
            self.betas[0].item(),
            self.betas[-1].item(),
        )

    def to(self, device: str) -> "OrdinalD3PMScheduler":
        self.device = device
        self.betas = self.betas.to(device)
        self.Q = self.Q.to(device)
        self.Q_bar = self.Q_bar.to(device)
        return self

    def _bins_from_skyline(self, skyline: torch.Tensor) -> torch.Tensor:
        if skyline.dim() == 4:
            return skyline.argmax(dim=2)
        if skyline.dim() == 3:
            return skyline.argmax(dim=1)
        raise ValueError(f"Expected skyline (B,1,H,W) or (B,H,W), got {skyline.shape}")

    def _skyline_from_bins(self, bins: torch.Tensor) -> torch.Tensor:
        bins = bins.long().clamp(0, self.num_classes - 1)
        one_hot = F.one_hot(bins, num_classes=self.num_classes)
        if one_hot.dim() == 3:
            return one_hot.permute(0, 2, 1).unsqueeze(1).float()
        return one_hot.permute(0, 1, 3, 2).float()

    def add_noise(
        self,
        x0_skyline: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffuse clean skyline to noisy skyline at timestep t."""
        if x0_skyline.dim() != 4:
            raise ValueError(f"x0_skyline must be (N,1,H,W), got {x0_skyline.shape}")
        n, _, h, w = x0_skyline.shape
        if h != self.num_classes:
            raise ValueError(f"skyline height {h} != num_classes {self.num_classes}")

        x0_bins = self._bins_from_skyline(x0_skyline).reshape(n, w)
        t_idx = t.reshape(-1).long().clamp(0, self.num_steps - 1)
        q_bar = self.Q_bar[t_idx]  # (N, H, H)
        one_hot = F.one_hot(x0_bins.clamp(0, h - 1), num_classes=h).float()  # (N, W, H)
        probs = torch.einsum("nwh,nhg->nwg", one_hot, q_bar)
        probs = probs.clamp_min(1e-12)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        flat = probs.reshape(-1, h)
        xt_bins = torch.multinomial(flat, 1).reshape(n, w)
        xt_skyline = self._skyline_from_bins(xt_bins)
        return xt_skyline, xt_bins

    def posterior_probs(
        self,
        xt_bins: torch.Tensor,
        x0_probs: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """q(x_{t-1} | x_t, x_0) ∝ row(x_{t-1}) Q_t ⊙ row(x_0) Q̄_{t-1}."""
        n, w = xt_bins.shape
        h = self.num_classes
        t = int(max(0, min(t, self.num_steps - 1)))
        q_t = self.Q[t]  # (H, H)
        if t == 0:
            q_prev_bar = torch.eye(h, device=x0_probs.device, dtype=x0_probs.dtype)
        else:
            q_prev_bar = self.Q_bar[t - 1]

        x0_probs = x0_probs.clamp_min(1e-12)
        x0_probs = x0_probs / x0_probs.sum(dim=1, keepdim=True)
        prior = torch.einsum("ngw,gi->nwi", x0_probs, q_prev_bar)
        xt_idx = xt_bins.long().clamp(0, h - 1)
        likelihood = q_t.t()[xt_idx]
        post = prior * likelihood
        post = post.clamp_min(1e-12)
        return post / post.sum(dim=2, keepdim=True)

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
        """Reverse sample ordinal skylines."""
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

        n = shape[0]
        h = self.num_classes
        w = shape[-1]
        xt_bins = torch.randint(0, h, (n, w), device=device)
        xt = self._skyline_from_bins(xt_bins)
        intermediates = []

        for i, t_val in enumerate(step_indices):
            t_idx = int(t_val.item())
            if yield_intermediates and (snapshot_set is None or t_idx in snapshot_set):
                intermediates.append((t_idx, xt.clone()))

            t_batch = torch.full((n,), t_idx, device=device, dtype=torch.long)
            logits = model_fn(xt, t_batch)
            if logits.dim() == 4:
                logits = logits[:, 0]
            x0_probs = F.softmax(logits, dim=1)

            if i < len(step_indices) - 1:
                t_next = int(step_indices[i + 1].item())
                post = self.posterior_probs(xt_bins, x0_probs, t_idx)
                flat = post.permute(0, 2, 1).reshape(-1, h).clamp_min(1e-12)
                flat = flat / flat.sum(dim=-1, keepdim=True)
                xt_bins = torch.multinomial(flat, 1).reshape(n, w)
                xt = self._skyline_from_bins(xt_bins)
            else:
                xt_bins = x0_probs.argmax(dim=1)
                xt = self._skyline_from_bins(xt_bins)

            if verbose and i % 5 == 0:
                logger.debug("  ordinal d3pm step %d/%d (t=%d)", i + 1, num_steps, t_idx)

        if yield_intermediates:
            if snapshot_set is None or 0 in snapshot_set:
                intermediates.append((0, xt.clone()))
            return xt, intermediates
        return xt


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

        self.betas = _build_transition_schedule(
            num_steps, beta_start, beta_end, schedule_type, device
        )
        self.schedule_type = schedule_type
        logger.debug(
            "BinaryDiffusionScheduler initialized: T=%d, schedule=%s, beta=[%.2e, %.3f]",
            num_steps,
            schedule_type,
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
