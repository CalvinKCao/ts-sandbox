"""Hybrid MSE + truncated signature loss for iTransformer forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import signatory
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "signatory is required for SignatureMSELoss. "
        "Install PyTorch first, then: pip install signatory --no-build-isolation"
    ) from exc


def prepare_signature_path(
    y: torch.Tensor,
    *,
    use_cumsum: bool = False,
) -> torch.Tensor:
    """Augment forecast paths for signature computation.

    Args:
        y: ``[batch, time, features]``
        use_cumsum: if True, integrate increments along time (helps mean-reverting series).

    Returns:
        ``[batch, time, features + 1]`` with a monotone time channel in ``[..., 0]``.
        Basepoint is applied inside ``signatory.signature(..., basepoint=True)``.
    """
    if y.dim() != 3:
        raise ValueError(f"expected [B, T, C], got shape {tuple(y.shape)}")

    path = y
    if use_cumsum:
        path = torch.cumsum(path, dim=1)

    batch, length, _ = path.shape
    device, dtype = path.device, path.dtype
    t = torch.linspace(0.0, 1.0, length, device=device, dtype=dtype)
    t = t.view(1, length, 1).expand(batch, length, 1)
    return torch.cat([t, path], dim=-1)


def truncated_signature(path: torch.Tensor, depth: int) -> torch.Tensor:
    """Differentiable truncated signature with zero basepoint prepended."""
    if path.size(1) < 2:
        raise ValueError(
            f"signature needs at least 2 time steps after basepoint; got length {path.size(1)}"
        )
    return signatory.signature(path, depth, basepoint=True)


@dataclass
class SignatureMSELossOutput:
    loss: torch.Tensor
    loss_mse: torch.Tensor
    loss_sig: torch.Tensor
    loss_sig_raw: torch.Tensor


class SignatureMSELoss(nn.Module):
    """``alpha * MSE + beta * ||Sig(y_hat) - Sig(y)||`` with optional sig normalization.

    Signature values grow quickly with depth and channel count. Set ``normalize_sig=True``
    (default) to divide the signature term by the detached target signature norm so
    ``beta`` is on a comparable scale to MSE. Without normalization, start with very
    small ``beta`` (e.g. 1e-4–1e-2) and ramp up after logging raw magnitudes.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.1,
        depth: int = 3,
        use_cumsum: bool = False,
        normalize_sig: bool = True,
        sig_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.depth = int(depth)
        self.use_cumsum = bool(use_cumsum)
        self.normalize_sig = bool(normalize_sig)
        self.sig_eps = float(sig_eps)

    def signature_channels(self, n_features: int) -> int:
        """Channel count after time augmentation (``n_features + 1``)."""
        return signatory.signature_channels(n_features + 1, self.depth)

    def _signature_l2(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        path_hat = prepare_signature_path(y_hat, use_cumsum=self.use_cumsum)
        path_y = prepare_signature_path(y, use_cumsum=self.use_cumsum)

        sig_hat = truncated_signature(path_hat, self.depth)
        sig_y = truncated_signature(path_y, self.depth)

        diff = sig_hat - sig_y
        raw = torch.linalg.vector_norm(diff, ord=2, dim=-1)
        if self.normalize_sig:
            denom = torch.linalg.vector_norm(sig_y.detach(), ord=2, dim=-1).clamp_min(self.sig_eps)
            scaled = raw / denom
            return scaled.mean(), raw.mean()
        return raw.mean(), raw.mean()

    def forward(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        return_parts: bool = False,
    ) -> torch.Tensor | SignatureMSELossOutput:
        loss_mse = F.mse_loss(y_hat, y)
        loss_sig, loss_sig_raw = self._signature_l2(y_hat, y)
        loss = self.alpha * loss_mse + self.beta * loss_sig

        if return_parts:
            return SignatureMSELossOutput(
                loss=loss,
                loss_mse=loss_mse,
                loss_sig=loss_sig,
                loss_sig_raw=loss_sig_raw,
            )
        return loss


def _synthetic_smoke_test() -> None:
    """Minimal gradient check on random multivariate paths."""
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch, time, features = 4, 32, 5

    model = nn.Linear(features, features, bias=False).to(device)
    criterion = SignatureMSELoss(alpha=1.0, beta=0.25, depth=3, use_cumsum=False).to(device)

    y = torch.randn(batch, time, features, device=device)
    y_hat = model(y)
    out = criterion(y_hat, y, return_parts=True)
    assert isinstance(out, SignatureMSELossOutput)
    out.loss.backward()

    assert any(p.grad is not None for p in model.parameters())
    print(
        f"[smoke] device={device} loss={out.loss.item():.4f} "
        f"mse={out.loss_mse.item():.4f} sig={out.loss_sig.item():.4f} "
        f"sig_raw={out.loss_sig_raw.item():.4f} sig_dim={criterion.signature_channels(features)}"
    )


if __name__ == "__main__":
    _synthetic_smoke_test()
