"""Hybrid MSE + truncated multivariate signature loss for iTransformer forecasts."""

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
    """Augment paths with a monotone time channel for signature computation.

    Args:
        y: ``[batch, time, features]`` multivariate path (all channels kept jointly).
        use_cumsum: if True, integrate increments along time (helps mean-reverting series).

    Returns:
        ``[batch, time, features + 1]`` with monotone time in ``[..., 0]``.
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


def extract_overlapping_patches(
    y: torch.Tensor,
    *,
    patch_size: int,
    stride: int,
) -> torch.Tensor:
    """Slide fixed windows along the forecast horizon.

    Args:
        y: ``[batch, time, features]``
        patch_size: window length (e.g. 24 on a 96-step horizon).
        stride: step between window starts (e.g. 12 for 50% overlap).

    Returns:
        ``[batch, num_patches, patch_size, features]``
    """
    if y.dim() != 3:
        raise ValueError(f"expected [B, T, C], got shape {tuple(y.shape)}")
    time_len = y.size(1)
    if time_len < patch_size:
        raise ValueError(
            f"sequence length {time_len} < patch_size {patch_size}; "
            "cannot extract overlapping patches"
        )
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    # unfold along time: [B, P, C, W] -> [B, P, W, C]
    patches = y.unfold(dimension=1, size=patch_size, step=stride)
    return patches.permute(0, 1, 3, 2).contiguous()


def truncated_signature(path: torch.Tensor, depth: int) -> torch.Tensor:
    """Differentiable truncated signature on a multivariate path (joint channels)."""
    if path.size(1) < 2:
        raise ValueError(
            f"signature needs at least 2 time steps after basepoint; got length {path.size(1)}"
        )
    return signatory.signature(path, depth, basepoint=True)


def truncated_logsignature(
    path: torch.Tensor,
    depth: int,
    *,
    mode: str = "words",
) -> torch.Tensor:
    """Truncated log-signature (Lyndon/word basis) for a multivariate path."""
    if path.size(1) < 2:
        raise ValueError(
            f"logsignature needs at least 2 time steps after basepoint; got length {path.size(1)}"
        )
    return signatory.logsignature(path, depth, basepoint=True, mode=mode)


def logsignature_dim(n_path_channels: int, depth: int) -> int:
    """Channel count for ``mode='words'`` log-signature."""
    return signatory.logsignature_channels(n_path_channels, depth)


def signature_dim(n_path_channels: int, depth: int) -> int:
    return signatory.signature_channels(n_path_channels, depth)


@dataclass
class SignatureMSELossOutput:
    loss: torch.Tensor
    loss_mse: torch.Tensor
    loss_sig: torch.Tensor
    loss_sig_raw: torch.Tensor


class SignatureMSELoss(nn.Module):
    """``alpha * MSE + beta * mean_patch ||Sig(y_hat) - Sig(y)||`` on overlapping subpatches.

    Signature is computed on the full multivariate path per patch (all channels jointly,
    including cross-integrals). Each patch is time-augmented before ``signatory.signature``.

    Default subpatch scheme for a 96-step horizon: window 24, stride 12 (50% overlap).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.1,
        depth: int = 3,
        use_cumsum: bool = False,
        normalize_sig: bool = True,
        sig_eps: float = 1e-6,
        patch_size: int = 24,
        stride: int = 12,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if patch_size < 2:
            raise ValueError("patch_size must be >= 2")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.depth = int(depth)
        self.use_cumsum = bool(use_cumsum)
        self.normalize_sig = bool(normalize_sig)
        self.sig_eps = float(sig_eps)
        self.patch_size = int(patch_size)
        self.stride = int(stride)

    def num_patches(self, sequence_length: int) -> int:
        if sequence_length < self.patch_size:
            return 0
        return (sequence_length - self.patch_size) // self.stride + 1

    def signature_channels(self, n_features: int) -> int:
        """Channel count after time augmentation (``n_features + 1``)."""
        return signatory.signature_channels(n_features + 1, self.depth)

    def _patch_signature_l2(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        patches_hat = extract_overlapping_patches(
            y_hat, patch_size=self.patch_size, stride=self.stride
        )
        patches_y = extract_overlapping_patches(
            y, patch_size=self.patch_size, stride=self.stride
        )

        batch, num_patches, _, _ = patches_hat.shape
        flat_hat = patches_hat.reshape(batch * num_patches, self.patch_size, -1)
        flat_y = patches_y.reshape(batch * num_patches, self.patch_size, -1)

        path_hat = prepare_signature_path(flat_hat, use_cumsum=self.use_cumsum)
        path_y = prepare_signature_path(flat_y, use_cumsum=self.use_cumsum)

        sig_hat = truncated_signature(path_hat, self.depth)
        sig_y = truncated_signature(path_y, self.depth)

        diff = sig_hat - sig_y
        raw = torch.linalg.vector_norm(diff, ord=2, dim=-1)
        if self.normalize_sig:
            denom = torch.linalg.vector_norm(sig_y.detach(), ord=2, dim=-1).clamp_min(self.sig_eps)
            scaled = raw / denom
            return scaled.mean(), raw.mean()
        return raw.mean(), raw.mean()

    def _signature_l2(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._patch_signature_l2(y_hat, y)

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
    """Minimal gradient check: 8-variate horizon-96 paths with overlapping subpatches."""
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch, time, features = 4, 96, 8

    model = nn.Linear(features, features, bias=False).to(device)
    criterion = SignatureMSELoss(
        alpha=1.0,
        beta=0.25,
        depth=3,
        use_cumsum=False,
        patch_size=24,
        stride=12,
    ).to(device)

    y = torch.randn(batch, time, features, device=device)
    y_hat = model(y)
    assert criterion.num_patches(time) == 7

    out = criterion(y_hat, y, return_parts=True)
    assert isinstance(out, SignatureMSELossOutput)
    out.loss.backward()

    assert any(p.grad is not None for p in model.parameters())
    print(
        f"[smoke] device={device} patches={criterion.num_patches(time)} "
        f"loss={out.loss.item():.4f} mse={out.loss_mse.item():.4f} "
        f"sig={out.loss_sig.item():.4f} sig_raw={out.loss_sig_raw.item():.4f} "
        f"sig_dim={criterion.signature_channels(features)}"
    )


if __name__ == "__main__":
    _synthetic_smoke_test()
