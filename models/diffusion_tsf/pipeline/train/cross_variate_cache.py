"""Phase-scoped GPU cache for frozen patch-decoder cross-variate tokens."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def raw_window_key(past: torch.Tensor) -> str:
    """Stable key for a raw window, independent of loader order or epoch."""
    cpu = past.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(cpu.numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class CachedCrossVariateContext:
    tokens: torch.Tensor
    token_variate_ids: Optional[torch.Tensor]


def _unwrap_cache_source(dataset) -> Tuple[object, List[int]]:
    from torch.utils.data import Subset

    base = dataset
    selected_indices = list(range(len(dataset)))
    while isinstance(base, Subset):
        selected_indices = [int(base.indices[i]) for i in selected_indices]
        base = base.dataset
    return base, selected_indices


def count_cache_windows(dataset) -> int:
    """Host rows that ``precompute_dataset`` will encode for ``dataset``."""
    from models.diffusion_tsf.patch_refine_segments import (
        UniquePatchSegmentDataset,
        parent_starts_for_segment,
    )

    base, selected_indices = _unwrap_cache_source(dataset)
    if not isinstance(base, UniquePatchSegmentDataset):
        return len(selected_indices)
    parents = set()
    for segment_index in selected_indices:
        segment_start = base.segment_starts[segment_index]
        for parent in parent_starts_for_segment(
            int(segment_start), lookback=base.lookback, horizon=base.horizon,
            overlap=base.overlap, patch_width=base.patch_width,
            series_len=int(base.data.shape[0]),
        ):
            parents.add(parent)
    return len(parents)


class CrossVariateTokenCache:
    """Frozen iTransformer encoder tokens keyed by immutable raw lookback windows.

    Stores ``(V, itrans_d_model)`` encoder tokens only — not post-adapter
    context. DiT train/val looks up a batch then runs ``iTransformerTokenAdapter``
    live (trainable, dropout on train). Tokens are built once before the first
    trial and reused across epochs and HP trials. ``pinned_cpu`` keeps the full
    cache out of VRAM, then transfers only the requested batch asynchronously.
    ``gpu`` is retained for explicit no-transfer experiments. A cache miss is
    an error: silent live fallback would make timing and numerical comparisons
    ambiguous.
    """

    def __init__(
        self,
        *,
        model,
        device: torch.device,
        storage: str = "pinned_cpu",
        token_kind: str = "mixed",
    ) -> None:
        if model.config.disable_cross_attention:
            raise ValueError("CrossVariateTokenCache requires cross-attention enabled")
        if storage not in {"pinned_cpu", "gpu"}:
            raise ValueError(
                "CrossVariateTokenCache storage must be 'pinned_cpu' or 'gpu', "
                f"got {storage!r}"
            )
        if storage == "pinned_cpu" and device.type != "cuda":
            raise ValueError("pinned_cpu token caching requires a CUDA training device")
        if token_kind != "mixed":
            raise ValueError(
                f"token_kind must be 'mixed' (frozen encoder tokens); got {token_kind!r}. "
                "pre_mixer caching has been removed."
            )
        self.model = model
        self.device = device
        self.storage = storage
        self.token_kind = token_kind
        self._entries: Dict[str, torch.Tensor | int] = {}
        self._packed_cpu: Optional[torch.Tensor] = None
        self._token_variate_ids: Optional[torch.Tensor] = None
        self._reserved_n = 0
        self._next_idx = 0
        self._finalize_logged = False

    @property
    def n_entries(self) -> int:
        return len(self._entries)

    @property
    def bytes(self) -> int:
        if self._packed_cpu is not None:
            used = int(self._next_idx)
            if used == 0:
                return 0
            return used * self._packed_cpu[0].numel() * self._packed_cpu.element_size()
        return sum(
            t.numel() * t.element_size()
            for t in self._entries.values()
            if isinstance(t, torch.Tensor)
        )

    @property
    def _memory_label(self) -> str:
        return "gpu_memory" if self.storage == "gpu" else "pinned_cpu_memory"

    def reserve(self, n: int) -> None:
        """Reserve pinned packed rows before the first ``add``.

        Call once with train+val window counts so packing never keeps a live
        list of individual tensors. Cannot grow after the packed buffer exists.
        """
        n = int(n)
        if n < 1:
            raise ValueError(f"token cache reserve must be >= 1, got {n}")
        if self._packed_cpu is not None:
            if n > self._packed_cpu.shape[0]:
                raise RuntimeError(
                    f"cannot grow pinned cache after allocation "
                    f"({self._packed_cpu.shape[0]} -> {n})"
                )
            return
        self._reserved_n = max(int(self._reserved_n), n)

    def _store_cpu_token(self, token: torch.Tensor) -> int:
        cpu_token = token.detach().cpu()
        if self._packed_cpu is None:
            reserved = int(self._reserved_n)
            if reserved < 1:
                raise RuntimeError(
                    "pinned_cpu CrossVariateTokenCache requires reserve(n) before add()"
                )
            self._packed_cpu = torch.empty(
                (reserved, *cpu_token.shape),
                dtype=cpu_token.dtype,
                device="cpu",
                pin_memory=True,
            )
            self._next_idx = 0
        idx = int(self._next_idx)
        if idx >= self._packed_cpu.shape[0]:
            raise RuntimeError(
                f"token cache exceeded reserve {self._packed_cpu.shape[0]}"
            )
        self._packed_cpu[idx].copy_(cpu_token)
        self._next_idx = idx + 1
        return idx

    @torch.no_grad()
    def add(self, past: torch.Tensor) -> None:
        """Encode windows not already present, retaining native GPU dtype."""
        if self.model is None:
            raise RuntimeError("token-cache encoder was released before precompute completed")
        if past.ndim != 3:
            raise ValueError(f"past must be (B,V,L), got {tuple(past.shape)}")
        keys = [raw_window_key(row) for row in past]
        missing = [i for i, key in enumerate(keys) if key not in self._entries]
        if not missing:
            return
        source = past[missing].to(self.device, non_blocking=True)
        tokens = self.model._encode_frozen_encoder_tokens(source)
        if tokens.dim() != 3:
            raise RuntimeError(
                f"encoder-token cache expects (B, V, d_model), got {tuple(tokens.shape)}"
            )
        d_model = int(self.model.config.itrans_d_model)
        if tokens.shape[-1] != d_model:
            raise RuntimeError(
                f"encoder-token cache last dim {tokens.shape[-1]} != itrans_d_model {d_model}"
            )
        for local_i, batch_i in enumerate(missing):
            token = tokens[local_i]
            if self.storage == "gpu":
                self._entries[keys[batch_i]] = token.detach().clone()
            else:
                self._entries[keys[batch_i]] = self._store_cpu_token(token)

    def _finalize_pinned_cpu(self) -> None:
        if self.storage != "pinned_cpu":
            return
        if self._packed_cpu is None:
            raise RuntimeError("cannot finalize an empty pinned CPU token cache")
        if getattr(self, "_finalize_logged", False):
            return
        used = int(self._next_idx)
        reserved = int(self._packed_cpu.shape[0])
        if used < reserved:
            logger.info(
                "  [cross-variate-cache] packed using %d/%d reserved slots",
                used, reserved,
            )
        self._finalize_logged = True

    def _pinned_cpu_batch(self, indices: List[int]) -> torch.Tensor:
        self._finalize_pinned_cpu()
        if self._packed_cpu is None:
            raise RuntimeError("pinned CPU token cache was not finalized")
        cpu_indices = torch.tensor(indices, dtype=torch.long)
        batch = torch.empty(
            (len(indices), *self._packed_cpu.shape[1:]),
            dtype=self._packed_cpu.dtype,
            device="cpu",
            pin_memory=True,
        )
        batch.copy_(self._packed_cpu.index_select(0, cpu_indices))
        return batch.to(self.device, non_blocking=True)

    def get(self, past: torch.Tensor) -> CachedCrossVariateContext:
        keys = [raw_window_key(row) for row in past]
        missing = [key for key in keys if key not in self._entries]
        if missing:
            raise KeyError(
                f"CrossVariateTokenCache miss for {len(missing)} window(s); "
                "build the phase cache before training."
            )
        if self.storage == "gpu":
            tokens = torch.stack(
                [self._entries[key] for key in keys], dim=0,
            )
        else:
            tokens = self._pinned_cpu_batch([int(self._entries[key]) for key in keys])
        return CachedCrossVariateContext(tokens=tokens, token_variate_ids=self._token_variate_ids)

    def release_encoder(self) -> None:
        """Finalize host storage and drop the model used only for cache construction."""
        self._finalize_pinned_cpu()
        self.model = None

    @torch.no_grad()
    def precompute_loader(self, loader) -> None:
        """Populate from a stable loader without touching train augmentations."""
        for batch in loader:
            past = batch[0] if isinstance(batch, (tuple, list)) else batch["past"]
            self.add(past)
        logger.info(
            "  [cross-variate-cache] storage=%s entries=%d %s=%.1f MiB",
            self.storage, self.n_entries, self._memory_label, self.bytes / 2**20,
        )

    @torch.no_grad()
    def precompute_dataset(self, dataset, *, batch_size: int) -> None:
        """Populate every stable parent window, including unique-segment parents."""
        from torch.utils.data import DataLoader
        from models.diffusion_tsf.patch_refine_segments import (
            UniquePatchSegmentDataset,
            parent_starts_for_segment,
        )

        n_expected = count_cache_windows(dataset)
        if self.storage == "pinned_cpu" and self._packed_cpu is None:
            self.reserve(max(int(self._reserved_n), n_expected))
        base, selected_indices = _unwrap_cache_source(dataset)
        if not isinstance(base, UniquePatchSegmentDataset):
            self.precompute_loader(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0))
            return
        source = base.rank_data if base.rank_data is not None else base.data
        parents = set()
        for segment_index in selected_indices:
            segment_start = base.segment_starts[segment_index]
            for parent in parent_starts_for_segment(
                int(segment_start), lookback=base.lookback, horizon=base.horizon,
                overlap=base.overlap, patch_width=base.patch_width,
                series_len=int(base.data.shape[0]),
            ):
                parents.add(parent)
        ordered = sorted(parents)
        for offset in range(0, len(ordered), batch_size):
            starts = ordered[offset : offset + batch_size]
            past = torch.stack([source[s : s + base.lookback].T for s in starts])
            self.add(past)
        logger.info(
            "  [cross-variate-cache] storage=%s unique-segment parents=%d entries=%d %s=%.1f MiB",
            self.storage, len(ordered), self.n_entries, self._memory_label, self.bytes / 2**20,
        )

    def release(self) -> None:
        self._entries.clear()
        self._packed_cpu = None
        self._token_variate_ids = None
        self._reserved_n = 0
        self._next_idx = 0
        self._finalize_logged = False
        self.model = None
