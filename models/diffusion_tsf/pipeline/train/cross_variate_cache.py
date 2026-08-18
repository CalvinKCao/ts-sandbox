"""Phase-scoped GPU cache for frozen patch-decoder cross-variate tokens."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

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


class CrossVariateTokenCache:
    """Exact frozen tokens keyed by immutable raw lookback windows.

    Tokens are built once before the first trial and reused across epochs and
    HP trials. ``pinned_cpu`` keeps the full cache out of VRAM, then transfers
    only the requested batch asynchronously. ``gpu`` is retained for explicit
    no-transfer experiments. A cache miss is an error: silent live fallback
    would make timing and numerical comparisons ambiguous.
    """

    def __init__(
        self,
        *,
        model,
        device: torch.device,
        storage: str = "pinned_cpu",
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
        self.model = model
        self.device = device
        self.storage = storage
        self._entries: Dict[str, torch.Tensor | int] = {}
        self._cpu_entries: List[torch.Tensor] = []
        self._packed_cpu: Optional[torch.Tensor] = None
        self._token_variate_ids: Optional[torch.Tensor] = None

    @property
    def n_entries(self) -> int:
        return len(self._entries)

    @property
    def bytes(self) -> int:
        if self._packed_cpu is not None:
            return self._packed_cpu.numel() * self._packed_cpu.element_size()
        if self.storage == "pinned_cpu":
            return sum(t.numel() * t.element_size() for t in self._cpu_entries)
        return sum(
            t.numel() * t.element_size()
            for t in self._entries.values()
            if isinstance(t, torch.Tensor)
        )

    @property
    def _memory_label(self) -> str:
        return "gpu_memory" if self.storage == "gpu" else "pinned_cpu_memory"

    @torch.no_grad()
    def add(self, past: torch.Tensor) -> None:
        """Encode windows not already present, retaining native GPU dtype."""
        if self.model is None:
            raise RuntimeError("token-cache encoder was released before precompute completed")
        if self._packed_cpu is not None:
            raise RuntimeError("cannot add entries after pinned CPU cache finalization")
        if past.ndim != 3:
            raise ValueError(f"past must be (B,V,L), got {tuple(past.shape)}")
        keys = [raw_window_key(row) for row in past]
        missing = [i for i, key in enumerate(keys) if key not in self._entries]
        if not missing:
            return
        source = past[missing].to(self.device, non_blocking=True)
        from models.diffusion_tsf.pipeline.train.checkpointing import amp_context

        with amp_context(bool(self.model.config.use_amp)):
            norm, _, _ = self.model._normalize_sequence(source, None)
            tokens = self.model._get_cross_variate_context(source, norm)
        if tokens is None:
            raise RuntimeError("cross-attention unexpectedly produced no context tokens")
        ids = self.model._ctx_token_variate_ids
        if ids is not None:
            ids = ids.detach().to(self.device).clone()
            if self._token_variate_ids is None:
                self._token_variate_ids = ids
            elif not torch.equal(self._token_variate_ids, ids):
                raise RuntimeError("token variate IDs changed while building frozen cache")
        for local_i, batch_i in enumerate(missing):
            token = tokens[local_i].detach()
            if self.storage == "gpu":
                self._entries[keys[batch_i]] = token.clone()
            else:
                self._entries[keys[batch_i]] = len(self._cpu_entries)
                self._cpu_entries.append(token.cpu().clone())

    def _finalize_pinned_cpu(self) -> None:
        if self.storage != "pinned_cpu" or self._packed_cpu is not None:
            return
        if not self._cpu_entries:
            raise RuntimeError("cannot finalize an empty pinned CPU token cache")
        prototype = self._cpu_entries[0]
        n = len(self._cpu_entries)
        packed = torch.empty(
            (n, *prototype.shape),
            dtype=prototype.dtype,
            device="cpu",
            pin_memory=True,
        )
        # Row copies instead of stack(): stack() kept the list tensors live
        # and added a third full copy, which oom-killed 160-var jobs at 150G.
        for i in range(n):
            packed[i].copy_(self._cpu_entries[i])
            self._cpu_entries[i] = None
        self._cpu_entries.clear()
        self._packed_cpu = packed

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
        from torch.utils.data import DataLoader, Subset
        from models.diffusion_tsf.patch_refine_segments import (
            UniquePatchSegmentDataset,
            parent_starts_for_segment,
        )

        base = dataset
        selected_indices = list(range(len(dataset)))
        while isinstance(base, Subset):
            selected_indices = [int(base.indices[i]) for i in selected_indices]
            base = base.dataset
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
        self._cpu_entries.clear()
        self._packed_cpu = None
        self._token_variate_ids = None
        self.model = None
