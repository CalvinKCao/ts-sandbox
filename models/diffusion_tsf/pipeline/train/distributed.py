"""Torch DDP helpers for the binary pipeline. No-op when WORLD_SIZE<=1."""
from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)

_INITIALIZED = False


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return int(os.environ.get("RANK", "0"))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_rank0() -> bool:
    return rank() == 0


def is_distributed() -> bool:
    return world_size() > 1


def init_distributed() -> None:
    """Init NCCL when launched via torchrun. Fail-fast if CUDA/world_size mismatch."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    ws = world_size()
    if ws <= 1:
        _INITIALIZED = True
        return
    if not torch.cuda.is_available():
        raise RuntimeError(f"DDP WORLD_SIZE={ws} requires CUDA")
    n_visible = torch.cuda.device_count()
    lr = local_rank()
    if lr < 0 or lr >= n_visible:
        raise RuntimeError(
            f"LOCAL_RANK={lr} out of range for cuda.device_count()={n_visible}"
        )
    torch.cuda.set_device(lr)
    dist.init_process_group(backend="nccl", init_method="env://")
    if dist.get_world_size() != ws:
        raise RuntimeError(
            f"process group world_size={dist.get_world_size()} != WORLD_SIZE={ws}"
        )
    _INITIALIZED = True
    if is_rank0():
        logger.info(
            "DDP init ok backend=nccl world_size=%d rank=%d local_rank=%d gpu=%s",
            ws, rank(), lr, torch.cuda.get_device_name(lr),
        )


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def destroy_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def device_for_rank() -> torch.device:
    if torch.cuda.is_available():
        if is_distributed():
            return torch.device("cuda", local_rank())
        return torch.device("cuda")
    return torch.device("cpu")


def wrap_ddp(model: torch.nn.Module) -> torch.nn.Module:
    if not is_distributed():
        return model
    if not next(model.parameters()).is_cuda:
        raise RuntimeError("wrap_ddp requires the model on CUDA")
    lr = local_rank()
    return DDP(
        model,
        device_ids=[lr],
        output_device=lr,
        find_unused_parameters=False,
        broadcast_buffers=False,
    )


def unwrap_module(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def dummy_ddp_backward(model: torch.nn.Module) -> None:
    """Keep DDP reducer in lockstep when a rank skips a real loss."""
    raw = unwrap_module(model)
    acc = None
    for param in raw.parameters():
        if param.requires_grad:
            piece = param.view(-1)[0] * 0
            acc = piece if acc is None else acc + piece
    if acc is None:
        raise RuntimeError("DDP dummy backward: no trainable parameters")
    acc.backward()
