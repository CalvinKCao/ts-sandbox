#!/usr/bin/env python3
"""Verify that the active GPU environment can run Torch Inductor."""

import torch
import triton


print(f"torch={torch.__version__}", flush=True)
print(f"triton={triton.__version__}", flush=True)
compiled = torch.compile(lambda x: x.sin() + x.cos(), backend="inductor")
value = compiled(torch.randn(1024, device="cuda").contiguous()).mean().item()
print(f"compiled_mean={value}", flush=True)
