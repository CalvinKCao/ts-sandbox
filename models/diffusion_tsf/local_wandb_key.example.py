"""Optional: copy to local_wandb_key.py (gitignored), implement apply() with your key,
and import it from a custom entrypoint.

Prefer setting WANDB_API_KEY in your shell environment (~/.bashrc, job exports, CI secrets).
"""

import os


def apply() -> None:
    os.environ["WANDB_API_KEY"] = "YOUR_WANDB_API_KEY_HERE"
