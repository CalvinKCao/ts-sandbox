"""Copy to local_wandb_key.py (gitignored) and set your key.

  cp models/diffusion_tsf/local_wandb_key.example.py models/diffusion_tsf/local_wandb_key.py

Prefer export WANDB_API_KEY=... in ~/.bashrc on clusters so nothing sensitive lives in the repo.
"""

import os


def apply() -> None:
    os.environ["WANDB_API_KEY"] = "PASTE_YOUR_KEY_HERE"
