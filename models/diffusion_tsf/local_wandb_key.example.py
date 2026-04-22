"""Optional: copy to local_wandb_key.py (gitignored), implement apply() with your key,
and import it from a custom entrypoint.

Prefer repo-root wandb_api_key.txt (see wandb_api_key.example.txt) — the pipeline loads
that automatically — or export WANDB_API_KEY in ~/.bashrc on clusters.
"""

import os


def apply() -> None:
    os.environ["WANDB_API_KEY"] = "YOUR_WANDB_API_KEY_HERE"
