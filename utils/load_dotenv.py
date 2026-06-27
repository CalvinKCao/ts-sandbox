"""Load repo-root `.env` into os.environ (no extra dependency)."""

from __future__ import annotations

import os
from typing import Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_LOADED = False


def load_repo_dotenv(repo_root: Optional[str] = None, *, override: bool = False) -> bool:
    """Parse ``{repo_root}/.env``. Returns True if the file was read."""
    global _LOADED
    root = os.path.abspath(repo_root or _REPO_ROOT)
    path = os.path.join(root, ".env")
    if not os.path.isfile(path):
        return False

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            if not key:
                continue
            if (val.startswith('"') and val.endswith('"')) or (
                val.startswith("'") and val.endswith("'")
            ):
                val = val[1:-1]
            if not override and key in os.environ and os.environ[key]:
                continue
            os.environ[key] = val

    _LOADED = True
    return True
