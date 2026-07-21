"""512 smoke: same vertical-only / 8x8 path at resolution 512."""

from __future__ import annotations

import sys

from experiments.ordinal_patch_refinement_killtest import smoke


if __name__ == "__main__":
    # Preserve CLI; force resolution=512 if not provided.
    if "--resolution" not in sys.argv:
        sys.argv.extend(["--resolution", "512"])
    if "--output" not in sys.argv:
        sys.argv.extend(["--output", str(smoke.DEFAULT_OUTPUT.parent / "smoke-512")])
    smoke.main()
