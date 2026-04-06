"""
Fast sanity checks for CI-latent training (imports, CUDA, datasets, checkpoint dirs).

Use before long Slurm jobs so failures show up in minutes, not after queue + instant crash.

  # After same module/venv setup as jobs (or inside slurm_ci_latent_precheck.sh):
  python -m models.diffusion_tsf.ci_latent_precheck                    # optional CUDA matmul if GPU present
  python -m models.diffusion_tsf.ci_latent_precheck --require-cuda    # fail if no GPU (match Slurm precheck job)
  python -m models.diffusion_tsf.ci_latent_precheck --no-cuda        # login node
  python -m models.diffusion_tsf.ci_latent_precheck --no-cuda --dataset ETTh2  # one CSV only

Exits 0 only if all checked steps pass.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


def _ok(msg: str) -> None:
    print(f"[ci_latent_precheck] OK  {msg}")


def _fail(msg: str) -> None:
    print(f"[ci_latent_precheck] FAIL {msg}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser(description="CI-latent environment / import precheck")
    p.add_argument(
        "--require-cuda",
        action="store_true",
        help="exit 1 if CUDA is not available (use on GPU batch jobs)",
    )
    p.add_argument(
        "--no-cuda",
        action="store_true",
        help="skip GPU checks (login node or CPU-only)",
    )
    p.add_argument(
        "--skip-heavy-import",
        action="store_true",
        help="only torch/optuna/deps; skip importing train_ci_latent_etth2 (faster)",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="",
        metavar="NAME",
        help="only check CSV for this DATASET_REGISTRY name (e.g. ETTh2); default checks every registry dataset",
    )
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    failed = False

    def check(name: str, fn) -> None:
        nonlocal failed
        try:
            fn()
            _ok(name)
        except Exception as e:
            failed = True
            _fail(f"{name}: {e}")

    def imp_torch():
        import torch

        assert hasattr(torch, "nn")

    def imp_stack():
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import einops  # noqa: F401
        import optuna  # noqa: F401
        import tqdm  # noqa: F401
        import matplotlib  # noqa: F401

    def imp_reformer():
        import reformer_pytorch  # noqa: F401

    def imp_train():
        importlib.import_module("models.diffusion_tsf.train_ci_latent_etth2")

    def imp_ldm():
        from models.diffusion_tsf.latent_diffusion_model import LatentDiffusionTSF  # noqa: F401
        from models.diffusion_tsf.vae import TimeSeriesVAE  # noqa: F401

    def datasets():
        from models.diffusion_tsf.latent_experiment_common import DATASET_REGISTRY, dataset_registry_row

        if args.dataset.strip():
            names = [args.dataset.strip()]
            for name in names:
                if name not in DATASET_REGISTRY:
                    raise ValueError(
                        f"unknown dataset {name!r}; registry has {sorted(DATASET_REGISTRY.keys())}",
                    )
        else:
            names = sorted(DATASET_REGISTRY.keys())
        for name in names:
            rel, _, _, _ = dataset_registry_row(name)
            csv = project_root / "datasets" / rel
            if not csv.is_file():
                raise FileNotFoundError(f"registry dataset {name}: missing {csv}")

    def dirs():
        (script_dir / "checkpoints_latent").mkdir(parents=True, exist_ok=True)
        (script_dir / "checkpoints_ci_etth2").mkdir(parents=True, exist_ok=True)
        (script_dir / "checkpoints_ci_runs").mkdir(parents=True, exist_ok=True)

    def cuda_skip():
        pass

    def cuda_required():
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False (--require-cuda)")
        d = torch.device("cuda:0")
        x = torch.randn(4, 4, device=d)
        y = x @ x
        torch.cuda.synchronize()
        assert y.shape == (4, 4)
        print(f"[ci_latent_precheck]      GPU {torch.cuda.get_device_name(0)!r}")

    def cuda_optional():
        import torch

        if not torch.cuda.is_available():
            print("[ci_latent_precheck]      (no GPU — skipped; use GPU job + --require-cuda to test CUDA)")
            return
        d = torch.device("cuda:0")
        x = torch.randn(4, 4, device=d)
        y = x @ x
        torch.cuda.synchronize()
        assert y.shape == (4, 4)
        print(f"[ci_latent_precheck]      GPU {torch.cuda.get_device_name(0)!r}")

    check("import torch", imp_torch)
    check("import numpy/pandas/optuna/einops/tqdm/matplotlib", imp_stack)
    check("import reformer_pytorch", imp_reformer)
    check("import LatentDiffusionTSF + TimeSeriesVAE", imp_ldm)
    if not args.skip_heavy_import:
        check("import train_ci_latent_etth2 (full dep chain)", imp_train)
    ds_label = (
        f"CSV for dataset {args.dataset.strip()!r}"
        if args.dataset.strip()
        else "DATASET_REGISTRY CSV files (all)"
    )
    check(ds_label, datasets)
    check("checkpoint dirs creatable under models/diffusion_tsf/", dirs)
    if args.no_cuda:
        check("CUDA (--no-cuda)", cuda_skip)
    elif args.require_cuda:
        check("CUDA (required)", cuda_required)
    else:
        check("CUDA (optional matmul)", cuda_optional)

    if failed:
        print("[ci_latent_precheck] Exiting 1 (see FAIL lines above)", file=sys.stderr)
        return 1
    print("[ci_latent_precheck] All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
