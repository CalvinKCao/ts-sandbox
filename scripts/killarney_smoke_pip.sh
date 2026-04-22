#!/bin/bash
# =============================================================================
# Alliance Killarney — same pip + import checks as slurm_etth2_compare.sh jobs.
#
# Run on a compute node (recommended) so $SLURM_TMPDIR exists and matches jobs:
#   salloc --time=0:30:00 --account=aip-boyuwang --gres=gpu:l40s:1 --mem=16G --cpus-per-task=4
#   bash $SCRATCH/ts-sandbox/scripts/killarney_smoke_pip.sh
#
# Or from an interactive GPU session you already have. Login-only works for pip,
# but torch wheels may differ; prefer one short salloc.
#
# Keeps logic aligned with PREAMBLE in slurm_etth2_compare.sh — update both if you change deps.
# =============================================================================
set -euo pipefail

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

ROOT="${SLURM_TMPDIR:-/tmp}/killarney-smoke-pip-$$"
virtualenv --no-download "$ROOT/env"
# shellcheck disable=SC1090
source "$ROOT/env/bin/activate"
pip install --no-index --upgrade pip -q

if pip install --no-index torch torchvision numpy pandas scipy scikit-learn tqdm -q 2>/dev/null; then
    :
else
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
    pip install numpy pandas scipy scikit-learn tqdm -q
fi

pip install "wandb>=0.25.0" optuna matplotlib einops -q
pip install "reformer-pytorch==1.4.4" -q

echo "== imports =="
python - <<'PY'
import einops
import matplotlib
import numpy
import optuna
import pandas
import scipy
import sklearn
import torch
import torchvision
import tqdm
import wandb
from reformer_pytorch import LSHSelfAttention  # PyPI package name reformer-pytorch

print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
print("reformer-pytorch import OK")
print("ALL OK")
PY

rm -rf "$ROOT"
echo "== done (temp venv removed) =="
