#!/bin/bash
# =============================================================================
# Pure compute-node worker script for diffusion pipeline.
#
# USAGE (do not call directly; use submit_grid.sh):
#   sbatch slurm_worker.sh --config configs/binary_anchor.yaml
# =============================================================================

set -euo pipefail

echo "=========================================="
echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID  Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "Started: $(date)"
echo "=========================================="

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

echo "[setup] Building venv on \$SLURM_TMPDIR..."
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q

# Fast wheel cache installs
pip install --no-index torch torchvision numpy pandas scipy scikit-learn tqdm -q 2>/dev/null || \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q && \
    pip install numpy pandas scipy scikit-learn tqdm -q

# Packages not in cache
pip install optuna wandb einops pyyaml -q
VIZ_WHEELS_OK=0
for _try in 1 2 3; do
    if pip install --no-index matplotlib -q; then
        VIZ_WHEELS_OK=1
        break
    fi
    sleep 30
done

export PYTHONUNBUFFERED=1

# Change to submit dir (repo root)
cd "$SLURM_SUBMIT_DIR"
if [[ ! -f "models/diffusion_tsf/train_multivariate_pipeline.py" ]]; then
    echo "ERROR: slurm_worker.sh must be submitted from repo root." >&2
    exit 1
fi

echo "[train] Starting pipeline with args: $*"
python -u -m models.diffusion_tsf.train_multivariate_pipeline "$@"

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
