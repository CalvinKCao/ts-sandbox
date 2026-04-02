#!/bin/bash
#SBATCH --job-name=latent-dim1
#SBATCH --account=aip-boyuwang
# L40S (Standard tier): request GPU type via GRES; do NOT use gpubase_h100_* here — those partitions are H100-only.
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ccao87@uwo.ca     # CHANGE to your email
#SBATCH --signal=B:USR1@120            # Send USR1 120s before wall-time kill

# =============================================================================
# Slurm: full 1-variate latent diffusion experiment (VAE → iTransformer → LDM → ETTh1)
# =============================================================================
#
# Prerequisites on Killarney:
#   - Repo under $SCRATCH/ts-sandbox (not /home)
#   - datasets/ETT-small/ETTh1.csv inside that repo
#   - One-time: ./setup/alliance_setup_killarney.sh
#
# Usage (from scratch repo root):
#   sbatch slurm_latent_experiment.sh
#   sbatch slurm_latent_experiment.sh -- --image-height 96
#   sbatch slurm_latent_experiment.sh -- --skip-vae-train
#   sbatch slurm_latent_experiment.sh -- --smoke-test
#
# Edit --account / --mail-user in the header to match your allocation.
#
# GPU choice (Killarney):
#   - This file: L40S via --gres=gpu:l40s:1 (same pattern as slurm_unet_fullvar.sh smoke). Often shorter queue.
#   - H100 long runs: comment out --gres=gpu:l40s:1 and use e.g.:
#       #SBATCH --partition=gpubase_h100_b4
#       #SBATCH --gpus-per-node=h100:1
#   - Confirm names on the cluster: sinfo -o "%P %G"  and  scontrol show node | grep -i gres
# =============================================================================

set -e

# Slurm redirects .out to a file → Python stdout/stderr are block-buffered unless unbuffered.
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1 || echo 'unknown')"
echo "Started: $(date)"
echo "=========================================="

module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

if [ -d "$SCRATCH/ts-sandbox" ]; then
    export PROJECT_ROOT="$SCRATCH/ts-sandbox"
elif [ -d "$HOME/ts-sandbox" ]; then
    export PROJECT_ROOT="$HOME/ts-sandbox"
else
    echo "ERROR: ts-sandbox not found in SCRATCH or HOME"
    exit 1
fi

if [ -z "$PROJECT" ]; then
    if [ -d "$HOME/projects" ]; then
        FIRST_PROJECT=$(ls -d $HOME/projects/def-* $HOME/projects/aip-* 2>/dev/null | head -1)
        [ -n "$FIRST_PROJECT" ] && export PROJECT=$(readlink -f "$FIRST_PROJECT")
    fi
fi

if [ -z "$PROJECT" ]; then
    echo "ERROR: PROJECT not found"
    exit 1
fi

export STORAGE_ROOT="$PROJECT/$USER/diffusion-tsf"
mkdir -p "$STORAGE_ROOT/checkpoints" "$STORAGE_ROOT/synthetic_cache" "$STORAGE_ROOT/results"

# Same venv contract as other Slurm scripts: $PROJECT/$USER/diffusion-tsf/venv + deps.
VENV_PATH="$STORAGE_ROOT/venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment at $VENV_PATH (first run)..."
    python -m venv "$VENV_PATH"
    export PATH="$VENV_PATH/bin:$PATH"
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install numpy pandas scipy scikit-learn optuna wandb tqdm matplotlib einops reformer_pytorch
    [ -f "$PROJECT_ROOT/requirements.txt" ] && pip install -r "$PROJECT_ROOT/requirements.txt"
else
    export PATH="$VENV_PATH/bin:$PATH"
fi
# shellcheck source=/dev/null
source "$VENV_PATH/bin/activate"

PY="$VENV_PATH/bin/python"
if ! "$PY" -c "import torch" 2>/dev/null; then
    echo "venv at $VENV_PATH exists but torch is missing; installing..."
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install numpy pandas scipy scikit-learn optuna wandb tqdm matplotlib einops reformer_pytorch
    [ -f "$PROJECT_ROOT/requirements.txt" ] && pip install -r "$PROJECT_ROOT/requirements.txt"
fi
echo "Python: $($PY -c 'import sys; print(sys.executable)')"
echo "Torch: $($PY -c 'import torch; print(torch.__version__)')"

LATENT_CACHE="$STORAGE_ROOT/synthetic_cache/latent_dim1"
mkdir -p "$LATENT_CACHE"

cd "$PROJECT_ROOT"

if [ ! -f datasets/ETT-small/ETTh1.csv ]; then
    if [ -f "$STORAGE_ROOT/datasets/ETT-small/ETTh1.csv" ]; then
        echo "Symlinking datasets from PROJECT storage..."
        ln -sf "$STORAGE_ROOT/datasets" "$PROJECT_ROOT/datasets"
    else
        echo "ERROR: Missing datasets/ETT-small/ETTh1.csv under $PROJECT_ROOT"
        exit 1
    fi
fi

cleanup() {
    trap '' EXIT ERR SIGTERM SIGINT SIGUSR1
    local code=${1:-$?}
    [ "$code" -ne 0 ] && echo "[SLURM CLEANUP] $(date)"
    kill -- -$$ 2>/dev/null || true
    pkill -P $$ 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT ERR SIGTERM SIGINT SIGUSR1

EXTRA_ARGS=""
for a in "$@"; do
    [ "$a" = "--" ] && continue
    EXTRA_ARGS="$EXTRA_ARGS $a"
done

echo ""
echo "Running: $PY -u -m models.diffusion_tsf.train_latent_experiment --stage all --cache-dir $LATENT_CACHE $EXTRA_ARGS"
echo ""

"$PY" -u -m models.diffusion_tsf.train_latent_experiment \
    --stage all \
    --cache-dir "$LATENT_CACHE" \
    $EXTRA_ARGS

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "Checkpoints: $PROJECT_ROOT/models/diffusion_tsf/checkpoints_latent/"
echo "Results JSON: $PROJECT_ROOT/models/diffusion_tsf/results/"
echo "=========================================="
