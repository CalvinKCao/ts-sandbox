#!/bin/bash
#SBATCH --job-name=ci-latent-etth1
#SBATCH --account=aip-boyuwang
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ccao87@uwo.ca
#SBATCH --signal=B:USR1@120

# =============================================================================
# CI latent diffusion ablation on ETTh1 — guided vs unguided
# =============================================================================
#
# Submit BOTH variants:
#   sbatch --job-name=ci-guided   slurm_ci_latent_etth1.sh
#   sbatch --job-name=ci-unguided slurm_ci_latent_etth1.sh -- --no-guidance
#
# Smoke test:
#   sbatch --job-name=ci-smoke slurm_ci_latent_etth1.sh -- --smoke-test
#
# They share the same VAE and iTransformer checkpoints, so safe to run in
# parallel (first one to reach training creates the checkpoint, second one
# loads it).
# =============================================================================

set -e
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
source "$VENV_PATH/bin/activate"

PY="$VENV_PATH/bin/python"
if ! "$PY" -c "import torch" 2>/dev/null; then
    echo "venv exists but torch missing; installing..."
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install numpy pandas scipy scikit-learn optuna wandb tqdm matplotlib einops reformer_pytorch
    [ -f "$PROJECT_ROOT/requirements.txt" ] && pip install -r "$PROJECT_ROOT/requirements.txt"
fi
echo "Python: $($PY -c 'import sys; print(sys.executable)')"
echo "Torch: $($PY -c 'import torch; print(torch.__version__)')"

CACHE_DIR="$STORAGE_ROOT/synthetic_cache/ci_latent"
mkdir -p "$CACHE_DIR"

cd "$PROJECT_ROOT"

if [ ! -f datasets/ETT-small/ETTh1.csv ]; then
    if [ -f "$STORAGE_ROOT/datasets/ETT-small/ETTh1.csv" ]; then
        echo "Symlinking datasets from PROJECT storage..."
        ln -sf "$STORAGE_ROOT/datasets" "$PROJECT_ROOT/datasets"
    else
        echo "ERROR: Missing datasets/ETT-small/ETTh1.csv"
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
echo "Running: $PY -u -m models.diffusion_tsf.train_ci_latent_etth1 --stage all --cache-dir $CACHE_DIR $EXTRA_ARGS"
echo ""

"$PY" -u -m models.diffusion_tsf.train_ci_latent_etth1 \
    --stage all \
    --cache-dir "$CACHE_DIR" \
    $EXTRA_ARGS

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "Checkpoints: $PROJECT_ROOT/models/diffusion_tsf/checkpoints_ci_latent/"
echo "Results: $PROJECT_ROOT/models/diffusion_tsf/results/"
echo "=========================================="
