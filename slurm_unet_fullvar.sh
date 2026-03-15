#!/bin/bash
# =============================================================================
# U-Net full-variate — self-resubmitting Slurm script for Killarney
#
# When run from the login node, it picks partition + wall time and sbatch's itself.
# When run inside a Slurm job (SLURM_JOB_ID is set), it does the actual work.
#
# USAGE (from login node):
#   ./slurm_unet_fullvar.sh --smoke-test                     # H100, 20GB, 30 min
#   ./slurm_unet_fullvar.sh                                  # H100, 60GB, 3 days
#   ./slurm_unet_fullvar.sh --dataset electricity             # H100, 60GB, 3 days
#   ./slurm_unet_fullvar.sh --resume --dataset traffic        # resume traffic
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ===========================================================================
# If NOT inside a Slurm job → submit ourselves with the right resources
# ===========================================================================

if [ -z "$SLURM_JOB_ID" ]; then
    IS_SMOKE=0
    for arg in "$@"; do
        [ "$arg" = "--smoke-test" ] && IS_SMOKE=1
    done

    if [ "$IS_SMOKE" -eq 1 ]; then
        echo "Submitting SMOKE TEST (L40S, 8GB, 15 min)..."
        sbatch \
            --job-name=unet-fullvar-smoke \
            --account=aip-boyuwang \
            --time=0:15:00 \
            --nodes=1 \
            --gres=gpu:l40s:1 \
            --cpus-per-task=2 \
            --mem=8G \
            --output=unet-fullvar-smoke-%j.out \
            --error=unet-fullvar-smoke-%j.err \
            --mail-type=END,FAIL \
            --mail-user=ccao87@uwo.ca \
            "$SCRIPT_DIR/slurm_unet_fullvar.sh" "$@"
    else
        echo "Submitting FULL RUN (H100, 60GB, b4 = 3 days)..."
        sbatch \
            --job-name=unet-fullvar \
            --account=aip-boyuwang \
            --partition=gpubase_h100_b4 \
            --time=3-00:00:00 \
            --nodes=1 \
            --gpus-per-node=h100:1 \
            --cpus-per-task=6 \
            --mem=60G \
            --output=unet-fullvar-%j.out \
            --error=unet-fullvar-%j.err \
            --mail-type=BEGIN,END,FAIL \
            --mail-user=ccao87@uwo.ca \
            "$SCRIPT_DIR/slurm_unet_fullvar.sh" "$@"
    fi
    exit 0
fi

# ===========================================================================
# We're inside a Slurm job — do the actual work
# ===========================================================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1 || echo 'unknown')"
echo "Started: $(date)"
echo "=========================================="

# ---- Environment ----

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

# Auto-detect PROJECT
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

# Separate storage root so it doesn't conflict with main pipeline or CI-DiT
export STORAGE_ROOT="$PROJECT/$USER/diffusion-tsf-fullvar"
echo "STORAGE_ROOT: $STORAGE_ROOT"

mkdir -p "$STORAGE_ROOT/checkpoints"
mkdir -p "$STORAGE_ROOT/results"

# Copy datasets to PROJECT if needed
if [ ! -d "$STORAGE_ROOT/datasets" ]; then
    echo "Copying datasets to PROJECT storage..."
    cp -r "$PROJECT_ROOT/datasets" "$STORAGE_ROOT/datasets"
fi

# Venv — reuse main pipeline venv if it exists
VENV_PATH="$PROJECT/$USER/diffusion-tsf/venv"
if [ ! -d "$VENV_PATH" ]; then
    VENV_PATH="$STORAGE_ROOT/venv"
    if [ ! -d "$VENV_PATH" ]; then
        echo "Creating virtual environment..."
        python -m venv "$VENV_PATH"
        source "$VENV_PATH/bin/activate"
        pip install --upgrade pip
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        pip install numpy pandas scipy scikit-learn optuna wandb tqdm matplotlib einops reformer_pytorch
        [ -f "$PROJECT_ROOT/requirements.txt" ] && pip install -r "$PROJECT_ROOT/requirements.txt"
    else
        source "$VENV_PATH/bin/activate"
    fi
else
    source "$VENV_PATH/bin/activate"
    echo "Reusing existing venv: $VENV_PATH"
fi

export WANDB_MODE=offline

# ---- Cleanup ----

cleanup() {
    trap '' EXIT ERR SIGTERM SIGINT SIGUSR1
    local code=${1:-$?}
    [ "$code" -ne 0 ] && echo "[CLEANUP] $(date) — killing child processes..."
    kill -- -$$ 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT ERR SIGTERM SIGINT SIGUSR1

# ---- Build args ----

PIPELINE_ARGS="--checkpoint-dir $STORAGE_ROOT/checkpoints --results-dir $STORAGE_ROOT/results"

for arg in "$@"; do
    PIPELINE_ARGS="$PIPELINE_ARGS $arg"
done

cd "$PROJECT_ROOT"

echo ""
echo "Running: ./run_unet_fullvar.sh $PIPELINE_ARGS"
echo ""

./run_unet_fullvar.sh $PIPELINE_ARGS

echo ""
echo "=========================================="
echo "Job completed: $(date)"
echo "Results: $STORAGE_ROOT/results"
echo "Checkpoints: $STORAGE_ROOT/checkpoints"
echo "=========================================="
