#!/bin/bash
#SBATCH --job-name=ci-dit-test
#SBATCH --account=aip-boyuwang
#SBATCH --partition=gpubase_h100_b1
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ccao87@uwo.ca

# =============================================================================
# CI-DiT smoke test on small datasets (ETTh1, exchange_rate)
#
# Quick test of the CI-DiT pipeline on 7-variate and 8-variate datasets.
# Uses b1 partition (3h) since these are small — plenty of time.
#
# USAGE:
#   sbatch slurm_ci_dit_test.sh                   # ETTh1 + exchange_rate
#   sbatch slurm_ci_dit_test.sh --smoke-test      # ~2 min sanity check
#   sbatch slurm_ci_dit_test.sh --dataset ETTh1   # single dataset
# =============================================================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $SLURM_GPUS_ON_NODE"
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

export STORAGE_ROOT="$PROJECT/diffusion-tsf-cidit"
echo "STORAGE_ROOT: $STORAGE_ROOT"

mkdir -p "$STORAGE_ROOT/checkpoints"
mkdir -p "$STORAGE_ROOT/results"

# Datasets
if [ ! -d "$STORAGE_ROOT/datasets" ]; then
    echo "Copying datasets to PROJECT storage..."
    cp -r "$PROJECT_ROOT/datasets" "$STORAGE_ROOT/datasets"
fi

# Virtual environment (reuse main one if it exists)
VENV_PATH="$PROJECT/diffusion-tsf/venv"
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

# Pass through sbatch cli args (--smoke-test, --dataset, etc.)
for arg in "$@"; do
    PIPELINE_ARGS="$PIPELINE_ARGS $arg"
done

# If no --dataset specified, default to the two smallest: ETTh1 (7) + exchange_rate (8)
HAS_DATASET=0
for arg in "$@"; do
    [ "$arg" = "--dataset" ] && HAS_DATASET=1
done

cd "$PROJECT_ROOT"

if [ "$HAS_DATASET" -eq 0 ]; then
    echo ""
    echo "No --dataset specified, running ETTh1 (7-var) and exchange_rate (8-var)"
    echo ""

    echo "=== ETTh1 (7 variates) ==="
    ./run_ci_dit.sh --dataset ETTh1 $PIPELINE_ARGS

    echo ""
    echo "=== exchange_rate (8 variates) ==="
    ./run_ci_dit.sh --dataset exchange_rate $PIPELINE_ARGS
else
    echo ""
    echo "Running: ./run_ci_dit.sh $PIPELINE_ARGS"
    echo ""
    ./run_ci_dit.sh $PIPELINE_ARGS
fi

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed: $(date) (exit $EXIT_CODE)"
echo "Results: $STORAGE_ROOT/results"
echo "Checkpoints: $STORAGE_ROOT/checkpoints"
echo "=========================================="
