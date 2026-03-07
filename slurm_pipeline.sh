#!/bin/bash
#SBATCH --job-name=diffusion-tsf
#SBATCH --account=def-boyuwang         # CHANGE to your allocation
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:4         # CHANGE: a100:N or h100:N
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ccao87@uwo.ca     # CHANGE to your email
#SBATCH --signal=B:USR1@120            # Send USR1 120s before wall-time kill

# =============================================================================
# Slurm wrapper for pipeline.sh
# =============================================================================
#
# Runs the full pipeline on an Alliance cluster node.
# The number of GPUs requested above is passed to pipeline.sh --gpus,
# so high-variate subset fine-tuning is parallelized across them.
#
# CLUSTER GPU OPTIONS:
#   Narval:   --gpus-per-node=a100:4    (40GB A100)
#   Fir:      --gpus-per-node=h100:4    (80GB H100)
#   Nibi:     --gpus-per-node=h100:4    (80GB H100)
#   Rorqual:  --gpus-per-node=h100:4    (80GB H100)
#
# USAGE:
#   sbatch slurm_pipeline.sh                    # Full training
#   sbatch slurm_pipeline.sh --smoke-test       # Quick test
#   sbatch slurm_pipeline.sh --resume           # Resume
#   sbatch slurm_pipeline.sh --dataset traffic  # Single dataset
#
# =============================================================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Started: $(date)"
echo "=========================================="

# ---- Environment ----

module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

export PROJECT_ROOT="$HOME/ts-sandbox"

# Auto-detect PROJECT
if [ -z "$PROJECT" ]; then
    if [ -d "$HOME/projects" ]; then
        FIRST_PROJECT=$(ls -d $HOME/projects/def-* 2>/dev/null | head -1)
        [ -n "$FIRST_PROJECT" ] && export PROJECT=$(readlink -f "$FIRST_PROJECT")
    fi
fi

if [ -z "$PROJECT" ]; then
    echo "ERROR: PROJECT not found"
    exit 1
fi

export STORAGE_ROOT="$PROJECT/diffusion-tsf"
echo "STORAGE_ROOT: $STORAGE_ROOT"

mkdir -p "$STORAGE_ROOT/checkpoints"
mkdir -p "$STORAGE_ROOT/results"
mkdir -p "$STORAGE_ROOT/wandb"

# Copy datasets to PROJECT if needed
if [ ! -d "$STORAGE_ROOT/datasets" ]; then
    echo "Copying datasets to PROJECT storage..."
    cp -r "$PROJECT_ROOT/datasets" "$STORAGE_ROOT/datasets"
fi

# Virtual environment
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

export WANDB_DIR="$STORAGE_ROOT/wandb"
export WANDB_CACHE_DIR="$STORAGE_ROOT/wandb/.cache"
export WANDB_MODE=offline

# ---- Detect GPU count ----

# SLURM_GPUS_ON_NODE may not always be set; fall back to parsing gres
if [ -z "$SLURM_GPUS_ON_NODE" ]; then
    SLURM_GPUS_ON_NODE=$(echo "$SLURM_JOB_GPUS" | tr ',' '\n' | wc -l 2>/dev/null || echo 1)
fi
NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}

echo "Detected $NUM_GPUS GPUs"

# ---- Cleanup on failure/cancel/timeout — release GPUs immediately ----

cleanup() {
    echo ""
    echo "[SLURM CLEANUP] $(date) — killing all child processes..."
    kill -- -$$ 2>/dev/null || true
    # pkill as fallback in case process group kill misses something
    pkill -P $$ 2>/dev/null || true
    wait 2>/dev/null || true
    echo "[SLURM CLEANUP] Done. GPU resources released."
}
trap cleanup EXIT ERR SIGTERM SIGINT SIGUSR1

# ---- Build pipeline args ----

PIPELINE_ARGS="--gpus $NUM_GPUS --wandb --checkpoint-dir $STORAGE_ROOT/checkpoints --results-dir $STORAGE_ROOT/results"

# Pass through any extra args from sbatch command line
for arg in "$@"; do
    PIPELINE_ARGS="$PIPELINE_ARGS $arg"
done

# ---- Run ----

cd "$PROJECT_ROOT"

echo ""
echo "Running: ./pipeline.sh $PIPELINE_ARGS"
echo ""

./pipeline.sh $PIPELINE_ARGS
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "JOB FAILED (exit $EXIT_CODE) at $(date)"
    exit $EXIT_CODE
fi

echo ""
echo "=========================================="
echo "Job completed: $(date)"
echo "Results: $STORAGE_ROOT/results"
echo "Checkpoints: $STORAGE_ROOT/checkpoints"
echo "=========================================="
