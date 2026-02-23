#!/bin/bash
#SBATCH --job-name=diffusion-tsf
#SBATCH --account=def-boyuwang         # CHANGE THIS to your allocation (e.g., def-smithj)
#SBATCH --time=96:00:00             # Max 7 days (168:00:00) on most clusters
#SBATCH --nodes=1                   # Single node
#SBATCH --gpus-per-node=a100:1      # 1 GPU (reliable, HP tuning is fast enough)
#SBATCH --cpus-per-task=12          # CPU cores
#SBATCH --mem=48G                   # Memory
#SBATCH --output=%x-%j.out          # Output: job-name-jobid.out
#SBATCH --error=%x-%j.err           # Error: job-name-jobid.err
#SBATCH --mail-type=BEGIN,END,FAIL  # Email notifications
#SBATCH --mail-user=ccao87@uwo.ca  # CHANGE THIS
 
# =============================================================================
# Digital Research Alliance Slurm Job Script for Diffusion TSF Training
# =============================================================================
# 
# BEFORE FIRST RUN:
#   1. Change --account to your allocation (run: sacctmgr show assoc user=$USER)
#   2. Change --mail-user to your email
#   3. Adjust --gpus-per-node based on your needs and cluster
#   4. Set WANDB_API_KEY in your ~/.bashrc or below
#
# CLUSTER GPU OPTIONS:
#   Narval:   --gpus-per-node=a100:4    (40GB A100)
#   Fir:      --gpus-per-node=h100:4    (80GB H100)
#   Nibi:     --gpus-per-node=h100:4    (80GB H100)
#   Rorqual:  --gpus-per-node=h100:4    (80GB H100)
#
# STORAGE PATHS (using PROJECT for persistence):
#   Checkpoints: $PROJECT/diffusion-tsf/checkpoints
#   Results:     $PROJECT/diffusion-tsf/results  
#   Synthetic:   $PROJECT/diffusion-tsf/synthetic_cache
#
# USAGE:
#   sbatch slurm_train_7var.sh                    # Full training
#   sbatch slurm_train_7var.sh --smoke-test      # Quick test
#   sbatch slurm_train_7var.sh --resume          # Resume training
#
# =============================================================================

# Don't use set -e: we capture exit codes manually to release GPU on failure

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Started: $(date)"
echo "=========================================="

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------

# Load required modules (adjust versions as needed)
module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

# Project paths - use PROJECT for persistent storage
export PROJECT_ROOT="$HOME/ts-sandbox"

# Auto-detect PROJECT if not set (common issue on Narval batch jobs)
if [ -z "$PROJECT" ]; then
    echo "PROJECT not set, auto-detecting from ~/projects..."
    if [ -d "$HOME/projects" ]; then
        FIRST_PROJECT=$(ls -d $HOME/projects/def-* 2>/dev/null | head -1)
        if [ -n "$FIRST_PROJECT" ]; then
            export PROJECT=$(readlink -f "$FIRST_PROJECT")
            echo "Found: $PROJECT"
        fi
    fi
fi

if [ -z "$PROJECT" ]; then
    echo "ERROR: PROJECT not found! Set manually:"
    echo "  export PROJECT=/project/def-boyuwang"
    exit 1
fi

export STORAGE_ROOT="$PROJECT/diffusion-tsf"
echo "STORAGE_ROOT: $STORAGE_ROOT"

# Create persistent storage directories
mkdir -p "$STORAGE_ROOT/checkpoints"
mkdir -p "$STORAGE_ROOT/results"
mkdir -p "$STORAGE_ROOT/synthetic_cache"
mkdir -p "$STORAGE_ROOT/wandb"

# Symlink datasets if needed (copy once, reuse)
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
    # Install from requirements if exists
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        pip install -r "$PROJECT_ROOT/requirements.txt"
    fi
else
    source "$VENV_PATH/bin/activate"
fi

# Wandb setup (set your API key in ~/.bashrc or uncomment below)
# export WANDB_API_KEY="your-key-here"
export WANDB_DIR="$STORAGE_ROOT/wandb"
export WANDB_CACHE_DIR="$STORAGE_ROOT/wandb/.cache"

# CUDA settings
export CUDA_VISIBLE_DEVICES=0

# -----------------------------------------------------------------------------
# Parse Arguments (passed after script name)
# -----------------------------------------------------------------------------

EXTRA_ARGS=""
for arg in "$@"; do
    EXTRA_ARGS="$EXTRA_ARGS $arg"
done

# Default to --fresh (start from scratch) unless --resume or --smoke-test passed
if [[ ! "$EXTRA_ARGS" =~ "--resume" ]] && [[ ! "$EXTRA_ARGS" =~ "--smoke-test" ]] && [[ ! "$EXTRA_ARGS" =~ "--fresh" ]]; then
    EXTRA_ARGS="--fresh $EXTRA_ARGS"
fi

# -----------------------------------------------------------------------------
# Run Training
# -----------------------------------------------------------------------------

cd "$PROJECT_ROOT"

# Purge cached synthetic data unless resuming (resume reuses existing cache)
if [[ ! "$EXTRA_ARGS" =~ "--resume" ]] && [ -d "$STORAGE_ROOT/synthetic_cache" ]; then
    echo "Clearing old synthetic cache to force regeneration..."
    rm -rf "$STORAGE_ROOT/synthetic_cache"/*
    echo "Synthetic cache cleared."
elif [[ "$EXTRA_ARGS" =~ "--resume" ]]; then
    echo "Resume mode: keeping existing synthetic cache."
fi

echo ""
echo "Starting training with args: $EXTRA_ARGS"
echo "Checkpoint dir: $STORAGE_ROOT/checkpoints"
echo "Results dir: $STORAGE_ROOT/results"
echo ""

# Wandb offline mode (compute nodes have no internet)
export WANDB_MODE=offline

echo "Running on single GPU..."
python -m models.diffusion_tsf.train_7var_pipeline \
    --wandb \
    --checkpoint-dir "$STORAGE_ROOT/checkpoints" \
    --results-dir "$STORAGE_ROOT/results" \
    $EXTRA_ARGS

EXIT_CODE=$?

# -----------------------------------------------------------------------------
# Auto-detect crash: release GPU immediately if script failed
# -----------------------------------------------------------------------------

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "JOB FAILED with exit code $EXIT_CODE at $(date)"
    echo "Releasing GPU resources immediately."
    echo "Check error log: diffusion-tsf-${SLURM_JOB_ID}.err"
    echo "=========================================="
    exit $EXIT_CODE
fi

echo ""
echo "=========================================="
echo "Job completed successfully: $(date)"
echo "Results in: $STORAGE_ROOT/results"
echo "Checkpoints in: $STORAGE_ROOT/checkpoints"
echo "=========================================="

