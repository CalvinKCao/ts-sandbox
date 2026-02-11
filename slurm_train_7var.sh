#!/bin/bash
#SBATCH --job-name=diffusion-tsf
#SBATCH --account=def-boyuwang         # CHANGE THIS to your allocation (e.g., def-smithj)
#SBATCH --time=24:00:00             # Max 7 days (168:00:00) on most clusters
#SBATCH --nodes=1                   # Single node
#SBATCH --gpus-per-node=a100:4      # 4 GPUs for parallel HP trials
#SBATCH --cpus-per-task=48          # CPU cores (12 per GPU)
#SBATCH --mem=192G                  # Memory (48G per GPU)
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

set -e  # Exit on error

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
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=WARN

# -----------------------------------------------------------------------------
# Parse Arguments (passed after script name)
# -----------------------------------------------------------------------------

EXTRA_ARGS=""
for arg in "$@"; do
    EXTRA_ARGS="$EXTRA_ARGS $arg"
done

# Default to resume mode for long jobs
if [[ ! "$EXTRA_ARGS" =~ "--resume" ]] && [[ ! "$EXTRA_ARGS" =~ "--smoke-test" ]]; then
    EXTRA_ARGS="--resume $EXTRA_ARGS"
fi

# -----------------------------------------------------------------------------
# Run Training
# -----------------------------------------------------------------------------

cd "$PROJECT_ROOT"

echo ""
echo "Starting training with args: $EXTRA_ARGS"
echo "Checkpoint dir: $STORAGE_ROOT/checkpoints"
echo "Results dir: $STORAGE_ROOT/results"
echo ""

# Wandb offline mode (compute nodes have no internet)
export WANDB_MODE=offline

NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
echo "Available GPUs: $NUM_GPUS"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Running $NUM_GPUS parallel Optuna workers (one per GPU)..."
    
    # Create a shared Optuna storage for coordination
    export OPTUNA_STORAGE="sqlite:///$STORAGE_ROOT/checkpoints/optuna_study.db"
    
    # Launch parallel workers, each with its own GPU
    # Worker 0 starts first and creates Optuna studies
    echo "Starting worker 0 (main) on GPU 0..."
    CUDA_VISIBLE_DEVICES=0 python -m models.diffusion_tsf.train_7var_pipeline \
        --checkpoint-dir "$STORAGE_ROOT/checkpoints" \
        --results-dir "$STORAGE_ROOT/results" \
        --parallel-worker 0 \
        --wandb \
        $EXTRA_ARGS &
    
    # Give worker 0 time to create studies
    sleep 10
    
    # Start other workers
    for gpu_id in $(seq 1 $((NUM_GPUS - 1))); do
        echo "Starting worker $gpu_id on GPU $gpu_id..."
        CUDA_VISIBLE_DEVICES=$gpu_id python -m models.diffusion_tsf.train_7var_pipeline \
            --checkpoint-dir "$STORAGE_ROOT/checkpoints" \
            --results-dir "$STORAGE_ROOT/results" \
            --parallel-worker $gpu_id \
            $EXTRA_ARGS &
        
        # Stagger worker starts
        sleep 3
    done
    
    # Wait for all workers to finish
    wait
    echo "All workers completed."
else
    echo "Running on single GPU..."
    python -m models.diffusion_tsf.train_7var_pipeline \
        --wandb \
        --checkpoint-dir "$STORAGE_ROOT/checkpoints" \
        --results-dir "$STORAGE_ROOT/results" \
        $EXTRA_ARGS
fi

# To sync wandb logs later from login node:
# wandb sync $STORAGE_ROOT/wandb/offline-*

# -----------------------------------------------------------------------------
# Completion
# -----------------------------------------------------------------------------

echo ""
echo "=========================================="
echo "Job completed: $(date)"
echo "Results in: $STORAGE_ROOT/results"
echo "Checkpoints in: $STORAGE_ROOT/checkpoints"
echo "=========================================="

