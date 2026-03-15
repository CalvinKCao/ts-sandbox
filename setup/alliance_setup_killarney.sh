#!/bin/bash
# =============================================================================
# Alliance Canada Setup Script — Killarney cluster
# =============================================================================
#
# Run this ONCE when first setting up on Killarney.
# Creates persistent storage in $PROJECT space.
#
# Usage:
#   ./alliance_setup_killarney.sh
#
# =============================================================================

set -e

CLUSTER="killarney"
CLUSTER_HOST="killarney.alliancecan.ca"

echo "=========================================="
echo "Alliance Canada Setup — $CLUSTER"
echo "=========================================="

# Check we're on an Alliance cluster - try to auto-detect PROJECT
if [ -z "$PROJECT" ]; then
    echo "WARNING: \$PROJECT not set, attempting auto-detection..."
    
    if [ -d "$HOME/projects" ]; then
        # Killarney uses aip- prefix, other clusters use def-
        FIRST_PROJECT=$(ls -d $HOME/projects/def-* $HOME/projects/aip-* 2>/dev/null | head -1)
        if [ -n "$FIRST_PROJECT" ]; then
            export PROJECT=$(readlink -f "$FIRST_PROJECT")
            echo "  Auto-detected PROJECT: $PROJECT"
        fi
    fi
    
    if [ -z "$PROJECT" ] && [ -n "$SCRATCH" ]; then
        echo "  SCRATCH is set to: $SCRATCH"
        echo ""
        echo "Please set PROJECT manually:"
        echo "  ls ~/projects/"
        echo "  export PROJECT=\$(readlink -f ~/projects/aip-YOURPI)  # or def-YOURPI"
        echo "  ./alliance_setup_killarney.sh"
        exit 1
    fi
    
    if [ -z "$PROJECT" ]; then
        echo ""
        echo "ERROR: Could not auto-detect PROJECT."
        echo ""
        echo "  1. Run: ls -la ~/projects/"
        echo "  2. Find your allocation (e.g., aip-boyuwang or def-boyuwang)"
        echo "  3. Run: export PROJECT=\$(readlink -f ~/projects/aip-boyuwang)"
        echo "  4. Run: ./alliance_setup_killarney.sh"
        echo ""
        echo "No allocation yet? Check: https://ccdb.alliancecan.ca/me/group_resources"
        exit 1
    fi
fi

echo "Using PROJECT: $PROJECT"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STORAGE_ROOT="$PROJECT/$USER/diffusion-tsf"

echo "Project root: $PROJECT_ROOT"
echo "Storage root: $STORAGE_ROOT"
echo ""

# Killarney Performance Compute tier: 10x Dell XE9680
#   GPUs: 8x NVIDIA H100 SXM 80GB per node (80 total)
#   CPU:  2x Intel Xeon Gold 6442Y = 48 cores, 2048 GB RAM
# Standard Compute tier (L40S) is available but we target H100s.
GPU_TYPE="h100"
GPU_COUNT=4          # 4 of 8 H100s per node — increase to 8 for max throughput
CPUS_PER_GPU=6       # 48 cores / 8 GPUs = 6 per GPU
MEM_PER_GPU="250G"   # ~2048 GB / 8 GPUs, leave headroom
echo "  Target GPU: ${GPU_COUNT}x H100 SXM 80GB (Killarney Performance Compute)"

# -----------------------------------------------------------------------------
# 1. Create persistent storage directories
# -----------------------------------------------------------------------------

echo ""
echo "Creating persistent storage in PROJECT space..."
mkdir -p "$STORAGE_ROOT/checkpoints"
mkdir -p "$STORAGE_ROOT/results"
mkdir -p "$STORAGE_ROOT/synthetic_cache"
mkdir -p "$STORAGE_ROOT/wandb"
mkdir -p "$STORAGE_ROOT/datasets"

echo "  ✓ $STORAGE_ROOT/checkpoints"
echo "  ✓ $STORAGE_ROOT/results"
echo "  ✓ $STORAGE_ROOT/synthetic_cache"
echo "  ✓ $STORAGE_ROOT/wandb"
echo "  ✓ $STORAGE_ROOT/datasets"

# -----------------------------------------------------------------------------
# 2. Copy datasets (if not already there)
# -----------------------------------------------------------------------------

echo ""
echo "Checking datasets..."
if [ -d "$PROJECT_ROOT/datasets" ]; then
    DATASETS_SIZE=$(du -sh "$PROJECT_ROOT/datasets" 2>/dev/null | cut -f1)
    echo "  Source datasets: $DATASETS_SIZE"
    
    if [ ! -f "$STORAGE_ROOT/datasets/.copied" ]; then
        echo "  Copying datasets to PROJECT storage (one-time)..."
        rsync -av --progress "$PROJECT_ROOT/datasets/" "$STORAGE_ROOT/datasets/"
        touch "$STORAGE_ROOT/datasets/.copied"
        echo "  ✓ Datasets copied"
    else
        echo "  ✓ Datasets already in PROJECT storage"
    fi
else
    echo "  WARNING: No datasets found at $PROJECT_ROOT/datasets"
    echo "           Upload datasets manually to $STORAGE_ROOT/datasets"
fi

# -----------------------------------------------------------------------------
# 3. Create virtual environment
# -----------------------------------------------------------------------------

echo ""
echo "Setting up Python environment..."

module purge 2>/dev/null || true
module load StdEnv/2023 2>/dev/null || module load StdEnv/2020 2>/dev/null || true
module load python/3.11 2>/dev/null || module load python/3.10 2>/dev/null || true
# Load CUDA matching the GPU - adjust if sinfo showed a different version needed
module load cuda/12.2 2>/dev/null || module load cuda/12.6 2>/dev/null || module load cuda/11.8 2>/dev/null || true

VENV_PATH="$STORAGE_ROOT/venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "  Creating virtual environment at $VENV_PATH..."
    python -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    
    echo "  Installing packages..."
    pip install --upgrade pip
    # cu121 works for CUDA 12.x; swap for cu118 if on older CUDA
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install numpy pandas scipy scikit-learn optuna wandb tqdm matplotlib einops reformer_pytorch
    
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        pip install -r "$PROJECT_ROOT/requirements.txt"
    fi
    
    echo "  ✓ Virtual environment created"
else
    echo "  ✓ Virtual environment already exists at $VENV_PATH"
fi

# -----------------------------------------------------------------------------
# 4. Generate cluster-specific convenience scripts
# -----------------------------------------------------------------------------

echo ""
echo "Generating convenience scripts..."

# Slurm job script for Killarney
cat > "$PROJECT_ROOT/slurm_train_7var_killarney.sh" << SLURM_EOF
#!/bin/bash
# =============================================================================
# Killarney — H100 SXM 80GB job script
# =============================================================================
# Hardware: Dell XE9680, 8x H100 SXM 80GB, 48 cores, 2048 GB RAM per node
#
# Scaling options (edit below):
#   Single GPU:  --gpus-per-node=h100:1  --cpus-per-task=6   --mem=250G
#   4x H100 DDP: --gpus-per-node=h100:4  --cpus-per-task=24  --mem=1000G
#   8x H100 DDP: --gpus-per-node=h100:8  --cpus-per-task=48  --mem=2000G
#
# If H100 partition requires a flag, check with: sinfo -o "%P %G"
# and add: #SBATCH --partition=<name>
#
# USAGE:
#   sbatch slurm_train_7var_killarney.sh              # Full training
#   sbatch slurm_train_7var_killarney.sh --smoke-test # Quick test (1 GPU)
#   sbatch slurm_train_7var_killarney.sh --resume     # Resume interrupted run
# =============================================================================
#SBATCH --job-name=diffusion-tsf
#SBATCH --account=aip-boyuwang
#SBATCH --partition=gpubase_h100_b4
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:${GPU_COUNT}
#SBATCH --cpus-per-task=$((CPUS_PER_GPU * GPU_COUNT))
#SBATCH --mem=${MEM_PER_GPU}
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ccao87@uwo.ca
#SBATCH --signal=B:USR1@120

# Kill all child processes on exit/cancel/timeout so GPUs are released
cleanup() {
    trap '' EXIT ERR SIGTERM SIGINT SIGUSR1  # prevent re-entry
    echo "[CLEANUP] \$(date) — releasing GPU resources..."
    kill -- -\$\$ 2>/dev/null || true
    pkill -P \$\$ 2>/dev/null || true
    wait 2>/dev/null || true
    echo "[CLEANUP] Done."
}
trap cleanup EXIT ERR SIGTERM SIGINT SIGUSR1

echo "=========================================="
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURMD_NODENAME"
echo "GPUs: \$SLURM_GPUS_ON_NODE"
echo "CPUs: \$SLURM_CPUS_PER_TASK"
echo "Started: \$(date)"
echo "=========================================="

module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

# Killarney forbids running from /home — use scratch copy
if [ -d "\$SCRATCH/ts-sandbox" ]; then
    export PROJECT_ROOT="\$SCRATCH/ts-sandbox"
elif [ -d "\$HOME/ts-sandbox" ]; then
    export PROJECT_ROOT="\$HOME/ts-sandbox"
else
    echo "ERROR: ts-sandbox not found"
    exit 1
fi

if [ -z "\$PROJECT" ]; then
    echo "PROJECT not set, auto-detecting..."
    if [ -d "\$HOME/projects" ]; then
        FIRST_PROJECT=\$(ls -d \$HOME/projects/def-* \$HOME/projects/aip-* 2>/dev/null | head -1)
        if [ -n "\$FIRST_PROJECT" ]; then
            export PROJECT=\$(readlink -f "\$FIRST_PROJECT")
            echo "Found: \$PROJECT"
        fi
    fi
fi

if [ -z "\$PROJECT" ]; then
    echo "ERROR: PROJECT not found. Set manually:"
    echo "  export PROJECT=/project/aip-boyuwang"
    exit 1
fi

export STORAGE_ROOT="\$PROJECT/\$USER/diffusion-tsf"
echo "STORAGE_ROOT: \$STORAGE_ROOT"

mkdir -p "\$STORAGE_ROOT/checkpoints"
mkdir -p "\$STORAGE_ROOT/results"
mkdir -p "\$STORAGE_ROOT/synthetic_cache"
mkdir -p "\$STORAGE_ROOT/wandb"

if [ ! -d "\$STORAGE_ROOT/datasets" ]; then
    echo "Copying datasets to PROJECT storage..."
    cp -r "\$PROJECT_ROOT/datasets" "\$STORAGE_ROOT/datasets"
fi

VENV_PATH="\$STORAGE_ROOT/venv"
if [ ! -d "\$VENV_PATH" ]; then
    echo "Creating virtual environment..."
    python -m venv "\$VENV_PATH"
    source "\$VENV_PATH/bin/activate"
    pip install --upgrade pip
    # cu121 is compatible with CUDA 12.x (H100 SXM requires CUDA 11.8+)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install numpy pandas scipy scikit-learn optuna wandb tqdm matplotlib einops reformer_pytorch
    if [ -f "\$PROJECT_ROOT/requirements.txt" ]; then
        pip install -r "\$PROJECT_ROOT/requirements.txt"
    fi
else
    source "\$VENV_PATH/bin/activate"
fi

# export WANDB_API_KEY="your-key-here"
export WANDB_DIR="\$STORAGE_ROOT/wandb"
export WANDB_CACHE_DIR="\$STORAGE_ROOT/wandb/.cache"
export WANDB_MODE=offline

EXTRA_ARGS=""
for arg in "\$@"; do
    EXTRA_ARGS="\$EXTRA_ARGS \$arg"
done

if [[ ! "\$EXTRA_ARGS" =~ "--resume" ]] && [[ ! "\$EXTRA_ARGS" =~ "--smoke-test" ]] && [[ ! "\$EXTRA_ARGS" =~ "--fresh" ]]; then
    EXTRA_ARGS="--fresh \$EXTRA_ARGS"
fi

cd "\$PROJECT_ROOT"

if [[ ! "\$EXTRA_ARGS" =~ "--resume" ]] && [ -d "\$STORAGE_ROOT/synthetic_cache" ]; then
    echo "Clearing synthetic cache..."
    rm -rf "\$STORAGE_ROOT/synthetic_cache"/*
fi

echo ""
echo "Starting training: \$EXTRA_ARGS"
echo ""

# Multi-GPU via DDP when more than 1 GPU is allocated
N_GPUS=\$(echo "\$SLURM_GPUS_ON_NODE" | tr ',' '\n' | wc -l)
if [ "\$N_GPUS" -gt 1 ]; then
    echo "Running DDP on \$N_GPUS GPUs..."
    torchrun --nproc_per_node=\$N_GPUS -m models.diffusion_tsf.train_7var_pipeline \\
        --ddp \\
        --wandb \\
        --checkpoint-dir "\$STORAGE_ROOT/checkpoints" \\
        --results-dir "\$STORAGE_ROOT/results" \\
        \$EXTRA_ARGS
else
    echo "Running single GPU..."
    export CUDA_VISIBLE_DEVICES=0
    python -m models.diffusion_tsf.train_7var_pipeline \\
        --wandb \\
        --checkpoint-dir "\$STORAGE_ROOT/checkpoints" \\
        --results-dir "\$STORAGE_ROOT/results" \\
        \$EXTRA_ARGS
fi

EXIT_CODE=\$?

if [ \$EXIT_CODE -ne 0 ]; then
    echo "JOB FAILED (exit \$EXIT_CODE) at \$(date)"
    echo "Check: diffusion-tsf-\${SLURM_JOB_ID}.err"
    exit \$EXIT_CODE
fi

echo "Job completed: \$(date)"
echo "Results: \$STORAGE_ROOT/results"
SLURM_EOF
chmod +x "$PROJECT_ROOT/slurm_train_7var_killarney.sh"
echo "  ✓ Created: slurm_train_7var_killarney.sh  (${GPU_COUNT}x H100 SXM 80GB, DDP-enabled)"

# Submit script
cat > "$PROJECT_ROOT/submit_train_killarney.sh" << 'SUBMIT_EOF'
#!/bin/bash
sbatch slurm_train_7var_killarney.sh "$@"
SUBMIT_EOF
chmod +x "$PROJECT_ROOT/submit_train_killarney.sh"
echo "  ✓ Created: submit_train_killarney.sh"

# Status checker
cat > "$PROJECT_ROOT/check_status.sh" << 'STATUS_EOF'
#!/bin/bash
echo "=== Your Jobs ==="
sq

echo ""
echo "=== Training Progress ==="
MANIFEST="$PROJECT/$USER/diffusion-tsf/checkpoints/training_manifest.json"
if [ -f "$MANIFEST" ]; then
    python3 -c "
import json
with open('$MANIFEST') as f:
    m = json.load(f)
print(f'iTransformer HP done: {m.get(\"itrans_hp_done\", False)}')
print(f'Diffusion HP done: {m.get(\"diffusion_hp_done\", False)}')
print(f'Pretrain complete: {m.get(\"pretrain_complete\", False)}')
complete = sum(1 for s in m.get('subsets', {}).values() if s.get('status') == 'complete')
pending = sum(1 for s in m.get('subsets', {}).values() if s.get('status') == 'pending')
in_prog = sum(1 for s in m.get('subsets', {}).values() if s.get('status') == 'in_progress')
print(f'Subsets: {complete} complete, {in_prog} in progress, {pending} pending')
"
else
    echo "No training manifest yet."
fi

echo ""
echo "=== Storage Usage ==="
diskusage_report 2>/dev/null || df -h $HOME $PROJECT $SCRATCH 2>/dev/null || echo "(usage info unavailable)"
STATUS_EOF
chmod +x "$PROJECT_ROOT/check_status.sh"
echo "  ✓ Created: check_status.sh"

# Sync script for LOCAL machine
cat > "$PROJECT_ROOT/sync_from_killarney.sh" << SYNC_EOF
#!/bin/bash
# Run on your LOCAL machine to pull results from Killarney.
# Usage: ./sync_from_killarney.sh [user@killarney.alliancecan.ca]

REMOTE="\${1:-$USER@$CLUSTER_HOST}"
LOCAL_DIR="./synced_results"
mkdir -p "\$LOCAL_DIR"

echo "Syncing from \$REMOTE..."
rsync -avz --progress "\$REMOTE:~/projects/*/\$USER/diffusion-tsf/results/" "\$LOCAL_DIR/results/"
rsync -avz --progress "\$REMOTE:~/projects/*/\$USER/diffusion-tsf/checkpoints/training_manifest.json" "\$LOCAL_DIR/"
rsync -avz --progress "\$REMOTE:~/projects/*/\$USER/diffusion-tsf/wandb/" "\$LOCAL_DIR/wandb/"

echo "Done — results in \$LOCAL_DIR"
SYNC_EOF
chmod +x "$PROJECT_ROOT/sync_from_killarney.sh"
echo "  ✓ Created: sync_from_killarney.sh"

# -----------------------------------------------------------------------------
# 5. Smoke-test the environment
# -----------------------------------------------------------------------------

echo ""
echo "Verifying Python + PyTorch install..."
source "$STORAGE_ROOT/venv/bin/activate"
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
else:
    print('  (CUDA not available on login node — normal, will be available in jobs)')
"

# -----------------------------------------------------------------------------
# 6. Done
# -----------------------------------------------------------------------------

echo ""
echo "=========================================="
echo "Setup Complete!  Cluster: $CLUSTER"
echo "=========================================="
echo ""
echo "NEXT STEPS:"
echo ""
echo "1. Check if H100 nodes need a partition flag:"
echo "   sinfo -o \"%P %G %l\" | grep h100"
echo "   # If yes, add to slurm_train_7var_killarney.sh: #SBATCH --partition=<name>"
echo ""
echo "2. Set up wandb (optional but recommended):"
echo "   wandb login"
echo "   # Or add to ~/.bashrc: export WANDB_API_KEY='your-key'"
echo ""
echo "3. Test the pipeline:"
echo "   ./submit_train_killarney.sh --smoke-test"
echo ""
echo "4. Full training run:"
echo "   ./submit_train_killarney.sh"
echo ""
echo "5. Monitor:"
echo "   ./check_status.sh"
echo "   sq"
echo "   tail -f diffusion-tsf-*.out"
echo ""
echo "6. Sync results to your local machine:"
echo "   ./sync_from_killarney.sh $USER@$CLUSTER_HOST"
echo ""
echo "Storage:"
echo "  Checkpoints: $STORAGE_ROOT/checkpoints"
echo "  Results:     $STORAGE_ROOT/results"
echo "  Datasets:    $STORAGE_ROOT/datasets"
echo "  Wandb:       $STORAGE_ROOT/wandb"
echo ""
