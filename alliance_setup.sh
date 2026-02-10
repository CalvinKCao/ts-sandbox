#!/bin/bash
# =============================================================================
# Digital Research Alliance Initial Setup Script
# =============================================================================
#
# Run this ONCE when first setting up on a new cluster.
# This sets up persistent storage in PROJECT space.
#
# Usage:
#   ./alliance_setup.sh
#
# =============================================================================

set -e

echo "=========================================="
echo "Digital Research Alliance Setup"
echo "=========================================="

# Check we're on an Alliance cluster
if [ -z "$PROJECT" ]; then
    echo "ERROR: \$PROJECT not set. Are you on an Alliance cluster?"
    echo "       This script is for Narval, Fir, Nibi, Rorqual, etc."
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
STORAGE_ROOT="$PROJECT/diffusion-tsf"

echo "Project root: $PROJECT_ROOT"
echo "Storage root: $STORAGE_ROOT"
echo ""

# -----------------------------------------------------------------------------
# 1. Create persistent storage directories
# -----------------------------------------------------------------------------

echo "Creating persistent storage in PROJECT space..."
mkdir -p "$STORAGE_ROOT/checkpoints"
mkdir -p "$STORAGE_ROOT/results"
mkdir -p "$STORAGE_ROOT/synthetic_cache"
mkdir -p "$STORAGE_ROOT/wandb"
mkdir -p "$STORAGE_ROOT/datasets"

echo "  ✓ Created: $STORAGE_ROOT/checkpoints"
echo "  ✓ Created: $STORAGE_ROOT/results"
echo "  ✓ Created: $STORAGE_ROOT/synthetic_cache"
echo "  ✓ Created: $STORAGE_ROOT/wandb"
echo "  ✓ Created: $STORAGE_ROOT/datasets"

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
    echo "  WARNING: No datasets directory found at $PROJECT_ROOT/datasets"
    echo "           You'll need to upload your datasets manually."
fi

# -----------------------------------------------------------------------------
# 3. Create virtual environment
# -----------------------------------------------------------------------------

echo ""
echo "Setting up Python environment..."

# Load modules
module purge 2>/dev/null || true
module load StdEnv/2023 2>/dev/null || module load StdEnv/2020 2>/dev/null || true
module load python/3.11 2>/dev/null || module load python/3.10 2>/dev/null || true
module load cuda/12.2 2>/dev/null || module load cuda/11.8 2>/dev/null || true

VENV_PATH="$STORAGE_ROOT/venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "  Creating virtual environment..."
    python -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    
    echo "  Installing packages..."
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install numpy pandas scipy scikit-learn optuna wandb tqdm matplotlib einops
    
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        pip install -r "$PROJECT_ROOT/requirements.txt"
    fi
    
    echo "  ✓ Virtual environment created"
else
    echo "  ✓ Virtual environment already exists"
fi

# -----------------------------------------------------------------------------
# 4. Create convenience scripts
# -----------------------------------------------------------------------------

echo ""
echo "Creating convenience scripts..."

# Quick submit script
cat > "$PROJECT_ROOT/submit_train.sh" << 'SUBMIT_EOF'
#!/bin/bash
# Quick submit script - edit slurm_train_7var.sh first!
sbatch slurm_train_7var.sh "$@"
SUBMIT_EOF
chmod +x "$PROJECT_ROOT/submit_train.sh"
echo "  ✓ Created: submit_train.sh"

# Check status script
cat > "$PROJECT_ROOT/check_status.sh" << 'STATUS_EOF'
#!/bin/bash
# Check job status and training progress

echo "=== Your Jobs ==="
sq

echo ""
echo "=== Training Progress ==="
MANIFEST="$PROJECT/diffusion-tsf/checkpoints/training_manifest.json"
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
    echo "No training manifest found yet."
fi

echo ""
echo "=== Storage Usage ==="
diskusage_report 2>/dev/null || echo "(diskusage_report not available)"
STATUS_EOF
chmod +x "$PROJECT_ROOT/check_status.sh"
echo "  ✓ Created: check_status.sh"

# Sync results script
cat > "$PROJECT_ROOT/sync_from_cluster.sh" << 'SYNC_EOF'
#!/bin/bash
# Run this on your LOCAL machine to sync results
# Usage: ./sync_from_cluster.sh user@narval.alliancecan.ca

if [ -z "$1" ]; then
    echo "Usage: $0 user@cluster.alliancecan.ca"
    echo "Example: $0 smithj@narval.alliancecan.ca"
    exit 1
fi

REMOTE="$1"
LOCAL_DIR="./synced_results"
mkdir -p "$LOCAL_DIR"

echo "Syncing from $REMOTE..."
rsync -avz --progress "$REMOTE:~/projects/*/diffusion-tsf/results/" "$LOCAL_DIR/results/"
rsync -avz --progress "$REMOTE:~/projects/*/diffusion-tsf/checkpoints/training_manifest.json" "$LOCAL_DIR/"
rsync -avz --progress "$REMOTE:~/projects/*/diffusion-tsf/wandb/" "$LOCAL_DIR/wandb/"

echo "Done! Results in $LOCAL_DIR"
SYNC_EOF
chmod +x "$PROJECT_ROOT/sync_from_cluster.sh"
echo "  ✓ Created: sync_from_cluster.sh"

# -----------------------------------------------------------------------------
# 5. Final instructions
# -----------------------------------------------------------------------------

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "NEXT STEPS:"
echo ""
echo "1. Edit slurm_train_7var.sh:"
echo "   - Change --account=def-YOURPI to your allocation"
echo "   - Change --mail-user to your email"
echo "   - Adjust GPU count if needed (--gpus-per-node)"
echo ""
echo "2. Set up wandb (optional but recommended):"
echo "   wandb login"
echo "   # Or add to ~/.bashrc: export WANDB_API_KEY='your-key'"
echo ""
echo "3. Submit your first job:"
echo "   ./submit_train.sh --smoke-test   # Quick test first"
echo "   ./submit_train.sh                # Full training"
echo ""
echo "4. Monitor progress:"
echo "   ./check_status.sh"
echo "   sq                               # Check job queue"
echo "   tail -f diffusion-tsf-*.out      # Watch output"
echo ""
echo "5. Sync results to local machine:"
echo "   # On your local machine:"
echo "   ./sync_from_cluster.sh $USER@$(hostname).alliancecan.ca"
echo ""
echo "Storage locations:"
echo "  Checkpoints: $STORAGE_ROOT/checkpoints"
echo "  Results:     $STORAGE_ROOT/results"
echo "  Datasets:    $STORAGE_ROOT/datasets"
echo "  Wandb:       $STORAGE_ROOT/wandb"
echo ""

