#!/bin/bash
#SBATCH --job-name=bench-px-lat
#SBATCH --account=aip-boyuwang
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ccao87@uwo.ca
#SBATCH --signal=B:USR1@120

# =============================================================================
# Alliance Canada — one-epoch timing: pixel DiffusionTSF vs latent LatentDiffusionTSF
# (same synthetic RealTS loader, matched windows / image height / U-Net widths).
#
# Edit #SBATCH --account (and mail-user) to your CCDB Group Name if different.
# Run from $SCRATCH copy of the repo on Killarney (not /home).
#
# Submit:
#   sbatch slurm_benchmark_pixel_vs_latent.sh
#   sbatch slurm_benchmark_pixel_vs_latent.sh -- --num-samples 256 --batch-size 8
#   sbatch slurm_benchmark_pixel_vs_latent.sh -- --amp --num-samples 512
#
# Logs: bench-px-lat-<jobid>.out / .err in the directory where you ran sbatch.
# =============================================================================

set -e
export PYTHONUNBUFFERED=1

cleanup() {
    trap '' EXIT ERR SIGTERM SIGINT SIGUSR1
    local code=${1:-$?}
    [ "$code" -ne 0 ] && echo "[SLURM CLEANUP] $(date)"
    kill -- -$$ 2>/dev/null || true
    pkill -P $$ 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT ERR SIGTERM SIGINT SIGUSR1

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
    echo "ERROR: PROJECT not found. Set export PROJECT=/path/to/your/allocation"
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
echo "CUDA: $($PY -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")')"

cd "$PROJECT_ROOT"

EXTRA_ARGS=""
for a in "$@"; do
    [ "$a" = "--" ] && continue
    EXTRA_ARGS="$EXTRA_ARGS $a"
done

# Default: one modest epoch + bf16 (typical training); override after --
DEFAULT_ARGS="--num-samples 512 --batch-size 8 --warmup-batches 3 --amp"
if [ -z "$EXTRA_ARGS" ]; then
    EXTRA_ARGS="$DEFAULT_ARGS"
else
    # User passed args only — still suggest --amp on GPU if they omitted it
    echo "Using custom args:$EXTRA_ARGS"
fi

echo ""
echo "Running: $PY -u -m models.diffusion_tsf.benchmark_pixel_vs_latent_epoch $EXTRA_ARGS"
echo ""

"$PY" -u -m models.diffusion_tsf.benchmark_pixel_vs_latent_epoch $EXTRA_ARGS

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
