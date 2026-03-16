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
    HOURS=""
    PASS_ARGS=()
    while [[ $# -gt 0 ]]; do
        case $1 in
            --smoke-test) IS_SMOKE=1; PASS_ARGS+=("$1"); shift ;;
            --hours)      HOURS="$2"; shift 2 ;;
            *)            PASS_ARGS+=("$1"); shift ;;
        esac
    done

    # pick partition based on requested hours (b1≤3h, b2≤12h, b3≤24h, b4≤72h, b5≤168h)
    pick_partition() {
        local h=$1
        if   [ "$h" -le 3   ]; then echo "gpubase_h100_b1"
        elif [ "$h" -le 12  ]; then echo "gpubase_h100_b2"
        elif [ "$h" -le 24  ]; then echo "gpubase_h100_b3"
        elif [ "$h" -le 72  ]; then echo "gpubase_h100_b4"
        else                        echo "gpubase_h100_b5"
        fi
    }

    if [ "$IS_SMOKE" -eq 1 ]; then
        WALL=${HOURS:-0}
        PART=$([ "$WALL" -gt 0 ] && pick_partition "$WALL" || echo "gpubase_h100_b1")
        TIME=$([ "$WALL" -gt 0 ] && printf "%d:00:00" "$WALL" || echo "0:30:00")
        echo "Submitting SMOKE TEST ($PART, $TIME)..."
        sbatch \
            --job-name=unet-fullvar-smoke \
            --account=aip-boyuwang \
            --partition="$PART" \
            --time="$TIME" \
            --nodes=1 \
            --gpus-per-node=h100:1 \
            --cpus-per-task=4 \
            --mem=20G \
            --output=unet-fullvar-smoke-%j.out \
            --error=unet-fullvar-smoke-%j.err \
            --mail-type=END,FAIL \
            --mail-user=ccao87@uwo.ca \
            "$SCRIPT_DIR/slurm_unet_fullvar.sh" "${PASS_ARGS[@]}"
    else
        WALL=${HOURS:-72}
        PART=$(pick_partition "$WALL")
        TIME=$(printf "%d:00:00" "$WALL")
        echo "Submitting FULL RUN ($PART, ${WALL}h)..."
        sbatch \
            --job-name=unet-fullvar \
            --account=aip-boyuwang \
            --partition="$PART" \
            --time="$TIME" \
            --nodes=1 \
            --gpus-per-node=h100:1 \
            --cpus-per-task=6 \
            --mem=80G \
            --output=unet-fullvar-%j.out \
            --error=unet-fullvar-%j.err \
            --mail-type=BEGIN,END,FAIL \
            --mail-user=ccao87@uwo.ca \
            "$SCRIPT_DIR/slurm_unet_fullvar.sh" "${PASS_ARGS[@]}"
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

# Venv — reuse main pipeline venv if it exists.
# Don't trust `source activate` on Alliance — module-loaded python shadows it.
# Instead, export PATH with venv/bin prepended explicitly.
VENV_PATH="$PROJECT/$USER/diffusion-tsf/venv"
if [ ! -d "$VENV_PATH" ]; then
    VENV_PATH="$STORAGE_ROOT/venv"
fi

if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment..."
    python -m venv "$VENV_PATH"
    export PATH="$VENV_PATH/bin:$PATH"
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install numpy pandas scipy scikit-learn optuna wandb tqdm matplotlib einops reformer_pytorch
    [ -f "$PROJECT_ROOT/requirements.txt" ] && pip install -r "$PROJECT_ROOT/requirements.txt"
else
    export PATH="$VENV_PATH/bin:$PATH"
    echo "Reusing venv: $VENV_PATH"
    echo "  python: $(command -v python) ($(python --version 2>&1))"
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
