# Sourced by slurm_ci_latent_*.sh after the caller sets EXTRA_ARGS (string, e.g. " --smoke-test").
# Caller should: set -e; export PYTHONUNBUFFERED=1; parse "$@" into EXTRA_ARGS; then source this file.

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
        FIRST_PROJECT=$(ls -d "$HOME/projects/def-"* "$HOME/projects/aip-"* 2>/dev/null | head -1)
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

CACHE_DIR="$STORAGE_ROOT/synthetic_cache/ci_latent"
mkdir -p "$CACHE_DIR"

cd "$PROJECT_ROOT"

if [ ! -d "$PROJECT_ROOT/datasets/ETT-small" ] || [ ! -d "$PROJECT_ROOT/datasets/exchange_rate" ]; then
    if [ -d "$STORAGE_ROOT/datasets" ]; then
        echo "Symlinking datasets from PROJECT storage..."
        ln -sf "$STORAGE_ROOT/datasets" "$PROJECT_ROOT/datasets"
    fi
fi

if [ ! -f "$PROJECT_ROOT/datasets/ETT-small/ETTh1.csv" ]; then
    echo "ERROR: Missing datasets (expected datasets/ETT-small/*.csv)"
    exit 1
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

DIFFUSION_TS="$PROJECT_ROOT/models/diffusion_tsf"
SHARED="$DIFFUSION_TS/checkpoints_ci_etth2"
RUNROOT="$DIFFUSION_TS/checkpoints_ci_runs"
IMAGE_H=128
STAGE4_TRIALS=12
EXCHANGE_SEED=42

mkdir -p "$SHARED" "$RUNROOT"

run_py() {
    "$PY" -u -m models.diffusion_tsf.train_ci_latent_etth2 \
        --cache-dir "$CACHE_DIR" \
        --shared-ckpt-dir "$SHARED" \
        --run-ckpt-dir "$RUNROOT" \
        --image-height "$IMAGE_H" \
        "$@"
}
