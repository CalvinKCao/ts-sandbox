# Sourced by slurm_ci_latent_*.sh after the caller sets EXTRA_ARGS (string, e.g. " --smoke-test").
# Caller should: set -e; export PYTHONUNBUFFERED=1; parse "$@" into EXTRA_ARGS; then source this file.
#
# Alliance layout: one combined log under ./results/logs/; checkpoints under ./results/ckpts/<stem>/;
# synthetic cache under ./results/datasets/<stem>_ci_latent_cache/ (submit dir = SLURM_SUBMIT_DIR).

if [ -n "${SLURM_JOB_ID:-}" ]; then
    cd "${SLURM_SUBMIT_DIR:-.}"
    mkdir -p results/logs results/ckpts results/datasets
    ALLIANCE_RUN_STEM="$(date +%m-%d)-${SLURM_JOB_ID: -4}-${SLURM_JOB_NAME:-ci-latent}"
    export ALLIANCE_RUN_STEM
    _log="results/logs/${ALLIANCE_RUN_STEM}.log"
    touch "$_log"
    exec >>"$_log" 2>&1
fi

module purge || true
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

if [ -z "${PROJECT:-}" ] && [ -d "$HOME/projects" ]; then
    shopt -s nullglob
    _m=("$HOME"/projects/def-* "$HOME"/projects/aip-*)
    shopt -u nullglob
    if [ "${#_m[@]}" -gt 0 ]; then
        export PROJECT=$(readlink -f "${_m[0]}")
    fi
fi

if [ -z "${PROJECT:-}" ]; then
    echo "ERROR: PROJECT not found"
    exit 1
fi

VENV_PATH="$PROJECT/$USER/diffusion-tsf/venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment at $VENV_PATH (first run)..."
    mkdir -p "$(dirname "$VENV_PATH")"
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

_RUN_STEM="${ALLIANCE_RUN_STEM:-local-ci}"
CACHE_DIR="${SLURM_SUBMIT_DIR:-$PROJECT_ROOT}/results/datasets/${_RUN_STEM}_ci_latent_cache"
mkdir -p "$CACHE_DIR"

cd "$PROJECT_ROOT"

DIFFUSION_TS="$PROJECT_ROOT/models/diffusion_tsf"
CKPT_PARENT="${SLURM_SUBMIT_DIR:-$PROJECT_ROOT}/results/ckpts/${_RUN_STEM}"
mkdir -p "$CKPT_PARENT/shared" "$CKPT_PARENT/runs"
ln -sfn "$CKPT_PARENT/shared" "$DIFFUSION_TS/checkpoints_ci_etth2"
ln -sfn "$CKPT_PARENT/runs" "$DIFFUSION_TS/checkpoints_ci_runs"
SHARED="$DIFFUSION_TS/checkpoints_ci_etth2"
RUNROOT="$DIFFUSION_TS/checkpoints_ci_runs"

if [ ! -d "$PROJECT_ROOT/datasets/ETT-small" ] || [ ! -d "$PROJECT_ROOT/datasets/exchange_rate" ]; then
    if [ ! -f "$PROJECT_ROOT/datasets/ETT-small/ETTh1.csv" ] && [ -d "$PROJECT/$USER/diffusion-tsf/datasets" ]; then
        echo "Symlinking datasets from \$PROJECT/\$USER/diffusion-tsf/datasets..."
        ln -sf "$PROJECT/$USER/diffusion-tsf/datasets" "$PROJECT_ROOT/datasets"
    fi
fi

if [ ! -d "$PROJECT_ROOT/datasets/ETT-small" ]; then
    echo "ERROR: Missing datasets/ETT-small"
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

IMAGE_H=128
STAGE4_TRIALS=12
EXCHANGE_SEED=42

run_py() {
    "$PY" -u -m models.diffusion_tsf.train_ci_latent_etth2 \
        --cache-dir "$CACHE_DIR" \
        --shared-ckpt-dir "$SHARED" \
        --run-ckpt-dir "$RUNROOT" \
        --image-height "$IMAGE_H" \
        "$@"
}
