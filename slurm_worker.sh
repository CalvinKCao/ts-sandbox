#!/bin/bash
# =============================================================================
# Pure compute-node worker script for diffusion pipeline.
#
# USAGE (do not call directly; use submit_binary.sh):
#   sbatch slurm_worker.sh --config configs/binary_anchor.yaml --dataset ETTh1
#
# Venv: node-local fast path — rebuilds on $SLURM_TMPDIR from
# setup/requirements-killarney.txt (generate via setup/killarney_freeze_requirements.sh).
# =============================================================================

set -euo pipefail

PY_ARGS=("$@")

ts() { date +'%d-%H:%M:%S'; }

echo "$(ts) =========================================="
echo "$(ts) Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID  Node: ${SLURMD_NODENAME:-unknown}"
echo "$(ts) =========================================="

STORE="${GRID_STORE:-$SCRATCH/ts-sandbox/results}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
REQ="$REPO/setup/requirements-killarney.txt"

# Fail fast before venv build if the repo checkout is missing the requested config.
CONFIG_REL=""
for ((i = 1; i < $#; i++)); do
    if [[ "${!i}" == "--config" ]]; then
        j=$((i + 1))
        CONFIG_REL="${!j}"
        break
    fi
done
if [[ -n "$CONFIG_REL" ]]; then
    CONFIG_PATH="$REPO/$CONFIG_REL"
    if [[ ! -f "$CONFIG_PATH" ]]; then
        BRANCH="$(git -C "$REPO" branch --show-current 2>/dev/null || echo unknown)"
        echo "ERROR: config not found: $CONFIG_PATH" >&2
        echo "ERROR: repo branch=$BRANCH — git checkout feat/patch-decoder-cross-variate-ctx && git pull" >&2
        exit 1
    fi
fi

[[ -f "$REQ" ]] || { echo "ERROR: missing $REQ — run ./setup/killarney_freeze_requirements.sh on login node" >&2; exit 1; }
[[ -n "${SLURM_TMPDIR:-}" ]] || { echo "ERROR: SLURM_TMPDIR is not set." >&2; exit 1; }

module purge 2>/dev/null || true
module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9 2>/dev/null || true
command -v virtualenv >/dev/null || { echo "ERROR: virtualenv not available after module load." >&2; exit 1; }

echo "$(ts) [setup] Building node-local venv on \$SLURM_TMPDIR from $REQ"
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck source=/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REQ" -q
python -c "import torch, optuna, wandb, einops, yaml; assert torch.cuda.is_available(), 'CUDA is not available (check driver compatibility)!'; print('torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))"

export PYTHONUNBUFFERED=1

cd "$REPO"
if [[ ! -f "models/diffusion_tsf/train_multivariate_pipeline.py" ]]; then
    echo "ERROR: slurm_worker.sh must be submitted from repo root." >&2
    exit 1
fi

if [[ "${GRID_EVAL_PATCH_REFINE:-0}" -eq 1 ]]; then
    [[ -n "${GRID_DATASET:-}" ]] || { echo "ERROR: GRID_DATASET is unset" >&2; exit 1; }
    [[ -n "${GRID_EXISTING_CKPT:-}" ]] || { echo "ERROR: GRID_EXISTING_CKPT is unset" >&2; exit 1; }
    [[ -n "${GRID_DISC_OUTPUT:-}" ]] || { echo "ERROR: GRID_DISC_OUTPUT is unset" >&2; exit 1; }
    [[ -d "$GRID_EXISTING_CKPT" ]] || { echo "ERROR: checkpoint root missing: $GRID_EXISTING_CKPT" >&2; exit 1; }

    echo "$(ts) [eval] fixed h96 patch-refine checkpoint: $GRID_EXISTING_CKPT"
    echo "$(ts) [eval] discriminator output: $GRID_DISC_OUTPUT"
    python -u temp/eval_univariate_patch_refine_vs_gt.py \
        --datasets "$GRID_DATASET" \
        --checkpoint-dir "$GRID_EXISTING_CKPT" \
        --output-dir "$GRID_DISC_OUTPUT" \
        --test-stride 4 \
        --slice-lengths 8 16 32 \
        --force-raw-eval \
        --force-train
    echo "$(ts) Done"
    exit 0
fi

DATE_STR="${GRID_DATE_STR:-$(date +%m-%d)}"
DS="${GRID_DATASET:-unknown}"
CFG_NAME="${GRID_CFG_NAME:-run}"
RUN_STEM="${GRID_RUN_STEM:-${DATE_STR}-${SLURM_JOB_ID}-${DS}-${CFG_NAME}}"

# Align Slurm display name with the full run stem once the job id is known.
if [[ -n "${SLURM_JOB_ID:-}" && "${SLURM_JOB_NAME:-}" != "$RUN_STEM" ]]; then
    scontrol update "JobId=${SLURM_JOB_ID}" "JobName=${RUN_STEM}" 2>/dev/null || true
fi


CKPT_DIR="$STORE/ckpts/${RUN_STEM}"
DATA_DIR="$STORE/datasets/${RUN_STEM}"
mkdir -p "$CKPT_DIR" "$DATA_DIR"

# Benchmark CSVs live in the repo clone ($SCRATCH/ts-sandbox/datasets), not under $STORE/datasets.
BENCHMARK_DATASETS="${DATASETS_DIR:-$REPO/datasets}"
[[ -d "$BENCHMARK_DATASETS" ]] || {
    echo "ERROR: benchmark data directory not found at $BENCHMARK_DATASETS" >&2
    exit 1
}

echo "$(ts) Repo: $PWD"
echo "$(ts) Checkpoints: $CKPT_DIR"
echo "$(ts) Results: $DATA_DIR"
echo "$(ts) Benchmark CSVs: $BENCHMARK_DATASETS"
echo "$(ts) GPUs: $(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ') ($(nvidia-smi -L 2>/dev/null | head -1 || echo none))"

PY_ARGS+=(
    --checkpoint-dir "$CKPT_DIR"
    --results-dir "$DATA_DIR"
    --datasets-dir "$BENCHMARK_DATASETS"
)

echo "$(ts) [train] Starting pipeline: ${PY_ARGS[*]}"
python -u -m models.diffusion_tsf.train_multivariate_pipeline "${PY_ARGS[@]}"

echo "$(ts) =========================================="
echo "$(ts) Done"
echo "$(ts) =========================================="
