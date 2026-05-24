#!/bin/bash
# =============================================================================
# Synthetic DiT capacity probe — parallel fan-out on Killarney (L40S default).
#
# Login node: submits one independent Slurm job per variant (runs in parallel).
# Compute node: trains a single variant passed via VARIANT=... in the environment.
#
# USAGE (login node, repo root):
#   ./slurm_synthetic_dit_capacity.sh --smoke-test
#   ./slurm_synthetic_dit_capacity.sh
#   ./slurm_synthetic_dit_capacity.sh --variants dit_tiny_no_guidance,dit_large_no_guidance
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ALL_VARIANTS=(
    dit_tiny_no_guidance
    dit_default_no_guidance
    dit_large_no_guidance
    dit_default_with_guidance
)

if [ -z "${SLURM_JOB_ID:-}" ]; then
    IS_SMOKE=0
    VARIANTS=("${ALL_VARIANTS[@]}")
    PASS_ARGS=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --smoke-test)
                IS_SMOKE=1
                PASS_ARGS+=("$1")
                shift
                ;;
            --variants)
                IFS=',' read -r -a VARIANTS <<< "$2"
                shift 2
                ;;
            *)
                PASS_ARGS+=("$1")
                shift
                ;;
        esac
    done

    mkdir -p "$SCRIPT_DIR/results/bootstrap"
    SB_OUT='results/bootstrap/%x-%j.out'
    SB_ERR='results/bootstrap/%x-%j.err'

    if [ "$IS_SMOKE" -eq 1 ]; then
        WALLTIME="8:00:00"
        MEM="8G"
        CPUS=4
        SUFFIX="-smoke"
    else
        WALLTIME="4:00:00"
        MEM="50G"
        CPUS=8
        SUFFIX=""
    fi

    JOB_IDS=()
    echo "Submitting ${#VARIANTS[@]} parallel job(s) (L40S, wall=$WALLTIME)..."
    for variant in "${VARIANTS[@]}"; do
        variant="${variant// /}"
        [ -z "$variant" ] && continue
        JOB_NAME="synth-dit-${variant}${SUFFIX}"
        JOB_NAME="${JOB_NAME//_/-}"

        job_id=$(sbatch --parsable \
            --job-name="$JOB_NAME" \
            --account=aip-boyuwang \
            --time="$WALLTIME" \
            --nodes=1 \
            --gres=gpu:l40s:1 \
            --cpus-per-task="$CPUS" \
            --mem="$MEM" \
            --chdir="$SCRIPT_DIR" \
            --output="$SB_OUT" \
            --error="$SB_ERR" \
            --mail-type=END,FAIL \
            --mail-user=ccao87@uwo.ca \
            --export="ALL,VARIANT=${variant},TS_SANDBOX_PROJECT_ROOT_SUBMIT_DIR=1" \
            "$SCRIPT_DIR/slurm_synthetic_dit_capacity.sh" "${PASS_ARGS[@]}")
        JOB_IDS+=("$job_id")
        echo "  -> $variant: job $job_id"
    done

    echo ""
    echo "=================================================================="
    for i in "${!VARIANTS[@]}"; do
        echo "  ${VARIANTS[$i]}  ${JOB_IDS[$i]}"
    done
    echo ""
    echo "  Monitor:    squeue -u \$USER"
    echo "  Cancel all: scancel ${JOB_IDS[*]}"
    echo "=================================================================="
    exit 0
fi

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}"

if [ -z "${VARIANT:-}" ]; then
    echo "ERROR: VARIANT not set inside Slurm job (login node should export VARIANT=...)"
    exit 1
fi

VARIANT_TAG="${VARIANT//_/-}"
STEM="$(date +%m-%d)-${SLURM_JOB_ID: -3}-synth-dit-${VARIANT_TAG}"
RUN_ROOT="$SLURM_SUBMIT_DIR/results/$STEM"
LOG_DIR="$RUN_ROOT/logs"
CKPT_DIR="$RUN_ROOT/ckpts"
DATA_DIR="$RUN_ROOT/datasets"
mkdir -p "$LOG_DIR" "$CKPT_DIR" "$DATA_DIR"
LOG_FILE="$LOG_DIR/${STEM}.log"
touch "$LOG_FILE"
exec >>"$LOG_FILE" 2>&1

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID   Node: ${SLURMD_NODENAME:-unknown}"
echo "Variant: $VARIANT"
echo "GPU:    $(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"
echo "Started: $(date)"
echo "Log: $LOG_FILE"
echo "=========================================="

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

if [ "${TS_SANDBOX_PROJECT_ROOT_SUBMIT_DIR:-}" = "1" ] && [ -d "$SLURM_SUBMIT_DIR/models/diffusion_tsf" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
elif [ -d "${SCRATCH:-}/ts-sandbox" ]; then
    PROJECT_ROOT="${SCRATCH}/ts-sandbox"
elif [ -d "$SLURM_SUBMIT_DIR/models/diffusion_tsf" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
elif [ -d "$HOME/ts-sandbox" ]; then
    PROJECT_ROOT="$HOME/ts-sandbox"
else
    echo "ERROR: ts-sandbox not found"
    exit 1
fi
cd "$PROJECT_ROOT"
echo "PROJECT_ROOT=$PROJECT_ROOT"

echo "[setup] venv on SLURM_TMPDIR..."
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck source=/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index \
    'torch==2.11.0+computecanada' \
    numpy pandas scipy scikit-learn tqdm einops -q
# iTransformer imports reformer_pytorch at module load; not in Alliance wheelhouse on Killarney.
# Standard Model uses FullAttention only — optional import in SelfAttention_Family.py covers that.
# Still try wheel/PyPI when available (e.g. login-built shared venv workflows).
pip install --no-index reformer-pytorch -q 2>/dev/null || pip install reformer-pytorch -q 2>/dev/null || true

python -c "import torch; assert torch.cuda.is_available(), 'CUDA required'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1

SMOKE_FLAG=""
PY_ARGS=(--variants "$VARIANT")
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test)
            SMOKE_FLAG="--smoke-test"
            shift
            ;;
        --variants)
            shift 2
            ;;
        --variants=*)
            shift
            ;;
        *)
            PY_ARGS+=("$1")
            shift
            ;;
    esac
done

python -u -m models.diffusion_tsf.train_synthetic_dit_capacity \
    --results-dir "$LOG_DIR" \
    $SMOKE_FLAG \
    "${PY_ARGS[@]}"

echo "=========================================="
echo "Done: $(date)"
echo "Variant: $VARIANT"
echo "Metrics: $LOG_DIR"
echo "Log: $LOG_FILE"
echo "=========================================="
