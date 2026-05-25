#!/bin/bash
# =============================================================================
# Compare direct RealTS training vs wave-curriculum -> RealTS fine-tuning.
#
# Login node: fans out one Slurm job per (arm, seed, lr) so runs happen in
# parallel. Compute node: executes one job config exported via ARM/SEED/LR.
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ALL_ARMS=(
    direct_realts
    wave_curriculum_realts
)
DEFAULT_SEEDS=(11 23 37)
DEFAULT_LRS=(1e-4 2e-4)
DEFAULT_VARIANT="dit_tiny_no_guidance"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    IS_SMOKE=0
    ARMS=("${ALL_ARMS[@]}")
    SEEDS=("${DEFAULT_SEEDS[@]}")
    LRS=("${DEFAULT_LRS[@]}")
    VARIANT="$DEFAULT_VARIANT"
    PASS_ARGS=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --smoke-test)
                IS_SMOKE=1
                PASS_ARGS+=("$1")
                shift
                ;;
            --arms)
                IFS=',' read -r -a ARMS <<< "$2"
                shift 2
                ;;
            --seeds)
                IFS=',' read -r -a SEEDS <<< "$2"
                shift 2
                ;;
            --lrs)
                IFS=',' read -r -a LRS <<< "$2"
                shift 2
                ;;
            --variant)
                VARIANT="$2"
                shift 2
                ;;
            *)
                PASS_ARGS+=("$1")
                shift
                ;;
        esac
    done

    if [ "$IS_SMOKE" -eq 1 ]; then
        SEEDS=(11)
        LRS=(2e-4)
        WALLTIME="6:00:00"
        MEM="10G"
        CPUS=4
        SUFFIX="-smoke"
    else
        WALLTIME="12:00:00"
        MEM="32G"
        CPUS=8
        SUFFIX=""
    fi

    mkdir -p "$SCRIPT_DIR/results/bootstrap"
    SB_OUT='results/bootstrap/%x-%j.out'
    SB_ERR='results/bootstrap/%x-%j.err'

    JOB_IDS=()
    echo "Submitting ${#ARMS[@]} arm(s) x ${#SEEDS[@]} seed(s) x ${#LRS[@]} lr(s) in parallel..."
    for arm in "${ARMS[@]}"; do
        arm="${arm// /}"
        [ -z "$arm" ] && continue
        for seed in "${SEEDS[@]}"; do
            seed="${seed// /}"
            [ -z "$seed" ] && continue
            for lr in "${LRS[@]}"; do
                lr="${lr// /}"
                [ -z "$lr" ] && continue
                arm_tag="${arm//_/-}"
                lr_tag="${lr//./p}"
                lr_tag="${lr_tag//-/m}"
                lr_tag="${lr_tag//+/}"
                JOB_NAME="realts-cur-${arm_tag}-s${seed}-lr${lr_tag}${SUFFIX}"
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
                    --export="ALL,ARM=${arm},SEED=${seed},LR=${lr},VARIANT_NAME=${VARIANT},TS_SANDBOX_PROJECT_ROOT_SUBMIT_DIR=1" \
                    "$SCRIPT_DIR/slurm_realts_curriculum_experiment.sh" "${PASS_ARGS[@]}")
                JOB_IDS+=("$job_id")
                echo "  -> arm=$arm seed=$seed lr=$lr job=$job_id"
            done
        done
    done

    echo ""
    echo "=================================================================="
    echo "Monitor:    squeue -u \$USER"
    echo "Cancel all: scancel ${JOB_IDS[*]}"
    echo "=================================================================="
    exit 0
fi

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}"

if [ -z "${ARM:-}" ] || [ -z "${SEED:-}" ] || [ -z "${LR:-}" ]; then
    echo "ERROR: ARM, SEED, and LR must be exported by the login-node fan-out."
    exit 1
fi

ARM_TAG="${ARM//_/-}"
LR_TAG="${LR//./p}"
LR_TAG="${LR_TAG//-/m}"
LR_TAG="${LR_TAG//+/}"
STEM="$(date +%m-%d)-${SLURM_JOB_ID: -3}-realts-cur-${ARM_TAG}-s${SEED}-lr${LR_TAG}"
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
echo "Arm: $ARM"
echo "Seed: $SEED"
echo "LR: $LR"
echo "Variant: ${VARIANT_NAME:-$DEFAULT_VARIANT}"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"
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

python -c "import torch; assert torch.cuda.is_available(), 'CUDA required'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1

mkdir -p "$SLURM_TMPDIR/realts-cache"

PY_ARGS=()
SMOKE_FLAG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test)
            SMOKE_FLAG="--smoke-test"
            shift
            ;;
        --arms|--seeds|--lrs|--variant)
            shift 2
            ;;
        --arms=*|--seeds=*|--lrs=*|--variant=*)
            shift
            ;;
        *)
            PY_ARGS+=("$1")
            shift
            ;;
    esac
done

python -u -m models.diffusion_tsf.train_realts_curriculum_experiment \
    --arm "$ARM" \
    --seed "$SEED" \
    --lr "$LR" \
    --variant "${VARIANT_NAME:-$DEFAULT_VARIANT}" \
    --checkpoint-dir "$CKPT_DIR" \
    --results-dir "$DATA_DIR" \
    --cache-dir "$SLURM_TMPDIR/realts-cache" \
    $SMOKE_FLAG \
    "${PY_ARGS[@]}"

echo "=========================================="
echo "Done: $(date)"
echo "Results: $DATA_DIR"
echo "Checkpoints: $CKPT_DIR"
echo "Log: $LOG_FILE"
echo "=========================================="
