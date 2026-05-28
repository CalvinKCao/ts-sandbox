#!/bin/bash
# =============================================================================
# Slurm: one-window probabilistic (dpmpp) shape test for completed bin-h64 ckpts.
# Skips datasets without best.pt unless --require-all.
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./slurm_binary_h64_prob_shape_test.sh
#   ./slurm_binary_h64_prob_shape_test.sh --date-tag 05-27 --height 64
#   ./slurm_binary_h64_prob_shape_test.sh --num-sampling-steps 20
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATE_TAG="$(date +%m-%d)"
HEIGHT=64
NUM_STEPS=5
REQUIRE_ALL=0
EXTRA_DATASETS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --date-tag) DATE_TAG="$2"; shift 2 ;;
        --height) HEIGHT="$2"; shift 2 ;;
        --num-sampling-steps) NUM_STEPS="$2"; shift 2 ;;
        --require-all) REQUIRE_ALL=1; shift ;;
        --datasets)
            shift
            EXTRA_DATASETS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                EXTRA_DATASETS+=("$1")
                shift
            done
            ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if [[ -d "${SCRATCH:-}/ts-sandbox" ]]; then
        REPO="${SCRATCH}/ts-sandbox"
    elif [[ -d "$HOME/ts-sandbox" ]]; then
        REPO="$HOME/ts-sandbox"
    else
        REPO="$SCRIPT_DIR"
    fi
    if [[ "$REPO" == /home/* ]]; then
        echo "ERROR: submit from \$SCRATCH/ts-sandbox on Killarney, not /home." >&2
        exit 1
    fi

    LOG_DIR="$REPO/results/logs"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/${DATE_TAG}-bin-h${HEIGHT}-prob-shape-test.log"
    JOB_NAME="bin-h${HEIGHT}-prob-shape"

    SUBMIT_ARGS=(
        --date-tag "$DATE_TAG"
        --height "$HEIGHT"
        --num-sampling-steps "$NUM_STEPS"
    )
    [[ "$REQUIRE_ALL" -eq 1 ]] && SUBMIT_ARGS+=(--require-all)
    [[ ${#EXTRA_DATASETS[@]} -gt 0 ]] && SUBMIT_ARGS+=(--datasets "${EXTRA_DATASETS[@]}")

    echo "Submitting ${JOB_NAME} (wall=0:30:00, L40S) log=${LOG_FILE}"
    JID=$(sbatch --parsable \
        --job-name="$JOB_NAME" \
        --account=aip-boyuwang \
        --time=0:30:00 \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task=4 \
        --mem=24G \
        --chdir="$REPO" \
        --output="$LOG_FILE" \
        --error="$LOG_FILE" \
        --mail-type=FAIL \
        --mail-user=ccao87@uwo.ca \
        "$REPO/slurm_binary_h64_prob_shape_test.sh" \
        "${SUBMIT_ARGS[@]}")
    echo "Submitted job ${JID}"
    echo "  tail -f ${LOG_FILE}"
    exit 0
fi

cd "$SLURM_SUBMIT_DIR"
PROJECT_ROOT="$SLURM_SUBMIT_DIR"
if [[ ! -f "$PROJECT_ROOT/utils/shape_test_binary_prob.py" ]]; then
    echo "ERROR: run from ts-sandbox repo root." >&2
    exit 1
fi

LOG_FILE="$PROJECT_ROOT/results/logs/${DATE_TAG}-bin-h${HEIGHT}-prob-shape-test.log"
mkdir -p "$(dirname "$LOG_FILE")"
exec >>"$LOG_FILE" 2>&1

echo "=========================================="
echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID  Node: ${SLURMD_NODENAME:-unknown}"
echo "Repo: $PROJECT_ROOT"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "Started: $(date)"
echo "=========================================="

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

echo "[setup] venv on SLURM_TMPDIR..."
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index \
    'torch==2.11.0+computecanada' numpy pandas scipy scikit-learn tqdm einops -q
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA required"
print("torch", torch.__version__, "gpu", torch.cuda.get_device_name(0))
PY

export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT"

SHAPE_ARGS=(
    --date-tag "$DATE_TAG"
    --height "$HEIGHT"
    --num-sampling-steps "$NUM_STEPS"
)
[[ "$REQUIRE_ALL" -eq 1 ]] && SHAPE_ARGS+=(--require-all)
if [[ ${#EXTRA_DATASETS[@]} -gt 0 ]]; then
    SHAPE_ARGS+=(--datasets "${EXTRA_DATASETS[@]}")
fi

echo "[shape-test] python utils/shape_test_binary_prob.py ${SHAPE_ARGS[*]}"
python -u utils/shape_test_binary_prob.py "${SHAPE_ARGS[@]}"
RC=$?
echo "Done: $(date)  exit=$RC"
exit "$RC"
