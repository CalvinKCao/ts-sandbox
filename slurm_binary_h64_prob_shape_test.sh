#!/bin/bash
# =============================================================================
# Slurm: probabilistic (dpmpp, non-anchor) shape or texture test for bin-h64/h128 ckpts.
# Skips datasets without best.pt unless --require-all.
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./slurm_binary_h64_prob_shape_test.sh
#   ./slurm_binary_h64_prob_shape_test.sh --mode texture --heights 64 128
#   ./slurm_binary_h64_prob_shape_test.sh --date-tag 05-27 --height 64
#   ./slurm_binary_h64_prob_shape_test.sh --num-sampling-steps 20
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATE_TAG="$(date +%m-%d)"
HEIGHT=64
HEIGHTS=()
MODE="shape"
NUM_STEPS=5
REQUIRE_ALL=0
TEST_MAX_ITEMS=""
INDICES_DIR=""
OUTPUT=""
EXTRA_DATASETS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --date-tag) DATE_TAG="$2"; shift 2 ;;
        --height) HEIGHT="$2"; shift 2 ;;
        --heights)
            shift
            HEIGHTS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                HEIGHTS+=("$1")
                shift
            done
            ;;
        --mode) MODE="$2"; shift 2 ;;
        --num-sampling-steps) NUM_STEPS="$2"; shift 2 ;;
        --test-max-items) TEST_MAX_ITEMS="$2"; shift 2 ;;
        --indices-dir) INDICES_DIR="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
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
    [[ ${#HEIGHTS[@]} -eq 0 ]] && HEIGHTS=("$HEIGHT")
    HEIGHT_TAG="${HEIGHTS[*]}"
    HEIGHT_TAG="${HEIGHT_TAG// /-}"
    LOG_SUFFIX="prob-${MODE}-h${HEIGHT_TAG}"
    LOG_FILE="$LOG_DIR/${DATE_TAG}-${LOG_SUFFIX}.log"
    JOB_NAME="bin-${LOG_SUFFIX}"

    SUBMIT_ARGS=(
        --date-tag "$DATE_TAG"
        --mode "$MODE"
        --heights "${HEIGHTS[@]}"
        --num-sampling-steps "$NUM_STEPS"
    )
    [[ "$REQUIRE_ALL" -eq 1 ]] && SUBMIT_ARGS+=(--require-all)
    [[ -n "$TEST_MAX_ITEMS" ]] && SUBMIT_ARGS+=(--test-max-items "$TEST_MAX_ITEMS")
    [[ -n "$INDICES_DIR" ]] && SUBMIT_ARGS+=(--indices-dir "$INDICES_DIR")
    [[ -n "$OUTPUT" ]] && SUBMIT_ARGS+=(--output "$OUTPUT")
    [[ ${#EXTRA_DATASETS[@]} -gt 0 ]] && SUBMIT_ARGS+=(--datasets "${EXTRA_DATASETS[@]}")

    WALL="0:30:00"
    [[ "$MODE" == "texture" ]] && WALL="2:00:00"

    echo "Submitting ${JOB_NAME} (wall=${WALL}, L40S) log=${LOG_FILE}"
    JID=$(sbatch --parsable \
        --job-name="$JOB_NAME" \
        --account=aip-boyuwang \
        --time="$WALL" \
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

[[ ${#HEIGHTS[@]} -eq 0 ]] && HEIGHTS=("$HEIGHT")
HEIGHT_TAG="${HEIGHTS[*]}"
HEIGHT_TAG="${HEIGHT_TAG// /-}"
LOG_SUFFIX="prob-${MODE}-h${HEIGHT_TAG}"
LOG_FILE="$PROJECT_ROOT/results/logs/${DATE_TAG}-${LOG_SUFFIX}.log"
mkdir -p "$(dirname "$LOG_FILE")"
exec >>"$LOG_FILE" 2>&1

export PYTHONUNBUFFERED=1
export PROJECT_ROOT
PROJECT_ROOT="$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID  Node: ${SLURMD_NODENAME:-unknown}"
echo "Repo: $PROJECT_ROOT"
echo "PWD: $(pwd)"
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
if ! pip install --no-index optuna -q 2>/dev/null; then
    echo "[setup] optuna missing from wheelhouse; trying PyPI"
    pip install optuna -q
fi
python -c "import optuna" || { echo "[setup] FATAL: optuna not installed"; exit 1; }
pip install --no-index reformer_pytorch -q 2>/dev/null \
    || pip install --no-index reformer-pytorch -q 2>/dev/null \
    || pip install reformer-pytorch -q 2>/dev/null \
    || echo "[setup] reformer-pytorch not installed (OK unless Reformer iTrans)"

# Minimal import smoke (repo on PYTHONPATH; no utils.* importlib)
python - <<'PY'
import os
import sys

repo = os.environ["PROJECT_ROOT"]
assert os.path.isdir(repo), repo
assert repo in sys.path or sys.path[0] in ("", repo), (sys.path[:3], repo)

import optuna
import torch

assert torch.cuda.is_available(), "CUDA required"
from models.diffusion_tsf.train_multivariate_pipeline import get_itransformer_class

get_itransformer_class()
print(
    "setup OK:",
    "torch", torch.__version__,
    "gpu", torch.cuda.get_device_name(0),
    "optuna", optuna.__version__,
    "cwd", os.getcwd(),
)
PY

SHAPE_ARGS=(
    --date-tag "$DATE_TAG"
    --mode "$MODE"
    --heights "${HEIGHTS[@]}"
    --num-sampling-steps "$NUM_STEPS"
)
[[ "$REQUIRE_ALL" -eq 1 ]] && SHAPE_ARGS+=(--require-all)
[[ -n "$TEST_MAX_ITEMS" ]] && SHAPE_ARGS+=(--test-max-items "$TEST_MAX_ITEMS")
[[ -n "$INDICES_DIR" ]] && SHAPE_ARGS+=(--indices-dir "$INDICES_DIR")
[[ -n "$OUTPUT" ]] && SHAPE_ARGS+=(--output "$OUTPUT")
if [[ ${#EXTRA_DATASETS[@]} -gt 0 ]]; then
    SHAPE_ARGS+=(--datasets "${EXTRA_DATASETS[@]}")
fi

echo "[prob-test] python utils/shape_test_binary_prob.py ${SHAPE_ARGS[*]}"
python -u utils/shape_test_binary_prob.py "${SHAPE_ARGS[@]}"
RC=$?
echo "Done: $(date)  exit=$RC"
exit "$RC"
