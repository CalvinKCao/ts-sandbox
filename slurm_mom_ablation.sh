#!/bin/bash
# =============================================================================
# Slurm: median-of-means vs mean diffusion eval on 5% test subset.
#
# Evaluates default full-pipeline (3539360–3539365) + exp A/B/A+B checkpoints
# under results/ and results/runs/ (sync from cluster first).
#
# USAGE (login node, repo root):
#   ./slurm_mom_ablation.sh --smoke-test
#   ./slurm_mom_ablation.sh
#   ./slurm_mom_ablation.sh --only default
#   ./slurm_mom_ablation.sh --only exp_A,ETTh1
#
# Checkout before submit (see reports or commit message for SHA):
#   git fetch origin && git checkout throwaway/learned-render-hybrid && git pull
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "$SCRIPT_DIR/results/logs"
    IS_SMOKE=0
    EXTRA_ARGS=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --smoke-test) IS_SMOKE=1; shift ;;
            *) EXTRA_ARGS+=("$1"); shift ;;
        esac
    done

    if [ "$IS_SMOKE" -eq 1 ]; then
        JOB_NAME="mom-ablation-smoke"
        WALLTIME="00:45:00"
        MEM="16G"
        DEFAULT_PY_ARGS=(--max-runs 2 --n-samples 5 --mom-repeats 2 --test-fraction 0.05 --batch-size 4)
    else
        JOB_NAME="mom-ablation"
        WALLTIME="12:00:00"
        MEM="50G"
        DEFAULT_PY_ARGS=(--test-fraction 0.05 --n-samples 30 --mom-blocks 5 --mom-repeats 10 --batch-size 8)
    fi

    echo "Submitting $JOB_NAME (${WALLTIME}) ..."
    sbatch \
        --job-name="$JOB_NAME" \
        --account=aip-boyuwang \
        --time="$WALLTIME" \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task=8 \
        --mem="$MEM" \
        --chdir="$SCRIPT_DIR" \
        --output=/dev/null \
        --error=/dev/null \
        --mail-type=END,FAIL \
        --mail-user=ccao87@uwo.ca \
        --export="ALL,IS_SMOKE=$IS_SMOKE" \
        "$SCRIPT_DIR/slurm_mom_ablation.sh" "${EXTRA_ARGS[@]}"
    exit 0
fi

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

ALLIANCE_RUN_STEM="$(date +%m-%d)-${SLURM_JOB_ID}-mom-ablation"
LOG_DIR="$SLURM_SUBMIT_DIR/results/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${ALLIANCE_RUN_STEM}.log"
exec >>"$LOG_FILE" 2>&1

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID   Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU:    $(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"
echo "Started: $(date)"
echo "Log: $LOG_FILE"
echo "=========================================="

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

if [ -d "${SCRATCH:-}/ts-sandbox" ]; then
    PROJECT_ROOT="${SCRATCH}/ts-sandbox"
elif [ -d "$HOME/ts-sandbox" ]; then
    PROJECT_ROOT="$HOME/ts-sandbox"
else
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
fi
cd "$PROJECT_ROOT"
echo "PROJECT_ROOT=$PROJECT_ROOT"
git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || true

if [ -n "${SLURM_TMPDIR:-}" ]; then
    python -m venv "$SLURM_TMPDIR/env"
    # shellcheck source=/dev/null
    source "$SLURM_TMPDIR/env/bin/activate"
    pip install --no-index --upgrade pip -q 2>/dev/null || pip install --upgrade pip -q
    pip install --no-index torch numpy scipy pandas scikit-learn tqdm einops -q 2>/dev/null || \
        pip install torch numpy scipy pandas scikit-learn tqdm einops -q
    pip install reformer-pytorch wandb optuna matplotlib -q 2>/dev/null || true
else
    source "$PROJECT_ROOT/.venv/bin/activate" 2>/dev/null || true
fi

export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

PY_ARGS=(--out-json "$LOG_DIR/${ALLIANCE_RUN_STEM}_results.json" --out-md "$LOG_DIR/${ALLIANCE_RUN_STEM}_report.md")
if [ "${IS_SMOKE:-0}" = "1" ]; then
    PY_ARGS+=(--max-runs 2 --n-samples 5 --mom-repeats 2 --test-fraction 0.05 --batch-size 4)
else
    PY_ARGS+=(--test-fraction 0.05 --n-samples 30 --mom-blocks 5 --mom-repeats 10 --batch-size 8)
fi
if [ "$#" -gt 0 ]; then
    PY_ARGS+=("$@")
fi

python -u utils/eval_mom_ablation.py "${PY_ARGS[@]}"

echo "=========================================="
echo "Done: $(date)"
echo "JSON: $LOG_DIR/${ALLIANCE_RUN_STEM}_results.json"
echo "MD:   $LOG_DIR/${ALLIANCE_RUN_STEM}_report.md"
echo "=========================================="
