#!/bin/bash
# One-off: staged_eval sample panels for 3 ETTh1 ablation ckpts
# (guided_p8 / window_norm_pr / ordinal_fine). Avoids pipeline KeyError('fine').
#
# USAGE (Killarney login, $SCRATCH/ts-sandbox):
#   ./temp/scripts/submit_viz_ablation_staged_eval_samples.sh
#   FORCE_FRESH_OUT=1 SAMPLER=quad_t NUM_STEPS=20 SKIP_EXISTING=0 \
#     ./temp/scripts/submit_viz_ablation_staged_eval_samples.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SAMPLER="${SAMPLER:-quad_t}"
NUM_STEPS="${NUM_STEPS:-20}"
# Default skip=1 for anchor reuse path; FORCE_FRESH_OUT=1 defaults skip=0.
if [ "${FORCE_FRESH_OUT:-0}" = "1" ]; then
    SKIP_EXISTING="${SKIP_EXISTING:-0}"
else
    SKIP_EXISTING="${SKIP_EXISTING:-1}"
fi
export SAMPLER NUM_STEPS SKIP_EXISTING FORCE_FRESH_OUT="${FORCE_FRESH_OUT:-0}"
# OUT_ROOT may be caller-set; export so sbatch --export=ALL carries it.
[ -n "${OUT_ROOT:-}" ] && export OUT_ROOT

if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "$REPO_ROOT/results/slurm"
    JOB_NAME="viz-abl-staged-${SAMPLER}"
    echo "Submitting ${JOB_NAME} (L40S, 1:30:00, exclude kn010)..."
    sbatch \
        --job-name="$JOB_NAME" \
        --account=aip-boyuwang \
        --time=1:30:00 \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task=8 \
        --mem=32G \
        --exclude=kn010 \
        --export=ALL \
        --output="$REPO_ROOT/results/slurm/%x-%j.out" \
        --error="$REPO_ROOT/results/slurm/%x-%j.err" \
        --mail-type=END,FAIL \
        --mail-user=ccao87@uwo.ca \
        "$SCRIPT_DIR/submit_viz_ablation_staged_eval_samples.sh" "$@"
    exit 0
fi

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID   Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU:    $(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"
echo "Started: $(date)"
echo "SAMPLER=$SAMPLER NUM_STEPS=$NUM_STEPS SKIP_EXISTING=$SKIP_EXISTING"
echo "=========================================="

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

if [ -d "${SCRATCH:-}/ts-sandbox" ]; then
    PROJECT_ROOT="$SCRATCH/ts-sandbox"
elif [ -d "$REPO_ROOT" ]; then
    PROJECT_ROOT="$REPO_ROOT"
else
    echo "ERROR: repo not found" >&2
    exit 1
fi
cd "$PROJECT_ROOT"
mkdir -p results/slurm results/datasets

# ordinal_fine was trained with past_cond_resize_to_horizon=false (native lb336).
# Main scratch ts-sandbox lacks _expand_horizon_cond_to_past_width; use OF worktree.
CODE_ROOT="${TS_SANDBOX_CODE_ROOT:-${SCRATCH:-/scratch/ccao87}/ts-sandbox-ordinal-fine}"
if [ ! -f "$CODE_ROOT/models/diffusion_tsf/diffusion_model.py" ]; then
    echo "ERROR: CODE_ROOT missing diffusion_model.py: $CODE_ROOT" >&2
    exit 1
fi
if ! grep -q "_expand_horizon_cond_to_past_width" \
    "$CODE_ROOT/models/diffusion_tsf/diffusion_model.py"; then
    echo "ERROR: $CODE_ROOT lacks _expand_horizon_cond_to_past_width" >&2
    exit 1
fi
export TS_SANDBOX_CODE_ROOT="$CODE_ROOT"
echo "CODE_ROOT=$CODE_ROOT"

if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
elif [ -f "$SLURM_TMPDIR/venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$SLURM_TMPDIR/venv/bin/activate"
fi

resolve_ckpt() {
    local jobid="$1"
    local main_glob="$2"
    local found=""
    # Prefer exact main path if present
    if [ -d "$PROJECT_ROOT/$main_glob" ]; then
        echo "$PROJECT_ROOT/$main_glob"
        return 0
    fi
    found="$(ls -d "$PROJECT_ROOT"/results/ckpts/*"${jobid}"* 2>/dev/null | head -1 || true)"
    if [ -n "$found" ] && [ -d "$found" ]; then
        echo "$found"
        return 0
    fi
    # Worktree fallbacks used by prior ablation jobs
    for base in \
        "$SCRATCH/ts-sandbox-window-norm" \
        "$SCRATCH/ts-sandbox-ordinal-fine" \
        "$SCRATCH/ts-sandbox-guided-p8" \
        "$SCRATCH"/ts-sandbox-*; do
        [ -d "$base/results/ckpts" ] || continue
        found="$(ls -d "$base"/results/ckpts/*"${jobid}"* 2>/dev/null | head -1 || true)"
        if [ -n "$found" ] && [ -d "$found" ]; then
            echo "$found"
            return 0
        fi
    done
    return 1
}

G_CKPT="$(resolve_ckpt 4519745 results/ckpts/08-01-4519745-ETTh1-binary_patch_refine_lb336_hz96_ordinal_tuned_guided_p8)" || {
    echo "ERROR: missing guided_p8 ckpt (4519745)" >&2; exit 1
}
WN_CKPT="$(resolve_ckpt 4524397 results/ckpts/08-01-4524397-ETTh1-binary_window_norm_patch_refine_earlyjuly_norm)" || {
    echo "ERROR: missing window_norm ckpt (4524397)" >&2; exit 1
}
OF_CKPT="$(resolve_ckpt 4525834 results/ckpts/08-02-4525834-ETTh1-binary_ordinal_fine_finer_earlyjuly_hps)" || {
    echo "ERROR: missing ordinal_fine ckpt (4525834)" >&2; exit 1
}

echo "guided_p8=$G_CKPT"
echo "window_norm_pr=$WN_CKPT"
echo "ordinal_fine=$OF_CKPT"

# Reuse panels from failed 4567965 (guided_p8 + window_norm_pr complete).
# Override with OUT_ROOT=... or FORCE_FRESH_OUT=1 for a new stamp dir.
if [ "${FORCE_FRESH_OUT:-0}" = "1" ]; then
    STAMP="$(date +%m-%d-%H%M)"
    OUT_ROOT="results/datasets/${STAMP}-ablation-staged-eval-samples-${SAMPLER}-ETTh1"
elif [ -n "${OUT_ROOT:-}" ]; then
    :
else
    OUT_ROOT="results/datasets/08-03-1859-ablation-staged-eval-samples-ETTh1"
fi
mkdir -p "$OUT_ROOT"
echo "OUT_ROOT=$OUT_ROOT SKIP_EXISTING=$SKIP_EXISTING SAMPLER=$SAMPLER NUM_STEPS=$NUM_STEPS"

SKIP_FLAG=()
if [ "$SKIP_EXISTING" = "1" ]; then
    SKIP_FLAG=(--skip-existing-runs)
fi

python -u temp/scripts/viz_ablation_staged_eval_samples.py \
    --code-root "$CODE_ROOT" \
    --output-root "$OUT_ROOT" \
    "${SKIP_FLAG[@]}" \
    --dataset ETTh1 \
    --n-samples 10 \
    --sampler "$SAMPLER" \
    --num-sampling-steps "$NUM_STEPS" \
    --runs \
      "guided_p8:${G_CKPT}:configs/binary_patch_refine_lb336_hz96_ordinal_tuned_guided_p8.yaml" \
      "window_norm_pr:${WN_CKPT}:configs/binary_window_norm_patch_refine_earlyjuly_norm.yaml" \
      "ordinal_fine:${OF_CKPT}:configs/binary_ordinal_fine_finer_earlyjuly_hps.yaml" \
    "$@"

echo "Finished: $(date)"
echo "output_root=$OUT_ROOT"
