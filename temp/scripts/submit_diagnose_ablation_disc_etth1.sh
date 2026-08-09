#!/bin/bash
# =============================================================================
# Util job: L8/L16 disc for window-norm (4524397) vs ordinal residual-fine
# (4525834) on ETTh1. Writes zoomed 256-row disc-input panels first, then
# trains the univariate discriminator.
#
# USAGE (from repo root on Killarney login node):
#   ./temp/scripts/submit_diagnose_ablation_disc_etth1.sh --viz-only   # L40S, panels only
#   ./temp/scripts/submit_diagnose_ablation_disc_etth1.sh              # full disc
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    IS_VIZ=0
    for arg in "$@"; do [ "$arg" = "--viz-only" ] && IS_VIZ=1; done
    WALL="2:00:00"
    JOB="disc-ablation-etth1"
    if [ "$IS_VIZ" -eq 1 ]; then
        WALL="0:45:00"
        JOB="disc-ablation-viz"
    fi
    echo "Submitting $JOB (L40S, exclude kn010)..."
    sbatch \
        --job-name="$JOB" \
        --account=aip-boyuwang \
        --time="$WALL" \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task=8 \
        --mem=50G \
        --exclude=kn010 \
        --output="$SCRIPT_DIR/results/logs/${JOB}-%j.log" \
        --error="$SCRIPT_DIR/results/logs/${JOB}-%j.log" \
        --mail-type=FAIL \
        --mail-user=ccao87@uwo.ca \
        "$SCRIPT_DIR/temp/scripts/submit_diagnose_ablation_disc_etth1.sh" "$@"
    exit 0
fi

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID   Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU:    $(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"
echo "Started: $(date)"
echo "=========================================="

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

if [ -d "$SCRATCH/ts-sandbox" ]; then
    PROJECT_ROOT="$SCRATCH/ts-sandbox"
elif [ -d "$SCRIPT_DIR" ]; then
    PROJECT_ROOT="$SCRIPT_DIR"
else
    echo "ERROR: repo not found" >&2
    exit 1
fi
cd "$PROJECT_ROOT"
mkdir -p results/logs results/datasets

if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
elif [ -f "$SLURM_TMPDIR/ts-sandbox-venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$SLURM_TMPDIR/ts-sandbox-venv/bin/activate"
fi

WN_CKPT="$PROJECT_ROOT/results/ckpts/08-01-4524397-ETTh1-binary_window_norm_patch_refine_earlyjuly_norm"
OF_CKPT="$PROJECT_ROOT/results/ckpts/08-02-4525834-ETTh1-binary_ordinal_fine_finer_earlyjuly_hps"
MMPD_ROOT="$PROJECT_ROOT/results/datasets/07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd"
OUT="results/datasets/08-03-disc-ablation-window-norm-vs-ordinal-fine"
RAW="${OUT}-raw"

# Prefer worktree ckpt locations if main checkout lacks the run dirs.
if [ ! -d "$WN_CKPT" ] && [ -d "$SCRATCH/ts-sandbox-window-norm/results/ckpts" ]; then
    WN_CKPT="$(ls -d "$SCRATCH"/ts-sandbox-window-norm/results/ckpts/*4524397* 2>/dev/null | head -1)"
fi
if [ ! -d "$OF_CKPT" ] && [ -d "$SCRATCH/ts-sandbox-ordinal-fine/results/ckpts" ]; then
    OF_CKPT="$(ls -d "$SCRATCH"/ts-sandbox-ordinal-fine/results/ckpts/*4525834* 2>/dev/null | head -1)"
fi

[[ -d "$WN_CKPT" ]] || { echo "ERROR: missing window-norm ckpt: $WN_CKPT" >&2; exit 1; }
[[ -d "$OF_CKPT" ]] || { echo "ERROR: missing ordinal-fine ckpt: $OF_CKPT" >&2; exit 1; }
[[ -d "$MMPD_ROOT/raw" ]] || { echo "ERROR: missing MMPD root: $MMPD_ROOT" >&2; exit 1; }

EXTRA_ARGS=()
for arg in "$@"; do
    EXTRA_ARGS+=("$arg")
done

python -u temp/scripts/eval_univariate_disc_two_ablations_vs_gt.py \
    --dataset ETTh1 \
    --models \
      "window_norm=${WN_CKPT}:configs/binary_window_norm_patch_refine_earlyjuly_norm.yaml" \
      "ordinal_fine=${OF_CKPT}:configs/binary_ordinal_fine_finer_earlyjuly_hps.yaml" \
    --mmpd-output-root "$MMPD_ROOT" \
    --output-dir "$OUT" \
    --raw-eval-dir "$RAW" \
    --force-raw-eval \
    --slice-lengths 8 16 \
    --fake-agg sample0 \
    --pack-test-stride 4 \
    --disc-index-stride 1 \
    --candidate-only \
    --disc-bin-center-shift \
    --viz-windows 2 \
    --viz-zoom-steps 8 \
    "${EXTRA_ARGS[@]}"

echo "Finished: $(date)"
