#!/bin/bash
# =============================================================================
# MSE loss weight ablation — iterates datasets x weight values.
#
# Sources the cluster job_preamble.sh if available, otherwise uses run.sh
# environment setup. Runs finetune-only (assumes pretrained checkpoints exist).
#
# USAGE (inside a Slurm job or local env with an activated venv):
#   bash run_mse_variants.sh
#   bash run_mse_variants.sh --smoke-test
#
# From login node via run.sh dispatch:
#   ./run.sh --dataset etth1 -- bash run_mse_variants.sh   (not recommended)
#   Instead, source job_preamble.sh and run directly.
#
# The script respects SUBMIT_ROOT_FOR_PROJECT / TS_SANDBOX_PROJECT_ROOT_SUBMIT_DIR
# by resolving PROJECT_ROOT through the same logic as run.sh.
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Resolve project root (same logic as run.sh) ----
if [ "${TS_SANDBOX_PROJECT_ROOT_SUBMIT_DIR:-}" = "1" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
elif [ -n "${SCRATCH:-}" ] && [ -d "$SCRATCH/ts-sandbox" ]; then
    PROJECT_ROOT="$SCRATCH/ts-sandbox"
elif [ -d "$HOME/ts-sandbox" ]; then
    PROJECT_ROOT="$HOME/ts-sandbox"
else
    PROJECT_ROOT="$SCRIPT_DIR"
fi
export PROJECT_ROOT
cd "$PROJECT_ROOT"

# Source cluster preamble if available (loads modules, activates venv)
if [ -n "${STORE:-}" ] && [ -f "$STORE/job_preamble.sh" ]; then
    echo "Sourcing $STORE/job_preamble.sh"
    source "$STORE/job_preamble.sh"
fi

SMOKE_FLAG=""
if [ "${1:-}" = "--smoke-test" ]; then
    SMOKE_FLAG="--smoke-test"
    shift
fi

# checkpoint and results dirs — use Slurm job layout if available
CKPT_DIR="${RUN_CKPT_DIR:-results/mse_ablation/ckpts}"
RES_DIR="${RUN_DATA_DIR:-results/mse_ablation/datasets}"
mkdir -p "$CKPT_DIR" "$RES_DIR"

PYTHON="python -u -m models.diffusion_tsf.train_multivariate_pipeline"

# ---- Experiment matrix ----
DATASETS=("ETTh1" "ETTm1" "exchange_rate")
WEIGHTS=(0.05 0.2 0.5)

# Per-dataset variate dim (matches run_experiments.sh / run.sh discover_dims)
declare -A DIM_MAP
DIM_MAP["ETTh1"]=7
DIM_MAP["ETTm1"]=7
DIM_MAP["exchange_rate"]=8

for ds in "${DATASETS[@]}"; do
    dim="${DIM_MAP[$ds]}"
    for w in "${WEIGHTS[@]}"; do
        tag="${ds}_mse${w}"
        echo ""
        echo "============================================================"
        echo "  $tag  (dataset=$ds  mse_loss_weight=$w  dim=$dim)"
        echo "============================================================"

        $PYTHON \
            --mode finetune \
            --dataset "$ds" \
            --n-variates "$dim" \
            --mse-loss-weight "$w" \
            --model-type dit \
            --guidance-penalty-weight 0.2 \
            --checkpoint-dir "$CKPT_DIR" \
            --results-dir "$RES_DIR" \
            --subset-id "$tag" \
            --fresh \
            --wandb \
            $SMOKE_FLAG \
            || echo "[WARN] $tag failed with exit code $?"
    done
done

echo ""
echo "============================================================"
echo "  MSE variant sweep complete"
echo "============================================================"
