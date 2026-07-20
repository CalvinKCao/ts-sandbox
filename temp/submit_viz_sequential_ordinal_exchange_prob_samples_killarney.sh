#!/bin/bash
# =============================================================================
# Viz sequential ordinal (coarse→fine) exchange_rate: 10 spaced windows × 10
# quad_t prob samples → coarse/fine 2D + combined 1D JPGs.
#
# Run: 07-12-4213914-exchange_rate-..._ordinal_norm_g10p0
#
# USAGE (Killarney login, $SCRATCH/ts-sandbox):
#   cd "$SCRATCH/ts-sandbox" && git pull
#   ./temp/submit_viz_sequential_ordinal_exchange_prob_samples_killarney.sh
#   ./temp/submit_viz_sequential_ordinal_exchange_prob_samples_killarney.sh --n-windows 4 --n-prob-samples 4
#   ./temp/submit_viz_sequential_ordinal_exchange_prob_samples_killarney.sh --n-windows 1 --n-prob-samples 1  # smoke
#
# Outputs under:
#   results/viz/sequential_ordinal_exchange_prob_samples/<run>/winXXXX/{coarse,fine,combined}/
# =============================================================================

set -euo pipefail

if [[ -d "${SCRATCH:-}/ts-sandbox" ]]; then
    REPO="$SCRATCH/ts-sandbox"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/temp/visualize_sequential_ordinal_exchange_prob_samples.py" ]]; then
    REPO="$SLURM_SUBMIT_DIR"
else
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
SCRIPT_DIR="$REPO/temp"
VIZ_PY="temp/visualize_sequential_ordinal_exchange_prob_samples.py"

PY_ARGS=("$@")
if [[ ${#PY_ARGS[@]} -eq 0 ]]; then
    PY_ARGS=(
        --n-windows 10
        --n-prob-samples 10
        --device cuda
    )
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    cd "$REPO"
    mkdir -p "$REPO/results/logs"
    echo "Submitting sequential ordinal exchange prob-sample viz (L40S, 2h) from $REPO ..."
    echo "  python $VIZ_PY ${PY_ARGS[*]}"
    sbatch \
        --chdir="$REPO" \
        --job-name="viz-seq-ord-exch" \
        --account=aip-boyuwang \
        --time=2:00:00 \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task=4 \
        --mem=40G \
        --output="$REPO/results/logs/viz-seq-ord-exch-%j.out" \
        --error="$REPO/results/logs/viz-seq-ord-exch-%j.err" \
        --mail-type=END,FAIL \
        --mail-user=ccao87@uwo.ca \
        "$SCRIPT_DIR/submit_viz_sequential_ordinal_exchange_prob_samples_killarney.sh" "${PY_ARGS[@]}"
    exit 0
fi

ts() { date +'%d-%H:%M:%S'; }
echo "$(ts) Job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-?} REPO=$REPO"
echo "$(ts) GPU=$(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"

REQ="$REPO/setup/requirements-killarney.txt"
[[ -f "$REQ" ]] || { echo "ERROR: missing $REQ"; exit 1; }
[[ -n "${SLURM_TMPDIR:-}" ]] || { echo "ERROR: SLURM_TMPDIR unset"; exit 1; }
[[ -f "$REPO/$VIZ_PY" ]] || { echo "ERROR: missing $REPO/$VIZ_PY — sync/pull temp viz script"; exit 1; }

CKPT_DIR=""
shopt -s nullglob
for d in "$REPO"/results/ckpts/*4213914*exchange_rate*ordinal_norm_g10p0; do
    [[ -d "$d" ]] || continue
    if [[ -f "$d/exchange_rate/coarse/best.pt" && -f "$d/exchange_rate/fine/best.pt" \
          && -f "$d/exchange_rate_patch_guidance.pt" ]]; then
        CKPT_DIR="$d"
        echo "$(ts) ckpt ok: $CKPT_DIR"
        break
    fi
done
shopt -u nullglob
[[ -n "$CKPT_DIR" ]] || {
    echo "ERROR: missing coarse/fine/guidance under results/ckpts/*4213914*ordinal_norm_g10p0"
    exit 1
}

HAS_CKPT_FLAG=0
for a in "${PY_ARGS[@]}"; do
    [[ "$a" == "--checkpoint-dir" ]] && HAS_CKPT_FLAG=1
done
if [[ "$HAS_CKPT_FLAG" -eq 0 ]]; then
    PY_ARGS+=(--checkpoint-dir "$CKPT_DIR")
fi

module purge 2>/dev/null || true
module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9 2>/dev/null || true

echo "$(ts) [setup] node-local venv from $REQ"
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck source=/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REQ" -q
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable after Killarney venv setup'"

cd "$REPO"
export PYTHONUNBUFFERED=1

echo "$(ts) running $VIZ_PY ${PY_ARGS[*]}"
python -u "$REPO/$VIZ_PY" "${PY_ARGS[@]}"
echo "$(ts) done → $REPO/results/viz/sequential_ordinal_exchange_prob_samples"
