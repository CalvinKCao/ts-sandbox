#!/bin/bash
# =============================================================================
# Offline synth-pretrain coherence viz (RealTS lookback / GT / pred panels).
#
# USAGE (Killarney login, repo = $SCRATCH/ts-sandbox):
#   cd "$SCRATCH/ts-sandbox" && git pull
#   # Auto-picks newest results/ckpts/*-ETTh2-..._vertical_dual_g1p0
#   ./temp/submit_viz_synth_pretrain_killarney.sh
#   ./temp/submit_viz_synth_pretrain_killarney.sh \
#     --run-dir results/ckpts/07-15-4243853-ETTh2-binary_noise_sched_ablation_vertical_dual_g1p0 \
#     --n-samples 4 --sampler anchor
# =============================================================================

set -euo pipefail

if [[ -d "${SCRATCH:-}/ts-sandbox" ]]; then
    REPO="$SCRATCH/ts-sandbox"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/temp/viz_synth_pretrain.py" ]]; then
    REPO="$SLURM_SUBMIT_DIR"
else
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
SCRIPT_DIR="$REPO/temp"
VIZ_PY="temp/viz_synth_pretrain.py"

# Defaults: g1 vertical_dual synth pretrain under results/ckpts (not reused/).
CONFIG="${VIZ_CONFIG:-configs/binary_noise_sched_ablation_vertical_dual_g1p0.yaml}"
DATASET="${VIZ_DATASET:-ETTh2}"
PY_ARGS=("$@")
if [[ ${#PY_ARGS[@]} -eq 0 ]]; then
    PY_ARGS=(
        --config "$CONFIG"
        --dataset "$DATASET"
        --n-samples 4
        --sampler anchor
    )
fi

# ---------------------------------------------------------------------------
# Login node → sbatch L40S
# ---------------------------------------------------------------------------
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    cd "$REPO"
    mkdir -p "$REPO/results/logs"
    echo "Submitting synth-pretrain viz (L40S, 30 min) from $REPO ..."
    echo "  python $VIZ_PY ${PY_ARGS[*]}"
    sbatch \
        --chdir="$REPO" \
        --job-name="viz-synth-pretrain" \
        --account=aip-boyuwang \
        --time=0:30:00 \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task=4 \
        --mem=32G \
        --output="$REPO/results/logs/viz-synth-pretrain-%j.out" \
        --error="$REPO/results/logs/viz-synth-pretrain-%j.err" \
        --mail-type=END,FAIL \
        --mail-user=ccao87@uwo.ca \
        "$SCRIPT_DIR/submit_viz_synth_pretrain_killarney.sh" "${PY_ARGS[@]}"
    exit 0
fi

# ---------------------------------------------------------------------------
# Compute node
# ---------------------------------------------------------------------------
ts() { date +'%d-%H:%M:%S'; }
echo "$(ts) Job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-?} REPO=$REPO"
echo "$(ts) GPU=$(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"

REQ="$REPO/setup/requirements-killarney.txt"
[[ -f "$REQ" ]] || { echo "ERROR: missing $REQ"; exit 1; }
[[ -n "${SLURM_TMPDIR:-}" ]] || { echo "ERROR: SLURM_TMPDIR unset"; exit 1; }
[[ -f "$REPO/$VIZ_PY" ]] || { echo "ERROR: missing $REPO/$VIZ_PY — git pull"; exit 1; }

module purge 2>/dev/null || true
module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9 2>/dev/null || true

echo "$(ts) [setup] node-local venv from $REQ"
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck source=/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REQ" -q

cd "$REPO"
echo "$(ts) running $VIZ_PY ${PY_ARGS[*]}"
python -u "$REPO/$VIZ_PY" "${PY_ARGS[@]}"
echo "$(ts) done"
