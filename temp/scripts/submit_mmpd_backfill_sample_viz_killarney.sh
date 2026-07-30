#!/bin/bash
# =============================================================================
# Backfill MMPD anchor + probabilistic sample panels → disk + wandb.
#
# USAGE (Killarney login):
#   cd "$SCRATCH/ts-sandbox" && git pull
#   ./temp/scripts/submit_mmpd_backfill_sample_viz_killarney.sh
#   ./temp/scripts/submit_mmpd_backfill_sample_viz_killarney.sh --local-only
#   ./temp/scripts/submit_mmpd_backfill_sample_viz_killarney.sh --n-windows 6
# =============================================================================

set -euo pipefail

if [[ -d "${SCRATCH:-}/ts-sandbox" ]]; then
    REPO="$SCRATCH/ts-sandbox"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/temp/scripts/mmpd_backfill_sample_viz.py" ]]; then
    REPO="$SLURM_SUBMIT_DIR"
else
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

VIZ_PY="temp/scripts/mmpd_backfill_sample_viz.py"
DEFAULT_OUT="results/datasets/07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd"

PY_ARGS=("$@")
if [[ ${#PY_ARGS[@]} -eq 0 ]]; then
    PY_ARGS=(
        --output-dir "$DEFAULT_OUT"
        --datasets electricity ETTh1 dynamic traffic
        --n-windows 4
    )
fi

# ---------------------------------------------------------------------------
# Login node → sbatch (CPU only; reads npz + matplotlib + wandb)
# ---------------------------------------------------------------------------
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    cd "$REPO"
    mkdir -p "$REPO/results/logs"
    echo "Submitting MMPD sample-viz backfill (CPU, 30 min) from $REPO ..."
    echo "  python $VIZ_PY ${PY_ARGS[*]}"
    sbatch \
        --chdir="$REPO" \
        --job-name="mmpd-sample-viz-backfill" \
        --account=aip-boyuwang \
        --time=0:30:00 \
        --nodes=1 \
        --cpus-per-task=4 \
        --mem=16G \
        --output="$REPO/results/logs/mmpd-sample-viz-backfill-%j.out" \
        --error="$REPO/results/logs/mmpd-sample-viz-backfill-%j.out" \
        --mail-type=FAIL \
        --mail-user=ccao87@uwo.ca \
        "$REPO/temp/scripts/submit_mmpd_backfill_sample_viz_killarney.sh" "${PY_ARGS[@]}"
    exit 0
fi

# ---------------------------------------------------------------------------
# Compute node
# ---------------------------------------------------------------------------
ts() { date +'%d-%H:%M:%S'; }
echo "$(ts) Job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-?} REPO=$REPO"

REQ="$REPO/setup/requirements-killarney.txt"
[[ -f "$REQ" ]] || { echo "ERROR: missing $REQ"; exit 1; }
[[ -n "${SLURM_TMPDIR:-}" ]] || { echo "ERROR: SLURM_TMPDIR unset"; exit 1; }
[[ -f "$REPO/$VIZ_PY" ]] || { echo "ERROR: missing $REPO/$VIZ_PY — git pull"; exit 1; }

module purge 2>/dev/null || true
module load StdEnv/2023 python/3.11 2>/dev/null || true

echo "$(ts) [setup] node-local venv from $REQ"
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck source=/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REQ" -q

cd "$REPO"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "$(ts) [wandb] WANDB_API_KEY set"
else
    echo "$(ts) [wandb] WARN: WANDB_API_KEY unset — use --local-only or export the key"
fi

echo "$(ts) running $VIZ_PY ${PY_ARGS[*]}"
python -u "$REPO/$VIZ_PY" "${PY_ARGS[@]}"
echo "$(ts) done"
