#!/bin/bash
# =============================================================================
# One-off: h96 binary vs MMPD shared-window anchor + probabilistic sample panels.
#
# USAGE (Killarney login, $SCRATCH/ts-sandbox):
#   cd "$SCRATCH/ts-sandbox" && git pull
#   sbatch --exclude=kn010 ./temp/scripts/submit_plot_binary_vs_mmpd_anchor_prob_h96.sh
#   sbatch --exclude=kn010 ./temp/scripts/submit_plot_binary_vs_mmpd_anchor_prob_h96.sh --smoke-test
# =============================================================================

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-}"
if [[ -z "$REPO" || ! -f "$REPO/temp/scripts/plot_binary_vs_mmpd_anchor_prob_h96.py" ]]; then
    if [[ -d "${SCRATCH:-}/ts-sandbox" ]]; then
        REPO="$SCRATCH/ts-sandbox"
    else
        REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    fi
fi
SCRIPT_DIR="$REPO/temp/scripts"
PY_SCRIPT="temp/scripts/plot_binary_vs_mmpd_anchor_prob_h96.py"
PY_ARGS=("$@")

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    cd "$REPO"
    mkdir -p "$REPO/results/logs"
    IS_SMOKE=0
    for a in "${PY_ARGS[@]:-}"; do [[ "$a" == "--smoke-test" ]] && IS_SMOKE=1; done
    if [[ "$IS_SMOKE" -eq 1 ]]; then
        TIME=0:30:00
        MEM=24G
        CPUS=4
    else
        TIME=2:00:00
        MEM=40G
        CPUS=6
    fi
    echo "Submitting binary-vs-MMPD anchor+prob panels from $REPO ..."
    echo "  python $PY_SCRIPT ${PY_ARGS[*]:-}"
    sbatch \
        --chdir="$REPO" \
        --exclude=kn010 \
        --job-name="viz-anchor-prob-h96" \
        --account=aip-boyuwang \
        --time="$TIME" \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --output="$REPO/results/logs/viz-anchor-prob-h96-%j.out" \
        --error="$REPO/results/logs/viz-anchor-prob-h96-%j.err" \
        --mail-type=END,FAIL \
        --mail-user=ccao87@uwo.ca \
        "$SCRIPT_DIR/submit_plot_binary_vs_mmpd_anchor_prob_h96.sh" "${PY_ARGS[@]}"
    exit 0
fi

ts() { date +'%d-%H:%M:%S'; }
echo "$(ts) Job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-?} REPO=$REPO"
echo "$(ts) GPU=$(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"

REQ="$REPO/setup/requirements-killarney.txt"
[[ -f "$REQ" ]] || { echo "ERROR: missing $REQ"; exit 1; }
[[ -n "${SLURM_TMPDIR:-}" ]] || { echo "ERROR: SLURM_TMPDIR unset"; exit 1; }
[[ -f "$REPO/$PY_SCRIPT" ]] || { echo "ERROR: missing $REPO/$PY_SCRIPT"; exit 1; }

module purge 2>/dev/null || true
module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9 2>/dev/null || true
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck source=/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REQ" -q
python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))"

cd "$REPO"
export PYTHONUNBUFFERED=1
python "$PY_SCRIPT" "${PY_ARGS[@]}"
echo "$(ts) done"
