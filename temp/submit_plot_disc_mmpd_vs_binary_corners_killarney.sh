#!/bin/bash
# Quick Killarney job: plot disc MMPD vs binary confusion corners.
#
# USAGE (login node, $SCRATCH/ts-sandbox):
#   ./temp/submit_plot_disc_mmpd_vs_binary_corners_killarney.sh
#   ./temp/submit_plot_disc_mmpd_vs_binary_corners_killarney.sh --dataset electricity --slice-len 8
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    cd "$REPO_ROOT"
    mkdir -p results/logs
    echo "Submitting disc corner plot job (L40S, 30 min)..."
    sbatch \
        --job-name=disc-corner-plots \
        --account=aip-boyuwang \
        --time=0:30:00 \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task=4 \
        --mem=32G \
        --output=results/logs/disc-corner-plots-%j.out \
        --error=results/logs/disc-corner-plots-%j.err \
        --mail-type=END,FAIL \
        --mail-user=ccao87@uwo.ca \
        "$SCRIPT_DIR/submit_plot_disc_mmpd_vs_binary_corners_killarney.sh" "$@"
    exit 0
fi

ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo "=========================================="
echo "$(ts) Job ID: $SLURM_JOB_ID   Node: ${SLURMD_NODENAME:-unknown}"
echo "$(ts) GPU:    $(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"
echo "=========================================="

REPO="${SLURM_SUBMIT_DIR:-$REPO_ROOT}"
cd "$REPO"
REQ="$REPO/setup/requirements-killarney.txt"
[[ -f "$REQ" ]] || { echo "ERROR: missing $REQ"; exit 1; }
[[ -n "${SLURM_TMPDIR:-}" ]] || { echo "ERROR: SLURM_TMPDIR unset"; exit 1; }

DATASET="ETTh1"
SLICE_LEN=8
DISC_RUN="results/datasets/disc-lb336-hz720-ordinal-four-native-stride2"
RAW_RUN="results/datasets/disc-lb336-hz720-ordinal-four-raw-trainval25"
NATIVE_STRIDE=2
N_TOTAL=24
PER_CORNER=2
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --slice-len) SLICE_LEN="$2"; shift 2 ;;
        --disc-run) DISC_RUN="$2"; shift 2 ;;
        --raw-run) RAW_RUN="$2"; shift 2 ;;
        --native-repr-stride) NATIVE_STRIDE="$2"; shift 2 ;;
        --n-total) N_TOTAL="$2"; shift 2 ;;
        --per-corner) PER_CORNER="$2"; shift 2 ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

for src in binary_staged mmpd; do
    npz="$REPO/$RAW_RUN/raw/${src}_${DATASET}.npz"
    ckpt="$REPO/$DISC_RUN/checkpoints/${DATASET}_${src}_L${SLICE_LEN}_discriminator.pt"
    [[ -f "$npz" ]] || { echo "ERROR: missing $npz"; exit 1; }
    [[ -f "$ckpt" ]] || { echo "ERROR: missing $ckpt"; exit 1; }
done

module purge 2>/dev/null || true
module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9 2>/dev/null || true

echo "$(ts) [setup] node-local venv from $REQ"
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck source=/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REQ" -q
python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))"

export PYTHONUNBUFFERED=1
python -u "$REPO/temp/plot_disc_mmpd_vs_binary_confusion_corners.py" \
    --dataset "$DATASET" \
    --slice-len "$SLICE_LEN" \
    --disc-run "$DISC_RUN" \
    --raw-run "$RAW_RUN" \
    --native-repr-stride "$NATIVE_STRIDE" \
    --n-total "$N_TOTAL" \
    --per-corner "$PER_CORNER" \
    "${EXTRA[@]}"

echo "$(ts) done → temp/disc_mmpd_vs_binary_corners/"
echo "$(ts) Finished"
