#!/bin/bash
# Binary vs MMPD staged diag — per-window eval + top-gap plots (lb336/hz720 ordinal).
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_binary_mmpd_staged_diag_killarney.sh --smoke-test
#   ./submit_binary_mmpd_staged_diag_killarney.sh
#   ./submit_binary_mmpd_staged_diag_killarney.sh --datasets ETTh1,weather --force-eval
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    SMOKE=0
    WALL="2:00:00"
    EXTRA=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --smoke-test|--smoke) SMOKE=1; shift ;;
            --time) WALL="$2"; shift 2 ;;
            *) EXTRA+=("$1"); shift ;;
        esac
    done

    if [[ "$SMOKE" -eq 1 ]]; then
        WALL="0:45:00"
        JOB_NAME="bin-mmpd-diag-smoke"
        EXTRA=(--datasets ETTh1 "${EXTRA[@]}")
    else
        JOB_NAME="bin-mmpd-diag"
    fi

    LOG_DIR="${SCRIPT_DIR}/results/logs"
    mkdir -p "$LOG_DIR"
    DATE_STR="$(date +%m-%d)"
    LOG_FILE="$LOG_DIR/${DATE_STR}-bin-mmpd-diag-%j.log"

    echo "Submitting $JOB_NAME (L40S, wall=$WALL)..."
    exec sbatch \
        --parsable \
        --job-name="$JOB_NAME" \
        --account=aip-boyuwang \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task=8 \
        --mem=60G \
        --time="$WALL" \
        --output="$LOG_FILE" \
        --error="$LOG_FILE" \
        --mail-type=FAIL \
        --mail-user="${USER}@uwo.ca" \
        "$SCRIPT_DIR/submit_binary_mmpd_staged_diag_killarney.sh" \
        ${SMOKE:+--smoke-test} \
        "${EXTRA[@]}"
fi

# ---------------------------------------------------------------------------
# Compute node
# ---------------------------------------------------------------------------
REPO="${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}"
REQ="$REPO/setup/requirements-killarney.txt"
STORE="${RESULTS_ROOT:-$SCRATCH/ts-sandbox/results}"
[[ -d "$REPO" ]] || { echo "ERROR: repo missing at $REPO" >&2; exit 1; }
[[ -f "$REQ" ]] || { echo "ERROR: missing $REQ — run ./setup/killarney_freeze_requirements.sh" >&2; exit 1; }
[[ -n "${SLURM_TMPDIR:-}" ]] || { echo "ERROR: SLURM_TMPDIR unset" >&2; exit 1; }

MMPD_DIR="results/datasets/07-08-mmpd-decoder-ordinal-norm-lb336-hz720"
BINARY_CONFIG="configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm.yaml"
MMPD_CONFIG="configs/mmpd_decoder_flat_subsets_paper_lb336_hz720_ordinal_norm.yaml"
DATASETS="ETTh1,weather,electricity,exchange_rate,traffic"
OUTPUT_DIR="reports/binary_vs_mmpd_ordinal_lb336_hz720"
FORCE_EVAL=0
EXTRA_PY=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) shift ;;
        --mmpd-dir) MMPD_DIR="$2"; shift 2 ;;
        --binary-config) BINARY_CONFIG="$2"; shift 2 ;;
        --mmpd-config) MMPD_CONFIG="$2"; shift 2 ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --force-eval) FORCE_EVAL=1; shift ;;
        *) EXTRA_PY+=("$1"); shift ;;
    esac
done

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID   Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU:    $(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"
echo "Started: $(date)"
echo "Repo:   $REPO"
echo "=========================================="

module purge 2>/dev/null || true
module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9 2>/dev/null || true
command -v virtualenv >/dev/null || { echo "ERROR: virtualenv missing after module load" >&2; exit 1; }

echo "[setup] node-local venv from $REQ"
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck source=/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REQ" -q
python -c "import torch, yaml, einops; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TS_SANDBOX_REPO="$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

MMPD_ABS="$REPO/$MMPD_DIR"
[[ -d "$MMPD_ABS" ]] || { echo "ERROR: mmpd dir missing: $MMPD_ABS" >&2; exit 1; }
[[ -d "$REPO/temp/MMPD" ]] || { echo "ERROR: MMPD repo missing at $REPO/temp/MMPD" >&2; exit 1; }

PY_ARGS=(
    utils/compare_binary_mmpd_staged_diag.py
    --mmpd-dir "$MMPD_ABS"
    --binary-config "$BINARY_CONFIG"
    --mmpd-config "$MMPD_CONFIG"
    --binary-ckpt-base "$STORE/ckpts"
    --datasets "$DATASETS"
    --output-dir "$REPO/$OUTPUT_DIR"
    --device cuda
)
[[ "$FORCE_EVAL" -eq 1 ]] && PY_ARGS+=(--force-eval)
[[ ${#EXTRA_PY[@]} -gt 0 ]] && PY_ARGS+=("${EXTRA_PY[@]}")

echo "[run] ${PY_ARGS[*]}"
python -u "${PY_ARGS[@]}"

echo "=========================================="
echo "Done: $(date)"
echo "Outputs: $REPO/$OUTPUT_DIR"
echo "Log: see Slurm output file"
echo "=========================================="
