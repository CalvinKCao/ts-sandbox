#!/bin/bash
# Diagnostic: L40S max train-step batch for electricity V=321 and V=160.
# Not a training wrapper. From repo root on the Killarney login node:
#   ./temp/scripts/submit_probe_electricity_l40s_max_batch.sh

set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
    echo "Submitting electricity L40S max-B probe (b1, 30 min)..."
    sbatch \
        --job-name=probe-l40s-maxb \
        --account=aip-boyuwang \
        --partition=gpubase_l40s_b1 \
        --time=0:30:00 \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task=8 \
        --mem=50G \
        --output="$SCRIPT_DIR/results/logs/probe-l40s-maxb-%j.log" \
        --error="$SCRIPT_DIR/results/logs/probe-l40s-maxb-%j.log" \
        --mail-type=FAIL \
        --mail-user=ccao87@uwo.ca \
        "$SCRIPT_DIR/temp/scripts/submit_probe_electricity_l40s_max_batch.sh"
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

REPO_ROOT="${SLURM_SUBMIT_DIR:?submit from repo root}"
cd "$REPO_ROOT"
mkdir -p results/logs
[[ -f models/diffusion_tsf/dit.py ]] || { echo "ERROR: not repo root: $REPO_ROOT" >&2; exit 1; }
REQ="$REPO_ROOT/setup/requirements-killarney.txt"
[[ -f "$REQ" ]] || { echo "ERROR: missing $REQ" >&2; exit 1; }
[[ -n "${SLURM_TMPDIR:-}" ]] || { echo "ERROR: SLURM_TMPDIR unset" >&2; exit 1; }

export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="$SLURM_TMPDIR/inductor"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

echo "[setup] Building node-local venv on \$SLURM_TMPDIR"
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck source=/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REQ" -q
python -c "import torch; assert torch.cuda.is_available(), 'CUDA required'; print('torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))"

PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python -u temp/scripts/probe_electricity_l40s_max_batch.py \
    --config configs/binary_window_norm_patch_refine_canvas128_p64x6_electricity_allv_msdefault_fixed.yaml \
    --max-bs 32 \
    --variates 321 160 \
    --stages coarse patch_refine

echo "Finished: $(date)"
