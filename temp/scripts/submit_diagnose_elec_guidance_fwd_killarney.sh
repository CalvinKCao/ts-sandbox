#!/bin/bash
# Diagnostic: electricity guidance fwd+bwd benchmark (patch vs iT vs no-xattn).
# Not a training wrapper. From repo root on Killarney login node:
#   ./temp/scripts/submit_diagnose_elec_guidance_fwd_killarney.sh

set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
    mkdir -p "$SCRIPT_DIR/results/logs"
    echo "Submitting electricity guidance fwd benchmark (L40S, 96G, 30 min)..."
    sbatch \
        --job-name=diag-elec-guid-fwd \
        --account=aip-boyuwang \
        --time=0:30:00 \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task=8 \
        --mem=96G \
        --output="$SCRIPT_DIR/results/logs/diag-elec-guid-fwd-%j.log" \
        --error="$SCRIPT_DIR/results/logs/diag-elec-guid-fwd-%j.log" \
        --mail-type=FAIL \
        --mail-user=ccao87@uwo.ca \
        "$SCRIPT_DIR/temp/scripts/submit_diagnose_elec_guidance_fwd_killarney.sh"
    exit 0
fi

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID   Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU:    $(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"
echo "Started: $(date)"
echo "Git:    $(git -C "$SLURM_SUBMIT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "=========================================="

if ! type module >/dev/null 2>&1; then
    if [ -f /cvmfs/soft.computecanada.ca/config/profile/bash.sh ]; then
        export SKIP_CC_CVMFS="${SKIP_CC_CVMFS:-0}"
        set +u
        # shellcheck disable=SC1091
        source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
        set -u
    elif [ -f /etc/profile.d/z00_lmod.sh ]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/z00_lmod.sh
    fi
fi
type module >/dev/null 2>&1 || {
    echo "ERROR: Lmod 'module' unavailable after profile source" >&2
    exit 127
}

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

STEM="$(date +%m-%d)-${SLURM_JOB_ID: -3}-elec-guid-fwd"
OUT="results/logs/${STEM}.json"

PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python -u temp/scripts/benchmark_elec_guidance_fwd.py \
    --config configs/binary_window_norm_patch_refine_canvas128_p64x6_electricity_allv_msdefault_fixed.yaml \
    --u-rows 217 \
    --n-variates 321 \
    --warmup 5 \
    --timed 100 \
    --out "$OUT"

echo "Results: $REPO_ROOT/$OUT"
echo "Log:     $REPO_ROOT/results/logs/diag-elec-guid-fwd-${SLURM_JOB_ID}.log"
echo "Finished: $(date)"
