#!/bin/bash
# Diagnostic: A100 40GB max unet_max_chunk_size for electricity coarse+fine generate.
# Not a training wrapper. From repo root on the Narval login node:
#   ./temp/scripts/submit_probe_electricity_a100_chunk.sh

set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
    mkdir -p "$SCRIPT_DIR/results/logs"
    echo "Submitting electricity A100 chunk probe (gpubase_bygpu_b5, 60 min)..."
    sbatch \
        --job-name=probe-elec-a100-chunk \
        --account=def-boyuwang \
        --partition=gpubase_bygpu_b5 \
        --time=0:60:00 \
        --nodes=1 \
        --gpus=a100:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --output="$SCRIPT_DIR/results/logs/probe-elec-a100-chunk-%j.log" \
        --error="$SCRIPT_DIR/results/logs/probe-elec-a100-chunk-%j.log" \
        --mail-type=FAIL \
        --mail-user=ccao87@uwo.ca \
        "$SCRIPT_DIR/temp/scripts/submit_probe_electricity_a100_chunk.sh"
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

CFG="configs/binary_window_norm_patch_refine_canvas128_p64x6_allv_fullT_hz720_nostitch_nopretrain_fixedhp_r0_ms35_kv_a100.yaml"
[[ -f "$CFG" ]] || { echo "ERROR: missing campaign YAML $CFG" >&2; exit 1; }

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

PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python -u temp/scripts/probe_electricity_a100_chunk.py \
    --config "$CFG" \
    --n-variates 321 \
    --lo 4096 \
    --hi 39563 \
    --steps 1

echo "Finished: $(date)"
