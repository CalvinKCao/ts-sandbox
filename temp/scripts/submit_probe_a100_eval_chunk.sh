#!/bin/bash
# Diagnostic: A100 40GB max unet_max_chunk_size for eval-shaped generate
# (n_samples=10, steps=4, real V, unique-seg, cache_cond_kv).
# Not a training wrapper. From repo root on the Narval login node:
#   ./temp/scripts/submit_probe_a100_eval_chunk.sh \
#       --dataset traffic --n-variates 862 \
#       --config configs/binary_window_norm_patch_refine_canvas128_p32x6_allv_fullT_hz720_nostitch_nopretrain_fixedhp_r0_ms35_kv_a100.yaml \
#       --lo 512 --hi 13753

set -euo pipefail

DATASET=""
N_VARIATES=""
CONFIG=""
LO="1024"
HI=""
STEPS="4"
N_SAMPLES="10"
WALL="0:60:00"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --n-variates) N_VARIATES="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --lo) LO="$2"; shift 2 ;;
        --hi) HI="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --n-samples) N_SAMPLES="$2"; shift 2 ;;
        --time) WALL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$DATASET" || -z "$N_VARIATES" || -z "$CONFIG" ]]; then
    echo "ERROR: --dataset --n-variates --config required" >&2
    exit 1
fi

DS_SLUG="${DATASET}"
DS_SLUG="${DS_SLUG//_/-}"
JOB_NAME="probe-${DS_SLUG}-a100-n${N_SAMPLES}"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
    mkdir -p "$SCRIPT_DIR/results/logs"
    echo "Submitting ${JOB_NAME} (gpubase_bygpu_b5, ${WALL})..."
    PY_FWD=(
        --dataset "$DATASET"
        --n-variates "$N_VARIATES"
        --config "$CONFIG"
        --lo "$LO"
        --steps "$STEPS"
        --n-samples "$N_SAMPLES"
        --time "$WALL"
    )
    if [[ -n "$HI" ]]; then
        PY_FWD+=(--hi "$HI")
    fi
    sbatch \
        --job-name="$JOB_NAME" \
        --account=def-boyuwang \
        --partition=gpubase_bygpu_b5 \
        --time="$WALL" \
        --nodes=1 \
        --gpus=a100:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --output="$SCRIPT_DIR/results/logs/probe-${DS_SLUG}-a100-n${N_SAMPLES}-%j.log" \
        --error="$SCRIPT_DIR/results/logs/probe-${DS_SLUG}-a100-n${N_SAMPLES}-%j.log" \
        --mail-type=FAIL \
        --mail-user=ccao87@uwo.ca \
        "$SCRIPT_DIR/temp/scripts/submit_probe_a100_eval_chunk.sh" \
        "${PY_FWD[@]}"
    exit 0
fi

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID   Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU:    $(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"
echo "Started: $(date)"
echo "dataset=$DATASET V=$N_VARIATES n_samples=$N_SAMPLES steps=$STEPS lo=$LO hi=${HI:-n_items+1}"
echo "config=$CONFIG"
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
[[ -f "$CONFIG" ]] || { echo "ERROR: missing campaign YAML $CONFIG" >&2; exit 1; }
PROBE="$REPO_ROOT/temp/scripts/probe_a100_eval_chunk.py"
[[ -f "$PROBE" ]] || { echo "ERROR: missing $PROBE" >&2; exit 1; }

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

PY_ARGS=(
    --config "$CONFIG"
    --dataset "$DATASET"
    --n-variates "$N_VARIATES"
    --n-samples "$N_SAMPLES"
    --lo "$LO"
    --steps "$STEPS"
)
if [[ -n "$HI" ]]; then
    PY_ARGS+=(--hi "$HI")
fi

PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python -u "$PROBE" "${PY_ARGS[@]}"

echo "Finished: $(date)"
