#!/bin/bash
# Materialize ETTh1 4678498 test forecasts + crop-level flat/wiggle pred accuracy.
#
# USAGE (Killarney login, prefer $SCRATCH/ts-sandbox-ordinal-fine):
#   ./temp/scripts/submit_etth1_flat_undersample_pred_acc.sh
#   ./temp/scripts/submit_etth1_flat_undersample_pred_acc.sh --max-windows 8   # smoke
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    IS_SMOKE=0
    for arg in "$@"; do [ "$arg" = "--max-windows" ] && IS_SMOKE=1; done
    # Heuristic: short wall if caller passes --max-windows
    if [ "$IS_SMOKE" -eq 1 ]; then
        WALL=0:30:00
        MEM=32G
        NAME=etth1-flat-pred-smoke
    else
        WALL=2:00:00
        MEM=40G
        NAME=etth1-flat-pred-acc
    fi
    mkdir -p "$REPO_ROOT/results/slurm"
    sbatch \
        --job-name="$NAME" \
        --account=aip-boyuwang \
        --time="$WALL" \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task=8 \
        --mem="$MEM" \
        --exclude=kn010 \
        --export=ALL \
        --output="$REPO_ROOT/results/slurm/%x-%j.out" \
        --error="$REPO_ROOT/results/slurm/%x-%j.err" \
        --mail-type=END,FAIL \
        --mail-user=ccao87@uwo.ca \
        "$SCRIPT_DIR/submit_etth1_flat_undersample_pred_acc.sh" "$@"
    exit 0
fi

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID   Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU:    $(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"
echo "Started: $(date)"
echo "=========================================="

if ! type module >/dev/null 2>&1; then
    if [ -f /cvmfs/soft.computecanada.ca/config/profile/bash.sh ]; then
        export SKIP_CC_CVMFS="${SKIP_CC_CVMFS:-0}"
        export FORCE_CC_CVMFS="${FORCE_CC_CVMFS:-0}"
        set +u
        # shellcheck disable=SC1091
        source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
        set -u
    elif [ -f /etc/profile.d/z00_lmod.sh ]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/z00_lmod.sh
    fi
fi

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9
module load scipy-stack/2024a
module load arrow/16.1.0

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
case "$PROJECT_ROOT" in
    /home/*|/project/*)
        if [ -n "${SCRATCH:-}" ] && [ -d "$SCRATCH/ts-sandbox-ordinal-fine" ]; then
            PROJECT_ROOT="$SCRATCH/ts-sandbox-ordinal-fine"
        elif [ -n "${SCRATCH:-}" ] && [ -d "$SCRATCH/ts-sandbox" ]; then
            PROJECT_ROOT="$SCRATCH/ts-sandbox"
        else
            echo "ERROR: cannot resolve PROJECT_ROOT under \$SCRATCH" >&2
            exit 1
        fi
        ;;
esac
cd "$PROJECT_ROOT"
echo "PROJECT_ROOT=$PROJECT_ROOT"

REQ="$PROJECT_ROOT/setup/requirements-killarney.txt"
[[ -f "$REQ" ]] || { echo "ERROR: missing $REQ" >&2; exit 1; }
[[ -n "${SLURM_TMPDIR:-}" ]] || { echo "ERROR: SLURM_TMPDIR unset" >&2; exit 1; }
command -v virtualenv >/dev/null || { echo "ERROR: virtualenv missing after module load" >&2; exit 1; }

echo "[setup] Building node-local venv on \$SLURM_TMPDIR from $REQ"
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck disable=SC1091
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REQ" -q
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable in job env"
print(f"torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")
PY

CKPT="${CKPT:-results/ckpts/08-09-4678498-ETTh1-binary_window_norm_patch_refine_canvas128_p64x6_etth1_flat_undersample}"
CFG="${CFG:-configs/binary_window_norm_patch_refine_canvas128_p64x6_etth1_flat_undersample.yaml}"
OUT_DIR="${OUT_DIR:-temp/lean_disc_c128_results/etth1_flat_undersample_pred_acc}"
mkdir -p "$OUT_DIR" results/slurm

python temp/scripts/etth1_flat_undersample_pred_acc.py \
    --ckpt "$CKPT" \
    --config "$CFG" \
    --output-dir "$OUT_DIR" \
    --pack-splits test \
    --pack-test-stride 4 \
    --lookback 336 \
    --horizon 96 \
    --batch-size 2 \
    --num-sampling-steps 20 \
    --probabilistic-sampler quad_t \
    "$@"

echo "Finished: $(date)"
echo "output_dir=$OUT_DIR"
ls -la "$OUT_DIR" || true
