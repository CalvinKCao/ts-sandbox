#!/bin/bash
# Ablation L8/L16 candidate-only disc. Ladder height comes from each run's
# patch_refine_canvas_height (256 legacy / 128 canvas128 leaf).
#
# USAGE (Killarney login, $SCRATCH/ts-sandbox):
#   ./temp/submit_ablation_disc_l8_l16.sh --viz-only --smoke-test
#   CKPT=results/ckpts/<stamp>-ETTh1-binary_window_norm_patch_refine_canvas128_p64x6 \
#     ./temp/submit_ablation_disc_l8_l16.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    IS_SMOKE=0
    IS_VIZ=0
    for arg in "$@"; do
        [ "$arg" = "--smoke-test" ] && IS_SMOKE=1
        [ "$arg" = "--viz-only" ] && IS_VIZ=1
    done
    if [ "$IS_SMOKE" -eq 1 ] || [ "$IS_VIZ" -eq 1 ]; then
        WALL=0:45:00
        MEM=32G
        NAME=ablation-disc-viz
    else
        WALL=4:00:00
        MEM=50G
        NAME=ablation-disc-l8l16
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
        "$SCRIPT_DIR/submit_ablation_disc_l8_l16.sh" "$@"
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

if [ -z "${SCRATCH:-}" ] || [ ! -d "$SCRATCH/ts-sandbox" ]; then
    echo "ERROR: require Killarney checkout at \$SCRATCH/ts-sandbox" >&2
    exit 1
fi
PROJECT_ROOT="$SCRATCH/ts-sandbox"
cd "$PROJECT_ROOT"

# Prefer node-local venv; fall back to shared scratch .venv (same as other util jobs).
if [ -f "$SLURM_TMPDIR/venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$SLURM_TMPDIR/venv/bin/activate"
elif [ -f "$SLURM_TMPDIR/ts-sandbox-venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$SLURM_TMPDIR/ts-sandbox-venv/bin/activate"
elif [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
else
    echo "ERROR: no venv under \$SLURM_TMPDIR or \$SCRATCH/ts-sandbox/.venv" >&2
    exit 1
fi
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable in job env"
print(f"torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")
PY

STAMP="$(date +%m-%d-%H%M)"
OUT_DIR="results/datasets/${STAMP}-ablation-disc-l8-l16-ETTh1"
mkdir -p "$OUT_DIR" results/slurm

CFG="${DISC_CONFIG:-configs/binary_window_norm_patch_refine_canvas128_p64x6.yaml}"
if [ -n "${CKPT:-}" ]; then
    RUN_SPEC="window_norm_c128:${CKPT}:${CFG}"
    RUN_ARGS=(--runs "$RUN_SPEC")
else
    # Caller must pass --runs name:ckpt:config (DEFAULT_RUNS has PLACEHOLDER).
    RUN_ARGS=()
fi

python temp/eval_ablation_disc_l8_l16.py \
    --dataset ETTh1 \
    --output-dir "$OUT_DIR" \
    --lookback 336 \
    --horizon 96 \
    --pack-test-stride 4 \
    --pack-splits test \
    --fake-agg sample0 \
    --slice-lengths 8 16 \
    --candidate-only \
    --disc-bin-center-shift \
    --num-sampling-steps 20 \
    --mmpd-output-root results/datasets/07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd \
    "${RUN_ARGS[@]}" \
    "$@"

echo "Finished: $(date)"
echo "output_dir=$OUT_DIR"
