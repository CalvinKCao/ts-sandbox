#!/bin/bash
# Ablation L8/L16 candidate-only disc. Ladder height comes from each run's
# patch_refine_canvas_height (256 legacy / 128 canvas128 leaf).
#
# USAGE (Killarney login, $SCRATCH/ts-sandbox-ordinal-fine):
#   ./temp/scripts/submit_ablation_disc_l8_l16.sh --viz-only --smoke-test
#   CKPT=results/ckpts/<stamp>-ETTh1-binary_window_norm_patch_refine_canvas128_p64x6 \
#     ./temp/scripts/submit_ablation_disc_l8_l16.sh
#
# Forecast packs auto-cache under results/datasets/disc_forecast_cache/ (keyed by
# ckpt+pack+protocol). Reuse without regenerate:
#   ./temp/scripts/submit_ablation_disc_l8_l16.sh --reuse-forecast-cache
# Disable shared cache: --no-forecast-cache
# Force regenerate: --force-raw-eval / --force-mmpd-eval
#
# Outputs (viz ON by default): auroc_table.json, auroc_by_variate.json,
#          viz/<run>/{snap_sanity,pre_post}/ (--viz-sanity all; pass none / --no-viz to skip),
#          viz/disc_disagreement/<run>/ (MMPD-wrong/binary-right and vice versa;
#            --no-disc-disagreement / --no-viz to skip),
#          viz/staged_eval_samples/<run>/ (full-horizon 1d + red-box; --no-redbox-viz to skip).
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

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
        WALL=8:00:00
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

# Non-interactive / bare --export=ALL submits may lack Lmod; init before module.
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
type module >/dev/null 2>&1 || {
    echo "ERROR: Lmod 'module' unavailable after profile source" >&2
    exit 127
}

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

# Prefer SLURM_SUBMIT_DIR (directory where sbatch was invoked) so spool copies
# of this script do not fall back to \$SCRATCH/ts-sandbox (often on main).
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
    PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
case "$PROJECT_ROOT" in
    "${SCRATCH}"/*) ;;
    *)
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

STAMP="$(date +%m-%d-%H%M)"
# Optional tag: OUT_TAG=lookback → ...-ablation-disc-l8-l16-lookback-ETTh1
# Default tag marks the val+test / 80/20 pack protocol (do not clobber old test-only outs).
OUT_TAG="${OUT_TAG:-valtest80}"
OUT_DIR="results/datasets/${STAMP}-ablation-disc-l8-l16-${OUT_TAG}"
mkdir -p "$OUT_DIR" results/slurm

CFG="${DISC_CONFIG:-configs/binary_window_norm_patch_refine_canvas128_p64x6.yaml}"
RUN_NAME="${DISC_RUN_NAME:-window_norm_c128}"
if [ -n "${CKPT:-}" ]; then
    RUN_SPEC="${RUN_NAME}:${CKPT}:${CFG}"
    RUN_ARGS=(--runs "$RUN_SPEC")
else
    # Caller must pass --runs name:ckpt:config (DEFAULT_RUNS has PLACEHOLDER).
    RUN_ARGS=()
fi

# Default protocol: paper val+test pack, chrono 80/20 (+ val-from-train early-stop).
# "$@" overrides (last argparse wins). Legacy test-only: --pack-splits test --train-fraction 0.7 --val-fraction 0.15
python temp/scripts/eval_ablation_disc_l8_l16.py \
    --dataset ETTh1 \
    --output-dir "$OUT_DIR" \
    --lookback 336 \
    --horizon 96 \
    --pack-test-stride 4 \
    --pack-splits val,test \
    --train-fraction 0.8 \
    --val-fraction 0 \
    --fake-agg sample0 \
    --slice-lengths 8 16 \
    --candidate-only \
    --disc-bin-center-shift \
    --num-sampling-steps 20 \
    --mmpd-output-root results/datasets/07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd \
    "${RUN_ARGS[@]}" \
    "$@"
# unique_abs defaults on in Python (--unique-absolute-slices); pass
# --no-unique-absolute-slices via "$@" to A/B the dense path.
# --dataset ETTh1 is hardcoded above; override with "$@" when sweeping.

echo "Finished: $(date)"
echo "output_dir=$OUT_DIR"
