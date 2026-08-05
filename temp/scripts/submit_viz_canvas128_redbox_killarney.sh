#!/bin/bash
# Full-horizon staged_eval red-box viz for canvas128 window_norm ckpts.
# Reuses temp/scripts/viz_ablation_staged_eval_samples.py (1d hz96 + refine_boxes).
#
# USAGE (Killarney login, $SCRATCH/ts-sandbox-ordinal-fine):
#   DATASET=traffic ./temp/scripts/submit_viz_canvas128_redbox_killarney.sh
#   DATASET=electricity N_SAMPLES=10 ./temp/scripts/submit_viz_canvas128_redbox_killarney.sh
#   DATASET=ETTh2 ./temp/scripts/submit_viz_canvas128_redbox_killarney.sh
#   # ETTh1 already has 08-04-1507 panels; re-run with SKIP_EXISTING=1 if desired:
#   DATASET=ETTh1 SKIP_EXISTING=1 ./temp/scripts/submit_viz_canvas128_redbox_killarney.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET="${DATASET:?set DATASET=ETTh1|ETTh2|traffic|electricity|exchange_rate}"
N_SAMPLES="${N_SAMPLES:-10}"
SAMPLER="${SAMPLER:-quad_t}"
NUM_STEPS="${NUM_STEPS:-20}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
export DATASET N_SAMPLES SAMPLER NUM_STEPS SKIP_EXISTING

JOB_TAG="$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')"
JOB_NAME="viz-c128-rb-${JOB_TAG}"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "$REPO_ROOT/results/slurm"
    echo "Submitting ${JOB_NAME} (L40S, 1h, exclude kn010) dataset=$DATASET from $REPO_ROOT ..."
    sbatch \
        --job-name="$JOB_NAME" \
        --account=aip-boyuwang \
        --time=1:00:00 \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task=8 \
        --mem=32G \
        --exclude=kn010 \
        --export=ALL \
        --output="$REPO_ROOT/results/slurm/%x-%j.out" \
        --error="$REPO_ROOT/results/slurm/%x-%j.err" \
        --mail-type=END,FAIL \
        --mail-user=ccao87@uwo.ca \
        "$SCRIPT_DIR/submit_viz_canvas128_redbox_killarney.sh" "$@"
    exit 0
fi

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID   Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU:    $(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"
echo "Started: $(date)"
echo "DATASET=$DATASET N_SAMPLES=$N_SAMPLES SAMPLER=$SAMPLER NUM_STEPS=$NUM_STEPS"
echo "=========================================="

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
    PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
    PROJECT_ROOT="$REPO_ROOT"
fi
case "$PROJECT_ROOT" in
    "${SCRATCH}"/*) ;;
    *)
        if [ -d "${SCRATCH:-}/ts-sandbox-ordinal-fine" ]; then
            PROJECT_ROOT="$SCRATCH/ts-sandbox-ordinal-fine"
        elif [ -d "${SCRATCH:-}/ts-sandbox" ]; then
            PROJECT_ROOT="$SCRATCH/ts-sandbox"
        else
            echo "ERROR: cannot resolve PROJECT_ROOT under \$SCRATCH" >&2
            exit 1
        fi
        ;;
esac
cd "$PROJECT_ROOT"
echo "PROJECT_ROOT=$PROJECT_ROOT"

CODE_ROOT="${TS_SANDBOX_CODE_ROOT:-$PROJECT_ROOT}"
export TS_SANDBOX_CODE_ROOT="$CODE_ROOT"
echo "CODE_ROOT=$CODE_ROOT"

case "$DATASET" in
    ETTh1)
        CFG="configs/binary_window_norm_patch_refine_canvas128_p64x6.yaml"
        CKPT_DEFAULT="results/ckpts/08-03-4571065-ETTh1-binary_window_norm_patch_refine_canvas128_p64x6"
        JOBID_HINT=4571065
        ;;
    ETTh2)
        CFG="configs/binary_window_norm_patch_refine_canvas128_p64x6_etth2.yaml"
        CKPT_DEFAULT="results/ckpts/08-04-4601319-ETTh2-binary_window_norm_patch_refine_canvas128_p64x6_etth2"
        JOBID_HINT=4601319
        ;;
    traffic)
        CFG="configs/binary_window_norm_patch_refine_canvas128_p64x6_traffic.yaml"
        CKPT_DEFAULT="results/ckpts/08-04-4597055-traffic-binary_window_norm_patch_refine_canvas128_p64x6_traffic"
        JOBID_HINT=4597055
        ;;
    electricity)
        CFG="configs/binary_window_norm_patch_refine_canvas128_p64x6_electricity.yaml"
        CKPT_DEFAULT="results/ckpts/08-04-4597054-electricity-binary_window_norm_patch_refine_canvas128_p64x6_electricity"
        JOBID_HINT=4597054
        ;;
    exchange_rate)
        CFG="configs/binary_window_norm_patch_refine_canvas128_p64x6_exchange_rate.yaml"
        CKPT_DEFAULT="results/ckpts/08-04-4597056-exchange_rate-binary_window_norm_patch_refine_canvas128_p64x6_exchange_rate"
        JOBID_HINT=4597056
        ;;
    *)
        echo "ERROR: unsupported DATASET=$DATASET" >&2
        exit 1
        ;;
esac

CKPT="${CKPT:-}"
if [ -z "$CKPT" ]; then
    for cand in "$CKPT_DEFAULT" \
        "$SCRATCH/ts-sandbox-ordinal-fine/$CKPT_DEFAULT" \
        "$SCRATCH/ts-sandbox/$CKPT_DEFAULT"; do
        if [ -d "$cand" ]; then
            CKPT="$cand"
            break
        fi
    done
fi
if [ -z "$CKPT" ] || [ ! -d "$CKPT" ]; then
    CKPT="$(ls -d "$PROJECT_ROOT"/results/ckpts/*"${JOBID_HINT}"* 2>/dev/null | head -1 || true)"
fi
[ -n "$CKPT" ] && [ -d "$CKPT" ] || {
    echo "ERROR: missing canvas128 ckpt for $DATASET (hint job $JOBID_HINT)" >&2
    exit 1
}
[ -f "$CFG" ] || { echo "ERROR: missing $CFG" >&2; exit 1; }
echo "CKPT=$CKPT CFG=$CFG"

REQ="$PROJECT_ROOT/setup/requirements-killarney.txt"
[[ -f "$REQ" ]] || { echo "ERROR: missing $REQ" >&2; exit 1; }
[[ -n "${SLURM_TMPDIR:-}" ]] || { echo "ERROR: SLURM_TMPDIR unset" >&2; exit 1; }

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
OUT_ROOT="${OUT_ROOT:-results/datasets/${STAMP}-ablation-staged-eval-samples-canvas128-${DATASET}}"
mkdir -p "$OUT_ROOT" results/slurm results/datasets
echo "OUT_ROOT=$OUT_ROOT"

VIZ_PY="temp/scripts/viz_ablation_staged_eval_samples.py"
[ -f "$VIZ_PY" ] || { echo "ERROR: missing $VIZ_PY" >&2; exit 1; }
[ -f temp/eval_ablation_disc_l8_l16.py ] || [ -f temp/scripts/eval_ablation_disc_l8_l16.py ] || {
    echo "ERROR: missing temp/eval_ablation_disc_l8_l16.py" >&2
    exit 1
}

SKIP_FLAG=()
if [ "$SKIP_EXISTING" = "1" ]; then
    SKIP_FLAG=(--skip-existing-runs)
fi

python -u "$VIZ_PY" \
    --code-root "$CODE_ROOT" \
    --output-root "$OUT_ROOT" \
    --dataset "$DATASET" \
    --n-samples "$N_SAMPLES" \
    --variables-to-plot 0 \
    --sampler "$SAMPLER" \
    --num-sampling-steps "$NUM_STEPS" \
    "${SKIP_FLAG[@]}" \
    --runs \
      "window_norm_c128:${CKPT}:${CFG}" \
    "$@"

echo "Finished: $(date)"
echo "output_root=$OUT_ROOT"
