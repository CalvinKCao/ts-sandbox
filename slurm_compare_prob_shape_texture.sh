#!/bin/bash
# One probabilistic sample per test window: texture metrics vs GT (MMPD vs anchor models).
#
# USAGE (Killarney login, repo root):
#   ./slurm_compare_prob_shape_texture.sh
#   MMPD_SHARED=./results/datasets/05-26-0688-mmpd-anchor-eval \
#     SHAPE_OUT=./results/datasets/05-26-prob-shape-texture \
#     ./slurm_compare_prob_shape_texture.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEED=2026
SHAPE_OUT="${SHAPE_OUT:-./results/datasets/05-26-prob-shape-texture}"
MMPD_SHARED="${MMPD_SHARED:-./results/datasets/05-26-0688-mmpd-anchor-eval}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    sbatch \
        --job-name=prob-shape-tex \
        --account=aip-boyuwang \
        --time=8:00:00 \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task=8 \
        --mem=60G \
        --output=/dev/null \
        --error=/dev/null \
        --mail-type=FAIL \
        --mail-user=ccao87@uwo.ca \
        "$SCRIPT_DIR/slurm_compare_prob_shape_texture.sh"
    exit 0
fi

cd "$SLURM_SUBMIT_DIR"
REPO="$SLURM_SUBMIT_DIR"
SHAPE_OUT="$(realpath -m "$SHAPE_OUT")"
MMPD_SHARED="$(realpath -m "$MMPD_SHARED")"
LOG="./results/logs/prob-shape-texture-${SLURM_JOB_ID}.log"
mkdir -p "$(dirname "$LOG")" "$SHAPE_OUT"
exec >>"$LOG" 2>&1

module purge || true
module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index \
    'torch==2.11.0+computecanada' numpy pandas scipy scikit-learn tqdm einops \
    -q
export PYTHONUNBUFFERED=1

ROOT_ARGS=""
declare -A GAUSS=(
    [ETTh1]=05-26-3037-gauss-anchor-etth1
    [ETTh2]=05-26-3038-gauss-anchor-etth2
    [ETTm1]=05-26-9040-gauss-anchor-ettm1
    [ETTm2]=05-26-9041-gauss-anchor-ettm2
    [exchange_rate]=05-26-3257-gauss-anchor-exchange_rate
    [illness]=05-26-9043-gauss-anchor-illness
)
declare -A BIN=(
    [ETTh1]=05-26-9033-binary-anchor-etth1
    [ETTh2]=05-26-9034-binary-anchor-etth2
    [ETTm1]=05-26-9035-binary-anchor-ettm1
    [ETTm2]=05-26-9036-binary-anchor-ettm2
    [exchange_rate]=05-26-9038-binary-anchor-exchange_rate
    [illness]=05-26-9039-binary-anchor-illness
)
for ds in ETTh1 ETTh2 ETTm1 ETTm2 exchange_rate illness; do
    [[ -d "$REPO/results/ckpts/${GAUSS[$ds]}" ]] && ROOT_ARGS+=" --anchor-root $REPO/results/ckpts/${GAUSS[$ds]}"
    [[ -d "$REPO/results/ckpts/${BIN[$ds]}" ]] && ROOT_ARGS+=" --binary-anchor-root $REPO/results/ckpts/${BIN[$ds]}"
done

python -u "$REPO/utils/compare_prob_shape_texture.py" \
    --output-dir "$SHAPE_OUT" \
    --indices-dir "$MMPD_SHARED/raw" \
    --mmpd-output-root "$MMPD_SHARED/mmpd_out" \
    --mmpd-raw-dir "$MMPD_SHARED/raw" \
    --mmpd-raw-fallback "$MMPD_SHARED/raw" \
    --ckpt-base "$REPO/results/ckpts" \
    --mmpd-repo "$REPO/temp/MMPD" \
    --mmpd-data-dir "$REPO/temp/mmpd_datasets" \
    --seed "$SEED" \
    --no-update-mmpd \
    --test-fraction 0.5 \
    --num-sampling-steps 20 \
    $ROOT_ARGS

echo "Done: $(date)"
echo "Results: $SHAPE_OUT/shape_metrics.json"
