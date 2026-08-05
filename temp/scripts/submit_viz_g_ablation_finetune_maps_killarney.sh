#!/bin/bash
# =============================================================================
# Viz GT vs pred coarse/fine maps for ETTh2 g=1..10 4ep finetune stubs.
#
# USAGE (Killarney login, $SCRATCH/ts-sandbox):
#   cd "$SCRATCH/ts-sandbox" && git pull
#   ./temp/scripts/submit_viz_g_ablation_finetune_maps_killarney.sh
#   ./temp/scripts/submit_viz_g_ablation_finetune_maps_killarney.sh --n-windows 4 --g-values 1-10
#
# Outputs:
#   results/viz/g_ablation_finetune_maps_ETTh2/g{N}p0/winXXXX/*.jpg
#   results/viz/g_ablation_finetune_maps_ETTh2/compare/winXXXX_var0_g1to10.jpg
# =============================================================================

set -euo pipefail

if [[ -d "${SCRATCH:-}/ts-sandbox" ]]; then
    REPO="$SCRATCH/ts-sandbox"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/temp/scripts/viz_g_ablation_finetune_maps.py" ]]; then
    REPO="$SLURM_SUBMIT_DIR"
else
    REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
fi
SCRIPT_DIR="$REPO/temp/scripts"
VIZ_PY="temp/scripts/viz_g_ablation_finetune_maps.py"

DATASET="${VIZ_DATASET:-ETTh2}"
PY_ARGS=("$@")
if [[ ${#PY_ARGS[@]} -eq 0 ]]; then
    PY_ARGS=(
        --dataset "$DATASET"
        --g-values 1-10
        --n-windows 3
        --n-vars-plot 3
        --sampler anchor
        --compare-var 0
    )
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    cd "$REPO"
    mkdir -p "$REPO/results/logs"
    echo "Submitting g-ablation finetune maps viz (L40S, 1h) from $REPO ..."
    echo "  python $VIZ_PY ${PY_ARGS[*]}"
    sbatch \
        --chdir="$REPO" \
        --job-name="viz-g-ablation-maps" \
        --account=aip-boyuwang \
        --time=1:00:00 \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task=4 \
        --mem=40G \
        --output="$REPO/results/logs/viz-g-ablation-maps-%j.out" \
        --error="$REPO/results/logs/viz-g-ablation-maps-%j.err" \
        --mail-type=END,FAIL \
        --mail-user=ccao87@uwo.ca \
        "$SCRIPT_DIR/submit_viz_g_ablation_finetune_maps_killarney.sh" "${PY_ARGS[@]}"
    exit 0
fi

ts() { date +'%d-%H:%M:%S'; }
echo "$(ts) Job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-?} REPO=$REPO"
echo "$(ts) GPU=$(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"

REQ="$REPO/setup/requirements-killarney.txt"
[[ -f "$REQ" ]] || { echo "ERROR: missing $REQ"; exit 1; }
[[ -n "${SLURM_TMPDIR:-}" ]] || { echo "ERROR: SLURM_TMPDIR unset"; exit 1; }
[[ -f "$REPO/$VIZ_PY" ]] || { echo "ERROR: missing $REPO/$VIZ_PY — git pull"; exit 1; }

# Fail fast if best.pt missing (common when only metadata was rsynced).
MISS=0
for g in 1 2 3 4 5 6 7 8 9 10; do
    stem="binary_noise_sched_ablation_vertical_dual_g${g}p0"
    found=0
    for d in "$REPO"/results/ckpts/*-"${DATASET}"-"${stem}"; do
        [[ -d "$d" ]] || continue
        if compgen -G "$d"'/*/vertical_dual/best.pt' >/dev/null; then
            found=1
            break
        fi
    done
    if [[ "$found" -eq 0 ]]; then
        echo "ERROR: no best.pt for g=${g} (*-${DATASET}-${stem})"
        MISS=1
    fi
done
[[ "$MISS" -eq 0 ]] || exit 1

module purge 2>/dev/null || true
module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9 2>/dev/null || true

echo "$(ts) [setup] node-local venv from $REQ"
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck source=/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REQ" -q

cd "$REPO"
echo "$(ts) running $VIZ_PY ${PY_ARGS[*]}"
python -u "$REPO/$VIZ_PY" "${PY_ARGS[@]}"
echo "$(ts) done → $REPO/results/viz/g_ablation_finetune_maps_${DATASET}"
