#!/bin/bash
# =============================================================================
# Backfill GT vs pred coarse/fine 2D maps for Jul-12 ETTh1 past_native g grid
# onto existing wandb pipeline runs (skip_eval_visualizations was true).
#
# USAGE (Killarney login, $SCRATCH/ts-sandbox):
#   cd "$SCRATCH/ts-sandbox" && git pull
#   ./temp/submit_viz_past_native_etth1_gt_pred_maps_killarney.sh
#   ./temp/submit_viz_past_native_etth1_gt_pred_maps_killarney.sh --g-labels 1.0,4.0 --n-windows 4
#
# Defaults pass --wandb (resume each leaderboard run). Needs WANDB_API_KEY in
# env or repo .env. Local JPGs also under results/viz/past_native_etth1_gt_pred_maps/.
# =============================================================================

set -euo pipefail

if [[ -d "${SCRATCH:-}/ts-sandbox" ]]; then
    REPO="$SCRATCH/ts-sandbox"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/temp/viz_past_native_etth1_gt_pred_maps.py" ]]; then
    REPO="$SLURM_SUBMIT_DIR"
else
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
SCRIPT_DIR="$REPO/temp"
VIZ_PY="temp/viz_past_native_etth1_gt_pred_maps.py"

PY_ARGS=("$@")
if [[ ${#PY_ARGS[@]} -eq 0 ]]; then
    PY_ARGS=(
        --n-windows 3
        --n-vars 3
        --sampler anchor
        --wandb
    )
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    cd "$REPO"
    mkdir -p "$REPO/results/logs"
    echo "Submitting past_native ETTh1 GT/pred maps viz (L40S, 1h) from $REPO ..."
    echo "  python $VIZ_PY ${PY_ARGS[*]}"
    sbatch \
        --chdir="$REPO" \
        --job-name="viz-past-native-maps" \
        --account=aip-boyuwang \
        --time=1:00:00 \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task=4 \
        --mem=40G \
        --output="$REPO/results/logs/viz-past-native-maps-%j.out" \
        --error="$REPO/results/logs/viz-past-native-maps-%j.err" \
        --mail-type=END,FAIL \
        --mail-user=ccao87@uwo.ca \
        "$SCRIPT_DIR/submit_viz_past_native_etth1_gt_pred_maps_killarney.sh" "${PY_ARGS[@]}"
    exit 0
fi

ts() { date +'%d-%H:%M:%S'; }
echo "$(ts) Job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-?} REPO=$REPO"
echo "$(ts) GPU=$(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"

REQ="$REPO/setup/requirements-killarney.txt"
[[ -f "$REQ" ]] || { echo "ERROR: missing $REQ"; exit 1; }
[[ -n "${SLURM_TMPDIR:-}" ]] || { echo "ERROR: SLURM_TMPDIR unset"; exit 1; }
[[ -f "$REPO/$VIZ_PY" ]] || { echo "ERROR: missing $REPO/$VIZ_PY — git pull"; exit 1; }

# Fail fast if sequential coarse/fine best.pt missing.
MISS=0
JOBS=(
    4205389:binary_noise_sched_ablation_past_native_g1p0
    4205429:binary_noise_sched_ablation_past_native_g1p0_s43
    4205433:binary_noise_sched_ablation_past_native_g1p0_s44
    4205393:binary_noise_sched_ablation_past_native_g1p5
    4205397:binary_noise_sched_ablation_past_native_g3p0
    4205401:binary_noise_sched_ablation_past_native_g4p0
    4205405:binary_noise_sched_ablation_past_native_g5p0
    4205417:binary_noise_sched_ablation_past_native_g6p0
    4205409:binary_noise_sched_ablation_past_native_g7p0
    4205421:binary_noise_sched_ablation_past_native_g8p0
    4205425:binary_noise_sched_ablation_past_native_g9p0
    4205413:binary_noise_sched_ablation_past_native_g10p0
)
# If user passed --g-labels, skip full inventory (python will fail per-run).
HAS_G_FILTER=0
for a in "${PY_ARGS[@]}"; do
    [[ "$a" == "--g-labels" ]] && HAS_G_FILTER=1
done
if [[ "$HAS_G_FILTER" -eq 0 ]]; then
    for spec in "${JOBS[@]}"; do
        job="${spec%%:*}"
        stem="${spec#*:}"
        found=0
        for d in "$REPO"/results/ckpts/*-"${job}-ETTh1-${stem}"; do
            [[ -d "$d" ]] || continue
            if [[ -f "$d/ETTh1/coarse/best.pt" && -f "$d/ETTh1/fine/best.pt" ]]; then
                if compgen -G "$d"'/*_patch_guidance*.pt' >/dev/null; then
                    found=1
                    break
                fi
            fi
        done
        if [[ "$found" -eq 0 ]]; then
            echo "ERROR: missing coarse/fine/guidance for job=${job} (${stem})"
            MISS=1
        fi
    done
    [[ "$MISS" -eq 0 ]] || exit 1
fi

module purge 2>/dev/null || true
module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9 2>/dev/null || true

echo "$(ts) [setup] node-local venv from $REQ"
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck source=/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REQ" -q

cd "$REPO"
# Online wandb upload for --wandb (python also loads repo .env).
unset WANDB_MODE || true
export PYTHONUNBUFFERED=1

echo "$(ts) running $VIZ_PY ${PY_ARGS[*]}"
python -u "$REPO/$VIZ_PY" "${PY_ARGS[@]}"
echo "$(ts) done → $REPO/results/viz/past_native_etth1_gt_pred_maps"
