#!/bin/bash
# Train flat-subset configs on datasets not yet in the 7-dataset sweep grid.
#
# Already trained (guidance_150 + grad_accum_200_lr_lo + maskae): ETTh1, ETTh2,
# exchange_rate, weather, electricity, traffic, solar_Alabama.
#
# Prerequisite: grad_accum_* configs reuse synthetic pretrain + iTrans from
# binary_anchor_stationary_flat_subsets_ema099 per dataset. This script submits
# ema099 first, then guidance_150 + lr_lo, then MMPD MaskAE (after lr_lo).
#
# Subsetting: ETTh1-sized dense arrays via configs/binary_anchor_stationary_flat_subsets.yaml
#   ETTm1 -> ETTm1_4v_s3 | ETTm2 -> ETTm2_7v_s4 | PeMS -> PeMS_7v_s1
#   dalia -> dalia_5v_s2 (native LB80/HZ20; MMPD eval overrides 96/96)
#   dynamic -> dynamic_7v_s29 (500k rows in datasets/dynamic/dynamic_500K.csv)
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_flat_subsets_remaining_datasets.sh --smoke
#   ./submit_flat_subsets_remaining_datasets.sh
#   ./submit_flat_subsets_remaining_datasets.sh --datasets ETTm1,illness
#   ./submit_flat_subsets_remaining_datasets.sh --skip-ema099 --prereq-jobs-file /tmp/ema099.jobs
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DONE_DATASETS="ETTh1,ETTh2,exchange_rate,weather,electricity,traffic,solar_Alabama"
ALL_DATASETS="ETTh1,ETTh2,ETTm1,ETTm2,illness,exchange_rate,weather,electricity,traffic,PeMS,solar_Alabama,dalia,dynamic"

EMA099_CFG="configs/binary_anchor_stationary_flat_subsets_ema099.yaml"
GUIDANCE_CFG="configs/binary_anchor_stationary_flat_subsets_grad_accum_guidance_150.yaml"
LR_LO_CFG="configs/binary_anchor_stationary_flat_subsets_grad_accum_200_lr_lo.yaml"
WALL_EMA099="11:00:00"
WALL_BINARY="4:00:00"
DATASETS_CSV=""
SMOKE=0
SKIP_MASKAE=0
SKIP_GUIDANCE=0
SKIP_LR_LO=0
SKIP_EMA099=0
PREREQ_JOBS_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke|--smoke-test) SMOKE=1; shift ;;
        --datasets) DATASETS_CSV="$2"; shift 2 ;;
        --skip-maskae) SKIP_MASKAE=1; shift ;;
        --skip-guidance) SKIP_GUIDANCE=1; shift ;;
        --skip-lr-lo) SKIP_LR_LO=1; shift ;;
        --skip-ema099) SKIP_EMA099=1; shift ;;
        --prereq-jobs-file) PREREQ_JOBS_FILE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$DATASETS_CSV" ]]; then
    IFS=',' read -ra ALL_ARR <<< "$ALL_DATASETS"
    IFS=',' read -ra DONE_ARR <<< "$DONE_DATASETS"
    PENDING=()
    for ds in "${ALL_ARR[@]}"; do
        skip=0
        for done in "${DONE_ARR[@]}"; do
            [[ "$ds" == "$done" ]] && skip=1 && break
        done
        [[ "$skip" -eq 0 ]] && PENDING+=("$ds")
    done
    DATASETS_CSV=$(IFS=,; echo "${PENDING[*]}")
fi

if [[ -z "$DATASETS_CSV" ]]; then
    echo "Nothing to submit: all datasets already in DONE set." >&2
    exit 0
fi

if [[ "$DATASETS_CSV" == *dynamic* ]] && [[ ! -f datasets/dynamic/dynamic_500K.csv ]]; then
    echo "ERROR: dynamic requested but datasets/dynamic/dynamic_500K.csv missing." >&2
    exit 1
fi

echo "Pending datasets: $DATASETS_CSV"
echo ""

slurm_dep_from_file() {
    local file="$1"
    [[ -s "$file" ]] || return 0
    # Slurm: afterok:id1:id2:... (one afterok prefix, not afterok:id1:afterok:id2)
    echo "afterok:$(paste -sd: "$file" | tr -d '[:space:]')"
}

EMA099_JOBS_FILE="$(mktemp)"
LR_LO_JOBS_FILE="$(mktemp)"
trap 'rm -f "$EMA099_JOBS_FILE" "$LR_LO_JOBS_FILE"' EXIT

if [[ "$SMOKE" -eq 1 ]]; then
    echo "=== smoke: ema099 on illness ==="
    ./submit_grid.sh --smoke \
        --configs "$EMA099_CFG" \
        --datasets illness \
        --time 1:00:00
  echo "=== smoke: guidance_150 on illness (needs ema099 ckpt on cluster; local may fail) ==="
    ./submit_grid.sh --smoke \
        --configs "$GUIDANCE_CFG" \
        --datasets illness \
        --time "$WALL_BINARY" || true
    echo "=== smoke: MMPD MaskAE on ETTh1 ==="
    ./submit_mmpd_maskae_flat_subsets_grad_accum_200_lr_lo.sh --smoke-test \
        --datasets ETTh1 \
        --output-dir "results/datasets/$(date +%m-%d)-mmpd-maskae-remaining-smoke"
    exit 0
fi

if [[ "$SKIP_EMA099" -eq 0 ]]; then
    echo "=== [1/3] binary_anchor_stationary_flat_subsets_ema099 (pretrain + iTrans reuse source) ==="
    ./submit_grid.sh \
        --configs "$EMA099_CFG" \
        --datasets "$DATASETS_CSV" \
        --time "$WALL_EMA099" \
        --job-ids-out "$EMA099_JOBS_FILE"
elif [[ -n "$PREREQ_JOBS_FILE" ]]; then
    cp "$PREREQ_JOBS_FILE" "$EMA099_JOBS_FILE"
fi

EMA_DEP="$(slurm_dep_from_file "$EMA099_JOBS_FILE")"
DEP_ARGS=()
if [[ -n "$EMA_DEP" ]]; then
    DEP_ARGS=(--dependency "$EMA_DEP")
    echo "ema099 dependency: $EMA_DEP"
fi

if [[ "$SKIP_GUIDANCE" -eq 0 ]]; then
    echo "=== [2a/3] binary guidance accum 1.5x ==="
    ./submit_grid.sh \
        --configs "$GUIDANCE_CFG" \
        --datasets "$DATASETS_CSV" \
        --time "$WALL_BINARY" \
        "${DEP_ARGS[@]}"
fi

if [[ "$SKIP_LR_LO" -eq 0 ]]; then
    echo "=== [2b/3] binary grad_accum_200_lr_lo (MaskAE anchor) ==="
    ./submit_grid.sh \
        --configs "$LR_LO_CFG" \
        --datasets "$DATASETS_CSV" \
        --time "$WALL_BINARY" \
        --job-ids-out "$LR_LO_JOBS_FILE" \
        "${DEP_ARGS[@]}"
fi

if [[ "$SKIP_MASKAE" -eq 0 ]]; then
    LR_DEP="$(slurm_dep_from_file "$LR_LO_JOBS_FILE")"
    if [[ "$SKIP_LR_LO" -ne 0 ]]; then
        echo "WARN: --skip-lr-lo; MaskAE submitted without lr_lo dependency." >&2
        ./submit_mmpd_maskae_flat_subsets_grad_accum_200_lr_lo.sh \
            --datasets "$DATASETS_CSV" \
            --output-dir "results/datasets/$(date +%m-%d)-mmpd-maskae-remaining-subset"
    else
        echo "=== [3/3] MMPD MaskAE (after lr_lo: ${LR_DEP:-pending}) ==="
        MASKAE_DEP_ARGS=()
        [[ -n "$LR_DEP" ]] && MASKAE_DEP_ARGS=(--dependency "$LR_DEP")
        ./submit_mmpd_maskae_flat_subsets_grad_accum_200_lr_lo.sh \
            --datasets "$DATASETS_CSV" \
            --output-dir "results/datasets/$(date +%m-%d)-mmpd-maskae-remaining-subset" \
            "${MASKAE_DEP_ARGS[@]}"
    fi
fi

echo "Done."
