#!/bin/bash
# Apples-to-apples binary flat vs MMPD on ETTh1-capped variate subsets (all 7 sweep datasets).
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_binary_mmpd_subset_compare.sh all            # binary + MMPD (MMPD waits on binary via Slurm)
#   ./submit_binary_mmpd_subset_compare.sh binary         # binary only
#   ./submit_binary_mmpd_subset_compare.sh mmpd           # MMPD only (ckpts must exist)
#   ./submit_binary_mmpd_subset_compare.sh smoke-all
#   ./submit_binary_mmpd_subset_compare.sh smoke-binary
#   ./submit_binary_mmpd_subset_compare.sh smoke-mmpd
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/binary_anchor_stationary_flat_subsets.yaml"
ANCHOR_CONFIG="binary_anchor_stationary_flat_subsets"
DATASETS="ETTh1,ETTh2,exchange_rate,weather,electricity,traffic,solar_Alabama"
MMPD_OUTPUT="results/datasets/$(date +%m-%d)-binary-mmpd-subset-compare"

usage() {
    echo "Usage: $0 {all|binary|mmpd|smoke-all|smoke-binary|smoke-mmpd}" >&2
    exit 1
}

build_binary_dependency() {
    local ids_file="$1"
    if [[ ! -s "$ids_file" ]]; then
        echo "ERROR: no binary job IDs captured in $ids_file" >&2
        exit 1
    fi
    local dep="afterok:$(paste -sd: "$ids_file")"
    echo "$dep"
}

[[ $# -eq 1 ]] || usage

case "$1" in
    all)
        IDS_FILE="$(mktemp)"
        trap 'rm -f "$IDS_FILE"' EXIT
        ./submit_grid.sh \
            --configs "$CONFIG" \
            --datasets "$DATASETS" \
            --job-ids-out "$IDS_FILE"
        DEP="$(build_binary_dependency "$IDS_FILE")"
        ./submit_mmpd_sweep_subset.sh \
            --anchor-config "$ANCHOR_CONFIG" \
            --datasets "$DATASETS" \
            --output-dir "$MMPD_OUTPUT" \
            --dependency "$DEP"
        ;;
    binary)
        ./submit_grid.sh \
            --configs "$CONFIG" \
            --datasets "$DATASETS"
        ;;
    mmpd)
        ./submit_mmpd_sweep_subset.sh \
            --anchor-config "$ANCHOR_CONFIG" \
            --datasets "$DATASETS" \
            --output-dir "$MMPD_OUTPUT"
        ;;
    smoke-all)
        IDS_FILE="$(mktemp)"
        trap 'rm -f "$IDS_FILE"' EXIT
        ./submit_grid.sh --smoke \
            --configs "$CONFIG" \
            --datasets ETTh1 \
            --job-ids-out "$IDS_FILE"
        DEP="$(build_binary_dependency "$IDS_FILE")"
        ./submit_mmpd_sweep_subset.sh --smoke \
            --anchor-config "$ANCHOR_CONFIG" \
            --datasets ETTh1 \
            --dependency "$DEP"
        ;;
    smoke-binary)
        ./submit_grid.sh --smoke \
            --configs "$CONFIG" \
            --datasets ETTh1
        ;;
    smoke-mmpd)
        ./submit_mmpd_sweep_subset.sh --smoke \
            --anchor-config "$ANCHOR_CONFIG" \
            --datasets ETTh1
        ;;
    *)
        usage
        ;;
esac
