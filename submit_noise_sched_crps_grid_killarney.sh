#!/bin/bash
# CRPS-targeted noise-schedule g calibration (336/720_uncompressed).
# Reuses synthetic pretrain + patch guidance; short coarse+fine (4 epochs) + staged_eval.
#
# Modes:
#   --mode refine       g∈{6,8,9} on ETTh1,exchange_rate (fine grid around g=7 peak)
#   --mode confirm      seed=43 at current recommended g (ETTh1/exchange g=7; traffic/elec g=3)
#   --mode remaining    full pipeline on untouched datasets (excl. dalia): endpoint + coarse
#                       grid incl. fine neighbors {6,8,9} + g=1.0 seed floor
#   --mode confirm_from_summary
#                       seed=43 at recommended_g from reports/noise_sched_crps_grid/summary.json
#   --mode extended     g∈{4,5,7,10} on prior four (legacy)
#   --mode electricity  full coarse g on electricity only (legacy)
#   --mode seeds        g=1.0 seed replicates s43/s44 (legacy)
#   --mode all          legacy electricity+extended+seeds
#   --mode v2           refine + confirm(current four) + remaining  (default for this campaign)
#
# USAGE (Killarney login, from $SCRATCH/ts-sandbox):
#   ./submit_noise_sched_crps_grid_killarney.sh --smoke-test --mode refine
#   ./submit_noise_sched_crps_grid_killarney.sh --mode v2
#   ./submit_noise_sched_crps_grid_killarney.sh --mode remaining --datasets ETTh2,weather
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="v2"
WALL_TIME="3:00:00"
SMOKE=0
RESUME=0
DATASETS_OVERRIDE=""
SUMMARY_JSON="${SCRIPT_DIR}/reports/noise_sched_crps_grid/summary.json"

# Datasets already through CRPS/anchor calibration
DONE_DS="ETTh1,traffic,exchange_rate,electricity"
# Untouched for this campaign: no other ETT, no illness, no dalia.
# weather had proxy-only diagnosis → treat as untouched for CRPS/anchor.
REMAINING_DS="weather,PeMS,solar_Alabama"

G_BASE=(
    configs/binary_noise_sched_ablation_elec_unc_g1p0.yaml
    configs/binary_noise_sched_ablation_elec_unc_g1p5.yaml
    configs/binary_noise_sched_ablation_elec_unc_g3p0.yaml
)
G_EXT=(
    configs/binary_noise_sched_ablation_elec_unc_g4p0.yaml
    configs/binary_noise_sched_ablation_elec_unc_g5p0.yaml
    configs/binary_noise_sched_ablation_elec_unc_g7p0.yaml
    configs/binary_noise_sched_ablation_elec_unc_g10p0.yaml
)
G_FINE=(
    configs/binary_noise_sched_ablation_elec_unc_g6p0.yaml
    configs/binary_noise_sched_ablation_elec_unc_g8p0.yaml
    configs/binary_noise_sched_ablation_elec_unc_g9p0.yaml
)
# Coarse + fine neighbors in one shot for remaining datasets (no second round wait)
G_FULL_PLUS_FINE=("${G_BASE[@]}" "${G_EXT[@]}" "${G_FINE[@]}")
G_FULL=("${G_BASE[@]}" "${G_EXT[@]}")
G_SEEDS=(
    configs/binary_noise_sched_ablation_elec_unc_g1p0_s43.yaml
    configs/binary_noise_sched_ablation_elec_unc_g1p0_s44.yaml
)

# Current recommended g → confirm seed=43 configs (Goal 2; refine may update ETTh1/exchange)
CONFIRM_G7=(configs/binary_noise_sched_ablation_elec_unc_g7p0_s43.yaml)
CONFIRM_G3=(configs/binary_noise_sched_ablation_elec_unc_g3p0_s43.yaml)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --smoke-test|--smoke) SMOKE=1; shift ;;
        --time) WALL_TIME="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        --datasets) DATASETS_OVERRIDE="$2"; shift 2 ;;
        --summary-json) SUMMARY_JSON="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

submit_one() {
    local config_csv="$1"
    local datasets="$2"
    local label="$3"
    local args=(
        --configs "$config_csv"
        --datasets "$datasets"
        --gpu l40s
        --time "$WALL_TIME"
        --wandb-project ts-sandbox-leaderboard
    )
    if [[ "$SMOKE" -eq 1 ]]; then
        args=(
            --configs "$config_csv"
            --datasets "$datasets"
            --gpu l40s
            --time "0:45:00"
            --smoke-test
            --wandb-project ts-sandbox-leaderboard
        )
    fi
    if [[ "$RESUME" -eq 1 ]]; then
        args+=(--resume)
    fi
    echo "=== submit [$label]: datasets=$datasets configs=$config_csv ==="
    ./test_submit.sh "${args[@]}"
}

submit_confirm_from_summary() {
    local summary="$1"
    if [[ ! -f "$summary" ]]; then
        echo "ERROR: summary json missing: $summary" >&2
        echo "Run: python utils/analyze_noise_sched_crps_grid.py --out-dir reports/noise_sched_crps_grid" >&2
        exit 1
    fi
    # Emit lines: dataset<TAB>config_path
    local pairs
    pairs="$(python3 - "$summary" <<'PY'
import json, sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text())
tags = {
    1.0: "g1p0", 1.5: "g1p5", 3.0: "g3p0", 4.0: "g4p0", 5.0: "g5p0",
    6.0: "g6p0", 7.0: "g7p0", 8.0: "g8p0", 9.0: "g9p0", 10.0: "g10p0",
}
for row in summary.get("summary_rows") or []:
    ds = row["dataset"]
    g = float(row["recommended_g"])
    tag = tags.get(g) or tags[min(tags, key=lambda x: abs(x - g))]
    cfg = f"configs/binary_noise_sched_ablation_elec_unc_{tag}_s43.yaml"
    print(f"{ds}\t{cfg}\t{g}")
PY
)"
    if [[ -z "$pairs" ]]; then
        echo "ERROR: no summary_rows in $summary" >&2
        exit 1
    fi
    # Group by config to batch datasets sharing the same recommended g
    declare -A CFG_TO_DS
    while IFS=$'\t' read -r ds cfg g; do
        [[ -z "$ds" ]] && continue
        if [[ -n "${CFG_TO_DS[$cfg]:-}" ]]; then
            CFG_TO_DS[$cfg]="${CFG_TO_DS[$cfg]},${ds}"
        else
            CFG_TO_DS[$cfg]="$ds"
        fi
        echo "  confirm plan: $ds @ g=$g → $cfg"
    done <<< "$pairs"
    for cfg in "${!CFG_TO_DS[@]}"; do
        submit_one "$cfg" "${CFG_TO_DS[$cfg]}" "confirm s43 via summary ($cfg)"
    done
}

submit_endpoint_remaining() {
    local ds="${1:-$REMAINING_DS}"
    echo "=== endpoint diagnostic (t=T bit-agreement) for: $ds ==="
    # One job per dataset keeps wall short and isolates failures
    IFS=',' read -ra ARR <<< "$ds"
    for d in "${ARR[@]}"; do
        [[ "$d" == "dalia" ]] && continue
        ./submit_diagnose_noise_schedule_killarney.sh --datasets "$d" --time "0:15:00"
    done
}

case "$MODE" in
    refine)
        DS="${DATASETS_OVERRIDE:-ETTh1,exchange_rate}"
        submit_one "$(IFS=,; echo "${G_FINE[*]}")" "$DS" "fine grid g=6/8/9"
        ;;
    confirm)
        # Goal 2: current recommendations (pre-refine). Override datasets splits if needed.
        if [[ -n "$DATASETS_OVERRIDE" ]]; then
            echo "NOTE: --datasets with --mode confirm ignored for split; use confirm_from_summary" >&2
        fi
        submit_one "$(IFS=,; echo "${CONFIRM_G7[*]}")" "ETTh1,exchange_rate" "confirm s43 @ g=7"
        submit_one "$(IFS=,; echo "${CONFIRM_G3[*]}")" "traffic,electricity" "confirm s43 @ g=3"
        ;;
    remaining)
        DS="${DATASETS_OVERRIDE:-$REMAINING_DS}"
        submit_endpoint_remaining "$DS"
        submit_one "$(IFS=,; echo "${G_FULL_PLUS_FINE[*]}")" "$DS" "remaining full g grid + fine"
        submit_one "$(IFS=,; echo "${G_SEEDS[*]}")" "$DS" "remaining g=1.0 seed floor"
        ;;
    confirm_from_summary)
        submit_confirm_from_summary "$SUMMARY_JSON"
        ;;
    extended)
        DS="${DATASETS_OVERRIDE:-ETTh1,traffic,exchange_rate,electricity}"
        submit_one "$(IFS=,; echo "${G_EXT[*]}")" "$DS" "extended g=4/5/7/10"
        ;;
    electricity)
        DS="${DATASETS_OVERRIDE:-electricity}"
        submit_one "$(IFS=,; echo "${G_FULL[*]}")" "$DS" "electricity full g grid"
        ;;
    seeds)
        DS="${DATASETS_OVERRIDE:-ETTh1,traffic,exchange_rate,electricity}"
        submit_one "$(IFS=,; echo "${G_SEEDS[*]}")" "$DS" "g=1.0 seed replicates"
        ;;
    all)
        submit_one "$(IFS=,; echo "${G_FULL[*]}")" \
            "${DATASETS_OVERRIDE:-electricity}" \
            "electricity full g grid"
        if [[ -n "$DATASETS_OVERRIDE" ]]; then
            submit_one "$(IFS=,; echo "${G_EXT[*]}")" "$DATASETS_OVERRIDE" "extended g"
        else
            submit_one "$(IFS=,; echo "${G_EXT[*]}")" \
                "ETTh1,traffic,exchange_rate" \
                "extended g on prior datasets"
        fi
        submit_one "$(IFS=,; echo "${G_SEEDS[*]}")" \
            "${DATASETS_OVERRIDE:-ETTh1,traffic,exchange_rate,electricity}" \
            "g=1.0 seed replicates"
        ;;
    v2)
        # Goal 1 + 2 + 3 (remaining). Confirmation for remaining → confirm_from_summary after pull.
        # If override is set, only run that slice (caller controls); else full campaign.
        if [[ -n "$DATASETS_OVERRIDE" ]]; then
            echo "v2 with --datasets override: treating as remaining-style submit for: $DATASETS_OVERRIDE"
            submit_endpoint_remaining "$DATASETS_OVERRIDE"
            submit_one "$(IFS=,; echo "${G_FULL_PLUS_FINE[*]}")" "$DATASETS_OVERRIDE" "override full+fine grid"
            submit_one "$(IFS=,; echo "${G_SEEDS[*]}")" "$DATASETS_OVERRIDE" "override seed floor"
        else
            submit_one "$(IFS=,; echo "${G_FINE[*]}")" "ETTh1,exchange_rate" "fine grid g=6/8/9"
            submit_one "$(IFS=,; echo "${CONFIRM_G7[*]}")" "ETTh1,exchange_rate" "confirm s43 @ g=7"
            submit_one "$(IFS=,; echo "${CONFIRM_G3[*]}")" "traffic,electricity" "confirm s43 @ g=3"
            submit_endpoint_remaining "$REMAINING_DS"
            submit_one "$(IFS=,; echo "${G_FULL_PLUS_FINE[*]}")" "$REMAINING_DS" "remaining full+fine grid"
            submit_one "$(IFS=,; echo "${G_SEEDS[*]}")" "$REMAINING_DS" "remaining g=1.0 seed floor"
        fi
        echo ""
        echo "After results land + analyze, confirm remaining (+ any refine-shifted g) with:"
        echo "  python utils/analyze_noise_sched_crps_grid.py --out-dir reports/noise_sched_crps_grid"
        echo "  ./submit_noise_sched_crps_grid_killarney.sh --mode confirm_from_summary"
        ;;
    *)
        echo "Unknown --mode $MODE (expected: refine|confirm|remaining|confirm_from_summary|extended|electricity|seeds|all|v2)" >&2
        exit 1
        ;;
esac
