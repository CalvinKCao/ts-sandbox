#!/bin/bash
# Resubmit missing 05-28 h128 cross-var eval partials (auto-detect from scratch).
#
# USAGE (Killarney login, $SCRATCH/ts-sandbox):
#   ./utils/resubmit_05_28_crossvar_gaps.sh --dry-run
#   ./utils/resubmit_05_28_crossvar_gaps.sh
#
# After partial count hits 40:
#   ./utils/submit_crossvar_merge_waiter.sh --run-stem 05-28-bin-h128-crossvar-ablation --cancel-pending

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${SCRATCH:-}/ts-sandbox"
[[ -d "$REPO" ]] || REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO"

DRY=0
AUTO=1
TRAIN_WALL="${TRAIN_WALL:-1-23:00:00}"
EVAL_WALL="${EVAL_WALL:-12:00:00}"
EXCLUDE_NODES="${EXCLUDE_NODES:-kn060,kn078}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY=1; shift ;;
        --auto) AUTO=1; shift ;;
        --minimal) AUTO=0; shift ;;
        --train-wall) TRAIN_WALL="$2"; shift 2 ;;
        --eval-wall) EVAL_WALL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

RUN_STEM="05-28-bin-h128-crossvar-ablation"
DATE_TAG="05-28"
EVAL_OUT="$REPO/results/datasets/${RUN_STEM}"
LOG_DIR="$REPO/results/logs/${RUN_STEM}"
JOB_DIR="$REPO/results/jobs/${RUN_STEM}"
TRAIN_SCRIPT="$REPO/slurm_binary_anchor_92d3.sh"
EVAL_SCRIPT="$REPO/utils/eval_binary_anchor_variants.py"
MERGE_INVOKE="$JOB_DIR/merge_invoke.sh"

mkdir -p "$LOG_DIR" "$JOB_DIR"
chmod +x "$TRAIN_SCRIPT" "$EVAL_SCRIPT" "$SCRIPT_DIR/write_crossvar_eval_preamble.sh"
"$SCRIPT_DIR/write_crossvar_eval_preamble.sh" "$JOB_DIR"
"$SCRIPT_DIR/write_crossvar_merge_invoke.sh" "$RUN_STEM" "$REPO" "$DATE_TAG"
export REPO

declare -A SPEC_SUBSET SPEC_NVARS SPEC_INDICES SPEC_EXTRA SPEC_STRIDE
# dataset|subset_id|nvars|indices|extra_flags|stride
_SPEC_ROWS=(
    "ETTh1|ETTh1|7|| |1"
    "ETTh2|ETTh2|7|| |1"
    "exchange_rate|exchange_rate|8|| |1"
    "illness|illness|7|| |1"
    "weather|weather_9v|9|0,1,2,3,4,5,6,7,8| |1"
    "ETTm1|ETTm1_2x|7|| |2"
    "ETTm2|ETTm2_2x|7|| |2"
    "electricity|electricity_18v|18|0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17| |1"
    "traffic|traffic_27v|27|0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26| |1"
    "dalia|dalia_5v|5|0,1,2,3,4| |2"
)
for row in "${_SPEC_ROWS[@]}"; do
    IFS='|' read -r ds subset nvars indices _extra stride <<< "$row"
    SPEC_SUBSET["$ds"]="$subset"
    SPEC_NVARS["$ds"]="$nvars"
    SPEC_INDICES["$ds"]="$indices"
    SPEC_STRIDE["$ds"]="$stride"
done

declare -A VARIANT_EXTRA
VARIANT_EXTRA[exp1-datasetnorm]="--disable-window-normalization"
VARIANT_EXTRA[exp2-tokens-only]="--zero-guidance-forecast"
VARIANT_EXTRA[exp3-datasetnorm-tokens-only]="--disable-window-normalization --zero-guidance-forecast"
VARIANT_EXTRA[exp4-baseline-xvar]=""

model_id_to_variant() {
    case "$1" in
        binary_exp1_datasetnorm) echo "exp1-datasetnorm" ;;
        binary_exp2_tokens_only) echo "exp2-tokens-only" ;;
        binary_exp3_datasetnorm_tokens_only) echo "exp3-datasetnorm-tokens-only" ;;
        binary_exp4_baseline_xvar) echo "exp4-baseline-xvar" ;;
        *) return 1 ;;
    esac
}

parse_partial_basename() {
  local base="$1"
  local ds variant model_id
  for ds in traffic electricity exchange_rate weather ETTh1 ETTh2 ETTm1 ETTm2 illness dalia; do
    if [[ "$base" == "${ds}_binary_"* ]]; then
      model_id="binary_${base#${ds}_binary_}"
      variant=$(model_id_to_variant "$model_id") || return 1
      printf '%s %s\n' "$ds" "$variant"
      return 0
    fi
  done
  return 1
}

train_ready() {
    local ckpt_dir="$1"
    [[ -f "${ckpt_dir}/diff_hp_best.pt" ]] || [[ -f "${ckpt_dir}/training_manifest.json" ]]
}

submit_train() {
    local variant="$1" subset="$2" dataset="$3" nvars="$4" indices="$5" extra="$6" stride="$7"
    local stem="${DATE_TAG}-bin-h128-xvar-${variant}-${subset,,}"
    local mem="60G" cpus=8
    [[ "$nvars" -ge 16 ]] && mem="72G"

    local -a args=(
        --dataset "$dataset"
        --n-variates "$nvars"
        --subset-id "$subset"
        --image-height 128
        --resume
        --run-stem "$stem"
        --walltime "$TRAIN_WALL"
        --wandb-project ts-sandbox-binary-h128-crossvar-ablation
    )
    [[ -n "$indices" ]] && args+=(--variate-indices "$indices")
    if [[ "$stride" -gt 1 ]]; then
        args+=(--window-stride "$stride")
    fi
    if [[ -n "$extra" ]]; then
        read -r -a extra_a <<< "$extra"
        args+=("${extra_a[@]}")
    fi
    if [[ -n "${VARIANT_EXTRA[$variant]:-}" ]]; then
        read -r -a vextra <<< "${VARIANT_EXTRA[$variant]}"
        args+=("${vextra[@]}")
    fi

    echo "TRAIN ${variant}/${subset} stem=${stem} wall=${TRAIN_WALL}"
    if [[ "$DRY" -eq 1 ]]; then
        echo "  -> sbatch ... $TRAIN_SCRIPT ${args[*]}"
        echo "__DRY_TRAIN__"
        return 0
    fi
    local -a sbatch_args=(
        --parsable
        --job-name="h128-${variant}-${subset,,}"
        --account=aip-boyuwang
        --time="$TRAIN_WALL"
        --nodes=1
        --gres=gpu:l40s:1
        --cpus-per-task="$cpus"
        --mem="$mem"
        --output=/dev/null
        --error=/dev/null
        --mail-type=FAIL
        --mail-user=ccao87@uwo.ca
    )
    [[ -n "$EXCLUDE_NODES" ]] && sbatch_args+=(--exclude="$EXCLUDE_NODES")
    sbatch "${sbatch_args[@]}" "$TRAIN_SCRIPT" "${args[@]}"
}

submit_eval() {
    local variant="$1" subset="$2" dataset="$3" train_jid="$4"
    local stem="${DATE_TAG}-bin-h128-xvar-${variant}-${subset,,}"
    local ckpt="$REPO/results/ckpts/${stem}"
    local worker="$JOB_DIR/eval-resubmit-${variant}-${subset}.sh"
    local eval_extra=""
    [[ "$dataset" == dalia ]] && eval_extra="--lookback 80 --horizon 20"

    cat > "$worker" <<EOF
#!/bin/bash
source "$JOB_DIR/eval_preamble.sh"
python -u "$EVAL_SCRIPT" \\
  --phase eval \\
  --variant-root "${variant}=${ckpt}" \\
  --datasets "$dataset" \\
  --output-dir "$EVAL_OUT" \\
  --test-fraction 0.5 \\
  --sample-num 9 \\
  --num-sampling-steps 20 \\
  --gmm-components 9 \\
  --anchor-batch-size 16 \\
  --texture-per-sample \\
  $eval_extra
EOF
    chmod +x "$worker"

    local dep=""
    [[ -n "$train_jid" && "$train_jid" != "__DRY_TRAIN__" ]] && dep="afterok:${train_jid}"

    echo "EVAL  ${variant}/${subset} dep=${dep:-none} wall=${EVAL_WALL}"
    if [[ "$DRY" -eq 1 ]]; then
        echo "  -> sbatch $worker"
        return 0
    fi
    local -a sbatch_args=(
        --parsable
        --job-name="eval-${variant}-${subset,,}"
        --account=aip-boyuwang
        --nodes=1
        --gres=gpu:l40s:1
        --cpus-per-task=8
        --mem=60G
        --time="$EVAL_WALL"
        --output="$LOG_DIR/eval-${variant}-${subset}-%j.out"
        --error="$LOG_DIR/eval-${variant}-${subset}-%j.err"
        --mail-type=FAIL
        --mail-user=ccao87@uwo.ca
    )
    [[ -n "$EXCLUDE_NODES" ]] && sbatch_args+=(--exclude="$EXCLUDE_NODES")
    [[ -n "$dep" ]] && sbatch_args+=(--dependency="$dep")
    sbatch "${sbatch_args[@]}" "$worker"
}

submit_cell() {
    local dataset="$1" variant="$2"
    local subset="${SPEC_SUBSET[$dataset]}"
    local nvars="${SPEC_NVARS[$dataset]}"
    local indices="${SPEC_INDICES[$dataset]:-}"
    local stride="${SPEC_STRIDE[$dataset]:-1}"
    local stem="${DATE_TAG}-bin-h128-xvar-${variant}-${subset,,}"
    local ckpt_dir="$REPO/results/ckpts/${stem}"

  if train_ready "$ckpt_dir"; then
        submit_eval "$variant" "$subset" "$dataset" ""
    else
        tid=$(submit_train "$variant" "$subset" "$dataset" "$nvars" "$indices" "" "$stride")
        submit_eval "$variant" "$subset" "$dataset" "$tid"
    fi
}

CELLS=()
if [[ "$AUTO" -eq 1 ]]; then
    if [[ ! -f "$MERGE_INVOKE" ]]; then
        echo "ERROR: missing $MERGE_INVOKE" >&2
        exit 1
    fi
    # shellcheck source=/dev/null
    source "$MERGE_INVOKE"
    missing_file="$(mktemp)"
    set +e
    python -u "$EVAL_SCRIPT" --phase check-partials "${CHECK_ARGS[@]}" >"$missing_file" 2>/dev/null
    check_st=$?
    set -e
    if [[ "$check_st" -eq 0 ]]; then
        rm -f "$missing_file"
        echo "All 40 partials present; nothing to resubmit."
        exit 0
    fi
    mapfile -t missing < <(sed 's|^partials/||; s|\.json$||' "$missing_file")
    rm -f "$missing_file"
    if [[ "${#missing[@]}" -eq 0 ]]; then
        echo "ERROR: check-partials failed but listed no paths" >&2
        exit 1
    fi
    echo "Auto-detected ${#missing[@]} missing partial(s):"
    for base in "${missing[@]}"; do
        [[ -z "$base" ]] && continue
        if ! read -r dataset variant < <(parse_partial_basename "$base"); then
            echo "WARN: skip unparseable partial: $base" >&2
            continue
        fi
        echo "  - ${dataset} / ${variant}"
        CELLS+=("${dataset}|${variant}")
    done
else
    CELLS=(
        "illness|exp2-tokens-only"
        "illness|exp3-datasetnorm-tokens-only"
        "illness|exp4-baseline-xvar"
        "dalia|exp3-datasetnorm-tokens-only"
    )
fi

echo "Repo: $REPO"
echo "Train wall: $TRAIN_WALL   Eval wall: $EVAL_WALL"
echo "Exclude nodes: ${EXCLUDE_NODES:-none}"
present=$(find "$EVAL_OUT/partials" -maxdepth 1 -name '*_binary_*.json' 2>/dev/null | wc -l)
echo "Partials now: ${present:-0} / 40"
echo ""

for row in "${CELLS[@]}"; do
    IFS='|' read -r dataset variant <<< "$row"
    submit_cell "$dataset" "$variant"
done

echo ""
echo "Verify: ls $EVAL_OUT/partials/*_binary_*.json 2>/dev/null | wc -l  # expect 40"
echo "Then: ./utils/submit_crossvar_merge_waiter.sh --run-stem $RUN_STEM --cancel-pending"
