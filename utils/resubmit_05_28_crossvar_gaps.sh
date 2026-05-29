#!/bin/bash
# Resubmit only the 05-28 h128 cross-var cells still missing on scratch (per merge-wait log).
#
# USAGE (Killarney login, $SCRATCH/ts-sandbox):
#   ./utils/resubmit_05_28_crossvar_gaps.sh
#   ./utils/resubmit_05_28_crossvar_gaps.sh --dry-run
#
# After trains/evals finish:
#   ./utils/submit_crossvar_merge_waiter.sh --run-stem 05-28-bin-h128-crossvar-ablation --cancel-pending

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${SCRATCH:-}/ts-sandbox"
[[ -d "$REPO" ]] || REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO"

DRY=0
TRAIN_WALL="${TRAIN_WALL:-1-23:00:00}"
EVAL_WALL="${EVAL_WALL:-12:00:00}"
EXCLUDE_NODES="${EXCLUDE_NODES:-kn060,kn078}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY=1; shift ;;
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

mkdir -p "$LOG_DIR" "$JOB_DIR"
chmod +x "$TRAIN_SCRIPT" "$EVAL_SCRIPT" "$SCRIPT_DIR/write_crossvar_eval_preamble.sh"
"$SCRIPT_DIR/write_crossvar_eval_preamble.sh" "$JOB_DIR"
"$SCRIPT_DIR/write_crossvar_merge_invoke.sh" "$RUN_STEM" "$REPO" "$DATE_TAG"
export REPO

# mode|variant|subset_id|dataset|nvars|indices|extra_flags|stride
# illness exp2-4: train died at venv (kn078); dalia exp3: train ok, eval failed (kn060).
CELLS=(
    "train+eval|exp2-tokens-only|illness|illness|7||--zero-guidance-forecast|1"
    "train+eval|exp3-datasetnorm-tokens-only|illness|illness|7||--disable-window-normalization --zero-guidance-forecast|1"
    "train+eval|exp4-baseline-xvar|illness|illness|7|||1"
    "eval-only|exp3-datasetnorm-tokens-only|dalia_5v|dalia|5|0,1,2,3,4|--disable-window-normalization --zero-guidance-forecast|2"
)

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

echo "Repo: $REPO"
echo "Train wall: $TRAIN_WALL   Eval wall: $EVAL_WALL"
echo "Exclude nodes: ${EXCLUDE_NODES:-none}"
echo ""

for row in "${CELLS[@]}"; do
    IFS='|' read -r mode variant subset dataset nvars indices extra stride <<< "$row"
    if [[ "$mode" == eval-only ]]; then
        submit_eval "$variant" "$subset" "$dataset" ""
    else
        tid=$(submit_train "$variant" "$subset" "$dataset" "$nvars" "$indices" "$extra" "$stride")
        submit_eval "$variant" "$subset" "$dataset" "$tid"
    fi
done

echo ""
echo "Verify partial count (expect 40):"
echo "  ls $EVAL_OUT/partials/*_binary_*.json 2>/dev/null | wc -l"
echo "Then refresh merge waiter:"
echo "  ./utils/submit_crossvar_merge_waiter.sh --run-stem $RUN_STEM --cancel-pending"
