#!/bin/bash
# =============================================================================
# Binary h128 cross-variate ablation matrix.
#
# Experiments:
#   exp1: dataset-level normalization only, forecast ghost kept
#   exp2: forecast ghost zeroed, iTransformer bottleneck tokens kept
#   exp3: exp1 + exp2
#   exp4: baseline, but with DiT bottleneck cross-variate tokens enabled
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./slurm_binary_h128_crossvar_ablation_chain.sh
#   ./slurm_binary_h128_crossvar_ablation_chain.sh --smoke-test
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE=0
DATE_TAG="$(date +%m-%d)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) SMOKE=1; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: run from the login node only (this script submits jobs)." >&2
    exit 1
fi

if [[ -d "${SCRATCH:-}/ts-sandbox" ]]; then
    REPO="${SCRATCH}/ts-sandbox"
elif [[ -d "$HOME/ts-sandbox" ]]; then
    REPO="$HOME/ts-sandbox"
else
    REPO="$(cd "$SCRIPT_DIR" && pwd)"
fi
if [[ "$REPO" == /home/* ]]; then
    echo "ERROR: submit from \$SCRATCH/ts-sandbox on Killarney, not /home." >&2
    exit 1
fi
cd "$REPO"

TRAIN_SCRIPT="$REPO/slurm_binary_anchor_92d3.sh"
EVAL_SCRIPT="$REPO/utils/eval_binary_anchor_variants.py"
chmod +x "$TRAIN_SCRIPT" "$EVAL_SCRIPT"

seq_csv() {
    local n="$1"
    local out="" i
    for ((i = 0; i < n; i++)); do
        if [[ -z "$out" ]]; then out="$i"; else out="${out},${i}"; fi
    done
    echo "$out"
}

# Format: dataset|subset_id|n_variates|comma_indices
DATASET_SPECS=(
    "ETTh1|ETTh1|7|"
    "ETTh2|ETTh2|7|"
    "exchange_rate|exchange_rate|8|"
    "illness|illness|7|"
    "weather|weather_9v|9|$(seq_csv 9)"
    "ETTm1|ETTm1_2x|7|"
    "ETTm2|ETTm2_2x|7|"
    "electricity|electricity_18v|18|$(seq_csv 18)"
    "traffic|traffic_27v|27|$(seq_csv 27)"
    "dalia|dalia_5v|5|0,1,2,3,4"
)

VARIANTS=(
    "exp1-datasetnorm|--disable-window-normalization"
    "exp2-tokens-only|--zero-guidance-forecast"
    "exp3-datasetnorm-tokens-only|--disable-window-normalization --zero-guidance-forecast"
    "exp4-baseline-xvar|"
)

SMOKE_SUFFIX=""
if [[ "$SMOKE" -eq 1 ]]; then
    SMOKE_SUFFIX="-smoke"
    DATASET_SPECS=("ETTh1|ETTh1|7|")
fi

RUN_STEM="${DATE_TAG}-bin-h128-crossvar-ablation${SMOKE_SUFFIX}"
EVAL_OUT="$REPO/results/datasets/${RUN_STEM}"
LOG_DIR="$REPO/results/logs/${RUN_STEM}"
JOB_DIR="$REPO/results/jobs/${RUN_STEM}"
mkdir -p "$EVAL_OUT" "$LOG_DIR" "$JOB_DIR"

declare -A TRAIN_JOB
declare -A CKPT_ROOT
declare -A SPEC_BY_DATASET
EVAL_JOBS=()

echo "Repo:       $REPO"
echo "Run stem:   $RUN_STEM"
echo "Eval out:   $EVAL_OUT"
echo "Datasets:   ${#DATASET_SPECS[@]}"
echo "Variants:   ${#VARIANTS[@]}"
echo "Cross-attn: enabled (no --disable-cross-attention)"

for spec in "${DATASET_SPECS[@]}"; do
    IFS='|' read -r dataset subset_id nvars indices <<< "$spec"
    SPEC_BY_DATASET["$dataset"]="$spec"
    for variant_spec in "${VARIANTS[@]}"; do
        IFS='|' read -r variant extra_flags <<< "$variant_spec"
        stem="${DATE_TAG}-bin-h128-xvar-${variant}-${subset_id,,}${SMOKE_SUFFIX}"
        CKPT_ROOT["${variant}:${dataset}"]="$REPO/results/ckpts/${stem}"

        wall="4:00:00"
        mem="60G"
        cpus=8
        if [[ "$SMOKE" -eq 1 ]]; then
            wall="0:30:00"; mem="24G"; cpus=4
        elif [[ "$nvars" -ge 16 ]]; then
            wall="4:00:00"; mem="72G"; cpus=8
        fi

        train_args=(
            --dataset "$dataset"
            --n-variates "$nvars"
            --subset-id "$subset_id"
            --image-height 128
            --fresh
            --run-stem "$stem"
            --wandb-project "ts-sandbox-binary-h128-crossvar-ablation"
        )
        if [[ -n "$indices" ]]; then
            train_args+=(--variate-indices "$indices")
        fi
        if [[ -n "$extra_flags" ]]; then
            read -r -a extra_array <<< "$extra_flags"
            train_args+=("${extra_array[@]}")
        fi
        if [[ "$dataset" == ETTm1 || "$dataset" == ETTm2 || "$dataset" == dalia ]]; then
            train_args+=(--window-stride 2)
        fi
        if [[ "$SMOKE" -eq 1 ]]; then
            train_args+=(--smoke-test)
        fi

        echo "Submitting train ${variant}/${subset_id} (n=${nvars}, stem=${stem})..."
        jid=$(sbatch --parsable \
            --job-name="h128-${variant}-${subset_id,,}" \
            --account=aip-boyuwang \
            --time="$wall" \
            --nodes=1 \
            --gres=gpu:l40s:1 \
            --cpus-per-task="$cpus" \
            --mem="$mem" \
            --output=/dev/null \
            --error=/dev/null \
            --mail-type=FAIL \
            --mail-user=ccao87@uwo.ca \
            "$TRAIN_SCRIPT" "${train_args[@]}")
        TRAIN_JOB["${variant}:${dataset}"]="$jid"
        echo "  -> $jid"
    done
done

PREAMBLE_FILE="$JOB_DIR/eval_preamble.sh"
"$REPO/utils/write_crossvar_eval_preamble.sh" "$JOB_DIR"

export REPO
for spec in "${DATASET_SPECS[@]}"; do
    IFS='|' read -r dataset subset_id nvars indices <<< "$spec"
    for variant_spec in "${VARIANTS[@]}"; do
        IFS='|' read -r variant _extra_flags <<< "$variant_spec"
        dep="afterok:${TRAIN_JOB[$variant:$dataset]}"
        root="${CKPT_ROOT[$variant:$dataset]}"
        worker="$JOB_DIR/eval-${variant}-${subset_id}.sh"
        eval_extra=""
        if [[ "$dataset" == dalia ]]; then
            eval_extra="--lookback 80 --horizon 20"
        fi
        cat > "$worker" <<EOF
#!/bin/bash
source "$PREAMBLE_FILE"
python -u "$EVAL_SCRIPT" \\
  --phase eval \\
  --variant-root "$variant=$root" \\
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
        echo "Submitting eval ${variant}/${subset_id} [$dep]..."
        ejid=$(sbatch --parsable \
            --job-name="eval-${variant}-${subset_id,,}" \
            --account=aip-boyuwang \
            --nodes=1 \
            --gres=gpu:l40s:1 \
            --cpus-per-task=8 \
            --mem=60G \
            --time="$([[ "$SMOKE" -eq 1 ]] && echo 0:45:00 || echo 4:00:00)" \
            --dependency="$dep" \
            --output="$LOG_DIR/eval-${variant}-${subset_id}-%j.out" \
            --error="$LOG_DIR/eval-${variant}-${subset_id}-%j.err" \
            --mail-type=FAIL \
            --mail-user=ccao87@uwo.ca \
            "$worker")
        EVAL_JOBS+=("$ejid")
        echo "  -> $ejid"
    done
done

"$REPO/utils/write_crossvar_merge_invoke.sh" "$RUN_STEM" "$REPO" "$DATE_TAG"
MERGE_INVOKE="$JOB_DIR/merge_invoke.sh"
WAIT_SCRIPT="$REPO/utils/wait_and_merge_binary_crossvar.sh"
chmod +x "$WAIT_SCRIPT"

printf '%s\n' "${EVAL_JOBS[@]}" > "$JOB_DIR/eval_job_ids.txt"

MERGE_DEP="afterany:${EVAL_JOBS[0]}"
for ejid in "${EVAL_JOBS[@]:1}"; do
    MERGE_DEP+=":$ejid"
done

echo "Submitting merge waiter (polls partials, then merge) [$MERGE_DEP]..."
MERGE_JOB=$(sbatch --parsable \
    --job-name="merge-wait-${RUN_STEM}" \
    --account=aip-boyuwang \
    --nodes=1 \
    --cpus-per-task=2 \
    --mem=16G \
    --time=7-00:00:00 \
    --dependency="$MERGE_DEP" \
    --output="$LOG_DIR/merge-wait-%j.out" \
    --error="$LOG_DIR/merge-wait-%j.err" \
    --mail-type=END,FAIL \
    --mail-user=ccao87@uwo.ca \
    --export=ALL,REPO="$REPO",EVAL_OUT="$EVAL_OUT",EVAL_SCRIPT="$EVAL_SCRIPT",MERGE_INVOKE="$MERGE_INVOKE" \
    "$WAIT_SCRIPT")
echo "$MERGE_JOB" > "$JOB_DIR/merge_waiter_job_id.txt"

echo ""
echo "=================================================================="
echo "  Cross-var ablation chain submitted"
echo "  Train jobs: ${#TRAIN_JOB[@]}   Eval jobs: ${#EVAL_JOBS[@]}   Merge waiter: $MERGE_JOB"
echo "  Checkpoints: results/ckpts/${DATE_TAG}-bin-h128-xvar-{variant}-{subset}"
echo "  Metrics:     $EVAL_OUT/metrics.csv"
echo "  Report:      reports/${RUN_STEM}_binary_variant_report.md"
echo "  Logs:        $LOG_DIR/"
echo "  Merge waiter polls partials/ then writes metrics.csv (resubmit evals OK)"
echo "  Re-submit waiter: utils/submit_crossvar_merge_waiter.sh --run-stem $RUN_STEM"
echo "  Monitor:     squeue -u \$USER"
echo "=================================================================="
