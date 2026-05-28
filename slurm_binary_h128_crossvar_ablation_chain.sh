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
cat > "$PREAMBLE_FILE" <<'PREAMBLE'
set -euo pipefail
echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID  Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "Started: $(date)"

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

echo "[setup] Building venv on $SLURM_TMPDIR..."
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index \
    'torch==2.11.0+computecanada' numpy pandas scipy scikit-learn tqdm optuna wandb einops \
    -q
export PYTHONUNBUFFERED=1
cd "$REPO"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA required"
print("torch", torch.__version__, "gpu", torch.cuda.get_device_name(0))
PY
PREAMBLE

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

merge_worker="$JOB_DIR/merge-report.sh"
variant_args=()
for variant_spec in "${VARIANTS[@]}"; do
    IFS='|' read -r variant _extra_flags <<< "$variant_spec"
    # Any existing root for the label is enough for merge ordering/report labels.
    first_spec="${DATASET_SPECS[0]}"
    IFS='|' read -r first_dataset _subset _n _idx <<< "$first_spec"
    variant_args+=(--variant-root "$variant=${CKPT_ROOT[$variant:$first_dataset]}")
done
dataset_args=()
for spec in "${DATASET_SPECS[@]}"; do
    IFS='|' read -r dataset _subset _n _idx <<< "$spec"
    dataset_args+=("$dataset")
done
cat > "$merge_worker" <<EOF
#!/bin/bash
source "$PREAMBLE_FILE"
python -u "$EVAL_SCRIPT" \\
  --phase merge \\
  ${variant_args[*]} \\
  --datasets ${dataset_args[*]} \\
  --output-dir "$EVAL_OUT" \\
  --test-fraction 0.5 \\
  --sample-num 9 \\
  --num-sampling-steps 20 \\
  --gmm-components 9 \\
  --anchor-batch-size 16 \\
  --texture-per-sample
EOF
chmod +x "$merge_worker"

MERGE_DEP="afterok:${EVAL_JOBS[0]}"
for ejid in "${EVAL_JOBS[@]:1}"; do
    MERGE_DEP+=":$ejid"
done

echo "Submitting merge/report [$MERGE_DEP]..."
MERGE_JOB=$(sbatch --parsable \
    --job-name="merge-${RUN_STEM}" \
    --account=aip-boyuwang \
    --nodes=1 \
    --cpus-per-task=2 \
    --mem=16G \
    --time=0:30:00 \
    --dependency="$MERGE_DEP" \
    --output="$LOG_DIR/merge-%j.out" \
    --error="$LOG_DIR/merge-%j.err" \
    --mail-type=FAIL \
    --mail-user=ccao87@uwo.ca \
    "$merge_worker")

echo ""
echo "=================================================================="
echo "  Cross-var ablation chain submitted"
echo "  Train jobs: ${#TRAIN_JOB[@]}   Eval jobs: ${#EVAL_JOBS[@]}   Merge: $MERGE_JOB"
echo "  Checkpoints: results/ckpts/${DATE_TAG}-bin-h128-xvar-{variant}-{subset}"
echo "  Metrics:     $EVAL_OUT/metrics.csv"
echo "  Report:      reports/${RUN_STEM}_binary_variant_report.md"
echo "  Logs:        $LOG_DIR/"
echo "  Monitor:     squeue -u \$USER"
echo "=================================================================="
