#!/bin/bash
# =============================================================================
# 100-sample probabilistic eval (50% test subset): mean-pred MSE/MAE, CRPS, top-3.
# Eval-only — reuses MMPD checkpoints from a finished matrix run when given
# --reference-run. Parallel fan-out: per dataset × (MMPD | binary) + merge.
#
# USAGE (from $SCRATCH/ts-sandbox on Killarney login node):
#   ./slurm_mmpd_anchor_prob100_eval.sh --reference-run results/datasets/05-27-XXXX-mmpd-anchor-matrix
#   ./slurm_mmpd_anchor_prob100_eval.sh --smoke-test --reference-run ...
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE=0
SEED=2026
REFERENCE_RUN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) SMOKE=1; shift ;;
        --seed) SEED="$2"; shift 2 ;;
        --reference-run) REFERENCE_RUN="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
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

  if [[ -n "$REFERENCE_RUN" ]]; then
    if [[ "$REFERENCE_RUN" != /* ]]; then
      REFERENCE_RUN="$REPO/$REFERENCE_RUN"
    fi
    if [[ ! -d "$REFERENCE_RUN/raw" ]]; then
      echo "ERROR: --reference-run must contain raw/indices_*.json (matrix init output)." >&2
      exit 1
    fi
  fi

  SMOKE_SUFFIX=""
  SKIP_INIT=0
  if [[ "$SMOKE" -eq 1 ]]; then
    SMOKE_SUFFIX="-smoke"
    DATASETS=(ETTh1)
    WALL_INIT="0:25:00"
    WALL_WORKER="0:45:00"
    WALL_MERGE="0:20:00"
    MEM="24G"
    CPUS=4
    EVAL_EXTRA=(
      --metrics-profile prob-core
      --sample-num 10
      --topk-max 3
      --test-fraction 0.02
      --test-max-items 32
      --num-sampling-steps 10
      --gmm-components 9
      --gmm-iterations 5
      --mmpd-eval-batch-size 4
      --anchor-batch-size 4
      --skip-mmpd-train
      --force-mmpd-eval
      --force-anchor-eval
    )
  else
    DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 illness exchange_rate weather electricity traffic PeMS solar_Alabama dalia)
    WALL_INIT="0:30:00"
    WALL_WORKER="4:00:00"
    WALL_MERGE="0:30:00"
    MEM="60G"
    CPUS=8
    EVAL_EXTRA=(
      --metrics-profile prob-core
      --sample-num 100
      --topk-max 3
      --test-fraction 0.5
      --num-sampling-steps 20
      --gmm-components 10
      --gmm-iterations 10
      --mmpd-eval-batch-size 8
      --anchor-batch-size 8
      --skip-mmpd-train
      --force-mmpd-eval
      --force-anchor-eval
    )
  fi

  if [[ -n "$REFERENCE_RUN" ]]; then
    SKIP_INIT=1
    EVAL_EXTRA+=(--indices-dir "$REFERENCE_RUN" --mmpd-output-root "$REFERENCE_RUN")
  fi

  RUN_STEM="$(date +%m-%d)-$$-mmpd-prob100${SMOKE_SUFFIX}"
  OUTPUT_DIR="$REPO/results/datasets/${RUN_STEM}"
  LOG_DIR="$REPO/results/logs/${RUN_STEM}"
  mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

  PREAMBLE_FILE="$REPO/results/job_preamble_mmpd_anchor_eval.sh"
  cat > "$PREAMBLE_FILE" << PREAMBLE
set -euo pipefail
echo "Job: \$SLURM_JOB_NAME  ID: \$SLURM_JOB_ID  Node: \${SLURMD_NODENAME:-unknown}"
echo "GPU: \$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "Started: \$(date)"

module --force purge || module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

echo "[setup] Building venv on \$SLURM_TMPDIR..."
virtualenv --no-download "\$SLURM_TMPDIR/env"
source "\$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index \\
    'torch==2.11.0+computecanada' numpy pandas scipy scikit-learn tqdm einops -q
pip install --no-index optuna wandb matplotlib -q 2>/dev/null || \\
    pip install optuna wandb matplotlib -q
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA required"
print("torch", torch.__version__, "gpu", torch.cuda.get_device_name(0))
PY

export PYTHONUNBUFFERED=1
export REPO="$REPO"
export OUTPUT_DIR="$OUTPUT_DIR"
cd "\$REPO"
PREAMBLE

  GPU_ARGS=(--gres=gpu:l40s:1)
  SBATCH_COMMON=(
    --account=aip-boyuwang
    --nodes=1
    --cpus-per-task="$CPUS"
    --mem="$MEM"
    "${GPU_ARGS[@]}"
    --mail-type=FAIL
    --mail-user=ccao87@uwo.ca
  )

  EVAL_BASE=(
    "$REPO/utils/eval_mmpd_gaussian_anchor.py"
    --output-dir "$OUTPUT_DIR"
    --ckpt-base "$REPO/results/ckpts"
    --mmpd-repo "$REPO/temp/MMPD"
    --mmpd-data-dir "$REPO/temp/mmpd_datasets"
    --seed "$SEED"
    --no-update-mmpd
    "${EVAL_EXTRA[@]}"
  )
  if [[ -n "${BINARY_ANCHOR_ROOTS:-}" ]]; then
    for _br in ${BINARY_ANCHOR_ROOTS}; do
      _br_abs="$_br"
      [[ "$_br_abs" != /* ]] && _br_abs="$REPO/$_br_abs"
      EVAL_BASE+=(--binary-anchor-root "$_br_abs")
    done
  fi

  echo "Repo:          $REPO"
  echo "Output:        $OUTPUT_DIR"
  echo "Logs:          $LOG_DIR"
  echo "Reference run: ${REFERENCE_RUN:-none}"
  echo "Datasets:      ${DATASETS[*]}"

  JOB_INIT=""
  WORKER_DEP=""
  if [[ "$SKIP_INIT" -eq 0 ]]; then
    echo "Submitting init (shared indices)..."
    JOB_INIT=$(sbatch --parsable \
      --job-name="prob100-init${SMOKE_SUFFIX}" \
      "${SBATCH_COMMON[@]}" \
      --time="$WALL_INIT" \
      --output="$LOG_DIR/init-%j.out" \
      --error="$LOG_DIR/init-%j.err" \
      <<ENDSCRIPT
#!/bin/bash
source "$PREAMBLE_FILE"
python -u ${EVAL_BASE[@]} \
  --phase init \
  --datasets ${DATASETS[*]}
echo "[init] done: \$(date)"
ENDSCRIPT
    )
    echo "  -> init: $JOB_INIT"
    WORKER_DEP="afterok:$JOB_INIT"
  fi

  WORKER_IDS=()
  for ds in "${DATASETS[@]}"; do
    DEP_ARGS=()
    [[ -n "$WORKER_DEP" ]] && DEP_ARGS=(--dependency="$WORKER_DEP")

    echo "Submitting mmpd-${ds} ${WORKER_DEP:+(after init)}..."
    JOB_MMPD=$(sbatch --parsable \
      --job-name="prob100-mmpd-${ds}${SMOKE_SUFFIX}" \
      "${SBATCH_COMMON[@]}" \
      --time="$WALL_WORKER" \
      "${DEP_ARGS[@]}" \
      --output="$LOG_DIR/mmpd-${ds}-%j.out" \
      --error="$LOG_DIR/mmpd-${ds}-%j.err" \
      <<ENDSCRIPT
#!/bin/bash
source "$PREAMBLE_FILE"
python -u ${EVAL_BASE[@]} \
  --phase mmpd \
  --datasets "$ds"
echo "[mmpd-${ds}] done: \$(date)"
ENDSCRIPT
    )
    echo "  -> mmpd-${ds}: $JOB_MMPD"
    WORKER_IDS+=("$JOB_MMPD")

    echo "Submitting bin-${ds}..."
    JOB_B=$(sbatch --parsable \
      --job-name="prob100-b-${ds}${SMOKE_SUFFIX}" \
      "${SBATCH_COMMON[@]}" \
      --time="$WALL_WORKER" \
      "${DEP_ARGS[@]}" \
      --output="$LOG_DIR/bin-${ds}-%j.out" \
      --error="$LOG_DIR/bin-${ds}-%j.err" \
      <<ENDSCRIPT
#!/bin/bash
source "$PREAMBLE_FILE"
python -u ${EVAL_BASE[@]} \
  --phase anchor \
  --anchor-variant binary \
  --datasets "$ds"
echo "[bin-${ds}] done: \$(date)"
ENDSCRIPT
    )
    echo "  -> bin-${ds}: $JOB_B"
    WORKER_IDS+=("$JOB_B")
  done

  MERGE_DEP="afterok:${WORKER_IDS[0]}"
  for wid in "${WORKER_IDS[@]:1}"; do
    MERGE_DEP+=":$wid"
  done

  echo "Submitting merge [$MERGE_DEP]..."
  JOB_MERGE=$(sbatch --parsable \
    --job-name="prob100-merge${SMOKE_SUFFIX}" \
    --account=aip-boyuwang \
    --nodes=1 \
    --cpus-per-task=2 \
    --mem=16G \
    "${GPU_ARGS[@]}" \
    --time="$WALL_MERGE" \
    --dependency="$MERGE_DEP" \
    --output="$LOG_DIR/merge-%j.out" \
    --error="$LOG_DIR/merge-%j.err" \
    --mail-type=END,FAIL \
    --mail-user=ccao87@uwo.ca \
    <<ENDSCRIPT
#!/bin/bash
source "$PREAMBLE_FILE"
python -u ${EVAL_BASE[@]} \
  --phase merge \
  --datasets ${DATASETS[*]} \
  --cpu
echo "[merge] done: \$(date)"
echo "Metrics: $OUTPUT_DIR/metrics.json"
ENDSCRIPT
  )
  echo "  -> merge: $JOB_MERGE"

  echo ""
  echo "=================================================================="
  echo "  Prob-100 eval submitted (${#DATASETS[@]} datasets × 3 workers + merge)"
  [[ -n "$JOB_INIT" ]] && echo "  init:  $JOB_INIT"
  echo "  merge: $JOB_MERGE"
  echo "  Output: $OUTPUT_DIR"
  echo "  Logs:   $LOG_DIR/"
  echo "  Monitor: squeue -u \$USER"
  CANCEL_IDS=("${WORKER_IDS[@]}" "$JOB_MERGE")
  [[ -n "$JOB_INIT" ]] && CANCEL_IDS=("$JOB_INIT" "${CANCEL_IDS[@]}")
  echo "  Cancel:  scancel ${CANCEL_IDS[*]}"
  echo "=================================================================="
  exit 0
fi

echo "ERROR: run from login node only." >&2
exit 1
