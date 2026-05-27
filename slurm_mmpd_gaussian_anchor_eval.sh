#!/bin/bash
# =============================================================================
# MMPD vs binary/Gaussian-anchor matrix eval — parallel fan-out (Killarney).
#
# Dependency graph (per dataset, after init):
#   init  -> shared indices + run_manifest.json
#   mmpd-{ds}, gauss-{ds}, bin-{ds}  (all parallel)
#   merge -> metrics.json / metrics.csv
#
# USAGE (from $SCRATCH/ts-sandbox on login node):
#   ./slurm_mmpd_gaussian_anchor_eval.sh --smoke-test
#   ./slurm_mmpd_gaussian_anchor_eval.sh
#   ./slurm_mmpd_gaussian_anchor_eval.sh --skip-mmpd-train
#   ./slurm_mmpd_gaussian_anchor_eval.sh --serial --smoke-test   # one GPU, debug
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE=0
SERIAL=0
SKIP_MMPD_TRAIN=0
SEED=2026

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) SMOKE=1; shift ;;
        --serial) SERIAL=1; shift ;;
        --skip-mmpd-train) SKIP_MMPD_TRAIN=1; shift ;;
        --seed) SEED="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Serial fallback: one self-submitting job (old behaviour)
# ---------------------------------------------------------------------------
if [[ "$SERIAL" -eq 1 && -z "${SLURM_JOB_ID:-}" ]]; then
    exec "$SCRIPT_DIR/slurm_mmpd_gaussian_anchor_eval_serial.sh" \
        $([[ "$SMOKE" -eq 1 ]] && echo --smoke-test) \
        $([[ "$SKIP_MMPD_TRAIN" -eq 1 ]] && echo --skip-mmpd-train) \
        --seed "$SEED" "$@"
fi

# ---------------------------------------------------------------------------
# Login node: fan-out parallel jobs
# ---------------------------------------------------------------------------
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

  SMOKE_SUFFIX=""
  if [[ "$SMOKE" -eq 1 ]]; then
    SMOKE_SUFFIX="-smoke"
    DATASETS=(ETTh1)
    WALL_INIT="0:25:00"
    WALL_MMPD="0:45:00"
    WALL_ANCHOR="0:45:00"
    WALL_MERGE="0:20:00"
    MEM="24G"
    CPUS=4
    EVAL_EXTRA=(
      --mmpd-train-epochs 1
      --mmpd-patience 1
      --test-fraction 0.02
      --test-max-items 32
      --sample-num 5
      --num-sampling-steps 5
      --gmm-components 5
      --gmm-iterations 3
      --mmpd-batch-size 16
      --mmpd-eval-batch-size 4
      --anchor-batch-size 4
    )
  else
    DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 illness exchange_rate)
    WALL_INIT="0:30:00"
    WALL_MMPD="4:00:00"
    WALL_ANCHOR="4:00:00"
    WALL_MERGE="0:30:00"
    MEM="60G"
    CPUS=8
    EVAL_EXTRA=(
      --mmpd-train-epochs 20
      --mmpd-patience 5
      --test-fraction 0.5
      --sample-num 9
      --num-sampling-steps 20
      --gmm-components 9
      --gmm-iterations 10
      --mmpd-batch-size 32
      --mmpd-eval-batch-size 16
      --anchor-batch-size 16
      --texture-per-sample
    )
  fi

  RUN_STEM="$(date +%m-%d)-$$-mmpd-anchor-matrix${SMOKE_SUFFIX}"
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

  export REPO OUTPUT_DIR PREAMBLE_FILE SEED SKIP_MMPD_TRAIN
  export -a EVAL_EXTRA DATASETS

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
  if [[ "$SKIP_MMPD_TRAIN" -eq 1 ]]; then
    EVAL_BASE+=(--skip-mmpd-train)
  fi

  echo "Repo:       $REPO"
  echo "Output:     $OUTPUT_DIR"
  echo "Logs:       $LOG_DIR"
  echo "Datasets:   ${DATASETS[*]}"

  echo "Submitting init (shared indices + manifest)..."
  JOB_INIT=$(sbatch --parsable \
    --job-name="mmpd-mx-init${SMOKE_SUFFIX}" \
    "${SBATCH_COMMON[@]}" \
    --time="$WALL_INIT" \
    --output="$LOG_DIR/init-%j.out" \
    --error="$LOG_DIR/init-%j.err" \
    <<ENDSCRIPT
#!/bin/bash
source "$PREAMBLE_FILE"
# Quoted "${EVAL_BASE[@]}" inside heredocs collapses to one argv; use unquoted.
python -u ${EVAL_BASE[@]} \
  --phase init \
  --datasets ${DATASETS[*]}
echo "[init] done: \$(date)"
ENDSCRIPT
  )
  echo "  -> init: $JOB_INIT"

  WORKER_IDS=()
  for ds in "${DATASETS[@]}"; do
    echo "Submitting mmpd-${ds} [afterok:$JOB_INIT]..."
    JOB_MMPD=$(sbatch --parsable \
      --job-name="mmpd-mx-${ds}${SMOKE_SUFFIX}" \
      "${SBATCH_COMMON[@]}" \
      --time="$WALL_MMPD" \
      --dependency="afterok:$JOB_INIT" \
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

    echo "Submitting gauss-${ds} [afterok:$JOB_INIT]..."
    JOB_G=$(sbatch --parsable \
      --job-name="mmpd-mx-g-${ds}${SMOKE_SUFFIX}" \
      "${SBATCH_COMMON[@]}" \
      --time="$WALL_ANCHOR" \
      --dependency="afterok:$JOB_INIT" \
      --output="$LOG_DIR/gauss-${ds}-%j.out" \
      --error="$LOG_DIR/gauss-${ds}-%j.err" \
      <<ENDSCRIPT
#!/bin/bash
source "$PREAMBLE_FILE"
python -u ${EVAL_BASE[@]} \
  --phase anchor \
  --anchor-variant gaussian \
  --datasets "$ds"
echo "[gauss-${ds}] done: \$(date)"
ENDSCRIPT
    )
    echo "  -> gauss-${ds}: $JOB_G"
    WORKER_IDS+=("$JOB_G")

    echo "Submitting bin-${ds} [afterok:$JOB_INIT]..."
    JOB_B=$(sbatch --parsable \
      --job-name="mmpd-mx-b-${ds}${SMOKE_SUFFIX}" \
      "${SBATCH_COMMON[@]}" \
      --time="$WALL_ANCHOR" \
      --dependency="afterok:$JOB_INIT" \
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

  MERGE_DEP="afterok:$JOB_INIT"
  for wid in "${WORKER_IDS[@]}"; do
    MERGE_DEP+=":$wid"
  done

  echo "Submitting merge [$MERGE_DEP]..."
  JOB_MERGE=$(sbatch --parsable \
    --job-name="mmpd-mx-merge${SMOKE_SUFFIX}" \
    --account=aip-boyuwang \
    --nodes=1 \
    --cpus-per-task=2 \
    --mem=16G \
    "${GPU_ARGS[@]}" \
    --time="$WALL_MERGE" \
    --dependency="$MERGE_DEP" \
    --output="$LOG_DIR/merge-%j.out" \
    --error="$LOG_DIR/merge-%j.err" \
    --mail-type=FAIL \
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
  echo "  Matrix eval submitted (${#DATASETS[@]} datasets × 3 workers + init + merge)"
  echo "  init:  $JOB_INIT"
  echo "  merge: $JOB_MERGE  ($MERGE_DEP)"
  echo "  Output: $OUTPUT_DIR"
  echo "  Logs:   $LOG_DIR/"
  echo "  Monitor: squeue -u \$USER"
  echo "  Cancel:  scancel $JOB_INIT ${WORKER_IDS[*]} $JOB_MERGE"
  echo "=================================================================="
  exit 0
fi

echo "ERROR: this script should only run worker bodies via sbatch heredoc." >&2
exit 1
