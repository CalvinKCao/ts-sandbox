#!/bin/bash
# =============================================================================
# Recompute metrics with per-sample texture from an existing matrix run.
# Loads raw/*.npz on disk (no re-sampling). Updates partials/ + metrics.json.
#
# USAGE (Killarney login node, after matrix raw/ exists):
#   ./slurm_mmpd_texture_per_sample.sh --reference-run results/datasets/05-27-XXXX-mmpd-anchor-matrix
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEED=2026
REFERENCE_RUN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --reference-run) REFERENCE_RUN="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
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
    echo "ERROR: submit from \$SCRATCH/ts-sandbox on Killarney." >&2
    exit 1
  fi
  cd "$REPO"

  if [[ -z "$REFERENCE_RUN" ]]; then
    echo "ERROR: --reference-run is required (matrix output dir with raw/*.npz)." >&2
    exit 1
  fi
  if [[ "$REFERENCE_RUN" != /* ]]; then
    REFERENCE_RUN="$REPO/$REFERENCE_RUN"
  fi
  if ! compgen -G "$REFERENCE_RUN/raw/mmpd_"*.npz >/dev/null && \
     ! compgen -G "$REFERENCE_RUN/raw/"*_anchor_*.npz >/dev/null; then
    echo "ERROR: no raw/*.npz under $REFERENCE_RUN/raw — run matrix eval first." >&2
    exit 1
  fi

  DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 illness exchange_rate)
  WALL_WORKER="0:45:00"
  WALL_MERGE="0:20:00"
  MEM="24G"
  CPUS=4

  LOG_DIR="$REPO/results/logs/$(basename "$REFERENCE_RUN")-texture-per-sample"
  mkdir -p "$LOG_DIR"

  PREAMBLE_FILE="$REPO/results/job_preamble_mmpd_anchor_eval.sh"
  cat > "$PREAMBLE_FILE" << PREAMBLE
set -euo pipefail
echo "Job: \$SLURM_JOB_NAME  ID: \$SLURM_JOB_ID  Node: \${SLURMD_NODENAME:-unknown}"
echo "Started: \$(date)"
module --force purge || module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9
virtualenv --no-download "\$SLURM_TMPDIR/env"
source "\$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index \\
    'torch==2.11.0+computecanada' numpy pandas scipy scikit-learn tqdm einops -q
pip install --no-index optuna wandb matplotlib -q 2>/dev/null || \\
    pip install optuna wandb matplotlib -q
export PYTHONUNBUFFERED=1
cd "$REPO"
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

  # Match the matrix run sample count (9). Only used if a raw file must be rebuilt.
  EVAL_BASE=(
    "$REPO/utils/eval_mmpd_gaussian_anchor.py"
    --output-dir "$REFERENCE_RUN"
    --indices-dir "$REFERENCE_RUN"
    --mmpd-output-root "$REFERENCE_RUN"
    --ckpt-base "$REPO/results/ckpts"
    --mmpd-repo "$REPO/temp/MMPD"
    --mmpd-data-dir "$REPO/temp/mmpd_datasets"
    --seed "$SEED"
    --no-update-mmpd
    --metrics-profile full
    --texture-per-sample
    --skip-mmpd-train
    --sample-num 9
    --gmm-components 9
    --topk-max 3
  )

  echo "Reference run: $REFERENCE_RUN"
  echo "Logs:          $LOG_DIR"
  echo "Updates partials + metrics in place under reference run."

  WORKER_IDS=()
  for ds in "${DATASETS[@]}"; do
    for spec in "mmpd:mmpd:" "gauss:anchor:gaussian" "bin:anchor:binary"; do
      IFS=: read -r label phase variant <<< "$spec"
      EXTRA=(--phase "$phase" --datasets "$ds")
      [[ -n "$variant" ]] && EXTRA+=(--anchor-variant "$variant")
      echo "Submitting tex-${label}-${ds}..."
      JOB=$(sbatch --parsable \
        --job-name="tex-${label}-${ds}" \
        "${SBATCH_COMMON[@]}" \
        --time="$WALL_WORKER" \
        --output="$LOG_DIR/tex-${label}-${ds}-%j.out" \
        --error="$LOG_DIR/tex-${label}-${ds}-%j.err" \
        <<ENDSCRIPT
#!/bin/bash
source "$PREAMBLE_FILE"
python -u ${EVAL_BASE[@]} ${EXTRA[@]}
echo "[tex-${label}-${ds}] done: \$(date)"
ENDSCRIPT
      )
      echo "  -> $JOB"
      WORKER_IDS+=("$JOB")
    done
  done

  MERGE_DEP="afterok:${WORKER_IDS[0]}"
  for wid in "${WORKER_IDS[@]:1}"; do
    MERGE_DEP+=":$wid"
  done

  JOB_MERGE=$(sbatch --parsable \
    --job-name="tex-merge" \
    "${SBATCH_COMMON[@]}" \
    --time="$WALL_MERGE" \
    --dependency="$MERGE_DEP" \
    --output="$LOG_DIR/tex-merge-%j.out" \
    --error="$LOG_DIR/tex-merge-%j.err" \
    <<ENDSCRIPT
#!/bin/bash
source "$PREAMBLE_FILE"
python -u ${EVAL_BASE[@]} \
  --phase merge \
  --datasets ${DATASETS[*]} \
  --cpu
echo "[tex-merge] done: \$(date)"
ENDSCRIPT
  )
  echo "  -> merge: $JOB_MERGE ($MERGE_DEP)"
  echo ""
  echo "Cancel all: scancel ${WORKER_IDS[*]} $JOB_MERGE"
  exit 0
fi

echo "ERROR: login node only." >&2
exit 1
