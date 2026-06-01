#!/bin/bash
# =============================================================================
# MMPD vs binary-anchor matrix eval — parallel fan-out (Killarney).
#
# Dependency graph (per dataset, after init):
#   init  -> shared indices + run_manifest.json
#   mmpd-{ds}, bin-{ds}  (parallel)
#   merge -> metrics.json / metrics.csv
#
# USAGE (from $SCRATCH/ts-sandbox on Killarney LOGIN node — do NOT sbatch this file):
#   ./slurm_mmpd_gaussian_anchor_eval.sh --smoke-test
#   ./slurm_mmpd_gaussian_anchor_eval.sh
#   ./slurm_mmpd_gaussian_anchor_eval.sh --skip-mmpd-train
#   ./slurm_mmpd_gaussian_anchor_eval.sh --reeval-only
#   ./slurm_mmpd_gaussian_anchor_eval.sh --serial --smoke-test   # serial uses sbatch once
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE=0
SERIAL=0
SKIP_MMPD_TRAIN=0
RETRY_MMPD_ONLY=0
RETRY_ANCHOR_ONLY=0
FORCE_MMPD_EVAL=0
FORCE_ANCHOR_EVAL=0
FORCE_INDICES=0
SKIP_MERGE=0
MERGE_ONLY=0
SEED=2026

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) SMOKE=1; shift ;;
        --serial) SERIAL=1; shift ;;
        --skip-mmpd-train) SKIP_MMPD_TRAIN=1; shift ;;
        --reeval-only) SKIP_MMPD_TRAIN=1; FORCE_MMPD_EVAL=1; FORCE_ANCHOR_EVAL=1; shift ;;
        --retry-mmpd-only) RETRY_MMPD_ONLY=1; SKIP_MMPD_TRAIN=1; FORCE_MMPD_EVAL=1; shift ;;
        --retry-anchor-only) RETRY_ANCHOR_ONLY=1; FORCE_ANCHOR_EVAL=1; shift ;;
        --force-indices) FORCE_INDICES=1; shift ;;
        --skip-merge) SKIP_MERGE=1; shift ;;
        --merge-only) MERGE_ONLY=1; SKIP_MERGE=0; shift ;;
        --seed) SEED="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ "$MERGE_ONLY" -eq 1 && -z "${MATRIX_OUTPUT_DIR:-}" ]]; then
    echo "ERROR: --merge-only requires MATRIX_OUTPUT_DIR (existing matrix run)." >&2
    exit 1
fi

if [[ "$RETRY_MMPD_ONLY" -eq 1 && -z "${MATRIX_OUTPUT_DIR:-}" ]]; then
    echo "ERROR: --retry-mmpd-only requires MATRIX_OUTPUT_DIR (existing matrix run)." >&2
    exit 1
fi
if [[ "$RETRY_ANCHOR_ONLY" -eq 1 && -z "${MATRIX_OUTPUT_DIR:-}" ]]; then
    echo "ERROR: --retry-anchor-only requires MATRIX_OUTPUT_DIR (existing matrix run)." >&2
    exit 1
fi

dataset_wall_mmpd() {
  if [[ "$1" == "dalia" ]]; then
    echo "${WALL_MMPD_DALIA:-24:00:00}"
  else
    echo "$WALL_MMPD"
  fi
}

dataset_wall_anchor() {
  echo "$WALL_ANCHOR"
}

# ---------------------------------------------------------------------------
# Serial fallback: one self-submitting job (old behaviour)
# ---------------------------------------------------------------------------
if [[ "$SERIAL" -eq 1 ]]; then
    exec "$SCRIPT_DIR/slurm_mmpd_gaussian_anchor_eval_serial.sh" \
        $([[ "$SMOKE" -eq 1 ]] && echo --smoke-test) \
        $([[ "$SKIP_MMPD_TRAIN" -eq 1 ]] && echo --skip-mmpd-train) \
        --seed "$SEED" "$@"
fi

# ---------------------------------------------------------------------------
# Login-node launcher: fan-out parallel sbatch jobs (each has its own --time).
# ---------------------------------------------------------------------------
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: This script is a login-node launcher; sbatch cannot submit child jobs from a compute node." >&2
  echo "  cd \"\$SCRATCH/ts-sandbox\" && ./slurm_mmpd_gaussian_anchor_eval.sh $*" >&2
  exit 1
fi

{
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
      --topk-max 3
      --num-sampling-steps 5
      --gmm-components 5
      --gmm-iterations 3
      --mmpd-batch-size 16
      --mmpd-eval-batch-size 4
      --anchor-batch-size 4
    )
  else
    DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 illness exchange_rate weather electricity traffic PeMS solar_Alabama dalia)
    WALL_INIT="0:30:00"
    WALL_MMPD="12:00:00"
    WALL_ANCHOR="12:00:00"
    WALL_MMPD_DALIA="24:00:00"
    WALL_MERGE="0:30:00"
    MEM="60G"
    CPUS=8
    EVAL_EXTRA=(
      --mmpd-train-epochs 20
      --mmpd-patience 5
      --test-fraction 0.5
      --sample-num 100
      --topk-max 3
      --num-sampling-steps 20
      --gmm-components 10
      --gmm-iterations 10
      --mmpd-batch-size 32
      --mmpd-eval-batch-size 16
      --anchor-batch-size 16
      --texture-per-sample
    )
  fi
  if [[ -n "${MATRIX_DATASETS:-}" ]]; then
    read -r -a DATASETS <<< "${MATRIX_DATASETS}"
  fi
  MERGE_DATASETS=("${DATASETS[@]}")
  if [[ -n "${MATRIX_MERGE_DATASETS:-}" ]]; then
    read -r -a MERGE_DATASETS <<< "${MATRIX_MERGE_DATASETS}"
  fi

  if [[ -n "${MATRIX_OUTPUT_DIR:-}" ]]; then
    OUTPUT_DIR="$MATRIX_OUTPUT_DIR"
    [[ "$OUTPUT_DIR" != /* ]] && OUTPUT_DIR="$REPO/$OUTPUT_DIR"
    RUN_STEM="$(basename "$OUTPUT_DIR")"
    LOG_DIR="$REPO/results/logs/${RUN_STEM}"
  else
    RUN_STEM="$(date +%m-%d)-$$-mmpd-anchor-matrix${SMOKE_SUFFIX}"
    OUTPUT_DIR="$REPO/results/datasets/${RUN_STEM}"
    LOG_DIR="$REPO/results/logs/${RUN_STEM}"
  fi
  mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

  PREAMBLE_FILE="$REPO/results/job_preamble_mmpd_anchor_eval.sh"
  cat > "$PREAMBLE_FILE" << PREAMBLE
set -euo pipefail
echo "Job: \$SLURM_JOB_NAME  ID: \$SLURM_JOB_ID  Node: \${SLURMD_NODENAME:-unknown}"
echo "GPU: \$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "Started: \$(date)"

USER="\${USER:-\$(whoami)}"
STORE="$REPO/results"

pip_retry() {
  local max_attempts=5 delay=20 attempt
  for attempt in \$(seq 1 "\$max_attempts"); do
    if "\$@"; then return 0; fi
    if [[ "\$attempt" -lt "\$max_attempts" ]]; then
      echo "[setup] pip failed (attempt \${attempt}/\${max_attempts}), retry in \${delay}s..."
      sleep "\$delay"
      delay=\$((delay + 2))
    fi
  done
  echo "[setup] pip failed after \${max_attempts} attempts: \$*" >&2
  return 1
}

install_pipeline_deps() {
  pip_retry pip install --no-index --upgrade pip -q 2>/dev/null || pip_retry pip install -U pip -q
  if ! python -c "import torch" 2>/dev/null; then
    if ! pip_retry pip install --no-index torch torchvision numpy pandas scipy scikit-learn tqdm -q 2>/dev/null; then
      pip_retry pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
      pip_retry pip install numpy pandas scipy scikit-learn tqdm -q
    fi
  fi
  pip_retry pip install optuna wandb einops pyyaml scikit-learn -q
  if ! pip_retry pip install --no-index matplotlib -q 2>/dev/null; then
    pip_retry pip install matplotlib -q 2>/dev/null || export DIFFUSION_TSF_SKIP_VIZ=1
  fi
  python -c "import optuna, wandb, einops, yaml"
}

_load_modules() {
  module purge 2>/dev/null || true
  module load StdEnv/2023 2>/dev/null || true
  module load python/3.11 2>/dev/null || true
  module load cuda/12.2 2>/dev/null || true
  module load cudnn/8.9 2>/dev/null || true
}

VENV=""
for cand in \\
  "\$STORE/venv" \\
  "\${SCRATCH:-}/\${USER}/ts-sandbox/results/venv" \\
  "\${SCRATCH:-}/ts-sandbox/results/venv" \\
  "$REPO/results/venv"; do
  if [[ -x "\${cand}/bin/python" ]]; then
    VENV="\$cand"
    break
  fi
done

if [[ -n "\$VENV" ]]; then
  echo "[setup] Using persistent venv: \$VENV"
  _load_modules
  # shellcheck source=/dev/null
  source "\$VENV/bin/activate"
  export PATH="\$VENV/bin:\$PATH"
  export PYTHON="\$VENV/bin/python"
  install_pipeline_deps
else
  echo "[setup] No persistent venv; loading modules and building on \${SLURM_TMPDIR:-/tmp}..."
  _load_modules
  if ! command -v python >/dev/null 2>&1; then
    echo "[setup] ERROR: python unavailable (Lmod/cvmfs issue?). Create \$SCRATCH/ts-sandbox/results/venv on login and resubmit." >&2
    exit 1
  fi
  python -m venv "\${SLURM_TMPDIR:-/tmp}/env"
  VENV="\${SLURM_TMPDIR:-/tmp}/env"
  # shellcheck source=/dev/null
  source "\$VENV/bin/activate"
  export PATH="\$VENV/bin:\$PATH"
  export PYTHON="\$VENV/bin/python"
  install_pipeline_deps
fi

"\$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || true

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export REPO="$REPO"
export OUTPUT_DIR="$OUTPUT_DIR"
export TS_SANDBOX_REPO="\$REPO"
export PYTHONPATH="\$REPO\${PYTHONPATH:+:\$PYTHONPATH}"
cd "\$REPO"
PREAMBLE

  export REPO OUTPUT_DIR PREAMBLE_FILE SEED SKIP_MMPD_TRAIN
  export -a EVAL_EXTRA DATASETS MERGE_DATASETS

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
    if [[ -z "${MATRIX_INDICES_DIR:-}" ]]; then
      EVAL_BASE+=(--indices-dir "$OUTPUT_DIR" --mmpd-output-root "$OUTPUT_DIR")
    fi
  fi
  if [[ -n "${MATRIX_INDICES_DIR:-}" ]]; then
    _indices="$MATRIX_INDICES_DIR"
    [[ "$_indices" != /* ]] && _indices="$REPO/$_indices"
    EVAL_BASE+=(--indices-dir "$_indices")
  fi
  if [[ -n "${MMPD_REUSE_DIR:-}" ]]; then
    _reuse="$MMPD_REUSE_DIR"
    [[ "$_reuse" != /* ]] && _reuse="$REPO/$_reuse"
    EVAL_BASE+=(--mmpd-output-root "$_reuse")
  fi
  if [[ "$FORCE_MMPD_EVAL" -eq 1 ]]; then
    EVAL_BASE+=(--force-mmpd-eval)
  fi
  if [[ "$FORCE_ANCHOR_EVAL" -eq 1 ]]; then
    EVAL_BASE+=(--force-anchor-eval)
  fi
  if [[ "$FORCE_INDICES" -eq 1 ]]; then
    EVAL_BASE+=(--force-indices)
  fi
  if [[ -n "${BINARY_ANCHOR_ROOTS:-}" ]]; then
    for _br in ${BINARY_ANCHOR_ROOTS}; do
      _br_abs="$_br"
      [[ "$_br_abs" != /* ]] && _br_abs="$REPO/$_br_abs"
      EVAL_BASE+=(--binary-anchor-root "$_br_abs")
    done
  fi

  echo "Repo:       $REPO"
  echo "Output:     $OUTPUT_DIR"
  echo "Logs:       $LOG_DIR"
  echo "Datasets:   ${DATASETS[*]}"
  if [[ "${#MERGE_DATASETS[@]}" -ne "${#DATASETS[@]}" ]] || [[ "${MERGE_DATASETS[*]}" != "${DATASETS[*]}" ]]; then
    echo "Merge over: ${MERGE_DATASETS[*]}"
  fi

  JOB_INIT=""
  WORKER_DEP=()
  WORKER_IDS=()
  if [[ "$MERGE_ONLY" -eq 1 ]]; then
    echo "Merge-only mode (skip init + per-dataset workers)."
  elif [[ "$RETRY_MMPD_ONLY" -eq 1 || "$RETRY_ANCHOR_ONLY" -eq 1 ]]; then
    [[ "$RETRY_MMPD_ONLY" -eq 1 ]] && echo "Retry mode: MMPD eval only (skip init + binary anchor workers)."
    [[ "$RETRY_ANCHOR_ONLY" -eq 1 ]] && echo "Retry mode: binary anchor eval only (skip init + MMPD workers)."
  elif [[ "$SKIP_MMPD_TRAIN" -eq 1 && ( -n "${MATRIX_OUTPUT_DIR:-}" || -n "${MATRIX_INDICES_DIR:-}" ) ]]; then
    echo "Re-eval mode: skip init (reuse existing indices/output)."
  else
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
"\$PYTHON" -u ${EVAL_BASE[@]} \
  --phase init \
  --datasets ${DATASETS[*]}
echo "[init] done: \$(date)"
ENDSCRIPT
    )
    echo "  -> init: $JOB_INIT"
    WORKER_DEP=(--dependency="afterok:$JOB_INIT")
  fi

  if [[ "$MERGE_ONLY" -eq 0 ]]; then
  for ds in "${DATASETS[@]}"; do
    if [[ "$RETRY_ANCHOR_ONLY" -eq 0 ]]; then
    echo "Submitting mmpd-${ds} ${WORKER_DEP[*]} (wall=$(dataset_wall_mmpd "$ds"))..."
    JOB_MMPD=$(sbatch --parsable \
      --job-name="mmpd-mx-${ds}${SMOKE_SUFFIX}" \
      "${SBATCH_COMMON[@]}" \
      --time="$(dataset_wall_mmpd "$ds")" \
      "${WORKER_DEP[@]}" \
      --output="$LOG_DIR/mmpd-${ds}-%j.out" \
      --error="$LOG_DIR/mmpd-${ds}-%j.err" \
      <<ENDSCRIPT
#!/bin/bash
source "$PREAMBLE_FILE"
"\$PYTHON" -u ${EVAL_BASE[@]} \
  --phase mmpd \
  --datasets "$ds"
echo "[mmpd-${ds}] done: \$(date)"
ENDSCRIPT
    )
    echo "  -> mmpd-${ds}: $JOB_MMPD"
    WORKER_IDS+=("$JOB_MMPD")
    fi

    if [[ "$RETRY_MMPD_ONLY" -eq 0 ]]; then
      echo "Submitting bin-${ds} ${WORKER_DEP[*]} (wall=$(dataset_wall_anchor "$ds"))..."
      JOB_B=$(sbatch --parsable \
        --job-name="mmpd-mx-b-${ds}${SMOKE_SUFFIX}" \
        "${SBATCH_COMMON[@]}" \
        --time="$(dataset_wall_anchor "$ds")" \
        "${WORKER_DEP[@]}" \
        --output="$LOG_DIR/bin-${ds}-%j.out" \
        --error="$LOG_DIR/bin-${ds}-%j.err" \
        <<ENDSCRIPT
#!/bin/bash
source "$PREAMBLE_FILE"
"\$PYTHON" -u ${EVAL_BASE[@]} \
  --phase anchor \
  --anchor-variant binary \
  --datasets "$ds"
echo "[bin-${ds}] done: \$(date)"
ENDSCRIPT
      )
      echo "  -> bin-${ds}: $JOB_B"
      WORKER_IDS+=("$JOB_B")
    fi
  done
  fi

  if [[ "$SKIP_MERGE" -eq 1 ]]; then
    echo ""
    echo "=================================================================="
    echo "  Workers submitted (--skip-merge); run --merge-only when all partials exist."
    echo "  Output: $OUTPUT_DIR"
    echo "  Logs:   $LOG_DIR/"
    echo "=================================================================="
    exit 0
  fi

  MERGE_DEP_ARGS=()
  if [[ "${#WORKER_IDS[@]}" -gt 0 ]]; then
    MERGE_DEP="afterok:${WORKER_IDS[0]}"
    for wid in "${WORKER_IDS[@]:1}"; do
      MERGE_DEP+=":$wid"
    done
    MERGE_DEP_ARGS=(--dependency="$MERGE_DEP")
  fi

  echo "Submitting merge ${MERGE_DEP_ARGS[*]}..."
  JOB_MERGE=$(sbatch --parsable \
    --job-name="mmpd-mx-merge${SMOKE_SUFFIX}" \
    --account=aip-boyuwang \
    --nodes=1 \
    --cpus-per-task=2 \
    --mem=16G \
    "${GPU_ARGS[@]}" \
    --time="$WALL_MERGE" \
    "${MERGE_DEP_ARGS[@]}" \
    --output="$LOG_DIR/merge-%j.out" \
    --error="$LOG_DIR/merge-%j.err" \
    --mail-type=FAIL \
    --mail-user=ccao87@uwo.ca \
    <<ENDSCRIPT
#!/bin/bash
source "$PREAMBLE_FILE"
"\$PYTHON" -u ${EVAL_BASE[@]} \
  --phase merge \
  --datasets ${MERGE_DATASETS[*]} \
  --cpu
echo "[merge] done: \$(date)"
echo "Metrics: $OUTPUT_DIR/metrics.json"
ENDSCRIPT
  )
  echo "  -> merge: $JOB_MERGE"

  echo ""
  echo "=================================================================="
  echo "  Matrix eval submitted (${#DATASETS[@]} datasets × 2 workers + init + merge)"
  echo "  init:  $JOB_INIT"
  echo "  merge: $JOB_MERGE  (${MERGE_DEP:-no worker deps})"
  echo "  Output: $OUTPUT_DIR"
  echo "  Logs:   $LOG_DIR/"
  echo "  Monitor: squeue -u \$USER"
  echo "  Cancel:  scancel $JOB_INIT ${WORKER_IDS[*]} $JOB_MERGE"
  echo "=================================================================="
  exit 0
}
