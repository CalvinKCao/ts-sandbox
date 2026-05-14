#!/bin/bash
# Batch joint pretrain + finetune for datasets with native dimensionality < 32.
#
# Alliance / Killarney: run from repo root on scratch (not /home for GPU jobs).
# From a login node this script **submits many Slurm GPU jobs in parallel** (one
# pretrain per native dim, one finetune per dataset). With RUN=all, finetune jobs
# depend on the matching-dim pretrain (afterok). Check sacct / squeue.
# Logs: ./results/logs/MM-DD-<jobid-last4>-joint-pre-d*.log | joint-ft-*.log
#
# Usage (repo root):
#   ./utils/run_joint_small_datasets.sh
#   ./utils/run_joint_small_datasets.sh --smoke-test
#   RUN=finetune ./utils/run_joint_small_datasets.sh
#   VENV_PATH=/path/to/venv ./utils/run_joint_small_datasets.sh
#
# One sequential Slurm job (legacy): JOINT_SMALL_SEQUENTIAL=1 ./utils/...
# Run on the login node without Slurm (debug): JOINT_SMALL_LOCAL=1 ./utils/...
#
# Extra args are forwarded to every python invocation.

set -euo pipefail

# Repo root: inside Slurm, trust submit directory (BASH_SOURCE points at spool copy).
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -n "${SLURM_JOB_ID:-}" ]]; then
  ROOT="$SLURM_SUBMIT_DIR"
  cd "$ROOT"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
  cd "$ROOT"
fi

MAX_V="${MAX_V:-31}" # native dim < 32  =>  d <= 31
RUN="${RUN:-all}"    # all | pretrain | finetune

_SELF="$ROOT/utils/run_joint_small_datasets.sh"

# Stdlib-only discovery for the login submit path (no torch / no venv).
discover_small_dataset_rows() {
  MAX_V="$1" ROOT="$2" python3 <<'PY'
import csv, os, sys

REGISTRY = {
    "ETTh1": ("ETT-small/ETTh1.csv", "date"),
    "ETTh2": ("ETT-small/ETTh2.csv", "date"),
    "ETTm1": ("ETT-small/ETTm1.csv", "date"),
    "ETTm2": ("ETT-small/ETTm2.csv", "date"),
    "illness": ("illness/national_illness.csv", "date"),
    "exchange_rate": ("exchange_rate/exchange_rate.csv", "date"),
    "electricity": ("electricity/electricity.csv", "date"),
    "traffic": ("traffic/traffic.csv", "date"),
    "PeMS": ("PeMS/PeMS.csv", "Time"),
    "solar_Alabama": ("solar_Alabama/solar_Alabama.csv", "Unnamed: 0"),
}

max_v = int(os.environ["MAX_V"])
root = os.environ["ROOT"]
datasets_dir = os.path.join(root, "datasets")

for name in sorted(REGISTRY.keys()):
    rel, date_col = REGISTRY[name]
    path = os.path.join(datasets_dir, rel)
    if not os.path.isfile(path):
        print(f"# skip (missing file): {name}", file=sys.stderr)
        continue
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader)
    except Exception as e:
        print(f"# skip {name}: {e}", file=sys.stderr)
        continue
    n = sum(1 for c in header if c.strip() != date_col)
    if n <= max_v:
        print(f"{name}\t{n}")
PY
}

# ---------------------------------------------------------------------------
# Single Slurm/local worker (one pretrain or one finetune) — internal use.
# ---------------------------------------------------------------------------
if [[ -n "${JOINT_SLURM_TASK:-}" ]]; then
  if [[ -z "${SLURM_JOB_ID:-}" && "${JOINT_SMALL_LOCAL:-0}" != "1" ]]; then
    echo "ERROR: JOINT_SLURM_TASK is internal (set by parallel sbatch). Unset it, or use JOINT_SMALL_LOCAL=1 for a local single task." >&2
    exit 1
  fi
  case "${JOINT_SLURM_TASK}" in
    pretrain)
      [[ -n "${JOINT_SLURM_DIM:-}" ]] || { echo "ERROR: JOINT_SLURM_DIM missing for pretrain worker" >&2; exit 1; }
      _stem_suffix="joint-pre-d${JOINT_SLURM_DIM}"
      ;;
    finetune)
      [[ -n "${JOINT_SLURM_DATASET:-}" && -n "${JOINT_SLURM_DIM:-}" ]] || {
        echo "ERROR: JOINT_SLURM_DATASET and JOINT_SLURM_DIM required for finetune worker" >&2
        exit 1
      }
      _stem_suffix="joint-ft-${JOINT_SLURM_DATASET}"
      ;;
    *)
      echo "ERROR: JOINT_SLURM_TASK must be pretrain or finetune" >&2
      exit 1
      ;;
  esac

  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    STEM="$(date +%m-%d)-${SLURM_JOB_ID: -4}-${_stem_suffix}"
    mkdir -p "$ROOT/results/logs"
    touch "$ROOT/results/logs/${STEM}.log"
    exec >>"$ROOT/results/logs/${STEM}.log" 2>&1
    echo "=========================================="
    echo "Job ID: $SLURM_JOB_ID  task=${JOINT_SLURM_TASK}  log: ./results/logs/${STEM}.log"
    echo "Node: ${SLURMD_NODENAME:-?}  started: $(date -Is)"
    echo "=========================================="
  fi

  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    module purge || true
    module load StdEnv/2023
    module load python/3.11
    module load cuda/12.2
    module load cudnn/8.9
  fi

  ACTIVATED_VENV_DIR=""
  _pick_python_in() {
    local d="$1"
    [[ -z "$d" ]] && return 1
    if [[ -x "$d/bin/python3" ]]; then printf '%s\n' "$d/bin/python3"; return 0; fi
    if [[ -x "$d/bin/python"  ]]; then printf '%s\n' "$d/bin/python";  return 0; fi
    return 1
  }
  resolve_venv_python() {
    local py
    if py="$(_pick_python_in "${ACTIVATED_VENV_DIR:-}")"; then printf '%s\n' "$py"; return 0; fi
    if py="$(_pick_python_in "${VIRTUAL_ENV:-}")";        then printf '%s\n' "$py"; return 0; fi
    return 1
  }
  activate_training_venv() {
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
      if py="$(resolve_venv_python)" && [[ -n "$py" ]] && "$py" -c "import torch" 2>/dev/null; then
        echo "Using already-active venv: $VIRTUAL_ENV"
        return 0
      fi
    fi
    if [[ -n "${VENV_PATH:-}" && -f "${VENV_PATH}/bin/activate" ]]; then
      # shellcheck source=/dev/null
      source "${VENV_PATH}/bin/activate"
      ACTIVATED_VENV_DIR="${VENV_PATH}"
      echo "Activated VENV_PATH=$VENV_PATH"
      return 0
    fi
    if [[ -f "$ROOT/.venv/bin/activate" ]]; then
      # shellcheck source=/dev/null
      source "$ROOT/.venv/bin/activate"
      ACTIVATED_VENV_DIR="$ROOT/.venv"
      echo "Activated repo .venv"
      return 0
    fi
    if [[ -z "${PROJECT:-}" ]] && [[ -d "$HOME/projects" ]]; then
      shopt -s nullglob
      local _m=("$HOME"/projects/def-* "$HOME"/projects/aip-*)
      shopt -u nullglob
      if [[ "${#_m[@]}" -gt 0 ]]; then
        export PROJECT
        PROJECT="$(readlink -f "${_m[0]}")"
      fi
    fi
    if [[ -n "${PROJECT:-}" ]]; then
      local pv="$PROJECT/${USER:-}/diffusion-tsf/venv"
      if [[ ! -d "$pv" ]]; then
        pv="$PROJECT/${USER:-}/diffusion-tsf-fullvar/venv"
      fi
      if [[ -f "$pv/bin/activate" ]]; then
        # shellcheck source=/dev/null
        source "$pv/bin/activate"
        ACTIVATED_VENV_DIR="$pv"
        echo "Activated cluster venv: $pv"
        return 0
      fi
    fi
    return 1
  }

  if ! activate_training_venv; then
    echo "ERROR: No venv with PyTorch found." >&2
    exit 1
  fi
  PY="$(resolve_venv_python)" || true
  if [[ -z "${PY:-}" ]] || ! "$PY" -c "import torch" 2>/dev/null; then
    echo "ERROR: venv / torch not usable (PY=${PY:-})" >&2
    exit 1
  fi

  run_py() { "$PY" -u -m models.diffusion_tsf.train_multivariate_pipeline "$@"; }

  case "${JOINT_SLURM_TASK}" in
    pretrain)
      echo "========== joint pretrain dim=${JOINT_SLURM_DIM} =========="
      run_py --mode pretrain --n-variates "${JOINT_SLURM_DIM}" "$@"
      ;;
    finetune)
      echo "========== joint finetune dataset=${JOINT_SLURM_DATASET} n_variates=${JOINT_SLURM_DIM} =========="
      run_py --mode finetune --dataset "${JOINT_SLURM_DATASET}" --n-variates "${JOINT_SLURM_DIM}" "$@"
      ;;
  esac
  echo "Done (worker)."
  exit 0
fi

# ---------------------------------------------------------------------------
# Login node: submit parallel Slurm jobs (default).
# JOINT_SMALL_SEQUENTIAL=1 → one job that runs everything in order (old behavior).
# ---------------------------------------------------------------------------
if [[ -z "${SLURM_JOB_ID:-}" && "${JOINT_SMALL_LOCAL:-0}" != "1" ]] && command -v sbatch >/dev/null 2>&1 && [[ "${JOINT_SMALL_SEQUENTIAL:-0}" != "1" ]]; then
  IS_SMOKE=0
  for _a in "$@"; do [[ "$_a" == "--smoke-test" ]] && IS_SMOKE=1; done
  ACCT="${SBATCH_ACCOUNT:-${SLURM_ACCOUNT:-aip-boyuwang}}"
  mkdir -p "$ROOT/results/logs" "$ROOT/results/bootstrap"

  mapfile -t ROWS < <(discover_small_dataset_rows "$MAX_V" "$ROOT")
  if [[ ${#ROWS[@]} -eq 0 ]]; then
    echo "No datasets with dim <= $MAX_V (or discovery failed)." >&2
    exit 1
  fi

  echo "Small datasets (dim <= $MAX_V):"
  printf '  %s\n' "${ROWS[@]}"
  echo ""

  if [[ "$IS_SMOKE" -eq 1 ]]; then
    _time=(--time=0:45:00)
    _mem=(--mem=16G)
    _cpus=(--cpus-per-task=4)
  else
    _time=(--time=2-00:00:00)
    _mem=(--mem=50G)
    _cpus=(--cpus-per-task=8)
  fi

  declare -A PRETRAIN_JID=()
  dims=$(printf '%s\n' "${ROWS[@]}" | cut -f2 | sort -u)

  if [[ "$RUN" == all || "$RUN" == pretrain ]]; then
    echo "Submitting parallel joint **pretrain** jobs (one per dim) account=$ACCT ..."
    for d in $dims; do
      jid="$(sbatch --parsable \
        --job-name="joint-pre-d${d}" \
        --account="$ACCT" \
        "${_time[@]}" \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        "${_cpus[@]}" \
        "${_mem[@]}" \
        --chdir="$ROOT" \
        --output=/dev/null \
        --error=/dev/null \
        --export=ALL,JOINT_SLURM_TASK=pretrain,JOINT_SLURM_DIM="${d}" \
        "$_SELF" "$@")"
      PRETRAIN_JID[$d]="$jid"
      echo "  pretrain dim=$d -> job $jid"
    done
  fi

  if [[ "$RUN" == all || "$RUN" == finetune ]]; then
    echo "Submitting parallel joint **finetune** jobs (one per dataset) account=$ACCT ..."
    for row in "${ROWS[@]}"; do
      name="$(printf '%s\n' "$row" | cut -f1)"
      dim="$(printf '%s\n' "$row" | cut -f2)"
      _dep=()
      if [[ "$RUN" == all && -n "${PRETRAIN_JID[$dim]:-}" ]]; then
        _dep=(--dependency="afterok:${PRETRAIN_JID[$dim]}")
      fi
      jid="$(sbatch --parsable \
        "${_dep[@]}" \
        --job-name="joint-ft-${name}" \
        --account="$ACCT" \
        "${_time[@]}" \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        "${_cpus[@]}" \
        "${_mem[@]}" \
        --chdir="$ROOT" \
        --output=/dev/null \
        --error=/dev/null \
        --export=ALL,JOINT_SLURM_TASK=finetune,JOINT_SLURM_DATASET="${name}",JOINT_SLURM_DIM="${dim}" \
        "$_SELF" "$@")"
      echo "  finetune $name (dim=$dim) -> job $jid${PRETRAIN_JID[$dim]:+ (after pretrain ${PRETRAIN_JID[$dim]})}"
    done
  fi

  echo ""
  echo "Submitted jobs listed above. Examples:"
  echo "  sacct --format=JobID,JobName%20,State,Elapsed,ExitCode -S \$(date -d yesterday +%F)"
  echo "  Logs under ./results/logs/ once each job starts."
  exit 0
fi

# Login: one Slurm job that runs the full sequential pipeline (no JOINT_SLURM_*).
if [[ -z "${SLURM_JOB_ID:-}" && "${JOINT_SMALL_LOCAL:-0}" != "1" ]] && command -v sbatch >/dev/null 2>&1 && [[ "${JOINT_SMALL_SEQUENTIAL:-0}" == "1" ]]; then
  IS_SMOKE=0
  for _a in "$@"; do [[ "$_a" == "--smoke-test" ]] && IS_SMOKE=1; done
  ACCT="${SBATCH_ACCOUNT:-${SLURM_ACCOUNT:-aip-boyuwang}}"
  mkdir -p "$ROOT/results/logs" "$ROOT/results/bootstrap"
  if [[ "$IS_SMOKE" -eq 1 ]]; then
    echo "Submitting single sequential Slurm smoke job (L40S, ~45 min) account=$ACCT ..."
    jid="$(sbatch --parsable \
      --job-name=joint-small-seq-smoke \
      --account="$ACCT" \
      --time=0:45:00 \
      --nodes=1 \
      --gres=gpu:l40s:1 \
      --cpus-per-task=4 \
      --mem=16G \
      --chdir="$ROOT" \
      --output=/dev/null \
      --error=/dev/null \
      --export=ALL \
      "$_SELF" "$@")"
  else
    echo "Submitting single sequential Slurm job (L40S, 2 d wall) account=$ACCT ..."
    jid="$(sbatch --parsable \
      --job-name=joint-small-seq \
      --account="$ACCT" \
      --time=2-00:00:00 \
      --nodes=1 \
      --gres=gpu:l40s:1 \
      --cpus-per-task=8 \
      --mem=50G \
      --chdir="$ROOT" \
      --output=/dev/null \
      --error=/dev/null \
      --export=ALL \
      "$_SELF" "$@")"
  fi
  echo "Submitted job id: $jid (sequential pretrain+finetune in one job)"
  echo "  sacct -j $jid --format=JobID,State,Elapsed,MaxRSS,ExitCode"
  exit 0
fi

# ---------------------------------------------------------------------------
# One sequential Slurm job (JOINT_SMALL_SEQUENTIAL=1) or local / no sbatch
# ---------------------------------------------------------------------------
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  STEM="$(date +%m-%d)-${SLURM_JOB_ID: -4}-joint-small-datasets"
  mkdir -p "$ROOT/results/logs"
  touch "$ROOT/results/logs/${STEM}.log"
  exec >>"$ROOT/results/logs/${STEM}.log" 2>&1
  echo "=========================================="
  echo "Job ID: $SLURM_JOB_ID  log: ./results/logs/${STEM}.log"
  echo "Node: ${SLURMD_NODENAME:-?}  started: $(date -Is)"
  echo "=========================================="
fi

# ---------------------------------------------------------------------------
# Environment (torch required — avoid importing train_multivariate_pipeline for discovery)
# ---------------------------------------------------------------------------

ACTIVATED_VENV_DIR=""

_pick_python_in() {
  local d="$1"
  [[ -z "$d" ]] && return 1
  if [[ -x "$d/bin/python3" ]]; then printf '%s\n' "$d/bin/python3"; return 0; fi
  if [[ -x "$d/bin/python"  ]]; then printf '%s\n' "$d/bin/python";  return 0; fi
  return 1
}

resolve_venv_python() {
  local py
  if py="$(_pick_python_in "${ACTIVATED_VENV_DIR:-}")"; then printf '%s\n' "$py"; return 0; fi
  if py="$(_pick_python_in "${VIRTUAL_ENV:-}")";        then printf '%s\n' "$py"; return 0; fi
  return 1
}

activate_training_venv() {
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    if py="$(resolve_venv_python)" && [[ -n "$py" ]] && "$py" -c "import torch" 2>/dev/null; then
      echo "Using already-active venv: $VIRTUAL_ENV"
      return 0
    fi
  fi
  if [[ -n "${VENV_PATH:-}" && -f "${VENV_PATH}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${VENV_PATH}/bin/activate"
    ACTIVATED_VENV_DIR="${VENV_PATH}"
    echo "Activated VENV_PATH=$VENV_PATH"
    return 0
  fi
  if [[ -f "$ROOT/.venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$ROOT/.venv/bin/activate"
    ACTIVATED_VENV_DIR="$ROOT/.venv"
    echo "Activated repo .venv"
    return 0
  fi
  if [[ -z "${PROJECT:-}" ]] && [[ -d "$HOME/projects" ]]; then
    shopt -s nullglob
    local _m=("$HOME"/projects/def-* "$HOME"/projects/aip-*)
    shopt -u nullglob
    if [[ "${#_m[@]}" -gt 0 ]]; then
      export PROJECT
      PROJECT="$(readlink -f "${_m[0]}")"
    fi
  fi
  if [[ -n "${PROJECT:-}" ]]; then
    local pv="$PROJECT/${USER:-}/diffusion-tsf/venv"
    if [[ ! -d "$pv" ]]; then
      pv="$PROJECT/${USER:-}/diffusion-tsf-fullvar/venv"
    fi
    if [[ -f "$pv/bin/activate" ]]; then
      # shellcheck source=/dev/null
      source "$pv/bin/activate"
      ACTIVATED_VENV_DIR="$pv"
      echo "Activated cluster venv: $pv"
      return 0
    fi
  fi
  return 1
}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  module purge || true
  module load StdEnv/2023
  module load python/3.11
  module load cuda/12.2
  module load cudnn/8.9
fi

if ! activate_training_venv; then
  echo "ERROR: No venv with PyTorch found." >&2
  echo "  Create repo .venv with torch, or set VENV_PATH=.../bin/activate parent dir," >&2
  echo "  or use the persistent venv under \$PROJECT/\$USER/diffusion-tsf/venv (see run.sh)." >&2
  exit 1
fi

PY="$(resolve_venv_python)" || true
if [[ -z "${PY:-}" ]]; then
  echo "ERROR: Could not find a python in the activated venv." >&2
  echo "  ACTIVATED_VENV_DIR=${ACTIVATED_VENV_DIR:-(unset)}" >&2
  echo "  VIRTUAL_ENV=${VIRTUAL_ENV:-(unset)}" >&2
  if [[ -n "${ACTIVATED_VENV_DIR:-}" && -d "$ACTIVATED_VENV_DIR/bin" ]]; then
    echo "  Contents of $ACTIVATED_VENV_DIR/bin:" >&2
    ls -la "$ACTIVATED_VENV_DIR/bin" >&2 || true
  fi
  exit 1
fi

if ! "$PY" -c "import torch" 2>/dev/null; then
  echo "ERROR: venv interpreter cannot import torch after activation." >&2
  echo "  ACTIVATED_VENV_DIR=${ACTIVATED_VENV_DIR:-(unset)}" >&2
  echo "  VIRTUAL_ENV=${VIRTUAL_ENV:-(unset)}" >&2
  echo "  Interpreter: $PY" >&2
  echo "  python3 from PATH (may differ): $(command -v python3 2>/dev/null || echo '(none)')" >&2
  echo "  Install torch into this venv (e.g. on a login node: source .../activate && pip install --no-index torch" >&2
  echo "    with python/3.11 loaded, or pip install torch), then re-run." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Discover small datasets — stdlib in $PY (must stay in sync with DATASET_REGISTRY
# in models/diffusion_tsf/train_multivariate_pipeline.py: path + date column).
# ---------------------------------------------------------------------------

mapfile -t ROWS < <(MAX_V="$MAX_V" ROOT="$ROOT" "$PY" <<'PY'
import csv, os, sys

REGISTRY = {
    "ETTh1": ("ETT-small/ETTh1.csv", "date"),
    "ETTh2": ("ETT-small/ETTh2.csv", "date"),
    "ETTm1": ("ETT-small/ETTm1.csv", "date"),
    "ETTm2": ("ETT-small/ETTm2.csv", "date"),
    "illness": ("illness/national_illness.csv", "date"),
    "exchange_rate": ("exchange_rate/exchange_rate.csv", "date"),
    "electricity": ("electricity/electricity.csv", "date"),
    "traffic": ("traffic/traffic.csv", "date"),
    "PeMS": ("PeMS/PeMS.csv", "Time"),
    "solar_Alabama": ("solar_Alabama/solar_Alabama.csv", "Unnamed: 0"),
}

max_v = int(os.environ["MAX_V"])
root = os.environ["ROOT"]
datasets_dir = os.path.join(root, "datasets")

for name in sorted(REGISTRY.keys()):
    rel, date_col = REGISTRY[name]
    path = os.path.join(datasets_dir, rel)
    if not os.path.isfile(path):
        print(f"# skip (missing file): {name}", file=sys.stderr)
        continue
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader)
    except Exception as e:
        print(f"# skip {name}: {e}", file=sys.stderr)
        continue
    n = sum(1 for c in header if c.strip() != date_col)
    if n <= max_v:
        print(f"{name}\t{n}")
PY
)

if [[ ${#ROWS[@]} -eq 0 ]]; then
  echo "No datasets with dim <= $MAX_V (or discovery failed)." >&2
  exit 1
fi

echo "Small datasets (dim <= $MAX_V):"
printf '  %s\n' "${ROWS[@]}"
echo ""

dims=$(printf '%s\n' "${ROWS[@]}" | cut -f2 | sort -u)

run_py() {
  "$PY" -u -m models.diffusion_tsf.train_multivariate_pipeline "$@"
}

if [[ "$RUN" == all || "$RUN" == pretrain ]]; then
  for d in $dims; do
    echo "========== joint pretrain dim=$d =========="
    run_py --mode pretrain --n-variates "$d" "$@"
  done
fi

if [[ "$RUN" == all || "$RUN" == finetune ]]; then
  for row in "${ROWS[@]}"; do
    name="$(printf '%s\n' "$row" | cut -f1)"
    dim="$(printf '%s\n' "$row" | cut -f2)"
    echo "========== joint finetune dataset=$name n_variates=$dim =========="
    run_py --mode finetune --dataset "$name" --n-variates "$dim" "$@"
  done
fi

echo "Done."
