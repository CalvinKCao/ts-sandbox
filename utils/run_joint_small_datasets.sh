#!/bin/bash
# Batch joint pretrain + finetune for datasets with native dimensionality < 32.
#
# Alliance / Killarney: run from repo root on scratch (not /home for GPU jobs).
# This script activates a venv with PyTorch before calling the pipeline — same
# idea as run.sh (PROJECT venv) and visualize_locally.sh (.venv).
#
# Usage (repo root):
#   ./utils/run_joint_small_datasets.sh
#   ./utils/run_joint_small_datasets.sh --smoke-test
#   RUN=finetune ./utils/run_joint_small_datasets.sh
#   VENV_PATH=/path/to/venv ./utils/run_joint_small_datasets.sh
#
# Extra args are forwarded to every python invocation.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$ROOT"

MAX_V="${MAX_V:-31}" # native dim < 32  =>  d <= 31
RUN="${RUN:-all}"    # all | pretrain | finetune

# ---------------------------------------------------------------------------
# Environment (torch required — avoid importing train_multivariate_pipeline for discovery)
# ---------------------------------------------------------------------------

activate_training_venv() {
  if [[ -n "${VIRTUAL_ENV:-}" ]] && python3 -c "import torch" 2>/dev/null; then
    echo "Using already-active venv: $VIRTUAL_ENV"
    return 0
  fi
  if [[ -n "${VENV_PATH:-}" && -f "${VENV_PATH}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${VENV_PATH}/bin/activate"
    echo "Activated VENV_PATH=$VENV_PATH"
    return 0
  fi
  if [[ -f "$ROOT/.venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$ROOT/.venv/bin/activate"
    echo "Activated repo .venv"
    return 0
  fi
  # Match run.sh: persistent venv under $PROJECT/$USER/…
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

if ! python3 -c "import torch" 2>/dev/null; then
  echo "ERROR: python3 in PATH cannot import torch after venv activation." >&2
  echo "  Fix the active environment, then re-run." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Discover small datasets — stdlib only (must stay in sync with DATASET_REGISTRY
# in models/diffusion_tsf/train_multivariate_pipeline.py: path + date column).
# ---------------------------------------------------------------------------

mapfile -t ROWS < <(MAX_V="$MAX_V" ROOT="$ROOT" python3 <<'PY'
import csv, os, sys

# (relative_path_under_datasets/, date_column_name)
REGISTRY = {
    "ETTh1": ("ETT-small/ETTh1.csv", "date"),
    "ETTh2": ("ETT-small/ETTh2.csv", "date"),
    "ETTm1": ("ETT-small/ETTm1.csv", "date"),
    "ETTm2": ("ETT-small/ETTm2.csv", "date"),
    "illness": ("illness/national_illness.csv", "date"),
    "exchange_rate": ("exchange_rate/exchange_rate.csv", "date"),
    "weather": ("weather/weather.csv", "date"),
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
  python3 -u -m models.diffusion_tsf.train_multivariate_pipeline "$@"
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
