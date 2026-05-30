#!/usr/bin/env bash
# Run on the login node with a Python that has torch, numpy, and pandas.
# Tries results/venv (built by slurm_worker.sh grid jobs) before system python.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

_pick_venv() {
    local v
    for v in "${VENV_PATH:-}" "$ROOT/results/venv" "$ROOT/.venv"; do
        [[ -n "$v" && -x "$v/bin/python" ]] || continue
        echo "$v"
        return 0
    done
    return 1
}

_activate_if_needed() {
    if python -c "import numpy, pandas, torch" 2>/dev/null; then
        return 0
    fi
    local venv
    if venv="$(_pick_venv)"; then
        echo "[convert_dalia] Using venv: $venv"
        # shellcheck source=/dev/null
        source "$venv/bin/activate"
        if python -c "import numpy, pandas, torch" 2>/dev/null; then
            return 0
        fi
    fi
  echo "ERROR: need Python with numpy, pandas, and torch." >&2
  echo "  which python: $(command -v python || echo none)" >&2
  echo "" >&2
  if [[ -x "$ROOT/results/venv/bin/python" ]]; then
    echo "  source $ROOT/results/venv/bin/activate" >&2
  else
    echo "  Grid jobs create $ROOT/results/venv — submit one smoke job first, or:" >&2
    echo "    module load StdEnv/2023 python/3.11" >&2
    echo "    virtualenv --no-download $ROOT/results/venv" >&2
    echo "    source $ROOT/results/venv/bin/activate" >&2
    echo "    pip install torch numpy pandas" >&2
  fi
  exit 1
}

_activate_if_needed
exec python "$ROOT/setup/convert_dalia_to_csv.py" "$@"
