# Shared module stack for classical baseline jobs (statsforecast needs pyarrow).
# Source after `deactivate` — arrow must load before any virtualenv is activated.
#
#   deactivate 2>/dev/null || true
#   unset VIRTUAL_ENV PYTHONHOME
#   source setup/killarney_classical_modules.sh
#   killarney_classical_modules || exit 1

killarney_classical_modules() {
    module purge 2>/dev/null || true
    if ! module load StdEnv/2023 python/3.11 2>/dev/null; then
        echo "ERROR: could not load StdEnv/2023 python/3.11" >&2
        return 1
    fi
    if ! module load gcc arrow 2>/dev/null; then
        echo "ERROR: could not load gcc/arrow (required for pyarrow)." >&2
        echo "Run: module spider arrow" >&2
        return 1
    fi
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        echo "ERROR: deactivate your venv before killarney_classical_modules (arrow/pyarrow order)." >&2
        echo "  deactivate && ./setup/killarney_bootstrap_classical_wheels.sh" >&2
        return 1
    fi
    return 0
}
