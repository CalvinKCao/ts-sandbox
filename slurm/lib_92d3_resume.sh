# Shared path/resume helpers for 92d3 Slurm drivers. Source from repo root context.
# Usage: source "$SCRIPT_DIR/slurm/lib_92d3_resume.sh"
#   resolve_92d3_run_dirs ckpts_flat|run_bundle <slug> <dataset> <job_suffix_len>

resolve_92d3_run_dirs() {
    local layout="$1"
    local slug="$2"
    local dataset_lower="$3"
    local id_suffix_len="${4:-4}"

    if [[ -n "${RUN_STEM:-}" ]]; then
        :
    elif [[ "$FRESH" -eq 0 ]]; then
        local candidate=""
        if [[ "$layout" == "ckpts_flat" ]]; then
            shopt -s nullglob
            local dirs=(./results/ckpts/*-"${slug}")
            shopt -u nullglob
            for d in "${dirs[@]}"; do
                [[ -f "${d}/training_manifest.json" ]] && candidate="$(basename "$d")"
            done
        else
            shopt -s nullglob
            local dirs=(./results/*-binary-92d3-"${dataset_lower}")
            shopt -u nullglob
            for d in "${dirs[@]}"; do
                [[ -f "${d}/ckpts/training_manifest.json" ]] && candidate="$(basename "$d")"
            done
        fi
        if [[ -n "$candidate" ]]; then
            RUN_STEM="$candidate"
            echo "[resume] Reusing run stem: $RUN_STEM"
        fi
    fi

    if [[ -z "${RUN_STEM:-}" ]]; then
        RUN_STEM="$(date +%m-%d)-${SLURM_JOB_ID: -id_suffix_len}-${slug}"
        echo "[resume] New run stem: $RUN_STEM"
    fi

    if [[ "$layout" == "ckpts_flat" ]]; then
        LOG_FILE="./results/logs/${RUN_STEM}.log"
        CKPT_DIR="./results/ckpts/${RUN_STEM}"
        DATA_DIR="./results/datasets/${RUN_STEM}"
    else
        RUN_DIR="./results/${RUN_STEM}"
        LOG_DIR="${RUN_DIR}/logs"
        LOG_FILE="${LOG_DIR}/${RUN_STEM}.log"
        CKPT_DIR="${RUN_DIR}/ckpts"
        DATA_DIR="${RUN_DIR}/datasets"
    fi
}
