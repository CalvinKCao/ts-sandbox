#!/bin/bash
# MMPD paper reproduction: MaskAE backbone, appendix D.3 hyperparams.
#
# Full variate sets (no binary-anchor subsets). One Slurm worker per dataset;
# each worker trains+tests horizons 96/192/336/720 at T=336.
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_mmpd_maskae_paper.sh
#   ./submit_mmpd_maskae_paper.sh --smoke-test
#   ./submit_mmpd_maskae_paper.sh --output-dir results/datasets/06-15-mmpd-paper-maskae
#   ./submit_mmpd_maskae_paper.sh --resume --output-dir results/datasets/06-15-mmpd-paper-maskae
#   ./submit_mmpd_maskae_paper.sh --datasets ETTh1,weather,ECL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE=0
RESUME=0
FORCE=0
OUTPUT_DIR=""
DATASETS_CSV="ETTh1,ETTh2,ETTm1,ETTm2,weather,ECL,Traffic"
DEPENDENCY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) SMOKE=1; shift ;;
        --resume) RESUME=1; shift ;;
        --force) FORCE=1; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --datasets) DATASETS_CSV="$2"; shift 2 ;;
        --dependency) DEPENDENCY="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: run from login node, not inside a Slurm job." >&2
    exit 1
fi

if [[ "$(hostname)" == *"narval"* ]]; then
    ACCOUNT="def-boyuwang"
else
    ACCOUNT="aip-boyuwang"
fi

if [[ -d "${SCRATCH:-}/ts-sandbox" ]]; then
    REPO="${SCRATCH}/ts-sandbox"
elif [[ -d "$HOME/ts-sandbox" ]]; then
    REPO="$HOME/ts-sandbox"
else
    REPO="$SCRIPT_DIR"
fi
if [[ "$REPO" == /home/* ]]; then
    echo "ERROR: submit from \$SCRATCH/ts-sandbox on Killarney, not /home." >&2
    exit 1
fi
cd "$REPO"

IFS=',' read -ra DATASETS <<< "$DATASETS_CSV"

pick_resume_output_dir() {
    local matches=()
    shopt -s nullglob
    matches=( "$REPO/results/datasets"/*-mmpd-paper-maskae "$REPO/results/datasets"/*-mmpd-paper-maskae-smoke )
    shopt -u nullglob
    if [[ ${#matches[@]} -eq 0 ]]; then
        echo "ERROR: --resume but no results/datasets/*-mmpd-paper-maskae found; pass --output-dir" >&2
        exit 1
    fi
    printf '%s\n' "${matches[@]}" | sort | tail -1
}

if [[ -z "$OUTPUT_DIR" ]]; then
    if [[ "$RESUME" -eq 1 ]]; then
        OUTPUT_DIR="$(pick_resume_output_dir)"
    else
        RUN_STEM="$(date +%m-%d)-$$-mmpd-paper-maskae$([[ "$SMOKE" -eq 1 ]] && echo -smoke)"
        OUTPUT_DIR="$REPO/results/datasets/${RUN_STEM}"
    fi
else
    [[ "$OUTPUT_DIR" != /* ]] && OUTPUT_DIR="$REPO/$OUTPUT_DIR"
fi
RUN_STEM="$(basename "$OUTPUT_DIR")"
LOG_DIR="$REPO/results/logs/${RUN_STEM}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

RUNNER="$REPO/utils/run_mmpd_paper_maskae.py"
RUN_BASE=(
    "$RUNNER"
    --output-dir "$OUTPUT_DIR"
    --mmpd-repo "$REPO/temp/MMPD"
    --mmpd-data-dir "$REPO/temp/mmpd_paper_datasets"
    --no-update-mmpd
)
FORCE_FLAGS=()
if [[ "$FORCE" -eq 1 ]]; then
    FORCE_FLAGS=(--force-train --force-test)
fi

if [[ "$SMOKE" -eq 1 ]]; then
    WALL_INIT="0:20:00"
    WALL_WORKER="0:45:00"
    WALL_MERGE="0:10:00"
    MEM="24G"
    CPUS=4
    RUN_EXTRA=(--smoke-test)
    DATASETS=(ETTh1)
else
    WALL_INIT="0:30:00"
    WALL_MERGE="0:20:00"
    MEM="60G"
    CPUS=8
    RUN_EXTRA=()
fi

dataset_walltime() {
    case "$1" in
        ECL|Traffic) echo "12:00:00" ;;
        weather) echo "6:00:00" ;;
        *) echo "4:00:00" ;;
    esac
}

dataset_mem() {
    case "$1" in
        Traffic) echo "80G" ;;
        ECL) echo "72G" ;;
        *) echo "$MEM" ;;
    esac
}

PREAMBLE_FILE="$REPO/results/job_preamble_mmpd_paper_maskae.sh"
cat > "$PREAMBLE_FILE" << PREAMBLE
set -euo pipefail
echo "Job: \$SLURM_JOB_NAME  ID: \$SLURM_JOB_ID  Node: \${SLURMD_NODENAME:-unknown}"
echo "GPU: \$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "Started: \$(date)"

REPO="$REPO"
REQ="\$REPO/setup/requirements-killarney.txt"
[[ -f "\$REQ" ]] || { echo "ERROR: missing \$REQ — run ./setup/killarney_freeze_requirements.sh on login node" >&2; exit 1; }
[[ -n "\${SLURM_TMPDIR:-}" ]] || { echo "ERROR: SLURM_TMPDIR is not set." >&2; exit 1; }

module purge 2>/dev/null || true
module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9 2>/dev/null || true
command -v virtualenv >/dev/null || { echo "ERROR: virtualenv not available after module load." >&2; exit 1; }

echo "[setup] Building node-local venv on \$SLURM_TMPDIR from \$REQ"
virtualenv --no-download "\$SLURM_TMPDIR/env"
# shellcheck source=/dev/null
source "\$SLURM_TMPDIR/env/bin/activate"
export PYTHON="\$SLURM_TMPDIR/env/bin/python"
pip install --no-index --upgrade pip -q
pip install --no-index -r "\$REQ" -q
"\$PYTHON" -c "import torch, einops; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TS_SANDBOX_REPO="\$REPO"
export PYTHONPATH="\$REPO\${PYTHONPATH:+:\$PYTHONPATH}"
cd "\$REPO"
PREAMBLE

write_worker_script() {
    local path="$1"
    shift
    cat > "$path" << SCRIPT
#!/bin/bash
source "$PREAMBLE_FILE"
exec "\$PYTHON" -u $(printf '%q ' "$@")
SCRIPT
    chmod +x "$path"
}

SBATCH_COMMON=(
    --account="$ACCOUNT"
    --nodes=1
    --cpus-per-task="$CPUS"
    --gres=gpu:l40s:1
    --mail-type=FAIL
    --mail-user=ccao87@uwo.ca
)

echo "Repo:     $REPO"
echo "Output:   $OUTPUT_DIR"
echo "Datasets: ${DATASETS[*]}"
echo "Smoke:    $SMOKE  Resume: $RESUME  Force: $FORCE"

SKIP_INIT=0
if [[ "$RESUME" -eq 1 && -f "$OUTPUT_DIR/run_manifest.json" ]]; then
    SKIP_INIT=1
    echo "Resume: reusing $OUTPUT_DIR/run_manifest.json"
fi

WORKER_DEP=()
INIT_SBATCH_EXTRA=()
if [[ -n "$DEPENDENCY" ]]; then
    INIT_SBATCH_EXTRA=(--dependency="$DEPENDENCY")
fi
if [[ "$SKIP_INIT" -eq 0 ]]; then
    INIT_SCRIPT="$LOG_DIR/submit-init.sh"
    write_worker_script "$INIT_SCRIPT" "${RUN_BASE[@]}" "${RUN_EXTRA[@]}" \
        --phase init --datasets "${DATASETS[@]}"
    echo "Submitting init..."
    JOB_INIT=$(sbatch --parsable \
        --job-name="mmpd-paper-init$([[ "$SMOKE" -eq 1 ]] && echo -smoke)" \
        "${SBATCH_COMMON[@]}" \
        --mem="$MEM" \
        --cpus-per-task=2 \
        --time="$WALL_INIT" \
        "${INIT_SBATCH_EXTRA[@]}" \
        --output="$LOG_DIR/init-%j.out" \
        --error="$LOG_DIR/init-%j.err" \
        "$INIT_SCRIPT")
    echo "  -> init: $JOB_INIT"
    WORKER_DEP=(--dependency="afterok:$JOB_INIT")
fi

WORKER_IDS=()
for ds in "${DATASETS[@]}"; do
    partial="$OUTPUT_DIR/partials/${ds}_maskae_paper.json"
    if [[ "$RESUME" -eq 1 && "$FORCE" -eq 0 && -f "$partial" ]]; then
        echo "Skip ${ds}: partial exists ($partial)"
        continue
    fi

    WALL="$(dataset_walltime "$ds")"
    DS_MEM="$(dataset_mem "$ds")"
    WORKER_SCRIPT="$LOG_DIR/submit-${ds}.sh"
    write_worker_script "$WORKER_SCRIPT" "${RUN_BASE[@]}" "${RUN_EXTRA[@]}" "${FORCE_FLAGS[@]}" \
        --phase run --datasets "$ds"
    echo "Submitting ${ds} (time=$WALL mem=$DS_MEM) ${WORKER_DEP[*]}..."
    JOB_ID=$(sbatch --parsable \
        --job-name="mmpd-paper-${ds}$([[ "$SMOKE" -eq 1 ]] && echo -smoke)" \
        "${SBATCH_COMMON[@]}" \
        --mem="$DS_MEM" \
        --time="$WALL" \
        "${WORKER_DEP[@]}" \
        --output="$LOG_DIR/${ds}-%j.out" \
        --error="$LOG_DIR/${ds}-%j.err" \
        "$WORKER_SCRIPT")
    echo "  -> ${ds}: $JOB_ID"
    WORKER_IDS+=("$JOB_ID")
done

if [[ ${#WORKER_IDS[@]} -eq 0 ]]; then
    echo "All partials present; merge only."
    MERGE_DEP_ARGS=()
else
    MERGE_DEP="afterok:${WORKER_IDS[0]}"
    for wid in "${WORKER_IDS[@]:1}"; do
        MERGE_DEP+=":$wid"
    done
    MERGE_DEP_ARGS=(--dependency="$MERGE_DEP")
fi

MERGE_SCRIPT="$LOG_DIR/submit-merge.sh"
write_worker_script "$MERGE_SCRIPT" "${RUN_BASE[@]}" \
    --phase merge --datasets "${DATASETS[@]}"
echo "Submitting merge ${MERGE_DEP_ARGS[*]}..."
JOB_MERGE=$(sbatch --parsable \
    --job-name="mmpd-paper-merge$([[ "$SMOKE" -eq 1 ]] && echo -smoke)" \
    "${SBATCH_COMMON[@]}" \
    --mem="8G" \
    --cpus-per-task=2 \
    --time="$WALL_MERGE" \
    "${MERGE_DEP_ARGS[@]}" \
    --output="$LOG_DIR/merge-%j.out" \
    --error="$LOG_DIR/merge-%j.err" \
    "$MERGE_SCRIPT")
echo "  -> merge: $JOB_MERGE"

echo ""
echo "MMPD paper MaskAE submitted"
echo "  Output: $OUTPUT_DIR"
echo "  Logs:   $LOG_DIR"
echo ""
echo "Hyperparams: T=336, horizons 96/192/336/720, P=12 (P=24 ECL/Traffic or tau 336/720)"
echo "  MaskAE backbone, point_weight=0.01 (lambda=0.99), N=100, K_infer=20, EM=10"
