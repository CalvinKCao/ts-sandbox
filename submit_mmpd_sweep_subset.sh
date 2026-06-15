#!/bin/bash
# Train + eval MMPD using variate/stride subsets from binary-anchor checkpoint metadata.
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_mmpd_sweep_subset.sh --anchor-config binary_anchor_stationary_flat_subsets
#   ./submit_mmpd_sweep_subset.sh --smoke-test
#   ./submit_mmpd_sweep_subset.sh --output-dir results/datasets/06-12-sweep-subset-mmpd
#   ./submit_mmpd_sweep_subset.sh --resume --output-dir results/datasets/06-12-sweep-subset-mmpd
#   ./submit_mmpd_sweep_subset.sh --datasets ETTh1,ETTh2,exchange_rate,weather,electricity,traffic,solar_Alabama
#   ./submit_mmpd_sweep_subset.sh --anchor-config binary_anchor_stationary_flat_subsets_ema099_lb336_hz96 \
#       --lookback 336 --horizon 96 --datasets ETTh1,exchange_rate,weather,traffic
#
# MMPD-only: does not submit binary-anchor re-eval workers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE=0
RESUME=0
FORCE=0
OUTPUT_DIR=""
ANCHOR_CONFIG="binary_anchor_stationary_flat_subsets"
DATASETS_CSV="ETTh1,ETTh2,exchange_rate,weather,electricity,traffic,solar_Alabama"
DEPENDENCY=""
SEED=2026
LOOKBACK=96
HORIZON=96
WALL_MMPD="3:00:00"
WALL_INIT="0:45:00"
WALL_MERGE="0:30:00"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) SMOKE=1; shift ;;
        --resume) RESUME=1; shift ;;
        --force) FORCE=1; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --anchor-config) ANCHOR_CONFIG="$2"; shift 2 ;;
        --datasets) DATASETS_CSV="$2"; shift 2 ;;
        --dependency) DEPENDENCY="$2"; shift 2 ;;
        --lookback) LOOKBACK="$2"; shift 2 ;;
        --horizon) HORIZON="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --time) WALL_MMPD="$2"; shift 2 ;;
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

pick_anchor_root() {
    local ds="$1"
    local matches=()
    shopt -s nullglob
    matches=( "$REPO/results/ckpts"/*-"${ds}"-"${ANCHOR_CONFIG}" )
    shopt -u nullglob
    if [[ ${#matches[@]} -eq 0 ]]; then
        return 1
    fi
    printf '%s\n' "${matches[@]}" | sort | tail -1
}

pick_resume_output_dir() {
    local matches=()
    shopt -s nullglob
    matches=( "$REPO/results/datasets"/*-sweep-subset-mmpd "$REPO/results/datasets"/*-sweep-subset-mmpd-smoke )
    shopt -u nullglob
    if [[ ${#matches[@]} -eq 0 ]]; then
        echo "ERROR: --resume but no results/datasets/*-sweep-subset-mmpd found; pass --output-dir" >&2
        exit 1
    fi
    printf '%s\n' "${matches[@]}" | sort | tail -1
}

ANCHOR_ROOTS=()
for ds in "${DATASETS[@]}"; do
    if root="$(pick_anchor_root "$ds")"; then
        ANCHOR_ROOTS+=( "$root" )
    else
        ANCHOR_ROOTS+=( "(pending: *-${ds}-${ANCHOR_CONFIG})" )
    fi
done

if [[ -z "$OUTPUT_DIR" ]]; then
    if [[ "$RESUME" -eq 1 ]]; then
        OUTPUT_DIR="$(pick_resume_output_dir)"
    else
        RUN_STEM="$(date +%m-%d)-$$-sweep-subset-mmpd$([[ "$SMOKE" -eq 1 ]] && echo -smoke)"
        OUTPUT_DIR="$REPO/results/datasets/${RUN_STEM}"
    fi
else
    [[ "$OUTPUT_DIR" != /* ]] && OUTPUT_DIR="$REPO/$OUTPUT_DIR"
fi
RUN_STEM="$(basename "$OUTPUT_DIR")"
LOG_DIR="$REPO/results/logs/${RUN_STEM}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

EVAL_BASE=(
    "$REPO/utils/eval_mmpd_gaussian_anchor.py"
    --output-dir "$OUTPUT_DIR"
    --ckpt-base "$REPO/results/ckpts"
    --anchor-config "$ANCHOR_CONFIG"
    --lookback "$LOOKBACK"
    --horizon "$HORIZON"
    --mmpd-repo "$REPO/temp/MMPD"
    --mmpd-data-dir "$REPO/temp/mmpd_datasets"
    --seed "$SEED"
    --no-update-mmpd
    --force-mmpd-eval
    --force-indices
)

if [[ "$FORCE" -eq 1 || "$RESUME" -eq 0 ]]; then
    EVAL_BASE+=(--force-mmpd-train)
fi

if [[ "$SMOKE" -eq 1 ]]; then
    WALL_MMPD="0:45:00"
    WALL_INIT="0:25:00"
    WALL_MERGE="0:15:00"
    MEM="24G"
    CPUS=4
    EVAL_EXTRA=(
        --mmpd-train-epochs 1
        --mmpd-patience 1
        --test-fraction 0.02
        --test-max-items 32
        --sample-num 5
        --num-sampling-steps 5
        --topk-max 3
        --gmm-components 5
        --gmm-iterations 3
        --mmpd-batch-size 16
        --mmpd-eval-batch-size 4
    )
    DATASETS=(ETTh1)
    EVAL_BASE=(
        "$REPO/utils/eval_mmpd_gaussian_anchor.py"
        --output-dir "$OUTPUT_DIR"
        --ckpt-base "$REPO/results/ckpts"
        --anchor-config "$ANCHOR_CONFIG"
        --lookback "$LOOKBACK"
        --horizon "$HORIZON"
        --mmpd-repo "$REPO/temp/MMPD"
        --mmpd-data-dir "$REPO/temp/mmpd_datasets"
        --seed "$SEED"
        --no-update-mmpd
        --force-mmpd-eval
        --force-indices
    )
    if [[ "$FORCE" -eq 1 || "$RESUME" -eq 0 ]]; then
        EVAL_BASE+=(--force-mmpd-train)
    fi
else
    MEM="60G"
    CPUS=8
    EVAL_EXTRA=(
        --mmpd-train-epochs 20
        --mmpd-patience 5
        --test-fraction 1.0
        --sample-num 20
        --num-sampling-steps 20
        --topk-max 3
        --gmm-components 10
        --gmm-iterations 10
        --mmpd-batch-size 32
        --mmpd-eval-batch-size 16
    )
fi

PREAMBLE_FILE="$REPO/results/job_preamble_mmpd_sweep_subset.sh"
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
"\$PYTHON" -c "import torch, optuna, wandb, einops, yaml; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

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
    --mem="$MEM"
    --gres=gpu:l40s:1
    --mail-type=FAIL
    --mail-user=ccao87@uwo.ca
)

echo "Repo:          $REPO"
echo "Output:        $OUTPUT_DIR"
echo "Anchor config: $ANCHOR_CONFIG"
echo "Lookback/horizon: $LOOKBACK / $HORIZON"
echo "Resume:        $RESUME  Force: $FORCE"
echo "Datasets:      ${DATASETS[*]}"
for i in "${!DATASETS[@]}"; do
    echo "  ${DATASETS[$i]} <- ${ANCHOR_ROOTS[$i]}"
done

SKIP_INIT=0
if [[ "$RESUME" -eq 1 && -f "$OUTPUT_DIR/run_manifest.json" ]]; then
    SKIP_INIT=1
    echo "Resume: reusing $OUTPUT_DIR/run_manifest.json"
fi

JOB_INIT=""
WORKER_DEP=()
INIT_SBATCH_EXTRA=()
if [[ -n "$DEPENDENCY" ]]; then
    INIT_SBATCH_EXTRA=(--dependency="$DEPENDENCY")
fi
if [[ "$SKIP_INIT" -eq 0 ]]; then
    INIT_SCRIPT="$LOG_DIR/submit-init.sh"
    write_worker_script "$INIT_SCRIPT" "${EVAL_BASE[@]}" "${EVAL_EXTRA[@]}" \
        --phase init --datasets "${DATASETS[@]}"
    echo "Submitting init..."
    JOB_INIT=$(sbatch --parsable \
        --job-name="mmpd-sw-init$([[ "$SMOKE" -eq 1 ]] && echo -smoke)" \
        "${SBATCH_COMMON[@]}" \
        --time="$WALL_INIT" \
        "${INIT_SBATCH_EXTRA[@]}" \
        --output="$LOG_DIR/init-%j.out" \
        --error="$LOG_DIR/init-%j.err" \
        "$INIT_SCRIPT")
    echo "  -> init: $JOB_INIT"
    WORKER_DEP=(--dependency="afterok:$JOB_INIT")
fi

WORKER_IDS=()
PENDING_DATASETS=()
for ds in "${DATASETS[@]}"; do
    partial="$OUTPUT_DIR/partials/${ds}_mmpd.json"
    if [[ "$RESUME" -eq 1 && "$FORCE" -eq 0 && -f "$partial" ]]; then
        echo "Skip mmpd-${ds}: partial exists ($partial)"
        continue
    fi
    PENDING_DATASETS+=("$ds")

    WORKER_SCRIPT="$LOG_DIR/submit-mmpd-${ds}.sh"
    write_worker_script "$WORKER_SCRIPT" "${EVAL_BASE[@]}" "${EVAL_EXTRA[@]}" \
        --phase mmpd --datasets "$ds"
    echo "Submitting mmpd-${ds} ${WORKER_DEP[*]}..."
    JOB_MMPD=$(sbatch --parsable \
        --job-name="mmpd-sw-${ds}$([[ "$SMOKE" -eq 1 ]] && echo -smoke)" \
        "${SBATCH_COMMON[@]}" \
        --time="$WALL_MMPD" \
        "${WORKER_DEP[@]}" \
        --output="$LOG_DIR/mmpd-${ds}-%j.out" \
        --error="$LOG_DIR/mmpd-${ds}-%j.err" \
        "$WORKER_SCRIPT")
    echo "  -> mmpd-${ds}: $JOB_MMPD"
    WORKER_IDS+=("$JOB_MMPD")
done

if [[ ${#WORKER_IDS[@]} -eq 0 && ${#PENDING_DATASETS[@]} -eq 0 ]]; then
    echo "All dataset partials present; submitting merge only."
    MERGE_DEP_ARGS=()
elif [[ ${#WORKER_IDS[@]} -eq 0 ]]; then
    echo "ERROR: nothing to submit (no pending datasets and no workers)." >&2
    exit 1
else
    MERGE_DEP="afterok:${WORKER_IDS[0]}"
    for wid in "${WORKER_IDS[@]:1}"; do
        MERGE_DEP+=":$wid"
    done
    MERGE_DEP_ARGS=(--dependency="$MERGE_DEP")
fi

MERGE_SCRIPT="$LOG_DIR/submit-merge.sh"
write_worker_script "$MERGE_SCRIPT" "${EVAL_BASE[@]}" "${EVAL_EXTRA[@]}" \
    --phase merge --datasets "${DATASETS[@]}" --cpu

echo "Submitting merge ${MERGE_DEP_ARGS[*]}..."
JOB_MERGE=$(sbatch --parsable \
    --job-name="mmpd-sw-merge$([[ "$SMOKE" -eq 1 ]] && echo -smoke)" \
    --account="$ACCOUNT" \
    --nodes=1 \
    --cpus-per-task=2 \
    --mem=16G \
    --gres=gpu:l40s:1 \
    --time="$WALL_MERGE" \
    "${MERGE_DEP_ARGS[@]}" \
    --output="$LOG_DIR/merge-%j.out" \
    --error="$LOG_DIR/merge-%j.err" \
    --mail-type=FAIL \
    --mail-user=ccao87@uwo.ca \
    "$MERGE_SCRIPT")
echo "  -> merge: $JOB_MERGE"

echo ""
echo "=================================================================="
echo "  MMPD sweep-subset submitted"
if [[ -n "$JOB_INIT" ]]; then echo "  init:  $JOB_INIT"; fi
echo "  workers: ${#WORKER_IDS[@]} pending dataset(s): ${PENDING_DATASETS[*]:-none}"
echo "  merge: $JOB_MERGE"
echo "  Output: $OUTPUT_DIR"
echo "  Logs:   $LOG_DIR/"
echo "  Monitor: squeue -u \$USER"
echo "=================================================================="
