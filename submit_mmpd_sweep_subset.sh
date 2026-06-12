#!/bin/bash
# Train + eval MMPD on the same variate/stride subsets as sweep_baseline
# (ETTh1, ETTm1_4v_s3, exchange_rate, weather_4v_s2).
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_mmpd_sweep_subset.sh
#   ./submit_mmpd_sweep_subset.sh --smoke-test
#   ./submit_mmpd_sweep_subset.sh --output-dir results/datasets/06-12-sweep-subset-mmpd
#
# MMPD-only: does not submit binary-anchor re-eval workers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE=0
OUTPUT_DIR=""
ANCHOR_CONFIG="sweep_baseline"
SEED=2026
WALL_MMPD="12:00:00"
WALL_INIT="0:30:00"
WALL_MERGE="0:30:00"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) SMOKE=1; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --anchor-config) ANCHOR_CONFIG="$2"; shift 2 ;;
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

DATASETS=(ETTh1 ETTm1 exchange_rate weather)

promote_staged_for_mmpd() {
    local root="$1" sub stage
    for sub in "$root"/*/; do
        [[ -d "$sub" ]] || continue
        base=$(basename "$sub")
        [[ "$base" == .* ]] && continue
        if [[ -f "${sub}metadata.json" && -f "${sub}best.pt" ]]; then
            continue
        fi
        for stage in finer fine coarse; do
            if [[ -f "${sub}${stage}/best.pt" ]]; then
                ln -sfn "${stage}/best.pt" "${sub}best.pt"
                cp -f "${sub}${stage}/metadata.json" "${sub}metadata.json"
                echo "[promote] ${root##*/}/${base}/${stage} -> flat metadata+best.pt"
                break
            fi
        done
    done
}

pick_anchor_root() {
    local ds="$1"
    local matches=()
    shopt -s nullglob
    matches=( "$REPO/results/ckpts"/*-"${ds}"-"${ANCHOR_CONFIG}" )
    shopt -u nullglob
    if [[ ${#matches[@]} -eq 0 ]]; then
        echo "ERROR: no ckpt dir matching results/ckpts/*-${ds}-${ANCHOR_CONFIG}" >&2
        exit 1
    fi
    printf '%s\n' "${matches[@]}" | sort | tail -1
}

ANCHOR_ROOTS=()
for ds in "${DATASETS[@]}"; do
    root=$(pick_anchor_root "$ds")
    promote_staged_for_mmpd "$root"
    ANCHOR_ROOTS+=( "$root" )
done

if [[ -z "$OUTPUT_DIR" ]]; then
    RUN_STEM="$(date +%m-%d)-$$-sweep-subset-mmpd$([[ "$SMOKE" -eq 1 ]] && echo -smoke)"
    OUTPUT_DIR="$REPO/results/datasets/${RUN_STEM}"
else
    [[ "$OUTPUT_DIR" != /* ]] && OUTPUT_DIR="$REPO/$OUTPUT_DIR"
    RUN_STEM="$(basename "$OUTPUT_DIR")"
fi
LOG_DIR="$REPO/results/logs/${RUN_STEM}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

EVAL_BASE=(
    "$REPO/utils/eval_mmpd_gaussian_anchor.py"
    --output-dir "$OUTPUT_DIR"
    --ckpt-base "$REPO/results/ckpts"
    --mmpd-repo "$REPO/temp/MMPD"
    --mmpd-data-dir "$REPO/temp/mmpd_datasets"
    --seed "$SEED"
    --no-update-mmpd
    --force-mmpd-train
    --force-mmpd-eval
    --force-indices
)

for root in "${ANCHOR_ROOTS[@]}"; do
    EVAL_BASE+=(--binary-anchor-root "$root")
done

if [[ "$SMOKE" -eq 1 ]]; then
    WALL_MMPD="0:45:00"
    WALL_INIT="0:20:00"
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
    ANCHOR_ROOTS=( "$(pick_anchor_root ETTh1)" )
    EVAL_BASE=(
        "$REPO/utils/eval_mmpd_gaussian_anchor.py"
        --output-dir "$OUTPUT_DIR"
        --ckpt-base "$REPO/results/ckpts"
        --mmpd-repo "$REPO/temp/MMPD"
        --mmpd-data-dir "$REPO/temp/mmpd_datasets"
        --seed "$SEED"
        --no-update-mmpd
        --force-mmpd-train
        --force-mmpd-eval
        --force-indices
        --binary-anchor-root "${ANCHOR_ROOTS[0]}"
    )
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
echo "Started: \$(date)"

USER="\${USER:-\$(whoami)}"
STORE="$REPO/results"
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
if [[ -z "\$VENV" ]]; then
  echo "ERROR: no results/venv found; create one on the login node first." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "\$VENV/bin/activate"
export PATH="\$VENV/bin:\$PATH"
export PYTHON="\$VENV/bin/python"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TS_SANDBOX_REPO="$REPO"
export PYTHONPATH="$REPO\${PYTHONPATH:+:\$PYTHONPATH}"
cd "$REPO"
PREAMBLE

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
echo "Datasets:      ${DATASETS[*]}"
for i in "${!DATASETS[@]}"; do
    echo "  ${DATASETS[$i]} <- ${ANCHOR_ROOTS[$i]}"
done

echo "Submitting init..."
JOB_INIT=$(sbatch --parsable \
    --job-name="mmpd-sw-init$([[ "$SMOKE" -eq 1 ]] && echo -smoke)" \
    "${SBATCH_COMMON[@]}" \
    --time="$WALL_INIT" \
    --output="$LOG_DIR/init-%j.out" \
    --error="$LOG_DIR/init-%j.err" \
    <<ENDSCRIPT
#!/bin/bash
source "$PREAMBLE_FILE"
"\$PYTHON" -u ${EVAL_BASE[@]} ${EVAL_EXTRA[@]} \
  --phase init \
  --datasets ${DATASETS[*]}
ENDSCRIPT
)
echo "  -> init: $JOB_INIT"

WORKER_IDS=()
for ds in "${DATASETS[@]}"; do
    echo "Submitting mmpd-${ds} (after init)..."
    JOB_MMPD=$(sbatch --parsable \
        --job-name="mmpd-sw-${ds}$([[ "$SMOKE" -eq 1 ]] && echo -smoke)" \
        "${SBATCH_COMMON[@]}" \
        --time="$WALL_MMPD" \
        --dependency="afterok:$JOB_INIT" \
        --output="$LOG_DIR/mmpd-${ds}-%j.out" \
        --error="$LOG_DIR/mmpd-${ds}-%j.err" \
        <<ENDSCRIPT
#!/bin/bash
source "$PREAMBLE_FILE"
"\$PYTHON" -u ${EVAL_BASE[@]} ${EVAL_EXTRA[@]} \
  --phase mmpd \
  --datasets "$ds"
ENDSCRIPT
    )
    echo "  -> mmpd-${ds}: $JOB_MMPD"
    WORKER_IDS+=("$JOB_MMPD")
done

MERGE_DEP="afterok:${WORKER_IDS[0]}"
for wid in "${WORKER_IDS[@]:1}"; do
    MERGE_DEP+=":$wid"
done

echo "Submitting merge..."
JOB_MERGE=$(sbatch --parsable \
    --job-name="mmpd-sw-merge$([[ "$SMOKE" -eq 1 ]] && echo -smoke)" \
    --account="$ACCOUNT" \
    --nodes=1 \
    --cpus-per-task=2 \
    --mem=16G \
    --gres=gpu:l40s:1 \
    --time="$WALL_MERGE" \
    --dependency="$MERGE_DEP" \
    --output="$LOG_DIR/merge-%j.out" \
    --error="$LOG_DIR/merge-%j.err" \
    --mail-type=FAIL \
    --mail-user=ccao87@uwo.ca \
    <<ENDSCRIPT
#!/bin/bash
source "$PREAMBLE_FILE"
"\$PYTHON" -u ${EVAL_BASE[@]} ${EVAL_EXTRA[@]} \
  --phase merge \
  --datasets ${DATASETS[*]} \
  --cpu
ENDSCRIPT
)
echo "  -> merge: $JOB_MERGE"

echo ""
echo "=================================================================="
echo "  MMPD sweep-subset matrix submitted (init + ${#DATASETS[@]} mmpd + merge)"
echo "  Output: $OUTPUT_DIR"
echo "  Logs:   $LOG_DIR/"
echo "  Monitor: squeue -u \$USER"
echo "=================================================================="
