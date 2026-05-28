#!/bin/bash
# Poll for eval partial JSON files, then run --phase merge.
# Survives failed/cancelled evals; picks up metrics from resubmitted eval jobs.
#
# Env:
#   REPO, EVAL_OUT, EVAL_SCRIPT, MERGE_INVOKE
#   MERGE_POLL_SEC (default 300), MERGE_MAX_WAIT_SEC (default 0 = unlimited)

set -euo pipefail

: "${REPO:?REPO required}"
: "${EVAL_OUT:?EVAL_OUT required}"
: "${EVAL_SCRIPT:?EVAL_SCRIPT required}"
: "${MERGE_INVOKE:?MERGE_INVOKE required}"

MERGE_POLL_SEC="${MERGE_POLL_SEC:-300}"
MERGE_MAX_WAIT_SEC="${MERGE_MAX_WAIT_SEC:-0}"

echo "=========================================="
echo "Job: $SLURM_JOB_NAME  ID: ${SLURM_JOB_ID:-local}  Node: ${SLURMD_NODENAME:-unknown}"
echo "Repo: $REPO"
echo "Eval out: $EVAL_OUT"
echo "Invoke: $MERGE_INVOKE"
echo "Poll: ${MERGE_POLL_SEC}s"
echo "Started: $(date)"
echo "=========================================="

# shellcheck source=/dev/null
source "$MERGE_INVOKE"

module purge || true
module load StdEnv/2023
module load python/3.11

echo "[setup] Building venv on \$SLURM_TMPDIR..."
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck source=/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index \
    'torch==2.11.0+computecanada' numpy pandas scipy scikit-learn tqdm wandb einops \
    -q
export PYTHONUNBUFFERED=1
cd "$REPO"

wait_start=$SECONDS
while true; do
    if python -u "$EVAL_SCRIPT" --phase check-partials "${CHECK_ARGS[@]}"; then
        echo "[wait] All eval partials present ($(date))"
        break
    fi
    elapsed=$((SECONDS - wait_start))
    if [[ "$MERGE_MAX_WAIT_SEC" -gt 0 && "$elapsed" -ge "$MERGE_MAX_WAIT_SEC" ]]; then
        echo "[wait] Timed out after ${elapsed}s; still missing partials" >&2
        exit 2
    fi
    echo "[wait] Missing partials (${elapsed}s elapsed); sleep ${MERGE_POLL_SEC}s ($(date))"
    sleep "$MERGE_POLL_SEC"
done

echo "[merge] Running aggregate/report..."
exec python -u "$EVAL_SCRIPT" --phase merge "${MERGE_ARGS[@]}"
