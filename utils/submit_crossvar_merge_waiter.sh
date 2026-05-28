#!/bin/bash
# Submit (or re-submit) the partial-file merge waiter for an existing ablation run.
#
# USAGE (Killarney login node):
#   ./utils/submit_crossvar_merge_waiter.sh --run-stem 05-28-bin-h128-crossvar-ablation
#   ./utils/submit_crossvar_merge_waiter.sh --run-stem 05-28-bin-h128-crossvar-ablation --cancel-pending
#
# Optional: pass eval job IDs so the waiter starts only after all first-wave evals finish:
#   ./utils/submit_crossvar_merge_waiter.sh --run-stem ... --after-eval 3793328:3793355:...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_STEM=""
CANCEL_PENDING=0
AFTER_EVAL=""
MERGE_WAIT_WALL="${MERGE_WAIT_WALL:-7-00:00:00}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-stem) RUN_STEM="$2"; shift 2 ;;
        --cancel-pending) CANCEL_PENDING=1; shift ;;
        --after-eval) AFTER_EVAL="$2"; shift 2 ;;
        --walltime) MERGE_WAIT_WALL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$RUN_STEM" ]]; then
    echo "ERROR: --run-stem required (e.g. 05-28-bin-h128-crossvar-ablation)" >&2
    exit 1
fi

if [[ -d "${SCRATCH:-}/ts-sandbox" ]]; then
    REPO="${SCRATCH}/ts-sandbox"
elif [[ -d "$HOME/ts-sandbox" ]]; then
    REPO="$HOME/ts-sandbox"
else
    REPO="$REPO_ROOT"
fi
cd "$REPO"

JOB_DIR="$REPO/results/jobs/${RUN_STEM}"
MERGE_INVOKE="$JOB_DIR/merge_invoke.sh"
WAIT_SCRIPT="$REPO/utils/wait_and_merge_binary_crossvar.sh"
EVAL_OUT="$REPO/results/datasets/${RUN_STEM}"
LOG_DIR="$REPO/results/logs/${RUN_STEM}"

if [[ ! -f "$MERGE_INVOKE" ]]; then
    echo "Generating $MERGE_INVOKE ..."
    "$REPO/utils/write_crossvar_merge_invoke.sh" "$RUN_STEM" "$REPO"
fi
chmod +x "$WAIT_SCRIPT"

mkdir -p "$LOG_DIR"

if [[ "$CANCEL_PENDING" -eq 1 && -f "$JOB_DIR/merge_waiter_job_id.txt" ]]; then
    old=$(tr -d '[:space:]' < "$JOB_DIR/merge_waiter_job_id.txt")
    if [[ -n "$old" ]]; then
        echo "Cancelling previous merge waiter: $old"
        scancel "$old" 2>/dev/null || true
    fi
fi

dep_args=()
if [[ -n "$AFTER_EVAL" ]]; then
    dep_args=(--dependency="afterany:${AFTER_EVAL}")
    echo "Merge waiter will start after eval jobs: ${AFTER_EVAL//:/, }"
else
    echo "Merge waiter starts immediately (polls until all partials exist)"
fi

jid=$(sbatch --parsable \
    "${dep_args[@]}" \
    --job-name="merge-wait-${RUN_STEM}" \
    --account=aip-boyuwang \
    --nodes=1 \
    --cpus-per-task=2 \
    --mem=16G \
    --time="$MERGE_WAIT_WALL" \
    --output="$LOG_DIR/merge-wait-%j.out" \
    --error="$LOG_DIR/merge-wait-%j.err" \
    --mail-type=END,FAIL \
    --mail-user=ccao87@uwo.ca \
    --export=ALL,REPO="$REPO",EVAL_OUT="$EVAL_OUT",EVAL_SCRIPT="$REPO/utils/eval_binary_anchor_variants.py",MERGE_INVOKE="$MERGE_INVOKE" \
    "$WAIT_SCRIPT")

echo "$jid" > "$JOB_DIR/merge_waiter_job_id.txt"
echo "Submitted merge waiter: $jid"
echo "  Log: $LOG_DIR/merge-wait-${jid}.out"
echo "  Poll partials in: $EVAL_OUT/partials/"
echo "  Cancel: scancel $jid"
