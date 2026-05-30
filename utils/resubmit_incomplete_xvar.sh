#!/bin/bash
# Resubmit incomplete 05-28 H=128 cross-variate ablation training jobs.
#
# Usage (Killarney login node, repo root):
#   ./utils/resubmit_incomplete_xvar.sh           # print commands
#   ./utils/resubmit_incomplete_xvar.sh --submit  # needs your ablation sbatch driver
#
# Do not re-submit eval-only jobs until train logs show PIPELINE COMPLETE.

set -euo pipefail

SUBMIT=0
[[ "${1:-}" == "--submit" ]] && SUBMIT=1

INCOMPLETE=(
    05-28-bin-h128-xvar-exp1-datasetnorm-electricity_18v
    05-28-bin-h128-xvar-exp1-datasetnorm-traffic_27v
    05-28-bin-h128-xvar-exp1-datasetnorm-weather_9v
    05-28-bin-h128-xvar-exp2-tokens-only-electricity_18v
    05-28-bin-h128-xvar-exp2-tokens-only-traffic_27v
    05-28-bin-h128-xvar-exp2-tokens-only-weather_9v
    05-28-bin-h128-xvar-exp3-datasetnorm-tokens-only-electricity_18v
    05-28-bin-h128-xvar-exp3-datasetnorm-tokens-only-traffic_27v
    05-28-bin-h128-xvar-exp3-datasetnorm-tokens-only-weather_9v
    05-28-bin-h128-xvar-exp4-baseline-xvar-electricity_18v
    05-28-bin-h128-xvar-exp4-baseline-xvar-traffic_27v
    05-28-bin-h128-xvar-exp4-baseline-xvar-weather_9v
)

WALL="3-00:00:00"
ACCOUNT="aip-boyuwang"
USER="${USER:-$(whoami)}"
STORE="/scratch/${USER}/results"

echo "# ${#INCOMPLETE[@]} incomplete xvar train jobs (72h wall)"
echo ""

for stem in "${INCOMPLETE[@]}"; do
    ckpt="${STORE}/ckpts/${stem}"
    data="${STORE}/datasets/${stem}"
    log="${STORE}/logs/${stem}.log"
    job_name="h128-resume-${stem#05-28-bin-h128-xvar-}"

    cmd=(
        sbatch
        --job-name="$job_name"
        --account="$ACCOUNT"
        --time="$WALL"
        --nodes=1
        --gres=gpu:l40s:1
        --cpus-per-task=8
        --mem=60G
        --output="$log"
        --error="$log"
        --export=ALL,GRID_STORE="$STORE",GRID_RUN_STEM="$stem",GRID_RESUME=1
    )

    # Ablation used a custom driver on cluster; set XVAR_WORKER to that script path.
    worker="${XVAR_WORKER:-./slurm_xvar_ablation.sh}"
    py_args=(--resume --checkpoint-dir "$ckpt" --results-dir "$data")

    if [[ "$SUBMIT" -eq 1 ]]; then
        [[ -d "$ckpt" ]] || { echo "SKIP missing $ckpt" >&2; continue; }
        [[ -x "$worker" ]] || { echo "ERROR: set XVAR_WORKER to your ablation sbatch script (not $worker)" >&2; exit 1; }
        "${cmd[@]}" "$worker" "${py_args[@]}"
    else
        printf '# %s\n' "$stem"
        printf '%q ' "${cmd[@]}" "$worker"
        printf '%q ' "${py_args[@]}"
        echo -e "\n"
    fi
done

if [[ "$SUBMIT" -eq 0 ]]; then
    echo "Dry run. Export XVAR_WORKER=/path/to/slurm_xvar_ablation.sh then --submit"
fi
