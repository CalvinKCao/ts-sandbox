#!/bin/bash
# Submit one Slurm job per dataset via run.sh. Each job pretrains then
# finetunes+evals on its dataset independently. All extra args are forwarded
# verbatim to run.sh (e.g. --variant, --smoke-test, --h100, --hours N).
#
# Default dataset list: ETTh1, ETTm1, exchange_rate, weather.
# Override with: --datasets ds1,ds2,...
#
# Usage:
#   ./run_multi_datasets.sh                      # 4-dataset sweep, defaults
#   ./run_multi_datasets.sh --smoke-test         # smoke each
#   ./run_multi_datasets.sh --variant h128 --hours 36
#   ./run_multi_datasets.sh --datasets ETTh1,weather --no-wandb

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASETS="ETTh1,ETTm1,exchange_rate,weather"
PASSTHRU=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets) DATASETS="$2"; shift 2 ;;
        *) PASSTHRU+=("$1"); shift ;;
    esac
done

IFS=',' read -ra DS_LIST <<< "$DATASETS"

echo "Submitting ${#DS_LIST[@]} job(s): ${DS_LIST[*]}"
echo "Extra args forwarded to run.sh: ${PASSTHRU[*]:-(none)}"
echo

for ds in "${DS_LIST[@]}"; do
    ds_trim="$(echo "$ds" | xargs)"
    [ -z "$ds_trim" ] && continue
    echo "--- $ds_trim ---"
    "$SCRIPT_DIR/run.sh" --dataset "$ds_trim" "${PASSTHRU[@]}"
done
