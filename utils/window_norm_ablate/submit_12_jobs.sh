#!/usr/bin/env bash
# Submit 12 Slurm jobs: 6 datasets × (A: no per-window norm, pen=0) vs (B: per-window norm, pen=0.03).
# Requires: repo branch with frozen packs, WANDB_API_KEY, datasets under ./datasets.
# Run from repo root on Killarney login node.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: export WANDB_API_KEY first."
  exit 2
fi

PACK_DIR="$ROOT/utils/window_norm_ablate/frozen_packs"

submit_one () {
  local arm="$1" dataset="$2" extra_py="$3"
  ./run.sh \
    --submit-root \
    --variant "$arm" \
    --dataset "$dataset" \
    --hours 24 \
    --frozen-hp-pack "$PACK_DIR/${dataset}.json" \
    --eval-test-fraction 0.2 \
    $extra_py
}

for ds in ETTh1 ETTh2 ETTm1 ETTm2 weather exchange_rate; do
  submit_one "wn-a" "$ds" "--disable-per-window-norm --guidance-penalty-weight 0"
  submit_one "wn-b" "$ds" "--guidance-penalty-weight 0.03"
done

echo "Submitted 12 jobs (Slurm names wn-a-<dataset> / wn-b-<dataset>)."
