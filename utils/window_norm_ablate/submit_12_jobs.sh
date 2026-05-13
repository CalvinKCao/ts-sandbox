#!/usr/bin/env bash
# Submit 18 Slurm jobs: 6 datasets × three arms:
#   wn-a — no per-window norm, guidance penalty 0
#   wn-b — default window norm, guidance penalty 0.03 (uniform MSE)
#   wn-c — default window norm, spatial ramped guidance penalty only (±5 row grace / col), max weight 0.2
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
  submit_one "wn-c" "$ds" "--guidance-spatial-penalty --guidance-penalty-weight 0.2"
done

echo "Submitted 18 jobs (wn-a-*, wn-b-*, wn-c-* per dataset)."
