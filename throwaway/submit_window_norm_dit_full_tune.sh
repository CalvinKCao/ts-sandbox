#!/usr/bin/env bash
# Throwaway: submit window-norm ablation (wn-a / wn-b / wn-c) × six datasets as separate Slurm jobs.
#
# - Full multivariate pipeline from scratch (no --frozen-hp-pack): Optuna + all phases as in pipeline_config.
# - Eval: 20% random subset of test windows (--eval-test-fraction 0.2). Diffusion eval still uses 3 DDIM
#   samples per window (hardcoded in train_multivariate_pipeline.py).
# - Slurm wall: 10h per job except ``weather`` (48h). Env: WALL_HOURS_DEFAULT, WALL_HOURS_WEATHER.
#
# Usage (repo root or anywhere — script cds to repo root):
#   export WANDB_API_KEY=...
#   ./throwaway/submit_window_norm_dit_full_tune.sh
#
# Wall time: 10h per job except dataset ``weather`` (48h). Override if needed:
#   WALL_HOURS_DEFAULT=12 WALL_HOURS_WEATHER=72 ./throwaway/submit_window_norm_dit_full_tune.sh
#   DATASETS="ETTh1 ETTh2" ./throwaway/submit_window_norm_dit_full_tune.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: export WANDB_API_KEY first."
  exit 2
fi

WALL_HOURS_DEFAULT="${WALL_HOURS_DEFAULT:-10}"
WALL_HOURS_WEATHER="${WALL_HOURS_WEATHER:-48}"
DATASETS="${DATASETS:-ETTh1 ETTh2 ETTm1 ETTm2 weather exchange_rate}"

submit_one() {
  local arm="$1" dataset="$2" extra_py="$3" hours="$4"
  # shellcheck disable=SC2086
  ./run.sh \
    --submit-root \
    --variant "$arm" \
    --dataset "$dataset" \
    --hours "$hours" \
    --eval-test-fraction 0.2 \
    $extra_py
}

n=0
for DS in $DATASETS; do
  if [[ "$DS" == weather ]]; then
    H="$WALL_HOURS_WEATHER"
  else
    H="$WALL_HOURS_DEFAULT"
  fi
  submit_one "wn-a" "$DS" "--disable-per-window-norm --guidance-penalty-weight 0" "$H"
  submit_one "wn-b" "$DS" "--guidance-penalty-weight 0.03" "$H"
  submit_one "wn-c" "$DS" "--guidance-spatial-penalty --guidance-penalty-weight 0.2" "$H"
  n=$((n + 3))
done

echo "Submitted $n jobs (wn-a/b/c × each dataset). Wall: ${WALL_HOURS_DEFAULT}h (weather: ${WALL_HOURS_WEATHER}h). No frozen HP packs."
