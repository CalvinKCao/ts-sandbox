#!/usr/bin/env bash
# Resubmit only the wn-a/b/c jobs that failed or timed out from the 3562485–3562502 batch.
# Bumps wall time to 12h since ETTm*/weather Phase 2B diffusion HP exceeded 4h.
# Run from repo root on the Killarney login node with WANDB_API_KEY exported.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: export WANDB_API_KEY first."
  exit 2
fi

PACK_DIR="$ROOT/utils/window_norm_ablate/frozen_packs"
HOURS="${HOURS:-12}"

submit_one () {
  local arm="$1" dataset="$2" extra_py="$3"
  ./run.sh \
    --submit-root \
    --variant "$arm" \
    --dataset "$dataset" \
    --hours "$HOURS" \
    --frozen-hp-pack "$PACK_DIR/${dataset}.json" \
    --eval-test-fraction 0.2 \
    $extra_py
}

ARM_A_EXTRA="--disable-per-window-norm --guidance-penalty-weight 0"
ARM_B_EXTRA="--guidance-penalty-weight 0.03"
ARM_C_EXTRA="--guidance-spatial-penalty --guidance-penalty-weight 0.2"

# (arm dataset) pairs to resubmit
PAIRS=(
  "wn-b ETTh2"            # 3562489 torch import
  "wn-c ETTh2"            # 3562490 torch import
  "wn-a ETTm1"            # 3562491 torch import
  "wn-b ETTm1"            # 3562492 torch import
  "wn-c ETTm1"            # 3562493 walltime
  "wn-a ETTm2"            # 3562494 walltime
  "wn-b ETTm2"            # 3562495 walltime
  "wn-c ETTm2"            # 3562496 walltime
  "wn-a weather"          # 3562497 walltime
  "wn-b weather"          # 3562498 walltime
  "wn-c weather"          # 3562499 walltime
  "wn-b exchange_rate"    # 3562501 dataloader bus error
)

for entry in "${PAIRS[@]}"; do
  arm="${entry%% *}"
  ds="${entry##* }"
  case "$arm" in
    wn-a) extra="$ARM_A_EXTRA" ;;
    wn-b) extra="$ARM_B_EXTRA" ;;
    wn-c) extra="$ARM_C_EXTRA" ;;
    *)    echo "unknown arm $arm"; exit 1 ;;
  esac
  submit_one "$arm" "$ds" "$extra"
done

echo "Resubmitted ${#PAIRS[@]} jobs at ${HOURS}h wall (override with HOURS=N ./utils/window_norm_ablate/resubmit_failed.sh)."
