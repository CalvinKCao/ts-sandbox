#!/usr/bin/env bash
# Run joint pipeline (pretrain per unique dim, then finetune) for every
# registered dataset whose native column count is < 32 (excludes electricity,
# traffic, weather, etc.). Skips datasets whose CSV is missing.
#
# Usage:
#   ./utils/run_joint_small_datasets.sh
#   ./utils/run_joint_small_datasets.sh --smoke-test
#   RUN=finetune ./utils/run_joint_small_datasets.sh   # only finetune (pretrain ckpts must exist)
#   RUN=pretrain ./utils/run_joint_small_datasets.sh     # only pretrain passes
#
# Extra args are forwarded to every python invocation, e.g.:
#   ./utils/run_joint_small_datasets.sh --checkpoint-dir /path/to/ckpt --results-dir /path/to/results

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MAX_V="${MAX_V:-31}" # native dim < 32  =>  d <= 31
RUN="${RUN:-all}"    # all | pretrain | finetune

mapfile -t ROWS < <(MAX_V="$MAX_V" ROOT="$ROOT" python3 <<'PY'
import os, sys
max_v = int(os.environ["MAX_V"])
root = os.environ["ROOT"]
sys.path.insert(0, root)
from models.diffusion_tsf.train_multivariate_pipeline import (
    DATASET_REGISTRY,
    get_dim_for_dataset,
    DATASETS_DIR,
)
for name in sorted(DATASET_REGISTRY.keys()):
    rel = DATASET_REGISTRY[name][0]
    path = os.path.join(DATASETS_DIR, rel)
    if not os.path.isfile(path):
        print(f"# skip (missing file): {name}", file=sys.stderr)
        continue
    try:
        d = get_dim_for_dataset(name)
    except Exception as e:
        print(f"# skip {name}: {e}", file=sys.stderr)
        continue
    if d <= max_v:
        print(f"{name}\t{d}")
PY
)

if [[ ${#ROWS[@]} -eq 0 ]]; then
  echo "No datasets with dim <= $MAX_V (or discovery failed)." >&2
  exit 1
fi

echo "Small datasets (dim <= $MAX_V):"
printf '  %s\n' "${ROWS[@]}"
echo ""

dims=$(printf '%s\n' "${ROWS[@]}" | cut -f2 | sort -u)

run_py() {
  python3 -m models.diffusion_tsf.train_multivariate_pipeline "$@"
}

if [[ "$RUN" == all || "$RUN" == pretrain ]]; then
  for d in $dims; do
    echo "========== joint pretrain dim=$d =========="
    run_py --mode pretrain --n-variates "$d" "$@"
  done
fi

if [[ "$RUN" == all || "$RUN" == finetune ]]; then
  for row in "${ROWS[@]}"; do
    name="$(printf '%s\n' "$row" | cut -f1)"
    dim="$(printf '%s\n' "$row" | cut -f2)"
    echo "========== joint finetune dataset=$name n_variates=$dim =========="
    run_py --mode finetune --dataset "$name" --n-variates "$dim" "$@"
  done
fi

echo "Done."
