#!/bin/bash
# sync_results.sh - Sync training results from remote cluster to local machine
#
# REMOTE_PATH should be $STORAGE_ROOT on the cluster, e.g.:
#   /lustre06/project/6054110/diffusion-tsf
#
# Usage (run from LOCAL machine):
#   ./sync_results.sh ccao87@narval.alliancecan.ca /lustre06/project/6054110/diffusion-tsf

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

DEFAULT_REMOTE_PATH="/lustre06/project/6054110/diffusion-tsf"
LOCAL_RESULTS_DIR="./synced_results"

REMOTE_HOST="${1:-}"
REMOTE_PATH="${2:-$DEFAULT_REMOTE_PATH}"

if [ -z "$REMOTE_HOST" ]; then
    echo -e "${RED}Usage: $0 user@remote-server [storage-root]${NC}"
    echo "  Default storage-root: $DEFAULT_REMOTE_PATH"
    exit 1
fi

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  Syncing Results from Remote${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""
echo -e "Remote: ${GREEN}$REMOTE_HOST:$REMOTE_PATH${NC}"
echo -e "Local:  ${GREEN}$LOCAL_RESULTS_DIR${NC}"
echo ""

mkdir -p "$LOCAL_RESULTS_DIR"

# ============================================================================
# [1] Results — per-subset subdirs (new) AND flat JSON files (old runs)
#     --delete keeps local in sync with remote (removes stale local files)
# ============================================================================
echo -e "${YELLOW}[1/3] Syncing evaluation results...${NC}"

rsync -avz --delete --progress \
    --include="*/" \
    --include="results.json" \
    --include="*_results.json" \
    --include="summary.csv" \
    --exclude="*" \
    "$REMOTE_HOST:$REMOTE_PATH/results/" \
    "$LOCAL_RESULTS_DIR/" \
    2>/dev/null || echo -e "${YELLOW}  (No results yet)${NC}"

# ============================================================================
# [2] Migrate any old flat *_results.json files → per-subset subdirs
#     Safe to re-run; skips files already migrated.
# ============================================================================
python3 - <<'PYEOF'
import json, os
from pathlib import Path

root = Path("./synced_results")
migrated = 0
for f in list(root.glob("*_results.json")):
    try:
        with open(f) as fh:
            data = json.load(fh)
        subset_id = data.get("subset_id") or f.name.replace("_results.json", "")
        # Rename old key if needed
        if "metrics" in data and "eval_metrics" not in data:
            data["eval_metrics"] = data.pop("metrics")
        subdir = root / subset_id
        subdir.mkdir(exist_ok=True)
        dest = subdir / "results.json"
        # Only overwrite if this flat file is newer / subdir doesn't exist yet
        if not dest.exists() or f.stat().st_mtime > dest.stat().st_mtime:
            with open(dest, "w") as fh:
                json.dump(data, fh, indent=2)
        f.unlink()
        migrated += 1
    except Exception as e:
        print(f"  Warning: could not migrate {f}: {e}")
if migrated:
    print(f"  Migrated {migrated} flat JSON files → per-subset subdirs")
PYEOF

# ============================================================================
# [3] Checkpoint manifests & metadata (JSON only, no .pt weights)
# ============================================================================
echo -e "${YELLOW}[2/3] Syncing training metadata...${NC}"

rsync -avz --progress \
    --include="*/" \
    --include="*.json" \
    --exclude="*" \
    "$REMOTE_HOST:$REMOTE_PATH/checkpoints/" \
    "$LOCAL_RESULTS_DIR/checkpoints/" \
    2>/dev/null || echo -e "${YELLOW}  (No checkpoints yet)${NC}"

# ============================================================================
# [4] Most recent SLURM log (*.out from ts-sandbox home dir)
# ============================================================================
echo -e "${YELLOW}[3/3] Syncing latest training log...${NC}"

LATEST_LOG=$(ssh "$REMOTE_HOST" "ls -t ~/ts-sandbox/diffusion-tsf-*.out 2>/dev/null | head -1" 2>/dev/null || true)
if [ -n "$LATEST_LOG" ]; then
    rsync -avz --progress \
        "$REMOTE_HOST:$LATEST_LOG" \
        "$LOCAL_RESULTS_DIR/latest_train.out" \
        2>/dev/null || echo -e "${YELLOW}  (Could not sync log)${NC}"
else
    echo -e "${YELLOW}  (No training log found)${NC}"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo -e "${BLUE}======================================${NC}"
echo -e "${GREEN}  Sync Complete!${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

N_SUBSETS=$(find "$LOCAL_RESULTS_DIR" -name "results.json" -not -path "*/checkpoints/*" 2>/dev/null | wc -l)
N_BASELINE=$(python3 -c "
import json
from pathlib import Path
n = sum(1 for p in Path('./synced_results').rglob('results.json')
        if 'itransformer_metrics' in json.load(open(p)))
print(n)
" 2>/dev/null || echo 0)

echo -e "  Subsets with diffusion eval:  ${GREEN}$N_SUBSETS${NC}"
echo -e "  Subsets with baseline:        ${GREEN}$N_BASELINE${NC}"

MANIFEST="$LOCAL_RESULTS_DIR/checkpoints/training_manifest.json"
if [ -f "$MANIFEST" ]; then
    COMPLETE=$(python3 -c "import json; m=json.load(open('$MANIFEST')); print(sum(1 for v in m.get('subsets',{}).values() if v.get('status')=='complete'))" 2>/dev/null || echo "?")
    PENDING=$(python3 -c "import json; m=json.load(open('$MANIFEST')); print(sum(1 for v in m.get('subsets',{}).values() if v.get('status')=='pending'))" 2>/dev/null || echo "?")
    echo -e "  Training manifest:            complete=${GREEN}$COMPLETE${NC}, pending=${YELLOW}$PENDING${NC}"
fi

echo ""
echo -e "Run: ${GREEN}python3 summarize_results.py --results-dir ./synced_results --output report.md${NC}"
echo ""
