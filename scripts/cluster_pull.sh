#!/bin/bash
# =============================================================================
# cluster_pull.sh — pull experiment artifacts from Killarney to local
#
# Creates a timestamped, descriptively-named local directory and rsyncs
# checkpoints, results, and Slurm logs from the cluster.  Synthetic pool
# .npy files are large and de-duplicated: they are only pulled if the same
# filename does not already exist locally anywhere under --local-dir.
#
# USAGE (run from repo root on local WSL):
#
#   ./scripts/cluster_pull.sh \
#       --host    ccao87@killarney.alliancecan.ca \
#       --store   /scratch/ccao87/diffusion-tsf-etth2 \
#       --out-dir ./runs \
#       [--name   etth2-gauss-binary]   # default: auto-detected from store path
#       [--synth]                        # also pull synth_pool_*.npy files
#       [--delete]                       # delete remote artifacts after pull
#       [--dry-run]                      # show what would be transferred
#
# The resulting directory layout:
#
#   runs/
#     20260412-1430_etth2-gauss-binary/
#       gauss/
#         checkpoints/   ← .pt files, hp.json, optuna db
#         results/       ← eval JSON, CSV
#       binary/
#         checkpoints/
#         results/
#       logs/            ← all Slurm .out/.err files
#       synth_cache/     ← synth_pool_*.npy (only with --synth)
#       pull_manifest.txt
# =============================================================================

set -euo pipefail

# ---- Defaults ---------------------------------------------------------------
HOST=""
REMOTE_STORE=""
OUT_DIR="$(pwd)/runs"
RUN_NAME=""
PULL_CHECKPOINTS=0
PULL_SYNTH=0
DELETE_REMOTE=0
DRY_RUN=0

# ---- Arg parsing ------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)         HOST="$2"; shift 2 ;;
        --store)        REMOTE_STORE="$2"; shift 2 ;;
        --out-dir)      OUT_DIR="$2"; shift 2 ;;
        --name)         RUN_NAME="$2"; shift 2 ;;
        --checkpoints)  PULL_CHECKPOINTS=1; shift ;;
        --synth)        PULL_SYNTH=1; shift ;;
        --delete)       DELETE_REMOTE=1; shift ;;
        --dry-run)      DRY_RUN=1; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

MISSING=()
[[ -z "$HOST" ]]         && MISSING+=("--host  (e.g. ccao87@killarney.alliancecan.ca)")
[[ -z "$REMOTE_STORE" ]] && MISSING+=("--store (e.g. /scratch/ccao87/diffusion-tsf-etth2)")
[[ -z "$RUN_NAME" ]]     && MISSING+=("--name  (e.g. etth2-gauss-binary-run1)")
if [[ "${#MISSING[@]}" -gt 0 ]]; then
    echo "ERROR: missing required flags:"
    for m in "${MISSING[@]}"; do echo "  $m"; done
    exit 1
fi

RSYNC_FLAGS=(-avz --progress --human-readable)
[[ "$DRY_RUN" -eq 1 ]] && RSYNC_FLAGS+=(--dry-run)

TIMESTAMP="$(date +%Y%m%d-%H%M)"
GIT_SHA="$(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo "nogit")"
LOCAL_DIR="$OUT_DIR/${TIMESTAMP}_${GIT_SHA}_${RUN_NAME}"

mkdir -p "$LOCAL_DIR/gauss/results" \
         "$LOCAL_DIR/binary/results" \
         "$LOCAL_DIR/logs"
[[ "$PULL_CHECKPOINTS" -eq 1 ]] && mkdir -p "$LOCAL_DIR/gauss/checkpoints" "$LOCAL_DIR/binary/checkpoints"

echo "=================================================================="
echo "  Pulling from: $HOST:$REMOTE_STORE"
echo "  Into:         $LOCAL_DIR"
echo "  Checkpoints:  $([ "$PULL_CHECKPOINTS" -eq 1 ] && echo yes || echo "no  (use --checkpoints to include)")"
echo "  Synth pool:   $([ "$PULL_SYNTH" -eq 1 ] && echo yes || echo "no  (use --synth to include)")"
echo "  Delete after: $([ "$DELETE_REMOTE" -eq 1 ] && echo yes || echo no)"
echo "  Dry run:      $([ "$DRY_RUN" -eq 1 ] && echo YES || echo no)"
echo "=================================================================="
echo ""

# Helper: rsync a remote path into a local dir, skip if remote dir is empty
pull() {
    local remote_path="$1"
    local local_dest="$2"
    local extra_excludes=("${@:3}")

    # Check if remote path exists and is non-empty
    if ! ssh "$HOST" "[ -d '$remote_path' ] && [ -n \"\$(ls -A '$remote_path' 2>/dev/null)\" ]" 2>/dev/null; then
        echo "  [skip] $remote_path — empty or missing"
        return 0
    fi

    local excl=()
    for ex in "${extra_excludes[@]:-}"; do
        excl+=(--exclude "$ex")
    done

    echo "  [pull] $remote_path → $local_dest"
    rsync "${RSYNC_FLAGS[@]}" "${excl[@]}" \
        "$HOST:$remote_path/" "$local_dest/"
}

# ---- Checkpoints (opt-in) ---------------------------------------------------
if [[ "$PULL_CHECKPOINTS" -eq 1 ]]; then
    echo "--- Checkpoints ---------------------------------------------------------"
    CKPT_EXCLUDES=("venv/" "env/" "synth_pool_*.npy")
    pull "$REMOTE_STORE/checkpoints_gauss"  "$LOCAL_DIR/gauss/checkpoints"  "${CKPT_EXCLUDES[@]}"
    pull "$REMOTE_STORE/checkpoints_binary" "$LOCAL_DIR/binary/checkpoints" "${CKPT_EXCLUDES[@]}"
    echo ""
else
    echo "--- Checkpoints ---------------------------------------------------------"
    echo "  [skip] not requested (pass --checkpoints to include .pt files)"
    echo ""
fi

# ---- Synth pool (de-duplicated by filename) ---------------------------------
if [[ "$PULL_SYNTH" -eq 1 ]]; then
    echo ""
    echo "--- Synth cache ---------------------------------------------------------"
    mkdir -p "$LOCAL_DIR/synth_cache"
    # Collect all remote .npy filenames first
    REMOTE_NPY=$(ssh "$HOST" "find '$REMOTE_STORE' -name 'synth_pool_*.npy' -maxdepth 3 2>/dev/null" || true)
    while IFS= read -r remote_file; do
        [[ -z "$remote_file" ]] && continue
        fname="$(basename "$remote_file")"
        # Skip if any copy of this filename already exists locally
        if find "$OUT_DIR" -name "$fname" -print -quit 2>/dev/null | grep -q .; then
            echo "  [skip-dup] $fname already present locally"
        else
            echo "  [pull] $remote_file → $LOCAL_DIR/synth_cache/$fname"
            if [[ "$DRY_RUN" -eq 0 ]]; then
                rsync "${RSYNC_FLAGS[@]}" "$HOST:$remote_file" "$LOCAL_DIR/synth_cache/$fname"
            else
                echo "    (dry-run: would rsync $remote_file)"
            fi
        fi
    done <<< "$REMOTE_NPY"
fi

# ---- Results ----------------------------------------------------------------
echo ""
echo "--- Results -------------------------------------------------------------"
pull "$REMOTE_STORE/results_gauss"  "$LOCAL_DIR/gauss/results"
pull "$REMOTE_STORE/results_binary" "$LOCAL_DIR/binary/results"

# ---- Slurm logs -------------------------------------------------------------
echo ""
echo "--- Logs ----------------------------------------------------------------"
pull "$REMOTE_STORE/logs" "$LOCAL_DIR/logs"

# ---- Manifest ---------------------------------------------------------------
echo ""
echo "--- Writing manifest ----------------------------------------------------"
MANIFEST="$LOCAL_DIR/pull_manifest.txt"
{
    echo "Pull timestamp:   $TIMESTAMP"
    echo "Remote host:      $HOST"
    echo "Remote store:     $REMOTE_STORE"
    echo "Local dir:        $LOCAL_DIR"
    echo "Synth pulled:     $PULL_SYNTH"
    echo "Deleted remote:   $DELETE_REMOTE"
    echo ""
    echo "Contents:"
    find "$LOCAL_DIR" -not -path '*/\.*' | sort
} > "$MANIFEST"
echo "  Written: $MANIFEST"

# ---- Delete remote (only after successful pull) -----------------------------
if [[ "$DELETE_REMOTE" -eq 1 && "$DRY_RUN" -eq 0 ]]; then
    echo ""
    echo "--- Deleting remote artifacts -------------------------------------------"
    echo "  Removing checkpoints, results, logs from $HOST:$REMOTE_STORE ..."
    ssh "$HOST" bash << REMOTE_DELETE
set -e
rm -rf "$REMOTE_STORE/results_gauss" \
       "$REMOTE_STORE/results_binary" \
       "$REMOTE_STORE/logs"
[ "$PULL_CHECKPOINTS" -eq 1 ] && rm -rf "$REMOTE_STORE/checkpoints_gauss" "$REMOTE_STORE/checkpoints_binary" || true
[ "$PULL_SYNTH" -eq 1 ] && find "$REMOTE_STORE" -name 'synth_pool_*.npy' -delete 2>/dev/null || true
echo "[remote] Cleanup done."
REMOTE_DELETE
fi

echo ""
echo "=================================================================="
echo "  Done. Artifacts in: $LOCAL_DIR"
echo "=================================================================="
