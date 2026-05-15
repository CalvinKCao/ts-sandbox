#!/bin/bash
# Pull Slurm run artifacts from Killarney.
#
# Alliance Best Practices:
# 1. Use Data Transfer Nodes (DTNs) if available (e.g. dtn.killarney.alliancecan.ca).
# 2. Use --no-g and --no-p to avoid permission/quota issues on /project.
# 3. Use --partial to allow resuming interrupted transfers of large model files.

REMOTE_HOST="${REMOTE_HOST:-killarney.alliancecan.ca}"
REMOTE_USER="ccao87"
LOCAL_PATH="./results"

# Regular SSH options
SSH_OPTS=(
    -o ConnectTimeout=12
    -o ConnectionAttempts=1
    -o StrictHostKeyChecking=accept-new
)

# Dynamically find all results directories under /scratch/ccao87, excluding archive.
# This ensures we pull from all current and future projects.
echo "Discovering project results on $REMOTE_HOST..."
mapfile -t REMOTE_PATHS < <(
    ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
        "find /scratch/${REMOTE_USER} -maxdepth 2 -mindepth 2 -name results -type d ! -path '/scratch/${REMOTE_USER}/archive/*' 2>/dev/null"
)

# Fallback: if no 'results' folders found, use all top-level subfolders (excluding archive)
if [ ${#REMOTE_PATHS[@]} -eq 0 ]; then
    mapfile -t REMOTE_PATHS < <(
        ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
            "find /scratch/${REMOTE_USER} -maxdepth 1 -mindepth 1 -type d ! -name archive 2>/dev/null"
    )
fi

RSYNC_OPTS=(
    -avz
    --progress
    --no-g    # Best practice: avoid group preservation issues on /project
    --no-p    # Best practice: avoid permission preservation issues on /project
    --partial # Best practice: allow resuming large .pt file transfers
    --include='*/'
    --include='best.pt'
    --include='metadata.json'
    --include='results.json'
    --include='itransformer.pt'
    --include='pretrained_itransformer.pt'
    --include='*_itransformer_finetuned.pt'
    --include='*_itrans_ft_hp_best.pt'
    --include='*_itrans_ft_hp.json'
    --include='*.log'
    --exclude='archive/'
    --exclude='*'
)

mkdir -p "$LOCAL_PATH"

SOURCES=()

if [ "$1" = "--recent" ]; then
    HOURS="${2:-24}"
    MINUTES=$((HOURS * 60))
    echo "Finding folders from the last $HOURS hours on $REMOTE_HOST..."
    
    for RP in "${REMOTE_PATHS[@]}"; do
        echo "  Checking $RP..."
        mapfile -t TARGET_FOLDERS < <(
            ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
                "find '${RP}' -maxdepth 1 -mindepth 1 -type d ! -name archive -mmin -${MINUTES} -printf '%f\\n' | sort" 2>/dev/null
        )
        for FOLDER in "${TARGET_FOLDERS[@]}"; do
            [ -n "$FOLDER" ] && SOURCES+=("${REMOTE_USER}@${REMOTE_HOST}:${RP}/${FOLDER}")
        done
    done

    if [ "${#SOURCES[@]}" -eq 0 ]; then
        echo "No folders found."
        exit 0
    fi
    echo "Pulling ${#SOURCES[@]} recent folders..."

elif [ $# -gt 0 ]; then
    while [ $# -gt 0 ]; do
        CLEAN="${1#results/}"
        if [ "$CLEAN" != "archive" ]; then
            # We don't know which path it's in, so we'll add both. 
            # Rsync will handle it (though it might warn if one doesn't exist, we can suppress or just let it be).
            # To be cleaner, we could check remote existence, but adding both is usually fine for specific pulls.
            for RP in "${REMOTE_PATHS[@]}"; do
                SOURCES+=("${REMOTE_USER}@${REMOTE_HOST}:${RP}/${CLEAN}")
            done
        fi
        shift
    done
    echo "Pulling specified folders (searching across all remote paths)..."
else
    # Default: sync everything from all remote paths
    for RP in "${REMOTE_PATHS[@]}"; do
        SOURCES+=("${REMOTE_USER}@${REMOTE_HOST}:${RP}/")
    done
    echo "Syncing all folders from all remote locations (excluding archive)..."
fi

# Single rsync call is much faster and more robust
# We use --ignore-missing-args to prevent errors if a specifically requested folder 
# only exists in one of the multiple remote paths.
rsync -e "ssh ${SSH_OPTS[*]}" "${RSYNC_OPTS[@]}" --ignore-missing-args "${SOURCES[@]}" "${LOCAL_PATH}/"

echo "Done."
