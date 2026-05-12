#!/bin/bash
# Pull Slurm run artifacts from Killarney.

REMOTE_HOST="killarney.alliancecan.ca"
REMOTE_USER="ccao87"
REMOTE_PATH="/scratch/ccao87/ts-sandbox/results"
LOCAL_PATH="./results"

# Regular SSH options (removed BatchMode=yes which can cause silent failures)
SSH_OPTS=(
    -o ConnectTimeout=12
    -o ConnectionAttempts=1
    -o StrictHostKeyChecking=accept-new
)

RSYNC_OPTS=(
    -avz
    --progress
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
    # We still need find to filter, but we won't print the list to the user.
    mapfile -t TARGET_FOLDERS < <(
        ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
            "find '${REMOTE_PATH}' -maxdepth 1 -mindepth 1 -type d ! -name archive -mmin -${MINUTES} -printf '%f\\n' | sort"
    )
    if [ "${#TARGET_FOLDERS[@]}" -eq 0 ]; then
        echo "No folders found from the last $HOURS hours."
        exit 0
    fi
    for FOLDER in "${TARGET_FOLDERS[@]}"; do
        [ -n "$FOLDER" ] && SOURCES+=("${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/${FOLDER}")
    done
    echo "Pulling ${#TARGET_FOLDERS[@]} recent folders..."
elif [ $# -gt 0 ]; then
    while [ $# -gt 0 ]; do
        CLEAN="${1#results/}"
        if [ "$CLEAN" != "archive" ]; then
            SOURCES+=("${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/${CLEAN}")
        fi
        shift
    done
    echo "Pulling specified folders..."
else
    # Default: sync everything
    SOURCES+=("${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/")
    echo "Syncing all folders (excluding archive)..."
fi

# Single rsync call is much faster and more robust
rsync -e "ssh ${SSH_OPTS[*]}" "${RSYNC_OPTS[@]}" "${SOURCES[@]}" "${LOCAL_PATH}/"

echo "Done."
