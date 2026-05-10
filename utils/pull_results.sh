#!/bin/bash
# Pull Slurm run artifacts from Killarney needed for local comparison plots + diffusion eval.
#
# run.sh layout (on cluster):
#   results/<MM-DD-JOBID-slug>/ckpts/
#     pretrained_dim<V>/itransformer.pt     — synthetic HP-best iTransformer (guidance pretrain)
#     <subset_id>/best.pt                   — fine-tuned diffusion U-Net
#     <subset_id>/metadata.json
#     <subset_id>_itransformer_finetuned.pt — real-data fine-tuned iTransformer (guidance at eval)
#     <subset_id>_itrans_ft_hp_best.pt      — optional; HP-trial best before copy to *_finetuned.pt
#
# visualize_comparison looks for *_itransformer_finetuned.pt next to each subset dir; without it,
# it falls back to pretrained_dim*/itransformer.pt (misleading orange curves on real data).

REMOTE_HOST="killarney.alliancecan.ca"
REMOTE_USER="ccao87"
REMOTE_PATH="/scratch/ccao87/ts-sandbox/results"
LOCAL_PATH="./results"
SSH_OPTS_INTERACTIVE=(
    -o BatchMode=no
    -o ConnectTimeout=12
    -o ConnectionAttempts=1
    -o StrictHostKeyChecking=accept-new
)

usage() {
    echo "Usage:"
    echo "  $0 --recent <hours>              Pull every run folder on the cluster touched in the last N hours"
    echo "  $0 <results-folder> [more...]   Pull named folders only"
    echo ""
    echo "The top-level remote folder \"archive\" is never pulled. Paths may omit a leading results/."
    echo "Examples:"
    echo "  $0 --recent 24"
    echo "  $0 05-08-3476425-default 05-08-3477032-h128"
}

rsync_one() {
    local FOLDER_CLEAN="$1"
    echo "------------------------------------------------------------"
    echo "Pulling visualization artifacts for: $FOLDER_CLEAN"
    echo "------------------------------------------------------------"
    rsync -e "ssh ${SSH_OPTS_INTERACTIVE[*]}" "${RSYNC_OPTS[@]}" \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/${FOLDER_CLEAN}" "${LOCAL_PATH}/"
}

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

mkdir -p "$LOCAL_PATH"

RSYNC_OPTS=(
    -avz
    --progress
    --include='*/'
    --include='best.pt'
    --include='metadata.json'
    --include='results.json'
    --include='itransformer.pt'
    --include='*_itransformer_finetuned.pt'
    --include='*_itrans_ft_hp_best.pt'
    --include='*_itrans_ft_hp.json'
    --include='*.log'
    --exclude='*'
)

TARGET_FOLDERS=()

if [ "$1" = "--recent" ]; then
    if [ -z "${2:-}" ] || ! [[ "${2}" =~ ^[0-9]+$ ]] || [ "${2}" -lt 1 ]; then
        echo "error: --recent requires a positive integer hour count (e.g. 24)" >&2
        exit 1
    fi
    HOURS="$2"
    MINUTES=$((HOURS * 60))
    echo "Listing run folders under ${REMOTE_PATH} modified in the last ${HOURS} hour(s) (excluding archive)..."
    mapfile -t TARGET_FOLDERS < <(
        ssh "${SSH_OPTS_INTERACTIVE[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
            "find '${REMOTE_PATH}' -maxdepth 1 -mindepth 1 -type d ! -name archive -mmin -${MINUTES} -printf '%f\\n' | sort"
    )
    if [ "${#TARGET_FOLDERS[@]}" -eq 0 ]; then
        echo "No matching folders on the cluster."
        exit 0
    fi
    echo "Will pull ${#TARGET_FOLDERS[@]} folder(s):"
    printf '  %s\n' "${TARGET_FOLDERS[@]}"
    echo ""
else
    while [ $# -gt 0 ]; do
        FOLDER_CLEAN="${1#results/}"
        if [ "$FOLDER_CLEAN" = "archive" ]; then
            echo "Skipping excluded folder: archive" >&2
        else
            TARGET_FOLDERS+=("$FOLDER_CLEAN")
        fi
        shift
    done
fi

for FOLDER in "${TARGET_FOLDERS[@]}"; do
    [ -n "$FOLDER" ] || continue
    rsync_one "$FOLDER"
done

echo ""
echo "Done. Pulled requested artifacts to $LOCAL_PATH"
