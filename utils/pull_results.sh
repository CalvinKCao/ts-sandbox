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

echo "Checking remote results in $REMOTE_PATH..."
FOLDERS=$(ssh "${REMOTE_USER}@${REMOTE_HOST}" "ls -1 $REMOTE_PATH" 2>/dev/null)

if [ -z "$FOLDERS" ]; then
    echo "No folders found in $REMOTE_PATH or SSH failed."
    exit 1
fi

# If arguments are provided, use them; otherwise, use interactive select
if [ $# -gt 0 ]; then
    TARGET_FOLDERS=("$@")
else
    echo "Available result folders on remote:"
    echo "------------------------------------"
    select FOLDER in $FOLDERS "All" "Cancel"; do
        case $FOLDER in
            "Cancel") exit 0 ;;
            "All") TARGET_FOLDERS=($FOLDERS); break ;;
            *)
                if [ -n "$FOLDER" ]; then
                    TARGET_FOLDERS=("$FOLDER")
                    break
                else
                    echo "Invalid selection."
                fi
                ;;
        esac
    done
fi

mkdir -p "$LOCAL_PATH"

# Rsync: traverse dirs, then allow only small artifacts (not full checkpoint trees).
RSYNC_OPTS=(
    -avz
    --progress
    --include='*/'
    --include='best.pt'
    --include='metadata.json'
    --include='itransformer.pt'
    --include='*_itransformer_finetuned.pt'
    --include='*_itrans_ft_hp_best.pt'
    --include='*_itrans_ft_hp.json'
    --exclude='*'
)

for FOLDER in "${TARGET_FOLDERS[@]}"; do
    # Remove 'results/' prefix if passed manually (as in user prompt)
    FOLDER_CLEAN="${FOLDER#results/}"
    echo "------------------------------------------------------------"
    echo "Pulling visualization artifacts for: $FOLDER_CLEAN"
    echo "------------------------------------------------------------"
    rsync "${RSYNC_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/${FOLDER_CLEAN}" "$LOCAL_PATH/"
done

echo ""
echo "Done. Optimized artifacts pulled to $LOCAL_PATH"
