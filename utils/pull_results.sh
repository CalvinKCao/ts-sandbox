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

if [ $# -eq 0 ]; then
    echo "Usage: $0 <results-folder> [more-folders...]"
    echo "Example:"
    echo "  $0 05-08-3476425-default 05-08-3477032-h128"
    exit 1
fi

TARGET_FOLDERS=("$@")

mkdir -p "$LOCAL_PATH"

# Rsync: traverse dirs, then allow only small artifacts (not full checkpoint trees).
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

for FOLDER in "${TARGET_FOLDERS[@]}"; do
    # Remove 'results/' prefix if passed manually (as in user prompt)
    FOLDER_CLEAN="${FOLDER#results/}"
    echo "------------------------------------------------------------"
    echo "Pulling visualization artifacts for: $FOLDER_CLEAN"
    echo "------------------------------------------------------------"
    rsync -e "ssh ${SSH_OPTS_INTERACTIVE[*]}" "${RSYNC_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/${FOLDER_CLEAN}" "$LOCAL_PATH/"
done

echo ""
echo "Done. Pulled requested artifacts to $LOCAL_PATH"
