#!/bin/bash
set -euo pipefail
SSH_PRE='source /etc/profile >/dev/null 2>&1; export PATH=/cm/shared/apps/slurm/current/bin:$PATH; [ -f /cm/shared/apps/slurm/var/etc/killarney/slurm.conf ] && export SLURM_CONF=/cm/shared/apps/slurm/var/etc/killarney/slurm.conf'
LOCAL_DIR=/home/cao/ts-sandbox/temp/lean_disc_c128_results/guidance_decoder_test_metrics
OUTLOG=/home/cao/ts-sandbox/temp/scripts/guid_dec_mse_pull_4679970.log
mkdir -p "$LOCAL_DIR"

REMOTE_ROOT=$(ssh -o BatchMode=yes -o ConnectTimeout=40 killarney "$SSH_PRE; echo \$SCRATCH")
echo "REMOTE_SCRATCH=$REMOTE_ROOT" | tee -a "$OUTLOG"
REMOTE_METRICS="$REMOTE_ROOT/ts-sandbox-ordinal-fine/temp/lean_disc_c128_results/guidance_decoder_test_metrics"
REMOTE_OUT="$REMOTE_ROOT/ts-sandbox-ordinal-fine/results/slurm/guid-dec-mse-all-4679970.out"

{
  echo "=== REMOTE SUMMARY ==="
  ssh -o BatchMode=yes -o ConnectTimeout=40 killarney "$SSH_PRE; ls -la '$REMOTE_METRICS' 2>&1; echo ---; cat '$REMOTE_METRICS/summary_latest.md' 2>&1"
  echo "=== SLURM OUT (tail) ==="
  ssh -o BatchMode=yes -o ConnectTimeout=40 killarney "$SSH_PRE; ls -la '$REMOTE_OUT' 2>&1; echo ---; tail -120 '$REMOTE_OUT' 2>&1"
} | tee -a "$OUTLOG"

echo "=== RSYNC ===" | tee -a "$OUTLOG"
rsync -avz -e "ssh -o BatchMode=yes -o ConnectTimeout=40" \
  "killarney:$REMOTE_METRICS/" "$LOCAL_DIR/" | tee -a "$OUTLOG"

echo "=== LOCAL FILES ===" | tee -a "$OUTLOG"
ls -la "$LOCAL_DIR" | tee -a "$OUTLOG"
echo "=== LOCAL SUMMARY ===" | tee -a "$OUTLOG"
cat "$LOCAL_DIR/summary_latest.md" | tee -a "$OUTLOG"
echo PULL_COMPLETE
