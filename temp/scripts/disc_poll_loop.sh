#!/usr/bin/env bash
set -u
IDS_FILE=/home/cao/ts-sandbox/temp/killarney_disc_ids.txt
STATUS=/home/cao/ts-sandbox/temp/killarney_disc_poll_status.txt
LOG=/home/cao/ts-sandbox/temp/killarney_disc_poll.log
RUN_NAME=07-31-2014-h96-ordinal-disc-bincenter-sample0
[[ -s "$IDS_FILE" ]] || echo '4510648,4510835,4510836,4510716,4510839' > "$IDS_FILE"
printf 'poll_wallflex %s ids=%s\n' "$(date -Is)" "$(cat "$IDS_FILE")" > "$STATUS"
START=$(date +%s); MAX=$((4*3600)); INTERVAL=900
SSH='source /etc/profile >/dev/null 2>&1; export PATH=/cm/shared/apps/slurm/current/bin:$PATH'

# pick wall: 8h if failed job elapsed >= 3.5h, else 6h
pick_wall() {
  local jid="$1"
  local el_sec
  el_sec=$(ssh -o BatchMode=yes killarney "$SSH; sacct -j $jid -X -n -o ElapsedRaw | head -1 | tr -d ' '")
  if [ -n "$el_sec" ] && [ "$el_sec" -ge 12600 ] 2>/dev/null; then
    echo "8:00:00"
  else
    echo "6:00:00"
  fi
}

resub_ds() {
  local datasets="$1" keep="$2" wall="$3"
  echo "RESUB $datasets wall=$wall keep=$keep" | tee -a "$LOG"
  local out
  out=$(ssh -o BatchMode=yes killarney "$SSH
cd \$SCRATCH/ts-sandbox
test -f temp/MMPD/exp/normalization.py || true
git pull --ff-only || true
BINARY_CONFIG=binary_patch_refine_lb336_hz96_ordinal_tuned_synth_fallback
MMPD_ROOT=results/datasets/07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd
BINARY_ROOTS=\"electricity=results/ckpts/07-29-4462979-electricity-\${BINARY_CONFIG},ETTh1=results/ckpts/07-29-4462980-ETTh1-\${BINARY_CONFIG},dynamic=results/ckpts/07-29-4462981-dynamic-\${BINARY_CONFIG},traffic=results/ckpts/07-29-4462982-traffic-\${BINARY_CONFIG}\"
RUN_NAME=$RUN_NAME
DAG_DIR=results/datasets/\${RUN_NAME}-dag
mkdir -p \"\$DAG_DIR\"
./submit_binary.sh --eval-ordinal-patch-refine-vs-mmpd --datasets $datasets \
  --existing-ckpt-roots \"\$BINARY_ROOTS\" --mmpd-root \"\$MMPD_ROOT\" \
  --ordinal-binary-config configs/\${BINARY_CONFIG}.yaml \
  --ordinal-disc-evaluator temp/scripts/eval_univariate_patch_refine_ordinal_vs_mmpd.py \
  --defer-checkpoint-check --disc-run \"\$RUN_NAME\" --raw-run \"\${RUN_NAME}-raw\" \
  --slice-lengths 8,16 --exclude kn010 --time $wall \
  --job-manifest \"\$DAG_DIR/discriminator_resub_\$(date +%H%M%S).json\"
")
  echo "$out" | tee -a "$LOG"
  local new_workers auto_merge
  new_workers=$(echo "$out" | python3 -c 'import sys,re; ids=re.findall(r"\"job_id\": \"(\d+)\"", sys.stdin.read()); print(",".join(ids[:-1]) if len(ids)>1 else ",".join(ids))')
  auto_merge=$(echo "$out" | python3 -c 'import sys,re; ids=re.findall(r"\"job_id\": \"(\d+)\"", sys.stdin.read()); print(ids[-1] if ids else "")')
  [ -n "$auto_merge" ] && ssh -o BatchMode=yes killarney "$SSH; scancel $auto_merge 2>/dev/null; true"
  local old_merge; old_merge=$(cut -d, -f5 "$IDS_FILE")
  ssh -o BatchMode=yes killarney "$SSH; scancel $old_merge 2>/dev/null; true"
  local workers="$keep"
  [ -n "$new_workers" ] && workers="${workers:+$workers,}$new_workers"
  workers=$(echo "$workers" | tr ',' '\n' | awk 'NF && !seen[$0]++' | paste -sd, -)
  local dep="afterok"
  for j in ${workers//,/ }; do dep="$dep:$j"; done
  local new_merge
  new_merge=$(ssh -o BatchMode=yes killarney "$SSH
cd \$SCRATCH/ts-sandbox
RUN_NAME=$RUN_NAME
DISC_OUTPUT=\$SCRATCH/ts-sandbox/results/datasets/\$RUN_NAME
RAW_OUTPUT=\$SCRATCH/ts-sandbox/results/datasets/\${RUN_NAME}-raw
sbatch --parsable --job-name=disc-opr96-merge --account=aip-boyuwang --nodes=1 \
  --cpus-per-task=2 --mem=8G --time=0:30:00 --dependency=$dep \
  --output=results/logs/disc-opr96-merge-%j.log --error=results/logs/disc-opr96-merge-%j.log \
  --mail-type=FAIL --mail-user=ccao87@uwo.ca \
  --export=ALL,GRID_EVAL_ORDINAL_PATCH_REFINE_MMPD=1,GRID_ORDINAL_DISC_MERGE=1,GRID_DISC_OUTPUT=\$DISC_OUTPUT,GRID_RAW_DISC_OUTPUT=\$RAW_OUTPUT,GRID_ORDINAL_DISC_EVALUATOR=temp/scripts/eval_univariate_patch_refine_ordinal_vs_mmpd.py,GRID_ORDINAL_BINARY_CONFIG=configs/binary_patch_refine_lb336_hz96_ordinal_tuned_synth_fallback.yaml \
  \$SCRATCH/ts-sandbox/slurm_worker.sh
")
  echo "${workers},${new_merge}" > "$IDS_FILE"
  echo "UPDATED_IDS=$(cat $IDS_FILE) wall=$wall" | tee -a "$STATUS" "$LOG"
}

while true; do
  NOW=$(date +%s); ELAPSED=$((NOW-START))
  JOBS=$(cat "$IDS_FILE")
  echo "===== poll $(date -Is) elapsed=${ELAPSED}s jobs=$JOBS =====" | tee -a "$LOG"
  OUT=$(ssh -o BatchMode=yes -o ConnectTimeout=45 killarney "$SSH; sacct -j ${JOBS} -X -o JobID,JobName%40,State,ExitCode,Elapsed,Timelimit,NodeList,End; echo ---; squeue -u ccao87 -o '%.18i %.8T %.10M %R %j' | head -40")
  echo "$OUT" | tee -a "$LOG"
  DONE=$(printf '%s\n' "$OUT" | awk 'NR>1 && /^[0-9]+/ && $3=="COMPLETED" {c++} END{print c+0}')
  TOTAL=$(printf '%s\n' "$OUT" | awk 'NR>1 && /^[0-9]+/ {c++} END{print c+0}')
  BAD=$(printf '%s\n' "$OUT" | awk 'NR>1 && /^[0-9]+/ && ($3=="FAILED"||$3=="TIMEOUT"||$3=="NODE_FAIL"||$3=="OUT_OF_MEMORY") {c++} END{print c+0}')
  BAD_EC=$(printf '%s\n' "$OUT" | awk 'NR>1 && /^[0-9]+/ && $3=="COMPLETED" && $4!="0:0" {c++} END{print c+0}')
  RUNN=$(printf '%s\n' "$OUT" | awk 'NR>1 && /^[0-9]+/ && ($3=="RUNNING"||$3=="PENDING"||$3=="COMPLETING") {c++} END{print c+0}')
  echo "DONE=$DONE TOTAL=$TOTAL BAD=$BAD BAD_EC=$BAD_EC RUN=$RUNN" | tee -a "$LOG"

  if [ "$BAD" -gt 0 ] || [ "$BAD_EC" -gt 0 ]; then
    echo FAILURE_DETECTED | tee -a "$STATUS"
    FAIL_LINES=$(printf '%s\n' "$OUT" | awk 'NR>1 && /^[0-9]+/ && ($3=="FAILED"||$3=="TIMEOUT"||$3=="NODE_FAIL"||$3=="OUT_OF_MEMORY"||($3=="COMPLETED"&&$4!="0:0")) {print $1,$2,$3}')
    ds_fail=""; wall="6:00:00"; max_el=0
    while read -r jid jname jst; do
      [ -z "$jid" ] && continue
      ssh -o BatchMode=yes killarney "$SSH; f=\$(ls -t /scratch/ccao87/ts-sandbox/results/logs/disc-opr96*-${jid}.log 2>/dev/null | head -1); echo ===== ${jid} ${jname} ${jst} =====; tail -40 \$f" | tee -a "$LOG"
      case "$jname" in
        *electricity*) ds_fail="${ds_fail:+$ds_fail,}electricity" ;;
        *ETTh1*) ds_fail="${ds_fail:+$ds_fail,}ETTh1" ;;
        *traffic*) ds_fail="${ds_fail:+$ds_fail,}traffic" ;;
        *dynamic*) ds_fail="${ds_fail:+$ds_fail,}dynamic" ;;
      esac
      w=$(pick_wall "$jid"); [ "$w" = "8:00:00" ] && wall="8:00:00"
    done <<< "$FAIL_LINES"
    keep=$(printf '%s\n' "$OUT" | awk 'NR>1 && /^[0-9]+/ && ($3=="COMPLETED"||$3=="RUNNING"||$3=="PENDING") && $2 !~ /merge/ {print $1}' | paste -sd, -)
    # drop failed from keep
    for jid in $(echo "$FAIL_LINES" | awk '{print $1}'); do
      keep=$(echo "$keep" | tr ',' '\n' | grep -v "^${jid}$" | paste -sd, -)
    done
    [ -z "$ds_fail" ] && { echo UNHANDLED; exit 2; }
    resub_ds "$ds_fail" "$keep" "$wall"
    sleep "$INTERVAL"; continue
  fi
  if [ "$TOTAL" -ge 1 ] && [ "$DONE" -eq "$TOTAL" ] && [ "$RUNN" -eq 0 ]; then
    echo ALL_COMPLETED_OK | tee -a "$STATUS"
    mkdir -p "/home/cao/ts-sandbox/results/pulled/$RUN_NAME"
    rsync -avz "killarney:/scratch/ccao87/ts-sandbox/results/datasets/$RUN_NAME/metrics.csv" \
      "killarney:/scratch/ccao87/ts-sandbox/results/datasets/$RUN_NAME/metrics.json" \
      "/home/cao/ts-sandbox/results/pulled/$RUN_NAME/" || true
    exit 0
  fi
  if [ "$ELAPSED" -ge "$MAX" ]; then echo CAP_REACHED | tee -a "$STATUS"; exit 3; fi
  sleep "$INTERVAL"
done
