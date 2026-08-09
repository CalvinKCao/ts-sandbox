#!/bin/bash
# Submit mlp ablation-disc for PeMS/solar/ETTm1/ETTm2 after MMPD workers.
# Usage (Killarney, ordinal-fine):
#   MMPD_ROOT=results/datasets/08-06-mmpd-decoder-paper-lb336-hz96-matched-c128-four \
#   DEPS=PeMS:4628316,solar_Alabama:4628317,ETTm1:4628318,ETTm2:4628319 \
#   ./temp/scripts/run_c128_four_mlp_disc_killarney.sh
set -euo pipefail
export PATH=/cm/shared/apps/slurm/current/bin:${PATH:-}
export SCRATCH="${SCRATCH:-/scratch/ccao87}"
if [ -z "${SLURM_CONF:-}" ]; then
  [ -f /cm/shared/apps/slurm/var/etc/killarney/slurm.conf ] && export SLURM_CONF=/cm/shared/apps/slurm/var/etc/killarney/slurm.conf
fi
ROOT="${SCRATCH}/ts-sandbox-ordinal-fine"
cd "$ROOT"

MMPD_ROOT="${MMPD_ROOT:?set MMPD_ROOT}"
DEPS="${DEPS:-}"
WALL="${DISC_WALL:-4:00:00}"
declare -A DEP_MAP=()
if [ -n "$DEPS" ]; then
  IFS=',' read -ra _pairs <<< "$DEPS"
  for kv in "${_pairs[@]}"; do
    DEP_MAP["${kv%%:*}"]="${kv#*:}"
  done
fi

SPECS=(
  "PeMS|results/ckpts/08-05-4623005-PeMS-binary_window_norm_patch_refine_canvas128_p64x6_pems|configs/binary_window_norm_patch_refine_canvas128_p64x6_pems.yaml|PeMS-c128-wn128grid-valtest80-byvar"
  "solar_Alabama|results/ckpts/08-05-4623006-solar_Alabama-binary_window_norm_patch_refine_canvas128_p64x6_solar_alabama|configs/binary_window_norm_patch_refine_canvas128_p64x6_solar_alabama.yaml|solar_Alabama-c128-wn128grid-valtest80-byvar"
  "ETTm1|results/ckpts/08-05-4623007-ETTm1-binary_window_norm_patch_refine_canvas128_p64x6_ettm1|configs/binary_window_norm_patch_refine_canvas128_p64x6_ettm1.yaml|ETTm1-c128-wn128grid-valtest80-byvar"
  "ETTm2|results/ckpts/08-05-4623008-ETTm2-binary_window_norm_patch_refine_canvas128_p64x6_ettm2|configs/binary_window_norm_patch_refine_canvas128_p64x6_ettm2.yaml|ETTm2-c128-wn128grid-valtest80-byvar"
)

mkdir -p results/slurm temp/lean_disc_c128_jobs
: > temp/lean_disc_c128_jobs/four_mlp_disc_submitted.txt

for spec in "${SPECS[@]}"; do
  IFS='|' read -r DS CKPT CFG OUT_TAG <<<"$spec"
  DEP_ARGS=()
  if [ -n "${DEP_MAP[$DS]:-}" ]; then
    DEP_ARGS=(--dependency="afterok:${DEP_MAP[$DS]}")
  fi
  JID=$(sbatch \
    --job-name="ablation-mlp-${DS}" \
    --account=aip-boyuwang \
    --time="$WALL" \
    --nodes=1 \
    --gres=gpu:l40s:1 \
    --cpus-per-task=8 \
    --mem=50G \
    --exclude=kn010 \
    --export=ALL,CKPT="$CKPT",DISC_CONFIG="$CFG",DISC_RUN_NAME=window_norm_c128,OUT_TAG="$OUT_TAG",SCRATCH,HOME,USER,PATH,SLURM_CONF \
    --output="$ROOT/results/slurm/%x-%j.out" \
    --error="$ROOT/results/slurm/%x-%j.err" \
    --mail-type=END,FAIL \
    --mail-user=ccao87@uwo.ca \
    "${DEP_ARGS[@]}" \
    "$ROOT/temp/scripts/submit_ablation_disc_l8_l16.sh" \
    --dataset "$DS" \
    --disc-arch mlp \
    --viz-n-windows 1 \
    --mmpd-output-root "$MMPD_ROOT" \
    | awk '{print $4}')
  echo "submitted $DS disc=$JID dep=${DEP_MAP[$DS]:-none} ckpt=$CKPT"
  echo "$JID $DS $OUT_TAG" >> temp/lean_disc_c128_jobs/four_mlp_disc_submitted.txt
done
echo '---'
cat temp/lean_disc_c128_jobs/four_mlp_disc_submitted.txt
squeue -u "${USER:-ccao87}" -o '%.18i %.40j %.2t %.10M %R' | head -30
