#!/bin/bash
# Killarney: lean disc arches (transformer/mlp/cnn1d/flatness) on all canvas128 leaves.
#
# One L40S job per dataset. Each job snaps once (window_norm_grid) from an existing
# wn128grid pack, then trains all arches × binary/MMPD × L∈{8,16} with per-variate
# metrics under unique_abs + bin-center + candidate_only.
#
# Viz defaults ON (inherited from smoke_lean_disc_arches.py): --viz-sanity all
# (snap+pre_post) and binary↔MMPD disagreement panels. Disable with
#   PY_EXTRA+=("--viz-sanity none") or --no-viz / --no-disc-disagreement
#
# USAGE (Killarney login, $SCRATCH/ts-sandbox-ordinal-fine):
#   ./temp/scripts/submit_lean_disc_c128_killarney.sh
#   ./temp/scripts/submit_lean_disc_c128_killarney.sh --smoke-test
#   DATASETS="ETTh2 electricity" ./temp/scripts/submit_lean_disc_c128_killarney.sh
#
set -euo pipefail
export PATH=/cm/shared/apps/slurm/current/bin:${PATH:-}
if [ -z "${SLURM_CONF:-}" ]; then
  if [ -f /cm/shared/apps/slurm/var/etc/killarney/slurm.conf ]; then
    export SLURM_CONF=/cm/shared/apps/slurm/var/etc/killarney/slurm.conf
  elif [ -f /cm/shared/apps/slurm/var/etc/slurm/slurm.conf ]; then
    export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
  fi
fi
export SCRATCH="${SCRATCH:-/scratch/ccao87}"

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -d "$SCRATCH/ts-sandbox-ordinal-fine" ]; then
  PROJECT_ROOT="$SCRATCH/ts-sandbox-ordinal-fine"
else
  PROJECT_ROOT="$REPO_ROOT"
fi
cd "$PROJECT_ROOT"
echo "PROJECT_ROOT=$PROJECT_ROOT branch=$(git branch --show-current 2>/dev/null || echo '?')"

IS_SMOKE=0
for arg in "$@"; do
  [ "$arg" = "--smoke-test" ] && IS_SMOKE=1
done

if [ "$IS_SMOKE" -eq 1 ]; then
  WALL=1:00:00
  MEM=32G
else
  WALL="${LEAN_DISC_WALL:-4:00:00}"
  MEM=50G
fi

mapfile -t SPECS <<'EOF'
ETTh1|results/ckpts/08-03-4571065-ETTh1-binary_window_norm_patch_refine_canvas128_p64x6|configs/binary_window_norm_patch_refine_canvas128_p64x6.yaml|ETTh1-c128-wn128grid-valtest80-byvar
ETTh2|results/ckpts/08-04-4601319-ETTh2-binary_window_norm_patch_refine_canvas128_p64x6_etth2|configs/binary_window_norm_patch_refine_canvas128_p64x6_etth2.yaml|ETTh2-c128-wn128grid-valtest80-byvar
electricity|results/ckpts/08-04-4597054-electricity-binary_window_norm_patch_refine_canvas128_p64x6_electricity|configs/binary_window_norm_patch_refine_canvas128_p64x6_electricity.yaml|electricity-c128-wn128grid-valtest80-byvar
traffic|results/ckpts/08-04-4597055-traffic-binary_window_norm_patch_refine_canvas128_p64x6_traffic|configs/binary_window_norm_patch_refine_canvas128_p64x6_traffic.yaml|traffic-c128-wn128grid-valtest80-byvar
exchange_rate|results/ckpts/08-04-4597056-exchange_rate-binary_window_norm_patch_refine_canvas128_p64x6_exchange_rate|configs/binary_window_norm_patch_refine_canvas128_p64x6_exchange_rate.yaml|exchange_rate-c128-wn128grid-valtest80-byvar
PeMS|results/ckpts/08-05-4623005-PeMS-binary_window_norm_patch_refine_canvas128_p64x6_pems|configs/binary_window_norm_patch_refine_canvas128_p64x6_pems.yaml|PeMS-c128-wn128grid-valtest80-byvar
solar_Alabama|results/ckpts/08-05-4623006-solar_Alabama-binary_window_norm_patch_refine_canvas128_p64x6_solar_alabama|configs/binary_window_norm_patch_refine_canvas128_p64x6_solar_alabama.yaml|solar_Alabama-c128-wn128grid-valtest80-byvar
ETTm1|results/ckpts/08-05-4623007-ETTm1-binary_window_norm_patch_refine_canvas128_p64x6_ettm1|configs/binary_window_norm_patch_refine_canvas128_p64x6_ettm1.yaml|ETTm1-c128-wn128grid-valtest80-byvar
ETTm2|results/ckpts/08-05-4623008-ETTm2-binary_window_norm_patch_refine_canvas128_p64x6_ettm2|configs/binary_window_norm_patch_refine_canvas128_p64x6_ettm2.yaml|ETTm2-c128-wn128grid-valtest80-byvar
EOF

WANT="${DATASETS:-}"
mkdir -p results/slurm results/datasets temp/lean_disc_c128_jobs
: > temp/lean_disc_c128_jobs/submitted.txt
JOB_IDS=()

resolve_pack() {
  local frag="$1"
  local hit
  hit="$(ls -d results/datasets/08-04-2341-ablation-disc-l8-l16-"${frag}" 2>/dev/null | head -1 || true)"
  if [ -z "$hit" ]; then
    hit="$(ls -d results/datasets/*-"${frag}" 2>/dev/null | head -1 || true)"
  fi
  echo "$hit"
}

for spec in "${SPECS[@]}"; do
  IFS='|' read -r DS CKPT CFG PACK_FRAG <<<"$spec"
  if [ -n "$WANT" ]; then
    skip=1
    for w in $WANT; do
      [ "$w" = "$DS" ] && skip=0
    done
    [ "$skip" -eq 1 ] && continue
  fi
  PACK="$(resolve_pack "$PACK_FRAG")"
  if [ -z "$PACK" ] || [ ! -d "$PACK/raw" ]; then
    echo "ERROR: missing pack for $DS (frag=$PACK_FRAG)" >&2
    exit 1
  fi
  if [ ! -d "$CKPT" ]; then
    echo "ERROR: missing ckpt $CKPT" >&2
    exit 1
  fi
  STAMP="$(date +%m-%d-%H%M)"
  OUT="results/datasets/${STAMP}-lean-disc-${DS}-c128-valtest80"
  mkdir -p "$OUT"

  PY_EXTRA=""
  [ "$IS_SMOKE" -eq 1 ] && PY_EXTRA="$PY_EXTRA --smoke-test"
  [ "$DS" = "ETTh2" ] && PY_EXTRA="$PY_EXTRA --also-lull-only"

  WRAP="temp/lean_disc_c128_jobs/run_${DS}.sh"
  cat >"$WRAP" <<EOF
#!/bin/bash
set -euo pipefail
export PATH=/cm/shared/apps/slurm/current/bin:\${PATH:-}
export SCRATCH=/scratch/ccao87
cd "$PROJECT_ROOT"
echo "Job \$SLURM_JOB_ID node=\${SLURMD_NODENAME:-?} dataset=$DS"
echo "Started: \$(date)"

if ! type module >/dev/null 2>&1; then
  if [ -f /cvmfs/soft.computecanada.ca/config/profile/bash.sh ]; then
    export SKIP_CC_CVMFS="\${SKIP_CC_CVMFS:-0}"
    set +u
    # shellcheck disable=SC1091
    source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
    set -u
  fi
fi
type module >/dev/null 2>&1 || { echo "ERROR: Lmod unavailable" >&2; exit 127; }
module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

REQ="$PROJECT_ROOT/setup/requirements-killarney.txt"
[[ -f "\$REQ" ]] || { echo "ERROR: missing \$REQ" >&2; exit 1; }
[[ -n "\${SLURM_TMPDIR:-}" ]] || { echo "ERROR: SLURM_TMPDIR unset" >&2; exit 1; }
command -v virtualenv >/dev/null || { echo "ERROR: virtualenv missing" >&2; exit 1; }

echo "[setup] Building node-local venv on \$SLURM_TMPDIR"
virtualenv --no-download "\$SLURM_TMPDIR/env"
# shellcheck disable=SC1091
source "\$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r "\$REQ" -q
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
print(f"torch={torch.__version__} device={torch.cuda.get_device_name(0)}")
PY

ARCHES="${LEAN_DISC_ARCHES:-mlp cnn1d}"
VIZ_N="${LEAN_DISC_VIZ_N_WINDOWS:-1}"
python temp/scripts/smoke_lean_disc_arches.py \\
  --dataset "$DS" \\
  --pack "$PACK" \\
  --ckpt "$CKPT" \\
  --config "$CFG" \\
  --output-dir "$OUT" \\
  --arches $ARCHES \\
  --sources binary_staged mmpd \\
  --slice-lengths 8 16 \\
  --viz-n-windows "$VIZ_N" \\
  $PY_EXTRA

echo "Finished: \$(date) out=$OUT"
EOF
  chmod +x "$WRAP"

  DEP_ARGS=()
  if [ -n "${LEAN_DISC_DEPENDENCY:-}" ]; then
    DEP_ARGS=(--dependency="$LEAN_DISC_DEPENDENCY")
  elif [ -n "${LEAN_DISC_DEPENDENCY_BY_DS:-}" ]; then
    # format: PeMS:123,solar_Alabama:456,...
    IFS=',' read -ra _deps <<< "$LEAN_DISC_DEPENDENCY_BY_DS"
    for _kv in "${_deps[@]}"; do
      _k="${_kv%%:*}"
      _v="${_kv#*:}"
      if [ "$_k" = "$DS" ] && [ -n "$_v" ]; then
        DEP_ARGS=(--dependency="afterok:${_v}")
        break
      fi
    done
  fi
  JID=$(sbatch \
    --job-name="lean-disc-${DS}" \
    --account=aip-boyuwang \
    --time="$WALL" \
    --nodes=1 \
    --gres=gpu:l40s:1 \
    --cpus-per-task=8 \
    --mem="$MEM" \
    --exclude=kn010 \
    --export=NONE,PATH,SLURM_CONF,SCRATCH,HOME,USER,LANG \
    --output="$PROJECT_ROOT/results/slurm/%x-%j.out" \
    --error="$PROJECT_ROOT/results/slurm/%x-%j.err" \
    --mail-type=END,FAIL \
    --mail-user=ccao87@uwo.ca \
    "${DEP_ARGS[@]}" \
    "$WRAP" | awk '{print $4}')
  echo "submitted $DS job=$JID pack=$PACK out=$OUT"
  JOB_IDS+=("$JID")
  echo "$JID $DS $OUT $PACK" >> temp/lean_disc_c128_jobs/submitted.txt
done

echo "--- job ids ---"
printf '%s\n' "${JOB_IDS[@]}"
squeue -u "${USER:-ccao87}" -o '%.18i %.40j %.2t %.10M %R' | head -20
