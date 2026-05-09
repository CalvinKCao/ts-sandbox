#!/usr/bin/env bash
# Submit all 6 U-Net architecture experiments as separate Slurm jobs (one variant each).
#
# These match the --variant cases in run.sh (4-var replication / architecture ablations):
#   default              — baseline U-Net + pipeline_config topology
#   h128                 — --image-height 128
#   attn-near-bottleneck — attention at level 1 (near bottleneck vs default level 2)
#   deeper-unet          — extra channel stage + attention level 3
#   penalty-0.1          — guidance penalty weight 0.1
#   penalty-0.3          — guidance penalty weight 0.3
#
# Usage (login node, repo root):
#   ./utils/submit_architecture_matrix.sh --hours 10
#   ./utils/submit_architecture_matrix.sh --hours 10 --dataset electricity
#   ./utils/submit_architecture_matrix.sh --hours 48 --dataset electricity --h100
#
# Does NOT bundle into one Slurm job: each variant needs its own GPU allocation and
# isolated results stem (MM-DD-JOBID-unet-fullvar-<variant>).
#
# After jobs finish, merge metrics:
#   python3 utils/collect_architecture_matrix_summaries.py results/architecture_matrix_manifest_<timestamp>.tsv

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "Run this script from the login node (not inside a Slurm job)." >&2
  exit 1
fi

HOURS=24
DATASET="electricity"
USE_H100=0
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hours)    HOURS="$2"; shift 2 ;;
    --dataset)  DATASET="$2"; shift 2 ;;
    --h100)     USE_H100=1; shift ;;
    --help|-h)
      sed -n '1,25p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    --)
      shift
      EXTRA+=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1 (use --help)" >&2
      exit 1
      ;;
  esac
done

VARIANTS=(
  default
  h128
  attn-near-bottleneck
  deeper-unet
  penalty-0.1
  penalty-0.3
)

DESCRIPTIONS=(
  "Baseline U-Net (IMAGE_HEIGHT and channels from pipeline_config.py)"
  "Taller 2D maps: image height 128"
  "Attention one level higher (near bottleneck): --attention-levels 1"
  "Deeper U-Net: channels 64–512 and --attention-levels 3"
  "Guidance penalty weight 0.1 (--guidance-penalty-weight)"
  "Guidance penalty weight 0.3 (--guidance-penalty-weight)"
)

TS="$(date +%Y%m%d_%H%M%S)"
MANIFEST="$ROOT/results/architecture_matrix_manifest_${TS}.tsv"
mkdir -p "$ROOT/results"

{
  echo "# architecture_matrix manifest — $(date -Iseconds)"
  echo "# dataset=$DATASET hours=$HOURS h100=$USE_H100"
  echo "# repo=$ROOT"
  echo -e "job_id\tvariant\tdescription"
} >"$MANIFEST"

echo ""
echo "================================================================"
echo "  Architecture matrix: 6 Slurm jobs (${DATASET}, ${HOURS}h wall each)"
echo "================================================================"
echo ""

declare -a JOB_IDS
declare -a SUBMIT_OUT

for i in "${!VARIANTS[@]}"; do
  v="${VARIANTS[$i]}"
  desc="${DESCRIPTIONS[$i]}"
  ARGS=(./run.sh --dataset "$DATASET" --hours "$HOURS" --variant "$v")
  [[ "$USE_H100" -eq 1 ]] && ARGS+=(--h100)
  [[ "${#EXTRA[@]}" -gt 0 ]] && ARGS+=("${EXTRA[@]}")

  echo ">>> ${ARGS[*]}"
  out="$("${ARGS[@]}" 2>&1)"
  printf '%s\n' "$out"
  jid="$(printf '%s\n' "$out" | sed -n 's/.*Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n1)"
  if [[ -z "$jid" ]]; then
    echo "[WARN] Could not parse Slurm job id for variant=$v" >&2
    jid="?"
  fi
  JOB_IDS+=("$jid")
  SUBMIT_OUT+=("$out")
  # shellcheck disable=SC2001
  esc_desc="$(echo "$desc" | sed 's/	/ /g')"
  echo -e "${jid}\t${v}\t${esc_desc}" >>"$MANIFEST"
  echo ""
done

echo ""
echo "================================================================"
echo "  Submitted architecture matrix — summary"
echo "================================================================"
echo ""
printf '%-22s  %-12s  %s\n' "VARIANT" "JOB_ID" "DESCRIPTION"
echo "--------------------------------------------------------------------------------"
for i in "${!VARIANTS[@]}"; do
  printf '%-22s  %-12s  %s\n' "${VARIANTS[$i]}" "${JOB_IDS[$i]}" "${DESCRIPTIONS[$i]}"
done
echo ""
echo "Manifest: $MANIFEST"
echo ""
echo "When jobs have finished, merge summary.csv rows:"
echo "  python3 utils/collect_architecture_matrix_summaries.py \\"
echo "    $MANIFEST"
echo ""
