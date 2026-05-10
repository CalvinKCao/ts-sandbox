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
#   ./utils/submit_architecture_matrix.sh --benchmark-full --hours 48
#   ./utils/submit_architecture_matrix.sh --hours 48 \
#     --datasets ETTm2,ETTh2,exchange_rate,weather
#   ./utils/submit_architecture_matrix.sh --hours 48 --dataset electricity --h100
#
# Full benchmark (native variates, no electricity 4-var subset):
#   ./utils/submit_architecture_matrix.sh --benchmark-full --hours 48
#
# Options:
#   --benchmark-full     Submit all four: ETTm2, ETTh2, exchange_rate, weather
#   --datasets A,B,C     Comma-separated dataset keys (same names as run.sh --dataset)
#   --electricity-4var   With --datasets, use the 4-var electricity subset when electricity is listed
#
# Does NOT bundle into one Slurm job: each variant needs its own GPU allocation and
# isolated results stem (MM-DD-JOBID-<variant>-<dataset>, optional -smoke / -h100 suffix from run.sh).
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
DATASETS_RAW=""
USE_H100=0
BENCHMARK_FULL=0
ELECTRICITY_4VAR=0
EXTRA=()

# Default 4-variate subset for standalone electricity runs (see notes/4-variate-subset-note.md).
SUBSET_INDICES="93,292,81,84"
SUBSET_ID="4var"

BENCHMARK_FULL_LIST="ETTm2,ETTh2,exchange_rate,weather"

usage() {
  sed -n '1,33p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hours)         HOURS="$2"; shift 2 ;;
    --dataset)       DATASET="$2"; shift 2 ;;
    --datasets)      DATASETS_RAW="$2"; shift 2 ;;
    --benchmark-full)
      BENCHMARK_FULL=1
      shift
      ;;
    --electricity-4var)
      ELECTRICITY_4VAR=1
      shift
      ;;
    --h100)          USE_H100=1; shift ;;
    --help|-h)
      usage
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

declare -a DATASETS
if [[ "$BENCHMARK_FULL" -eq 1 && -z "$DATASETS_RAW" ]]; then
  DATASETS_RAW="$BENCHMARK_FULL_LIST"
fi

if [[ -n "$DATASETS_RAW" ]]; then
  IFS=',' read -r -a _parts <<< "$DATASETS_RAW"
  for x in "${_parts[@]}"; do
    x="$(echo "$x" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [[ -z "$x" ]] && continue
    DATASETS+=("$x")
  done
  if [[ "${#DATASETS[@]}" -eq 0 ]]; then
    echo "[ERROR] No datasets parsed from --datasets / --benchmark-full" >&2
    exit 1
  fi
else
  DATASETS=("$DATASET")
fi

# Legacy: default single `--dataset electricity` (no `--datasets`) uses the 4-variate subset.
# Any explicit `--datasets …` / `--benchmark-full` uses native variates per dataset; pass
# `--electricity-4var` if electricity appears in that list and you want the 4-var subset.
use_electricity_subset_for() {
  local ds="$1"
  [[ "$ds" != "electricity" ]] && return 1
  if [[ -n "$DATASETS_RAW" ]]; then
    [[ "$ELECTRICITY_4VAR" -eq 1 ]] && return 0
    return 1
  fi
  [[ "${#DATASETS[@]}" -eq 1 && "${DATASETS[0]}" == "electricity" ]]
}

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
  echo "# datasets=${DATASETS[*]} hours=$HOURS h100=$USE_H100 benchmark_full=$BENCHMARK_FULL electricity_4var=$ELECTRICITY_4VAR"
  echo "# repo=$ROOT"
  echo -e "job_id\tvariant\tdataset\tdescription"
} >"$MANIFEST"

total_jobs=$((${#VARIANTS[@]} * ${#DATASETS[@]}))

echo ""
echo "================================================================"
echo "  Architecture matrix: ${total_jobs} Slurm jobs (${#DATASETS[@]} dataset(s), ${#VARIANTS[@]} variants, ${HOURS}h wall each)"
echo "================================================================"
echo ""

declare -a JOB_IDS
declare -a SUBMIT_OUT

for ds in "${DATASETS[@]}"; do
  for i in "${!VARIANTS[@]}"; do
    v="${VARIANTS[$i]}"
    desc="${DESCRIPTIONS[$i]}"
    ARGS=(./run.sh --dataset "$ds" --hours "$HOURS" --variant "$v")
    [[ "$USE_H100" -eq 1 ]] && ARGS+=(--h100)
    if use_electricity_subset_for "$ds"; then
      ARGS+=(--variate-indices "${SUBSET_ID}:${SUBSET_INDICES}")
    fi
    [[ "${#EXTRA[@]}" -gt 0 ]] && ARGS+=("${EXTRA[@]}")

    echo ">>> ${ARGS[*]}"
    out="$("${ARGS[@]}" 2>&1)"
    printf '%s\n' "$out"
    jid="$(printf '%s\n' "$out" | sed -n 's/.*Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n1)"
    if [[ -z "$jid" ]]; then
      echo "[WARN] Could not parse Slurm job id for dataset=$ds variant=$v" >&2
      jid="?"
    fi
    JOB_IDS+=("$jid")
    SUBMIT_OUT+=("$out")
    # shellcheck disable=SC2001
    esc_desc="$(echo "$desc" | sed 's/	/ /g')"
    echo -e "${jid}\t${v}\t${ds}\t${esc_desc}" >>"$MANIFEST"
    echo ""
  done
done

echo ""
echo "================================================================"
echo "  Submitted architecture matrix — summary"
echo "================================================================"
echo ""
printf '%-12s  %-22s  %s\n' "DATASET" "VARIANT" "JOB_ID"
echo "--------------------------------------------------------------------------------"
idx=0
for ds in "${DATASETS[@]}"; do
  for i in "${!VARIANTS[@]}"; do
    printf '%-12s  %-22s  %s\n' "$ds" "${VARIANTS[$i]}" "${JOB_IDS[$idx]}"
    idx=$((idx + 1))
  done
done
echo ""
echo "Manifest: $MANIFEST"
echo ""
echo "When jobs have finished, merge summary.csv rows:"
echo "  python3 utils/collect_architecture_matrix_summaries.py \\"
echo "    $MANIFEST"
echo ""
