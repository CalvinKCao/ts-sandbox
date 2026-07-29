#!/bin/bash
# Submit the matched h96 ordinal binary + non-ordinal MMPD campaign.
#
# This is a coordinator only. Training is submitted exclusively through
# submit_binary.sh / submit_mmpd.sh; the final discriminator is the explicit
# deferred mode of submit_binary.sh. Job IDs move through JSON manifests, never
# by scraping human-readable sbatch output.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASETS="electricity,ETTh1,dynamic,traffic"
BINARY_CONFIG="binary_patch_refine_lb336_hz96_ordinal_tuned"
MMPD_CONFIG="mmpd_decoder_flat_subsets_paper_lb336_hz96_matched_binary"
DONOR_CONFIG="binary_patch_refine_lb336_hz96_full"
RUN_NAME="$(date +%m-%d-%H%M)-h96-ordinal-binary-nonordinal-mmpd"
BINARY_TIME="12:00:00"
MMPD_TIME="12:00:00"
DISC_TIME="3:00:00"
PREFLIGHT_ONLY=0
ALLOW_SYNTHETIC_PRETRAIN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets) DATASETS="$2"; shift 2 ;;
        --binary-config) BINARY_CONFIG="$2"; shift 2 ;;
        --mmpd-config) MMPD_CONFIG="$2"; shift 2 ;;
        --donor-config) DONOR_CONFIG="$2"; shift 2 ;;
        --run-name) RUN_NAME="$2"; shift 2 ;;
        --binary-time) BINARY_TIME="$2"; shift 2 ;;
        --mmpd-time) MMPD_TIME="$2"; shift 2 ;;
        --disc-time) DISC_TIME="$2"; shift 2 ;;
        --preflight-only) PREFLIGHT_ONLY=1; shift ;;
        --allow-synthetic-pretrain) ALLOW_SYNTHETIC_PRETRAIN=1; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ "$ALLOW_SYNTHETIC_PRETRAIN" -eq 1 ]]; then
    [[ "$BINARY_CONFIG" == "binary_patch_refine_lb336_hz96_ordinal_tuned" ]] || {
        echo "ERROR: --allow-synthetic-pretrain supports only the default ordinal h96 binary config." >&2
        exit 1
    }
    BINARY_CONFIG="binary_patch_refine_lb336_hz96_ordinal_tuned_synth_fallback"
fi

[[ -z "${SLURM_JOB_ID:-}" ]] || { echo "ERROR: run this coordinator from a login node." >&2; exit 1; }
[[ -d "${SCRATCH:-}/ts-sandbox" ]] || {
    echo "ERROR: expected Killarney checkout at \$SCRATCH/ts-sandbox." >&2
    exit 1
}
[[ "$REPO" == "${SCRATCH}/ts-sandbox" ]] || {
    echo "ERROR: run $0 from \$SCRATCH/ts-sandbox, not $REPO." >&2
    exit 1
}
# shellcheck source=utils/mmpd_submit_helpers.sh
source "$REPO/utils/mmpd_submit_helpers.sh"

resolve_config() {
    local token="$1" candidate
    candidate="$token"
    [[ "$candidate" == *.yaml ]] || candidate+=".yaml"
    if [[ -f "$REPO/configs/$candidate" ]]; then
        printf '%s\n' "$REPO/configs/$candidate"
    elif [[ -f "$REPO/$candidate" ]]; then
        printf '%s\n' "$REPO/$candidate"
    else
        echo "ERROR: config not found: $token" >&2
        return 1
    fi
}

BINARY_CONFIG_PATH="$(resolve_config "$BINARY_CONFIG")"
MMPD_CONFIG_PATH="$(resolve_config "$MMPD_CONFIG")"

grep -Eq "^[[:space:]]*reuse_pretrain_from_config:[[:space:]]*${DONOR_CONFIG}[[:space:]]*$" "$BINARY_CONFIG_PATH" || {
    echo "ERROR: $BINARY_CONFIG_PATH must explicitly set reuse_pretrain_from_config: $DONOR_CONFIG" >&2
    exit 1
}
grep -Eq "^[[:space:]]*use_window_normalization:[[:space:]]*false[[:space:]]*$" "$BINARY_CONFIG_PATH" || {
    echo "ERROR: binary config must disable window normalization." >&2
    exit 1
}
grep -Eq "^[[:space:]]*use_ordinal_window_norm:[[:space:]]*true[[:space:]]*$" "$BINARY_CONFIG_PATH" || {
    echo "ERROR: binary config must enable ordinal window normalization." >&2
    exit 1
}
grep -Eq "^[[:space:]]*horizon:[[:space:]]*96[[:space:]]*$" "$MMPD_CONFIG_PATH" || {
    echo "ERROR: MMPD config must set horizon: 96." >&2
    exit 1
}
if [[ "$ALLOW_SYNTHETIC_PRETRAIN" -eq 0 ]]; then
    grep -Eq "^[[:space:]]*require_reuse_pretrain:[[:space:]]*true[[:space:]]*$" "$BINARY_CONFIG_PATH" || {
        echo "ERROR: strict binary config must require reused synthetic pretrains." >&2
        exit 1
    }
else
    grep -Eq "^[[:space:]]*require_reuse_pretrain:[[:space:]]*false[[:space:]]*$" "$BINARY_CONFIG_PATH" || {
        echo "ERROR: synthetic fallback config must explicitly allow a new synthetic pretrain." >&2
        exit 1
    }
fi
find_dataset_donor() {
    local dataset="$1" stage="$2" global_path newest="" candidate
    global_path="$REPO/reused/pretrain/$DONOR_CONFIG/pretrained_${stage}/pretrained_diffusion.pt"
    if [[ -f "$global_path" ]]; then
        printf '%s\n' "$global_path"
        return 0
    fi
    shopt -s nullglob
    local candidates=("$REPO"/results/ckpts/*-"$dataset"-"$DONOR_CONFIG"/pretrained_"$stage"/pretrained_diffusion.pt)
    shopt -u nullglob
    for candidate in "${candidates[@]}"; do
        [[ -f "$candidate" ]] || continue
        if [[ -z "$newest" || "$candidate" -nt "$newest" ]]; then
            newest="$candidate"
        fi
    done
    [[ -n "$newest" ]] || return 1
    printf '%s\n' "$newest"
}

IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"
declare -A DONOR_COARSE_BY_DATASET=()
declare -A DONOR_REFINE_BY_DATASET=()
declare -A MMPD_DATA_BY_DATASET=()
for dataset in "${DATASET_ARRAY[@]}"; do
    mmpd_dataset_path="$(mmpd_dataset_file_path "$dataset" "$REPO" 2>/dev/null || true)"
    [[ -n "$mmpd_dataset_path" && -f "$mmpd_dataset_path" ]] || {
        echo "ERROR: locked MMPD dataset is unavailable: $dataset (${mmpd_dataset_path:-unknown path})" >&2
        exit 1
    }
    MMPD_DATA_BY_DATASET["$dataset"]="$mmpd_dataset_path"
    if donor="$(find_dataset_donor "$dataset" coarse)"; then
        DONOR_COARSE_BY_DATASET["$dataset"]="$donor"
    elif [[ "$ALLOW_SYNTHETIC_PRETRAIN" -eq 1 ]]; then
        DONOR_COARSE_BY_DATASET["$dataset"]="MISSING: will train synthetic coarse stage"
    else
        echo "ERROR: missing synthetic coarse donor for $dataset. Expected reused/pretrain/$DONOR_CONFIG or results/ckpts/*-$dataset-$DONOR_CONFIG/pretrained_coarse/pretrained_diffusion.pt" >&2
        exit 1
    fi
    if donor="$(find_dataset_donor "$dataset" patch_refine)"; then
        DONOR_REFINE_BY_DATASET["$dataset"]="$donor"
    elif [[ "$ALLOW_SYNTHETIC_PRETRAIN" -eq 1 ]]; then
        DONOR_REFINE_BY_DATASET["$dataset"]="MISSING: will train synthetic patch-refine stage"
    else
        echo "ERROR: missing synthetic patch-refine donor for $dataset. Expected reused/pretrain/$DONOR_CONFIG or results/ckpts/*-$dataset-$DONOR_CONFIG/pretrained_patch_refine/pretrained_diffusion.pt" >&2
        exit 1
    fi
done

printf '%s\n' "[preflight] binary config: $BINARY_CONFIG_PATH"
printf '%s\n' "[preflight] MMPD config:   $MMPD_CONFIG_PATH"
for dataset in "${DATASET_ARRAY[@]}"; do
    printf '%s\n' "[preflight] $dataset MMPD data: ${MMPD_DATA_BY_DATASET[$dataset]}"
    printf '%s\n' "[preflight] $dataset coarse donor: ${DONOR_COARSE_BY_DATASET[$dataset]}"
    printf '%s\n' "[preflight] $dataset refine donor: ${DONOR_REFINE_BY_DATASET[$dataset]}"
done
printf '%s\n' "[preflight] datasets:      $DATASETS"
printf '%s\n' "[preflight] synthetic fallback: $([[ "$ALLOW_SYNTHETIC_PRETRAIN" -eq 1 ]] && echo enabled || echo disabled)"
if [[ "$PREFLIGHT_ONLY" -eq 1 ]]; then
    exit 0
fi

DAG_DIR="$REPO/results/datasets/$RUN_NAME-dag"
MMPD_OUTPUT="$REPO/results/datasets/$RUN_NAME-mmpd"
mkdir -p "$DAG_DIR"
BINARY_MANIFEST="$DAG_DIR/binary_submission.json"
MMPD_MANIFEST="$DAG_DIR/mmpd_submission.json"
DISC_MANIFEST="$DAG_DIR/discriminator_submission.json"
ASSERT_MANIFEST="$DAG_DIR/assertion_submission.json"
MANIFEST_TOOL=(python3 "$REPO/temp/submission_manifest.py")

cd "$REPO"
./submit_binary.sh \
    --configs "$BINARY_CONFIG" \
    --datasets "$DATASETS" \
    --seeds 42 \
    --time "$BINARY_TIME" \
    --job-manifest "$BINARY_MANIFEST"

./submit_mmpd.sh \
    --mmpd-run-config "$MMPD_CONFIG" \
    --datasets "$DATASETS" \
    --output-dir "$MMPD_OUTPUT" \
    --time "$MMPD_TIME" \
    --mmpd-instance-norm \
    --job-manifest "$MMPD_MANIFEST"

BINARY_ROOTS="$("${MANIFEST_TOOL[@]}" checkpoint-root-pairs --path "$BINARY_MANIFEST")"
BINARY_TERMINALS="$("${MANIFEST_TOOL[@]}" terminal-job-ids --path "$BINARY_MANIFEST" --roles binary_train)"
MMPD_TERMINAL="$("${MANIFEST_TOOL[@]}" terminal-job-ids --path "$MMPD_MANIFEST" --roles mmpd_merge)"
MMPD_ROOT="$("${MANIFEST_TOOL[@]}" value --path "$MMPD_MANIFEST" --key output_root)"
ALL_DEPENDENCY="afterok:${BINARY_TERMINALS}:${MMPD_TERMINAL}"

./submit_binary.sh \
    --eval-ordinal-patch-refine-vs-mmpd \
    --ordinal-assert-only \
    --datasets "$DATASETS" \
    --existing-ckpt-roots "$BINARY_ROOTS" \
    --mmpd-root "$MMPD_ROOT" \
    --ordinal-binary-config "$BINARY_CONFIG_PATH" \
    --ordinal-disc-evaluator temp/eval_univariate_patch_refine_ordinal_vs_mmpd.py \
    --defer-checkpoint-check \
    --dependency "$ALL_DEPENDENCY" \
    --disc-run "$RUN_NAME-assert" \
    --raw-run "$RUN_NAME-assert-raw" \
    --time "$DISC_TIME" \
    --job-manifest "$ASSERT_MANIFEST"

ASSERT_TERMINALS="$("${MANIFEST_TOOL[@]}" terminal-job-ids --path "$ASSERT_MANIFEST" --roles ordinal_assert)"

./submit_binary.sh \
    --eval-ordinal-patch-refine-vs-mmpd \
    --datasets "$DATASETS" \
    --existing-ckpt-roots "$BINARY_ROOTS" \
    --mmpd-root "$MMPD_ROOT" \
    --ordinal-binary-config "$BINARY_CONFIG_PATH" \
    --ordinal-disc-evaluator temp/eval_univariate_patch_refine_ordinal_vs_mmpd.py \
    --defer-checkpoint-check \
    --dependency "afterok:${ASSERT_TERMINALS}" \
    --disc-run "$RUN_NAME-disc" \
    --raw-run "$RUN_NAME-disc-raw" \
    --time "$DISC_TIME" \
    --job-manifest "$DISC_MANIFEST"

printf '%s\n' "[submitted] binary manifest: $BINARY_MANIFEST"
printf '%s\n' "[submitted] MMPD manifest:   $MMPD_MANIFEST"
printf '%s\n' "[submitted] assertion manifest: $ASSERT_MANIFEST"
printf '%s\n' "[submitted] disc manifest:   $DISC_MANIFEST"
