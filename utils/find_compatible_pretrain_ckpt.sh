#!/usr/bin/env bash
# Find synthetic pretrain ckpts the pipeline would reuse for reuse_pretrain_from_config.
#
# Mirrors staged_diffusion_pretrain lookup order:
#   1) $SCRATCH/ts-sandbox/reused/pretrain/<config>/pretrained_{coarse,fine}/pretrained_diffusion.pt
#   2) newest *-<dataset>-<config>/pretrained_*/pretrained_diffusion.pt under results/ckpts
#   3) cross-dataset fallback: newest *-*-<config>/ with both stages
#
# USAGE (Killarney login or WSL):
#   ./utils/find_compatible_pretrain_ckpt.sh
#   ./utils/find_compatible_pretrain_ckpt.sh --dataset electricity
#   ./utils/find_compatible_pretrain_ckpt.sh --config binary_anchor_ar_patch_decoder_ctx_lb336_hz720_fixed --dataset traffic,electricity
#   ./utils/find_compatible_pretrain_ckpt.sh --all-ckpt-roots   # brute list every matching file

set -euo pipefail

CONFIG="binary_anchor_ar_patch_decoder_ctx_lb336_hz720_fixed"
DATASETS="ETTh1,traffic,electricity,exchange_rate"
STAGES="coarse fine"
ALL_ROOTS=0

usage() {
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage 0 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --dataset|--datasets) DATASETS="$2"; shift 2 ;;
        --stage) STAGES="$2"; shift 2 ;;
        --all-ckpt-roots) ALL_ROOTS=1; shift ;;
        *) echo "Unknown arg: $1" >&2; usage 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

ckpt_roots=()
add_root() {
    local p="${1:-}"
    [[ -z "$p" || ! -d "$p" ]] && return 0
    p="$(cd "$p" && pwd)"
    local r
    for r in "${ckpt_roots[@]:-}"; do
        [[ "$r" == "$p" ]] && return 0
    done
    ckpt_roots+=("$p")
}

USER_NAME="${USER:-$(whoami)}"
SCRATCH="${SCRATCH:-}"

add_root "$REPO/results/ckpts"
if [[ -n "$SCRATCH" ]]; then
    add_root "$SCRATCH/ts-sandbox/results/ckpts"
    add_root "$SCRATCH/$USER_NAME/ts-sandbox/results/ckpts"
fi
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    add_root "$SLURM_SUBMIT_DIR/results/ckpts"
fi
if [[ -n "${RESULTS_ROOT:-}" ]]; then
    add_root "$RESULTS_ROOT/ckpts"
fi

REUSED_ROOT="${SCRATCH:+$SCRATCH/ts-sandbox/reused}"
[[ -z "$REUSED_ROOT" || ! -d "$REUSED_ROOT" ]] && REUSED_ROOT="$REPO/reused"

run_matches_dataset_config() {
    local name="$1" dataset="$2" cfg="$3"
    [[ "$name" == *"-${dataset}-${cfg}" ]] || return 1
    local suffix="${name#*-${dataset}-${cfg}}"
    [[ -z "$suffix" ]]
}

run_matches_any_dataset_config() {
    local name="$1" cfg="$2"
    [[ "$name" == *"-${cfg}" ]] || return 1
    local body="${name%-${cfg}}"
    [[ "$(grep -o '-' <<< "$body" | wc -l)" -ge 3 ]]
}

newest_run_with_file() {
    local dataset="$1" require_same_ds="$2" file_rel="$3"
    local root name path best="" best_mtime=0 mtime
    for root in "${ckpt_roots[@]}"; do
        shopt -s nullglob
        for path in "$root"/*; do
            shopt -u nullglob
            [[ -d "$path" ]] || continue
            name="$(basename "$path")"
            if [[ "$require_same_ds" -eq 1 ]]; then
                run_matches_dataset_config "$name" "$dataset" "$CONFIG" || continue
            else
                run_matches_any_dataset_config "$name" "$CONFIG" || continue
            fi
            [[ -f "$path/$file_rel" ]] || continue
            mtime=$(stat -c %Y "$path" 2>/dev/null || echo 0)
            if [[ "$mtime" -gt "$best_mtime" ]]; then
                best_mtime="$mtime"
                best="$path"
            fi
        done
        shopt -u nullglob 2>/dev/null || true
    done
    [[ -n "$best" ]] && echo "$best"
}

print_hit() {
    local label="$1" path="$2" stage="$3"
    if [[ -f "$path" ]]; then
        local sz mtime
        sz=$(du -h "$path" | awk '{print $1}')
        mtime=$(stat -c '%y' "$path" 2>/dev/null | cut -d. -f1)
        printf '  OK  %-28s %s\n' "[$label/$stage]" "$path"
        printf '      %-28s %s bytes  %s\n' '' "$sz" "$mtime"
    else
        printf '  --  %-28s (missing)\n' "[$label/$stage]"
    fi
}

IFS=',' read -ra DS_ARR <<< "$DATASETS"

echo "config:  $CONFIG"
echo "reused:  $REUSED_ROOT/pretrain/$CONFIG/"
echo "ckpt roots:"
for root in "${ckpt_roots[@]}"; do
    echo "  - $root"
done
echo

if [[ "$ALL_ROOTS" -eq 1 ]]; then
    echo "=== all pretrained_{coarse,fine}/pretrained_diffusion.pt under *-*-${CONFIG} ==="
    for root in "${ckpt_roots[@]}"; do
        while IFS= read -r -d '' f; do
            echo "$f"
        done < <(find "$root" -maxdepth 3 -type f -path "*/pretrained_*/pretrained_diffusion.pt" 2>/dev/null \
            | while read -r f; do
                d="$(basename "$(dirname "$(dirname "$f")")")"
                if [[ "$d" == *"-${CONFIG}" ]]; then echo "$f"; fi
              done | tr '\n' '\0')
    done
    exit 0
fi

for DS in "${DS_ARR[@]}"; do
    echo "=== $DS ==="
    for ST in $STAGES; do
        rel="pretrained_${ST}/pretrained_diffusion.pt"
        reused_path="$REUSED_ROOT/pretrain/$CONFIG/pretrained_${ST}/pretrained_diffusion.pt"
        print_hit "reused" "$reused_path" "$ST"

        same_dir=$(newest_run_with_file "$DS" 1 "$rel" || true)
        if [[ -n "$same_dir" ]]; then
            print_hit "same-dataset" "$same_dir/$rel" "$ST"
        else
            printf '  --  %-28s (no *-%s-%s)\n' "[same-dataset/$ST]" "$DS" "$CONFIG"
        fi

        any_dir=$(newest_run_with_file "$DS" 0 "$rel" || true)
        if [[ -n "$any_dir" ]]; then
            if [[ "$any_dir" != "${same_dir:-}" ]]; then
                print_hit "cross-dataset" "$any_dir/$rel" "$ST"
            fi
        else
            printf '  --  %-28s (no *-*-%s)\n' "[cross-dataset/$ST]" "$CONFIG"
        fi

        # What the pipeline would pick (first hit in order).
        pick=""
        pick_label=""
        if [[ -f "$reused_path" ]]; then
            pick="$reused_path"; pick_label="reused"
        elif [[ -n "$same_dir" ]]; then
            pick="$same_dir/$rel"; pick_label="same-dataset"
        elif [[ -n "$any_dir" ]]; then
            pick="$any_dir/$rel"; pick_label="cross-dataset"
        fi
        if [[ -n "$pick" ]]; then
            printf '  => pipeline pick (%s): %s\n' "$pick_label" "$pick"
        else
            printf '  => pipeline pick: NONE (would train synthetic pretrain)\n'
        fi
        echo
    done
done
