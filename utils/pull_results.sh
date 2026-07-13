#!/bin/bash
# Pull Slurm run artifacts and reports from Killarney.
#
# Usage:
#   ./utils/pull_results.sh [--light|--all] [--recent [HOURS]] [--no-npy-ckpt] [--dry-run] [subpath ...]
#
# Modes (default: --light):
#   --light   discover all *.log, *.err, *.out, *.json via remote find, then rsync
#   --all     sync all files (no extension filter)
#
# Filters:
#   --no-npy-ckpt  skip *.npy, *.npy.*, *.npz, *.ckpt, *.pt (useful with --all)
#
# Scope:
#   No args     sync results/ + reports/ from all configured remote repo roots
#   --recent H  only files with mtime in the last H hours (default 24)
#   subpath     optional paths under results/ (e.g. logs, logs/foo.log)
#               use subpath "reports" to sync only ./reports (repo root, not under results/)
#
# Light mode always builds the file list on the cluster with find (not rsync
# include rules) so nothing is missed in nested dirs like logs/ or logs/run/.
#
# Remote find prunes heavy Phase-1 / synthetic trees (see architecture.md §2):
#   synthetic_cache, synth_data, synth_pool*.npy*, pretrained_dim*/
#   and promoted Phase-1 .pt names (itrans_hp_best.pt, diffusion.pt, …).
# --all still walks Phase-1 .pt trees; both modes skip synthetic pools.
# --no-npy-ckpt additionally drops all npy / npz / ckpt / pt regardless of mode.

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-killarney.alliancecan.ca}"
REMOTE_USER="ccao87"
LOCAL_RESULTS_PATH="./results"
LOCAL_REPORTS_PATH="./reports"

if [[ -n "${REMOTE_REPO_ROOTS:-}" ]]; then
    IFS=',' read -ra REMOTE_REPO_ROOTS <<< "${REMOTE_REPO_ROOTS}"
else
    REMOTE_REPO_ROOTS=(
        "/scratch/ccao87/ts-sandbox"
        "/scratch/ccao87/ts-sandbox-dit-parallel"
    )
fi

REMOTE_PATHS=()
for _root in "${REMOTE_REPO_ROOTS[@]}"; do
    REMOTE_PATHS+=("${_root}/results")
done

# Match ~/.ssh/config ControlPath so pull/rsync reuse an interactive ssh master.
SSH_CONTROL_DIR="${HOME}/.ssh/sockets"
mkdir -p "$SSH_CONTROL_DIR"
SSH_OPTS=(
    -o ConnectTimeout=12
    -o ConnectionAttempts=1
    -o StrictHostKeyChecking=accept-new
    -o ControlMaster=auto
    -o "ControlPath=${SSH_CONTROL_DIR}/%r@%h:%p"
)

RSYNC_BASE=(
    -avz
    --progress
    --no-g
    --no-p
    --partial
    --update
)

PULL_MODE="light"
USE_RECENT=0
RECENT_HOURS=24
DRY_RUN=0
SKIP_NPY_CKPT=0

while [[ $# -gt 0 && $1 == --* ]]; do
    case "$1" in
        --light) PULL_MODE="light"; shift ;;
        --all)   PULL_MODE="all"; shift ;;
        --recent)
            USE_RECENT=1
            shift
            if [[ -n "${1:-}" && "$1" =~ ^[0-9]+$ ]]; then
                RECENT_HOURS="$1"
                shift
            fi
            ;;
        --no-npy-ckpt) SKIP_NPY_CKPT=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--light|--all] [--recent [HOURS]] [--no-npy-ckpt] [--dry-run] [subpath ...]" >&2
            exit 1
            ;;
    esac
done

build_rsync_opts() {
    local -n out=$1
    out=("${RSYNC_BASE[@]}")
    if [ "$DRY_RUN" -eq 1 ]; then
        out+=(--dry-run -i)
    fi
    if [ "$SKIP_NPY_CKPT" -eq 1 ]; then
        out+=(--exclude '*.npy' --exclude '*.npy.*' --exclude '*.npz' --exclude '*.ckpt' --exclude '*.pt')
    fi
}

normalize_subpath() {
    local raw="$1"
    raw="${raw%/}"
    if [[ "$raw" == "reports" || "$raw" == "./reports" ]]; then
        echo "reports"
        return 0
    fi
    local rp
    for rp in "${REMOTE_REPO_ROOTS[@]}"; do
        if [[ "$raw" == "${rp}/reports" || "$raw" == "${rp}/reports/"* ]]; then
            echo "reports"
            return 0
        fi
    done
    for rp in "${REMOTE_PATHS[@]}"; do
        if [[ "$raw" == "${rp}/"* ]]; then
            echo "${raw#"${rp}/"}"
            return 0
        fi
        if [[ "$raw" == "$rp" ]]; then
            echo ""
            return 0
        fi
    done
    raw="${raw#results/}"
    raw="${raw#/}"
    echo "$raw"
}

remote_src_dir() {
    local remote_root="$1"
    local rel_subpath="${2:-}"
    if [ -z "$rel_subpath" ]; then
        echo "${REMOTE_USER}@${REMOTE_HOST}:${remote_root}/"
        return
    fi
    echo "${REMOTE_USER}@${REMOTE_HOST}:${remote_root}/${rel_subpath%/}/"
}

# Print paths relative to remote_root (what rsync --files-from expects).
# tree_kind: results | reports
remote_list_files() {
    local remote_root="$1"
    local rel_subpath="${2:-}"
    local tree_kind="${3:-results}"

    # SSH drops empty args; use a sentinel so pull_mode/use_recent/hours keep $3-$5.
    local rel_arg="${rel_subpath:-__ROOT__}"

    ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
        bash -s -- "$remote_root" "$rel_arg" "$PULL_MODE" "$USE_RECENT" "$RECENT_HOURS" "$tree_kind" "$SKIP_NPY_CKPT" <<'REMOTE_FIND'
set -euo pipefail
remote_root=${1:?}
pull_mode=${3:?}
use_recent=${4:?}
recent_hours=${5:?}
tree_kind=${6:-results}
skip_npy_ckpt=${7:-0}
remote_root=${remote_root%/}
rel=$2
[ "$rel" = "__ROOT__" ] && rel=""

search="$remote_root"
[ -n "$rel" ] && search="${remote_root}/${rel}"

is_npy_ckpt() {
    case "$1" in
        *.npy|*.npy.*|*.npz|*.ckpt|*.pt) return 0 ;;
        *) return 1 ;;
    esac
}

light_ok=0
if [ "$pull_mode" = "light" ]; then
    case "$search" in
        results)
            case "$tree_kind" in
                results) light_ok=1 ;;
            esac
            ;;
        reports)
            case "$tree_kind" in
                reports) light_ok=1 ;;
            esac
            ;;
        *.log|*.err|*.out|*.json)
            [ "$tree_kind" = "results" ] && light_ok=1
            ;;
        *.png|*.jpg|*.jpeg|*.csv|*.md)
            [ "$tree_kind" = "reports" ] && light_ok=1
            ;;
        *.json)
            light_ok=1
            ;;
    esac
else
    light_ok=1
fi

if [ -f "$search" ]; then
    [ "$light_ok" -eq 1 ] || exit 0
    if [ "$skip_npy_ckpt" = "1" ] && is_npy_ckpt "$search"; then
        exit 0
    fi
    printf '%s\n' "${search#"${remote_root}/"}"
    exit 0
fi

[ -d "$search" ] || exit 0

prune=(\( -path '*/archive/*')
if [ "$tree_kind" = "results" ]; then
    prune+=(
        -o -path '*/synthetic_cache/*'
        -o -path '*/synth_data/*'
        -o -name 'synth_pool*.npy'
        -o -name 'synth_pool*.npy.*'
    )
    if [ "$pull_mode" = "light" ]; then
        prune+=(
            -o -path '*/pretrained_dim*'
            -o -path '*/pretrain_dim*'
            -o -name 'itrans_hp_best.pt'
            -o -name 'diff_hp_best.pt'
            -o -name 'pretrained_itransformer.pt'
            -o -name 'pretrained_diffusion.pt'
            -o -name 'itransformer.pt'
            -o -name 'diffusion.pt'
        )
    fi
fi
if [ "$skip_npy_ckpt" = "1" ]; then
    prune+=(
        -o -name '*.npy'
        -o -name '*.npy.*'
        -o -name '*.npz'
        -o -name '*.ckpt'
        -o -name '*.pt'
    )
fi
prune+=(\) -prune -o)

file_args=(-type f)
if [ "$use_recent" = "1" ]; then
    file_args+=(-mmin "-$(( recent_hours * 60 ))")
fi
if [ "$pull_mode" = "light" ]; then
    if [ "$tree_kind" = "reports" ]; then
        file_args+=(
            \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg'
            -o -name '*.csv' -o -name '*.json' -o -name '*.md' \)
        )
    else
        file_args+=(\( -name '*.log' -o -name '*.err' -o -name '*.out' -o -name '*.json' \))
    fi
fi

find "$search" "${prune[@]}" "${file_args[@]}" -print 2>/dev/null \
    | sed "s|^${remote_root}/||" \
    | sort -u
REMOTE_FIND
}

pull_via_file_list() {
    local remote_root="$1"
    local tree_kind="${2:-results}"
    local local_dest="${3:-$LOCAL_RESULTS_PATH}"
    shift 3
    local -a rel_subpaths=("$@")

    local src
    src="$(remote_src_dir "$remote_root" "")"

    local -a rsync_opts
    build_rsync_opts rsync_opts

    local scope="all files"
    if [ "$PULL_MODE" = "light" ]; then
        if [ "$tree_kind" = "reports" ]; then
            scope="*.png *.jpg *.jpeg *.csv *.json *.md"
        else
            scope="*.log *.err *.out *.json"
        fi
    fi
    local prune_note=""
    [ "$PULL_MODE" = "light" ] && [ "$tree_kind" = "results" ] && prune_note=", skip synth pools + phase-1 ckpts"
    [ "$SKIP_NPY_CKPT" -eq 1 ] && prune_note="${prune_note}, skip npy/npz/ckpt/pt"
    local label="$remote_root"
    if [ "${#rel_subpaths[@]}" -eq 1 ]; then
        label="${remote_root}/${rel_subpaths[0]}"
    elif [ "${#rel_subpaths[@]}" -gt 1 ]; then
        label="${remote_root} (${#rel_subpaths[@]} subpaths)"
    fi
    if [ "$USE_RECENT" -eq 1 ]; then
        echo "  -> ${label} (${scope}, mtime < ${RECENT_HOURS}h${prune_note})"
    else
        echo "  -> ${label} (${scope}${prune_note})"
    fi

    local file_list
    file_list="$(mktemp)"
    if [ "${#rel_subpaths[@]}" -eq 0 ]; then
        remote_list_files "$remote_root" "" "$tree_kind" >"$file_list" || true
    else
        local rel
        for rel in "${rel_subpaths[@]}"; do
            remote_list_files "$remote_root" "$rel" "$tree_kind" >>"$file_list" || true
        done
        sort -u -o "$file_list" "$file_list"
    fi

    local n
    n="$(wc -l <"$file_list" | tr -d ' ')"
    if [ "$n" -eq 0 ]; then
        rm -f "$file_list"
        echo "     0 files on remote (check SSH / path / --recent window)"
        return
    fi
    echo "     ${n} files on remote (one rsync)"

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "     (dry-run) newest 10 paths:"
        tail -10 "$file_list" | sed 's/^/       /'
    fi

    rsync -e "ssh ${SSH_OPTS[*]}" "${rsync_opts[@]}" --ignore-missing-args \
        --files-from="$file_list" "${src}" "${local_dest}/"
    rm -f "$file_list"
}

pull_full_tree() {
    local remote_root="$1"
    local rel_subpath="${2:-}"
    local local_dest="${3:-$LOCAL_RESULTS_PATH}"
    local label="$remote_root"
    [ -n "$rel_subpath" ] && label="${remote_root}/${rel_subpath}"

    local src
    if [ -n "$rel_subpath" ]; then
        src="${REMOTE_USER}@${REMOTE_HOST}:${remote_root}/${rel_subpath}"
    else
        src="${REMOTE_USER}@${REMOTE_HOST}:${remote_root}/"
    fi

    local -a rsync_opts
    build_rsync_opts rsync_opts

    local skip_note=""
    [ "$SKIP_NPY_CKPT" -eq 1 ] && skip_note=", skip npy/npz/ckpt/pt"
    echo "  -> ${label} (full tree${skip_note})"
    rsync -e "ssh ${SSH_OPTS[*]}" "${rsync_opts[@]}" --ignore-missing-args \
        "${src}" "${local_dest}/"
}

pull_tree() {
    local remote_root="$1"
    local tree_kind="${2:-results}"
    local local_dest="${3:-$LOCAL_RESULTS_PATH}"
    shift 3
    local -a rel_subpaths=("$@")

    if [ "$PULL_MODE" = "light" ] || [ "$USE_RECENT" -eq 1 ]; then
        pull_via_file_list "$remote_root" "$tree_kind" "$local_dest" "${rel_subpaths[@]}"
    elif [ "${#rel_subpaths[@]}" -eq 0 ]; then
        pull_full_tree "$remote_root" "" "$local_dest"
    elif [ "${#rel_subpaths[@]}" -eq 1 ]; then
        pull_full_tree "$remote_root" "${rel_subpaths[0]}" "$local_dest"
    else
        pull_via_file_list "$remote_root" "$tree_kind" "$local_dest" "${rel_subpaths[@]}"
    fi
}

mkdir -p "$LOCAL_RESULTS_PATH" "$LOCAL_REPORTS_PATH"

SUBPATHS=()
REPORTS_EXPLICIT=0
if [ $# -gt 0 ]; then
    while [ $# -gt 0 ]; do
        rel="$(normalize_subpath "$1")"
        if [ "$rel" = "archive" ]; then
            :
        elif [ "$rel" = "reports" ]; then
            REPORTS_EXPLICIT=1
        else
            SUBPATHS+=("$rel")
        fi
        shift
    done
fi

PULL_RESULTS=1
PULL_REPORTS=0
if [ "$REPORTS_EXPLICIT" -eq 0 ] && [ "${#SUBPATHS[@]}" -eq 0 ]; then
    PULL_REPORTS=1
elif [ "$REPORTS_EXPLICIT" -eq 1 ] && [ "${#SUBPATHS[@]}" -eq 0 ]; then
    PULL_RESULTS=0
    PULL_REPORTS=1
elif [ "$REPORTS_EXPLICIT" -eq 1 ]; then
    PULL_REPORTS=1
fi

echo "Pulling from ${REMOTE_HOST} (${PULL_MODE} mode, dest=${LOCAL_RESULTS_PATH})..."
[ "$PULL_REPORTS" -eq 1 ] && echo "Also syncing reports -> ${LOCAL_REPORTS_PATH}"
[ "$DRY_RUN" -eq 1 ] && echo "(dry-run: no files written)"
[ "$USE_RECENT" -eq 1 ] && echo "Only files modified in the last ${RECENT_HOURS} hours"
[ "$SKIP_NPY_CKPT" -eq 1 ] && echo "Skipping *.npy / *.npz / *.ckpt / *.pt"

# One SSH session (shared ControlMaster socket) for find + rsync below.
ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" true

if [ "$PULL_RESULTS" -eq 1 ]; then
    for RP in "${REMOTE_PATHS[@]}"; do
        echo "Remote: ${RP}"
        pull_tree "$RP" "results" "$LOCAL_RESULTS_PATH" "${SUBPATHS[@]}"
    done
fi

if [ "$PULL_REPORTS" -eq 1 ]; then
    for RR in "${REMOTE_REPO_ROOTS[@]}"; do
        echo "Remote: ${RR}/reports"
        pull_tree "${RR}/reports" "reports" "$LOCAL_REPORTS_PATH"
    done
fi

ssh "${SSH_OPTS[@]}" -O exit "${REMOTE_USER}@${REMOTE_HOST}" 2>/dev/null || true
echo "Done."
