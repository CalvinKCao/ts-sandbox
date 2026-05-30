#!/bin/bash
# Pull Slurm run artifacts from Killarney.
#
# Usage:
#   ./utils/pull_results.sh [--light|--all] [--recent [HOURS]] [--dry-run] [subpath ...]
#
# Modes (default: --light):
#   --light   discover all *.log, *.err, *.out, *.json via remote find, then rsync
#   --all     sync all files (no extension filter)
#
# Scope:
#   No args     sync from all configured remote results roots
#   --recent H  only files with mtime in the last H hours (default 24)
#   subpath     optional paths under results/ (e.g. logs, logs/foo.log)
#
# Light mode always builds the file list on the cluster with find (not rsync
# include rules) so nothing is missed in nested dirs like logs/ or logs/run/.

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-killarney.alliancecan.ca}"
REMOTE_USER="ccao87"
LOCAL_PATH="./results"

REMOTE_PATHS=(
    "/scratch/ccao87/ts-sandbox/results"
    "/scratch/ccao87/ts-sandbox-dit-parallel/results"
)

SSH_OPTS=(
    -o ConnectTimeout=12
    -o ConnectionAttempts=1
    -o StrictHostKeyChecking=accept-new
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
        --dry-run) DRY_RUN=1; shift ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--light|--all] [--recent [HOURS]] [--dry-run] [subpath ...]" >&2
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
}

normalize_subpath() {
    local raw="$1"
    raw="${raw%/}"
    local rp
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
remote_list_files() {
    local remote_root="$1"
    local rel_subpath="${2:-}"

    # SSH drops empty args; use a sentinel so pull_mode/use_recent/hours keep $3-$5.
    local rel_arg="${rel_subpath:-__ROOT__}"

    ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
        bash -s -- "$remote_root" "$rel_arg" "$PULL_MODE" "$USE_RECENT" "$RECENT_HOURS" <<'REMOTE_FIND'
set -euo pipefail
remote_root=${1:?}
pull_mode=${3:?}
use_recent=${4:?}
recent_hours=${5:?}
remote_root=${remote_root%/}
rel=$2
[ "$rel" = "__ROOT__" ] && rel=""

search="$remote_root"
[ -n "$rel" ] && search="${remote_root}/${rel}"

if [ -f "$search" ]; then
    case "$pull_mode" in
        light)
            case "$search" in
                *.log|*.err|*.out|*.json) ;;
                *) exit 0 ;;
            esac
            ;;
    esac
    printf '%s\n' "${search#"${remote_root}/"}"
    exit 0
fi

[ -d "$search" ] || exit 0

args=(-type f ! -path '*/archive/*')
if [ "$use_recent" = "1" ]; then
    args+=(-mmin "-$(( recent_hours * 60 ))")
fi
if [ "$pull_mode" = "light" ]; then
    args+=(\( -name '*.log' -o -name '*.err' -o -name '*.out' -o -name '*.json' \))
fi

find "$search" "${args[@]}" -print 2>/dev/null \
    | sed "s|^${remote_root}/||" \
    | sort -u
REMOTE_FIND
}

pull_via_file_list() {
    local remote_root="$1"
    local rel_subpath="${2:-}"
    local label="$remote_root"
    [ -n "$rel_subpath" ] && label="${remote_root}/${rel_subpath}"

    local src
    src="$(remote_src_dir "$remote_root" "$rel_subpath")"

    local -a rsync_opts
    build_rsync_opts rsync_opts

    local scope="all files"
    [ "$PULL_MODE" = "light" ] && scope="*.log *.err *.out *.json"
    if [ "$USE_RECENT" -eq 1 ]; then
        echo "  -> ${label} (${scope}, mtime < ${RECENT_HOURS}h)"
    else
        echo "  -> ${label} (${scope})"
    fi

    local file_list
    file_list="$(mktemp)"
    remote_list_files "$remote_root" "$rel_subpath" >"$file_list" || true

    local n
    n="$(wc -l <"$file_list" | tr -d ' ')"
    if [ "$n" -eq 0 ]; then
        rm -f "$file_list"
        echo "     0 files on remote (check SSH / path / --recent window)"
        return
    fi
    echo "     ${n} files on remote"

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "     (dry-run) newest 10 paths:"
        tail -10 "$file_list" | sed 's/^/       /'
    fi

    rsync -e "ssh ${SSH_OPTS[*]}" "${rsync_opts[@]}" --ignore-missing-args \
        --files-from="$file_list" "${src}" "${LOCAL_PATH}/"
    rm -f "$file_list"
}

pull_full_tree() {
    local remote_root="$1"
    local rel_subpath="${2:-}"
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

    echo "  -> ${label} (full tree)"
    rsync -e "ssh ${SSH_OPTS[*]}" "${rsync_opts[@]}" --ignore-missing-args \
        "${src}" "${LOCAL_PATH}/"
}

pull_tree() {
    local remote_root="$1"
    local rel_subpath="${2:-}"

    if [ "$PULL_MODE" = "light" ] || [ "$USE_RECENT" -eq 1 ]; then
        pull_via_file_list "$remote_root" "$rel_subpath"
    else
        pull_full_tree "$remote_root" "$rel_subpath"
    fi
}

mkdir -p "$LOCAL_PATH"

SUBPATHS=()
if [ $# -gt 0 ]; then
    while [ $# -gt 0 ]; do
        rel="$(normalize_subpath "$1")"
        if [ "$rel" != "archive" ]; then
            SUBPATHS+=("$rel")
        fi
        shift
    done
fi

echo "Pulling from ${REMOTE_HOST} (${PULL_MODE} mode, dest=${LOCAL_PATH})..."
[ "$DRY_RUN" -eq 1 ] && echo "(dry-run: no files written)"
[ "$USE_RECENT" -eq 1 ] && echo "Only files modified in the last ${RECENT_HOURS} hours"

for RP in "${REMOTE_PATHS[@]}"; do
    echo "Remote: ${RP}"
    if [ "${#SUBPATHS[@]}" -eq 0 ]; then
        pull_tree "$RP" ""
    else
        for rel in "${SUBPATHS[@]}"; do
            pull_tree "$RP" "$rel"
        done
    fi
done

echo "Done."
