#!/bin/bash
# Pull Slurm artifacts from Narval in one SSH session (one Duo prompt).
# Passes extra args to pull_results.sh (--all, --recent 6, subpaths, etc.).

export REMOTE_HOST="narval.alliancecan.ca"
export REMOTE_REPO_ROOTS="/scratch/ccao87/ts-sandbox"
exec "$(dirname "$0")/pull_results.sh" "$@"
