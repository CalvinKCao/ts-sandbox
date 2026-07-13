#!/bin/bash
# Pull Slurm artifacts from Narval. Uses the same ~/.ssh/sockets multiplex
# socket as pull_results.sh (add a Host narval block in ~/.ssh/config to share
# with interactive ssh). Passes extra args (--all, --recent 6, subpaths, etc.).

export REMOTE_HOST="narval.alliancecan.ca"
export REMOTE_REPO_ROOTS="/scratch/ccao87/ts-sandbox"
exec "$(dirname "$0")/pull_results.sh" "$@"
