#!/bin/bash
# Wrapper to pull Slurm run artifacts from Narval instead of Killarney.
# Passes all arguments directly to pull_results.sh

export REMOTE_HOST="narval.alliancecan.ca"
exec "$(dirname "$0")/pull_results.sh" "$@"
