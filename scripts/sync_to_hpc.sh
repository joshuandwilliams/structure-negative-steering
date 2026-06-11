#!/bin/bash
#
# Sync the local repo to the HPC via SSH (uses the ~/.ssh/config alias 'slurm').
#
# Usage:
#   ./scripts/sync_to_hpc.sh           # real sync
#   ./scripts/sync_to_hpc.sh --dry     # dry run, no changes
#
# Direction & authority:
#   The Mac repo is authoritative for CODE.  Develop here, sync up, run the
#   engine on the HPC.  The HPC has no git.
#
# CRITICAL — experiments/ is protected:
#   The HPC holds the real experiments/ tree (prior runs migrated from the old
#   receptor_design/negative_steering scratch folder, plus new runs).  Those
#   are HPC-only and gitignored.  This push EXCLUDES experiments/ entirely, so
#   --delete can never wipe your results on the cluster.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HPC_USER_HOST="slurm"
HPC_DEST="receptor_design/structure-negative-steering/"

DRY_RUN=""
if [ "${1:-}" = "--dry" ] || [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN="--dry-run"
    echo "=== DRY RUN -- no files will be changed ==="
fi

cd "$REPO_ROOT"

rsync -av --delete $DRY_RUN -e ssh \
    --exclude='.git/' \
    --exclude='experiments/' \
    --exclude='__pycache__/' \
    --exclude='.pytest_cache/' \
    --exclude='.ruff_cache/' \
    --exclude='.mypy_cache/' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='~$*' \
    --exclude='*.img' \
    --exclude='*.sif' \
    --exclude='work/' \
    --exclude='.nextflow*' \
    --exclude='*_results/' \
    --exclude='*.out' \
    --exclude='*.err' \
    --exclude='rsync_dryrun*.txt' \
    --exclude='rsync_deletions*.txt' \
    "$REPO_ROOT/" "${HPC_USER_HOST}:${HPC_DEST}"

echo
echo "Sync complete -> ${HPC_USER_HOST}:${HPC_DEST}"
echo "(experiments/ deliberately NOT pushed — it is HPC-authoritative.)"
