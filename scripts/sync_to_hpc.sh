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
#
# Also excluded:
#   - analysis/  The Quarto report is authored and rendered on the Mac from
#     results pulled down by sync_from_hpc.sh.  Nothing on the HPC reads it, and
#     its rendered .html plus the *_files/ JS/CSS bundle are gitignored build
#     artefacts that do not belong on the cluster.
#   - .claude/  Local editor/agent config, Mac-side only.

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
    --exclude='analysis/' \
    --exclude='.claude/' \
    --exclude='.vscode/' \
    --exclude='.idea/' \
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
    --exclude='jdk-*/' \
    --exclude='nxf_home/' \
    --exclude='*_results/' \
    --exclude='*.out' \
    --exclude='*.err' \
    --exclude='rsync_dryrun*.txt' \
    --exclude='rsync_deletions*.txt' \
    "$REPO_ROOT/" "${HPC_USER_HOST}:${HPC_DEST}"

# Push experiment DEFINITIONS additively (NO --delete): the benchmarking test
# specs (config.yml + reference PDBs) must reach the HPC, but the HPC
# experiments/ tree (prior results + run outputs) is authoritative and must
# never be pruned by a code sync. Derived inputs/ and run/ outputs are excluded.
if [ -d "$REPO_ROOT/experiments/benchmarking" ]; then
    rsync -av --no-perms --no-owner --no-group $DRY_RUN -e ssh \
        --prune-empty-dirs \
        --include='benchmarking/' \
        --include='benchmarking/**/' \
        --include='benchmarking/**/config.yml' \
        --include='benchmarking/**/*.pdb' \
        --exclude='*' \
        "$REPO_ROOT/experiments/" "${HPC_USER_HOST}:${HPC_DEST}experiments/"
fi

echo
echo "Sync complete -> ${HPC_USER_HOST}:${HPC_DEST}"
echo "(code pushed with --delete; experiments/ results untouched —"
echo " only benchmarking/ test definitions are pushed, additively.)"
