#!/bin/bash
#
# Sync the local repo to the HPC via SSH (uses the ~/.ssh/config alias 'slurm').
#
# Usage:
#   ./scripts/sync_to_hpc.sh              # real sync
#   ./scripts/sync_to_hpc.sh --dry        # dry run, no changes
#   ./scripts/sync_to_hpc.sh --list-stale # report cluster-only paths, change nothing
#
# Why --list-stale exists:
#   rsync --delete does NOT delete excluded paths. It protects them, which is
#   the behaviour that stops a code sync destroying results. The cost is that
#   anything matching an exclude survives on the cluster forever, so a renamed
#   or dropped directory is never cleaned up and the cluster silently
#   accumulates two of everything. --list-stale reports what is up there and
#   not down here so pruning is a deliberate, visible act.
#
#   It only ever reads. It prints commands; it does not run them, and this
#   script never deletes anything on the cluster. --delete-excluded would
#   "fix" the accumulation by wiping every result. Never use it.
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
#   - tests/single_run_test/inputs/ and tests/single_run_test/run/  The single-run test is a real engine
#     output produced on a GPU node, and the only thing that reaches the plan,
#     collect, cycle and reversion-harvest code. Only tests/single_run_test/config.yml is
#     tracked; the derived inputs and the run tree are HPC-authoritative for
#     exactly the same reason experiments/ is, and a code sync must not wipe
#     them. This was found by reading a --dry run rather than by thinking about
#     it, which is the argument for always reading one.
#   - tests/workflow_check/inputs/ and tests/workflow_check/run/  Same rule, same
#     reason. The 6G10 workflow check is a real engine output produced on the
#     cluster and only tests/workflow_check/config.yml is tracked.
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
case "${1:-}" in
    --dry|--dry-run)
        DRY_RUN="--dry-run"
        echo "=== DRY RUN -- no files will be changed ==="
        ;;
    --list-stale)
        cd "$REPO_ROOT"
        echo "=== cluster paths with no local counterpart ==="
        echo "Reads only. Nothing here is deleted, by this script or otherwise."
        echo

        # A benchmark target is a directory holding a config.yml. Comparing on
        # that rather than on directory names keeps derived inputs/ and run/
        # trees out of the report.
        find_targets="find . -name config.yml -maxdepth 3 \
                      | sed -e 's|^\./||' -e 's|/config.yml\$||' | sort"

        local_targets=$(cd experiments/benchmarking 2>/dev/null \
            && eval "$find_targets")
        remote_targets=$(ssh "$HPC_USER_HOST" \
            "cd ${HPC_DEST}experiments/benchmarking 2>/dev/null && $find_targets" 2>/dev/null)

        echo "--- benchmarking targets on the cluster only ---"
        stale=$(comm -23 <(echo "$remote_targets") <(echo "$local_targets"))
        [ -n "$stale" ] && echo "$stale" | sed 's/^/  /' || echo "  (none)"

        echo
        echo "--- repo-root entries on the cluster only ---"
        local_root=$(ls -A | sort)
        remote_root=$(ssh "$HPC_USER_HOST" "cd ${HPC_DEST} && ls -A" 2>/dev/null | sort)
        comm -23 <(echo "$remote_root") <(echo "$local_root") | sed 's/^/  /' || true

        echo
        echo "Nothing above has been touched. To retire one, rename rather than"
        echo "delete, so the name is freed but the data survives:"
        echo "  ssh $HPC_USER_HOST 'cd ${HPC_DEST} && mv <path> <path>_superseded_\$(date +%Y%m%d)'"
        exit 0
        ;;
esac

cd "$REPO_ROOT"

rsync -av --delete $DRY_RUN -e ssh \
    --exclude='.git/' \
    --exclude='experiments/' \
    --exclude='analysis/' \
    --exclude='tests/single_run_test/inputs/' \
    --exclude='tests/single_run_test/run/' \
    --exclude='tests/single_run_test/run_cli/' \
    --exclude='tests/single_run_test/.cli_view*' \
    --exclude='tests/workflow_check/inputs/' \
    --exclude='tests/workflow_check/run/' \
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
# The dose-sweep configs ride along the same way. They carry no PDB of their own,
# each one points back at the benchmark's copy.
if [ -d "$REPO_ROOT/experiments/benchmarking" ]; then
    rsync -av --no-perms --no-owner --no-group $DRY_RUN -e ssh \
        --prune-empty-dirs \
        --include='benchmarking/' \
        --include='benchmarking/**/' \
        --include='benchmarking/**/config.yml' \
        --include='benchmarking/**/*.pdb' \
        --include='dose-sweep/' \
        --include='dose-sweep/**/' \
        --include='dose-sweep/**/config.yml' \
        --exclude='*' \
        "$REPO_ROOT/experiments/" "${HPC_USER_HOST}:${HPC_DEST}experiments/"
fi

echo
echo "Sync complete -> ${HPC_USER_HOST}:${HPC_DEST}"
echo "(code pushed with --delete; experiments/ results untouched —"
echo " only benchmarking/ test definitions are pushed, additively.)"
