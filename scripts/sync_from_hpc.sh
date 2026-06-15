#!/bin/bash
#
# Pull a negative-steering experiment back from the HPC to the Mac for local
# analysis.  Experiments live under experiments/<name>/ on the cluster, are
# gitignored, and are never pushed up by sync_to_hpc.sh.  This script pulls a
# chosen one down (plots, CSVs) without committing the large artefacts.
#
# Usage:
#   ./scripts/sync_from_hpc.sh --name <experiment>                 # full pull
#   ./scripts/sync_from_hpc.sh --name <experiment> --results-only  # summaries only
#   ./scripts/sync_from_hpc.sh --name <experiment> --dry           # dry-run
#   ./scripts/sync_from_hpc.sh --list                              # list HPC experiments
#
# --results-only pulls only the lightweight result/summary artefacts (CSV/JSON/
# TXT/FASTA tables, plan.json, SLURM .out logs) and skips the multi-GB raw model
# outputs (per-prediction npz/pdb/cif/a3m subdirectories). Use it for benchmarking,
# where a full pull would be ~6 GB but the analysis only needs ~1 MB/target.
#
# Transport: rsync over SSH using the 'slurm' host alias in ~/.ssh/config.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HPC_USER_HOST="slurm"
HPC_BASE="receptor_design/structure-negative-steering"

DRY_RUN=""
NAME=""
LIST=""
RESULTS_ONLY=""

usage() {
    cat <<EOF
Usage: $(basename "$0") (--name NAME | --list) [--results-only] [--dry]

  --name NAME      Pull experiments/NAME/ from the HPC into the local repo.
  --list           List experiments/* present on the HPC.
  --results-only   Pull only lightweight result/summary files (CSV/JSON/TXT/
                   FASTA/logs); skip raw model outputs (npz/pdb/cif/a3m). Best
                   for benchmarking, where a full pull is ~6 GB.
  --dry            rsync dry-run; show what would change without copying.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry|--dry-run) DRY_RUN="--dry-run"; shift ;;
        --name)          NAME="$2"; shift 2 ;;
        --list)          LIST="1"; shift ;;
        --results-only)  RESULTS_ONLY="1"; shift ;;
        -h|--help)       usage; exit 0 ;;
        *)               echo "ERROR: unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

cd "${REPO_ROOT}"

if [ -n "${LIST}" ]; then
    echo "Experiments on ${HPC_USER_HOST}:${HPC_BASE}/experiments :"
    ssh -q -o BatchMode=yes "${HPC_USER_HOST}" \
        "ls -1 ${HPC_BASE}/experiments 2>/dev/null" | grep -vE '^README' || echo "  (none)"
    exit 0
fi

if [ -z "${NAME}" ]; then
    echo "ERROR: provide --name NAME (or --list)." >&2
    usage
    exit 2
fi

REL="experiments/${NAME}"
SRC="${HPC_USER_HOST}:${HPC_BASE}/${REL}/"
DST="${REPO_ROOT}/${REL}/"

if ! ssh -q -o BatchMode=yes "${HPC_USER_HOST}" "[ -d ${HPC_BASE}/${REL} ]" 2>/dev/null; then
    echo "ERROR: no remote dir at ${HPC_BASE}/${REL}" >&2
    exit 2
fi

[ -n "${DRY_RUN}" ] && echo "=== DRY RUN — no files will be changed ==="
mkdir -p "${DST}"

# Shared excludes (never want these locally, in either mode).
RSYNC_RULES=(
    --exclude='.DS_Store'
    --exclude='work/'
    --exclude='.nextflow*'
    --exclude='tmp/'
)

if [ -n "${RESULTS_ONLY}" ]; then
    echo "[${NAME}] results-only pull (summaries/logs; skipping raw model outputs)"
    # First match wins: drop the heavy per-prediction subdirs and run/logs/
    # outright, then keep only the small tabular/summary/log files anywhere in
    # the surviving tree, then exclude everything else. --prune-empty-dirs keeps
    # the local layout free of empty skeleton directories.
    RSYNC_RULES+=(
        --prune-empty-dirs
        --exclude='run/logs/'
        --exclude='run/cycle_*/initial*/'
        --exclude='run/cycle_*/steered/'
        --exclude='run/cycle_*/reversions/'
        --exclude='run/cycle_*/contamination_scratch/'
        --include='*/'
        --include='*.csv'
        --include='*.json'
        --include='*.txt'
        --include='*.fasta'
        --include='*.tsv'
        --include='launch.sh'
        --include='negsteer_*.out'
        --include='*.err'
        --exclude='*'
    )
fi

echo "[${NAME}] rsync ${SRC} -> ${DST}"
rsync -av ${DRY_RUN} -e ssh "${RSYNC_RULES[@]}" "${SRC}" "${DST}"

echo "Done."
if [ -n "${DRY_RUN}" ]; then
    echo "(dry-run — nothing was actually pulled)"
fi
