#!/bin/bash
#
# Pull a negative-steering experiment back from the HPC to the Mac for local
# analysis.  Experiments live under experiments/<name>/ on the cluster, are
# gitignored, and are never pushed up by sync_to_hpc.sh.  This script pulls a
# chosen one down (plots, CSVs) without committing the large artefacts.
#
# Usage:
#   ./scripts/sync_from_hpc.sh --name <experiment>        # one experiment
#   ./scripts/sync_from_hpc.sh --name <experiment> --dry  # dry-run
#   ./scripts/sync_from_hpc.sh --list                     # list what's on the HPC
#
# Transport: rsync over SSH using the 'slurm' host alias in ~/.ssh/config.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HPC_USER_HOST="slurm"
HPC_BASE="receptor_design/structure-negative-steering"

DRY_RUN=""
NAME=""
LIST=""

usage() {
    cat <<EOF
Usage: $(basename "$0") (--name NAME | --list) [--dry]

  --name NAME  Pull experiments/NAME/ from the HPC into the local repo.
  --list       List experiments/* present on the HPC.
  --dry        rsync dry-run; show what would change without copying.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry|--dry-run) DRY_RUN="--dry-run"; shift ;;
        --name)          NAME="$2"; shift 2 ;;
        --list)          LIST="1"; shift ;;
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
echo "[${NAME}] rsync ${SRC} -> ${DST}"
rsync -av ${DRY_RUN} -e ssh \
    --exclude='.DS_Store' \
    --exclude='work/' \
    --exclude='.nextflow*' \
    --exclude='tmp/' \
    "${SRC}" "${DST}"

echo "Done."
[ -n "${DRY_RUN}" ] && echo "(dry-run — nothing was actually pulled)"
