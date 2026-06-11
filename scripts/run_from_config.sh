#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# run_from_config.sh — airgapped-HPC entry point for a config-driven run.
#
# NOTHING runs outside a container.  This host script uses only base-OS tools
# (bash, grep, sed, realpath, singularity).  All Python — config parsing and
# input preparation — runs via `singularity exec <img> python ...`.  The GPU
# engine run is launched on the host via the generated launch.sh, whose
# orchestrator does its OWN `singularity exec --nv`.
#
# Usage:
#   ./scripts/run_from_config.sh experiments/benchmarking/6G10/config.yml
#   ./scripts/run_from_config.sh <config> --prepare-only     # stop after prepare
#   ./scripts/run_from_config.sh <config> --dry-run          # print commands only
#   ./scripts/run_from_config.sh <config> --container /path/to.img   # override
#
# The container defaults to the config's `boltz_container:` value.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_DRIVER="${REPO_ROOT}/scripts/run_from_config.py"

CONFIG=""
CONTAINER=""
PREPARE_ONLY=""
DRY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --container)    CONTAINER="$2"; shift 2 ;;
        --prepare-only) PREPARE_ONLY="1"; shift ;;
        --dry-run|--dry) DRY="1"; shift ;;
        -h|--help)      sed -n '2,20p' "$0"; exit 0 ;;
        -*)             echo "unknown arg: $1" >&2; exit 2 ;;
        *)              CONFIG="$1"; shift ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "usage: $(basename "$0") CONFIG [--container IMG] [--prepare-only] [--dry-run]" >&2
    exit 2
fi
CONFIG="$(realpath "$CONFIG")"
[[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 2; }
CFG_DIR="$(dirname "$CONFIG")"

# Resolve the container: --container override, else the boltz_container: line in
# the YAML (plain text extraction — no parser, no install).
if [[ -z "$CONTAINER" ]]; then
    CONTAINER="$(grep -E '^[[:space:]]*boltz_container:' "$CONFIG" | head -1 \
        | sed -E 's/^[^:]*:[[:space:]]*//; s/^["'"'"']//; s/["'"'"'][[:space:]]*$//')"
fi
if [[ -z "$CONTAINER" ]]; then
    echo "ERROR: no container — pass --container or set boltz_container in $CONFIG" >&2
    exit 2
fi
if [[ ! -f "$CONTAINER" ]]; then
    echo "ERROR: container image not found: $CONTAINER" >&2
    exit 2
fi

# All Python runs in-container (CPU; the engine run adds --nv itself).
SING=(singularity exec --bind "$REPO_ROOT" "$CONTAINER" python "$PY_DRIVER" "$CONFIG")

if [[ -n "$DRY" ]]; then
    exec "${SING[@]}" --dry-run
fi

echo "=== prepare (in container: $(basename "$CONTAINER")) ==="
if [[ -n "$PREPARE_ONLY" ]]; then
    exec "${SING[@]}" --prepare-only --boltz-container "$CONTAINER"
fi
"${SING[@]}" --boltz-container "$CONTAINER"

LAUNCH="${CFG_DIR}/run/launch.sh"
[[ -f "$LAUNCH" ]] || { echo "ERROR: launch script not generated: $LAUNCH" >&2; exit 1; }
echo
echo "=== launch engine on host (orchestrator runs its own singularity --nv) ==="
echo "bash $LAUNCH"
exec bash "$LAUNCH"
