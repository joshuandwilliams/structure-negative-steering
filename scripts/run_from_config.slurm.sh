#!/bin/bash
#SBATCH --job-name="negsteer"
#SBATCH -p jic-gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=negsteer_%j.out
#SBATCH --error=negsteer_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jowillia@nbi.ac.uk
# ─────────────────────────────────────────────────────────────────────
# run_from_config.slurm.sh — config-driven negative-steering run.
#
# The header mirrors the pipeline's NEGSTEER_RUN_ONE process (nextflow.config):
# queue jic-gpu, 4 CPUs, 12 GB, --gres=gpu:1 — BUT with --time raised to 48 h.
# The pipeline's 6 h is sized for small RFDiffusion design fragments; the
# standalone benchmark runs full experimental complexes (up to ~1400 residues)
# under a heavier n_designs×num_seeds×diffusion config, which can take >12 h.
# jic-gpu has no walltime cap (TIMELIMIT=infinite), so don't let a short --time
# kill a long structure. Raise it further for anything larger.
# The GPU allocation is REQUIRED: the engine's `singularity exec --nv` needs the
# host GPU driver libraries, or Boltz fails with "No supported gpu backend found!".
#
# NOTHING runs outside a container.  This host script uses only base-OS
# tools (bash, grep, sed, realpath, singularity).  All Python — config
# parsing and input preparation — runs via `singularity exec <img>
# python ...`.  The engine run is launched via the generated launch.sh,
# whose orchestrator does its OWN `singularity exec --nv`.
#
# Usage:
#   sbatch scripts/run_from_config.slurm.sh experiments/benchmarking/6G10/config.yml
#
#   # CPU-only steps need no GPU node — run interactively with bash
#   # (the #SBATCH lines are just comments outside sbatch):
#   bash scripts/run_from_config.slurm.sh <config> --prepare-only
#   bash scripts/run_from_config.slurm.sh <config> --dry-run
#   bash scripts/run_from_config.slurm.sh <config> --container /path/to.img
#
# The container defaults to the config's `boltz_container:` value.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

# REPO_ROOT is resolved AFTER the config is known (see below): under `sbatch`
# SLURM spools this script to /var/spool/slurmd/..., so ${BASH_SOURCE[0]} is NOT
# the repo path. We walk up from the absolute config path to find scripts/ instead.

CONFIG=""
CONTAINER=""
PREPARE_ONLY=""
DRY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --container)     CONTAINER="$2"; shift 2 ;;
        --prepare-only)  PREPARE_ONLY="1"; shift ;;
        --dry-run|--dry) DRY="1"; shift ;;
        -h|--help)       sed -n '14,40p' "$0"; exit 0 ;;
        -*)              echo "unknown arg: $1" >&2; exit 2 ;;
        *)               CONFIG="$1"; shift ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "usage: sbatch $(basename "$0") CONFIG [--container IMG] [--prepare-only] [--dry-run]" >&2
    exit 2
fi
CONFIG="$(realpath "$CONFIG")"
[[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 2; }
CFG_DIR="$(dirname "$CONFIG")"

# Resolve REPO_ROOT robustly (sbatch spools this script, so BASH_SOURCE is
# unreliable). Walk up from the config dir to the repo (contains scripts/), then
# fall back to $SLURM_SUBMIT_DIR and finally the script's own dir.
REPO_ROOT=""
_d="$CFG_DIR"
while [[ "$_d" != "/" ]]; do
    if [[ -f "$_d/scripts/run_from_config.py" ]]; then REPO_ROOT="$_d"; break; fi
    _d="$(dirname "$_d")"
done
if [[ -z "$REPO_ROOT" && -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/run_from_config.py" ]]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
fi
if [[ -z "$REPO_ROOT" ]]; then
    _bs="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd || true)"
    [[ -n "$_bs" && -f "$_bs/scripts/run_from_config.py" ]] && REPO_ROOT="$_bs"
fi
if [[ -z "$REPO_ROOT" ]]; then
    echo "ERROR: could not locate the repo root (scripts/run_from_config.py) from $CONFIG" >&2
    exit 2
fi
PY_DRIVER="${REPO_ROOT}/scripts/run_from_config.py"

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
echo "=== launch engine (orchestrator runs its own singularity --nv) ==="
echo "bash $LAUNCH"
exec bash "$LAUNCH"
