#!/bin/bash
#SBATCH --job-name="negsteer"
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=negsteer_%j.out
#SBATCH --error=negsteer_%j.err
#SBATCH --mail-type=END,FAIL
#
# The partition is not in this header. It is the one resource whose name
# differs per site, so it comes from site.conf through scripts/submit.sh, or
# from an sbatch -p on the command line.
# ─────────────────────────────────────────────────────────────────────
# negative_steering.slurm.sh - run negative steering on one config.
#
# The host half of the runner. It resolves the container, runs the Python half
# (bin/negative_steering.py) inside it, executes the generated launch.sh,
# then assigns the run's tier.
#
# The header mirrors the pipeline's NEGSTEER_RUN_ONE process, 4 CPUs, 12 GB,
# --gres=gpu:1, but with --time raised to 48 h. The pipeline's 6 h is sized
# for small RFDiffusion design fragments. The benchmark runs full experimental
# complexes up to ~1400 residues under a heavier n_designs x num_seeds x
# diffusion config, which can exceed 12 h. A short --time risks killing a long
# structure for nothing, so raise it rather than trim it unless your partition
# caps walltime.
#
# The GPU allocation is REQUIRED. The engine's `singularity exec --nv` needs the
# host GPU driver libraries, or Boltz fails with "No supported gpu backend found!".
#
# NOTHING runs outside a container. This script uses only base-OS tools (bash,
# grep, sed, realpath, singularity). All Python runs via `singularity exec`.
# The engine run is launched via launch.sh, whose orchestrator does its OWN
# `singularity exec --nv`, which is why it cannot run from inside a container.
#
# Usage:
#   ./scripts/submit.sh steer tests/test_run/config.yml
#
#   # CPU-only steps need no GPU node. Run them interactively with bash,
#   # since the #SBATCH lines are just comments outside sbatch:
#   bash scripts/negative_steering.slurm.sh <config> --prepare-only
#   bash scripts/negative_steering.slurm.sh <config> --dry-run
#   bash scripts/negative_steering.slurm.sh <config> --container /path/to.img
#
# The container defaults to the config's `boltz_container:` value.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

# REPO_ROOT is resolved AFTER the config is known (see below). Under `sbatch`
# SLURM spools this script to /var/spool/slurmd/..., so ${BASH_SOURCE[0]} is NOT
# the repo path. NEGSTEER_REPO_DIR names it when the config lives in another
# repository. Otherwise we walk up from the absolute config path to find it.

CONFIG=""
CONTAINER=""
PREPARE_ONLY=""
DRY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --container)     CONTAINER="$2"; shift 2 ;;
        --prepare-only)  PREPARE_ONLY="1"; shift ;;
        --dry-run|--dry) DRY="1"; shift ;;
        -h|--help)       sed -n '14,45p' "$0"; exit 0 ;;
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

# NEGSTEER_REPO_DIR first, then walk up from the config dir to the repo root
# (the directory containing bin/negative_steering.py), then fall back to
# $SLURM_SUBMIT_DIR and finally to this script's own location.
REPO_ROOT=""
if [[ -n "${NEGSTEER_REPO_DIR:-}" && -f "${NEGSTEER_REPO_DIR}/bin/negative_steering.py" ]]; then
    REPO_ROOT="$(realpath "$NEGSTEER_REPO_DIR")"
fi
_d="$CFG_DIR"
while [[ -z "$REPO_ROOT" && "$_d" != "/" ]]; do
    if [[ -f "$_d/bin/negative_steering.py" ]]; then REPO_ROOT="$_d"; break; fi
    _d="$(dirname "$_d")"
done
if [[ -z "$REPO_ROOT" && -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/bin/negative_steering.py" ]]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
fi
if [[ -z "$REPO_ROOT" ]]; then
    _bs="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd || true)"
    [[ -n "$_bs" && -f "$_bs/bin/negative_steering.py" ]] && REPO_ROOT="$_bs"
fi
if [[ -z "$REPO_ROOT" ]]; then
    echo "ERROR: could not locate the repo root (bin/negative_steering.py) from $CONFIG" >&2
    exit 2
fi
PY_DRIVER="${REPO_ROOT}/bin/negative_steering.py"
ENGINE_BIN="${REPO_ROOT}/bin"

# Resolve the container through bin/resolve_boltz_container.sh, which is the
# same resolver the workflow's READ_CONFIG stage uses: --container override,
# then the config's boltz_container, then NEGSTEER_BOLTZ_IMG.
#
# This used to extract the YAML line here with grep and sed, and that was a
# real bug rather than a duplication. Every committed config carries
# `boltz_container: null`, which the extraction returned as the literal string
# "null", so the check below reported "container image not found: null" and the
# site setting was never consulted. It broke the moment the configs were
# emptied in 11b9f01, and only on this path, because the workflow already went
# through the resolver.
#
# NEGSTEER_BOLTZ_IMG reaches here by inheritance: scripts/submit.sh sources
# scripts/site_conf.sh and sbatch exports the submitting environment. A plain
# `bash scripts/negative_steering.slurm.sh <config>` with no site.conf loaded
# therefore needs --container, and says so below.
RESOLVER="${REPO_ROOT}/bin/resolve_boltz_container.sh"
if [[ ! -f "$RESOLVER" ]]; then
    echo "ERROR: missing $RESOLVER" >&2
    exit 2
fi
CONTAINER="$(bash "$RESOLVER" "$CONFIG" "$CONTAINER")"
if [[ -z "$CONTAINER" ]]; then
    echo "ERROR: no Boltz image. The config leaves boltz_container null, so" >&2
    echo "       set NEGSTEER_BOLTZ_IMG in site.conf, or pass --container." >&2
    exit 2
fi
if [[ ! -f "$CONTAINER" ]]; then
    echo "ERROR: container image not found: $CONTAINER" >&2
    exit 2
fi

# All Python runs in-container on CPU. The engine run adds --nv itself.
#
# The driver reads the config and its reference_pdb, and writes inputs/ beside
# the config, so the config's directory is bound as well as the engine. A
# reference_pdb outside both, as in a project whose structures sit in their own
# data/ tree, needs its directory in NEGSTEER_BIND_PATHS, comma-separated.
# launch.sh needs none of this: negative_steering_run_one.sh binds every
# directory it is given.
BINDS=(--bind "$REPO_ROOT" --bind "$CFG_DIR")
if [[ -n "${NEGSTEER_BIND_PATHS:-}" ]]; then
    IFS=',' read -r -a _extra <<< "$NEGSTEER_BIND_PATHS"
    for _b in "${_extra[@]}"; do
        [[ -n "$_b" ]] && BINDS+=(--bind "$_b")
    done
fi
SING=(singularity exec "${BINDS[@]}" "$CONTAINER" python "$PY_DRIVER" "$CONFIG")

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
bash "$LAUNCH"

# ── Tiering: pick the representative steered design and assign cross_tier ────
# Mirrors the pipeline's NEGSTEER_CROSS_SUMMARY stage. cross_summary.py reads
# the run's passing_summary.csv, picks ONE representative and writes
# cross_sequence_summary.csv with cross_tier:
#   A: n_pass == n_seeds   B: 1 < n_pass < n_seeds   C: n_pass == 1   none: 0
# Runs in-container on CPU. NON-FATAL, because the engine results are already
# written and a tiering hiccup must not fail the run.
WORKDIR="${CFG_DIR}/run"
NAME="$(grep -E '^[[:space:]]*name:' "$CONFIG" | head -1 \
    | sed -E 's/^[^:]*:[[:space:]]*//; s/^["'"'"']//; s/["'"'"'][[:space:]]*$//')"
NAME="${NAME:-$(basename "$CFG_DIR")}"
PASSING="${WORKDIR}/passing_summary.csv"
XSUMM="${WORKDIR}/cross_sequence_summary.csv"

if [[ -f "$PASSING" ]]; then
    echo
    echo "=== tiering: representative + cross_tier -> cross_sequence_summary.csv ==="
    if singularity exec --bind "$WORKDIR" --bind "$ENGINE_BIN" "$CONTAINER" \
            python "${ENGINE_BIN}/cross_summary.py" \
                --passing-summary "${NAME}=${PASSING}" \
                --output "$XSUMM"; then
        echo "wrote $XSUMM"
    else
        echo "WARNING: tiering step failed. Engine results preserved in $WORKDIR." >&2
    fi
else
    echo "WARNING: no passing_summary.csv in $WORKDIR, skipping tiering." >&2
fi
