#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# benchmark_submission.sh — submit a negative-steering job per benchmark target.
#
# Submits one SLURM job (scripts/run_from_config.slurm.sh) for every
# experiments/benchmarking/<name>/config.yml.  Every output for a target lands
# INSIDE that target's folder — nothing floats in the repo root or scripts/:
#
#   experiments/benchmarking/<name>/
#     ├── config.yml, <name>.pdb        (tracked test spec)
#     ├── negsteer_<jobid>.out/.err     (SLURM logs — directed here via --output/--error)
#     ├── inputs/                       (derived by prepare)
#     └── run/                          (engine workdir + launch.sh)
#
# Run this on the HPC login node (it only calls sbatch — no GPU itself).
#
# Usage:
#   ./scripts/benchmark_submission.sh                # submit all targets
#   ./scripts/benchmark_submission.sh --dry-run      # print the sbatch commands only
#   ./scripts/benchmark_submission.sh --only 6G10    # one target (repeatable)
#   ./scripts/benchmark_submission.sh --only 6G10 --only 7B1I
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="${REPO_ROOT}/experiments/benchmarking"
SLURM_SCRIPT="${REPO_ROOT}/scripts/run_from_config.slurm.sh"

DRY=""
ONLY=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|--dry) DRY="1"; shift ;;
        --only)          ONLY+=("$2"); shift 2 ;;
        -h|--help)       sed -n '2,24p' "$0"; exit 0 ;;
        *)               echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

[[ -f "$SLURM_SCRIPT" ]] || { echo "ERROR: missing $SLURM_SCRIPT" >&2; exit 2; }
if [[ -z "$DRY" ]] && ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch not found — run this on the HPC (or use --dry-run)." >&2
    exit 2
fi

_wanted() {  # name -> 0 if it should be submitted
    [[ ${#ONLY[@]} -eq 0 ]] && return 0
    local o
    for o in "${ONLY[@]}"; do [[ "$o" == "$1" ]] && return 0; done
    return 1
}

shopt -s nullglob
configs=("$BENCH_DIR"/*/config.yml)
if [[ ${#configs[@]} -eq 0 ]]; then
    echo "No targets found under $BENCH_DIR/*/config.yml" >&2
    exit 1
fi

submitted=0
for cfg in "${configs[@]}"; do
    dir="$(dirname "$cfg")"
    name="$(basename "$dir")"
    _wanted "$name" || continue

    sbatch_cmd=(sbatch --parsable
        --job-name="negsteer_${name}"
        --output="${dir}/negsteer_%j.out"
        --error="${dir}/negsteer_%j.err"
        "$SLURM_SCRIPT" "$cfg")

    if [[ -n "$DRY" ]]; then
        printf 'sbatch'; printf ' %q' "${sbatch_cmd[@]:1}"; printf '\n'
    else
        jobid="$("${sbatch_cmd[@]}")"
        echo "submitted ${name}: job ${jobid}  (outputs -> ${dir}/)"
    fi
    submitted=$((submitted + 1))
done

echo "--- ${submitted} target(s) ${DRY:+would be }submitted ---"
