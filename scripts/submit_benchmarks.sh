#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# submit_benchmarks.sh - submit a negative-steering job per benchmark target.
#
# A wrapper around scripts/negative_steering.slurm.sh. It submits one SLURM job
# for every experiments/benchmarking/<variant>/<target>/config.yml, directing
# every output INSIDE that target's own folder so nothing floats in the repo
# root or in scripts/:
#
#   experiments/benchmarking/<variant>/<target>/
#     ├── config.yml, <target>.pdb      (tracked test spec)
#     ├── negsteer_<jobid>.out/.err     (SLURM logs, via --output/--error)
#     ├── inputs/                       (derived by prepare)
#     └── run/                          (engine workdir + launch.sh)
#
# <variant> is `unconstrained` or `constrained`. The constrained variant is the
# same config with negsteer_boltz_constraints: true, so every Boltz-2 call in the
# run also carries pocket + contact restraints derived from the reference
# complex. Jobs are named negsteer_<target> and negsteer_<target>_constrained,
# so the two never collide in the scheduler or in Slurm accounting.
#
# By default a target whose run/ directory already exists is SKIPPED, so
# re-running only picks up outstanding work. Pass --force to resubmit.
#
# Run this on the HPC login node. It only calls sbatch, and needs no GPU itself.
#
# Usage:
#   ./scripts/submit_benchmarks.sh                          # everything outstanding
#   ./scripts/submit_benchmarks.sh --dry-run                # print the sbatch commands only
#   ./scripts/submit_benchmarks.sh --variant unconstrained  # one variant
#   ./scripts/submit_benchmarks.sh --variant constrained
#   ./scripts/submit_benchmarks.sh --only 6G10              # one target, both variants
#   ./scripts/submit_benchmarks.sh --only constrained/6G10  # one target, one variant
#   ./scripts/submit_benchmarks.sh --force                  # include targets that already have run/
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="${REPO_ROOT}/experiments/benchmarking"
SLURM_SCRIPT="${REPO_ROOT}/scripts/negative_steering.slurm.sh"

DRY=""
FORCE=""
VARIANT=""
ONLY=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|--dry) DRY="1"; shift ;;
        --force)         FORCE="1"; shift ;;
        --variant)       VARIANT="$2"; shift 2 ;;
        --only)          ONLY+=("$2"); shift 2 ;;
        -h|--help)       sed -n '2,36p' "$0"; exit 0 ;;
        *)               echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -n "$VARIANT" && "$VARIANT" != "unconstrained" && "$VARIANT" != "constrained" ]]; then
    echo "ERROR: --variant must be unconstrained or constrained (got: $VARIANT)" >&2
    exit 2
fi

[[ -f "$SLURM_SCRIPT" ]] || { echo "ERROR: missing $SLURM_SCRIPT" >&2; exit 2; }
if [[ -z "$DRY" ]] && ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch not found. Run this on the HPC, or use --dry-run." >&2
    exit 2
fi

_wanted() {  # <variant> <target> -> 0 if it should be submitted
    [[ ${#ONLY[@]} -eq 0 ]] && return 0
    local o
    for o in "${ONLY[@]}"; do
        [[ "$o" == "$2" || "$o" == "$1/$2" ]] && return 0
    done
    return 1
}

shopt -s nullglob
configs=("$BENCH_DIR"/${VARIANT:-*}/*/config.yml)
if [[ ${#configs[@]} -eq 0 ]]; then
    echo "No targets found under $BENCH_DIR/${VARIANT:-*}/*/config.yml" >&2
    exit 1
fi

submitted=0
skipped=()
for cfg in "${configs[@]}"; do
    dir="$(dirname "$cfg")"
    target="$(basename "$dir")"
    variant="$(basename "$(dirname "$dir")")"
    _wanted "$variant" "$target" || continue

    if [[ -z "$FORCE" && -d "${dir}/run" ]]; then
        skipped+=("${variant}/${target}")
        continue
    fi

    # Job name keeps the pre-split convention, so Slurm accounting stays
    # comparable with the runs already on the cluster.
    name="$target"
    [[ "$variant" == "constrained" ]] && name="${target}_constrained"

    sbatch_cmd=(sbatch --parsable
        --job-name="negsteer_${name}"
        --output="${dir}/negsteer_%j.out"
        --error="${dir}/negsteer_%j.err"
        "$SLURM_SCRIPT" "$cfg")

    if [[ -n "$DRY" ]]; then
        printf 'sbatch'; printf ' %q' "${sbatch_cmd[@]:1}"; printf '\n'
    else
        jobid="$("${sbatch_cmd[@]}")"
        echo "submitted ${variant}/${target}: job ${jobid}  (outputs -> ${dir}/)"
    fi
    submitted=$((submitted + 1))
done

if [[ ${#skipped[@]} -gt 0 ]]; then
    echo "already have run/, skipping (use --force to resubmit): ${skipped[*]}"
fi
echo "--- ${submitted} target(s) ${DRY:+would be }submitted ---"
