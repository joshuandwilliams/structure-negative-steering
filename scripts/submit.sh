#!/bin/bash
#
# Submit a launcher with this site's partition and accounting.
#
#   ./scripts/submit.sh workflow  tests/test_run/config.yml
#   ./scripts/submit.sh workflow  'experiments/benchmarking/**/config.yml'
#   ./scripts/submit.sh sweep                       # all five doses
#   ./scripts/submit.sh sweep     n10
#   ./scripts/submit.sh ablation
#   ./scripts/submit.sh steer     experiments/benchmarking/unconstrained/6G10/config.yml
#   ./scripts/submit.sh test-run
#
# Each launcher carries the resources it needs in its own #SBATCH header, and
# none of those are site-specific. What IS site-specific is the partition
# name, and sometimes an account or a QOS. Those live in site.conf, and this
# passes them on the command line, where they beat both the environment and
# the header (sbatch(1), INPUT ENVIRONMENT VARIABLES).
#
# So `sbatch scripts/run_workflow.slurm.sh <targets>` still works with no
# site.conf at all. The job then lands on the cluster's default partition,
# which is right for the head jobs and usually wrong for the GPU ones.
#
# --dry-run prints the sbatch command and submits nothing.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"

# shellcheck source=scripts/site_conf.sh
. "${HERE}/site_conf.sh"

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
    shift
fi

ROLE="${1:-}"
[ "$#" -gt 0 ] && shift

# role -> launcher, and which partition it wants. The GPU roles are the ones
# that break on a default partition, because a node with no card fails the
# whole job rather than queueing for one.
case "${ROLE}" in
    workflow)
        SCRIPT="${REPO}/scripts/run_workflow.slurm.sh"
        PARTITION="${NEGSTEER_PARTITION_HEAD:-}"
        GPU=0
        ;;
    sweep)
        SCRIPT="${REPO}/scripts/run_dose_sweep.slurm.sh"
        PARTITION="${NEGSTEER_PARTITION_HEAD_LONG:-${NEGSTEER_PARTITION_HEAD:-}}"
        GPU=0
        ;;
    ablation)
        SCRIPT="${REPO}/scripts/run_ablation_controls.slurm.sh"
        PARTITION="${NEGSTEER_PARTITION_HEAD:-}"
        GPU=0
        ;;
    steer)
        SCRIPT="${REPO}/scripts/negative_steering.slurm.sh"
        PARTITION="${NEGSTEER_PARTITION_GPU:-}"
        GPU=1
        ;;
    test-run)
        SCRIPT="${REPO}/tests/test_run.slurm.sh"
        PARTITION="${NEGSTEER_PARTITION_GPU:-}"
        GPU=1
        ;;
    *)
        echo "Usage: ./scripts/submit.sh [--dry-run] <role> [args]" >&2
        echo "" >&2
        echo "  workflow <targets>   the Nextflow head job" >&2
        echo "  sweep    [dose]      the dose sweep, one head job for all of it" >&2
        echo "  ablation [variant]   the interface-ablation controls" >&2
        echo "  steer    <config>    one target through the engine, no Nextflow" >&2
        echo "  test-run             produce the run the check_test_run tier asserts on" >&2
        echo "" >&2
        echo "Partitions and accounting come from site.conf." >&2
        echo "Copy site.conf.example to site.conf to set them." >&2
        exit 1
        ;;
esac

if [ ! -f "${SCRIPT}" ]; then
    echo "ERROR: no launcher at ${SCRIPT}" >&2
    exit 1
fi

CMD=(sbatch)
[ -n "${PARTITION}" ]                     && CMD+=(-p "${PARTITION}")
[ -n "${NEGSTEER_SLURM_ACCOUNT:-}" ]      && CMD+=(-A "${NEGSTEER_SLURM_ACCOUNT}")
[ -n "${NEGSTEER_SLURM_QOS:-}" ]          && CMD+=(--qos "${NEGSTEER_SLURM_QOS}")
if [ "${GPU}" -eq 1 ] && [ -n "${NEGSTEER_GPU_CLUSTER_OPTIONS:-}" ]; then
    # Word-split on purpose. A site may need more than one flag here, and
    # quoting the whole string would hand sbatch one unparseable argument.
    # shellcheck disable=SC2206
    CMD+=(${NEGSTEER_GPU_CLUSTER_OPTIONS})
fi
CMD+=("${SCRIPT}" "$@")

if [ "${GPU}" -eq 1 ] && [ -z "${PARTITION}" ]; then
    echo "WARNING: no NEGSTEER_PARTITION_GPU set, so this GPU job goes to the" >&2
    echo "         cluster's default partition. Boltz dies with \"No supported" >&2
    echo "         gpu backend found!\" if that partition has no card." >&2
fi

if [ "${DRY_RUN}" -eq 1 ]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

cd "${REPO}"
exec "${CMD[@]}"
