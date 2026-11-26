#!/bin/bash
#
# Load site.conf, if there is one.
#
#   . scripts/site_conf.sh
#
# site.conf holds everything that is specific to one cluster or one person:
# partition names, container paths, an ssh host. It is gitignored, and
# site.conf.example documents every setting. Nothing here is required, so a
# clone with no site.conf still works, falling back to the cluster's defaults
# and to whatever is on PATH.
#
# Sourced rather than executed, because the caller needs the variables. It
# therefore must not call `exit`, and must not set -e: a failure here would
# kill a shell the caller still owns.
#
# A variable already set in the environment wins over site.conf. That is what
# lets one run override a setting without editing the file:
#
#   NEGSTEER_PARTITION_GPU=gpu-long ./scripts/submit.sh steer <config>

# Resolve the repo from this script's own location, not from the caller's cwd.
_SITE_CONF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEGSTEER_SITE_CONF="${NEGSTEER_SITE_CONF:-${_SITE_CONF_DIR}/site.conf}"

if [ -f "${NEGSTEER_SITE_CONF}" ]; then
    # Snapshot anything already set so the environment keeps precedence.
    for _v in NEGSTEER_REPO_DIR NEGSTEER_PARTITION_GPU NEGSTEER_PARTITION_HEAD \
              NEGSTEER_PARTITION_HEAD_LONG NEGSTEER_PARTITION_CPU \
              NEGSTEER_GPU_CLUSTER_OPTIONS NEGSTEER_SLURM_ACCOUNT \
              NEGSTEER_SLURM_QOS NEGSTEER_MAX_GPU_JOBS NEGSTEER_BOLTZ_IMG \
              NEGSTEER_JAVA_HOME NEGSTEER_NEXTFLOW_IMG NEGSTEER_HPC_HOST \
              NEGSTEER_HPC_DEST; do
        if [ -n "${!_v:-}" ]; then
            eval "_SITE_PRESET_${_v}=\${${_v}}"
        fi
    done

    # shellcheck source=/dev/null
    . "${NEGSTEER_SITE_CONF}"

    for _v in NEGSTEER_REPO_DIR NEGSTEER_PARTITION_GPU NEGSTEER_PARTITION_HEAD \
              NEGSTEER_PARTITION_HEAD_LONG NEGSTEER_PARTITION_CPU \
              NEGSTEER_GPU_CLUSTER_OPTIONS NEGSTEER_SLURM_ACCOUNT \
              NEGSTEER_SLURM_QOS NEGSTEER_MAX_GPU_JOBS NEGSTEER_BOLTZ_IMG \
              NEGSTEER_JAVA_HOME NEGSTEER_NEXTFLOW_IMG NEGSTEER_HPC_HOST \
              NEGSTEER_HPC_DEST; do
        _preset="_SITE_PRESET_${_v}"
        if [ -n "${!_preset:-}" ]; then
            eval "${_v}=\${${_preset}}"
        fi
        unset "${_preset}"
    done
    unset _v _preset
fi

# Defaults for the two settings that have a sensible one. A partition has no
# sensible default, because every site names its own, so an unset partition
# means "the cluster's default" and the job goes wherever that is.
NEGSTEER_GPU_CLUSTER_OPTIONS="${NEGSTEER_GPU_CLUSTER_OPTIONS:---gres=gpu:1}"
NEGSTEER_MAX_GPU_JOBS="${NEGSTEER_MAX_GPU_JOBS:-8}"

export NEGSTEER_SITE_CONF NEGSTEER_GPU_CLUSTER_OPTIONS NEGSTEER_MAX_GPU_JOBS
export NEGSTEER_REPO_DIR NEGSTEER_PARTITION_GPU NEGSTEER_PARTITION_HEAD
export NEGSTEER_PARTITION_HEAD_LONG NEGSTEER_PARTITION_CPU
export NEGSTEER_SLURM_ACCOUNT NEGSTEER_SLURM_QOS NEGSTEER_BOLTZ_IMG
export NEGSTEER_JAVA_HOME NEGSTEER_NEXTFLOW_IMG
export NEGSTEER_HPC_HOST NEGSTEER_HPC_DEST
unset _SITE_CONF_DIR
