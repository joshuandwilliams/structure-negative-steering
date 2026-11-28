#!/bin/bash
#SBATCH --job-name="nf_negsteer"
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=nf_negsteer_%j.out
#SBATCH --error=nf_negsteer_%j.err
#SBATCH --mail-type=END,FAIL
# ─────────────────────────────────────────────────────────────────────────────
# run_workflow.slurm.sh — submit main.nf as a batch job.
#
#   sbatch scripts/run_workflow.slurm.sh tests/test_run/config.yml
#   sbatch scripts/run_workflow.slurm.sh 'my_targets/**/config.yml'
#
# Two directories matter, and they need not be the same one.
#
#   the engine   this checkout, holding main.nf. NEGSTEER_REPO_DIR, else the
#                directory sbatch was called from.
#   the launch   where Nextflow runs. --targets resolves against it, and
#                work/, nxf_home/ and .nextflow-reports/ are written into it.
#                NEGSTEER_LAUNCH_DIR, else the directory sbatch was called
#                from, else the engine.
#
# Submitting from the engine's root makes them one directory. A project that
# keeps its targets in its own repository submits from that repository with
# NEGSTEER_REPO_DIR pointing at the engine. The containers then bind the
# launch directory as well as the engine, through params.bind_paths.
#
# The Nextflow head process is CPU-only. It parses the DAG, submits each stage
# to SLURM and waits. It is NOT run on the login node: this is a shared cluster,
# and a process that sits for hours polling squeue belongs in a job like any
# other. The same pattern is what receptor-resurfacing-pipeline's
# run_pipeline.slurm.sh does, so submitting from a compute node is known to work
# here.
#
# The header is sized for the head process only, 2 CPUs and 4 GB. Every stage
# that does real work gets its own allocation from nextflow.config's `cpu` and
# `gpu` labels, and executor.queueSize caps this run at 40 tasks in flight.
#
# The partition is not here. It is the one resource whose name differs per
# site, so it comes from site.conf through scripts/submit.sh, or from an
# sbatch -p on the command line. Without either, the job takes the cluster's
# default partition, which is fine for a CPU-only head process.
#
# --time must outlive the whole run, including time the GPU tasks spend queued,
# because the head job being killed at its walltime orphans whatever it has
# already submitted. 24 h is the default. Trim it with sbatch's own flags,
# which override the header, but leave room for the queue:
#
#   sbatch -t 12:00:00 scripts/run_workflow.slurm.sh tests/test_run/config.yml
#
# -resume is always passed, so a head job that hits its walltime can be
# resubmitted with the identical command and will skip what already finished.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: ./scripts/submit.sh workflow <targets> [extra nextflow args]"
    echo ""
    echo "  <targets>  one config.yml, or a quoted glob matching several."
    echo ""
    echo "Examples:"
    echo "  ./scripts/submit.sh workflow tests/test_run/config.yml"
    echo "  ./scripts/submit.sh workflow 'my_targets/**/config.yml'"
    echo ""
    echo "submit.sh adds this site's partition from site.conf. Plain"
    echo "\`sbatch scripts/run_workflow.slurm.sh <targets>\` also works and"
    echo "takes the cluster's default partition."
    exit 1
fi

TARGETS="$1"
shift

# ── Engine and launch directory ──────────────────────────────────────────────
# SLURM copies the batch script to a spool directory, so BASH_SOURCE does not
# resolve to the repo. SLURM_SUBMIT_DIR is where sbatch was called from, which
# is the repo root in the documented usage.
REPO_DIR="${NEGSTEER_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [ ! -f "${REPO_DIR:-}/main.nf" ]; then
    echo "ERROR: no main.nf under '${REPO_DIR:-}'."
    echo "Submit from the repo root, or set NEGSTEER_REPO_DIR in site.conf."
    exit 1
fi
LAUNCH_DIR="${NEGSTEER_LAUNCH_DIR:-${SLURM_SUBMIT_DIR:-${REPO_DIR}}}"
if [ ! -d "${LAUNCH_DIR}" ]; then
    echo "ERROR: launch directory '${LAUNCH_DIR}' does not exist."
    exit 1
fi
cd "${LAUNCH_DIR}"

# site.conf carries this cluster's container and launcher paths. Sourced after
# the cd so a relative NEGSTEER_* path resolves against the launch directory.
# shellcheck source=scripts/site_conf.sh
. "${REPO_DIR}/scripts/site_conf.sh"

# ── Why this one process runs outside a container ────────────────────────────
# Every stage of the pipeline runs inside the Boltz image. The head process
# cannot, because its whole job is to submit and poll SLURM jobs and sbatch
# and squeue are host binaries. Run it inside an image and it has no
# scheduler to talk to.
#
# So this one process needs a host JVM and a host Nextflow launcher. The
# Nextflow VERSION is still not the host's: the launcher is extracted from
# NEGSTEER_NEXTFLOW_IMG below, which pins it in its own %post.

# ── Java ─────────────────────────────────────────────────────────────────────
# NEGSTEER_JAVA_HOME in site.conf, or java already on PATH.
if [ -n "${NEGSTEER_JAVA_HOME:-}" ]; then
    export JAVA_HOME="${NEGSTEER_JAVA_HOME}"
    export PATH="${JAVA_HOME}/bin:${PATH}"
fi
if ! command -v java >/dev/null 2>&1; then
    echo "ERROR: no java on PATH. Set NEGSTEER_JAVA_HOME in site.conf."
    exit 1
fi

# ── Nextflow environment ─────────────────────────────────────────────────────
# Offline and plugins-off because the host cannot reach the internet, and a
# Nextflow that tries to resolve a plugin there hangs rather than failing.
export NXF_OFFLINE=true
export NXF_PLUGINS_DEFAULT=false
export NXF_HOME="${LAUNCH_DIR}/nxf_home"
export NXF_WORK="${LAUNCH_DIR}/work"
export NXF_TEMP="${NXF_WORK}/tmp"

# .nextflow.log's location is fixed before nextflow.config is read, so it needs
# the environment variable rather than a config setting. See
# docs/implementation-notes.md.
export NXF_LOG_FILE="${LAUNCH_DIR}/.nextflow-reports/nextflow.log"

mkdir -p "${NXF_HOME}" "${NXF_WORK}" "${NXF_TEMP}" "${LAUNCH_DIR}/.nextflow-reports"

# ── Nextflow launcher ────────────────────────────────────────────────────────
# Extracted from the container into the launch directory's own NXF_HOME rather
# than borrowed from a sibling repo's, so a run here cannot write into that one.
NEXTFLOW_IMG="${NEGSTEER_NEXTFLOW_IMG:-}"
NEXTFLOW_BIN="${NXF_HOME}/nextflow"

# No image configured and a nextflow already on PATH is a complete answer.
if [ ! -x "${NEXTFLOW_BIN}" ] && [ -z "${NEXTFLOW_IMG}" ] \
   && command -v nextflow >/dev/null 2>&1; then
    NEXTFLOW_BIN="$(command -v nextflow)"
fi

if [ ! -x "${NEXTFLOW_BIN}" ]; then
    if [ -z "${NEXTFLOW_IMG}" ] || [ ! -f "${NEXTFLOW_IMG}" ]; then
        echo "ERROR: no nextflow launcher at ${NEXTFLOW_BIN}, none on PATH, and"
        echo "no image at '${NEXTFLOW_IMG}'. Set NEGSTEER_NEXTFLOW_IMG in"
        echo "site.conf, or install Nextflow 25.10.4."
        exit 1
    fi
    echo "Extracting the nextflow launcher from ${NEXTFLOW_IMG} ..."
    singularity exec "${NEXTFLOW_IMG}" cat /usr/local/bin/nextflow > "${NEXTFLOW_BIN}"
    chmod +x "${NEXTFLOW_BIN}"
fi

if [ -n "${NEXTFLOW_IMG}" ] && { [ ! -d "${NXF_HOME}/plugins" ] \
   || [ -z "$(ls -A "${NXF_HOME}/plugins" 2>/dev/null)" ]; }; then
    echo "Seeding NXF_HOME from ${NEXTFLOW_IMG} ..."
    singularity exec --bind "${NXF_HOME}:/mnt/out" "${NEXTFLOW_IMG}" \
        bash -c "cp -r /opt/nextflow/* /mnt/out" 2>/dev/null || true
fi

# ── The head job must be able to submit ──────────────────────────────────────
# If sbatch is not on PATH here, every stage silently falls back to nothing and
# the run dies deep inside the first task. Fail now, with a reason.
if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch is not on PATH on this node, so Nextflow cannot submit stages."
    exit 1
fi

# ── Launch ───────────────────────────────────────────────────────────────────
echo "============================================================"
echo "structure-negative-steering — Nextflow head job"
echo "============================================================"
echo "Engine:      ${REPO_DIR}"
echo "Launch dir:  ${LAUNCH_DIR}"
echo "Targets:     ${TARGETS}"
echo "NXF_HOME:    ${NXF_HOME}"
echo "NXF_WORK:    ${NXF_WORK}"
echo "Log:         ${NXF_LOG_FILE}"
echo "Java:        $(java -version 2>&1 | head -1)"
echo "Nextflow:    $("${NEXTFLOW_BIN}" -version 2>&1 | grep -m1 version || echo unknown)"
echo "sbatch:      $(command -v sbatch)"
echo "Node:        $(hostname)"
echo "Job:         ${SLURM_JOB_ID:-none}"
echo "Date:        $(date)"
echo "============================================================"

# -profile hpc selects the cluster. The directives themselves live in the
# `process` block of nextflow.config, and this profile deliberately does not
# redefine any `withLabel:` selector, because Nextflow 25.10 REPLACES such a
# block rather than merging it. That is what dropped --gres=gpu:1 and put a
# Boltz job on a partition with no GPU.
"${NEXTFLOW_BIN}" run "${REPO_DIR}/main.nf" \
    -profile hpc \
    --targets "${TARGETS}" \
    -resume \
    "$@"

echo ""
echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
