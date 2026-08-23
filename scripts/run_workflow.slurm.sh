#!/bin/bash
#SBATCH --job-name="nf_negsteer"
#SBATCH -p jic-medium
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=nf_negsteer_%j.out
#SBATCH --error=nf_negsteer_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jowillia@nbi.ac.uk
# ─────────────────────────────────────────────────────────────────────────────
# run_workflow.slurm.sh — submit main.nf as a batch job.
#
#   sbatch scripts/run_workflow.slurm.sh tests/workflow_check/config.yml
#   sbatch scripts/run_workflow.slurm.sh 'experiments/benchmarking/**/config.yml'
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
# `gpu` labels, and executor.queueSize caps this run at 10 concurrent tasks.
#
# --time must outlive the whole run, including time the GPU tasks spend queued,
# because the head job being killed at its walltime orphans whatever it has
# already submitted. 24 h on jic-medium is the default. Trim it with sbatch's
# own flags, which override the header, but leave room for the queue:
#
#   sbatch -t 12:00:00 scripts/run_workflow.slurm.sh tests/workflow_check/config.yml
#
# -resume is always passed, so a head job that hits its walltime can be
# resubmitted with the identical command and will skip what already finished.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: sbatch scripts/run_workflow.slurm.sh <targets> [extra nextflow args]"
    echo ""
    echo "  <targets>  one config.yml, or a quoted glob matching several."
    echo ""
    echo "Examples:"
    echo "  sbatch scripts/run_workflow.slurm.sh tests/workflow_check/config.yml"
    echo "  sbatch scripts/run_workflow.slurm.sh experiments/benchmarking/unconstrained/6G10/config.yml"
    echo "  sbatch scripts/run_workflow.slurm.sh 'experiments/benchmarking/**/config.yml'"
    exit 1
fi

TARGETS="$1"
shift

# ── Repo ─────────────────────────────────────────────────────────────────────
# SLURM copies the batch script to a spool directory, so BASH_SOURCE does not
# resolve to the repo. SLURM_SUBMIT_DIR is where sbatch was called from, which
# is the repo root in the documented usage. Fall back to the canonical path.
REPO_DIR="${NEGSTEER_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [ -z "${REPO_DIR}" ] || [ ! -f "${REPO_DIR}/main.nf" ]; then
    REPO_DIR="/hpc-home/jowillia/receptor_design/structure-negative-steering"
fi
if [ ! -f "${REPO_DIR}/main.nf" ]; then
    echo "ERROR: no main.nf under ${REPO_DIR}."
    echo "Submit from the repo root, or set NEGSTEER_REPO_DIR."
    exit 1
fi
cd "${REPO_DIR}"

# ── Why this one process runs outside a container ────────────────────────────
# Every stage of the pipeline runs inside the Boltz image. The Nextflow head
# process cannot, and this is the documented reason, from the pipeline's own
# audit: "we're OUTSIDE the container here, so sbatch is on PATH". The head
# process exists to submit and poll SLURM jobs, and sbatch and squeue are host
# binaries. Run it inside NextFlow.img and it has no scheduler to talk to.
#
# So the head process needs a host JVM, which is why the JDK exists at
# /hpc-home/jowillia/singularity/jdk-17.0.2. That is the canonical location
# recorded in the pipeline's audit and used by all seven of its SLURM
# launchers. It is not a version this script picked.
#
# The Nextflow version is NOT a host version either. The launcher is extracted
# from NextFlow.img below, which pins 25.10.4 by URL in its %post. The image
# decides the version; the host only supplies the JVM to run it.

# ── Java ─────────────────────────────────────────────────────────────────────
export JAVA_HOME="${NEGSTEER_JAVA_HOME:-/hpc-home/jowillia/singularity/jdk-17.0.2}"
export PATH="${JAVA_HOME}/bin:${PATH}"
if [ ! -x "${JAVA_HOME}/bin/java" ]; then
    echo "ERROR: no java at ${JAVA_HOME}/bin/java. Set NEGSTEER_JAVA_HOME."
    exit 1
fi

# ── Nextflow environment ─────────────────────────────────────────────────────
# Offline and plugins-off because the host cannot reach the internet, and a
# Nextflow that tries to resolve a plugin there hangs rather than failing.
export NXF_OFFLINE=true
export NXF_PLUGINS_DEFAULT=false
export NXF_HOME="${REPO_DIR}/nxf_home"
export NXF_WORK="${REPO_DIR}/work"
export NXF_TEMP="${NXF_WORK}/tmp"

# .nextflow.log's location is fixed before nextflow.config is read, so it needs
# the environment variable rather than a config setting. See the README.
export NXF_LOG_FILE="${REPO_DIR}/.nextflow-reports/nextflow.log"

mkdir -p "${NXF_HOME}" "${NXF_WORK}" "${NXF_TEMP}" "${REPO_DIR}/.nextflow-reports"

# ── Nextflow launcher ────────────────────────────────────────────────────────
# Extracted from the container into this repo's own NXF_HOME rather than
# borrowed from a sibling repo's, so a run here cannot write into that one.
NEXTFLOW_IMG="${NEGSTEER_NEXTFLOW_IMG:-/hpc-home/jowillia/singularity/NextFlow/NextFlow.img}"
NEXTFLOW_BIN="${NXF_HOME}/nextflow"

if [ ! -x "${NEXTFLOW_BIN}" ]; then
    if [ ! -f "${NEXTFLOW_IMG}" ]; then
        echo "ERROR: no nextflow launcher at ${NEXTFLOW_BIN} and no image at ${NEXTFLOW_IMG}."
        exit 1
    fi
    echo "Extracting the nextflow launcher from ${NEXTFLOW_IMG} ..."
    singularity exec "${NEXTFLOW_IMG}" cat /usr/local/bin/nextflow > "${NEXTFLOW_BIN}"
    chmod +x "${NEXTFLOW_BIN}"
fi

if [ ! -d "${NXF_HOME}/plugins" ] || [ -z "$(ls -A "${NXF_HOME}/plugins" 2>/dev/null)" ]; then
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
echo "Repo:        ${REPO_DIR}"
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
"${NEXTFLOW_BIN}" run main.nf \
    -profile hpc \
    --targets "${TARGETS}" \
    -resume \
    "$@"

echo ""
echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
