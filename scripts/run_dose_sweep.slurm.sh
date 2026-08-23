#!/bin/bash
#SBATCH --job-name="nf_dosesweep"
#SBATCH -p jic-long
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH --time=1-12:00:00
#SBATCH --output=nf_dosesweep_%j.out
#SBATCH --error=nf_dosesweep_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jowillia@nbi.ac.uk
# ─────────────────────────────────────────────────────────────────────────────
# run_dose_sweep.slurm.sh — the mutation-set dose sweep, as one batch job.
#
#   sbatch scripts/run_dose_sweep.slurm.sh              # all five doses
#   sbatch scripts/run_dose_sweep.slurm.sh n10          # one dose
#   sbatch scripts/run_dose_sweep.slurm.sh 10           # same thing
#
# A thin wrapper. It resolves the dose selector to a targets glob and hands over
# to run_workflow.slurm.sh, which owns the Nextflow bootstrap. Running that
# script with bash rather than sbatch is deliberate: its own #SBATCH lines are
# comments to bash, and we are already inside an allocation.
#
# ONE head job for the whole sweep, not five, so the concurrency bound applies
# once rather than five times over. Five sessions would each claim the full
# bound and queue behind each other while monopolising the partition.
#
# GPU concurrency is params.max_gpu_jobs in nextflow.config, applied as maxForks on
# the gpu label. It is 26, every A100 on jic-gpu. Lower it on a busy weekday:
#
#   sbatch scripts/run_dose_sweep.slurm.sh all --max_gpu_jobs 16
#
# Sizing, from the committed benchmark's own per-target runtimes and from the
# 6G10 workflow run:
#
#   dose   configs   predictions   ~GPU-hours   ~wall at 26   ~wall at 16
#     3       18          135          13          0.6 h         0.9 h
#     5       18          222          22          0.9 h         1.4 h
#    10       18          432          42          1.7 h         2.7 h
#    20       18          828          81          3.2 h         5.2 h
#    50       18         1992         194          7.6 h        12.3 h
#   all       90         3609         351         ~14 h         ~23 h
#
# GPU-hours are the SUM across tasks, wall is that divided by the concurrency
# cap. The per-task figure used is ~350 s, which is the workflow's rate, not the
# old single job's ~100 s. Each task now pays its own container start and model
# load, and that overhead is real. Do not size from the committed benchmark's
# run_one_runtime_sec.txt.
#
# The default header, 36 h, is sized for the full sweep at 26-way concurrency
# with headroom for queue contention. For a single dose, trim it:
#
#   sbatch -p jic-medium -t 8:00:00 scripts/run_dose_sweep.slurm.sh n10
#
# If it dies partway, resubmit the identical command. run_workflow.slurm.sh
# always passes -resume, so finished targets are skipped. The cpu-labelled
# stages carry errorStrategy 'terminate', so one broken target stops the whole
# session rather than being dropped silently. That is the intended behaviour,
# and -resume is what makes it cheap to recover from.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SELECTOR="${1:-all}"
shift || true

REPO_DIR="${NEGSTEER_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [ -z "${REPO_DIR}" ] || [ ! -f "${REPO_DIR}/main.nf" ]; then
    REPO_DIR="/hpc-home/jowillia/receptor_design/structure-negative-steering"
fi
if [ ! -f "${REPO_DIR}/main.nf" ]; then
    echo "ERROR: no main.nf under ${REPO_DIR}. Submit from the repo root." >&2
    exit 1
fi

SWEEP_DIR="${REPO_DIR}/experiments/dose-sweep"
if [ ! -d "${SWEEP_DIR}" ]; then
    echo "ERROR: no ${SWEEP_DIR}. Generate it with scripts/make_dose_sweep.py." >&2
    exit 1
fi

# ── Resolve the selector ─────────────────────────────────────────────────────
case "${SELECTOR}" in
    all)
        GLOB="experiments/dose-sweep/*/*/config.yml"
        LABEL="all doses"
        ;;
    n[0-9][0-9]|n[0-9])
        DOSE="${SELECTOR}"
        GLOB="experiments/dose-sweep/${DOSE}/*/config.yml"
        LABEL="dose ${DOSE}"
        ;;
    [0-9]|[0-9][0-9])
        DOSE="$(printf 'n%02d' "${SELECTOR}")"
        GLOB="experiments/dose-sweep/${DOSE}/*/config.yml"
        LABEL="dose ${DOSE}"
        ;;
    *)
        echo "ERROR: unknown selector '${SELECTOR}'." >&2
        echo "Use 'all', or a dose such as n10 or 10. Available:" >&2
        (cd "${SWEEP_DIR}" && ls -d n* 2>/dev/null | sed 's/^/  /') >&2
        exit 1
        ;;
esac

# shellcheck disable=SC2086
N_CONFIGS=$(ls -1 ${REPO_DIR}/${GLOB} 2>/dev/null | wc -l | tr -d ' ')
if [ "${N_CONFIGS}" -eq 0 ]; then
    echo "ERROR: ${GLOB} matched no configs under ${REPO_DIR}." >&2
    exit 1
fi

echo "============================================================"
echo "Mutation-set dose sweep"
echo "============================================================"
echo "Selector:  ${SELECTOR}  (${LABEL})"
echo "Targets:   ${GLOB}"
echo "Configs:   ${N_CONFIGS}"
echo "Job:       ${SLURM_JOB_ID:-none}"
echo "Date:      $(date)"
echo "============================================================"
echo

exec bash "${REPO_DIR}/scripts/run_workflow.slurm.sh" "${GLOB}" "$@"
