#!/bin/bash
#SBATCH --job-name="negsteer_smoke"
#SBATCH -p jic-gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=smoke_%j.out
#SBATCH --error=smoke_%j.err

# -----------------------------------------------------------------------------
# Run the engine end to end on the HPC, as cheaply as it will go, then assert
# on what it produced.
#
# This is the test that covers the half of the engine a laptop cannot reach:
# boltz2_negative_steering (plan, predict-one, collect), boltz2_iterate_steering,
# reversion and extract_passing. All of them need a GPU and the Boltz image, so
# they are untestable anywhere else.
#
# tests/smoke/config.yml is 6Q76 with every cost knob at its floor, one design
# by one seed by one diffusion sample. That is roughly 3 Boltz calls against the
# ~305 a benchmark target runs. Expect minutes, not hours.
#
# Usage:
#   sbatch tests/run_smoke_negative_steering.slurm.sh
#   sbatch tests/run_smoke_negative_steering.slurm.sh --keep    # keep a previous run
#
# The run lands in tests/smoke/run/ and is gitignored. Once it exists, the
# assertions can also be re-run on their own, anywhere:
#
#   pytest -m hpc tests/characterization/test_engine_smoke_run.py
#
# To measure how much of bin/ the run actually exercised, set
# NEGSTEER_SMOKE_COVERAGE=1. That wraps every in-container Python call in
# `coverage run`, so the engine itself is instrumented rather than just the
# assertions. It needs coverage installed in the Boltz image, and is skipped
# with a warning if it is not.
# -----------------------------------------------------------------------------

set -euo pipefail

# sbatch copies this script into /var/spool/slurmd, so ${BASH_SOURCE[0]} is
# NOT the repo. Prefer the directory the job was submitted from, then the
# script's own location for a plain `bash` invocation.
if [ -n "${NEGSTEER_REPO:-}" ]; then
    REPO_DIR="${NEGSTEER_REPO}"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/tests/smoke/config.yml" ]; then
    REPO_DIR="${SLURM_SUBMIT_DIR}"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/../tests/smoke/config.yml" ]; then
    REPO_DIR="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
else
    REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
SMOKE_DIR="${REPO_DIR}/tests/smoke"
CONFIG="${SMOKE_DIR}/config.yml"
ENTRY="${REPO_DIR}/scripts/negative_steering.slurm.sh"

KEEP=""
[ "${1:-}" = "--keep" ] && KEEP=1

for f in "$CONFIG" "$ENTRY"; do
    [ -f "$f" ] || {
        echo "ERROR: missing $f" >&2
        echo "       REPO_DIR resolved to ${REPO_DIR}" >&2
        echo "       Submit from the repo root, or set NEGSTEER_REPO." >&2
        exit 2
    }
done

# A stale run/ would let the assertions pass against the previous attempt.
if [ -d "${SMOKE_DIR}/run" ] && [ -z "$KEEP" ]; then
    echo "Clearing previous smoke run at ${SMOKE_DIR}/run"
    rm -rf "${SMOKE_DIR}/run" "${SMOKE_DIR}/inputs"
fi

echo "=== smoke run: 6Q76, 1 design x 1 seed x 1 sample ==="
echo "config: ${CONFIG}"
date

START=$(date +%s)
bash "$ENTRY" "$CONFIG"
END=$(date +%s)

echo
echo "=== engine finished in $((END - START)) s ==="

# Assert on what it produced. Without this the job is just a run, not a test.
echo
echo "=== asserting on the run ==="
PYTEST_CONTAINER="${PYTEST_CONTAINER:-/hpc-home/jowillia/singularity/pytest/pytest_runner.img}"
if [ -f "$PYTEST_CONTAINER" ]; then
    exec singularity exec --bind "$REPO_DIR" "$PYTEST_CONTAINER" \
        python -m pytest "${REPO_DIR}/tests/characterization/test_engine_smoke_run.py" \
            -m hpc -ra --no-header
else
    echo "WARNING: no pytest image at ${PYTEST_CONTAINER}, skipping the assertions." >&2
    echo "         Build it from containers/pytest_runner.def, or run them yourself:" >&2
    echo "         pytest -m hpc tests/characterization/test_engine_smoke_run.py" >&2
fi
