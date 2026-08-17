#!/bin/bash
#SBATCH --job-name="negsteer_pytest"
#SBATCH -p jic-short
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=pytest_%j.out
#SBATCH --error=pytest_%j.err

# Run the test suite on the HPC.
#
# Three tiers, one marker each:
#   local_unit         pure helpers on synthetic input, runs anywhere
#   local_integration  reads the committed fixtures and reference structures
#   hpc                needs a GPU, the Boltz container, or a finished run
#
# The two local tiers run on a Mac and need nothing from the cluster. This
# script exists for the hpc tier, which needs a GPU node and the Boltz image.
# Tests skip cleanly when their inputs are absent rather than failing.
#
# The runner image is built from containers/pytest_runner.def. It carries
# pytest, numpy, pandas, gemmi and biopython, so nothing is installed on the
# airgapped host.
#
# Usage:
#   sbatch tests/run_pytest.slurm.sh                    # full suite
#   sbatch tests/run_pytest.slurm.sh -m hpc             # the cluster tier only
#   sbatch tests/run_pytest.slurm.sh -m local_unit -q   # any pytest args
#
# The hpc tier needs a GPU, so submit it to a GPU partition:
#   sbatch -p jic-gpu --gres=gpu:1 tests/run_pytest.slurm.sh -m hpc

set -euo pipefail

REPO_DIR="${NEGSTEER_REPO:-/hpc-home/jowillia/receptor_design/structure-negative-steering}"
CONTAINER="${PYTEST_CONTAINER:-/hpc-home/jowillia/singularity/pytest/pytest_runner.img}"

if [ ! -d "${REPO_DIR}" ]; then
    echo "ERROR: repo not found at ${REPO_DIR}. Set NEGSTEER_REPO." >&2
    exit 2
fi
if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: pytest image not found at ${CONTAINER}. Set PYTEST_CONTAINER." >&2
    echo "       Build it from containers/pytest_runner.def." >&2
    exit 2
fi

cd "${REPO_DIR}"

# Default to the whole suite if no pytest args are supplied.
if [ "$#" -eq 0 ]; then
    set -- tests/ -ra
fi

echo "=== pytest in $(basename "${CONTAINER}") ==="
echo "repo: ${REPO_DIR}"
echo "args: $*"
echo

# --nv exposes the GPU driver libraries. Harmless without a GPU allocation, and
# required by anything in the hpc tier that reaches Boltz.
exec singularity exec --nv --bind "${REPO_DIR}" "${CONTAINER}" \
    python -m pytest "$@"
