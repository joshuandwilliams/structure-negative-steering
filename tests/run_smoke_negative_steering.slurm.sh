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
#   sbatch tests/run_smoke_negative_steering.slurm.sh --keep     # keep a previous run
#   sbatch tests/run_smoke_negative_steering.slurm.sh --via-cli  # through `negsteer run`
#
# --via-cli runs the SAME config through the tool interface instead of the
# shell orchestrator, into tests/smoke/run_cli/. Everything else is identical,
# so the two run directories are directly comparable and
# test_smoke_cli_matches_orchestrator.py asserts they agree.
#
# That comparison is the only thing that proves the tool end to end. The unit
# tests assert the CLI emits the same stage commands, which is an argv-level
# claim; this asserts the same commands produce the same numbers on a GPU.
# Run the default first, then --via-cli, then the comparison has both halves.
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
VIA_CLI=""
for arg in "$@"; do
    case "$arg" in
        --keep)    KEEP=1 ;;
        --via-cli) VIA_CLI=1 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

for f in "$CONFIG" "$ENTRY"; do
    [ -f "$f" ] || {
        echo "ERROR: missing $f" >&2
        echo "       REPO_DIR resolved to ${REPO_DIR}" >&2
        echo "       Submit from the repo root, or set NEGSTEER_REPO." >&2
        exit 2
    }
done

RUN_DIR="${SMOKE_DIR}/run"
[ -n "$VIA_CLI" ] && RUN_DIR="${SMOKE_DIR}/run_cli"

# A stale run would let the assertions pass against the previous attempt.
if [ -d "$RUN_DIR" ] && [ -z "$KEEP" ]; then
    echo "Clearing previous smoke run at ${RUN_DIR}"
    rm -rf "$RUN_DIR"
    [ -z "$VIA_CLI" ] && rm -rf "${SMOKE_DIR}/inputs"
fi

echo "=== smoke run: 6Q76, 1 design x 1 seed x 1 sample ==="
echo "config: ${CONFIG}"
echo "path:   $([ -n "$VIA_CLI" ] && echo 'negsteer run (tool interface)' || echo 'shell orchestrator')"
date

START=$(date +%s)
if [ -n "$VIA_CLI" ]; then
    # The tool needs the derived inputs, which the config driver produces. Reuse
    # the orchestrator run's if present, otherwise derive them now, so this can
    # be submitted on its own.
    if [ ! -f "${SMOKE_DIR}/inputs/receptor.fasta" ]; then
        echo "--- deriving inputs (no orchestrator run to reuse) ---"
        bash "$ENTRY" "$CONFIG" --prepare-only
    fi

    CONTAINER="$(grep -E '^[[:space:]]*boltz_container:' "$CONFIG" | head -1 \
        | sed -E 's/^[^:]*:[[:space:]]*//; s/^"//; s/"[[:space:]]*$//')"
    REFERENCE="$(cd "$SMOKE_DIR" && realpath ../../experiments/benchmarking/unconstrained/6Q76/6Q76.pdb)"

    # python -m rather than the bare `negsteer`, because the image has not been
    # rebuilt yet. Both reach the same CLI; PYTHONPATH is what makes the
    # bind-mounted checkout importable.
    singularity exec --nv \
        --bind "${REPO_DIR}:${REPO_DIR}" \
        --bind "$(dirname "$REFERENCE"):$(dirname "$REFERENCE")" \
        "$CONTAINER" \
        env PYTHONPATH="${REPO_DIR}" python -m negsteer.cli run \
            --name              6Q76_smoke \
            --receptor-fasta    "${SMOKE_DIR}/inputs/receptor.fasta" \
            --effector-fasta    "${SMOKE_DIR}/inputs/effector.fasta" \
            --reference-pdb     "$REFERENCE" \
            --design-region     "${SMOKE_DIR}/inputs/design_region.txt" \
            --true-interface    "${SMOKE_DIR}/inputs/true_interface.txt" \
            --receptor-chain    A \
            --effector-chain    B \
            --outdir            "$RUN_DIR" \
            --n-designs         1 \
            --num-seeds         1 \
            --diffusion-samples 1 \
            --recycling-steps   1 \
            --max-mutations     3 \
            --candidate-pool-size 6 \
            --protected-set-source true_interface \
            --seed              0 \
            --contact-cutoff    5.0 \
            --rmsd-threshold    5.0 \
            --metric-column     steered_ra_eff_vs_truth \
            --postprocess-contact-cutoff 5.0
else
    bash "$ENTRY" "$CONFIG"
fi
END=$(date +%s)

echo
echo "=== engine finished in $((END - START)) s ==="

# Assert on what it produced. Without this the job is just a run, not a test.
echo
echo "=== asserting on the run ==="
PYTEST_CONTAINER="${PYTEST_CONTAINER:-/hpc-home/jowillia/singularity/pytest/pytest_runner.img}"
if [ -f "$PYTEST_CONTAINER" ]; then
    # The same assertions run against whichever path produced the run.
    # NEGSTEER_SMOKE_DIR is how test_engine_smoke_run.py is pointed elsewhere.
    SMOKE_ASSERT_DIR="$SMOKE_DIR"
    TESTS="${REPO_DIR}/tests/characterization/test_engine_smoke_run.py"
    if [ -n "$VIA_CLI" ]; then
        # run_cli/ sits where run/ normally does, so point the tests at a
        # directory whose "run" subdir IS run_cli. A symlink is cheaper than
        # teaching the tests a second layout.
        ln -sfn "$RUN_DIR" "${SMOKE_DIR}/.cli_view_run"
        mkdir -p "${SMOKE_DIR}/.cli_view"
        ln -sfn "$RUN_DIR" "${SMOKE_DIR}/.cli_view/run"
        ln -sfn "${SMOKE_DIR}/inputs" "${SMOKE_DIR}/.cli_view/inputs"
        SMOKE_ASSERT_DIR="${SMOKE_DIR}/.cli_view"
        # Plus the comparison against the orchestrator run, which is the
        # assertion that actually validates the tool.
        TESTS="$TESTS ${REPO_DIR}/tests/characterization/test_smoke_cli_matches_orchestrator.py"
    fi
    exec singularity exec --bind "$REPO_DIR" "$PYTEST_CONTAINER" \
        env NEGSTEER_SMOKE_DIR="$SMOKE_ASSERT_DIR" \
        python -m pytest $TESTS -m hpc -ra --no-header
else
    echo "WARNING: no pytest image at ${PYTEST_CONTAINER}, skipping the assertions." >&2
    echo "         Build it from containers/pytest_runner.def, or run them yourself:" >&2
    echo "         pytest -m hpc tests/characterization/test_engine_smoke_run.py" >&2
fi
