#!/bin/bash
#SBATCH --job-name="negsteer_test_run"
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=single_run_%j.out
#SBATCH --error=single_run_%j.err
#
# The partition is not in this header. It is the one resource whose name
# differs per site, so it comes from site.conf through scripts/submit.sh, or
# from an sbatch -p on the command line.

# -----------------------------------------------------------------------------
# Run the engine end to end on the HPC, as cheaply as it will go, then assert
# on what it produced.
#
# This is the test that covers the half of the engine a laptop cannot reach:
# boltz2_negative_steering (plan, predict-one, collect), boltz2_iterate_steering,
# reversion and extract_passing. All of them need a GPU and the Boltz image, so
# they are untestable anywhere else.
#
# tests/test_run/config.yml is 6G10 at 3 designs x 3 seeds x 5 samples. Its cold
# start fails at 23.9 A against the 5 A threshold, so the steering fan-out runs,
# and its committed benchmark run contaminated designs, so the reversion
# fan-out runs too. Only negsteer_n_designs is turned down, 3 against the
# shipped 20, so the fan-out is 9 predictions rather than 60.
#
# Usage:
#   ./scripts/submit.sh test-run
#   ./scripts/submit.sh test-run --keep     # keep a previous run
#   ./scripts/submit.sh test-run --via-cli  # through `negsteer run`
#
# --via-cli runs the SAME config through the tool interface instead of the
# shell orchestrator, into tests/test_run/run_cli/. Everything else is identical,
# so the two run directories are directly comparable and
# test_single_run_cli_matches_orchestrator.py asserts they agree.
#
# That comparison is the only thing that proves the tool end to end. The unit
# tests assert the CLI emits the same stage commands, which is an argv-level
# claim; this asserts the same commands produce the same numbers on a GPU.
# Run the default first, then --via-cli, then the comparison has both halves.
#
# The run lands in tests/test_run/run/ and is gitignored. Once it exists, the
# assertions can also be re-run on their own, anywhere:
#
#   pytest -m check_test_run tests/characterization/test_engine_single_run.py
#
# To measure how much of bin/ the run actually exercised, set
# NEGSTEER_SINGLE_RUN_COVERAGE=1. That wraps every in-container Python call in
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
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/tests/test_run/config.yml" ]; then
    REPO_DIR="${SLURM_SUBMIT_DIR}"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/../tests/test_run/config.yml" ]; then
    REPO_DIR="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
else
    REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
SINGLE_RUN_DIR="${REPO_DIR}/tests/test_run"
CONFIG="${SINGLE_RUN_DIR}/config.yml"
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

RUN_DIR="${SINGLE_RUN_DIR}/run"
[ -n "$VIA_CLI" ] && RUN_DIR="${SINGLE_RUN_DIR}/run_cli"

# A stale run would let the assertions pass against the previous attempt.
if [ -d "$RUN_DIR" ] && [ -z "$KEEP" ]; then
    echo "Clearing previous single-run test at ${RUN_DIR}"
    rm -rf "$RUN_DIR"
    [ -z "$VIA_CLI" ] && rm -rf "${SINGLE_RUN_DIR}/inputs"
fi

echo "=== test run: $(grep -E '^[[:space:]]*name:' "$CONFIG" | head -1 | sed -E 's/^[^:]*:[[:space:]]*//; s/"//g') ==="
echo "config: ${CONFIG}"
echo "path:   $([ -n "$VIA_CLI" ] && echo 'negsteer run (tool interface)' || echo 'shell orchestrator')"
date

START=$(date +%s)
if [ -n "$VIA_CLI" ]; then
    # The tool needs the derived inputs, which the config driver produces. Reuse
    # the orchestrator run's if present, otherwise derive them now, so this can
    # be submitted on its own.
    if [ ! -f "${SINGLE_RUN_DIR}/inputs/receptor.fasta" ]; then
        echo "--- deriving inputs (no orchestrator run to reuse) ---"
        bash "$ENTRY" "$CONFIG" --prepare-only
    fi

    CONTAINER="$(grep -E '^[[:space:]]*boltz_container:' "$CONFIG" | head -1 \
        | sed -E 's/^[^:]*:[[:space:]]*//; s/^"//; s/"[[:space:]]*$//')"
    # Read the reference from the config rather than hardcoding a target. The
    # config used to be 6Q76 and is now 6G10; a literal path silently pointed
    # the CLI run at a different complex than the orchestrator run it is
    # compared against, which is the one thing this comparison must not do.
    REFERENCE="$(cd "$SINGLE_RUN_DIR" && realpath "$(grep -E '^[[:space:]]*reference_pdb:' "$CONFIG" \
        | head -1 | sed -E 's/^[^:]*:[[:space:]]*//; s/^"//; s/"[[:space:]]*$//')")"

    # python -m rather than the bare `negsteer`, because the image has not been
    # rebuilt yet. Both reach the same CLI; PYTHONPATH is what makes the
    # bind-mounted checkout importable.
    singularity exec --nv \
        --bind "${REPO_DIR}:${REPO_DIR}" \
        --bind "$(dirname "$REFERENCE"):$(dirname "$REFERENCE")" \
        "$CONTAINER" \
        env PYTHONPATH="${REPO_DIR}" python -m negsteer.cli run \
            --name              "$(grep -E '^[[:space:]]*name:' "$CONFIG" | head -1 | sed -E 's/^[^:]*:[[:space:]]*//; s/"//g')" \
            --receptor-fasta    "${SINGLE_RUN_DIR}/inputs/receptor.fasta" \
            --effector-fasta    "${SINGLE_RUN_DIR}/inputs/effector.fasta" \
            --reference-pdb     "$REFERENCE" \
            --design-region     "${SINGLE_RUN_DIR}/inputs/design_region.txt" \
            --true-interface    "${SINGLE_RUN_DIR}/inputs/true_interface.txt" \
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
            --mode              mild \
            --seed              0 \
            --contact-cutoff    5.0 \
            --rmsd-threshold    5.0 \
            --metric-column     steered_ra_eff_vs_truth \
            --postprocess-rmsd-threshold 5.0 \
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
# The Boltz image now carries pytest, pytest-cov and coverage, so the
# assertions run in the SAME image the engine ran in. That is what makes
# coverage possible: coverage has to be installed in the process that
# executes the code, and the engine executes inside this image.
TEST_CONTAINER="${NEGSTEER_TEST_CONTAINER:-$(grep -E '^[[:space:]]*boltz_container:' "$CONFIG" \
    | head -1 | sed -E 's/.*:[[:space:]]*"?([^"]*)"?[[:space:]]*$/\1/')}"
if [ -f "$TEST_CONTAINER" ]; then
    # The same assertions run against whichever path produced the run.
    # NEGSTEER_SINGLE_RUN_DIR is how test_engine_single_run.py is pointed elsewhere.
    SINGLE_RUN_ASSERT_DIR="$SINGLE_RUN_DIR"
    TESTS="${REPO_DIR}/tests/characterization/test_engine_single_run.py"
    if [ -n "$VIA_CLI" ]; then
        # run_cli/ sits where run/ normally does, so point the tests at a
        # directory whose "run" subdir IS run_cli. A symlink is cheaper than
        # teaching the tests a second layout.
        ln -sfn "$RUN_DIR" "${SINGLE_RUN_DIR}/.cli_view_run"
        mkdir -p "${SINGLE_RUN_DIR}/.cli_view"
        ln -sfn "$RUN_DIR" "${SINGLE_RUN_DIR}/.cli_view/run"
        ln -sfn "${SINGLE_RUN_DIR}/inputs" "${SINGLE_RUN_DIR}/.cli_view/inputs"
        SINGLE_RUN_ASSERT_DIR="${SINGLE_RUN_DIR}/.cli_view"
        # Plus the comparison against the orchestrator run, which is the
        # assertion that actually validates the tool.
        TESTS="$TESTS ${REPO_DIR}/tests/characterization/test_single_run_cli_matches_orchestrator.py"
    fi
    # addopts is cleared here on purpose. These tests only READ the run's
    # output, so measuring them would report the assertions rather than the
    # engine. The engine's own coverage comes from NEGSTEER_COVERAGE below.
    singularity exec --bind "$REPO_DIR" "$TEST_CONTAINER" \
        env NEGSTEER_SINGLE_RUN_DIR="$SINGLE_RUN_ASSERT_DIR" \
        python -m pytest -o addopts="" $TESTS -m check_test_run -ra --no-header
    PYTEST_RC=$?

    # Merge the per-stage .coverage.* files the engine dropped, if any.
    if [ -n "${NEGSTEER_COVERAGE:-}" ]; then
        echo
        echo "=== combining engine coverage ==="
        singularity exec --bind "$REPO_DIR" "$TEST_CONTAINER" \
            sh -c "cd '$REPO_DIR' && python -m coverage combine && python -m coverage report --skip-covered | tail -25"
        echo
        echo "Merge with a laptop run:  coverage combine .coverage <this .coverage>"
    fi
    exit $PYTEST_RC
else
    echo "WARNING: no image at ${TEST_CONTAINER}, skipping the assertions." >&2
    echo "         Build it from containers/boltz2_negsteer_testing.def, or:" >&2
    echo "         pytest -m check_test_run tests/characterization/test_engine_single_run.py" >&2
fi
