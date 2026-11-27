"""tests/test_run.slurm.sh must hand each assertion module the tree it needs.

This is a text-level check on the runner, in the laptop tier, because the path
it guards runs only on a GPU node with NEGSTEER_TEST_CONTAINER set. Nothing
else exercises it, which is exactly how it came to be wrong.

What was wrong: the runner builds a `.cli_view` directory whose `run` is a
symlink to `run_cli`, so test_engine_single_run.py can assert on the CLI output
without learning a second layout. It then handed that same directory to
test_single_run_cli_matches_orchestrator.py, which reads <dir>/run and
<dir>/run_cli as two separate trees. Against `.cli_view` the orchestrator side
resolves to the CLI run and the CLI side does not exist, so the comparison
could only skip. Had both resolved it would have compared the CLI run against
itself and passed without comparing anything.

The comparison module carries its own guard against that, in
test_the_two_run_directories_are_not_the_same_tree. These assert the runner
does not put it in that position to begin with.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "tests" / "test_run.slurm.sh"

COMPARISON = "test_single_run_cli_matches_orchestrator.py"


@pytest.fixture(scope="module")
def body() -> str:
    """The runner with its comment lines removed.

    The module names appear in the header comments too, so matching against
    the raw text would find prose rather than an invocation.
    """
    return "\n".join(line for line in RUNNER.read_text().splitlines()
                      if not line.lstrip().startswith("#"))


@pytest.mark.logic
def test_the_runner_still_runs_the_comparison(body):
    assert COMPARISON in body, (
        "the runner no longer invokes the orchestrator-to-CLI comparison, so "
        "--via-cli proves only that the tool ran, not that it agrees")


@pytest.mark.logic
def test_the_comparison_is_given_the_real_tree_not_the_cli_view(body):
    """The invocation that names the comparison must set
    NEGSTEER_SINGLE_RUN_DIR to SINGLE_RUN_DIR, the directory holding both
    run/ and run_cli/, and not to SINGLE_RUN_ASSERT_DIR, which is .cli_view
    under --via-cli."""
    # The comparison's own invocation: the nearest env assignment above the
    # line that names the module.
    upto = body[:body.index(COMPARISON)]
    assignment = upto[upto.rindex('env NEGSTEER_SINGLE_RUN_DIR'):].splitlines()[0]
    assert '"$SINGLE_RUN_DIR"' in assignment, (
        f"the comparison is invoked with {assignment.strip()}, which under "
        "--via-cli is the .cli_view shim. It would then compare the CLI run "
        "against itself.")


@pytest.mark.logic
def test_the_engine_assertions_keep_the_cli_view(body):
    """The other half of the same split. test_engine_single_run.py does want
    the view, because it asserts on one run and the view is how the CLI run is
    presented as `run`."""
    head = body[:body.index(COMPARISON)]
    assignment = head[head.index('env NEGSTEER_SINGLE_RUN_DIR'):].splitlines()[0]
    assert '"$SINGLE_RUN_ASSERT_DIR"' in assignment, (
        "the engine assertions are no longer given SINGLE_RUN_ASSERT_DIR, so "
        "a --via-cli run would assert on the orchestrator's output instead of "
        "the CLI's")


@pytest.mark.logic
def test_a_failing_comparison_cannot_be_masked(body):
    """Two pytest invocations mean two exit codes. The script must not report
    success because the first one passed."""
    assert '[ "$PYTEST_RC" -eq 0 ] && PYTEST_RC=$CMP_RC' in body, (
        "the comparison's exit code is not folded into the script's, so a "
        "failing comparison would leave the job green")
