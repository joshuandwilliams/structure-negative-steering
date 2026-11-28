"""The tool and the runner behind the benchmark produce the same science.

test_negsteer_cli.py asserts the CLI emits the same stage commands as
bin/negative_steering_run_one.sh. That is an argv-level claim. It cannot catch
a difference in what those commands do once a GPU is involved.

This closes that gap. Both paths run tests/test_run/config.yml on the same
hardware with the same seed, and their run directories are compared:

    sbatch tests/test_run.slurm.sh            -> run/
    sbatch tests/test_run.slurm.sh --via-cli  -> run_cli/

Skips cleanly unless both exist, so it costs nothing when only one has run.
The tree comes from single_run.py, so NEGSTEER_SINGLE_RUN_DIR points this and
test_engine_single_run.py at the same place. It used to read
NEGSTEER_SINGLE_RUN_BASE, which nothing sets, so on a relocated tree it looked
at the checkout, found nothing and skipped.

On tolerance: Boltz is seeded, but CUDA kernel scheduling is not bit-exact
across runs, so structural metrics are compared with a tolerance rather than
for equality. The tolerance is deliberately tight. It is there to absorb
floating-point drift, not a different pose. Anything categorical, meaning row
counts, statuses, mutation positions and tier, must match exactly, because
nondeterminism cannot explain a difference in those.
"""

from __future__ import annotations

import json

import pytest

# Shared with test_engine_single_run.py, which asserts on the same tree.
# single_run.py is a sibling module and pytest puts this directory on sys.path.
from single_run import RUN_NAME, SINGLE_RUN
from single_run import read_csv as _read

ORCH = SINGLE_RUN / "run"
CLI = SINGLE_RUN / "run_cli"

# Kernel scheduling moves the last decimal place, not the answer.
RMSD_TOL_A = 0.15

needs_both = pytest.mark.skipif(
    not ((ORCH / "raw_per_seed_results.csv").is_file()
         and (CLI / "raw_per_seed_results.csv").is_file()),
    reason=f"needs both {ORCH} and {CLI}. Run the single-run job twice, once with "
           "--via-cli.")


@pytest.mark.check_test_run
def test_the_two_run_directories_are_not_the_same_tree():
    """Not a comparison, a guard on the comparison.

    tests/test_run.slurm.sh runs the --via-cli job against a `.cli_view`
    directory whose `run` is a symlink to `run_cli`, so that
    test_engine_single_run.py can assert on the CLI output without learning a
    second layout. Point this module at that view and ORCH resolves to the CLI
    run, CLI resolves to nothing, and every test below either skips or compares
    the CLI run against itself and passes for free.

    This runs unconditionally, so a tree that cannot support the comparison
    fails here rather than passing quietly below.
    """
    if not ORCH.exists():
        pytest.skip(f"no run/ under {SINGLE_RUN}")

    wrong_dir = (f"Point NEGSTEER_SINGLE_RUN_DIR at the directory holding both "
                 f"run/ and run_cli/, not at the .cli_view shim. It is at "
                 f"{SINGLE_RUN}.")

    # The shim's only possible shape: run/ is a link whose target is run_cli/.
    assert ORCH.resolve().name != "run_cli", (
        f"{ORCH} resolves to {ORCH.resolve()}, so the orchestrator side is "
        f"really the CLI run. {wrong_dir}")

    if CLI.exists():
        assert ORCH.resolve() != CLI.resolve(), (
            f"{ORCH} and {CLI} are one tree at {ORCH.resolve()}, so this "
            f"module would compare a run against itself. {wrong_dir}")


def _key(row: dict) -> tuple[str, str]:
    return row.get("design", ""), row.get("seed_index", "")


@pytest.mark.check_test_run
@needs_both
def test_both_paths_produced_the_same_rows():
    """A different row count means one path ran a different number of designs,
    which no amount of GPU nondeterminism explains."""
    orch = {_key(r) for r in _read(ORCH / "raw_per_seed_results.csv")}
    cli = {_key(r) for r in _read(CLI / "raw_per_seed_results.csv")}
    assert orch == cli, (
        f"only the orchestrator produced {sorted(orch - cli)}; "
        f"only the CLI produced {sorted(cli - orch)}")


@pytest.mark.check_test_run
@needs_both
def test_both_paths_wrote_the_same_columns():
    """A missing column is invisible to the analyses rather than an error, so
    a column the tool drops would silently degrade every downstream table."""
    orch = set(_read(ORCH / "raw_per_seed_results.csv")[0])
    cli = set(_read(CLI / "raw_per_seed_results.csv")[0])
    assert orch == cli, f"column difference: {orch ^ cli}"


@pytest.mark.check_test_run
@needs_both
def test_every_row_agrees_on_status():
    orch = {_key(r): r for r in _read(ORCH / "raw_per_seed_results.csv")}
    for row in _read(CLI / "raw_per_seed_results.csv"):
        assert row.get("status") == orch[_key(row)].get("status"), \
            f"{_key(row)}: status differs"


@pytest.mark.check_test_run
@needs_both
def test_the_pose_agrees_within_tolerance():
    """The number the whole method is judged on. If ra_eff differs by more
    than kernel noise, the tool is not doing what the benchmark runner did."""
    orch = {_key(r): r for r in _read(ORCH / "raw_per_seed_results.csv")}
    compared = 0
    for row in _read(CLI / "raw_per_seed_results.csv"):
        a, b = row.get("steered_ra_eff_vs_truth"), orch[_key(row)].get(
            "steered_ra_eff_vs_truth")
        if a in ("", None) or b in ("", None):
            continue
        assert abs(float(a) - float(b)) <= RMSD_TOL_A, (
            f"{_key(row)}: ra_eff {a} via the CLI against {b} via the "
            f"orchestrator, beyond the {RMSD_TOL_A} A tolerance")
        compared += 1
    assert compared, "no comparable ra_eff values, so this asserted nothing"


@pytest.mark.check_test_run
@needs_both
def test_the_same_residues_were_mutated():
    """Steering picks positions from a seeded ranking. Different positions
    mean a different steering decision, not a different rounding."""
    orch = {_key(r): r for r in _read(ORCH / "raw_per_seed_results.csv")}
    for row in _read(CLI / "raw_per_seed_results.csv"):
        assert row.get("steered_mutations_chimerax", "") == \
            orch[_key(row)].get("steered_mutations_chimerax", ""), \
            f"{_key(row)}: different residues mutated"


@pytest.mark.check_test_run
@needs_both
def test_the_representative_gets_the_same_tier():
    """Tier is what the cohort-level analyses key on."""
    a = _read(ORCH / "cross_sequence_summary.csv")
    b = _read(CLI / "cross_sequence_summary.csv")
    assert len(a) == len(b) == 1
    assert a[0]["cross_tier"] == b[0]["cross_tier"]


@pytest.mark.check_test_run
@needs_both
def test_only_the_cli_run_carries_a_result_manifest():
    """The manifest is the tool's contract and the orchestrator has no such
    thing. Its absence on one side is the expected asymmetry, and asserting it
    stops the comparison being read as "these are interchangeable"."""
    manifest = CLI / "negsteer_result.json"
    assert manifest.is_file(), "the CLI run wrote no negsteer_result.json"
    assert not (ORCH / "negsteer_result.json").exists()

    payload = json.loads(manifest.read_text())
    assert payload["name"] == RUN_NAME
    assert payload["status"] in {"ok", "skipped_steering"}
    assert payload["runtime_seconds"] > 0
