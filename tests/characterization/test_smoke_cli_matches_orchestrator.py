"""The tool and the runner behind the benchmark produce the same science.

test_negsteer_cli.py asserts the CLI emits the same stage commands as
bin/negative_steering_run_one.sh. That is an argv-level claim. It cannot catch
a difference in what those commands do once a GPU is involved.

This closes that gap. Both paths run tests/smoke/config.yml on the same
hardware with the same seed, and their run directories are compared:

    sbatch tests/run_smoke_negative_steering.slurm.sh            -> run/
    sbatch tests/run_smoke_negative_steering.slurm.sh --via-cli  -> run_cli/

Skips cleanly unless both exist, so it costs nothing when only one has run.

On tolerance: Boltz is seeded, but CUDA kernel scheduling is not bit-exact
across runs, so structural metrics are compared with a tolerance rather than
for equality. The tolerance is deliberately tight. It is there to absorb
floating-point drift, not a different pose. Anything categorical, meaning row
counts, statuses, mutation positions and tier, must match exactly, because
nondeterminism cannot explain a difference in those.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE = Path(os.environ.get("NEGSTEER_SMOKE_BASE", REPO_ROOT / "tests" / "smoke"))
ORCH = SMOKE / "run"
CLI = SMOKE / "run_cli"

# Kernel scheduling moves the last decimal place, not the answer.
RMSD_TOL_A = 0.15

needs_both = pytest.mark.skipif(
    not ((ORCH / "raw_per_seed_results.csv").is_file()
         and (CLI / "raw_per_seed_results.csv").is_file()),
    reason=f"needs both {ORCH} and {CLI}. Run the smoke job twice, once with "
           "--via-cli.")


def _read(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _key(row: dict) -> tuple[str, str]:
    return row.get("design", ""), row.get("seed_index", "")


@pytest.mark.hpc
@needs_both
def test_both_paths_produced_the_same_rows():
    """A different row count means one path ran a different number of designs,
    which no amount of GPU nondeterminism explains."""
    orch = {_key(r) for r in _read(ORCH / "raw_per_seed_results.csv")}
    cli = {_key(r) for r in _read(CLI / "raw_per_seed_results.csv")}
    assert orch == cli, (
        f"only the orchestrator produced {sorted(orch - cli)}; "
        f"only the CLI produced {sorted(cli - orch)}")


@pytest.mark.hpc
@needs_both
def test_both_paths_wrote_the_same_columns():
    """A missing column is invisible to the analyses rather than an error, so
    a column the tool drops would silently degrade every downstream table."""
    orch = set(_read(ORCH / "raw_per_seed_results.csv")[0])
    cli = set(_read(CLI / "raw_per_seed_results.csv")[0])
    assert orch == cli, f"column difference: {orch ^ cli}"


@pytest.mark.hpc
@needs_both
def test_every_row_agrees_on_status():
    orch = {_key(r): r for r in _read(ORCH / "raw_per_seed_results.csv")}
    for row in _read(CLI / "raw_per_seed_results.csv"):
        assert row.get("status") == orch[_key(row)].get("status"), \
            f"{_key(row)}: status differs"


@pytest.mark.hpc
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


@pytest.mark.hpc
@needs_both
def test_the_same_residues_were_mutated():
    """Steering picks positions from a seeded ranking. Different positions
    mean a different steering decision, not a different rounding."""
    orch = {_key(r): r for r in _read(ORCH / "raw_per_seed_results.csv")}
    for row in _read(CLI / "raw_per_seed_results.csv"):
        assert row.get("steered_mutations_chimerax", "") == \
            orch[_key(row)].get("steered_mutations_chimerax", ""), \
            f"{_key(row)}: different residues mutated"


@pytest.mark.hpc
@needs_both
def test_the_representative_gets_the_same_tier():
    """Tier is what the cohort-level analyses key on."""
    a = _read(ORCH / "cross_sequence_summary.csv")
    b = _read(CLI / "cross_sequence_summary.csv")
    assert len(a) == len(b) == 1
    assert a[0]["cross_tier"] == b[0]["cross_tier"]


@pytest.mark.hpc
@needs_both
def test_only_the_cli_run_carries_a_result_manifest():
    """The manifest is the tool's contract and the orchestrator has no such
    thing. Its absence on one side is the expected asymmetry, and asserting it
    stops the comparison being read as "these are interchangeable"."""
    manifest = CLI / "negsteer_result.json"
    assert manifest.is_file(), "the CLI run wrote no negsteer_result.json"
    assert not (ORCH / "negsteer_result.json").exists()

    payload = json.loads(manifest.read_text())
    assert payload["name"] == "6Q76_smoke"
    assert payload["status"] in {"ok", "skipped_steering"}
    assert payload["runtime_seconds"] > 0
