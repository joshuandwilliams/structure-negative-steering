"""boltz2_iterate_steering's post-prediction stages, on a real run.

The aggregate stages read a finished run directory and rebuild the tables the
tiering and the analyses consume. They call no predictor, so a committed run
drives them exactly as the cluster does.

The fixture is 6G10's real run, 540 KB, copied into a temp directory per test
so nothing writes back into the fixture. Regenerating from it must reproduce
the committed tables, which is the strongest available check that the
aggregation still means what the published numbers mean.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from unittest import mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin"))

import boltz2_iterate_steering as its  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "full_run" / "6G10"
needs_run = pytest.mark.skipif(
    not (FIXTURE / "cycle_0").is_dir(), reason="full-run fixture not present")


@pytest.fixture
def run_dir(tmp_path) -> Path:
    """A scratch copy of the committed run, so tests never mutate the fixture."""
    dst = tmp_path / "run"
    shutil.copytree(FIXTURE, dst)
    return dst


def _main(*args: str) -> int:
    with mock.patch.object(sys, "argv", ["boltz2_iterate_steering.py", *args]):
        rc = its.main()
    return 0 if rc is None else rc


def _read(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


# ── aggregate: per-seed rows into the multicycle table ───────────────────────

@pytest.mark.local_integration
@needs_run
def test_aggregate_rebuilds_the_multicycle_table(run_dir):
    before = _read(run_dir / "all_results_multicycle.csv")
    (run_dir / "all_results_multicycle.csv").unlink()

    assert _main("aggregate", "--experiment-root", str(run_dir)) == 0

    after = _read(run_dir / "all_results_multicycle.csv")
    assert len(after) == len(before), \
        f"rebuilt {len(after)} rows from a run that recorded {len(before)}"


@pytest.mark.local_integration
@needs_run
def test_aggregate_writes_the_pathway_index(run_dir):
    (run_dir / "pathways.json").unlink()
    _main("aggregate", "--experiment-root", str(run_dir))
    data = json.loads((run_dir / "pathways.json").read_text())
    assert data, "no pathways recorded"


@pytest.mark.local_integration
@needs_run
def test_aggregate_leaves_cycle_statistics_in_place(run_dir):
    """The bell-curve table survives a re-aggregate rather than being dropped."""
    assert (run_dir / "cycle_statistics.csv").is_file(), "fixture lacks the table"
    _main("aggregate", "--experiment-root", str(run_dir))
    assert (run_dir / "cycle_statistics.csv").is_file(), \
        "aggregate removed cycle_statistics.csv without rewriting it"


@pytest.mark.local_integration
@needs_run
def test_aggregate_is_idempotent(run_dir):
    """Running twice must not double the table."""
    _main("aggregate", "--experiment-root", str(run_dir))
    once = _read(run_dir / "all_results_multicycle.csv")
    _main("aggregate", "--experiment-root", str(run_dir))
    twice = _read(run_dir / "all_results_multicycle.csv")
    assert len(once) == len(twice), \
        f"a second aggregate changed the row count, {len(once)} -> {len(twice)}"


# ── aggregate-per-sequence: the per-design verdicts ──────────────────────────

@pytest.mark.local_integration
@needs_run
def test_per_sequence_aggregate_reproduces_the_committed_table(run_dir):
    """The verdict for every design must come back the same.

    This is the table the tiering reads, so a change here silently changes
    every published tier.
    """
    before = _read(run_dir / "aggregated_results.csv")
    (run_dir / "aggregated_results.csv").unlink()

    assert _main("aggregate-per-sequence", "--experiment-root", str(run_dir)) == 0

    after = _read(run_dir / "aggregated_results.csv")
    assert len(after) == len(before), \
        f"rebuilt {len(after)} designs from a run that recorded {len(before)}"

    key = "design" if "design" in before[0] else list(before[0])[0]
    before_outcomes = {r[key]: r.get("outcome") for r in before}
    after_outcomes = {r[key]: r.get("outcome") for r in after}
    changed = {k for k in before_outcomes
               if before_outcomes[k] != after_outcomes.get(k)}
    assert not changed, f"the outcome changed for {sorted(changed)}"


@pytest.mark.local_integration
@needs_run
def test_every_aggregated_design_carries_a_known_outcome(run_dir):
    _main("aggregate-per-sequence", "--experiment-root", str(run_dir))
    rows = _read(run_dir / "aggregated_results.csv")
    known = {"no_reversion", "pose_holds", "pose_collapses", "clean_steered",
             "singleton", ""}
    unknown = {r.get("outcome") for r in rows} - known
    assert not unknown, f"unrecognised outcome labels: {unknown}"


@pytest.mark.local_integration
@needs_run
@pytest.mark.parametrize("drop", [
    ["all_results_multicycle.csv"],
    ["all_results_multicycle_with_metrics.csv"],
])
def test_aggregate_per_sequence_falls_back_between_its_two_inputs(run_dir, drop):
    """It prefers the with-metrics table and falls back to the plain one.

    Either alone is enough, which is worth pinning because the two names differ
    by one word and losing the preferred one degrades silently rather than
    failing.
    """
    for name in drop:
        (run_dir / name).unlink()
    (run_dir / "aggregated_results.csv").unlink()
    assert _main("aggregate-per-sequence", "--experiment-root", str(run_dir)) == 0
    assert _read(run_dir / "aggregated_results.csv"), \
        f"dropping {drop} left the aggregate empty"


@pytest.mark.local_integration
@needs_run
def test_aggregate_per_sequence_with_no_input_at_all_emits_nothing(run_dir):
    """With every candidate input gone it must not invent a populated table."""
    for name in ("all_results_multicycle.csv",
                 "all_results_multicycle_with_metrics.csv",
                 "raw_per_seed_results.csv", "aggregated_results.csv"):
        (run_dir / name).unlink(missing_ok=True)
    rc = _main("aggregate-per-sequence", "--experiment-root", str(run_dir))
    out = run_dir / "aggregated_results.csv"
    assert rc != 0 or not out.is_file() or not _read(out), \
        "with no input at all a populated aggregate was produced"


# ── the run the fixture came from ────────────────────────────────────────────

@pytest.mark.local_integration
@needs_run
def test_the_fixture_is_a_complete_run():
    """Guards the fixture itself, so a truncated copy fails loudly."""
    for name in ("all_results_multicycle.csv", "aggregated_results.csv",
                 "raw_per_seed_results.csv", "passing_summary.csv",
                 "cross_sequence_summary.csv"):
        assert (FIXTURE / name).is_file(), f"fixture is missing {name}"
    assert (FIXTURE / "cycle_0" / "plan.json").is_file()
    assert (FIXTURE / "cycle_0" / "steered_results.csv").is_file()


@pytest.mark.local_integration
@needs_run
def test_the_fixture_cold_start_failed_so_steering_ran():
    """6G10 was picked because it steers. If that changes these tests weaken."""
    plan = json.loads((FIXTURE / "cycle_0" / "plan.json").read_text())
    assert float(plan["initial_receptor_aligned_effector_rmsd"]) > 5.0


# ── pathway bookkeeping against the real tree ────────────────────────────────

@pytest.mark.local_integration
@needs_run
def test_recorded_pathways_parse_back_to_cycle_and_design(run_dir):
    _main("aggregate", "--experiment-root", str(run_dir))
    data = json.loads((run_dir / "pathways.json").read_text())
    labels = data if isinstance(data, list) else list(data)
    for label in labels:
        if not isinstance(label, str):
            continue
        parsed = its.parse_pathway_label(label)
        assert isinstance(parsed, list), f"{label!r} did not parse"


@pytest.mark.local_integration
@needs_run
def test_the_multicycle_table_carries_the_columns_downstream_reads(run_dir):
    _main("aggregate", "--experiment-root", str(run_dir))
    rows = _read(run_dir / "all_results_multicycle.csv")
    assert rows, "the multicycle table is empty"
    for col in ("cycle", "design", "seed_index"):
        assert col in rows[0], f"the multicycle table lacks {col}"
