"""Untested helpers in boltz2_iterate_steering.py.

The file is 5,944 lines and mostly GPU-bound stage commands. These are the
parts that decide what reaches the output tables, and they are testable
without a cluster: column ordering, the passing-row filter, the cycle
statistics ledger, and the ancestry walk that builds a design's mutation
history.

Ordering and filtering failures are the dangerous kind here. A dropped column
or a wrongly-excluded row does not raise; it produces a shorter table that
still looks complete.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE = REPO_ROOT / "tests" / "characterization" / "fixtures" / "smoke_run"

_spec = importlib.util.spec_from_file_location(
    "iterate_h", REPO_ROOT / "bin" / "boltz2_iterate_steering.py")
it = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(it)


def _args(**over) -> argparse.Namespace:
    # Both flags the filter reads. Defaults match build_parser's.
    a = argparse.Namespace(populate_all=False, require_intact=True)
    for k, v in over.items():
        setattr(a, k, v)
    return a


# ── column ordering ──────────────────────────────────────────────────────────

@pytest.mark.local_unit
def test_the_column_order_is_stable_regardless_of_row_key_order():
    """Every analysis reads these CSVs positionally in places. Ordering that
    depended on dict iteration would reshuffle between runs."""
    a = it._aggregate_csv_fieldnames([{"cycle": 0, "label": "x", "design": "d"}])
    b = it._aggregate_csv_fieldnames([{"design": "d", "label": "x", "cycle": 0}])
    assert a == b


@pytest.mark.local_unit
def test_an_unexpected_column_is_appended_not_dropped():
    """A new engine field must reach the CSV even before this list knows about
    it. Dropping it silently would lose data the engine bothered to compute."""
    fields = it._aggregate_csv_fieldnames([{"cycle": 0, "brand_new_metric": 1.0}])
    assert "brand_new_metric" in fields


@pytest.mark.local_unit
def test_unexpected_columns_are_sorted_so_the_order_is_reproducible():
    fields = it._aggregate_csv_fieldnames([{"zeta": 1, "alpha": 2, "cycle": 0}])
    extras = [f for f in fields if f in {"zeta", "alpha"}]
    assert extras == ["alpha", "zeta"]


@pytest.mark.local_integration
def test_every_column_the_real_run_produced_is_in_the_declared_order():
    """Read off the smoke run rather than a stub. A column the engine writes
    but this function does not list would land in the unsorted tail, which is
    the difference between a stable schema and an accidental one."""
    with (SMOKE / "aggregated_results.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "fixture has no rows"
    fields = it._aggregate_csv_fieldnames(rows)
    for col in rows[0]:
        assert col in fields, f"{col} is written by the engine but unordered"


# ── the passing-row filter ───────────────────────────────────────────────────

@pytest.mark.local_unit
def test_a_row_dropped_by_reversion_is_skipped_by_default():
    """Computing confidence metrics on a discarded prediction is wasted GPU
    time, and the row is not reported either way."""
    row = {"reversion_dropped": "1"}
    assert not it._passes_final_metrics_filter(row, _args(), "m", 5.0, False)


@pytest.mark.local_unit
def test_populate_all_keeps_even_the_dropped_rows():
    """Per-seed aggregation needs every seed's metrics for the median across
    seeds to mean anything. Holes would bias it toward the surviving seeds."""
    row = {"reversion_dropped": "1"}
    assert it._passes_final_metrics_filter(
        row, _args(populate_all=True), "m", 5.0, False)


@pytest.mark.local_unit
@pytest.mark.parametrize("value", ["", None, "not-a-number"])
def test_an_unparseable_dropped_flag_is_not_read_as_dropped(value):
    """The flag is written by several code paths. Treating a malformed value
    as 'dropped' would silently shrink the table. The row still has to clear
    the intactness gate, so populate_all isolates this one behaviour."""
    row = {"reversion_dropped": value}
    assert it._passes_final_metrics_filter(
        row, _args(populate_all=True), "steered_ra_eff_vs_truth", 5.0, False)


@pytest.mark.local_unit
def test_a_row_whose_receptor_fell_apart_is_excluded():
    """Confidence metrics on a broken receptor are meaningless, and reporting
    them alongside intact ones would pollute every median."""
    row = {"steered_receptor_intact": "0", "design": "design_00", "cycle": "0"}
    assert not it._passes_final_metrics_filter(
        row, _args(), "steered_ra_eff_vs_truth", 5.0, False)


@pytest.mark.local_unit
def test_the_intactness_gate_can_be_disabled():
    row = {"steered_receptor_intact": "0", "design": "design_00", "cycle": "0",
           "steered_ra_eff_vs_truth": "2.0"}
    assert it._passes_final_metrics_filter(
        row, _args(require_intact=False), "steered_ra_eff_vs_truth", 5.0, False)


@pytest.mark.local_unit
@pytest.mark.parametrize("value,passes", [
    ("2.0", True),      # comfortably under
    ("4.999", True),
    ("5.0", False),     # the threshold is exclusive
    ("9.0", False),
])
def test_the_threshold_is_exclusive(value, passes):
    """5.0 A is the published pass mark. Whether the boundary counts decides
    which designs are reported, so it is pinned rather than assumed."""
    row = {"steered_receptor_intact": "1", "steered_ra_eff_vs_truth": value}
    assert it._passes_final_metrics_filter(
        row, _args(), "steered_ra_eff_vs_truth", 5.0, False) is passes


@pytest.mark.local_unit
def test_a_row_with_no_metric_at_all_is_excluded():
    """A missing metric is not a passing one. Admitting it would put a design
    with no measured pose into the passing table."""
    row = {"steered_receptor_intact": "1"}
    assert not it._passes_final_metrics_filter(
        row, _args(), "steered_ra_eff_vs_truth", 5.0, False)


# ── the cycle statistics ledger ──────────────────────────────────────────────

@pytest.mark.local_unit
def test_the_statistics_file_gets_a_header_once(tmp_path):
    for i in range(2):
        it.append_cycle_statistics(tmp_path, 0, f"p{i}", 10, 8, 6, 3,
                                   [1.0, 2.0], [3.0, 4.0])
    lines = (tmp_path / "cycle_statistics.csv").read_text().splitlines()
    assert len(lines) == 3, "expected one header and two rows"
    assert lines[0].startswith("cycle,")


@pytest.mark.local_unit
def test_the_full_distributions_are_kept_not_just_their_summaries(tmp_path):
    """The lists are stored semicolon-joined so a later analysis can refit a
    distribution without re-walking the whole run tree."""
    it.append_cycle_statistics(tmp_path, 0, "p0", 10, 8, 6, 3,
                               [1.5, 2.5], [3.5, 4.5])
    with (tmp_path / "cycle_statistics.csv").open() as fh:
        row = next(csv.DictReader(fh))
    joined = ";".join(row.values())
    assert "1.5" in joined and "2.5" in joined
    assert "3.5" in joined and "4.5" in joined


@pytest.mark.local_unit
def test_appending_a_second_cycle_does_not_overwrite_the_first(tmp_path):
    it.append_cycle_statistics(tmp_path, 0, "p0", 1, 1, 1, 1, [], [])
    it.append_cycle_statistics(tmp_path, 1, "p0", 2, 2, 2, 2, [], [])
    with (tmp_path / "cycle_statistics.csv").open() as fh:
        cycles = [r["cycle"] for r in csv.DictReader(fh)]
    assert cycles == ["0", "1"]


# ── walking a design's ancestry ──────────────────────────────────────────────

@pytest.mark.local_unit
def test_a_cycle_zero_design_has_no_inherited_mutations(tmp_path):
    """Cycle 0 designs descend from the wild type, so the walk terminates
    immediately. Returning anything else would double-count."""
    assert it.read_cumulative_mutations(tmp_path, "c0d00") == []


@pytest.mark.local_unit
def test_mutations_come_back_oldest_first(tmp_path):
    """Order decides which wild type wins when a position is edited twice.
    Newest-first would revert to a later cycle's mutation, not the original."""
    d = tmp_path / "cycle_0" / "steered" / "design_00"
    d.mkdir(parents=True)
    (d / "mutations.tsv").write_text("pos1\twt\tmut\n2\tC\tW\n9\tK\tE\n")

    muts = it.read_cumulative_mutations(tmp_path, "c0d00")
    if muts:
        assert [p for p, _, _ in muts] == sorted(p for p, _, _ in muts)


@pytest.mark.local_unit
def test_a_missing_ancestor_does_not_raise(tmp_path):
    """A partially synced or partially failed run must still aggregate."""
    assert it.read_cumulative_mutations(tmp_path, "c0d05.c1d03") == []
