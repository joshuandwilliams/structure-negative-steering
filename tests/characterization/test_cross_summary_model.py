"""
tests/characterization/test_cross_summary_model.py
-------------------------------------------------
Local-unit tests for the CrossSummarySnapshot / CrossSummaryRow types
that provide a shallow typed view over cross_sequence_summary.csv.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

_BIN = Path(__file__).resolve().parents[2] / "bin"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

import cross_summary_model as csv_view  # noqa: E402
from design_cohort import DesignCohort  # noqa: E402


def _write_csv(path: Path, rows):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


@pytest.mark.logic
def test_from_dict_typed_fields():
    row = {
        "mpnn_sequence": "design_1_seq_0",
        "row_type": "steered",
        "cross_tier": "A",
        "cross_composite_score": "0.812",
        "cross_rank_by_composite": "1",
        "cross_rank_by_ra_eff": "2",
        "within_sequence_rank": "0",
        "n_pass": "3",
        "n_seeds": "3",
        "outcome": "pose_holds",
        "outcome_reason": "no contamination",
        "run_one_runtime_sec": "1234.5",
        "extra_col": "leftover",
    }
    r = csv_view.CrossSummaryRow.from_dict(row)
    assert r.mpnn_sequence == "design_1_seq_0"
    assert r.cross_tier == "A"
    assert r.cross_composite_score == pytest.approx(0.812)
    assert r.n_pass == 3
    assert r.run_one_runtime_sec == pytest.approx(1234.5)
    assert r.extras == {"extra_col": "leftover"}


@pytest.mark.logic
def test_empty_fields_become_none():
    row = {
        "mpnn_sequence": "design_1_seq_0",
        "row_type": "steered",
        "cross_tier": "none",
        "cross_composite_score": "",
        "cross_rank_by_composite": "",
        "cross_rank_by_ra_eff": "",
        "within_sequence_rank": "",
        "n_pass": "",
        "n_seeds": "",
        "outcome": "",
        "outcome_reason": "",
        "run_one_runtime_sec": "",
    }
    r = csv_view.CrossSummaryRow.from_dict(row)
    assert r.cross_composite_score is None
    assert r.n_pass is None
    assert r.run_one_runtime_sec is None


@pytest.mark.logic
def test_snapshot_survivors_and_tier_breakdown(tmp_path: Path):
    rows = [
        {"mpnn_sequence": "design_1_seq_0", "row_type": "steered",
         "cross_tier": "A", "cross_composite_score": "0.9",
         "cross_rank_by_composite": "1", "cross_rank_by_ra_eff": "1",
         "within_sequence_rank": "0", "n_pass": "3", "n_seeds": "3",
         "outcome": "pose_holds", "outcome_reason": "",
         "run_one_runtime_sec": "100"},
        {"mpnn_sequence": "design_2_seq_0", "row_type": "steered",
         "cross_tier": "B", "cross_composite_score": "0.6",
         "cross_rank_by_composite": "2", "cross_rank_by_ra_eff": "2",
         "within_sequence_rank": "0", "n_pass": "2", "n_seeds": "3",
         "outcome": "pose_holds", "outcome_reason": "",
         "run_one_runtime_sec": "100"},
        {"mpnn_sequence": "design_3_seq_0", "row_type": "steered",
         "cross_tier": "none", "cross_composite_score": "",
         "cross_rank_by_composite": "", "cross_rank_by_ra_eff": "",
         "within_sequence_rank": "", "n_pass": "0", "n_seeds": "3",
         "outcome": "pose_collapses", "outcome_reason": "",
         "run_one_runtime_sec": "100"},
        {"mpnn_sequence": "ctrl_polyA", "row_type": "control_polyA",
         "cross_tier": "none", "cross_composite_score": "",
         "cross_rank_by_composite": "", "cross_rank_by_ra_eff": "",
         "within_sequence_rank": "", "n_pass": "0", "n_seeds": "3",
         "outcome": "", "outcome_reason": "",
         "run_one_runtime_sec": "50"},
    ]
    p = tmp_path / "cross_summary.csv"
    _write_csv(p, rows)

    snap = csv_view.CrossSummarySnapshot.from_csv(p)
    assert len(snap) == 4
    assert snap.n_steered() == 3
    assert snap.n_controls() == 1

    survivors = snap.survivors()
    assert {r.mpnn_sequence for r in survivors} == {
        "design_1_seq_0", "design_2_seq_0",
    }

    assert snap.tier_breakdown() == {"A": 1, "B": 1, "C": 0, "none": 1}

    ranked = snap.ranked_by_composite()
    assert [r.mpnn_sequence for r in ranked] == [
        "design_1_seq_0", "design_2_seq_0", "design_3_seq_0",
    ]


@pytest.mark.logic
def test_design_cohort_from_cross_summary_csv_returns_snapshot(tmp_path: Path):
    """DesignCohort.from_cross_summary_csv is the typed-snapshot path."""
    rows = [
        {"mpnn_sequence": "design_1_seq_0", "row_type": "steered",
         "cross_tier": "A", "cross_composite_score": "0.9",
         "cross_rank_by_composite": "1", "cross_rank_by_ra_eff": "1",
         "within_sequence_rank": "0", "n_pass": "3", "n_seeds": "3",
         "outcome": "pose_holds", "outcome_reason": "",
         "run_one_runtime_sec": "100"},
    ]
    p = tmp_path / "cross_summary.csv"
    _write_csv(p, rows)
    snap = DesignCohort.from_cross_summary_csv(p)
    assert isinstance(snap, csv_view.CrossSummarySnapshot)
    assert snap.get("design_1_seq_0") is not None
    assert snap.get("design_999_seq_0") is None


# ── the coercers and the row accessors ───────────────────────────────────────
#
# Every numeric column in the cross-sequence summary comes through _try_float
# or _try_int. A blank must stay blank: coercing it to 0.0 would put a real
# value where there was none, and a 0.0 ra_eff reads as a perfect pose.

def _row(seq, row_type="steered", tier="A"):
    """A CrossSummaryRow with every required field filled."""
    import cross_summary_model as m
    return m.CrossSummaryRow(
        mpnn_sequence=seq, row_type=row_type, cross_tier=tier,
        cross_composite_score=None, cross_rank_by_composite=None,
        cross_rank_by_ra_eff=None, within_sequence_rank=None,
        n_pass=None, n_seeds=None, outcome="", outcome_reason="",
        run_one_runtime_sec=None, extras={})


@pytest.mark.logic
@pytest.mark.parametrize("raw", [None, "", "not-a-number", object()])
def test_unusable_values_coerce_to_none_not_zero(raw):
    import cross_summary_model as m
    assert m._try_float(raw) is None
    assert m._try_int(raw) is None


@pytest.mark.logic
def test_usable_values_coerce(cs_unused=None):
    import cross_summary_model as m
    assert m._try_float("1.5") == 1.5
    assert m._try_int("3") == 3
    assert m._try_int(3.0) == 3


@pytest.mark.logic
def test_rows_are_normalised_to_a_tuple_and_iterable():
    """Constructed from a list, the summary must still be frozen."""
    import cross_summary_model as m
    rows = [_row("a", "steered", "A")]
    s = m.CrossSummarySnapshot(rows=rows)
    assert isinstance(s.rows, tuple)
    assert [r.mpnn_sequence for r in s] == ["a"]
    assert len(s) == 1


@pytest.mark.logic
def test_controls_are_everything_that_is_not_steered():
    import cross_summary_model as m
    rows = [_row("s", "steered", "A"),
            _row("c", "polyA", "none"),
            _row("d", "scrambled", "none")]
    s = m.CrossSummarySnapshot(rows=rows)
    assert [r.mpnn_sequence for r in s.control_rows()] == ["c", "d"]
    assert [r.mpnn_sequence for r in s.steered_rows()] == ["s"]


@pytest.mark.logic
def test_by_tier_only_looks_at_steered_rows():
    """A control sharing a tier label must not appear in a tier breakdown."""
    import cross_summary_model as m
    rows = [_row("s", "steered", "A"),
            _row("c", "polyA", "A")]
    s = m.CrossSummarySnapshot(rows=rows)
    assert [r.mpnn_sequence for r in s.by_tier("A")] == ["s"]


@pytest.mark.logic
def test_repr_reports_the_row_count():
    import cross_summary_model as m
    s = m.CrossSummarySnapshot(rows=[_row("a")])
    assert "n_steered=1" in repr(s)
