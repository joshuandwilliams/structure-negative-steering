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


@pytest.mark.local_unit
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


@pytest.mark.local_unit
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


@pytest.mark.local_unit
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


@pytest.mark.local_unit
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
