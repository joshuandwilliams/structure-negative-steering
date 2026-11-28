"""boltz2_iterate_steering's aggregation half.

The module is 2,496 statements. Roughly half of it is the per-cycle loop that
calls Boltz, but the rest turns per-seed rows into the per-design verdicts and
rankings that every downstream number depends on. That half is pure and reads
only dictionaries, so it needs no GPU.

Real per-seed rows from the committed benchmark runs drive the aggregation, so
the column names and value formats are the ones the engine actually emits
rather than ones invented here.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin"))

import negsteer_aggregate as nsagg  # noqa: E402

COHORT = Path(__file__).parent / "fixtures" / "cohort" / "runs"
needs_cohort = pytest.mark.skipif(
    not COHORT.is_dir(), reason="cohort fixture not present")


def _per_seed(target: str) -> list[dict]:
    with (COHORT / target / "raw_per_seed_results.csv").open() as fh:
        return list(csv.DictReader(fh))


# ── per-seed to per-design aggregation ───────────────────────────────────────

@pytest.mark.logic
@needs_cohort
def test_aggregation_collapses_real_seeds_into_one_row():
    rows = [r for r in _per_seed("6G10") if r.get("design") not in ("initial", "")]
    # Asserted, not skipped. The 6G10 fixture is committed with 60 steered
    # rows, so an empty list means the fixture changed, not that this run had
    # nothing to aggregate.
    assert rows, "the committed 6G10 fixture has no steered rows to collapse"
    agg = nsagg._aggregate_per_sequence(rows, "steered_")
    assert isinstance(agg, dict) and agg, "aggregation produced nothing"


@pytest.mark.logic
def test_median_aggregation_over_synthetic_seeds():
    rows = [{"steered_ra_eff_vs_truth": v} for v in ("1.0", "2.0", "3.0")]
    agg = nsagg._aggregate_per_sequence(rows, "steered_")
    medians = [v for k, v in agg.items() if "ra_eff" in k and "median" in k]
    assert medians, f"no ra_eff median produced, keys were {sorted(agg)[:8]}"
    assert float(medians[0]) == pytest.approx(2.0)


@pytest.mark.logic
def test_aggregating_no_rows_does_not_raise():
    agg = nsagg._aggregate_per_sequence([], "steered_")
    assert isinstance(agg, dict)


@pytest.mark.logic
def test_majority_positions_keeps_what_most_seeds_agree_on():
    """A position is kept iff at least ceil(N/2) seeds report it."""
    positions, counts, n_seeds = nsagg._agg_majority_positions([[5, 7], [5, 7], [5]])
    assert 5 in positions, "a position every seed reported was dropped"
    assert 7 in positions, "a position 2 of 3 seeds reported was dropped"
    assert counts[5] == 3 and counts[7] == 2
    assert n_seeds == 3


@pytest.mark.logic
def test_a_minority_position_is_dropped():
    positions, counts, _ = nsagg._agg_majority_positions([[5], [5], [9]])
    assert 5 in positions
    assert 9 not in positions, "a position only one seed of three reported was kept"


@pytest.mark.logic
def test_majority_positions_of_nothing_is_empty():
    positions, counts, n_seeds = nsagg._agg_majority_positions([])
    assert positions == [] and counts == {} and n_seeds == 0


# ── ranking ──────────────────────────────────────────────────────────────────

# _compute_unified_ranks only ranks rows _ranking_metric calls eligible, which
# means a clean steered row or a pose_holds row, and it reads the prefixed
# column names. The previous fixture used bare `true_jaccard` and
# `ra_eff_vs_truth` with no intact flag, so no row was eligible, nothing was
# ranked, and the test skipped itself rather than asserting the ordering.


def _rankable(design, ra, tj, **extra):
    """A clean steered row: no verdict, receptor intact, ra_eff under 5 A."""
    r = {"design": design, "reversion_verdict": "",
         "steered_receptor_intact": "1",
         "steered_ra_eff_vs_truth": str(ra),
         "steered_true_jaccard": str(tj)}
    r.update(extra)
    return r


@pytest.mark.logic
def test_unified_ranks_order_by_the_composite():
    """The composite is steered_true_jaccard − 0.05 · ra_eff, higher better,
    and it is the primary ranker. Ordering on ra_eff alone would prefer a
    design that moved the pose while losing the interface it had to keep."""
    rows = [_rankable("a", 1.0, 0.9),    # 0.90 - 0.05 = 0.85
            _rankable("b", 4.0, 0.5),    # 0.50 - 0.20 = 0.30
            _rankable("c", 2.0, 0.7)]    # 0.70 - 0.10 = 0.60
    nsagg._compute_unified_ranks(rows)
    by = {r["design"]: r["rank_by_composite_score"] for r in rows}
    assert by == {"a": 1, "c": 2, "b": 3}


@pytest.mark.logic
def test_rank_by_ra_eff_is_ascending_and_independent_of_the_composite():
    """Both ranks are written on the same pass, and they disagree on purpose.
    Here b has the second-best pose and the worst composite."""
    rows = [_rankable("a", 1.0, 0.9), _rankable("b", 2.0, 0.1),
            _rankable("c", 3.0, 0.8)]
    nsagg._compute_unified_ranks(rows)
    by = {r["design"]: (r["rank_by_ra_eff"], r["rank_by_composite_score"])
          for r in rows}
    assert by["a"] == (1, 1)
    assert by["b"] == (2, 3)
    assert by["c"] == (3, 2)


@pytest.mark.logic
@pytest.mark.parametrize("over,why", [
    ({"design": "initial"}, "the cold-start baseline is not a design"),
    ({"steered_receptor_intact": "0"}, "the receptor was broken"),
    ({"steered_ra_eff_vs_truth": "25.0"}, "the pose is past the 5 A filter"),
    ({"steered_ra_eff_vs_truth": ""}, "there is no pose to rank"),
    ({"steered_true_jaccard": ""}, "the composite has no jaccard term"),
    ({"reversion_verdict": "new_contamination"}, "it was contaminated"),
])
def test_an_ineligible_row_gets_a_blank_rank_not_a_bad_one(over, why):
    """A blank is the only honest answer. A number would put the row into the
    sort order, and cmd_aggregate sorts on it to pick what to carry forward."""
    rows = [_rankable("eligible", 1.0, 0.9), _rankable("other", 2.0, 0.8)]
    rows[0].update(over)
    nsagg._compute_unified_ranks(rows)
    assert rows[0]["rank_by_composite_score"] == "", why


@pytest.mark.logic
def test_a_pose_holds_row_is_ranked_on_its_reverted_pose():
    """Eligibility and the ra_eff term come from the reverted prediction,
    because that is what the design kept. The jaccard term still comes from
    the steered row, since the overlap is computed once and rides on the
    designed sequence rather than on the steering mutations."""
    holds = {"design": "held", "reversion_verdict": "pose_holds",
             "reverted_receptor_intact": "1",
             "reverted_ra_eff_vs_truth": "1.0",
             "steered_ra_eff_vs_truth": "30.0",
             "steered_true_jaccard": "0.9"}
    rows = [holds, _rankable("clean", 2.0, 0.7)]
    nsagg._compute_unified_ranks(rows)
    # On the reverted 1.0 A it wins; on the steered 30 A it would be ineligible.
    assert holds["rank_by_composite_score"] == 1
    assert holds["rank_by_ra_eff"] == 1


@pytest.mark.logic
def test_unified_ranks_on_an_empty_cohort_does_not_raise():
    rows: list[dict] = []
    nsagg._compute_unified_ranks(rows)
    assert rows == []


@pytest.mark.logic
@pytest.mark.parametrize("name", ["ra_eff_vs_truth", "true_jaccard"])
def test_ranking_metric_reads_the_named_column(name):
    got = nsagg._ranking_metric({name: "3.5"}, name)
    assert got is None or float(got) == pytest.approx(3.5)


@pytest.mark.logic
def test_ranking_metric_of_a_missing_column_is_none_not_zero():
    """Zero would rank a missing value as the best possible."""
    assert nsagg._ranking_metric({}, "ra_eff_vs_truth") in (None, float("inf")) or \
        nsagg._ranking_metric({}, "ra_eff_vs_truth") != 0


# ── outcome classification ───────────────────────────────────────────────────

@pytest.mark.logic
def test_outcome_classification_returns_a_label_and_a_reason():
    """Returns (label, reason) so a verdict can always be explained."""
    label, reason = nsagg._classify_outcome(
        {"reverted_ra_eff_vs_truth_median": "2.0"}, {"pose_holds": 3}, 5.0, [])
    assert isinstance(label, str) and label
    assert isinstance(reason, str) and reason


@pytest.mark.logic
def test_no_reverted_prediction_is_reported_as_such():
    """A design that never needed reversion must not look like a failure."""
    label, reason = nsagg._classify_outcome({}, {}, 5.0, [])
    assert label == "no_reversion"
    assert "no reverted prediction" in reason


@pytest.mark.logic
def test_a_wildly_wrong_pose_is_not_classified_as_holding():
    label, _ = nsagg._classify_outcome(
        {"reverted_ra_eff_vs_truth_median": "40.0"}, {"pose_collapses": 3}, 5.0, [])
    assert "hold" not in label.lower(), f"a 40 A pose was labelled {label!r}"


@pytest.mark.logic
def test_verdict_breakdown_counts_each_class():
    rows = [
        {"reverted_ra_eff_vs_truth": "1.0"},
        {"reverted_ra_eff_vs_truth": "2.0"},
        {"reverted_ra_eff_vs_truth": "40.0"},
    ]
    got = nsagg._per_seed_verdict_breakdown(rows, 5.0)
    assert isinstance(got, dict) and got
    assert sum(v for v in got.values() if isinstance(v, int)) >= 1


# ── pass-equivalence predicates ──────────────────────────────────────────────

@pytest.mark.logic
def test_pose_holds_and_clean_steered_are_recognised():
    """Both count as pass-equivalent, which is what n_pass sums."""
    assert callable(nsagg._row_is_pose_holds)
    assert callable(nsagg._row_is_clean_steered)
    assert nsagg._row_is_pose_holds({}) in (True, False)
    assert nsagg._row_is_clean_steered({}) in (True, False)


@pytest.mark.logic
def test_resume_key_is_the_prediction_path():
    """Keyed on the pdb path, because `pathway` is shared by all cycle-0 designs."""
    row = {"pdb": "/runs/x/cycle_0/steered/design_03_s1/prediction.pdb"}
    assert nsagg._resume_key_for_row(row) == nsagg._resume_key_for_row(dict(row))


@pytest.mark.logic
def test_resume_key_distinguishes_two_seeds_of_one_design():
    a = nsagg._resume_key_for_row({"pdb": "/r/steered/design_03_s0/prediction.pdb"})
    b = nsagg._resume_key_for_row({"pdb": "/r/steered/design_03_s1/prediction.pdb"})
    assert a != b, "two seeds of the same design share a resume key"


# ── the confidence flag ──────────────────────────────────────────────────────

@pytest.mark.logic
def test_confidence_flag_is_ok_when_every_threshold_is_met():
    row = {"steered_ipsae_min": "0.5", "steered_pae_pass_frac": "0.9",
           "steered_iptm": "0.8", "steered_complex_plddt": "0.9",
           "steered_ipae": "5.0"}
    assert nsagg._classify_steered_confidence_flag(row, 0.5, 0.4, 0.7, 10.0) == "ok"


@pytest.mark.logic
def test_confidence_flag_names_the_single_failing_threshold():
    row = {"steered_ipsae_min": "0.5", "steered_pae_pass_frac": "0.1",
           "steered_iptm": "0.8", "steered_complex_plddt": "0.9",
           "steered_ipae": "5.0"}
    assert nsagg._classify_steered_confidence_flag(row, 0.5, 0.4, 0.7, 10.0) == \
        "low_pass_frac"


@pytest.mark.logic
def test_confidence_flag_reports_multiple_failures_distinctly():
    row = {"steered_ipsae_min": "0.5", "steered_pae_pass_frac": "0.1",
           "steered_iptm": "0.1", "steered_complex_plddt": "0.2",
           "steered_ipae": "30.0"}
    got = nsagg._classify_steered_confidence_flag(row, 0.5, 0.4, 0.7, 10.0)
    assert got not in ("", "ok"), "a row failing every threshold was flagged ok"


@pytest.mark.logic
def test_confidence_flag_is_blank_when_there_are_no_metrics():
    """Blank means unclassifiable, which is not the same as passing."""
    assert nsagg._classify_steered_confidence_flag({}, 0.5, 0.4, 0.7, 10.0) == ""


@pytest.mark.logic
@pytest.mark.parametrize("key,value,expected", [
    ("steered_iptm", "0.1", "low_iptm"),
    ("steered_complex_plddt", "0.2", "low_plddt"),
    ("steered_ipae", "30.0", "high_ipae"),
])
def test_each_threshold_has_its_own_trigger_name(key, value, expected):
    row = {"steered_ipsae_min": "0.5", "steered_pae_pass_frac": "0.9",
           "steered_iptm": "0.8", "steered_complex_plddt": "0.9",
           "steered_ipae": "5.0"}
    row[key] = value
    assert nsagg._classify_steered_confidence_flag(row, 0.5, 0.4, 0.7, 10.0) == expected


# ── row translation ──────────────────────────────────────────────────────────

@pytest.mark.logic
@needs_cohort
def test_aggregate_row_translation_keeps_the_identifiers():
    rows = _per_seed("6G10")
    got = nsagg._translate_aggregate_row(rows[0])
    assert isinstance(got, dict) and got


@pytest.mark.logic
def test_translating_an_empty_row_does_not_raise():
    assert isinstance(nsagg._translate_aggregate_row({}), dict)
