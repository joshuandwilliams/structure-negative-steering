"""The data-selection policy inside the two plot modules.

The existing figure tests render every plot from the cohort fixture and assert
the PNGs are not blank. That covers the drawing. What it cannot cover is which
number each figure draws, and that is where the two modules have already been
wrong once.

Three decisions are made before anything is plotted.

Which prediction is the final one. A sequence that went through reversion has
both a steered and a reverted prediction, and the reverted one is the last
prediction made for it. Reading `steered_*` unconditionally was the previous
behaviour and it glossed over reversion failures: the module comment records
d0_s3 appearing with steered ra_eff 3.013 and true Jaccard 0.7 while its
reverted prediction had collapsed to ra_eff 30.5. That is the exact failure
negative steering exists to detect, shown as a success.

Which sequence group represents a sequence. A sequence has several sequence
groups and cross_summary names a representative for tiered sequences. For
tier-none sequences it names none, so the modules pick the best composite
among non-singleton rows. Picking the first row instead would show a
sequence's worst attempt as its result.

What a seed's outcome is. classify_seed sorts each per-seed row into one of
three stages and six outcomes, from six metric columns. The heatmap, the bar
chart and the trajectory plot are all built from it.

None of this needs a figure, so it is tested directly against constructed
rows with known answers.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin"))

FIXTURES = Path(__file__).parent / "fixtures" / "cohort"
RUNS = FIXTURES / "runs"
COHORT_CSV = FIXTURES / "cross_sequence_summary.csv"

needs_fixture = pytest.mark.skipif(
    not COHORT_CSV.is_file(), reason="cohort fixture not present")


@pytest.fixture(scope="module")
def npl():
    import negsteer_plots
    return negsteer_plots


@pytest.fixture(scope="module")
def nwp():
    import negsteer_within_sequence_plots
    return negsteer_within_sequence_plots


# ── which prediction is the final one ────────────────────────────────────────
#
# A verdict in {pose_holds, pose_collapses, new_contamination} means reversion
# ran, so the reverted prediction is the last one made. Anything else means it
# did not, and the steered prediction is the only one that exists.

AGG_BOTH = {
    "outcome": "pose_holds",
    "steered_ra_eff_vs_truth_median": "3.013",
    "reverted_ra_eff_vs_truth_median": "30.5",
    "steered_true_jaccard_median": "0.7",
    "reverted_true_jaccard_median": "0.02",
    "n_seeds": "4", "n_pass": "3",
}


@pytest.mark.logic
@pytest.mark.parametrize("verdict", ["pose_holds", "pose_collapses",
                                     "new_contamination"])
def test_a_reverted_sequence_is_read_off_its_reverted_prediction(npl,
                                                                 verdict):
    """The numbers here are the ones from the module's own bug note. Reading
    the steered side would show a collapsed reversion as a 3 A success."""
    row = dict(AGG_BOTH, outcome=verdict)
    assert npl._final_prediction_value(
        row, "rep_ra_eff_vs_truth_median") == "30.5"
    assert npl._final_prediction_value(
        row, "rep_true_jaccard_median") == "0.02"


@pytest.mark.logic
@pytest.mark.parametrize("verdict", ["clean_steered", "no_reversion",
                                     "singleton", ""])
def test_a_sequence_that_never_reverted_is_read_off_its_steered_prediction(
        npl, verdict):
    """There is no reverted prediction for these, so the steered one is the
    only prediction the sequence has."""
    row = dict(AGG_BOTH, outcome=verdict)
    assert npl._final_prediction_value(
        row, "rep_ra_eff_vs_truth_median") == "3.013"


@pytest.mark.logic
def test_a_reversion_that_produced_nothing_falls_back_to_steered(npl):
    """The verdict says reversion ran but the prediction failed, so the
    reverted column is blank. Returning the blank would drop the sequence
    from every figure rather than showing its steered stage."""
    row = dict(AGG_BOTH, reverted_ra_eff_vs_truth_median="")
    assert npl._final_prediction_value(
        row, "rep_ra_eff_vs_truth_median") == "3.013"


@pytest.mark.logic
def test_the_seed_counts_have_no_reverted_equivalent(npl):
    """n_seeds and n_pass are counted once per sequence group, not per
    stage, so there is a single source of truth for them."""
    row = dict(AGG_BOTH)
    assert npl._final_prediction_value(row, "rep_n_seeds") == "4"
    assert npl._final_prediction_value(row, "rep_n_pass") == "3"
    assert npl._final_prediction_value(row, "rep_outcome") == "pose_holds"


@pytest.mark.logic
def test_a_field_outside_the_map_is_blank_rather_than_an_error(npl):
    assert npl._final_prediction_value(AGG_BOTH, "rep_not_a_field") == ""


@pytest.mark.logic
def test_the_projection_marks_an_aggregate_row_as_tier_none(npl):
    """cross_tier is computed by cross_summary after the passing filter, so
    a sequence reached only through aggregated_results.csv has no tier. A
    default of A or B would place it in the cohort ranking."""
    out = npl._project_agg_row("design_0_seq_1", AGG_BOTH)
    assert out["mpnn_sequence"] == "design_0_seq_1"
    assert out["cross_tier"] == "none"
    assert out["cross_rank_by_composite"] == ""
    assert out["rep_ra_eff_vs_truth_median"] == "30.5"


# ── which sequence group represents a sequence ───────────────────────────────

def _agg(sg, *, ra, tj, outcome="clean_steered", cycle="0", pathway=""):
    return {"sequence_group": sg, "cycle": cycle, "pathway": pathway,
            "outcome": outcome,
            "steered_ra_eff_vs_truth_median": str(ra),
            "steered_true_jaccard_median": str(tj)}


@pytest.mark.logic
def test_the_representative_named_by_cross_summary_is_used_exactly(npl):
    """It is the row the cross-rank was computed on, so substituting a
    better-looking row would make the plotted point disagree with the rank
    beside it."""
    rows = [_agg("0", ra=2.0, tj=0.8), _agg("1", ra=30.0, tj=0.01)]
    got = npl._pick_best_agg_row(rows, prefer_cycle="0", prefer_pathway="",
                                 prefer_sequence_group="1")
    assert got["sequence_group"] == "1"


@pytest.mark.logic
def test_a_tier_none_sequence_falls_back_to_its_best_composite(npl):
    """tier-none means no row passed the filters, but the per-sequence best
    is still the informative summary. Taking the first row instead would
    show a sequence's worst attempt as its result."""
    rows = [_agg("0", ra=30.0, tj=0.01), _agg("1", ra=2.0, tj=0.8)]
    got = npl._pick_best_agg_row(rows, prefer_sequence_group="")
    assert got["sequence_group"] == "1"


@pytest.mark.logic
def test_the_composite_is_jaccard_minus_a_twentieth_of_the_rmsd(npl):
    """0.05 is the weight, so 1 A of pose error costs 0.05 of Jaccard. These
    two rows are constructed to cross over: 0.50 - 0.05*2 = 0.40 beats
    0.60 - 0.05*6 = 0.30."""
    rows = [_agg("0", ra=6.0, tj=0.60), _agg("1", ra=2.0, tj=0.50)]
    assert npl._pick_best_agg_row(rows,
                                  prefer_sequence_group="")["sequence_group"] == "1"
    assert npl.COMPOSITE_RA_EFF_WEIGHT == 0.05


@pytest.mark.logic
def test_a_singleton_row_is_not_a_representative(npl):
    """A singleton is a fallback row with no real seed counts, so a
    dispersion or pass-rate figure built on it would be describing one
    prediction as if it were a group."""
    rows = [_agg("0", ra=1.0, tj=0.9, outcome="singleton"),
            _agg("1", ra=4.0, tj=0.5)]
    assert npl._pick_best_agg_row(
        rows, prefer_sequence_group="")["sequence_group"] == "1"


@pytest.mark.logic
def test_a_sequence_of_only_singletons_still_gets_a_row(npl):
    """Returning nothing would drop the sequence from the cohort entirely."""
    rows = [_agg("0", ra=1.0, tj=0.9, outcome="singleton")]
    assert npl._pick_best_agg_row(
        rows, prefer_sequence_group="")["sequence_group"] == "0"


@pytest.mark.logic
def test_a_row_with_missing_numbers_loses_to_a_real_one(npl):
    """A missing composite has to rank last. Treating it as zero would let
    a row with no data outscore a real row with a negative composite."""
    rows = [{"sequence_group": "0", "outcome": "clean_steered",
             "steered_ra_eff_vs_truth_median": "",
             "steered_true_jaccard_median": ""},
            _agg("1", ra=40.0, tj=0.0)]
    assert npl._pick_best_agg_row(
        rows, prefer_sequence_group="")["sequence_group"] == "1"


@pytest.mark.logic
def test_no_rows_at_all_is_none(npl):
    assert npl._pick_best_agg_row([]) is None


@pytest.mark.logic
def test_a_representative_pointer_that_matches_nothing_falls_through(npl):
    """A stale cross_summary naming a sequence group the aggregate no longer
    has must not return nothing, or the sequence vanishes from the figures."""
    rows = [_agg("0", ra=2.0, tj=0.8)]
    got = npl._pick_best_agg_row(rows, prefer_cycle="0", prefer_pathway="",
                                 prefer_sequence_group="9")
    assert got["sequence_group"] == "0"


# ── the same choice, in the within-sequence module ───────────────────────────

@pytest.mark.logic
def test_the_within_module_prefers_the_cross_summary_pointer(nwp, tmp_path):
    d = tmp_path / "design_0_seq_1"
    d.mkdir()
    _write_agg(d, [_agg("0", ra=2.0, tj=0.8), _agg("1", ra=30.0, tj=0.0)])
    lookup = {"design_0_seq_1": {"rep_sequence_group": "1"}}
    assert nwp._rep_sg(d, lookup) == "1"


@pytest.mark.logic
def test_the_within_module_falls_back_to_the_best_composite(nwp, tmp_path):
    d = tmp_path / "design_0_seq_1"
    d.mkdir()
    _write_agg(d, [_agg("0", ra=30.0, tj=0.0), _agg("1", ra=2.0, tj=0.8)])
    assert nwp._rep_sg(d, {}) == "1"


@pytest.mark.logic
def test_the_within_modules_fallback_scores_on_the_final_prediction(nwp,
                                                                    tmp_path):
    """Group 0's steered pose is the better one but its reversion collapsed,
    so on final numbers group 1 wins. Scoring the steered side would pick
    the sequence group whose result did not survive reversion."""
    d = tmp_path / "design_0_seq_1"
    d.mkdir()
    collapsed = dict(_agg("0", ra=2.0, tj=0.8, outcome="pose_collapses"),
                     reverted_ra_eff_vs_truth_median="30.5",
                     reverted_true_jaccard_median="0.02")
    _write_agg(d, [collapsed, _agg("1", ra=4.0, tj=0.5)])
    assert nwp._rep_sg(d, {}) == "1"


@pytest.mark.logic
def test_a_sequence_with_no_aggregate_has_no_representative(nwp, tmp_path):
    d = tmp_path / "design_0_seq_1"
    d.mkdir()
    assert nwp._rep_sg(d, {}) is None


@pytest.mark.logic
def test_a_blank_pointer_is_not_a_pointer(nwp, tmp_path):
    """cross_summary writes an empty rep_sequence_group for tier-none, so an
    empty string has to trigger the fallback rather than select nothing."""
    d = tmp_path / "design_0_seq_1"
    d.mkdir()
    _write_agg(d, [_agg("0", ra=2.0, tj=0.8)])
    assert nwp._rep_sg(d, {"design_0_seq_1": {"rep_sequence_group": ""}}) == "0"


def _write_agg(seq_dir: Path, rows):
    fields = sorted({k for r in rows for k in r})
    with (seq_dir / "aggregated_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ── what a seed's outcome is ─────────────────────────────────────────────────

PASSING = {"steered_total_mutations": "4",
           "steered_ra_eff_vs_truth": "2.0",
           "steered_receptor_intact": "1",
           "steered_complex_plddt": "0.9",
           "steered_ipae": "3.0",
           "steered_pae_pass_frac": "0.9",
           "steered_iptm": "0.8"}


@pytest.fixture(params=["cohort", "within"])
def classify(request, npl, nwp):
    """Both modules carry their own copy of this function. They have to
    agree, because the cohort heatmap and the per-sequence trajectory plot
    are read side by side in the thesis."""
    return (npl if request.param == "cohort" else nwp).classify_seed


@pytest.mark.logic
def test_a_mutated_row_that_passes_everything_is_a_steering_pass(classify):
    assert classify(dict(PASSING)) == ("steering", "pass")


@pytest.mark.logic
def test_a_row_with_no_mutations_is_a_cold_start(classify):
    """Zero mutations means the prediction was made from the input sequence,
    so it is the baseline rather than a steering result."""
    assert classify(dict(PASSING, steered_total_mutations="0")) == (
        "cold_start", "pass")


@pytest.mark.logic
def test_a_populated_verdict_makes_it_a_reversion_row(classify):
    """The verdict only exists on rows reversion ran for, and those rows
    have to be read off the reverted columns."""
    row = dict(PASSING, reversion_verdict="pose_holds",
               reverted_ra_eff_vs_truth="2.0",
               reverted_receptor_intact="1",
               reverted_complex_plddt="0.9", reverted_ipae="3.0",
               reverted_pae_pass_frac="0.9", reverted_iptm="0.8")
    assert classify(row) == ("reversion", "pass")


@pytest.mark.logic
def test_a_reversion_row_ignores_the_steered_columns(classify):
    """The steered numbers on this row are perfect and the reverted ones
    are not. Reading the wrong family reports a collapse as a pass."""
    row = dict(PASSING, reversion_verdict="pose_collapses",
               reverted_ra_eff_vs_truth="30.5",
               reverted_receptor_intact="1",
               reverted_complex_plddt="0.9", reverted_ipae="3.0",
               reverted_pae_pass_frac="0.9", reverted_iptm="0.8")
    assert classify(row) == ("reversion", "off_target")


@pytest.mark.logic
def test_new_contamination_is_taken_from_the_verdict_not_recomputed(classify):
    """The pipeline already decided it, and the metrics on the row cannot
    express it — a contaminated pose can be geometrically perfect."""
    row = dict(PASSING, reversion_verdict="new_contamination")
    assert classify(row) == ("reversion", "new_contamination")


@pytest.mark.logic
def test_a_pose_that_misses_the_cutoff_with_good_confidence_is_off_target(
        classify):
    """Confident about a pose in the wrong place. That is the failure mode
    negative steering exists to remove."""
    assert classify(dict(PASSING, steered_ra_eff_vs_truth="30.0")) == (
        "steering", "off_target")


@pytest.mark.logic
def test_a_good_pose_with_bad_confidence_is_a_poor_prediction(classify):
    assert classify(dict(PASSING, steered_iptm="0.01")) == (
        "steering", "poor_prediction")


@pytest.mark.logic
def test_both_failing_is_reported_as_multiple_failures(classify):
    """Naming one of them would suggest fixing that one would make the seed
    pass."""
    row = dict(PASSING, steered_ra_eff_vs_truth="30.0", steered_iptm="0.01")
    assert classify(row) == ("steering", "multiple_failures")


@pytest.mark.logic
def test_a_broken_receptor_fails_the_structural_test(classify):
    """Intactness is part of the structural gate, not a separate outcome.
    A collapsed receptor can still have a small effector RMSD."""
    assert classify(dict(PASSING, steered_receptor_intact="0")) == (
        "steering", "off_target")


@pytest.mark.logic
@pytest.mark.parametrize("ra,outcome", [("4.99", "pass"),
                                        ("5.0", "off_target")])
def test_the_structural_cutoff_is_exclusive(classify, ra, outcome):
    """Every gate in this pipeline is < rather than <=, so a seed reported
    at exactly 5.0 A is a failure."""
    assert classify(dict(PASSING, steered_ra_eff_vs_truth=ra))[1] == outcome


@pytest.mark.logic
@pytest.mark.parametrize("field,value", [
    ("steered_complex_plddt", "0.70"),
    ("steered_pae_pass_frac", "0.10"),
    ("steered_iptm", "0.30"),
    ("steered_ipae", "15.0"),
])
def test_the_confidence_gates_are_inclusive(classify, field, value):
    """Unlike the structural one. A seed exactly at a confidence threshold
    passes, so the four published thresholds are attainable values."""
    assert classify(dict(PASSING, **{field: value}))[1] == "pass"


@pytest.mark.logic
def test_a_row_with_no_mutation_count_has_no_stage(classify):
    """Without it there is no way to tell a cold start from a steering
    result, and guessing either would put the row in the wrong panel."""
    row = {k: v for k, v in PASSING.items()
           if k != "steered_total_mutations"}
    assert classify(row) == ("no_data", "no_data")


@pytest.mark.logic
@pytest.mark.parametrize("field", ["steered_ra_eff_vs_truth",
                                   "steered_receptor_intact",
                                   "steered_complex_plddt",
                                   "steered_ipae",
                                   "steered_pae_pass_frac",
                                   "steered_iptm"])
def test_any_missing_metric_is_no_data_rather_than_a_verdict(classify,
                                                             field):
    """A seed scored on five of six metrics is not a pass. The stage is
    still known, so the row stays in its panel as no_data."""
    stage, outcome = classify(dict(PASSING, **{field: ""}))
    assert (stage, outcome) == ("steering", "no_data")


@pytest.mark.logic
def test_a_nan_metric_is_no_data_not_a_failure(classify):
    """collect writes nan for a prediction that never landed. nan fails
    every comparison, so without this it would be reported as a genuine
    off-target result."""
    assert classify(dict(PASSING,
                         steered_ra_eff_vs_truth="nan"))[1] == "no_data"


@pytest.mark.logic
def test_the_two_modules_use_the_same_thresholds(npl, nwp):
    """They are read side by side in the thesis, so a divergence would make
    the cohort heatmap and the per-sequence plots disagree about which
    seeds passed."""
    assert npl.SEED_RA_EFF_MAX == nwp.RA_EFF_MAX
    assert npl.SEED_PLDDT_MIN == nwp.COMPLEX_PLDDT_MIN
    assert npl.SEED_IPAE_MAX == nwp.IPAE_MAX
    assert npl.SEED_PAE_PASS_MIN == nwp.PAE_PASS_FRAC_MIN
    assert npl.SEED_IPTM_MIN == nwp.IPTM_MIN


# ── the cold-start baseline ──────────────────────────────────────────────────

@pytest.mark.logic
def test_the_blank_sequence_group_row_becomes_the_cold_start_baseline(
        nwp, tmp_path):
    """The cycle-0 initial prediction is one prediction shared across every
    sequence group, so it has no sequence group and no seed index. The
    trajectory plot draws steering as movement away from it, which needs it
    captured rather than skipped."""
    runs = tmp_path / "runs"
    d = runs / "design_0_seq_1"
    d.mkdir(parents=True)
    _write_agg(d, [_agg("0", ra=2.0, tj=0.8)])
    rows = [dict(PASSING, sequence_group="", seed_index="",
                 steered_total_mutations="0",
                 steered_ra_eff_vs_truth="41.2", design="initial"),
            dict(PASSING, sequence_group="0", seed_index="0",
                 design="design_0_seq_1")]
    _write_raw(d, rows)
    cross = tmp_path / "cross.csv"
    _write_cross(cross, ["design_0_seq_1"])

    rep, allс, composites, tiers, cold = nwp.load_cohort_seeds(runs, cross)
    assert cold["design_0_seq_1"] == pytest.approx(41.2)
    assert len(allс["design_0_seq_1"]) == 1, "the baseline row was plotted"


@pytest.mark.logic
def test_a_control_sequence_is_not_loaded_as_a_design(nwp, tmp_path):
    """Controls have no design region and no steering history, so a panel
    per control would be a panel of one point with nothing to compare."""
    runs = tmp_path / "runs"
    for name in ("design_0_seq_1", "input_control_polyA"):
        d = runs / name
        d.mkdir(parents=True)
        _write_agg(d, [_agg("0", ra=2.0, tj=0.8)])
        _write_raw(d, [dict(PASSING, sequence_group="0", seed_index="0",
                            design=name)])
    cross = tmp_path / "cross.csv"
    _write_cross(cross, ["design_0_seq_1", "input_control_polyA"])

    rep, all_, composites, tiers, cold = nwp.load_cohort_seeds(runs, cross)
    assert set(all_) == {"design_0_seq_1"}


@pytest.mark.logic
def test_a_sequence_with_no_per_seed_table_is_skipped(nwp, tmp_path):
    runs = tmp_path / "runs"
    d = runs / "design_0_seq_1"
    d.mkdir(parents=True)
    _write_agg(d, [_agg("0", ra=2.0, tj=0.8)])
    cross = tmp_path / "cross.csv"
    _write_cross(cross, ["design_0_seq_1"])
    rep, all_, composites, tiers, cold = nwp.load_cohort_seeds(runs, cross)
    assert all_ == {}


@pytest.mark.logic
def test_an_unparseable_seed_index_becomes_zero(nwp, tmp_path):
    """Rather than dropping the seed. The index orders the x axis; it is not
    the measurement."""
    runs = tmp_path / "runs"
    d = runs / "design_0_seq_1"
    d.mkdir(parents=True)
    _write_agg(d, [_agg("0", ra=2.0, tj=0.8)])
    _write_raw(d, [dict(PASSING, sequence_group="0", seed_index="first",
                        design="design_0_seq_1")])
    cross = tmp_path / "cross.csv"
    _write_cross(cross, ["design_0_seq_1"])
    rep, all_, composites, tiers, cold = nwp.load_cohort_seeds(runs, cross)
    assert all_["design_0_seq_1"][0]["seed"] == 0


def _write_raw(seq_dir: Path, rows):
    fields = sorted({k for r in rows for k in r})
    with (seq_dir / "raw_per_seed_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_cross(path: Path, names):
    fields = ["mpnn_sequence", "cross_tier", "rep_sequence_group",
              "cross_composite_score", "cross_rank_by_composite"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for n in names:
            w.writerow({"mpnn_sequence": n, "cross_tier": "none",
                        "rep_sequence_group": "", "cross_composite_score": "",
                        "cross_rank_by_composite": ""})


# ── the coercion helpers ─────────────────────────────────────────────────────
#
# Both modules read every number out of a CSV, so every metric arrives as a
# string and every missing metric arrives as "". These two functions are the
# only thing standing between that and a matplotlib call.

@pytest.fixture(params=["cohort", "within"])
def mod(request, npl, nwp):
    return npl if request.param == "cohort" else nwp


@pytest.mark.logic
@pytest.mark.parametrize("value,want", [
    ("2.5", 2.5), (" 2.5 ", 2.5), (2.5, 2.5), (3, 3.0), ("-1", -1.0),
    ("", None), ("   ", None), (None, None), ("n/a", None),
])
def test_floats_are_parsed_or_rejected(mod, value, want):
    assert mod._try_float(value) == want


@pytest.mark.logic
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", float("nan"),
                                   float("inf")])
def test_nan_and_infinity_are_rejected_as_if_missing(mod, value):
    """collect writes nan for a prediction that never landed, and plotting
    it would draw a point at an arbitrary place or silently drop it from an
    axis range. Treating it as missing is what makes the figures honest."""
    assert mod._try_float(value) is None


@pytest.mark.logic
@pytest.mark.parametrize("value,want", [
    ("4", 4), (" 4 ", 4), ("4.0", 4), ("4.7", 4),
    ("", None), (None, None), ("four", None),
])
def test_integers_are_parsed_or_rejected(mod, value, want):
    """A float string has to parse, because the intact flag is written as
    1 by one stage and 1.0 by another."""
    assert mod._try_int(value) == want


# ── the compact labels ───────────────────────────────────────────────────────

@pytest.mark.logic
@pytest.mark.parametrize("name,want", [
    ("design_0_seq_1", "d0_s1"),
    ("design_12_seq_3", "d12_s3"),
    ("input_control_polyA", "ctrl_poly"),
    ("input_control_scrambled", "ctrl_scra"),
    ("something_else", "something_else"),
])
def test_names_are_shortened_for_the_axis(mod, name, want):
    assert mod._short_name(name) == want


@pytest.mark.logic
@pytest.mark.parametrize("name,want", [
    ("design_0_seq_1", "d0"), ("design_7_seq_12", "d7"),
    ("input_control_polyA", "input_control_polyA"),
])
def test_the_design_id_groups_the_sequences_of_one_design(nwp, name, want):
    """The trajectory plot draws one line per design across its sequences,
    so this is the grouping key."""
    assert nwp._design_id(name) == want


@pytest.mark.logic
@pytest.mark.parametrize("name,want", [
    ("design_0_seq_3", 3), ("design_0_seq_0", 0), ("not_a_design", 0),
])
def test_the_sequence_number_is_the_colour_key(nwp, name, want):
    """Colour is assigned by sequence number so one sequence keeps its
    colour across every design's panel."""
    assert nwp._seq_num(name) == want


@pytest.mark.logic
@pytest.mark.parametrize("n,shape", [
    (0, (1, 1)), (1, (1, 1)), (3, (1, 3)),
    (4, (2, 2)), (9, (3, 3)), (12, (3, 4)), (20, (5, 4)),
])
def test_the_grid_stays_within_four_columns(nwp, n, shape):
    """Past four columns the per-panel axes are too narrow to read at
    thesis figure width."""
    assert nwp._grid_shape(n) == shape


@pytest.mark.logic
def test_every_panel_has_a_cell(nwp):
    for n in range(1, 40):
        rows, cols = nwp._grid_shape(n)
        assert rows * cols >= n, f"{n} panels do not fit in {rows}x{cols}"


# ── the empty-data path ──────────────────────────────────────────────────────

@pytest.mark.logic
def test_an_empty_figure_is_still_a_figure(mod, tmp_path):
    """Nextflow publishes a fixed list of figure paths, so a figure that is
    not written fails the stage rather than reporting that there was
    nothing to draw."""
    out = tmp_path / "empty.png"
    mod._make_empty_plot("no data", str(out))
    assert out.is_file() and out.stat().st_size > 0


# ── the CLI's own input handling ─────────────────────────────────────────────

@pytest.mark.logic
def test_a_missing_csv_argument_is_refused(npl):
    """Nextflow injects --csv, so an absent one means the channel was empty
    and the figures would be drawn from nothing."""
    with pytest.raises(SystemExit, match="--csv is required"):
        npl._resolve_csv_path(None)


@pytest.mark.logic
def test_a_csv_path_that_does_not_exist_is_refused(npl, tmp_path):
    with pytest.raises(SystemExit, match="does not exist"):
        npl._resolve_csv_path(str(tmp_path / "nope.csv"))


@pytest.mark.logic
@needs_fixture
def test_an_existing_csv_resolves(npl):
    assert npl._resolve_csv_path(str(COHORT_CSV)) == COHORT_CSV


@pytest.mark.logic
def test_the_design_region_file_is_parsed(npl, tmp_path):
    f = tmp_path / "region.txt"
    f.write_text("# 1-based positions\n1,2,3\n30,31\n")
    positions, length = npl._load_input_design_region(str(f))
    assert positions == {1, 2, 3, 30, 31}


@pytest.mark.logic
def test_the_second_return_is_the_highest_position_not_the_length(npl,
                                                                  tmp_path):
    """The file carries no record of the receptor length, so the highest
    design-region position is a lower bound on it. main() uses it as the
    mutation-impact x-axis extent only when --pdb-length was not given, so a
    run that wants the whole receptor on that axis has to pass --pdb-length.
    The docstring used to claim this returned None, which is why it is
    pinned here."""
    f = tmp_path / "region.txt"
    f.write_text("1,2,3,30,31\n")
    positions, highest = npl._load_input_design_region(str(f))
    assert positions == {1, 2, 3, 30, 31}
    assert highest == 31


@pytest.mark.logic
def test_a_file_that_parses_to_no_positions_is_nothing(npl, tmp_path,
                                                        capsys):
    """A file of only comments shades nothing and sets no axis extent,
    rather than setting an extent of zero."""
    f = tmp_path / "region.txt"
    f.write_text("# nothing here\n\n")
    assert npl._load_input_design_region(str(f)) == (None, None)
    assert "no positions parsed" in capsys.readouterr().out


@pytest.mark.logic
def test_no_design_region_argument_shades_nothing(npl):
    assert npl._load_input_design_region(None) == (None, None)
    assert npl._load_input_design_region("") == (None, None)


@pytest.mark.logic
def test_a_missing_design_region_file_warns_and_shades_nothing(npl, tmp_path,
                                                               capsys):
    """The mutation-impact figure is still worth drawing without the
    shading, so this degrades rather than failing."""
    positions, length = npl._load_input_design_region(
        str(tmp_path / "gone.txt"))
    assert (positions, length) == (None, None)
    assert "path not found" in capsys.readouterr().out


@pytest.mark.logic
def test_an_unreadable_design_region_file_warns_and_shades_nothing(
        npl, tmp_path, capsys):
    d = tmp_path / "region.txt"
    d.mkdir()
    positions, length = npl._load_input_design_region(str(d))
    assert (positions, length) == (None, None)
    assert "could not read" in capsys.readouterr().out


# ── the mutation position parser ─────────────────────────────────────────────

@pytest.mark.logic
@pytest.mark.parametrize("spec,want", [
    ("/A:5,7,22", [5, 7, 22]),
    ("/A:5", [5]),
    ("  /A:5, 7 ", [5, 7]),
    ("", []),
    ("5,7", []),                 # no chain prefix, so no colon to split on
    ("/A:", []),
    ("/A:5,x,7", [5, 7]),
])
def test_chimerax_specs_are_parsed(npl, spec, want):
    """The mutation-impact figure counts how often each receptor position
    was mutated, and this string is where those positions come from."""
    assert npl._parse_chimerax_positions(spec) == want


# ── building the cohort ──────────────────────────────────────────────────────
#
# load_unified_cohort merges two sources. cross_sequence_summary.csv has the
# tier and the cohort rank but leaves the rep_* metric columns blank for
# tier-none sequences, because those are computed after the passing filter and
# a tier-none sequence passed nothing. Each sequence's own
# aggregated_results.csv has the metrics regardless.
#
# Four cases fall out of that, and each one decides whether a sequence appears
# in the figures at all.

def _cross(path: Path, rows):
    fields = sorted({k for r in rows for k in r} | {"mpnn_sequence",
                                                    "cross_tier"})
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _seq(runs: Path, name: str, agg_rows):
    d = runs / name
    d.mkdir(parents=True, exist_ok=True)
    _write_agg(d, agg_rows)
    return d


@pytest.mark.logic
def test_without_a_runs_directory_the_cross_summary_is_used_as_is(npl,
                                                                   tmp_path,
                                                                   capsys):
    """There is nothing to backfill from, so tier-none sequences keep their
    blank metric columns and drop out of the figures that read them. The
    message says so rather than the figures silently thinning."""
    cross = tmp_path / "cross.csv"
    _cross(cross, [{"mpnn_sequence": "design_0_seq_1", "cross_tier": "none",
                    "rep_n_pass": ""}])
    rows = npl.load_unified_cohort(cross, None, verbose=True)
    assert [r["mpnn_sequence"] for r in rows] == ["design_0_seq_1"]
    out = capsys.readouterr().out
    assert "backfill skipped" in out
    assert "blank rep_n_pass: 1" in out


@pytest.mark.logic
def test_a_runs_directory_that_does_not_exist_is_the_same_as_none(npl,
                                                                   tmp_path):
    cross = tmp_path / "cross.csv"
    _cross(cross, [{"mpnn_sequence": "design_0_seq_1", "cross_tier": "A"}])
    rows = npl.load_unified_cohort(cross, tmp_path / "gone", verbose=False)
    assert len(rows) == 1


@pytest.mark.logic
def test_the_tier_breakdown_is_reported(npl, tmp_path, capsys):
    """It is how a run with no tiered sequences at all is noticed, which
    otherwise looks the same as a run whose figures came out empty."""
    cross = tmp_path / "cross.csv"
    _cross(cross, [{"mpnn_sequence": "a", "cross_tier": "A"},
                   {"mpnn_sequence": "b", "cross_tier": "B"},
                   {"mpnn_sequence": "c", "cross_tier": "none"}])
    npl.load_unified_cohort(cross, None, verbose=True)
    out = capsys.readouterr().out
    assert "A=1 B=1 C=0 none=1" in out


@pytest.mark.logic
def test_the_metrics_are_re_sourced_from_the_final_prediction(npl, tmp_path):
    """cross_summary computes its rep_*_median columns from the steered side
    whether or not reversion ran, so a pose_collapses sequence arrives
    carrying its steered intermediate. Overwriting with the final value is
    what stops a collapsed reversion being plotted as a 3 A success."""
    runs = tmp_path / "runs"
    _seq(runs, "design_0_seq_1", [dict(
        _agg("0", ra=3.013, tj=0.7, outcome="pose_collapses"),
        reverted_ra_eff_vs_truth_median="30.5",
        reverted_true_jaccard_median="0.02")])
    cross = tmp_path / "cross.csv"
    _cross(cross, [{"mpnn_sequence": "design_0_seq_1", "cross_tier": "C",
                    "rep_sequence_group": "0", "rep_cycle": "0",
                    "rep_pathway": "",
                    "rep_ra_eff_vs_truth_median": "3.013",
                    "rep_true_jaccard_median": "0.7"}])
    rows = npl.load_unified_cohort(cross, runs, verbose=False)
    assert rows[0]["rep_ra_eff_vs_truth_median"] == "30.5"
    assert rows[0]["rep_true_jaccard_median"] == "0.02"


@pytest.mark.logic
def test_the_cross_summary_tier_and_rank_survive_the_backfill(npl, tmp_path):
    """Only the metric columns are re-sourced. The tier and the cohort rank
    are cross_summary's to decide, and recomputing them here would let the
    figures disagree with the ranking table."""
    runs = tmp_path / "runs"
    _seq(runs, "design_0_seq_1", [_agg("0", ra=2.0, tj=0.8)])
    cross = tmp_path / "cross.csv"
    _cross(cross, [{"mpnn_sequence": "design_0_seq_1", "cross_tier": "A",
                    "rep_sequence_group": "0", "rep_cycle": "0",
                    "rep_pathway": "", "cross_rank_by_composite": "1",
                    "cross_composite_score": "0.70"}])
    rows = npl.load_unified_cohort(cross, runs, verbose=False)
    assert rows[0]["cross_tier"] == "A"
    assert rows[0]["cross_rank_by_composite"] == "1"
    assert rows[0]["cross_composite_score"] == "0.70"


@pytest.mark.logic
def test_a_populated_verdict_count_is_not_overwritten(npl, tmp_path):
    """n_seeds, n_pass and the mutation strings have no reverted equivalent,
    so cross_summary's value is the authoritative one when it has one."""
    runs = tmp_path / "runs"
    _seq(runs, "design_0_seq_1", [dict(_agg("0", ra=2.0, tj=0.8),
                                       n_seeds="4", n_pass="1")])
    cross = tmp_path / "cross.csv"
    _cross(cross, [{"mpnn_sequence": "design_0_seq_1", "cross_tier": "A",
                    "rep_sequence_group": "0", "rep_cycle": "0",
                    "rep_pathway": "", "rep_n_seeds": "4", "rep_n_pass": "3"}])
    rows = npl.load_unified_cohort(cross, runs, verbose=False)
    assert rows[0]["rep_n_pass"] == "3"


@pytest.mark.logic
def test_a_blank_verdict_count_is_filled_from_the_aggregate(npl, tmp_path):
    """Which is the whole reason the backfill exists: a tier-none sequence
    has blank rep_* columns and would otherwise be dropped by every figure
    that reads a seed count."""
    runs = tmp_path / "runs"
    _seq(runs, "design_0_seq_1", [dict(_agg("0", ra=2.0, tj=0.8),
                                       n_seeds="4", n_pass="1")])
    cross = tmp_path / "cross.csv"
    _cross(cross, [{"mpnn_sequence": "design_0_seq_1", "cross_tier": "none",
                    "rep_sequence_group": "", "rep_n_seeds": "",
                    "rep_n_pass": ""}])
    rows = npl.load_unified_cohort(cross, runs, verbose=False)
    assert rows[0]["rep_n_pass"] == "1"
    assert rows[0]["rep_n_seeds"] == "4"


@pytest.mark.logic
def test_a_sequence_only_in_runs_is_synthesised(npl, tmp_path, capsys):
    """Older workdirs and partial runs have sequences cross_summary never
    saw. Dropping them would make a partial run look like a smaller one."""
    runs = tmp_path / "runs"
    _seq(runs, "design_9_seq_9", [_agg("0", ra=2.0, tj=0.8)])
    cross = tmp_path / "cross.csv"
    _cross(cross, [])
    rows = npl.load_unified_cohort(cross, runs, verbose=True)
    assert [r["mpnn_sequence"] for r in rows] == ["design_9_seq_9"]
    assert rows[0]["cross_tier"] == "none"
    assert "synthesised): 1" in capsys.readouterr().out


@pytest.mark.logic
@pytest.mark.parametrize("name,row_type", [
    ("input_control_polyA", "control_polyA"),
    ("input_control_scrambled", "control_scrambled"),
    ("design_0_seq_1", "steered"),
])
def test_a_synthesised_row_keeps_its_control_identity(npl, tmp_path, name,
                                                       row_type):
    """The controls are excluded from the cohort ranking and drawn in their
    own figure, so a control synthesised as a steered row would enter the
    tier landscape as a design."""
    runs = tmp_path / "runs"
    _seq(runs, name, [_agg("0", ra=2.0, tj=0.8)])
    cross = tmp_path / "cross.csv"
    _cross(cross, [])
    rows = npl.load_unified_cohort(cross, runs, verbose=False)
    assert rows[0]["row_type"] == row_type


@pytest.mark.logic
def test_a_runs_dir_with_no_aggregate_is_warned_and_skipped(npl, tmp_path,
                                                             capsys):
    runs = tmp_path / "runs"
    (runs / "design_0_seq_1").mkdir(parents=True)
    cross = tmp_path / "cross.csv"
    _cross(cross, [])
    rows = npl.load_unified_cohort(cross, runs, verbose=True)
    assert rows == []
    assert "no aggregated_results.csv" in capsys.readouterr().out


@pytest.mark.logic
def test_an_aggregate_with_no_usable_row_is_skipped(npl, tmp_path):
    """An empty aggregated_results.csv means the sequence has no numbers at
    all, so there is nothing to synthesise a row from."""
    runs = tmp_path / "runs"
    _seq(runs, "design_0_seq_1", [])
    cross = tmp_path / "cross.csv"
    _cross(cross, [])
    assert npl.load_unified_cohort(cross, runs, verbose=False) == []


@pytest.mark.logic
def test_a_loose_file_in_the_runs_directory_is_ignored(npl, tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "README.txt").write_text("not a sequence\n")
    cross = tmp_path / "cross.csv"
    _cross(cross, [])
    assert npl.load_unified_cohort(cross, runs, verbose=False) == []


@pytest.mark.logic
def test_a_sequence_only_in_cross_summary_is_passed_through(npl, tmp_path,
                                                             capsys):
    """Its run directory was not synced, or was cleaned. The tier and rank
    are still real, so the sequence stays in the ranking figures."""
    runs = tmp_path / "runs"
    _seq(runs, "design_0_seq_1", [_agg("0", ra=2.0, tj=0.8)])
    cross = tmp_path / "cross.csv"
    _cross(cross, [{"mpnn_sequence": "design_0_seq_1", "cross_tier": "A",
                    "rep_sequence_group": "0", "rep_cycle": "0",
                    "rep_pathway": ""},
                   {"mpnn_sequence": "design_5_seq_2", "cross_tier": "B"}])
    rows = npl.load_unified_cohort(cross, runs, verbose=True)
    assert {r["mpnn_sequence"] for r in rows} == {"design_0_seq_1",
                                                  "design_5_seq_2"}
    out = capsys.readouterr().out
    assert "no matching runs/ dir" in out
    assert "design_5_seq_2" in out


@pytest.mark.logic
def test_a_row_with_no_sequence_name_is_dropped(npl, tmp_path):
    """Nothing can be merged against it and nothing can label it on a plot."""
    cross = tmp_path / "cross.csv"
    _cross(cross, [{"mpnn_sequence": "", "cross_tier": "A"},
                   {"mpnn_sequence": "design_0_seq_1", "cross_tier": "A"}])
    rows = npl.load_unified_cohort(cross, None, verbose=False)
    assert [r["mpnn_sequence"] for r in rows] == ["design_0_seq_1"]


@pytest.mark.logic
def test_every_sequence_appears_exactly_once(npl, tmp_path):
    """A sequence counted twice would be two bars in the tier landscape and
    two points in every scatter."""
    runs = tmp_path / "runs"
    for n in ("design_0_seq_1", "design_0_seq_2"):
        _seq(runs, n, [_agg("0", ra=2.0, tj=0.8)])
    cross = tmp_path / "cross.csv"
    _cross(cross, [{"mpnn_sequence": "design_0_seq_1", "cross_tier": "A",
                    "rep_sequence_group": "0", "rep_cycle": "0",
                    "rep_pathway": ""},
                   {"mpnn_sequence": "design_7_seq_0", "cross_tier": "B"}])
    names = [r["mpnn_sequence"]
             for r in npl.load_unified_cohort(cross, runs, verbose=False)]
    assert sorted(names) == ["design_0_seq_1", "design_0_seq_2",
                             "design_7_seq_0"]
    assert len(names) == len(set(names))
