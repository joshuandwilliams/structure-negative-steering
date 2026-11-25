"""The degenerate-data guards inside the figure functions.

test_negsteer_plots.py renders every figure from the four-run cohort fixture
and test_plot_data_policy.py pins which number each figure draws. Neither
reaches the guards, because the fixture is a healthy cohort. What is left is
the behaviour on data that is partly missing, and that is the data a real run
produces. A sequence whose reversion prediction never landed has blank metric
columns. A cohort run before the controls were added has no control rows. A
run stopped after one cycle has one stage populated and two empty.

Each guard decides whether something appears in a published figure or
disappears from it silently. The skip inside plot_ra_eff_vs_jaccard is the
sharpest case. A sequence with a blank Jaccard is dropped from the scatter and
from the RMSD histogram below it, so a design that was predicted and failed
looks like a design that was never attempted.

The figure functions are called directly rather than through main, which is
what allows one category to be emptied at a time. Where a row has to vanish
from a figure, the test renders the same figure with the row absent and
compares the bytes. Two identical PNGs is the only available proof that the
row contributed nothing.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from unittest import mock

import matplotlib
import pytest
from test_plot_data_policy import PASSING, _agg, _cross, _seq, _write_raw

matplotlib.use("Agg")

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin"))

# A PNG smaller than this is an axes frame with nothing drawn on it. Same
# constant as test_negsteer_plots.py, for the same reason: every empty-data
# path still writes a file, so existence proves nothing and size does.
MIN_PNG_BYTES = 5_000


@pytest.fixture(scope="module")
def npl():
    import negsteer_plots
    return negsteer_plots


@pytest.fixture(scope="module")
def nwp():
    import negsteer_within_sequence_plots
    return negsteer_within_sequence_plots


def _plot_row(name="design_0_seq_1", *, tier="A", ra=2.0, tj=0.8,
              row_type="steered", **extra):
    """One cross-summary-shaped row, the shape every plot_* takes.

    ra or tj passed as None writes the blank string, which is what
    cross_summary writes for a median it could not compute.
    """
    row = {
        "mpnn_sequence": name,
        "row_type": row_type,
        "cross_tier": tier,
        "cross_rank_by_composite": "",
        "cross_composite_score": "",
        "rep_ra_eff_vs_truth_median": "" if ra is None else str(ra),
        "rep_true_jaccard_median": "" if tj is None else str(tj),
    }
    row.update(extra)
    return row


def _seed_rec(seed=0, sg="0", stage="steering", ra=2.0, tj=0.8, wj=0.8,
              **extra):
    """One per-seed record, the shape load_cohort_seeds hands the figures."""
    rec = {"seed": seed, "sg": sg, "stage": stage,
           "ra_final": ra, "tj_final": tj, "wj_final": wj,
           "steered_ra_raw": ra, "reverted_ra_raw": None}
    rec.update(extra)
    return rec


THRESHOLDS = {"complex_plddt_min": 0.7, "ipae_max": 15.0,
              "pae_pass_frac_min": 0.1, "iptm_min": 0.3,
              "control_ra_eff_min": 10.0}


def _assert_placeholder(npl, path: Path, message: str) -> None:
    """Assert `path` is the empty-data placeholder carrying `message`.

    Size is not enough. _make_empty_plot writes about 8 kB, which is over
    the MIN_PNG_BYTES floor the figure tests use, so a placeholder would pass
    a size check for a drawn figure. Rendering the placeholder here and
    comparing bytes also pins each message, which is what a reader of the
    published figure set sees when a figure came out empty.
    """
    expected = path.parent / "_expected_placeholder.png"
    npl._make_empty_plot(message, str(expected))
    assert path.read_bytes() == expected.read_bytes(), \
        f"{path.name} is not the '{message}' placeholder"


# ── the scatter's silent skip ────────────────────────────────────────────────
#
# plot_ra_eff_vs_jaccard is the cohort's headline figure. It takes one point
# per sequence from rep_ra_eff_vs_truth_median and rep_true_jaccard_median,
# and it drops any row where either is missing. Both columns are blank for a
# sequence whose final prediction never landed, so the drop is not rare.
#
# Nothing on the figure records how many rows were dropped. The histogram
# panel below the scatter is built from the same filtered points, so a dropped
# sequence is absent from the distribution as well.

@pytest.mark.logic
@pytest.mark.parametrize("ra,tj", [(2.0, None), (None, 0.8), (None, None)])
def test_a_sequence_missing_either_metric_leaves_the_scatter_entirely(
        npl, tmp_path, ra, tj):
    """A blank median is written whenever the final prediction produced no
    structure to measure. The sequence is then absent from the scatter and
    from the RMSD histogram, so a design that failed is indistinguishable
    from one never attempted. Here it is the only row, so the whole figure
    falls back to the empty-data message."""
    out = tmp_path / "scatter.png"
    assert npl.plot_ra_eff_vs_jaccard(
        [_plot_row(ra=ra, tj=tj)], str(out)) is False
    _assert_placeholder(npl, out, "No RMSD/jaccard data available")


@pytest.mark.logic
def test_the_dropped_row_changes_nothing_about_the_figure(npl, tmp_path):
    """The proof that the skip is total. One good row plus one row with a
    blank Jaccard renders byte for byte the same figure as the good row
    alone, including the histogram's per-tier counts in the legend."""
    good = _plot_row("design_0_seq_1", tier="A", ra=2.0, tj=0.8)
    blank = _plot_row("design_0_seq_2", tier="A", ra=4.0, tj=None)

    one = tmp_path / "one.png"
    two = tmp_path / "two.png"
    assert npl.plot_ra_eff_vs_jaccard([good], str(one)) is True
    assert npl.plot_ra_eff_vs_jaccard([good, blank], str(two)) is True
    assert one.read_bytes() == two.read_bytes(), \
        "the row with a blank Jaccard left a mark on the figure"


@pytest.mark.logic
def test_the_controls_are_drawn_in_the_histogram_only(npl, tmp_path):
    """polyA and scrambled are excluded from the cross-sequence ranking by
    design, so they have no tier and cannot be a tier-coloured dot. They
    carry the RMSD a wrong sequence produces, which is the reference the
    histogram panel exists to show."""
    rows = [_plot_row("design_0_seq_1", tier="A", ra=2.0, tj=0.8),
            _plot_row("input_control_scrambled", tier="none", ra=28.0,
                      tj=0.01, row_type="control_scrambled"),
            _plot_row("input_control_polyA", tier="none", ra=31.0, tj=0.0,
                      row_type="control_polyA")]
    out = tmp_path / "scatter.png"
    assert npl.plot_ra_eff_vs_jaccard(rows, str(out)) is True
    assert out.stat().st_size > MIN_PNG_BYTES


@pytest.mark.logic
def test_a_cohort_of_only_controls_gets_a_fixed_ten_angstrom_axis(npl,
                                                                   tmp_path):
    """No tiered and no tier-none sequence means no steered point to scale
    the scatter from. The axis falls back to 0-10 A rather than collapsing,
    which keeps the iso-composite contours drawable."""
    rows = [_plot_row("input_control_polyA", tier="none", ra=31.0, tj=0.0,
                      row_type="control_polyA")]
    out = tmp_path / "scatter.png"
    assert npl.plot_ra_eff_vs_jaccard(rows, str(out)) is True
    assert out.stat().st_size > MIN_PNG_BYTES


@pytest.mark.logic
def test_a_cohort_with_no_tiered_sequence_scales_to_its_own_worst_rmsd(
        npl, tmp_path):
    """A run where nothing passed has only tier-none points. They set the
    axis themselves, instead of being held off-scale by the 1.5x rule that
    applies when tiered sequences exist."""
    rows = [_plot_row("design_0_seq_1", tier="none", ra=12.0, tj=0.2)]
    out = tmp_path / "scatter.png"
    assert npl.plot_ra_eff_vs_jaccard(rows, str(out)) is True
    assert out.stat().st_size > MIN_PNG_BYTES


@pytest.mark.logic
def test_a_tier_label_outside_a_b_c_none_is_counted_but_never_drawn(npl,
                                                                     tmp_path):
    """The scatter and the histogram both enumerate the four known tiers, so
    a fifth label passes the not-empty check and then appears in neither
    panel. The figure is drawn with an axis-off histogram reading "no data"
    while the row was counted as data. Pinned as the current behaviour."""
    out_d = tmp_path / "tier_d.png"
    out_none = tmp_path / "tier_none.png"
    assert npl.plot_ra_eff_vs_jaccard(
        [_plot_row(tier="D", ra=3.0, tj=0.6)], str(out_d)) is True
    assert npl.plot_ra_eff_vs_jaccard(
        [_plot_row(tier="none", ra=3.0, tj=0.6)], str(out_none)) is True
    assert out_d.stat().st_size < out_none.stat().st_size, \
        "the unknown tier drew something after all"


# ── the composite nobody can compute ─────────────────────────────────────────

@pytest.mark.logic
@pytest.mark.parametrize("ra,tj", [(2.0, None), (None, 0.8)])
def test_a_half_measured_sequence_has_no_composite(npl, ra, tj):
    """The composite is Jaccard minus a twentieth of the RMSD. Substituting
    zero for the missing half would score a sequence that was never measured
    against sequences that were."""
    assert npl._composite_from_row(_plot_row(ra=ra, tj=tj)) is None


# ── the tier landscape ───────────────────────────────────────────────────────

@pytest.mark.logic
def test_a_control_is_not_a_bar_in_the_tier_landscape(npl, tmp_path):
    """Controls are excluded from the cross-ranking, so a control bar would
    sit in the ranking with no rank. A cohort of only controls therefore has
    nothing to rank and says so."""
    rows = [_plot_row("input_control_polyA", tier="none", ra=31.0, tj=0.0,
                      row_type="control_polyA"),
            _plot_row("input_control_scrambled", tier="none", ra=28.0,
                      tj=0.01, row_type="control_scrambled")]
    out = tmp_path / "tiers.png"
    assert npl.plot_tier_landscape(rows, str(out)) is False
    _assert_placeholder(npl, out, "No steered rows in cross-sequence summary")


@pytest.mark.logic
def test_a_sequence_with_no_composite_is_marked_rather_than_drawn_at_zero(
        npl, tmp_path):
    """A bar of height zero is a real reading. It means Jaccard exactly
    equal to a twentieth of the RMSD. An unmeasurable sequence gets a cross
    at zero so the two cannot be read as the same result."""
    rows = [_plot_row("design_0_seq_1", tier="A", ra=2.0, tj=0.8),
            _plot_row("design_0_seq_2", tier="none", ra=None, tj=None)]
    out = tmp_path / "tiers.png"
    assert npl.plot_tier_landscape(rows, str(out)) is True
    assert out.stat().st_size > MIN_PNG_BYTES


# ── the filter cascade ───────────────────────────────────────────────────────

@pytest.mark.logic
def test_the_cascade_needs_a_steered_sequence(npl, tmp_path):
    """The cascade counts how many designs survive each confidence filter.
    Controls are never candidates, so a cohort of controls has no cascade
    rather than a cascade of zeros."""
    rows = [_plot_row("input_control_polyA", tier="none", ra=31.0, tj=0.0,
                      row_type="control_polyA")]
    out = tmp_path / "cascade.png"
    assert npl.plot_filter_cascade(rows, THRESHOLDS, str(out)) is False
    _assert_placeholder(npl, out, "No steered rows available")


# ── the controls diagnostic ──────────────────────────────────────────────────
#
# This is the figure the controls exist for. It puts the steered distribution
# beside the scrambled and polyA distributions, and the pipeline is
# mis-calibrated if they overlap. The controls reach it through row_type,
# which is the one figure that reads a control row as data rather than
# skipping it.

@pytest.mark.logic
def test_both_controls_get_their_own_boxes(npl, tmp_path):
    """polyA and scrambled are separate groups, not one control group. They
    fail differently. polyA has no interface at all, while scrambled has a
    plausible one in the wrong place."""
    rows = [_plot_row("design_0_seq_1", tier="A", ra=2.0, tj=0.8),
            _plot_row("design_0_seq_2", tier="none", ra=22.0, tj=0.05),
            _plot_row("input_control_scrambled", tier="none", ra=28.0,
                      tj=0.01, row_type="control_scrambled"),
            _plot_row("input_control_polyA", tier="none", ra=31.0, tj=0.0,
                      row_type="control_polyA")]
    out = tmp_path / "controls.png"
    assert npl.plot_controls_diagnostic(rows, THRESHOLDS, str(out)) is True
    assert out.stat().st_size > MIN_PNG_BYTES


@pytest.mark.logic
def test_a_control_keeps_the_half_of_its_row_that_was_measured(npl, tmp_path):
    """RMSD and Jaccard are collected independently, and the two panels are
    independent. A control with an RMSD but no Jaccard belongs in the RMSD
    panel rather than being dropped from both."""
    rows = [_plot_row("input_control_polyA", tier="none", ra=31.0, tj=None,
                      row_type="control_polyA")]
    out = tmp_path / "controls.png"
    assert npl.plot_controls_diagnostic(rows, THRESHOLDS, str(out)) is True
    assert out.stat().st_size > MIN_PNG_BYTES


@pytest.mark.logic
def test_a_cohort_with_no_numbers_has_no_controls_diagnostic(npl, tmp_path):
    """Every group empty means the boxplot would be four [nan] placeholders
    with a threshold band across them, which reads as four measured groups
    that all sit at zero."""
    rows = [_plot_row("design_0_seq_1", tier="A", ra=None, tj=None),
            _plot_row("input_control_polyA", tier="none", ra=None, tj=None,
                      row_type="control_polyA")]
    out = tmp_path / "controls.png"
    assert npl.plot_controls_diagnostic(rows, THRESHOLDS, str(out)) is False
    _assert_placeholder(npl, out, "No data for controls diagnostic")


# ── the mutation-impact figure ───────────────────────────────────────────────

@pytest.mark.logic
def test_a_control_contributes_no_mutated_position(npl, tmp_path):
    """Controls are not steered, so they have no mutation set. Counting
    polyA's substitutions as mutations at every position would swamp the
    per-position medians."""
    rows = [_plot_row("input_control_polyA", tier="none", ra=31.0, tj=0.0,
                      row_type="control_polyA",
                      rep_steered_mutations_chimerax="/A:5,7,22")]
    out = tmp_path / "mut.png"
    assert npl.plot_mutation_impact(rows, None, None, str(out)) is False
    _assert_placeholder(npl, out, "No steered_mutations data found in cross-summary")


@pytest.mark.logic
def test_a_sequence_with_no_composite_contributes_no_mutated_position(
        npl, tmp_path):
    """The figure plots the median composite of the sequences mutating each
    position. A sequence with no composite has nothing to contribute to that
    median, and its positions must not inflate the n= counts either."""
    rows = [_plot_row(ra=None, tj=None,
                      rep_steered_mutations_chimerax="/A:5,7,22")]
    out = tmp_path / "mut.png"
    assert npl.plot_mutation_impact(rows, None, None, str(out)) is False
    _assert_placeholder(npl, out, "No steered_mutations data found in cross-summary")


@pytest.mark.logic
def test_a_negative_median_composite_is_labelled_below_its_bar(npl, tmp_path):
    """Composite goes negative whenever the pose error costs more than the
    interface recovers. 0.1 - 0.05*30 is -1.4, and the n= count has to sit
    under a downward bar rather than inside it."""
    rows = [_plot_row(ra=30.0, tj=0.1,
                      rep_steered_mutations_chimerax="/A:10")]
    out = tmp_path / "mut.png"
    assert npl._composite_from_row(rows[0]) == pytest.approx(-1.4)
    assert npl.plot_mutation_impact(rows, None, 20, str(out)) is True
    assert out.stat().st_size > MIN_PNG_BYTES


@pytest.mark.logic
def test_a_split_design_region_is_shaded_as_separate_blocks(npl, tmp_path):
    """The contig can replace two separate stretches of the input PDB. One
    span from the first position to the last would shade the gap between
    them, where mutations are allowed and did occur."""
    rows = [_plot_row(ra=2.0, tj=0.8,
                      rep_steered_mutations_chimerax="/A:4")]
    out = tmp_path / "mut.png"
    assert npl.plot_mutation_impact(rows, {1, 2, 7, 8}, 12, str(out)) is True
    assert out.stat().st_size > MIN_PNG_BYTES


# ── the per-seed figures with no seeds ───────────────────────────────────────

@pytest.mark.logic
def test_the_seed_heatmap_needs_seeds(npl, tmp_path):
    """Both per-seed figures read raw_per_seed_results.csv, which is absent
    from a workdir synced without the per-seed tables. Nextflow publishes a
    fixed list of figure paths, so each one still writes a file."""
    out = tmp_path / "heatmap.png"
    assert npl.plot_seed_outcomes_heatmap({}, {}, str(out)) is False
    _assert_placeholder(npl, out, "No seed-outcome data available")


@pytest.mark.logic
def test_the_seed_bars_need_seeds(npl, tmp_path):
    out = tmp_path / "bars.png"
    assert npl.plot_seed_outcomes_bars({}, {}, str(out)) is False
    _assert_placeholder(npl, out, "No seed-outcome data available")


@pytest.mark.logic
def test_a_stage_no_seed_reached_is_not_a_legend_entry(npl, tmp_path):
    """A run stopped before reversion has every seed in the steering stage.
    Stacking a zero-height cold-start and reversion segment would put two
    stages in the legend that no seed reached."""
    outcomes = {"design_0_seq_1": [("steering", "pass", 0),
                                   ("steering", "off_target", 1)]}
    out = tmp_path / "bars.png"
    assert npl.plot_seed_outcomes_bars(
        outcomes, {"design_0_seq_1": 0.7}, str(out)) is True
    assert out.stat().st_size > MIN_PNG_BYTES


# ── choosing a representative sequence group for the seed figures ────────────
#
# _rep_sg_for_outcomes answers "which three seeds are this sequence's result".
# cross_summary names the group for a tiered sequence. For a tier-none
# sequence it names nothing, so the function reads aggregated_results.csv and
# picks the best composite itself.

@pytest.mark.logic
def test_a_sequence_with_no_aggregate_has_no_representative(npl, tmp_path):
    """A directory created by a run that died before aggregation. Guessing
    group 0 would report one arbitrary group's seeds as the sequence's
    result."""
    d = tmp_path / "design_0_seq_1"
    d.mkdir()
    assert npl._rep_sg_for_outcomes(d, {}) is None


@pytest.mark.logic
def test_an_empty_aggregate_has_no_representative(npl, tmp_path):
    """Header-only, which is what aggregation writes when no sequence group
    produced a complete row."""
    d = _seq(tmp_path, "design_0_seq_1", [])
    assert npl._rep_sg_for_outcomes(d, {}) is None


@pytest.mark.logic
def test_a_group_with_no_numbers_still_names_a_representative(npl, tmp_path):
    """Scoring an unmeasurable group as -1e9 ranks it last without dropping
    it. Group 1 here is the only group with numbers, so it wins; the
    sequence keeps its seeds in the heatmap either way."""
    d = _seq(tmp_path, "design_0_seq_1", [
        {"sequence_group": "0", "outcome": "clean_steered",
         "steered_ra_eff_vs_truth_median": "",
         "steered_true_jaccard_median": ""},
        _agg("1", ra=4.0, tj=0.5)])
    assert npl._rep_sg_for_outcomes(d, {}) == "1"


@pytest.mark.logic
def test_every_group_without_numbers_still_names_one(npl, tmp_path):
    """All groups tie at -1e9, so the sort is stable and the first group is
    named. Returning nothing would drop the sequence from both per-seed
    figures although its seeds were predicted."""
    d = _seq(tmp_path, "design_0_seq_1", [
        {"sequence_group": "0", "outcome": "clean_steered",
         "steered_ra_eff_vs_truth_median": "",
         "steered_true_jaccard_median": ""},
        {"sequence_group": "1", "outcome": "clean_steered",
         "steered_ra_eff_vs_truth_median": "",
         "steered_true_jaccard_median": ""}])
    assert npl._rep_sg_for_outcomes(d, {}) == "0"


# ── load_seed_outcomes, the per-seed merge ───────────────────────────────────

@pytest.mark.logic
def test_a_control_directory_contributes_no_seeds(npl, tmp_path):
    """The per-seed figures rank sequences by composite. A control has no
    rank, so it would sort to the bottom of the heatmap as if it were the
    worst design rather than a reference."""
    runs = tmp_path / "runs"
    for name in ("design_0_seq_1", "input_control_polyA"):
        d = _seq(runs, name, [_agg("0", ra=2.0, tj=0.8)])
        _write_raw(d, [dict(PASSING, sequence_group="0", seed_index="0")])
    cross = tmp_path / "cross.csv"
    _cross(cross, [{"mpnn_sequence": "design_0_seq_1", "cross_tier": "A",
                    "rep_sequence_group": "0"}])

    outcomes, _comps = npl.load_seed_outcomes(runs, cross, all_sgs=False)
    assert set(outcomes) == {"design_0_seq_1"}


@pytest.mark.logic
def test_a_sequence_with_no_representative_is_dropped_only_from_the_heatmap(
        npl, tmp_path):
    """The heatmap draws the representative group's seeds, so a sequence
    with no representative has no row. The bar chart counts every group, so
    the same sequence still appears there. The two figures disagreeing about
    which sequences exist is the designed behaviour, not a loading bug."""
    runs = tmp_path / "runs"
    d = runs / "design_0_seq_1"
    d.mkdir(parents=True)
    _write_raw(d, [dict(PASSING, sequence_group="0", seed_index="0")])
    cross = tmp_path / "cross.csv"
    _cross(cross, [])

    rep, _c1 = npl.load_seed_outcomes(runs, cross, all_sgs=False)
    every, _c2 = npl.load_seed_outcomes(runs, cross, all_sgs=True)
    assert rep == {}
    assert list(every) == ["design_0_seq_1"]


@pytest.mark.logic
def test_a_sequence_with_no_per_seed_table_is_skipped(npl, tmp_path):
    """aggregated_results.csv is published and raw_per_seed_results.csv is
    not always kept, so this is the common state of an archived run."""
    runs = tmp_path / "runs"
    _seq(runs, "design_0_seq_1", [_agg("0", ra=2.0, tj=0.8)])
    cross = tmp_path / "cross.csv"
    _cross(cross, [{"mpnn_sequence": "design_0_seq_1", "cross_tier": "A",
                    "rep_sequence_group": "0"}])
    outcomes, _comps = npl.load_seed_outcomes(runs, cross, all_sgs=False)
    assert outcomes == {}


@pytest.mark.logic
def test_an_unparseable_seed_index_becomes_zero(npl, tmp_path):
    """The index only orders the cells along the heatmap row. It is not a
    measurement, so a bad one must not cost the seed its cell."""
    runs = tmp_path / "runs"
    d = _seq(runs, "design_0_seq_1", [_agg("0", ra=2.0, tj=0.8)])
    _write_raw(d, [dict(PASSING, sequence_group="0", seed_index="first")])
    cross = tmp_path / "cross.csv"
    _cross(cross, [{"mpnn_sequence": "design_0_seq_1", "cross_tier": "A",
                    "rep_sequence_group": "0"}])
    outcomes, _comps = npl.load_seed_outcomes(runs, cross, all_sgs=False)
    assert outcomes["design_0_seq_1"] == [("steering", "pass", 0)]


@pytest.mark.logic
def test_a_table_of_only_the_cold_start_baseline_yields_nothing(npl,
                                                                tmp_path):
    """The cycle-0 prediction has no sequence group, because it is one
    prediction shared by every group. A sequence whose steering never ran
    has only that row, and an empty seed list must not become a heatmap row
    of zero cells."""
    runs = tmp_path / "runs"
    d = _seq(runs, "design_0_seq_1", [_agg("0", ra=2.0, tj=0.8)])
    _write_raw(d, [dict(PASSING, sequence_group="", seed_index="",
                        steered_total_mutations="0")])
    cross = tmp_path / "cross.csv"
    _cross(cross, [{"mpnn_sequence": "design_0_seq_1", "cross_tier": "A",
                    "rep_sequence_group": "0"}])
    outcomes, comps = npl.load_seed_outcomes(runs, cross, all_sgs=False)
    assert outcomes == {}
    assert comps == {}


# ── the design-region file's tolerant parse ──────────────────────────────────

@pytest.mark.logic
def test_a_trailing_comma_is_not_a_position(npl, tmp_path):
    """DERIVE_INPUT_INDICES writes the list with a trailing comma. An empty
    token read as zero would shade a residue that does not exist and extend
    the mutation-impact axis to zero."""
    f = tmp_path / "region.txt"
    f.write_text("1,2,3,\n")
    assert npl._load_input_design_region(str(f)) == ({1, 2, 3}, 3)


@pytest.mark.logic
def test_a_non_numeric_token_is_dropped_and_the_rest_kept(npl, tmp_path):
    """The shading is context on a figure, so a malformed token loses that
    one residue rather than the whole figure's shading."""
    f = tmp_path / "region.txt"
    f.write_text("1,A:2,3\n")
    assert npl._load_input_design_region(str(f)) == ({1, 3}, 3)


# ── the cohort CLI's own argument handling ───────────────────────────────────
#
# main resolves the runs directory, loads the cohort, and then calls seven
# figure functions inside one try each. Every branch here decides whether the
# stage reports a problem or produces a thinner figure set in silence.

def _run_cohort_main(npl, *args):
    with mock.patch.object(sys, "argv", ["negsteer_plots.py", *args]):
        return npl.main()


def _one_sequence_cohort(tmp_path, name="design_0_seq_1"):
    """A cross-summary and a runs tree for one healthy tier-A sequence."""
    runs = tmp_path / "runs"
    _seq(runs, name, [_agg("0", ra=2.0, tj=0.8)])
    cross = tmp_path / "cross.csv"
    _cross(cross, [{"mpnn_sequence": name, "cross_tier": "A",
                    "rep_sequence_group": "0", "rep_cycle": "0",
                    "rep_pathway": ""}])
    return cross, runs


@pytest.mark.logic
def test_a_runs_dir_that_does_not_exist_warns_and_keeps_going(npl, tmp_path,
                                                               capsys):
    """Nextflow stages runs/ as a symlink, and a broken stage-in is a path
    that is not there. The cross-summary figures are still correct without
    it, so the stage degrades and names what it lost."""
    cross, _runs = _one_sequence_cohort(tmp_path)
    out = tmp_path / "out"
    assert _run_cohort_main(npl, "--csv", str(cross),
                            "--runs-dir", str(tmp_path / "gone"),
                            "--outdir", str(out)) == 0
    assert "WARNING: --runs-dir not found" in capsys.readouterr().out
    assert (out / "negsteer_tier_landscape.png").is_file()


@pytest.mark.logic
def test_without_the_flag_the_csv_sibling_is_tried_and_reported(npl, tmp_path,
                                                                capsys):
    """The fallback is <csv>/../runs. When it is absent the zero-passing
    sequences become invisible, which looks the same as a cohort that had
    none. The note is the only record that the cohort was reduced."""
    lonely = tmp_path / "lonely"
    lonely.mkdir()
    cross = lonely / "cross.csv"
    _cross(cross, [{"mpnn_sequence": "design_0_seq_1", "cross_tier": "A"}])
    assert not (lonely / "runs").exists()
    out = tmp_path / "out"
    assert _run_cohort_main(npl, "--csv", str(cross),
                            "--outdir", str(out)) == 0
    printed = capsys.readouterr().out
    assert "NOTE: no runs/ found" in printed
    assert "zero-passing sequences will be invisible" in printed


@pytest.mark.logic
def test_an_empty_cohort_writes_no_figures_and_still_succeeds(npl, tmp_path,
                                                              capsys):
    """Exit zero with no figures, rather than seven empty-data PNGs. A
    header-only cross-summary means the upstream channel was empty, and the
    publish step failing on a missing path is the louder signal."""
    cross = tmp_path / "cross.csv"
    _cross(cross, [])
    runs = tmp_path / "runs"
    runs.mkdir()
    out = tmp_path / "out"
    assert _run_cohort_main(npl, "--csv", str(cross), "--runs-dir", str(runs),
                            "--outdir", str(out)) == 0
    assert "No rows — nothing to plot." in capsys.readouterr().out
    assert list(out.glob("*.png")) == []


@pytest.mark.logic
def test_the_design_region_max_becomes_the_pdb_length(npl, tmp_path, capsys):
    """Without --pdb-length the highest design-region position is the only
    lower bound available for the receptor length, and it sets the
    mutation-impact axis. A run that wants the full receptor on that axis
    has to pass --pdb-length."""
    cross, runs = _one_sequence_cohort(tmp_path)
    region = tmp_path / "region.txt"
    region.write_text("1,2,3,60,61\n")
    out = tmp_path / "out"
    assert _run_cohort_main(npl, "--csv", str(cross), "--runs-dir", str(runs),
                            "--input-design-region", str(region),
                            "--outdir", str(out)) == 0
    printed = capsys.readouterr().out
    assert "loaded 5 input-PDB design-region positions" in printed
    assert "using PDB length: 61" in printed


@pytest.mark.logic
def test_a_seed_outcome_load_that_raises_leaves_the_other_figures_standing(
        npl, tmp_path, capsys):
    """load_seed_outcomes indexes the cross-summary by mpnn_sequence and
    raises when the column is absent, which is what an older cross-summary
    looks like. The five figures that do not need per-seed data are still
    correct, so the failure is reported and the stage continues."""
    runs = tmp_path / "runs"
    _seq(runs, "design_0_seq_1", [_agg("0", ra=2.0, tj=0.8)])
    cross = tmp_path / "cross.csv"
    with cross.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "cross_tier"])
        w.writeheader()
        w.writerow({"name": "design_0_seq_1", "cross_tier": "A"})

    out = tmp_path / "out"
    assert _run_cohort_main(npl, "--csv", str(cross), "--runs-dir", str(runs),
                            "--outdir", str(out)) == 0
    printed = capsys.readouterr().out
    assert "WARN: seed-outcomes load failed" in printed
    assert "(skipped — no data)" in printed
    assert (out / "negsteer_ra_eff_vs_jaccard.png").is_file()


@pytest.mark.logic
def test_one_figure_failing_does_not_cost_the_others(npl, tmp_path, capsys):
    """Each figure is drawn inside its own try. Seven figures take one
    cohort load, and losing the whole set to one matplotlib failure would
    mean re-running the stage to recover the six that worked."""
    cross, runs = _one_sequence_cohort(tmp_path)
    out = tmp_path / "out"
    # A directory where the first figure's file belongs. savefig cannot
    # write it, which is the true external failure this exercises.
    (out / "negsteer_tier_landscape.png").mkdir(parents=True)

    assert _run_cohort_main(npl, "--csv", str(cross), "--runs-dir", str(runs),
                            "--outdir", str(out)) == 0
    assert "FAILED: IsADirectoryError" in capsys.readouterr().out
    assert (out / "negsteer_ra_eff_vs_jaccard.png").is_file()
    assert (out / "negsteer_controls_diagnostic.png").is_file()


# ── the within-sequence module's representative choice ───────────────────────

@pytest.mark.logic
def test_a_group_with_no_numbers_loses_but_is_not_dropped(nwp, tmp_path):
    """Same -1e9 penalty as the cohort module, and it has to stay the same.
    The per-sequence panels are read beside the cohort heatmap, so the two
    modules naming different representative groups would show two different
    sets of seeds as one sequence's result."""
    d = _seq(tmp_path, "design_0_seq_1", [
        {"sequence_group": "0", "outcome": "clean_steered",
         "steered_ra_eff_vs_truth_median": "",
         "steered_true_jaccard_median": ""},
        _agg("1", ra=4.0, tj=0.5)])
    assert nwp._rep_sg(d, {}) == "1"


@pytest.mark.logic
def test_an_empty_aggregate_names_no_representative(nwp, tmp_path):
    d = _seq(tmp_path, "design_0_seq_1", [])
    assert nwp._rep_sg(d, {}) is None


@pytest.mark.logic
def test_the_composite_fallback_skips_singletons_and_uses_the_reverted_side(
        nwp, tmp_path):
    """The fallback runs when cross_summary has no composite, which is every
    tier-none sequence. The singleton row here looks the best at 0.9 - 0.05
    and has no real seed counts behind it. The surviving row reverted to
    30.5 A, so the sequence sorts to 0.02 - 0.05*30.5 = -1.505 rather than
    to its steered intermediate."""
    runs = tmp_path / "runs"
    collapsed = dict(_agg("0", ra=2.0, tj=0.8, outcome="pose_collapses"),
                     reverted_ra_eff_vs_truth_median="30.5",
                     reverted_true_jaccard_median="0.02")
    d = _seq(runs, "design_0_seq_1",
             [_agg("9", ra=1.0, tj=0.9, outcome="singleton"), collapsed])
    _write_raw(d, [dict(PASSING, sequence_group="0", seed_index="0")])
    cross = tmp_path / "cross.csv"
    _cross(cross, [{"mpnn_sequence": "design_0_seq_1", "cross_tier": "none",
                    "rep_sequence_group": "",
                    "rep_ra_eff_vs_truth_median": "",
                    "rep_true_jaccard_median": ""}])

    _rep, _all, composites, _tiers, _cold = nwp.load_cohort_seeds(runs, cross)
    assert composites["design_0_seq_1"] == pytest.approx(-1.505)


# ── the within-sequence figures on partial data ──────────────────────────────

@pytest.mark.logic
def test_a_seed_with_no_final_metrics_is_absent_from_the_dispersion_scatter(
        nwp, tmp_path):
    """A seed whose final prediction never landed has no RMSD and no
    Jaccard. The figure renders byte for byte as if the seed did not exist,
    including the tier legend, so the panel cannot imply it was measured."""
    empty = tmp_path / "empty.png"
    skipped = tmp_path / "skipped.png"
    nwp.plot_dispersion_overview({}, {}, str(empty))
    nwp.plot_dispersion_overview(
        {"design_0_seq_1": [_seed_rec(ra=None, tj=None)]},
        {"design_0_seq_1": "A"}, str(skipped))
    assert empty.read_bytes() == skipped.read_bytes()


@pytest.mark.logic
def test_the_dispersion_grid_writes_a_placeholder_with_no_rmsd(nwp, tmp_path):
    """The grid takes its x-axis bound from the cohort's RMSDs, so with none
    there is nothing to scale. It writes the placeholder and returns False.

    It used to return before saving, so no file was written, while main's
    `or True` wrapper printed the path as a success. Nextflow publishes a
    fixed figure list, so the stage then failed on a missing path with
    nothing in the log to explain it."""
    out = tmp_path / "grid.png"
    seeds = {"design_0_seq_1": [_seed_rec(ra=None, tj=None)]}
    assert nwp.plot_dispersion_grid(
        seeds, seeds, {"design_0_seq_1": "A"},
        {"design_0_seq_1": 0.7}, str(out)) is False
    assert out.is_file()
    expected = tmp_path / "expected.png"
    nwp._make_empty_plot("No per-seed RMSD data", str(expected))
    assert out.read_bytes() == expected.read_bytes()


@pytest.mark.logic
def test_the_dispersion_grid_blanks_the_cells_it_does_not_fill(nwp, tmp_path):
    """Five sequences go into a 2x3 grid. The sixth cell keeps its axes and
    its 0-to-5 A threshold line unless they are turned off, which reads as a
    sixth sequence with no seeds."""
    names = [f"design_{i}_seq_1" for i in range(5)]
    rep = {n: [_seed_rec(sg="0", seed=0, ra=2.0 + i, tj=0.5)]
           for i, n in enumerate(names)}
    every = {n: list(v) for n, v in rep.items()}
    # One sequence also carries a second group's seed and two unmeasurable
    # seeds, one representative and one not.
    every[names[0]] = [
        _seed_rec(sg="1", seed=0, ra=9.0, tj=0.2),
        _seed_rec(sg="1", seed=1, ra=None, tj=0.2),
        _seed_rec(sg="0", seed=0, ra=2.0, tj=0.5),
        _seed_rec(sg="0", seed=1, ra=None, tj=None),
    ]
    rep[names[0]] = [_seed_rec(sg="0", seed=0, ra=2.0, tj=0.5),
                     _seed_rec(sg="0", seed=1, ra=None, tj=None)]
    out = tmp_path / "grid.png"
    nwp.plot_dispersion_grid(rep, every, {n: "A" for n in names},
                             {n: 0.7 for n in names}, str(out))
    assert out.stat().st_size > MIN_PNG_BYTES


@pytest.mark.logic
def test_the_trajectory_plot_writes_a_placeholder_without_an_rmsd(nwp,
                                                                  tmp_path):
    """Same cause as the grid. The y axis is the RMSD and there is no bound
    for it, so the placeholder goes out instead of nothing."""
    out = tmp_path / "traj.png"
    seeds = {"design_0_seq_1": [_seed_rec(ra=None, tj=None,
                                          steered_ra_raw=None)]}
    assert nwp.plot_stage_trajectories(
        seeds, {"design_0_seq_1": "A"}, {}, str(out)) is False
    assert out.is_file()


@pytest.mark.logic
def test_the_trajectory_grid_blanks_the_cells_it_does_not_fill(nwp, tmp_path):
    """One panel per RFDiffusion design. Five designs leave a sixth cell in
    the 2x3 grid, and an axes frame with a threshold line in it reads as a
    design whose seeds all failed."""
    every = {f"design_{i}_seq_1": [_seed_rec(ra=2.0 + i, tj=0.5)]
             for i in range(5)}
    cold = {n: 41.2 for n in every}
    out = tmp_path / "traj.png"
    nwp.plot_stage_trajectories(every, {n: "A" for n in every}, cold,
                                str(out))
    assert out.stat().st_size > MIN_PNG_BYTES


@pytest.mark.logic
def test_a_seed_with_no_weighted_jaccard_leaves_that_figure_empty(nwp,
                                                                   tmp_path):
    """weighted_jaccard is computed only when the interface could be
    resolved, so it is blank more often than true_jaccard. A seed missing
    either value cannot be a point on a diagonal comparing the two, and a
    figure with no points says so in words."""
    empty = tmp_path / "empty.png"
    skipped = tmp_path / "skipped.png"
    nwp.plot_weighted_vs_true_jaccard({}, {}, str(empty))
    nwp.plot_weighted_vs_true_jaccard(
        {"design_0_seq_1": [_seed_rec(tj=0.8, wj=None)]},
        {"design_0_seq_1": "A"}, str(skipped))
    assert empty.read_bytes() == skipped.read_bytes()


@pytest.mark.logic
def test_a_sequence_with_no_composite_is_absent_from_every_panel(nwp,
                                                                  tmp_path):
    """The y axis of all ten panels is the composite. A sequence without one
    has no y, so it is dropped before the panels are chosen rather than
    plotted at zero."""
    out = tmp_path / "conf.png"
    empty = tmp_path / "empty.png"
    nwp.plot_composite_vs_confidence({}, {}, {}, str(empty))
    nwp.plot_composite_vs_confidence(
        {"design_0_seq_1": [_seed_rec(iptm=0.8)]},
        {}, {"design_0_seq_1": "A"}, str(out))
    assert out.read_bytes() == empty.read_bytes(), \
        "a sequence with no composite put a panel on the figure"


@pytest.mark.logic
def test_a_panel_only_plots_the_sequences_that_have_its_metric(nwp, tmp_path):
    """A panel exists when any sequence has the metric, and the sequences
    without it are skipped in that panel alone. Reusing the previous
    sequence's value would correlate the composite against a number from a
    different design."""
    rep = {"design_0_seq_1": [_seed_rec(iptm=0.8)],
           "design_0_seq_2": [_seed_rec(ipae=4.0)]}
    out = tmp_path / "conf.png"
    nwp.plot_composite_vs_confidence(
        rep, {"design_0_seq_1": 0.7, "design_0_seq_2": 0.3},
        {"design_0_seq_1": "A", "design_0_seq_2": "none"}, str(out))
    assert out.stat().st_size > MIN_PNG_BYTES


# ── the within-sequence CLI ──────────────────────────────────────────────────

def _run_within_main(nwp, *args):
    argv = ["negsteer_within_sequence_plots.py", *args]
    with mock.patch.object(sys, "argv", argv):
        return nwp.main()


@pytest.mark.logic
def test_a_missing_runs_dir_is_refused_rather_than_degraded(nwp, tmp_path):
    """Unlike the cohort module, every figure here is built from the
    per-seed tables. Without runs/ there is nothing to draw, so exiting is
    honest where degrading would publish five empty figures."""
    cross = tmp_path / "cross.csv"
    _cross(cross, [])
    with pytest.raises(SystemExit, match="--runs-dir does not exist"):
        _run_within_main(nwp, "--runs-dir", str(tmp_path / "gone"),
                         "--cross-summary-csv", str(cross),
                         "--outdir", str(tmp_path / "out"))


@pytest.mark.logic
def test_a_missing_cross_summary_is_refused(nwp, tmp_path):
    """It names the representative sequence group and carries the composite
    the panels are sorted by. Both would otherwise be guessed."""
    runs = tmp_path / "runs"
    runs.mkdir()
    with pytest.raises(SystemExit, match="--cross-summary-csv does not exist"):
        _run_within_main(nwp, "--runs-dir", str(runs),
                         "--cross-summary-csv", str(tmp_path / "gone.csv"),
                         "--outdir", str(tmp_path / "out"))


@pytest.mark.logic
def test_an_empty_runs_dir_plots_nothing_and_succeeds(nwp, tmp_path, capsys):
    """A cohort where no sequence has a representative group. Exit zero with
    no figures, so the publish step is what reports the gap."""
    runs = tmp_path / "runs"
    runs.mkdir()
    cross = tmp_path / "cross.csv"
    _cross(cross, [])
    out = tmp_path / "out"
    assert _run_within_main(nwp, "--runs-dir", str(runs),
                            "--cross-summary-csv", str(cross),
                            "--outdir", str(out)) == 0
    assert "Nothing to plot." in capsys.readouterr().out
    assert list(out.glob("*.png")) == []


@pytest.mark.logic
def test_one_within_sequence_figure_failing_does_not_cost_the_others(
        nwp, tmp_path, capsys):
    """Five figures share one walk of the runs tree, which is the expensive
    part. One unwritable path must not discard the other four."""
    runs = tmp_path / "runs"
    d = _seq(runs, "design_0_seq_1", [_agg("0", ra=2.0, tj=0.8)])
    _write_raw(d, [dict(PASSING, sequence_group="0", seed_index="0",
                        steered_true_jaccard="0.8",
                        steered_weighted_jaccard="0.75")])
    cross = tmp_path / "cross.csv"
    _cross(cross, [{"mpnn_sequence": "design_0_seq_1", "cross_tier": "A",
                    "rep_sequence_group": "0"}])
    out = tmp_path / "out"
    (out / "negsteer_per_seed_dispersion_overview.png").mkdir(parents=True)

    assert _run_within_main(nwp, "--runs-dir", str(runs),
                            "--cross-summary-csv", str(cross),
                            "--outdir", str(out)) == 0
    assert "FAILED: IsADirectoryError" in capsys.readouterr().out
    assert (out / "negsteer_weighted_vs_true_jaccard.png").is_file()


# ── main reports what it wrote ────────────────────────────────────────────────

@pytest.mark.logic
def test_main_reports_the_path_of_every_figure_that_drew(nwp, tmp_path,
                                                          capsys):
    """main reports a figure by the truth of its return value, so each plot_*
    returns True once it has saved.

    All five used to fall off the end on success and return None, so the log
    read "(skipped — no data)" for five figures that had just written a PNG.
    Nextflow publishes a fixed figure list, so the stage read as having
    produced nothing on every successful run. bin/negsteer_plots.py returns
    True from all seven of its figures, which is the contract main was
    written against.
    """
    runs = tmp_path / "runs"
    d = _seq(runs, "design_0_seq_1", [_agg("0", ra=2.0, tj=0.8)])
    _write_raw(d, [dict(PASSING, sequence_group="0", seed_index="0",
                        steered_true_jaccard="0.8",
                        steered_weighted_jaccard="0.75")])
    cross = tmp_path / "cross.csv"
    _cross(cross, [{"mpnn_sequence": "design_0_seq_1", "cross_tier": "A",
                    "rep_sequence_group": "0"}])
    out = tmp_path / "out"
    argv = ["negsteer_within_sequence_plots.py", "--runs-dir", str(runs),
            "--cross-summary-csv", str(cross), "--outdir", str(out)]
    with mock.patch.object(sys, "argv", argv):
        assert nwp.main() == 0

    text = capsys.readouterr().out
    assert len(list(out.glob("*.png"))) == 5
    assert text.count("→") == 5, f"a drawn figure went unreported:\n{text}"
    assert "(skipped — no data)" not in text


@pytest.mark.logic
def test_main_reports_a_figure_with_nothing_to_draw_as_skipped(nwp, tmp_path,
                                                                capsys):
    """The other half of the same report. A figure that returned False wrote
    only a placeholder, and saying so is what tells a reader the stage ran
    but the cohort had no measurable seed.

    load_cohort_seeds reads a steering seed's RMSD straight out of
    steered_ra_eff_vs_truth, so blanking that column leaves the record in
    place with no measurable value. The record is what keeps main past its
    "Nothing to plot" exit, and the two figures that scale an axis on the
    RMSD then have nothing to scale.
    """
    runs = tmp_path / "runs"
    d = _seq(runs, "design_0_seq_1", [_agg("0", ra=2.0, tj=0.8)])
    _write_raw(d, [dict(PASSING, sequence_group="0", seed_index="0",
                        steered_ra_eff_vs_truth="",
                        steered_true_jaccard="", reversion_verdict="")])
    cross = tmp_path / "cross.csv"
    _cross(cross, [{"mpnn_sequence": "design_0_seq_1", "cross_tier": "A",
                    "rep_sequence_group": "0"}])
    out = tmp_path / "out"
    argv = ["negsteer_within_sequence_plots.py", "--runs-dir", str(runs),
            "--cross-summary-csv", str(cross), "--outdir", str(out)]
    with mock.patch.object(sys, "argv", argv):
        assert nwp.main() == 0

    text = capsys.readouterr().out
    assert "(skipped — no data)" in text, text
    assert "FAILED" not in text
