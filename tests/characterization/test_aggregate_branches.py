"""Tests for the aggregation branches that decide what counts as a rescue.

`negsteer_aggregate.py` turns per-seed Boltz predictions into the table the
thesis reports. Three layers of that pipeline decide which designs appear as
successes, and all three fail silently rather than loudly.

  - The per-row eligibility predicates (`_row_is_clean_steered`,
    `_row_is_pose_holds`) decide which rows enter the unified ranking.
    A row admitted wrongly becomes a reported rescue; a row excluded
    wrongly disappears from the figures entirely.
  - `_classify_outcome` re-derives the reversion verdict from the
    median/majority columns. It is the per-design outcome taxonomy, and
    every downgrade path in it exists because a weaker version of the rule
    once over-reported pose_holds.
  - `_populate_reverted_mutations` computes the post-reversion mutation set
    as cumulative mutations minus reverted positions. Getting that set wrong
    misstates how many mutations a reported design actually carries.

Inputs here are constructed rows and constructed run trees whose correct
output is stated before the code runs, so each assertion is a derived value
rather than a snapshot.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def agg():
    sys.path.insert(0, str(REPO_ROOT / "bin"))
    spec = importlib.util.spec_from_file_location(
        "negsteer_aggregate", REPO_ROOT / "bin" / "negsteer_aggregate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class Args:
    def __init__(self, **kw):
        self.populate_all = False
        self.require_intact = True
        self.__dict__.update(kw)


# ── eligibility: the clean steered row ───────────────────────────────────────
#
# A clean steered row is one that was never contaminated and whose steered
# prediction passes the structural filter. The predicate used to gate on
# `steered_ipsae_min` being populated, as a proxy for "compute-final-metrics
# ran on this row". `--populate-all` populates ipSAE on every non-dropped row,
# which broke the proxy and let wrong-interface rows into the ranking. The
# proxy is now a direct ra_eff < 5 A check, so the unparseable and non-finite
# cases have to be excluded explicitly.


def clean_row(**kw):
    r = {"design": "design_00", "reversion_verdict": "",
         "steered_receptor_intact": "1", "steered_ra_eff_vs_truth": "2.0",
         "steered_true_jaccard": "0.8"}
    r.update(kw)
    return r


@pytest.mark.logic
def test_a_sound_uncontaminated_row_is_ranking_eligible(agg):
    assert agg._row_is_clean_steered(clean_row()) is True


@pytest.mark.logic
def test_an_unparseable_intact_flag_is_not_an_intact_receptor(agg):
    """A garbled flag must not be read as intact. Treating it as 1 would put
    a collapsed receptor into the ranking."""
    assert agg._row_is_clean_steered(
        clean_row(steered_receptor_intact="yes")) is False


@pytest.mark.logic
def test_an_unparseable_steered_rmsd_is_not_ranking_eligible(agg):
    assert agg._row_is_clean_steered(
        clean_row(steered_ra_eff_vs_truth="n/a")) is False


@pytest.mark.logic
@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
def test_a_non_finite_steered_rmsd_is_not_ranking_eligible(agg, bad):
    """`collect` writes nan for a prediction that never landed, and nan fails
    every comparison, so `ra < 5.0` would admit it without this check."""
    assert agg._row_is_clean_steered(
        clean_row(steered_ra_eff_vs_truth=bad)) is False


@pytest.mark.logic
def test_a_steered_row_at_five_angstroms_is_excluded(agg):
    """The threshold is exclusive. 5.0 A is the pipeline's rmsd_threshold and
    a design exactly at it did not beat it."""
    assert agg._row_is_clean_steered(
        clean_row(steered_ra_eff_vs_truth="5.0")) is False
    assert agg._row_is_clean_steered(
        clean_row(steered_ra_eff_vs_truth="4.999")) is True


# ── eligibility: the pose_holds row ──────────────────────────────────────────
#
# A pose_holds design had its contaminating mutations reverted and the
# reverted prediction re-run. This predicate mirrors the ra_eff check in
# _row_is_clean_steered so the two are consistent. Before that mirroring, a
# pose_holds row whose reverted prediction sat at 25 A was still ranking
# eligible, because the reversion harvest only filters on receptor intactness
# and never looks at ra_eff. Such a row is reported as a rescue while its
# effector is nowhere near the true interface.


def ph_row(**kw):
    r = {"reversion_verdict": "pose_holds",
         "reverted_receptor_intact": "1", "reverted_ra_eff_vs_truth": "2.0",
         "steered_true_jaccard": "0.8"}
    r.update(kw)
    return r


@pytest.mark.logic
def test_a_sound_reverted_pose_is_ranking_eligible(agg):
    assert agg._row_is_pose_holds(ph_row()) is True


@pytest.mark.logic
@pytest.mark.parametrize("verdict", ["", "pose_collapses", "new_contamination",
                                     "pose_holds_not_intact"])
def test_only_the_exact_pose_holds_verdict_qualifies(agg, verdict):
    assert agg._row_is_pose_holds(ph_row(reversion_verdict=verdict)) is False


@pytest.mark.logic
def test_an_unparseable_reverted_intact_flag_is_not_intact(agg):
    assert agg._row_is_pose_holds(
        ph_row(reverted_receptor_intact="maybe")) is False


@pytest.mark.logic
def test_a_reverted_row_with_no_rmsd_is_not_ranking_eligible(agg):
    """A blank reverted ra_eff means the reverted prediction was never
    scored. Ranking it would rank an unmeasured structure."""
    assert agg._row_is_pose_holds(
        ph_row(reverted_ra_eff_vs_truth="")) is False
    row = ph_row()
    del row["reverted_ra_eff_vs_truth"]
    assert agg._row_is_pose_holds(row) is False


@pytest.mark.logic
def test_an_unparseable_reverted_rmsd_is_not_ranking_eligible(agg):
    assert agg._row_is_pose_holds(
        ph_row(reverted_ra_eff_vs_truth="n/a")) is False


@pytest.mark.logic
@pytest.mark.parametrize("bad", ["nan", "inf"])
def test_a_non_finite_reverted_rmsd_is_not_ranking_eligible(agg, bad):
    assert agg._row_is_pose_holds(
        ph_row(reverted_ra_eff_vs_truth=bad)) is False


@pytest.mark.logic
def test_a_pose_holds_row_at_twenty_five_angstroms_is_excluded(agg):
    """This is the case the mirrored check was added for. The harvest called
    it pose_holds on receptor intactness alone, so without the ra_eff check a
    25 A effector placement enters the ranking as a rescue."""
    assert agg._row_is_pose_holds(
        ph_row(reverted_ra_eff_vs_truth="25.0")) is False


# ── the ranking metric and the composite score ───────────────────────────────
#
# _ranking_metric is the single place that decides WHICH prediction a design
# is scored on. A clean steered row is scored on its steered prediction; a
# pose_holds row on its reverted one. Reading the steered value for a
# pose_holds row would score the contaminated structure that was discarded.


@pytest.mark.logic
def test_a_clean_row_is_scored_on_its_steered_prediction(agg):
    row = clean_row(steered_ra_eff_vs_truth="2.5",
                    reverted_ra_eff_vs_truth="9.0")
    assert agg._ranking_metric(row, "ra_eff_vs_truth") == 2.5


@pytest.mark.logic
def test_a_pose_holds_row_is_scored_on_its_reverted_prediction(agg):
    """The steered structure of a pose_holds design was contaminated and
    thrown away. Scoring it would report a number for a discarded pose."""
    row = ph_row(reverted_ra_eff_vs_truth="2.5",
                 steered_ra_eff_vs_truth="9.0")
    assert agg._ranking_metric(row, "ra_eff_vs_truth") == 2.5


@pytest.mark.logic
def test_an_ineligible_row_has_no_ranking_metric(agg):
    row = clean_row(steered_receptor_intact="0")
    assert agg._ranking_metric(row, "ra_eff_vs_truth") is None


@pytest.mark.logic
def test_a_missing_metric_on_an_eligible_row_is_none(agg):
    """None, not zero. Zero on an ascending rank is the best score there is,
    so a design with no ipSAE would rank first."""
    assert agg._ranking_metric(clean_row(), "ipsae_min") is None
    assert agg._ranking_metric(clean_row(steered_ipsae_min=""),
                               "ipsae_min") is None


@pytest.mark.logic
@pytest.mark.parametrize("bad", ["nan", "inf"])
def test_a_non_finite_metric_is_none(agg, bad):
    assert agg._ranking_metric(clean_row(steered_ipsae_min=bad),
                               "ipsae_min") is None


@pytest.mark.logic
def test_an_unparseable_metric_is_none(agg):
    assert agg._ranking_metric(clean_row(steered_ipsae_min="high"),
                               "ipsae_min") is None


@pytest.mark.logic
def test_the_composite_is_jaccard_minus_a_twentieth_of_the_rmsd(agg):
    """0.9 - 0.05 x 4.0 = 0.7. Higher is better, so the trade is 1 point of
    interface overlap against 20 A of effector displacement."""
    row = clean_row(steered_ra_eff_vs_truth="4.0",
                    steered_true_jaccard="0.9")
    assert agg._ranking_composite(row) == pytest.approx(0.7)


@pytest.mark.logic
def test_a_pose_holds_composite_mixes_reverted_rmsd_with_steered_jaccard(agg):
    """true_jaccard is computed once on the steered prediction and not
    recomputed after reversion, because interface identity is carried by the
    designed sequence rather than by the steering mutations. ra_eff is the
    reverted value. 0.8 - 0.05 x 2.0 = 0.7."""
    row = ph_row(reverted_ra_eff_vs_truth="2.0", steered_true_jaccard="0.8")
    assert agg._ranking_composite(row) == pytest.approx(0.7)


@pytest.mark.logic
@pytest.mark.parametrize("bad", ["nan", "inf"])
def test_a_non_finite_jaccard_gives_no_composite(agg, bad):
    """An eligible row is needed to reach this check. With ra_eff over 5 A
    the row is ineligible and the composite is None for the wrong reason."""
    row = clean_row(steered_ra_eff_vs_truth="2.0", steered_true_jaccard=bad)
    assert agg._ranking_metric(row, "ra_eff_vs_truth") == 2.0
    assert agg._ranking_composite(row) is None


@pytest.mark.logic
def test_an_unparseable_jaccard_gives_no_composite(agg):
    row = clean_row(steered_ra_eff_vs_truth="2.0",
                    steered_true_jaccard="most of it")
    assert agg._ranking_composite(row) is None


# ── the majority votes ───────────────────────────────────────────────────────


@pytest.mark.logic
def test_a_binary_vote_ignores_a_value_that_is_not_zero_or_one(agg):
    """A flag of 2 is not a vote and is dropped from n_used, exactly as a
    blank is. Three usable votes with one positive is below the majority of
    two; admitting the 2 as a fourth vote would change the denominator the
    threshold is derived from."""
    majority, n_pos, n_used = agg._agg_majority_binary([1, 2, 0, 0])
    assert (majority, n_pos, n_used) == (0, 1, 3)


@pytest.mark.logic
def test_a_position_vote_ignores_an_unparseable_position(agg):
    """A non-integer token must not be counted as a position. Two seeds
    report 7, so 7 reaches the majority of 2 and the junk token does not."""
    positions, counts, n_seeds = agg._agg_majority_positions(
        [[7, "x"], [7, None]])
    assert positions == [7]
    assert counts == {7: 2}
    assert n_seeds == 2


@pytest.mark.logic
def test_one_seed_reporting_a_position_twice_counts_once(agg):
    """Per-position counts are per seed, not per contact. Across three seeds
    the majority is two. Position 7 is listed twice by one seed and would
    reach that majority on its own, which would read as two seeds agreeing."""
    positions, counts, _ = agg._agg_majority_positions([[7, 7], [9], [9]])
    assert counts == {7: 1, 9: 2}
    assert positions == [9]


# ── the mutation strings ─────────────────────────────────────────────────────


@pytest.mark.logic
def test_a_position_edited_twice_appears_once_in_the_chimerax_spec(agg):
    """The ChimeraX spec is a position list, so a repeat is a duplicate
    selection. The amino-acid string keeps both edits, because a re-edit of
    one position is a real event in the mutation history."""
    cx, aa = agg.format_mutation_strings(
        [(5, "A", "K"), (5, "K", "W"), (7, "G", "P")], chain_id="A")
    assert cx == "/A:5,7"
    assert aa == "A5K K5W G7P"


# ── the aggregated outcome taxonomy ──────────────────────────────────────────
#
# _classify_outcome re-derives the verdict from the median and majority
# columns. Every downgrade path in it was added after a weaker rule
# over-reported pose_holds. The structural gate comes first, then the
# contamination gate, then two gates on how the individual seeds voted.


def agg_doc(**kw):
    d = {"reverted_ra_eff_vs_truth_n_used": 3,
         "reverted_receptor_intact_majority": 1,
         "reverted_receptor_intact_n_positive": 3,
         "reverted_receptor_intact_n_used": 3,
         "reverted_ra_eff_vs_truth_median": "2.0",
         "reverted_mutated_contact_positions_majority": ""}
    d.update(kw)
    return d


def counts(**kw):
    c = {"pose_holds": 0, "pose_collapses": 0, "new_contamination": 0,
         "no_data": 0, "clean_steered": 0}
    c.update(kw)
    return c


@pytest.mark.logic
def test_a_blank_reverted_median_collapses_the_pose(agg):
    """No median means no seed produced a usable reverted ra_eff. The design
    cannot be called a rescue on no measurement."""
    outcome, reason = agg._classify_outcome(
        agg_doc(reverted_ra_eff_vs_truth_median=""), counts(pose_holds=3))
    assert outcome == "pose_collapses"
    assert "median unavailable" in reason


@pytest.mark.logic
def test_an_unparseable_reverted_median_collapses_the_pose(agg):
    outcome, reason = agg._classify_outcome(
        agg_doc(reverted_ra_eff_vs_truth_median="n/a"), counts(pose_holds=3))
    assert outcome == "pose_collapses"
    assert "median unparseable" in reason


@pytest.mark.logic
def test_three_agreeing_seeds_inside_the_cutoff_hold_the_pose(agg):
    outcome, reason = agg._classify_outcome(
        agg_doc(), counts(pose_holds=3))
    assert outcome == "pose_holds"
    assert "median ra_eff 2.00" in reason
    assert "3/3 seeds individually pass" in reason


@pytest.mark.logic
def test_a_clean_steered_seed_counts_as_a_pass(agg):
    """A seed with zero contamination on its mutated positions had reversion
    skipped on purpose, so its steered structure is its final answer. Without
    counting it, a design whose only sound seed needed no reversion is
    downgraded."""
    outcome, reason = agg._classify_outcome(
        agg_doc(), counts(clean_steered=2, pose_collapses=1))
    assert outcome == "pose_holds"
    assert "0 pose_holds + 2 clean_steered" in reason


@pytest.mark.logic
def test_an_ungated_majority_contact_does_not_downgrade(agg):
    """Only positions inside the design region union the true interface count
    as contamination. An off-interface contact from a reaching bit is
    acceptable by construction, so position 99 must not downgrade."""
    outcome, _ = agg._classify_outcome(
        agg_doc(reverted_mutated_contact_positions_majority="99"),
        counts(pose_holds=3), 5.0, {12, 13})
    assert outcome == "pose_holds"


@pytest.mark.logic
def test_a_gated_majority_contact_is_new_contamination(agg):
    outcome, reason = agg._classify_outcome(
        agg_doc(reverted_mutated_contact_positions_majority="12"),
        counts(pose_holds=3), 5.0, {12, 13})
    assert outcome == "new_contamination"
    assert "[12]" in reason


@pytest.mark.logic
def test_a_mixed_contact_set_names_both_halves(agg):
    """The reason string has to distinguish the contact that downgraded the
    design from the one that was tolerated, or the downgrade is unreadable."""
    outcome, reason = agg._classify_outcome(
        agg_doc(reverted_mutated_contact_positions_majority="12,99"),
        counts(pose_holds=3), 5.0, {12, 13})
    assert outcome == "new_contamination"
    assert "gated position(s)" in reason and "[12]" in reason
    assert "ungated positions [99] (acceptable)" in reason


@pytest.mark.logic
def test_with_no_gating_set_every_majority_contact_downgrades(agg):
    """Runs too old to carry a plan.json gating set fall back to the legacy
    ungated rule, which treats every majority contact as contamination."""
    outcome, reason = agg._classify_outcome(
        agg_doc(reverted_mutated_contact_positions_majority="12,99"),
        counts(pose_holds=3), 5.0, None)
    assert outcome == "new_contamination"
    assert "[12, 99]" in reason
    assert "acceptable" not in reason


@pytest.mark.logic
def test_no_seed_passing_individually_downgrades_to_collapse(agg):
    """The aggregated median can pass while no single seed does, because the
    median of three different failures is not a success. Ties favour
    pose_collapses as the more conservative call."""
    outcome, reason = agg._classify_outcome(
        agg_doc(), counts(pose_collapses=2, new_contamination=1))
    assert outcome == "pose_collapses"
    assert "0/3 seeds individually agree" in reason
    assert "downgrading to pose_collapses" in reason


@pytest.mark.logic
def test_no_seed_passing_with_contamination_leading_downgrades_to_that(agg):
    outcome, reason = agg._classify_outcome(
        agg_doc(), counts(pose_collapses=1, new_contamination=2))
    assert outcome == "new_contamination"
    assert "downgrading to new_contamination" in reason


@pytest.mark.logic
def test_one_passing_seed_against_two_collapses_is_downgraded(agg):
    """The Bug-E residual guard. One seed passing while two fail the same way
    at unrelated positions reaches no position majority, so the contamination
    gate does not fire and this count comparison is the only thing stopping
    it being called a rescue."""
    outcome, reason = agg._classify_outcome(
        agg_doc(), counts(pose_holds=1, pose_collapses=2))
    assert outcome == "pose_collapses"
    assert "1/3 seed(s) individually pass" in reason
    assert "failure modes dominate" in reason


@pytest.mark.logic
def test_one_passing_seed_against_two_contaminations_is_downgraded(agg):
    outcome, reason = agg._classify_outcome(
        agg_doc(), counts(pose_holds=1, new_contamination=2))
    assert outcome == "new_contamination"
    assert "downgrading to new_contamination" in reason


@pytest.mark.logic
@pytest.mark.parametrize("failure", ["pose_collapses", "new_contamination"])
def test_an_evenly_split_design_is_currently_called_a_rescue(agg, failure):
    """One seed passing against one failing is reported as pose_holds. The
    guard is `non_failing < max(npc, nnc)`, so it needs the passes to equal
    the failures, while the docstring above it states the passes must EXCEED
    each failure count. On the stricter reading a 1-1 split downgrades. This
    pins the implemented rule, which decides whether a split design is
    counted in the rescue total."""
    outcome, _ = agg._classify_outcome(
        agg_doc(), counts(pose_holds=1, **{failure: 1}))
    assert outcome == "pose_holds"


@pytest.mark.logic
def test_a_no_data_seed_counts_towards_not_failing(agg):
    """no_data is an absent measurement, not a failure. One pass plus one
    no_data equals the single pose_collapses, so the guard does not fire."""
    outcome, _ = agg._classify_outcome(
        agg_doc(), counts(pose_holds=1, no_data=1, pose_collapses=1))
    assert outcome == "pose_holds"


# ── the per-seed verdict for a skipped reversion ─────────────────────────────


def rev_row(**kw):
    r = {"reversion_verdict": "", "steered_receptor_intact": "1",
         "steered_ra_eff_vs_truth": "2.0",
         "steered_n_contacts_on_mutated_positions": "0",
         "reverted_receptor_intact": "", "reverted_ra_eff_vs_truth": "",
         "reverted_total_mutations": ""}
    r.update(kw)
    return r


@pytest.mark.logic
def test_an_unreadable_steered_intact_flag_is_not_a_clean_pass(agg):
    """Zero contamination is not enough on its own. A garbled intact flag
    must not promote the seed to pass-equivalent, because a steered miss with
    no contamination would then count as a rescue."""
    got = agg._per_seed_verdict_breakdown(
        [rev_row(steered_receptor_intact="yes")], 5.0)
    assert got["clean_steered"] == 0
    assert got["no_data"] == 1


@pytest.mark.logic
def test_an_unreadable_steered_rmsd_is_not_a_clean_pass(agg):
    got = agg._per_seed_verdict_breakdown(
        [rev_row(steered_ra_eff_vs_truth="")], 5.0)
    assert got["clean_steered"] == 0
    assert got["no_data"] == 1


# ── the post-reversion mutation set ──────────────────────────────────────────
#
# _populate_reverted_mutations computes what a pose_holds design actually
# carries after reversion: its cumulative mutations minus the positions that
# were reverted. That number is what a reader sees as the design's mutation
# load, so an over-count makes a rescue look more invasive than it is and an
# under-count hides mutations that are still there.
#
# The reverted contact columns are populated for EVERY reverted row, not only
# pose_holds ones. Blanking them on non-pose_holds rows was the pre-audit
# behaviour and it made the per-seed aggregator see empty contact sets for
# seeds whose harvest had flagged contamination, inflating the pose_holds
# count and reporting new_contamination designs as pose_holds.


def mut_design(d: Path, positions) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "mutations.tsv").write_text(
        "pos1\twt\tmut\n" + "".join(f"{p}\tA\tK\n" for p in positions))
    (d / "prediction.pdb").write_text("ATOM\n")
    return d


def cycle0(root: Path, *, plan=None, contaminated=None, results=None,
           rev_plan=None) -> Path:
    c0 = root / "cycle_0"
    c0.mkdir(parents=True, exist_ok=True)
    if plan is not None:
        (c0 / "plan.json").write_text(
            plan if isinstance(plan, str) else json.dumps(plan))
    if contaminated is not None:
        (c0 / "contaminated.json").write_text(
            contaminated if isinstance(contaminated, str)
            else json.dumps(contaminated))
    if results is not None:
        (c0 / "reversion_results.json").write_text(
            results if isinstance(results, str) else json.dumps(results))
    if rev_plan is not None:
        (c0 / "reversion_plan.json").write_text(
            rev_plan if isinstance(rev_plan, str) else json.dumps(rev_plan))
    return c0


def ph_tree_row(design_dir: Path, *, label="design_00", **kw):
    r = {"cycle": 0, "design": label, "pathway": "cycle_0",
         "parent_pathway": "", "reversion_verdict": "pose_holds",
         "pdb": str(design_dir / "prediction.pdb")}
    r.update(kw)
    return r


@pytest.mark.logic
def test_the_reverted_set_is_the_cumulative_set_minus_the_reverted_ones(
        agg, tmp_path):
    """The design mutated 5, 7 and 12 and position 7 was reverted, so two
    mutations remain. The chain letter comes from the plan, because the
    ChimeraX spec has to name the chain the reader will select."""
    d = mut_design(tmp_path / "design_00", [5, 7, 12])
    cycle0(tmp_path,
           plan={"pred_receptor_chain": "C", "contact_cutoff": 4.5},
           contaminated={"contaminated": [
               {"label": "design_00", "positions_to_revert": [7]}]},
           results={"design_00": {
               "reverted_mutated_contact_positions": [],
               "reverted_n_contact_residues": 9,
               "reverted_contact_residues": "10,11"}})
    rows = [ph_tree_row(d)]
    agg._populate_reverted_mutations(rows, tmp_path)
    assert rows[0]["reverted_total_mutations"] == 2
    assert rows[0]["reverted_mutations_chimerax"] == "/C:5,12"
    assert rows[0]["reverted_mutations_aa"] == "A5K A12K"
    assert rows[0]["reverted_contact_cutoff_used"] == "4.5"
    assert rows[0]["reverted_n_contact_residues"] == 9


@pytest.mark.logic
def test_the_steered_contact_positions_are_surfaced_for_every_reverted_row(
        agg, tmp_path):
    """The contamination that triggered the reversion is reported on rows
    that failed too, not only on the rescues. Blanking it for a failed row
    made the downstream aggregator see no contamination for that seed."""
    d = mut_design(tmp_path / "design_00", [5, 7])
    cycle0(tmp_path, plan={"pred_receptor_chain": "A"},
           contaminated={"contaminated": [
               {"label": "design_00", "positions_to_revert": [7, 5]}]},
           results={"design_00": {
               "reverted_mutated_contact_positions": [7],
               "reverted_n_contact_residues": 3,
               "reverted_contact_residues": "7"}})
    rows = [ph_tree_row(d, reversion_verdict="new_contamination")]
    agg._populate_reverted_mutations(rows, tmp_path)
    assert rows[0]["steered_mutated_contact_positions"] == "5,7"
    assert rows[0]["reverted_mutated_contact_positions"] == "7"
    assert rows[0]["reverted_total_mutations"] == ""


@pytest.mark.logic
def test_a_row_that_never_went_through_reversion_stays_blank(agg, tmp_path):
    d = mut_design(tmp_path / "design_00", [5, 7])
    cycle0(tmp_path, plan={"pred_receptor_chain": "A"},
           contaminated={"contaminated": [
               {"label": "design_00", "positions_to_revert": [7]}]})
    rows = [ph_tree_row(d, reversion_verdict="")]
    agg._populate_reverted_mutations(rows, tmp_path)
    assert rows[0]["steered_mutated_contact_positions"] == ""
    assert rows[0]["reverted_mutated_contact_positions"] == ""


@pytest.mark.logic
def test_the_reversion_plan_is_the_fallback_source_of_reverted_positions(
        agg, tmp_path):
    """contaminated.json is authoritative, but a label it does not carry
    falls back to reversion_plan.json. Reverting 5 and 12 leaves position 7,
    so one mutation remains."""
    d = mut_design(tmp_path / "design_00", [5, 7, 12])
    cycle0(tmp_path, plan={"pred_receptor_chain": "A"},
           contaminated={"contaminated": []},
           results={"design_00": {}},
           rev_plan={"entries": [
               {"label": "design_00", "positions_to_revert": [5, 12]}]})
    rows = [ph_tree_row(d)]
    agg._populate_reverted_mutations(rows, tmp_path)
    assert rows[0]["reverted_total_mutations"] == 1
    assert rows[0]["reverted_mutations_chimerax"] == "/A:7"
    assert rows[0]["reverted_mutations_aa"] == "A7K"


@pytest.mark.logic
def test_two_designs_in_one_workdir_read_the_plan_once(agg, tmp_path):
    """The reads are cached per workdir. Each design still gets its own
    positions out of the shared plan."""
    d0 = mut_design(tmp_path / "design_00", [5, 7])
    d1 = mut_design(tmp_path / "design_01", [8, 9])
    cycle0(tmp_path, plan={"pred_receptor_chain": "A"},
           contaminated={"contaminated": []},
           rev_plan={"entries": [
               {"label": "design_00", "positions_to_revert": [5]},
               {"label": "design_01", "positions_to_revert": [9]}]})
    rows = [ph_tree_row(d0, label="design_00"),
            ph_tree_row(d1, label="design_01")]
    agg._populate_reverted_mutations(rows, tmp_path)
    assert rows[0]["reverted_mutations_aa"] == "A7K"
    assert rows[1]["reverted_mutations_aa"] == "A8K"


@pytest.mark.logic
def test_a_design_with_no_mutations_file_reverts_to_nothing(agg, tmp_path):
    """A missing mutations.tsv gives an empty cumulative set, so the reverted
    set is empty rather than a crash mid-aggregation."""
    d = tmp_path / "design_00"
    d.mkdir()
    (d / "prediction.pdb").write_text("ATOM\n")
    cycle0(tmp_path, plan={"pred_receptor_chain": "A"},
           contaminated={"contaminated": [
               {"label": "design_00", "positions_to_revert": [7]}]})
    rows = [ph_tree_row(d)]
    agg._populate_reverted_mutations(rows, tmp_path)
    assert rows[0]["reverted_total_mutations"] == 0
    assert rows[0]["reverted_mutations_chimerax"] == ""


@pytest.mark.logic
def test_a_row_with_no_prediction_path_reverts_to_nothing(agg, tmp_path):
    cycle0(tmp_path, plan={"pred_receptor_chain": "A"},
           contaminated={"contaminated": [
               {"label": "design_00", "positions_to_revert": [7]}]})
    rows = [{"cycle": 0, "design": "design_00", "pathway": "cycle_0",
             "parent_pathway": "", "reversion_verdict": "pose_holds",
             "pdb": ""}]
    agg._populate_reverted_mutations(rows, tmp_path)
    assert rows[0]["reverted_total_mutations"] == 0


@pytest.mark.logic
def test_an_unparseable_plan_falls_back_to_chain_a(agg, tmp_path):
    """Chain A is the default receptor chain. A plan that will not parse must
    not stop the mutation strings being written."""
    d = mut_design(tmp_path / "design_00", [5, 7])
    cycle0(tmp_path, plan="{ not json",
           contaminated={"contaminated": [
               {"label": "design_00", "positions_to_revert": [7]}]})
    rows = [ph_tree_row(d)]
    agg._populate_reverted_mutations(rows, tmp_path)
    assert rows[0]["reverted_mutations_chimerax"] == "/A:5"
    assert rows[0]["reverted_contact_cutoff_used"] == ""


@pytest.mark.logic
@pytest.mark.parametrize("which", ["contaminated", "results", "rev_plan"])
def test_an_unparseable_sidecar_leaves_the_columns_empty(agg, tmp_path, which):
    """A truncated JSON sidecar is treated as absent. Raising here would
    abort the whole aggregation over one unreadable file."""
    d = mut_design(tmp_path / "design_00", [5, 7])
    kw = {"contaminated": {"contaminated": []}, "results": {},
          "rev_plan": {"entries": []}}
    kw[which] = "{ truncated"
    cycle0(tmp_path, plan={"pred_receptor_chain": "A"}, **kw)
    rows = [ph_tree_row(d)]
    agg._populate_reverted_mutations(rows, tmp_path)
    assert rows[0]["reverted_mutated_contact_positions"] == ""
    assert rows[0]["reverted_total_mutations"] == 2


@pytest.mark.logic
def test_a_later_cycle_row_is_keyed_by_its_leaf_label(agg, tmp_path):
    """Cycle-0 rows carry the per-design name in `design`, because their
    `pathway` is the constant "cycle_0". A later cycle's per-design identity
    is in `pathway`, and its sidecars live under
    cycle_N/pathway_<parent>/."""
    d = mut_design(tmp_path / "c1d03", [5, 7, 12])
    wd = tmp_path / "cycle_1" / "pathway_c0d05"
    wd.mkdir(parents=True)
    (wd / "contaminated.json").write_text(json.dumps({"contaminated": []}))
    (wd / "reversion_results.json").write_text(json.dumps(
        {"c0d05.c1d03": {"reverted_mutated_contact_positions": [12]}}))
    (wd / "reversion_plan.json").write_text(json.dumps(
        {"entries": [{"label": "c0d05.c1d03",
                      "positions_to_revert": [5]}]}))
    cycle0(tmp_path, plan={"pred_receptor_chain": "A"})
    rows = [{"cycle": 1, "design": "design_03", "pathway": "c0d05.c1d03",
             "parent_pathway": "c0d05", "reversion_verdict": "pose_holds",
             "pdb": str(d / "prediction.pdb")}]
    agg._populate_reverted_mutations(rows, tmp_path)
    assert rows[0]["reverted_mutated_contact_positions"] == "12"
    assert rows[0]["reverted_mutations_aa"] == "A7K A12K"


# ── the aggregate stage's cycle-0 reconstruction ─────────────────────────────


@pytest.mark.logic
def test_a_tree_with_neither_a_table_nor_a_prediction_says_so(
        agg, tmp_path, capsys):
    """A cycle-0 directory with a plan but no outputs is a run that never
    produced anything. The stage still writes a header-only table so readers
    get an empty table instead of a FileNotFoundError."""
    c0 = tmp_path / "cycle_0"
    c0.mkdir()
    (c0 / "plan.json").write_text(json.dumps({"designs": []}))
    assert agg.cmd_aggregate(Args(experiment_root=tmp_path)) == 0
    assert "no steered_results.csv AND no" in capsys.readouterr().out
    with open(tmp_path / "all_results_multicycle.csv") as f:
        assert list(csv.DictReader(f)) == []


@pytest.mark.logic
def test_a_reconstructed_baseline_with_no_truth_set_leaves_jaccard_blank(
        agg, tmp_path):
    """Without a true interface there is nothing to compute an overlap
    against. Writing 0.0 would report the baseline as having no overlap with
    the truth, which is a measurement rather than a missing one."""
    c0 = tmp_path / "cycle_0"
    c0.mkdir()
    (c0 / "plan.json").write_text(json.dumps({
        "designs": [], "initial_receptor_aligned_effector_rmsd": 23.9,
        "initial_independent_receptor_rmsd": 1.0,
        "initial_independent_effector_rmsd": 1.0,
        "initial_wrong_interface_idx": [40, 41], "true_interface_idx": []}))
    (c0 / "initial_prediction.pdb").write_text("ATOM\n")
    assert agg.cmd_aggregate(Args(experiment_root=tmp_path)) == 0
    with open(tmp_path / "all_results_multicycle.csv") as f:
        row = list(csv.DictReader(f))[0]
    assert row["design"] == "initial"
    assert row["steered_true_jaccard"] == ""
    assert row["steered_n_shared_true"] == ""
    assert row["steered_wrong_jaccard"] == ""
    assert row["steered_receptor_intact"] == "1"


# ── the Ca counter and the float parser ──────────────────────────────────────
#
# Chain lengths go to compute_metrics.py as --chain-lengths. A miscount makes
# the two chains swap, and every interface metric is then computed on the
# wrong pair.


@pytest.mark.logic
def test_only_ca_atoms_of_atom_records_are_counted(agg, tmp_path):
    """HETATM and TER lines are not residues of the chain, and a CB atom is
    the same residue counted twice."""
    pdb = tmp_path / "p.pdb"
    pdb.write_text(
        "REMARK  generated\n"
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 80.00\n"
        "ATOM      2  CB  ALA A   1       1.000   0.000   0.000  1.00 80.00\n"
        "ATOM      3  CA  GLY A   2       2.000   0.000   0.000  1.00 80.00\n"
        "TER       4      GLY A   2\n"
        "HETATM    5  CA  HOH B   1       3.000   0.000   0.000  1.00 80.00\n"
        "ATOM      6  CA  LEU B   1       4.000   0.000   0.000  1.00 80.00\n")
    assert agg._count_ca_per_chain(pdb) == {"A": 2, "B": 1}


@pytest.mark.logic
def test_an_unreadable_pdb_counts_nothing_and_warns(agg, tmp_path, capsys):
    """An empty count is what the caller treats as "cannot determine the
    chains". Raising would end the stage on one bad file."""
    assert agg._count_ca_per_chain(tmp_path / "missing.pdb") == {}
    assert "failed to count CA atoms" in capsys.readouterr().err


@pytest.mark.logic
@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
def test_a_non_finite_metric_parses_as_none(agg, bad):
    """nan fails every comparison, so `v < threshold` on a nan silently
    rejects. None makes the missing value explicit."""
    assert agg._float_or_none(bad) is None


@pytest.mark.logic
@pytest.mark.parametrize("bad", ["n/a", "2.0A", [], {}])
def test_an_unparseable_metric_parses_as_none(agg, bad):
    assert agg._float_or_none(bad) is None


@pytest.mark.logic
def test_a_cold_start_exemption_still_rejects_an_unreadable_intact_flag(agg):
    """The skip_steering baseline is exempt from the metric threshold but not
    from needing an intact receptor. A garbled flag is not an intact one."""
    row = {"design": "initial", "cycle": "0",
           "steered_receptor_intact": "yes", "ra": "1.0"}
    assert agg._passes_final_metrics_filter(
        row, Args(), "ra", 5.0, True) is False


# ── the per-sequence stage's aggregate ranker ────────────────────────────────
#
# The ranker runs a second time on the collapsed rows, keyed on the _median
# columns. It has to make the same steered-or-reverted choice the per-seed
# ranker makes, or a pose_holds sequence is ranked on the contaminated
# structure that was discarded.

@pytest.mark.logic
def test_an_interpreter_that_cannot_be_executed_is_recorded_not_raised(
        agg, tmp_path):
    """Every non-timeout failure of the compute_metrics subprocess has to land
    as a steered_error on the row. An exception here ends the stage after
    hours of per-row subprocess work and loses every row already computed."""
    out, status = _one_row(agg, tmp_path,
                           python_executable=str(tmp_path / "designs"))
    assert status == "failed"
    assert out["steered_error"].startswith("subprocess failed:")
    assert out["steered_iptm"] == ""


@pytest.mark.logic
def test_a_verbose_failure_prints_the_script_stdout(agg, tmp_path, capsys):
    """The exit code and stderr tail go on the row. stdout only appears under
    --verbose, and it is where compute_metrics.py reports which chain pair it
    actually used."""
    script = tmp_path / "stub_noisy.py"
    script.write_text("import sys\n"
                      "print('chains A/B, 12 and 7 residues')\n"
                      "sys.stderr.write('no PAE sidecar\\n')\n"
                      "sys.exit(3)\n")
    out, status = _one_row(agg, tmp_path, script=script, verbose=True)
    assert status == "failed"
    assert "exited with 3" in out["steered_error"]
    assert "chains A/B, 12 and 7 residues" in capsys.readouterr().out


def _one_row(agg, tmp_path, *, script=None, **argkw):
    """Drive _process_passing_row once over a minimal Boltz-shaped design."""
    design = tmp_path / "designs" / "d0"
    design.mkdir(parents=True, exist_ok=True)
    pdb = "".join(
        f"ATOM  {i:5d}  CA  ALA {c}{i:4d}       0.000   0.000   0.000"
        f"  1.00 80.00\n"
        for c, i in (("A", 1), ("A", 2), ("B", 3)))
    (design / "prediction.pdb").write_text(pdb)
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    plan_path = tmp_path / "cycle_0" / "plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text("{}")
    args = Args(skip_existing=False, pae_cutoff=10.0, contact_cutoff=4.5,
                subprocess_timeout=60, verbose=False,
                python_executable=sys.executable)
    args.__dict__.update(argkw)
    if script is None:
        script = tmp_path / "stub_ok.py"
        script.write_text("import sys\nsys.exit(0)\n")
    return agg._process_passing_row(
        row={"cycle": "0", "design": "d0", "pathway": "cycle_0",
             "ra": "2.0", "pdb": str(design / "prediction.pdb")},
        args=args,
        prefixed_metric_keys=["steered_iptm", "steered_ptm"],
        metric_keys_raw=["iptm", "ptm"],
        metric_key_prefix="steered_",
        existing_by_key={},
        scratch_dir=scratch,
        compute_metrics_script=script,
        cycle_0_plan={},
        cycle_0_plan_path=plan_path,
        pred_rec_chain="A",
        pred_eff_chain="B",
        model_name="boltz2",
        metric_col="ra",
        passing_progress=1,
        n_passing_total=1,
    )


PER_SEED_COLUMNS = [
    "cycle", "pathway", "design", "sequence_group", "seed_index", "status",
    "pdb", "steered_total_mutations", "steered_ra_eff_vs_truth",
    "steered_independent_receptor_rmsd", "steered_independent_effector_rmsd",
    "steered_receptor_intact", "steered_true_jaccard",
    "steered_n_contacts_on_mutated_positions", "reversion_verdict",
    "reverted_ra_eff_vs_truth", "reverted_receptor_intact",
    "reverted_independent_receptor_rmsd", "reverted_independent_effector_rmsd",
    "reverted_mutated_contact_positions", "reverted_total_mutations",
]


def seed(group, idx, **kw):
    r = {"cycle": "0", "pathway": "cycle_0", "design": f"design_{group}",
         "sequence_group": str(group), "seed_index": str(idx), "status": "ok",
         "pdb": f"/runs/g{group}_s{idx}.pdb", "steered_total_mutations": "3",
         "steered_ra_eff_vs_truth": "2.0",
         "steered_independent_receptor_rmsd": "1.0",
         "steered_independent_effector_rmsd": "1.0",
         "steered_receptor_intact": "1", "steered_true_jaccard": "0.8",
         "steered_n_contacts_on_mutated_positions": "0",
         "reversion_verdict": "", "reverted_ra_eff_vs_truth": "",
         "reverted_receptor_intact": "",
         "reverted_independent_receptor_rmsd": "",
         "reverted_independent_effector_rmsd": "",
         "reverted_mutated_contact_positions": "",
         "reverted_total_mutations": ""}
    r.update({k: str(v) for k, v in kw.items()})
    return r


def ph_seed(group, idx, *, reverted_ra=2.0, steered_ra=9.0, **kw):
    return seed(group, idx, reversion_verdict="pose_holds",
                steered_ra_eff_vs_truth=steered_ra,
                reverted_ra_eff_vs_truth=reverted_ra,
                reverted_receptor_intact="1",
                reverted_independent_receptor_rmsd="1.0",
                reverted_independent_effector_rmsd="1.0",
                reverted_total_mutations="2", **kw)


def write_per_seed(root: Path, rows, plan=None):
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "all_results_multicycle.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PER_SEED_COLUMNS,
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    if plan is not None:
        c0 = root / "cycle_0"
        c0.mkdir(parents=True, exist_ok=True)
        (c0 / "plan.json").write_text(json.dumps(plan))
    return root


def aggregated(root: Path):
    with open(root / "aggregated_results.csv") as f:
        return list(csv.DictReader(f))


@pytest.mark.logic
def test_a_pose_holds_sequence_is_ranked_on_its_reverted_pose(agg, tmp_path):
    """Steered median 9.0 A, reverted median 2.0 A, jaccard median 0.8. The
    rank has to come off the reverted pose, so the composite is
    0.8 - 0.05 x 2.0 = 0.7 and not 0.8 - 0.05 x 9.0 = 0.35."""
    write_per_seed(tmp_path, [ph_seed(1, s) for s in range(3)])
    assert agg.cmd_aggregate_per_sequence(Args(experiment_root=tmp_path)) == 0
    row = aggregated(tmp_path)[0]
    assert row["outcome"] == "pose_holds"
    assert float(row["reverted_ra_eff_vs_truth_median"]) == 2.0
    assert float(row["steered_ra_eff_vs_truth_median"]) == 9.0
    assert row["rank_by_ra_eff"] == "1"
    assert row["rank_by_composite_score"] == "1"


@pytest.mark.logic
def test_the_better_reverted_pose_outranks_the_worse_one(agg, tmp_path):
    """Both sequences hold, so they are ranked against each other on the
    reverted median. Group 2 at 1.0 A beats group 1 at 4.0 A."""
    rows = ([ph_seed(1, s, reverted_ra=4.0) for s in range(3)]
            + [ph_seed(2, s, reverted_ra=1.0) for s in range(3)])
    write_per_seed(tmp_path, rows)
    assert agg.cmd_aggregate_per_sequence(Args(experiment_root=tmp_path)) == 0
    by_group = {r["sequence_group"]: r for r in aggregated(tmp_path)}
    assert by_group["2"]["rank_by_ra_eff"] == "1"
    assert by_group["1"]["rank_by_ra_eff"] == "2"
    assert by_group["2"]["rank_by_composite_score"] == "1"


@pytest.mark.logic
def test_a_pose_holds_sequence_with_no_jaccard_still_gets_an_rmsd_rank(
        agg, tmp_path):
    """The composite needs both components. A sequence missing the interface
    overlap gets a blank composite rank rather than a score computed from
    half the formula, and keeps its rmsd rank."""
    write_per_seed(tmp_path, [ph_seed(1, s, steered_true_jaccard="")
                              for s in range(3)])
    assert agg.cmd_aggregate_per_sequence(Args(experiment_root=tmp_path)) == 0
    row = aggregated(tmp_path)[0]
    assert row["outcome"] == "pose_holds"
    assert row["rank_by_ra_eff"] == "1"
    assert row["rank_by_composite_score"] == ""


@pytest.mark.logic
def test_a_sequence_with_no_steered_median_is_not_ranked(agg, tmp_path):
    """No seed produced a usable ra_eff, so there is nothing to rank on. A
    blank rank keeps it out of the figures rather than placing it first."""
    write_per_seed(tmp_path, [seed(1, s, steered_ra_eff_vs_truth="")
                              for s in range(3)])
    assert agg.cmd_aggregate_per_sequence(Args(experiment_root=tmp_path)) == 0
    row = aggregated(tmp_path)[0]
    assert row["steered_ra_eff_vs_truth_median"] == ""
    assert row["rank_by_ra_eff"] == ""
    assert row["rank_by_composite_score"] == ""


@pytest.mark.logic
def test_an_unusable_rmsd_sorts_last_when_the_canonical_seed_is_picked(
        agg, tmp_path):
    """The canonical seed names the structure a reader opens. A seed whose
    ra_eff is nan sorts to the end, so the middle of three is the 3.0 A seed
    and not the unmeasured one."""
    write_per_seed(tmp_path, [
        seed(1, 0, steered_ra_eff_vs_truth="1.0"),
        seed(1, 1, steered_ra_eff_vs_truth="3.0"),
        seed(1, 2, steered_ra_eff_vs_truth="nan")])
    assert agg.cmd_aggregate_per_sequence(Args(experiment_root=tmp_path)) == 0
    row = aggregated(tmp_path)[0]
    assert row["canonical_seed_index"] == "1"
    assert row["canonical_pdb"] == "/runs/g1_s1.pdb"
    assert float(row["steered_ra_eff_vs_truth_median"]) == 2.0


@pytest.mark.logic
def test_a_sequence_whose_receptor_collapsed_by_majority_is_not_ranked(
        agg, tmp_path):
    """Two of three seeds lost the receptor fold, so the sequence is not a
    usable binder however good its effector rmsd looks. Ranking it would put
    a collapsed structure at the top of the table."""
    write_per_seed(tmp_path, [
        seed(1, 0, steered_receptor_intact="1"),
        seed(1, 1, steered_receptor_intact="0"),
        seed(1, 2, steered_receptor_intact="0")])
    assert agg.cmd_aggregate_per_sequence(Args(experiment_root=tmp_path)) == 0
    row = aggregated(tmp_path)[0]
    assert row["steered_receptor_intact_majority"] == "0"
    assert float(row["steered_ra_eff_vs_truth_median"]) == 2.0
    assert row["rank_by_ra_eff"] == ""
    assert row["rank_by_composite_score"] == ""


@pytest.mark.logic
def test_an_unparseable_cold_start_threshold_falls_back_to_five(
        agg, tmp_path, capsys):
    """rmsd_threshold sets the looser cap for cold-start aggregates. A plan
    whose value will not parse must leave the legacy 5 A cap in place rather
    than abort the stage."""
    write_per_seed(tmp_path, [seed(1, 0)],
                   plan={"design_region_idx": [11], "rmsd_threshold": "lots"})
    assert agg.cmd_aggregate_per_sequence(Args(experiment_root=tmp_path)) == 0
    assert "contamination gating set: 1 positions" in capsys.readouterr().out
    assert aggregated(tmp_path)[0]["rank_by_ra_eff"] == "1"
