"""The statistics and selection rules behind the confidence chapters.

`analysis/04-confidence-cost/confidence_common.py` is 547 statements and is the
only analysis module imported by more than one document. Its docstring says why
it exists: `04-confidence-cost.qmd` runs on the 18-target benchmark and
`05-campaign-confidence.qmd` on the 128-sequence campaign, "they ask the same
two questions and must therefore use identical selection rules, statistics and
figures". Nothing in `tests/` imported it before this file, and it sits outside
the coverage gate's scope, so none of it was checked.

Two kinds of thing in it decide published claims.

The tests. scipy is not a dependency here, so the Wilcoxon signed-rank and the
two-sample rank-sum are both built by permutation, and the bootstrap median CI
by resampling. Each takes an explicit seed, so each is deterministic and each
has inputs whose exact p-value is derivable rather than merely plausible. A
symmetric set of paired differences must give p = 1 exactly, because the
observed signed-rank sum is 0 and every permutation of the null is an absolute
value, so every one of them is greater than or equal to it. Two identical
groups must give p = 1 for the same reason. A monotone pair must give a
Spearman of exactly 1. Those four are the anchors here.

The selection rule. `select_representative` picks one design per unit, and
every before-and-after figure in both chapters is drawn from its output. It
orders by reliability first — all seeds passing, then most, then one, then none
— and only then by median ra_eff. A unit whose representative is picked by
median alone would show its luckiest seed rather than its most reliable design.

`_final_seed_value` is underneath that. A seed that went through reversion is
scored on the reverted structure, not the steered prediction, which is the same
rule the pipeline's own aggregator applies.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE = (_REPO_ROOT / "analysis" / "04-confidence-cost"
           / "confidence_common.py")


def _deps() -> bool:
    return all(importlib.util.find_spec(m) is not None
               for m in ("numpy", "pandas", "matplotlib"))


pytestmark = [
    pytest.mark.skipif(not _MODULE.is_file(),
                       reason="confidence_common.py absent"),
    pytest.mark.skipif(not _deps(),
                       reason="needs numpy, pandas and matplotlib"),
]


@pytest.fixture(scope="module")
def cc():
    import matplotlib
    matplotlib.use("Agg")
    sys.path.insert(0, str(_MODULE.parent))
    spec = importlib.util.spec_from_file_location("confidence_common", _MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── the signed-rank test ─────────────────────────────────────────────────────
#
# Wilcoxon by sign-flip permutation. The statistic is the sum of signed ranks
# of the absolute differences and the null flips each pair's sign at random,
# which is the Wilcoxon construction. The module's own comment says why a
# permutation and not a median: "A median statistic is too coarse at n = 15 and
# a mean is dragged around by one or two large-effect units."

@pytest.mark.logic
def test_a_symmetric_set_of_differences_is_exactly_p_one(cc):
    """The anchor. Signed ranks cancel to an observed statistic of 0, and
    every permutation's statistic is an absolute value, so every one of them
    is at least as extreme. p = (1 + n) / (n + 1) = 1 exactly."""
    assert cc.signed_rank_p([1.0, -1.0, 2.0, -2.0, 3.0, -3.0]) == 1.0


@pytest.mark.logic
def test_a_consistent_improvement_is_significant(cc):
    """Ten differences all the same sign. Only 2 of the 2^10 sign patterns
    reach the observed statistic, so p is of order 2/1024."""
    p = cc.signed_rank_p([0.4, 0.9, 1.3, 2.1, 0.6, 1.8, 2.4, 0.7, 1.1, 3.0])
    assert p < 0.01


@pytest.mark.logic
def test_the_magnitudes_matter_not_just_the_signs(cc):
    """The ranks are over the absolute differences, so one large opposing
    difference outweighs several small agreeing ones. A sign test would miss
    this, which is why the statistic is a rank sum."""
    agree = cc.signed_rank_p([0.1, 0.1, 0.1, 0.1, 0.1, 10.0])
    oppose = cc.signed_rank_p([0.1, 0.1, 0.1, 0.1, 0.1, -10.0])
    assert agree < oppose


@pytest.mark.logic
def test_the_test_is_deterministic_for_a_given_seed(cc):
    """A published p-value has to be the same number on the next run. The
    permutation draw is seeded for exactly that reason."""
    diffs = [0.4, -0.2, 1.3, 0.1, -0.9, 2.1, 0.3]
    assert cc.signed_rank_p(diffs) == cc.signed_rank_p(diffs)
    assert cc.signed_rank_p(diffs, seed=0) != cc.signed_rank_p(diffs, seed=7)


@pytest.mark.logic
def test_zero_differences_are_dropped_not_counted(cc):
    """A unit that did not move carries no information about direction.
    Ranking it would dilute the statistic toward no effect."""
    assert cc.signed_rank_p([1.0, -1.0, 0.0, 0.0]) == cc.signed_rank_p(
        [1.0, -1.0])


@pytest.mark.logic
def test_a_set_of_only_zeros_has_no_p_value(cc):
    import numpy as np
    assert np.isnan(cc.signed_rank_p([0.0, 0.0, 0.0]))


@pytest.mark.logic
def test_non_finite_differences_are_dropped(cc):
    """A unit whose metric was never computed must not be read as no change."""
    import numpy as np
    assert cc.signed_rank_p([1.0, -1.0, np.nan, np.inf]) == cc.signed_rank_p(
        [1.0, -1.0])


@pytest.mark.logic
def test_an_empty_set_has_no_p_value(cc):
    import numpy as np
    assert np.isnan(cc.signed_rank_p([]))


@pytest.mark.logic
def test_the_p_value_can_never_be_zero(cc):
    """The estimator adds one to both the count and the draw, so the smallest
    reportable value is 1/(n+1). A p of exactly 0 from 20,000 permutations
    would be a claim the method cannot support."""
    p = cc.signed_rank_p([5.0] * 30, n=200)
    assert p == pytest.approx(1.0 / 201, abs=1e-12)
    assert p > 0.0


# ── the bootstrap median interval ────────────────────────────────────────────

@pytest.mark.logic
def test_a_constant_set_has_a_zero_width_interval(cc):
    """Every resample is the same set, so both percentiles are the median.
    An interval that opened up here would be an artefact of the resampling."""
    med, lo, hi = cc.median_ci([5.0] * 20)
    assert (med, lo, hi) == (5.0, 5.0, 5.0)


@pytest.mark.logic
def test_the_interval_brackets_the_median(cc):
    med, lo, hi = cc.median_ci([0.1, 0.4, 0.9, 1.2, 1.6, 2.0, 2.4, 3.1])
    assert lo <= med <= hi


@pytest.mark.logic
def test_the_reported_median_is_of_the_data_not_the_resamples(cc):
    """A bootstrap median of medians is biased on an even-sized set. The
    point estimate has to be the observed median."""
    import numpy as np
    data = [0.1, 0.4, 0.9, 1.2, 1.6, 2.0]
    med, _, _ = cc.median_ci(data)
    assert med == pytest.approx(float(np.median(data)))


@pytest.mark.logic
def test_an_empty_set_has_no_interval(cc):
    import numpy as np
    assert all(np.isnan(x) for x in cc.median_ci([]))


@pytest.mark.logic
def test_non_finite_values_are_dropped_before_resampling(cc):
    import numpy as np
    assert cc.median_ci([1.0, 2.0, 3.0, np.nan]) == cc.median_ci(
        [1.0, 2.0, 3.0])


@pytest.mark.logic
def test_the_interval_endpoints_are_observed_values(cc):
    """A bootstrap resamples with replacement, so the median of any resample
    of a 7-point set is one of those 7 points. The interval is therefore
    quantised to the data, and on a set this small it does not move with the
    seed at all. Both chapters report intervals on 15 and 18 units, so this
    is the regime they are in."""
    data = [0.1, 0.4, 0.9, 1.2, 1.6, 2.0, 2.4]
    med, lo, hi = cc.median_ci(data)
    assert {lo, hi} <= set(data)
    assert cc.median_ci(data, seed=1) == cc.median_ci(data, seed=99)


# ── the two-sample rank-sum test ─────────────────────────────────────────────
#
# The unpaired counterpart, for comparing rescued units against still-wrong
# ones. The module's comment states the reason it is not the paired test:
# "These are different units rather than a before/after of the same unit, so
# the paired test does not apply."

@pytest.mark.logic
def test_two_identical_groups_are_exactly_p_one(cc):
    """The second anchor. Group a's rank sum equals its expectation, so the
    observed deviation is 0 and every permutation matches or exceeds it."""
    g = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert cc.rank_sum_p(g, list(g)) == 1.0


@pytest.mark.logic
def test_fully_separated_groups_are_significant(cc):
    """Every value in a below every value in b. Only the permutations that
    recover that exact split are as extreme."""
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    b = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    assert cc.rank_sum_p(a, b) < 0.01


@pytest.mark.logic
def test_the_groups_are_symmetric_only_to_monte_carlo_error(cc):
    """The observed statistic is a two-sided deviation from the expected rank
    sum, so it does not depend on which group is named first. The null draw
    does: `argsort(rng.random((n, pooled.size)))` indexes into the pooled
    array, and swapping the groups changes that array's order. On this input
    the two orders give 0.0526 and 0.0539, a gap of the size one standard
    error of a 20,000-draw estimate at p = 0.05. A reported p that close to
    a threshold can therefore sit on either side of it depending on argument
    order, which is a property of the permutation estimate and not of the
    data."""
    a = [1.0, 4.0, 2.0, 8.0, 3.0]
    b = [9.0, 5.0, 7.0, 6.0, 10.0]
    forward, reverse = cc.rank_sum_p(a, b), cc.rank_sum_p(b, a)
    assert forward == pytest.approx(reverse, abs=0.005)
    assert forward != reverse


@pytest.mark.logic
def test_it_works_on_groups_of_different_sizes(cc):
    """Rescued and still-wrong are rarely the same size, so an implementation
    that assumed balance would be wrong on every real comparison."""
    p = cc.rank_sum_p([1.0, 2.0], [8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
    assert 0.0 < p < 1.0


@pytest.mark.logic
def test_an_empty_group_has_no_p_value(cc):
    import numpy as np
    assert np.isnan(cc.rank_sum_p([], [1.0, 2.0]))
    assert np.isnan(cc.rank_sum_p([1.0, 2.0], []))


@pytest.mark.logic
def test_non_finite_values_are_dropped_from_both_groups(cc):
    import numpy as np
    assert cc.rank_sum_p([1.0, 2.0, np.nan], [8.0, 9.0]) == cc.rank_sum_p(
        [1.0, 2.0], [8.0, 9.0])


# ── Spearman ─────────────────────────────────────────────────────────────────

@pytest.mark.logic
def test_a_monotone_pair_correlates_one(cc):
    """The third anchor. It is a correlation of ranks, so any increasing
    transform of one axis gives 1. np.corrcoef returns it to within 1e-16
    rather than exactly, which is why this is an approx."""
    assert cc.spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == pytest.approx(
        1.0, abs=1e-12)


@pytest.mark.logic
def test_a_reversed_pair_correlates_minus_one(cc):
    assert cc.spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) == pytest.approx(
        -1.0, abs=1e-12)


@pytest.mark.logic
def test_a_non_linear_monotone_pair_still_correlates_one(cc):
    """This is the reason the rank correlation is used rather than Pearson.
    ipSAE against ipTM is monotone but not linear."""
    assert cc.spearman([1, 2, 3, 4, 5], [1, 4, 9, 16, 25]) == pytest.approx(
        1.0, abs=1e-12)


@pytest.mark.logic
def test_pairs_with_a_missing_value_are_dropped(cc):
    """Dropping the pair, not the value. Keeping one side would pair a metric
    with another unit's."""
    import numpy as np
    assert cc.spearman(
        [1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, np.nan]) == pytest.approx(
            1.0, abs=1e-12)


@pytest.mark.logic
def test_fewer_than_four_usable_pairs_has_no_correlation(cc):
    """A correlation over three points is not a measurement."""
    import numpy as np
    assert np.isnan(cc.spearman([1, 2, 3], [1, 2, 3]))
    assert np.isnan(cc.spearman([1, 2, 3, 4], [1, 2, np.nan, np.nan]))


# ── the reliability-first ordering ───────────────────────────────────────────
#
# Its docstring: "Reliability-first ordering: all seeds pass, then most, then
# one, then none." A design whose every seed passes beats a design with a
# better median that only passed once, because the second one cannot be
# relied on to reproduce.

@pytest.mark.logic
@pytest.mark.parametrize("n_pass,n_seeds,rank", [
    (4, 4, 0),        # every seed passed
    (3, 4, 1),        # most of them
    (2, 4, 1),        # still more than one
    (1, 4, 2),        # exactly one
    (0, 4, 3),        # none
    (1, 1, 0),        # a single-seed design that passed is "all seeds"
    (0, 1, 3),
])
def test_the_rank_orders_by_how_many_seeds_passed(cc, n_pass, n_seeds, rank):
    assert cc._rank(n_pass, n_seeds) == rank


@pytest.mark.logic
def test_a_design_with_no_seeds_ranks_last(cc):
    """Not first. An empty set passes vacuously, and ranking it top would put
    a design with no data above every real one."""
    assert cc._rank(0, 0) == 3


# ── the per-seed final value ─────────────────────────────────────────────────
#
# A seed whose steering mutations touched a protected position goes through
# reversion, and its real final value is the reverted structure's ra_eff. This
# is the same rule the pipeline's aggregator applies, and reading the steered
# value instead reports a collapsed reversion as a success.

def _row(**kw):
    base = {"reversion_verdict": "", "reverted_ra_eff_vs_truth": "",
            "steered_n_contacts_on_mutated_positions": "0",
            "steered_receptor_intact": "1", "ra_eff": "2.0"}
    base.update(kw)
    return base


@pytest.mark.logic
def test_a_held_reversion_is_scored_on_the_reverted_structure(cc):
    ra, ok = cc._final_seed_value(_row(reversion_verdict="pose_holds",
                                       reverted_ra_eff_vs_truth="3.2",
                                       ra_eff="2.0"))
    assert ra == 3.2
    assert ok is True


@pytest.mark.logic
def test_a_held_reversion_above_the_threshold_does_not_pass(cc):
    ra, ok = cc._final_seed_value(_row(reversion_verdict="pose_holds",
                                       reverted_ra_eff_vs_truth="7.5"))
    assert ra == 7.5
    assert ok is False


@pytest.mark.logic
@pytest.mark.parametrize("verdict", ["pose_collapses", "new_contamination",
                                     "pose_holds_not_intact",
                                     "missing_verdict", "unknown"])
def test_a_failed_reversion_never_passes(cc, verdict):
    """Its value is still reported, so the figure can show where it landed,
    but it cannot count toward the design's pass count."""
    ra, ok = cc._final_seed_value(_row(reversion_verdict=verdict,
                                       reverted_ra_eff_vs_truth="2.0",
                                       ra_eff="2.0"))
    assert ra == 2.0
    assert ok is False


@pytest.mark.logic
def test_a_seed_that_never_needed_reversion_uses_its_steered_value(cc):
    """A blank verdict with zero contacts on the mutated positions means
    reversion was correctly skipped, so the steered prediction is final."""
    ra, ok = cc._final_seed_value(_row(ra_eff="1.5"))
    assert ra == 1.5
    assert ok is True


@pytest.mark.logic
def test_a_broken_receptor_does_not_pass(cc):
    """Intactness is part of passing, because a collapsed receptor can still
    place the effector close to the truth."""
    ra, ok = cc._final_seed_value(_row(steered_receptor_intact="0"))
    assert ra == 2.0
    assert ok is False


@pytest.mark.logic
def test_a_blank_verdict_with_contacts_on_mutations_has_no_value(cc):
    """Contacts on the mutated positions with no verdict means the reversion
    pass should have run and its result is missing. Reporting the steered
    value would report a number the construct cannot reproduce."""
    ra, ok = cc._final_seed_value(
        _row(steered_n_contacts_on_mutated_positions="2"))
    assert ra is None
    assert ok is False


@pytest.mark.logic
def test_the_threshold_is_exclusive(cc):
    """5.0 A is the pass cutoff everywhere in this project, and every gate
    uses < rather than <=."""
    assert cc.RA_EFF_THRESHOLD == 5.0
    _, ok_under = cc._final_seed_value(_row(ra_eff="4.99"))
    _, ok_at = cc._final_seed_value(_row(ra_eff="5.0"))
    assert ok_under is True
    assert ok_at is False


@pytest.mark.logic
def test_an_unparseable_number_is_no_value(cc):
    assert cc._num("not a number") is None
    assert cc._num(None) is None
    assert cc._num(float("nan")) is None
    assert cc._num("3.5") == 3.5


# ── choosing a unit's representative design ──────────────────────────────────

def _steered(rows):
    import pandas as pd
    return pd.DataFrame(rows)


@pytest.mark.logic
def test_the_most_reliable_design_wins_over_the_better_median(cc):
    """design_a passes on both seeds at 4.0 A. design_b has a better median
    but only passes once. Picking by median would show the luckier design as
    the unit's result."""
    import pandas as pd
    rows = []
    for seed in (0, 1):
        rows.append(_row(ra_eff="4.0") | {"base": "design_a"})
    rows.append(_row(ra_eff="1.0") | {"base": "design_b"})
    rows.append(_row(ra_eff="40.0") | {"base": "design_b"})
    got = cc.select_representative(_steered(rows), pd.DataFrame())
    assert got["rep_design"] == "design_a"
    assert got["rep_n_pass"] == 2
    assert got["rep_n_seeds"] == 2
    assert got["rep_ra_eff"] == 4.0


@pytest.mark.logic
def test_the_median_breaks_a_tie_on_reliability(cc):
    """Both designs pass on every seed, so the better pose wins."""
    import pandas as pd
    rows = [_row(ra_eff="4.0") | {"base": "design_a"},
            _row(ra_eff="1.5") | {"base": "design_b"}]
    got = cc.select_representative(_steered(rows), pd.DataFrame())
    assert got["rep_design"] == "design_b"
    assert got["rep_ra_eff"] == 1.5


@pytest.mark.logic
def test_the_representative_value_is_the_median_across_seeds(cc):
    """Not the best seed. One lucky seed is not what the design would give
    again."""
    import pandas as pd
    rows = [_row(ra_eff=v) | {"base": "design_a"}
            for v in ("1.0", "4.0", "9.0")]
    got = cc.select_representative(_steered(rows), pd.DataFrame())
    assert got["rep_ra_eff"] == 4.0
    assert got["rep_n_pass"] == 2


@pytest.mark.logic
def test_a_reverted_seed_contributes_its_reverted_value(cc):
    """The whole point of routing through _final_seed_value. This design's
    steered numbers are good and its reversion collapsed, so its
    representative value has to be the collapsed one."""
    import pandas as pd
    rows = [_row(reversion_verdict="pose_collapses",
                 reverted_ra_eff_vs_truth="30.0", ra_eff="2.0")
            | {"base": "design_a"}]
    got = cc.select_representative(_steered(rows), pd.DataFrame())
    assert got["rep_ra_eff"] == 30.0
    assert got["rep_n_pass"] == 0


@pytest.mark.logic
def test_a_short_circuited_unit_falls_back_to_its_cold_start(cc):
    """When the cold start already passed on every seed the engine skips
    steering, so there are no steered rows. The unit still has a result, and
    dropping it would remove an already-correct target from the comparison."""
    import pandas as pd
    cold = pd.DataFrame([{"ra_eff": 1.0}, {"ra_eff": 1.4}, {"ra_eff": 1.2}])
    got = cc.select_representative(pd.DataFrame(columns=["base"]), cold)
    assert got["rep_design"] == "initial"
    assert got["rep_ra_eff"] == 1.2
    assert got["rep_n_pass"] == 3
    assert got["rep_n_seeds"] == 3


@pytest.mark.logic
def test_the_cold_start_fallback_only_fires_with_no_steered_candidate(cc):
    """A unit that has both must be scored on its steered designs, or
    steering would be credited with the cold start's result."""
    import pandas as pd
    cold = pd.DataFrame([{"ra_eff": 0.5}])
    rows = [_row(ra_eff="4.0") | {"base": "design_a"}]
    got = cc.select_representative(_steered(rows), cold)
    assert got["rep_design"] == "design_a"


@pytest.mark.logic
def test_a_unit_with_nothing_usable_has_no_representative(cc):
    """None rather than a row of blanks, so the caller drops the unit rather
    than plotting a point at an arbitrary place."""
    import pandas as pd
    rows = [_row(steered_n_contacts_on_mutated_positions="3")
            | {"base": "design_a"}]
    assert cc.select_representative(_steered(rows), pd.DataFrame()) is None


@pytest.mark.logic
def test_a_design_whose_every_seed_is_unusable_is_not_a_candidate(cc):
    """design_a has no final value on any seed, so design_b represents the
    unit even though design_a would have ranked first on pass count."""
    import pandas as pd
    rows = [_row(steered_n_contacts_on_mutated_positions="3")
            | {"base": "design_a"},
            _row(ra_eff="20.0") | {"base": "design_b"}]
    got = cc.select_representative(_steered(rows), pd.DataFrame())
    assert got["rep_design"] == "design_b"
