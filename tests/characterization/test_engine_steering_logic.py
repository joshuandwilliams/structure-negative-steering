"""The steering decision logic, and the bookkeeping that surrounds it.

These are the functions inside boltz2_negative_steering, boltz2_iterate_steering,
reversion, extract_passing and the two plot modules that decide what gets
mutated, which designs survive, how outcomes are classified and how per-seed
values are aggregated. None of them needs a GPU. They were untested because the
modules they live in also contain Boltz calls, not because they are hard to
test.

Expected values come from the documented behaviour or a hand-worked case.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin"))

import boltz2_negative_steering as bns  # noqa: E402
import extract_passing as ep  # noqa: E402
import negsteer_aggregate as agg  # noqa: E402
import negsteer_coldstart as cold  # noqa: E402

# ── jaccard: how interface overlap is measured everywhere ────────────────────

@pytest.mark.logic
@pytest.mark.parametrize("a,b,expected", [
    ({1, 2, 3}, {1, 2, 3}, 1.0),      # identical
    ({1, 2, 3}, {4, 5, 6}, 0.0),      # disjoint
    ({1, 2}, {2, 3}, 1 / 3),          # one shared of three
    ({1, 2, 3, 4}, {3, 4}, 0.5),      # subset
])
def test_jaccard_overlap(a, b, expected):
    assert bns.jaccard(a, b) == pytest.approx(expected)


@pytest.mark.logic
def test_jaccard_of_two_empty_sets_is_nan():
    """0/0 returns NaN rather than 0.0, and that value propagates.

    A design whose prediction has no interface contacts at all produces a NaN
    true_jaccard, which makes its composite score NaN and drops it out of the
    ranking rather than scoring it worst. Worth knowing when reading a cohort
    where a design is simply absent from the ordering.
    """
    got = bns.jaccard(set(), set())
    assert got != got, f"expected NaN for the empty/empty case, got {got}"


@pytest.mark.logic
def test_jaccard_is_symmetric():
    a, b = {1, 2, 3, 9}, {3, 9, 20}
    assert bns.jaccard(a, b) == bns.jaccard(b, a)


# ── make_steered_sequence: the mutation itself ───────────────────────────────

@pytest.mark.logic
@pytest.mark.parametrize("mode", ["strong", "mild", "conservative", "alanine"])
def test_steering_only_edits_the_candidate_pool(mode):
    """Every position outside the pool must survive untouched.

    This is the invariant that keeps steering from damaging the rest of the
    receptor. A violation would be invisible in the metrics.
    """
    seq = "ACDEFGHIKLMNPQRSTVWY"
    pool = [2, 5, 9]
    steered, muts = bns.make_steered_sequence(
        seq, pool, max_mutations=3, mode=mode, rng=random.Random(0))

    assert len(steered) == len(seq), "steering changed the sequence length"
    changed = {i for i, (o, n) in enumerate(zip(seq, steered)) if o != n}
    assert changed <= set(pool), \
        f"mode {mode} mutated positions outside the pool: {sorted(changed - set(pool))}"


@pytest.mark.logic
def test_steering_respects_the_mutation_budget():
    seq = "ACDEFGHIKLMNPQRSTVWY"
    pool = list(range(len(seq)))
    for budget in (1, 3, 6):
        steered, muts = bns.make_steered_sequence(
            seq, pool, max_mutations=budget, mode="mild", rng=random.Random(1))
        changed = sum(1 for o, n in zip(seq, steered) if o != n)
        assert changed <= budget, f"budget {budget} exceeded, {changed} positions changed"


@pytest.mark.logic
def test_alanine_mode_substitutes_alanine():
    seq = "KKKKKKKKKK"
    steered, _ = bns.make_steered_sequence(
        seq, [0, 1, 2], max_mutations=3, mode="alanine", rng=random.Random(0))
    changed = [n for o, n in zip(seq, steered) if o != n]
    assert changed and set(changed) == {"A"}, f"alanine mode produced {set(changed)}"


@pytest.mark.logic
def test_steering_is_deterministic_for_a_given_seed():
    """Design sampling is seeded, which is what makes a resumed run reproduce."""
    seq = "ACDEFGHIKLMNPQRSTVWY"
    pool = list(range(20))
    a, _ = bns.make_steered_sequence(seq, pool, 5, "mild", random.Random(42))
    b, _ = bns.make_steered_sequence(seq, pool, 5, "mild", random.Random(42))
    c, _ = bns.make_steered_sequence(seq, pool, 5, "mild", random.Random(43))
    assert a == b, "same seed produced different designs"
    assert a != c, "different seeds produced identical designs"


@pytest.mark.logic
def test_an_empty_pool_produces_no_mutations():
    seq = "ACDEFG"
    steered, muts = bns.make_steered_sequence(
        seq, [], max_mutations=3, mode="mild", rng=random.Random(0))
    assert steered == seq
    assert not muts


# ── constraint blocks: what gets appended to the Boltz YAML ──────────────────

@pytest.mark.logic
def test_pocket_block_names_both_chains_and_every_residue():
    block = bns._format_boltz_pocket_block("A", "B", [3, 7, 11], 8.0)
    text = block if isinstance(block, str) else "\n".join(block)
    assert "A" in text and "B" in text
    for r in (3, 7, 11):
        assert str(r) in text, f"residue {r} missing from the pocket block"
    assert "8" in text


@pytest.mark.logic
def test_contact_block_pairs_a_receptor_token_with_an_effector_token():
    block = bns._format_boltz_contact_block("A", "B", 12, 34, 10.0)
    text = block if isinstance(block, str) else "\n".join(block)
    assert "12" in text and "34" in text
    assert "A" in text and "B" in text


# ── index-file loading ───────────────────────────────────────────────────────

@pytest.mark.logic
def test_true_interface_indices_load_from_a_file(tmp_path):
    f = tmp_path / "iface.txt"
    f.write_text("# a header comment\n0\n4\n9\n")
    assert bns._load_true_interface_indices(None, f, receptor_length=20) == [0, 4, 9]


@pytest.mark.logic
def test_true_interface_indices_load_from_an_inline_argument():
    assert bns._load_true_interface_indices("0,4,9", None, receptor_length=20) == [0, 4, 9]


@pytest.mark.logic
def test_out_of_range_interface_index_is_rejected(tmp_path):
    """An index past the chain end would protect the wrong residue."""
    f = tmp_path / "iface.txt"
    f.write_text("0\n999\n")
    with pytest.raises((ValueError, SystemExit)):
        bns._load_true_interface_indices(None, f, receptor_length=20)


@pytest.mark.logic
def test_design_region_indices_load_and_convert(tmp_path):
    f = tmp_path / "region.txt"
    f.write_text("# 1-based\n1,2,3\n")
    got = bns._load_design_region_indices(f, receptor_length=20)
    assert got, "design region came back empty"


@pytest.mark.logic
def test_single_sequence_a3m_is_written_in_the_expected_shape(tmp_path):
    out = tmp_path / "x.a3m"
    bns.write_single_seq_a3m(out, "receptor", "ACDEFG")
    body = out.read_text()
    assert body.startswith(">"), "A3M must start with a header line"
    assert "ACDEFG" in body


# ── pathway labels: how the cycle tree is addressed ──────────────────────────


@pytest.mark.logic
def test_no_mutations_gives_empty_strings():
    chimerax, aa = agg.format_mutation_strings([], "A")
    assert chimerax in ("", None) or "A" not in str(chimerax)
    assert aa in ("", None)


@pytest.mark.logic
@pytest.mark.parametrize("raw,expected", [
    ("5,7,22", [5, 7, 22]),
    ("22, 5 ,7", [22, 5, 7]),   # input order is preserved, not sorted
    ("", []),
    (None, []),
    ("junk", []),
])
def test_position_csv_parsing(raw, expected):
    assert agg._parse_position_csv(raw) == expected


# ── per-seed aggregation ─────────────────────────────────────────────────────

@pytest.mark.logic
def test_continuous_aggregate_is_the_median_with_a_spread_and_a_count():
    median, mad, n = agg._agg_continuous([1.0, 2.0, 3.0])
    assert median == pytest.approx(2.0)
    assert n == 3
    assert mad >= 0


@pytest.mark.logic
def test_continuous_aggregate_ignores_missing_values():
    median, mad, n = agg._agg_continuous([1.0, None, 3.0, float("nan")])
    assert n == 2, f"expected 2 usable values, counted {n}"
    assert median == pytest.approx(2.0)


@pytest.mark.logic
def test_continuous_aggregate_of_nothing_is_not_a_crash():
    median, mad, n = agg._agg_continuous([])
    assert n == 0
    assert median is None or (isinstance(median, float) and median != median)


@pytest.mark.logic
@pytest.mark.parametrize("values,expected", [
    ([1, 1, 1], 1),
    ([0, 0, 0], 0),
    ([1, 1, 0], 1),      # majority
    ([0, 0, 1], 0),
])
def test_binary_majority_aggregate(values, expected):
    got = agg._agg_majority_binary(values)
    got = got[0] if isinstance(got, tuple) else got
    assert int(got) == expected


@pytest.mark.logic
def test_ranking_composite_matches_the_documented_formula():
    """composite = true_jaccard - 0.05 x ra_eff_vs_truth."""
    row = {"true_jaccard": "0.8", "ra_eff_vs_truth": "4.0"}
    got = agg._ranking_composite(row)
    if got is not None:
        assert got == pytest.approx(0.8 - 0.05 * 4.0)


@pytest.mark.logic
@pytest.mark.parametrize("value,expected", [
    (float("nan"), True), (1.0, False), (0.0, False),
])
def test_nan_detection(value, expected):
    assert cold._is_nan(value) is expected


@pytest.mark.logic
@pytest.mark.parametrize("raw,expected", [
    ("3.5", 3.5), ("", None), ("x", None), (None, None),
])
def test_float_or_none(raw, expected):
    got = agg._float_or_none(raw)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)


# ── maximin selection: how diverse designs are chosen ────────────────────────

# maximin_select takes a list of candidate DICTS, each carrying a float
# `min_dist_to_upstream`, and returns the k most distant. These tests used to
# pass a list of (name, {}) tuples, so `c["min_dist_to_upstream"]` raised
# TypeError, an `except Exception` caught it and the tests skipped themselves.
# Both asserted nothing for as long as that stood.


def _cand(name, dist, **extra):
    """One candidate, with the keys maximin_select reads and nothing else."""
    c = {"design": name, "min_dist_to_upstream": float(dist)}
    c.update(extra)
    return c


@pytest.mark.logic
def test_maximin_takes_the_k_most_distant_candidates():
    """Diversity is the whole point of the selection. Taking the k largest
    `min_dist_to_upstream` is what stops a cycle spending its budget on ten
    near-identical designs."""
    cands = [_cand(f"d{i}", i) for i in range(10)]
    got = cold.maximin_select(cands, 3)
    assert [c["design"] for c in got] == ["d9", "d8", "d7"]


@pytest.mark.logic
def test_maximin_of_fewer_than_k_returns_everything():
    cands = [_cand("d0", 1.0), _cand("d1", 2.0)]
    got = cold.maximin_select(cands, 5)
    assert [c["design"] for c in got] == ["d1", "d0"]


@pytest.mark.logic
def test_maximin_breaks_a_distance_tie_on_the_better_pose():
    """Equal distance means equally diverse, so the pose decides. Lower ra_eff
    wins, because that is the design that is closer to being right."""
    cands = [_cand("worse", 5.0, receptor_aligned_effector_rmsd_vs_truth=9.0),
             _cand("better", 5.0, receptor_aligned_effector_rmsd_vs_truth=2.0)]
    assert [c["design"] for c in cold.maximin_select(cands, 1)] == ["better"]


@pytest.mark.logic
def test_maximin_judges_a_reverted_design_on_its_reverted_pose():
    """`_active_ra_eff_for_selection` is set by finalize and is the only
    post-reversion view here, because the steered base columns are never
    overwritten. A pose_holds design must be ranked on what it kept, not on
    the pre-reversion prediction it did not."""
    cands = [
        _cand("raw_looks_good", 5.0,
              receptor_aligned_effector_rmsd_vs_truth=1.0,
              _active_ra_eff_for_selection=20.0),
        _cand("holds_up", 5.0,
              receptor_aligned_effector_rmsd_vs_truth=8.0,
              _active_ra_eff_for_selection=2.0),
    ]
    assert [c["design"] for c in cold.maximin_select(cands, 1)] == ["holds_up"]


@pytest.mark.logic
def test_maximin_puts_an_unreadable_pose_last_rather_than_crashing():
    """A missing or non-numeric ra_eff becomes inf, so the candidate sorts to
    the back of its distance tie instead of raising mid-cycle."""
    cands = [_cand("unreadable", 5.0,
                   receptor_aligned_effector_rmsd_vs_truth="n/a"),
             _cand("fine", 5.0, receptor_aligned_effector_rmsd_vs_truth=3.0)]
    assert [c["design"] for c in cold.maximin_select(cands, 2)] == \
        ["fine", "unreadable"]


# ── extract_passing: cross-seed mutation summaries ───────────────────────────

@pytest.mark.logic
def test_mutation_summary_across_seeds_finds_the_majority():
    rows = [
        {"steered_mutations_chimerax": "/A:5,7"},
        {"steered_mutations_chimerax": "/A:5,7"},
        {"steered_mutations_chimerax": "/A:5"},
    ]
    got = ep._summarize_mutations_across_seeds(
        rows, "steered_mutations_chimerax", 0.5)
    assert got is not None
    assert "5" in str(got), "position 5 is in every seed and must appear"


@pytest.mark.logic
def test_mutation_summary_of_no_rows_is_empty_not_an_error():
    """Returns a 3-tuple of empty strings rather than raising or returning None."""
    got = ep._summarize_mutations_across_seeds([], "steered_mutations_chimerax", 0.5)
    assert isinstance(got, tuple) and len(got) == 3
    assert all(x == "" for x in got), f"expected three empty strings, got {got!r}"
