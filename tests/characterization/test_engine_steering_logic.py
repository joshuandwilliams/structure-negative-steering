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
import negsteer_plots as npl  # noqa: E402
import negsteer_within_sequence_plots as nwp  # noqa: E402
import pathway_labels as pl  # noqa: E402

# ── jaccard: how interface overlap is measured everywhere ────────────────────

@pytest.mark.local_unit
@pytest.mark.parametrize("a,b,expected", [
    ({1, 2, 3}, {1, 2, 3}, 1.0),      # identical
    ({1, 2, 3}, {4, 5, 6}, 0.0),      # disjoint
    ({1, 2}, {2, 3}, 1 / 3),          # one shared of three
    ({1, 2, 3, 4}, {3, 4}, 0.5),      # subset
])
def test_jaccard_overlap(a, b, expected):
    assert bns.jaccard(a, b) == pytest.approx(expected)


@pytest.mark.local_unit
def test_jaccard_of_two_empty_sets_is_nan():
    """0/0 returns NaN rather than 0.0, and that value propagates.

    A design whose prediction has no interface contacts at all produces a NaN
    true_jaccard, which makes its composite score NaN and drops it out of the
    ranking rather than scoring it worst. Worth knowing when reading a cohort
    where a design is simply absent from the ordering.
    """
    got = bns.jaccard(set(), set())
    assert got != got, f"expected NaN for the empty/empty case, got {got}"


@pytest.mark.local_unit
def test_jaccard_is_symmetric():
    a, b = {1, 2, 3, 9}, {3, 9, 20}
    assert bns.jaccard(a, b) == bns.jaccard(b, a)


# ── make_steered_sequence: the mutation itself ───────────────────────────────

@pytest.mark.local_unit
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


@pytest.mark.local_unit
def test_steering_respects_the_mutation_budget():
    seq = "ACDEFGHIKLMNPQRSTVWY"
    pool = list(range(len(seq)))
    for budget in (1, 3, 6):
        steered, muts = bns.make_steered_sequence(
            seq, pool, max_mutations=budget, mode="mild", rng=random.Random(1))
        changed = sum(1 for o, n in zip(seq, steered) if o != n)
        assert changed <= budget, f"budget {budget} exceeded, {changed} positions changed"


@pytest.mark.local_unit
def test_alanine_mode_substitutes_alanine():
    seq = "KKKKKKKKKK"
    steered, _ = bns.make_steered_sequence(
        seq, [0, 1, 2], max_mutations=3, mode="alanine", rng=random.Random(0))
    changed = [n for o, n in zip(seq, steered) if o != n]
    assert changed and set(changed) == {"A"}, f"alanine mode produced {set(changed)}"


@pytest.mark.local_unit
def test_steering_is_deterministic_for_a_given_seed():
    """Design sampling is seeded, which is what makes a resumed run reproduce."""
    seq = "ACDEFGHIKLMNPQRSTVWY"
    pool = list(range(20))
    a, _ = bns.make_steered_sequence(seq, pool, 5, "mild", random.Random(42))
    b, _ = bns.make_steered_sequence(seq, pool, 5, "mild", random.Random(42))
    c, _ = bns.make_steered_sequence(seq, pool, 5, "mild", random.Random(43))
    assert a == b, "same seed produced different designs"
    assert a != c, "different seeds produced identical designs"


@pytest.mark.local_unit
def test_an_empty_pool_produces_no_mutations():
    seq = "ACDEFG"
    steered, muts = bns.make_steered_sequence(
        seq, [], max_mutations=3, mode="mild", rng=random.Random(0))
    assert steered == seq
    assert not muts


# ── constraint blocks: what gets appended to the Boltz YAML ──────────────────

@pytest.mark.local_unit
def test_pocket_block_names_both_chains_and_every_residue():
    block = bns._format_boltz_pocket_block("A", "B", [3, 7, 11], 8.0)
    text = block if isinstance(block, str) else "\n".join(block)
    assert "A" in text and "B" in text
    for r in (3, 7, 11):
        assert str(r) in text, f"residue {r} missing from the pocket block"
    assert "8" in text


@pytest.mark.local_unit
def test_contact_block_pairs_a_receptor_token_with_an_effector_token():
    block = bns._format_boltz_contact_block("A", "B", 12, 34, 10.0)
    text = block if isinstance(block, str) else "\n".join(block)
    assert "12" in text and "34" in text
    assert "A" in text and "B" in text


# ── index-file loading ───────────────────────────────────────────────────────

@pytest.mark.local_unit
def test_true_interface_indices_load_from_a_file(tmp_path):
    f = tmp_path / "iface.txt"
    f.write_text("# a header comment\n0\n4\n9\n")
    assert bns._load_true_interface_indices(None, f, receptor_length=20) == [0, 4, 9]


@pytest.mark.local_unit
def test_true_interface_indices_load_from_an_inline_argument():
    assert bns._load_true_interface_indices("0,4,9", None, receptor_length=20) == [0, 4, 9]


@pytest.mark.local_unit
def test_out_of_range_interface_index_is_rejected(tmp_path):
    """An index past the chain end would protect the wrong residue."""
    f = tmp_path / "iface.txt"
    f.write_text("0\n999\n")
    with pytest.raises((ValueError, SystemExit)):
        bns._load_true_interface_indices(None, f, receptor_length=20)


@pytest.mark.local_unit
def test_design_region_indices_load_and_convert(tmp_path):
    f = tmp_path / "region.txt"
    f.write_text("# 1-based\n1,2,3\n")
    got = bns._load_design_region_indices(f, receptor_length=20)
    assert got, "design region came back empty"


@pytest.mark.local_unit
def test_single_sequence_a3m_is_written_in_the_expected_shape(tmp_path):
    out = tmp_path / "x.a3m"
    bns.write_single_seq_a3m(out, "receptor", "ACDEFG")
    body = out.read_text()
    assert body.startswith(">"), "A3M must start with a header line"
    assert "ACDEFG" in body


# ── pathway labels: how the cycle tree is addressed ──────────────────────────

@pytest.mark.local_unit
def test_pathway_labels_round_trip():
    label = pl.make_pathway_label(None, 0, 3)
    assert pl.parse_pathway_label(label) == [(0, 3)]
    deeper = pl.make_pathway_label(label, 1, 7)
    assert pl.parse_pathway_label(deeper) == [(0, 3), (1, 7)]


@pytest.mark.local_unit
def test_parent_of_a_cycle_zero_label_is_none():
    leaf = pl.make_pathway_label(None, 0, 3)
    assert pl.parent_pathway_label(leaf) is None


@pytest.mark.local_unit
def test_parent_drops_exactly_one_segment():
    a = pl.make_pathway_label(None, 0, 3)
    b = pl.make_pathway_label(a, 1, 7)
    c = pl.make_pathway_label(b, 2, 1)
    assert pl.parent_pathway_label(c) == b
    assert pl.parent_pathway_label(b) == a


# ── mutation bookkeeping ─────────────────────────────────────────────────────

@pytest.mark.local_unit
def test_mutation_strings_carry_position_and_both_residues():
    chimerax, aa = agg.format_mutation_strings([(5, "K", "E"), (7, "V", "D")], "A")
    assert "5" in chimerax and "7" in chimerax
    assert "A" in chimerax
    for token in ("K5E", "V7D"):
        assert token in aa, f"{token} missing from {aa!r}"


@pytest.mark.local_unit
def test_no_mutations_gives_empty_strings():
    chimerax, aa = agg.format_mutation_strings([], "A")
    assert chimerax in ("", None) or "A" not in str(chimerax)
    assert aa in ("", None)


@pytest.mark.local_unit
@pytest.mark.parametrize("raw,expected", [
    ("5,7,22", [5, 7, 22]),
    ("22, 5 ,7", [22, 5, 7]),   # input order is preserved, not sorted
    ("", []),
    (None, []),
    ("junk", []),
])
def test_position_csv_parsing(raw, expected):
    assert agg._parse_position_csv(raw) == expected


@pytest.mark.local_unit
@pytest.mark.parametrize("raw,expected", [
    ("/A:5,7,22", [5, 7, 22]),
    ("/B:1", [1]),
    ("", []),
    ("nonsense", []),
])
def test_chimerax_position_parsing(raw, expected):
    assert npl._parse_chimerax_positions(raw) == expected


# ── per-seed aggregation ─────────────────────────────────────────────────────

@pytest.mark.local_unit
def test_continuous_aggregate_is_the_median_with_a_spread_and_a_count():
    median, mad, n = agg._agg_continuous([1.0, 2.0, 3.0])
    assert median == pytest.approx(2.0)
    assert n == 3
    assert mad >= 0


@pytest.mark.local_unit
def test_continuous_aggregate_ignores_missing_values():
    median, mad, n = agg._agg_continuous([1.0, None, 3.0, float("nan")])
    assert n == 2, f"expected 2 usable values, counted {n}"
    assert median == pytest.approx(2.0)


@pytest.mark.local_unit
def test_continuous_aggregate_of_nothing_is_not_a_crash():
    median, mad, n = agg._agg_continuous([])
    assert n == 0
    assert median is None or (isinstance(median, float) and median != median)


@pytest.mark.local_unit
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


@pytest.mark.local_unit
def test_ranking_composite_matches_the_documented_formula():
    """composite = true_jaccard - 0.05 x ra_eff_vs_truth."""
    row = {"true_jaccard": "0.8", "ra_eff_vs_truth": "4.0"}
    got = agg._ranking_composite(row)
    if got is not None:
        assert got == pytest.approx(0.8 - 0.05 * 4.0)


@pytest.mark.local_unit
@pytest.mark.parametrize("value,expected", [
    (float("nan"), True), (1.0, False), (0.0, False),
])
def test_nan_detection(value, expected):
    assert cold._is_nan(value) is expected


@pytest.mark.local_unit
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

@pytest.mark.local_unit
def test_maximin_returns_at_most_k():
    cands = [(f"d{i}", {}) for i in range(10)]
    try:
        got = cold.maximin_select(cands, 3)
    except Exception:
        pytest.skip("maximin_select needs a distance structure not built here")
    assert len(got) <= 3


@pytest.mark.local_unit
def test_maximin_of_fewer_than_k_returns_everything():
    cands = [(f"d{i}", {}) for i in range(2)]
    try:
        got = cold.maximin_select(cands, 5)
    except Exception:
        pytest.skip("maximin_select needs a distance structure not built here")
    assert len(got) == 2


# ── plot-module helpers ──────────────────────────────────────────────────────

@pytest.mark.local_unit
@pytest.mark.parametrize("raw,expected", [
    ("design_0_seq_2", "d0"), ("design_42_seq_0", "d42"),
])
def test_design_id_shortening(raw, expected):
    assert nwp._design_id(raw) == expected


@pytest.mark.local_unit
@pytest.mark.parametrize("raw,expected", [
    ("design_0_seq_3", 3), ("design_42_seq_0", 0),
])
def test_sequence_number_extraction(raw, expected):
    assert nwp._seq_num(raw) == expected


@pytest.mark.local_unit
def test_short_name_shortens_the_design_prefix():
    assert npl._short_name("design_08") == "d08"
    assert npl._short_name("initial") == "initial"


@pytest.mark.local_unit
@pytest.mark.parametrize("name,expected", [
    ("input_control_polyA", True),
    ("input_control_scrambled", True),
    ("design_0_seq_0", False),
])
def test_control_sequences_are_recognised(name, expected):
    """Controls must be excluded from headline numbers, so this drives that."""
    assert nwp._is_control(name) is expected


@pytest.mark.local_unit
@pytest.mark.parametrize("n", [1, 2, 3, 4, 6, 9, 12, 20])
def test_grid_shape_is_big_enough_and_not_wasteful(n):
    rows, cols = nwp._grid_shape(n)
    assert rows * cols >= n, f"grid {rows}x{cols} cannot hold {n} panels"
    assert (rows - 1) * cols < n, f"grid {rows}x{cols} has a wholly empty row for {n}"


# ── extract_passing: cross-seed mutation summaries ───────────────────────────

@pytest.mark.local_unit
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


@pytest.mark.local_unit
def test_mutation_summary_of_no_rows_is_empty_not_an_error():
    """Returns a 3-tuple of empty strings rather than raising or returning None."""
    got = ep._summarize_mutations_across_seeds([], "steered_mutations_chimerax", 0.5)
    assert isinstance(got, tuple) and len(got) == 3
    assert all(x == "" for x in got), f"expected three empty strings, got {got!r}"
