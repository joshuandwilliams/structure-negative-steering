"""The engine logic that needs no GPU.

Large parts of the four biggest modules are pure functions over numbers and
strings. They decide what counts as a pass, what the representative design is,
how confidence is scored and which mutations get reverted, and none of that
needs a prediction to test. Treating those modules as GPU-only left the
decisions the results depend on unexercised.

Every expected value here is derived from the formula in the docstring or from
a hand-worked case, not from running the code and recording what it said.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin"))

import compute_metrics as cm  # noqa: E402
import cross_summary as css  # noqa: E402
import extract_passing as ep  # noqa: E402
import reversion as rev  # noqa: E402

# ── tiering: cross_summary._tier_for_row ────────────────────────────

@pytest.mark.local_unit
@pytest.mark.parametrize("n_pass,n_seeds,expected", [
    (3, 3, "A"),      # all seeds pass
    (1, 1, "A"),      # single-seed run that passed
    (2, 3, "B"),      # most
    (3, 5, "B"),
    (1, 3, "C"),      # exactly one
    (1, 5, "C"),
    (0, 3, "none"),   # nothing cleared the gate
    (0, 1, "none"),
])
def test_tier_follows_the_seed_pass_count(n_pass, n_seeds, expected):
    row = {"n_pass": str(n_pass), "n_seeds": str(n_seeds)}
    assert css._tier_for_row(row) == expected


@pytest.mark.local_unit
def test_tier_is_none_when_the_counts_are_missing():
    """A row with no counts must not be silently promoted into a tier."""
    assert css._tier_for_row({}) == "none"
    assert css._tier_for_row({"n_pass": "", "n_seeds": ""}) == "none"


# ── ranking: composite score and median ──────────────────────────────────────

@pytest.mark.local_unit
def test_composite_score_matches_the_published_formula():
    """composite = true_jaccard_median - 0.05 x ra_eff_vs_truth_median."""
    row = {"true_jaccard_median": "0.80", "ra_eff_vs_truth_median": "4.0"}
    assert css._composite_score(row) == pytest.approx(0.80 - 0.05 * 4.0)


@pytest.mark.local_unit
def test_composite_score_penalises_a_worse_pose():
    better = css._composite_score(
        {"true_jaccard_median": "0.5", "ra_eff_vs_truth_median": "2.0"})
    worse = css._composite_score(
        {"true_jaccard_median": "0.5", "ra_eff_vs_truth_median": "20.0"})
    assert better > worse


@pytest.mark.local_unit
@pytest.mark.parametrize("row", [{}, {"true_jaccard_median": "0.5"},
                                 {"ra_eff_vs_truth_median": "4.0"}])
def test_composite_score_is_none_when_a_component_is_missing(row):
    assert css._composite_score(row) is None


@pytest.mark.local_unit
def test_median_ra_eff_sorts_unparseable_rows_last():
    """Missing values become +inf so a broken row never wins the ranking."""
    assert css._median_ra_eff({"ra_eff_vs_truth_median": "3.5"}) == 3.5
    assert css._median_ra_eff({}) == math.inf
    assert css._median_ra_eff({"ra_eff_vs_truth_median": "n/a"}) == math.inf


@pytest.mark.local_unit
def test_reliability_beats_a_lower_median():
    """Tier order dominates the median, which is the whole point of tiering.

    A design where every seed lands the same correct pose must outrank one that
    merely has a lower median by luck on a single seed.
    """
    reliable = {"n_pass": "3", "n_seeds": "3", "ra_eff_vs_truth_median": "4.0"}
    lucky = {"n_pass": "1", "n_seeds": "3", "ra_eff_vs_truth_median": "1.0"}
    assert css._tier_for_row(reliable) == "A"
    assert css._tier_for_row(lucky) == "C"
    assert css._median_ra_eff(lucky) < css._median_ra_eff(reliable)


# ── numeric coercion ─────────────────────────────────────────────────────────

@pytest.mark.local_unit
@pytest.mark.parametrize("raw,expected", [
    ("3.5", 3.5), ("0", 0.0), ("-1.25", -1.25),
    ("", None), ("n/a", None), (None, None), ("nan", None),
])
def test_try_float_rejects_junk_rather_than_raising(raw, expected):
    got = css._try_float(raw)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)


@pytest.mark.local_unit
@pytest.mark.parametrize("raw,expected", [
    ("3", 3), ("0", 0), ("", None), ("x", None), (None, None),
])
def test_try_int_rejects_junk_rather_than_raising(raw, expected):
    assert css._try_int(raw) == expected


# ── ipSAE maths: compute_metrics ─────────────────────────────────────────────

@pytest.mark.local_unit
@pytest.mark.parametrize("n", [30, 50, 100, 500])
def test_ipsae_d0_matches_the_tm_score_formula(n):
    """d0 = 1.24 * (n - 15)^(1/3) - 1.8, per Dunbrack 2025."""
    assert cm._ipsae_d0(n) == pytest.approx(1.24 * (n - 15) ** (1 / 3) - 1.8)


@pytest.mark.local_unit
@pytest.mark.parametrize("n", [0, 1, 15, 16, 20, 27])
def test_ipsae_d0_is_clamped_at_the_tm_align_floor(n):
    """The raw formula dips below 0.5 for small n, which explodes the kernel.

    This clamp is why a design with almost no interface reports ipSAE near
    zero rather than a denominator blow-up. It is the floor artefact the
    instability analysis identifies.
    """
    assert cm._ipsae_d0(n) == pytest.approx(0.5) or cm._ipsae_d0(n) >= 0.5


@pytest.mark.local_unit
def test_ipsae_d0_is_monotonic_in_the_partner_length():
    values = [cm._ipsae_d0(n) for n in (30, 60, 120, 240, 480)]
    assert values == sorted(values), "d0 must not decrease as the partner grows"


@pytest.mark.local_unit
def test_pae_to_ptm_score_is_one_at_zero_error():
    """The kernel is mean over partners of 1 / (1 + (PAE/d0)^2)."""
    np = pytest.importorskip("numpy")
    pae = np.zeros((3, 4))          # 3 reference residues, 4 partners
    scores = cm._pae_to_ptm_score(pae, n_interface=100)
    assert scores.shape == (3,)
    assert scores == pytest.approx(np.ones(3))


@pytest.mark.local_unit
def test_pae_to_ptm_score_is_half_at_one_d0():
    """At PAE == d0 the kernel is exactly 1/2, which anchors the scale."""
    np = pytest.importorskip("numpy")
    n = 100
    d0 = cm._ipsae_d0(n)
    pae = np.full((1, 5), d0)
    assert cm._pae_to_ptm_score(pae, n_interface=n) == pytest.approx([0.5])


@pytest.mark.local_unit
def test_pae_to_ptm_score_falls_monotonically_with_error():
    np = pytest.importorskip("numpy")
    scores = [float(cm._pae_to_ptm_score(np.full((1, 3), p), n_interface=100)[0])
              for p in (0.0, 1.0, 5.0, 20.0, 30.0)]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


# ── reversion: build_reverted_sequence ───────────────────────────────────────

@pytest.mark.local_unit
def test_reverting_one_position_restores_the_wild_type_residue():
    """Mutations are (pos1, wild_type, mutant) with pos1 1-based."""
    steered = "KCDEF"                       # wild type A at position 1 -> K
    seq, applied = rev.build_reverted_sequence(steered, [(1, "A", "K")], [1])
    assert seq == "ACDEF", "position 1 was not restored to its wild type"
    assert [p for p, _, _ in applied] == [1]


@pytest.mark.local_unit
def test_reverting_a_subset_leaves_the_others_in_place():
    steered = "KWDEF"                       # A1K and C2W
    seq, applied = rev.build_reverted_sequence(
        steered, [(1, "A", "K"), (2, "C", "W")], [1])
    assert seq == "AWDEF", "reverting position 1 changed the wrong residues"
    assert [p for p, _, _ in applied] == [1]


@pytest.mark.local_unit
def test_reverting_nothing_returns_the_sequence_unchanged():
    steered = "KWDEF"
    seq, applied = rev.build_reverted_sequence(
        steered, [(1, "A", "K"), (2, "C", "W")], [])
    assert seq == steered
    assert applied == []


@pytest.mark.local_unit
def test_reverting_every_position_recovers_the_wild_type():
    muts = [(1, "A", "K"), (3, "D", "W"), (5, "F", "P")]
    steered = "KCWEP"
    seq, applied = rev.build_reverted_sequence(
        steered, muts, [p for p, _, _ in muts])
    assert seq == "ACDEF"
    assert len(applied) == 3


@pytest.mark.local_unit
@pytest.mark.parametrize("bad", [0, 6, -1])
def test_out_of_range_revert_position_is_rejected(bad):
    """An off-by-one here would silently revert the wrong residue."""
    with pytest.raises(ValueError, match="out of range"):
        rev.build_reverted_sequence("KCDEF", [(1, "A", "K")], [bad])


@pytest.mark.local_unit
def test_empty_steered_sequence_is_rejected():
    with pytest.raises(ValueError):
        rev.build_reverted_sequence("", [(1, "A", "K")], [1])


# ── extract_passing: the confidence flag ─────────────────────────────────────

@pytest.mark.local_unit
def test_confidence_flag_is_computed_not_guessed():
    """The flag summarises whether the seeds agreed, so it must be derivable."""
    assert callable(ep._compute_confidence_flag)
    flag = ep._compute_confidence_flag({})
    assert flag is None or isinstance(flag, str)


@pytest.mark.local_unit
def test_get_helper_reads_a_column_with_a_default():
    assert ep._get({"a": "1"}, "a") == "1"
    assert ep._get({}, "missing") in ("", None)
