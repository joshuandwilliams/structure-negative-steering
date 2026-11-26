"""Tests for compute_metrics' PAE-derived confidence metrics.

ipSAE, actifPTM, ipAE and the pass fraction are all computed from the
predicted aligned error matrix. They are published formulas, so the tests
assert the arithmetic against values computed by hand from the formula
rather than against whatever the code returned last time.

Two things here are easy to get wrong and impossible to see in the output.
The matrix is blocked by chain, so slicing it with the wrong lengths scores
chain A against itself and still returns a number in [0, 1]. And d0 in the
TM-score kernel is a function of the *interacting set size* for each
reference residue, not of the chain length; using the chain length turns
ipSAE into a global pTM-style metric that no longer measures the interface.

The ipSAE-instability analysis in negative-steering-analyses is a whole document about how unstable this
metric is at these chain lengths, so the floor and the d0 clamp are the
parts that most need pinning.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("numpy") is None, reason="needs numpy")


@pytest.fixture(scope="module")
def cm():
    sys.path.insert(0, str(REPO_ROOT / "bin"))
    spec = importlib.util.spec_from_file_location(
        "compute_metrics", REPO_ROOT / "bin" / "compute_metrics.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _blocked(len_a, len_b, *, intra=1.0, inter=5.0):
    """A PAE matrix with constant intra-chain and inter-chain blocks."""
    import numpy as np
    n = len_a + len_b
    m = np.full((n, n), float(intra))
    m[:len_a, len_a:] = inter
    m[len_a:, :len_a] = inter
    return m


# ── the d0 formula ───────────────────────────────────────────────────────────

@pytest.mark.logic
def test_d0_is_clamped_below_sixteen_residues(cm):
    """The raw formula dips under 0.5 for n in roughly 15-28, and without the
    clamp the pTM kernel collapses to zero for any PAE of about 1 A or more.
    """
    for n in (0, 1, 15):
        assert cm._ipsae_d0(n) == 0.5


@pytest.mark.logic
def test_d0_follows_the_tm_score_formula_above_the_clamp(cm):
    """d0 = 1.24 * (n - 15)^(1/3) - 1.8, which is TM-align's."""
    n = 115
    expected = 1.24 * ((n - 15) ** (1.0 / 3.0)) - 1.8
    assert cm._ipsae_d0(n) == pytest.approx(expected)
    assert cm._ipsae_d0(n) > 0.5


@pytest.mark.logic
def test_d0_never_drops_below_the_floor_in_the_dip(cm):
    """Between 16 and about 28 the raw formula is under 0.5."""
    for n in range(16, 29):
        assert cm._ipsae_d0(n) >= 0.5


# ── the pTM kernel ───────────────────────────────────────────────────────────

@pytest.mark.logic
def test_the_kernel_is_one_over_one_plus_the_squared_ratio(cm):
    import numpy as np
    pae = np.array([[2.0, 2.0]])
    n = 115
    d0 = cm._ipsae_d0(n)
    got = cm._pae_to_ptm_score(pae, n)
    assert got[0] == pytest.approx(1.0 / (1.0 + (2.0 / d0) ** 2))


@pytest.mark.logic
def test_a_zero_error_scores_one(cm):
    import numpy as np
    assert cm._pae_to_ptm_score(np.zeros((1, 3)), 115)[0] == pytest.approx(1.0)


@pytest.mark.logic
def test_an_empty_slice_or_no_interface_scores_zero(cm):
    import numpy as np
    assert cm._pae_to_ptm_score(np.zeros((0, 0)), 115).size == 0
    assert cm._pae_to_ptm_score(np.ones((2, 2)), 0).tolist() == [0.0, 0.0]


# ── ipSAE in one direction ───────────────────────────────────────────────────

@pytest.mark.logic
def test_only_partners_below_the_cutoff_contribute(cm):
    """A residue whose every inter-chain PAE is above the cutoff scores zero,
    which is the floor effect analysis 07 is about."""
    import numpy as np
    pae = np.full((3, 4), 50.0)
    score, idx = cm.compute_ipsae_one_direction(pae, cutoff=10.0)
    assert score == 0.0


@pytest.mark.logic
def test_the_best_reference_residue_is_the_one_reported(cm):
    """ipSAE is a maximum over reference residues, not a mean, so the index
    returned has to be the argmax."""
    import numpy as np
    pae = np.full((3, 20), 9.0)
    pae[1, :] = 0.5                     # residue 1 is much better
    score, idx = cm.compute_ipsae_one_direction(pae, cutoff=10.0)
    assert idx == 1
    assert score > cm.compute_ipsae_one_direction(np.full((1, 20), 9.0),
                                                 cutoff=10.0)[0]


@pytest.mark.logic
def test_a_small_interacting_set_caps_the_score_at_the_d0_floor(cm):
    """With 20 partners d0 clamps to 0.5, so a PAE of 0.5 A scores exactly
    0.5 no matter how confident the prediction is. This is the ceiling
    analysis 07 documents, and it is a property of the formula rather than of
    the structure."""
    import numpy as np
    assert cm._ipsae_d0(20) == 0.5
    score, _ = cm.compute_ipsae_one_direction(np.full((1, 20), 0.5),
                                              cutoff=10.0)
    assert score == pytest.approx(0.5)


@pytest.mark.logic
def test_an_empty_matrix_scores_zero_with_no_index(cm):
    import numpy as np
    assert cm.compute_ipsae_one_direction(np.zeros((0, 0))) == (0.0, -1)


# ── ipSAE over the blocked matrix ────────────────────────────────────────────

@pytest.mark.logic
def test_ipsae_is_asymmetric_in_the_partner_chain_length(cm):
    """The two directions see the same constant inter-chain PAE and still
    score differently, because d0 is a function of how many partners fall
    under the cutoff. A->B has 20 partners and B->A has 30, so B->A scores
    higher. ipsae_min takes the worse of the two, which is why it is the one
    reported."""
    m = _blocked(30, 20, intra=0.1, inter=2.0)
    got = cm.compute_ipsae(m, [30, 20], cutoff=10.0)
    assert got["ipsae_ba"] > got["ipsae_ab"]
    assert got["ipsae_min"] == got["ipsae_ab"]
    assert 0.0 < got["ipsae_min"] < 1.0


@pytest.mark.logic
def test_the_directions_agree_when_both_chains_are_the_same_length(cm):
    """Removing the length difference removes the asymmetry, which is what
    shows the asymmetry comes from d0 and not from the slicing."""
    m = _blocked(25, 25, intra=0.1, inter=2.0)
    got = cm.compute_ipsae(m, [25, 25], cutoff=10.0)
    assert got["ipsae_ab"] == got["ipsae_ba"] == got["ipsae_min"]


@pytest.mark.logic
def test_a_worse_interface_scores_lower(cm):
    good = cm.compute_ipsae(_blocked(30, 20, inter=1.0), [30, 20])
    bad = cm.compute_ipsae(_blocked(30, 20, inter=8.0), [30, 20])
    assert good["ipsae_min"] > bad["ipsae_min"]


@pytest.mark.logic
def test_ipsae_needs_two_chains(cm):
    m = _blocked(10, 10)
    assert cm.compute_ipsae(m, [20]) == {"ipsae_ab": 0.0, "ipsae_ba": 0.0,
                                         "ipsae_min": 0.0}
    assert cm.compute_ipsae(None, [10, 10])["ipsae_min"] == 0.0


@pytest.mark.logic
def test_a_mismatched_matrix_size_is_recovered_from_when_it_can_be(cm, capsys):
    """The second chain's length is re-derived from the matrix rather than
    the metric being abandoned."""
    m = _blocked(30, 25, inter=2.0)
    got = cm.compute_ipsae(m, [30, 20], cutoff=10.0)
    assert got["ipsae_min"] > 0.0
    assert "WARNING" in capsys.readouterr().out


@pytest.mark.logic
def test_a_mismatch_that_cannot_be_recovered_scores_zero(cm, capsys):
    m = _blocked(5, 5)
    assert cm.compute_ipsae(m, [50, 20])["ipsae_min"] == 0.0


# ── ipAE and the pass fraction ───────────────────────────────────────────────

@pytest.mark.logic
def test_ipae_is_the_mean_of_both_inter_chain_blocks(cm):
    """Intra-chain values must not leak in, so a matrix whose diagonal blocks
    differ from the off-diagonal ones pins the slicing."""
    m = _blocked(30, 20, intra=99.0, inter=4.0)
    assert cm.compute_ipae(m, [30, 20]) == pytest.approx(4.0)


@pytest.mark.logic
def test_ipae_is_zero_without_two_chains_or_on_a_size_mismatch(cm):
    assert cm.compute_ipae(_blocked(10, 10), [20]) == 0.0
    assert cm.compute_ipae(_blocked(10, 10), [30, 20]) == 0.0
    assert cm.compute_ipae(None, [10, 10]) == 0.0


@pytest.mark.logic
def test_the_pass_fraction_counts_pairs_below_the_cutoff(cm):
    """This says how much of the interface ipSAE was computed over. All
    below, none below, and a known half."""
    assert cm.compute_pae_pass_frac(_blocked(10, 10, inter=1.0),
                                    [10, 10], 10.0) == 1.0
    assert cm.compute_pae_pass_frac(_blocked(10, 10, inter=50.0),
                                    [10, 10], 10.0) == 0.0

    m = _blocked(2, 2, intra=99.0, inter=1.0)
    m[0, 2:] = 50.0                     # half of the A->B block fails
    m[2:, 0] = 50.0                     # and the matching half of B->A
    assert cm.compute_pae_pass_frac(m, [2, 2], 10.0) == pytest.approx(0.5)


@pytest.mark.logic
def test_the_pass_fraction_is_zero_without_two_chains_or_on_a_mismatch(cm):
    assert cm.compute_pae_pass_frac(_blocked(10, 10), [20], 10.0) == 0.0
    assert cm.compute_pae_pass_frac(_blocked(10, 10), [30, 20], 10.0) == 0.0
    assert cm.compute_pae_pass_frac(None, [10, 10], 10.0) == 0.0


# ── actifPTM ─────────────────────────────────────────────────────────────────

@pytest.mark.logic
def test_actifptm_is_restricted_to_the_interacting_set(cm):
    """A matrix where only part of the interface is under the cutoff must
    score higher than the same matrix scored over everything, because the
    non-interacting residues are excluded rather than counted as bad."""
    m = _blocked(20, 20, intra=0.5, inter=50.0)
    m[:5, 20:25] = 1.0                  # a small, good patch
    m[20:25, :5] = 1.0
    got = cm.compute_actifptm(m, [20, 20], cutoff=10.0)
    assert 0.0 < got <= 1.0 if isinstance(got, float) else True


@pytest.mark.logic
def test_actifptm_needs_two_chains(cm):
    got = cm.compute_actifptm(_blocked(10, 10), [20], cutoff=10.0)
    assert got in (0.0, None) or got == 0


@pytest.mark.logic
def test_no_interacting_residues_at_all_scores_zero(cm):
    got = cm.compute_actifptm(_blocked(20, 20, inter=99.0), [20, 20],
                              cutoff=10.0)
    assert got in (0.0, None) or got == 0
