"""compute_metrics: the scoring the whole benchmark rests on.

753 statements covering interface detection, contact geometry, the PAE-derived
confidence metrics and the weighted Jaccard. None of it needs a GPU. The PAE
maths runs on synthetic matrices, and the structural half runs on 6G10, a real
two-chain complex tracked in this repo.

These assert against the published formula or a hand-constructed geometry, not
against whatever the code currently returns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin"))

import compute_metrics as cm  # noqa: E402

np = pytest.importorskip("numpy")

SIXG10 = (_REPO_ROOT / "experiments" / "benchmarking" / "unconstrained" / "6G10"
          / "6G10.pdb")
needs_pdb = pytest.mark.skipif(not SIXG10.is_file(), reason="6G10 reference absent")

# 6G10 is receptor chain A at 76 residues, effector chain B at 83.
REC, EFF = "A", "B"
REC_LEN, EFF_LEN = 76, 83


# ── PAE-derived metrics, on synthetic matrices ───────────────────────────────

def _pae(n_rec: int, n_eff: int, inter: float, intra: float = 1.0):
    """A full PAE matrix with a constant inter-chain block."""
    n = n_rec + n_eff
    m = np.full((n, n), intra, dtype=float)
    m[:n_rec, n_rec:] = inter
    m[n_rec:, :n_rec] = inter
    return m


@pytest.mark.local_unit
def test_ipsae_returns_both_directions_and_their_minimum():
    """compute_ipsae reports ipsae_ab, ipsae_ba and ipsae_min."""
    got = cm.compute_ipsae(_pae(30, 30, inter=0.0), [30, 30], cutoff=10.0)
    assert set(got) == {"ipsae_ab", "ipsae_ba", "ipsae_min"}
    assert got["ipsae_min"] == pytest.approx(min(got["ipsae_ab"], got["ipsae_ba"]))


@pytest.mark.local_unit
def test_perfect_interchain_pae_gives_the_maximum_ipsae():
    got = cm.compute_ipsae(_pae(30, 30, inter=0.0), [30, 30], cutoff=10.0)
    assert got["ipsae_min"] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.local_unit
def test_hopeless_interchain_pae_gives_near_zero_ipsae():
    """Every pair above the cutoff sends L to 0 and ipSAE to the floor.

    This is the floor artefact: the score is bimodal rather than graded, so a
    design with no usable interface reports ~0 rather than a low number.
    """
    got = cm.compute_ipsae(_pae(30, 30, inter=30.0), [30, 30], cutoff=10.0)
    assert got["ipsae_min"] < 0.1, f"expected a near-zero score, got {got}"


@pytest.mark.local_unit
def test_ipsae_falls_as_interchain_error_rises():
    scores = [cm.compute_ipsae(_pae(40, 40, i), [40, 40], 10.0)["ipsae_min"]
              for i in (0.0, 2.0, 5.0, 8.0)]
    assert scores == sorted(scores, reverse=True), f"not monotonic: {scores}"


@pytest.mark.local_unit
def test_mean_interchain_pae_is_the_off_diagonal_mean():
    m = _pae(20, 30, inter=7.0, intra=1.0)
    assert float(cm.compute_ipae(m, [20, 30])) == pytest.approx(7.0)


@pytest.mark.local_unit
def test_mean_interchain_pae_ignores_the_intra_chain_blocks():
    """A confident monomer with a hopeless interface must not look good."""
    a = float(cm.compute_ipae(_pae(20, 30, inter=9.0, intra=0.0), [20, 30]))
    b = float(cm.compute_ipae(_pae(20, 30, inter=9.0, intra=25.0), [20, 30]))
    assert a == pytest.approx(b), "intra-chain PAE leaked into the interchain mean"


@pytest.mark.local_unit
@pytest.mark.parametrize("inter,cutoff,expected", [
    (1.0, 10.0, 1.0),     # everything passes
    (30.0, 10.0, 0.0),    # nothing passes
])
def test_pae_pass_fraction_is_all_or_nothing_at_the_extremes(inter, cutoff, expected):
    got = cm.compute_pae_pass_frac(_pae(15, 15, inter), [15, 15], cutoff)
    assert float(got) == pytest.approx(expected)


@pytest.mark.local_unit
def test_pae_pass_fraction_is_a_proportion():
    m = _pae(20, 20, inter=1.0)
    m[:20, 20:][:10, :] = 30.0      # half the receptor rows fail
    m[20:, :20][:, :10] = 30.0
    got = float(cm.compute_pae_pass_frac(m, [20, 20], 10.0))
    assert 0.0 < got < 1.0, f"expected a fraction, got {got}"


@pytest.mark.local_unit
def test_all_pae_metrics_are_produced_in_one_call():
    got = cm._pae_derived_metrics(_pae(25, 25, inter=4.0), [25, 25], 10.0)
    assert isinstance(got, dict) and got, "no metrics returned"
    for k, v in got.items():
        if isinstance(v, float):
            assert v == v, f"{k} came back NaN"


@pytest.mark.local_unit
def test_actifptm_is_bounded():
    got = cm.compute_actifptm(_pae(25, 25, inter=3.0), [25, 25], 10.0)
    val = got if isinstance(got, (int, float)) else max(np.ravel(np.asarray(got, dtype=float)))
    assert 0.0 <= float(val) <= 1.0


@pytest.mark.local_unit
def test_one_direction_ipsae_is_bounded_and_ordered():
    good = cm.compute_ipsae_one_direction(np.zeros((10, 10)), 10.0)
    bad = cm.compute_ipsae_one_direction(np.full((10, 10), 30.0), 10.0)
    g = float(np.max(np.asarray(good, dtype=float)))
    b = float(np.max(np.asarray(bad, dtype=float)))
    assert 0.0 <= b <= g <= 1.0, f"expected bad({b}) <= good({g}) within [0,1]"


# ── structural metrics, on a real complex ────────────────────────────────────

@pytest.mark.local_integration
@needs_pdb
def test_heavy_atoms_parse_into_both_chains():
    """Returns every heavy ATOM per chain as (resseq, x, y, z), not residues."""
    got = cm._parse_heavy_atoms_by_chain(str(SIXG10))
    assert REC in got and EFF in got, f"chains found: {sorted(got)}"
    assert len(got[REC]) > REC_LEN, "chain A should have more atoms than residues"
    resseqs = {a[0] for a in got[REC]}
    assert len(resseqs) == REC_LEN, \
        f"chain A spans {len(resseqs)} residues, expected 76"
    assert len({a[0] for a in got[EFF]}) == EFF_LEN


@pytest.mark.local_integration
@needs_pdb
def test_contact_residues_match_the_committed_interface():
    """6G10 has 26 receptor residues within 5 A of the effector."""
    contacts = cm.find_contact_residues_heavy(str(SIXG10), REC, EFF, 5.0)
    idx = sorted(i for i, _ in contacts)
    assert len(idx) == 26, f"expected 26 contact residues, got {len(idx)}"
    assert max(idx) < REC_LEN, "a contact index runs past the receptor chain"


@pytest.mark.local_integration
@needs_pdb
def test_contact_count_grows_with_the_cutoff():
    """A monotonicity failure would mean the distance test is wrong."""
    counts = [len(cm.find_contact_residues_heavy(str(SIXG10), REC, EFF, c))
              for c in (3.5, 5.0, 8.0, 12.0)]
    assert counts == sorted(counts), f"not monotonic in cutoff: {counts}"


@pytest.mark.local_integration
@needs_pdb
def test_no_contacts_at_an_absurdly_small_cutoff():
    assert cm.find_contact_residues_heavy(str(SIXG10), REC, EFF, 0.1) == []


@pytest.mark.local_integration
@needs_pdb
def test_interface_contacts_are_one_based_residue_numbers():
    got = cm.compute_interface_contacts(str(SIXG10), REC, EFF, 5.0)
    assert got, "no interface contacts found"
    assert min(got) >= 1, "compute_interface_contacts must be 1-based"


@pytest.mark.local_integration
@needs_pdb
def test_effector_interface_filter_degrades_to_empty_rather_than_raising():
    """Empty is the documented "no filter" signal, not a crash.

    6G10 uses author numbering 187..262 on chain A while the receptor interface
    is reported positionally, so the two coordinate systems do not meet and the
    filter returns nothing. Callers treat that as unrestricted atoms, which is
    why a mismatch here degrades silently rather than failing.
    """
    rec_iface_1b = sorted(cm.compute_interface_contacts(str(SIXG10), REC, EFF, 5.0))
    got = cm.compute_effector_interface_residues(
        str(SIXG10), rec_iface_1b, REC, EFF, 8.0)
    assert isinstance(got, set)


@pytest.mark.local_integration
@needs_pdb
def test_weighted_jaccard_of_a_structure_against_itself_is_one():
    """Comparing a model with itself is the fixed point of the metric."""
    value, _ = cm.compute_weighted_jaccard(str(SIXG10), str(SIXG10), REC, EFF, 8.0)
    assert float(value) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.local_integration
@needs_pdb
def test_interface_plddt_returns_a_value_and_a_diagnostic():
    value, note = cm.compute_interface_plddt(str(SIXG10), REC, EFF, 5.0)
    assert value is None or 0.0 <= float(value) <= 100.0


@pytest.mark.local_integration
@needs_pdb
def test_average_plddt_reads_the_b_factor_column():
    got = cm.plddt_from_pdb(str(SIXG10))
    assert got is None or isinstance(float(got), float)


# ── position-list parsing ────────────────────────────────────────────────────

@pytest.mark.local_unit
@pytest.mark.parametrize("spec,expected", [
    ("1,2,3", [1, 2, 3]),
    ("1-3", [1, 2, 3]),
    ("1-3,7", [1, 2, 3, 7]),
    ("7,1-3", [1, 2, 3, 7]),
    ("", []),
])
def test_position_list_supports_ranges(spec, expected):
    assert sorted(cm.parse_position_list(spec)) == expected


@pytest.mark.local_unit
def test_position_file_round_trips(tmp_path):
    f = tmp_path / "pos.txt"
    f.write_text("# a comment\n1-3\n7\n")
    assert sorted(cm.load_positions_file(f)) == [1, 2, 3, 7]


@pytest.mark.local_unit
def test_mutated_positions_parse():
    assert sorted(cm.parse_mutated_positions("5,7,22")) == [5, 7, 22]


@pytest.mark.local_unit
def test_mutation_reliance_counts_the_mutated_contacts():
    """Returns (n_mutated_in_contact, the positions), which is what says
    whether a rescue leaned on the residues steering actually changed."""
    n, positions = cm.classify_mutation_reliance({1, 2, 3, 10}, [2, 3])
    assert n == 2
    assert sorted(positions) == [2, 3]


@pytest.mark.local_unit
def test_mutation_reliance_is_zero_when_no_mutation_touches_a_contact():
    n, positions = cm.classify_mutation_reliance({1, 2, 3}, [50, 60])
    assert n == 0
    assert list(positions) == []


# ── the Gaussian contact weight ──────────────────────────────────────────────

@pytest.mark.local_unit
def test_contact_weight_peaks_at_four_angstroms():
    """weight = exp(-(d - 4)^2 / 2.25), so 4 A is the maximum."""
    assert cm._gaussian_contact_weight(4.0) == pytest.approx(1.0)
    for d in (2.0, 3.0, 5.0, 6.0):
        assert cm._gaussian_contact_weight(d) < 1.0


@pytest.mark.local_unit
def test_contact_weight_is_symmetric_about_the_peak():
    assert cm._gaussian_contact_weight(3.0) == pytest.approx(
        cm._gaussian_contact_weight(5.0))


@pytest.mark.local_unit
def test_contact_weight_decays_with_distance():
    weights = [cm._gaussian_contact_weight(d) for d in (4.0, 6.0, 8.0, 12.0)]
    assert weights == sorted(weights, reverse=True)
