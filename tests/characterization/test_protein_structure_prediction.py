"""Unit tests for bin/protein_structure_prediction.py (Phase 4 Tier 2).

Construction-time + geometry-helper coverage runs without gemmi.
Loading-real-structures tests skip cleanly when no fixture is present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin"))

import protein_structure_prediction as psp  # noqa: E402


@pytest.mark.logic
class TestConstruction:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            psp.ProteinStructurePrediction(
                path=tmp_path / "nope.pdb",
                receptor_chain="A",
                effector_chain="B",
            )

    def test_empty_receptor_chain_raises(self, tmp_path):
        p = tmp_path / "f.pdb"
        p.write_text("")
        with pytest.raises(ValueError, match="receptor_chain"):
            psp.ProteinStructurePrediction(
                path=p, receptor_chain="", effector_chain="B")

    def test_same_chain_raises(self, tmp_path):
        p = tmp_path / "f.pdb"
        p.write_text("")
        with pytest.raises(ValueError, match="cannot both be"):
            psp.ProteinStructurePrediction(
                path=p, receptor_chain="A", effector_chain="A")

    def test_invalid_predictor_raises(self, tmp_path):
        p = tmp_path / "f.pdb"
        p.write_text("")
        with pytest.raises(ValueError, match="predictor"):
            psp.ProteinStructurePrediction(
                path=p, receptor_chain="A", effector_chain="B",
                predictor="rosettafold")

    def test_basic_construction(self, tmp_path):
        p = tmp_path / "f.pdb"
        p.write_text("")
        x = psp.ProteinStructurePrediction(
            path=p, receptor_chain="A", effector_chain="B")
        assert x.receptor_chain == "A"
        assert x.effector_chain == "B"
        assert x.path == p
        assert x.predictor is None


@pytest.mark.logic
class TestThreeToOne:
    def test_standard_aas(self):
        assert psp.THREE_TO_ONE["ALA"] == "A"
        assert psp.THREE_TO_ONE["VAL"] == "V"
        assert psp.THREE_TO_ONE["TRP"] == "W"

    def test_non_standard(self):
        assert psp.THREE_TO_ONE["MSE"] == "M"


@pytest.mark.logic
class TestKabschRmsd:
    def test_identical_returns_zero(self):
        P = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        assert psp._kabsch_rmsd(P, P.copy()) == pytest.approx(0.0)

    def test_translated_returns_zero(self):
        P = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        Q = P + np.array([10.0, 20.0, 30.0])
        assert psp._kabsch_rmsd(P, Q) == pytest.approx(0.0, abs=1e-10)

    def test_rotated_returns_zero(self):
        P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        Q = P @ R.T
        assert psp._kabsch_rmsd(P, Q) == pytest.approx(0.0, abs=1e-10)

    def test_fewer_than_3_points_returns_none(self):
        P = np.array([[0, 0, 0]], dtype=float)
        assert psp._kabsch_rmsd(P, P.copy()) is None

    def test_pairs_to_shorter(self):
        P = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [9, 9, 9]], dtype=float)
        Q = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        assert psp._kabsch_rmsd(P, Q) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.logic
class TestRaEffRmsd:
    def test_perfect_overlap(self):
        rec = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        eff = np.array([[5, 5, 5], [6, 5, 5], [5, 6, 5]], dtype=float)
        assert psp._ra_eff_rmsd(rec, eff, rec, eff) == pytest.approx(0.0)

    def test_translated_complex_still_zero(self):
        rec = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        eff = np.array([[5, 5, 5], [6, 5, 5], [5, 6, 5]], dtype=float)
        t = np.array([10.0, 20.0, 30.0])
        assert psp._ra_eff_rmsd(rec + t, eff + t, rec, eff) == pytest.approx(
            0.0, abs=1e-10)

    def test_effector_displaced_independently(self):
        rec = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        eff = np.array([[5, 5, 5], [6, 5, 5], [5, 6, 5]], dtype=float)
        eff_d = eff + np.array([1.0, 0.0, 0.0])
        # With perfect receptor alignment (rec == rec), effector shift
        # of (1,0,0) means every atom is 1Å off → RMSD = 1
        assert psp._ra_eff_rmsd(rec, eff_d, rec, eff) == pytest.approx(1.0)

    def test_too_few_residues_returns_none(self):
        rec = np.array([[0, 0, 0]], dtype=float)
        eff = np.array([[0, 0, 0]], dtype=float)
        assert psp._ra_eff_rmsd(rec, eff, rec, eff) is None


def _has_gemmi():
    try:
        import gemmi  # noqa: F401
        return True
    except ImportError:
        return False


_REAL_PDB_DIR = (
    _REPO_ROOT / "data" / "reference_complexes"
)


def _find_real_pdb():
    if not _REAL_PDB_DIR.is_dir():
        return None
    candidates = list(_REAL_PDB_DIR.glob("*.pdb"))
    return candidates[0] if candidates else None


@pytest.mark.logic
@pytest.mark.skipif(not _has_gemmi(), reason="gemmi not installed")
class TestRealStructure:
    def test_load_and_read_sequences(self):
        pdb = _find_real_pdb()
        if pdb is None:
            pytest.skip("no real PDB fixture available")
        x = psp.ProteinStructurePrediction(
            path=pdb, receptor_chain="A", effector_chain="B",
            predictor="boltz")
        rec = x.receptor_sequence()
        eff = x.effector_sequence()
        assert len(rec) > 10 and len(eff) > 10
        assert all(c in "ACDEFGHIKLMNPQRSTVWYUOX" for c in rec)
        assert all(c in "ACDEFGHIKLMNPQRSTVWYUOX" for c in eff)

    def test_lazy_loading(self):
        pdb = _find_real_pdb()
        if pdb is None:
            pytest.skip("no real PDB fixture available")
        x = psp.ProteinStructurePrediction(
            path=pdb, receptor_chain="A", effector_chain="B")
        assert x._structure_cache is None
        _ = x.receptor_sequence()
        assert x._structure_cache is not None

    def test_chain_ca_coords_shape(self):
        pdb = _find_real_pdb()
        if pdb is None:
            pytest.skip("no real PDB fixture available")
        x = psp.ProteinStructurePrediction(
            path=pdb, receptor_chain="A", effector_chain="B")
        ca = x.receptor_ca_coords()
        assert ca.ndim == 2 and ca.shape[1] == 3
        assert ca.shape[0] == x.receptor_length()

    def test_contact_residues_returns_position_set(self):
        pdb = _find_real_pdb()
        if pdb is None:
            pytest.skip("no real PDB fixture available")
        x = psp.ProteinStructurePrediction(
            path=pdb, receptor_chain="A", effector_chain="B")
        cs = x.receptor_contact_residues(cutoff=4.5)
        assert cs.chain == "A"
        assert cs.frame == "prediction"
        assert len(cs) > 0

    def test_ra_eff_against_self_is_zero(self):
        pdb = _find_real_pdb()
        if pdb is None:
            pytest.skip("no real PDB fixture available")
        x = psp.ProteinStructurePrediction(
            path=pdb, receptor_chain="A", effector_chain="B")
        x2 = psp.ProteinStructurePrediction(
            path=pdb, receptor_chain="A", effector_chain="B")
        assert x.ra_eff_vs(x2) == pytest.approx(0.0, abs=1e-6)

    def test_jaccard_against_self_is_one(self):
        pdb = _find_real_pdb()
        if pdb is None:
            pytest.skip("no real PDB fixture available")
        x = psp.ProteinStructurePrediction(
            path=pdb, receptor_chain="A", effector_chain="B")
        x2 = psp.ProteinStructurePrediction(
            path=pdb, receptor_chain="A", effector_chain="B")
        assert x.jaccard_interface_vs(x2, cutoff=4.5) == pytest.approx(1.0)

    def test_independent_rmsds_against_self_are_zero(self):
        pdb = _find_real_pdb()
        if pdb is None:
            pytest.skip("no real PDB fixture available")
        x = psp.ProteinStructurePrediction(
            path=pdb, receptor_chain="A", effector_chain="B")
        x2 = psp.ProteinStructurePrediction(
            path=pdb, receptor_chain="A", effector_chain="B")
        assert x.independent_receptor_rmsd_vs(x2) == pytest.approx(0.0, abs=1e-6)
        assert x.independent_effector_rmsd_vs(x2) == pytest.approx(0.0, abs=1e-6)


# ── caching, repr, and the parse paths ───────────────────────────────────────
#
# Every accessor caches per chain. A cache keyed wrongly would return one
# chain's coordinates for another, which is the kind of error that produces a
# plausible RMSD against the wrong thing.

def _two_chain_pdb(path, *, alt=False, hydrogens=False):
    lines, serial = [], 1
    for chain, n, aa in (("A", 3, "ALA"), ("B", 2, "GLY")):
        for r in range(1, n + 1):
            atoms = [("CA", "C")]
            if hydrogens:
                atoms.append(("H", "H"))
            for name, elem in atoms:
                lines.append(
                    f"ATOM  {serial:5d}  {name:<3s} {aa} {chain}{r:4d}"
                    f"    {r:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00"
                    f"          {elem:>2s}")
                serial += 1
            if alt and r == 1:
                lines.append(
                    f"ATOM  {serial:5d}  CA  {aa} {chain}{r:4d}"
                    f"    {99.0:8.3f}{0.0:8.3f}{0.0:8.3f}  0.50  0.00"
                    f"           C")
                serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.mark.logic
@pytest.mark.skipif(not _has_gemmi(), reason="gemmi not installed")
class TestParsingAndCaching:
    def _psp(self, tmp_path, **kw):
        p = _two_chain_pdb(tmp_path / "x.pdb", **kw)
        return psp.ProteinStructurePrediction(
            path=p, receptor_chain="A", effector_chain="B")

    def test_repr_names_the_file_and_both_chains(self, tmp_path):
        r = repr(self._psp(tmp_path))
        assert "x.pdb" in r and "'A'" in r and "'B'" in r

    def test_sequences_are_cached_per_chain(self, tmp_path):
        x = self._psp(tmp_path)
        assert x.receptor_sequence() == "AAA"
        assert x.effector_sequence() == "GG"
        # Second call comes from the cache, and must not cross chains.
        assert x.receptor_sequence() == "AAA"
        assert x.effector_length() == 2

    def test_alternate_conformations_are_counted_once(self, tmp_path):
        """A duplicated residue would lengthen the sequence and shift every
        positional index after it."""
        x = self._psp(tmp_path, alt=True)
        assert x.receptor_sequence() == "AAA"

    def test_ca_coordinates_are_cached_per_chain(self, tmp_path):
        x = self._psp(tmp_path)
        a = x.receptor_ca_coords()
        assert a.shape == (3, 3)
        assert x.receptor_ca_coords() is a
        assert x.effector_ca_coords().shape == (2, 3)

    def test_a_chain_with_no_atoms_gives_an_empty_array(self, tmp_path):
        """Empty rather than raising, so a missing chain degrades to 'no
        contacts' instead of taking the metrics step down."""
        p = _two_chain_pdb(tmp_path / "x.pdb")
        x = psp.ProteinStructurePrediction(
            path=p, receptor_chain="A", effector_chain="Z")
        assert x.effector_ca_coords().shape == (0, 3)

    def test_hydrogens_are_excluded_from_heavy_atoms(self, tmp_path):
        x = self._psp(tmp_path, hydrogens=True)
        atoms = x._chain_heavy_atoms_list("A")
        assert len(atoms) == 3          # the three CAs, not the three Hs

    def test_heavy_atoms_are_cached_per_chain(self, tmp_path):
        x = self._psp(tmp_path)
        first = x._chain_heavy_atoms_list("A")
        assert x._chain_heavy_atoms_list("A") is first
        assert len(x._chain_heavy_atoms_list("B")) == 2


@pytest.mark.logic
def test_an_empty_effector_chain_is_refused(tmp_path):
    p = tmp_path / "x.pdb"
    p.write_text("END\n")
    with pytest.raises(ValueError, match="effector_chain"):
        psp.ProteinStructurePrediction(path=p, receptor_chain="A",
                                       effector_chain="")
