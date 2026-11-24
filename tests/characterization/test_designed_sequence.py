"""Unit tests for bin/designed_sequence.py (Phase 4 Tier 4.1)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin"))

import contig_spec as cs  # noqa: E402
import designed_backbone as db  # noqa: E402
import designed_sequence as ds  # noqa: E402
import protein_structure_prediction as psp  # noqa: E402


def _make_backbone(tmp_path):
    p = tmp_path / "f.pdb"
    p.write_text("")
    contig = cs.ContigSpec.from_resolved_string("A1-10/5/A15-20 B")
    s = psp.ProteinStructurePrediction(
        path=p, receptor_chain="A", effector_chain="B")
    return db.DesignedBackbone(
        pdb_path=p, design_id="design_03", contig=contig, structure=s)


_RECEPTOR_21 = "A" * 10 + "MNRSV" + "L" * 6   # 21 residues
_NATIVE_DR = "MNRSV"   # the wild-type residues at design positions
_MPNN_DR = "MPRSY"     # MPNN-designed at design positions


@pytest.mark.logic
class TestConstruction:
    def test_basic(self, tmp_path):
        b = _make_backbone(tmp_path)
        d = ds.DesignedSequence(
            backbone=b, sequence_id="design_03_seq_01",
            corrected_receptor=_RECEPTOR_21,
            designed_residues=_MPNN_DR,
            native_residues=_NATIVE_DR,
            mpnn_score=0.8)
        assert d.sequence_id == "design_03_seq_01"
        assert d.design_id == "design_03"
        assert d.receptor_chain == "A"

    def test_empty_sequence_id_raises(self, tmp_path):
        b = _make_backbone(tmp_path)
        with pytest.raises(ValueError, match="sequence_id"):
            ds.DesignedSequence(
                backbone=b, sequence_id="",
                corrected_receptor=_RECEPTOR_21,
                designed_residues=_MPNN_DR, native_residues=_NATIVE_DR)

    def test_wrong_backbone_type_raises(self):
        with pytest.raises(TypeError, match="DesignedBackbone"):
            ds.DesignedSequence(
                backbone="not a backbone", sequence_id="d",
                corrected_receptor=_RECEPTOR_21,
                designed_residues=_MPNN_DR, native_residues=_NATIVE_DR)

    def test_short_corrected_receptor_raises(self, tmp_path):
        b = _make_backbone(tmp_path)
        with pytest.raises(ValueError, match="too short"):
            ds.DesignedSequence(
                backbone=b, sequence_id="d",
                corrected_receptor="A" * 5,  # too short for the 21-residue contig
                designed_residues=_MPNN_DR, native_residues=_NATIVE_DR)


@pytest.mark.logic
class TestSequenceViews:
    def test_to_fasta(self, tmp_path):
        b = _make_backbone(tmp_path)
        d = ds.DesignedSequence(
            backbone=b, sequence_id="d",
            corrected_receptor=_RECEPTOR_21,
            designed_residues=_MPNN_DR, native_residues=_NATIVE_DR)
        fasta = d.to_fasta()
        assert fasta.startswith(">d\n")
        assert _RECEPTOR_21 in fasta

    def test_mutations_vs_native(self, tmp_path):
        b = _make_backbone(tmp_path)
        d = ds.DesignedSequence(
            backbone=b, sequence_id="d",
            corrected_receptor=_RECEPTOR_21,
            designed_residues=_MPNN_DR, native_residues=_NATIVE_DR)
        # _MPNN_DR = "MPRSY" vs _NATIVE_DR = "MNRSV"
        # Position 0: M vs M (same)
        # Position 1: P vs N (different) -> at design position 12
        # Position 2: R vs R (same)
        # Position 3: S vs S (same)
        # Position 4: Y vs V (different) -> at design position 15
        muts = d.mutations_vs_native()
        # Design region is 11..15
        assert (12, "N", "P") in muts
        assert (15, "V", "Y") in muts
        assert len(muts) == 2

    def test_n_changes(self, tmp_path):
        b = _make_backbone(tmp_path)
        d = ds.DesignedSequence(
            backbone=b, sequence_id="d",
            corrected_receptor=_RECEPTOR_21,
            designed_residues=_MPNN_DR, native_residues=_NATIVE_DR)
        assert d.n_changes() == 2

    def test_no_changes_when_identical(self, tmp_path):
        b = _make_backbone(tmp_path)
        d = ds.DesignedSequence(
            backbone=b, sequence_id="d",
            corrected_receptor=_RECEPTOR_21,
            designed_residues=_NATIVE_DR, native_residues=_NATIVE_DR)
        assert d.mutations_vs_native() == []
        assert d.n_changes() == 0


@pytest.mark.logic
class TestImmutability:
    def test_frozen(self, tmp_path):
        b = _make_backbone(tmp_path)
        d = ds.DesignedSequence(
            backbone=b, sequence_id="d",
            corrected_receptor=_RECEPTOR_21,
            designed_residues=_MPNN_DR, native_residues=_NATIVE_DR)
        with pytest.raises((AttributeError, Exception)):
            d.sequence_id = "new"  # type: ignore


# ── the remaining type guards and the repr ───────────────────────────────────
#
# Each guard turns a wrong-typed field into an error at construction. Without
# them a non-string slips through and fails much later, inside a string
# operation, where the traceback no longer names the field that was wrong.

@pytest.mark.logic
class TestFieldTypeGuards:
    def _kwargs(self, tmp_path):
        return dict(backbone=_make_backbone(tmp_path),
                    sequence_id="design_03_seq_01",
                    corrected_receptor=_RECEPTOR_21,
                    designed_residues=_MPNN_DR,
                    native_residues=_NATIVE_DR)

    @pytest.mark.parametrize("field", ["corrected_receptor", "designed_residues",
                                       "native_residues"])
    def test_a_non_string_field_is_a_type_error(self, tmp_path, field):
        kw = self._kwargs(tmp_path)
        kw[field] = 123
        with pytest.raises(TypeError, match=field):
            ds.DesignedSequence(**kw)

    @pytest.mark.parametrize("bad", ["", None])
    def test_an_empty_sequence_id_is_rejected(self, tmp_path, bad):
        kw = self._kwargs(tmp_path)
        kw["sequence_id"] = bad
        with pytest.raises((ValueError, TypeError)):
            ds.DesignedSequence(**kw)


@pytest.mark.logic
class TestRepr:
    def _one(self, tmp_path, score):
        return ds.DesignedSequence(
            backbone=_make_backbone(tmp_path), sequence_id="design_03_seq_01",
            corrected_receptor=_RECEPTOR_21, designed_residues=_MPNN_DR,
            native_residues=_NATIVE_DR, mpnn_score=score)

    def test_repr_formats_the_score_to_three_places(self, tmp_path):
        r = repr(self._one(tmp_path, 1.23456))
        assert "design_03_seq_01" in r and "1.235" in r

    def test_repr_says_none_when_there_is_no_score(self, tmp_path):
        assert "mpnn_score=None" in repr(self._one(tmp_path, None))


@pytest.mark.logic
def test_n_changes_counts_the_differing_positions(tmp_path):
    d = ds.DesignedSequence(
        backbone=_make_backbone(tmp_path), sequence_id="s",
        corrected_receptor=_RECEPTOR_21, designed_residues=_MPNN_DR,
        native_residues=_NATIVE_DR)
    # MNRSV -> MPRSY differs at two positions.
    assert d.n_changes() == len(d.mutations_vs_native()) == 2
