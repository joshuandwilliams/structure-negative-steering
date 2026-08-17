"""Unit tests for bin/designed_backbone.py (Phase 4 Tier 3)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin"))

import contig_spec as cs  # noqa: E402
import designed_backbone as db  # noqa: E402
import protein_structure_prediction as psp  # noqa: E402


def _make_fake_psp(tmp_path, receptor="A", effector="B"):
    p = tmp_path / "fake.pdb"
    p.write_text("")
    return psp.ProteinStructurePrediction(
        path=p, receptor_chain=receptor, effector_chain=effector)


@pytest.fixture
def contig():
    return cs.ContigSpec.from_resolved_string("A1-10/5/A15-20 B")


@pytest.mark.local_unit
class TestConstruction:
    def test_basic(self, tmp_path, contig):
        s = _make_fake_psp(tmp_path)
        b = db.DesignedBackbone(
            pdb_path=tmp_path / "fake.pdb",
            design_id="design_03",
            contig=contig,
            structure=s,
            sc_score=0.7,
        )
        assert b.design_id == "design_03"
        assert b.sc_score == 0.7
        assert b.rfdiff_metrics is None

    def test_empty_design_id_raises(self, tmp_path, contig):
        s = _make_fake_psp(tmp_path)
        with pytest.raises(ValueError, match="design_id"):
            db.DesignedBackbone(
                pdb_path=tmp_path / "fake.pdb", design_id="",
                contig=contig, structure=s)

    def test_wrong_contig_type_raises(self, tmp_path):
        s = _make_fake_psp(tmp_path)
        with pytest.raises(TypeError, match="ContigSpec"):
            db.DesignedBackbone(
                pdb_path=tmp_path / "fake.pdb",
                design_id="design_03",
                contig="A1-10/5/A15-20",  # raw string
                structure=s)

    def test_wrong_structure_type_raises(self, tmp_path, contig):
        with pytest.raises(TypeError, match="ProteinStructurePrediction"):
            db.DesignedBackbone(
                pdb_path=tmp_path / "fake.pdb",
                design_id="design_03",
                contig=contig,
                structure=None)


@pytest.mark.local_unit
class TestAccessors:
    def test_design_region(self, tmp_path, contig):
        s = _make_fake_psp(tmp_path)
        b = db.DesignedBackbone(
            pdb_path=tmp_path / "fake.pdb",
            design_id="design_03", contig=contig, structure=s)
        dr = b.design_region()
        assert sorted(dr) == [11, 12, 13, 14, 15]
        assert dr.chain == "A"
        assert dr.frame == "designed"
        assert dr.contig is contig

    def test_fixed_anchor_positions(self, tmp_path, contig):
        s = _make_fake_psp(tmp_path)
        b = db.DesignedBackbone(
            pdb_path=tmp_path / "fake.pdb",
            design_id="design_03", contig=contig, structure=s)
        anchors = b.fixed_anchor_positions()
        assert sorted(anchors) == list(range(1, 11)) + list(range(16, 22))
        assert anchors.frame == "designed"

    def test_receptor_length(self, tmp_path, contig):
        s = _make_fake_psp(tmp_path)
        b = db.DesignedBackbone(
            pdb_path=tmp_path / "fake.pdb",
            design_id="design_03", contig=contig, structure=s)
        assert b.receptor_length() == 21

    def test_chain_passthrough(self, tmp_path, contig):
        s = _make_fake_psp(tmp_path, receptor="X", effector="Y")
        b = db.DesignedBackbone(
            pdb_path=tmp_path / "fake.pdb",
            design_id="d", contig=contig, structure=s)
        assert b.receptor_chain == "X"
        assert b.effector_chain == "Y"


@pytest.mark.local_unit
class TestFromPdbAndContig:
    def test_constructs_via_convenience(self, tmp_path):
        p = tmp_path / "f.pdb"
        p.write_text("")
        b = db.DesignedBackbone.from_pdb_and_contig(
            pdb_path=p, design_id="design_07",
            contig_string="A1-10/5/A15-20 B", sc_score=0.65)
        assert b.design_id == "design_07"
        assert b.receptor_length() == 21
        assert b.sc_score == 0.65


@pytest.mark.local_unit
class TestImmutability:
    def test_frozen(self, tmp_path, contig):
        s = _make_fake_psp(tmp_path)
        b = db.DesignedBackbone(
            pdb_path=tmp_path / "fake.pdb",
            design_id="d", contig=contig, structure=s)
        with pytest.raises((AttributeError, Exception)):
            b.design_id = "new"  # type: ignore


@pytest.mark.local_unit
class TestRepr:
    def test_repr_includes_id_and_sc(self, tmp_path, contig):
        s = _make_fake_psp(tmp_path)
        b = db.DesignedBackbone(
            pdb_path=tmp_path / "fake.pdb",
            design_id="design_03", contig=contig, structure=s, sc_score=0.7)
        r = repr(b)
        assert "design_03" in r
        assert "0.700" in r
