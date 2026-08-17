"""Unit tests for bin/position_set.py (Phase 4 Tier 1)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin"))

import contig_spec as cs  # noqa: E402
import position_set as ps  # noqa: E402


@pytest.fixture
def contig():
    return cs.ContigSpec.from_resolved_string("A1-10/5/A15-20 B")


@pytest.mark.local_unit
class TestConstruction:
    def test_basic(self):
        s = ps.PositionSet([1, 5, 10], chain="A", frame="designed")
        assert s.chain == "A"
        assert s.frame == "designed"
        assert len(s) == 3

    def test_dedups(self):
        s = ps.PositionSet([1, 1, 2, 3, 3], chain="A", frame="designed")
        assert len(s) == 3
        assert sorted(s) == [1, 2, 3]

    def test_sorted_iteration(self):
        s = ps.PositionSet([3, 1, 2], chain="A", frame="designed")
        assert list(s) == [1, 2, 3]

    def test_invalid_frame_raises(self):
        with pytest.raises(ValueError, match="frame"):
            ps.PositionSet([1], chain="A", frame="bogus")

    def test_invalid_chain_raises(self):
        with pytest.raises(ValueError, match="chain"):
            ps.PositionSet([1], chain="", frame="designed")

    def test_empty(self):
        s = ps.PositionSet([], chain="A", frame="designed")
        assert s.is_empty()
        assert len(s) == 0


@pytest.mark.local_unit
class TestSetOperations:
    def test_union(self):
        a = ps.PositionSet([1, 2], chain="A", frame="designed")
        b = ps.PositionSet([2, 3], chain="A", frame="designed")
        u = a.union(b)
        assert sorted(u) == [1, 2, 3]
        assert (a | b) == u

    def test_intersection(self):
        a = ps.PositionSet([1, 2, 3], chain="A", frame="designed")
        b = ps.PositionSet([2, 3, 4], chain="A", frame="designed")
        assert sorted(a.intersection(b)) == [2, 3]
        assert (a & b) == a.intersection(b)

    def test_difference(self):
        a = ps.PositionSet([1, 2, 3], chain="A", frame="designed")
        b = ps.PositionSet([2], chain="A", frame="designed")
        assert sorted(a.difference(b)) == [1, 3]
        assert (a - b) == a.difference(b)

    def test_contains(self):
        s = ps.PositionSet([1, 5, 10], chain="A", frame="designed")
        assert 1 in s
        assert 99 not in s

    def test_frame_mismatch_raises(self):
        a = ps.PositionSet([1], chain="A", frame="designed")
        b = ps.PositionSet([1], chain="A", frame="native")
        with pytest.raises(ps.FrameMismatchError, match="frame"):
            a.union(b)

    def test_chain_mismatch_raises(self):
        a = ps.PositionSet([1], chain="A", frame="designed")
        b = ps.PositionSet([1], chain="B", frame="designed")
        with pytest.raises(ps.FrameMismatchError, match="chain"):
            a.intersection(b)

    def test_non_position_set_raises(self):
        a = ps.PositionSet([1], chain="A", frame="designed")
        with pytest.raises(TypeError):
            a.union([1, 2, 3])

    def test_protected_set_pattern(self):
        design = ps.PositionSet([11, 12, 13, 14, 15], chain="A", frame="designed")
        interface = ps.PositionSet([10, 11, 16], chain="A", frame="designed")
        assert sorted(design | interface) == [10, 11, 12, 13, 14, 15, 16]


@pytest.mark.local_unit
class TestFrameConversion:
    def test_designed_to_native_drops_unmappable(self, contig):
        s = ps.PositionSet(
            list(range(1, 17)), chain="A", frame="designed", contig=contig
        )
        n = s.in_frame("native")
        assert n.frame == "native"
        assert sorted(n) == list(range(1, 11)) + [15]

    def test_strict_raises_on_unmappable(self, contig):
        s = ps.PositionSet([5, 12], chain="A", frame="designed", contig=contig)
        with pytest.raises(ps.FrameConversionError, match="strict=True"):
            s.in_frame("native", strict=True)

    def test_native_to_designed(self, contig):
        s = ps.PositionSet([1, 10, 15, 20], chain="A", frame="native", contig=contig)
        assert sorted(s.in_frame("designed")) == [1, 10, 16, 21]

    def test_native_to_designed_drops_unanchored(self, contig):
        s = ps.PositionSet([5, 11, 12, 15], chain="A", frame="native", contig=contig)
        assert sorted(s.in_frame("designed")) == [5, 16]

    def test_no_contig_raises(self):
        s = ps.PositionSet([1], chain="A", frame="designed")
        with pytest.raises(ps.FrameConversionError, match="no ContigSpec"):
            s.in_frame("native")

    def test_same_frame_returns_self(self, contig):
        s = ps.PositionSet([1], chain="A", frame="designed", contig=contig)
        assert s.in_frame("designed") is s

    def test_invalid_target_frame_raises(self, contig):
        s = ps.PositionSet([1], chain="A", frame="designed", contig=contig)
        with pytest.raises(ValueError, match="target_frame"):
            s.in_frame("bogus")

    def test_prediction_to_designed_identity(self, contig):
        s = ps.PositionSet([5, 10, 15], chain="A", frame="prediction", contig=contig)
        assert sorted(s.in_frame("designed")) == [5, 10, 15]


@pytest.mark.local_unit
class TestImmutability:
    def test_no_attribute_assignment(self):
        s = ps.PositionSet([1], chain="A", frame="designed")
        with pytest.raises((AttributeError, Exception)):
            s.chain = "B"  # type: ignore

    def test_set_op_returns_new_instance(self):
        a = ps.PositionSet([1, 2], chain="A", frame="designed")
        u = a | ps.PositionSet([3], chain="A", frame="designed")
        assert u is not a
        assert sorted(a) == [1, 2]


@pytest.mark.local_unit
class TestChimeraXFormat:
    def test_to_chimerax(self):
        s = ps.PositionSet([5, 7, 22], chain="A", frame="designed")
        assert s.to_chimerax_string() == "/A:5,7,22"

    def test_to_chimerax_empty(self):
        assert ps.PositionSet([], chain="A", frame="designed").to_chimerax_string() == ""

    def test_round_trip(self):
        a = ps.PositionSet([5, 7, 22], chain="A", frame="designed")
        b = ps.PositionSet.from_chimerax_string(a.to_chimerax_string(), frame="designed")
        assert a == b

    def test_round_trip_different_chain(self):
        a = ps.PositionSet([1, 2, 3], chain="B", frame="native")
        b = ps.PositionSet.from_chimerax_string(a.to_chimerax_string(), frame="native")
        assert b.chain == "B"

    def test_from_empty_string(self):
        s = ps.PositionSet.from_chimerax_string("", frame="designed")
        assert s.is_empty()

    def test_from_malformed_raises(self):
        with pytest.raises(ValueError, match="chimerax"):
            ps.PositionSet.from_chimerax_string("A:1,2", frame="designed")

    def test_from_chimerax_non_int_raises(self):
        with pytest.raises(ValueError, match="non-integer"):
            ps.PositionSet.from_chimerax_string("/A:1,foo,3", frame="designed")


@pytest.mark.local_unit
class TestTextFile:
    def test_round_trip(self, tmp_path):
        a = ps.PositionSet([5, 7, 22], chain="A", frame="designed")
        path = tmp_path / "i.txt"
        a.to_text_file(path)
        b = ps.PositionSet.from_text_file(path, chain="A", frame="designed")
        assert a == b

    def test_writes_sorted_one_per_line(self, tmp_path):
        s = ps.PositionSet([22, 5, 7], chain="A", frame="designed")
        path = tmp_path / "out.txt"
        s.to_text_file(path)
        assert path.read_text().strip() == "5\n7\n22"

    def test_reads_blank_and_comments(self, tmp_path):
        path = tmp_path / "in.txt"
        path.write_text("# comment\n1\n\n2\n# another\n3\n")
        loaded = ps.PositionSet.from_text_file(path, chain="A", frame="designed")
        assert sorted(loaded) == [1, 2, 3]

    def test_non_integer_raises(self, tmp_path):
        path = tmp_path / "bad.txt"
        path.write_text("1\n2\nfoo\n")
        with pytest.raises(ValueError, match="non-integer"):
            ps.PositionSet.from_text_file(path, chain="A", frame="designed")


@pytest.mark.local_unit
class TestEqualityAndHash:
    def test_equal_when_same_state(self):
        a = ps.PositionSet([1, 2], chain="A", frame="designed")
        b = ps.PositionSet([2, 1], chain="A", frame="designed")
        assert a == b
        assert hash(a) == hash(b)

    def test_unequal_when_frame_differs(self):
        a = ps.PositionSet([1], chain="A", frame="designed")
        b = ps.PositionSet([1], chain="A", frame="native")
        assert a != b

    def test_unequal_when_chain_differs(self):
        a = ps.PositionSet([1], chain="A", frame="designed")
        b = ps.PositionSet([1], chain="B", frame="designed")
        assert a != b

    def test_contig_does_not_affect_equality(self, contig):
        a = ps.PositionSet([1], chain="A", frame="designed")
        b = ps.PositionSet([1], chain="A", frame="designed", contig=contig)
        assert a == b


@pytest.mark.local_unit
class TestRepr:
    def test_small(self):
        s = ps.PositionSet([1, 2], chain="A", frame="designed")
        r = repr(s)
        assert "PositionSet" in r and "chain='A'" in r

    def test_large_truncated(self):
        s = ps.PositionSet(range(100), chain="A", frame="designed")
        assert "100 total" in repr(s)
