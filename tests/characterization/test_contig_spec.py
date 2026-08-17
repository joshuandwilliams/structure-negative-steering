"""Unit tests for bin/contig_spec.py (Phase 4 Tier 0.5).

Covers parsing of the canonical RFDiffusion contig format, frame
conversion (designed ↔ native), design-region queries, error paths."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin"))

import contig_spec as cs  # noqa: E402


@pytest.mark.local_unit
class TestParseCanonicalForm:
    """Canonical: A1-10/5/A15-20 B"""

    def test_parses_canonical(self):
        c = cs.ContigSpec.from_resolved_string("A1-10/5/A15-20 B")
        assert c.chain_ids == ["A", "B"]
        # Chain A: fixed A1-10 (len 10) + denovo 5 + fixed A15-20 (len 6)
        a = c.chain("A")
        assert len(a.segments) == 3
        assert isinstance(a.segments[0], cs.FixedSegment)
        assert isinstance(a.segments[1], cs.DeNovoSegment)
        assert isinstance(a.segments[2], cs.FixedSegment)
        assert a.segments[0].start == 1 and a.segments[0].end == 10
        assert a.segments[1].length == 5
        assert a.segments[2].start == 15 and a.segments[2].end == 20
        assert a.total_length == 10 + 5 + 6
        # Chain B: bare passthrough
        b = c.chain("B")
        assert b.segments == ()
        assert b.total_length == 0  # unknown until PDB lookup

    def test_round_trip_string_kept(self):
        c = cs.ContigSpec.from_resolved_string("A1-10/5/A15-20 B")
        assert c.contig_string == "A1-10/5/A15-20 B"

    def test_strips_surrounding_whitespace(self):
        c = cs.ContigSpec.from_resolved_string("  A1-10/5/A15-20 B  ")
        assert c.contig_string == "A1-10/5/A15-20 B"


@pytest.mark.local_unit
class TestParseEdgeCases:
    def test_single_fixed_segment(self):
        c = cs.ContigSpec.from_resolved_string("A1-100")
        assert c.chain_ids == ["A"]
        assert c.chain("A").total_length == 100

    def test_single_residue_fixed(self):
        c = cs.ContigSpec.from_resolved_string("A42")
        # Single residue fixed: start=end=42, length=1
        seg = c.chain("A").segments[0]
        assert isinstance(seg, cs.FixedSegment)
        assert seg.start == 42 and seg.end == 42
        assert seg.length == 1

    def test_multiple_denovo_regions(self):
        c = cs.ContigSpec.from_resolved_string("A1-10/3/A11-20/4/A21-30")
        a = c.chain("A")
        denovos = [s for s in a.segments if isinstance(s, cs.DeNovoSegment)]
        assert len(denovos) == 2
        assert denovos[0].length == 3
        assert denovos[1].length == 4

    def test_bare_chain_alone(self):
        c = cs.ContigSpec.from_resolved_string("X")
        assert c.chain("X").segments == ()


@pytest.mark.local_unit
class TestParseErrors:
    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="empty"):
            cs.ContigSpec.from_resolved_string("")

    def test_range_denovo_rejected_by_from_resolved_string(self):
        """Constraint-form (e.g. ``5-7``) must be resolved upstream when
        the caller asks for the strict form."""
        with pytest.raises(ValueError, match="unresolved"):
            cs.ContigSpec.from_resolved_string("A1-10/5-7/A15-20")

    def test_invalid_fixed_range(self):
        with pytest.raises(ValueError):
            cs.ContigSpec.from_resolved_string("Abad-10")

    def test_mixed_chain_in_same_block(self):
        with pytest.raises(ValueError, match="mixes chains"):
            cs.ContigSpec.from_resolved_string("A1-10/5/B15-20")

    def test_no_chain_anchor_in_block(self):
        with pytest.raises(ValueError, match="no chain-prefixed"):
            cs.ContigSpec.from_resolved_string("5/3")


@pytest.mark.local_unit
class TestDesignedToNative:
    """For contig A1-10/5/A15-20 on chain A:
       designed 1..10 -> native 1..10 (fixed)
       designed 11..15 -> None (de novo)
       designed 16..21 -> native 15..20 (fixed)"""

    @pytest.fixture
    def spec(self):
        return cs.ContigSpec.from_resolved_string("A1-10/5/A15-20 B")

    def test_first_fixed_segment(self, spec):
        assert spec.designed_to_native(1, "A") == 1
        assert spec.designed_to_native(10, "A") == 10

    def test_de_novo_region_yields_none(self, spec):
        for designed in (11, 12, 13, 14, 15):
            assert spec.designed_to_native(designed, "A") is None

    def test_second_fixed_segment(self, spec):
        assert spec.designed_to_native(16, "A") == 15
        assert spec.designed_to_native(21, "A") == 20

    def test_out_of_range_yields_none(self, spec):
        assert spec.designed_to_native(0, "A") is None
        assert spec.designed_to_native(99, "A") is None
        assert spec.designed_to_native(-5, "A") is None


@pytest.mark.local_unit
class TestNativeToDesigned:
    """Inverse mapping."""

    @pytest.fixture
    def spec(self):
        return cs.ContigSpec.from_resolved_string("A1-10/5/A15-20 B")

    def test_first_fixed(self, spec):
        assert spec.native_to_designed(1, "A") == 1
        assert spec.native_to_designed(10, "A") == 10

    def test_second_fixed(self, spec):
        assert spec.native_to_designed(15, "A") == 16
        assert spec.native_to_designed(20, "A") == 21

    def test_native_in_gap_yields_none(self, spec):
        """Native residue 11-14 aren't anchored by any fixed segment."""
        for native in (11, 12, 13, 14):
            assert spec.native_to_designed(native, "A") is None

    def test_native_outside_yields_none(self, spec):
        assert spec.native_to_designed(50, "A") is None


@pytest.mark.local_unit
class TestDesignRegionQueries:
    @pytest.fixture
    def spec(self):
        return cs.ContigSpec.from_resolved_string("A1-10/5/A15-20")

    def test_design_region_positions(self, spec):
        # Designed positions 11..15 are the de novo region
        assert spec.design_region_positions("A") == [11, 12, 13, 14, 15]

    def test_fixed_anchor_positions(self, spec):
        # 1..10 plus 16..21
        positions = spec.fixed_anchor_positions("A")
        assert positions == list(range(1, 11)) + list(range(16, 22))

    def test_is_design_region_position(self, spec):
        assert spec.is_design_region_position(11, "A")
        assert spec.is_design_region_position(15, "A")
        assert not spec.is_design_region_position(10, "A")
        assert not spec.is_design_region_position(16, "A")
        assert not spec.is_design_region_position(99, "A")


@pytest.mark.local_unit
class TestChainAccess:
    def test_chain_length(self):
        spec = cs.ContigSpec.from_resolved_string("A1-10/5/A15-20")
        assert spec.chain_length("A") == 21

    def test_chain_missing_raises_keyerror(self):
        spec = cs.ContigSpec.from_resolved_string("A1-10")
        with pytest.raises(KeyError):
            spec.chain("Z")

    def test_fixed_segments_accessor(self):
        spec = cs.ContigSpec.from_resolved_string("A1-10/5/A15-20")
        fs = spec.fixed_segments("A")
        assert len(fs) == 2
        assert fs[0].start == 1 and fs[0].end == 10
        assert fs[1].start == 15 and fs[1].end == 20

    def test_denovo_segments_accessor(self):
        spec = cs.ContigSpec.from_resolved_string("A1-10/5/A15-20")
        ds = spec.denovo_segments("A")
        assert len(ds) == 1
        assert ds[0].length == 5


@pytest.mark.local_unit
class TestPredictionFrame:
    """prediction_to_designed is a no-op for the receptor chain — the
    method exists for caller-side explicitness."""

    def test_no_op(self):
        spec = cs.ContigSpec.from_resolved_string("A1-10/5/A15-20")
        assert spec.prediction_to_designed(7, "A") == 7
        assert spec.prediction_to_designed(15, "A") == 15


@pytest.mark.local_unit
class TestFixedSegment:
    def test_invalid_range_raises(self):
        with pytest.raises(ValueError):
            cs.FixedSegment(start=10, end=5)

    def test_length(self):
        assert cs.FixedSegment(start=1, end=10).length == 10
        assert cs.FixedSegment(start=42, end=42).length == 1


@pytest.mark.local_unit
class TestDeNovoSegment:
    def test_negative_length_raises(self):
        with pytest.raises(ValueError):
            cs.DeNovoSegment(min_len=-1, max_len=-1)

    def test_zero_length_allowed(self):
        # Edge case: a 0-length de novo segment is degenerate but allowed
        s = cs.DeNovoSegment.resolved(0)
        assert s.length == 0
        assert s.is_resolved

    def test_range_form(self):
        s = cs.DeNovoSegment(min_len=5, max_len=7)
        assert not s.is_resolved
        with pytest.raises(ValueError):
            _ = s.length

    def test_max_lt_min_raises(self):
        with pytest.raises(ValueError):
            cs.DeNovoSegment(min_len=7, max_len=5)

    def test_to_string(self):
        assert cs.DeNovoSegment.resolved(5).to_string() == "5"
        assert cs.DeNovoSegment(min_len=5, max_len=7).to_string() == "5-7"


@pytest.mark.local_unit
class TestBreakAndPassthrough:
    def test_break_marker(self):
        spec = cs.ContigSpec.from_string("A1-10/0/A15-20")
        types = [type(s).__name__ for s in spec.chain("A").segments]
        assert types == ["FixedSegment", "BreakSegment", "FixedSegment"]
        # Break does not contribute to length
        assert spec.chain("A").total_length == 10 + 6

    def test_passthrough_within_block(self):
        spec = cs.ContigSpec.from_string("A1-10/5/B/A15-20")
        types = [type(s).__name__ for s in spec.chain("A").segments]
        assert "PassthroughSegment" in types
        # Passthrough makes the chain unresolved
        assert not spec.chain("A").is_resolved
        # but min/max_total_length still computes (passthrough contributes 0)
        assert spec.chain("A").min_total_length == 10 + 5 + 6

    def test_break_does_not_break_position_math(self):
        """Designed-to-native lookup should ignore break markers."""
        spec = cs.ContigSpec.from_string("A1-10/0/A15-20")
        chain = spec.chain("A")
        assert chain.designed_position_to_native(1) == 1
        assert chain.designed_position_to_native(10) == 10
        # Position 11 is the first native residue after the break — i.e. A15
        assert chain.designed_position_to_native(11) == 15


@pytest.mark.local_unit
class TestConstraintForm:
    def test_from_string_accepts_ranges(self):
        spec = cs.ContigSpec.from_string("A1-10/5-7/A15-20 B")
        assert not spec.is_resolved
        denovos = spec.denovo_segments("A")
        assert denovos[0].min_len == 5 and denovos[0].max_len == 7

    def test_from_resolved_string_rejects_ranges(self):
        with pytest.raises(ValueError):
            cs.ContigSpec.from_resolved_string("A1-10/5-7/A15-20 B")

    def test_resolved_round_trip_via_from_string(self):
        spec = cs.ContigSpec.from_string("A1-10/5/A15-20 B")
        assert spec.is_resolved
        denovos = spec.denovo_segments("A")
        assert denovos[0].length == 5

    def test_min_max_total_length(self):
        spec = cs.ContigSpec.from_string("A1-10/5-7/A15-20")
        chain = spec.chain("A")
        assert chain.min_total_length == 10 + 5 + 6
        assert chain.max_total_length == 10 + 7 + 6
        with pytest.raises(ValueError):
            _ = chain.total_length


@pytest.mark.local_unit
class TestImmutability:
    def test_contig_spec_is_frozen(self):
        spec = cs.ContigSpec.from_resolved_string("A1-10")
        with pytest.raises((AttributeError, Exception)):
            spec.contig_string = "X"  # type: ignore

    def test_chain_is_frozen(self):
        spec = cs.ContigSpec.from_resolved_string("A1-10")
        chain = spec.chain("A")
        with pytest.raises((AttributeError, Exception)):
            chain.chain_id = "Z"  # type: ignore

    def test_segments_tuple(self):
        spec = cs.ContigSpec.from_resolved_string("A1-10/5/A15-20")
        chain = spec.chain("A")
        assert isinstance(chain.segments, tuple)
