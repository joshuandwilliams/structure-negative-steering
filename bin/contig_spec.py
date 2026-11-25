"""
contig_spec.py
--------------
ContigSpec — RFDiffusion contig strings parsed into structured per-chain
segments.  A typical string looks like ``A1-10/5/A15-20 B``.

A "contig" is RFDiffusion's own notation for the layout of a designed
chain.  It says which stretches are copied from the input PDB and which
are generated from scratch.  This module and ``contig_utils.py`` are the
only places that notation is interpreted, so they are the single source
of truth for where the design region sits.  See ``from_string`` below
for the full grammar.

Tier 0 type — no upstream dependencies on other typed modules.

ContigSpec normally holds the *resolved* form.  Once RFDiffusion has
sampled a concrete length for each de novo segment, the contig describes
one fixed layout.  The constraint form, with length ranges like ``5-7``,
is only an input string to RFDiffusion.  It is passed through as text
and is not a runtime type.

``bin/contig_utils.py`` holds the older parser (``parse_block_segments``,
``resolve_contigs``, ``parse_design_region``).  ContigSpec absorbs its
behaviour, and contig_utils keeps working until every caller has moved
over.  Two more parsers once existed in
``bin/derive_input_design_region.py`` and
``bin/pipeline_correct_sequences.py``.  Both files have since been
deleted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ═════════════════════════════════════════════════════════════════════════
# Segment types
# ═════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class FixedSegment:
    """A fixed, anchored segment.  It is a native residue range copied
    from the input PDB.  Both ends are inclusive."""
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(
                f"FixedSegment start={self.start} > end={self.end}"
            )

    @property
    def length(self) -> int:
        """Number of native residues in this fixed segment."""
        return self.end - self.start + 1


@dataclass(frozen=True)
class BreakSegment:
    """A chain-break marker, written as a literal ``0`` in RFDiffusion's
    contig grammar.  It carries no length and no positions.  It only
    tells RFDiffusion that the residues on either side of it are not
    contiguous.  The position-math methods on ContigChain ignore breaks.
    """

    def __post_init__(self) -> None:
        return None


@dataclass(frozen=True)
class PassthroughSegment:
    """A bare chain letter whose residue range needs a reference PDB to
    resolve.  It is carried through the contig pipeline unchanged, and
    RFDiffusion expands it from the input PDB at run time.  Like
    BreakSegment, it has no length here.
    """
    chain: str

    def __post_init__(self) -> None:
        if not isinstance(self.chain, str) or not self.chain:
            raise ValueError("PassthroughSegment.chain must be non-empty")


@dataclass(frozen=True)
class DeNovoSegment:
    """A de novo segment, meaning a stretch RFDiffusion designs from
    scratch.

    It always carries a length range (min_len, max_len).  The constraint
    form ``5-7`` becomes (5, 7), and the resolved form ``5`` becomes
    (5, 5).  One representation covers both, so there is no separate
    resolved type.

    Most downstream code sees only resolved segments, because
    RFDiffusion has already sampled a concrete length.  ``length``
    returns that integer when ``is_resolved`` is True and raises
    otherwise.
    """
    min_len: int
    max_len: int

    def __post_init__(self) -> None:
        if self.min_len < 0:
            raise ValueError(f"DeNovoSegment min_len={self.min_len} must be >= 0")
        if self.max_len < self.min_len:
            raise ValueError(
                f"DeNovoSegment max_len={self.max_len} < min_len={self.min_len}"
            )

    @property
    def is_resolved(self) -> bool:
        """True only if the segment has one concrete length, min == max."""
        return self.min_len == self.max_len

    @property
    def length(self) -> int:
        """The resolved length.  Raises ValueError if the segment is
        still a range.  Read min_len and max_len for the bounds."""
        if not self.is_resolved:
            raise ValueError(
                f"DeNovoSegment is unresolved ({self.min_len}-{self.max_len}); "
                f"no single length.  Use min_len / max_len explicitly."
            )
        return self.min_len

    @classmethod
    def resolved(cls, length: int) -> "DeNovoSegment":
        """Build a fixed-length de novo segment.  Same as
        ``DeNovoSegment(length, length)``."""
        return cls(min_len=length, max_len=length)

    def to_string(self) -> str:
        """Render back to contig text.  A resolved segment gives ``5``
        and a range gives ``5-7``."""
        if self.is_resolved:
            return str(self.min_len)
        return f"{self.min_len}-{self.max_len}"


# ═════════════════════════════════════════════════════════════════════════
# ContigChain
# ═════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ContigChain:
    """The contig segments for a single chain, in order."""
    chain_id: str
    segments: Tuple  # Tuple[Union[FixedSegment, DeNovoSegment], ...]

    def __post_init__(self) -> None:
        # Coerce a list to a tuple so the dataclass stays frozen and
        # hashable.
        if not isinstance(self.segments, tuple):
            object.__setattr__(self, "segments", tuple(self.segments))

    @property
    def is_resolved(self) -> bool:
        """True only if every de novo segment in this chain has one
        concrete length, and the chain has no Passthrough segments.  A
        passthrough segment's length depends on the input PDB, so it is
        unknown here."""
        for s in self.segments:
            if isinstance(s, DeNovoSegment) and not s.is_resolved:
                return False
            if isinstance(s, PassthroughSegment):
                return False
        return True

    @property
    def _length_bearing(self) -> List:
        """The segments that have a length, meaning Fixed and DeNovo.
        Break and Passthrough segments are left out.  The position-math
        methods walk this list, not ``self.segments``."""
        return [
            s for s in self.segments
            if isinstance(s, (FixedSegment, DeNovoSegment))
        ]

    @property
    def total_length(self) -> int:
        """Total residues in the resolved chain, summing the fixed and
        de novo lengths.  Raises ValueError if any de novo segment is
        still a range.  For the constraint form use min_total_length or
        max_total_length instead."""
        if not self.is_resolved:
            raise ValueError(
                "ContigChain is unresolved; total_length is undefined.  "
                "Resolve de novo segment lengths or use min_total_length / "
                "max_total_length."
            )
        return sum(s.length for s in self._length_bearing)

    @property
    def min_total_length(self) -> int:
        """Smallest total residue count the chain can have.
        Passthrough segments contribute 0, since their length is not
        known until the input PDB is read."""
        return sum(
            s.length if isinstance(s, FixedSegment) else s.min_len
            for s in self._length_bearing
        )

    @property
    def max_total_length(self) -> int:
        """Largest total residue count the chain can have.
        Passthrough segments contribute 0."""
        return sum(
            s.length if isinstance(s, FixedSegment) else s.max_len
            for s in self._length_bearing
        )

    @property
    def fixed_segments(self) -> List[FixedSegment]:
        return [s for s in self.segments if isinstance(s, FixedSegment)]

    @property
    def denovo_segments(self) -> List[DeNovoSegment]:
        return [s for s in self.segments if isinstance(s, DeNovoSegment)]

    @property
    def has_breaks(self) -> bool:
        return any(isinstance(s, BreakSegment) for s in self.segments)

    @property
    def passthrough_segments(self) -> List[PassthroughSegment]:
        return [s for s in self.segments if isinstance(s, PassthroughSegment)]

    def designed_position_to_native(self, designed_pos: int) -> Optional[int]:
        """Map a 1-based designed-frame position to its native residue
        number.  Returns None if the position falls in a de novo
        segment, because designed residues have no native counterpart.

        The method walks the segments in order and tracks the running
        designed position.  When the cursor passes ``designed_pos``,
        that position lies in the current segment.
        """
        if designed_pos < 1:
            return None
        cursor = 0
        for seg in self._length_bearing:
            seg_start = cursor + 1
            seg_end = cursor + seg.length
            if seg_start <= designed_pos <= seg_end:
                if isinstance(seg, FixedSegment):
                    offset = designed_pos - seg_start
                    return seg.start + offset
                return None  # de novo — no native counterpart
            cursor = seg_end
        return None

    def native_position_to_designed(self, native_pos: int) -> Optional[int]:
        """Map a native residue number to its 1-based designed-frame
        position.  Returns None if no fixed segment anchors that native
        position, which happens inside or between de novo regions.
        """
        cursor = 0
        for seg in self._length_bearing:
            if isinstance(seg, FixedSegment):
                if seg.start <= native_pos <= seg.end:
                    offset = native_pos - seg.start
                    return cursor + 1 + offset
            cursor += seg.length
        return None

    def is_design_region_position(self, designed_pos: int) -> bool:
        """True only if designed_pos sits in a de novo segment on this
        chain."""
        cursor = 0
        for seg in self._length_bearing:
            seg_start = cursor + 1
            seg_end = cursor + seg.length
            if seg_start <= designed_pos <= seg_end:
                return isinstance(seg, DeNovoSegment)
            cursor = seg_end
        return False

    def design_region_positions(self) -> List[int]:
        """The 1-based designed-frame positions that fall in de novo
        segments on this chain."""
        out: List[int] = []
        cursor = 0
        for seg in self._length_bearing:
            if isinstance(seg, DeNovoSegment):
                out.extend(range(cursor + 1, cursor + seg.length + 1))
            cursor += seg.length
        return out

    def fixed_anchor_positions(self) -> List[int]:
        """The 1-based designed-frame positions that do have native
        counterparts, meaning those inside fixed segments."""
        out: List[int] = []
        cursor = 0
        for seg in self._length_bearing:
            if isinstance(seg, FixedSegment):
                out.extend(range(cursor + 1, cursor + seg.length + 1))
            cursor += seg.length
        return out


# ═════════════════════════════════════════════════════════════════════════
# ContigSpec
# ═════════════════════════════════════════════════════════════════════════


# A bare chain identifier, such as the ` B` in `A1-10/5 B`, means the
# whole chain is included with no design regions.  This is usually the
# effector chain.  A whole block that is one bare letter parses to a
# ContigChain with an empty segments tuple.  A bare letter inside a
# slash-separated block parses to a PassthroughSegment.
#
# Neither form records a residue range, because the range is not known
# until the input PDB is read, and RFDiffusion expands it at run time.
# Code that needs the real range reads the input PDB directly.

# Match a chain segment block (slash-separated) for one chain.
_SEG_PATTERN = re.compile(r"([A-Za-z]\d+(?:-\d+)?|\d+(?:-\d+)?)")


@dataclass(frozen=True)
class ContigSpec:
    """A resolved RFDiffusion contig, held as structured segments per
    chain.

    Build one with :meth:`from_resolved_string`.  Direct construction
    also works, and is used by tests and by code that generates contigs
    programmatically.
    """
    chains: Tuple  # Tuple[ContigChain, ...]
    contig_string: str

    def __post_init__(self) -> None:
        if not isinstance(self.chains, tuple):
            object.__setattr__(self, "chains", tuple(self.chains))

    # ── Construction ─────────────────────────────────────────────────

    @classmethod
    def from_string(cls, s: str) -> "ContigSpec":
        """Parse a contig string in either the resolved or the
        constraint form.

        The grammar:
        - Chain blocks are separated by spaces.
        - Inside a block, segments are separated by slashes.
        - A fixed segment is a chain letter plus a native residue range.
          ``A1-10`` means chain A residues 1 to 10.
        - A de novo segment is a bare length.  ``5`` is resolved, so
          min_len == max_len == 5.  ``5-7`` is the constraint form, and
          RFDiffusion samples a length between 5 and 7.  Both round-trip
          through DeNovoSegment(min_len, max_len).
        - A ``0`` on its own is a chain break.
        - A block that is one bare chain letter, such as ``B``, means
          the whole chain with no design regions.  It parses to a
          ContigChain with no segments, because the chain's length is
          unknown until the input PDB is read.
        """
        raw = s.strip()
        if not raw:
            raise ValueError("ContigSpec.from_string: empty input")
        blocks = raw.split()
        chains: List[ContigChain] = []
        for block in blocks:
            chain = _parse_block(block)
            chains.append(chain)
        return cls(chains=tuple(chains), contig_string=raw)

    @classmethod
    def from_resolved_string(cls, s: str) -> "ContigSpec":
        """Parse a contig string and require every de novo segment to
        be resolved.  Raises ValueError if any segment is a range.  Use
        ``from_string`` when ranges are acceptable."""
        spec = cls.from_string(s)
        if not spec.is_resolved:
            raise ValueError(
                f"ContigSpec.from_resolved_string({s!r}): contains "
                f"unresolved de novo segment(s).  Resolve lengths upstream "
                f"or call from_string instead."
            )
        return spec

    @property
    def is_resolved(self) -> bool:
        """True only if every de novo segment on every chain has one
        concrete length, so min_len == max_len."""
        return all(c.is_resolved for c in self.chains)

    # ── Per-chain accessors ──────────────────────────────────────────

    def chain(self, chain_id: str) -> ContigChain:
        for c in self.chains:
            if c.chain_id == chain_id:
                return c
        raise KeyError(f"chain {chain_id!r} not in contig (have: {self.chain_ids})")

    @property
    def chain_ids(self) -> List[str]:
        return [c.chain_id for c in self.chains]

    def chain_length(self, chain_id: str) -> int:
        return self.chain(chain_id).total_length

    def fixed_segments(self, chain_id: str) -> List[FixedSegment]:
        return self.chain(chain_id).fixed_segments

    def denovo_segments(self, chain_id: str) -> List[DeNovoSegment]:
        return self.chain(chain_id).denovo_segments

    # ── Frame conversion.  PositionSet calls into these. ─────────────

    def designed_to_native(
        self, pos: int, chain: str
    ) -> Optional[int]:
        """Map a designed-frame position to its native counterpart, or
        None if the position falls in a de novo segment.
        """
        return self.chain(chain).designed_position_to_native(pos)

    def native_to_designed(
        self, pos: int, chain: str
    ) -> Optional[int]:
        """Map a native residue number to its designed-frame position,
        or None if the native position is not anchored by any fixed
        segment.
        """
        return self.chain(chain).native_position_to_designed(pos)

    def prediction_to_designed(self, pos: int, chain: str) -> int:
        """Return the position unchanged.

        Boltz and AF3 both emit per-chain 1-based numbering.  For the
        receptor chain that matches the designed frame exactly, because
        the receptor PDB went straight into Boltz.  The method exists so
        callers do not have to special-case the prediction frame.
        """
        return pos

    # ── Design region access ─────────────────────────────────────────

    def design_region_positions(self, chain: str) -> List[int]:
        return self.chain(chain).design_region_positions()

    def fixed_anchor_positions(self, chain: str) -> List[int]:
        return self.chain(chain).fixed_anchor_positions()

    def is_design_region_position(self, pos: int, chain: str) -> bool:
        return self.chain(chain).is_design_region_position(pos)


# ═════════════════════════════════════════════════════════════════════════
# Parsing helpers
# ═════════════════════════════════════════════════════════════════════════


def _parse_block(block: str) -> ContigChain:
    """Parse one space-separated chain block, e.g. ``A1-10/5/A15-20`` or
    ``B`` (bare chain).
    """
    block = block.strip()
    if not block:
        raise ValueError("empty chain block")

    # Bare chain identifier: a single letter.
    if len(block) == 1 and block.isalpha():
        return ContigChain(chain_id=block, segments=())

    # Multi-segment block.  The chain ID comes from the first segment
    # with a chain prefix.  Every fixed segment in a block must name the
    # same chain.
    parts = block.split("/")
    segments: List = []
    chain_id: Optional[str] = None
    for raw_seg in parts:
        seg = raw_seg.strip()
        if not seg:
            continue
        if seg == "0":
            # RFDiffusion's chain-break marker.  It has no length.
            segments.append(BreakSegment())
            continue
        if seg[0].isalpha():
            chain_letter = seg[0]
            rest = seg[1:]
            if not rest:
                # A bare chain letter inside a block is a passthrough.
                if chain_id is None:
                    chain_id = chain_letter
                segments.append(PassthroughSegment(chain=chain_letter))
                continue
            if "-" in rest:
                lo_s, hi_s = rest.split("-", 1)
                try:
                    lo, hi = int(lo_s), int(hi_s)
                except ValueError:
                    raise ValueError(
                        f"fixed segment {seg!r}: residue range must be int-int"
                    )
            else:
                try:
                    lo = hi = int(rest)
                except ValueError:
                    raise ValueError(
                        f"fixed segment {seg!r}: residue ref must be an int"
                    )
            if chain_id is None:
                chain_id = chain_letter
            elif chain_letter != chain_id:
                raise ValueError(
                    f"block {block!r} mixes chains: {chain_id!r} and "
                    f"{chain_letter!r}.  Use space-separated blocks for "
                    f"different chains."
                )
            segments.append(FixedSegment(start=lo, end=hi))
        elif seg[0].isdigit():
            # De novo segment.  A bare length `5` is resolved, a range
            # `5-7` is the constraint form.  Both are stored as
            # (min_len, max_len), with min == max when resolved.
            if "-" in seg:
                lo_s, hi_s = seg.split("-", 1)
                try:
                    lo_n, hi_n = int(lo_s), int(hi_s)
                except ValueError:
                    raise ValueError(
                        f"de novo segment {seg!r}: range must be int-int"
                    )
                segments.append(DeNovoSegment(min_len=lo_n, max_len=hi_n))
            else:
                try:
                    length = int(seg)
                except ValueError:
                    raise ValueError(
                        f"de novo segment {seg!r}: length must be an integer"
                    )
                segments.append(DeNovoSegment(min_len=length, max_len=length))
        else:
            raise ValueError(
                f"segment {seg!r}: unrecognised first character"
            )

    if chain_id is None:
        # The block held only de novo segments, with nothing to anchor
        # it to a chain.  Refuse rather than guess the chain.
        raise ValueError(
            f"block {block!r} has no chain-prefixed fixed segment to "
            f"determine the chain identifier"
        )

    return ContigChain(chain_id=chain_id, segments=tuple(segments))
