"""ContigSpec — RFDiffusion contig strings parsed into structured
per-chain segments.  A typical string looks like ``A1-10/5/A15-20 B``.

A "contig" is RFDiffusion's notation for a designed chain's layout,
saying which stretches are copied from the input PDB and which are
generated from scratch.  This module and ``contig_utils.py`` are the
only interpreters of that notation, so they are the single source of
truth for where the design region sits.  ``from_string`` has the
grammar.

Tier 0 type — no upstream dependencies on other typed modules.

ContigSpec normally holds the *resolved* form, one fixed layout, once
RFDiffusion has sampled each de novo length.  The constraint form, with
ranges like ``5-7``, is only an input string to RFDiffusion.  It is
passed through as text and is not a runtime type.

``contig_utils.py`` holds the older parser (``parse_block_segments``,
``resolve_contigs``, ``parse_design_region``), which ContigSpec absorbs
and which stays until every caller has moved over.  Two more parsers
lived in receptor-resurfacing-pipeline's ``derive_input_design_region.py``
and ``pipeline_correct_sequences.py``.  Neither file has ever been in this
repo, and ``bin/derive_design_region.py`` is the descendant of the first.
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
    """A native residue range copied from the input PDB, both ends
    inclusive."""
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(
                f"FixedSegment start={self.start} > end={self.end}"
            )

    @property
    def length(self) -> int:
        """Native residues in this segment."""
        return self.end - self.start + 1


@dataclass(frozen=True)
class BreakSegment:
    """A chain-break marker, a literal ``0`` in RFDiffusion's grammar.
    It has no length or positions, and only says the residues either
    side are not contiguous.  ContigChain's position math
    ignores breaks.
    """

    def __post_init__(self) -> None:
        return None


@dataclass(frozen=True)
class PassthroughSegment:
    """A bare chain letter whose residue range needs a reference PDB.
    It passes through unchanged, and RFDiffusion expands it from the
    input PDB at run time.  No length here.
    """
    chain: str

    def __post_init__(self) -> None:
        if not isinstance(self.chain, str) or not self.chain:
            raise ValueError("PassthroughSegment.chain must be non-empty")


@dataclass(frozen=True)
class DeNovoSegment:
    """A stretch RFDiffusion designs from scratch.

    It always carries a length range (min_len, max_len).  ``5-7``
    becomes (5, 7) and ``5`` becomes (5, 5), so there is no separate
    resolved type.  ``length`` returns that integer when
    ``is_resolved`` is True and raises otherwise.
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
        """The resolved length.  Raises ValueError on a range, where
        min_len and max_len give the bounds."""
        if not self.is_resolved:
            raise ValueError(
                f"DeNovoSegment is unresolved ({self.min_len}-{self.max_len}); "
                f"no single length.  Use min_len / max_len explicitly."
            )
        return self.min_len

    @classmethod
    def resolved(cls, length: int) -> "DeNovoSegment":
        """Build a fixed-length de novo segment."""
        return cls(min_len=length, max_len=length)

    def to_string(self) -> str:
        """Render back to contig text, ``5`` resolved or ``5-7`` a
        range."""
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
        # Coerce to a tuple so the dataclass stays frozen and hashable.
        if not isinstance(self.segments, tuple):
            object.__setattr__(self, "segments", tuple(self.segments))

    @property
    def is_resolved(self) -> bool:
        """True only if every de novo segment has one concrete length
        and there are no Passthrough segments, whose lengths depend on
        the input PDB."""
        for s in self.segments:
            if isinstance(s, DeNovoSegment) and not s.is_resolved:
                return False
            if isinstance(s, PassthroughSegment):
                return False
        return True

    @property
    def _length_bearing(self) -> List:
        """Fixed and DeNovo segments only.  The position-math methods
        walk this list, not ``self.segments``."""
        return [
            s for s in self.segments
            if isinstance(s, (FixedSegment, DeNovoSegment))
        ]

    @property
    def total_length(self) -> int:
        """Total residues in the resolved chain.  Raises ValueError on
        an unresolved de novo segment, where min_total_length and
        max_total_length apply."""
        if not self.is_resolved:
            raise ValueError(
                "ContigChain is unresolved; total_length is undefined.  "
                "Resolve de novo segment lengths or use min_total_length / "
                "max_total_length."
            )
        return sum(s.length for s in self._length_bearing)

    @property
    def min_total_length(self) -> int:
        """Smallest total residue count.  Passthrough segments
        contribute 0, their length unknown until the PDB is read."""
        return sum(
            s.length if isinstance(s, FixedSegment) else s.min_len
            for s in self._length_bearing
        )

    @property
    def max_total_length(self) -> int:
        """Largest total residue count.  Passthrough segments
        contribute 0."""
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
        number.  Returns None inside a de novo segment, where designed
        residues have no native counterpart.
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
        position.  Returns None when no fixed segment anchors it, which
        happens in and between de novo regions.
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
        """True if designed_pos sits in a de novo segment."""
        cursor = 0
        for seg in self._length_bearing:
            seg_start = cursor + 1
            seg_end = cursor + seg.length
            if seg_start <= designed_pos <= seg_end:
                return isinstance(seg, DeNovoSegment)
            cursor = seg_end
        return False

    def design_region_positions(self) -> List[int]:
        """The 1-based designed-frame positions in de novo segments."""
        out: List[int] = []
        cursor = 0
        for seg in self._length_bearing:
            if isinstance(seg, DeNovoSegment):
                out.extend(range(cursor + 1, cursor + seg.length + 1))
            cursor += seg.length
        return out

    def fixed_anchor_positions(self) -> List[int]:
        """The 1-based designed-frame positions inside fixed segments,
        the ones with native counterparts."""
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
# whole chain with no design regions, usually the effector.  A whole
# bare-letter block gives a ContigChain with an empty segments tuple,
# while a bare letter inside a slash-separated block gives a
# PassthroughSegment.  Neither records a residue range, so code needing
# the real range reads the input PDB directly.

# Matches one slash-separated segment.
_SEG_PATTERN = re.compile(r"([A-Za-z]\d+(?:-\d+)?|\d+(?:-\d+)?)")


@dataclass(frozen=True)
class ContigSpec:
    """A resolved RFDiffusion contig, held as structured segments per
    chain.

    Build one with :meth:`from_resolved_string`.  Direct construction
    also works, as tests and programmatic generators do.
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

        Chain blocks are space-separated, segments inside a block
        slash-separated.  A fixed segment is a chain letter plus a
        native residue range, so ``A1-10`` is chain A residues 1 to 10.
        A de novo segment is a bare length, ``5`` resolved or ``5-7`` a
        constraint.  A ``0`` is a chain break.  A block of one bare
        chain letter is the whole chain with no design regions, and
        gives a ContigChain with no segments.
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
        """Parse a contig string, requiring every de novo segment to
        be resolved.  ``from_string`` allows ranges."""
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
        """True only if every de novo segment on every chain has
        min_len == max_len."""
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
        """Delegate to the chain's designed_position_to_native."""
        return self.chain(chain).designed_position_to_native(pos)

    def native_to_designed(
        self, pos: int, chain: str
    ) -> Optional[int]:
        """Delegate to the chain's native_position_to_designed."""
        return self.chain(chain).native_position_to_designed(pos)

    def prediction_to_designed(self, pos: int, chain: str) -> int:
        """Return the position unchanged.

        Boltz and AF3 emit per-chain 1-based numbering, which matches
        the designed frame for the receptor chain, since the receptor
        PDB went straight into Boltz.  It exists so callers need not
        special-case the prediction frame.
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

    # The chain ID comes from the first chain-prefixed segment, and
    # every fixed segment in a block must name the same chain.
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
        # Nothing anchors the block to a chain, so refuse rather than
        # guess.
        raise ValueError(
            f"block {block!r} has no chain-prefixed fixed segment to "
            f"determine the chain identifier"
        )

    return ContigChain(chain_id=chain_id, segments=tuple(segments))
