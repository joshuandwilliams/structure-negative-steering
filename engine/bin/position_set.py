"""
position_set.py
---------------
PositionSet — an immutable set of receptor residue positions, tagged
with the numbering frame the integers refer to.  Per Phase 4 spec §2.3.

Three frames in use across the codebase:
- ``native``     : input PDB's residue numbers (may have gaps).
- ``designed``   : RFDiffusion's 1-based contiguous renumbering.
- ``prediction`` : Boltz / AF3's 1-based per-chain numbering.  For
                    the receptor chain this matches the designed frame
                    (the receptor PDB went straight into Boltz).

Cross-frame operations raise.  Conversion is explicit via
``in_frame(target)`` and requires a `ContigSpec` reference (carried
optionally on the PositionSet at construction).

Tier 1 type — depends on ContigSpec (Tier 0).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import FrozenSet, Iterable, Iterator, List, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from contig_spec import ContigSpec  # noqa: E402


# Frames allowed in the codebase today.  Locked-down list rather than a
# free-form string so typos at construction surface as a clear error
# instead of silently disabling cross-frame checks downstream.
_VALID_FRAMES = frozenset(("native", "designed", "prediction"))


class FrameMismatchError(ValueError):
    """Raised when set operations are attempted across different frames
    (or different chains)."""


class FrameConversionError(ValueError):
    """Raised when in_frame() is asked to convert without the necessary
    ContigSpec, or (with strict=True) when some positions cannot be
    mapped to the target frame."""


class PositionSet:
    """An immutable set of receptor positions with frame + chain tags."""

    __slots__ = ("_positions", "_chain", "_frame", "_contig")

    def __init__(
        self,
        positions: Iterable[int],
        chain: str,
        frame: str,
        contig: Optional[ContigSpec] = None,
    ):
        if frame not in _VALID_FRAMES:
            raise ValueError(
                f"frame {frame!r} not in {sorted(_VALID_FRAMES)}"
            )
        if not isinstance(chain, str) or not chain:
            raise ValueError(f"chain must be a non-empty string, got {chain!r}")
        # Deduplicate and normalise to a frozen, sorted tuple.
        self._positions: FrozenSet[int] = frozenset(int(p) for p in positions)
        self._chain: str = chain
        self._frame: str = frame
        self._contig: Optional[ContigSpec] = contig

    # ── Properties ───────────────────────────────────────────────────

    @property
    def chain(self) -> str:
        return self._chain

    @property
    def frame(self) -> str:
        return self._frame

    @property
    def contig(self) -> Optional[ContigSpec]:
        return self._contig

    # ── Set operations (same-frame, same-chain only) ─────────────────

    def union(self, other: "PositionSet") -> "PositionSet":
        self._check_compatible(other)
        return PositionSet(
            self._positions | other._positions,
            chain=self._chain,
            frame=self._frame,
            contig=self._contig or other._contig,
        )

    def intersection(self, other: "PositionSet") -> "PositionSet":
        self._check_compatible(other)
        return PositionSet(
            self._positions & other._positions,
            chain=self._chain,
            frame=self._frame,
            contig=self._contig or other._contig,
        )

    def difference(self, other: "PositionSet") -> "PositionSet":
        self._check_compatible(other)
        return PositionSet(
            self._positions - other._positions,
            chain=self._chain,
            frame=self._frame,
            contig=self._contig or other._contig,
        )

    def __or__(self, other: "PositionSet") -> "PositionSet":
        return self.union(other)

    def __and__(self, other: "PositionSet") -> "PositionSet":
        return self.intersection(other)

    def __sub__(self, other: "PositionSet") -> "PositionSet":
        return self.difference(other)

    def __contains__(self, pos: int) -> bool:
        return int(pos) in self._positions

    # ── Frame conversion ─────────────────────────────────────────────

    def in_frame(
        self, target_frame: str, strict: bool = False
    ) -> "PositionSet":
        """Return a new PositionSet in the target frame.

        Requires ``self.contig is not None`` (otherwise raises
        FrameConversionError — there's no source of truth for how to
        translate without the contig).

        Positions with no counterpart in the target frame (e.g. de novo
        positions translating to native) are silently dropped by
        default; pass ``strict=True`` to raise instead.
        """
        if target_frame not in _VALID_FRAMES:
            raise ValueError(
                f"target_frame {target_frame!r} not in {sorted(_VALID_FRAMES)}"
            )
        if target_frame == self._frame:
            return self
        if self._contig is None:
            raise FrameConversionError(
                f"cannot convert PositionSet from {self._frame!r} to "
                f"{target_frame!r}: no ContigSpec attached.  Pass "
                f"contig=... at construction or use ContigSpec methods "
                f"directly."
            )

        mapping = self._select_mapper(target_frame)
        translated: List[int] = []
        dropped: List[int] = []
        for pos in sorted(self._positions):
            mapped = mapping(pos, self._chain)
            if mapped is None:
                dropped.append(pos)
            else:
                translated.append(mapped)

        if dropped and strict:
            raise FrameConversionError(
                f"in_frame(strict=True): {len(dropped)} position(s) "
                f"have no counterpart in frame {target_frame!r}: "
                f"{dropped[:10]}{'...' if len(dropped) > 10 else ''}"
            )
        return PositionSet(
            translated, chain=self._chain, frame=target_frame,
            contig=self._contig,
        )

    def _select_mapper(self, target_frame: str):
        """Return the per-position mapper function for the requested
        conversion direction."""
        src, dst = self._frame, target_frame
        if src == "designed" and dst == "native":
            return self._contig.designed_to_native
        if src == "native" and dst == "designed":
            return self._contig.native_to_designed
        if src == "prediction" and dst == "designed":
            # Identity for the receptor chain; documented as no-op
            return lambda p, c: self._contig.prediction_to_designed(p, c)
        if src == "designed" and dst == "prediction":
            return lambda p, c: p
        if src == "prediction" and dst == "native":
            return lambda p, c: self._contig.designed_to_native(p, c)
        if src == "native" and dst == "prediction":
            return lambda p, c: self._contig.native_to_designed(p, c)
        raise FrameConversionError(
            f"no conversion mapper for {src!r} -> {dst!r}"
        )

    # ── Predicates and access ────────────────────────────────────────

    def is_empty(self) -> bool:
        return len(self._positions) == 0

    def __len__(self) -> int:
        return len(self._positions)

    def __iter__(self) -> Iterator[int]:
        yield from sorted(self._positions)

    def as_sorted_list(self) -> List[int]:
        return sorted(self._positions)

    def __eq__(self, other: object) -> bool:
        """Equal iff same positions, chain, and frame.  ContigSpec
        reference is informational and does NOT affect equality."""
        if not isinstance(other, PositionSet):
            return NotImplemented
        return (
            self._positions == other._positions
            and self._chain == other._chain
            and self._frame == other._frame
        )

    def __hash__(self) -> int:
        return hash((self._positions, self._chain, self._frame))

    def __repr__(self) -> str:
        n = len(self._positions)
        preview = sorted(self._positions)[:5]
        if n > 5:
            preview_str = f"{preview} ... ({n} total)"
        else:
            preview_str = str(preview)
        return (
            f"PositionSet(chain={self._chain!r}, frame={self._frame!r}, "
            f"positions={preview_str})"
        )

    # ── I/O ──────────────────────────────────────────────────────────

    def to_chimerax_string(self) -> str:
        """Render as the ``/A:5,7,22`` ChimeraX selection format used
        widely across the codebase.  Empty set → empty string."""
        if not self._positions:
            return ""
        joined = ",".join(str(p) for p in sorted(self._positions))
        return f"/{self._chain}:{joined}"

    @classmethod
    def from_chimerax_string(
        cls,
        s: str,
        frame: str,
        contig: Optional[ContigSpec] = None,
    ) -> "PositionSet":
        """Parse a ChimeraX selection string.  Accepts empty string
        (returns an empty PositionSet — chain defaults to 'A' in that
        case; if you need a different chain for an empty set, construct
        directly)."""
        s = s.strip()
        if not s:
            return cls(positions=[], chain="A", frame=frame, contig=contig)
        if not s.startswith("/") or ":" not in s:
            raise ValueError(
                f"chimerax string {s!r} must be '/CHAIN:pos1,pos2,...'"
            )
        chain_part, pos_part = s[1:].split(":", 1)
        if not chain_part:
            raise ValueError(f"chimerax string {s!r} missing chain")
        positions: List[int] = []
        for token in pos_part.split(","):
            tok = token.strip()
            if not tok:
                continue
            try:
                positions.append(int(tok))
            except ValueError:
                raise ValueError(
                    f"chimerax string {s!r}: non-integer position {tok!r}"
                )
        return cls(
            positions=positions, chain=chain_part, frame=frame, contig=contig
        )

    def to_text_file(self, path: Path) -> None:
        """Write integers, one per line, sorted.  Matches the format of
        the existing ``*_design_region.txt`` / ``*_true_interface.txt``
        files.  Chain and frame are NOT serialised — callers must
        provide them at read time."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for p in sorted(self._positions):
                f.write(f"{p}\n")

    @classmethod
    def from_text_file(
        cls,
        path: Path,
        chain: str,
        frame: str,
        contig: Optional[ContigSpec] = None,
    ) -> "PositionSet":
        """Read integers one per line.  Blank lines and lines starting
        with ``#`` are ignored."""
        path = Path(path)
        positions: List[int] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    positions.append(int(line))
                except ValueError:
                    raise ValueError(
                        f"{path}: non-integer position {line!r}"
                    )
        return cls(
            positions=positions, chain=chain, frame=frame, contig=contig
        )

    # ── Compatibility check ──────────────────────────────────────────

    def _check_compatible(self, other: "PositionSet") -> None:
        if not isinstance(other, PositionSet):
            raise TypeError(
                f"PositionSet operation expects another PositionSet, "
                f"got {type(other).__name__}"
            )
        if self._chain != other._chain:
            raise FrameMismatchError(
                f"PositionSet chain mismatch: {self._chain!r} vs "
                f"{other._chain!r}.  Convert chains explicitly before "
                f"combining."
            )
        if self._frame != other._frame:
            raise FrameMismatchError(
                f"PositionSet frame mismatch: {self._frame!r} vs "
                f"{other._frame!r}.  Use in_frame(target) to convert "
                f"before combining."
            )
