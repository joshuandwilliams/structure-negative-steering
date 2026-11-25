"""
designed_sequence.py
--------------------
DesignedSequence — one MPNN-designed sequence overlaid on one
DesignedBackbone.

Tier 4 type.  It depends on DesignedBackbone, which is Tier 3, and
PositionSet, which is Tier 1.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))  # pragma: no cover - import order

from designed_backbone import DesignedBackbone  # noqa: E402
from position_set import PositionSet  # noqa: E402


@dataclass(frozen=True)
class DesignedSequence:
    """One MPNN-designed sequence for one DesignedBackbone.

    `corrected_receptor` is the full receptor amino-acid string.  It is
    native at the fixed positions and MPNN-designed at the design
    positions.  `designed_residues` and `native_residues` are the two
    slices at the design positions, so they can be compared directly.
    `mpnn_score` is MPNN's own score for the sequence.
    """
    backbone: DesignedBackbone
    sequence_id: str
    corrected_receptor: str
    designed_residues: str
    native_residues: str
    mpnn_score: Optional[float] = None
    qc_metadata: Optional[dict] = None

    def __post_init__(self) -> None:
        if not isinstance(self.backbone, DesignedBackbone):
            raise TypeError(
                f"backbone must be a DesignedBackbone, got "
                f"{type(self.backbone).__name__}"
            )
        if not isinstance(self.sequence_id, str) or not self.sequence_id:
            raise ValueError("sequence_id must be a non-empty string")
        if not isinstance(self.corrected_receptor, str):
            raise TypeError("corrected_receptor must be a string")
        if not isinstance(self.designed_residues, str):
            raise TypeError("designed_residues must be a string")
        if not isinstance(self.native_residues, str):
            raise TypeError("native_residues must be a string")
        # corrected_receptor must be at least as long as the contig's
        # receptor.  It need not match exactly, because it can carry
        # trailing native residues from outside the contig region.
        expected_min = self.backbone.receptor_length()
        if len(self.corrected_receptor) < expected_min:
            raise ValueError(
                f"corrected_receptor too short: got "
                f"{len(self.corrected_receptor)}, contig expects "
                f"at least {expected_min} residues"
            )

    # ── Delegated to backbone ────────────────────────────────────────

    def design_region(self) -> PositionSet:
        return self.backbone.design_region()

    @property
    def design_id(self) -> str:
        return self.backbone.design_id

    @property
    def receptor_chain(self) -> str:
        return self.backbone.receptor_chain

    # ── Sequence views ───────────────────────────────────────────────

    def to_fasta(self) -> str:
        """Return a FASTA record for the corrected receptor."""
        return f">{self.sequence_id}\n{self.corrected_receptor}\n"

    def mutations_vs_native(self) -> List[Tuple[int, str, str]]:
        """Return (designed_position, native_aa, designed_aa) for every
        position where the designed residue differs from the native
        one.

        It looks at the design-region slice only.  Positions outside
        the design region equal native by construction, so they can
        never differ.

        The position numbers are in the `designed` frame, which counts
        from 1 along the corrected receptor.
        """
        out: List[Tuple[int, str, str]] = []
        dr_positions = self.design_region().as_sorted_list()
        # Walk the two slices in step.  A pipe-joined input is
        # flattened to one string first, so the comparison is
        # position by position.
        designed = self.designed_residues.replace("|", "")
        native = self.native_residues.replace("|", "")
        n = min(len(designed), len(native), len(dr_positions))
        for i in range(n):
            d_aa = designed[i]
            n_aa = native[i]
            if d_aa != n_aa:
                out.append((dr_positions[i], n_aa, d_aa))
        return out

    def n_changes(self) -> int:
        """Count the positions where the designed residue differs from
        the native one."""
        return len(self.mutations_vs_native())

    def __repr__(self) -> str:
        score = (f"{self.mpnn_score:.3f}"
                 if self.mpnn_score is not None else "None")
        return (
            f"DesignedSequence(sequence_id={self.sequence_id!r}, "
            f"design_id={self.design_id!r}, mpnn_score={score})"
        )
