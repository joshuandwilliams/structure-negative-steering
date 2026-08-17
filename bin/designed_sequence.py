"""
designed_sequence.py
--------------------
DesignedSequence — one MPNN-designed sequence overlaid on one
DesignedBackbone.

Per Phase 4 spec §2.7.  Tier 4 type — depends on DesignedBackbone
(Tier 3) and PositionSet (Tier 1).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from designed_backbone import DesignedBackbone  # noqa: E402
from position_set import PositionSet  # noqa: E402


@dataclass(frozen=True)
class DesignedSequence:
    """One MPNN-designed sequence for one DesignedBackbone.

    Carries the corrected receptor amino-acid string (native at fixed
    positions, MPNN-designed at design positions), the MPNN-only and
    native-only slices at the design positions, and the MPNN score.
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
        # Length sanity check: corrected_receptor should be at least
        # as long as the contig's receptor length.  Doesn't have to
        # match exactly because the corrected receptor may include
        # trailing native residues outside the contig region.
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
        """Return a list of (designed_position, native_aa, designed_aa)
        for positions where the designed residue differs from the
        native residue.

        Operates on the design-region slice only; positions outside
        the design region are by construction equal to native.
        Position numbers are in the `designed` frame (1-based along
        the corrected receptor) of the de novo segment(s).
        """
        out: List[Tuple[int, str, str]] = []
        dr_positions = self.design_region().as_sorted_list()
        # Walk the parallel slices.  Pipe-joined inputs are flattened
        # to single strings for the per-position comparison.
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
        """Count of positions where designed differs from native."""
        return len(self.mutations_vs_native())

    def __repr__(self) -> str:
        score = (f"{self.mpnn_score:.3f}"
                 if self.mpnn_score is not None else "None")
        return (
            f"DesignedSequence(sequence_id={self.sequence_id!r}, "
            f"design_id={self.design_id!r}, mpnn_score={score})"
        )
