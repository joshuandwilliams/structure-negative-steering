"""DesignedBackbone — one RFDiffusion output.

Tier 3 type, depending on ProteinStructurePrediction, ContigSpec and
PositionSet.

It wraps a ``ProteinStructurePrediction`` rather than extending it,
since a backbone is not a kind of prediction, so the structure is
reached through ``backbone.structure.X``.  It adds the
RFDiffusion-only fields design_id, contig and Rosetta filter score,
plus the design-region accessor the contig answers.

A DesignedBackbone has a structure but no sequence.  The de novo
region's residue identities arrive later from MPNN, and adding a
sequence gives a Tier 4 `DesignedSequence`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from contig_spec import ContigSpec  # noqa: E402
from position_set import PositionSet  # noqa: E402
from protein_structure_prediction import ProteinStructurePrediction  # noqa: E402


@dataclass(frozen=True)
class DesignedBackbone:
    """One RFDiffusion-generated receptor backbone.

    The structure file sits in a composed ProteinStructurePrediction.
    """
    pdb_path: Path
    design_id: str
    contig: ContigSpec
    structure: ProteinStructurePrediction
    sc_score: Optional[float] = None
    rfdiff_metrics: Optional[dict] = None

    def __post_init__(self) -> None:
        if not isinstance(self.design_id, str) or not self.design_id:
            raise ValueError("design_id must be a non-empty string")
        if not isinstance(self.contig, ContigSpec):
            raise TypeError(
                f"contig must be a ContigSpec, got "
                f"{type(self.contig).__name__}"
            )
        if not isinstance(self.structure, ProteinStructurePrediction):
            raise TypeError(
                f"structure must be a ProteinStructurePrediction, got "
                f"{type(self.structure).__name__}"
            )

    # ── Convenience delegating to ContigSpec ─────────────────────────

    def design_region(self) -> PositionSet:
        """Receptor positions inside de novo segments, in the
        `designed` frame.  The result carries the contig, so a caller
        can convert it with `set.in_frame('native')`."""
        chain = self.receptor_chain
        positions = self.contig.design_region_positions(chain)
        return PositionSet(
            positions=positions, chain=chain, frame="designed",
            contig=self.contig,
        )

    def fixed_anchor_positions(self) -> PositionSet:
        """Receptor positions in fixed segments, so anchored to native
        residues, in the `designed` frame."""
        chain = self.receptor_chain
        positions = self.contig.fixed_anchor_positions(chain)
        return PositionSet(
            positions=positions, chain=chain, frame="designed",
            contig=self.contig,
        )

    def receptor_length(self) -> int:
        """Total residues in the resolved receptor chain, fixed plus de
        novo segment lengths."""
        return self.contig.chain_length(self.receptor_chain)

    # ── Pass-through to the composed structure ───────────────────────

    @property
    def receptor_chain(self) -> str:
        return self.structure.receptor_chain

    @property
    def effector_chain(self) -> str:
        return self.structure.effector_chain

    # ── Factory ─────────────────────────────────────────────────────

    @classmethod
    def from_pdb_and_contig(
        cls,
        pdb_path: Path,
        design_id: str,
        contig_string: str,
        receptor_chain: str = "A",
        effector_chain: str = "B",
        sc_score: Optional[float] = None,
        rfdiff_metrics: Optional[dict] = None,
    ) -> "DesignedBackbone":
        """Build a DesignedBackbone from a contig string and the file
        paths, parsing the contig and constructing the
        ProteinStructurePrediction.
        """
        contig = ContigSpec.from_resolved_string(contig_string)
        structure = ProteinStructurePrediction(
            path=pdb_path,
            receptor_chain=receptor_chain,
            effector_chain=effector_chain,
        )
        return cls(
            pdb_path=Path(pdb_path),
            design_id=design_id,
            contig=contig,
            structure=structure,
            sc_score=sc_score,
            rfdiff_metrics=rfdiff_metrics,
        )

    def __repr__(self) -> str:
        sc = f"{self.sc_score:.3f}" if self.sc_score is not None else "None"
        return (
            f"DesignedBackbone(design_id={self.design_id!r}, "
            f"sc_score={sc}, receptor_length={self.receptor_length()})"
        )
