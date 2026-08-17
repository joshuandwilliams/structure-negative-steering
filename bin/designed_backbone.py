"""
designed_backbone.py
--------------------
DesignedBackbone — one RFDiffusion output.

Per Phase 4 spec §2.6.  Tier 3 type — depends on
ProteinStructurePrediction, ContigSpec, PositionSet.

Composition over inheritance (per Q8): wraps a
``ProteinStructurePrediction`` for the structural file rather than
extending it.  Accessing the structure goes through
``backbone.structure.X``.  DesignedBackbone adds the RFDiffusion-
specific fields (design_id, contig, Rosetta filter score) and the
design-region accessor that delegates to the contig.

A DesignedBackbone has STRUCTURE but no SEQUENCE — the residue
identities for the de novo region come later from MPNN.  Adding a
sequence produces a `DesignedSequence` (Tier 4).
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

    Carries the structural file via a composed ProteinStructurePrediction
    (per Q8 — composition, not inheritance), plus the contig that
    produced it and the Rosetta filter score that gated it forward.
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
        """Receptor positions that fall in de novo (design) segments,
        in `designed` frame.  Carries the contig reference so callers
        can convert with `set.in_frame('native')`."""
        chain = self.receptor_chain
        positions = self.contig.design_region_positions(chain)
        return PositionSet(
            positions=positions, chain=chain, frame="designed",
            contig=self.contig,
        )

    def fixed_anchor_positions(self) -> PositionSet:
        """Receptor positions anchored to native (fixed segments),
        in `designed` frame."""
        chain = self.receptor_chain
        positions = self.contig.fixed_anchor_positions(chain)
        return PositionSet(
            positions=positions, chain=chain, frame="designed",
            contig=self.contig,
        )

    def receptor_length(self) -> int:
        """Total residues in the resolved receptor chain (= fixed +
        de novo segment lengths summed)."""
        return self.contig.chain_length(self.receptor_chain)

    # ── Pass-through to the composed structure ───────────────────────

    @property
    def receptor_chain(self) -> str:
        return self.structure.receptor_chain

    @property
    def effector_chain(self) -> str:
        return self.structure.effector_chain

    # ── Factory: parse contig + load PSP in one shot ─────────────────

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
        """Convenience: takes a contig STRING + paths, returns a
        DesignedBackbone with the contig parsed and the PSP constructed.
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
