"""
protein_structure_prediction.py
-------------------------------
ProteinStructurePrediction — one prediction's structural payload.

Wraps a Boltz PDB or an AF3 mmCIF together with its chain composition,
and exposes everything derivable from the atom coordinates: per-chain
sequences, contact residues between chains, RMSDs against another
prediction, jaccard overlap of interfaces.

Per Phase 4 spec §2.1.  Tier 2 type — depends on PositionSet (which
depends on ContigSpec).  No predictor-specific knowledge (confidence
metrics live on `BoltzConfidenceMetrics` / `AF3ConfidenceAggregate`).

This type consolidates many existing implementations:

| What                    | Existing copies                              |
|-------------------------|----------------------------------------------|
| chain sequence          | boltz2_negative_steering, build_control_     |
|                         | sequences, extract_survivor_manifest,        |
|                         | pipeline_correct_sequences                   |
| Cα reading              | boltz2_negative_steering, rfdiffusion_filter,|
|                         | parse_af3_output, reversion                  |
| Heavy-atom contacts     | boltz2_negative_steering, compute_metrics    |
| Jaccard / weighted      | boltz2_negative_steering, compute_metrics,   |
|                         | compute_interface_metrics                    |
| Receptor-aligned effector RMSD | parse_af3_output, binding_rmsds       |
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Local import — PositionSet (Tier 1).
import sys
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from position_set import PositionSet  # noqa: E402


# Three-letter → one-letter AA map.  Single canonical home in
# contig_utils.py (post-Phase-3); re-exported here for the
# tests/characterization tests that reach in for it.  All other
# callers should import from contig_utils directly.
from contig_utils import THREE_TO_ONE  # noqa: E402, F401


class ProteinStructurePrediction:
    """One prediction's structural payload.

    Construct with the path + chain roles (receptor / effector).
    Validates the path at construction; structure data is loaded
    lazily on first access and cached for the instance's lifetime.
    """

    def __init__(
        self,
        path: Path,
        receptor_chain: str,
        effector_chain: str,
        predictor: Optional[str] = None,
    ):
        self._path: Path = Path(path)
        if not self._path.is_file():
            raise FileNotFoundError(f"structure file not found: {self._path}")
        if not isinstance(receptor_chain, str) or not receptor_chain:
            raise ValueError(f"receptor_chain must be non-empty string")
        if not isinstance(effector_chain, str) or not effector_chain:
            raise ValueError(f"effector_chain must be non-empty string")
        if receptor_chain == effector_chain:
            raise ValueError(
                f"receptor_chain and effector_chain cannot both be "
                f"{receptor_chain!r}"
            )
        self._receptor_chain = receptor_chain
        self._effector_chain = effector_chain
        # Optional predictor tag — diagnostic only, no behaviour depends on it.
        if predictor is not None and predictor not in ("boltz", "af3"):
            raise ValueError(
                f"predictor must be 'boltz' | 'af3' | None, got {predictor!r}"
            )
        self._predictor = predictor

        # Lazy caches.
        self._structure_cache = None
        self._chain_seqs: Dict[str, str] = {}
        self._chain_ca_coords: Dict[str, np.ndarray] = {}
        self._chain_heavy_atoms: Dict[str, List[Tuple[int, np.ndarray]]] = {}

    # ── Identity ────────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        return self._path

    @property
    def receptor_chain(self) -> str:
        return self._receptor_chain

    @property
    def effector_chain(self) -> str:
        return self._effector_chain

    @property
    def predictor(self) -> Optional[str]:
        return self._predictor

    def __repr__(self) -> str:
        return (
            f"ProteinStructurePrediction(path={self._path.name!r}, "
            f"receptor={self._receptor_chain!r}, "
            f"effector={self._effector_chain!r})"
        )

    # ── Lazy structure load ─────────────────────────────────────────

    def _structure(self):
        """Return the loaded gemmi structure, parsing on first call."""
        if self._structure_cache is None:
            try:
                import gemmi
            except ImportError as e:
                raise ImportError(
                    f"gemmi is required to read structure files "
                    f"({self._path}): {e}"
                ) from e
            try:
                self._structure_cache = gemmi.read_structure(str(self._path))
            except Exception as e:  # noqa: BLE001
                raise OSError(
                    f"failed to read structure {self._path}: "
                    f"{type(e).__name__}: {e}"
                ) from e
        return self._structure_cache

    # ── Sequence access ─────────────────────────────────────────────

    def chain_sequence(self, chain: str) -> str:
        """Return the 1-letter amino-acid sequence of the named chain.

        Reads residues in their structural order from the file.  Unknown
        residues map to 'X' per convention.  Returns empty string if
        the chain has no atoms in the structure.
        """
        if chain in self._chain_seqs:
            return self._chain_seqs[chain]
        s = self._structure()
        seq_chars: List[str] = []
        seen_resnums: set = set()
        for model in s:
            for ch in model:
                if ch.name != chain:
                    continue
                for res in ch:
                    # Some residues appear multiple times (alternate
                    # conformations).  De-dup by (resnum, segid).
                    key = (res.seqid.num, getattr(res.seqid, "icode", ""))
                    if key in seen_resnums:
                        continue
                    seen_resnums.add(key)
                    seq_chars.append(THREE_TO_ONE.get(res.name, "X"))
        seq = "".join(seq_chars)
        self._chain_seqs[chain] = seq
        return seq

    def receptor_sequence(self) -> str:
        return self.chain_sequence(self._receptor_chain)

    def effector_sequence(self) -> str:
        return self.chain_sequence(self._effector_chain)

    def receptor_length(self) -> int:
        return len(self.receptor_sequence())

    def effector_length(self) -> int:
        return len(self.effector_sequence())

    # ── Coordinate access ──────────────────────────────────────────

    def chain_ca_coords(self, chain: str) -> np.ndarray:
        """Return ordered Cα coordinates (shape `(N, 3)`) for the chain.

        Ordered as they appear in the file — NOT keyed by residue
        number.  See parse_af3_output._chain_ca_coords for the
        position-pairing rationale (AF3 uses per-chain 1-based;
        Boltz/RFDiffusion use continuous numbering).  Returns an
        empty `(0, 3)` array if the chain has no atoms.
        """
        if chain in self._chain_ca_coords:
            return self._chain_ca_coords[chain]
        coords: List[List[float]] = []
        s = self._structure()
        for model in s:
            for ch in model:
                if ch.name != chain:
                    continue
                for res in ch:
                    for atom in res:
                        if atom.name == "CA":
                            coords.append(
                                [atom.pos.x, atom.pos.y, atom.pos.z]
                            )
                            break
        if not coords:
            arr = np.zeros((0, 3), dtype=float)
        else:
            arr = np.array(coords, dtype=float)
        self._chain_ca_coords[chain] = arr
        return arr

    def receptor_ca_coords(self) -> np.ndarray:
        return self.chain_ca_coords(self._receptor_chain)

    def effector_ca_coords(self) -> np.ndarray:
        return self.chain_ca_coords(self._effector_chain)

    def _chain_heavy_atoms_list(
        self, chain: str
    ) -> List[Tuple[int, np.ndarray]]:
        """Return list of (residue_number, xyz) for every heavy
        (non-H) atom in the chain.  Cached."""
        if chain in self._chain_heavy_atoms:
            return self._chain_heavy_atoms[chain]
        out: List[Tuple[int, np.ndarray]] = []
        s = self._structure()
        for model in s:
            for ch in model:
                if ch.name != chain:
                    continue
                for res in ch:
                    rn = res.seqid.num
                    for atom in res:
                        if atom.name.startswith("H"):
                            continue
                        out.append(
                            (rn, np.array(
                                [atom.pos.x, atom.pos.y, atom.pos.z],
                                dtype=float,
                            ))
                        )
        self._chain_heavy_atoms[chain] = out
        return out

    # ── Contact analysis ───────────────────────────────────────────

    def receptor_contact_residues(self, cutoff: float) -> PositionSet:
        """Receptor residues whose heavy atoms come within `cutoff` Å
        of any effector heavy atom.

        Returns a PositionSet in the `prediction` frame (per-chain
        1-based numbering as emitted by the predictor — matches the
        residue numbers in the file).
        """
        rec = self._chain_heavy_atoms_list(self._receptor_chain)
        eff = self._chain_heavy_atoms_list(self._effector_chain)
        if not rec or not eff:
            return PositionSet(
                positions=[], chain=self._receptor_chain, frame="prediction"
            )
        # Numpy-based pairwise distance check.  At negsteer scale
        # (~100 residues × ~50 atoms per chain) this is O(N*M) but
        # tiny in absolute terms.
        rec_coords = np.array([xyz for _, xyz in rec])
        eff_coords = np.array([xyz for _, xyz in eff])
        # |rec_coords[i] - eff_coords[j]|^2 < cutoff^2
        diff = rec_coords[:, None, :] - eff_coords[None, :, :]
        dist_sq = (diff * diff).sum(axis=-1)
        in_contact = (dist_sq < cutoff * cutoff).any(axis=1)
        contact_residues = sorted({
            rec[i][0] for i in range(len(rec)) if in_contact[i]
        })
        return PositionSet(
            positions=contact_residues,
            chain=self._receptor_chain,
            frame="prediction",
        )

    # ── Pair operations (asymmetric — `self` vs `other`) ───────────

    def ra_eff_vs(self, other: "ProteinStructurePrediction") -> Optional[float]:
        """Receptor-aligned effector Cα RMSD: align `self`'s receptor
        onto `other`'s receptor, then measure effector atom displacement.

        Returns None if either chain has fewer than 3 residues to
        align (matches parse_af3_output behaviour).
        """
        return _ra_eff_rmsd(
            self.receptor_ca_coords(),
            self.effector_ca_coords(),
            other.receptor_ca_coords(),
            other.effector_ca_coords(),
        )

    def independent_receptor_rmsd_vs(
        self, other: "ProteinStructurePrediction"
    ) -> Optional[float]:
        """Receptor-only Cα RMSD after Kabsch alignment."""
        return _kabsch_rmsd(
            self.receptor_ca_coords(), other.receptor_ca_coords()
        )

    def independent_effector_rmsd_vs(
        self, other: "ProteinStructurePrediction"
    ) -> Optional[float]:
        """Effector-only Cα RMSD after Kabsch alignment."""
        return _kabsch_rmsd(
            self.effector_ca_coords(), other.effector_ca_coords()
        )

    def jaccard_interface_vs(
        self,
        other: "ProteinStructurePrediction",
        cutoff: float,
    ) -> float:
        """Jaccard overlap between this prediction's receptor contact
        residues and `other`'s, at the given heavy-atom cutoff."""
        s_set = set(self.receptor_contact_residues(cutoff).as_sorted_list())
        o_set = set(other.receptor_contact_residues(cutoff).as_sorted_list())
        if not s_set and not o_set:
            return 0.0
        return len(s_set & o_set) / len(s_set | o_set)


# ═════════════════════════════════════════════════════════════════════════
# Private geometry helpers
# ═════════════════════════════════════════════════════════════════════════


def _kabsch_rmsd(
    P: np.ndarray, Q: np.ndarray
) -> Optional[float]:
    """Kabsch-align P onto Q and return the resulting Cα RMSD.
    Returns None if either array has fewer than 3 points."""
    n = min(len(P), len(Q))
    if n < 3:
        return None
    P, Q = P[:n], Q[:n]
    P_c = P - P.mean(axis=0)
    Q_c = Q - Q.mean(axis=0)
    H = P_c.T @ Q_c
    try:
        U, _, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return None
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    P_aligned = P_c @ R.T
    diff = P_aligned - Q_c
    return float(np.sqrt((diff * diff).sum(axis=1).mean()))


def _ra_eff_rmsd(
    pred_rec: np.ndarray,
    pred_eff: np.ndarray,
    ref_rec: np.ndarray,
    ref_eff: np.ndarray,
) -> Optional[float]:
    """Receptor-aligned effector RMSD.  Aligns pred_rec onto ref_rec,
    applies the same transform to pred_eff, measures pred_eff vs
    ref_eff.

    Position-paired up to the shorter of the two on each chain (same
    convention as parse_af3_output)."""
    rec_n = min(len(pred_rec), len(ref_rec))
    eff_n = min(len(pred_eff), len(ref_eff))
    if rec_n < 3 or eff_n < 3:
        return None
    P = pred_rec[:rec_n]
    Q = ref_rec[:rec_n]
    P_c, P_m = P - P.mean(axis=0), P.mean(axis=0)
    Q_c, Q_m = Q - Q.mean(axis=0), Q.mean(axis=0)
    H = P_c.T @ Q_c
    try:
        U, _, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return None
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    P_eff = pred_eff[:eff_n]
    Q_eff = ref_eff[:eff_n]
    P_eff_aligned = (P_eff - P_m) @ R.T + Q_m
    diff = P_eff_aligned - Q_eff
    return float(np.sqrt((diff * diff).sum(axis=1).mean()))
