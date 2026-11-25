"""ProteinStructurePrediction — one prediction's structural payload.

Wraps a Boltz PDB or an AF3 mmCIF with its chain composition.  What
the atom coordinates give: per-chain sequences, inter-chain contact
residues, RMSDs against another prediction, jaccard interface overlap.

Tier 2 type, depending on PositionSet and through it ContigSpec.  It
holds no predictor-specific knowledge, since confidence metrics live on
`BoltzConfidenceMetrics` and `AF3ConfidenceAggregate`.

It replaces copies of the same geometry code, of whose earlier homes
only boltz2_negative_steering.py, compute_metrics.py and reversion.py
remain in the repo.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Re-exported from contig_utils.py only for the characterization tests
# that reach in for it.  Other callers import it from contig_utils.
from contig_utils import THREE_TO_ONE  # noqa: E402, F401
from position_set import PositionSet  # noqa: E402


class ProteinStructurePrediction:
    """One prediction's structural payload.

    Build one with the file path and the receptor and effector chain
    roles.  The path is checked at construction, and the structure
    loads lazily and stays cached.
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
            raise ValueError("receptor_chain must be non-empty string")
        if not isinstance(effector_chain, str) or not effector_chain:
            raise ValueError("effector_chain must be non-empty string")
        if receptor_chain == effector_chain:
            raise ValueError(
                f"receptor_chain and effector_chain cannot both be "
                f"{receptor_chain!r}"
            )
        self._receptor_chain = receptor_chain
        self._effector_chain = effector_chain
        # The predictor tag is diagnostic.  No behaviour depends on
        # it.
        if predictor is not None and predictor not in ("boltz", "af3"):
            raise ValueError(
                f"predictor must be 'boltz' | 'af3' | None, got {predictor!r}"
            )
        self._predictor = predictor

        # Caches filled on first use.
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
        """Return the gemmi structure, parsing on first call."""
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

        Residues come in file order.  An unknown residue maps to 'X',
        and a chain with no atoms gives "".
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
                    # Alternate conformations repeat a residue, so
                    # deduplicate on (resnum, insertion code).
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

        Coordinates come in file order, not keyed by residue number.
        Pairing by order is deliberate, because AF3 numbers residues
        1-based per chain while Boltz and RFDiffusion number
        continuously across chains, so residue numbers cannot be
        matched.  A chain with no atoms gives an empty `(0, 3)` array.
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
        """Return a (resnum, xyz) pair for every non-hydrogen atom in
        the chain.  Cached."""
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
        """Receptor residues within `cutoff` Å of any effector heavy
        atom.

        Returns a PositionSet in the `prediction` frame, the
        predictor's per-chain 1-based numbering, so the values match
        the file.
        """
        rec = self._chain_heavy_atoms_list(self._receptor_chain)
        eff = self._chain_heavy_atoms_list(self._effector_chain)
        if not rec or not eff:
            return PositionSet(
                positions=[], chain=self._receptor_chain, frame="prediction"
            )
        # O(N*M), but at negsteer scale, roughly 100 residues and 50
        # atoms per chain, the cost is tiny.
        rec_coords = np.array([xyz for _, xyz in rec])
        eff_coords = np.array([xyz for _, xyz in eff])
        # Squared distances avoid a square root per pair.
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

    # ── Pair operations, asymmetric `self` vs `other` ──────────────

    def ra_eff_vs(self, other: "ProteinStructurePrediction") -> Optional[float]:
        """Return the receptor-aligned effector Cα RMSD.

        Aligning `self`'s receptor onto `other`'s and measuring the
        effector shift isolates binding-pose error from receptor fold
        error.  Returns None if either chain has under 3 residues.
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
        """Jaccard overlap of this prediction's receptor contact
        residues with `other`'s, at the given heavy-atom cutoff."""
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
    """Kabsch-align P onto Q and return the Cα RMSD.  Returns None if
    either array holds under 3 points."""
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
    """Receptor-aligned effector RMSD.

    It aligns pred_rec onto ref_rec, applies that transform to
    pred_eff, then measures pred_eff against ref_eff.  Atoms pair by
    order, up to the shorter of the two arrays on each chain.
    """
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
