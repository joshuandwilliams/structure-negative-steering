"""
boltz_confidence.py
-------------------
BoltzConfidenceMetrics — wrapper around one Boltz prediction sample's
confidence sidecar JSON (and optional pLDDT npz).

Paired with one `ProteinStructurePrediction` from Boltz (1:1 — each
sample produces one PDB and one confidence JSON; this type wraps the
non-structural side).

Tier 0 scope (per Phase 4 spec §2.4): the JSON-direct metrics + mean
pLDDT.  PAE-derived metrics (`ipae`, `pae_pass_frac`, `ipsae*`) are
deferred to a later tier because they require the chain lengths from
the matching PDB (a `ProteinStructurePrediction`-dependent operation —
that type doesn't exist yet at Tier 0).  Until then,
`compute_metrics.py` keeps its existing PAE-derived computations and
`BoltzConfidenceMetrics` is augmented at Tier 2+ when it can take a
`ProteinStructurePrediction` as a parameter.

Tier 0 type — no upstream dependencies on other Phase 4 types.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class BoltzConfidenceMetrics:
    """One sample's confidence metrics from a Boltz prediction.

    Construction takes the path to a `confidence_<name>.json` file (and
    optionally a matching `plddt_<name>.npz`).  Lazy loading: the JSON
    is read on first access and cached for the instance's lifetime.

    Properties return `float | None`:
    - `None` if the sidecar was missing, unparseable, or the field was
      absent from the JSON.
    - Otherwise the value as a float (rounded to 4 dp where the
      existing codebase rounds, matching compute_metrics.py behaviour).
    """

    def __init__(
        self,
        confidence_json: Path,
        plddt_npz: Optional[Path] = None,
    ):
        self._confidence_json = Path(confidence_json)
        self._plddt_npz = Path(plddt_npz) if plddt_npz is not None else None
        if not self._confidence_json.is_file():
            raise FileNotFoundError(
                f"confidence JSON not found: {self._confidence_json}"
            )
        self._cached_json: Optional[Dict[str, Any]] = None
        self._cached_avg_plddt: Optional[float] = None
        self._avg_plddt_computed = False

    # ── Properties (lazy from JSON) ──────────────────────────────────

    def _load_json(self) -> Dict[str, Any]:
        if self._cached_json is None:
            try:
                with open(self._confidence_json) as f:
                    self._cached_json = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._cached_json = {}
        return self._cached_json

    @property
    def confidence_score(self) -> Optional[float]:
        return self._field_float("confidence_score")

    @property
    def ptm(self) -> Optional[float]:
        return self._field_float("ptm")

    @property
    def iptm(self) -> Optional[float]:
        """Boltz writes iptm directly; falls back to protein_iptm when
        only the protein-protein variant exists (matches compute_metrics
        behaviour at line 1241-1243)."""
        val = self._field_float("iptm")
        if val is not None:
            return val
        return self._field_float("protein_iptm")

    @property
    def protein_iptm(self) -> Optional[float]:
        return self._field_float("protein_iptm")

    @property
    def ligand_iptm(self) -> Optional[float]:
        return self._field_float("ligand_iptm")

    @property
    def complex_plddt(self) -> Optional[float]:
        """Whole-complex pLDDT on a 0-1 scale (Boltz convention)."""
        return self._field_float("complex_plddt")

    @property
    def complex_iplddt(self) -> Optional[float]:
        return self._field_float("complex_iplddt")

    @property
    def complex_pde(self) -> Optional[float]:
        return self._field_float("complex_pde")

    @property
    def complex_ipde(self) -> Optional[float]:
        return self._field_float("complex_ipde")

    @property
    def chains_ptm(self) -> Dict[str, float]:
        """Per-chain pTM dict keyed by chain index (string)."""
        raw = self._load_json().get("chains_ptm", {})
        out: Dict[str, float] = {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                try:
                    out[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
        return out

    @property
    def pair_chains_iptm(self) -> Dict[str, Dict[str, float]]:
        """Pairwise inter-chain ipTM dict, keyed by chain index strings."""
        raw = self._load_json().get("pair_chains_iptm", {})
        out: Dict[str, Dict[str, float]] = {}
        if isinstance(raw, dict):
            for k, inner in raw.items():
                if not isinstance(inner, dict):
                    continue
                inner_out: Dict[str, float] = {}
                for k2, v in inner.items():
                    try:
                        inner_out[str(k2)] = float(v)
                    except (TypeError, ValueError):
                        continue
                out[str(k)] = inner_out
        return out

    # ── Properties (from PLDDT npz, if provided) ─────────────────────

    @property
    def avg_plddt(self) -> Optional[float]:
        """Mean pLDDT across all residues, normalised to 0-100.

        Boltz writes the per-residue .npz on a 0-1 scale, conventional
        PDB B-factor pLDDT is 0-100; we always return 0-100 to match
        the rest of the pipeline (compute_metrics.py:1277-1278)."""
        if self._avg_plddt_computed:
            return self._cached_avg_plddt
        self._avg_plddt_computed = True
        if self._plddt_npz is None or not self._plddt_npz.is_file():
            self._cached_avg_plddt = None
            return None
        try:
            import numpy as np  # local — keep top-level import light
            data = np.load(self._plddt_npz)
            arr = data[list(data.keys())[0]]
            if arr.ndim > 1:
                arr = arr.flatten()
            mean = float(arr.mean())
            if float(arr.max()) <= 1.5:
                mean *= 100.0
            self._cached_avg_plddt = round(mean, 2)
        except (OSError, ValueError, KeyError, ImportError):
            self._cached_avg_plddt = None
        return self._cached_avg_plddt

    # ── Derived ──────────────────────────────────────────────────────

    def confidence_flag(self, thresholds) -> str:
        """Returns one of: ok | low_pass_frac | low_iptm | low_plddt |
        high_ipae | multiple.

        Tier 0 scope: pae_pass_frac and ipae are NOT yet available from
        this type (PAE matrix read deferred to Tier 2+).  Until then,
        confidence_flag checks iptm and complex_plddt only.  Migration
        of the full check (matching extract_passing.py:_compute_
        confidence_flag) lands when BoltzConfidenceMetrics gains PAE
        access via a ProteinStructurePrediction reference.
        """
        triggers: List[str] = []
        if self.iptm is not None and self.iptm < thresholds.iptm_min:
            triggers.append("low_iptm")
        if self.complex_plddt is not None and self.complex_plddt < thresholds.complex_plddt_min:
            triggers.append("low_plddt")
        if not triggers:
            return "ok"
        if len(triggers) == 1:
            return triggers[0]
        return "multiple"

    # ── Helpers ──────────────────────────────────────────────────────

    def _field_float(self, key: str) -> Optional[float]:
        raw = self._load_json().get(key)
        if raw is None:
            return None
        try:
            return round(float(raw), 4)
        except (TypeError, ValueError):
            return None

    @classmethod
    def from_prediction_dir(
        cls, prediction_dir: Path, base_name: str = "input"
    ) -> List["BoltzConfidenceMetrics"]:
        """Find every per-sample confidence JSON in `prediction_dir` and
        return a list of BoltzConfidenceMetrics, one per sample, sorted
        by model index.

        Boltz writes one `confidence_<base_name>_model_<sample>.json`
        per sample (sample = 0..N-1 where N = diffusion_samples).
        Default `base_name="input"` matches the typical Boltz output
        layout used everywhere in this pipeline.

        Each instance is also linked to the matching `plddt_*.npz` if
        present in the same directory.
        """
        prediction_dir = Path(prediction_dir)
        if not prediction_dir.is_dir():
            raise FileNotFoundError(
                f"prediction dir not found: {prediction_dir}"
            )
        confs = sorted(
            prediction_dir.glob(f"confidence_{base_name}_model_*.json")
        )
        out: List["BoltzConfidenceMetrics"] = []
        for cj in confs:
            # Match plddt npz by replacing the prefix.
            stem = cj.name[len("confidence_"):-len(".json")]
            plddt = prediction_dir / f"plddt_{stem}.npz"
            out.append(cls(cj, plddt if plddt.is_file() else None))
        return out
