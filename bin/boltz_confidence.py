"""
boltz_confidence.py
-------------------
BoltzConfidenceMetrics — a wrapper around one Boltz sample's
confidence sidecar JSON, plus its pLDDT npz when that is available.

Each Boltz sample produces one PDB and one confidence JSON, so this
type pairs one-to-one with a `ProteinStructurePrediction`.  That type
holds the structure, and this one holds everything else.

This type covers the metrics read straight from the JSON, plus the mean
pLDDT.  It does not cover the PAE-derived metrics `ipae`,
`pae_pass_frac` or `ipsae*`.  Those need the per-chain lengths, which
only the matching PDB can supply, and reading that would make this a
Tier 2 type.  So `compute_metrics.py` keeps computing them for now.
This class can take them over once it accepts a
`ProteinStructurePrediction` argument.

Tier 0 type — no upstream dependencies on other typed modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class BoltzConfidenceMetrics:
    """One sample's confidence metrics from a Boltz prediction.

    Build one with the path to a `confidence_<name>.json` file, and
    optionally the matching `plddt_<name>.npz`.  The JSON is read on
    first access and cached for the life of the instance.

    Each property returns a float or None.  It returns None when the
    sidecar was missing or unreadable, or when the JSON had no such
    field.  Otherwise it returns the value rounded to 4 decimal places,
    which matches what compute_metrics.py does.
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

    # ── Properties read from the JSON on first access ────────────────

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
        """Return ipTM.  Boltz usually writes `iptm` directly.  Some
        outputs carry only the protein-protein variant, so this falls
        back to `protein_iptm`.  compute_metrics.py does the same."""
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
        """Whole-complex pLDDT on a 0 to 1 scale, which is Boltz's
        convention."""
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
        """Per-chain pTM, keyed by the chain index as a string."""
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
        """Pairwise inter-chain ipTM, keyed by chain index strings in
        both the outer and the inner dict."""
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

    # ── Properties read from the pLDDT npz, when one was given ───────

    @property
    def avg_plddt(self) -> Optional[float]:
        """Mean pLDDT across all residues, always on a 0 to 100 scale.

        Boltz writes the per-residue .npz on a 0 to 1 scale, while the
        conventional pLDDT stored in a PDB B-factor runs 0 to 100.
        This property returns 0 to 100 so it matches the rest of the
        pipeline, including compute_metrics.py.
        """
        if self._avg_plddt_computed:
            return self._cached_avg_plddt
        self._avg_plddt_computed = True
        if self._plddt_npz is None or not self._plddt_npz.is_file():
            self._cached_avg_plddt = None
            return None
        try:
            # Imported here to keep the module's top-level imports light.
            import numpy as np
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
        """Return one of ok, low_pass_frac, low_iptm, low_plddt,
        high_ipae or multiple.

        This class cannot read the PAE matrix yet, so pae_pass_frac and
        ipae are unavailable here.  The check therefore looks at iptm
        and complex_plddt only.  The full check lives in
        `_compute_confidence_flag` in extract_passing.py.  This method
        can take it over once the class can reach a
        ProteinStructurePrediction for the PAE data.
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
        """Return one BoltzConfidenceMetrics per sample found in
        `prediction_dir`, sorted by model index.

        Boltz writes one `confidence_<base_name>_model_<sample>.json`
        per sample, numbered 0 to N-1 where N is diffusion_samples.
        The default `base_name="input"` matches the Boltz output layout
        this pipeline uses throughout.

        Each instance is also linked to the matching `plddt_*.npz` when
        that file sits in the same directory.
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
            # The npz has the same name with a different prefix.
            stem = cj.name[len("confidence_"):-len(".json")]
            plddt = prediction_dir / f"plddt_{stem}.npz"
            out.append(cls(cj, plddt if plddt.is_file() else None))
        return out
