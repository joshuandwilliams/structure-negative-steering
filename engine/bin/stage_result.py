"""
stage_result.py
---------------
StageResult — one stage's worth of predictions for one configuration.

Per Phase 4 spec §2.8.  Tier 4 type — depends on
ProteinStructurePrediction, BoltzConfidenceMetrics, PositionSet,
PipelineInternalThresholds.

A stage is `cold_start`, `steered`, or `reversion`.  A configuration is
the receptor sequence being predicted in this stage.  Each stage runs
`num_seeds` predictions for the configuration.  The collective verdict
drives whether the *next* stage runs (see ``triggers_next_stage``).

**CL-3 lands here**: for ``stage_type='steered'`` the gating rule is the
new majority-of-correctly-placed-contaminated rule, NOT the per-(design,
seed) gate currently in ``cmd_build_contaminated``.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from boltz_confidence import BoltzConfidenceMetrics  # noqa: E402
from position_set import PositionSet  # noqa: E402
from protein_structure_prediction import ProteinStructurePrediction  # noqa: E402


VALID_STAGE_TYPES = frozenset(("cold_start", "steered", "reversion"))


@dataclass(frozen=True)
class StageResult:
    """One stage's predictions for one configuration + collective verdict.

    `mutated_positions` is the PositionSet of receptor residues changed
    from native in this stage's configuration.  Required for steered
    and reversion stages; should be None for cold_start (which has no
    mutations).

    `applies_to_designs` is reversion-only: the steered design_ids
    whose contamination produced this unique reverted sequence (the
    dedup-by-reverted-sequence mapping; per Step 4.3 of the architecture
    spec).  None for cold_start and steered.
    """
    stage_type: str
    config_id: str
    predictions: Tuple[ProteinStructurePrediction, ...]
    confidences: Tuple[BoltzConfidenceMetrics, ...]
    seed_indices: Tuple[int, ...]
    mutated_positions: Optional[PositionSet] = None
    applies_to_designs: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if self.stage_type not in VALID_STAGE_TYPES:
            raise ValueError(
                f"stage_type {self.stage_type!r} not in "
                f"{sorted(VALID_STAGE_TYPES)}"
            )
        if not isinstance(self.config_id, str) or not self.config_id:
            raise ValueError("config_id must be a non-empty string")
        # Coerce to tuples for hashability/immutability.
        if not isinstance(self.predictions, tuple):
            object.__setattr__(self, "predictions", tuple(self.predictions))
        if not isinstance(self.confidences, tuple):
            object.__setattr__(self, "confidences", tuple(self.confidences))
        if not isinstance(self.seed_indices, tuple):
            object.__setattr__(self, "seed_indices", tuple(self.seed_indices))
        if self.applies_to_designs is not None and not isinstance(
            self.applies_to_designs, tuple
        ):
            object.__setattr__(
                self, "applies_to_designs", tuple(self.applies_to_designs)
            )
        n = len(self.predictions)
        if n == 0:
            raise ValueError("StageResult must have at least 1 prediction")
        if len(self.confidences) != n:
            raise ValueError(
                f"confidences length ({len(self.confidences)}) must match "
                f"predictions length ({n})"
            )
        if len(self.seed_indices) != n:
            raise ValueError(
                f"seed_indices length ({len(self.seed_indices)}) must "
                f"match predictions length ({n})"
            )
        if self.stage_type == "cold_start" and self.mutated_positions is not None:
            raise ValueError(
                "cold_start stage cannot carry mutated_positions "
                "(by definition cold-start uses the native sequence)"
            )
        if self.stage_type in ("steered", "reversion"):
            if self.mutated_positions is None:
                raise ValueError(
                    f"{self.stage_type} stage requires mutated_positions"
                )
        if self.applies_to_designs is not None and self.stage_type != "reversion":
            raise ValueError(
                "applies_to_designs is reversion-only"
            )

    @property
    def num_seeds(self) -> int:
        return len(self.predictions)

    # ── Per-seed verdicts ────────────────────────────────────────────

    def per_seed_verdicts(
        self,
        thresholds,
        truth: ProteinStructurePrediction,
        contamination_positions: Optional[PositionSet] = None,
    ) -> List[str]:
        """Return one verdict per seed (parallel to seed_indices).

        Vocabulary per stage type:
          - cold_start: 'clean' | 'fail_structural' | 'no_data'
          - steered: 'clean_steered' | 'contaminated' | 'pose_collapses' | 'no_data'
          - reversion: 'pose_holds' | 'new_contamination' | 'pose_collapses' | 'no_data'
        """
        if self.stage_type in ("steered", "reversion") and contamination_positions is None:
            raise ValueError(
                f"{self.stage_type} stage requires contamination_positions"
            )
        return [
            self._classify_one_seed(pred, thresholds, truth, contamination_positions)
            for pred in self.predictions
        ]

    def _classify_one_seed(
        self,
        pred: ProteinStructurePrediction,
        thresholds,
        truth: ProteinStructurePrediction,
        contamination_positions: Optional[PositionSet],
    ) -> str:
        """Apply the stage-specific per-seed verdict rule for one prediction."""
        # Structural check: intact + ra_eff below the stage's cutoff.
        ra_eff_cutoff = (
            thresholds.reverted_ra_eff
            if self.stage_type == "reversion"
            else thresholds.steered_ra_eff
        )
        ra_eff = pred.ra_eff_vs(truth)
        intact_rmsd = pred.independent_receptor_rmsd_vs(truth)
        if ra_eff is None or intact_rmsd is None:
            return "no_data"
        structurally_ok = (
            intact_rmsd < thresholds.intact_threshold
            and ra_eff < ra_eff_cutoff
        )
        if self.stage_type == "cold_start":
            return "clean" if structurally_ok else "fail_structural"
        if not structurally_ok:
            return "pose_collapses"
        # Contamination check: any contact in (mutated_positions ∩
        # contamination_positions).
        contacts = pred.receptor_contact_residues(thresholds.contact_cutoff)
        try:
            triple_intersection = (
                contacts.intersection(self.mutated_positions)
                       .intersection(contamination_positions)
            )
        except Exception:
            # Frame mismatch is the only realistic failure here; treat
            # as data unavailability for a single seed rather than
            # raising and bringing down the whole stage.
            return "no_data"
        has_contamination = not triple_intersection.is_empty()
        if self.stage_type == "steered":
            return "contaminated" if has_contamination else "clean_steered"
        # reversion
        return "new_contamination" if has_contamination else "pose_holds"

    # ── Aggregate counts ─────────────────────────────────────────────

    def n_correctly_placed(
        self, thresholds, truth: ProteinStructurePrediction
    ) -> int:
        """Count of seeds that pass intact + ra_eff < cutoff."""
        ra_eff_cutoff = (
            thresholds.reverted_ra_eff
            if self.stage_type == "reversion"
            else thresholds.steered_ra_eff
        )
        count = 0
        for pred in self.predictions:
            ra_eff = pred.ra_eff_vs(truth)
            intact_rmsd = pred.independent_receptor_rmsd_vs(truth)
            if ra_eff is None or intact_rmsd is None:
                continue
            if intact_rmsd < thresholds.intact_threshold and ra_eff < ra_eff_cutoff:
                count += 1
        return count

    def n_contaminated(
        self,
        thresholds,
        truth: ProteinStructurePrediction,
        contamination_positions: PositionSet,
    ) -> int:
        """Count of correctly-placed seeds whose contact residues
        intersect mutated_positions ∩ contamination_positions.  Only
        meaningful for steered/reversion (cold_start has no mutations)."""
        if self.stage_type == "cold_start":
            return 0
        verdicts = self.per_seed_verdicts(thresholds, truth, contamination_positions)
        if self.stage_type == "steered":
            return sum(1 for v in verdicts if v == "contaminated")
        # reversion
        return sum(1 for v in verdicts if v == "new_contamination")

    def n_pass(
        self,
        thresholds,
        truth: ProteinStructurePrediction,
        contamination_positions: Optional[PositionSet] = None,
    ) -> int:
        """Count of seeds whose stage-level verdict is pass-equivalent.

        Pass-equivalent per stage:
          - cold_start: 'clean'
          - steered: 'clean_steered'
          - reversion: 'pose_holds'

        NOTE: at the NegativeSteeringRun (Tier 5) level, the cross-stage
        per-seed verdict aggregation overrides a seed's steered verdict
        with its reversion verdict if reversion ran for that design.
        This method gives the per-stage count only.
        """
        verdicts = self.per_seed_verdicts(thresholds, truth, contamination_positions)
        passing = {
            "cold_start": {"clean"},
            "steered": {"clean_steered"},
            "reversion": {"pose_holds"},
        }[self.stage_type]
        return sum(1 for v in verdicts if v in passing)

    # ── Stage-level gating (CL-3 lives here) ─────────────────────────

    def triggers_next_stage(
        self,
        thresholds,
        truth: ProteinStructurePrediction,
        contamination_positions: Optional[PositionSet] = None,
    ) -> bool:
        """Apply the stage-specific gating rule:

        - cold_start: True if fewer than majority of seeds were clean
          (i.e. steering should run).  Existing rule.
        - steered: True if n_correctly_placed > 0 AND n_contaminated >=
          ceil(n_correctly_placed / 2).  **NEW rule from CL-3** —
          replaces the per-(design, seed) gate currently in
          cmd_build_contaminated.
        - reversion: False always (no next stage).
        """
        if self.stage_type == "reversion":
            return False
        if self.stage_type == "cold_start":
            verdicts = self.per_seed_verdicts(thresholds, truth)
            n_clean = sum(1 for v in verdicts if v == "clean")
            majority = math.ceil(self.num_seeds / 2)
            return n_clean < majority
        # steered — CL-3
        if contamination_positions is None:
            raise ValueError(
                "steered.triggers_next_stage requires contamination_positions"
            )
        n_correct = self.n_correctly_placed(thresholds, truth)
        if n_correct == 0:
            return False  # nothing to revert
        n_contam = self.n_contaminated(thresholds, truth, contamination_positions)
        return n_contam >= math.ceil(n_correct / 2)

    # ── Selection ─────────────────────────────────────────────────────

    def canonical_prediction(
        self,
        thresholds,
        truth: ProteinStructurePrediction,
        contamination_positions: Optional[PositionSet] = None,
    ) -> ProteinStructurePrediction:
        """Return the best-by-ra_eff prediction (pass-equivalent seeds
        preferred over non-passing).  Used by NegativeSteeringRun /
        DesignCohort for the per-stage representative."""
        verdicts = self.per_seed_verdicts(thresholds, truth, contamination_positions)
        passing_idx: List[int] = []
        all_with_ra: List[Tuple[float, int]] = []
        passing = {
            "cold_start": {"clean"},
            "steered": {"clean_steered"},
            "reversion": {"pose_holds"},
        }[self.stage_type]
        for i, (pred, verdict) in enumerate(zip(self.predictions, verdicts)):
            ra_eff = pred.ra_eff_vs(truth)
            if ra_eff is None:
                continue
            all_with_ra.append((ra_eff, i))
            if verdict in passing:
                passing_idx.append(i)
        if passing_idx:
            # Among passing, pick lowest ra_eff
            best = min(passing_idx, key=lambda i: all_with_ra[
                next(j for j, (_, ji) in enumerate(all_with_ra) if ji == i)
            ][0])
        elif all_with_ra:
            # No passing — pick lowest ra_eff overall
            best = min(all_with_ra, key=lambda x: x[0])[1]
        else:
            # No ra_eff anywhere — fall back to first
            best = 0
        return self.predictions[best]
