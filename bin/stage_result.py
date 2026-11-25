"""StageResult — one stage's predictions for one configuration.

Tier 4 type, depending on ProteinStructurePrediction,
BoltzConfidenceMetrics, PositionSet and PipelineInternalThresholds.

A stage is `cold_start`, `steered` or `reversion`.  A configuration is
the receptor sequence predicted in this stage, with `num_seeds`
predictions each.  The collective verdict drives whether the *next*
stage runs, see ``triggers_next_stage``.

The steered gating rule lives here, at stage level, not the
per-(design, seed) gate in ``cmd_build_contaminated`` in
``reversion.py``.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
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
    from native, required for steered and reversion and None for
    cold_start, which has no mutations.

    `applies_to_designs` is reversion-only, listing the steered
    design_ids whose contamination produced this reverted sequence.
    Designs reverting to the same sequence share one stage.
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
        # Coerce to tuples for immutability.
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
        # Rejected at construction because the gating maths would be
        # wrong, not empty.  ceil(0 / 2) is 0, so a seedless cold_start
        # counts 0 clean seeds as "fewer than a majority" and
        # triggers steering, and canonical_prediction would hit
        # predictions[0] and raise IndexError.
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
        """Return one verdict per seed, parallel to seed_indices.

        Vocabulary:
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
        """Apply the stage's per-seed verdict rule to one prediction."""
        # Intact + ra_eff below the stage's cutoff.
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
        # Any contact in (mutated_positions ∩
        # contamination_positions).
        contacts = pred.receptor_contact_residues(thresholds.contact_cutoff)
        try:
            triple_intersection = (
                contacts.intersection(self.mutated_positions)
                       .intersection(contamination_positions)
            )
        except Exception:
            # A numbering-frame mismatch is the only realistic failure.
            # Treat it as missing data rather than lose the whole stage.
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
        """Seeds passing intact + ra_eff < cutoff."""
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
        """Count of correctly-placed seeds whose contacts intersect
        mutated_positions ∩ contamination_positions.  Only
        meaningful for steered and reversion, cold_start having no
        mutations."""
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
        """Count of seeds whose verdict is pass-equivalent, meaning
        'clean', 'clean_steered' or 'pose_holds' by stage.

        Per-stage only.  NegativeSteeringRun (Tier 5) replaces a seed's steered verdict with its reversion verdict
        when reversion ran for that design.
        """
        verdicts = self.per_seed_verdicts(thresholds, truth, contamination_positions)
        passing = {
            "cold_start": {"clean"},
            "steered": {"clean_steered"},
            "reversion": {"pose_holds"},
        }[self.stage_type]
        return sum(1 for v in verdicts if v in passing)

    # ── Stage-level gating ───────────────────────────────────────────

    def triggers_next_stage(
        self,
        thresholds,
        truth: ProteinStructurePrediction,
        contamination_positions: Optional[PositionSet] = None,
    ) -> bool:
        """The gating rule:

        - cold_start: True if fewer than a majority of seeds were
          clean, so steering should run.
        - steered: True if n_correctly_placed > 0 and n_contaminated >=
          ceil(n_correctly_placed / 2).
        - reversion: always False, no stage follows it.
        """
        if self.stage_type == "reversion":
            return False
        if self.stage_type == "cold_start":
            verdicts = self.per_seed_verdicts(thresholds, truth)
            n_clean = sum(1 for v in verdicts if v == "clean")
            majority = math.ceil(self.num_seeds / 2)
            return n_clean < majority
        # steered
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
        """Return the best-by-ra_eff prediction, preferring
        pass-equivalent seeds.  The per-stage representative for
        NegativeSteeringRun and DesignCohort."""
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
            # Lowest ra_eff among passing
            best = min(passing_idx, key=lambda i: all_with_ra[
                next(j for j, (_, ji) in enumerate(all_with_ra) if ji == i)
            ][0])
        elif all_with_ra:
            # Lowest ra_eff overall
            best = min(all_with_ra, key=lambda x: x[0])[1]
        else:
            # No ra_eff anywhere
            best = 0
        return self.predictions[best]
