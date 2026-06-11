"""
cross_summary_view.py
---------------------
CrossSummaryRow + CrossSummarySnapshot — a typed read-only view over
``cross_sequence_summary.csv``.

The full Phase 4 DesignCohort / NegativeSteeringRun types model the
underlying experiment (StageResults, PSPs, contamination positions).
Building them faithfully requires the per-prediction PDB files, the
gemmi-loaded structures, and the ground-truth PSP — that is the
"deep" form.

But many cohort-level queries (which sequences survived, what tier did
each land in, what is the composite-ranked order) only need the
already-aggregated values cross_sequence_summary.csv has stamped onto
each row.  This module provides a shallow typed view for those
queries, so callers can opt for the cheap path when the deep form is
overkill.

Both forms expose the same DesignCohort-style interface:
``survivors()``, ``tier_breakdown()``, ``ranked_by_composite()``.

Tier 6.5 — sits beside DesignCohort.  No dependency on PSP / gemmi.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _try_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _try_int(v) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class CrossSummaryRow:
    """One row of cross_sequence_summary.csv as a typed record.

    Carries the cohort-level decisions stamped onto each MPNN sequence:
    tier, n_pass, outcome, composite score, run-one runtime.  Native
    columns not yet typed remain accessible via the ``extras`` dict.
    """
    mpnn_sequence: str
    row_type: str
    cross_tier: str  # "A" / "B" / "C" / "none"
    cross_composite_score: Optional[float]
    cross_rank_by_composite: Optional[int]
    cross_rank_by_ra_eff: Optional[int]
    within_sequence_rank: Optional[int]
    n_pass: Optional[int]
    n_seeds: Optional[int]
    outcome: str
    outcome_reason: str
    run_one_runtime_sec: Optional[float]
    extras: Dict[str, str]

    @classmethod
    def from_dict(cls, row: Dict[str, str]) -> "CrossSummaryRow":
        """Build a CrossSummaryRow from a csv.DictReader row."""
        known = {
            "mpnn_sequence", "row_type", "cross_tier",
            "cross_composite_score", "cross_rank_by_composite",
            "cross_rank_by_ra_eff", "within_sequence_rank",
            "n_pass", "n_seeds", "outcome", "outcome_reason",
            "run_one_runtime_sec",
        }
        extras = {k: v for k, v in row.items() if k not in known and v is not None}
        return cls(
            mpnn_sequence=row.get("mpnn_sequence", ""),
            row_type=row.get("row_type", "steered"),
            cross_tier=row.get("cross_tier", "none"),
            cross_composite_score=_try_float(row.get("cross_composite_score")),
            cross_rank_by_composite=_try_int(row.get("cross_rank_by_composite")),
            cross_rank_by_ra_eff=_try_int(row.get("cross_rank_by_ra_eff")),
            within_sequence_rank=_try_int(row.get("within_sequence_rank")),
            n_pass=_try_int(row.get("n_pass")),
            n_seeds=_try_int(row.get("n_seeds")),
            outcome=row.get("outcome", ""),
            outcome_reason=row.get("outcome_reason", ""),
            run_one_runtime_sec=_try_float(row.get("run_one_runtime_sec")),
            extras=extras,
        )


@dataclass(frozen=True)
class CrossSummarySnapshot:
    """Typed view over a cross_sequence_summary.csv.

    Read-only.  Exposes the cohort queries DesignCohort exposes
    (survivors / tier_breakdown / ranked_by_composite) on a row-level
    basis without needing the deep StageResult / PSP scaffolding.
    """
    rows: Tuple[CrossSummaryRow, ...]
    source_csv: Optional[Path] = None

    def __post_init__(self) -> None:
        if not isinstance(self.rows, tuple):
            object.__setattr__(self, "rows", tuple(self.rows))

    @classmethod
    def from_csv(cls, path: Path) -> "CrossSummarySnapshot":
        path = Path(path)
        rows: List[CrossSummaryRow] = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(CrossSummaryRow.from_dict(row))
        return cls(rows=tuple(rows), source_csv=path)

    # ── Lookup / filter ────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def get(self, mpnn_sequence: str) -> Optional[CrossSummaryRow]:
        for r in self.rows:
            if r.mpnn_sequence == mpnn_sequence:
                return r
        return None

    def steered_rows(self) -> List[CrossSummaryRow]:
        return [r for r in self.rows if r.row_type == "steered"]

    def control_rows(self) -> List[CrossSummaryRow]:
        return [r for r in self.rows if r.row_type != "steered"]

    def by_tier(self, tier: str) -> List[CrossSummaryRow]:
        return [r for r in self.steered_rows() if r.cross_tier == tier]

    def survivors(self) -> List[CrossSummaryRow]:
        """Tier A/B/C steered rows — the set fed to the orthogonal stage."""
        return [
            r for r in self.steered_rows()
            if r.cross_tier in ("A", "B", "C")
        ]

    def tier_breakdown(self) -> Dict[str, int]:
        out = {"A": 0, "B": 0, "C": 0, "none": 0}
        for r in self.steered_rows():
            out[r.cross_tier] = out.get(r.cross_tier, 0) + 1
        return out

    _TIER_ORDER = {"A": 0, "B": 1, "C": 2, "none": 3}

    def ranked_by_composite(self) -> List[CrossSummaryRow]:
        """Sort steered rows by (tier_order, -composite_score).

        Equivalent to the existing ``_assign_cross_ranks`` in
        cross_sequence_summary.py — used here as a verification path."""
        def key(r: CrossSummaryRow):
            tier_idx = self._TIER_ORDER.get(r.cross_tier, 99)
            composite = r.cross_composite_score
            score = composite if composite is not None else float("-inf")
            return (tier_idx, -score)
        return sorted(self.steered_rows(), key=key)

    # ── Stats ──────────────────────────────────────────────────────

    def n_steered(self) -> int:
        return sum(1 for r in self.rows if r.row_type == "steered")

    def n_controls(self) -> int:
        return sum(1 for r in self.rows if r.row_type != "steered")

    def __repr__(self) -> str:
        return (
            f"CrossSummarySnapshot(n_steered={self.n_steered()}, "
            f"n_controls={self.n_controls()}, "
            f"source={self.source_csv})"
        )
