"""
design_cohort.py
----------------
DesignCohort — the set of all MPNN sequences in one pipeline run,
each carrying its NegativeSteeringRun.

Per Phase 4 spec §2.10.  Tier 6 type — depends on NegativeSteeringRun,
OrthogonalMetrics (optional, attached separately), and
PipelineInternalThresholds.  Top of the hierarchy.

Knows tier breakdowns, the composite-score ranking across sequences,
and which sequences are survivors eligible for the orthogonal stage.
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from negative_steering_run import NegativeSteeringRun  # noqa: E402


@dataclass(frozen=True)
class DesignCohort:
    """All MPNN sequences from one pipeline run.

    Holds NegativeSteeringRun instances keyed by mpnn_sequence_id.
    Tier 6 constructor takes a list of runs directly; the workdir-
    walking factory (from_runs_directory) is implemented at migration
    time when the existing cross_sequence_summary.py absorbs.
    """
    runs: Tuple[NegativeSteeringRun, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.runs, tuple):
            object.__setattr__(self, "runs", tuple(self.runs))
        seen_ids = set()
        for r in self.runs:
            if not isinstance(r, NegativeSteeringRun):
                raise TypeError(
                    f"every run must be a NegativeSteeringRun, "
                    f"got {type(r).__name__}"
                )
            if r.mpnn_sequence_id in seen_ids:
                raise ValueError(
                    f"duplicate mpnn_sequence_id {r.mpnn_sequence_id!r}"
                )
            seen_ids.add(r.mpnn_sequence_id)

    # ── Lookup / filter ─────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.runs)

    def __iter__(self):
        return iter(self.runs)

    def get(self, mpnn_sequence_id: str) -> Optional[NegativeSteeringRun]:
        for r in self.runs:
            if r.mpnn_sequence_id == mpnn_sequence_id:
                return r
        return None

    def steered_runs(self) -> List[NegativeSteeringRun]:
        """Only steered rows (excludes controls)."""
        return [r for r in self.runs if r.row_type == "steered"]

    def control_runs(self) -> List[NegativeSteeringRun]:
        """Only the negative controls."""
        return [r for r in self.runs if r.row_type != "steered"]

    def by_tier(self, tier: str, thresholds) -> List[NegativeSteeringRun]:
        """Steered runs at the given tier (A/B/C/none)."""
        return [r for r in self.steered_runs() if r.tier(thresholds) == tier]

    def survivors(self, thresholds) -> List[NegativeSteeringRun]:
        """Tier A/B/C steered rows — the set fed to the orthogonal stage."""
        return [
            r for r in self.steered_runs()
            if r.tier(thresholds) in ("A", "B", "C")
        ]

    def tier_breakdown(self, thresholds) -> Dict[str, int]:
        """Per-tier count of STEERED runs (controls excluded)."""
        counts = Counter(
            r.tier(thresholds) for r in self.steered_runs()
        )
        return {
            "A": counts.get("A", 0),
            "B": counts.get("B", 0),
            "C": counts.get("C", 0),
            "none": counts.get("none", 0),
        }

    # ── Ranking ─────────────────────────────────────────────────────

    _TIER_ORDER = {"A": 0, "B": 1, "C": 2, "none": 3}

    def ranked_by_composite(
        self, thresholds
    ) -> List[NegativeSteeringRun]:
        """Sort steered runs by (tier_order, -composite_score).

        composite_score = `n_pass / n_seeds_total` (a simple ratio for
        Tier 6; the existing composite that weights jaccard − 0.05 ×
        ra_eff requires per-prediction metrics not yet accessible here
        and is computed in `to_cross_summary_csv` at migration time).
        """
        def key(r):
            tier = r.tier(thresholds)
            tier_idx = self._TIER_ORDER.get(tier, 99)
            n_seeds = r.n_seeds_total()
            ratio = r.n_pass(thresholds) / n_seeds if n_seeds > 0 else 0.0
            return (tier_idx, -ratio)
        return sorted(self.steered_runs(), key=key)

    # ── Stats ─────────────────────────────────────────────────────

    def n_steered(self) -> int:
        return sum(1 for r in self.runs if r.row_type == "steered")

    def n_controls(self) -> int:
        return sum(1 for r in self.runs if r.row_type != "steered")

    # ── Cross-summary CSV round-trip ────────────────────────────────

    @staticmethod
    def from_cross_summary_csv(path):
        """Build a CrossSummarySnapshot (shallow typed view) from an
        already-emitted cross_sequence_summary.csv.

        Returns a ``CrossSummarySnapshot`` — NOT a full DesignCohort —
        because the CSV doesn't carry enough information to reconstruct
        the StageResult / PSP scaffolding the full DesignCohort needs.
        The snapshot exposes the same survivors / tier_breakdown /
        ranked_by_composite query interface using the per-row
        cross_tier / cross_composite_score columns directly.
        """
        from cross_summary_view import CrossSummarySnapshot  # local import to keep deps lazy
        return CrossSummarySnapshot.from_csv(path)

    def to_cross_summary_csv(
        self,
        output_path,
        published_runs_dir=None,
        scored_metadata_path=None,
        strict: bool = False,
    ) -> int:
        """Emit a cross_sequence_summary.csv from this cohort.

        Delegates to the existing ``cross_sequence_summary.aggregate``
        function so the column set and aggregation behaviour stay
        bit-for-bit identical to the pre-Phase-4 emitter.  Each run's
        ``workdir / 'passing_summary.csv'`` is used as the input.

        Returns the aggregate return code (0 on success).
        """
        from cross_sequence_summary import aggregate  # local import
        sequences = [
            (r.mpnn_sequence_id, Path(r.workdir) / "passing_summary.csv")
            for r in self.runs
        ]
        return aggregate(
            sequences=sequences,
            output_path=Path(output_path),
            strict=strict,
            published_runs_dir=published_runs_dir,
            scored_metadata_path=scored_metadata_path,
        )

    @classmethod
    def emit_cross_summary_from_dirs(
        cls,
        runs_dirs,
        output_path,
        published_runs_dir=None,
        scored_metadata_path=None,
        strict: bool = False,
        explicit_pairs=None,
    ) -> int:
        """Walk one or more runs directories and emit cross_sequence_summary.csv.

        The Phase 4 typed entry point for cohort CSV emission.  Building
        full hydrated NegativeSteeringRun objects requires per-seed PDB
        + confidence parsing for every sequence; that cost is only
        warranted when downstream code consumes the StageResult chain.
        For CSV emission alone the workdir + mpnn_sequence_id pairs are
        sufficient — this classmethod gathers those and delegates to
        the existing ``cross_sequence_summary.aggregate`` so the column
        set stays bit-identical.

        Arguments
        ---------
        runs_dirs : iterable of Path-like
            Each directory contains one subdir per MPNN sequence with
            its own passing_summary.csv.  Repeat-tolerant — duplicates
            are deduplicated by sequence name.
        explicit_pairs : optional list of (name, path) tuples
            Per-sequence overrides for direct file paths (mirrors the
            existing --passing-summary CLI flag).
        """
        from cross_sequence_summary import aggregate  # local import

        # Sequence-name validation regex (matches the existing CLI).
        import re as _re
        _SEQ_NAME_RE = _re.compile(r"^[A-Za-z0-9_.+\-]+$")

        sequences: List[Tuple[str, Path]] = []
        seen: set = set()

        if explicit_pairs is not None:
            for name, path in explicit_pairs:
                if name in seen:
                    continue
                seen.add(name)
                sequences.append((name, Path(path)))

        for d in runs_dirs:
            d = Path(d)
            if not d.is_dir():
                continue
            for sub in sorted(d.iterdir()):
                if not sub.is_dir():
                    continue
                seq_name = sub.name
                if not _SEQ_NAME_RE.match(seq_name):
                    continue
                if seq_name in seen:
                    continue
                seen.add(seq_name)
                sequences.append((seq_name, sub / "passing_summary.csv"))

        if not sequences:
            return 2

        return aggregate(
            sequences=sequences,
            output_path=Path(output_path),
            strict=strict,
            published_runs_dir=published_runs_dir,
            scored_metadata_path=scored_metadata_path,
        )

    @classmethod
    def from_runs_directory(
        cls,
        runs_dir,
        num_seeds: int = 3,
        truth_psp=None,
        contamination_positions=None,
    ) -> "DesignCohort":
        """Walk a runs directory (one subdir per MPNN sequence) and build
        a DesignCohort of view-only NegativeSteeringRun instances.

        The deep form of NegativeSteeringRun (with full StageResult
        scaffolding) requires PSP loading per prediction.  This factory
        produces SHALLOW NegativeSteeringRun objects by delegating to
        ``NegativeSteeringRun.from_workdir``, which fabricates minimal
        cold-start stages from the per-sequence CSV so the dataclass
        invariants hold.  Verdict computation via
        ``per_seed_final_verdicts`` requires the runs to be hydrated
        with real StageResult / PSP data — see the from_workdir
        docstring for the boundary.
        """
        runs_dir = Path(runs_dir)
        runs: List[NegativeSteeringRun] = []
        for sub in sorted(runs_dir.iterdir()):
            if not sub.is_dir():
                continue
            run = NegativeSteeringRun.from_workdir(
                workdir=sub,
                mpnn_sequence_id=sub.name,
                num_seeds=num_seeds,
                truth_psp=truth_psp,
                contamination_positions=contamination_positions,
            )
            if run is not None:
                runs.append(run)
        return cls(runs=tuple(runs))

    def __repr__(self) -> str:
        return (
            f"DesignCohort(n_steered={self.n_steered()}, "
            f"n_controls={self.n_controls()})"
        )
