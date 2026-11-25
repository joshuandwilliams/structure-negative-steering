"""DesignCohort — every MPNN sequence in one pipeline run, each
carrying its NegativeSteeringRun.

Tier 6 type, the top of the hierarchy, depending on
NegativeSteeringRun, PipelineInternalThresholds and optionally
OrthogonalMetrics, attached separately.

It owns the tier breakdowns, the composite-score ranking and the
survivors eligible for the orthogonal stage.
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from negative_steering_run import NegativeSteeringRun  # noqa: E402


@dataclass(frozen=True)
class DesignCohort:
    """All MPNN sequences from one pipeline run.

    It holds NegativeSteeringRun instances keyed by mpnn_sequence_id.
    The constructor takes a list of runs, and from_runs_directory walks
    a workdir.  It will take over from cross_summary.py's parsers when
    those callers migrate.
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
        """The steered rows only."""
        return [r for r in self.runs if r.row_type == "steered"]

    def control_runs(self) -> List[NegativeSteeringRun]:
        """The control rows."""
        return [r for r in self.runs if r.row_type != "steered"]

    def by_tier(self, tier: str, thresholds) -> List[NegativeSteeringRun]:
        """The steered runs at one tier, A, B, C or none."""
        return [r for r in self.steered_runs() if r.tier(thresholds) == tier]

    def survivors(self, thresholds) -> List[NegativeSteeringRun]:
        """The tier A, B and C steered rows, the set fed to the
        orthogonal stage."""
        return [
            r for r in self.steered_runs()
            if r.tier(thresholds) in ("A", "B", "C")
        ]

    def tier_breakdown(self, thresholds) -> Dict[str, int]:
        """Count the steered runs per tier, leaving out controls."""
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
        """Sort the steered runs by tier, then composite score
        descending.

        The composite score here is `n_pass / n_seeds_total`.  The
        richer `true_jaccard - 0.05 * ra_eff` needs per-prediction
        metrics this tier cannot reach, so `to_cross_summary_csv`
        computes it.
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
        """Build a shallow typed view from a written
        cross_sequence_summary.csv.

        It returns a ``CrossSummarySnapshot``, not a full DesignCohort,
        because the CSV cannot rebuild the StageResult and prediction
        scaffolding.  The snapshot still answers survivors,
        tier_breakdown and ranked_by_composite from the per-row
        cross_tier and cross_composite_score columns.
        """
        # Lazy, to keep module dependencies lazy.
        from cross_summary_model import CrossSummarySnapshot
        return CrossSummarySnapshot.from_csv(path)

    def to_cross_summary_csv(
        self,
        output_path,
        published_runs_dir=None,
        scored_metadata_path=None,
        strict: bool = False,
    ) -> int:
        """Write a cross_sequence_summary.csv for this cohort.

        ``cross_summary.aggregate`` does the work, so the columns and
        aggregation stay byte-identical to the older untyped emitter.
        Each run contributes its ``workdir / 'passing_summary.csv'``.
        Returns aggregate's code, 0 on success.
        """
        from cross_summary import aggregate
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
        """Walk runs directories and write cross_sequence_summary.csv.

        Fully loaded NegativeSteeringRun objects would parse every
        seed's PDB and confidence file, only worth it when downstream
        code uses the StageResult chain.  CSV emission needs only the
        workdir and mpnn_sequence_id pairs, which this hands to
        ``cross_summary.aggregate``.

        Each `runs_dirs` directory holds one subdirectory per MPNN
        sequence with its passing_summary.csv inside.  Repeats are
        fine, duplicates dropping by name.  `explicit_pairs` is
        (name, path) tuples mirroring the --passing-summary CLI flag.
        """
        # Matches the CLI's own sequence-name pattern.
        import re as _re

        from cross_summary import aggregate
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
        """Walk a runs directory and build a DesignCohort of view-only
        NegativeSteeringRun instances, one per MPNN subdirectory.

        The deep form loads a ProteinStructurePrediction per
        prediction, which is expensive, so this factory produces
        shallow runs through ``NegativeSteeringRun.from_workdir``.  That
        reads each seed's Boltz PDB and confidence JSON and builds the
        minimum scaffolding the dataclass invariants need, with no CSV.

        A shallow run cannot be asked for verdicts, since
        ``per_seed_final_verdicts`` needs real StageResult and
        prediction data.  Hydrate first.
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
