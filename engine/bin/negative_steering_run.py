"""
negative_steering_run.py
------------------------
NegativeSteeringRun — one MPNN sequence's full negsteer experiment.

Per Phase 4 spec §2.9.  Tier 5 type — depends on StageResult,
DesignedSequence, PipelineInternalThresholds.

Read-only view: constructed from already-completed stages (the
orchestration stays in `negative_steering_run_one.sh` + the Nextflow
process body per Q10).  Owns the cross-stage per-seed verdict
aggregation and the outcome / n_pass / tier decisions.
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from boltz_confidence import BoltzConfidenceMetrics  # noqa: E402
from designed_sequence import DesignedSequence  # noqa: E402
from position_set import PositionSet  # noqa: E402
from protein_structure_prediction import ProteinStructurePrediction  # noqa: E402
from stage_result import StageResult  # noqa: E402


VALID_ROW_TYPES = frozenset(
    ("steered", "control_scrambled", "control_polyA")
)


# ═══════════════════════════════════════════════════════════════════════
# Workdir discovery helpers (used by from_workdir)
# ═══════════════════════════════════════════════════════════════════════


def _find_boltz_outputs(
    pred_dir: Path, model_sample: int = 0
) -> Tuple[Optional[Path], Optional[Path]]:
    """Locate the canonical Boltz PDB + confidence JSON in a prediction
    directory.  Returns (None, None) if either is missing.

    Layout (per boltz2_iterate_steering.py:4426):
      <pred_dir>/boltz_results_input/predictions/input/
          pdb/input_model_<M>.pdb
          confidence_input_model_<M>.json
    """
    pred_root = pred_dir / "boltz_results_input" / "predictions" / "input"
    if not pred_root.is_dir():
        return None, None
    pdb_path = pred_root / "pdb" / f"input_model_{model_sample}.pdb"
    if not pdb_path.is_file():
        # Some pipeline versions write the PDB directly without the
        # pdb/ subdir.  Try the flat variant too.
        alt = pred_root / f"input_model_{model_sample}.pdb"
        pdb_path = alt if alt.is_file() else None
    conf_path = pred_root / f"confidence_input_model_{model_sample}.json"
    if not conf_path.is_file():
        conf_path = None
    return pdb_path, conf_path


def _read_mutations_positionset(
    tsv_path: Path, chain: str
) -> Optional["PositionSet"]:
    """Parse mutations.tsv (columns: pos1, wt, mut) into a designed-
    frame PositionSet.  Returns None when the file is missing."""
    import csv as _csv
    if not tsv_path.is_file():
        return None
    positions: List[int] = []
    with open(tsv_path) as f:
        reader = _csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                positions.append(int(row["pos1"]))
            except (KeyError, ValueError, TypeError):
                continue
    return PositionSet(positions=positions, chain=chain, frame="designed")


def _read_seed_stage_dirs(
    cycle_dir: Path,
    num_seeds: int,
    receptor_chain: str,
    effector_chain: str,
    model_sample: int,
    require_pdb: bool,
) -> Tuple[List, List, List[int]]:
    """Walk the cold-start seed subdirectories under cycle_dir.
    Convention: seed 0 is in ``initial/``; seeds 1..N are in
    ``initial_s1/``, ``initial_s2/``, ...

    Returns three parallel lists (predictions, confidences, seed_indices),
    skipping seeds whose PDB or confidence sidecar is missing.
    """
    preds: List = []
    confs: List = []
    seed_idx_out: List[int] = []
    for seed_idx in range(num_seeds):
        sub_name = "initial" if seed_idx == 0 else f"initial_s{seed_idx}"
        sub = cycle_dir / sub_name
        if not sub.is_dir():
            continue
        pdb, conf = _find_boltz_outputs(sub, model_sample)
        if pdb is None or conf is None:
            if require_pdb:
                continue
            # If require_pdb is False, we can't construct without files.
            # Skip silently.
            continue
        preds.append(ProteinStructurePrediction(
            path=pdb,
            receptor_chain=receptor_chain,
            effector_chain=effector_chain,
            predictor="boltz",
        ))
        confs.append(BoltzConfidenceMetrics(conf))
        seed_idx_out.append(seed_idx)
    return preds, confs, seed_idx_out


@dataclass(frozen=True)
class NegativeSteeringRun:
    """One MPNN sequence's complete negsteer chain.

    Tier 5 constructor takes pre-built StageResults; the workdir-
    walking factory (``from_workdir``) is implemented at migration time
    when the existing parsers in cross_sequence_summary.py are absorbed.
    """
    mpnn_sequence_id: str
    workdir: Path
    row_type: str
    cold_start: StageResult
    steered: Dict[str, StageResult] = field(default_factory=dict)
    reversion: Dict[str, StageResult] = field(default_factory=dict)
    truth: Optional[ProteinStructurePrediction] = None
    contamination_positions: Optional[PositionSet] = None
    designed_sequence: Optional[DesignedSequence] = None
    run_one_runtime_sec: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.mpnn_sequence_id, str) or not self.mpnn_sequence_id:
            raise ValueError("mpnn_sequence_id must be a non-empty string")
        if self.row_type not in VALID_ROW_TYPES:
            raise ValueError(
                f"row_type {self.row_type!r} not in {sorted(VALID_ROW_TYPES)}"
            )
        if not isinstance(self.cold_start, StageResult):
            raise TypeError("cold_start must be a StageResult")
        if self.cold_start.stage_type != "cold_start":
            raise ValueError(
                f"cold_start stage must have stage_type='cold_start', "
                f"got {self.cold_start.stage_type!r}"
            )
        for design_id, st in self.steered.items():
            if not isinstance(st, StageResult):
                raise TypeError(
                    f"steered[{design_id!r}] must be a StageResult"
                )
            if st.stage_type != "steered":
                raise ValueError(
                    f"steered[{design_id!r}] must have stage_type='steered'"
                )
        for rev_id, st in self.reversion.items():
            if not isinstance(st, StageResult):
                raise TypeError(
                    f"reversion[{rev_id!r}] must be a StageResult"
                )
            if st.stage_type != "reversion":
                raise ValueError(
                    f"reversion[{rev_id!r}] must have stage_type='reversion'"
                )

    @property
    def num_seeds(self) -> int:
        return self.cold_start.num_seeds

    # ── Construction from a per-sequence workdir ──────────────────

    @classmethod
    def from_workdir(
        cls,
        workdir,
        mpnn_sequence_id: str,
        num_seeds: int = 3,
        truth_psp=None,
        contamination_positions=None,
        receptor_chain: str = "A",
        effector_chain: str = "B",
        row_type: Optional[str] = None,
        cycle: int = 0,
        model_sample: int = 0,
        require_pdb: bool = True,
    ) -> Optional["NegativeSteeringRun"]:
        """Build a NegativeSteeringRun by reading a per-MPNN workdir.

        Walks the canonical workdir layout produced by
        ``negative_steering_run_one.sh``:

            <workdir>/cycle_<C>/initial[_sN]/                ← cold-start seeds
            <workdir>/cycle_<C>/steered/design_NN_sS/        ← steered (design, seed)
            <workdir>/cycle_<C>/reversions/rev_design_NN_sS_sR/  ← reversion

        For each prediction directory, the canonical Boltz output is
        ``boltz_results_input/predictions/input/pdb/input_model_<M>.pdb``
        with sidecar ``confidence_input_model_<M>.json``.  M is the
        Boltz sample index within a seed (default 0 — first sample;
        matches the post-canonicalisation convention used elsewhere).

        Mutations for steered/reverted stages are read from
        ``mutations.tsv`` in the prediction directory (columns:
        ``pos1 wt mut``); the resulting positions become the
        ``mutated_positions`` PositionSet (frame="designed", since
        mutation positions are 1-based on the designed-frame receptor).

        Returns None if the workdir has no cold-start data — signals
        "skip this sequence" to ``DesignCohort.from_runs_directory``.

        Arguments
        ---------
        receptor_chain / effector_chain : Chain labels Boltz emits.
            "A"/"B" for the standard layout.
        row_type : Override the row_type read from row_type.txt.  If
            None and no sidecar exists, defaults to "steered".
        cycle : Which cycle directory to read (default 0; multi-cycle
            workdirs not yet exercised here).
        model_sample : Which Boltz model sample to use as canonical
            (default 0).  The pipeline's post-hoc canonicalisation
            chooses one of 5; choosing 0 is a defensible default
            without re-running canonicalisation logic.
        require_pdb : If True (default) skip seeds whose PDB file is
            missing.  If False, allow PSP construction to raise.
        """
        import csv
        import re
        from pathlib import Path

        workdir = Path(workdir)
        cycle_dir = workdir / f"cycle_{cycle}"
        if not cycle_dir.is_dir():
            return None

        # ── row_type from sidecar ────────────────────────────────────
        if row_type is None:
            sidecar = workdir / "row_type.txt"
            if sidecar.is_file():
                row_type = sidecar.read_text().strip() or "steered"
            else:
                row_type = "steered"

        # ── Cold start ──────────────────────────────────────────────
        cold_preds, cold_confs, cold_seeds = _read_seed_stage_dirs(
            cycle_dir, num_seeds, receptor_chain, effector_chain,
            model_sample, require_pdb,
        )
        if not cold_preds:
            return None
        cold_start_stage = StageResult(
            stage_type="cold_start",
            config_id=mpnn_sequence_id,
            predictions=tuple(cold_preds),
            confidences=tuple(cold_confs),
            seed_indices=tuple(cold_seeds),
            mutated_positions=None,
        )

        # ── Steered ─────────────────────────────────────────────────
        steered: Dict[str, StageResult] = {}
        steered_root = cycle_dir / "steered"
        if steered_root.is_dir():
            by_design: Dict[str, List[Tuple[int, Path]]] = {}
            for sub in sorted(steered_root.iterdir()):
                if not sub.is_dir():
                    continue
                m = re.match(r"(design_\d+)_s(\d+)$", sub.name)
                if not m:
                    continue
                design_id = m.group(1)
                seed_idx = int(m.group(2))
                by_design.setdefault(design_id, []).append((seed_idx, sub))

            for design_id, items in by_design.items():
                items.sort(key=lambda x: x[0])
                preds, confs, seeds = [], [], []
                mut_positions = None
                for seed_idx, sub in items:
                    pdb, conf = _find_boltz_outputs(sub, model_sample)
                    if pdb is None or conf is None:
                        if require_pdb:
                            continue
                    if pdb is not None and conf is not None:
                        preds.append(ProteinStructurePrediction(
                            path=pdb,
                            receptor_chain=receptor_chain,
                            effector_chain=effector_chain,
                            predictor="boltz",
                        ))
                        confs.append(BoltzConfidenceMetrics(conf))
                        seeds.append(seed_idx)
                        if mut_positions is None:
                            mut_positions = _read_mutations_positionset(
                                sub / "mutations.tsv", receptor_chain,
                            )
                if not preds:
                    continue
                # mutated_positions REQUIRED for steered stage by StageResult
                # invariants; synthesise an empty designed-frame set if the
                # mutations.tsv was missing rather than dropping the stage.
                if mut_positions is None:
                    mut_positions = PositionSet(
                        positions=(),
                        chain=receptor_chain,
                        frame="designed",
                    )
                steered[design_id] = StageResult(
                    stage_type="steered",
                    config_id=f"{mpnn_sequence_id}__{design_id}",
                    predictions=tuple(preds),
                    confidences=tuple(confs),
                    seed_indices=tuple(seeds),
                    mutated_positions=mut_positions,
                )

        # ── Reversions ──────────────────────────────────────────────
        reversion: Dict[str, StageResult] = {}
        rev_root = cycle_dir / "reversions"
        if rev_root.is_dir():
            # Reverted-sequence id pattern: rev_design_NN_sS_sR
            #   NN = source design, S = source seed, R = reversion seed.
            # The (design_NN, source_seed) pair identifies the
            # PARENT steered slot; the reversion stage groups seeds
            # by parent.  applies_to_designs gets the parent design_id.
            by_parent: Dict[
                Tuple[str, int], List[Tuple[int, Path]]
            ] = {}
            for sub in sorted(rev_root.iterdir()):
                if not sub.is_dir():
                    continue
                m = re.match(r"rev_(design_\d+)_s(\d+)_s(\d+)$", sub.name)
                if not m:
                    continue
                parent_design = m.group(1)
                parent_seed = int(m.group(2))
                rev_seed = int(m.group(3))
                by_parent.setdefault(
                    (parent_design, parent_seed), []
                ).append((rev_seed, sub))

            for (parent_design, parent_seed), items in by_parent.items():
                items.sort(key=lambda x: x[0])
                preds, confs, seeds = [], [], []
                mut_positions = None
                for rev_seed, sub in items:
                    pdb, conf = _find_boltz_outputs(sub, model_sample)
                    if pdb is None or conf is None:
                        if require_pdb:
                            continue
                    if pdb is not None and conf is not None:
                        preds.append(ProteinStructurePrediction(
                            path=pdb,
                            receptor_chain=receptor_chain,
                            effector_chain=effector_chain,
                            predictor="boltz",
                        ))
                        confs.append(BoltzConfidenceMetrics(conf))
                        seeds.append(rev_seed)
                        if mut_positions is None:
                            mut_positions = _read_mutations_positionset(
                                sub / "mutations.tsv", receptor_chain,
                            )
                if not preds:
                    continue
                if mut_positions is None:
                    mut_positions = PositionSet(
                        positions=(),
                        chain=receptor_chain,
                        frame="designed",
                    )
                rev_id = f"rev_{parent_design}_s{parent_seed}"
                reversion[rev_id] = StageResult(
                    stage_type="reversion",
                    config_id=f"{mpnn_sequence_id}__{rev_id}",
                    predictions=tuple(preds),
                    confidences=tuple(confs),
                    seed_indices=tuple(seeds),
                    mutated_positions=mut_positions,
                    applies_to_designs=(parent_design,),
                )

        # ── Optional run_one_runtime_sec sidecar ────────────────────
        runtime_sidecar = workdir / "run_one_runtime_sec.txt"
        runtime_sec: Optional[float] = None
        if runtime_sidecar.is_file():
            try:
                runtime_sec = float(runtime_sidecar.read_text().strip())
            except ValueError:
                runtime_sec = None

        return cls(
            mpnn_sequence_id=mpnn_sequence_id,
            workdir=workdir,
            row_type=row_type,
            cold_start=cold_start_stage,
            steered=steered,
            reversion=reversion,
            truth=truth_psp,
            contamination_positions=contamination_positions,
            run_one_runtime_sec=runtime_sec,
        )

    # ── Reversion mapping (per spec §2.9 + Step 4.3 addition) ──────

    def _reversion_for_design(self, design_id: str) -> Optional[StageResult]:
        """Walk reversion stages, return the one whose
        ``applies_to_designs`` contains ``design_id``, or None."""
        for st in self.reversion.values():
            if st.applies_to_designs and design_id in st.applies_to_designs:
                return st
        return None

    # ── Per-seed cross-stage aggregation ────────────────────────────

    def per_seed_final_verdicts(self, thresholds) -> List[Dict]:
        """Final verdict per (design, seed) across all stages.

        Rule (per spec §2.9):
        - If steering didn't run (no `steered` entries), use the
          cold-start stage's per-seed verdicts.  This is the cold-start
          path; outcome will be ``no_reversion``.
        - Otherwise, for each (design, seed) in `steered`:
            - If reversion ran for that design (matched by
              applies_to_designs), use the matching reversion verdict
              (paired by seed_index — current convention).
            - Otherwise, use the steered verdict.

        Returns list of dicts: ``{design_id, seed_index, stage_run,
        verdict}``.
        """
        if self._truth_required():
            self._ensure_truth()

        # Cold-start path
        if not self.steered:
            verdicts = self.cold_start.per_seed_verdicts(thresholds, self.truth)
            return [
                {
                    "design_id": "initial",
                    "seed_index": si,
                    "stage_run": "cold_start",
                    "verdict": v,
                }
                for si, v in zip(self.cold_start.seed_indices, verdicts)
            ]

        # Normal path: steered (and possibly reversion)
        out: List[Dict] = []
        for design_id, steered_stage in self.steered.items():
            rev_stage = self._reversion_for_design(design_id)
            if rev_stage is not None:
                # Pair by seed_index — current convention
                rev_verdicts = rev_stage.per_seed_verdicts(
                    thresholds, self.truth, self.contamination_positions,
                )
                rev_by_seed = dict(zip(rev_stage.seed_indices, rev_verdicts))
                for si in steered_stage.seed_indices:
                    if si in rev_by_seed:
                        out.append({
                            "design_id": design_id,
                            "seed_index": si,
                            "stage_run": "reverted",
                            "verdict": rev_by_seed[si],
                        })
                    else:
                        # Reversion missing this seed — keep the steered verdict
                        steered_verdicts = steered_stage.per_seed_verdicts(
                            thresholds, self.truth, self.contamination_positions,
                        )
                        st_by_seed = dict(
                            zip(steered_stage.seed_indices, steered_verdicts)
                        )
                        out.append({
                            "design_id": design_id,
                            "seed_index": si,
                            "stage_run": "steered",
                            "verdict": st_by_seed.get(si, "no_data"),
                        })
            else:
                # No reversion → steered verdict stands
                steered_verdicts = steered_stage.per_seed_verdicts(
                    thresholds, self.truth, self.contamination_positions,
                )
                for si, v in zip(steered_stage.seed_indices, steered_verdicts):
                    out.append({
                        "design_id": design_id,
                        "seed_index": si,
                        "stage_run": "steered",
                        "verdict": v,
                    })
        return out

    def _truth_required(self) -> bool:
        # Truth always needed for verdicts.
        return True

    def _ensure_truth(self) -> None:
        if self.truth is None:
            raise ValueError(
                "NegativeSteeringRun verdict methods require a 'truth' "
                "ProteinStructurePrediction at construction"
            )

    # ── Outcome + n_pass + tier ─────────────────────────────────────

    PASS_EQUIVALENT_VERDICTS = frozenset(
        ("clean", "clean_steered", "pose_holds")
    )

    def n_pass(self, thresholds) -> int:
        """Count seeds whose final verdict is pass-equivalent."""
        verdicts = self.per_seed_final_verdicts(thresholds)
        return sum(
            1 for e in verdicts if e["verdict"] in self.PASS_EQUIVALENT_VERDICTS
        )

    def n_seeds_total(self) -> int:
        """Total seed count for this MPNN sequence.

        For the cold-start path, this is the cold-start stage's
        num_seeds.  For the steered path, it's `n_designs × num_seeds`
        summed across all steered designs (matches the existing per-
        design × per-seed accounting in the codebase).
        """
        if not self.steered:
            return self.cold_start.num_seeds
        return sum(st.num_seeds for st in self.steered.values())

    def outcome(self, thresholds) -> str:
        """Aggregate outcome label per spec §2.9.

        Returns one of: ``no_reversion`` | ``pose_holds`` |
        ``pose_collapses`` | ``new_contamination`` | ``singleton``.

        Logic mirrors the existing `_classify_outcome` rules with the
        post-rename labels.
        """
        if not self.steered:
            # No steering ran — cold-start path
            return "no_reversion"
        # Did any reversion run?
        if not self.reversion:
            return "no_reversion"  # steering ran, no contamination triggered reversion
        verdicts = self.per_seed_final_verdicts(thresholds)
        counts = Counter(e["verdict"] for e in verdicts)
        nph = counts.get("pose_holds", 0)
        npc = counts.get("pose_collapses", 0)
        nnc = counts.get("new_contamination", 0)
        ncs = counts.get("clean_steered", 0)
        # Per the existing logic: if any pose_holds AND failure modes
        # don't dominate, return pose_holds.  Otherwise pick by
        # plurality between failure modes.
        nph_eff = nph + ncs
        if nph_eff > 0 and nph_eff >= max(npc, nnc):
            return "pose_holds"
        if nnc > npc:
            return "new_contamination"
        return "pose_collapses"

    def outcome_reason(self, thresholds) -> str:
        """Short diagnostic explanation of the outcome decision."""
        if not self.steered:
            return "cold-start path: no steering ran"
        if not self.reversion:
            return ("steering ran but no design triggered reversion under "
                    "the CL-3 majority rule")
        verdicts = self.per_seed_final_verdicts(thresholds)
        counts = Counter(e["verdict"] for e in verdicts)
        parts = [f"{k}={v}" for k, v in sorted(counts.items())]
        return "; ".join(parts)

    def tier(self, thresholds) -> str:
        """Tier classification (A / B / C / none) from n_pass / n_seeds.

        Per the post-rename rule (no `outcome=='no_reversion'` shortcut):
            A: n_pass == n_seeds (e.g. 3/3)
            B: 1 < n_pass < n_seeds (e.g. 2/3)
            C: n_pass == 1
            none: 0
        """
        n_pass = self.n_pass(thresholds)
        n_seeds = self.n_seeds_total()
        if n_seeds <= 0:
            return "none"
        if n_pass == n_seeds:
            return "A"
        if 1 < n_pass < n_seeds:
            return "B"
        if n_pass == 1:
            return "C"
        return "none"

    # ── Identification / passthrough ────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"NegativeSteeringRun(mpnn_sequence_id="
            f"{self.mpnn_sequence_id!r}, row_type={self.row_type!r}, "
            f"n_steered_designs={len(self.steered)}, "
            f"n_reversions={len(self.reversion)})"
        )
