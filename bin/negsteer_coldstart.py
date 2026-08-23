"""Cold-start selection: score the unsteered prediction and pick what survives.

Three stages, run in this order by bin/negative_steering_run_one.sh:
kickoff-distances, kickoff-prefilter, kickoff-finalize. Split out of
boltz2_iterate_steering.py, which no longer exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Single source of truth for the reverted-confidence column set, defined in
# reversion.py next to the per_seed_records dict it must stay in sync with.
from reversion import _REVERTED_CONFIDENCE_FIELDS  # noqa: E402

# Receptor-fold "intact" threshold, with a literal fallback so this module stays
# importable where the thresholds catalogue is absent.
try:  # noqa: E402
    from pipeline_thresholds import PipelineInternalThresholds as _PIT
    _INTACT_THRESHOLD = _PIT.default().intact_threshold
except Exception:
    _INTACT_THRESHOLD = 5.0

# Lazy import of boltz2_negative_steering — it transitively imports
# numpy and Bio.Align which are only present inside the singularity
# container.  Phase-2 entry points (iterate-collect, kickoff,
# aggregate) must NOT trigger this import.
_BNS_MODULE = None


def _lazy_bns():
    global _BNS_MODULE
    if _BNS_MODULE is None:
        sys.path.insert(0, str(SCRIPT_DIR))
        import boltz2_negative_steering as bns
        _BNS_MODULE = bns
    return _BNS_MODULE


# ───────────────────────────────────────────────────────────────────────
# Novelty + maximin selection
# ───────────────────────────────────────────────────────────────────────
def compute_pose_distance(candidate_pdb: Path, reference_pdb: Path,
                          pred_rec_chain: str, pred_eff_chain: str) -> float:
    """receptor_aligned_effector_rmsd between two predicted complexes
    that share both protein sequences and chain naming.  Aligns the
    receptors via Kabsch then measures the effector displacement —
    same machinery as binding_rmsds, just both arguments are
    predictions instead of one being a ground truth.

    Container-only — calls into boltz2_negative_steering which needs
    numpy and Bio.Align.
    """
    bns = _lazy_bns()
    result = bns.compute_binding_rmsds(
        str(candidate_pdb), str(reference_pdb),
        rec_chain=pred_rec_chain, eff_chain=pred_eff_chain,
        des_rec_chain=pred_rec_chain, des_eff_chain=pred_eff_chain,
    )
    v = result.get("receptor_aligned_effector_rmsd")
    return float(v) if v is not None else float("nan")


def maximin_select(candidates: List[Dict], k: int) -> List[Dict]:
    """Select up to k candidates from `candidates` by max-min distance
    to the upstream pose set.  Each candidate dict must have a key
    'min_dist_to_upstream' (float).  Picks the k with the largest
    such value, ties broken by lower ra_eff vs ground truth.

    For the tie-break, prefers `_active_ra_eff_for_selection` if
    present (set by finalize on `pose_holds` rows so the selector
    sees the reverted ra_eff), falling back to
    `receptor_aligned_effector_rmsd_vs_truth` (the steered value).
    Steered base columns are never overwritten — the dedicated
    `_active_*` keys carry the post-reversion view used for selection.
    """
    def _ra_eff(c: Dict) -> float:
        v = c.get("_active_ra_eff_for_selection")
        if v is None:
            v = c.get("receptor_aligned_effector_rmsd_vs_truth", float("inf"))
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("inf")
    sorted_c = sorted(
        candidates,
        key=lambda c: (-c["min_dist_to_upstream"], _ra_eff(c)),
    )
    return sorted_c[:k]


# ───────────────────────────────────────────────────────────────────────
# Statistics tracking
# ───────────────────────────────────────────────────────────────────────
def append_cycle_statistics(experiment_root: Path,
                            cycle: int,
                            pathway_label: str,
                            n_generated: int,
                            n_intact: int,
                            n_novel: int,
                            n_selected: int,
                            min_dists: List[float],
                            ra_eff_vs_truth: List[float]):
    """Append one row per (cycle, pathway) to cycle_statistics.csv.
    The file is created on first write.

    Stats include the full list of min_dists and ra_eff_vs_truth as
    semicolon-joined strings so downstream analysis can fit
    distributions / draw bell curves without re-walking the tree.
    """
    stats_path = experiment_root / "cycle_statistics.csv"
    is_new = not stats_path.exists()

    with open(stats_path, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow([
                "cycle", "pathway", "n_generated", "n_intact",
                "n_novel", "n_selected",
                "min_dist_to_upstream_values",
                "receptor_aligned_effector_rmsd_vs_truth_values",
            ])
        w.writerow([
            cycle, pathway_label, n_generated, n_intact,
            n_novel, n_selected,
            ";".join(f"{v:.3f}" for v in min_dists),
            ";".join(f"{v:.3f}" for v in ra_eff_vs_truth),
        ])


def _is_nan(x):
    """Tiny helper that handles either numpy NaN or Python NaN
    without forcing a numpy import on the caller."""
    try:
        return math.isnan(x)
    except TypeError:
        return False


def _copy_reverted_confidence(dst: Dict, rev: Dict) -> None:
    """Copy every reverted confidence metric from `rev` (a single
    entry from reversion_results.json) into the candidate dict `dst`
    that will be written into passing.json.  Missing keys are
    skipped so the column stays blank on rows that don't have
    that metric (e.g. chai predictions that don't emit actifptm)."""
    for k in _REVERTED_CONFIDENCE_FIELDS:
        v = rev.get(k)
        if v is not None:
            dst[k] = v


def cmd_kickoff_distances(args: argparse.Namespace) -> int:
    """Phase 1 of kickoff — compute pose distances for every cycle-0
    design vs the wild-type prediction.  Writes cycle_0/kickoff_distances.json.

    Container-only — uses compute_pose_distance which needs numpy and
    Bio.Align via boltz2_negative_steering.

    Phase 2 (cmd_kickoff) reads kickoff_distances.json and runs the
    filter/maximin/sbatch logic in pure stdlib outside the container.
    """
    experiment_root = args.experiment_root.resolve()
    cycle0_dir = experiment_root / "cycle_0"
    if not cycle0_dir.is_dir():
        print(f"ERROR: no cycle_0 under {experiment_root}", file=sys.stderr)
        return 2

    plan_path = cycle0_dir / "plan.json"
    if not plan_path.exists():
        print(f"ERROR: no plan.json at {plan_path}", file=sys.stderr)
        return 2
    plan = json.loads(plan_path.read_text())

    if plan.get("skip_steering"):
        print("[kickoff-distances] cycle 0 skipped steering — writing empty distances")
        (cycle0_dir / "kickoff_distances.json").write_text(json.dumps({
            "skip_steering": True,
            "n_generated": 0,
            "candidates": [],
        }, indent=2))
        return 0

    pred_rec_chain = plan["pred_receptor_chain"]
    pred_eff_chain = plan["pred_effector_chain"]
    wild_type_pdb = cycle0_dir / "initial_prediction.pdb"
    if not wild_type_pdb.exists():
        print(f"ERROR: missing {wild_type_pdb}", file=sys.stderr)
        return 2

    candidates: List[Dict] = []
    for design in plan["designs"]:
        d = Path(design["dir"])
        rj = d / "result.json"
        if not rj.exists():
            continue
        r = json.loads(rj.read_text())
        if r.get("status") != "ok":
            continue
        ra_eff_truth = r.get("receptor_aligned_effector_rmsd")
        ind_rec      = r.get("independent_receptor_rmsd")
        ind_eff      = r.get("independent_effector_rmsd")
        if ra_eff_truth is None or ind_rec is None or ind_eff is None:
            continue

        candidate_pdb = d / "prediction.pdb"
        try:
            dist = compute_pose_distance(
                candidate_pdb, wild_type_pdb,
                pred_rec_chain, pred_eff_chain,
            )
        except Exception as e:
            print(f"  WARN: pose distance failed for {design['name']}: {e}",
                  file=sys.stderr)
            dist = None
        # Use None instead of NaN for JSON portability
        dist_val = None if (dist is None or _is_nan(dist)) else float(dist)

        candidates.append({
            "design": r["design"],
            "design_idx": design["index"],
            # P0 audit (multi-seed): carry sequence_group + seed_index
            # from the plan into the candidate dict so per-sequence
            # aggregation downstream can group on them.
            "sequence_group": design.get("sequence_group", ""),
            "seed_index": design.get("seed_index", ""),
            "pdb": str(candidate_pdb),
            "receptor_aligned_effector_rmsd_vs_truth": float(ra_eff_truth),
            "independent_receptor_rmsd": float(ind_rec),
            "independent_effector_rmsd": float(ind_eff),
            "dists_to_upstream": [dist_val],
            "min_dist_to_upstream": dist_val,
            # Carry the full result.json through so the aggregator
            # can surface every per-design metric in cycle_0/passing.json
            # alongside the kickoff distance/selection info.
            "result": r,
        })

    doc = {
        "n_generated": len(plan["designs"]),
        "candidates": candidates,
        "wild_type_pdb": str(wild_type_pdb),
    }
    (cycle0_dir / "kickoff_distances.json").write_text(json.dumps(doc, indent=2))
    print(f"[kickoff-distances] wrote {len(candidates)} candidate(s) to kickoff_distances.json")
    return 0


# ───────────────────────────────────────────────────────────────────────
# Subcommand: kickoff-prefilter  (host, cycle-0 phase 2a)
# ───────────────────────────────────────────────────────────────────────
#
# Cycle-0 equivalent of iterate-collect-prefilter.  Reads
# kickoff_distances.json, applies the intact filter, writes
# cycle_0/prefilter.json.  Does not run novelty/maximin/resubmit —
# those happen in kickoff-finalize after the reversion chain runs.
def cmd_kickoff_prefilter(args: argparse.Namespace) -> int:
    experiment_root = args.experiment_root.resolve()
    cycle0_dir = experiment_root / "cycle_0"
    if not cycle0_dir.is_dir():
        print(f"ERROR: no cycle_0 under {experiment_root}", file=sys.stderr)
        return 2

    distances_path = cycle0_dir / "kickoff_distances.json"
    if not distances_path.exists():
        msg = (f"ERROR: {distances_path} does not exist — phase 1 "
               f"(kickoff-distances) probably failed.\n")
        sys.stderr.write(msg)
        (cycle0_dir / "prefilter.json").write_text(json.dumps({"error": msg}))
        return 1

    distances = json.loads(distances_path.read_text())
    n_generated = int(distances.get("n_generated", 0))

    if distances.get("skip_steering"):
        print("[kickoff-prefilter] cycle 0 skipped steering — "
              "writing empty prefilter")
        (cycle0_dir / "prefilter.json").write_text(json.dumps({
            "cycle": 0,
            "parent_pathway": None,
            "skip_steering": True,
            "exhausted": True,
            "n_generated": 0,
            "n_intact": 0,
            "intact_candidates": [],
        }, indent=2))
        return 0

    intact_candidates: List[Dict] = []
    for c in distances.get("candidates", []):
        intact = (
            c["independent_receptor_rmsd"] <= _INTACT_THRESHOLD
            and c["independent_effector_rmsd"] <= _INTACT_THRESHOLD
        )
        c2 = dict(c)
        c2["receptor_intact"] = intact
        if intact:
            intact_candidates.append(c2)

    print(f"[kickoff-prefilter] cycle 0: {n_generated} generated -> "
          f"{len(intact_candidates)} intact")

    (cycle0_dir / "prefilter.json").write_text(json.dumps({
        "cycle": 0,
        "parent_pathway": None,
        "n_generated": n_generated,
        "n_intact": len(intact_candidates),
        "intact_candidates": intact_candidates,
    }, indent=2, default=str))
    return 0


# ───────────────────────────────────────────────────────────────────────
# Subcommand: kickoff-finalize  (host, cycle-0 phase 2c)
# ───────────────────────────────────────────────────────────────────────
#
# Cycle-0 equivalent of iterate-collect-finalize.  Reads
# cycle_0/prefilter.json + (optionally) cycle_0/reversion_results.json,
# routes verdicts, re-applies structural filter on reverted designs,
# merges pose_holds back into the intact list with their reverted
# sequence rewritten in place, runs novelty/maximin, writes
# cycle_0/passing.json, and submits cycle-1 chains.
def cmd_kickoff_finalize(args: argparse.Namespace) -> int:
    experiment_root = args.experiment_root.resolve()
    cycle0_dir = experiment_root / "cycle_0"
    if not cycle0_dir.is_dir():
        print(f"ERROR: no cycle_0 under {experiment_root}", file=sys.stderr)
        return 2

    prefilter_path = cycle0_dir / "prefilter.json"
    if not prefilter_path.exists():
        print(f"ERROR: {prefilter_path} not found — run "
              f"kickoff-prefilter first", file=sys.stderr)
        return 1
    prefilter = json.loads(prefilter_path.read_text())

    novelty_cutoff = float(args.novelty_cutoff)
    max_passing = int(args.max_passing)
    max_cycles = int(args.max_cycles)

    if prefilter.get("exhausted") or prefilter.get("skip_steering"):
        print("cycle 0 exhausted — nothing to iterate.")
        return 0

    n_generated = int(prefilter.get("n_generated", 0))
    intact_candidates: List[Dict] = list(prefilter.get("intact_candidates", []))

    # Load reversion results if any contaminated designs were processed
    reversion_results: Dict[str, Dict] = {}
    reversion_results_path = cycle0_dir / "reversion_results.json"
    if reversion_results_path.exists():
        reversion_results = json.loads(reversion_results_path.read_text())

    # Bug 1 guard: load contaminated.json so we can detect designs that
    # were flagged contaminated by build-contaminated but never received
    # a verdict.  Such designs MUST be dropped — silently keeping them
    # would let unvalidated contaminated designs pass through.
    contaminated_labels: set = set()
    contaminated_path = cycle0_dir / "contaminated.json"
    if contaminated_path.exists():
        cdoc = json.loads(contaminated_path.read_text())
        contaminated_labels = {
            e.get("label") for e in cdoc.get("contaminated", [])
            if e.get("label")
        }
    missing_verdict_labels: set = (
        contaminated_labels - set(reversion_results.keys())
    )
    if missing_verdict_labels:
        print(f"[kickoff-finalize] WARN: "
              f"{len(missing_verdict_labels)} contaminated design(s) have "
              f"no verdict in reversion_results.json — will be dropped: "
              f"{sorted(missing_verdict_labels)}", file=sys.stderr)

    # Route verdicts.  Cycle-0 labels are just the design name (e.g.
    # "design_40") since there's no parent pathway.
    pose_holds_count = 0
    pose_collapses_count = 0
    new_contamination_count = 0
    dropped_labels: List[str] = []
    reverted_labels: List[str] = []
    # Designs that got dropped by verdict routing — surfaced into
    # passing.json's all_candidates with a reversion_verdict marker
    # so the aggregator CSV can render them.  They do NOT flow into
    # novelty/maximin.
    dropped_rows: List[Dict] = []

    def _label_for(c: Dict) -> str:
        return c.get("design", "?")

    kept: List[Dict] = []
    missing_verdict_count = 0
    for c in intact_candidates:
        label = _label_for(c)
        rev = reversion_results.get(label)
        if rev is None:
            # Bug 1 guard: if this label was flagged contaminated but
            # has no verdict, drop it instead of silently keeping it.
            if label in contaminated_labels:
                missing_verdict_count += 1
                dropped_labels.append(label)
                print(f"  [missing_verdict] {label}: dropped — flagged "
                      f"contaminated but no entry in reversion_results.json")
                c_drop = dict(c)
                c_drop["reversion_verdict"] = "missing_verdict"
                c_drop["reversion_dropped"] = 1
                # Steered metrics already in base columns of c_drop;
                # no overlay required (and no reverted prediction exists).
                dropped_rows.append(c_drop)
                continue
            kept.append(c)
            continue

        verdict = rev.get("verdict", "")
        if verdict == "pose_holds":
            pose_holds_count += 1
            reverted_labels.append(label)
            design_workdir = Path(c["pdb"]).parent
            # Locate reverted FASTA from reversion_plan.json
            reverted_fasta = None
            rp_path = cycle0_dir / "reversion_plan.json"
            if rp_path.exists():
                rp = json.loads(rp_path.read_text())
                for entry in rp.get("entries", []):
                    if entry.get("label") == label:
                        reverted_fasta = Path(entry["reverted_receptor_fasta"])
                        break
            if reverted_fasta and reverted_fasta.exists():
                try:
                    seq_lines = reverted_fasta.read_text().splitlines()
                    reverted_seq = "".join(
                        line.strip() for line in seq_lines
                        if line.strip() and not line.startswith(">")
                    )
                    (design_workdir / "receptor.fasta").write_text(
                        f">receptor_reverted\n{reverted_seq}\n"
                    )
                    (design_workdir / "reversion_applied.json").write_text(
                        json.dumps({
                            "label": label,
                            "verdict": verdict,
                            "reverted_sequence": reverted_seq,
                            "reversion_result": rev,
                        }, indent=2, default=str)
                    )
                    print(f"  [pose_holds] {label}: rewrote "
                          f"{design_workdir/'receptor.fasta'}")
                except Exception as e:
                    print(f"  WARN {label}: failed to rewrite "
                          f"receptor.fasta: {e}", file=sys.stderr)
            else:
                print(f"  WARN {label}: reverted FASTA not found",
                      file=sys.stderr)

            # Build the kept candidate dict.  Critical invariant:
            # the base columns hold the steered values; reverted
            # values live only in dedicated reverted_* keys.  For
            # selection, _active_ra_eff_for_selection carries the
            # post-reversion ra_eff so maximin operates on the
            # reverted pose.
            c2 = dict(c)
            ra = rev.get("reverted_ra_eff_vs_truth")
            if ra is not None:
                c2["reverted_ra_eff_vs_truth"] = float(ra)
                c2["_active_ra_eff_for_selection"] = float(ra)
            ind_rec = rev.get("reverted_independent_receptor_rmsd")
            if ind_rec is not None:
                c2["reverted_independent_receptor_rmsd"] = float(ind_rec)
            ind_eff = rev.get("reverted_independent_effector_rmsd")
            if ind_eff is not None:
                c2["reverted_independent_effector_rmsd"] = float(ind_eff)
            _copy_reverted_confidence(c2, rev)
            c2["reversion_verdict"] = verdict
            c2["reverted_sequence_used_as_parent"] = 1
            reverted_intact = (
                (ind_rec is not None and float(ind_rec) <= 5.0)
                and (ind_eff is not None and float(ind_eff) <= 5.0)
            )
            if not reverted_intact:
                print(f"  [pose_holds→drop] {label}: reverted fails "
                      f"intact filter, dropping anyway")
                dropped_labels.append(label)
                c2["reversion_verdict"] = "pose_holds_not_intact"
                c2["reversion_dropped"] = 1
                dropped_rows.append(c2)
                continue
            c2["reversion_dropped"] = 0
            kept.append(c2)
        elif verdict == "pose_collapses":
            pose_collapses_count += 1
            dropped_labels.append(label)
            print(f"  [pose_collapses] {label}: {rev.get('reason','')}")
            c_drop = dict(c)
            c_drop["reversion_verdict"] = verdict
            c_drop["reversion_dropped"] = 1
            # Steered metrics already in base columns of c_drop.
            # Overlay reverted metrics so the CSV shows what the
            # reverted prediction looked like.
            if rev.get("reverted_ra_eff_vs_truth") is not None:
                c_drop["reverted_ra_eff_vs_truth"] = float(rev["reverted_ra_eff_vs_truth"])
            if rev.get("reverted_independent_receptor_rmsd") is not None:
                c_drop["reverted_independent_receptor_rmsd"] = float(rev["reverted_independent_receptor_rmsd"])
            if rev.get("reverted_independent_effector_rmsd") is not None:
                c_drop["reverted_independent_effector_rmsd"] = float(rev["reverted_independent_effector_rmsd"])
            _copy_reverted_confidence(c_drop, rev)
            dropped_rows.append(c_drop)
        elif verdict == "new_contamination":
            new_contamination_count += 1
            dropped_labels.append(label)
            print(f"  [new_contamination] {label}: {rev.get('reason','')}")
            c_drop = dict(c)
            c_drop["reversion_verdict"] = verdict
            c_drop["reversion_dropped"] = 1
            if rev.get("reverted_ra_eff_vs_truth") is not None:
                c_drop["reverted_ra_eff_vs_truth"] = float(rev["reverted_ra_eff_vs_truth"])
            if rev.get("reverted_independent_receptor_rmsd") is not None:
                c_drop["reverted_independent_receptor_rmsd"] = float(rev["reverted_independent_receptor_rmsd"])
            if rev.get("reverted_independent_effector_rmsd") is not None:
                c_drop["reverted_independent_effector_rmsd"] = float(rev["reverted_independent_effector_rmsd"])
            _copy_reverted_confidence(c_drop, rev)
            dropped_rows.append(c_drop)
        else:
            dropped_labels.append(label)
            print(f"  WARN {label}: unknown verdict {verdict!r}, dropping")
            c_drop = dict(c)
            c_drop["reversion_verdict"] = verdict or "unknown"
            c_drop["reversion_dropped"] = 1
            dropped_rows.append(c_drop)

    intact_candidates = kept

    novel = [
        c for c in intact_candidates
        if c.get("min_dist_to_upstream") is not None
        and c["min_dist_to_upstream"] >= novelty_cutoff
    ]
    n_intact = len(intact_candidates)
    n_novel = len(novel)
    selected = maximin_select(novel, max_passing)
    n_selected = len(selected)

    print(f"[kickoff-finalize] cycle 0: {n_generated} designs -> "
          f"{n_intact} intact (after verdicts) -> {n_novel} novel -> "
          f"{n_selected} selected for cycle 1")
    print(f"  verdicts: pose_holds={pose_holds_count} "
          f"pose_collapses={pose_collapses_count} "
          f"new_contamination={new_contamination_count}")

    (cycle0_dir / "passing.json").write_text(json.dumps({
        "cycle": 0,
        "parent_pathway": None,
        "novelty_cutoff": novelty_cutoff,
        "max_passing": max_passing,
        "n_generated": n_generated,
        "n_intact": n_intact,
        "n_novel": n_novel,
        "n_selected": n_selected,
        "pose_holds_count": pose_holds_count,
        "pose_collapses_count": pose_collapses_count,
        "new_contamination_count": new_contamination_count,
        "reverted_labels": reverted_labels,
        "dropped_labels": dropped_labels,
        "all_candidates": intact_candidates + dropped_rows,
        "selected": selected,
    }, indent=2, default=str))

    append_cycle_statistics(
        experiment_root, 0, "cycle_0",
        n_generated=n_generated,
        n_intact=n_intact, n_novel=n_novel, n_selected=n_selected,
        min_dists=[c["min_dist_to_upstream"] for c in selected],
        ra_eff_vs_truth=[c["receptor_aligned_effector_rmsd_vs_truth"] for c in selected],
    )

    if n_selected == 0:
        print("  no qualifying designs at cycle 0 — nothing to iterate.")
        return 0
    # Follow-up cycles were removed with the rest of the multi-cycle harness
    # (see archive/boltz2_iterate_steering_multicycle.py). This stage finalises
    # cycle 0 and stops. --max-cycles is retained because callers pass it, but
    # any value now behaves as 0.
    if max_cycles >= 1:
        print(f"  max-cycles={max_cycles}: follow-up cycles are not implemented; "
              "finalising cycle 0 only.", file=sys.stderr)
    return 0



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pkd = sub.add_parser("kickoff-distances",
                         help="Phase 1 (in container) of kickoff: compute "
                              "pose distances for every cycle-0 design vs "
                              "the wild-type prediction")
    pkd.add_argument("--experiment-root", type=Path, required=True)
    pkd.set_defaults(func=cmd_kickoff_distances)

    # ─── kickoff-prefilter (host, cycle-0 phase 2a) ─────────────────
    pkp = sub.add_parser(
        "kickoff-prefilter",
        help="Host, cycle-0 phase 2a: apply intact filter to "
             "kickoff_distances.json, write cycle_0/prefilter.json",
    )
    pkp.add_argument("--experiment-root", type=Path, required=True)
    pkp.set_defaults(func=cmd_kickoff_prefilter)

    # ─── kickoff-finalize (host, cycle-0 phase 2c) ──────────────────
    pkf = sub.add_parser(
        "kickoff-finalize",
        help="Host, cycle-0 phase 2c: read cycle_0/prefilter.json + "
             "(optional) cycle_0/reversion_results.json, route "
             "verdicts, run novelty/maximin, write cycle_0/passing.json, "
             "submit cycle-1 chains",
    )
    pkf.add_argument("--experiment-root", type=Path, required=True)
    pkf.add_argument("--max-cycles", type=int, required=True)
    pkf.add_argument("--max-passing", type=int, default=5)
    pkf.add_argument("--novelty-cutoff", type=float, default=10.0)
    pkf.add_argument("--n-designs", type=int, default=None)
    pkf.set_defaults(func=cmd_kickoff_finalize)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
