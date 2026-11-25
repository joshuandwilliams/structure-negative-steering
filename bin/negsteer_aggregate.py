"""Aggregation: per-seed rows to the per-sequence tables the analyses read.

Three stages, run in this order by bin/negative_steering_run_one.sh:
aggregate, compute-final-metrics, aggregate-per-sequence. Split out of
boltz2_iterate_steering.py, which no longer exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pathway_labels import (  # noqa: E402
    _locate_boltz_sidecar_pdb,
    read_mutations_tsv_full,
)
from reversion import _REVERTED_CONFIDENCE_FIELDS  # noqa: E402

try:  # noqa: E402
    from pipeline_thresholds import PipelineInternalThresholds as _PIT
    _INTACT_THRESHOLD = _PIT.default().intact_threshold
except Exception:
    _INTACT_THRESHOLD = 5.0


# Internal key -> CSV column name.  Keys absent here pass through
# unchanged.
_AGG_RENAME = {
    # Steering RMSDs
    "receptor_aligned_effector_rmsd":  "steered_ra_eff_vs_truth",
    "independent_receptor_rmsd":       "steered_independent_receptor_rmsd",
    "independent_effector_rmsd":       "steered_independent_effector_rmsd",
    "receptor_intact":                 "steered_receptor_intact",
    # Steering metadata + interface
    "total_mutations":                 "steered_total_mutations",
    "mutations_chimerax":              "steered_mutations_chimerax",
    "mutations_aa":                    "steered_mutations_aa",
    "wrong_jaccard":                   "steered_wrong_jaccard",
    "n_shared_wrong":                  "steered_n_shared_wrong",
    "true_jaccard":                    "steered_true_jaccard",
    "n_shared_true":                   "steered_n_shared_true",
    "n_design_interface_residues":     "steered_n_design_interface_residues",
}


# Keys dropped at write time, each a duplicate or superseded.
_AGG_DROP = {
    "delta_receptor_aligned_effector_rmsd",  # trivially derivable
    "rank",                                  # legacy cycle-0 row index
    "rank_by_truth",                         # superseded by rank_by_ra_eff
    # Written under these names back when finalize overwrote the base
    # columns.  It no longer does, so drop them if present.
    "steered_ra_eff_vs_truth",
    "steered_independent_receptor_rmsd",
    "steered_independent_effector_rmsd",
    # Internal selection-only key, never user-facing
    "_active_ra_eff_for_selection",
}


# Aggregated by median+MAD.  Mirrors the Block-B/C continuous columns
# of _aggregate_csv_fieldnames.
_AGG_CONTINUOUS_METRICS = (
    "ra_eff_vs_truth",
    "independent_receptor_rmsd",
    "independent_effector_rmsd",
    "wrong_jaccard",
    "true_jaccard",
    "avg_plddt",
    "complex_plddt",
    "ptm",
    "iptm",
    "pae_mean",
    "ipae",
    "pae_pass_frac",
    "ipsae_ab",
    "ipsae_ba",
    "ipsae_min",
    # 15Å iPSAE — same machinery, more permissive PAE cutoff.
    "ipsae_ab_15",
    "ipsae_ba_15",
    "ipsae_min_15",
    "actifptm",
    # These two need the ground-truth PDB.  Moved here from
    # compute_interface_metrics.py and run_biophysical_metrics.py so
    # they aggregate alongside the Boltz-derived metrics.
    "interface_plddt",
    "weighted_jaccard",
)


# 0/1 categorical metrics aggregated by majority.
_AGG_BINARY_METRICS = (
    "receptor_intact",
    # intact_core: per-prediction Kabsch-aligned receptor-core RMSD
    # under threshold.
    "intact_core",
)


# Integer counts.  Median + MAD too, since they are discrete but
# ordered.
_AGG_INTEGER_METRICS = (
    "total_mutations",
    "n_shared_wrong",
    "n_shared_true",
    "n_design_interface_residues",
    "n_contact_residues",
    "n_contacts_on_mutated_positions",
)


# Position-set columns (CSV strings of 1-based positions) aggregated
# by per-position majority.
_AGG_POSITION_SET_METRICS = (
    "mutated_contact_positions",
    "contact_residues",
)


def format_mutation_strings(
    mutations: List[Tuple[int, str, str]],
    chain_id: str = "A",
) -> Tuple[str, str]:
    """Build (chimerax_spec, aa_changes) from a list of (pos1, wt, mut)
    tuples.

      chimerax_spec   "/A:5,12,37", sorted and deduplicated.  The user
                      prepends their own "#N" model number.
      aa_changes      "L5A V12K G37P", sorted by position.

    An empty mutation list gives ("", "").
    """
    if not mutations:
        return ("", "")
    # A position appears twice when one edit landed on top of another.
    # The chimerax spec keeps the earliest edit.  The aa-changes string
    # keeps every entry, because re-edits are scientifically meaningful
    # even though burnt-positions logic usually prevents them.
    seen = set()
    unique_positions = []
    for pos, _wt, _mut in mutations:
        if pos in seen:
            continue
        seen.add(pos)
        unique_positions.append(pos)
    unique_positions.sort()
    chimerax_spec = f"/{chain_id}:" + ",".join(str(p) for p in unique_positions)

    aa_changes_sorted = sorted(mutations, key=lambda t: t[0])
    aa_changes = " ".join(f"{wt}{pos}{mut}" for pos, wt, mut in aa_changes_sorted)

    return (chimerax_spec, aa_changes)


def _translate_aggregate_row(r: Dict) -> Dict:
    """Apply _AGG_RENAME and _AGG_DROP to produce a row dict ready for CSV
    writing, and derive reverted_receptor_intact from the two reverted
    RMSDs. A pure dict transformation that never reads another row.
    """
    out: Dict = {}
    # Drop the keys that never appear in output
    for k, v in r.items():
        if k in _AGG_DROP:
            continue
        out_key = _AGG_RENAME.get(k, k)
        out[out_key] = v

    # Derive reverted_receptor_intact from reverted RMSDs (if present)
    rev_rec = out.get("reverted_independent_receptor_rmsd", "")
    rev_eff = out.get("reverted_independent_effector_rmsd", "")
    if rev_rec != "" and rev_eff != "":
        try:
            out["reverted_receptor_intact"] = (
                1 if (float(rev_rec) <= 5.0 and float(rev_eff) <= 5.0) else 0
            )
        except (ValueError, TypeError):
            out["reverted_receptor_intact"] = ""
    else:
        out["reverted_receptor_intact"] = ""

    return out


def _populate_reverted_mutations(rows: List[Dict],
                                 experiment_root: Path) -> None:
    """For pose_holds rows, compute the post-reversion mutation set
    (cumulative mutations minus reverted positions) and write
    reverted_total_mutations / reverted_mutations_chimerax /
    reverted_mutations_aa.  All other rows leave these blank.

    Reads cycle_N/(pathway_<parent>/)?reversion_results.json keyed
    by pathway label to get positions_to_revert.  Reads cycle_0/
    plan.json once for pred_receptor_chain.
    """
    # Resolve pred_receptor_chain and contact_cutoff from cycle_0 plan
    pred_rec_chain = "A"
    cycle0_contact_cutoff = ""
    cycle0_plan_path = experiment_root / "cycle_0" / "plan.json"
    if cycle0_plan_path.exists():
        try:
            cp = json.loads(cycle0_plan_path.read_text())
            pred_rec_chain = cp.get("pred_receptor_chain", "A")
            cycle0_contact_cutoff = str(cp.get("contact_cutoff", ""))
        except Exception:
            pass

    # Cache reversion_results.json reads by workdir
    rev_cache: Dict[Path, Dict] = {}

    def _load_rev_results(cycle: int, parent_pathway: str) -> Dict:
        if cycle == 0:
            wd = experiment_root / "cycle_0"
        else:
            wd = (experiment_root / f"cycle_{cycle}"
                  / f"pathway_{parent_pathway}")
        if wd in rev_cache:
            return rev_cache[wd]
        rr_path = wd / "reversion_results.json"
        data: Dict = {}
        if rr_path.exists():
            try:
                data = json.loads(rr_path.read_text())
            except Exception:
                data = {}
        rev_cache[wd] = data
        return data

    # Cache contaminated.json reads by workdir
    contam_cache: Dict[Path, Dict] = {}

    def _load_contaminated_doc(cycle: int, parent_pathway: str) -> Dict:
        if cycle == 0:
            wd = experiment_root / "cycle_0"
        else:
            wd = (experiment_root / f"cycle_{cycle}"
                  / f"pathway_{parent_pathway}")
        if wd in contam_cache:
            return contam_cache[wd]
        path = wd / "contaminated.json"
        data: Dict = {}
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except Exception:
                data = {}
        contam_cache[wd] = data
        return data

    # Cache reversion_plan.json reads by workdir (for positions_to_revert)
    rev_plan_cache: Dict[Path, Dict] = {}

    def _load_rev_plan(cycle: int, parent_pathway: str) -> Dict:
        if cycle == 0:
            wd = experiment_root / "cycle_0"
        else:
            wd = (experiment_root / f"cycle_{cycle}"
                  / f"pathway_{parent_pathway}")
        if wd in rev_plan_cache:
            return rev_plan_cache[wd]
        rp_path = wd / "reversion_plan.json"
        data: Dict = {}
        if rp_path.exists():
            try:
                data = json.loads(rp_path.read_text())
            except Exception:
                data = {}
        rev_plan_cache[wd] = data
        return data

    for r in rows:
        # Default blank
        r.setdefault("reverted_total_mutations", "")
        r.setdefault("reverted_mutations_chimerax", "")
        r.setdefault("reverted_mutations_aa", "")
        r.setdefault("steered_mutated_contact_positions", "")
        r.setdefault("reverted_mutated_contact_positions", "")

        verdict = r.get("reversion_verdict") or ""
        cycle = int(r.get("cycle", 0) or 0)
        parent_pathway = r.get("parent_pathway", "") or ""

        # The label keying contaminated.json and
        # reversion_results.json.  Cycle-0 rows all have
        # pathway="cycle_0", so the identifier comes from the "design"
        # column, such as "design_04".  Cycle-1+ rows set pathway to a
        # leaf label like "c0d05.c1d03".
        if cycle == 0:
            leaf_label = r.get("design", "")
        else:
            leaf_label = r.get("pathway", "")

        # Every reverted row, pose_holds or dropped, gets the mutated
        # positions that contacted the effector in the steered
        # prediction.  That is the contamination that triggered it.
        if verdict != "":
            cdoc = _load_contaminated_doc(cycle, parent_pathway)
            for e in cdoc.get("contaminated", []):
                if e.get("label") == leaf_label:
                    pos = e.get("positions_to_revert") or []
                    r["steered_mutated_contact_positions"] = ",".join(
                        str(p) for p in sorted(pos)
                    )
                    break

        # Reverted contact data for every verdict, not just pose_holds.
        # The per-position majority needs each seed's real
        # contamination set.  An earlier version blanked these columns
        # on non-pose_holds rows, so the aggregator saw empty contact
        # sets and called the outcome pose_holds when some seeds had
        # said new_contamination.
        if verdict != "":
            rev_data = _load_rev_results(cycle, parent_pathway)
            rev_entry = rev_data.get(leaf_label) or {}
            rmcp = rev_entry.get("reverted_mutated_contact_positions") or []
            r["reverted_mutated_contact_positions"] = ",".join(
                str(p) for p in sorted(rmcp)
            )
            r["reverted_n_contact_residues"] = rev_entry.get(
                "reverted_n_contact_residues", "")
            r["reverted_contact_residues"] = rev_entry.get(
                "reverted_contact_residues", "")
            r["reverted_contact_cutoff_used"] = cycle0_contact_cutoff

        if verdict != "pose_holds":
            continue

        # pose_holds: reverted_mutations_* from the remaining mutations.
        rev_data = _load_rev_results(cycle, parent_pathway)
        rev_entry = rev_data.get(leaf_label) or {}

        # contaminated.json is authoritative for positions_to_revert.
        # reversion_results.json does not carry the field, and
        # reversion_plan.json entries are the fallback.
        positions_to_revert: set = set()
        cdoc = _load_contaminated_doc(cycle, parent_pathway)
        for e in cdoc.get("contaminated", []):
            if e.get("label") == leaf_label:
                positions_to_revert = set(
                    int(p) for p in (e.get("positions_to_revert") or [])
                )
                break
        if not positions_to_revert:
            # Fallback.
            rp = _load_rev_plan(cycle, parent_pathway)
            for e in rp.get("entries", []):
                if e.get("label") == leaf_label:
                    positions_to_revert = set(
                        int(p) for p in (e.get("positions_to_revert") or [])
                    )
                    break

        # A cycle-0 design's mutations.tsv is its full history, with no
        # ancestors to walk, and the run is single-cycle.  Cycle 1+
        # walked the lineage with read_cumulative_mutations instead.
        # See archive/cycle1plus_aggregation.py.
        cum_muts: List[Tuple[int, str, str]] = []
        pdb_col = r.get("pdb", "")
        if pdb_col:
            mut_tsv = Path(pdb_col).parent / "mutations.tsv"
            cum_muts = read_mutations_tsv_full(mut_tsv)

        # Where a position was touched twice, keep the most recent edit.
        latest_by_pos: Dict[int, Tuple[int, str, str]] = {}
        for pos, wt, mut in cum_muts:
            latest_by_pos[pos] = (pos, wt, mut)
        remaining = [
            t for pos, t in latest_by_pos.items()
            if pos not in positions_to_revert
        ]
        remaining.sort(key=lambda t: t[0])
        cx, aa = format_mutation_strings(remaining, chain_id=pred_rec_chain)
        r["reverted_total_mutations"] = len(remaining)
        r["reverted_mutations_chimerax"] = cx
        r["reverted_mutations_aa"] = aa


def _row_is_clean_steered(r: Dict) -> bool:
    """Row is a clean steered design, uncontaminated and passing the
    structural filter.

    The 5 Å threshold is hardcoded to match the steering pipeline's
    default rmsd_threshold.  The check used to be a proxy on
    `steered_ipsae_min` being populated, which broke under
    `compute-final-metrics --populate-all`.  That flag populates ipSAE
    on every non-dropped row, wrong-interface ones included, and let
    them into the composite ranking.
    """
    if r.get("design") == "initial":
        return False
    verdict = (r.get("reversion_verdict") or "").strip()
    if verdict != "":
        return False
    try:
        if int(r.get("steered_receptor_intact", 0) or 0) != 1:
            return False
    except (ValueError, TypeError):
        return False
    # See the docstring for what this check replaced.
    ra_str = r.get("steered_ra_eff_vs_truth", "")
    if ra_str in ("", None):
        return False
    try:
        ra = float(ra_str)
    except (ValueError, TypeError):
        return False
    if math.isnan(ra) or math.isinf(ra):
        return False
    if ra >= 5.0:
        return False
    return True


def _row_is_pose_holds(r: Dict) -> bool:
    """Row is a pose_holds design whose reverted prediction passes the
    structural filter.

    The ra_eff check mirrors _row_is_clean_steered, so a pose_holds row
    with a reverted ra_eff of 25 Å is not ranking-eligible.  The only
    earlier guard was the reversion harvest's intact filter, which does
    not cover ra_eff."""
    if r.get("reversion_verdict") != "pose_holds":
        return False
    try:
        if int(r.get("reverted_receptor_intact", 0) or 0) != 1:
            return False
    except (ValueError, TypeError):
        return False
    ra_str = r.get("reverted_ra_eff_vs_truth", "")
    if ra_str in ("", None):
        return False
    try:
        ra = float(ra_str)
    except (ValueError, TypeError):
        return False
    if math.isnan(ra) or math.isinf(ra):
        return False
    if ra >= 5.0:
        return False
    return True


def _ranking_metric(r: Dict, name: str) -> Optional[float]:
    """Canonical value of `name` for ranking, one of 'ra_eff_vs_truth'
    or 'ipsae_min'.  Reads the steered_ value on a clean steered row
    and the reverted_ value on a pose_holds row.  None if the row is
    ineligible or the value is missing.
    """
    if _row_is_clean_steered(r):
        prefix = "steered_"
    elif _row_is_pose_holds(r):
        prefix = "reverted_"
    else:
        return None
    v = r.get(prefix + name, "")
    if v in ("", None):
        return None
    try:
        v = float(v)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (ValueError, TypeError):
        return None


def _ranking_composite(r: Dict) -> Optional[float]:
    """Composite ranker score, true_jaccard − 0.05 · ra_eff_vs_truth.
    Higher is better.

    Eligibility matches _ranking_metric.  true_jaccard always comes
    from steered_true_jaccard, because the overlap is computed once on
    the steered prediction and never recomputed after reversion.  The
    interface identity rides on the designed sequence, not on the
    steering mutations.  None if a component is missing or the row is
    ineligible.
    """
    ra = _ranking_metric(r, "ra_eff_vs_truth")
    if ra is None:
        return None
    v = r.get("steered_true_jaccard", "")
    if v in ("", None):
        return None
    try:
        tj = float(v)
        if math.isnan(tj) or math.isinf(tj):
            return None
    except (ValueError, TypeError):
        return None
    return tj - 0.05 * ra


def _compute_unified_ranks(rows: List[Dict]) -> None:
    """Add rank_by_ra_eff and rank_by_composite_score.  Both are
    1-based and rank 1 is best.  Eligible rows are clean steered or
    pose_holds, and ineligible rows get blank.

    rank_by_composite_score is the primary ranker.  A third rank,
    rank_by_ipsae_min, used to sit alongside these two and is now
    dropped entirely.
    """
    # Initialise blanks
    for r in rows:
        r.setdefault("rank_by_ra_eff", "")
        r.setdefault("rank_by_composite_score", "")

    # rank_by_ra_eff: ascending (lower is better)
    eligible = []
    for r in rows:
        v = _ranking_metric(r, "ra_eff_vs_truth")
        if v is not None:
            eligible.append((v, id(r), r))
    eligible.sort(key=lambda t: (t[0], t[1]))
    for rank_idx, (_, _, r) in enumerate(eligible, start=1):
        r["rank_by_ra_eff"] = rank_idx

    # rank_by_composite_score: descending (higher is better).  Rows
    # missing ra_eff or steered_true_jaccard get a blank rank, usually
    # rows where compute-final-metrics never populated interface
    # metrics.
    eligible = []
    for r in rows:
        v = _ranking_composite(r)
        if v is not None:
            eligible.append((v, id(r), r))
    eligible.sort(key=lambda t: (-t[0], t[1]))
    for rank_idx, (_, _, r) in enumerate(eligible, start=1):
        r["rank_by_composite_score"] = rank_idx


def _agg_continuous(values: List[Optional[float]]) -> Tuple[Optional[float], Optional[float], int]:
    """Aggregate per-seed values into (median, MAD, n_used), dropping
    None, NaN and Inf.

    MAD is the median absolute deviation from the median.  It beats SD
    at small N, where SD on N=3 is dominated by one outlier and swings
    from run to run.  (None, None, 0) if nothing is usable.
    """
    clean: List[float] = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(f) or math.isinf(f):
            continue
        clean.append(f)
    if not clean:
        return (None, None, 0)
    med = statistics.median(clean)
    if len(clean) == 1:
        # MAD is undefined at N=1.  0.0 means no spread visible.
        return (med, 0.0, 1)
    mad = statistics.median(abs(v - med) for v in clean)
    return (med, mad, len(clean))


def _agg_majority_binary(values: List[Optional[int]]) -> Tuple[Optional[int], int, int]:
    """Aggregate per-seed 0/1 values into (majority_value, n_positive,
    n_used).  The threshold is ≥ ceil(N/2).

    None and unparseable entries are dropped, which tolerates partial
    failures.  (None, 0, 0) if nothing is usable.
    """
    clean: List[int] = []
    for v in values:
        if v is None:
            continue
        try:
            i = int(v)
        except (TypeError, ValueError):
            continue
        if i not in (0, 1):
            continue
        clean.append(i)
    if not clean:
        return (None, 0, 0)
    n_used = len(clean)
    n_pos = sum(clean)
    threshold = (n_used + 1) // 2  # ceil(N/2)
    return (1 if n_pos >= threshold else 0, n_pos, n_used)


def _agg_majority_positions(
    per_seed_position_sets: List[List[int]],
) -> Tuple[List[int], Dict[int, int], int]:
    """Per-position majority vote over per-seed position sets.  A
    position is in the aggregate iff ≥ ceil(N/2) seeds report it.
    Returns (aggregate_positions, position_to_seed_count, n_seeds_used).

    `position_to_seed_count` lets the aggregator record near-misses,
    such as a position reported by 1 of 3 seeds.  Empty input →
    ([], {}, 0).
    """
    if not per_seed_position_sets:
        return ([], {}, 0)
    n_seeds = len(per_seed_position_sets)
    threshold = (n_seeds + 1) // 2
    counts: Dict[int, int] = {}
    for ps in per_seed_position_sets:
        seen_in_this_seed = set()
        for p in ps:
            try:
                pi = int(p)
            except (TypeError, ValueError):
                continue
            if pi in seen_in_this_seed:
                continue
            seen_in_this_seed.add(pi)
            counts[pi] = counts.get(pi, 0) + 1
    aggregate = sorted(p for p, c in counts.items() if c >= threshold)
    return (aggregate, counts, n_seeds)


def _parse_position_csv(s: str) -> List[int]:
    """Parse a comma-separated string of 1-based positions into ints,
    tolerating blanks and non-integer tokens.  Used on the
    contact_residues and mutated_contact_positions columns."""
    if not s:
        return []
    out = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            continue
    return out


def _aggregate_per_sequence(
    per_seed_rows: List[Dict],
    prefix: str,
) -> Dict:
    """Aggregate per-seed rows sharing one unique sequence into a
    single summary dict.

    `prefix` is "steered_" or "reverted_", and every metric column in
    the per-seed rows must start with it.  Suffixes on
    `<prefix><metric>`:
      continuous     `_median`, `_mad`, `_n_used`
      binary         `_majority`, `_n_positive`, `_n_used`
      position sets  `_majority` as a CSV string,
                     `_per_seed_counts` as a JSON map, `_n_used`

    Identification fields come from the FIRST row.  Rows in one
    sequence_group share them, because one unique sequence gives one
    design name once the `_sN` suffix is removed.  The caller merges
    the steered and reverted records into one row.
    """
    if not per_seed_rows:
        return {}

    out: Dict[str, any] = {}

    for metric in _AGG_CONTINUOUS_METRICS + _AGG_INTEGER_METRICS:
        col = prefix + metric
        vals = [r.get(col) for r in per_seed_rows]
        med, mad, n_used = _agg_continuous(vals)
        out[f"{prefix}{metric}_median"] = "" if med is None else round(med, 4)
        out[f"{prefix}{metric}_mad"] = "" if mad is None else round(mad, 4)
        out[f"{prefix}{metric}_n_used"] = n_used

    for metric in _AGG_BINARY_METRICS:
        col = prefix + metric
        vals = [r.get(col) for r in per_seed_rows]
        majority, n_pos, n_used = _agg_majority_binary(vals)
        out[f"{prefix}{metric}_majority"] = "" if majority is None else majority
        out[f"{prefix}{metric}_n_positive"] = n_pos
        out[f"{prefix}{metric}_n_used"] = n_used

    for metric in _AGG_POSITION_SET_METRICS:
        col = prefix + metric
        position_sets = [_parse_position_csv(r.get(col, "")) for r in per_seed_rows]
        agg_positions, counts, n_used = _agg_majority_positions(position_sets)
        out[f"{prefix}{metric}_majority"] = ",".join(
            str(p) for p in agg_positions
        )
        # Empty → "" rather than "{}", so downstream readers can use a
        # truthiness test.
        out[f"{prefix}{metric}_per_seed_counts"] = (
            json.dumps({str(p): c for p, c in sorted(counts.items())})
            if counts else ""
        )
        out[f"{prefix}{metric}_n_used"] = n_used

    return out


def _classify_outcome(
    agg: Dict,
    verdict_counts: Dict[str, int],
    structural_ra_eff_cutoff: float = 5.0,
    contamination_gating_positions: Optional[Set[int]] = None,
) -> Tuple[str, str]:
    """Re-derive the reversion outcome from the AGGREGATED reverted
    metrics.  Mirrors classify_reversion_verdict but reads the
    median / majority columns from _aggregate_per_sequence.

    Returns (outcome, reason).  The wet-lab construct carries none of
    the steering mutations, so a pose that depends on a mutated residue
    touching the effector is not evidence about the real molecule.
    Reversion puts those mutations back to wild type and re-predicts.
      pose_holds        the pose survives reversion, so it is usable
                        for wet-lab triage.
      pose_collapses    the reverted prediction fails the structural
                        filter, so the pose needed the mutations.
                        Dropped.
      new_contamination the reverted pose leans on a different mutated
                        contact.  Dropped.
      no_reversion      nothing needed reverting, so the record has no
                        reverted metrics.  Either cold start skipped
                        steering or no seed was contaminated on a
                        mutated position.

    pose_holds requires ALL of:
      1. the aggregated structural filter passes (intact_majority = 1,
         median ra_eff < cutoff, no position-majority contamination)
      2. at least ONE per-seed verdict is pose-passing, counting
         pose_holds and clean_steered
      3. the non-failing count (pose_holds + clean_steered + no_data)
         is at least as large as each failure count

    One seed suffices for rule 2 because `num_seeds = 3` and a per-seed
    pose_holds rate around 10-15% on this scaffold put majority-of-3
    out of reach.  Rule 1 already excludes the "every seed failed
    differently" artefact, since a position reaches
    `reverted_mutated_contact_positions_majority` only when ≥ ceil(N/2)
    seeds contaminate at that same position.

    Rule 3 downgrades only when the non-failing count is strictly
    smaller than the larger failure count, so an even split stays
    pose_holds.  Where the two failure modes tie, the downgrade picks
    pose_collapses as the more conservative call.

    `verdict_counts` comes from _per_seed_verdict_breakdown, with keys
    pose_holds, pose_collapses, new_contamination, no_data and
    clean_steered.
    """
    n_used = int(agg.get("reverted_ra_eff_vs_truth_n_used", 0) or 0)
    if n_used == 0:
        return ("no_reversion", "no reverted prediction available")

    nph = int(verdict_counts.get("pose_holds", 0))
    npc = int(verdict_counts.get("pose_collapses", 0))
    nnc = int(verdict_counts.get("new_contamination", 0))
    nnd = int(verdict_counts.get("no_data", 0))
    ncs = int(verdict_counts.get("clean_steered", 0))
    n_total_verdicts = nph + npc + nnc + nnd + ncs
    # A clean_steered seed was structurally fine and uncontaminated,
    # so its steered structure is the final verdict for that seed.
    nph_eff = nph + ncs

    # Structural filter on the aggregated reverted metrics.
    intact_majority = agg.get("reverted_receptor_intact_majority", "")
    if intact_majority != 1:
        return ("pose_collapses",
                f"reverted not intact by majority "
                f"({agg.get('reverted_receptor_intact_n_positive', 0)}/"
                f"{agg.get('reverted_receptor_intact_n_used', 0)} seeds intact)")

    ra_med = agg.get("reverted_ra_eff_vs_truth_median", "")
    if ra_med == "" or ra_med is None:
        return ("pose_collapses", "reverted_ra_eff_vs_truth median unavailable")
    try:
        ra_med_f = float(ra_med)
    except (TypeError, ValueError):
        return ("pose_collapses", "reverted_ra_eff_vs_truth median unparseable")
    if ra_med_f >= structural_ra_eff_cutoff:
        return ("pose_collapses",
                f"reverted ra_eff median {ra_med_f:.2f} ≥ "
                f"cutoff {structural_ra_eff_cutoff:.2f}")

    # Only positions inside the design region ∪ true interface count
    # as contamination, which matches the pre-reversion policy.
    # Positions outside the gated set, such as off-interface "reaching
    # bit" contacts, are acceptable by construction and must not
    # downgrade the aggregated verdict.
    contam_majority = agg.get(
        "reverted_mutated_contact_positions_majority", "")
    if contam_majority:
        all_positions = [int(p) for p in str(contam_majority).split(",")
                         if p.strip().lstrip("-").isdigit()]
        if contamination_gating_positions is not None:
            gated_positions = [p for p in all_positions
                               if p in contamination_gating_positions]
            ungated_positions = [p for p in all_positions
                                 if p not in contamination_gating_positions]
        else:
            gated_positions = all_positions
            ungated_positions = []

        if gated_positions:
            reason = (f"reverted pose has mutated contact(s) by majority "
                      f"at gated position(s) "
                      f"(design region ∪ true interface): {gated_positions}")
            if ungated_positions:
                reason += (f"; also touching ungated positions "
                           f"{ungated_positions} (acceptable)")
            return ("new_contamination", reason)
        # else: all majority contacts are at ungated positions — fall
        # through to the pose_holds path

    # Rule 2, the any-seed gate.  See the docstring for why one seed
    # is enough.
    if nph_eff < 1:
        # Ties favour pose_collapses as the more conservative call.
        if nnc > npc:
            downgraded = "new_contamination"
        else:
            downgraded = "pose_collapses"
        return (downgraded,
                f"aggregated structural/position rule said pose_holds but "
                f"0/{n_total_verdicts} seeds individually agree "
                f"(pose_collapses={npc}, new_contamination={nnc}, "
                f"no_data={nnd}, clean_steered={ncs}); "
                f"downgrading to {downgraded}")

    # Rule 3.  This stops "1 passes, 2 fail the same way at different
    # positions, so no position reaches majority" being pose_holds.
    non_failing = nph_eff + nnd
    if non_failing < max(npc, nnc):
        if nnc > npc:
            downgraded = "new_contamination"
        else:
            downgraded = "pose_collapses"
        return (downgraded,
                f"{nph_eff}/{n_total_verdicts} seed(s) individually pass "
                f"({nph} pose_holds + {ncs} clean_steered) "
                f"but failure modes dominate (pose_collapses={npc}, "
                f"new_contamination={nnc}, no_data={nnd}); "
                f"downgrading to {downgraded}")

    return ("pose_holds",
            f"reverted pose holds "
            f"({agg.get('reverted_receptor_intact_n_positive', 0)}/"
            f"{agg.get('reverted_receptor_intact_n_used', 0)} seeds intact, "
            f"median ra_eff {ra_med_f:.2f} Å, "
            f"{nph_eff}/{n_total_verdicts} seeds individually pass "
            f"[{nph} pose_holds + {ncs} clean_steered])")


def _per_seed_verdict_breakdown(
    per_seed_reverted_rows: List[Dict],
    structural_ra_eff_cutoff: float = 5.0,
) -> Dict[str, int]:
    """Run the SINGLE-SEED verdict logic on each reverted per-seed row
    and return {verdict: count} over five buckets: pose_holds,
    pose_collapses, new_contamination, no_data, clean_steered.  Feeds
    n_pass (= pose_holds + clean_steered) and _classify_outcome.

    The `reversion_verdict` column is authoritative and preferred when
    present.  classify_reversion_verdict set it in
    cmd_harvest_reversions from fresh reversion_results.json data,
    before _populate_reverted_mutations blanked contact columns on
    non-pose_holds rows.  Without that preference, a row whose
    reverted_mutated_contact_positions had been blanked for display
    would read as pose_holds and inflate the count.  The raw reverted_*
    columns are the fallback.

    `clean_steered` covers a steered prediction with ZERO contamination
    on the mutated positions, where reversion was correctly skipped.
    Those rows have a blank reversion_verdict, blank reverted_*
    columns, and steered_n_contacts_on_mutated_positions == 0.  They
    used to land in no_data, which pushed otherwise-strong designs down
    the per-sequence triage.  clean_steered counts as pose-passing in
    _classify_outcome and in the cross-sequence tier classifier.
    """
    counts = {
        "pose_holds": 0,
        "pose_collapses": 0,
        "new_contamination": 0,
        "no_data": 0,
        "clean_steered": 0,
    }
    for r in per_seed_reverted_rows:
        # pose_holds-but-dropped states count as pose_collapses.
        authoritative = (r.get("reversion_verdict") or "").strip()
        if authoritative in ("pose_holds", "pose_collapses",
                             "new_contamination"):
            counts[authoritative] += 1
            continue
        if authoritative in ("pose_holds_not_intact", "missing_verdict",
                             "unknown"):
            counts["pose_collapses"] += 1
            continue
        if authoritative != "":
            # Unknown verdict string.  no_data keeps it visible.
            counts["no_data"] += 1
            continue

        # Verdict column is blank.  Two sub-cases:
        # (i)  zero contamination on mutated positions, so reversion
        #      was skipped on purpose and this seed is clean_steered.
        # (ii) reversion was attempted but the reverted_* fields are
        #      missing, so re-derive from the raw columns and fall back
        #      to no_data.
        steered_n_contacts_str = r.get(
            "steered_n_contacts_on_mutated_positions", "")
        try:
            steered_n_contacts = int(steered_n_contacts_str)
        except (TypeError, ValueError):
            steered_n_contacts = None

        # (i) also demands the steered row be structurally sound.
        # Otherwise a steered miss that happened to have no
        # contamination would silently count as a pass.
        if steered_n_contacts == 0:
            try:
                steered_intact = int(r.get("steered_receptor_intact", ""))
            except (TypeError, ValueError):
                steered_intact = None
            try:
                steered_ra = float(r.get("steered_ra_eff_vs_truth", ""))
            except (TypeError, ValueError):
                steered_ra = None
            reverted_attempted = any(
                (r.get(k) or "") != ""
                for k in ("reverted_ra_eff_vs_truth",
                          "reverted_receptor_intact",
                          "reverted_total_mutations")
            )
            if (steered_intact == 1
                    and steered_ra is not None
                    and not math.isnan(steered_ra)
                    and not math.isinf(steered_ra)
                    and steered_ra < structural_ra_eff_cutoff
                    and not reverted_attempted):
                counts["clean_steered"] += 1
                continue
            # Otherwise fall through to the (ii) re-derivation path.

        # (ii) re-derivation.
        intact = r.get("reverted_receptor_intact", "")
        try:
            intact_i = int(intact)
        except (TypeError, ValueError):
            counts["no_data"] += 1
            continue
        if intact_i != 1:
            counts["pose_collapses"] += 1
            continue
        try:
            ra = float(r.get("reverted_ra_eff_vs_truth", ""))
        except (TypeError, ValueError):
            counts["no_data"] += 1
            continue
        if math.isnan(ra) or math.isinf(ra):
            counts["no_data"] += 1
            continue
        if ra >= structural_ra_eff_cutoff:
            counts["pose_collapses"] += 1
            continue
        contam_str = r.get("reverted_mutated_contact_positions", "") or ""
        contam = _parse_position_csv(contam_str)
        if contam:
            counts["new_contamination"] += 1
            continue
        counts["pose_holds"] += 1
    return counts


def _aggregate_csv_fieldnames(rows: List[Dict]) -> List[str]:
    """Column order for all_results_multicycle.csv and the joined
    all_results_multicycle_with_metrics.csv.  Three blocks (A
    identification, B steering, C reversion) plus a tail of selection
    and ranks.  Keys not in the list are appended sorted at the end, so
    unexpected fields are never silently dropped.

    The Block B `steered_*` confidence columns are listed even though
    compute-final-metrics populates them, so the column order is the
    same whether or not that stage has run.
    """
    preferred = [
        # ── A. Identification ────────────────────────────────────
        "cycle", "pathway", "design",
        # Two rows with the same sequence_group differ only by Boltz
        # seed.
        "sequence_group", "seed_index",
        "status", "pdb",
        # ── B. Steering details ──────────────────────────────────
        # Metadata
        "steered_total_mutations",
        "steered_mutations_chimerax",
        "steered_mutations_aa",
        # Structural RMSDs vs truth (the steered prediction's values,
        # NEVER overwritten by reversion)
        "steered_ra_eff_vs_truth",
        "steered_independent_receptor_rmsd",
        "steered_independent_effector_rmsd",
        "steered_receptor_intact",
        # Interface-similarity
        "steered_wrong_jaccard",
        "steered_n_shared_wrong",
        "steered_true_jaccard",
        "steered_n_shared_true",
        "steered_n_design_interface_residues",
        # Confidence metrics on the steered prediction (populated by
        # compute-final-metrics for rows that pass its filter)
        "steered_avg_plddt",
        "steered_complex_plddt",
        "steered_ptm",
        "steered_iptm",
        "steered_pae_mean",
        "steered_ipae",
        "steered_pae_pass_frac",
        "steered_pae_cutoff_used",
        "steered_ipsae_ab",
        "steered_ipsae_ba",
        "steered_ipsae_min",
        "steered_actifptm",
        # Contact analysis + contamination check.  Any mutated residue
        # contacting the effector invalidates the metrics, and
        # mutated_contact_positions names which positions those are.
        "steered_n_contact_residues",
        "steered_contact_residues",
        "steered_n_contacts_on_mutated_positions",
        "steered_mutated_contact_positions",
        "steered_contact_cutoff_used",
        # Confidence flag + error column
        "steered_confidence_flag",
        "steered_error",
        # ── C. Reversion details ─────────────────────────────────
        "reversion_verdict",
        "reversion_dropped",
        "reverted_sequence_used_as_parent",
        # Reverted metadata
        "reverted_total_mutations",
        "reverted_mutations_chimerax",
        "reverted_mutations_aa",
        # Reverted structural RMSDs vs truth
        "reverted_ra_eff_vs_truth",
        "reverted_independent_receptor_rmsd",
        "reverted_independent_effector_rmsd",
        "reverted_receptor_intact",
        # Any remaining mutation contacting the effector in the
        # reverted prediction.  Empty for pose_holds by definition.
        "reverted_mutated_contact_positions",
        # Reverted contact analysis (from harvest-reversions)
        "reverted_n_contact_residues",
        "reverted_contact_residues",
        "reverted_contact_cutoff_used",
        # Reverted confidence suite. harvest-reversions populates these
        # for pose_holds rows and leaves them blank otherwise.
        "reverted_avg_plddt",
        "reverted_complex_plddt",
        "reverted_ptm",
        "reverted_iptm",
        "reverted_pae_mean",
        "reverted_ipae",
        "reverted_pae_pass_frac",
        "reverted_ipsae_ab",
        "reverted_ipsae_ba",
        "reverted_ipsae_min",
        "reverted_actifptm",
        # ── End. Selection / pathway-internal / ranks ────────────
        "min_dist_to_upstream",
        "selected_for_next_cycle",
        "rank_by_ra_eff",
        "rank_by_composite_score",
    ]
    all_keys = {k for r in rows for k in r.keys()}
    extra = sorted(k for k in all_keys
                   if k not in preferred and k != "parent_pathway")
    # parent_pathway: silently dropped (derivable from pathway)
    # With no rows there is nothing to intersect, so fall back to the
    # full preferred order.  A run where nothing passed still has to
    # leave a readable empty table.
    if not all_keys:
        return list(preferred)
    return [k for k in preferred if k in all_keys] + extra


# ───────────────────────────────────────────────────────────────────────
# Subcommand: aggregate
# ───────────────────────────────────────────────────────────────────────
def cmd_aggregate(args: argparse.Namespace) -> int:
    """Walk the entire experiment tree, build pathways.json + a flat
    all_results_multicycle.csv, and report the bell-curve summary."""
    experiment_root = args.experiment_root.resolve()
    if not experiment_root.is_dir():
        print(f"ERROR: {experiment_root} not a directory", file=sys.stderr)
        return 2

    # Two cycle-0 layouts, depending on the submit script's
    # --n-cycles:
    #   1:  cycle-0 files (plan.json, steered_results.csv, steered/,
    #       initial_prediction.pdb) sit directly in the experiment
    #       root, with no cycle_0/ subdirectory.
    #   2+: cycle-0 files sit under experiment_root/cycle_0/, with
    #       further cycle_N/ subdirectories per chain step.
    # plan.json, the canonical cycle-0 artefact, tells them apart.
    pathways: List[Dict] = []
    nested_cycle_0 = experiment_root / "cycle_0"
    if (nested_cycle_0 / "plan.json").exists():
        cycle_0_dir = nested_cycle_0
        print(f"  cycle_0 layout: nested ({cycle_0_dir})")
    elif (experiment_root / "plan.json").exists():
        cycle_0_dir = experiment_root
        print(f"  cycle_0 layout: flat ({cycle_0_dir})")
    else:
        print(f"ERROR: no cycle_0 plan.json under {experiment_root} "
              f"(checked {nested_cycle_0}/plan.json and "
              f"{experiment_root}/plan.json)", file=sys.stderr)
        return 2

    # Cycle 0 — special, the single-cycle script writes here
    plan0 = json.loads((cycle_0_dir / "plan.json").read_text())
    pathways.append({
        "label": "cycle_0",
        "cycle": 0,
        "workdir": str(cycle_0_dir),
        "n_designs": len(plan0.get("designs", [])),
    })

    # Cycles 1+ are gone.  A cycle re-steered the previous round's
    # survivors on top of their already-mutated sequences, so the
    # mutation count compounded.  The harness went to
    # archive/boltz2_iterate_steering_multicycle.py and its reader, the
    # pathway_<label> discovery and row builder that stood here, went
    # to archive/cycle1plus_aggregation.py.  This refuses a cycle_N
    # directory rather than ignoring it, because skipping one would
    # write an all_results_multicycle.csv that looks complete while
    # omitting every row of that cycle.
    stray_cycles = [d.name for d in sorted(experiment_root.glob("cycle_*"))
                    if d.is_dir() and d.name != "cycle_0"]
    if stray_cycles:
        print(f"ERROR: follow-up cycle directories under {experiment_root}: "
              f"{', '.join(stray_cycles)}\n"
              f"  This aggregator is single-cycle.  The cycle-1+ reader was "
              f"archived to archive/cycle1plus_aggregation.py; aggregating "
              f"this root would silently drop those rows.",
              file=sys.stderr)
        return 2

    (experiment_root / "pathways.json").write_text(
        json.dumps(pathways, indent=2)
    )

    # The wild-type baseline ra_eff was read here to give cycle-1+ rows
    # a delta_receptor_aligned_effector_rmsd on cycle 0's baseline.
    # Cycle 0 already gets that column from steered_results.csv, so
    # nothing is left to compute.
    cycle0_plan = json.loads((cycle_0_dir / "plan.json").read_text())
    # Receptor chain ID for the mutations_chimerax column.  "A" if the
    # plan does not record it (old runs).
    pred_rec_chain_for_spec = cycle0_plan.get("pred_receptor_chain", "A")

    # Cycle 0's kickoff writes pose distances and selection info into
    # cycle_0/passing.json.  Index by design so it merges onto the
    # steered_results.csv rows.
    cycle0_passing_path = cycle_0_dir / "passing.json"
    cycle0_extra_by_design: Dict[str, Dict] = {}
    if cycle0_passing_path.exists():
        cp = json.loads(cycle0_passing_path.read_text())
        sel_idxs_c0 = {s["design_idx"] for s in cp.get("selected", [])}
        for c in cp.get("all_candidates", []):
            cycle0_extra_by_design[c["design"]] = {
                "min_dist_to_upstream": c.get("min_dist_to_upstream"),
                "selected_for_next_cycle": 1 if c["design_idx"] in sel_idxs_c0 else 0,
                # Blank for non-reverted designs.  Steered values
                # stay in receptor_aligned_effector_rmsd and
                # independent_*_rmsd, which _translate_aggregate_row
                # renames at write time.  Reverted values live only in
                # reverted_* keys.
                "reversion_verdict": c.get("reversion_verdict", ""),
                "reversion_dropped": c.get("reversion_dropped", ""),
                "reverted_sequence_used_as_parent": c.get(
                    "reverted_sequence_used_as_parent", ""
                ),
                "reverted_ra_eff_vs_truth": c.get("reverted_ra_eff_vs_truth", ""),
                "reverted_independent_receptor_rmsd": c.get(
                    "reverted_independent_receptor_rmsd", ""
                ),
                "reverted_independent_effector_rmsd": c.get(
                    "reverted_independent_effector_rmsd", ""
                ),                **{k: c.get(k, "") for k in _REVERTED_CONFIDENCE_FIELDS},
            }

    # Flat CSV: one row per prediction, ranked by ra_eff vs truth
    rows = []
    for p in pathways:
        wd = Path(p["workdir"])
        # Cycle 0 has its own steered_results.csv from the single-cycle script
        if p["label"] == "cycle_0":
            csv_path = wd / "steered_results.csv"
            if not csv_path.exists():
                # Skip_steering under an older driver wrote no CSV,
                # though the cold-start prediction and plan.json
                # exist.  Reconstruct one initial row from them.  This
                # path fires for 7B1I-style workdirs aggregated before
                # the driver was fixed.
                initial_pdb = wd / "initial_prediction.pdb"
                if (cycle_0_dir / "plan.json").exists() and initial_pdb.exists():
                    print("  cycle_0: no steered_results.csv — "
                          "reconstructing initial row from plan.json + "
                          "initial_prediction.pdb")
                    plan_data = cycle0_plan
                    initial_ra = plan_data.get(
                        "initial_receptor_aligned_effector_rmsd", "")
                    initial_ind_rec = plan_data.get(
                        "initial_independent_receptor_rmsd", "")
                    initial_ind_eff = plan_data.get(
                        "initial_independent_effector_rmsd", "")
                    try:
                        intact_flag = (
                            1 if (float(initial_ind_rec) <= 5.0
                                  and float(initial_ind_eff) <= 5.0)
                            else 0
                        )
                    except (TypeError, ValueError):
                        intact_flag = ""
                    # Interface fields are present only if interface
                    # analysis ran, which not every skip type does.
                    initial_wrong = plan_data.get("initial_wrong_interface_idx") or []
                    true_set = plan_data.get("true_interface_idx") or []
                    # Only the truth set is required here.  An empty
                    # initial_wrong gives jaccard = 0, not undefined.
                    # boltz2_negative_steering.py's
                    # _write_initial_only_csv does the same.
                    if true_set:
                        n_iface = len(initial_wrong)
                        n_shared_true = len(set(initial_wrong) & set(true_set))
                        true_j = n_shared_true / max(
                            len(set(initial_wrong) | set(true_set)), 1)
                        wrong_j = 1.0 if initial_wrong else float("nan")
                    else:
                        n_iface = ""
                        n_shared_true = ""
                        true_j = ""
                        wrong_j = ""
                    reconstructed = {
                        "rank": "0",
                        "design": "initial",
                        "total_mutations": "0",
                        "receptor_aligned_effector_rmsd": str(initial_ra),
                        "delta_receptor_aligned_effector_rmsd": "0.0",
                        "independent_receptor_rmsd": str(initial_ind_rec),
                        "independent_effector_rmsd": str(initial_ind_eff),
                        "receptor_intact": str(intact_flag),
                        "wrong_jaccard": str(wrong_j),
                        "n_shared_wrong": str(n_iface),
                        "true_jaccard": str(true_j),
                        "n_shared_true": str(n_shared_true),
                        "n_design_interface_residues": str(n_iface),
                        "status": "initial",
                        "pdb": str(initial_pdb.resolve()),
                        "pathway": "cycle_0",
                        "parent_pathway": "",
                        "cycle": 0,
                        # A pseudo-group with seed_index 0 keeps the
                        # grouping logic uniform for the unsteered,
                        # single-seed baseline.
                        "sequence_group": "initial",
                        "seed_index": 0,
                        "min_dist_to_upstream": "",
                        "selected_for_next_cycle": "",
                        "mutations_chimerax": "",
                        "mutations_aa": "",
                    }
                    rows.append(reconstructed)
                else:
                    print("  cycle_0: no steered_results.csv AND no "
                          "initial_prediction.pdb — skipping")
                continue
            with open(csv_path) as f:
                for r in csv.DictReader(f):
                    r["pathway"] = "cycle_0"
                    r["parent_pathway"] = ""
                    r["cycle"] = 0
                    # The 'initial' baseline row has no design
                    # entry, so it stays blank.
                    extra = cycle0_extra_by_design.get(r.get("design", ""), {})
                    r["min_dist_to_upstream"] = extra.get("min_dist_to_upstream", "")
                    r["selected_for_next_cycle"] = extra.get("selected_for_next_cycle", "")
                    # Reversion fields — blank for non-reverted designs
                    for k in (
                        "reversion_verdict",
                        "reversion_dropped",
                        "reverted_sequence_used_as_parent",
                        "reverted_ra_eff_vs_truth",
                        "reverted_independent_receptor_rmsd",
                        "reverted_independent_effector_rmsd",                        *_REVERTED_CONFIDENCE_FIELDS,
                    ):
                        r[k] = extra.get(k, "")
                    # mutations.tsv sits in the design dir, derived
                    # from the pdb path.  Baseline rows have none, and
                    # multi-seed cold start names them "initial",
                    # "initial_s1", "initial_s2" and so on.
                    design_name = r.get("design", "")
                    if design_name == "initial" or design_name.startswith("initial_s"):
                        r["mutations_chimerax"] = ""
                        r["mutations_aa"] = ""
                    else:
                        pdb_col = r.get("pdb", "")
                        if pdb_col:
                            mut_tsv = Path(pdb_col).parent / "mutations.tsv"
                            muts = read_mutations_tsv_full(mut_tsv)
                        else:
                            muts = []
                        cx, aa = format_mutation_strings(
                            muts, chain_id=pred_rec_chain_for_spec,
                        )
                        r["mutations_chimerax"] = cx
                        r["mutations_aa"] = aa
                    rows.append(r)
            continue

        # Cycles 1+ built one row per candidate from the pathway's
        # passing.json, labelled with its lineage.  That reader is in
        # archive/cycle1plus_aggregation.py.  `pathways` now only ever
        # holds cycle_0, so nothing reaches here.

    out_csv = experiment_root / "all_results_multicycle.csv"
    if not rows:
        # The "Wrote" line below used to print on a zero-row run with
        # no file created, so readers got a FileNotFoundError after
        # being told it was written.  Write the header.
        with open(out_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=_aggregate_csv_fieldnames([]),
                           extrasaction="ignore").writeheader()
    if rows:
        # ── Translate internal dict keys to the CSV schema ──
        # Internal keys keep their historical names so upstream code
        # keeps working.  The write boundary is where they gain
        # consistent steered_ and reverted_ prefixes.
        rows = [_translate_aggregate_row(r) for r in rows]

        # ── reverted_mutations_* on pose_holds rows ──
        _populate_reverted_mutations(rows, experiment_root)

        # ── Unified ranks ──
        _compute_unified_ranks(rows)

        # ── Sort ──
        # Cycle, then the initial baseline, then
        # rank_by_composite_score, with blank ranks at the bottom.
        # The key used to be rank_by_ipsae_min, which is gone.
        def _sort_key(r):
            try:
                cyc = int(r.get("cycle", 0) or 0)
            except (ValueError, TypeError):
                cyc = 0
            is_initial = 0 if r.get("design") == "initial" else 1
            rk = r.get("rank_by_composite_score", "")
            try:
                rk_num = int(rk) if rk != "" else 10**9
            except (ValueError, TypeError):
                rk_num = 10**9
            return (cyc, is_initial, rk_num)
        rows.sort(key=_sort_key)

        # ── Write with the new column order ──
        fieldnames = _aggregate_csv_fieldnames(rows)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"Wrote {out_csv} ({len(rows)} rows)")
    print(f"Wrote {experiment_root / 'pathways.json'} ({len(pathways)} pathways)")

    stats_path = experiment_root / "cycle_statistics.csv"
    if stats_path.exists():
        print(f"Bell-curve data: {stats_path}")

    return 0


# ───────────────────────────────────────────────────────────────────────
# compute-final-metrics
# ───────────────────────────────────────────────────────────────────────
def _count_ca_per_chain(pdb_path: Path) -> "Dict[str, int]":
    """Count Cα atoms per chain ID from a PDB file.  Stdlib only, so
    this subcommand still runs outside the numpy/gemmi container."""
    counts: Dict[str, int] = {}
    try:
        with open(pdb_path) as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                if line[12:16].strip() != "CA":
                    continue
                chain_id = line[21:22].strip() or " "
                counts[chain_id] = counts.get(chain_id, 0) + 1
    except Exception as e:
        print(f"  WARNING: failed to count CA atoms in {pdb_path}: {e}",
              file=sys.stderr)
    return counts


def _float_or_none(x) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


# ───────────────────────────────────────────────────────────────────────
# Subcommand: aggregate-per-sequence
# ───────────────────────────────────────────────────────────────────────
def cmd_aggregate_per_sequence(args: argparse.Namespace) -> int:
    """Collapse per-seed rows into one aggregated row per unique
    sequence.

    Reads all_results_multicycle_with_metrics.csv if
    compute-final-metrics has run and all_results_multicycle.csv
    otherwise.  Groups by `sequence_group` and writes
    raw_per_seed_results.csv, an untouched copy of the input, and
    aggregated_results.csv, one row per unique sequence with the
    verdict re-derived from the aggregated reverted record.

    _aggregate_per_sequence holds the per-metric rules.  The canonical
    seed, which supplies `pdb`, is chosen separately:
      odd N   the seed at the median ra_eff
      even N  the WORSE (higher) ra_eff of the two middle seeds, plus
              a warning column
      N=1     that seed, no warning

    Reverted metrics are populated only on rows that flowed through the
    reversion pass.  The aggregator deduplicates reverted records by
    sequence_hash via the `reversion_plan.json` manifest, aggregates
    per unique reverted sequence, then maps each aggregate back to
    every steered sequence pointing at that hash.  One reverted result
    can serve several steering results.
    """
    experiment_root: Path = args.experiment_root.resolve()
    if not experiment_root.is_dir():
        print(f"ERROR: {experiment_root} not a directory", file=sys.stderr)
        return 2

    # Prefer the with_metrics CSV, so the aggregate covers the
    # confidence metrics too.
    metrics_csv = experiment_root / "all_results_multicycle_with_metrics.csv"
    plain_csv = experiment_root / "all_results_multicycle.csv"
    if metrics_csv.exists():
        per_seed_csv = metrics_csv
    elif plain_csv.exists():
        per_seed_csv = plain_csv
        print(f"  NOTE: {metrics_csv.name} not found; aggregating "
              f"{plain_csv.name} (no confidence metrics will be "
              f"available in the aggregate)")
    else:
        print(f"ERROR: neither {metrics_csv.name} nor {plain_csv.name} "
              f"found under {experiment_root}.  Run `aggregate` first.",
              file=sys.stderr)
        return 2

    with open(per_seed_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"  read {len(rows)} per-seed rows from {per_seed_csv.name}")

    # ── Load contamination gating set from cycle_0/plan.json ──────────
    # Only positions inside the design region ∪ true interface count
    # as contamination, the same rule classify_reversion_verdict
    # applies per seed.  plan.json stores the indices 0-based, so
    # convert to 1-based to match
    # reverted_mutated_contact_positions_majority.  No plan.json means
    # the gate is disabled and the older ungated behaviour applies.
    gating_1b: Set[int] = set()
    # Cold-start trigger threshold, also from plan.json, so cold-start
    # aggregates with ra_eff in (5.0, rmsd_threshold] still get a rank.
    # It matches the all-clean trigger in boltz2_negative_steering.py,
    # and defaults to 5.0.
    cold_start_rmsd_threshold: float = 5.0
    for candidate in (experiment_root / "cycle_0" / "plan.json",
                      experiment_root / "plan.json"):
        if candidate.exists():
            try:
                pl = json.loads(candidate.read_text())
                for i in (pl.get("design_region_idx") or []):
                    gating_1b.add(int(i) + 1)
                for i in (pl.get("true_interface_idx") or []):
                    gating_1b.add(int(i) + 1)
                try:
                    cold_start_rmsd_threshold = float(
                        pl.get("rmsd_threshold", cold_start_rmsd_threshold))
                except (TypeError, ValueError):
                    pass
                if gating_1b:
                    print(f"  contamination gating set: {len(gating_1b)} "
                          f"positions from {candidate.relative_to(experiment_root)} "
                          f"(design region ∪ true interface)")
                break
            except (ValueError, OSError) as e:
                print(f"  WARN: could not parse {candidate}: {e}")
    if not gating_1b:
        print("  WARN: no plan.json gating set found — aggregated verdict "
              "contamination check will use legacy (ungated) behaviour")

    # ── Copy per-seed input to raw_per_seed_results.csv ───────────────
    # One stable filename for "every seed's raw outputs", whether or
    # not compute-final-metrics ran.
    raw_csv = experiment_root / "raw_per_seed_results.csv"
    shutil.copy2(per_seed_csv, raw_csv)
    print(f"  wrote {raw_csv.name} (copy of {per_seed_csv.name})")

    # ── Group rows by (cycle, pathway, sequence_group) ────────────────
    # cycle and pathway scope the sequence_group.  Two designs in
    # different cycles sharing a sequence_group integer are NOT the
    # same unique sequence.  An empty sequence_group is the initial
    # baseline or a pre-multi-seed row, and becomes its own group of
    # size 1.
    from collections import defaultdict
    groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    singletons: List[Dict] = []
    for r in rows:
        sg = r.get("sequence_group", "")
        if sg in ("", None):
            # Singleton group, no aggregation.
            singletons.append(r)
            continue
        key = (str(r.get("cycle", "0")), str(r.get("pathway", "")), str(sg))
        groups[key].append(r)

    n_groups = len(groups)
    n_singletons = len(singletons)
    n_seeds_total = sum(len(v) for v in groups.values())
    print(f"  {n_groups} unique sequence group(s) × ~{n_seeds_total // max(n_groups, 1)} "
          f"seeds + {n_singletons} singleton(s)")

    # ── Aggregate each group ──────────────────────────────────────────
    aggregated_rows: List[Dict] = []

    def _canonical_seed(group: List[Dict]) -> Tuple[Dict, str]:
        """Pick the canonical seed for diagnostics.  Returns (row,
        warning), with warning empty when none is needed.

        Sorts by steered_ra_eff_vs_truth ascending, with NaN and
        missing values as infinity so they land at the end.  See the
        caller's docstring for the index rule.
        """
        def _ra(r):
            v = r.get("steered_ra_eff_vs_truth", "")
            try:
                f = float(v)
                if math.isnan(f) or math.isinf(f):
                    return float("inf")
                return f
            except (TypeError, ValueError):
                return float("inf")
        sorted_group = sorted(group, key=_ra)
        n = len(sorted_group)
        if n == 1:
            return (sorted_group[0], "")
        if n % 2 == 1:
            return (sorted_group[(n - 1) // 2], "")
        # Even N: pick the upper-middle (worse) seed and warn.
        return (sorted_group[n // 2],
                f"even_n={n}_chose_worse_of_two_middle_seeds")

    for key, group in sorted(groups.items()):
        cycle_str, pathway_str, sg_str = key

        canonical_row, canonical_warning = _canonical_seed(group)

        # Bare design name, with any _sN suffix stripped.
        design_name = canonical_row.get("design", "")
        if "_s" in design_name:
            base = design_name.rsplit("_s", 1)[0]
            # Validate the suffix is actually a numeric seed index
            suf = design_name.rsplit("_s", 1)[1]
            if suf.isdigit():
                design_name = base

        agg = {
            "cycle": cycle_str,
            "pathway": pathway_str,
            "sequence_group": sg_str,
            "design": design_name,
            "n_seeds": len(group),
            "canonical_seed_index": canonical_row.get("seed_index", ""),
            "canonical_pdb": canonical_row.get("pdb", ""),
            "canonical_seed_choice_warning": canonical_warning,
        }

        # Steered aggregation
        agg.update(_aggregate_per_sequence(group, prefix="steered_"))

        # Reverted aggregation runs only if some group member has a
        # reversion_verdict, meaning reversion ran on this sequence.
        any_reverted = any(
            (r.get("reversion_verdict") or "").strip() != ""
            for r in group
        )
        # Always computed.  It drives the outcome classifier and
        # n_pass, which is path-agnostic.
        verdict_counts = _per_seed_verdict_breakdown(group)

        if any_reverted:
            agg.update(_aggregate_per_sequence(group, prefix="reverted_"))
            outcome, reason = _classify_outcome(
                agg, verdict_counts,
                contamination_gating_positions=gating_1b if gating_1b else None,
            )
            agg["outcome"] = outcome
            agg["outcome_reason"] = reason
        else:
            # No seed needed reversion.  n_pass still comes from the
            # per-seed verdicts below.  A group with the wrong
            # placement and no contamination has every seed as
            # no_data, so n_pass is 0 rather than n_seeds, which
            # correctly lands it in tier none.
            agg["outcome"] = "no_reversion"
            agg["outcome_reason"] = ""

        # Seeds that passed the structural and contamination filters,
        # as pose_holds plus clean_steered.
        agg["n_pass"] = (verdict_counts["pose_holds"]
                        + verdict_counts["clean_steered"])

        aggregated_rows.append(agg)

    # Singletons, usually the cycle-0 initial baseline.  They pass
    # through with n_seeds=1 and no aggregation suffixes.
    # canonical_pdb IS populated, so consumers need no special case.
    for r in singletons:
        passthrough = dict(r)
        passthrough["n_seeds"] = 1
        passthrough["canonical_seed_index"] = r.get("seed_index", "")
        passthrough["canonical_pdb"] = r.get("pdb", "")
        passthrough["canonical_seed_choice_warning"] = ""
        passthrough["outcome"] = "singleton"
        passthrough["outcome_reason"] = (
            "no sequence_group — singleton row, no aggregation applied"
        )
        passthrough["n_pass"] = ""
        aggregated_rows.append(passthrough)

    # ── Recompute composite ranks on the aggregated rows ──────────────
    # These adapters apply _ranking_composite's logic to the
    # aggregated column names, such as steered_ra_eff_vs_truth_median.
    def _rank_continuous(r: Dict, name_median: str) -> Optional[float]:
        v = r.get(name_median, "")
        if v in ("", None):
            return None
        try:
            f = float(v)
            if math.isnan(f) or math.isinf(f):
                return None
            return f
        except (ValueError, TypeError):
            return None

    def _agg_row_is_clean_steered(r: Dict) -> bool:
        # A cold-start all-clean aggregate, design == "initial" with
        # outcome == "no_reversion", MUST be ranking-eligible, because
        # it is the final output for a cold-start binder.  It is
        # admitted under the looser cold-start trigger threshold from
        # plan.json.  Steered aggregates keep the 5.0 cap.
        is_cold_start = (
            r.get("design") == "initial"
            and (r.get("outcome") or "").strip() == "no_reversion"
        )
        if r.get("design") == "initial" and not is_cold_start:
            # design=="initial" without no_reversion is the older
            # single-row baseline extract_passing.py used as a sanity
            # check.  Excluding it stops reconstructed initial rows
            # from old workdirs being promoted.
            return False
        outcome = (r.get("outcome") or "").strip()
        if outcome not in ("", "no_reversion"):
            return False
        if r.get("steered_receptor_intact_majority") != 1:
            return False
        ra = _rank_continuous(r, "steered_ra_eff_vs_truth_median")
        if ra is None:
            return False
        cap = cold_start_rmsd_threshold if is_cold_start else 5.0
        if ra >= cap:
            return False
        return True

    def _agg_row_is_pose_holds(r: Dict) -> bool:
        if r.get("outcome") != "pose_holds":
            return False
        if r.get("reverted_receptor_intact_majority") != 1:
            return False
        ra = _rank_continuous(r, "reverted_ra_eff_vs_truth_median")
        if ra is None or ra >= 5.0:
            return False
        return True

    def _agg_composite(r: Dict) -> Optional[float]:
        if _agg_row_is_clean_steered(r):
            ra = _rank_continuous(r, "steered_ra_eff_vs_truth_median")
        elif _agg_row_is_pose_holds(r):
            ra = _rank_continuous(r, "reverted_ra_eff_vs_truth_median")
        else:
            return None
        if ra is None:
            return None
        tj = _rank_continuous(r, "steered_true_jaccard_median")
        if tj is None:
            return None
        return tj - 0.05 * ra

    def _agg_ra_eff(r: Dict) -> Optional[float]:
        if _agg_row_is_clean_steered(r):
            return _rank_continuous(r, "steered_ra_eff_vs_truth_median")
        if _agg_row_is_pose_holds(r):
            return _rank_continuous(r, "reverted_ra_eff_vs_truth_median")
        return None

    # Initialise blanks
    for r in aggregated_rows:
        r.setdefault("rank_by_ra_eff", "")
        r.setdefault("rank_by_composite_score", "")

    # rank_by_ra_eff: ascending (lower is better)
    eligible = []
    for r in aggregated_rows:
        v = _agg_ra_eff(r)
        if v is not None:
            eligible.append((v, id(r), r))
    eligible.sort(key=lambda t: (t[0], t[1]))
    for rank_idx, (_, _, r) in enumerate(eligible, start=1):
        r["rank_by_ra_eff"] = rank_idx

    # rank_by_composite_score: descending (higher is better)
    eligible = []
    for r in aggregated_rows:
        v = _agg_composite(r)
        if v is not None:
            eligible.append((v, id(r), r))
    eligible.sort(key=lambda t: (-t[0], t[1]))
    for rank_idx, (_, _, r) in enumerate(eligible, start=1):
        r["rank_by_composite_score"] = rank_idx

    # ── Write aggregated_results.csv ──────────────────────────────────
    # A deterministic column list keeps the layout stable across runs.
    # Block order is identification, steered, reverted, verdict, ranks.
    # Unexpected keys, such as singleton passthrough's, go sorted at
    # the end.
    preferred_cols = [
        # Identification
        "cycle", "pathway", "sequence_group", "design",
        "n_seeds", "canonical_seed_index", "canonical_pdb",
        "canonical_seed_choice_warning",
        # Steered aggregates — generated by _aggregate_per_sequence
        # in this order:
    ]
    for metric in (_AGG_INTEGER_METRICS + _AGG_CONTINUOUS_METRICS):
        preferred_cols.extend([
            f"steered_{metric}_median",
            f"steered_{metric}_mad",
            f"steered_{metric}_n_used",
        ])
    for metric in _AGG_BINARY_METRICS:
        preferred_cols.extend([
            f"steered_{metric}_majority",
            f"steered_{metric}_n_positive",
            f"steered_{metric}_n_used",
        ])
    for metric in _AGG_POSITION_SET_METRICS:
        preferred_cols.extend([
            f"steered_{metric}_majority",
            f"steered_{metric}_per_seed_counts",
            f"steered_{metric}_n_used",
        ])
    # Reverted aggregates (same shape, reverted_ prefix)
    for metric in (_AGG_INTEGER_METRICS + _AGG_CONTINUOUS_METRICS):
        preferred_cols.extend([
            f"reverted_{metric}_median",
            f"reverted_{metric}_mad",
            f"reverted_{metric}_n_used",
        ])
    for metric in _AGG_BINARY_METRICS:
        preferred_cols.extend([
            f"reverted_{metric}_majority",
            f"reverted_{metric}_n_positive",
            f"reverted_{metric}_n_used",
        ])
    for metric in _AGG_POSITION_SET_METRICS:
        preferred_cols.extend([
            f"reverted_{metric}_majority",
            f"reverted_{metric}_per_seed_counts",
            f"reverted_{metric}_n_used",
        ])
    # Outcome + ranks
    preferred_cols.extend([
        "outcome",
        "outcome_reason",
        "n_pass",
        "rank_by_ra_eff",
        "rank_by_composite_score",
    ])
    all_keys = {k for r in aggregated_rows for k in r.keys()}
    extras = sorted(k for k in all_keys if k not in preferred_cols)
    fieldnames = [k for k in preferred_cols if k in all_keys] + extras

    out_csv = experiment_root / "aggregated_results.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in aggregated_rows:
            w.writerow(r)
    print(f"  wrote {out_csv.name} ({len(aggregated_rows)} rows)")

    # Quick outcome tally for the log
    from collections import Counter
    outcomes = Counter(
        r.get("outcome", "") for r in aggregated_rows
    )
    print("  Outcome tally:")
    for v, c in sorted(outcomes.items()):
        print(f"    {v:<22s} {c:>4d}")

    # Even-N warnings tally
    n_warned = sum(
        1 for r in aggregated_rows
        if r.get("canonical_seed_choice_warning")
    )
    if n_warned:
        print(f"  Even-N canonical-seed warnings: {n_warned} row(s) "
              f"(degraded — some seeds failed; see "
              f"canonical_seed_choice_warning column)")

    return 0


def _passes_final_metrics_filter(row: Dict, args: argparse.Namespace,
                                  metric_col: str, threshold: float,
                                  skip_steering_cycle0: bool) -> bool:
    """Apply the cmd_compute_final_metrics passing-row filter to one row."""
    # --populate-all means every row, reversion_dropped included.  A
    # dropped seed with no metrics is a hole in the per-seed CSV, and a
    # hole shifts the median across seeds towards the survivors.
    if args.populate_all:
        return True
    # By default, skip rows dropped by verdict routing, because
    # confidence metrics on a discarded prediction are wasted work.
    # pose_holds rows are NOT dropped (reversion_dropped == 0), and
    # their steered_* metrics show the pre-reversion pose.
    try:
        if int(row.get("reversion_dropped", "0") or "0") == 1:
            return False
    except (TypeError, ValueError):
        pass
    # Skip_steering exemption for the initial row or rows.  Only
    # cycle-0 baselines are named "initial", or "initial_s1",
    # "initial_s2" and so on under multi-seed cold start.
    _design = row.get("design", "")
    if (skip_steering_cycle0
            and (_design == "initial" or _design.startswith("initial_s"))
            and str(row.get("cycle", "0")) == "0"):
        if args.require_intact:
            try:
                if int(row.get("steered_receptor_intact", "0") or "0") != 1:
                    return False
            except (TypeError, ValueError):
                return False
        return True
    # Require steered_receptor_intact == 1 unless explicitly disabled.
    if args.require_intact:
        try:
            if int(row.get("steered_receptor_intact", "0") or "0") != 1:
                return False
        except (TypeError, ValueError):
            return False
    v = _float_or_none(row.get(metric_col))
    if v is None:
        return False
    return v < threshold


def _resume_key_for_row(row: Dict) -> str:
    """Stable per-design resume key.  The pdb path is unique per design."""
    # `pathway` cannot serve, because cmd_aggregate gives every
    # cycle-0 design the literal string "cycle_0", so all of them would
    # collide.  It is still the fallback.
    return row.get("pdb") or row.get("pathway") or ""


def _load_existing_metrics_rows(output_csv: Path,
                                 prefixed_metric_keys: List[str]) -> Dict[str, Dict]:
    """Load existing-output rows keyed by resume key, keeping only those with at least one populated metric."""
    existing_by_key: Dict[str, Dict] = {}
    if not output_csv.exists():
        return existing_by_key
    with open(output_csv) as f:
        er = csv.DictReader(f)
        for row in er:
            key = _resume_key_for_row(row)
            if not key:
                continue
            has_any = any(row.get(k, "") not in ("", None)
                          for k in prefixed_metric_keys)
            if has_any:
                existing_by_key[key] = row
    return existing_by_key


def _classify_steered_confidence_flag(
    row: Dict,
    flag_pass_frac_min: float,
    flag_iptm_min: float,
    flag_complex_plddt_min: float,
    flag_ipae_max: float,
) -> str:
    """Return the confidence flag: ok, the one threshold crossed, or multiple.
    A row with no metrics returns the empty string."""
    ipsae = _float_or_none(row.get("steered_ipsae_min"))
    if ipsae is None:
        # No metrics — nothing to classify.
        return ""
    triggered = []
    pf = _float_or_none(row.get("steered_pae_pass_frac"))
    if pf is not None and pf < flag_pass_frac_min:
        triggered.append("low_pass_frac")
    ip = _float_or_none(row.get("steered_iptm"))
    if ip is not None and ip < flag_iptm_min:
        triggered.append("low_iptm")
    cp = _float_or_none(row.get("steered_complex_plddt"))
    if cp is not None and cp < flag_complex_plddt_min:
        triggered.append("low_plddt")
    ipae = _float_or_none(row.get("steered_ipae"))
    if ipae is not None and ipae > flag_ipae_max:
        triggered.append("high_ipae")
    if not triggered:
        return "ok"
    if len(triggered) == 1:
        return triggered[0]
    return "multiple"


def _write_metrics_csv(output_csv: Path, joined_rows: List[Dict]) -> None:
    """Write joined rows to output_csv using canonical Block A/B/C/End column order."""
    fieldnames = _aggregate_csv_fieldnames(joined_rows)
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r2 in joined_rows:
            w.writerow(r2)


def _process_passing_row(
    row: Dict,
    args: argparse.Namespace,
    prefixed_metric_keys: List[str],
    metric_keys_raw: List[str],
    metric_key_prefix: str,
    existing_by_key: Dict[str, Dict],
    scratch_dir: Path,
    compute_metrics_script: Path,
    cycle_0_plan: Dict,
    cycle_0_plan_path: Path,
    pred_rec_chain: str,
    pred_eff_chain: str,
    model_name: str,
    metric_col: str,
    passing_progress: int,
    n_passing_total: int,
) -> Tuple[Dict, str]:
    """Process one passing row: returns (out_row, status) where status is 'resumed', 'computed', or 'failed'."""
    pathway_key = _resume_key_for_row(row)
    # pathway_label kept for human-readable progress messages
    pathway_label = row.get("pathway") or row.get("pdb") or "?"
    pdb_path_str = row.get("pdb", "")
    pdb_path = Path(pdb_path_str) if pdb_path_str else None

    # Resume from the CURRENT base row, which may carry fresh columns
    # a previous run lacked, and overlay only the cached steered_* and
    # reverted_* metric values.  Otherwise stale data from an older
    # output CSV would overwrite fresh base columns.
    if args.skip_existing and pathway_key in existing_by_key:
        cached = existing_by_key[pathway_key]
        merged = dict(row)  # start from fresh base row
        for k in prefixed_metric_keys:
            v = cached.get(k, "")
            if v not in ("", None):
                merged[k] = v
        # Carry over steered_error so failure state survives for
        # diagnostics.
        if cached.get("steered_error"):
            merged["steered_error"] = cached["steered_error"]
        return merged, "resumed"

    out_row = dict(row)
    for k in prefixed_metric_keys:
        out_row.setdefault(k, "")
    out_row.setdefault("steered_error", "")

    if pdb_path is None or not pdb_path.exists():
        msg = f"pdb not found: {pdb_path_str}"
        print(f"  [{passing_progress}/{n_passing_total}] {pathway_label}: SKIP ({msg})")
        out_row["steered_error"] = msg
        return out_row, "failed"

    # compute_metrics.py's Boltz-2 parser looks for
    # confidence_<stem>.json, pae_<stem>.npz and plddt_<stem>.npz
    # beside each PDB.  The steering pipeline copies the chosen output
    # to `prediction.pdb` but leaves the sidecars in Boltz's layout,
    # `<design>/predictions/<stem>/pdb/<stem>_model_0.pdb` with
    # `confidence_<stem>_model_0.json` beside it.  --prediction-dir on
    # `prediction.pdb`'s parent would give ptm=0, iptm=0 and pae=0,
    # with only pLDDT surviving the B-factor fallback.  Point the
    # parser at the nested sidecar's directory instead.
    sidecar_pdb = _locate_boltz_sidecar_pdb(pdb_path)
    if sidecar_pdb is not None:
        pred_dir = sidecar_pdb.parent
        lookup_stem = sidecar_pdb.stem
        metric_pdb = sidecar_pdb
    else:
        pred_dir = pdb_path.parent
        lookup_stem = pdb_path.stem
        metric_pdb = pdb_path

    # CA counts must match whatever compute_metrics.py ingests.
    ca_counts = _count_ca_per_chain(metric_pdb)
    rec_len = ca_counts.get(pred_rec_chain, 0)
    eff_len = ca_counts.get(pred_eff_chain, 0)
    if rec_len == 0 or eff_len == 0:
        # Fallback: use the two largest chains
        ordered = sorted(ca_counts.items(), key=lambda kv: -kv[1])
        if len(ordered) >= 2:
            rec_len = rec_len or ordered[0][1]
            eff_len = eff_len or ordered[1][1]
    if rec_len == 0 or eff_len == 0:
        msg = (f"could not determine chain lengths from {metric_pdb.name} "
               f"(CA counts: {ca_counts})")
        print(f"  [{passing_progress}/{n_passing_total}] {pathway_label}: SKIP ({msg})")
        out_row["steered_error"] = msg
        return out_row, "failed"

    # One output CSV per row in the scratch dir, keyed on cycle plus
    # design name so the filename stays short and safe.
    cyc = row.get("cycle", "0")
    des = row.get("design", "unknown")
    safe_key = f"cycle{cyc}_{des}".replace("/", "_").replace(" ", "_")
    per_row_csv = scratch_dir / f"{safe_key}.csv"

    # _translate_aggregate_row renames `mutations_chimerax` to
    # `steered_mutations_chimerax`, so read the new name and fall back
    # to the old one for paths that skip that pass.  The format is
    # "/A:5,26,44,46", a chain letter plus 1-based positions, which
    # compute_metrics.py's parse_mutated_positions accepts as-is.
    # Blank means cold start with no mutations.
    #
    # Bug history: reading `mutations_chimerax` alone, always blank
    # once aggregate has run, dropped --mutated-positions from every
    # call and blanked steered_mutated_contact_positions and
    # steered_n_contacts_on_mutated_positions for every design,
    # including ones _populate_reverted_mutations had already filled.
    mutations_cx = (
        row.get("steered_mutations_chimerax", "")
        or row.get("mutations_chimerax", "")
        or ""
    )

    cmd = [
        args.python_executable, str(compute_metrics_script),
        "--model", model_name,
        "--prediction-dir", str(pred_dir),
        "--chain-lengths", str(rec_len), str(eff_len),
        "--output-csv", str(per_row_csv),
        "--pae-cutoff", str(args.pae_cutoff),
        "--receptor-chain", pred_rec_chain,
        "--effector-chain", pred_eff_chain,
        "--contact-cutoff", str(args.contact_cutoff),
        # compute_metrics.py reads 'effector_interface_atoms' out of
        # cycle_0/plan.json to restrict which effector atoms count.
        "--effector-atom-filter-json", str(cycle_0_plan_path),
    ]
    # intact_core and weighted_jaccard need the ground-truth PDB,
    # whose absolute path is in cycle_0/plan.json under
    # "ground_truth".  Without it the flag is omitted and those columns
    # come out blank.
    _native_pdb = cycle_0_plan.get("ground_truth", "")
    if _native_pdb:
        cmd.extend(["--native-pdb", str(_native_pdb)])
    if mutations_cx:
        cmd.extend(["--mutated-positions", mutations_cx])
    # No --reference-pdb.  The structural RMSDs come from
    # all_results_multicycle.csv, with the correct chain mapping.

    print(f"  [{passing_progress}/{n_passing_total}] {pathway_label}  "
          f"{metric_col}={row.get(metric_col)}  pdb={metric_pdb.name}")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=args.subprocess_timeout,
        )
    except subprocess.TimeoutExpired:
        msg = f"compute_metrics.py timed out after {args.subprocess_timeout}s"
        print(f"    ERROR: {msg}")
        out_row["steered_error"] = msg
        return out_row, "failed"
    except Exception as e:
        msg = f"subprocess failed: {e}"
        print(f"    ERROR: {msg}")
        out_row["steered_error"] = msg
        return out_row, "failed"

    if proc.returncode != 0:
        msg = (f"compute_metrics.py exited with {proc.returncode}: "
               f"{(proc.stderr or '').strip()[:300]}")
        print(f"    ERROR: {msg}")
        if args.verbose and proc.stdout:
            print("    --- stdout ---")
            print(proc.stdout)
        out_row["steered_error"] = msg
        return out_row, "failed"

    if not per_row_csv.exists():
        msg = f"compute_metrics.py produced no CSV at {per_row_csv}"
        print(f"    ERROR: {msg}")
        out_row["steered_error"] = msg
        return out_row, "failed"

    # With multiple models in the CSV, which one-design-per-dir should
    # prevent, take the row whose model_name matches the pdb stem.
    try:
        with open(per_row_csv) as f:
            cm_rows = list(csv.DictReader(f))
    except Exception as e:
        msg = f"could not read {per_row_csv}: {e}"
        print(f"    ERROR: {msg}")
        out_row["steered_error"] = msg
        return out_row, "failed"

    if not cm_rows:
        msg = "compute_metrics.py CSV was empty"
        print(f"    ERROR: {msg}")
        out_row["steered_error"] = msg
        return out_row, "failed"

    chosen = None
    for cr in cm_rows:
        if cr.get("model_name") == lookup_stem:
            chosen = cr
            break
    if chosen is None:
        chosen = cm_rows[0]  # fall back to first (and usually only)

    for k in metric_keys_raw:
        out_row[metric_key_prefix + k] = chosen.get(k, "")
    return out_row, "computed"


def cmd_compute_final_metrics(args: argparse.Namespace) -> int:
    """Shell out to compute_metrics.py for every row in
    all_results_multicycle.csv that passes the final ranking filter,
    and write a joined CSV beside it.

    Opt-in, after `aggregate`, which keeps the fast `aggregate` pass
    stdlib-only.
    """
    experiment_root: Path = args.experiment_root.resolve()
    if not experiment_root.is_dir():
        print(f"ERROR: {experiment_root} not a directory", file=sys.stderr)
        return 2

    all_results_csv = experiment_root / "all_results_multicycle.csv"
    if not all_results_csv.exists():
        print(f"ERROR: {all_results_csv} not found — run `aggregate` first.",
              file=sys.stderr)
        return 2

    # ── Resolve compute_metrics.py ─────────────────────────────────────
    compute_metrics_script = args.compute_metrics_script
    if compute_metrics_script is None:
        # Default: sibling of this script, i.e. the benchmark pipeline root.
        compute_metrics_script = Path(__file__).resolve().parent / "compute_metrics.py"
    compute_metrics_script = Path(compute_metrics_script).resolve()
    if not compute_metrics_script.exists():
        print(f"ERROR: compute_metrics.py not found at {compute_metrics_script}.\n"
              f"       Pass --compute-metrics-script to point at it explicitly.",
              file=sys.stderr)
        return 2

    # ── Read prediction chain IDs from cycle_0/plan.json ───────────────
    # Only the prediction chain IDs are needed, to count Cα atoms per
    # chain for --chain-lengths.  compute_metrics.py is used here only
    # for the confidence metrics (pTM, ipTM, ipSAE, actifPTM, pLDDT,
    # PAE), none of which need a reference.  The layout detection
    # matches cmd_aggregate's.
    nested_cycle_0 = experiment_root / "cycle_0"
    if (nested_cycle_0 / "plan.json").exists():
        cycle_0_plan_path = nested_cycle_0 / "plan.json"
    elif (experiment_root / "plan.json").exists():
        cycle_0_plan_path = experiment_root / "plan.json"
    else:
        print(f"ERROR: no cycle_0 plan.json under {experiment_root} "
              f"(checked {nested_cycle_0}/plan.json and "
              f"{experiment_root}/plan.json)", file=sys.stderr)
        return 2
    cycle_0_plan = json.loads(cycle_0_plan_path.read_text())

    pred_rec_chain = cycle_0_plan.get("pred_receptor_chain", "A")
    pred_eff_chain = cycle_0_plan.get("pred_effector_chain", "B")

    # ── Read all_results_multicycle.csv and apply the filter ──────────
    with open(all_results_csv) as f:
        reader = csv.DictReader(f)
        base_rows = list(reader)
        base_fieldnames = list(reader.fieldnames or [])

    metric_col = args.metric_column
    threshold = args.rmsd_threshold

    # With skip_steering=true at cycle 0 there are no steered designs
    # to filter against, and the initial row is the only row.  It still
    # needs confidence metrics so the comparison table has something
    # to rank.
    _skip_steering_cycle0 = bool(cycle_0_plan.get("skip_steering"))
    if _skip_steering_cycle0:
        print("  cycle_0 plan has skip_steering=true — will include "
              "the initial row in metrics regardless of ra_eff filter")

    passing_rows = [
        r for r in base_rows
        if _passes_final_metrics_filter(
            r, args, metric_col, threshold, _skip_steering_cycle0,
        )
    ]
    print(f"Read {len(base_rows)} rows from {all_results_csv.name}")
    if args.populate_all:
        # See _passes_final_metrics_filter for why the dropped seeds
        # are needed.
        print("  Filter: --populate-all set — every row, "
              "reversion_dropped included")
    else:
        print(f"  Filter: {metric_col} < {threshold} Å"
              + (" AND steered_receptor_intact == 1" if args.require_intact else "")
              + " AND reversion_dropped != 1")
    print(f"  Passing: {len(passing_rows)}")

    # ── Resume support: load any existing output and skip finished rows
    output_csv: Path = args.output_csv
    if output_csv is None:
        output_csv = experiment_root / "all_results_multicycle_with_metrics.csv"
    output_csv = Path(output_csv).resolve()

    # Harvested from compute_metrics.py under a `steered_` prefix,
    # which never collides with a base column and records that the
    # value came from the STEERED prediction.  On a pose_holds row
    # these describe the pre-reversion pose, and harvest-reversions
    # fills the matching reverted_* metrics.
    #
    # The structural RMSD columns (rmsd_receptor, rmsd_effector_*,
    # n_receptor_ca, n_effector_ca) are deliberately NOT harvested.
    # all_results_multicycle.csv already carries the authoritative
    # per-design RMSDs with the correct truth-vs-pred chain mapping, in
    # receptor_aligned_effector_rmsd, independent_receptor_rmsd and
    # independent_effector_rmsd.  compute_metrics.py uses one
    # --receptor-chain and --effector-chain pair for both the
    # prediction and the reference, so its RMSDs are wrong wherever
    # those chains differ, as in 7QPX and 7QZD with truth B/C against
    # pred A/B.  Skipping them also drops --reference-pdb and its
    # per-design Kabsch fit, a large speed-up.
    METRIC_KEYS_RAW = [
        "avg_plddt", "complex_plddt",
        "ptm", "iptm", "pae_mean", "ipae", "pae_pass_frac",
        "ipsae_ab", "ipsae_ba", "ipsae_min",
        # 15Å iPSAE, the same Dunbrack machinery at a more permissive
        # PAE cutoff.  One compute_metrics.py call produces both
        # trios, loading the PAE matrix once.
        "ipsae_ab_15", "ipsae_ba_15", "ipsae_min_15",
        "actifptm", "af_rank_score",
        # Formerly computed after the fact, and only on the survivors,
        # by compute_interface_metrics.py and by
        # run_biophysical_metrics.py for interface_plddt.  Computing
        # them here scores every steered prediction as it lands.
        # intact_core and weighted_jaccard need --native-pdb, naming
        # this design's cycle_0 ground truth, and are blank without it.
        # interface_plddt needs only the model PDB.
        "interface_plddt",
        "intact_core",
        "weighted_jaccard",
        # Interface contacts and the contamination check, defined by
        # compute_interface_contacts and classify_mutation_reliance in
        # compute_metrics.py.  n_contact_residues and contact_residues
        # are populated on every row.  The contamination columns need a
        # mutation list, so they stay blank on cold-start and initial
        # rows.
        #
        # `mutated_contact_positions` is the authoritative
        # contamination signal.  Any mutated residue contacting the
        # effector invalidates the metrics, and only this column names
        # which positions those are.  The count beside it,
        # n_contacts_on_mutated_positions, cannot.
        "n_contact_residues", "contact_residues",
        "n_contacts_on_mutated_positions",
        "mutated_contact_positions",
        "contact_cutoff_used",
    ]
    METRIC_KEY_PREFIX = "steered_"
    prefixed_metric_keys = [METRIC_KEY_PREFIX + k for k in METRIC_KEYS_RAW]

    # Resume from the existing output CSV.
    existing_by_key: Dict[str, Dict] = {}
    if args.skip_existing:
        existing_by_key = _load_existing_metrics_rows(
            output_csv, prefixed_metric_keys,
        )
        if existing_by_key:
            print(f"  Resume: {len(existing_by_key)} rows already have metrics "
                  f"in {output_csv.name}")

    # ── Set up a scratch dir for per-row compute_metrics CSVs ──────────
    scratch_dir = experiment_root / "final_metrics_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # Built up front so the output CSV is consistent across partial
    # runs.
    augmented_fieldnames = list(base_fieldnames)
    for k in prefixed_metric_keys:
        if k not in augmented_fieldnames:
            augmented_fieldnames.append(k)
    # Derived columns added by the post-processing pass below.
    for k in ("steered_pae_cutoff_used",
              "steered_confidence_flag",
              "steered_error"):
        if k not in augmented_fieldnames:
            augmented_fieldnames.append(k)

    # ── Iterate over EVERY row in all_results_multicycle.csv ──────────
    # Rows failing the filter are still written out with empty
    # steered_* and reverted_* fields, so the joined CSV is a true
    # superset.  Only passing rows cost a subprocess.
    model_name = args.model
    joined_rows: List[Dict] = []
    n_computed = 0
    n_skipped_resume = 0
    n_skipped_filter = 0
    n_failed = 0
    passing_set = {id(r) for r in passing_rows}
    n_passing_total = len(passing_rows)
    passing_progress = 0  # 1-based counter for the printed progress

    for row in base_rows:
        # Preserved as-is.
        if id(row) not in passing_set:
            out_row = dict(row)
            for k in prefixed_metric_keys:
                out_row.setdefault(k, "")
            out_row.setdefault("steered_error", "")
            joined_rows.append(out_row)
            n_skipped_filter += 1
            continue

        passing_progress += 1

        out_row, status = _process_passing_row(
            row, args,
            prefixed_metric_keys, METRIC_KEYS_RAW, METRIC_KEY_PREFIX,
            existing_by_key, scratch_dir,
            compute_metrics_script, cycle_0_plan, cycle_0_plan_path,
            pred_rec_chain, pred_eff_chain,
            model_name, metric_col,
            passing_progress, n_passing_total,
        )
        joined_rows.append(out_row)
        if status == "resumed":
            n_skipped_resume += 1
        elif status == "computed":
            n_computed += 1
            # The resume contract: the output table is valid after
            # every computed row, so a job killed mid-loop leaves a
            # readable CSV the next run can resume from.  The layout is
            # the final writer's canonical Block A/B/C/End order, so
            # column order does not depend on how the job ended.
            _write_metrics_csv(output_csv, joined_rows)
        else:  # "failed"
            n_failed += 1

    # ── Post-processing ───────────────────────────────────────────────
    # steered_pae_cutoff_used, steered_confidence_flag and the unified
    # ranks.  These run after the harvest loop because the ranks need a
    # global view, and the threshold flags must treat computed and
    # resumed rows alike.
    pae_cutoff_used = float(args.pae_cutoff)
    flag_pass_frac_min     = float(args.flag_pass_frac_min)
    flag_iptm_min          = float(args.flag_iptm_min)
    flag_complex_plddt_min = float(args.flag_complex_plddt_min)
    flag_ipae_max          = float(args.flag_ipae_max)

    for r in joined_rows:
        r["steered_pae_cutoff_used"] = pae_cutoff_used
        r["steered_confidence_flag"] = _classify_steered_confidence_flag(
            r, flag_pass_frac_min, flag_iptm_min,
            flag_complex_plddt_min, flag_ipae_max,
        )

    # aggregate already set rank_by_ra_eff and
    # rank_by_composite_score, since both read only steering-pipeline
    # outputs.  Re-deriving them is harmless, because the values are
    # deterministic.
    _compute_unified_ranks(joined_rows)

    # Final write.  It covers the run where every row was
    # resume-skipped or failed, so the loop never reached the
    # incremental write, and it puts the steered_* metrics in Block B
    # next to the steering RMSDs rather than at the end.
    _write_metrics_csv(output_csv, joined_rows)

    print()
    print(f"Wrote {output_csv} ({len(joined_rows)} rows)")
    print(f"  computed now    : {n_computed}")
    print(f"  resumed         : {n_skipped_resume}")
    print(f"  failed          : {n_failed}")
    print(f"  filtered out    : {n_skipped_filter} (preserved with empty steered_* / reverted_*)")
    if n_failed == 0:
        # Tidy up scratch only on clean runs so failed rows remain
        # inspectable.
        try:
            shutil.rmtree(scratch_dir)
        except Exception:
            pass
    else:
        print(f"  scratch dir kept for inspection: {scratch_dir}")

    return 0 if n_failed == 0 else 1



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("aggregate",
                        help="Walk the tree and produce summary CSVs")
    pa.add_argument("--experiment-root", type=Path, required=True)
    pa.set_defaults(func=cmd_aggregate)

    pcfm = sub.add_parser(
        "compute-final-metrics",
        help="Run compute_metrics.py on every design in "
             "all_results_multicycle.csv that passes a final "
             "ranking filter (opt-in post-aggregation step)",
    )
    pcfm.add_argument("--experiment-root", type=Path, required=True,
                      help="Experiment root containing all_results_multicycle.csv "
                           "and cycle_0/plan.json")
    pcfm.add_argument("--rmsd-threshold", type=float, default=5.0,
                      help="Only compute metrics for designs whose "
                           "--metric-column value is strictly less than "
                           "this (default: 5.0 Å)")
    pcfm.add_argument("--metric-column", default="steered_ra_eff_vs_truth",
                      help="Column in all_results_multicycle.csv used for "
                           "the final filter (default: "
                           "steered_ra_eff_vs_truth)")
    pcfm.add_argument("--require-intact", dest="require_intact",
                      action="store_true", default=True,
                      help="Also require steered_receptor_intact == 1 (default: on)")
    pcfm.add_argument("--no-require-intact", dest="require_intact",
                      action="store_false",
                      help="Disable the steered_receptor_intact == 1 filter")
    pcfm.add_argument("--model", default="boltz2",
                      help="Model name passed to compute_metrics.py "
                           "(default: boltz2)")
    pcfm.add_argument("--compute-metrics-script", type=Path, default=None,
                      help="Path to compute_metrics.py "
                           "(default: sibling of this script)")
    pcfm.add_argument("--python-executable", default="python",
                      help="Python interpreter used to run compute_metrics.py "
                           "(default: 'python' — the interpreter on $PATH, "
                           "which inside the boltz container is the one "
                           "with numpy installed).  Use 'python3' or an "
                           "absolute path if your environment differs.")
    pcfm.add_argument("--pae-cutoff", type=float, default=10.0,
                      help="PAE cutoff (Å) for ipSAE.  Forwarded to "
                           "compute_metrics.py.  Default 10 Å follows "
                           "Dunbrack 2025; 15 is also reasonable for "
                           "permissive scoring.  Inspect steered_pae_pass_frac "
                           "in the output to see what fraction of "
                           "interface pairs survived this cutoff for "
                           "each design.")
    pcfm.add_argument("--contact-cutoff", type=float, default=5.0,
                      help="Heavy-atom distance cutoff (Å) for the "
                           "interface contact definition used by the "
                           "mutation-reliance check.  A receptor "
                           "residue is counted as a contact if any of "
                           "its heavy atoms is strictly closer than "
                           "this to any effector heavy atom.  "
                           "steered_n_contacts_on_mutated_positions is then set: "
                           "'clean' (contact set disjoint from the "
                           "row's steering mutations) or 'contaminated' "
                           "(at least one contact residue is a "
                           "steering mutation).  Default 5.0.")
    pcfm.add_argument("--flag-pass-frac-min", type=float, default=0.10,
                      help="Confidence flag: rows with steered_pae_pass_frac "
                           "below this get flagged as 'low_pass_frac' "
                           "(ipSAE summarising too few residue pairs to "
                           "be statistically reliable).  Default 0.10.")
    pcfm.add_argument("--flag-iptm-min", type=float, default=0.30,
                      help="Confidence flag: rows with steered_iptm below "
                           "this get flagged as 'low_iptm' (Boltz not "
                           "confident the interface is real).  "
                           "Default 0.30 — informally used as the AF-"
                           "Multimer 'no interaction' boundary.")
    pcfm.add_argument("--flag-complex-plddt-min", type=float, default=0.70,
                      help="Confidence flag: rows with steered_complex_plddt "
                           "below this get flagged as 'low_plddt' "
                           "(overall structure suspect).  Default 0.70.")
    pcfm.add_argument("--flag-ipae-max", type=float, default=15.0,
                      help="Confidence flag: rows with steered_ipae ABOVE "
                           "this (Å) get flagged as 'high_ipae' (Boltz "
                           "uncertain about inter-chain placement).  "
                           "Default 15.0.")
    pcfm.add_argument("--populate-all", dest="populate_all",
                      action="store_true", default=False,
                      help="Populate confidence and interface metrics on "
                           "EVERY row, not just the rows that pass the "
                           "ra_eff and intact filter.  Rows with "
                           "reversion_dropped=1 are included too.  A "
                           "dropped seed with no metrics is a hole in the "
                           "per-seed table, and a hole shifts the median "
                           "across seeds towards whichever seeds "
                           "survived.  Also needed for an AUROC analysis, "
                           "where the filtered-away rows are the "
                           "wrong-interface side of the curve.  Default: "
                           "off, so only filter-passing rows get metrics.")
    pcfm.add_argument("--output-csv", type=Path, default=None,
                      help="Output CSV path "
                           "(default: <experiment-root>/all_results_multicycle_with_metrics.csv)")
    pcfm.add_argument("--skip-existing", dest="skip_existing",
                      action="store_true", default=True,
                      help="Resume: skip rows that already have metrics "
                           "in the output CSV (default: on)")
    pcfm.add_argument("--no-skip-existing", dest="skip_existing",
                      action="store_false",
                      help="Force recomputation of every passing row")
    pcfm.add_argument("--subprocess-timeout", type=int, default=1800,
                      help="Per-design compute_metrics.py timeout in "
                           "seconds (default: 1800)")
    pcfm.add_argument("--verbose", action="store_true",
                      help="Print compute_metrics.py stdout on failure")
    pcfm.set_defaults(func=cmd_compute_final_metrics)

    paps = sub.add_parser(
        "aggregate-per-sequence",
        help="Collapse per-seed rows into one aggregated row per "
             "unique sequence (median + MAD on continuous metrics, "
             "majority vote on categoricals).  Reads "
             "all_results_multicycle_with_metrics.csv (or the "
             "unmetriced fallback) and writes raw_per_seed_results.csv "
             "+ aggregated_results.csv.  Run AFTER aggregate "
             "(and ideally after compute-final-metrics so confidence "
             "metrics are part of the aggregate).",
    )
    paps.add_argument("--experiment-root", type=Path, required=True)
    paps.set_defaults(func=cmd_aggregate_per_sequence)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
