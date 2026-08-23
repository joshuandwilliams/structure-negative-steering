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
    make_pathway_label,
    read_cumulative_mutations,
    read_mutations_tsv_full,
)
from reversion import _REVERTED_CONFIDENCE_FIELDS  # noqa: E402

try:  # noqa: E402
    from pipeline_thresholds import PipelineInternalThresholds as _PIT
    _INTACT_THRESHOLD = _PIT.default().intact_threshold
except Exception:
    _INTACT_THRESHOLD = 5.0


# Map of (internal key) -> (new CSV column name).  Only keys that need
# renaming appear here; others pass through unchanged.
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


# Internal keys that must be DROPPED at write time (they were
# duplicates or are now superseded).
_AGG_DROP = {
    "delta_receptor_aligned_effector_rmsd",  # trivially derivable
    "rank",                                  # legacy cycle-0 row index
    "rank_by_truth",                         # superseded by rank_by_ra_eff
    # The old "steered_*" fields the code used to write under those
    # exact names (when the base columns had been overwritten).  After
    # the finalize refactor, base columns are no longer overwritten,
    # so these duplicates are unnecessary.  Drop if present (defensive).
    "steered_ra_eff_vs_truth",
    "steered_independent_receptor_rmsd",
    "steered_independent_effector_rmsd",
    # Internal selection-only key, never user-facing
    "_active_ra_eff_for_selection",
}


# Continuous metrics that should be aggregated by median+MAD.  Mirrors
# the Block-B/C continuous columns from _aggregate_csv_fieldnames.
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
    # Per-prediction native-PDB-dependent metrics (P0-29 / P0-38),
    # moved from compute_interface_metrics.py / run_biophysical_metrics.py
    # so they aggregate alongside the Boltz-derived metrics.
    "interface_plddt",
    "weighted_jaccard",
)


# 0/1 categorical metrics aggregated by majority.
_AGG_BINARY_METRICS = (
    "receptor_intact",
    # intact_core: per-prediction Kabsch-aligned receptor-core RMSD
    # < threshold flag.  Binary 0/1 → majority across seeds.
    "intact_core",
)


# Integer count metrics.  These get median + MAD too — they're
# discrete but ordered, and median across N=3 still makes sense.
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
    """Build the two portable mutation-description strings from a
    list of (pos1, wt, mut) tuples.

    Returns (chimerax_spec, aa_changes):
      chimerax_spec:  "/A:5,12,37"  — copy-paste ready; user prepends
                      their own "#N" model number.  Positions are sorted
                      ascending and deduplicated.
      aa_changes:     "L5A V12K G37P" — space-separated wt+pos+mut
                      tokens, also sorted by position.

    Empty mutation list → ("", "").
    """
    if not mutations:
        return ("", "")
    # Sort by position and deduplicate: if the same position appears
    # twice (cumulative history across cycles where an ancestor touched
    # a position and a descendant touched it again), keep the first
    # occurrence in the chain order — which is the earliest edit, not
    # the most recent one.  Dedup by position only for the chimerax
    # spec; the aa-changes string keeps every entry because re-edits
    # are scientifically meaningful (even though the pipeline's
    # burnt-positions logic usually prevents them).
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
    """Apply _AGG_RENAME and _AGG_DROP to produce a row dict ready
    for CSV writing.  Also adds derived columns:
      - steered_passes_filter (computed from steered_receptor_intact
        and steered_ra_eff_vs_truth — populated by the writer if the
        downstream compute-final-metrics step recorded filter inputs)
      - reverted_passes_filter (likewise from reverted_receptor_intact)
      - reverted_receptor_intact (derived from reverted RMSDs)

    Pure dict transformation; does not touch other rows.
    """
    out: Dict = {}
    # First, drop the keys that should never appear in output
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

        # Derive the per-design label used by contaminated.json and
        # reversion_results.json.  Cycle-0 rows have pathway="cycle_0"
        # (a constant, not a per-design identifier); the per-design
        # identifier is in the "design" column (e.g. "design_04").
        # Cycle-1+ rows have pathway = the leaf label (e.g. "c0d05.c1d03").
        if cycle == 0:
            leaf_label = r.get("design", "")
        else:
            leaf_label = r.get("pathway", "")

        # For ANY row that went through reversion (pose_holds or any
        # dropped verdict), surface the set of mutated positions that
        # were contacting the effector in the steered prediction —
        # the contamination that triggered the reversion.
        if verdict != "":
            cdoc = _load_contaminated_doc(cycle, parent_pathway)
            for e in cdoc.get("contaminated", []):
                if e.get("label") == leaf_label:
                    pos = e.get("positions_to_revert") or []
                    r["steered_mutated_contact_positions"] = ",".join(
                        str(p) for p in sorted(pos)
                    )
                    break

        # For ANY reverted row (pose_holds, pose_collapses,
        # new_contamination), populate the reverted contact data
        # from reversion_results.json.  The per-seed aggregator
        # downstream needs the REAL contamination set per seed to
        # compute the per-position majority correctly.  Blanking
        # these columns for non-pose_holds rows (the pre-audit
        # behaviour) caused the aggregator to see empty contact sets
        # for seeds whose harvest actually flagged contamination,
        # inflating the pose_holds count and misclassifying the
        # outcome as pose_holds when some seeds actually
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

        # pose_holds: compute reverted_mutations_* from remaining mutations.
        # (reverted contact data was already populated above for all
        # reverted rows.)
        rev_data = _load_rev_results(cycle, parent_pathway)
        rev_entry = rev_data.get(leaf_label) or {}

        # Read positions_to_revert from contaminated.json (the
        # authoritative source — reversion_results.json does not
        # carry this field).  Fall back to reversion_plan.json
        # entries if contaminated.json lookup fails.
        positions_to_revert: set = set()
        cdoc = _load_contaminated_doc(cycle, parent_pathway)
        for e in cdoc.get("contaminated", []):
            if e.get("label") == leaf_label:
                positions_to_revert = set(
                    int(p) for p in (e.get("positions_to_revert") or [])
                )
                break
        if not positions_to_revert:
            # Fallback: reversion_plan.json entries also carry the list
            rp = _load_rev_plan(cycle, parent_pathway)
            for e in rp.get("entries", []):
                if e.get("label") == leaf_label:
                    positions_to_revert = set(
                        int(p) for p in (e.get("positions_to_revert") or [])
                    )
                    break

        # Cumulative mutation history.  For cycle-0 rows the leaf_label
        # is "design_NN" which parse_pathway_label cannot parse, so
        # read_cumulative_mutations would crash.  Read mutations.tsv
        # directly from the design's workdir instead.
        cum_muts: List[Tuple[int, str, str]] = []
        if cycle == 0:
            pdb_col = r.get("pdb", "")
            if pdb_col:
                mut_tsv = Path(pdb_col).parent / "mutations.tsv"
                cum_muts = read_mutations_tsv_full(mut_tsv)
        else:
            try:
                cum_muts = read_cumulative_mutations(
                    experiment_root, leaf_label)
            except Exception:
                cum_muts = []

        # Filter out reverted positions; for positions touched twice
        # (rare) keep the most recent edit
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
    """Row represents a clean steered design that survived everything:
    not contaminated AND steered prediction passes the structural
    filter (intact AND ra_eff < 5 Å).

    P0.4 (changed from earlier behaviour): the eligibility predicate
    used to gate on `steered_ipsae_min` being populated, as a proxy
    for "compute-final-metrics ran on this row, which it only does
    for ra_eff-passing rows".  That proxy breaks under
    `compute-final-metrics --populate-all`, which populates ipSAE
    for every non-dropped row including wrong-interface ones, and
    would otherwise let those rows into the composite ranking.  The
    proxy is now replaced with a direct ra_eff check against a
    hardcoded 5 Å threshold matching the steering pipeline's default
    (rmsd_threshold).  Rows missing ra_eff entirely are still
    excluded.
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
    # Direct ra_eff check (replaces the legacy ipSAE-presence proxy).
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
    """Row represents a pose_holds design whose reverted prediction
    passes the structural filter (intact AND ra_eff < 5 Å).

    Mirrors the ra_eff check in _row_is_clean_steered so the two
    eligibility predicates are consistent: a pose_holds row whose
    reverted prediction has terrible ra_eff (e.g. 25 Å) should not
    be ranking-eligible, just as a clean steered row with terrible
    steered ra_eff is excluded.  This was previously only enforced
    implicitly by the reversion harvest's intact filter, which does
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
    """Return the canonical value of `name` for ranking purposes.
    `name` is one of 'ra_eff_vs_truth', 'ipsae_min'.

    For clean steered rows, reads the steered_ value.
    For pose_holds rows, reads the reverted_ value.
    Returns None if the row is ineligible or the value is missing.
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
    """Composite ranker score (P0.3): true_jaccard − 0.05 · ra_eff_vs_truth.
    Higher is better.

    Eligibility matches _ranking_metric.  ra_eff is sourced from
    steered_ or reverted_ per the row type; true_jaccard is always
    sourced from steered_true_jaccard because the jaccard overlap is
    computed once on the steered prediction and not recomputed after
    reversion (the interface identity is carried by the designed
    sequence, not the steering mutations).

    Returns None if either component is missing / unparseable or the
    row is not ranking-eligible.
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
    """Add rank_by_ra_eff and rank_by_composite_score columns.  Both
    ranks are 1-based; rank 1 is best.  Eligible rows: clean steered
    (no contamination, passes filter) OR pose_holds (passes reverted
    filter).  Ineligible rows get blank.

    P0.3: rank_by_composite_score is the primary ranker.  Score is
    true_jaccard − 0.05 · ra_eff_vs_truth; higher is better.
    P0.4: rank_by_ipsae_min is dropped entirely.
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
    # missing either component (ra_eff or steered_true_jaccard) get a
    # blank rank — typically the ra_eff-only rows on which
    # compute-final-metrics never populated interface metrics.
    eligible = []
    for r in rows:
        v = _ranking_composite(r)
        if v is not None:
            eligible.append((v, id(r), r))
    # Sort descending (higher composite = better = lower rank)
    eligible.sort(key=lambda t: (-t[0], t[1]))
    for rank_idx, (_, _, r) in enumerate(eligible, start=1):
        r["rank_by_composite_score"] = rank_idx


def _agg_continuous(values: List[Optional[float]]) -> Tuple[Optional[float], Optional[float], int]:
    """Aggregate a list of per-seed values into (median, MAD, n_used).

    None / NaN / Inf entries are dropped.  MAD is the median of absolute
    deviations from the median; pairs naturally with median for small N
    (SD on N=3 is dominated by single outliers and gives wildly variable
    estimates run-to-run, MAD is robust).

    Returns (None, None, 0) if no usable values.
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
        # MAD undefined for N=1; report 0.0 to mean "no spread visible"
        return (med, 0.0, 1)
    mad = statistics.median(abs(v - med) for v in clean)
    return (med, mad, len(clean))


def _agg_majority_binary(values: List[Optional[int]]) -> Tuple[Optional[int], int, int]:
    """Aggregate a list of per-seed 0/1 values into
    (majority_value, n_positive, n_used).

    Majority threshold is ≥ ceil(N/2).  None / unparseable entries are
    dropped from the count (tolerates partial failures).  Returns
    (None, 0, 0) if no usable values.
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
    """Aggregate per-seed position sets into a per-position majority vote.

    A position is in the aggregate set iff ≥ ceil(N/2) seeds report it.
    Returns (aggregate_positions, position_to_seed_count, n_seeds_used).

    `position_to_seed_count` is exposed so the aggregator can record
    near-miss positions (e.g. a position that 1 of 3 seeds reported)
    as diagnostic context, not just the majority result.

    Empty input → ([], {}, 0).
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
    """Parse a comma-separated string of 1-based positions into a list
    of ints.  Tolerates blanks and non-integer tokens.  Used to convert
    contact_residues / mutated_contact_positions CSV strings back into
    sets for aggregation."""
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
    """Aggregate a list of per-seed rows (all sharing one unique
    sequence) into a single summary dict.

    `prefix` is "steered_" or "reverted_".  All metric column names
    in the per-seed rows are expected to start with this prefix; the
    aggregated output uses `<prefix><metric>_median`, `_mad`, `_n_used`
    for continuous; `<prefix><metric>_majority`, `_n_positive`,
    `_n_used` for binary; `<prefix><metric>_majority` (CSV string),
    `<prefix><metric>_per_seed_counts` (JSON map), `_n_used` for
    position sets.

    Identification fields (`design`, `pdb`, `cycle`, `pathway`) are
    pulled from the FIRST row.  Rows in the same sequence_group should
    share these by construction (same unique sequence → same design
    name minus the `_sN` suffix).  The aggregated `design` field is
    the bare name without the `_sN` suffix.

    Returns the aggregated dict.  Caller is responsible for merging
    aggregated steered + aggregated reverted records into one row.
    """
    if not per_seed_rows:
        return {}

    out: Dict[str, any] = {}

    # Continuous metrics → median + MAD
    for metric in _AGG_CONTINUOUS_METRICS + _AGG_INTEGER_METRICS:
        col = prefix + metric
        vals = [r.get(col) for r in per_seed_rows]
        med, mad, n_used = _agg_continuous(vals)
        out[f"{prefix}{metric}_median"] = "" if med is None else round(med, 4)
        out[f"{prefix}{metric}_mad"] = "" if mad is None else round(mad, 4)
        out[f"{prefix}{metric}_n_used"] = n_used

    # Binary metrics → majority vote
    for metric in _AGG_BINARY_METRICS:
        col = prefix + metric
        vals = [r.get(col) for r in per_seed_rows]
        majority, n_pos, n_used = _agg_majority_binary(vals)
        out[f"{prefix}{metric}_majority"] = "" if majority is None else majority
        out[f"{prefix}{metric}_n_positive"] = n_pos
        out[f"{prefix}{metric}_n_used"] = n_used

    # Position-set metrics → per-position majority
    for metric in _AGG_POSITION_SET_METRICS:
        col = prefix + metric
        position_sets = [_parse_position_csv(r.get(col, "")) for r in per_seed_rows]
        agg_positions, counts, n_used = _agg_majority_positions(position_sets)
        out[f"{prefix}{metric}_majority"] = ",".join(
            str(p) for p in agg_positions
        )
        # Per-position counts as a compact JSON map for diagnostics.
        # Empty → empty string (not "{}") so downstream readers can use
        # a simple truthiness test.
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
    metrics.  Mirrors classify_reversion_verdict but operates on the
    median / majority columns produced by _aggregate_per_sequence.

    Returns (outcome, reason).  Outcome is one of pose_holds,
    pose_collapses, new_contamination, no_reversion.

    no_reversion is returned when the aggregated record carries no
    reverted metrics at all (i.e. no seed in this group needed
    reversion: either cold-start skip_steering, or steering ran but
    no seed had contamination on mutated positions).

    The aggregated pose_holds outcome requires ALL of:
      1. The aggregated structural filter passes (intact_majority = 1
         AND median ra_eff < cutoff AND no position-majority contamination)
      2. At least ONE individual per-seed verdict says pose_holds
         (verdict_counts["pose_holds"] >= 1)

    Rationale for the any-seed gate (v8, replaces the earlier ceil(N/2)
    majority gate from Bug E): with `num_seeds = 3` and a per-seed
    pose_holds rate around 10-15% on this scaffold (measured from v7),
    majority-of-3 is architecturally unreachable even when several
    designs individually have one clean-pass seed.  The aggregated
    median/position-majority rules already exclude the "every seed
    failed differently" artefact that Bug E was originally aimed at:
    if no seed individually passes, pose_holds count == 0 and this
    gate short-circuits.  If exactly one seed passes and the other
    two fail in different ways, the aggregated position-majority
    check on `reverted_mutated_contact_positions_majority` only
    flags contamination when ≥ ceil(N/2) seeds contaminate at a
    shared position — so sporadic per-seed contamination at unrelated
    positions does NOT trigger it, which is what Bug E was worried
    about.  That concern is now handled by also requiring the
    per-seed failure distribution not to be a majority-failure:
    we additionally require pose_holds + no_data
    > each of (pose_collapses, new_contamination).
    In words: among seeds that produced a verdict, the "pass or no-op"
    count must exceed either of the failure counts.

    `verdict_counts` is the {verdict: count} dict produced by
    _per_seed_verdict_breakdown.  Keys: pose_holds, pose_collapses,
    new_contamination, no_data, clean_steered.
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
    # Pose-passing-equivalent count: explicit per-seed pose_holds
    # PLUS clean_steered (steered prediction was structurally fine
    # AND had no contamination, so no reversion was needed — the
    # steered structure is the final verdict for that seed).
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

    # Contamination check on the aggregated reverted contact set.
    # A position only appears in `reverted_mutated_contact_positions_majority`
    # if ≥ ceil(N/2) seeds individually contacted at that position, so
    # sporadic per-seed contamination at unrelated positions does not
    # trigger this — which is the Bug E "every seed failed differently"
    # artefact we still need to guard against.
    #
    # notes11 gating: Only positions inside the design region ∪ true
    # interface count as contamination, matching the pre-reversion
    # policy.  Positions outside the gated set (e.g. off-interface
    # "reaching bit" contacts) are acceptable by construction and
    # should not downgrade the aggregated verdict.
    contam_majority = agg.get(
        "reverted_mutated_contact_positions_majority", "")
    if contam_majority:
        # Parse the majority list and filter to gated positions only.
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

    # Any-seed gate: at least one seed must individually pass —
    # either as an explicit pose_holds verdict OR as clean_steered
    # (no reversion needed because steered prediction had zero
    # contamination on mutated positions).  This is the weaker
    # gate that replaces the Bug E majority-of-N gate; see
    # docstring for rationale.
    if nph_eff < 1:
        # Zero seeds passed individually.  Choose downgraded verdict
        # based on plurality of failure modes; ties favour
        # pose_collapses as the more conservative call.
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

    # Bug-E residual guard: if the minority-of-failures is still the
    # majority of seeds in either direction, that's a weak signal —
    # we need the number of seeds that didn't individually fail
    # (pose_holds + clean_steered + no_data) to be at least as large
    # as each failure count.  This prevents "1 passes, 2 fail the
    # same way (but at different positions so no position reaches
    # majority)" from being called pose_holds.
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
    and return {verdict: count}.  Counts seeds in five buckets:
    pose_holds, pose_collapses, new_contamination, no_data,
    clean_steered.  Feeds into n_pass (= pose_holds + clean_steered)
    and into the aggregate outcome classifier.

    The authoritative per-seed verdict is the `reversion_verdict`
    column, which was set by `classify_reversion_verdict` in
    cmd_harvest_reversions on the fresh reversion_results.json data
    (before the aggregate CSV blanked contact columns for non-pose_holds
    rows in _populate_reverted_mutations).  We prefer that column when
    present and fall back to re-deriving the verdict from the raw
    reverted_* columns only when it isn't.

    The `clean_steered` bucket (added later) covers the case where
    the steered prediction had ZERO contamination on the mutated
    positions, so the reversion pass was correctly skipped — there
    was nothing to revert.  These rows have a blank reversion_verdict
    AND blank reverted_* columns AND
    steered_n_contacts_on_mutated_positions == 0.  Without this
    bucket they were misclassified as no_data, which incorrectly
    demoted otherwise-strong designs in the per-sequence triage
    (notes12 / chat 'rank-1 issue').  clean_steered is treated as
    pose-passing-equivalent in _classify_outcome and in
    the cross-sequence tier classifier.

    Without the verdict-column preference for the explicit-verdict
    cases, rows whose reverted_mutated_contact_positions column was
    blanked for display would be misclassified here as pose_holds,
    inflating the pose_holds count.
    """
    counts = {
        "pose_holds": 0,
        "pose_collapses": 0,
        "new_contamination": 0,
        "no_data": 0,
        "clean_steered": 0,
    }
    for r in per_seed_reverted_rows:
        # Prefer the authoritative verdict column when present.  Map
        # any pose_holds-but-dropped states (e.g. pose_holds_not_intact,
        # missing_verdict) to pose_collapses for the count.
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
            # Unknown non-empty verdict string — surface as no_data so
            # it's visible but doesn't silently get bucketed.
            counts["no_data"] += 1
            continue

        # Verdict column is blank.  Two sub-cases:
        #
        # (i)  Steered prediction had ZERO contamination on mutated
        #      positions, so reversion was skipped intentionally.
        #      The steered structure is the final result for this
        #      seed — count it as clean_steered (pass-equivalent).
        #
        # (ii) Reversion was attempted but the row is missing the
        #      reverted_* fields needed to classify it.  Re-derive
        #      from raw reverted_* columns; fall back to no_data
        #      if anything is missing or unparseable.
        steered_n_contacts_str = r.get(
            "steered_n_contacts_on_mutated_positions", "")
        try:
            steered_n_contacts = int(steered_n_contacts_str)
        except (TypeError, ValueError):
            steered_n_contacts = None

        # Heuristic for (i): steered row is intact, has zero
        # contamination, and the reverted_* fields are entirely
        # blank (no attempt was made).  Require the steered row
        # itself to also be structurally OK (intact + ra_eff in
        # range) — otherwise a "no contamination" steered miss
        # would silently count as a pass.
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

        # Fall back to re-deriving from raw columns for rows that
        # never went through reversion (verdict column blank).
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
    """Return the column order for all_results_multicycle.csv (and the
    joined all_results_multicycle_with_metrics.csv).  Three logical
    blocks (A: identification, B: steering, C: reversion) plus a tail
    of selection / ranks.  Any row keys not in this list get appended
    (sorted) at the very end so unexpected fields aren't silently
    dropped.

    The Block B `steered_*` confidence columns are populated by
    compute-final-metrics; they're listed here so that whether or not
    compute-final-metrics has run, the column order in the produced
    CSV is the same and consumers can rely on it.
    """
    preferred = [
        # ── A. Identification ────────────────────────────────────
        "cycle", "pathway", "design",
        # P0 audit (multi-seed): per-seed grouping metadata.  Two
        # rows with the same sequence_group differ only by Boltz
        # seed; per-sequence aggregation collapses them.
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
        # that contacts the effector invalidates the metrics, so the
        # count (n_contacts_on_mutated_positions) and the list
        # (mutated_contact_positions) together identify contamination.
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
        # Reverted contamination check: any remaining mutation
        # contacting the effector in the reverted prediction.
        # Should be empty for pose_holds by definition of the verdict.
        "reverted_mutated_contact_positions",
        # Reverted contact analysis (from harvest-reversions)
        "reverted_n_contact_residues",
        "reverted_contact_residues",
        "reverted_contact_cutoff_used",
        # Reverted confidence suite (populated for pose_holds rows
        # by harvest-reversions; blank otherwise)
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

    # Discover every cycle dir and every pathway under it.
    # The submit script uses two different layouts depending on
    # --n-cycles:
    #   --n-cycles 1: cycle-0 files (plan.json, steered_results.csv,
    #       steered/, initial_prediction.pdb) live directly in the
    #       experiment root.  There is no cycle_0/ subdirectory.
    #   --n-cycles 2+: cycle-0 files live under experiment_root/cycle_0/
    #       and additional cycle_N/ subdirectories are created as the
    #       chain runs.
    # Detect which layout we're in by checking for the canonical
    # cycle-0 artefact (plan.json) in each location.
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

    # Cycles 1+ — pathway_<label> subdirs under cycle_N/
    for cycle_dir in sorted(experiment_root.glob("cycle_*")):
        if not cycle_dir.is_dir():
            continue   # skip e.g. cycle_statistics.csv which the glob also matches
        if cycle_dir.name == "cycle_0":
            continue
        cycle_num = int(cycle_dir.name.split("_")[1])
        for pdir in sorted(cycle_dir.glob("pathway_*")):
            label = pdir.name[len("pathway_"):]
            pj = pdir / "plan.json"
            if not pj.exists():
                continue
            pl = json.loads(pj.read_text())
            pathways.append({
                "label": label,
                "cycle": cycle_num,
                "workdir": str(pdir),
                "parent_pathway": pl.get("parent_pathway"),
                "n_designs": len(pl.get("designs", [])),
                "exhausted": pl.get("exhausted", False),
            })

    (experiment_root / "pathways.json").write_text(
        json.dumps(pathways, indent=2)
    )

    # Look up the wild-type baseline ra_eff so we can compute a
    # consistent delta_receptor_aligned_effector_rmsd column for
    # cycle 1+ rows (cycle 0 already has it from steered_results.csv,
    # using the same baseline).
    cycle0_plan = json.loads((cycle_0_dir / "plan.json").read_text())
    wt_ra_eff = cycle0_plan.get("initial_receptor_aligned_effector_rmsd")
    # Prediction receptor chain ID, used to build the
    # mutations_chimerax column.  Default "A" if the plan doesn't
    # record it (old runs).
    pred_rec_chain_for_spec = cycle0_plan.get("pred_receptor_chain", "A")

    # Cycle 0's kickoff (if it ran) writes pose distances + selection
    # info into cycle_0/passing.json.  Index by design index so we
    # can merge it onto the steered_results.csv rows.
    cycle0_passing_path = cycle_0_dir / "passing.json"
    cycle0_extra_by_design: Dict[str, Dict] = {}
    if cycle0_passing_path.exists():
        cp = json.loads(cycle0_passing_path.read_text())
        sel_idxs_c0 = {s["design_idx"] for s in cp.get("selected", [])}
        for c in cp.get("all_candidates", []):
            cycle0_extra_by_design[c["design"]] = {
                "min_dist_to_upstream": c.get("min_dist_to_upstream"),
                "selected_for_next_cycle": 1 if c["design_idx"] in sel_idxs_c0 else 0,
                # Reversion fields — blank for non-reverted designs.
                # Note: steered_* base columns are preserved in the
                # row's existing receptor_aligned_effector_rmsd /
                # independent_*_rmsd fields and will be renamed by
                # _translate_aggregate_row at write time.  Reverted
                # values live only in dedicated reverted_* keys.
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

    # Flat CSV: every prediction from every cycle, ranked by ra_eff vs truth
    rows = []
    for p in pathways:
        wd = Path(p["workdir"])
        # Cycle 0 has its own steered_results.csv from the single-cycle script
        if p["label"] == "cycle_0":
            csv_path = wd / "steered_results.csv"
            if not csv_path.exists():
                # Skip_steering case with a pre-patch driver: no CSV
                # was written, but the cold-start prediction + plan.json
                # still exist.  Reconstruct a single initial row from
                # them so the comparison table gets SOMETHING for this
                # complex.  This path fires for existing 7B1I-style
                # workdirs that were aggregated before the patch.
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
                    # Interface fields: only present if interface
                    # analysis ran (not the case for all skip types)
                    initial_wrong = plan_data.get("initial_wrong_interface_idx") or []
                    true_set = plan_data.get("true_interface_idx") or []
                    # P0 jaccard-empty-wrong fix: only the truth set is
                    # required.  Empty initial_wrong gives jaccard = 0,
                    # not undefined.  See same fix in
                    # boltz2_negative_steering.py _write_initial_only_csv.
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
                        # P0 audit: initial baseline is a single
                        # prediction (no steering, no multi-seed),
                        # so it gets its own pseudo-group with
                        # seed_index 0 — keeps grouping logic uniform.
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
                    # Merge in min_dist + selected info from kickoff,
                    # if available.  The 'initial' baseline row has
                    # no design entry so it stays blank.
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
                    # Mutation strings: parse mutations.tsv from the
                    # design dir (derived from the pdb path).  The
                    # 'initial' baseline row(s) have no mutations so
                    # both columns stay empty.  Multi-seed cold-start
                    # produces design names "initial", "initial_s1",
                    # "initial_s2", ... — all are baseline rows with
                    # zero mutations and no mutations.tsv (notes12).
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

        # Cycles 1+: read passing.json (which contains all candidates with metrics)
        passing_path = wd / "passing.json"
        if not passing_path.exists():
            continue
        passing = json.loads(passing_path.read_text())
        sel_idxs = {s["design_idx"] for s in passing.get("selected", [])}
        # p["label"] is the PARENT pathway (e.g. "c0d05"); each candidate
        # in this directory becomes a LEAF pathway by appending its
        # (cycle, design_idx) to the parent.
        parent_label = p["label"]
        for c in passing.get("all_candidates", []):
            leaf_label = make_pathway_label(parent_label, p["cycle"], c["design_idx"])
            ra_eff = c["receptor_aligned_effector_rmsd_vs_truth"]
            # delta vs the wild-type baseline (positive = improvement,
            # i.e. effector pulled closer to the true site)
            delta = (wt_ra_eff - ra_eff) if wt_ra_eff is not None else ""

            # The full result.json content from compute-distances —
            # contains every per-design metric we want to surface.
            # Older runs (pre-result-passthrough fix) won't have this
            # embedded; fall back to reading result.json directly from
            # the design directory so the aggregator is self-healing.
            result_full = c.get("result")
            if not result_full:
                rj_path = Path(c["pdb"]).parent / "result.json"
                if rj_path.exists():
                    try:
                        result_full = json.loads(rj_path.read_text())
                    except Exception:
                        result_full = {}
                else:
                    result_full = {}

            row = {
                "pathway": leaf_label,
                "parent_pathway": parent_label,
                "cycle": p["cycle"],
                "design": c["design"],
                # P0 audit (multi-seed): expose sequence_group +
                # seed_index in the per-seed CSV so the per-sequence
                # aggregator can group on them without parsing the
                # _sN suffix in the design name.
                "sequence_group": c.get("sequence_group", ""),
                "seed_index": c.get("seed_index", ""),
                # Prefer total_mutations; fall back to legacy
                # n_mutations key for old runs (where it meant
                # per-cycle mutation count, not cumulative).
                "total_mutations": result_full.get(
                    "total_mutations",
                    result_full.get("n_mutations", ""),
                ),
                "receptor_aligned_effector_rmsd": ra_eff,
                "delta_receptor_aligned_effector_rmsd": delta,
                "independent_receptor_rmsd": c["independent_receptor_rmsd"],
                "independent_effector_rmsd": c["independent_effector_rmsd"],
                "receptor_intact": 1 if c["receptor_intact"] else 0,
                "wrong_jaccard": result_full.get("wrong_jaccard", ""),
                "n_shared_wrong": result_full.get("n_shared_wrong", ""),
                "true_jaccard": result_full.get("true_jaccard", ""),
                "n_shared_true": result_full.get("n_shared_true", ""),
                "n_design_interface_residues": result_full.get(
                    "n_design_interface_residues", ""),
                "min_dist_to_upstream": c["min_dist_to_upstream"],
                "selected_for_next_cycle": 1 if c["design_idx"] in sel_idxs else 0,
                "status": result_full.get("status", ""),
                "pdb": c["pdb"],
                # Reversion columns — populated when iterate-collect-
                # finalize ran a reversion pass over this design.  Rows
                # that were never reverted (non-contaminated) leave
                # these blank.  Rows that were dropped
                # (pose_collapses, new_contamination) carry
                # reversion_dropped=1 and DO NOT flow to next cycle.
                # The base columns above hold the steered values
                # untouched; reverted values live ONLY in the
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
            # Cumulative mutations walked up the ancestry chain.
            cum_muts = read_cumulative_mutations(experiment_root, leaf_label)
            cx, aa = format_mutation_strings(
                cum_muts, chain_id=pred_rec_chain_for_spec,
            )
            row["mutations_chimerax"] = cx
            row["mutations_aa"] = aa
            rows.append(row)

    out_csv = experiment_root / "all_results_multicycle.csv"
    if rows:
        # ── Translate internal dict keys to the new clean CSV schema ──
        # Internal keys preserve historical names so upstream code keeps
        # working.  Here at the write boundary we rename to a clean
        # schema with consistent steered_/reverted_ prefixes, drop
        # redundant columns, and add new derived columns.
        rows = [_translate_aggregate_row(r) for r in rows]

        # ── Compute reverted_mutations_chimerax / _aa / _total for ──
        # pose_holds rows: cumulative mutation set minus the positions
        # that were reverted.  All other rows leave these blank.
        # The reversion data lives in cycle_N/(pathway_<label>/)?
        # reversion_results.json keyed by pathway label.
        _populate_reverted_mutations(rows, experiment_root)

        # ── Compute unified ranks: for each ranking-eligible row, ──
        # use the canonical metric source (steered or reverted) per
        # the eligibility rules.  Both ranks are 1-based; rank 1 is
        # best.  Blank for ineligible rows.
        _compute_unified_ranks(rows)

        # ── Sort rows: cycle ascending, then initial baseline first ──
        # within cycle 0, then by rank_by_composite_score ascending
        # (blank ranks at the bottom).  P0.4: this sort key was
        # rank_by_ipsae_min; since that column is dropped, we key on
        # the new primary ranker instead.
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
# compute-final-metrics: run compute_metrics.py on every design in
# all_results_multicycle.csv that passes a final ranking filter.
# ───────────────────────────────────────────────────────────────────────
def _count_ca_per_chain(pdb_path: Path) -> "Dict[str, int]":
    """Count Cα atoms per chain ID from a PDB file.  Stdlib only —
    keeps this subcommand runnable outside the numpy/gemmi container."""
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
    """Collapse per-seed rows into one aggregated row per unique sequence.

    Reads the per-seed CSV (all_results_multicycle_with_metrics.csv if
    compute-final-metrics has run, else all_results_multicycle.csv),
    groups rows by `sequence_group`, and writes:

      raw_per_seed_results.csv         — copy of the input, untouched
                                         (renamed for clarity)
      aggregated_results.csv           — one row per unique sequence,
                                         median + MAD on continuous,
                                         majority on categorical, plus
                                         the verdict re-derived from the
                                         aggregated reverted record and
                                         per-seed verdict counts

    Aggregation rules (per P0 audit decisions):
      - continuous metrics      → median + MAD across seeds
      - 0/1 categoricals        → majority vote, ≥ ceil(N/2)
      - position-set categoricals → per-position majority
      - canonical seed (for `pdb`):
          * odd N  → seed at the median ra_eff
          * even N → seed with the WORSE (higher) ra_eff among the
                     two middle seeds, plus a warning column
          * N=1    → that seed, no warning

    Reversion-side aggregation:
      - Reverted metrics in the per-seed CSV are populated only on
        rows that flowed through the reversion pass.  The aggregator
        deduplicates reverted records by sequence_hash via the
        `reversion_plan.json` manifest, aggregates per unique reverted
        sequence, then maps the aggregate back to every steered
        sequence whose contaminated state pointed at that reverted
        hash.  This matches the user's stage-4 spec: one reverted
        result mapped onto several steering results.
    """
    experiment_root: Path = args.experiment_root.resolve()
    if not experiment_root.is_dir():
        print(f"ERROR: {experiment_root} not a directory", file=sys.stderr)
        return 2

    # Prefer the with_metrics CSV (if compute-final-metrics has run)
    # because we want the aggregated record to reflect confidence
    # metrics too.  Fall back to the unmetriced CSV with a warning.
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
    # See classify_reversion_verdict (reversion.py) and the notes11
    # policy: only positions inside the design region ∪ true interface
    # count as contamination.  cycle_0/plan.json stores the indices
    # 0-based; convert to 1-based to match the coordinate system of
    # reverted_mutated_contact_positions_majority.  If plan.json can't
    # be located (unusual layouts / very old runs), the gate is
    # disabled and legacy behaviour prevails.
    gating_1b: Set[int] = set()
    # cold-start trigger threshold (from plan.json).  Used by the
    # ranking-eligibility gate below so that cold-start aggregates
    # whose ra_eff is in (5.0, rmsd_threshold] still get a rank
    # (matches the cold-start all-clean trigger in
    # boltz2_negative_steering.py).  Defaults to 5.0 (legacy
    # behaviour) when no plan can be loaded.
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
    # Provides a stable filename downstream that means "every seed's
    # raw outputs", regardless of whether compute-final-metrics ran.
    raw_csv = experiment_root / "raw_per_seed_results.csv"
    shutil.copy2(per_seed_csv, raw_csv)
    print(f"  wrote {raw_csv.name} (copy of {per_seed_csv.name})")

    # ── Group rows by (cycle, pathway, sequence_group) ────────────────
    # cycle and pathway scope the sequence_group: two designs in
    # different cycles or pathways with the same sequence_group integer
    # are NOT the same unique sequence.  Empty sequence_group means a
    # row that pre-dates the multi-seed plumbing (or the initial
    # baseline); each such row is its own group of size 1.
    from collections import defaultdict
    groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    singletons: List[Dict] = []
    for r in rows:
        sg = r.get("sequence_group", "")
        if sg in ("", None):
            # Initial-baseline rows or pre-multi-seed rows: treat each
            # as its own singleton group with no aggregation.
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
        """Pick the canonical seed for diagnostics.  Returns (row, warning).

        Sort group by steered_ra_eff_vs_truth ascending (NaN/missing →
        infinity, so they sort to the end).  Pick:
          - odd N: index (N-1)//2 (the middle)
          - even N: index N//2  (the upper-middle, i.e. the WORSE of
            the two middle seeds — conservative)
          - N=1: that seed
        Returns the canonical row and a warning string (empty if no
        warning needed).
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

        # Bare design name (strip _sN suffix if present).  Falls back to
        # the canonical seed's design field if no suffix is present.
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

        # Reverted aggregation — only run if any group member has
        # reversion_verdict populated (i.e. the steering pipeline
        # flagged this sequence as contaminated and reversion ran).
        any_reverted = any(
            (r.get("reversion_verdict") or "").strip() != ""
            for r in group
        )
        # Per-seed verdict breakdown is always computed: it drives both
        # the outcome classifier (when reversion ran) and n_pass (which
        # is path-agnostic — pose_holds + clean_steered regardless of
        # whether reversion was triggered).
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
            # No seed needed reversion (cold-start skip_steering, OR
            # steering ran but no seed had contamination on mutated
            # positions).  Outcome is no_reversion; n_pass is still
            # derived from per-seed verdicts below — a wrong-placement
            # no-contamination group will have all seeds = no_data and
            # n_pass = 0 (NOT n_seeds), correctly landing in tier none.
            agg["outcome"] = "no_reversion"
            agg["outcome_reason"] = ""

        # n_pass: how many seeds passed structural+contamination filters.
        # Counted as pose_holds (reversion succeeded) + clean_steered
        # (reversion correctly skipped because steered already passed).
        agg["n_pass"] = (verdict_counts["pose_holds"]
                        + verdict_counts["clean_steered"])

        aggregated_rows.append(agg)

    # Singletons (typically the cycle-0 initial baseline) — pass
    # through with n_seeds=1 and no aggregation suffixes on the
    # value columns; they go in as-is so the aggregated CSV still
    # carries them.  We DO populate the canonical_pdb field so
    # consumers don't have to special-case singletons.
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
    # The ranker now operates on the median-aggregated rows.  We use
    # the existing _ranking_composite logic but on the aggregated
    # column names (e.g. steered_ra_eff_vs_truth_median in place of
    # steered_ra_eff_vs_truth).  Wrap in an adapter:
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
        # Cold-start all-clean aggregates (notes12) have
        # design == "initial" AND outcome == "no_reversion".
        # They MUST be ranking-eligible — they're the final output
        # for cold-start binders.  We admit them via a slightly looser
        # ra_eff cap (matching the cold-start trigger threshold from
        # plan.json, default 5.0 for legacy compatibility).  Steered
        # aggregates use the legacy 5.0 cap unchanged.
        is_cold_start = (
            r.get("design") == "initial"
            and (r.get("outcome") or "").strip() == "no_reversion"
        )
        if r.get("design") == "initial" and not is_cold_start:
            # design=="initial" without no_reversion outcome was the
            # legacy single-row baseline used by extract_passing.py
            # to sanity-check; preserve that exclusion to avoid
            # promoting reconstructed initial rows from old workdirs.
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
    # Build the column list deterministically so the CSV layout is
    # stable across runs.  Block order: identification, steered,
    # reverted, verdict + ranks.  Any unexpected keys (e.g. from
    # singleton passthrough) get appended in sorted order at the end.
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
    # P0.2 + P0-audit: --populate-all truly means "populate every
    # row" — including rows dropped by reversion verdict routing.
    # Per-seed aggregation requires every seed's steered_*
    # confidence metrics so the median across seeds is meaningful;
    # skipping reversion_dropped rows would leave the dropped
    # seeds as holes in the per-seed CSV and bias the aggregate.
    if args.populate_all:
        return True
    # Default (no --populate-all): skip rows dropped by reversion
    # verdict routing — confidence metrics on a discarded
    # prediction are wasted work.  pose_holds rows are NOT
    # dropped (reversion_dropped == 0); we still compute steered_*
    # cm metrics on them so the CSV shows what the steered
    # prediction looked like before reversion.
    try:
        if int(row.get("reversion_dropped", "0") or "0") == 1:
            return False
    except (TypeError, ValueError):
        pass
    # Skip_steering exemption: include the initial row(s) if their
    # parent workdir had skip_steering=true.  Applies only to
    # cycle-0 initial baselines (cycle-1+ rows never have
    # design=="initial").  Multi-seed cold-start (notes12) writes
    # additional rows with design names "initial_s1", "initial_s2",
    # ... — all are baseline rows that need confidence metrics.
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
    """Stable per-design resume key (pdb path is unique per design)."""
    # Resume key is the absolute pdb path because the `pathway` field
    # is shared across all cycle-0 designs (they all get the literal
    # string "cycle_0" from cmd_aggregate, which would collide).  The
    # pdb path is unique per design.  Falls back to pathway then to
    # an empty string for safety.
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
            # Only treat as "done" if at least one metric field is non-empty.
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
    """Classify steered confidence flag (ok, single trigger, or multiple); '' for rows without metrics."""
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

    # Resume.  Take the CURRENT base row (which may have fresh
    # columns like mutations_aa, rank_by_truth, or new derived
    # columns that a previous run didn't have) and overlay just
    # the steered_* / reverted_* metric values from the cached row onto it.  This
    # prevents the resume path from stomping fresh base columns
    # with stale data from an older output CSV.
    if args.skip_existing and pathway_key in existing_by_key:
        cached = existing_by_key[pathway_key]
        merged = dict(row)  # start from fresh base row
        for k in prefixed_metric_keys:
            v = cached.get(k, "")
            if v not in ("", None):
                merged[k] = v
        # Also carry over steered_error if it was set (usually blank
        # on success, but preserve failure state for diagnostics).
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
    # confidence_<stem>.json, pae_<stem>.npz, plddt_<stem>.npz
    # alongside each PDB.  The steering pipeline copies the chosen
    # Boltz output to `prediction.pdb` but leaves those sidecar
    # files nested under Boltz's native output layout (e.g.
    # `<design>/predictions/<stem>/pdb/<stem>_model_0.pdb` with
    # `confidence_<stem>_model_0.json` next to it).  If we pointed
    # --prediction-dir at `prediction.pdb`'s parent we'd get
    # metrics with ptm=0, iptm=0, pae=0 (only pLDDT would work,
    # via the B-factor-column fallback).  Instead, locate the
    # nested sidecar PDB and point the parser at its directory.
    sidecar_pdb = _locate_boltz_sidecar_pdb(pdb_path)
    if sidecar_pdb is not None:
        pred_dir = sidecar_pdb.parent
        lookup_stem = sidecar_pdb.stem
        metric_pdb = sidecar_pdb
    else:
        pred_dir = pdb_path.parent
        lookup_stem = pdb_path.stem
        metric_pdb = pdb_path

    # Determine chain lengths from the PDB we'll actually parse,
    # so CA counts match whatever compute_metrics.py ingests.
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

    # One output CSV per row, lives in the scratch dir.  Use a
    # short, filesystem-safe key derived from cycle + design name
    # rather than the full pdb path.
    cyc = row.get("cycle", "0")
    des = row.get("design", "unknown")
    safe_key = f"cycle{cyc}_{des}".replace("/", "_").replace(" ", "_")
    per_row_csv = scratch_dir / f"{safe_key}.csv"

    # Look up this row's cumulative mutation set.  After
    # cmd_aggregate's _translate_aggregate_row pass, the column
    # is named `steered_mutations_chimerax` (renamed from the
    # legacy `mutations_chimerax`).  We read the post-rename name
    # with a fallback to the legacy name for any path that
    # bypasses _translate_aggregate_row.  Format is
    # "/A:5,26,44,46" (chain letter + 1-based positions).
    # compute_metrics.py's parse_mutated_positions accepts this
    # form directly, so we forward the string as-is.  Blank string
    # (cold start, no mutations) is fine: compute_metrics.py will
    # emit the contact columns with blank mutated_contact_positions.
    #
    # Bug history: this previously read only `mutations_chimerax`,
    # which after the _AGG_RENAME refactor is always blank after
    # aggregate runs.  Consequence: compute_metrics.py ran without
    # --mutated-positions on every row, blanking the
    # steered_mutated_contact_positions and
    # steered_n_contacts_on_mutated_positions columns for every
    # design — INCLUDING the ones that HAD been correctly populated
    # by _populate_reverted_mutations in cmd_aggregate (which
    # compute-final-metrics runs after).
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
        # v8 Level 1: effector atom filter from cycle_0 plan.json.
        # compute_metrics.py reads the 'effector_interface_atoms'
        # key directly; no materialised filter file needed.
        "--effector-atom-filter-json", str(cycle_0_plan_path),
    ]
    # Per-prediction native-PDB-dependent metrics (intact_core +
    # weighted_jaccard).  cycle_0/plan.json carries the absolute
    # path to the ground-truth PDB under "ground_truth" — pass it
    # through so compute_metrics.py can populate those columns
    # alongside the PAE-derived ones.  When the cycle_0 plan
    # doesn't carry a ground_truth (defensive), we just omit the
    # flag and the metrics columns come out blank.
    _native_pdb = cycle_0_plan.get("ground_truth", "")
    if _native_pdb:
        cmd.extend(["--native-pdb", str(_native_pdb)])
    if mutations_cx:
        cmd.extend(["--mutated-positions", mutations_cx])
    # No --reference-pdb: we only want confidence metrics here.
    # Structural RMSDs come from all_results_multicycle.csv (base
    # columns), which already uses the correct truth-vs-pred
    # chain mapping.

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

    # Read back the per-row CSV.  If multiple models were picked
    # up (shouldn't happen with one-design-per-dir, but defend
    # against it), select the row whose model_name matches the
    # pdb stem we asked about.
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
    """For every row in all_results_multicycle.csv that passes the
    final ranking filter, shell out to compute_metrics.py to compute
    confidence / interface / structural-RMSD metrics, and write a
    single joined CSV alongside all_results_multicycle.csv.

    This is an opt-in post-aggregation step.  The fast `aggregate`
    pass stays stdlib-only; the heavy Kabsch/numpy work only runs
    when explicitly requested.
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
    # We only need the prediction chain IDs (to count Cα atoms per
    # chain when building --chain-lengths for compute_metrics.py).
    # The reference PDB and truth chain IDs are NOT needed: structural
    # RMSDs are already in all_results_multicycle.csv from the
    # steering pipeline, and compute_metrics.py is only being used
    # here for confidence metrics (pTM / ipTM / ipSAE / actifPTM /
    # pLDDT / PAE), none of which depend on a reference.
    # Detect cycle-0 layout (nested under cycle_0/ for multi-cycle runs,
    # flat in the experiment root for --n-cycles 1 runs).  Same logic
    # as cmd_aggregate.
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

    # Detect skip_steering at cycle 0 — if the cycle-0 plan has
    # skip_steering=true (or no `designs` array at all), the
    # filter criterion doesn't apply to the initial row: there
    # are no steered designs to filter against, and the initial
    # row IS the only row.  We still want confidence metrics on
    # it so the comparison table has something to rank.
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
        print("  Filter: --populate-all set — all rows except "
              "reversion_dropped==1")
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

    # The per-design metrics we harvest from compute_metrics.py output.
    # We NEVER add these as new columns if they collide with existing
    # base columns — we prefix them with `steered_` so the join is lossless
    # AND the prefix correctly indicates that these confidence metrics
    # were computed on the STEERED prediction.  For pose_holds rows the
    # steered_* metrics describe the pre-reversion pose; corresponding
    # reverted_* metrics are populated by harvest-reversions.
    #
    # We deliberately DO NOT harvest the structural RMSD columns
    # (rmsd_receptor, rmsd_effector_*, n_receptor_ca, n_effector_ca)
    # from compute_metrics.py:
    #   1. `all_results_multicycle.csv` already carries the
    #      authoritative per-design RMSDs from the steering pipeline
    #      (receptor_aligned_effector_rmsd, independent_receptor_rmsd,
    #      independent_effector_rmsd) in its base columns, and those
    #      use the correct truth-vs-pred chain mapping.
    #   2. compute_metrics.py uses a single --receptor-chain /
    #      --effector-chain pair for BOTH the prediction and the
    #      reference, so on complexes where truth chains != pred
    #      chains (7QPX/7QZD — truth B/C vs pred A/B) its structural
    #      RMSDs would be wrong.
    # Skipping them also means we don't need --reference-pdb at all,
    # which in turn means no Kabsch fit per design — much faster.
    METRIC_KEYS_RAW = [
        "avg_plddt", "complex_plddt",
        "ptm", "iptm", "pae_mean", "ipae", "pae_pass_frac",
        "ipsae_ab", "ipsae_ba", "ipsae_min",
        # 15Å iPSAE — same Dunbrack iPSAE machinery as the 10Å trio
        # above, evaluated at a more permissive PAE cutoff.  Computed
        # in the same compute_metrics.py call so no extra subprocess
        # work — the PAE matrix is already loaded.
        "ipsae_ab_15", "ipsae_ba_15", "ipsae_min_15",
        "actifptm", "af_rank_score",
        # Per-prediction native-PDB-dependent metrics.  These were
        # previously computed post-hoc by compute_interface_metrics.py
        # (and run_biophysical_metrics.py for interface_plddt) on
        # survivors only — moved here so every steered prediction is
        # scored on the same metric set as it lands, without a
        # separate sidecar-finding pass.  intact_core and
        # weighted_jaccard are populated when --native-pdb was passed
        # to compute_metrics.py (the cycle_0 ground truth for this
        # design); blank otherwise.  interface_plddt only needs the
        # model PDB and is always populated.
        "interface_plddt",
        "intact_core",
        "weighted_jaccard",
        # Interface contacts / contamination check.  See
        # compute_metrics.py:compute_interface_contacts /
        # classify_mutation_reliance for definitions.  n_contact_residues
        # and contact_residues are populated for every row; the
        # contamination columns (n_contacts_on_mutated_positions,
        # mutated_contact_positions) are
        # populated only when the mutation list was supplied — i.e.
        # never for cold-start / initial rows.
        #
        # The authoritative contamination check: any mutated residue
        # that contacts the effector invalidates the metrics.  The
        # list of such positions is in `mutated_contact_positions`.
        "n_contact_residues", "contact_residues",
        "n_contacts_on_mutated_positions",
        "mutated_contact_positions",
        "contact_cutoff_used",
    ]
    METRIC_KEY_PREFIX = "steered_"
    prefixed_metric_keys = [METRIC_KEY_PREFIX + k for k in METRIC_KEYS_RAW]

    # Load existing output CSV for resume (helper handles the file-exists
    # gate and the "row has at least one populated metric" predicate).
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

    # ── Build the augmented fieldnames up front so the output CSV is
    # consistent across rows / partial runs.
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
    # Rows that don't pass the filter are still preserved in the
    # output (with empty steered_* / reverted_* fields) so the joined CSV is a true
    # superset of all_results_multicycle.csv.  We only shell out to
    # compute_metrics.py for rows that DO pass the filter.
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
        # Non-passing rows: preserve as-is with empty steered_* / reverted_* fields.
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
            # Write output CSV after every successful row so resume works
            # cleanly even if the job is killed mid-loop.  Use the same
            # canonical Block A/B/C/End layout as the final writer so the
            # column order is consistent whether the job completes or is
            # killed.
            _write_metrics_csv(output_csv, joined_rows)
        else:  # "failed"
            n_failed += 1

    # ── Post-processing pass: derive steered_pae_cutoff_used,
    # steered_confidence_flag, and recompute the unified ranks ─────────
    # We compute these AFTER the harvest loop because they need a
    # global view of all populated rows (for the rank) and they need
    # to apply the threshold flags consistently across both freshly-
    # computed rows and resumed rows from a previous run.
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

    # Recompute the unified ranks now that compute-final-metrics has
    # populated the steered_* confidence columns.  rank_by_ra_eff and
    # rank_by_composite_score were already populated by aggregate
    # (both use only steering-pipeline outputs that are available
    # pre-compute-final-metrics) and will be re-derived here too —
    # values are deterministic, so the second computation produces
    # identical output.
    _compute_unified_ranks(joined_rows)

    # Final write (covers the case where every row was resume-skipped
    # or failed and the loop never hit the incremental write path).
    # Reorder columns into the canonical Block A / B / C / End layout
    # so the CSV the user actually reads has steered_* metrics inserted
    # into Block B alongside the steering RMSDs, not appended at the
    # end as legacy steered_* / reverted_* columns used to be.
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
                      help="P0.2: populate confidence / interface metrics "
                           "on EVERY non-dropped row, not just rows that "
                           "pass the ra_eff + intact filter.  Useful for "
                           "cross-task AUROC analysis where filtered-away "
                           "rows are needed as the wrong-interface side "
                           "of the ROC curve.  reversion_dropped==1 rows "
                           "are still skipped because their metrics are "
                           "meaningless.  Default: off (preserves legacy "
                           "behaviour where only filter-passing rows get "
                           "metrics).")
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
