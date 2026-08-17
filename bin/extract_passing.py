#!/usr/bin/env python3
"""
extract_passing.py
------------------
Read aggregated_results.csv and produce passing_summary.csv: the
ranked-and-filtered wet-lab triage list.

Input
-----
aggregated_results.csv (produced by `aggregate-per-sequence`).  One
row per UNIQUE SEQUENCE, with median + MAD aggregates across the
num_seeds Boltz predictions of that sequence.

Output
------
passing_summary.csv: one row per passing unique sequence, ranked by
composite score = true_jaccard − 0.05 · ra_eff_vs_truth (P0.3).

Filter chain (a row must pass ALL):
  - outcome in {"" / "no_reversion" / "pose_holds"}
    (i.e. the sequence is either clean steered or a passing
    pose_holds)
  - rank_by_composite_score is populated (i.e. the sequence has both
    a usable ra_eff and a usable true_jaccard)

Schema notes (P0 audit — multi-seed):
  - All numeric columns are MEDIANS across seeds.  The full per-seed
    distribution is available in raw_per_seed_results.csv.
  - For the two ranking-driver metrics (ra_eff_vs_truth and
    true_jaccard) we surface median + MAD, since spread on those
    directly affects how trustworthy the rank is.
  - For confidence metrics we surface only median; MAD is in
    aggregated_results.csv if needed.
  - canonical_pdb is the seed-of-median PDB; for even-N degraded
    cases (some seeds failed) it's the worse of the two middle
    seeds (see canonical_seed_choice_warning).
  - n_pass = number of seeds that passed structural+contamination
    filters (pose_holds + clean_steered).  Tiered against n_seeds:
    A = all pass, B = some pass, C = one, none = zero.  n_pass is
    path-agnostic and authoritative — outcome=='no_reversion' with
    n_pass<n_seeds correctly lands in a lower tier.
  - Cold-start binders (notes12 / patch b): when ALL N cold-start
    seeds pass the rmsd_threshold + intact filters, plan() skips
    negative steering entirely and writes one row per seed under
    sequence_group=0.  These aggregate as a single
    outcome="no_reversion" row with n_pass=n_seeds, get a composite
    rank, and appear in passing_summary as tier-A candidates.

Usage
-----
    python3 extract_passing.py \\
        --input  runs/design_3_v5/aggregated_results.csv \\
        --output runs/design_3_v5/passing_summary.csv

    # Or default output (input dir / passing_summary.csv):
    python3 extract_passing.py \\
        --input  runs/design_3_v5/aggregated_results.csv
"""

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

# ── Output schema ────────────────────────────────────────────────────
#
# Median values throughout (those are the sortable / thresholdable
# numbers).  MAD only on the two ranking-driver metrics where spread
# materially affects how much we trust the rank.  Per-seed verdict
# counts surfaced for tie-breaking.

OUTPUT_FIELDS = [
    # Identification
    "rank_by_composite_score",
    "rank_by_ra_eff",
    "cycle",
    "pathway",
    "sequence_group",
    "design",
    "canonical_pdb",
    "canonical_seed_index",
    "canonical_seed_choice_warning",
    "n_seeds",

    # Per-seed pass count + aggregate outcome label
    "n_pass",
    "outcome",

    # Mutations on the sequence (median is meaningless here — the
    # mutation set is identical across seeds of one sequence; we just
    # surface the median as a proxy for "the value")
    "total_mutations_median",

    # Mutations as a chain selection.  "steered" = full set of
    # steering mutations in the sequence going into Boltz; "reverted
    # majority" = positions remaining (i.e. NOT reverted) in a
    # majority of the seeds' reverted predictions; "reverted all" =
    # union of remaining positions across all seeds (worst case for
    # what the construct will retain).  For the wet-lab construct
    # itself, NONE of these mutations are present — these describe
    # the sequence Boltz was scored on, not the wet-lab construct.
    "steered_mutations_chimerax",
    "steered_mutations_aa",
    "reverted_mutations_majority_chimerax",
    "reverted_mutations_majority_aa",
    "reverted_mutations_all_chimerax",
    "reverted_mutations_all_aa",
    "reverted_mutations_seeds_identical",

    # Structural RMSDs vs ground truth — primary triage metric.
    # Spread (MAD / n_used) and intact-majority columns deliberately
    # omitted: MAD is available in aggregated_results.csv, intact-
    # majority is tautologically 1 for any row that reaches this file.
    "ra_eff_vs_truth_median",
    "independent_receptor_rmsd_median",
    "independent_effector_rmsd_median",

    # Interface similarity (always sourced from steered side — the
    # designed interface is a sequence property, not a pose property,
    # and reverted predictions don't recompute jaccard).  Spread
    # omitted — look it up in aggregated_results.csv if needed.
    "wrong_jaccard_median",
    "true_jaccard_median",
    "n_shared_true_median",
    "n_shared_wrong_median",

    # Contact analysis
    "n_contact_residues_median",
    "contact_residues_majority",
    "mutated_contact_positions_majority",

    # Confidence metrics — medians only.  Full spread in
    # aggregated_results.csv.
    "iptm_median",
    "ptm_median",
    "actifptm_median",
    "complex_plddt_median",
    "avg_plddt_median",
    "ipae_median",
    "pae_mean_median",
    "pae_pass_frac_median",
    "ipsae_min_median",
    # 15Å iPSAE median — same machinery, more permissive PAE cutoff.
    "ipsae_ab_15_median",
    "ipsae_ba_15_median",
    "ipsae_min_15_median",

    # Per-prediction interface metrics (P0-29 / P0-38) — moved from
    # the post-hoc orthogonal-metrics pass so they aggregate alongside
    # the other Boltz-derived metrics.
    "interface_plddt_median",
    "weighted_jaccard_median",
    "intact_core_majority",

    # Confidence flag (computed from medians)
    "confidence_flag",

    # Run-level diagnostics (runtime instrumentation).  Cumulative
    # wall-clock seconds for this MPNN sequence's
    # negative_steering_run_one.sh invocation, covering plan +
    # predict-one + collect + reversion + harvest + finalize +
    # aggregate + compute-final-metrics + aggregate-per-sequence.
    # Sourced from run_one_runtime_sec.txt in the input dir.  Same
    # value is repeated on every row of passing_summary.csv for this
    # sequence (one run_one invocation covers one sequence).  Blank
    # if the sidecar file was missing.
    "run_one_runtime_sec",
]


def _get(row, key, fallback_key=None):
    """Return row[key] if non-empty, else row[fallback_key], else ''."""
    v = row.get(key, "")
    if v not in ("", None):
        return v
    if fallback_key:
        return row.get(fallback_key, "")
    return ""


def _compute_confidence_flag(
    out,
    pass_frac_min=None,
    iptm_min=None,
    complex_plddt_min=None,
    ipae_max=None,
):
    """Compute confidence_flag from the median metrics in `out`.
    Operates on median values: a sequence is flagged if its TYPICAL
    prediction fails the confidence filter, ignoring single rogue seeds.

    Thresholds default to PipelineInternalThresholds.default()
    (Phase 4 caller migration — was hard-coded 0.10/0.30/0.70/15.0
    literals).  Callers can still pass explicit values to override."""
    if pass_frac_min is None or iptm_min is None or complex_plddt_min is None or ipae_max is None:
        # Lazy import — keep extract_passing.py importable in environments
        # that don't have the pipeline_thresholds module available.
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from pipeline_thresholds import PipelineInternalThresholds
            _t = PipelineInternalThresholds.default()
            if pass_frac_min is None:
                pass_frac_min = _t.pae_pass_frac_min
            if iptm_min is None:
                iptm_min = _t.iptm_min
            if complex_plddt_min is None:
                complex_plddt_min = _t.complex_plddt_min
            if ipae_max is None:
                ipae_max = _t.ipae_max
        except ImportError:
            # Fallback to the historical hard-coded values if the new
            # module isn't on the path (e.g. older standalone invocations).
            pass_frac_min = pass_frac_min if pass_frac_min is not None else 0.10
            iptm_min = iptm_min if iptm_min is not None else 0.30
            complex_plddt_min = complex_plddt_min if complex_plddt_min is not None else 0.70
            ipae_max = ipae_max if ipae_max is not None else 15.0
    triggers = []
    try:
        if float(out.get("pae_pass_frac_median", 1)) < pass_frac_min:
            triggers.append("low_pass_frac")
    except (ValueError, TypeError):
        pass
    try:
        if float(out.get("iptm_median", 1)) < iptm_min:
            triggers.append("low_iptm")
    except (ValueError, TypeError):
        pass
    try:
        if float(out.get("complex_plddt_median", 1)) < complex_plddt_min:
            triggers.append("low_plddt")
    except (ValueError, TypeError):
        pass
    try:
        if float(out.get("ipae_median", 0)) > ipae_max:
            triggers.append("high_ipae")
    except (ValueError, TypeError):
        pass
    if not triggers:
        return "ok"
    if len(triggers) == 1:
        return triggers[0]
    return "multiple"


def _summarize_mutations_across_seeds(
    per_seed_rows: List[Dict],
    column: str,
    threshold_majority_frac: float = 0.5,
) -> Tuple[str, str, str]:
    """For one sequence group, summarise the contents of a per-seed
    mutation-position column across the group's seeds.

    Returns (majority_chimerax, all_chimerax, identical_flag) where:
      - majority_chimerax: ChimeraX selection of positions present in
        STRICTLY MORE than `threshold_majority_frac` of seeds (default
        > 50% — for n=3 that means "in 2 or more seeds").
      - all_chimerax: ChimeraX selection of every position that
        appears in ANY seed (the union — worst case for "what
        mutations might end up in the construct").
      - identical_flag: "1" if every seed reverted to the same set,
        "0" if seeds differ, "" if there is only one or zero seeds
        with non-empty data.

    `column` should be one of `steered_mutations_chimerax` or
    `reverted_mutations_chimerax`.  Empty values are skipped.
    """
    # Each seed's selection string parses to a set of int positions.
    sets: List[set] = []
    for r in per_seed_rows:
        s = (r.get(column) or "").strip()
        if not s:
            continue
        # Format: "/A:5,7,22"
        parts = s.split(":", 1)
        if len(parts) != 2:
            continue
        positions = {
            int(p) for p in parts[1].split(",")
            if p.strip().lstrip("-").isdigit()
        }
        sets.append(positions)
    n = len(sets)
    if n == 0:
        return ("", "", "")

    # Position-occurrence counter
    counter: Counter = Counter()
    for s in sets:
        for p in s:
            counter[p] += 1

    threshold = max(1, int(n * threshold_majority_frac) + 1
                    if n * threshold_majority_frac
                    == int(n * threshold_majority_frac)
                    else int(n * threshold_majority_frac) + 1)
    # Above is "strictly greater than half" — for n=3, threshold=2;
    # for n=2, threshold=2; for n=1, threshold=1.  Equivalently:
    threshold = (n // 2) + 1

    majority = sorted(p for p, c in counter.items() if c >= threshold)
    all_pos = sorted(counter.keys())

    # Chain prefix: take from first non-empty seed
    chain_prefix = "/A"
    for r in per_seed_rows:
        s = (r.get(column) or "").strip()
        if s and ":" in s:
            chain_prefix = s.split(":", 1)[0]
            break

    maj_str = (f"{chain_prefix}:" + ",".join(str(p) for p in majority)
               if majority else "")
    all_str = (f"{chain_prefix}:" + ",".join(str(p) for p in all_pos)
               if all_pos else "")
    identical = "1" if all(s == sets[0] for s in sets) and n > 1 else \
                ("0" if n > 1 else "")

    return (maj_str, all_str, identical)


def _muts_aa_for_positions(
    per_seed_rows: List[Dict],
    column_aa: str,
    positions: List[int],
) -> str:
    """Return an AA mutation string (e.g. 'K5E V7D M22R') restricted
    to the given positions.  Looks across ALL seeds for each
    position's AA identity — a single seed's reverted_mutations_aa
    only covers the positions THAT seed retained, so positions
    retained only in other seeds need to be looked up there.

    The AA identities are sequence-determined (the same sequence_group
    has the same residue at each position) so any seed that mentions a
    position can supply its AA code.

    Falls back to looking up identities in steered_mutations_aa, which
    always contains every mutated position, if reverted columns can't
    cover all requested positions.
    """
    if not positions:
        return ""
    pos_set = set(positions)

    def _parse_aa_string(s: str) -> Dict[int, str]:
        """Parse 'K5E V7D M22R' into {5: 'K5E', 7: 'V7D', 22: 'M22R'}."""
        out: Dict[int, str] = {}
        for tok in s.split():
            if len(tok) < 3:
                continue
            i = 1
            while i < len(tok) and tok[i].isdigit():
                i += 1
            digits = tok[1:i]
            if not digits:
                continue
            try:
                p = int(digits)
            except ValueError:
                continue
            out[p] = tok
        return out

    # Build position → AA-token map by merging across all seeds.
    # First try the requested column; fall back to steered_mutations_aa
    # for any positions still missing (should be rare for reverted
    # lookups since the steered set is a superset).
    pos_to_token: Dict[int, str] = {}
    for r in per_seed_rows:
        s = (r.get(column_aa) or "").strip()
        if not s:
            continue
        for p, tok in _parse_aa_string(s).items():
            pos_to_token.setdefault(p, tok)
    # Fallback for any positions still missing
    missing = pos_set - set(pos_to_token.keys())
    if missing:
        for r in per_seed_rows:
            s = (r.get("steered_mutations_aa") or "").strip()
            if not s:
                continue
            for p, tok in _parse_aa_string(s).items():
                if p in missing:
                    pos_to_token[p] = tok

    def _pos(t: str) -> int:
        i = 1
        while i < len(t) and t[i].isdigit():
            i += 1
        return int(t[1:i])

    kept = [pos_to_token[p] for p in positions if p in pos_to_token]
    return " ".join(sorted(kept, key=_pos))


def extract_row(r, per_seed_rows=None):
    """Build a passing_summary row from one aggregated_results row.

    Stage-aware metric selection:
      - For verdicts where reversion was ATTEMPTED (any reversion-
        classifier output: pose_holds, pose_collapses, new_contamination):
        structural and confidence metrics come from reverted_*.  The
        reverted prediction is the final design — even if reversion
        failed, the reverted metrics are what describes the failure.
        Showing steered_* in those cases would misleadingly present
        the pre-reversion (still-mutated) prediction's metrics as if
        they were the final result.
      - For no_reversion (reversion was not triggered because the
        steered prediction had no contamination) and singleton (only
        one seed had data, no aggregation possible): the steered
        prediction IS the final design, so structural and confidence
        come from steered_*.
      - Interface similarity (true_jaccard, wrong_jaccard, n_shared_*)
        ALWAYS comes from steered_* — it's a property of the designed
        sequence's cold-start binding mode, not the reverted pose.

    Pre-2026-04-29 history note: this function previously used
    steered_* for everything except pose_holds, which gave
    semantically-wrong results for pose_collapses and
    new_contamination rows that surface their cold-start metrics in
    the cohort summary and made the rows look unrealistically good.
    The bug never manifested in tier-A/B/C rows because main()
    filters passing_summary.csv to {pose_holds, no_reversion, ""}
    only — pose_collapses and new_contamination never reach
    passing_summary.csv via the normal flow.  The bug surfaced once
    cross_summary.py's tier-none fallback (added 2026-04-29)
    started reading aggregated_results.csv directly, which DOES
    contain those verdicts.

    `per_seed_rows` is the list of raw_per_seed_results rows for this
    sequence group, used to populate the mutation columns.
    """
    outcome = r.get("outcome") or ""
    # Reversion was attempted iff the outcome is one the reversion
    # classifier produces.  See _classify_outcome in
    # boltz2_iterate_steering.py for the canonical list.
    reversion_attempted = outcome in (
        "pose_holds",
        "pose_collapses",
        "new_contamination",
    )
    struct_prefix = "reverted_" if reversion_attempted else "steered_"
    conf_prefix = "reverted_" if reversion_attempted else "steered_"

    out = {}

    # Identification
    out["rank_by_composite_score"] = r.get("rank_by_composite_score", "")
    out["rank_by_ra_eff"] = r.get("rank_by_ra_eff", "")
    out["cycle"] = r.get("cycle", "")
    out["pathway"] = r.get("pathway", "")
    out["sequence_group"] = r.get("sequence_group", "")
    out["design"] = r.get("design", "")
    out["canonical_pdb"] = r.get("canonical_pdb", "")
    out["canonical_seed_index"] = r.get("canonical_seed_index", "")
    out["canonical_seed_choice_warning"] = r.get(
        "canonical_seed_choice_warning", "")
    out["n_seeds"] = r.get("n_seeds", "")

    # Per-seed pass count + aggregate outcome
    out["n_pass"] = r.get("n_pass", "")
    out["outcome"] = r.get("outcome", "")

    # Mutations.  For clean steered, the mutation set lives on
    # steered_total_mutations_median.  For pose_holds, the
    # post-reversion mutation set lives on
    # reverted_total_mutations_median.
    out["total_mutations_median"] = _get(
        r, struct_prefix + "total_mutations_median",
        "steered_total_mutations_median")

    # Per-seed mutation strings (steered always identical across
    # seeds of a sequence group; reverted may differ because each
    # seed's reverted set depends on its own steered prediction's
    # contact list).  Populated from raw_per_seed_results.csv.
    if per_seed_rows:
        # Steered mutation set — identical across seeds, take from
        # any seed.
        steered_chimx = ""
        steered_aa = ""
        for ps in per_seed_rows:
            if ps.get("steered_mutations_chimerax"):
                steered_chimx = ps.get("steered_mutations_chimerax", "")
                steered_aa = ps.get("steered_mutations_aa", "")
                break
        out["steered_mutations_chimerax"] = steered_chimx
        out["steered_mutations_aa"] = steered_aa

        # Reverted mutation set — may differ across seeds.
        rev_maj_chimx, rev_all_chimx, identical = (
            _summarize_mutations_across_seeds(
                per_seed_rows, "reverted_mutations_chimerax")
        )
        # Convert majority/all chimerax → AA strings using any seed's
        # AA column (residue identities don't vary between seeds).
        def _positions_from_chimx(s: str) -> List[int]:
            if not s or ":" not in s:
                return []
            return [int(p) for p in s.split(":", 1)[1].split(",")
                    if p.strip().lstrip("-").isdigit()]

        out["reverted_mutations_majority_chimerax"] = rev_maj_chimx
        out["reverted_mutations_majority_aa"] = _muts_aa_for_positions(
            per_seed_rows, "reverted_mutations_aa",
            _positions_from_chimx(rev_maj_chimx))
        out["reverted_mutations_all_chimerax"] = rev_all_chimx
        out["reverted_mutations_all_aa"] = _muts_aa_for_positions(
            per_seed_rows, "reverted_mutations_aa",
            _positions_from_chimx(rev_all_chimx))
        out["reverted_mutations_seeds_identical"] = identical
    else:
        # No per-seed data available — leave mutation columns blank.
        out["steered_mutations_chimerax"] = ""
        out["steered_mutations_aa"] = ""
        out["reverted_mutations_majority_chimerax"] = ""
        out["reverted_mutations_majority_aa"] = ""
        out["reverted_mutations_all_chimerax"] = ""
        out["reverted_mutations_all_aa"] = ""
        out["reverted_mutations_seeds_identical"] = ""

    # Structural RMSDs — median only.  Spread (MAD / n_used) and
    # intact-majority columns available in aggregated_results.csv.
    out["ra_eff_vs_truth_median"] = _get(
        r, struct_prefix + "ra_eff_vs_truth_median")
    out["independent_receptor_rmsd_median"] = _get(
        r, struct_prefix + "independent_receptor_rmsd_median")
    out["independent_effector_rmsd_median"] = _get(
        r, struct_prefix + "independent_effector_rmsd_median")

    # Interface similarity: always from the steered side.  The
    # designed interface is a property of the steered sequence's
    # cold-start prediction, not the reverted pose.  Spread available
    # in aggregated_results.csv if needed.
    out["wrong_jaccard_median"] = r.get("steered_wrong_jaccard_median", "")
    out["true_jaccard_median"] = r.get("steered_true_jaccard_median", "")
    out["n_shared_true_median"] = r.get("steered_n_shared_true_median", "")
    out["n_shared_wrong_median"] = r.get(
        "steered_n_shared_wrong_median", "")

    # Contact analysis: from the scored prediction (matches structural)
    out["n_contact_residues_median"] = _get(
        r, struct_prefix + "n_contact_residues_median",
        "steered_n_contact_residues_median")
    out["contact_residues_majority"] = _get(
        r, struct_prefix + "contact_residues_majority",
        "steered_contact_residues_majority")
    out["mutated_contact_positions_majority"] = _get(
        r, struct_prefix + "mutated_contact_positions_majority",
        "steered_mutated_contact_positions_majority")

    # Confidence metrics — medians only
    for metric in ("iptm", "ptm", "actifptm", "complex_plddt",
                   "avg_plddt", "ipae", "pae_mean", "pae_pass_frac",
                   "ipsae_min",
                   # 15Å iPSAE trio
                   "ipsae_ab_15", "ipsae_ba_15", "ipsae_min_15",
                   # Per-prediction interface metrics moved from
                   # compute_interface_metrics.py / run_biophysical_metrics.py
                   "interface_plddt", "weighted_jaccard"):
        out[f"{metric}_median"] = _get(
            r, conf_prefix + metric + "_median",
            "steered_" + metric + "_median")

    # intact_core is binary → majority across seeds, not a median
    out["intact_core_majority"] = _get(
        r, conf_prefix + "intact_core_majority",
        "steered_intact_core_majority")

    # Confidence flag computed from medians
    out["confidence_flag"] = _compute_confidence_flag(out)

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Extract ranked survivors from aggregated_results.csv "
                    "into a clean wet-lab triage CSV"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to aggregated_results.csv "
             "(produced by `aggregate-per-sequence`)")
    parser.add_argument(
        "--per-seed", default=None,
        help="Path to raw_per_seed_results.csv (default: same dir as "
             "--input).  Used to populate the steered/reverted "
             "mutation columns, since aggregated_results.csv does not "
             "carry the per-seed mutation strings.")
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path (default: <input-dir>/passing_summary.csv)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found", file=sys.stderr)
        return 2

    per_seed_path = (Path(args.per_seed) if args.per_seed
                     else input_path.parent / "raw_per_seed_results.csv")
    per_seed_by_group: Dict[Tuple[str, str, str], List[Dict]] = {}
    if per_seed_path.exists():
        with open(per_seed_path) as f:
            for r in csv.DictReader(f):
                key = (str(r.get("cycle", "")),
                       str(r.get("pathway", "")),
                       str(r.get("sequence_group", "")))
                per_seed_by_group.setdefault(key, []).append(r)
        print(f"Loaded {sum(len(v) for v in per_seed_by_group.values())} "
              f"per-seed rows from {per_seed_path.name}")
    else:
        print(f"WARNING: {per_seed_path} not found — mutation columns "
              f"will be blank in the output.  Pass --per-seed if you "
              f"want them populated.")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / "passing_summary.csv"

    # Read the cumulative run_one wall-clock, if the orchestrator
    # wrote it.  Silently blank if missing — older workdirs or
    # partial runs won't have it, which is fine.
    run_one_runtime_sec = ""
    runtime_file = input_path.parent / "run_one_runtime_sec.txt"
    if runtime_file.exists():
        try:
            run_one_runtime_sec = runtime_file.read_text().strip()
            # Validate it parses as a number so we don't propagate
            # corrupt content downstream.
            float(run_one_runtime_sec)
        except (OSError, ValueError):
            run_one_runtime_sec = ""

    with open(input_path) as f:
        rows = list(csv.DictReader(f))

    # Filter: row must (a) be a clean steered or pose_holds aggregate,
    # and (b) have a populated rank_by_composite_score.  Singletons
    # (initial baseline) and dropped outcomes (pose_collapses,
    # new_contamination) are excluded.
    PASSING_OUTCOMES = {"", "no_reversion", "pose_holds"}
    passing = [
        r for r in rows
        if (r.get("outcome") or "") in PASSING_OUTCOMES
        and r.get("rank_by_composite_score")
    ]

    if not passing:
        print("WARNING: no ranked rows found — writing empty CSV")

    # Build output rows.  For each passing aggregate, look up the
    # per-seed rows in the same (cycle, pathway, sequence_group)
    # group so we can populate mutation columns.
    out_rows = []
    for r in passing:
        key = (str(r.get("cycle", "")),
               str(r.get("pathway", "")),
               str(r.get("sequence_group", "")))
        per_seed_rows = per_seed_by_group.get(key, [])
        out_row = extract_row(r, per_seed_rows=per_seed_rows)
        # Stamp the sequence-level runtime on every row.  All rows
        # in this passing_summary.csv share one run_one invocation,
        # so the value is identical for each — repeating it per row
        # keeps the CSV self-contained and lets downstream
        # cross_summary read it without a separate lookup.
        out_row["run_one_runtime_sec"] = run_one_runtime_sec
        out_rows.append(out_row)

    # Sort by rank_by_composite_score ascending (rank 1 = best).
    def sort_key(r):
        try:
            return int(r["rank_by_composite_score"])
        except (ValueError, TypeError, KeyError):
            return 10**9
    out_rows.sort(key=sort_key)

    # Write
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
