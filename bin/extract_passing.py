#!/usr/bin/env python3
"""Read aggregated_results.csv and write passing_summary.csv, the ranked
wet-lab triage list.

Input is aggregated_results.csv from `aggregate-per-sequence`, one row
per unique sequence with median and MAD aggregates across its
num_seeds Boltz predictions.  Output is one row per passing
sequence, ranked by composite score
``true_jaccard - 0.05 * ra_eff_vs_truth``.

A row appears only if it passes both filters:
  - outcome is "", "no_reversion" or "pose_holds", so the sequence is
    clean steered or a passing pose_holds
  - rank_by_composite_score is populated, which needs both a usable
    ra_eff and a usable true_jaccard

Schema notes
------------
  - All numeric columns are medians across seeds.  The per-seed
    distribution stays in raw_per_seed_results.csv.
  - ra_eff_vs_truth and true_jaccard also carry a MAD, which decides
    how much to trust the rank.  Confidence metrics carry a median
    only, with their MAD in aggregated_results.csv.
  - canonical_pdb is the PDB of the seed at the median.  With an even
    number of surviving seeds it is the worse of the two middle seeds,
    and canonical_seed_choice_warning records that.
  - n_pass counts the seeds that passed the structural and
    contamination filters, so pose_holds plus clean_steered.  Against
    n_seeds it gives the tier: A when all seeds pass, B when some pass,
    C when one passes, none when zero pass.  n_pass overrides the
    outcome label, so outcome=="no_reversion" with n_pass < n_seeds
    lands in a lower tier.
  - When all N cold-start seeds pass the rmsd_threshold and intact
    filters, plan() skips negative steering and writes one row per seed under sequence_group=0.  Those
    aggregate into one outcome="no_reversion" row with
    n_pass == n_seeds, which ranks as a tier-A candidate.

    python3 extract_passing.py \\
        --input  runs/design_3_v5/aggregated_results.csv \\
        --output runs/design_3_v5/passing_summary.csv

With no --output the file goes to the input directory.
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
# Medians throughout, since those are the numbers worth sorting and
# thresholding on.  MAD appears only on the two ranking metrics.
# Per-seed verdict counts are exposed so ties can be broken.

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

    # Every seed of one sequence has the same mutation set, so the
    # median just reports that single value.
    "total_mutations_median",

    # Mutations as a chain selection.  "steered" is every steering
    # mutation that went into Boltz.  "reverted majority" is the
    # positions still mutated in a majority of the seeds' reverted
    # predictions.  "reverted all" is the union across all seeds, the
    # worst case for what the construct would retain.  None of these
    # mutations are in the wet-lab construct.  They describe the
    # sequence Boltz scored.
    "steered_mutations_chimerax",
    "steered_mutations_aa",
    "reverted_mutations_majority_chimerax",
    "reverted_mutations_majority_aa",
    "reverted_mutations_all_chimerax",
    "reverted_mutations_all_aa",
    "reverted_mutations_seeds_identical",

    # Structural RMSDs against ground truth, the primary triage metric.
    # The spread columns (MAD, n_used) are in aggregated_results.csv.
    # intact-majority is left out, being always 1 in this file.
    "ra_eff_vs_truth_median",
    "independent_receptor_rmsd_median",
    "independent_effector_rmsd_median",

    # Interface similarity, always from the steered side.  The designed
    # interface is a property of the sequence, not the pose, and
    # reverted predictions do not recompute jaccard.
    "wrong_jaccard_median",
    "true_jaccard_median",
    "n_shared_true_median",
    "n_shared_wrong_median",

    # Contact analysis
    "n_contact_residues_median",
    "contact_residues_majority",
    "mutated_contact_positions_majority",

    # Confidence metrics, medians only.  Spread is in
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
    # 15Å iPSAE median, the same calculation with a more permissive PAE
    # cutoff.
    "ipsae_ab_15_median",
    "ipsae_ba_15_median",
    "ipsae_min_15_median",

    # Per-prediction interface metrics, moved out of the post-hoc
    # orthogonal-metrics pass to aggregate with the other Boltz
    # metrics.
    "interface_plddt_median",
    "weighted_jaccard_median",
    "intact_core_majority",

    "confidence_flag",

    # Cumulative wall-clock seconds for this MPNN sequence's
    # negative_steering_run_one.sh invocation, covering plan,
    # predict-one, collect, reversion, harvest, finalize, aggregate,
    # compute-final-metrics and aggregate-per-sequence.  The value
    # comes from run_one_runtime_sec.txt in the input directory and
    # repeats on every row, since one invocation covers one sequence.
    # Blank when that file was missing.
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

    Working on medians means a sequence is flagged only when its typical
    prediction fails the filter, so a single rogue seed does not flag
    it.

    Thresholds default to PipelineInternalThresholds.default(), falling
    back to the hard-coded 0.10, 0.30, 0.70 and 15.0.  A caller can
    override either source.
    """
    if pass_frac_min is None or iptm_min is None or complex_plddt_min is None or ipae_max is None:
        # Lazy import, so this script loads without
        # pipeline_thresholds.
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
            # Fall back to the hard-coded values when the module is off
            # the path.
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
    """Summarise one per-seed mutation-position column across a
    sequence group's seeds.

    Returns (majority_chimerax, all_chimerax, identical_flag):
      - majority_chimerax: ChimeraX selection of the positions present
        in strictly more than half the seeds, so two or more for n=3
      - all_chimerax: every position present in any seed, the worst
        case for which mutations could end up in the construct
      - identical_flag: "1" when every seed gave the same set, "0" when
        they differ, "" when fewer than two seeds had data

    `column` is either `steered_mutations_chimerax` or
    `reverted_mutations_chimerax`.  Empty values are skipped.
    """
    sets: List[set] = []
    for r in per_seed_rows:
        s = (r.get(column) or "").strip()
        if not s:
            continue
        # The format is "/A:5,7,22".
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

    counter: Counter = Counter()
    for s in sets:
        for p in s:
            counter[p] += 1

    # Strictly more than half the seeds, so 2 for n=3, 2 for n=2, 1 for
    # n=1.  threshold_majority_frac stays in the signature because
    # callers pass it, but it has no effect.  A four-line conditional
    # on the fraction used to sit above, and this line overwrote its
    # result unconditionally, so it has been removed.
    threshold = (n // 2) + 1

    majority = sorted(p for p, c in counter.items() if c >= threshold)
    all_pos = sorted(counter.keys())

    # Chain prefix comes from the first non-empty seed.
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
    """Return an amino-acid mutation string such as 'K5E V7D M22R',
    restricted to the given positions.

    The lookup covers all seeds, because one seed's
    reverted_mutations_aa lists only the positions that seed retained.
    Residue identities are fixed by the sequence, so any seed
    mentioning a position can supply its code.

    If the requested column cannot cover every position, the lookup
    falls back to steered_mutations_aa, which always lists every
    mutated position.
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

    pos_to_token: Dict[int, str] = {}
    for r in per_seed_rows:
        s = (r.get(column_aa) or "").strip()
        if not s:
            continue
        for p, tok in _parse_aa_string(s).items():
            pos_to_token.setdefault(p, tok)
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

    Which stage's metrics are used depends on the outcome:
      - Reversion was attempted for pose_holds, pose_collapses and
        new_contamination, so structural and confidence metrics come
        from reverted_*.  The reverted prediction is the final
        design, even when reversion failed.  steered_* would present
        the still-mutated pre-reversion prediction as the result.
      - Reversion was not attempted for no_reversion, where the steered
        prediction had no contamination, nor for a singleton, where
        only one seed had data.  Both metric groups then come from
        steered_*.
      - Interface similarity (true_jaccard, wrong_jaccard, n_shared_*)
        always comes from steered_*.  It describes the designed
        sequence's cold-start binding mode, not the reverted pose.

    Past bug, fixed 2026-04-29.  This function used steered_* for
    everything except pose_holds, so pose_collapses and
    new_contamination rows reported their cold-start metrics and looked
    unrealistically good in the cohort summary.  It never showed in
    tier-A, B or C rows, since main() filters down to
    {pose_holds, no_reversion, ""}.  It surfaced when
    cross_summary.py's tier-none fallback, added the same day, began
    reading aggregated_results.csv directly, which does contain them.

    `per_seed_rows` is the raw_per_seed_results rows for this sequence
    group, used to fill the mutation columns.
    """
    outcome = r.get("outcome") or ""
    # _classify_outcome in negsteer_aggregate.py holds the
    # authoritative list of reversion outcomes.
    reversion_attempted = outcome in (
        "pose_holds",
        "pose_collapses",
        "new_contamination",
    )
    struct_prefix = "reverted_" if reversion_attempted else "steered_"
    conf_prefix = "reverted_" if reversion_attempted else "steered_"

    out = {}

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

    out["n_pass"] = r.get("n_pass", "")
    out["outcome"] = r.get("outcome", "")

    # A clean steered row reads steered_total_mutations_median.  A
    # pose_holds row reads the post-reversion count.
    out["total_mutations_median"] = _get(
        r, struct_prefix + "total_mutations_median",
        "steered_total_mutations_median")

    # Per-seed mutation strings from raw_per_seed_results.csv.  The
    # steered set is identical across a sequence group's seeds.  The
    # reverted set can differ, because each seed reverts against its
    # own steered prediction's contact list.
    if per_seed_rows:
        # Take the first seed that has the steered set.
        steered_chimx = ""
        steered_aa = ""
        for ps in per_seed_rows:
            if ps.get("steered_mutations_chimerax"):
                steered_chimx = ps.get("steered_mutations_chimerax", "")
                steered_aa = ps.get("steered_mutations_aa", "")
                break
        out["steered_mutations_chimerax"] = steered_chimx
        out["steered_mutations_aa"] = steered_aa

        rev_maj_chimx, rev_all_chimx, identical = (
            _summarize_mutations_across_seeds(
                per_seed_rows, "reverted_mutations_chimerax")
        )

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
        out["steered_mutations_chimerax"] = ""
        out["steered_mutations_aa"] = ""
        out["reverted_mutations_majority_chimerax"] = ""
        out["reverted_mutations_majority_aa"] = ""
        out["reverted_mutations_all_chimerax"] = ""
        out["reverted_mutations_all_aa"] = ""
        out["reverted_mutations_seeds_identical"] = ""

    out["ra_eff_vs_truth_median"] = _get(
        r, struct_prefix + "ra_eff_vs_truth_median")
    out["independent_receptor_rmsd_median"] = _get(
        r, struct_prefix + "independent_receptor_rmsd_median")
    out["independent_effector_rmsd_median"] = _get(
        r, struct_prefix + "independent_effector_rmsd_median")

    out["wrong_jaccard_median"] = r.get("steered_wrong_jaccard_median", "")
    out["true_jaccard_median"] = r.get("steered_true_jaccard_median", "")
    out["n_shared_true_median"] = r.get("steered_n_shared_true_median", "")
    out["n_shared_wrong_median"] = r.get(
        "steered_n_shared_wrong_median", "")

    # Contact analysis follows the structural metrics' prediction.
    out["n_contact_residues_median"] = _get(
        r, struct_prefix + "n_contact_residues_median",
        "steered_n_contact_residues_median")
    out["contact_residues_majority"] = _get(
        r, struct_prefix + "contact_residues_majority",
        "steered_contact_residues_majority")
    out["mutated_contact_positions_majority"] = _get(
        r, struct_prefix + "mutated_contact_positions_majority",
        "steered_mutated_contact_positions_majority")

    for metric in ("iptm", "ptm", "actifptm", "complex_plddt",
                   "avg_plddt", "ipae", "pae_mean", "pae_pass_frac",
                   "ipsae_min",
                   "ipsae_ab_15", "ipsae_ba_15", "ipsae_min_15",
                   # These moved here from compute_interface_metrics.py
                   # and run_biophysical_metrics.py, both now deleted.
                   "interface_plddt", "weighted_jaccard"):
        out[f"{metric}_median"] = _get(
            r, conf_prefix + metric + "_median",
            "steered_" + metric + "_median")

    # intact_core is binary, so it takes a majority across seeds rather
    # than a median.
    out["intact_core_majority"] = _get(
        r, conf_prefix + "intact_core_majority",
        "steered_intact_core_majority")

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

    # A missing run_one_runtime_sec.txt leaves the column blank, which
    # is expected for older workdirs and partial runs.
    run_one_runtime_sec = ""
    runtime_file = input_path.parent / "run_one_runtime_sec.txt"
    if runtime_file.exists():
        try:
            run_one_runtime_sec = runtime_file.read_text().strip()
            # Check it parses, so corrupt content does not travel
            # downstream.
            float(run_one_runtime_sec)
        except (OSError, ValueError):
            run_one_runtime_sec = ""

    with open(input_path) as f:
        rows = list(csv.DictReader(f))

    # This drops singletons from the initial baseline, plus the
    # pose_collapses and new_contamination outcomes.
    PASSING_OUTCOMES = {"", "no_reversion", "pose_holds"}
    passing = [
        r for r in rows
        if (r.get("outcome") or "") in PASSING_OUTCOMES
        and r.get("rank_by_composite_score")
    ]

    if not passing:
        print("WARNING: no ranked rows found — writing empty CSV")

    # Each aggregate looks up per-seed rows by
    # (cycle, pathway, sequence_group).
    out_rows = []
    for r in passing:
        key = (str(r.get("cycle", "")),
               str(r.get("pathway", "")),
               str(r.get("sequence_group", "")))
        per_seed_rows = per_seed_by_group.get(key, [])
        out_row = extract_row(r, per_seed_rows=per_seed_rows)
        # Repeating it keeps the CSV self-contained, so cross_summary
        # needs no extra lookup.
        out_row["run_one_runtime_sec"] = run_one_runtime_sec
        out_rows.append(out_row)

    # Sort by rank_by_composite_score ascending, so rank 1 is best.
    def sort_key(r):
        try:
            return int(r["rank_by_composite_score"])
        except (ValueError, TypeError, KeyError):
            return 10**9
    out_rows.sort(key=sort_key)

    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
