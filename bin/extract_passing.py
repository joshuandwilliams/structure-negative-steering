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

# ── the output columns, in order, each with the rule that fills it ───────────
# One list, so a column cannot be in the CSV header and unfilled by
# extract_row, which csv.DictWriter does not catch in that direction.
#
#   AS_IS    r[name]
#   STEERED  r["steered_" + name], whatever the outcome
#   FINAL    r[final + name], where final is reverted_ when reversion ran
#   OR_STEER r[final + name], falling back to r["steered_" + name]
#   CONF     the same, with a _median suffix on a bare metric name
#   SEEDS    from raw_per_seed_results.csv, not from this row
#   LATER    main() fills it
AS_IS, STEERED, FINAL, OR_STEER, CONF, SEEDS, LATER = range(7)

COLUMNS = (
    ("rank_by_composite_score", AS_IS), ("rank_by_ra_eff", AS_IS),
    ("cycle", AS_IS), ("pathway", AS_IS), ("sequence_group", AS_IS),
    ("design", AS_IS), ("canonical_pdb", AS_IS),
    ("canonical_seed_index", AS_IS),
    ("canonical_seed_choice_warning", AS_IS), ("n_seeds", AS_IS),
    ("n_pass", AS_IS), ("outcome", AS_IS),

    # Every seed of one sequence has the same mutation set, so the median
    # just reports that single value.
    ("total_mutations_median", OR_STEER),

    # Mutations as a chain selection.  "steered" is every steering mutation
    # that went into Boltz.  "majority" is the positions still mutated in a
    # majority of seeds' reverted predictions, "all" the union across seeds,
    # the worst case for what the construct would retain.  None of these are
    # in the wet-lab construct: they describe the sequence Boltz scored.
    ("steered_mutations_chimerax", SEEDS), ("steered_mutations_aa", SEEDS),
    ("reverted_mutations_majority_chimerax", SEEDS),
    ("reverted_mutations_majority_aa", SEEDS),
    ("reverted_mutations_all_chimerax", SEEDS),
    ("reverted_mutations_all_aa", SEEDS),
    ("reverted_mutations_seeds_identical", SEEDS),

    # Structural RMSDs against ground truth, the primary triage metric.  The
    # spread columns (MAD, n_used) are in aggregated_results.csv.  No
    # fallback: a reverted row missing these is broken, not steered.
    ("ra_eff_vs_truth_median", FINAL),
    ("independent_receptor_rmsd_median", FINAL),
    ("independent_effector_rmsd_median", FINAL),

    # Interface similarity, always the steered side.  The designed interface
    # is a property of the sequence rather than the pose, and reverted
    # predictions do not recompute jaccard.
    ("wrong_jaccard_median", STEERED), ("true_jaccard_median", STEERED),
    ("n_shared_true_median", STEERED), ("n_shared_wrong_median", STEERED),

    # Contact analysis follows the structural metrics' prediction.
    ("n_contact_residues_median", OR_STEER),
    ("contact_residues_majority", OR_STEER),
    ("mutated_contact_positions_majority", OR_STEER),

    # Confidence, medians only; spread is in aggregated_results.csv.
    # ipsae_*_15 is the same calculation at a more permissive PAE cutoff.
    # interface_plddt and weighted_jaccard moved here from the post-hoc
    # orthogonal-metrics pass.
    ("iptm", CONF), ("ptm", CONF), ("actifptm", CONF),
    ("complex_plddt", CONF), ("avg_plddt", CONF), ("ipae", CONF),
    ("pae_mean", CONF), ("pae_pass_frac", CONF), ("ipsae_min", CONF),
    ("ipsae_ab_15", CONF), ("ipsae_ba_15", CONF), ("ipsae_min_15", CONF),
    ("interface_plddt", CONF), ("weighted_jaccard", CONF),

    # Binary per seed, so a majority rather than a median.
    ("intact_core_majority", OR_STEER),

    ("confidence_flag", LATER),

    # Cumulative wall-clock seconds for this sequence's
    # negative_steering_run_one.sh invocation, from run_one_runtime_sec.txt.
    # Repeats on every row, since one invocation covers one sequence.
    ("run_one_runtime_sec", LATER),
)

OUTPUT_FIELDS = [c + "_median" if rule is CONF else c for c, rule in COLUMNS]

_REVERSION_OUTCOMES = ("pose_holds", "pose_collapses", "new_contamination")


def _get(row, key, fallback_key=None):
    """Return row[key] if non-empty, else row[fallback_key], else ''."""
    v = row.get(key, "")
    if v not in ("", None):
        return v
    if fallback_key:
        return row.get(fallback_key, "")
    return ""


# Resolved once at import rather than on every call.  The old code did the
# sys.path insert and the import inside _compute_confidence_flag, so a run
# over N rows pushed N copies of bin/ onto sys.path and re-resolved
# __file__ N times.  500 calls added 500 duplicate entries, which
# tests/characterization/test_modules_repair_sys_path.py asserts must not
# happen for the modules it covers.
#
# The fallbacks are the values that were hard-coded across bin/ before
# pipeline_thresholds existed, kept so this script still runs with that
# module off the path.
try:
    from pipeline_thresholds import PipelineInternalThresholds
    _T = PipelineInternalThresholds.default()
    _DEFAULT_PASS_FRAC_MIN = _T.pae_pass_frac_min
    _DEFAULT_IPTM_MIN = _T.iptm_min
    _DEFAULT_COMPLEX_PLDDT_MIN = _T.complex_plddt_min
    _DEFAULT_IPAE_MAX = _T.ipae_max
except ImportError:  # pragma: no cover - the module sits beside this one
    _DEFAULT_PASS_FRAC_MIN = 0.10
    _DEFAULT_IPTM_MIN = 0.30
    _DEFAULT_COMPLEX_PLDDT_MIN = 0.70
    _DEFAULT_IPAE_MAX = 15.0

# (column, comparison, threshold, label).  "<" flags a value below the
# threshold, ">" above it.  The default used when the column is missing is
# the one that does NOT trigger, so an absent metric is not a failure.
_CONFIDENCE_CHECKS = (
    ("pae_pass_frac_median", "<", _DEFAULT_PASS_FRAC_MIN, "low_pass_frac", 1),
    ("iptm_median", "<", _DEFAULT_IPTM_MIN, "low_iptm", 1),
    ("complex_plddt_median", "<", _DEFAULT_COMPLEX_PLDDT_MIN, "low_plddt", 1),
    ("ipae_median", ">", _DEFAULT_IPAE_MAX, "high_ipae", 0),
)


def _compute_confidence_flag(out, pass_frac_min=None, iptm_min=None,
                             complex_plddt_min=None, ipae_max=None):
    """Compute confidence_flag from the median metrics in `out`.

    Working on medians means a sequence is flagged only when its typical
    prediction fails the filter, so a single rogue seed does not flag it.

    Thresholds come from PipelineInternalThresholds at import time.  A
    caller can override any of the four.  A metric that is absent or
    unparseable does not trigger: the row is reporting what it has, and a
    missing column is not evidence of low confidence.
    """
    overrides = (pass_frac_min, iptm_min, complex_plddt_min, ipae_max)
    triggers = []
    for (col, op, default, label, missing), override in zip(
            _CONFIDENCE_CHECKS, overrides):
        limit = default if override is None else override
        try:
            value = float(out.get(col, missing))
        except (ValueError, TypeError):
            continue
        if (value < limit) if op == "<" else (value > limit):
            triggers.append(label)
    if not triggers:
        return "ok"
    return triggers[0] if len(triggers) == 1 else "multiple"


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


# Derived once at import, so extract_row branches per RULE rather than per
# column. Five tight comprehensions instead of a 47-iteration if/elif chain.
_AS_IS_COLS    = tuple(c for c, r in COLUMNS if r is AS_IS)
_STEERED_COLS  = tuple(c for c, r in COLUMNS if r is STEERED)
_FINAL_COLS    = tuple(c for c, r in COLUMNS if r is FINAL)
_OR_STEER_COLS = tuple(c for c, r in COLUMNS if r is OR_STEER)
_CONF_COLS     = tuple(c for c, r in COLUMNS if r is CONF)
_SEED_COLS     = tuple(c for c, r in COLUMNS if r is SEEDS)


def _positions_from_chimx(s):
    """Residue numbers out of a "/A:5,7,22" chain selection."""
    if not s or ":" not in s:
        return []
    return [int(p) for p in s.split(":", 1)[1].split(",")
            if p.strip().lstrip("-").isdigit()]


def _mutation_columns(per_seed_rows):
    """The seven SEEDS columns, or blanks without any seed rows.

    The steered set is identical across a sequence group's seeds, so the
    first seed carrying one supplies it. The reverted set can differ,
    because each seed reverts against its own steered prediction's contact
    list, so it is summarised across seeds.
    """
    if not per_seed_rows:
        return dict.fromkeys(_SEED_COLS, "")

    steered = next((ps for ps in per_seed_rows
                    if ps.get("steered_mutations_chimerax")), {})
    majority, union, identical = _summarize_mutations_across_seeds(
        per_seed_rows, "reverted_mutations_chimerax")

    def aa_for(chimx):
        return _muts_aa_for_positions(per_seed_rows, "reverted_mutations_aa",
                                      _positions_from_chimx(chimx))

    return {
        "steered_mutations_chimerax": steered.get(
            "steered_mutations_chimerax", ""),
        "steered_mutations_aa": steered.get("steered_mutations_aa", ""),
        "reverted_mutations_majority_chimerax": majority,
        "reverted_mutations_majority_aa": aa_for(majority),
        "reverted_mutations_all_chimerax": union,
        "reverted_mutations_all_aa": aa_for(union),
        "reverted_mutations_seeds_identical": identical,
    }


def extract_row(r, per_seed_rows=None):
    """Build a passing_summary row from one aggregated_results row.

    COLUMNS above names every column and the rule that fills it. The rule
    that matters is which stage's prediction a metric comes from. Reversion
    was attempted for pose_holds, pose_collapses and new_contamination, so
    the reverted prediction is the final design even when reversion failed,
    and steered_* would present the still-mutated pre-reversion prediction
    as the result. It was not attempted for no_reversion, where the steered
    prediction had no contamination, nor for a singleton, where only one
    seed had data.

    Past bug, fixed 2026-04-29. This used steered_* for everything except
    pose_holds, so pose_collapses and new_contamination rows reported their
    cold-start metrics and looked unrealistically good. It never showed in
    tier-A, B or C rows, since main() filters down to
    {pose_holds, no_reversion, ""}. It surfaced when cross_summary.py's
    tier-none fallback, added the same day, began reading
    aggregated_results.csv directly, which does contain them.

    `per_seed_rows` is the raw_per_seed_results rows for this sequence group.
    """
    final = ("reverted_" if (r.get("outcome") or "") in _REVERSION_OUTCOMES
             else "steered_")
    out = {}
    for c in _AS_IS_COLS:
        out[c] = r.get(c, "")
    for c in _STEERED_COLS:
        out[c] = r.get("steered_" + c, "")
    for c in _FINAL_COLS:
        out[c] = _get(r, final + c)
    for c in _OR_STEER_COLS:
        out[c] = _get(r, final + c, "steered_" + c)
    for c in _CONF_COLS:
        out[c + "_median"] = _get(r, final + c + "_median",
                                  "steered_" + c + "_median")
    out.update(_mutation_columns(per_seed_rows))
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
