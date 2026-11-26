#!/usr/bin/env python3
"""
cross_summary.py
-------------------------
Aggregate per-MPNN-sequence ``passing_summary.csv`` files into a
single cross-sequence ranked table.

Context
=======
The NextFlow negative-steering pipeline runs one full single-cycle
negative-steering experiment per MPNN sequence.  Each produces its
own ``passing_summary.csv``, one row per unique (re-)folded sequence
ranked by ``rank_by_composite_score`` within that experiment.

Cross-sequence triage needs one representative row PER MPNN SEQUENCE.
Within each sequence the best non-empty tier wins, and within that tier
the lowest ``ra_eff_vs_truth_median``.  Reliability (tier) comes first
and accuracy (median) second, so a design with a slightly better
interface overlap cannot outrank one with a meaningfully better pose.
See _pick_representative.  Tiering uses n_pass and n_seeds only, with
no outcome shortcut.  See _tier_for_row.

Representatives are then stacked across sequences and re-ranked two
ways.  ``cross_rank_by_composite`` uses the jaccard-weighted
composite.  ``cross_rank_by_ra_eff`` uses tier then ra_eff, the same
policy as the per-sequence pick.  See _assign_cross_ranks.

Inputs
======
One ``passing_summary.csv`` per MPNN sequence, named by the flags
below.  A file with zero rows means that sequence had no passing
designs.

Each input path must yield the MPNN sequence name, such as
``design_12_seq_3``.  In precedence order: an explicit
``--passing-summary NAME=PATH`` pair, then the parent directory's
name, then the grandparent's.  The parent-directory default matches
the NextFlow layout, as in ``runs/design_12_seq_3/passing_summary.csv``.

Outputs
=======
``cross_sequence_summary.csv``, written to ``--output``, with one row
per MPNN sequence.  Columns:
  * mpnn_sequence              — MPNN sequence name
  * row_type                   — 'steered', 'control_scrambled' or
                                 'control_polyA'.  Controls are
                                 excluded from the cross-rank by
                                 design and appear at the bottom with
                                 every cross_rank column blank.
                                 Defaults to 'steered' when the
                                 workdir has no row_type.txt sidecar.
  * cross_rank_by_composite    — rank across sequences, 1 = best,
                                 tier A/B/C steered rows only
  * cross_rank_by_ra_eff       — parallel rank using raw ra_eff
  * cross_tier                 — A / B / C / none
  * cross_composite_score      — recomputed composite for ranking
  * within_sequence_rank       — rank_by_composite_score inside the
                                 sequence's own passing_summary.csv
  * passing_rows_in_sequence   — rows in that passing_summary
  * n_tier_A_in_sequence,      — how many landed in each tier
    n_tier_B_in_sequence,
    n_tier_C_in_sequence
  * every passing_summary.csv column for the picked row, prefixed
    ``rep_``

A sequence with an empty passing_summary.csv still appears, with
``cross_tier = none`` and the rep_* and cross-rank columns blank.
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make sibling scripts in bin/ importable, as negsteer_aggregate.py does.
# The tier-none fallback below calls extract_passing.extract_row to turn one
# aggregated_results row into a passing_summary-shaped dict.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from extract_passing import extract_row as _extract_passing_row  # noqa: E402

# ── Helpers ──────────────────────────────────────────────────────────

def _rewrite_workdir_path_to_published(
    raw_path: str,
    seq_name: str,
    published_runs_dir: Optional[Path],
) -> str:
    """Rewrite an in-work-dir absolute path to its published equivalent.

      in:  /…/work/b6/afc…/design_7_seq_3/cycle_0/steered/<…>.pdb
      out: <published_runs_dir>/design_7_seq_3/cycle_0/steered/<…>.pdb

    NEGSTEER_RUN_ONE stamps the per-prediction PDB's path in Nextflow's
    per-task ``work/<hash>/<seq_name>/...`` directory into the aggregated
    CSV's ``canonical_pdb`` columns.  Those paths are fragile.  The work/
    tree is scratch by design and ``nextflow clean`` deletes it, and a
    downstream task that does not bind that exact path into its container
    cannot read the file even while it exists on the host.  NEGSTEER_RUN_ONE
    also publishes the whole per-sequence workdir to
    ``${params.outdir}/negative_steering/runs/<seq_name>/``, a stable path,
    which is where this helper points the value.

    ``raw_path`` comes back unchanged when ``published_runs_dir`` is None
    (outside Nextflow and for older callers), when it has no seq_name segment
    (an unexpected layout), and when it is blank.  So the helper is safe to
    call on every row.
    """
    if not raw_path or published_runs_dir is None or not seq_name:
        return raw_path
    # Match /<seq_name>/ as a path segment, not a substring, so unrelated
    # paths containing the sequence name are left alone.
    parts = Path(raw_path).parts
    try:
        idx = parts.index(seq_name)
    except ValueError:
        # The rewrite is best-effort, so leaving the path alone is safe.
        return raw_path
    suffix_parts = parts[idx + 1:]
    if not suffix_parts:
        # The path ended at <seq_name>/ with nothing to rewrite.
        return raw_path
    return str(published_runs_dir / seq_name / Path(*suffix_parts))


def _try_float(v) -> Optional[float]:
    """Parse a string/number to float, returning None on failure or NaN."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        f = float(v)
    else:
        s = str(v).strip()
        if s == "":
            return None
        try:
            f = float(s)
        except ValueError:
            return None
    if f != f or f == float("inf") or f == float("-inf"):
        return None
    return f


def _try_int(v) -> Optional[int]:
    """Parse a string/number to int, returning None on failure."""
    if v is None:
        return None
    if isinstance(v, int):
        return v
    s = str(v).strip()
    if s == "":
        return None
    try:
        return int(s)
    except ValueError:
        # Tolerate "3.0".
        try:
            return int(float(s))
        except ValueError:
            return None


def _tier_for_row(row: Dict) -> str:
    """Determine tier from a passing_summary.csv row.

    Tier A: n_pass == n_seeds   (e.g. 3/3 pass-equivalent)
    Tier B: 1 < n_pass < n_seeds (e.g. 2/3)
    Tier C: n_pass == 1
    else:   "none"

    Tier comes from n_pass and n_seeds only.  The aggregator emits n_pass as
    pose_holds plus clean_steered, because both count as a pass.  A
    clean_steered seed had zero contamination on its mutated positions and
    passed the structural check, so reversion was correctly skipped and its
    steered structure is final.

    outcome=='no_reversion' is NOT a tier shortcut.  A group whose seeds all
    show wrong placement with no contamination carries it with n_pass==0,
    because those seeds are no_data or pose_collapses rather than
    clean_steered, and correctly lands in tier none.  A cold-start group
    where every seed passed carries it with n_pass==n_seeds, landing in
    tier A.
    """
    n_pass = _try_int(row.get("n_pass"))
    n_seeds = _try_int(row.get("n_seeds"))
    if n_pass is None or n_seeds is None or n_seeds <= 0:
        # Too little data to place, so "none" keeps cross-ranking from
        # guessing.
        return "none"
    if n_pass == n_seeds:
        return "A"
    if 1 < n_pass < n_seeds:
        return "B"
    if n_pass == 1:
        return "C"
    return "none"


_TIER_ORDER = {"A": 0, "B": 1, "C": 2, "none": 3}


def _composite_score(row: Dict) -> Optional[float]:
    """Recompute composite score from a passing_summary.csv row, matching
    ``_ranking_composite`` in negsteer_aggregate.py:

        composite = true_jaccard_median − 0.05 × ra_eff_vs_truth_median

    One angstrom of pose error therefore costs 0.05 of Jaccard, and higher is
    better.  Returns None when either component is missing.  Both are
    medians, because passing_summary.csv columns are already medians after
    the per-sequence aggregation.
    """
    ra = _try_float(row.get("ra_eff_vs_truth_median"))
    tj = _try_float(row.get("true_jaccard_median"))
    if ra is None or tj is None:
        return None
    return tj - 0.05 * ra


def _within_sequence_rank(row: Dict) -> int:
    """Return the row's rank_by_composite_score, 1 = best.  A missing value
    becomes a large sentinel, sorting such rows last."""
    v = _try_int(row.get("rank_by_composite_score"))
    if v is None:
        return 10**9
    return v


def _median_ra_eff(row: Dict) -> float:
    """The row's ra_eff_vs_truth_median, where lower is better.

    The column is already stage-aware.  extract_passing.extract_row sources it
    from reverted_* where reversion was attempted and steered_* otherwise, so
    no correction is needed here.  A missing or unparseable value becomes
    +inf, sorting such rows last.
    """
    v = _try_float(row.get("ra_eff_vs_truth_median"))
    return v if v is not None else float("inf")


# ── Input parsing ────────────────────────────────────────────────────

_SEQ_NAME_RE = re.compile(r"^[A-Za-z0-9_.+\-]+$")


def _infer_seq_name(path: Path, mode: str) -> str:
    """Derive the MPNN sequence name from a passing_summary.csv path.

    mode: "parent" (default) or "grandparent".
    """
    path = path.resolve()
    if mode == "parent":
        name = path.parent.name
    elif mode == "grandparent":
        name = path.parent.parent.name
    else:
        raise ValueError(f"unknown seq-name inference mode: {mode}")
    if not name or not _SEQ_NAME_RE.match(name):
        raise ValueError(
            f"inferred sequence name {name!r} from {path} is empty or "
            f"contains unexpected characters.  Use "
            f"--passing-summary NAME=PATH for an explicit name."
        )
    return name


def _parse_passing_summary_arg(arg: str, seq_name_mode: str
                               ) -> Tuple[str, Path]:
    """Parse a --passing-summary argument, returning (seq_name, path).

    Accepts either "NAME=PATH" or bare "PATH" (name inferred).
    """
    # Split on the FIRST '=' only, since HPC paths sometimes contain '='.
    if "=" in arg:
        head, sep, rest = arg.partition("=")
        candidate_path = Path(rest)
        if _SEQ_NAME_RE.match(head) and candidate_path.exists():
            return head, candidate_path
        # A head containing '/' or similar is not a seq name, so the whole
        # argument is treated as a path.
    path = Path(arg)
    name = _infer_seq_name(path, seq_name_mode)
    return name, path


# ── Core aggregation ─────────────────────────────────────────────────

def _pick_representative(rows: List[Dict]) -> Optional[Dict]:
    """Tier-then-median pick for one sequence's passing_summary.csv rows.
    Returns None for empty input, meaning no passing designs.

    Reliability comes first, in tier order A (all seeds pass), B (some), C
    (one), none.  A design where more seeds independently land the same
    correct pose outranks one that merely has a lower median by chance.

    Within a tier, rank by median ra_eff ascending.  The composite score
    (true_jaccard - 0.05*ra_eff) was used here before, but it can let a
    design with a slightly better interface overlap outrank one with a
    meaningfully better pose.  Downstream cross-ranking still offers both
    cross_rank_by_composite and cross_rank_by_ra_eff, so this choice only
    affects which single design represents a sequence.
    """
    if not rows:
        return None
    tagged = []
    for r in rows:
        tier = _tier_for_row(r)
        tagged.append((_TIER_ORDER.get(tier, 99),
                       _median_ra_eff(r),
                       tier,
                       r))
    tagged.sort(key=lambda t: (t[0], t[1]))
    best_tier_order, _, best_tier, best_row = tagged[0]
    # A best tier of "none" means every row is ineligible.  That should
    # never happen for a non-empty passing_summary.csv, where every row is at
    # least clean_steered or pose_holds and so tier A/B/C.  Guard anyway.
    if best_tier == "none":
        return None
    out = dict(best_row)
    out["_picked_tier"] = best_tier
    return out


def _pick_rep_from_aggregated(
    passing_summary_path: Path,
) -> Optional[Dict]:
    """Tier-none fallback for a sequence with an empty ``passing_summary.csv``.

    Empty means every prediction failed the eligibility gate in
    ``_compute_unified_ranks`` (negsteer_aggregate.py), usually because
    steered ra_eff was far above the 5 Å threshold.  The cohort plot still
    needs something, so score every row of the sibling
    ``aggregated_results.csv`` raw and pass the best through
    extract_passing.extract_row.  Returns None when that file is missing or
    no row has both composite components.

    The returned dict carries ``_picked_tier="none"``, keeping the row out of
    ``aggregate``'s cross-rank, and the raw composite as
    ``_fallback_composite`` for the caller to stamp onto
    ``cross_composite_score``.

    Scoring is stage-aware, matching ``_ranking_metric`` and
    ``_ranking_composite`` in negsteer_aggregate.py.  ``true_jaccard`` always
    comes from ``steered_true_jaccard_median``, because interface identity
    belongs to the steered prediction rather than the reverted pose.
    ``ra_eff`` comes from ``reverted_*`` for ``pose_holds`` rows and
    ``steered_*`` otherwise.

    Skipping the eligibility gate is deliberate.  A best-of-bad-attempts view
    beats grey-hatched cells everywhere, and the tier stripe already tells the
    reader these are not passing designs.
    """
    agg_path = passing_summary_path.parent / "aggregated_results.csv"
    if not agg_path.is_file():
        return None
    try:
        with open(agg_path) as f:
            rows = list(csv.DictReader(f))
    except OSError as e:
        print(f"WARNING: could not read {agg_path}: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return None
    if not rows:
        return None

    # Best row by composite, skipping any row where a component does not
    # parse as a finite float.  The stage-aware key choice mirrors the
    # struct_prefix logic in extract_passing.extract_row and _ranking_metric
    # in negsteer_aggregate.py.
    best_row = None
    best_composite = None
    for r in rows:
        is_pose_holds = (r.get("outcome") == "pose_holds")
        ra_key = ("reverted_ra_eff_vs_truth_median"
                  if is_pose_holds
                  else "steered_ra_eff_vs_truth_median")
        tj_key = "steered_true_jaccard_median"
        ra = _try_float(r.get(ra_key, ""))
        tj = _try_float(r.get(tj_key, ""))
        if ra is None or tj is None:
            continue
        comp = tj - 0.05 * ra
        if best_composite is None or comp > best_composite:
            best_composite = comp
            best_row = r

    if best_row is None:
        return None

    # per_seed_rows=None leaves the mutation columns blank, which is fine
    # here because the mutation set is informational and gates nothing.
    rep = _extract_passing_row(best_row, per_seed_rows=None)
    rep["_picked_tier"] = "none"
    rep["_fallback_composite"] = best_composite
    return rep


class _AggregateInputError(ValueError):
    """Hard input error raised by _read_passing_summary_inputs."""


def _read_passing_summary_inputs(
    sequences: List[Tuple[str, Path]],
    strict: bool,
) -> Tuple[Dict[str, List[Dict]], Dict[str, Path], List[str]]:
    """Read every passing_summary.csv. Returns the rows, their paths and the
    canonical fieldname order."""
    per_seq_rows: Dict[str, List[Dict]] = {}
    source_paths: Dict[str, Path] = {}
    passing_summary_fieldnames: List[str] = []
    seen_fieldnames = False

    for seq_name, path in sequences:
        if seq_name in per_seq_rows:
            raise _AggregateInputError(
                f"duplicate sequence name {seq_name!r} (second seen at {path})"
            )
        if not path.exists():
            msg = f"passing_summary.csv not found for {seq_name}: {path}"
            if strict:
                raise _AggregateInputError(msg)
            print(f"WARNING: {msg}", file=sys.stderr)
            per_seq_rows[seq_name] = []
            source_paths[seq_name] = path
            continue
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = reader.fieldnames or []
        except Exception as e:
            msg = f"failed to read {path} for {seq_name}: {e}"
            if strict:
                raise _AggregateInputError(msg)
            print(f"WARNING: {msg}", file=sys.stderr)
            per_seq_rows[seq_name] = []
            source_paths[seq_name] = path
            continue
        per_seq_rows[seq_name] = rows
        source_paths[seq_name] = path
        # The first non-empty fieldnames become the canonical rep_* column
        # list.  A mismatch means pipeline version drift, so warn only.
        if fieldnames and not seen_fieldnames:
            passing_summary_fieldnames = list(fieldnames)
            seen_fieldnames = True
        elif (fieldnames and seen_fieldnames
              and fieldnames != passing_summary_fieldnames):
            print(f"WARNING: fieldnames in {path} differ from the first "
                  f"file's fieldnames.  Will emit superset of both.",
                  file=sys.stderr)
            for col in fieldnames:
                if col not in passing_summary_fieldnames:
                    passing_summary_fieldnames.append(col)
    return per_seq_rows, source_paths, passing_summary_fieldnames


def _pick_or_fallback_representative(
    rows: List[Dict],
    source_path: Path,
) -> Tuple[Optional[Dict], bool]:
    """Pick the representative row. A sequence with no passing rows falls back
    to its aggregated_results.csv."""
    rep = _pick_representative(rows)
    if rep is not None:
        return rep, False
    rep = _pick_rep_from_aggregated(source_path)
    return rep, rep is not None


def _resolve_run_one_runtime_sec(
    rep: Optional[Dict], source_path: Path
) -> str:
    """Cumulative wall-clock for a sequence: prefer rep column, fall back to sidecar."""
    run_one_runtime_sec = ""
    if rep is not None:
        rep_runtime = rep.get("run_one_runtime_sec", "") or ""
        if rep_runtime:
            run_one_runtime_sec = rep_runtime
    if not run_one_runtime_sec:
        runtime_file = source_path.parent / "run_one_runtime_sec.txt"
        if runtime_file.exists():
            try:
                raw = runtime_file.read_text().strip()
                float(raw)   # validate
                run_one_runtime_sec = raw
            except (OSError, ValueError):
                pass
    return run_one_runtime_sec


def _read_row_type_sidecar(source_path: Path) -> str:
    """Read the row_type.txt sidecar. Absent or unreadable defaults to
    'steered'."""
    row_type = "steered"
    row_type_file = source_path.parent / "row_type.txt"
    if row_type_file.exists():
        try:
            rt = row_type_file.read_text().strip()
            if rt:
                row_type = rt
        except OSError:
            pass
    return row_type


def _stamp_ranks(scored, column: str) -> None:
    """Sort (value, rec) pairs tier-first and stamp 1-based ranks.

    Tier first in both rankings: a tier B record with a slightly higher raw
    composite still ranks below every tier A record, because tier is a
    reliability measure and the composite is not.

    `value` is already signed so that ascending order is "better first", so
    the caller negates a composite and leaves an ra_eff alone.

    id() is the third key, which is what the two separate rank functions did
    before this was factored out.  It is a memory address, so two records
    tied on tier and value rank in an order that depends on allocation.
    Dropping it would make list.sort's own stability decide, which is input
    order and reproducible.  That changes committed output for any cohort
    containing ties, so it is left as it was.
    """
    scored.sort(key=lambda pair: (_TIER_ORDER[pair[1]["cross_tier"]],
                                  pair[0], id(pair[1])))
    for rank_idx, (_value, rec) in enumerate(scored, start=1):
        rec[column] = rank_idx


def _assign_cross_ranks(records: List[Dict]) -> None:
    """Compute and stamp cross_rank_by_composite / cross_rank_by_ra_eff in place."""
    scored_ra = []       # list of (ra_eff_value, rec) for ra-eff rank
    scored_composite = []  # list of (composite_value, rec) for composite rank
    for rec in records:
        if rec.get("row_type", "steered") != "steered":
            continue
        rep_cols = {
            "ra_eff_vs_truth_median":
                rec.get("rep_ra_eff_vs_truth_median", ""),
            "true_jaccard_median":
                rec.get("rep_true_jaccard_median", ""),
        }
        # Requiring a placed tier keeps the "none"-tier stubs out.
        if rec["cross_tier"] in ("A", "B", "C"):
            comp = _composite_score(rep_cols)
            if comp is not None:
                rec["cross_composite_score"] = f"{comp:.6f}"
                scored_composite.append((-comp, rec))   # higher is better
            ra = _try_float(rep_cols["ra_eff_vs_truth_median"])
            if ra is not None:
                scored_ra.append((ra, rec))             # lower is better

    _stamp_ranks(scored_composite, "cross_rank_by_composite")
    _stamp_ranks(scored_ra, "cross_rank_by_ra_eff")


def _emit_aggregate_summary(records: List[Dict], output_path: Path) -> None:
    """Print the cross-sequence summary stats to stderr."""
    n_total = len(records)
    n_ranked = sum(1 for r in records
                   if r.get("cross_rank_by_composite") != "")
    # Steered rows only, the same filter _assign_cross_ranks applies.  An
    # unfiltered count would disagree with n_ranked above, counting a tier-A
    # control under A while its cross_rank_by_composite stayed blank.
    by_tier = {"A": 0, "B": 0, "C": 0, "none": 0}
    for r in records:
        if (r.get("row_type") or "steered") != "steered":
            continue
        by_tier[r["cross_tier"]] += 1
    by_row_type: Dict[str, int] = {}
    for r in records:
        rt = r.get("row_type", "steered")
        by_row_type[rt] = by_row_type.get(rt, 0) + 1
    print(f"Wrote {output_path}", file=sys.stderr)
    print(f"  {n_total} sequences, {n_ranked} placed in tiers A/B/C",
          file=sys.stderr)
    print(f"  tier breakdown (steered only): A={by_tier['A']} B={by_tier['B']} "
          f"C={by_tier['C']} none={by_tier['none']}",
          file=sys.stderr)
    if any(rt != "steered" for rt in by_row_type):
        rt_summary = ", ".join(
            f"{rt}={n}" for rt, n in sorted(by_row_type.items())
        )
        print(f"  row_type breakdown: {rt_summary}", file=sys.stderr)


def _load_scored_metadata(path: Optional[Path]) -> Dict[str, Dict]:
    """Load scored_metadata.csv into a {mpnn_sequence: row} lookup.

    The join key is f"design_{row['design']}_seq_{row['seq']}", matching the
    mpnn_sequence convention used in FASTA headers and per-sequence workdir
    names.  Returns an empty dict when `path` is None or the file is missing,
    and the joined columns then come out blank as the intended no-op.
    """
    if path is None:
        return {}
    if not path.is_file():
        print(f"WARNING: --scored-metadata file not found: {path} "
              f"(corrected_receptor / designed_residues / native_residues "
              f"columns will be blank)", file=sys.stderr)
        return {}
    lookup: Dict[str, Dict] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            design = row.get("design", "").strip()
            seq = row.get("seq", "").strip()
            if not design or not seq:
                continue
            key = f"design_{design}_seq_{seq}"
            lookup[key] = row
    print(f"  loaded {len(lookup)} scored_metadata rows from {path.name}")
    return lookup


def aggregate(
    sequences: List[Tuple[str, Path]],
    output_path: Path,
    strict: bool,
    published_runs_dir: Optional[Path] = None,
    scored_metadata_path: Optional[Path] = None,
) -> int:
    """Read every (seq_name, passing_summary.csv) pair, pick a
    representative, and write a ranked cross-sequence CSV.

    strict: True aborts on a missing or unreadable passing_summary.csv.
            False, the default, writes a diagnostic stub row and continues.
    published_runs_dir: see _rewrite_workdir_path_to_published.
    scored_metadata_path: see _load_scored_metadata.
    """
    if not sequences:
        print("ERROR: no inputs supplied", file=sys.stderr)
        return 2

    try:
        per_seq_rows, source_paths, passing_summary_fieldnames = (
            _read_passing_summary_inputs(sequences, strict)
        )
    except _AggregateInputError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    scored_metadata = _load_scored_metadata(scored_metadata_path)

    records = []
    for seq_name, rows in per_seq_rows.items():
        # The tier-none fallback stamps real metric values onto tier-none
        # rows, so they appear in the cohort summary plot instead of showing
        # as grey-hatched cells.  See _pick_rep_from_aggregated.
        rep, rep_is_fallback = _pick_or_fallback_representative(
            rows, source_paths[seq_name]
        )
        if rep_is_fallback and rep is not None:
            # extract_passing.extract_row can return columns the fieldname
            # union lacks, which the CSV writer would silently drop.
            for col in rep.keys():
                if col.startswith("_"):
                    continue
                if col not in passing_summary_fieldnames:
                    passing_summary_fieldnames.append(col)

        tier_counts = {"A": 0, "B": 0, "C": 0, "none": 0}
        for r in rows:
            tier_counts[_tier_for_row(r)] += 1

        # Cumulative wall-clock.  extract_passing.py stamps it onto every
        # passing_summary row and writes it as a sidecar in the sequence's
        # workdir.  The representative row's column is canonical and matches
        # what wet-lab users see.  The sidecar fallback covers tier-none
        # sequences whose empty passing_summary.csv has no representative.
        run_one_runtime_sec = _resolve_run_one_runtime_sec(
            rep, source_paths[seq_name]
        )

        # NEGSTEER_CONTROLS writes 'control_scrambled' or 'control_polyA'
        # next to its passing_summary.csv, so this needs no column changes to
        # extract_passing.py or negative_steering_run_one.sh.  Defaulting to
        # 'steered' keeps workdirs made before controls existed readable.
        row_type = _read_row_type_sidecar(source_paths[seq_name])

        rec = {
            "mpnn_sequence": seq_name,
            "row_type": row_type,
            "source_passing_summary": str(source_paths[seq_name]),
            "passing_rows_in_sequence": len(rows),
            "n_tier_A_in_sequence": tier_counts["A"],
            "n_tier_B_in_sequence": tier_counts["B"],
            "n_tier_C_in_sequence": tier_counts["C"],
            "cross_tier": (rep.get("_picked_tier", "none")
                           if rep else "none"),
            "within_sequence_rank": (
                _within_sequence_rank(rep)
                if (rep and _within_sequence_rank(rep) < 10**9)
                else ""
            ),
            "cross_composite_score": "",   # filled below
            "cross_rank_by_composite": "",
            "cross_rank_by_ra_eff": "",
            # Top-level, so triage can read sequence-level throughput
            # without unpacking the rep_* columns.  Blank when neither the
            # sidecar nor the representative row exists, as in old workdirs.
            "run_one_runtime_sec": run_one_runtime_sec,
        }
        # canonical_pdb is rewritten from the in-work-dir absolute path to
        # the published equivalent.  See
        # _rewrite_workdir_path_to_published.
        for col in passing_summary_fieldnames:
            val = rep.get(col, "") if rep else ""
            if col == "canonical_pdb":
                val = _rewrite_workdir_path_to_published(
                    val, seq_name, published_runs_dir,
                )
            rec[f"rep_{col}"] = val

        # The cross-rank pass below rightly skips tier-none rows, since they
        # should not compete with tier A/B/C.  But the cohort summary plot
        # reads cross_composite_score as its first column, so the score from
        # _pick_rep_from_aggregated is stamped on while
        # cross_rank_by_composite and cross_rank_by_ra_eff stay blank.
        if rep_is_fallback and rep is not None:
            comp = rep.get("_fallback_composite")
            if comp is not None:
                rec["cross_composite_score"] = f"{comp:.6f}"

        # With --scored-metadata supplied, the designed receptor sequence
        # and the native residues it replaced go straight into
        # cross_sequence_summary.csv, so downstream consumers need not
        # re-parse mpnn_corrected.fasta.
        meta = scored_metadata.get(seq_name)
        if meta is not None:
            rec["corrected_receptor"] = meta.get("corrected_receptor", "")
            rec["designed_residues"] = meta.get("designed_residues", "")
            rec["native_residues"] = meta.get("native_residues", "")
        else:
            rec["corrected_receptor"] = ""
            rec["designed_residues"] = ""
            rec["native_residues"] = ""

        records.append(rec)

    # The score comes from the representative row's median columns, and a
    # sequence with no representative gets None.  Controls are excluded from
    # the ranking by design, being diagnostic rather than promotable, and
    # still appear with every cross_rank_* column blank.  The onComplete
    # miscalibration check reads them back from the final CSV.
    _assign_cross_ranks(records)

    # cross_rank_by_composite, then cross_tier, then name for determinism.
    def _final_sort_key(rec):
        rc = rec.get("cross_rank_by_composite", "")
        try:
            rc_val = int(rc)
        except (ValueError, TypeError):
            rc_val = 10**9
        tier_val = _TIER_ORDER[rec["cross_tier"]]
        return (rc_val, tier_val, rec["mpnn_sequence"])
    records.sort(key=_final_sort_key)

    # ─── Write CSV ─────────────────────────────────────────────────
    # Metadata first, then the rep_* columns in passing_summary.csv order.
    meta_cols = [
        "mpnn_sequence",
        "row_type",
        "cross_rank_by_composite",
        "cross_rank_by_ra_eff",
        "cross_tier",
        "cross_composite_score",
        "within_sequence_rank",
        "passing_rows_in_sequence",
        "n_tier_A_in_sequence",
        "n_tier_B_in_sequence",
        "n_tier_C_in_sequence",
        "run_one_runtime_sec",
        "source_passing_summary",
        # Blank when --scored-metadata is not supplied.
        # corrected_receptor: full receptor, native at fixed positions and
        #                     MPNN at design positions
        # designed_residues:  MPNN AA slice at design positions only,
        #                     pipe-joined per de novo segment
        # native_residues:    native AA slice at the same positions
        "corrected_receptor",
        "designed_residues",
        "native_residues",
    ]
    rep_cols = [f"rep_{c}" for c in passing_summary_fieldnames]
    fieldnames = meta_cols + rep_cols

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)

    # ─── Summary to stderr ─────────────────────────────────────────
    _emit_aggregate_summary(records, output_path)
    return 0


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--passing-summary",
        action="append",
        default=[],
        metavar="[NAME=]PATH",
        help="Path to a per-sequence passing_summary.csv.  Optional "
             "'NAME=' prefix sets the MPNN sequence name explicitly; "
             "otherwise it is inferred from the path.  Repeat the "
             "flag for each sequence."
    )
    parser.add_argument(
        "--passing-summary-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="Directory containing one subdirectory per MPNN "
             "sequence, each with its own passing_summary.csv.  "
             "Equivalent to passing every <DIR>/*/passing_summary.csv "
             "via --passing-summary.  Repeat to combine multiple "
             "source dirs."
    )
    parser.add_argument(
        "--seq-name-from-parent-dir",
        dest="seq_name_mode", action="store_const", const="parent",
        default="parent",
        help="Infer sequence name from the parent directory of each "
             "passing_summary.csv (default).",
    )
    parser.add_argument(
        "--seq-name-from-grandparent-dir",
        dest="seq_name_mode", action="store_const", const="grandparent",
        help="Infer sequence name from the grandparent directory of "
             "each passing_summary.csv.  Use this if files live at "
             "<seq>/cycle_0/passing_summary.csv.",
    )
    parser.add_argument(
        "--output", "-o", type=Path, required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Abort on missing/unreadable input files instead of "
             "recording them as empty stubs.",
    )
    parser.add_argument(
        "--published-runs-dir", type=Path, default=None,
        metavar="DIR",
        help="Path to ${params.outdir}/negative_steering/runs (the "
             "published per-sequence workdir tree).  When provided, "
             "every rep_canonical_pdb value (which is "
             "stamped by NEGSTEER_RUN_ONE as an in-work-dir absolute "
             "path) is rewritten to the published equivalent under "
             "<DIR>/<seq_name>/<…>.  Without this flag the canonical "
             "PDB paths in the output CSV point into a Nextflow work/ "
             "tree that's scratch by design and unbound in downstream "
             "container tasks — leaving the rewrite off is the legacy "
             "behaviour, kept for non-Nextflow callers."
    )
    parser.add_argument(
        "--scored-metadata", type=Path, default=None,
        metavar="CSV",
        help="Optional path to scored_metadata.csv (emitted by "
             "MPNN_DESIGN_REGION_SCORE).  When provided, three extra "
             "columns are joined onto every output row by mpnn_sequence "
             "(= design_<design>_seq_<seq>): corrected_receptor (full "
             "receptor sequence with native at fixed + MPNN at design "
             "positions), designed_residues (pipe-joined MPNN AA slice "
             "at design positions only), and native_residues (the same "
             "slice on the native receptor — what each MPNN residue is "
             "replacing).  Blank columns when the join key is missing."
    )

    args = parser.parse_args()

    # A relative --published-runs-dir would be stamped verbatim into
    # rep_canonical_pdb, then resolve against the wrong directory
    # downstream, because each container task runs in its own work dir.
    # main.nf already normalises params.outdir to absolute, so this is a
    # second guard for standalone or other-workflow callers.
    if args.published_runs_dir is not None:
        args.published_runs_dir = args.published_runs_dir.resolve()

    sequences: List[Tuple[str, Path]] = []
    seen_names = set()

    for arg in args.passing_summary:
        try:
            seq_name, path = _parse_passing_summary_arg(
                arg, args.seq_name_mode
            )
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        if seq_name in seen_names:
            print(f"ERROR: duplicate sequence name {seq_name!r}",
                  file=sys.stderr)
            return 2
        seen_names.add(seq_name)
        sequences.append((seq_name, path))

    for dir_arg in args.passing_summary_dir:
        d = Path(dir_arg)
        if not d.is_dir():
            print(f"ERROR: not a directory: {d}", file=sys.stderr)
            return 2
        # Each sequence subdirectory should hold a passing_summary.csv.
        # Missing files are handled downstream.
        for sub in sorted(d.iterdir()):
            if not sub.is_dir():
                continue
            path = sub / "passing_summary.csv"
            seq_name = sub.name
            if not _SEQ_NAME_RE.match(seq_name):
                print(f"WARNING: skipping {sub} (name not a valid "
                      f"sequence identifier)", file=sys.stderr)
                continue
            if seq_name in seen_names:
                # Already added via --passing-summary.
                continue
            seen_names.add(seq_name)
            sequences.append((seq_name, path))

    if not sequences:
        print("ERROR: no inputs found.  Supply --passing-summary "
              "and/or --passing-summary-dir.", file=sys.stderr)
        return 2

    return aggregate(sequences, args.output, args.strict,
                     published_runs_dir=args.published_runs_dir,
                     scored_metadata_path=args.scored_metadata)


if __name__ == "__main__":
    raise SystemExit(main())
