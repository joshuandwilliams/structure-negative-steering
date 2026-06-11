#!/usr/bin/env python3
"""
cross_sequence_summary.py
-------------------------
Aggregate per-MPNN-sequence ``passing_summary.csv`` files into a
single cross-sequence ranked table.

Context
=======
The NextFlow negative-steering pipeline runs one full single-cycle
negative-steering experiment per MPNN sequence.  Each experiment
produces its own ``passing_summary.csv`` containing one row per
unique (re-)folded sequence ranked by ``rank_by_composite_score``
within that experiment.

For cross-sequence triage we want one representative row PER MPNN
SEQUENCE, picked using the tier-then-composite policy documented in
notes 11.  The aggregator emits n_pass directly (= number of seeds
classified as pose_holds or clean_steered — both pass-equivalent).
Tiering is from n_pass / n_seeds only, with no outcome shortcut:

    Tier A: n_pass == n_seeds   (e.g. 3/3 pass-equivalent)
    Tier B: 1 < n_pass < n_seeds   (e.g. 2/3)
    Tier C: n_pass == 1            (e.g. 1/3)
    else:   "none"                 (includes wrong-placement
                                    no-contamination groups whose
                                    outcome label is "no_reversion"
                                    but whose seeds all sit in no_data
                                    / pose_collapses — n_pass == 0)

Within each sequence, the best non-empty tier wins and the top-
ranked row from that tier (lowest ``rank_by_composite_score``) is
that sequence's representative.  Representatives are then stacked
across sequences and re-ranked using the same composite score
(``true_jaccard − 0.05 × ra_eff_vs_truth``; higher is better) so
the cross-sequence rank is directly comparable to per-sequence
ranks.

Inputs
======
One ``passing_summary.csv`` per MPNN sequence, passed via one or more
``--passing-summary`` flags or as the inner files of a glob under
``--passing-summary-glob``.  Each input file carries one or more
rows (may be empty if the sequence had no passing designs).

Sequence identification
=======================
The MPNN sequence name (e.g. ``design_12_seq_3``) must be inferable
from each input file's path.  Three options, in precedence order:

1. Explicit pairs: ``--passing-summary design_12_seq_3=<path>``
2. ``--seq-name-from-parent-dir`` (default): use the name of the
   parent directory of each passing_summary.csv.  This matches the
   NextFlow layout where each sequence's experiment root is named
   after the sequence (e.g. ``runs/design_12_seq_3/passing_summary.csv``).
3. ``--seq-name-from-grandparent-dir``: use the name of the
   grandparent directory instead.  Useful if the file is nested one
   level deeper (e.g. ``<seq>/cycle_0/passing_summary.csv``).

Outputs
=======
``cross_sequence_summary.csv`` — one row per MPNN sequence
(representative pick + cross-sequence rank), written to
``--output`` or ``<output-dir>/cross_sequence_summary.csv``.

Columns:
  * mpnn_sequence              — the MPNN sequence name
  * row_type                   — 'steered' (production), 'control_scrambled',
                                 or 'control_polyA' (Task 6 negative
                                 controls).  Defaults to 'steered' when
                                 the per-sequence workdir has no
                                 row_type.txt sidecar.  Only 'steered'
                                 rows participate in the cross-rank
                                 ranking — controls are diagnostic-only,
                                 surfaced at the bottom of the table
                                 with all cross_rank columns blank.
  * cross_rank_by_composite    — rank across all sequences, 1 = best,
                                 only populated for tier A/B/C steered rows
  * cross_rank_by_ra_eff       — parallel rank using raw ra_eff
  * cross_tier                 — A / B / C / none
  * cross_composite_score      — recomputed composite for ranking
  * within_sequence_rank       — row's rank_by_composite_score inside
                                 its own passing_summary.csv
  * passing_rows_in_sequence   — how many rows that sequence's
                                 passing_summary had
  * n_pose_holds_sequences     — same, counting rows at tier A (3/3)
  * (then every column from passing_summary.csv, prefixed with
    ``rep_``, for the picked row).

Sequences whose passing_summary.csv is empty are still included in
the output with ``cross_tier = none``, all rep_* columns
blank, and cross-rank columns blank.
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make sibling scripts in the same bin/ directory importable.  Mirrors
# the pattern boltz2_iterate_steering.py uses (line 91).  Needed for
# the tier-none representative fallback below, which calls
# extract_passing.extract_row to translate one aggregated_results row
# into a passing_summary-shaped dict.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from extract_passing import extract_row as _extract_passing_row


# ── Helpers ──────────────────────────────────────────────────────────

def _rewrite_workdir_path_to_published(
    raw_path: str,
    seq_name: str,
    published_runs_dir: Optional[Path],
) -> str:
    """Rewrite an in-work-dir absolute path to its published equivalent.

    NEGSTEER_RUN_ONE writes per-prediction PDBs into Nextflow's
    per-task ``work/<hash>/<seq_name>/...`` directory and stamps that
    absolute path into ``canonical_pdb`` columns of the per-sequence
    aggregated CSV.  Those paths are inherently fragile: the work/
    tree is scratch by design; ``nextflow clean`` deletes it; and
    downstream tasks that don't bind that exact work-dir-path into
    their container can't read the file even when it exists on the
    host.

    Because ``NEGSTEER_RUN_ONE`` publishes the entire per-sequence
    workdir to ``${params.outdir}/negative_steering/runs/<seq_name>/``,
    the same file always exists at a stable, predictable path.  This
    helper rewrites the in-work absolute path to point there:

      in:  /…/work/b6/afc…/design_7_seq_3/cycle_0/steered/<…>.pdb
      out: <published_runs_dir>/design_7_seq_3/cycle_0/steered/<…>.pdb

    When ``published_runs_dir`` is None (script invoked outside
    Nextflow / legacy callers), or when ``raw_path`` doesn't contain
    the seq_name segment (unexpected layout), or when raw_path is
    blank, the value is returned unchanged.  This keeps the helper
    safe to call unconditionally on every row.
    """
    if not raw_path or published_runs_dir is None or not seq_name:
        return raw_path
    # Find /<seq_name>/ as a path segment (not a substring match,
    # to avoid mangling unrelated paths that happen to contain the
    # sequence name as a substring).
    parts = Path(raw_path).parts
    try:
        idx = parts.index(seq_name)
    except ValueError:
        # No matching segment — leave the path alone.  This is the
        # safe default; the rewrite is a best-effort improvement,
        # not a hard requirement.
        return raw_path
    suffix_parts = parts[idx + 1:]
    if not suffix_parts:
        # Path ended at <seq_name>/ with no trailing components —
        # nothing meaningful to rewrite.
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
    # NaN and +/-inf are not valid ranking values.
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
        # Could be "3.0"; tolerate.
        try:
            return int(float(s))
        except ValueError:
            return None


def _tier_for_row(row: Dict) -> str:
    """Determine tier from a passing_summary.csv row.

    Tier is derived from n_pass / n_seeds only.  n_pass is emitted by
    the aggregator as (pose_holds + clean_steered) — both verdicts
    count as pass-equivalent (a clean_steered seed had zero
    contamination on mutated positions AND passed structural, so
    reversion was correctly skipped and the steered structure is the
    final verdict).

    Tier A: n_pass == n_seeds   (e.g. 3/3 pass-equivalent)
    Tier B: 1 < n_pass < n_seeds (e.g. 2/3)
    Tier C: n_pass == 1
    else:   "none"

    Note: outcome=='no_reversion' is NOT a tier shortcut.  A group
    whose seeds are all wrong-placement-no-contamination has
    outcome=='no_reversion' but n_pass==0 (seeds are no_data /
    pose_collapses, not clean_steered) and correctly lands in tier
    none.  Cold-start groups where every seed passed have
    outcome=='no_reversion' and n_pass==n_seeds → tier A.
    """
    n_pass = _try_int(row.get("n_pass"))
    n_seeds = _try_int(row.get("n_seeds"))
    if n_pass is None or n_seeds is None or n_seeds <= 0:
        # Insufficient data to place in any tier.  Treat as "none"
        # so cross-ranking ignores it rather than guessing.
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
    """Recompute composite score from a passing_summary.csv row.

    Formula matches ``_ranking_composite`` in boltz2_iterate_steering:
        composite = true_jaccard_median − 0.05 × ra_eff_vs_truth_median

    Higher is better.  Returns None if either component is missing
    or unparseable.  Uses medians throughout — passing_summary.csv
    columns are always medians after the per-sequence aggregation.
    """
    ra = _try_float(row.get("ra_eff_vs_truth_median"))
    tj = _try_float(row.get("true_jaccard_median"))
    if ra is None or tj is None:
        return None
    return tj - 0.05 * ra


def _within_sequence_rank(row: Dict) -> int:
    """Return the row's rank_by_composite_score (ascending: 1 = best).
    Missing/unparseable -> large sentinel so bad rows sort last."""
    v = _try_int(row.get("rank_by_composite_score"))
    if v is None:
        return 10**9
    return v


# ── Input parsing ────────────────────────────────────────────────────

_SEQ_NAME_RE = re.compile(r"^[A-Za-z0-9_.+\-]+$")


def _infer_seq_name(path: Path, mode: str) -> str:
    """Derive the MPNN sequence name from a passing_summary.csv path.

    mode:
      - "parent":       parent directory name  (default)
      - "grandparent":  parent of parent directory name
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
    # Only split on the FIRST '=' because HPC paths sometimes contain
    # '=' in unexpected places (rare, but we want to be robust).
    if "=" in arg:
        head, sep, rest = arg.partition("=")
        # If head looks like a plausible seq name and rest is a real
        # path, treat as explicit pair.  Otherwise fall through to
        # bare-path handling.
        candidate_path = Path(rest)
        if _SEQ_NAME_RE.match(head) and candidate_path.exists():
            return head, candidate_path
        # If head doesn't look like a seq name (contains '/' etc.),
        # fall back to treating the whole arg as a path.
    path = Path(arg)
    name = _infer_seq_name(path, seq_name_mode)
    return name, path


# ── Core aggregation ─────────────────────────────────────────────────

def _pick_representative(rows: List[Dict]) -> Optional[Dict]:
    """Tier-then-composite pick for a single sequence's
    passing_summary.csv rows.  Returns None for an empty input
    (sequence had no passing designs)."""
    if not rows:
        return None
    # Tag every row with its tier + within-seq rank, then pick the
    # row that minimises (tier_order, within_seq_rank).
    tagged = []
    for r in rows:
        tier = _tier_for_row(r)
        tagged.append((_TIER_ORDER.get(tier, 99),
                       _within_sequence_rank(r),
                       tier,
                       r))
    tagged.sort(key=lambda t: (t[0], t[1]))
    best_tier_order, _, best_tier, best_row = tagged[0]
    # If the best tier is "none", all rows are ineligible and there
    # is no representative (this should never happen for a non-empty
    # passing_summary.csv, because any row there is at least a clean
    # steered or pose_holds — both implying tier A/B/C).  Still, be
    # defensive.
    if best_tier == "none":
        return None
    # Return a copy annotated with its tier for downstream use.
    out = dict(best_row)
    out["_picked_tier"] = best_tier
    return out


def _pick_rep_from_aggregated(
    passing_summary_path: Path,
) -> Optional[Dict]:
    """Tier-none fallback: when ``passing_summary.csv`` has no rows
    (every prediction failed the eligibility gate at boltz2_iterate_
    steering.py:_compute_unified_ranks — typically because steered
    ra_eff was way above the 5 Å threshold), the cohort summary plot
    still wants something to show for that sequence.  Read the
    sibling ``aggregated_results.csv``, compute the composite score
    on every row in raw form (ignoring the eligibility gate that
    would otherwise blank rank_by_composite_score), pick the best,
    and feed it through extract_passing.extract_row to produce a
    passing_summary-shaped dict.

    Returns None if aggregated_results.csv is missing or has no rows
    with both composite-score components present.

    The returned dict is annotated with ``_picked_tier="none"`` and
    ``_fallback_composite`` (the raw composite value) so the cross-
    sequence aggregator can stamp ``cross_composite_score`` onto the
    record.  The tier itself stays "none" — these rows are still
    excluded from the cross-rank ranking by the existing tier gate
    in ``aggregate``.

    Stage-aware following the same convention as
    ``_ranking_metric`` / ``_ranking_composite`` in
    boltz2_iterate_steering.py:
      - ``true_jaccard`` always sourced from ``steered_true_jaccard_median``
        (the interface identity is a property of the steered
        prediction, not the reverted pose).
      - ``ra_eff`` sourced from ``reverted_*`` for ``pose_holds``
        rows, ``steered_*`` otherwise.

    The eligibility gate is intentionally skipped here.  The point of
    this fallback is to surface SOMETHING for tier-none rows in the
    cohort summary; a "best of bad attempts" view is more useful than
    grey-hatched cells everywhere, and the tier stripe already warns
    the reader these aren't passing designs.
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

    # Find best by composite (true_jaccard - 0.05 * ra_eff).  Both
    # components must parse as finite floats; rows missing either
    # are skipped.  Stage-aware key selection mirrors extract_passing.
    # extract_row's struct_prefix logic and iterate_steering's
    # _ranking_metric (steered_*_median for non-pose_holds; the
    # _median-suffixed reverted_* for pose_holds).
    best_row = None
    best_composite = None
    for r in rows:
        is_pose_holds = (r.get("outcome") == "pose_holds")
        ra_key = ("reverted_ra_eff_vs_truth_median"
                  if is_pose_holds
                  else "steered_ra_eff_vs_truth_median")
        # true_jaccard always sourced from steered_ — interface identity
        # is a property of the steered (cold-start) prediction, not the
        # reverted pose.  Mirrors _ranking_composite in
        # boltz2_iterate_steering.py:3194-3198.
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

    # Reuse extract_passing.extract_row to translate the aggregated row
    # into a passing_summary-shaped dict.  per_seed_rows=None — the
    # mutation columns will be blank, which is fine for a tier-none
    # fallback (mutation set is informational, not gating).
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
    """Read every passing_summary.csv; return rows / paths / canonical fieldnames."""
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
        # Adopt the first non-empty fieldnames we see as the canonical
        # list for rep_* columns.  Mismatches across files
        # would indicate a pipeline version drift; warn but don't
        # abort.
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
    """Pick representative; on empty-rows fall back to aggregated_results."""
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
    """Read row_type.txt sidecar; default 'steered' when absent or unreadable."""
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
        # Only score sequences that have a placed tier AND both
        # components present.  This keeps the "none"-tier stubs out
        # of the ranking.
        if rec["cross_tier"] in ("A", "B", "C"):
            comp = _composite_score(rep_cols)
            if comp is not None:
                rec["cross_composite_score"] = f"{comp:.6f}"
                scored_composite.append((comp, rec))
            ra = _try_float(rep_cols["ra_eff_vs_truth_median"])
            if ra is not None:
                scored_ra.append((ra, rec))

    # Assign cross-composite ranks: descending score (higher is better)
    # and tier first (A before B before C).  That is, a tier B record
    # with a slightly higher raw composite still ranks below any
    # tier A record — policy from notes 11.
    def _composite_rank_key(item):
        comp, rec = item
        tier_order = _TIER_ORDER[rec["cross_tier"]]
        # Negate composite so higher composites come first within a
        # tier.  id() breaks ties deterministically in the same
        # stable way Python's sort would.
        return (tier_order, -comp, id(rec))
    scored_composite.sort(key=_composite_rank_key)
    for rank_idx, (_, rec) in enumerate(scored_composite, start=1):
        rec["cross_rank_by_composite"] = rank_idx

    # Assign cross-ra-eff ranks: tier first, then ra_eff ascending
    # (lower is better).
    def _ra_rank_key(item):
        ra, rec = item
        tier_order = _TIER_ORDER[rec["cross_tier"]]
        return (tier_order, ra, id(rec))
    scored_ra.sort(key=_ra_rank_key)
    for rank_idx, (_, rec) in enumerate(scored_ra, start=1):
        rec["cross_rank_by_ra_eff"] = rank_idx


def _emit_aggregate_summary(records: List[Dict], output_path: Path) -> None:
    """Print the cross-sequence summary stats to stderr."""
    n_total = len(records)
    n_ranked = sum(1 for r in records
                   if r.get("cross_rank_by_composite") != "")
    by_tier = {"A": 0, "B": 0, "C": 0, "none": 0}
    for r in records:
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

    The join key is f"design_{row['design']}_seq_{row['seq']}", matching
    the mpnn_sequence convention used elsewhere in the pipeline (FASTA
    headers, per-sequence workdir names).  Returns an empty dict if
    `path` is None or the file is missing — the columns then come out
    blank in the output, which is the desired no-op behaviour.
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

    strict: if True, a missing or unreadable passing_summary.csv
            aborts the run; if False (default), emits a diagnostic
            stub row and continues.
    published_runs_dir: when provided, used by
            _rewrite_workdir_path_to_published to stamp stable
            published-tree paths into rep_canonical_pdb
            in place of the work-dir absolute paths
            NEGSTEER_RUN_ONE writes.  See helper docstring.
    scored_metadata_path: when provided, three columns joined onto
            every output row by mpnn_sequence:
            corrected_receptor, designed_residues, native_residues.
            See _load_scored_metadata.
    """
    if not sequences:
        print("ERROR: no inputs supplied", file=sys.stderr)
        return 2

    # First pass: read every file.  Gather per-sequence row lists,
    # tier counts, and (eventually) the representative row.
    try:
        per_seq_rows, source_paths, passing_summary_fieldnames = (
            _read_passing_summary_inputs(sequences, strict)
        )
    except _AggregateInputError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    # Optional MPNN-sequence join (corrected_receptor / designed_residues /
    # native_residues).  Empty dict when --scored-metadata wasn't supplied.
    scored_metadata = _load_scored_metadata(scored_metadata_path)

    # Build per-sequence records.
    records = []
    for seq_name, rows in per_seq_rows.items():
        # Tier-none fallback.  When passing_summary.csv has no rows
        # (typically because every prediction's steered_ra_eff
        # exceeded the 5 Å eligibility gate at boltz2_iterate_steering.
        # py:_compute_unified_ranks), fall back to picking the best
        # row from aggregated_results.csv directly.  See helper
        # docstring for rationale.  This stamps real metric values
        # onto tier-none rows so they show up in the cohort summary
        # plot instead of all grey-hatched cells.
        rep, rep_is_fallback = _pick_or_fallback_representative(
            rows, source_paths[seq_name]
        )
        if rep_is_fallback and rep is not None:
            # Extract_passing.extract_row's output may contain
            # columns the existing passing_summary fieldname union
            # doesn't.  Union them in so the CSV writer doesn't
            # silently drop the values.
            for col in rep.keys():
                if col.startswith("_"):
                    continue
                if col not in passing_summary_fieldnames:
                    passing_summary_fieldnames.append(col)

        # Tier counts inside the sequence — useful context:
        tier_counts = {"A": 0, "B": 0, "C": 0, "none": 0}
        for r in rows:
            tier_counts[_tier_for_row(r)] += 1

        # Cumulative wall-clock for this sequence.  Stamped onto
        # every passing_summary row by extract_passing.py when the
        # run finishes, and also written as a sidecar file in the
        # sequence's workdir so we can surface it for tier-none
        # sequences (which have an empty passing_summary.csv and
        # therefore no representative row to read from).
        #
        # Order of preference:
        #   1. The representative row's run_one_runtime_sec column
        #      (canonical; matches what wet-lab users see in the
        #      per-sequence passing_summary.csv).
        #   2. run_one_runtime_sec.txt in the same directory as
        #      passing_summary.csv (covers tier-none sequences).
        run_one_runtime_sec = _resolve_run_one_runtime_sec(
            rep, source_paths[seq_name]
        )

        # Task 6 (P0-6) row_type.  Default 'steered' when no sidecar
        # is present — preserves backward compatibility with
        # pre-controls workdirs.  The NEGSTEER_CONTROLS process writes
        # 'control_scrambled' or 'control_polyA' next to its
        # passing_summary.csv so this loop can read them off without
        # any column-level changes to extract_passing.py or
        # negative_steering_run_one.sh.
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
            # Top-level runtime column so triage can see sequence-level
            # throughput directly without having to unpack
            # rep_*.  Blank when the sidecar + representative
            # row are both unavailable (e.g. very old workdirs).
            "run_one_runtime_sec": run_one_runtime_sec,
        }
        # Attach the representative row columns with a prefix.
        # canonical_pdb gets path-rewritten from its in-work-dir
        # absolute (NEGSTEER_RUN_ONE writes them) to the published
        # equivalent under <published_runs_dir>/<seq_name>/...
        # See _rewrite_workdir_path_to_published for rationale.
        for col in passing_summary_fieldnames:
            val = rep.get(col, "") if rep else ""
            if col == "canonical_pdb":
                val = _rewrite_workdir_path_to_published(
                    val, seq_name, published_runs_dir,
                )
            rec[f"rep_{col}"] = val

        # Tier-none fallback: stamp the composite score we computed
        # in _pick_rep_from_aggregated directly onto the
        # rec.  The cross-rank pass below skips tier-none rows
        # (correctly — they shouldn't compete with tier A/B/C in
        # ranking), but the cohort summary plot reads
        # cross_composite_score as its first column and we want
        # tier-none rows to show their score.  cross_rank_by_composite
        # and cross_rank_by_ra_eff stay blank; only the score is
        # populated.
        if rep_is_fallback and rep is not None:
            comp = rep.get("_fallback_composite")
            if comp is not None:
                rec["cross_composite_score"] = f"{comp:.6f}"

        # MPNN-sequence join.  When --scored-metadata is supplied,
        # surface the receptor sequence the MPNN designed for this
        # mpnn_sequence (and the native residues it replaced at the
        # design positions) directly in cross_sequence_summary.csv so
        # downstream consumers don't need to re-parse mpnn_corrected.fasta.
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

    # Score + cross-rank.  Score comes from the representative row's
    # median columns; sequences without a representative get None.
    # Control rows (row_type != 'steered') are deliberately excluded
    # from the ranking — they are diagnostic output, not promotable
    # designs.  They still appear in the output with all cross_rank_*
    # columns blank.  The onComplete miscalibration check reads them
    # back from the final CSV.
    _assign_cross_ranks(records)

    # Final ordering for output: by cross_rank_by_composite ascending,
    # then by cross_tier, then by mpnn_sequence name for determinism.
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
    # Column order: metadata first, then rep_* columns in
    # the order they appeared in passing_summary.csv.
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
        # MPNN-sequence join (blank when --scored-metadata not supplied).
        # corrected_receptor: full receptor sequence (native at fixed +
        #                     MPNN at design positions)
        # designed_residues:  MPNN AA slice at design positions only
        #                     (pipe-joined per de novo segment)
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

    # Defensive: resolve --published-runs-dir to absolute.  A relative
    # path here would be stamped verbatim into rep_canonical_pdb,
    # then resolve against the wrong CWD in downstream container tasks
    # (each runs in its own work dir, not the launch dir).  main.nf
    # also normalises params.outdir to absolute, so this is belt-and-
    # braces — but the script can be invoked standalone or by other
    # workflows, and the cost of an unconditional .resolve() is zero.
    if args.published_runs_dir is not None:
        args.published_runs_dir = args.published_runs_dir.resolve()

    # Build the (seq_name, path) list.
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
        # Enumerate sequence subdirectories, each expected to contain
        # a passing_summary.csv.  Missing files are handled downstream.
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
                # Already added via --passing-summary; skip.
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