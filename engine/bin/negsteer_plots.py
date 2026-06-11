#!/usr/bin/env python3
"""
negsteer_plots.py
-----------------
Production cohort plot script for negative steering (Task 43).

Reads cross_sequence_summary.csv produced by NEGSTEER_CROSS_SEQUENCE
and emits the diagnostic plot suite to --outdir.  Invoked from
modules/negative_steering.nf::NEGSTEER_PLOTS.

Lifted verbatim from tests/negative_steering/test_negsteer_plots.py;
the only differences are (a) the test-path fallback in
_resolve_csv_path is removed (production always passes --csv
explicitly) and (b) this header.  The two scripts must stay in sync;
when iterating on plots, edit the test script first, validate against
real cluster data, then mirror the changes here.

Plots produced
--------------

negsteer_tier_landscape.png
    Per-sequence bar of cross_composite_score, sorted by
    cross_rank_by_composite, coloured by tier.  The primary ranker
    visualisation: shows the cohort's composite-score distribution
    and the gap between tier A and the tail.

negsteer_seed_outcomes_heatmap.png
    Per-MPNN-sequence dot grid (representative sg only, 3 seeds per
    row).  Sorted by composite score descending.  Shape encodes which
    stage produced the seed's final prediction (cold-start /
    steering / reversion); colour encodes how that prediction fared
    (pass / off_target / poor_prediction / multiple_failures /
    new_contamination / no_data).  Replaces the old seed_verdicts
    plot — separates "what stage" from "what outcome" cleanly.

negsteer_seed_outcomes_bars.png
    Two side-by-side stacked bars per sequence using ALL sequence
    groups (typically 9 seeds per sequence: 3 sg × 3 Boltz seeds).
    Left panel: stage breakdown.  Right panel: outcome breakdown.
    Shows the marginal distributions across the cohort.

negsteer_ra_eff_vs_jaccard.png
    Scatter of rep_ra_eff_vs_truth_median (x) vs
    rep_true_jaccard_median (y), coloured by tier, with
    iso-composite contours overlaid.  Controls drawn as different
    markers.  The composite score made geometric.

negsteer_filter_cascade.png
    Waterfall: how many sequences pass each Boltz confidence filter
    in succession (complex_plddt → ipae → pae_pass_frac → tiered).
    Frames Boltz confidence metrics as filters (their actual job
    per the metric reliability memo) rather than as rankers.

negsteer_controls_diagnostic.png
    Two-panel boxplot, ra_eff (top) and true_jaccard (bottom),
    grouped by row_type (steered / control_scrambled / control_polyA).
    Reference lines at controls_warning_ra_eff_min=5.0.  If the
    control distributions overlap the steered distribution, the
    pipeline is mis-calibrated.

negsteer_mutation_impact.png
    Per-input-PDB-position summary of how steering mutations
    correlate with composite score.  For each position p that
    appears in any sequence's rep_steered_mutations_chimerax,
    plot the median composite score of sequences mutating p vs the
    cohort median.  Bars sized by number of sequences mutating p.
    Faint background shading marks design-region positions for
    context (mutations should never fall there by construction).
    The "which mutations make negative steering work" plot.

Usage (production, called from NEGSTEER_PLOTS process):
    python negsteer_plots.py \\
        --csv cross_sequence_summary.csv \\
        --runs-dir runs \\
        --input-design-region input_design_region.txt \\
        --outdir plots
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


# ── Threshold defaults ────────────────────────────────────────────────────
# Default values mirror extract_passing._compute_confidence_flag.
# Override at the CLI to reflect the values pinned in your run.
COMPOSITE_RA_EFF_WEIGHT = 0.05      # composite = true_jaccard − 0.05·ra_eff
CONTROL_RA_EFF_MIN     = 5.0        # controls_warning_ra_eff_min
CONFIDENCE_PLDDT_MIN   = 0.70       # complex_plddt_min in extract_passing
CONFIDENCE_IPAE_MAX    = 15.0       # ipae_max in extract_passing
CONFIDENCE_PAEPF_MIN   = 0.10       # pass_frac_min in extract_passing
CONFIDENCE_IPTM_MIN    = 0.30       # iptm_min in extract_passing

# Minimum number of sequences sharing a mutated position before the
# position is plotted with a confidence-bearing colour in the
# mutation-impact plot.  Below this, the bar is rendered faint to
# convey the lower n.
MUT_IMPACT_MIN_N = 3


# ── Styling ──────────────────────────────────────────────────────────────
COLOUR_TIER = {
    "A":    "#2CA02C",   # green
    "B":    "#FFB000",   # amber
    "C":    "#FF7F0E",   # orange
    "none": "#B0B0B0",   # grey
}
COLOUR_STEERED       = "#4C72B0"   # blue (steered cohort)
COLOUR_CTRL_SCRAM    = "#8C564B"   # brown (scrambled control)
COLOUR_CTRL_POLYA    = "#9467BD"   # purple (polyA control)
COLOUR_THRESHOLD     = "#D62728"   # red (threshold reference lines)
COLOUR_CONTOUR       = "#888888"   # grey (iso-composite contours)
COLOUR_DESIGN_REGION = "#FFE9B0"   # very pale yellow (design-region shading)
COLOUR_BETTER_THAN   = "#2CA02C"   # green (mutation associated with above-cohort composite)
COLOUR_WORSE_THAN    = "#D62728"   # red (mutation associated with below-cohort composite)


# ═══════════════════════════════════════════════════════════════════════════
# Data loaders — cohort merge from cross_summary + per-sequence aggregated_results
# ═══════════════════════════════════════════════════════════════════════════

# Per-sequence "final" prediction.  Each MPNN sequence has multiple
# possible prediction stages depending on what the negative-steering
# pipeline did with it:
#
#   - cold start only (status=initial, never steered):
#         final = cold-start prediction (stored as steered_* by the
#         pipeline; no reverted_* exists)
#   - steered, verdict=no_reversion (steered prediction was clean of
#     contamination on mutated positions, so reversion never ran):
#         final = steered prediction (steered_*; no reverted_* exists)
#   - steered, verdict=pose_holds | pose_collapses | new_contamination
#     (reversion ran):
#         final = reverted prediction (reverted_*) — this is the
#         actual last prediction made for the sequence; the steered
#         metrics describe an intermediate stage
#
# Plots should read final_*.  Reading steered_* unconditionally
# (the previous behaviour) silently glossed over reversion failures
# and made e.g. d0_s3 look great (steered ra=3.013, jaccard=0.7)
# when its reverted prediction collapsed to ra=30.5 — exactly the
# scenario negative steering exists to detect.

# Mapping: how each rep_* field is derived from agg row.
# Each entry maps the destination field to a tuple of:
#   (steered_column_name, reverted_column_name_or_None)
# When reverted_col is None, the steered column is always used (e.g.
# verdict-bucket counts that don't have a reverted equivalent).
_AGG_FINAL_FIELD_MAP = {
    # n_seeds, n_pass and outcome — single source of truth (no reverted).
    "rep_n_seeds":                       ("n_seeds", None),
    "rep_n_pass":                        ("n_pass", None),
    "rep_outcome":                       ("outcome", None),
    # Steered/reverted metric pairs — picked per row by verdict.
    "rep_ra_eff_vs_truth_median": (
        "steered_ra_eff_vs_truth_median", "reverted_ra_eff_vs_truth_median"),
    "rep_true_jaccard_median": (
        "steered_true_jaccard_median",    "reverted_true_jaccard_median"),
    "rep_complex_plddt_median": (
        "steered_complex_plddt_median",   "reverted_complex_plddt_median"),
    "rep_ipae_median": (
        "steered_ipae_median",            "reverted_ipae_median"),
    "rep_pae_pass_frac_median": (
        "steered_pae_pass_frac_median",   "reverted_pae_pass_frac_median"),
    "rep_iptm_median": (
        "steered_iptm_median",            "reverted_iptm_median"),
    "rep_interface_plddt_median": (
        "steered_interface_plddt_median", "reverted_interface_plddt_median"),
    "rep_weighted_jaccard_median": (
        "steered_weighted_jaccard_median","reverted_weighted_jaccard_median"),
    # Receptor-intact majority flag (used by the cascade's structural
    # filter step).  Verdict-aware: clean_steered uses steered side,
    # pose_holds/collapses/contamination use reverted side.
    "rep_receptor_intact_majority": (
        "steered_receptor_intact_majority",
        "reverted_receptor_intact_majority"),
    # Mutations: cross_summary has BOTH rep_steered_*
    # and rep_reverted_mutations_majority_* populated.
    # The aggregated_results.csv however only has the unpopulated
    # raw columns — they get filled in later by extract_passing when
    # building cross_summary.  So this mapping projects steered into
    # the steered slot only; the verdict-aware choice between
    # steered/reverted final mutations happens at the consumer end
    # (plot_mutation_impact reads the right cross_summary column
    # directly based on rep_outcome).
    "rep_steered_mutations_chimerax":    ("steered_mutations_chimerax", None),
    "rep_steered_mutations_aa":          ("steered_mutations_aa", None),
    "rep_total_mutations_median":        ("steered_total_mutations_median", None),
}

# Verdicts that mean reversion ran → final = reverted prediction.
_VERDICTS_WITH_REVERSION = {"pose_holds", "pose_collapses", "new_contamination"}


def _final_prediction_value(agg_row: Dict, dest_field: str) -> str:
    """Return the value of `dest_field` from `agg_row`, picking
    the steered or reverted column based on the row's
    outcome.  See _AGG_FINAL_FIELD_MAP and module docstring
    for the policy."""
    spec = _AGG_FINAL_FIELD_MAP.get(dest_field)
    if spec is None:
        return ""
    steered_col, reverted_col = spec
    if reverted_col is None:
        return agg_row.get(steered_col, "") or ""
    verdict = (agg_row.get("outcome") or "").strip()
    if verdict in _VERDICTS_WITH_REVERSION:
        # Prefer reverted; fall back to steered if reverted is blank
        # (e.g. reversion failed to produce a prediction at all).
        v = agg_row.get(reverted_col, "")
        if v not in ("", None):
            return v
        return agg_row.get(steered_col, "") or ""
    # no_reversion / singleton / blank — use steered (cold start or
    # clean-steered prediction; the only prediction that exists).
    return agg_row.get(steered_col, "") or ""


def _project_agg_row(seq_name: str, agg_row: Dict, row_type: str = "steered") -> Dict:
    """Convert an aggregated_results.csv row into a cross_summary-shaped
    dict.  Only minimal columns needed by the plots are populated.
    Tier defaults to 'none' (cross_tier is computed by
    cross_sequence_summary.py post-passing-filter, so a non-passing
    sequence gets no tier)."""
    out = {
        "mpnn_sequence": seq_name,
        "row_type":      row_type,
        "cross_tier":    "none",                # not in cohort cross-rank
        "cross_rank_by_composite": "",
        "cross_composite_score":   "",
    }
    for dst in _AGG_FINAL_FIELD_MAP:
        out[dst] = _final_prediction_value(agg_row, dst)
    return out


def _pick_best_agg_row(agg_rows: List[Dict],
                        prefer_cycle: Optional[str] = None,
                        prefer_pathway: Optional[str] = None,
                        prefer_sequence_group: Optional[str] = None,
                        ) -> Optional[Dict]:
    """A sequence's aggregated_results.csv may have multiple rows
    (multiple cycles, pathways, sequence_groups, plus singleton).

    When cross_summary points at a representative row (tiered
    sequence), return that exact row — it's the row the cross-rank
    was computed on.

    For tier-none sequences (no representative pointer): pick the row
    with the best composite score (true_jaccard − 0.05·ra_eff), so
    the cohort plots show the sequence's "best attempt" rather than
    its worst.  This is the right choice because tier-none means no
    row crossed extract_passing's filters, but the per-sequence best
    aggregate is still the most informative summary for plots.
    Singletons skipped (no real seed counts)."""
    if not agg_rows:
        return None

    # Exact-match path for tiered sequences.
    if prefer_sequence_group is not None and prefer_sequence_group != "":
        for r in agg_rows:
            if (str(r.get("cycle", "")) == str(prefer_cycle or "")
                and str(r.get("pathway", "")) == str(prefer_pathway or "")
                and str(r.get("sequence_group", "")) == str(prefer_sequence_group)):
                return r

    # Skip singleton fallback rows (no real seed counts).
    candidates = [r for r in agg_rows
                  if (r.get("outcome") or "").strip()
                     not in ("singleton",)]
    if not candidates:
        candidates = agg_rows

    def _composite(r: Dict) -> float:
        ra = _try_float(r.get("steered_ra_eff_vs_truth_median"))
        tj = _try_float(r.get("steered_true_jaccard_median"))
        if ra is None or tj is None:
            # Penalise rows with missing numbers — they would
            # otherwise outscore real rows.
            return -1e9
        return tj - 0.05 * ra

    # Pick by best composite (highest = best attempt).
    return sorted(candidates, key=lambda r: -_composite(r))[0]


def load_unified_cohort(
    cross_csv: Path,
    runs_dir: Optional[Path],
    verbose: bool = True,
) -> List[Dict]:
    """Build the cohort by merging cross_sequence_summary.csv (which
    has tier / composite-rank info plus tier-none stub rows for every
    sequence) with per-sequence aggregated_results.csv files (which
    have the seed-count / ra_eff / jaccard data even for tier-none
    sequences).

    Returns a unified list of dicts in the cross_summary vocabulary.
    Sequences in cross_summary keep their identification + tier info;
    if their tier is 'none' (i.e. no passing rows, so the
    rep_* columns are blank), the seed counts / ra_eff /
    jaccard / mutation columns are backfilled from
    aggregated_results.csv when available.

    Sequences in runs/ but missing entirely from cross_summary (older
    workdirs / partial runs) get a synthetic entry built from their
    best aggregated_results row.

    Controls (control_polyA, control_scrambled) are passed through
    from cross_summary unchanged; cross_summary handles them.
    """
    # Step 1: read cross_summary into a dict keyed by mpnn_sequence.
    cross_rows: Dict[str, Dict] = {}
    with open(cross_csv) as f:
        for row in csv.DictReader(f):
            name = (row.get("mpnn_sequence") or "").strip()
            if name:
                cross_rows[name] = row

    if verbose:
        print(f"  cross_summary contains {len(cross_rows)} rows")
        # Show tier breakdown for visibility — tier-none rows often have
        # blank rep_* and need backfill from aggregated_results.
        from collections import Counter
        tier_counter = Counter(
            (r.get("cross_tier") or "none").strip() for r in cross_rows.values()
        )
        print(f"  cross_summary tier breakdown: "
              f"A={tier_counter.get('A',0)} B={tier_counter.get('B',0)} "
              f"C={tier_counter.get('C',0)} none={tier_counter.get('none',0)}")
        # Count rows missing n_pass (the tell that backfill is needed).
        n_blank_seedcounts = sum(
            1 for r in cross_rows.values()
            if not (r.get("rep_n_pass") or "").strip()
        )
        print(f"  cross_summary rows with blank "
              f"rep_n_pass: {n_blank_seedcounts}")

    if runs_dir is None or not runs_dir.exists():
        if verbose:
            print("  no --runs-dir → backfill skipped; tier-none "
                  "sequences with blank seed counts will be filtered "
                  "out of plots that read those columns.")
        return list(cross_rows.values())

    # Step 2: enumerate sequence dirs under runs/, read aggregated_results.csv.
    augmented: List[Dict] = []
    n_dirs_seen = 0
    n_in_cross = 0
    n_backfilled = 0
    n_synthesised = 0
    for entry in sorted(runs_dir.iterdir()):
        if not entry.is_dir():
            continue
        seq_name = entry.name
        n_dirs_seen += 1
        agg_path = entry / "aggregated_results.csv"
        if not agg_path.exists():
            if verbose:
                print(f"  WARN: no aggregated_results.csv in {entry}")
            continue
        with open(agg_path) as f:
            agg_rows = list(csv.DictReader(f))

        # When cross_summary already names a representative row, use
        # its (cycle, pathway, sequence_group) to find the matching
        # aggregated_results row exactly.  When tier is "none", tell
        # the picker to skip no_reversion rows so seed_verdicts gets
        # real counts instead of a synthesised all-clean stack.
        cross = cross_rows.get(seq_name)
        if cross is not None:
            best_agg = _pick_best_agg_row(
                agg_rows,
                prefer_cycle=cross.get("rep_cycle"),
                prefer_pathway=cross.get("rep_pathway"),
                prefer_sequence_group=cross.get("rep_sequence_group"),
            )
        else:
            best_agg = _pick_best_agg_row(agg_rows)

        if seq_name in cross_rows:
            n_in_cross += 1
            row = dict(cross_rows[seq_name])
            if best_agg is not None:
                # Always OVERWRITE metric fields with verdict-aware
                # final-prediction values.  cross_summary's
                # rep_*_median values are computed from the
                # steered side regardless of whether reversion ran;
                # re-sourcing them here ensures pose_collapses /
                # new_contamination sequences carry their reverted
                # metrics (the actual final prediction) rather than
                # the misleading steered intermediate.  Verdict counts
                # and mutation strings (no reverted equivalent) are
                # only filled when blank, preserving cross_summary's
                # values when present.
                for dst, (steered_col, reverted_col) in _AGG_FINAL_FIELD_MAP.items():
                    final_val = _final_prediction_value(best_agg, dst)
                    if reverted_col is not None:
                        # Steered/reverted pair → always take final.
                        row[dst] = final_val
                    else:
                        # Verdict count / mutation string → only fill blanks.
                        if not row.get(dst):
                            row[dst] = final_val
                n_backfilled += 1
            augmented.append(row)
            del cross_rows[seq_name]
        else:
            if best_agg is None:
                continue
            n_synthesised += 1
            row_type = "steered"
            if seq_name.startswith("input_control_polyA"):
                row_type = "control_polyA"
            elif seq_name.startswith("input_control_scrambled"):
                row_type = "control_scrambled"
            augmented.append(_project_agg_row(seq_name, best_agg, row_type))

    if verbose:
        print(f"  runs/ contained {n_dirs_seen} sequence dirs")
        print(f"    in cross_summary: {n_in_cross} "
              f"(of which backfilled from agg: {n_backfilled})")
        print(f"    not in cross_summary (synthesised): {n_synthesised}")

    # Step 3: any cross_summary rows we didn't see in runs_dir get
    # appended verbatim.
    leftover_count = len(cross_rows)
    for leftover in cross_rows.values():
        augmented.append(leftover)
    if verbose and leftover_count:
        print(f"  cross_summary rows with no matching runs/ dir "
              f"(passed through unchanged): {leftover_count}")
        for name in list(cross_rows.keys())[:5]:
            print(f"      e.g.: {name}")

    return augmented


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _try_float(v) -> Optional[float]:
    """Parse to float, returning None on failure / NaN / inf / empty."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        f = float(v)
    else:
        s = str(v).strip()
        if not s:
            return None
        try:
            f = float(s)
        except ValueError:
            return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _try_int(v) -> Optional[int]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except ValueError:
            return None


def _parse_chimerax_positions(s: str) -> List[int]:
    """Parse '/A:5,7,22' → [5, 7, 22].  Returns [] for blank/malformed."""
    if not s:
        return []
    s = s.strip()
    if ":" not in s:
        return []
    _, _, tail = s.partition(":")
    out = []
    for tok in tail.split(","):
        tok = tok.strip()
        if tok.lstrip("-").isdigit():
            out.append(int(tok))
    return out


def _composite_from_row(row: Dict) -> Optional[float]:
    """Recompute composite from rep_* medians.
    composite = true_jaccard − 0.05·ra_eff.  Returns None if either
    component is missing."""
    ra = _try_float(row.get("rep_ra_eff_vs_truth_median"))
    tj = _try_float(row.get("rep_true_jaccard_median"))
    if ra is None or tj is None:
        return None
    return tj - COMPOSITE_RA_EFF_WEIGHT * ra


def _row_type(row: Dict) -> str:
    return (row.get("row_type") or "steered").strip() or "steered"


def _is_steered(row: Dict) -> bool:
    return _row_type(row) == "steered"


def _make_empty_plot(message: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center",
            transform=ax.transAxes, fontsize=12, color="grey")
    ax.set_axis_off()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _short_name(name: str) -> str:
    """Strip the 'design_' prefix for compact x-axis labels.
    'design_0_seq_1' → 'd0_s1', 'input_control_polyA' → 'ctrl_pA'."""
    n = name
    if n.startswith("design_"):
        n = n[len("design_"):]
        n = n.replace("_seq_", "_s")
        return f"d{n}"
    if n.startswith("input_control_"):
        tail = n[len("input_control_"):]
        return f"ctrl_{tail[:4]}"
    return n


# ═══════════════════════════════════════════════════════════════════════════
# Plot 1: Tier + composite-score landscape
# ═══════════════════════════════════════════════════════════════════════════

def plot_tier_landscape(rows: List[Dict], out_path: str) -> bool:
    """Per-sequence bar of cross_composite_score, sorted by
    cross_rank_by_composite (tier A first, then B, then C, then none),
    coloured by tier.  Controls excluded — they have blank cross_rank
    by design (excluded from cross-ranking; shown separately in the
    controls_diagnostic plot)."""
    items = []
    for row in rows:
        if not _is_steered(row):
            continue
        rk = _try_int(row.get("cross_rank_by_composite"))
        cs = _try_float(row.get("cross_composite_score"))
        # Tier-none sequences have blank cross_rank_by_composite by
        # construction.  Recompute composite from the representative
        # row if possible so they still appear in the bar chart.
        if cs is None:
            cs = _composite_from_row(row)
        tier = (row.get("cross_tier") or "none").strip()
        items.append({
            "name": row.get("mpnn_sequence", "?"),
            "rank": rk if rk is not None else 10**6,
            "comp": cs,
            "tier": tier,
        })

    if not items:
        _make_empty_plot("No steered rows in cross-sequence summary", out_path)
        return False

    # Tier-then-composite ordering (tier-none at the tail, sorted by
    # composite descending within the none tail).
    items.sort(key=lambda d: (
        d["rank"],
        -(d["comp"] if d["comp"] is not None else -1e6),
    ))

    n = len(items)
    # Min width 9 — the lifted legend has 5 entries and needs the
    # full plot width to lay out in a single row without squashing.
    fig_w = max(9, min(22, n * 0.50 + 4))
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    xs = np.arange(n)
    ys = [d["comp"] if d["comp"] is not None else 0.0 for d in items]
    colors = [COLOUR_TIER.get(d["tier"], "#B0B0B0") for d in items]

    ax.bar(xs, ys, color=colors, edgecolor="black", linewidth=0.4)

    # Mark sequences whose composite couldn't be computed (e.g. blank
    # true_jaccard) with a small "×" at y=0 — without this they're
    # indistinguishable from a real zero-composite sequence.
    for i, d in enumerate(items):
        if d["comp"] is None:
            ax.scatter(i, 0, marker="x", s=40,
                       color="black", linewidth=1.5, zorder=4)

    # Vertical dotted separators between tier boundaries.
    tier_starts: Dict[str, int] = {}
    for i, d in enumerate(items):
        tier_starts.setdefault(d["tier"], i)
    for tier in ("B", "C", "none"):
        if tier in tier_starts and tier_starts[tier] > 0:
            ax.axvline(tier_starts[tier] - 0.5, color="black",
                       linewidth=0.6, linestyle=":", alpha=0.5)

    # Reference at composite = 0 — no signal.
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)

    ax.set_xticks(xs)
    ax.set_xticklabels(
        [_short_name(d["name"]) for d in items],
        rotation=90, fontsize=max(6, 9 - n // 25),
    )
    ax.set_xlabel("MPNN sequence", fontsize=10)
    ax.set_ylabel("Composite score\n"
                  "(Interface Jaccard − 0.05 × Receptor Aligned Effector RMSD)",
                  fontsize=10)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.3, alpha=0.4, color="grey")
    ax.set_axisbelow(True)

    # Legend with per-tier counts.  Lifted above the panel so it
    # never overlaps the top bar (which is always the tallest in
    # composite-sorted order).
    tier_counts = defaultdict(int)
    for d in items:
        tier_counts[d["tier"]] += 1
    legend_handles = [
        mpatches.Patch(color=COLOUR_TIER["A"],
                       label=f"Tier A (n={tier_counts['A']})"),
        mpatches.Patch(color=COLOUR_TIER["B"],
                       label=f"Tier B (n={tier_counts['B']})"),
        mpatches.Patch(color=COLOUR_TIER["C"],
                       label=f"Tier C (n={tier_counts['C']})"),
        mpatches.Patch(color=COLOUR_TIER["none"],
                       label=f"None (n={tier_counts['none']})"),
        plt.Line2D([0], [0], marker="x", color="black", linestyle="",
                   markersize=7, markeredgewidth=1.5,
                   label="composite n/a"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              bbox_to_anchor=(1.0, 1.02),
              fontsize=9, framealpha=0.95, ncol=len(legend_handles))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Plot 2: Per-seed outcomes — heatmap + side-by-side bars
# ═══════════════════════════════════════════════════════════════════════════
#
# Two visualisations of per-seed outcomes that replace the old
# negsteer_seed_verdicts plot.  Both classify each seed into:
#   - stage:    cold_start | steering | reversion
#   - outcome:  pass | off_target | poor_prediction | multiple_failures
#               | new_contamination | no_data
# Stage tells you which prediction was the FINAL one for that seed
# (i.e. which one cross_summary's rep_* fields would have
# pulled from).  Outcome combines structural (ra_eff, intact) and
# confidence (plddt, ipae, paepf, iptm) checks.
#
# Heatmap (negsteer_seed_outcomes_heatmap.png):
#   one row per MPNN sequence, sorted by composite (best at top), 3
#   cells per row (representative sequence_group only).  Shape =
#   stage; colour = outcome.  Lets the reader see at a glance which
#   sequences passed and at which stage, plus the variability across
#   seeds within the representative attempt.
#
# Bars (negsteer_seed_outcomes_bars.png):
#   two side-by-side stacked-bar panels — one per stage, one per
#   outcome — using ALL seeds across ALL sequence_groups (typically
#   9 seeds per sequence: 3 sg × 3 Boltz seeds).  Shows the marginal
#   distributions across the cohort.

# ── Classifier thresholds (matching extract_passing.py) ─────────────────
SEED_RA_EFF_MAX     = 5.0
SEED_PLDDT_MIN      = 0.70
SEED_IPAE_MAX       = 15.0
SEED_PAE_PASS_MIN   = 0.10
SEED_IPTM_MIN       = 0.30

# ── Outcome / stage palettes ────────────────────────────────────────────
OUTCOME_COLOUR = {
    "pass":               "#2CA02C",
    "off_target":         "#FF7F0E",
    "poor_prediction":    "#FFD700",
    "multiple_failures":  "#D62728",
    "new_contamination":  "#8B0000",
    "no_data":            "#B0B0B0",
}
OUTCOME_LABEL = {
    "pass":              "Pass",
    "off_target":        "Off-target",
    "poor_prediction":   "Poor prediction",
    "multiple_failures": "Multiple failures",
    "new_contamination": "New contamination",
    "no_data":           "No data",
}
OUTCOME_ORDER = [
    "pass", "off_target", "poor_prediction",
    "multiple_failures", "new_contamination", "no_data",
]

STAGE_MARKER = {"cold_start": "o", "steering": "s", "reversion": "^"}
STAGE_LABEL  = {"cold_start": "Cold-start",
                "steering":   "Steering",
                "reversion":  "Reversion"}
STAGE_COLOUR = {"cold_start": "#1F77B4",
                "steering":   "#9467BD",
                "reversion":  "#17BECF"}
STAGE_ORDER  = ["cold_start", "steering", "reversion"]


def classify_seed(row: Dict) -> Tuple[str, str]:
    """Return (stage, outcome) for one raw_per_seed_results.csv row.

    Stage:
      - reversion: reversion_verdict is populated (any value)
      - cold_start: steered_total_mutations==0 AND no reversion ran
      - steering:   steered_total_mutations>0  AND no reversion ran

    Outcome (based on the row's stage-appropriate metric columns):
      - new_contamination: pipeline labelled it directly (only valid
        for reversion stage)
      - pass: ra_eff < 5 AND intact AND all 4 confidence metrics pass
      - off_target: ra_eff fails but confidence passes
      - poor_prediction: confidence fails but ra_eff passes
      - multiple_failures: both ra_eff and confidence fail
      - no_data: required fields missing/unparseable
    """
    reversion_verdict = (row.get("reversion_verdict") or "").strip()
    if reversion_verdict:
        stage = "reversion"
        if reversion_verdict == "new_contamination":
            return stage, "new_contamination"
        ra      = _try_float(row.get("reverted_ra_eff_vs_truth"))
        intact  = _try_int(row.get("reverted_receptor_intact"))
        plddt   = _try_float(row.get("reverted_complex_plddt"))
        ipae    = _try_float(row.get("reverted_ipae"))
        paepf   = _try_float(row.get("reverted_pae_pass_frac"))
        iptm    = _try_float(row.get("reverted_iptm"))
    else:
        n_mut = _try_int(row.get("steered_total_mutations"))
        if n_mut is None:
            return "no_data", "no_data"
        stage = "cold_start" if n_mut == 0 else "steering"
        ra      = _try_float(row.get("steered_ra_eff_vs_truth"))
        intact  = _try_int(row.get("steered_receptor_intact"))
        plddt   = _try_float(row.get("steered_complex_plddt"))
        ipae    = _try_float(row.get("steered_ipae"))
        paepf   = _try_float(row.get("steered_pae_pass_frac"))
        iptm    = _try_float(row.get("steered_iptm"))

    if any(x is None for x in (ra, intact, plddt, ipae, paepf, iptm)):
        return stage, "no_data"

    structural_pass = (intact == 1) and (ra < SEED_RA_EFF_MAX)
    confidence_pass = (
        plddt >= SEED_PLDDT_MIN
        and ipae <= SEED_IPAE_MAX
        and paepf >= SEED_PAE_PASS_MIN
        and iptm >= SEED_IPTM_MIN
    )
    if structural_pass and confidence_pass:
        return stage, "pass"
    if not structural_pass and not confidence_pass:
        return stage, "multiple_failures"
    if not structural_pass and confidence_pass:
        return stage, "off_target"
    return stage, "poor_prediction"


def _rep_sg_for_outcomes(
    seq_dir: Path,
    cs_lookup: Dict[str, Dict],
) -> Optional[str]:
    """Pick the representative sequence_group for one MPNN sequence.

    For tier A/B/C sequences cross_summary already names a representative
    (rep_sequence_group); use that.  For tier-none sequences
    cross_summary leaves it blank — pick the aggregated_results row
    with the best composite (true_jaccard − 0.05·ra_eff) among non-
    singleton rows."""
    seq_name = seq_dir.name
    cs_row = cs_lookup.get(seq_name)
    if cs_row:
        rep = (cs_row.get("rep_sequence_group") or "").strip()
        if rep:
            return rep
    agg_path = seq_dir / "aggregated_results.csv"
    if not agg_path.exists():
        return None
    with open(agg_path) as f:
        agg_rows = list(csv.DictReader(f))
    candidates = [
        r for r in agg_rows
        if (r.get("outcome") or "").strip() != "singleton"
    ] or agg_rows

    def _comp(r: Dict) -> float:
        verdict = (r.get("outcome") or "").strip()
        if verdict in _VERDICTS_WITH_REVERSION:
            ra_v = (r.get("reverted_ra_eff_vs_truth_median")
                    or r.get("steered_ra_eff_vs_truth_median"))
            tj_v = (r.get("reverted_true_jaccard_median")
                    or r.get("steered_true_jaccard_median"))
        else:
            ra_v = r.get("steered_ra_eff_vs_truth_median")
            tj_v = r.get("steered_true_jaccard_median")
        ra = _try_float(ra_v)
        tj = _try_float(tj_v)
        if ra is None or tj is None:
            return -1e9
        return tj - 0.05 * ra

    if not candidates:
        return None
    best = sorted(candidates, key=lambda r: -_comp(r))[0]
    return (best.get("sequence_group") or "").strip()


def load_seed_outcomes(
    runs_dir: Path,
    cross_summary_path: Path,
    all_sgs: bool = False,
) -> Tuple[Dict[str, List[Tuple[str, str, int]]], Dict[str, float]]:
    """Build per-sequence (stage, outcome, seed_index) lists from
    raw_per_seed_results.csv.  Returns (seq_outcomes, composites).

    When all_sgs=False (default), filter to the representative
    sequence_group only (3 seeds per MPNN sequence).  When True,
    include all sequence_groups (typically 9 seeds per sequence)."""
    cs_lookup: Dict[str, Dict] = {}
    with open(cross_summary_path) as f:
        for r in csv.DictReader(f):
            cs_lookup[r["mpnn_sequence"]] = r

    out: Dict[str, List[Tuple[str, str, int]]] = {}
    composites: Dict[str, float] = {}
    for seq_dir_name in sorted(os.listdir(runs_dir)):
        if seq_dir_name.startswith("input_control"):
            continue
        seq_dir = runs_dir / seq_dir_name
        rep_sg = None
        if not all_sgs:
            rep_sg = _rep_sg_for_outcomes(seq_dir, cs_lookup)
            if rep_sg is None:
                continue
        raw = seq_dir / "raw_per_seed_results.csv"
        if not raw.exists():
            continue
        seeds: List[Tuple[str, str, int]] = []
        with open(raw) as f:
            for row in csv.DictReader(f):
                sg = (row.get("sequence_group") or "").strip()
                if sg == "":
                    continue   # skip cycle-0 baseline
                if rep_sg is not None and sg != rep_sg:
                    continue
                stage, outcome = classify_seed(row)
                try:
                    seed_idx = int(row.get("seed_index", "0"))
                except (TypeError, ValueError):
                    seed_idx = 0
                seeds.append((stage, outcome, seed_idx))
        if not seeds:
            continue
        out[seq_dir_name] = seeds
        # Composite for ranking — uses cross_summary's verdict-aware
        # rep_*_median when available, else falls back to
        # the best aggregate row's composite.
        cs = cs_lookup.get(seq_dir_name)
        comp_value = float("-inf")
        if cs is not None:
            ra = _try_float(cs.get("rep_ra_eff_vs_truth_median"))
            tj = _try_float(cs.get("rep_true_jaccard_median"))
            if ra is not None and tj is not None:
                comp_value = tj - 0.05 * ra
        if comp_value == float("-inf"):
            agg_path = seq_dir / "aggregated_results.csv"
            if agg_path.exists():
                with open(agg_path) as f:
                    agg_rows_local = list(csv.DictReader(f))
                cands = [
                    r for r in agg_rows_local
                    if (r.get("outcome") or "").strip()
                       != "singleton"
                ] or agg_rows_local
                best_comp = float("-inf")
                for r in cands:
                    verdict = (r.get("outcome") or "").strip()
                    if verdict in _VERDICTS_WITH_REVERSION:
                        ra_v = (r.get("reverted_ra_eff_vs_truth_median")
                                or r.get("steered_ra_eff_vs_truth_median"))
                        tj_v = (r.get("reverted_true_jaccard_median")
                                or r.get("steered_true_jaccard_median"))
                    else:
                        ra_v = r.get("steered_ra_eff_vs_truth_median")
                        tj_v = r.get("steered_true_jaccard_median")
                    ra = _try_float(ra_v)
                    tj = _try_float(tj_v)
                    if ra is not None and tj is not None:
                        c = tj - 0.05 * ra
                        if c > best_comp:
                            best_comp = c
                if best_comp != float("-inf"):
                    comp_value = best_comp
        composites[seq_dir_name] = comp_value
    return out, composites


def plot_seed_outcomes_heatmap(
    seq_outcomes: Dict[str, List[Tuple[str, str, int]]],
    composites: Dict[str, float],
    out_path: str,
) -> bool:
    """One row per MPNN sequence, sorted by composite descending
    (best at top).  Each row has 3 cells (representative sg seeds).
    Cell shape = stage; cell colour = outcome."""
    if not seq_outcomes:
        _make_empty_plot("No seed-outcome data available", out_path)
        return False
    names = sorted(
        seq_outcomes.keys(),
        key=lambda n: -composites.get(n, float("-inf")),
    )
    n_rows = len(names)
    max_cells = max(len(seq_outcomes[n]) for n in names)

    fig_h = max(2.6, n_rows * 0.36 + 1.0)
    fig_w = max(5.5, max_cells * 0.22 + 4.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    for r, name in enumerate(names):
        cells_sorted = sorted(seq_outcomes[name], key=lambda t: t[2])
        for c, (stage, outcome, _seed) in enumerate(cells_sorted):
            colour = OUTCOME_COLOUR.get(outcome, "#000000")
            marker = STAGE_MARKER.get(stage, "x")
            ax.scatter(c, n_rows - 1 - r, marker=marker, s=140,
                       c=colour, edgecolor="black", linewidth=0.6,
                       zorder=3)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([_short_name(n) for n in reversed(names)],
                       fontsize=10)
    ax.set_xticks(range(max_cells))
    ax.set_xticklabels([f"#{i+1}" for i in range(max_cells)], fontsize=8)
    ax.set_xlim(-0.3, max_cells - 0.7)
    ax.set_ylim(-0.4, n_rows - 0.6)
    ax.set_xlabel("Seed (representative sequence_group)", fontsize=10)
    ax.set_ylabel("MPNN sequence", fontsize=10)
    ax.grid(True, axis="both", linestyle="-", linewidth=0.3,
            alpha=0.3, color="grey")
    ax.set_axisbelow(True)
    ax.set_title("Per-seed outcomes  (shape = stage, colour = outcome)",
                 fontsize=11, pad=8)

    outcome_handles = [
        mpatches.Patch(color=OUTCOME_COLOUR[k], label=OUTCOME_LABEL[k])
        for k in OUTCOME_ORDER
    ]
    stage_handles = [
        Line2D([0], [0], marker=STAGE_MARKER[k], color="w",
               markerfacecolor="#888888", markeredgecolor="black",
               markersize=10, linewidth=0, label=STAGE_LABEL[k])
        for k in STAGE_ORDER
    ]
    leg1 = ax.legend(handles=outcome_handles, loc="upper left",
                     bbox_to_anchor=(1.02, 1.0), fontsize=9,
                     framealpha=0.95, title="Outcome",
                     title_fontsize=10)
    ax.add_artist(leg1)
    n_outcome_rows = len(outcome_handles) + 1
    stage_anchor_y = max(0.05, 1.0 - n_outcome_rows * 0.085)
    ax.legend(handles=stage_handles, loc="upper left",
              bbox_to_anchor=(1.02, stage_anchor_y), fontsize=9,
              framealpha=0.95, title="Stage", title_fontsize=10)

    plt.subplots_adjust(right=0.62, top=0.90, bottom=0.20, left=0.16)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_seed_outcomes_bars(
    seq_outcomes: Dict[str, List[Tuple[str, str, int]]],
    composites: Dict[str, float],
    out_path: str,
) -> bool:
    """Two side-by-side stacked bars per sequence, sorted by composite
    descending.  Left panel: stage breakdown (which final stage did
    each seed end at?).  Right panel: outcome breakdown (how did each
    seed fare?)."""
    if not seq_outcomes:
        _make_empty_plot("No seed-outcome data available", out_path)
        return False
    names = sorted(
        seq_outcomes.keys(),
        key=lambda n: -composites.get(n, float("-inf")),
    )
    n = len(names)
    short = [_short_name(x) for x in names]

    stage_counts = {k: np.zeros(n, dtype=int) for k in STAGE_ORDER}
    outcome_counts = {k: np.zeros(n, dtype=int) for k in OUTCOME_ORDER}
    for i, name in enumerate(names):
        for stage, outcome, _seed in seq_outcomes[name]:
            if stage in stage_counts:
                stage_counts[stage][i] += 1
            if outcome in outcome_counts:
                outcome_counts[outcome][i] += 1

    fig_w = max(9.0, n * 0.5 + 4.0)
    fig, (ax_stage, ax_outcome) = plt.subplots(
        1, 2, figsize=(fig_w, 5),
        gridspec_kw={"wspace": 0.18},
    )
    xs = np.arange(n)

    # Stage panel
    bottoms = np.zeros(n, dtype=float)
    for k in STAGE_ORDER:
        heights = stage_counts[k].astype(float)
        if heights.sum() == 0:
            continue
        ax_stage.bar(xs, heights, bottom=bottoms,
                     color=STAGE_COLOUR[k],
                     edgecolor="black", linewidth=0.3,
                     label=STAGE_LABEL[k])
        bottoms += heights
    ax_stage.set_xticks(xs)
    ax_stage.set_xticklabels(short, rotation=90, fontsize=9)
    ax_stage.set_ylabel("Number of seeds (per stage)", fontsize=10)
    ax_stage.set_xlabel("MPNN sequence", fontsize=10)
    ax_stage.yaxis.grid(True, linestyle="-", linewidth=0.3,
                        alpha=0.4, color="grey")
    ax_stage.set_axisbelow(True)
    ax_stage.set_title("Which stage did each seed end at?", fontsize=10)
    ax_stage.legend(loc="lower center", bbox_to_anchor=(0.5, -0.45),
                    fontsize=8, framealpha=0.95, ncol=3)

    # Outcome panel
    bottoms = np.zeros(n, dtype=float)
    for k in OUTCOME_ORDER:
        heights = outcome_counts[k].astype(float)
        if heights.sum() == 0:
            continue
        ax_outcome.bar(xs, heights, bottom=bottoms,
                       color=OUTCOME_COLOUR[k],
                       edgecolor="black", linewidth=0.3,
                       label=OUTCOME_LABEL[k])
        bottoms += heights
    ax_outcome.set_xticks(xs)
    ax_outcome.set_xticklabels(short, rotation=90, fontsize=9)
    ax_outcome.set_ylabel("Number of seeds (per outcome)", fontsize=10)
    ax_outcome.set_xlabel("MPNN sequence", fontsize=10)
    ax_outcome.yaxis.grid(True, linestyle="-", linewidth=0.3,
                          alpha=0.4, color="grey")
    ax_outcome.set_axisbelow(True)
    ax_outcome.set_title("How did each seed fare?", fontsize=10)
    ax_outcome.legend(loc="lower center", bbox_to_anchor=(0.5, -0.45),
                      fontsize=8, framealpha=0.95, ncol=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True

# ═══════════════════════════════════════════════════════════════════════════
# Plot 3: ra_eff vs true_jaccard scatter, with iso-composite contours
# ═══════════════════════════════════════════════════════════════════════════

def plot_ra_eff_vs_jaccard(rows: List[Dict], out_path: str) -> bool:
    """Two-panel layout, stacked vertically:

    Top: scatter of receptor-aligned effector RMSD (x) vs interface
    Jaccard (y), one dot per steered sequence, coloured by tier.
    Iso-composite contours overlaid (composite = jaccard − 0.05·rmsd).

    Bottom: histogram of receptor-aligned effector RMSD across the
    full cohort.  Stacked: tier A (green), tier-none (grey), control
    scrambled (brown), control polyA (purple).  Lets the reader see
    the RMSD distribution including controls and tier-none failures
    that the scatter would otherwise compress into a corner.
    """
    pts_steered: Dict[str, List[Tuple[float, float, str]]] = defaultdict(list)
    pts_ctrl_s: List[Tuple[float, float, str]] = []
    pts_ctrl_p: List[Tuple[float, float, str]] = []

    for row in rows:
        ra = _try_float(row.get("rep_ra_eff_vs_truth_median"))
        tj = _try_float(row.get("rep_true_jaccard_median"))
        if ra is None or tj is None:
            continue
        name = row.get("mpnn_sequence", "?")
        rt = _row_type(row)
        if rt == "control_scrambled":
            pts_ctrl_s.append((ra, tj, name))
        elif rt == "control_polyA":
            pts_ctrl_p.append((ra, tj, name))
        else:
            tier = (row.get("cross_tier") or "none").strip()
            pts_steered[tier].append((ra, tj, name))

    n_pts = (sum(len(v) for v in pts_steered.values())
             + len(pts_ctrl_s) + len(pts_ctrl_p))
    if n_pts == 0:
        _make_empty_plot("No RMSD/jaccard data available", out_path)
        return False

    # Scatter axis range — clip to the meaningful steered region.
    abc_xs = []
    for tier in ("A", "B", "C"):
        abc_xs.extend(p[0] for p in pts_steered.get(tier, []))
    none_xs = [p[0] for p in pts_steered.get("none", [])]

    if abc_xs:
        scatter_x_max = max(abc_xs)
        # Include tier-none in scatter if within 1.5× the worst tier
        # A/B/C; otherwise leave them for the histogram below.
        for x in none_xs:
            if x <= scatter_x_max * 1.5:
                scatter_x_max = max(scatter_x_max, x)
        scatter_x_max += 1.0
    elif none_xs:
        scatter_x_max = max(none_xs) + 1.0
    else:
        scatter_x_max = 10.0
    scatter_x_min = max(0.0, (min(abc_xs) if abc_xs else 0.0) - 1.0)

    # Y range — anchored to (0, 1) but extended if data spills.
    all_y = []
    for v in pts_steered.values():
        all_y.extend(p[1] for p in v)
    all_y.extend(p[1] for p in pts_ctrl_s)
    all_y.extend(p[1] for p in pts_ctrl_p)
    y_min = min(0.0, min(all_y) - 0.05)
    y_max = max(1.0, max(all_y) + 0.05)

    # ── Layout: scatter on top (taller), histogram below (shorter) ────
    fig, (ax, ax_h) = plt.subplots(
        2, 1, figsize=(7.5, 7.5),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.35},
    )

    # ── Top panel: iso-composite contours + scatter ───────────────────
    ra_grid = np.linspace(scatter_x_min, scatter_x_max, 50)
    composite_levels = [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
    for comp in composite_levels:
        tj_line = comp + COMPOSITE_RA_EFF_WEIGHT * ra_grid
        ax.plot(ra_grid, tj_line, color=COLOUR_CONTOUR,
                linewidth=0.7, linestyle="--", alpha=0.55, zorder=1)
        x_label = scatter_x_min + (scatter_x_max - scatter_x_min) * 0.5
        y_label = comp + COMPOSITE_RA_EFF_WEIGHT * x_label
        if y_min <= y_label <= y_max:
            ax.annotate(
                f"comp={comp:.1f}",
                xy=(x_label, y_label),
                xytext=(0, 3), textcoords="offset points",
                fontsize=7, color=COLOUR_CONTOUR,
                ha="center", va="bottom", alpha=0.9,
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.7, pad=1),
            )

    for tier in ("A", "B", "C", "none"):
        pts = [p for p in pts_steered.get(tier, []) if p[0] <= scatter_x_max]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(xs, ys, s=44, c=COLOUR_TIER[tier],
                   edgecolor="black", linewidth=0.4, alpha=0.9,
                   label=f"Tier {tier} (n={len(pts_steered.get(tier, []))})",
                   zorder=3)

    ax.set_xlim(scatter_x_min, scatter_x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Receptor Aligned Effector RMSD (Å)", fontsize=10)
    ax.set_ylabel("Interface Jaccard", fontsize=10)
    ax.grid(True, linestyle="-", linewidth=0.3, alpha=0.4, color="grey")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02),
              fontsize=8, framealpha=0.95, ncol=4)

    # ── Bottom panel: stacked histogram of RMSD across full cohort ────
    # Build the full RMSD distribution including controls + everything
    # off-scale.  Stack: tier A / B / C / none / ctrl_scrambled /
    # ctrl_polyA so the reader can see at a glance where each cohort
    # group sits along the RMSD axis.
    series = []
    series_colours = []
    series_labels = []
    for tier in ("A", "B", "C", "none"):
        vals = [p[0] for p in pts_steered.get(tier, [])]
        if vals:
            series.append(vals)
            series_colours.append(COLOUR_TIER[tier])
            series_labels.append(f"Tier {tier} (n={len(vals)})")
    if pts_ctrl_s:
        series.append([p[0] for p in pts_ctrl_s])
        series_colours.append(COLOUR_CTRL_SCRAM)
        series_labels.append(f"ctrl: scrambled (n={len(pts_ctrl_s)})")
    if pts_ctrl_p:
        series.append([p[0] for p in pts_ctrl_p])
        series_colours.append(COLOUR_CTRL_POLYA)
        series_labels.append(f"ctrl: polyA (n={len(pts_ctrl_p)})")

    if series:
        all_rmsd = [v for s in series for v in s]
        h_x_min = 0.0
        h_x_max = max(all_rmsd) * 1.05
        # Bin width — aim for ~25 bins across the full range.
        n_bins = max(15, min(40, int(np.sqrt(len(all_rmsd)) * 3)))
        bins = np.linspace(h_x_min, h_x_max, n_bins + 1)
        ax_h.hist(series, bins=bins, stacked=True,
                  color=series_colours, label=series_labels,
                  edgecolor="black", linewidth=0.3)
        ax_h.set_xlim(h_x_min, h_x_max)
        ax_h.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax_h.set_xlabel("Receptor Aligned Effector RMSD (Å)", fontsize=10)
        ax_h.set_ylabel("Count", fontsize=10)
        ax_h.grid(True, axis="y", linestyle="-", linewidth=0.3,
                  alpha=0.4, color="grey")
        ax_h.set_axisbelow(True)
        ax_h.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02),
                    fontsize=7, framealpha=0.95, ncol=4)
    else:
        ax_h.text(0.5, 0.5, "no data", ha="center", va="center",
                  transform=ax_h.transAxes, color="grey")
        ax_h.set_axis_off()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Plot 4: Confidence-filter cascade
# ═══════════════════════════════════════════════════════════════════════════

def plot_filter_cascade(rows: List[Dict], thresholds: Dict, out_path: str) -> bool:
    """Waterfall: how many sequences pass each Boltz confidence filter
    in succession.  Thresholds taken from `thresholds` dict (forwarded
    from CLI flags), so the bar labels reflect the values actually
    used by extract_passing.py rather than hard-coded constants.

    All four thresholds applied here are confidence FILTERS (their
    job is to drop catastrophes, not rank candidates) — see the
    metric-reliability memo.
    """
    steered = [r for r in rows if _is_steered(r)]
    if not steered:
        _make_empty_plot("No steered rows available", out_path)
        return False

    n_total = len(steered)
    plddt_min = thresholds["complex_plddt_min"]
    ipae_max = thresholds["ipae_max"]
    paepf_min = thresholds["pae_pass_frac_min"]
    iptm_min = thresholds["iptm_min"]
    # Structural ra_eff cap matches iterate_steering's
    # _agg_row_is_clean_steered / _agg_row_is_pose_holds (5.0 Å).
    # This is the SAME structural filter that decides whether a
    # sequence's aggregate gets a rank_by_composite_score and
    # therefore appears in passing_summary / cross_summary.  Adding
    # it here makes the cascade's final-step count match the actual
    # tier-A/B/C count rather than diverging from it.
    ra_eff_max = thresholds.get("ra_eff_max", 5.0)

    # Per-sequence boolean pass arrays — one per filter, in cascade order.
    pass_plddt, pass_ipae, pass_paepf, pass_iptm, pass_ra = (
        [], [], [], [], [])
    for row in steered:
        pl   = _try_float(row.get("rep_complex_plddt_median"))
        ip   = _try_float(row.get("rep_ipae_median"))
        ppf  = _try_float(row.get("rep_pae_pass_frac_median"))
        iptm = _try_float(row.get("rep_iptm_median"))
        ra   = _try_float(row.get("rep_ra_eff_vs_truth_median"))
        pass_plddt.append(pl is not None and pl >= plddt_min)
        pass_ipae.append(ip is not None and ip <= ipae_max)
        pass_paepf.append(ppf is not None and ppf >= paepf_min)
        pass_iptm.append(iptm is not None and iptm >= iptm_min)
        pass_ra.append(ra is not None and ra < ra_eff_max)

    pa = np.array(pass_plddt)
    pb = np.array(pass_ipae)
    pc = np.array(pass_paepf)
    pd = np.array(pass_iptm)
    pe = np.array(pass_ra)

    # Cumulative-AND survival counts.
    n_plddt              = int(pa.sum())
    n_plddt_ipae         = int((pa & pb).sum())
    n_plddt_ipae_pf      = int((pa & pb & pc).sum())
    n_all_confidence     = int((pa & pb & pc & pd).sum())
    n_all_confidence_ra  = int((pa & pb & pc & pd & pe).sum())

    drops = [
        0,
        n_total - n_plddt,
        n_plddt - n_plddt_ipae,
        n_plddt_ipae - n_plddt_ipae_pf,
        n_plddt_ipae_pf - n_all_confidence,
        n_all_confidence - n_all_confidence_ra,
    ]
    survivors = [n_total, n_plddt, n_plddt_ipae,
                 n_plddt_ipae_pf, n_all_confidence,
                 n_all_confidence_ra]

    # Labels read thresholds from the dict so they reflect what the
    # pipeline actually applied.
    labels = [
        "All steered",
        f"complex_plddt\n≥ {plddt_min:g}",
        f"+ ipae\n≤ {ipae_max:g}",
        f"+ pae_pass_frac\n≥ {paepf_min:g}",
        f"+ iptm\n≥ {iptm_min:g}",
        f"+ ra_eff\n< {ra_eff_max:g}",
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(len(labels))

    ax.bar(xs, survivors, color=COLOUR_STEERED, edgecolor="black",
           linewidth=0.4, label="Surviving", zorder=2)
    ax.bar(xs, drops, bottom=survivors, color=COLOUR_THRESHOLD,
           edgecolor="black", linewidth=0.4, alpha=0.85,
           label="Dropped at this step", zorder=2)

    for i, (s, d) in enumerate(zip(survivors, drops)):
        if s > 0:
            ax.text(i, s / 2, f"{s}", ha="center", va="center",
                    fontsize=10, color="white", fontweight="bold")
        if d > 0:
            ax.text(i, s + d / 2, f"−{d}", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Number of sequences", fontsize=10)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.grid(True, linestyle="-", linewidth=0.3, alpha=0.4, color="grey")
    ax.set_axisbelow(True)
    # Legend lifted above the panel area, single row, so it never
    # overlaps the bars (the leftmost "All steered" bar is always
    # the tallest).
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02),
              ncol=2, fontsize=9, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Plot 5: Controls diagnostic
# ═══════════════════════════════════════════════════════════════════════════

def plot_controls_diagnostic(rows: List[Dict], thresholds: Dict, out_path: str) -> bool:
    """Two-panel boxplot: ra_eff (top), true_jaccard (bottom).
    Grouped by row_type.  Pale-red shading above the
    controls_warning_ra_eff_min threshold marks the "expected control
    region" — values above this line are where controls SHOULD sit;
    values below it are where steered candidates should sit.

    If the control distributions overlap the steered distribution,
    the pipeline is mis-calibrated.
    """
    groups: Dict[str, Dict[str, List[float]]] = {
        "steered_passing":   {"ra": [], "tj": []},   # tier A/B/C
        "steered_failing":   {"ra": [], "tj": []},   # tier none
        "control_scrambled": {"ra": [], "tj": []},
        "control_polyA":     {"ra": [], "tj": []},
    }
    for row in rows:
        rt = _row_type(row)
        ra = _try_float(row.get("rep_ra_eff_vs_truth_median"))
        tj = _try_float(row.get("rep_true_jaccard_median"))
        if rt == "steered":
            tier = (row.get("cross_tier") or "none").strip()
            bucket = "steered_passing" if tier in ("A", "B", "C") else "steered_failing"
            if ra is not None:
                groups[bucket]["ra"].append(ra)
            if tj is not None:
                groups[bucket]["tj"].append(tj)
        elif rt in groups:
            if ra is not None:
                groups[rt]["ra"].append(ra)
            if tj is not None:
                groups[rt]["tj"].append(tj)

    if all(not (g["ra"] or g["tj"]) for g in groups.values()):
        _make_empty_plot("No data for controls diagnostic", out_path)
        return False

    ra_threshold = thresholds.get("control_ra_eff_min", CONTROL_RA_EFF_MIN)

    fig, (ax_ra, ax_tj) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    group_order   = ["steered_passing",  "steered_failing",   "control_scrambled", "control_polyA"]
    group_labels  = ["steered: passing",  "steered: failing",  "ctrl: scrambled",   "ctrl: polyA"]
    group_colours = [COLOUR_TIER["A"],     COLOUR_TIER["none"], COLOUR_CTRL_SCRAM,   COLOUR_CTRL_POLYA]

    # ── Panel 1: ra_eff ────────────────────────────────────────────
    ra_data = [groups[g]["ra"] for g in group_order]
    counts_ra = [len(d) for d in ra_data]
    # Replace empty groups with [nan] so positions are preserved.
    # showfliers=False — outlier markers double up with the jittered
    # data points overlaid below, making the panel look noisier than
    # it is.
    ra_plot = [d if d else [np.nan] for d in ra_data]
    bp_ra = ax_ra.boxplot(ra_plot, positions=range(len(group_order)),
                           widths=0.55, patch_artist=True, showfliers=False)
    for patch, c in zip(bp_ra["boxes"], group_colours):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for i, d in enumerate(ra_data):
        if d:
            jitter = np.random.uniform(-0.08, 0.08, size=len(d))
            ax_ra.scatter(np.full(len(d), i) + jitter, d,
                          c=group_colours[i], edgecolor="black",
                          linewidth=0.3, s=22, alpha=0.85, zorder=3)

    # Plot the data first so matplotlib computes the y-axis range.
    # Then set the y-limits explicitly: top is just slightly above the
    # max data point (to avoid a huge empty red expanse when controls
    # sit at e.g. 25 Å), with a minimum of 1.5x threshold so the
    # threshold itself is visibly placed.
    y_top_data = max([v for d in ra_data for v in d] + [ra_threshold])
    y_top = max(y_top_data * 1.05, ra_threshold * 1.5)
    ax_ra.set_ylim(0, y_top)

    # Pale-red shading from threshold to the top of the panel.  Drawn
    # before the threshold line so the line sits on top of the shading.
    ax_ra.axhspan(ra_threshold, ax_ra.get_ylim()[1],
                  color=COLOUR_THRESHOLD, alpha=0.07, zorder=0)
    ax_ra.axhline(ra_threshold, color=COLOUR_THRESHOLD,
                  linestyle="--", linewidth=1.2, zorder=4,
                  label=f"ra_eff control threshold = {ra_threshold:g} Å")
    ax_ra.set_ylabel("Receptor Aligned Effector RMSD (Å)", fontsize=10)
    ax_ra.yaxis.grid(True, linestyle="-", linewidth=0.3, alpha=0.4, color="grey")
    ax_ra.set_axisbelow(True)
    # Legend ABOVE the panel area so it never overlaps data points or
    # the threshold line.  bbox_to_anchor y > 1.0 puts it outside the
    # axes box at the top.
    ax_ra.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02),
                 fontsize=9, framealpha=0.95)

    # ── Panel 2: true_jaccard ─────────────────────────────────────
    tj_data = [groups[g]["tj"] for g in group_order]
    counts_tj = [len(d) for d in tj_data]
    tj_plot = [d if d else [np.nan] for d in tj_data]
    bp_tj = ax_tj.boxplot(tj_plot, positions=range(len(group_order)),
                           widths=0.55, patch_artist=True, showfliers=False)
    for patch, c in zip(bp_tj["boxes"], group_colours):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for i, d in enumerate(tj_data):
        if d:
            jitter = np.random.uniform(-0.08, 0.08, size=len(d))
            ax_tj.scatter(np.full(len(d), i) + jitter, d,
                          c=group_colours[i], edgecolor="black",
                          linewidth=0.3, s=22, alpha=0.85, zorder=3)

    ax_tj.set_ylabel("Interface Jaccard", fontsize=10)
    ax_tj.yaxis.grid(True, linestyle="-", linewidth=0.3, alpha=0.4, color="grey")
    ax_tj.set_axisbelow(True)

    # Single n per group: prefer ra_eff count, fall back to tj count if
    # ra_eff count is zero.  Drop the "row_type" axis label entirely.
    ax_tj.set_xticks(range(len(group_order)))
    ax_tj.set_xticklabels(
        [f"{lbl}\n(n={n_ra if n_ra else n_tj})"
         for lbl, n_ra, n_tj in zip(group_labels, counts_ra, counts_tj)],
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Plot 6: Mutation impact (per input-PDB position)
# ═══════════════════════════════════════════════════════════════════════════

def plot_mutation_impact(
    rows: List[Dict],
    design_region_positions: Optional[set],
    pdb_length: Optional[int],
    out_path: str,
) -> bool:
    """For each input-PDB position p that appears in any steered
    sequence's FINAL mutation set, compute the median composite score
    of sequences mutating p.  Bars coloured by delta vs cohort median
    (green = better, red = worse), faded for low-n.

    Final mutation set = the mutations present in the prediction that
    cross_summary calls "representative".  When reversion ran
    (verdict in {pose_holds, pose_collapses, new_contamination}),
    that's the reverted majority set — reversion may have reverted
    some of the steered mutations back to wild-type, so the steered
    set would over-count positions not in the final structure.  When
    reversion didn't run, the steered set IS the final set.

    The x-axis spans the full input PDB length (1 → pdb_length) so
    positions with no mutations show as gaps.  When the
    input_design_region.txt is available, the design-region footprint
    on the input PDB is shaded — by construction steering mutations
    are forbidden from those positions.
    """
    # Build per-position: list of (sequence_name, composite_score) pairs.
    per_pos: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
    cohort_composites: List[float] = []

    for row in rows:
        if not _is_steered(row):
            continue
        comp = _composite_from_row(row)
        if comp is None:
            continue
        cohort_composites.append(comp)
        # Pick the final mutation set based on verdict.  When reversion
        # ran, the reverted_majority set is what survived.  Otherwise
        # the steered set is the final.
        verdict = (row.get("rep_outcome") or "").strip()
        if verdict in _VERDICTS_WITH_REVERSION:
            mut_str = (row.get("rep_reverted_mutations_majority_chimerax")
                       or row.get("rep_steered_mutations_chimerax", ""))
        else:
            mut_str = row.get("rep_steered_mutations_chimerax", "")
        positions = _parse_chimerax_positions(mut_str)
        name = row.get("mpnn_sequence", "?")
        for p in positions:
            per_pos[p].append((name, comp))

    if not per_pos:
        _make_empty_plot(
            "No steered_mutations data found in cross-summary",
            out_path,
        )
        return False

    cohort_median = float(np.median(cohort_composites)) if cohort_composites else 0.0

    # Per-position medians.
    positions = sorted(per_pos.keys())
    medians = [float(np.median([c for _, c in per_pos[p]])) for p in positions]
    deltas  = [m - cohort_median for m in medians]
    counts  = [len(per_pos[p]) for p in positions]

    # X-axis range: 1 to full PDB length when known, else just the
    # range of mutated positions.
    if pdb_length and pdb_length > 0:
        x_min, x_max = 1, pdb_length
    else:
        x_min = min(positions)
        x_max = max(positions)

    # Compressed width: ~0.10 fig-units per residue, capped to 14.
    fig_w = max(8, min(14, (x_max - x_min) * 0.10 + 2))
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))

    # Design-region shading: positions on the input PDB whose ranges
    # the contig replaces with de novo segments.  Mutations CANNOT
    # fall here (they're forbidden by the contig) — the shading is
    # context, not a data slot.
    if design_region_positions:
        sorted_dr = sorted(design_region_positions)
        run_start = sorted_dr[0]
        prev = sorted_dr[0]
        for p in sorted_dr[1:] + [None]:
            if p is None or p != prev + 1:
                ax.axvspan(run_start - 0.5, prev + 0.5,
                           color=COLOUR_DESIGN_REGION, alpha=0.6, zorder=0)
                if p is not None:
                    run_start = p
            prev = p if p is not None else prev

    # Cohort-median reference line.
    ax.axhline(cohort_median, color="black", linewidth=0.6,
               linestyle=":", alpha=0.5, zorder=1)

    # Bars on the plot — ALWAYS the green/red colour from delta vs
    # cohort median.  Low-n bars are drawn faded (low alpha) but stay
    # green or red — they are NOT grey.  The grey swatch in the legend
    # is a symbol for "the faded versions of the colours above"; no
    # bar on the plot is ever drawn grey.
    for p, m, d, c in zip(positions, medians, deltas, counts):
        colour = COLOUR_BETTER_THAN if d >= 0 else COLOUR_WORSE_THAN
        alpha = 0.30 if c < MUT_IMPACT_MIN_N else 0.85
        ax.bar(p, m, width=0.8, color=colour, alpha=alpha,
               edgecolor="black", linewidth=0.4, zorder=2)
        # Count label above/below the bar.
        if m >= 0:
            va = "bottom"
            y_text = m + 0.005
        else:
            va = "top"
            y_text = m - 0.005
        ax.text(p, y_text, f"n={c}", ha="center", va=va,
                fontsize=6, color="black", alpha=0.75, zorder=3)

    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_xlabel("Input-PDB residue position (chain A)", fontsize=10)
    ax.set_ylabel("Median composite score across\n"
                  "sequences mutating this position", fontsize=10)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.3, alpha=0.4, color="grey")
    ax.set_axisbelow(True)

    # Legend.  Cohort median line, two bright bar swatches (n ≥ MIN_N),
    # one GREY symbolic swatch labelled "Faded = ..." that represents
    # the faded green/red versions on the plot, and the design-region
    # shading.  No grey bar is ever drawn on the plot itself; the
    # grey swatch is purely a legend symbol for the faded states.
    legend_handles = [
        plt.Line2D([0], [0], color="black", linestyle=":", linewidth=0.6,
                   label=f"Cohort median = {cohort_median:.3f}"),
        mpatches.Patch(color=COLOUR_BETTER_THAN, alpha=0.85,
                       label="Median ≥ cohort"),
        mpatches.Patch(color=COLOUR_WORSE_THAN, alpha=0.85,
                       label="Median < cohort"),
        mpatches.Patch(color="#B0B0B0", alpha=0.55,
                       label=f"Faded = n < {MUT_IMPACT_MIN_N} (low confidence)"),
    ]
    if design_region_positions:
        legend_handles.append(
            mpatches.Patch(color=COLOUR_DESIGN_REGION, alpha=0.6,
                           label="Design region (no mutations possible)")
        )
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, 1.18), fontsize=8, framealpha=0.95,
              ncol=len(legend_handles))

    # X-tick spacing — aim for ~20 ticks max.
    span = x_max - x_min
    step = max(1, span // 20)
    ax.set_xticks(np.arange(x_min, x_max + 1, step))
    ax.tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Top-level orchestration
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_csv_path(supplied: Optional[str]) -> Path:
    # Production: --csv is required and injected by NEGSTEER_PLOTS;
    # the test-script equivalent has a canonical-test-path fallback
    # for fast iteration without Nextflow.  Keep them in sync, but
    # the fallback never belongs in production.
    if not supplied:
        raise SystemExit("--csv is required")
    p = Path(supplied)
    if not p.exists():
        raise SystemExit(f"--csv path does not exist: {p}")
    return p


def _load_input_design_region(
    input_design_region_path: Optional[str],
) -> Tuple[Optional[set], Optional[int]]:
    """Load input-PDB-numbered design-region positions from
    input_design_region.txt (1-based, comma-separated, written by
    derive_input_design_region.py and published to
    .../negative_steering/controls_inputs/).

    Returns (positions_set, pdb_length) — either may be None if file
    missing or unreadable.  pdb_length is unknown from this file alone
    (it lists only the design-region positions; the file has no record
    of the receptor length), so callers should derive pdb_length from
    elsewhere if it's needed.

    File format:
        # comments lines starting with #
        1,2,3,30,31,32,...     comma-separated 1-based positions
    """
    if not input_design_region_path:
        return None, None
    p = Path(input_design_region_path)
    if not p.exists():
        print(f"WARNING: --input-design-region path not found: {p}")
        print("         (mutation-impact plot will render without "
              "design-region shading)")
        return None, None
    try:
        with open(p) as f:
            data = f.read()
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not read {p}: {type(e).__name__}: {e}")
        return None, None

    positions: set = set()
    for line in data.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for tok in line.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                positions.add(int(tok))
            except ValueError:
                pass

    if not positions:
        print(f"NOTE: no positions parsed from {p}")
        return None, None
    return positions, max(positions)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--csv", required=True,
        help="cross_sequence_summary.csv from NEGSTEER_CROSS_SEQUENCE.")
    ap.add_argument("--runs-dir", default=None,
        help="Path to runs/ directory containing per-sequence subdirs "
             "(each with aggregated_results.csv).  When provided, "
             "the cohort plots include sequences with zero passing "
             "rows by reading their aggregated_results.csv directly.  "
             "Falls back to <csv>/../runs if omitted.")
    ap.add_argument("--input-design-region", default=None,
        help="Path to input_design_region.txt produced by "
             "DERIVE_INPUT_INDICES (lives at "
             ".../negative_steering/controls_inputs/).  Used by the "
             "mutation-impact plot to shade input-PDB residues that "
             "fall within the contig's design-region footprint.  "
             "Optional — plot still renders without it.")
    ap.add_argument("--pdb-length", type=int, default=None,
        help="Length of the input receptor chain.  Optional — used "
             "by the mutation-impact plot to extend the x-axis to "
             "the full PDB length even when no mutations occur near "
             "the C-terminus.  When omitted, falls back to the "
             "maximum of (mutation positions, design-region max).")
    ap.add_argument("--complex-plddt-min", type=float,
                     default=CONFIDENCE_PLDDT_MIN,
        help=f"complex_plddt threshold used in the cascade plot "
             f"(default {CONFIDENCE_PLDDT_MIN}; matches extract_passing).")
    ap.add_argument("--ipae-max", type=float, default=CONFIDENCE_IPAE_MAX,
        help=f"ipae threshold used in the cascade plot "
             f"(default {CONFIDENCE_IPAE_MAX}).")
    ap.add_argument("--pae-pass-frac-min", type=float,
                     default=CONFIDENCE_PAEPF_MIN,
        help=f"pae_pass_frac threshold used in the cascade plot "
             f"(default {CONFIDENCE_PAEPF_MIN}).")
    ap.add_argument("--iptm-min", type=float, default=CONFIDENCE_IPTM_MIN,
        help=f"iptm threshold used in the cascade plot "
             f"(default {CONFIDENCE_IPTM_MIN}).")
    ap.add_argument("--outdir", required=True,
        help="Directory to write plots into; created if missing.")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    csv_path = _resolve_csv_path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Resolve runs_dir: explicit --runs-dir, else <csv>/../runs.
    if args.runs_dir:
        runs_dir: Optional[Path] = Path(args.runs_dir)
        if not runs_dir.exists():
            print(f"WARNING: --runs-dir not found: {runs_dir}")
            runs_dir = None
    else:
        candidate = csv_path.parent / "runs"
        runs_dir = candidate if candidate.exists() else None
        if runs_dir is None:
            print(f"NOTE: no runs/ found at {candidate}; falling back to "
                  f"cross-summary-only mode (zero-passing sequences will "
                  f"be invisible).")

    print(f"Reading cross-sequence summary: {csv_path}")
    if runs_dir is not None:
        print(f"Reading per-sequence aggregated_results from: {runs_dir}")
    rows = load_unified_cohort(csv_path, runs_dir)
    print(f"  unified cohort rows: {len(rows)}")
    if not rows:
        print("No rows — nothing to plot.")
        return 0

    n_steered = sum(1 for r in rows if _is_steered(r))
    n_ctrl = len(rows) - n_steered
    n_with_tier = sum(1 for r in rows
                      if _is_steered(r)
                      and (r.get("cross_tier") or "none") != "none")
    print(f"  steered: {n_steered}  (with tier: {n_with_tier}, "
          f"tier-none: {n_steered - n_with_tier})")
    print(f"  controls: {n_ctrl}")

    design_region, dr_max = _load_input_design_region(args.input_design_region)
    if design_region:
        print(f"  loaded {len(design_region)} input-PDB design-region positions")

    # PDB length: explicit --pdb-length first, else fall back to the
    # design-region max (which is at least an upper bound on the part
    # of the PDB any plot cares about).
    pdb_length: Optional[int] = args.pdb_length
    if pdb_length is None and dr_max is not None:
        pdb_length = dr_max
    if pdb_length:
        print(f"  using PDB length: {pdb_length}")

    # Pack thresholds into a dict for plots that need them.
    thresholds = {
        "complex_plddt_min":   args.complex_plddt_min,
        "ipae_max":            args.ipae_max,
        "pae_pass_frac_min":   args.pae_pass_frac_min,
        "iptm_min":            args.iptm_min,
        "control_ra_eff_min":  CONTROL_RA_EFF_MIN,
    }

    # Per-seed outcomes — loaded once for both heatmap and bars.
    # Heatmap uses representative sg only (3 seeds per sequence).
    # Bars use all sgs (typically 9 seeds per sequence: 3 sg × 3 Boltz
    # seeds).  Both require runs_dir to find raw_per_seed_results.csv.
    seed_outcomes_rep: Dict[str, List[Tuple[str, str, int]]] = {}
    seed_composites: Dict[str, float] = {}
    seed_outcomes_all: Dict[str, List[Tuple[str, str, int]]] = {}
    if runs_dir is not None:
        try:
            seed_outcomes_rep, seed_composites = load_seed_outcomes(
                runs_dir, csv_path, all_sgs=False)
            seed_outcomes_all, _ = load_seed_outcomes(
                runs_dir, csv_path, all_sgs=True)
            print(f"Loaded seed outcomes for "
                  f"{len(seed_outcomes_rep)} sequences (rep sg) / "
                  f"{len(seed_outcomes_all)} sequences (all sgs)")
        except Exception as e:  # noqa: BLE001
            print(f"WARN: seed-outcomes load failed: {e}")

    plots = [
        ("Tier landscape (composite score per sequence)",
         "negsteer_tier_landscape.png",
         lambda p: plot_tier_landscape(rows, p)),
        ("Per-seed outcomes — heatmap (representative sg)",
         "negsteer_seed_outcomes_heatmap.png",
         lambda p: plot_seed_outcomes_heatmap(
             seed_outcomes_rep, seed_composites, p)),
        ("Per-seed outcomes — stacked bars (all sgs)",
         "negsteer_seed_outcomes_bars.png",
         lambda p: plot_seed_outcomes_bars(
             seed_outcomes_all, seed_composites, p)),
        ("ra_eff vs true_jaccard",
         "negsteer_ra_eff_vs_jaccard.png",
         lambda p: plot_ra_eff_vs_jaccard(rows, p)),
        ("Confidence-filter cascade",
         "negsteer_filter_cascade.png",
         lambda p: plot_filter_cascade(rows, thresholds, p)),
        ("Controls diagnostic (steered vs scrambled vs polyA)",
         "negsteer_controls_diagnostic.png",
         lambda p: plot_controls_diagnostic(rows, thresholds, p)),
        ("Mutation impact per input-PDB position",
         "negsteer_mutation_impact.png",
         lambda p: plot_mutation_impact(rows, design_region, pdb_length, p)),
    ]

    for desc, fname, fn in plots:
        out = outdir / fname
        print(f"Building {desc}...")
        try:
            ok = fn(str(out))
            if ok:
                print(f"  → {out}")
            else:
                print("  (skipped — no data)")
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {type(e).__name__}: {e}")

    print(f"\nDone.  Inspect: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())