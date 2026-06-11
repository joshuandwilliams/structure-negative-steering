#!/usr/bin/env python3
"""
negsteer_within_sequence_plots.py
---------------------------------
Production within-sequence (per-seed) plot script for negative
steering (Task 43).  Invoked from
modules/negative_steering.nf::NEGSTEER_WITHIN_SEQUENCE_PLOTS.

Lifted verbatim from
tests/negative_steering/test_negsteer_within_sequence_plots.py;
the only difference is this header.  The two scripts must stay in
sync; when iterating on plots, edit the test script first, validate
against real cluster data, then mirror the changes here.

Reads per-sequence raw_per_seed_results.csv files from the runs/
directory of a previous test_negative_steering run.  Each CSV holds
one row per Boltz seed prediction; this script visualises that
per-seed dispersion across the cohort so you can see whether
cohort-level medians (in cross_sequence_summary.csv) are trustworthy
or driven by a single rogue seed.

The "final" prediction for each seed is verdict-aware: when reversion
ran (verdict in pose_holds / pose_collapses / new_contamination) the
reverted_* metrics are the seed's final values; otherwise the
steered_* metrics are.  For cold-start-only paths (no mutations
applied), the steered_* values ARE the cold-start prediction.

Plots produced
--------------

negsteer_per_seed_dispersion_overview.png
    Single scatter, one point per (MPNN sequence, Boltz seed) of the
    representative sequence_group.  X = ra_eff (final), Y =
    true_jaccard (final).  Shape encodes which stage produced the
    final prediction (circle = cold-start, square = steering, triangle
    = reversion); colour encodes tier.

negsteer_per_seed_dispersion_grid.png
    Small-multiples grid, one panel per MPNN sequence, sorted by
    composite descending.  Shows ALL seeds across all sequence_groups:
    other-sg seeds in grey, the representative-sg seeds highlighted
    green (ra_eff < 5 Å, pass) or red (ra_eff ≥ 5 Å, fail).  Vertical
    threshold line at ra_eff = 5 Å.

negsteer_per_design_stage_trajectories.png
    One panel per RFDiffusion design.  X-axis: stage (cold-start →
    steering → reversion).  One line per (MPNN sequence, sg, seed)
    showing ra_eff at each stage where data exists.  Cold-start point
    is shared across seeds (single pre-steering prediction); steered
    and reverted points are per-seed.  Marker placed on the FINAL
    stage.  Lines coloured by MPNN sequence number (so seq 3 is the
    same colour wherever it appears across designs).  Dashed line at
    ra_eff = 5 Å.

negsteer_weighted_vs_true_jaccard.png
    Single scatter, one point per (MPNN sequence, Boltz seed) of the
    representative sequence_group.  X = true_jaccard (final), Y =
    weighted_jaccard (final).  y = x diagonal drawn.  Small jitter
    applied so stacked seeds remain visible.

negsteer_composite_vs_confidence.png
    Multi-panel scatter.  Per panel: X = a confidence metric (median
    across the representative sg's seeds, per MPNN sequence), Y =
    composite score (per MPNN sequence).  Bounded [0, 1] metrics are
    shown on a fixed 0-1 axis; pLDDT is shown on its native 0-100
    scale; ipae and pae_mean autoscale.  Threshold lines drawn for
    the 4 gating metrics (complex_pLDDT, ipTM, ipae, pae_pass_frac).

Usage
-----
    python test_negsteer_within_sequence_plots.py \\
        --runs-dir tests/negative_steering/receptor_resurfacing_results/negative_steering/runs \\
        --cross-summary-csv tests/negative_steering/receptor_resurfacing_results/negative_steering/cross_sequence_summary.csv \\
        --outdir tests/negative_steering/receptor_resurfacing_results/plots_iter

Sequences with no raw_per_seed_results.csv (or empty CSVs) are
silently skipped.  Controls (input_control_*) are excluded; the
within-sequence plots are about per-seed dispersion of steered
sequences, not control behaviour.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


# ── Styling (mirrors cohort script for consistency) ─────────────────────
COLOUR_TIER = {
    "A":    "#2CA02C",
    "B":    "#FFB000",
    "C":    "#FF7F0E",
    "none": "#B0B0B0",
}
COLOUR_SEED         = "#4C72B0"   # default seed scatter (when no tier known)
COLOUR_DIAG         = "#888888"   # y=x reference
COLOUR_VERDICT_HOLD = "#2CA02C"   # reversion_verdict = pose_holds (good)
COLOUR_VERDICT_NEW  = "#D62728"   # new_contamination
COLOUR_VERDICT_COL  = "#FFB000"   # pose_collapses
COLOUR_VERDICT_NA   = "#888888"   # no_data / clean_steered / other

# Constants used by the per-seed classifier and for confidence-metric
# threshold lines.  Mirror extract_passing.py / cohort plot script.
RA_EFF_MAX        = 5.0
COMPLEX_PLDDT_MIN = 0.70
IPAE_MAX          = 15.0
PAE_PASS_FRAC_MIN = 0.10
IPTM_MIN          = 0.30

# Stage encoding — same shapes as the cohort script's seed-outcomes
# heatmap so the two scripts are visually consistent.
STAGE_MARKER = {"cold_start": "o", "steering": "s", "reversion": "^"}
STAGE_LABEL  = {"cold_start": "Cold-start",
                "steering":   "Steering",
                "reversion":  "Reversion"}
STAGE_ORDER  = ["cold_start", "steering", "reversion"]

# Verdict bucket used by load_cohort_seeds when picking final metrics.
_VERDICTS_WITH_REVERSION = {"pose_holds", "pose_collapses",
                             "new_contamination"}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _try_float(v) -> Optional[float]:
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


def _short_name(name: str) -> str:
    n = name
    if n.startswith("design_"):
        n = n[len("design_"):]
        n = n.replace("_seq_", "_s")
        return f"d{n}"
    if n.startswith("input_control_"):
        return f"ctrl_{n[len('input_control_'):][:4]}"
    return n


def _design_id(name: str) -> str:
    """design_0_seq_2 → d0  (used for grouping in trajectory plot)."""
    m = re.match(r"design_(\d+)_seq_\d+", name)
    return f"d{m.group(1)}" if m else name


def _seq_num(name: str) -> int:
    """design_0_seq_3 → 3  (for stable colour assignment across designs)."""
    m = re.match(r"design_\d+_seq_(\d+)", name)
    return int(m.group(1)) if m else 0


def _make_empty_plot(message: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center",
            transform=ax.transAxes, fontsize=12, color="grey")
    ax.set_axis_off()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _is_control(seq_name: str) -> bool:
    return seq_name.startswith("input_control_")


def _grid_shape(n: int) -> Tuple[int, int]:
    """Pick a reasonable (n_rows, n_cols) for a small-multiples grid
    of `n` panels.  Aim for landscape-ish shape, max 4 columns."""
    if n <= 0:
        return (1, 1)
    if n <= 3:
        return (1, n)
    n_cols = min(4, math.ceil(math.sqrt(n)))
    n_rows = math.ceil(n / n_cols)
    return (n_rows, n_cols)



def classify_seed(row: Dict) -> Tuple[str, str]:
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
    structural_pass = (intact == 1) and (ra < RA_EFF_MAX)
    confidence_pass = (
        plddt >= COMPLEX_PLDDT_MIN
        and ipae <= IPAE_MAX
        and paepf >= PAE_PASS_FRAC_MIN
        and iptm >= IPTM_MIN
    )
    if structural_pass and confidence_pass:
        return stage, "pass"
    if not structural_pass and not confidence_pass:
        return stage, "multiple_failures"
    if not structural_pass and confidence_pass:
        return stage, "off_target"
    return stage, "poor_prediction"


def _rep_sg(seq_dir: Path, cs_lookup: Dict[str, Dict]) -> Optional[str]:
    """Pick representative sg: cross_summary's pointer if set, else best
    composite among non-singleton agg rows."""
    cs_row = cs_lookup.get(seq_dir.name)
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
    _VERDICTS_WITH_REV = {"pose_holds", "pose_collapses", "new_contamination"}

    def _comp(r: Dict) -> float:
        verdict = (r.get("outcome") or "").strip()
        if verdict in _VERDICTS_WITH_REV:
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


# ── Data loaders ────────────────────────────────────────────────────────
def load_cohort_seeds(
    runs_dir: Path,
    cross_csv: Path,
) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
    """Returns:
      rep_seeds: {seq_name: [{seed, stage, outcome, ra_final, ...}, ...]}
      all_seeds: same shape but with seeds across all sgs
      composites: {seq_name: composite}
      tiers: {seq_name: tier}
      cold_baseline: {seq_name: float}  ra_eff of the cycle-0 initial
                                         prediction (cold-start, single
                                         prediction shared across sgs)
    """
    cs_lookup: Dict[str, Dict] = {}
    with open(cross_csv) as f:
        for r in csv.DictReader(f):
            cs_lookup[r["mpnn_sequence"]] = r

    rep_seeds: Dict[str, List[Dict]] = {}
    all_seeds: Dict[str, List[Dict]] = {}
    composites: Dict[str, float] = {}
    tiers: Dict[str, str] = {}
    cold_baseline: Dict[str, float] = {}

    for seq_dir_name in sorted(os.listdir(runs_dir)):
        if seq_dir_name.startswith("input_control"):
            continue
        seq_dir = runs_dir / seq_dir_name
        rep_sg = _rep_sg(seq_dir, cs_lookup)
        raw = seq_dir / "raw_per_seed_results.csv"
        if not raw.exists():
            continue
        with open(raw) as f:
            raw_rows = list(csv.DictReader(f))
        rep_list: List[Dict] = []
        all_list: List[Dict] = []
        for row in raw_rows:
            sg = (row.get("sequence_group") or "").strip()
            if sg == "":
                # Cycle-0 initial baseline row — capture as cold-start
                # ra_eff for the trajectory plot, then skip (it has no
                # seed_index / sg).
                v = _try_float(row.get("steered_ra_eff_vs_truth"))
                if v is not None:
                    cold_baseline[seq_dir_name] = v
                continue
            stage, outcome = classify_seed(row)
            # Pick the right column family for FINAL metrics.
            if stage == "reversion":
                ra = _try_float(row.get("reverted_ra_eff_vs_truth"))
                tj = _try_float(row.get("reverted_true_jaccard"))
                wj = _try_float(row.get("reverted_weighted_jaccard"))
                conf = {
                    "complex_plddt":   _try_float(row.get("reverted_complex_plddt")),
                    "avg_plddt":       _try_float(row.get("reverted_avg_plddt")),
                    "interface_plddt": _try_float(row.get("reverted_interface_plddt")),
                    "iptm":            _try_float(row.get("reverted_iptm")),
                    "ptm":             _try_float(row.get("reverted_ptm")),
                    "actifptm":        _try_float(row.get("reverted_actifptm")),
                    "ipae":            _try_float(row.get("reverted_ipae")),
                    "pae_pass_frac":   _try_float(row.get("reverted_pae_pass_frac")),
                    "pae_mean":        _try_float(row.get("reverted_pae_mean")),
                    "ipsae_min":       _try_float(row.get("reverted_ipsae_min")),
                    "ipsae_ab":        _try_float(row.get("reverted_ipsae_ab")),
                    "ipsae_ba":        _try_float(row.get("reverted_ipsae_ba")),
                    "ipsae_min_15":    _try_float(row.get("reverted_ipsae_min_15")),
                    "ipsae_ab_15":     _try_float(row.get("reverted_ipsae_ab_15")),
                    "ipsae_ba_15":     _try_float(row.get("reverted_ipsae_ba_15")),
                }
            else:
                ra = _try_float(row.get("steered_ra_eff_vs_truth"))
                tj = _try_float(row.get("steered_true_jaccard"))
                wj = _try_float(row.get("steered_weighted_jaccard"))
                conf = {
                    "complex_plddt":   _try_float(row.get("steered_complex_plddt")),
                    "avg_plddt":       _try_float(row.get("steered_avg_plddt")),
                    "interface_plddt": _try_float(row.get("steered_interface_plddt")),
                    "iptm":            _try_float(row.get("steered_iptm")),
                    "ptm":             _try_float(row.get("steered_ptm")),
                    "actifptm":        _try_float(row.get("steered_actifptm")),
                    "ipae":            _try_float(row.get("steered_ipae")),
                    "pae_pass_frac":   _try_float(row.get("steered_pae_pass_frac")),
                    "pae_mean":        _try_float(row.get("steered_pae_mean")),
                    "ipsae_min":       _try_float(row.get("steered_ipsae_min")),
                    "ipsae_ab":        _try_float(row.get("steered_ipsae_ab")),
                    "ipsae_ba":        _try_float(row.get("steered_ipsae_ba")),
                    "ipsae_min_15":    _try_float(row.get("steered_ipsae_min_15")),
                    "ipsae_ab_15":     _try_float(row.get("steered_ipsae_ab_15")),
                    "ipsae_ba_15":     _try_float(row.get("steered_ipsae_ba_15")),
                }

            try:
                seed_idx = int(row.get("seed_index", "0"))
            except (TypeError, ValueError):
                seed_idx = 0

            # Also capture per-stage values for trajectory plot.
            cs_ra = _try_float(row.get("steered_ra_eff_vs_truth")) \
                    if stage == "cold_start" else None
            st_ra = _try_float(row.get("steered_ra_eff_vs_truth")) \
                    if stage in ("steering", "reversion") else None
            rev_ra = _try_float(row.get("reverted_ra_eff_vs_truth")) \
                     if stage == "reversion" else None

            rec = {
                "seed": seed_idx, "sg": sg,
                "stage": stage, "outcome": outcome,
                "ra_final": ra, "tj_final": tj, "wj_final": wj,
                # All confidence metrics expanded out of `conf`.
                **conf,
                # Trajectory data per row:
                "steered_ra_raw": _try_float(row.get("steered_ra_eff_vs_truth")),
                "reverted_ra_raw": _try_float(row.get("reverted_ra_eff_vs_truth")),
                "n_mut": _try_int(row.get("steered_total_mutations")),
            }
            all_list.append(rec)
            if sg == rep_sg:
                rep_list.append(rec)
        if rep_list:
            rep_seeds[seq_dir_name] = rep_list
        if all_list:
            all_seeds[seq_dir_name] = all_list
            cs = cs_lookup.get(seq_dir_name)
            tiers[seq_dir_name] = (cs.get("cross_tier") if cs else "none") or "none"
            # Composite for sorting (verdict-aware via cross_summary).
            comp = float("-inf")
            if cs is not None:
                ra = _try_float(cs.get("rep_ra_eff_vs_truth_median"))
                tj = _try_float(cs.get("rep_true_jaccard_median"))
                if ra is not None and tj is not None:
                    comp = tj - 0.05 * ra
            if comp == float("-inf"):
                # Fallback: best agg composite.
                agg_path = seq_dir / "aggregated_results.csv"
                if agg_path.exists():
                    with open(agg_path) as f:
                        agg_rows = list(csv.DictReader(f))
                    best = float("-inf")
                    _VR = {"pose_holds", "pose_collapses", "new_contamination"}
                    for r in agg_rows:
                        v = (r.get("outcome") or "").strip()
                        if v == "singleton":
                            continue
                        if v in _VR:
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
                            if c > best:
                                best = c
                    if best != float("-inf"):
                        comp = best
            composites[seq_dir_name] = comp
    return rep_seeds, all_seeds, composites, tiers, cold_baseline


# ── Plot 1a: per-seed dispersion overview ───────────────────────────────
def plot_dispersion_overview(rep_seeds, tiers, out_path):
    """Single scatter, one point per (sequence, rep-seed): x = ra_eff
    (final), y = true_jaccard (final).  Shape = stage, colour = tier."""
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    seen_tiers = set()
    seen_stages = set()
    for name, seeds in rep_seeds.items():
        tier = tiers.get(name, "none")
        colour = COLOUR_TIER.get(tier, "#B0B0B0")
        for s in seeds:
            ra, tj, stage = s["ra_final"], s["tj_final"], s["stage"]
            if ra is None or tj is None:
                continue
            marker = STAGE_MARKER.get(stage, "x")
            ax.scatter(ra, tj, marker=marker, s=80, c=colour,
                       edgecolor="black", linewidth=0.5, alpha=0.85,
                       zorder=3)
            seen_tiers.add(tier); seen_stages.add(stage)

    ax.set_xlabel("Receptor Aligned Effector RMSD (Å, final)", fontsize=11)
    ax.set_ylabel("Interface Jaccard (final)", fontsize=11)
    ax.set_title("Per-seed dispersion — representative sg, final prediction",
                 fontsize=12, pad=10)
    ax.grid(True, linestyle="-", linewidth=0.3, alpha=0.4, color="grey")
    ax.set_axisbelow(True)
    ax.set_ylim(-0.05, 1.05)

    # Legend: tiers + stages.
    tier_handles = [mpatches.Patch(color=COLOUR_TIER[t], label=f"Tier {t}")
                    for t in ["A", "B", "C", "none"] if t in seen_tiers]
    stage_handles = [Line2D([0], [0], marker=STAGE_MARKER[s], color="w",
                            markerfacecolor="#888", markeredgecolor="black",
                            markersize=10, linewidth=0, label=STAGE_LABEL[s])
                     for s in STAGE_ORDER if s in seen_stages]
    leg1 = ax.legend(handles=tier_handles, loc="upper right",
                     bbox_to_anchor=(1.0, 1.0), fontsize=9, title="Tier")
    ax.add_artist(leg1)
    ax.legend(handles=stage_handles, loc="upper right",
              bbox_to_anchor=(1.0, 0.78), fontsize=9, title="Stage (final)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Plot 1b: per-seed dispersion grid ───────────────────────────────────
def plot_dispersion_grid(rep_seeds, all_seeds, tiers, composites, out_path):
    """Small-multiples: one panel per MPNN sequence, scatter of ALL
    its seeds (across all sgs) — ra_eff (final) vs true_jaccard
    (final).  Non-representative seeds are drawn in grey; the
    representative sg's seeds are coloured green (pass) or red (fail)
    based on whether ra_eff < 5 Å.  Vertical line at ra_eff = 5.
    Sorted by composite descending."""
    names = sorted(rep_seeds.keys(),
                   key=lambda n: -composites.get(n, float("-inf")))
    n = len(names)
    nr, nc = _grid_shape(n)
    fig, axes = plt.subplots(nr, nc, figsize=(2.8 * nc, 2.6 * nr),
                             sharex=True, sharey=True)
    axes_flat = np.array(axes).reshape(-1) if n > 1 else np.array([axes])

    # Build a set of (seq_name, sg, seed_index) for representative seeds
    # so we can identify them when iterating ALL seeds.
    rep_set = set()
    for sn, seeds in rep_seeds.items():
        for s in seeds:
            rep_set.add((sn, s["sg"], s["seed"]))

    # Axis bounds across cohort (use all seeds, not just rep).
    all_ra, all_tj = [], []
    for seeds in all_seeds.values():
        for s in seeds:
            if s["ra_final"] is not None: all_ra.append(s["ra_final"])
            if s["tj_final"] is not None: all_tj.append(s["tj_final"])
    if not all_ra:
        return
    x_max = max(all_ra) * 1.05
    y_min = min(0.0, min(all_tj) - 0.05) if all_tj else 0.0
    y_max = max(1.0, max(all_tj) * 1.05) if all_tj else 1.0

    GREY     = "#B0B0B0"
    GREEN    = "#2CA02C"
    RED      = "#D62728"

    for i, ax in enumerate(axes_flat):
        if i >= n:
            ax.set_axis_off(); continue
        name = names[i]
        seeds = all_seeds.get(name, [])
        tier = tiers.get(name, "none")
        for spine in ax.spines.values():
            spine.set_edgecolor(COLOUR_TIER.get(tier, "#888"))
            spine.set_linewidth(1.6 if tier in ("A", "B", "C") else 1.0)

        # Plot non-rep seeds in grey first (background).
        for s in seeds:
            ra, tj, stage = s["ra_final"], s["tj_final"], s["stage"]
            if ra is None or tj is None:
                continue
            is_rep = (name, s["sg"], s["seed"]) in rep_set
            if is_rep:
                continue
            marker = STAGE_MARKER.get(stage, "x")
            ax.scatter(ra, tj, marker=marker, s=42, c=GREY,
                       edgecolor="#666", linewidth=0.4, alpha=0.6,
                       zorder=2)

        # Then plot rep seeds in green (pass) / red (fail).  Pass =
        # ra_eff < 5 AND structure intact (we treat any rep-sg seed
        # whose ra_final < 5 as the pass).
        for s in seeds:
            ra, tj, stage = s["ra_final"], s["tj_final"], s["stage"]
            if ra is None or tj is None:
                continue
            if (name, s["sg"], s["seed"]) not in rep_set:
                continue
            marker = STAGE_MARKER.get(stage, "x")
            colour = GREEN if ra < RA_EFF_MAX else RED
            ax.scatter(ra, tj, marker=marker, s=78, c=colour,
                       edgecolor="black", linewidth=0.6, alpha=0.95,
                       zorder=3)

        # Vertical threshold line at ra_eff = 5.
        ax.axvline(RA_EFF_MAX, color="grey", linestyle="--",
                   linewidth=0.7, alpha=0.55, zorder=1)

        ax.set_xlim(0, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle="-", linewidth=0.25, alpha=0.4, color="grey")
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=7)
        ax.set_title(_short_name(name), fontsize=8)

    # Legend below the figure.
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY,
               markeredgecolor="#666", markersize=7, linewidth=0,
               label="Other sg seed"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GREEN,
               markeredgecolor="black", markersize=8, linewidth=0,
               label="Representative seed — pass (ra_eff < 5 Å)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=RED,
               markeredgecolor="black", markersize=8, linewidth=0,
               label="Representative seed — fail (ra_eff ≥ 5 Å)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
               markeredgecolor="black", markersize=8, linewidth=0,
               label="Cold-start"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#888",
               markeredgecolor="black", markersize=8, linewidth=0,
               label="Steering"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#888",
               markeredgecolor="black", markersize=8, linewidth=0,
               label="Reversion"),
        Line2D([0], [0], color="grey", linestyle="--", linewidth=0.8,
               label="ra_eff = 5 Å threshold"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=8,
               framealpha=0.95)

    fig.text(0.5, 0.06, "ra_eff (Å, final)", ha="center", fontsize=10)
    fig.text(0.02, 0.5, "true_jaccard (final)",
             rotation="vertical", ha="center", fontsize=10)
    plt.tight_layout(rect=(0.04, 0.10, 1.0, 0.97))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Plot 2: per-design stage trajectories ───────────────────────────────
def plot_stage_trajectories(all_seeds, tiers, cold_baseline, out_path):
    """Panel per RFDiffusion design.  X-axis: stage (cold-start →
    steered → reverted).  One line per (MPNN sequence, sg, seed).
    Cold-start point is shared across all seeds of a sequence (it's
    a single pre-steering prediction); steering and reversion points
    are per-seed.  Marker on the FINAL stage.  Lines coloured by MPNN
    sequence number (so seq 3 is the same colour wherever it appears
    across designs)."""
    by_design: Dict[str, List[str]] = defaultdict(list)
    for name in all_seeds:
        by_design[_design_id(name)].append(name)

    designs = sorted(by_design.keys(),
                     key=lambda d: int(d[1:]) if d[1:].isdigit() else 0)
    n = len(designs)
    nr, nc = _grid_shape(n)
    fig, axes = plt.subplots(nr, nc, figsize=(3.2 * nc, 3.0 * nr),
                             sharex=True, sharey=True)
    axes_flat = np.array(axes).reshape(-1) if n > 1 else np.array([axes])

    STAGES = ["cold_start", "steering", "reversion"]
    stage_x = {s: i for i, s in enumerate(STAGES)}
    stage_labels = ["Cold-start", "Steering", "Reversion"]

    # Colour per MPNN sequence NUMBER (s0, s1, …) so seq 3 is always
    # the same colour across designs.  tab10 has 10 distinguishable
    # colours; cycle for >10.
    cmap = plt.get_cmap("tab10")

    def _seq_colour(name: str):
        return cmap(_seq_num(name) % 10)

    # Y axis bounds.
    all_ra = list(cold_baseline.values())
    for seeds in all_seeds.values():
        for s in seeds:
            for k in ("steered_ra_raw", "reverted_ra_raw"):
                v = s.get(k)
                if v is not None: all_ra.append(v)
    if not all_ra:
        return
    y_max = max(all_ra) * 1.05

    for i, ax in enumerate(axes_flat):
        if i >= n:
            ax.set_axis_off(); continue
        design = designs[i]
        seq_names = sorted(by_design[design])

        for sn in seq_names:
            seeds = all_seeds[sn]
            colour = _seq_colour(sn)
            cold_ra = cold_baseline.get(sn)
            for s in seeds:
                pts = []
                if cold_ra is not None:
                    pts.append((stage_x["cold_start"], cold_ra))
                if s["stage"] == "cold_start":
                    pass
                elif s["steered_ra_raw"] is not None:
                    pts.append((stage_x["steering"], s["steered_ra_raw"]))
                if s["stage"] == "reversion" and s["reverted_ra_raw"] is not None:
                    pts.append((stage_x["reversion"], s["reverted_ra_raw"]))
                if len(pts) < 1:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, "-", color=colour, linewidth=1.0,
                        alpha=0.55, zorder=2)
                ax.scatter(xs[-1], ys[-1], marker="o", s=42,
                           c=[colour], edgecolor="black", linewidth=0.5,
                           zorder=3)

        ax.axhline(RA_EFF_MAX, color="grey", linestyle="--",
                   linewidth=0.8, alpha=0.5, zorder=1)
        ax.set_xticks(list(range(len(STAGES))))
        ax.set_xticklabels(stage_labels, fontsize=8)
        ax.set_xlim(-0.4, len(STAGES) - 0.6)
        ax.set_ylim(0, y_max)
        ax.grid(True, linestyle="-", linewidth=0.25, alpha=0.4, color="grey")
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=7)
        # "d0" → "Design 0"
        design_num = design[1:] if design.startswith("d") else design
        ax.set_title(f"Design {design_num}", fontsize=10)

        if len(seq_names) <= 6:
            handles = [
                Line2D([0], [0], color=_seq_colour(sn), marker="o",
                       markersize=6, linewidth=1.5,
                       label=_short_name(sn))
                for sn in seq_names
            ]
            ax.legend(handles=handles, loc="upper right", fontsize=7,
                      framealpha=0.85)

    fig.text(0.5, 0.02, "Stage", ha="center", fontsize=10)
    fig.text(0.02, 0.5, "ra_eff (Å)",
             rotation="vertical", ha="center", fontsize=10)
    fig.suptitle("Per-design stage trajectories — line per (sequence, "
                 "sg, seed); marker = final stage; dashed = ra_eff "
                 "threshold (5 Å)", fontsize=10, y=0.99)
    plt.tight_layout(rect=(0.04, 0.04, 1.0, 0.96))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Plot 3: weighted vs true jaccard ────────────────────────────────────
def plot_weighted_vs_true_jaccard(rep_seeds, tiers, out_path):
    """Single scatter, one point per (MPNN sequence, Boltz seed) of the
    representative sequence_group — typically 3 points per sequence.
    x = true jaccard, y = weighted jaccard.  y=x diagonal.

    Small random jitter applied to expose stacked points (different
    seeds with identical metric values otherwise plot on top of each
    other and look like fewer points)."""
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    seen_tiers = set()
    rng = np.random.default_rng(seed=42)   # deterministic jitter
    n_total = 0
    for name, seeds in rep_seeds.items():
        tier = tiers.get(name, "none")
        colour = COLOUR_TIER.get(tier, "#B0B0B0")
        for s in seeds:
            tj, wj = s["tj_final"], s["wj_final"]
            if tj is None or wj is None:
                continue
            jx = rng.uniform(-0.005, 0.005)
            jy = rng.uniform(-0.005, 0.005)
            ax.scatter(tj + jx, wj + jy, s=70, c=colour,
                       edgecolor="black", linewidth=0.5, alpha=0.85,
                       zorder=3)
            seen_tiers.add(tier); n_total += 1
    if n_total == 0:
        ax.text(0.5, 0.5, "No weighted_jaccard data available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="grey")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--",
            linewidth=0.8, alpha=0.7, zorder=1)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("True Jaccard (final prediction)", fontsize=11)
    ax.set_ylabel("Weighted Jaccard (final prediction)", fontsize=11)
    ax.set_title(f"Weighted vs True Jaccard\n"
                 f"one point per Boltz seed of each MPNN sequence's "
                 f"representative design ({n_total} points)",
                 fontsize=11, pad=10)
    ax.grid(True, linestyle="-", linewidth=0.3, alpha=0.4, color="grey")
    ax.set_axisbelow(True)
    handles = [mpatches.Patch(color=COLOUR_TIER[t], label=f"Tier {t}")
               for t in ["A", "B", "C", "none"] if t in seen_tiers]
    handles.append(Line2D([0], [0], color="grey", linestyle="--",
                          linewidth=0.8, label="y = x"))
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Plot 4: composite vs confidence metrics ─────────────────────────────
def plot_composite_vs_confidence(rep_seeds, composites, tiers, out_path):
    """Multi-panel scatter: composite (y) vs each confidence metric (x).
    One point per MPNN sequence (composite is per-sequence; confidence
    metric is the median across that sequence's representative
    seeds).  Colour by tier.

    Each tuple: (key, label, threshold, comparison, x_lim).
    x_lim is None for "auto-scale", else a (min, max) tuple to force
    the natural range (e.g. metrics that are bounded [0, 1] should
    show that range so the reader can place each value in context)."""
    metrics = [
        # Bounded-in-[0,1] confidence metrics — fix the axis so we can
        # see how close each sequence is to the limits.
        # avg_plddt is on a 0-100 scale (complex_plddt is the same
        # metric / 100); we drop complex_plddt and keep avg_plddt with
        # a 0-100 axis to preserve the natural pLDDT scale.
        ("avg_plddt",       "average pLDDT (0-100)", None,            None, (0, 100)),
        ("interface_plddt", "interface pLDDT",       None,             None, (0, 1)),
        ("iptm",            "ipTM",                  IPTM_MIN,         ">=", (0, 1)),
        ("ptm",             "pTM",                   None,             None, (0, 1)),
        ("actifptm",        "actifPTM",              None,             None, (0, 1)),
        ("pae_pass_frac",   "pae_pass_frac",         PAE_PASS_FRAC_MIN, ">=", (0, 1)),
        # Only the _min variants of ipSAE — _ab and _ba just show
        # the per-direction asymmetric components and don't add
        # information to the symmetric min for the cohort plot.
        ("ipsae_min",       "ipSAE_min (10 Å)",      None,             None, (0, 1)),
        ("ipsae_min_15",    "ipSAE_min (15 Å)",      None,             None, (0, 1)),
        # Unbounded metrics — auto-scale.
        ("ipae",            "ipae",                  IPAE_MAX,         "<=", None),
        ("pae_mean",        "pae_mean",              None,             None, None),
    ]
    # Drop metrics where every sequence has no value.
    seq_data: Dict[str, Dict[str, float]] = {}
    for name, seeds in rep_seeds.items():
        comp = composites.get(name, float("-inf"))
        if comp == float("-inf"):
            continue
        rec: Dict[str, float] = {"composite": comp,
                                 "tier": tiers.get(name, "none")}
        for key, _label, _thresh, _cmp, _xlim in metrics:
            vals = [s.get(key) for s in seeds if s.get(key) is not None]
            if vals:
                rec[key] = float(np.median(vals))
        seq_data[name] = rec

    metrics_present = [m for m in metrics
                        if any(m[0] in rec for rec in seq_data.values())]
    n = len(metrics_present)
    nr, nc = _grid_shape(n)
    fig, axes = plt.subplots(nr, nc, figsize=(3.4 * nc, 3.0 * nr))
    axes_flat = np.array(axes).reshape(-1)

    seen_tiers = set()
    for i, ax in enumerate(axes_flat):
        if i >= n:
            ax.set_axis_off(); continue
        key, label, thresh, _cmp, xlim = metrics_present[i]
        xs, ys, cols = [], [], []
        for name, rec in seq_data.items():
            if key not in rec:
                continue
            xs.append(rec[key]); ys.append(rec["composite"])
            cols.append(COLOUR_TIER.get(rec["tier"], "#B0B0B0"))
            seen_tiers.add(rec["tier"])
        if not xs:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="grey")
            ax.set_title(label, fontsize=10); continue
        ax.scatter(xs, ys, s=60, c=cols, edgecolor="black",
                   linewidth=0.4, alpha=0.85, zorder=3)
        if thresh is not None:
            ax.axvline(thresh, color="grey", linestyle="--",
                       linewidth=0.7, alpha=0.6, zorder=1)
        if xlim is not None:
            ax.set_xlim(xlim[0] - 0.02, xlim[1] + 0.02)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Composite (tj − 0.05·ra_eff)", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.grid(True, linestyle="-", linewidth=0.25, alpha=0.4, color="grey")
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=7)

    handles = [mpatches.Patch(color=COLOUR_TIER[t], label=f"Tier {t}")
               for t in ["A", "B", "C", "none"] if t in seen_tiers]
    handles.append(Line2D([0], [0], color="grey", linestyle="--",
                          linewidth=0.8, label="threshold"))
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
               fontsize=9, framealpha=0.95)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)




# ═══════════════════════════════════════════════════════════════════════════
# Top-level orchestration
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--runs-dir", required=True,
        help="Path to the runs/ directory containing per-sequence "
             "subdirs, each with a raw_per_seed_results.csv.")
    ap.add_argument("--cross-summary-csv", required=True,
        help="Path to cross_sequence_summary.csv.  Used to identify "
             "the representative sequence_group per MPNN sequence and "
             "to compute composite scores for sorting.")
    ap.add_argument("--outdir", required=True,
        help="Directory to write plots into; created if missing.")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    runs_dir = Path(args.runs_dir)
    cross_csv = Path(args.cross_summary_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not runs_dir.exists():
        raise SystemExit(f"--runs-dir does not exist: {runs_dir}")
    if not cross_csv.exists():
        raise SystemExit(f"--cross-summary-csv does not exist: {cross_csv}")

    print(f"Walking runs dir: {runs_dir}")
    print(f"Cross-summary csv: {cross_csv}")
    rep_seeds, all_seeds, composites, tiers, cold_baseline = (
        load_cohort_seeds(runs_dir, cross_csv)
    )
    print(f"  sequences with rep-sg data: {len(rep_seeds)}")
    print(f"  total seeds across all sgs: "
          f"{sum(len(v) for v in all_seeds.values())}")
    print(f"  cold-start baselines: {len(cold_baseline)}")
    for sn in sorted(rep_seeds.keys()):
        seed_strs = []
        for s in rep_seeds[sn]:
            sg = s.get('sg')
            sd = s.get('seed')
            st = s.get('stage')
            ra = s.get('ra_final')
            tj = s.get('tj_final')
            seed_strs.append(f'sg={sg} seed={sd} stage={st} ra={ra} tj={tj}')
        print('DEBUG ' + sn + ': ' + ' | '.join(seed_strs))
    if not rep_seeds:
        print("Nothing to plot.")
        return 0

    plots = [
        ("Per-seed dispersion overview (single scatter)",
         "negsteer_per_seed_dispersion_overview.png",
         lambda p: (plot_dispersion_overview(rep_seeds, tiers, p) or True)),
        ("Per-seed dispersion grid (panel per MPNN sequence)",
         "negsteer_per_seed_dispersion_grid.png",
         lambda p: (plot_dispersion_grid(
             rep_seeds, all_seeds, tiers, composites, p) or True)),
        ("Per-design stage trajectories",
         "negsteer_per_design_stage_trajectories.png",
         lambda p: (plot_stage_trajectories(
             all_seeds, tiers, cold_baseline, p) or True)),
        ("Weighted vs True Jaccard",
         "negsteer_weighted_vs_true_jaccard.png",
         lambda p: (plot_weighted_vs_true_jaccard(
             rep_seeds, tiers, p) or True)),
        ("Composite vs confidence metrics (multi-panel)",
         "negsteer_composite_vs_confidence.png",
         lambda p: (plot_composite_vs_confidence(
             rep_seeds, composites, tiers, p) or True)),
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