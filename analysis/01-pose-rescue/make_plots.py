#!/usr/bin/env python3
"""
Negative-steering benchmark plots.

Scans experiments/benchmarking/<TARGET>/, restricted to the 18 HMA decoy + AVR
targets that are also covered by structure-prediction-benchmarking's Tier-1
comparison, and produces seven plots plus a summary CSV. By default it reads the
repo-local benchmarking tree (populated by ``scripts/sync_from_hpc.sh --name
benchmarking --results-only``) and writes its outputs alongside this script.

Per-target data sources
-----------------------
* run/cross_sequence_summary.csv   -> the single REPRESENTATIVE steered design.
                                      rep_ra_eff_vs_truth_median is the steered
                                      ("representative prediction") ra_eff.
* run/cycle_0/plan.json            -> initial_receptor_aligned_effector_rmsd is
                                      the COLD-START (initial prediction) ra_eff.
* inputs/receptor.fasta + effector.fasta -> combined sequence length.

A run is treated as FINISHED-with-representative when its cross_sequence_summary.csv
exists and has a data row. A run that reached harvest but had zero passing designs
(no cross_sequence_summary.csv, log ended at "STAGE 6") is FINISHED but contributes
a blank row to the summary CSV and is excluded from the plots (it has no
representative ra_eff). Runs still predicting are marked unfinished.

ra_eff = receptor_aligned_effector_rmsd vs truth (Angstrom). Lower is better; the
5 A threshold marks a "rescued" (correct) pose.

Plots
-----
1. Stacked histogram of representative ra_eff across finished runs, vertical
   line at x=5.
2. Histogram of change in ra_eff (representative - cold start). 0 = no change;
   positive (worse) bars red, negative (better) bars green.
3. Scatter of combined sequence length vs representative ra_eff.
4. Scatter of representative ra_eff vs its change from cold start; y=0
   separates improved/worsened, x=5 marks the pass threshold.
5. Boxplot of representative ra_eff across all finished runs.
6. Scatter of initial (cold start) vs representative ra_eff, one series for
   negative steering alone (blue) and one for negative steering plus reference
   restraints (red), each against its own cold start. The y=x diagonal marks no
   change (below = improved), y=5 marks the pass threshold.
7. 2x2 transition matrix of cold start -> steered correctness: how many poses
   negative steering rescued / held / missed / broke. Uses the pipeline's own
   cold start as the before state (a true before->after), so "lost" is not
   confounded by run-to-run variance.

Outputs
-------
   benchmark_summary.csv                                       (in --outdir)
   thesis-figures/cold_start_vs_steered_ra_eff.png
   supplementary-figures/steered_ra_eff_distribution.png
   supplementary-figures/ra_eff_change_distribution.png
   supplementary-figures/sequence_length_vs_ra_eff.png
   supplementary-figures/steered_ra_eff_vs_change.png
   supplementary-figures/ra_eff_by_seed_pass_count.png
   supplementary-figures/rescue_outcome_matrix.png   (if baseline available)

thesis-figures/ also holds model_choice_comparison.png, written by
02-model-choice.qmd rather than by this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Repo root: analysis/01-pose-rescue/make_plots.py -> parents[2].
REPO_ROOT = Path(__file__).resolve().parents[2]

# ── Scope: the 18 HMA decoy + AVR targets ─────────────────────────────────────
# The same 18 targets structure-prediction-benchmarking's Tier 1 covers, so the
# two repos' comparisons line up on identical target sets. Everything else under
# experiments/benchmarking/ (larger NLR-sensor, host-target and resistosome
# complexes) is out of scope for this analysis and is skipped.
TARGETS = ["5A6W", "6FU9", "6FUB", "6FUD", "6G10", "6G11", "6Q76", "6R8K", "6R8M",
           "7A8W", "7A8X", "7BNT", "7QPX", "7QZD", "5ZNG", "9IP6", "9IMU", "8B2R"]

RA_EFF_THRESHOLD = 5.0  # Angstrom; vertical reference line / pass cutoff

DEFAULT_BENCH = REPO_ROOT / "experiments" / "benchmarking"


# ── Representative-design selection ─────────────────────────────────────────
# Ranks a target's steered designs by (how many of its seeds cleanly pass) then
# (median ra_eff of those seeds), rather than reading cross_sequence_summary.csv's
# rep_ra_eff_vs_truth_median directly. The two mostly agree, but this is the more
# defensible ranking: it prioritises reproducibility (a design where all/most
# seeds land the same correct pose) over a design that merely has a low median
# by chance. Critically it computes that median on each seed's REVERSION-
# CORRECTED final value (the reverted structure's ra_eff when a seed's mutations
# needed reverting, not its raw pre-reversion prediction), matching what the
# engine's own per-seed verdict logic actually scores.


def _try_float(v) -> float | None:
    try:
        f = float(v)
        return None if (f != f) else f   # filter NaN
    except (TypeError, ValueError):
        return None


def _try_int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None


def _final_seed_value(row) -> tuple[float | None, bool]:
    """One raw_per_seed_results.csv row -> (final ra_eff, did it cleanly pass).

    Mirrors boltz2_iterate_steering._per_seed_verdict_breakdown: a seed whose
    steered mutations landed on/near a protected position goes through
    reversion, and its real final value is the REVERTED structure's ra_eff,
    not the raw steered prediction. A seed with zero contamination on its
    mutated positions skips reversion entirely (correctly) and the raw
    steered value stands.
    """
    verdict = str(row.get("reversion_verdict") or "").strip()
    if verdict == "pose_holds":
        ra = _try_float(row.get("reverted_ra_eff_vs_truth"))
        return (ra, ra is not None and ra < RA_EFF_THRESHOLD)
    if verdict in ("pose_collapses", "new_contamination",
                   "pose_holds_not_intact", "missing_verdict", "unknown"):
        return (_try_float(row.get("reverted_ra_eff_vs_truth")), False)
    # Blank verdict: either reversion was correctly skipped (zero
    # contamination on the mutated positions) or the row is missing data.
    n_contacts = _try_int(row.get("steered_n_contacts_on_mutated_positions"))
    intact = _try_int(row.get("steered_receptor_intact"))
    ra = _try_float(row.get("steered_ra_eff_vs_truth"))
    if n_contacts == 0 and ra is not None:
        return (ra, intact == 1 and ra < RA_EFF_THRESHOLD)
    return (None, False)


def _design_group(n_pass: int, n_seeds: int) -> tuple[int, str]:
    """(sort_rank, label) for a design's seed pass-count, best first.

    Rank order is all seeds pass, then most, then one, then none. The
    reliability-first ordering ("prioritise 3/3 over 2/3, but 2/3 over
    nothing") rather than ranking purely on median ra_eff, which can let an
    unreliable design with a lucky low median outrank a fully-reproducible one.
    """
    if n_seeds <= 0:
        return (3, "no design available")
    if n_pass == n_seeds:
        rank = 0
    elif n_pass == 0:
        rank = 3
    else:
        rank = 1 if n_pass > 1 else 2
    return (rank, f"{n_pass}/{n_seeds} seeds passing")


def select_representative(rundir: Path) -> dict | None:
    """Pick the representative design for one target's run.

    Ranks candidate designs by (seed pass-count group, median ra_eff of that
    design's seeds ascending) and returns the best, using each seed's
    reversion-corrected final value (see _final_seed_value), not the raw
    steered prediction. This is applied uniformly, including to the
    cold-start-all-clean case (steering skipped because the cold start
    itself already passed on every seed): there the "design" is the cold
    start's own multi-seed set from plan.json.

    Returns None if there is no per-seed data to select from at all.
    """
    plan_path = rundir / "cycle_0" / "plan.json"
    per_seed_path = rundir / "raw_per_seed_results.csv"

    cold_start_seeds = []
    if plan_path.is_file():
        try:
            cold_start_seeds = json.loads(plan_path.read_text()).get(
                "cold_start_seeds") or []
        except Exception:
            cold_start_seeds = []

    candidates = []   # (rank, median, n_pass, n_seeds, design_label)

    if per_seed_path.is_file():
        try:
            df = pd.read_csv(per_seed_path)
        except Exception:
            df = None
        if df is not None and len(df):
            steered = df[(df.get("status") == "ok") & (df.get("design") != "initial")]
            if len(steered):
                steered = steered.copy()
                steered["base"] = steered["design"].str.replace(
                    r"_s\d+$", "", regex=True)
                for base, grp in steered.groupby("base"):
                    finals, n_pass = [], 0
                    for _, row in grp.iterrows():
                        ra, ok = _final_seed_value(row)
                        if ra is not None:
                            finals.append(ra)
                        if ok:
                            n_pass += 1
                    if not finals:
                        continue
                    rank, _ = _design_group(n_pass, len(grp))
                    candidates.append(
                        (rank, float(np.median(finals)), n_pass, len(grp), base))

    if not candidates and cold_start_seeds:
        finals = [_try_float(s.get("ra_eff")) for s in cold_start_seeds]
        finals = [f for f in finals if f is not None]
        if finals:
            n_pass = sum(1 for f in finals if f < RA_EFF_THRESHOLD)
            rank, _ = _design_group(n_pass, len(finals))
            candidates.append(
                (rank, float(np.median(finals)), n_pass, len(finals), "initial"))

    if not candidates:
        return None

    candidates.sort(key=lambda c: (c[0], c[1]))
    rank, median, n_pass, n_seeds, label = candidates[0]
    return {
        "rep_ra_eff": median,
        "rep_n_pass": n_pass,
        "rep_n_seeds": n_seeds,
        "rep_design": label,
    }


def fasta_len(path: Path) -> int | None:
    """Residue count of the (single-record) FASTA, or None if unreadable."""
    if not path.is_file():
        return None
    seq = "".join(
        line.strip()
        for line in path.read_text().splitlines()
        if line and not line.startswith(">")
    )
    return len(seq) or None


def classify(rundir: Path, css: Path) -> str:
    """representative | none_passed | unfinished."""
    if css.is_file():
        try:
            if len(pd.read_csv(css)) > 0:
                return "representative"
        except Exception:
            pass
    # No representative. Did the run reach harvest (finished, nothing passed)?
    logs = sorted(rundir.parent.glob("negsteer_*.out"))
    for log in logs:
        try:
            txt = log.read_text(errors="replace")
        except Exception:
            continue
        if "STAGE 6: harvest" in txt:
            return "none_passed"
    return "unfinished"


def constrained_arm(bench: Path, target: str):
    """Cold-start and representative ra_eff for <target>_constrained.

    That folder is the same config with negsteer_boltz_constraints: true, so
    every Boltz-2 call in the run, the cold start included, also carries the
    benchmark-style pocket + contact restraints derived from the ground truth.
    Returns (None, None) when the run has not landed yet.
    """
    rundir = bench / "constrained" / target / "run"
    if not rundir.is_dir():
        return None, None

    cold = None
    plan_path = rundir / "cycle_0" / "plan.json"
    if plan_path.is_file():
        try:
            cold = json.load(open(plan_path)).get(
                "initial_receptor_aligned_effector_rmsd"
            )
        except Exception:
            cold = None

    sel = select_representative(rundir)
    return cold, (sel["rep_ra_eff"] if sel else None)


def collect(bench: Path) -> pd.DataFrame:
    rows = []
    unconstrained = bench / "unconstrained"
    for target in sorted(p.name for p in unconstrained.iterdir()
                         if p.is_dir() and p.name in TARGETS):
        tdir = bench / "unconstrained" / target
        rundir = tdir / "run"
        css = rundir / "cross_sequence_summary.csv"
        plan_path = rundir / "cycle_0" / "plan.json"

        cold = None
        if plan_path.is_file():
            try:
                cold = json.load(open(plan_path)).get(
                    "initial_receptor_aligned_effector_rmsd"
                )
            except Exception:
                cold = None

        comb_len = None
        rlen = fasta_len(tdir / "inputs" / "receptor.fasta")
        elen = fasta_len(tdir / "inputs" / "effector.fasta")
        if rlen is not None and elen is not None:
            comb_len = rlen + elen

        status = classify(rundir, css)

        # Reference-restrained + negative-steered variant for the same target, if
        # its run has landed.  Its cold start is ALSO restrained (the engine's
        # --boltz-constraints applies to every Boltz-2 call, initial included),
        # so it is kept as its own before-value rather than being paired with
        # the unrestrained cold start above.
        con_cold, con_rep = constrained_arm(bench, target)

        # Full cross_sequence_summary.csv row, kept for its other columns
        # (jaccard, confidence metrics, mutation lists, ...), but its own
        # rep_ra_eff / rep_n_pass / rep_n_seeds are superseded below by
        # select_representative's reliability-first, reversion-corrected
        # selection (see that function's docstring).
        rep = {}
        sel = None
        if status == "representative":
            r = pd.read_csv(css).iloc[0]
            rep = r.to_dict()
            sel = select_representative(rundir)

        rep_ra_eff = sel["rep_ra_eff"] if sel else None
        rep_n_pass = sel["rep_n_pass"] if sel else None
        rep_n_seeds = sel["rep_n_seeds"] if sel else None

        change = (
            rep_ra_eff - cold
            if (rep_ra_eff is not None and cold is not None)
            else None
        )

        # Lead with our derived columns, then the full representative row
        # (whose own rep_ra_eff/rep_n_pass/rep_n_seeds we then override).
        row = {
            "target": target,
            "status": status,
            "receptor_len": rlen,
            "effector_len": elen,
            "combined_seq_len": comb_len,
            "cold_start_ra_eff": cold,
            "change_ra_eff": change,
            "constrained_cold_start_ra_eff": con_cold,
            "constrained_rep_ra_eff": con_rep,
        }
        row.update(rep)
        row["rep_ra_eff"] = rep_ra_eff
        row["rep_n_pass"] = rep_n_pass
        row["rep_n_seeds"] = rep_n_seeds
        rows.append(row)
    return pd.DataFrame(rows)


# Legends sit beside the axes by default. A long legend goes above the axes
# instead, so it never squeezes the plotting area.
LEGEND_KW = dict(loc="center left", bbox_to_anchor=(1.01, 0.5),
                 borderaxespad=0.0, frameon=False)


def place_legend(ax, handles=None, labels=None, **kw):
    """Beside the axes for a short legend, above it for a long one."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    elif labels is None:
        labels = [h.get_label() for h in handles]
    n = len(labels)
    longest = max((len(str(x)) for x in labels), default=0)
    if n > 3 or longest > 22:
        base = dict(loc="lower center", bbox_to_anchor=(0.5, 1.02),
                    borderaxespad=0.0, frameon=False, ncol=min(n, 2))
    else:
        base = dict(LEGEND_KW)
    base.update(kw)
    return ax.legend(handles, labels, **base)


# ── Colour by how many of the representative design's seeds passed ─────────
# Deliberately NOT labelled "tier". These are the same reliability groups the
# representative-selection ranking above uses (see _design_group), just
# surfaced as colour instead of only affecting which design got picked.
_PASS_CMAP = plt.get_cmap("RdYlGn")


def _pass_group_color(n_pass, n_seeds) -> str:
    if n_pass is None or not n_seeds:
        return "#999999"
    return matplotlib.colors.to_hex(_PASS_CMAP(n_pass / n_seeds))


def _pass_group_label(n_pass, n_seeds) -> str:
    if n_pass is None or not n_seeds:
        return "no design available"
    return f"{int(n_pass)}/{int(n_seeds)} seeds passing"


def _ordered_pass_groups(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Distinct (n_pass, n_seeds) pairs present in df, best group first."""
    sub = df.dropna(subset=["rep_n_pass", "rep_n_seeds"])
    pairs = {(int(p), int(s)) for p, s in
             zip(sub["rep_n_pass"], sub["rep_n_seeds"]) if s}
    return sorted(pairs, key=lambda ps: (-(ps[0] / ps[1]), -ps[0]))


def save_tight(fig, out: Path):
    """Save as PNG with whitespace cropped tight."""
    fig.savefig(out.with_suffix(".png"), dpi=200,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_ra_eff_hist(df: pd.DataFrame, out: Path):
    """Stacked histogram of representative ra_eff, segmented by how many of
    the representative design's seeds cleanly passed."""
    sub = df.dropna(subset=["rep_ra_eff"])
    vals = sub["rep_ra_eff"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.histogram_bin_edges(vals, bins=20) if len(vals) else 20
    groups = _ordered_pass_groups(sub)
    per_group = [
        sub.loc[(sub["rep_n_pass"] == p) & (sub["rep_n_seeds"] == s),
                "rep_ra_eff"].to_numpy()
        for p, s in groups
    ]
    ax.hist(per_group, bins=bins, stacked=True,
            color=[_pass_group_color(p, s) for p, s in groups],
            label=[f"{_pass_group_label(p, s)} (n = {len(v)})"
                   for (p, s), v in zip(groups, per_group)],
            edgecolor="white")
    ax.axvline(RA_EFF_THRESHOLD, color="black", linestyle="--", linewidth=1.5,
               label=f"x = {RA_EFF_THRESHOLD:g} Å (pass threshold)")
    ax.set_xlabel("Representative ra_eff vs truth (Å)")
    ax.set_ylabel("Number of finished runs")
    place_legend(ax)
    save_tight(fig, out)


def plot_change_hist(df: pd.DataFrame, out: Path,
                     no_change: float = 2.0, bin_width: float = 4.0):
    """Histogram of change in ra_eff.

    A central grey bin spans [-no_change, +no_change] and catches runs that
    stayed roughly the same. Every other bin is `bin_width` Å wide: bins fully
    below the no-change band are green (improved), fully above are red (worsened).

    The central "no change" bar is split (stacked) by whether the cold start was
    already correct (ra_eff <= RA_EFF_THRESHOLD): the bottom blue/hatched segment
    is runs the model already got right at cold start (so there was nothing to
    rescue); the top grey segment is runs that stayed incorrect.
    """
    sub = df.dropna(subset=["change_ra_eff", "cold_start_ra_eff"])
    vals = sub["change_ra_eff"].to_numpy()
    cold = sub["cold_start_ra_eff"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 5))

    # Build edges outward from the central [-no_change, +no_change] band.
    lo = float(np.min(vals)) if len(vals) else -no_change
    hi = float(np.max(vals)) if len(vals) else no_change
    left = [-no_change]
    while left[-1] > lo:
        left.append(left[-1] - bin_width)
    right = [no_change]
    while right[-1] < hi:
        right.append(right[-1] + bin_width)
    edges = np.array(sorted(set(left)) + sorted(set(right)))

    counts, edges = np.histogram(vals, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)

    for i, c in enumerate(centers):
        if -no_change <= c <= no_change:
            # Central no-change bin: split by cold-start correctness.
            in_bin = (vals >= edges[i]) & (vals < edges[i + 1])
            already_ok = int(np.sum(in_bin & (cold <= RA_EFF_THRESHOLD)))
            still_wrong = int(np.sum(in_bin & (cold > RA_EFF_THRESHOLD)))
            ax.bar(edges[i], already_ok, width=widths[i], align="edge",
                   color="#6baed6", edgecolor="white", hatch="//")
            ax.bar(edges[i], still_wrong, width=widths[i], align="edge",
                   bottom=already_ok, color="#999999", edgecolor="white")
        else:
            color = "#2ca02c" if c < 0 else "#d62728"  # green good, red bad
            ax.bar(edges[i], counts[i], width=widths[i], align="edge",
                   color=color, edgecolor="white")

    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_xlabel("Change in ra_eff: representative − cold start (Å)")
    ax.set_ylabel("Number of finished runs")
    # Legend proxies
    from matplotlib.patches import Patch
    place_legend(ax, handles=[
        Patch(facecolor="#2ca02c", label="improved (good)"),
        Patch(facecolor="#6baed6", hatch="//",
              label=f"no change, already correct at cold start "
                    f"(ra_eff ≤ {RA_EFF_THRESHOLD:g} Å)"),
        Patch(facecolor="#999999", label="no change, still incorrect"),
        Patch(facecolor="#d62728", label="worsened (bad)"),
    ])
    save_tight(fig, out)


def plot_length_scatter(df: pd.DataFrame, out: Path):
    sub = df.dropna(subset=["rep_ra_eff", "combined_seq_len"])
    fig, ax = plt.subplots(figsize=(8, 5))
    for p, s in _ordered_pass_groups(sub):
        g = sub[(sub["rep_n_pass"] == p) & (sub["rep_n_seeds"] == s)]
        ax.scatter(g["combined_seq_len"], g["rep_ra_eff"],
                   color=_pass_group_color(p, s), label=_pass_group_label(p, s),
                   s=55, edgecolor="black", linewidth=0.4, alpha=0.9)
    ax.axhline(RA_EFF_THRESHOLD, color="black", linestyle="--", linewidth=1.2,
               label=f"ra_eff = {RA_EFF_THRESHOLD:g} Å")
    ax.set_xlabel("Combined sequence length (receptor + effector, residues)")
    ax.set_ylabel("Representative ra_eff vs truth (Å)")
    place_legend(ax)
    save_tight(fig, out)


def plot_ra_eff_vs_change(df: pd.DataFrame, out: Path):
    """Scatter of representative ra_eff vs its change from cold start.

    x = representative ra_eff (lower better); y = change (negative = improved).
    The horizontal y=0 line separates improved (below) from worsened (above);
    the vertical x=5 Å line marks the pass threshold.
    """
    sub = df.dropna(subset=["rep_ra_eff", "change_ra_eff"])
    fig, ax = plt.subplots(figsize=(8, 5))
    for p, s in _ordered_pass_groups(sub):
        g = sub[(sub["rep_n_pass"] == p) & (sub["rep_n_seeds"] == s)]
        ax.scatter(g["rep_ra_eff"], g["change_ra_eff"],
                   color=_pass_group_color(p, s), label=_pass_group_label(p, s),
                   s=55, edgecolor="black", linewidth=0.4, alpha=0.9)
    ax.axhline(0, color="black", linewidth=1.2)
    ax.axvline(RA_EFF_THRESHOLD, color="black", linestyle="--", linewidth=1.2,
               label=f"ra_eff = {RA_EFF_THRESHOLD:g} Å")
    ax.set_xlabel("Representative ra_eff vs truth (Å)")
    ax.set_ylabel("Change in ra_eff: representative − cold start (Å)")
    place_legend(ax)
    save_tight(fig, out)


def plot_initial_vs_rep(df: pd.DataFrame, out: Path):
    """Scatter of cold-start (initial) ra_eff vs representative ra_eff, for
    negative steering alone and for negative steering plus reference restraints.

    x = that configuration's own cold start, y = its representative. Each
    series is paired with its OWN cold start because the restrained variant applies
    restraints to the initial prediction too, so the two configurations do not
    share a starting pose.

    The y = x diagonal marks no change: points BELOW it improved
    (representative < initial), points above worsened. The horizontal y = 5 Å
    dashed line marks the pass threshold for the representative prediction.
    """
    SERIES = [
        ("cold_start_ra_eff", "rep_ra_eff",
         "#0072B2", "negative-steered"),
        ("constrained_cold_start_ra_eff", "constrained_rep_ra_eff",
         "#CC3311", "reference-restrained + negative-steered"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    all_vals = []
    for xcol, ycol, colour, label in SERIES:
        if xcol not in df.columns or ycol not in df.columns:
            continue
        g = df.dropna(subset=[xcol, ycol])
        if not len(g):
            continue
        ax.scatter(g[xcol], g[ycol], color=colour,
                   label=f"{label} (n = {len(g)})",
                   s=55, edgecolor="black", linewidth=0.4, alpha=0.9)
        all_vals.append(g[xcol].to_numpy())
        all_vals.append(g[ycol].to_numpy())

    # y = x diagonal spanning the full data range (square the axes so it reads
    # as a true 45 degree line).
    both = np.concatenate(all_vals) if all_vals else np.array([0.0, 1.0])
    lo, hi = float(np.min(both)), float(np.max(both))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    lims = (lo - pad, hi + pad)
    ax.plot(lims, lims, color="grey", linewidth=1.2, zorder=0)
    ax.axhline(RA_EFF_THRESHOLD, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Initial ra_eff vs ground truth (Å)")
    ax.set_ylabel("Steered ra_eff vs ground truth (Å)")
    place_legend(ax)
    save_tight(fig, out)


def plot_steering_confusion(df: pd.DataFrame, out: Path,
                            baseline_col: str = "cold_start_ra_eff",
                            baseline_label: str = "Cold start (initial Boltz-2)",
                            scope_label: str | None = None):
    """2×2 transition matrix: the cold-start prediction vs negative steering.

    Each target is classified by whether its pose is correct (ra_eff < 5 Å)
    BEFORE steering (the cold-start initial prediction) and AFTER (the steered
    representative). Both come from the SAME pipeline run, so this is a true
    before→after and the four cells are unambiguous:
        rescued        incorrect → correct   (the win condition)
        both correct   correct   → correct   (steering held a good pose)
        both incorrect incorrect → incorrect (steering didn't reach the cutoff)
        lost           correct   → incorrect (steering broke a good pose)

    (Using the cold start rather than an independent predictor avoids attributing
    run-to-run prediction variance to steering. An external Boltz-2 run can pass
    a target whose cold-start draw failed, which is not steering "losing" a pose.)
    """
    sub = df.dropna(subset=["rep_ra_eff", baseline_col]).copy()
    n = len(sub)

    base_ok = sub[baseline_col] < RA_EFF_THRESHOLD
    steer_ok = sub["rep_ra_eff"] < RA_EFF_THRESHOLD
    rescued        = int((~base_ok & steer_ok).sum())
    both_correct   = int(( base_ok & steer_ok).sum())
    both_incorrect = int((~base_ok & ~steer_ok).sum())
    lost           = int(( base_ok & ~steer_ok).sum())

    # (row, col) -> (label, count, colour). row 0 = steered correct (top);
    # col 0 = baseline incorrect (left).
    cells = {
        (0, 0): ("Rescued\n(incorrect → correct)", rescued, "#2ca02c"),
        (0, 1): ("Both correct", both_correct, "#6baed6"),
        (1, 0): ("Both incorrect", both_incorrect, "#bdbdbd"),
        (1, 1): ("Lost\n(correct → incorrect)", lost, "#d62728"),
    }
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    for (r, c), (label, count, color) in cells.items():
        y = 1 - r
        ax.add_patch(plt.Rectangle((c, y), 1, 1, facecolor=color,
                                   alpha=0.85 if count else 0.20,
                                   edgecolor="white", linewidth=3))
        ax.text(c + 0.5, y + 0.60, str(count), ha="center", va="center",
                fontsize=30, fontweight="bold", color="black")
        pct = f"{100 * count / n:.0f}%" if n else ""
        ax.text(c + 0.5, y + 0.28, f"{label}\n{pct}", ha="center", va="center",
                fontsize=10.5, color="black")

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect("equal")
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Incorrect", "Correct"])
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Incorrect", "Correct"], rotation=90, va="center")
    ax.set_xlabel(f"{baseline_label} pose  (< {RA_EFF_THRESHOLD:g} Å = correct)")
    ax.set_ylabel(f"After negative steering  (< {RA_EFF_THRESHOLD:g} Å = correct)")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    save_tight(fig, out)
    return dict(rescued=rescued, both_correct=both_correct,
                both_incorrect=both_incorrect, lost=lost, n=n)


def plot_tier_boxplot(df: pd.DataFrame, out: Path):
    """Grouped boxplot of representative ra_eff, one box per pass-count group
    (how many of that design's seeds cleanly passed)."""
    sub = df.dropna(subset=["rep_ra_eff"])
    groups = _ordered_pass_groups(sub)
    data = [
        sub.loc[(sub["rep_n_pass"] == p) & (sub["rep_n_seeds"] == s),
                "rep_ra_eff"].to_numpy()
        for p, s in groups
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    # No mean marker (showmeans) and no outlier fliers (showfliers): every point
    # is already drawn via the jittered overlay below.
    bp = ax.boxplot(data, patch_artist=True, showmeans=False, showfliers=False,
                    medianprops=dict(color="black"))
    for patch, (p, s) in zip(bp["boxes"], groups):
        patch.set_facecolor(_pass_group_color(p, s))
        patch.set_alpha(0.7)
    # Overlay the individual points (jittered deterministically) for context.
    for i, ((p, s), vals) in enumerate(zip(groups, data), start=1):
        if not len(vals):
            continue
        jitter = np.linspace(-0.18, 0.18, len(vals)) if len(vals) > 1 else [0.0]
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=_pass_group_color(p, s), edgecolor="black", linewidth=0.4,
                   s=28, alpha=0.9, zorder=3)
    ax.axhline(RA_EFF_THRESHOLD, color="black", linestyle="--", linewidth=1.2,
               label=f"ra_eff = {RA_EFF_THRESHOLD:g} Å")
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels([f"{_pass_group_label(p, s)}\n(n = {len(v)})"
                        for (p, s), v in zip(groups, data)], fontsize=8)
    ax.set_ylabel("Representative ra_eff vs truth (Å)")
    place_legend(ax)
    save_tight(fig, out)


def main():
    ap = argparse.ArgumentParser(
        description="Generate the negative-steering benchmark plots + summary CSV."
    )
    ap.add_argument("--bench", type=Path, default=DEFAULT_BENCH,
                    help=f"benchmarking tree to scan (default: {DEFAULT_BENCH})")
    ap.add_argument("--outdir", type=Path,
                    default=Path(__file__).resolve().parent,
                    help="where to write benchmark_summary.csv and figures/")
    args = ap.parse_args()

    if not args.bench.is_dir():
        raise SystemExit(
            f"ERROR: benchmarking tree not found at {args.bench}\n"
            f"Populate it first with:\n"
            f"  scripts/sync_from_hpc.sh --name benchmarking --results-only"
        )

    # Two output folders: the plots that go into the thesis chapter, and
    # everything else.  model_choice_comparison.png is the other thesis figure
    # and is written into thesis-figures/ by 02-model-choice.qmd.
    thesisdir = args.outdir / "thesis-figures"
    suppdir = args.outdir / "supplementary-figures"
    thesisdir.mkdir(parents=True, exist_ok=True)
    suppdir.mkdir(parents=True, exist_ok=True)

    df = collect(args.bench)

    summary_path = args.outdir / "benchmark_summary.csv"
    df.to_csv(summary_path, index=False)

    plot_ra_eff_hist(df, suppdir / "steered_ra_eff_distribution.png")
    plot_change_hist(df, suppdir / "ra_eff_change_distribution.png")
    plot_length_scatter(df, suppdir / "sequence_length_vs_ra_eff.png")
    plot_ra_eff_vs_change(df, suppdir / "steered_ra_eff_vs_change.png")
    plot_tier_boxplot(df, suppdir / "ra_eff_by_seed_pass_count.png")
    plot_initial_vs_rep(df, thesisdir / "cold_start_vs_steered_ra_eff.png")

    # Rescue outcome matrix: cold-start -> steered. Baselines against the pipeline's OWN
    # cold start (not an external predictor), so it is a true before->after and
    # "lost" cannot be a run-to-run artefact.
    confusion = plot_steering_confusion(
        df, suppdir / "rescue_outcome_matrix.png")

    # ── Console report ───────────────────────────────────────────────────────
    n_rep = (df["status"] == "representative").sum()
    n_none = (df["status"] == "none_passed").sum()
    n_unf = (df["status"] == "unfinished").sum()
    print(f"Targets scanned:            {len(df)}")
    print(f"  with representative:      {n_rep}  (used in plots)")
    print(f"  finished, none passed:    {n_none}  (blank row, excluded from plots)")
    print(f"  unfinished:               {n_unf}")
    rep = df[df["status"] == "representative"]
    print(f"\nrep ra_eff: min={rep['rep_ra_eff'].min():.2f} "
          f"median={rep['rep_ra_eff'].median():.2f} "
          f"max={rep['rep_ra_eff'].max():.2f}")
    print(f"  <= {RA_EFF_THRESHOLD:g} A: "
          f"{(rep['rep_ra_eff'] <= RA_EFF_THRESHOLD).sum()} / {n_rep}")
    print(f"improved (change<0): {(rep['change_ra_eff'] < 0).sum()}, "
          f"worsened (change>0): {(rep['change_ra_eff'] > 0).sum()}")
    print(f"\nRescue vs cold start (n={confusion['n']}): "
          f"rescued={confusion['rescued']}, "
          f"both_correct={confusion['both_correct']}, "
          f"both_incorrect={confusion['both_incorrect']}, "
          f"lost={confusion['lost']}")
    written = [
        (thesisdir, ["cold_start_vs_steered_ra_eff"]),
        (suppdir, ["steered_ra_eff_distribution", "ra_eff_change_distribution",
                   "sequence_length_vs_ra_eff",
                   "steered_ra_eff_vs_change",
                   "ra_eff_by_seed_pass_count",
                   "rescue_outcome_matrix"]),
    ]
    print(f"\nWrote:\n  {summary_path}")
    for d, names in written:
        for p in names:
            print(f"  {d / p}.png")


if __name__ == "__main__":
    main()
