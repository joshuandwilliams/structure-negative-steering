#!/usr/bin/env python3
"""
Negative-steering benchmark plots.

Scans experiments/benchmarking/<TARGET>/ and produces six plots plus a summary
CSV. By default it reads the repo-local benchmarking tree (populated by
``scripts/sync_from_hpc.sh --name benchmarking --results-only``) and writes its
outputs alongside this script.

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
1. Stacked histogram of representative ra_eff across finished runs (segments by
   tier), vertical line at x=5.
2. Histogram of change in ra_eff (representative - cold start). 0 = no change;
   positive (worse) bars red, negative (better) bars green.
3. Scatter of combined sequence length vs representative ra_eff, coloured by tier.
4. Scatter of representative ra_eff vs its change from cold start, coloured by
   tier; y=0 separates improved/worsened, x=5 marks the pass threshold.
5. Grouped boxplot of representative ra_eff, one box per tier (1-4).
6. Scatter of initial (cold start) vs representative ra_eff, coloured by tier;
   y=x diagonal marks no change (below = improved), y=5 marks the pass threshold.
7. 2x2 transition matrix of cold start -> steered correctness, restricted to
   Tier 1 (HMA decoy + AVR) targets: how many poses negative steering rescued /
   held / missed / broke. Uses the pipeline's own cold start as the before state
   (a true before->after), so "lost" is not confounded by run-to-run variance.

Outputs
-------
   benchmark_summary.csv                         (in --outdir)
   figures/plot1_ra_eff_hist.png
   figures/plot2_ra_eff_change_hist.png
   figures/plot3_length_vs_ra_eff_scatter.png
   figures/plot4_ra_eff_vs_change_scatter.png
   figures/plot5_ra_eff_by_tier_boxplot.png
   figures/plot6_initial_vs_rep_scatter.png
   figures/plot7_steering_rescue_confusion.png   (if baseline available)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Repo root: analysis/negsteer_benchmark/make_plots.py -> parents[2].
REPO_ROOT = Path(__file__).resolve().parents[2]

# ── Tier assignment (provided) ────────────────────────────────────────────────
TIERS = {
    1: ["5A6W", "6FU9", "6FUB", "6FUD", "6G10", "6G11", "6Q76", "6R8K", "6R8M",
        "7A8W", "7A8X", "7BNT", "7QPX", "7QZD", "5ZNG", "9IP6", "9IMU", "8B2R"],
    2: ["7XVG", "7CRB", "7JLU", "7P8K"],
    3: ["7B1I", "7NLJ", "7NMM", "9RSV", "8R7A", "8R7D", "7PP2", "8PQ7", "8PFC",
        "8PFD", "9RDC"],
    4: ["8BV0", "9QWW", "9RIA", "7XC2", "7XX2", "7CRC", "9QLV", "9QT4", "9QU9",
        "9R8W"],
}
TIER_OF = {t: tier for tier, ts in TIERS.items() for t in ts}
TIER_COLORS = {1: "#1f77b4", 2: "#2ca02c", 3: "#ff7f0e", 4: "#d62728"}

# Biological category for each difficulty tier. The label names both parts of the
# modelled complex (receptor + effector). Category 4 is flagged as a single
# protomer to make clear it is one receptor/effector pair, not the full
# oligomeric resistosome.
CATEGORY_LABEL = {
    1: "HMA decoy + AVR",            # isolated integrated HMA decoy domain / AVR
    2: "NLR sensor + effector",      # single NLR sensor protomer, 1:1 binary
    3: "Host target + effector",     # genuine host virulence target / effector
    4: "Resistosome protomer + effector",  # one protomer of an NLR resistosome
}

RA_EFF_THRESHOLD = 5.0  # Angstrom; vertical reference line / pass cutoff

DEFAULT_BENCH = REPO_ROOT / "experiments" / "benchmarking"


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


def collect(bench: Path) -> pd.DataFrame:
    rows = []
    for target in sorted(p.name for p in bench.iterdir() if p.is_dir()):
        tdir = bench / target
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

        rep = {}
        rep_ra_eff = None
        if status == "representative":
            r = pd.read_csv(css).iloc[0]
            rep = r.to_dict()
            rep_ra_eff = r.get("rep_ra_eff_vs_truth_median")

        change = (
            rep_ra_eff - cold
            if (rep_ra_eff is not None and cold is not None)
            else None
        )

        # Lead with our derived columns, then the full representative row.
        tier = TIER_OF.get(target)
        row = {
            "target": target,
            "tier": tier,
            "category": CATEGORY_LABEL.get(tier),
            "status": status,
            "receptor_len": rlen,
            "effector_len": elen,
            "combined_seq_len": comb_len,
            "cold_start_ra_eff": cold,
            "rep_ra_eff": rep_ra_eff,
            "change_ra_eff": change,
        }
        row.update(rep)
        rows.append(row)
    return pd.DataFrame(rows)


# Place every legend outside the axes (to the right) so it never overlaps data
# and Inkscape gets a clean plotting area.
LEGEND_KW = dict(loc="center left", bbox_to_anchor=(1.01, 0.5),
                 borderaxespad=0.0, frameon=False)


def save_tight(fig, out: Path):
    """Save as PNG with whitespace cropped tight."""
    fig.savefig(out.with_suffix(".png"), dpi=200,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_ra_eff_hist(df: pd.DataFrame, out: Path):
    """Stacked histogram of representative ra_eff, split by tier.

    All tiers share the same bin edges (derived from the full value range) so the
    stacked bars line up; each tier's contribution is a coloured segment.
    """
    sub = df.dropna(subset=["rep_ra_eff"])
    vals = sub["rep_ra_eff"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.histogram_bin_edges(vals, bins=20) if len(vals) else 20
    tiers = sorted(TIERS)
    per_tier = [sub.loc[sub["tier"] == t, "rep_ra_eff"].to_numpy() for t in tiers]
    counts = [len(v) for v in per_tier]
    ax.hist(per_tier, bins=bins, stacked=True,
            color=[TIER_COLORS[t] for t in tiers],
            label=[f"{CATEGORY_LABEL[t]} (n = {c})" for t, c in zip(tiers, counts)],
            edgecolor="white")
    ax.axvline(RA_EFF_THRESHOLD, color="black", linestyle="--", linewidth=1.5,
               label=f"x = {RA_EFF_THRESHOLD:g} Å (pass threshold)")
    ax.set_xlabel("Representative ra_eff vs truth (Å)")
    ax.set_ylabel("Number of finished runs")
    ax.set_title(f"Representative ra_eff across finished runs (n = {len(vals)})")
    ax.legend(**LEGEND_KW)
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
    ax.set_title("Change in ra_eff from cold start to representative "
                 f"(n = {len(vals)})")
    # Legend proxies
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#2ca02c", label="improved (good)"),
        Patch(facecolor="#6baed6", hatch="//",
              label=f"no change — already correct at cold start "
                    f"(ra_eff ≤ {RA_EFF_THRESHOLD:g} Å)"),
        Patch(facecolor="#999999", label="no change — still incorrect"),
        Patch(facecolor="#d62728", label="worsened (bad)"),
    ], **LEGEND_KW)
    save_tight(fig, out)


def plot_length_scatter(df: pd.DataFrame, out: Path):
    sub = df.dropna(subset=["rep_ra_eff", "combined_seq_len"])
    fig, ax = plt.subplots(figsize=(8, 5))
    for tier in sorted(TIERS):
        g = sub[sub["tier"] == tier]
        if g.empty:
            continue
        ax.scatter(g["combined_seq_len"], g["rep_ra_eff"],
                   color=TIER_COLORS[tier], label=CATEGORY_LABEL[tier],
                   s=55, edgecolor="black", linewidth=0.4, alpha=0.9)
    ax.axhline(RA_EFF_THRESHOLD, color="black", linestyle="--", linewidth=1.2,
               label=f"ra_eff = {RA_EFF_THRESHOLD:g} Å")
    ax.set_xlabel("Combined sequence length (receptor + effector, residues)")
    ax.set_ylabel("Representative ra_eff vs truth (Å)")
    ax.set_title("Sequence length vs representative ra_eff")
    ax.legend(**LEGEND_KW)
    save_tight(fig, out)


def plot_ra_eff_vs_change(df: pd.DataFrame, out: Path):
    """Scatter of representative ra_eff vs its change from cold start.

    x = representative ra_eff (lower better); y = change (negative = improved).
    Coloured by tier. The horizontal y=0 line separates improved (below) from
    worsened (above); the vertical x=5 Å line marks the pass threshold.
    """
    sub = df.dropna(subset=["rep_ra_eff", "change_ra_eff"])
    fig, ax = plt.subplots(figsize=(8, 5))
    for tier in sorted(TIERS):
        g = sub[sub["tier"] == tier]
        if g.empty:
            continue
        ax.scatter(g["rep_ra_eff"], g["change_ra_eff"],
                   color=TIER_COLORS[tier], label=CATEGORY_LABEL[tier],
                   s=55, edgecolor="black", linewidth=0.4, alpha=0.9)
    ax.axhline(0, color="black", linewidth=1.2)
    ax.axvline(RA_EFF_THRESHOLD, color="black", linestyle="--", linewidth=1.2,
               label=f"ra_eff = {RA_EFF_THRESHOLD:g} Å")
    ax.set_xlabel("Representative ra_eff vs truth (Å)")
    ax.set_ylabel("Change in ra_eff: representative − cold start (Å)")
    ax.set_title(f"Representative ra_eff vs change from cold start "
                 f"(n = {len(sub)})")
    ax.legend(**LEGEND_KW)
    save_tight(fig, out)


def plot_initial_vs_rep(df: pd.DataFrame, out: Path):
    """Scatter of cold-start (initial) ra_eff vs representative ra_eff.

    x = initial (cold start), y = representative. Coloured by tier. The y = x
    diagonal marks no change: points BELOW it improved (representative < initial),
    points above worsened. The horizontal y = 5 Å dashed line marks the pass
    threshold for the representative prediction.
    """
    sub = df.dropna(subset=["cold_start_ra_eff", "rep_ra_eff"])
    fig, ax = plt.subplots(figsize=(8, 5))
    for tier in sorted(TIERS):
        g = sub[sub["tier"] == tier]
        if g.empty:
            continue
        ax.scatter(g["cold_start_ra_eff"], g["rep_ra_eff"],
                   color=TIER_COLORS[tier], label=CATEGORY_LABEL[tier],
                   s=55, edgecolor="black", linewidth=0.4, alpha=0.9)
    # y = x diagonal spanning the full data range (square the axes so it reads
    # as a true 45° line).
    both = np.concatenate([sub["cold_start_ra_eff"].to_numpy(),
                           sub["rep_ra_eff"].to_numpy()]) if len(sub) else [0, 1]
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
    ax.legend(loc="upper left", frameon=True)
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
    run-to-run prediction variance to steering — an external Boltz-2 run can pass
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
    scope = f"{scope_label}; " if scope_label else ""
    ax.set_title(f"Pose rescue by negative steering vs {baseline_label}\n"
                 f"({scope}n = {n} targets)")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    save_tight(fig, out)
    return dict(rescued=rescued, both_correct=both_correct,
                both_incorrect=both_incorrect, lost=lost, n=n)


def plot_tier_boxplot(df: pd.DataFrame, out: Path):
    """Grouped boxplot of representative ra_eff, one box per tier (1-4)."""
    sub = df.dropna(subset=["rep_ra_eff"])
    tiers = sorted(TIERS)
    data = [sub.loc[sub["tier"] == t, "rep_ra_eff"].to_numpy() for t in tiers]
    counts = [len(v) for v in data]
    fig, ax = plt.subplots(figsize=(8, 5))
    # No mean marker (showmeans) and no outlier fliers (showfliers): every point
    # is already drawn via the jittered overlay below.
    bp = ax.boxplot(data, patch_artist=True, showmeans=False, showfliers=False,
                    medianprops=dict(color="black"))
    for patch, t in zip(bp["boxes"], tiers):
        patch.set_facecolor(TIER_COLORS[t])
        patch.set_alpha(0.7)
    # Overlay the individual points (jittered deterministically) for context.
    for i, (t, vals) in enumerate(zip(tiers, data), start=1):
        if not len(vals):
            continue
        jitter = (np.linspace(-0.18, 0.18, len(vals)) if len(vals) > 1 else [0.0])
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=TIER_COLORS[t], edgecolor="black", linewidth=0.4,
                   s=28, alpha=0.9, zorder=3)
    ax.axhline(RA_EFF_THRESHOLD, color="black", linestyle="--", linewidth=1.2,
               label=f"ra_eff = {RA_EFF_THRESHOLD:g} Å")
    ax.set_xticks(range(1, len(tiers) + 1))
    ax.set_xticklabels([f"{CATEGORY_LABEL[t]}\n(n = {c})"
                        for t, c in zip(tiers, counts)], fontsize=8)
    ax.set_ylabel("Representative ra_eff vs truth (Å)")
    ax.legend(loc="upper left", frameon=False)
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

    figdir = args.outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    df = collect(args.bench)

    summary_path = args.outdir / "benchmark_summary.csv"
    df.to_csv(summary_path, index=False)

    plot_ra_eff_hist(df, figdir / "plot1_ra_eff_hist.png")
    plot_change_hist(df, figdir / "plot2_ra_eff_change_hist.png")
    plot_length_scatter(df, figdir / "plot3_length_vs_ra_eff_scatter.png")
    plot_ra_eff_vs_change(df, figdir / "plot4_ra_eff_vs_change_scatter.png")
    plot_tier_boxplot(df, figdir / "plot5_ra_eff_by_tier_boxplot.png")
    plot_initial_vs_rep(df, figdir / "plot6_initial_vs_rep_scatter.png")

    # Plot 7: cold-start -> steered rescue, restricted to Tier 1 (HMA decoy + AVR).
    # Baselines against the pipeline's OWN cold start (not an external predictor),
    # so it is a true before->after and "lost" cannot be a run-to-run artefact.
    confusion = plot_steering_confusion(
        df[df["tier"] == 1], figdir / "plot7_steering_rescue_confusion.png",
        scope_label="Tier 1 (HMA decoy + AVR)")

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
    print(f"\nTier-1 rescue vs cold start (n={confusion['n']}): "
          f"rescued={confusion['rescued']}, "
          f"both_correct={confusion['both_correct']}, "
          f"both_incorrect={confusion['both_incorrect']}, "
          f"lost={confusion['lost']}")
    figs = ["plot1_ra_eff_hist", "plot2_ra_eff_change_hist",
            "plot3_length_vs_ra_eff_scatter",
            "plot4_ra_eff_vs_change_scatter",
            "plot5_ra_eff_by_tier_boxplot",
            "plot6_initial_vs_rep_scatter",
            "plot7_steering_rescue_confusion"]
    print(f"\nWrote:\n  {summary_path}")
    for p in figs:
        print(f"  {figdir / p}.png")


if __name__ == "__main__":
    main()
