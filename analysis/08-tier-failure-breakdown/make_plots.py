#!/usr/bin/env python3
"""
Three candidate visualisations of tier_failure_breakdown_n50.csv: how the
(up to) 50 steering sets per target split into Tier A/B/C and the two
failure modes (misfolded vs wrong pose).

1. stacked_counts   -- vertical stacked bar, raw counts, targets ordered by
                        n_sets (so the varying denominator, 4-50, is visible
                        via bar height).
2. heatmap          -- targets x outcome grid, count in each cell, coloured
                        by that cell's share of the target's own n_sets (so
                        the varying denominator does not wash out colour
                        intensity for low-n_sets targets like 6Q76).
3. ranked_counts     -- horizontal raw-count stacked bar, targets ordered by
                        rescued-set count (A+B+C), pastel colours. Thesis
                        figure (fig-tier-failure-breakdown).

Run after tier_failure_breakdown.py.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

COLS = ["A", "B", "C", "failed_misfolded", "failed_wrong_pose"]
LABELS = {
    "A": "Tier A (all seeds pass)",
    "B": "Tier B (some seeds pass)",
    "C": "Tier C (one seed passes)",
    "failed_misfolded": "Failed: misfolded",
    "failed_wrong_pose": "Failed: wrong pose",
}
COLORS = {
    "A": "#1a9850",
    "B": "#91cf60",
    "C": "#d9ef8b",
    "failed_misfolded": "#4575b4",
    "failed_wrong_pose": "#d73027",
}
# Pastel variant + fraction labels: used for the thesis figure
# (plot_ranked_counts), which reports raw counts rather than percentage and
# drops the title/no-steering-colour/divider seen in earlier drafts.
LABELS_PASTEL = {
    "A": "Tier A (3/3)",
    "B": "Tier B (2/3)",
    "C": "Tier C (1/3)",
    "failed_misfolded": "Failed: misfolded",
    "failed_wrong_pose": "Failed: wrong pose",
}
COLORS_PASTEL = {
    "A": "#a1d99b",
    "B": "#c7e9c0",
    "C": "#e5f5e0",
    "failed_misfolded": "#9ecae1",
    "failed_wrong_pose": "#fcae91",
}

FIGDIR = HERE / "supplementary-figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(FIGDIR / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_stacked_counts(df: pd.DataFrame):
    d = df.sort_values("n_sets", ascending=False)
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bottom = np.zeros(len(d))
    for c in COLS:
        ax.bar(x, d[c], bottom=bottom, color=COLORS[c], label=LABELS[c],
               edgecolor="white", linewidth=0.5)
        bottom += d[c].to_numpy()
    ax.set_xticks(x)
    ax.set_xticklabels(d["target"], rotation=45, ha="right")
    ax.set_ylabel("Steering sets (of up to 50)")
    ax.set_title("Outcome of every steering set, n = 50 dose")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
               frameon=False, fontsize=9)
    for xi, n in zip(x, d["n_sets"]):
        ax.text(xi, n + 0.6, str(int(n)), ha="center", va="bottom", fontsize=7.5,
                color="#555555")
    save(fig, "tier_failure_stacked_counts")


def plot_heatmap(df: pd.DataFrame):
    d = df.sort_values("n_sets", ascending=False).set_index("target")
    frac = d[COLS].div(d["n_sets"].replace(0, np.nan), axis=0)
    fig, ax = plt.subplots(figsize=(6.5, 8))
    im = ax.imshow(frac.to_numpy(), aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(COLS)))
    ax.set_xticklabels([LABELS[c] for c in COLS], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d.index, fontsize=8)
    for i, t in enumerate(d.index):
        for j, c in enumerate(COLS):
            n = int(d.loc[t, c])
            ax.text(j, i, str(n), ha="center", va="center", fontsize=7.5,
                     color="black" if 0.25 < frac.loc[t, c] < 0.75 else "white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Share of that target's steering sets")
    ax.set_title("Outcome composition per target (n50)")
    save(fig, "tier_failure_heatmap")


def plot_ranked_counts(df: pd.DataFrame):
    """Raw-count composition, ranked A > B > C > misfolded > wrong pose (each
    descending, each breaking ties in the one before). Thesis figure.

    Targets with n_sets == 0 (5ZNG, 9IP6, 8B2R: cold start already passed on
    every seed, so steering never ran) have no steering-set data to draw a
    bar from at all. Rather than a zero-length bar, they get their own row
    with left-aligned text in place of a bar, and sit above every steered
    target: an already-correct cold start is the best outcome a target can
    have, better than any amount of rescuing.
    """
    RANK_COLS = ["A", "B", "C", "failed_misfolded", "failed_wrong_pose"]

    no_steer = df[df["n_sets"] == 0].sort_values("target")
    steered = df[df["n_sets"] > 0].sort_values(RANK_COLS, ascending=False)

    # barh with y = arange plots y=0 at the bottom, so "best at the top" means
    # best needs the HIGHEST y index: reverse steered to worst-first (bottom
    # to top improves), then the no-steer rows (best of all) go on top of that.
    steered = steered.iloc[::-1]
    d = pd.concat([steered, no_steer], ignore_index=True)

    fig, ax = plt.subplots(figsize=(9, 7.2))
    y = np.arange(len(d))
    y_steered = y[: len(steered)]
    y_no_steer = y[len(steered):]

    left = np.zeros(len(steered))
    for c in COLS:
        ax.barh(y_steered, steered[c], left=left, color=COLORS_PASTEL[c],
                 label=LABELS_PASTEL[c], edgecolor="white", linewidth=0.5)
        left += steered[c].to_numpy()
    for yi in y_no_steer:
        ax.text(0.5, yi, "Initial Prediction Correct", ha="left", va="center",
                fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{t} (n={n})" for t, n in zip(steered["target"], steered["n_sets"])] +
        [f"{t} (n=0)" for t in no_steer["target"]],
        fontsize=8.5)
    ax.set_ylim(-0.6, len(d) - 0.4)  # headroom so the top text row clears the axes border
    ax.set_xlabel("Steering sets")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
               frameon=False, fontsize=9)
    save(fig, "tier_failure_ranked_counts")


def main():
    df = pd.read_csv(HERE / "tier_failure_breakdown_n50.csv")
    plot_stacked_counts(df)
    plot_heatmap(df)
    plot_ranked_counts(df)
    print("Wrote:")
    for p in sorted(FIGDIR.glob("*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
