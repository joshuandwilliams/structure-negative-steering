#!/usr/bin/env python3
"""
Three-way variant of 02-model-choice.qmd's comparison figure: unrestrained,
reference-restrained, and negative-steered, with the fourth
(reference-restrained + negative-steered) "combined" series dropped.

That combined series is a ceiling reference (it needs the true interface for
its restraints, so it cannot run on a novel target) rather than a method
being compared on equal footing with the other three, and it only landed for
a subset of the 18 targets. Data prep matches 02-model-choice.qmd exactly:
neg_steered is the n=50 dose-sweep arm (see that qmd for why).

Run after analysis/01-pose-rescue/make_plots.py (needs benchmark_summary.csv).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
REPO_ROOT = ANALYSIS.parent
POSE = ANALYSIS / "01-pose-rescue"

sys.path.insert(0, str(POSE))
from make_plots import select_representative  # noqa: E402

RA_EFF_THRESHOLD = 5.0
RA_TICKS = [0.5, 1, 2, 5, 10, 20, 40]

ns = pd.read_csv(POSE / "benchmark_summary.csv").set_index("target")
bench = pd.read_csv(HERE / "boltz2_benchmark_comparison.csv").set_index("target")

DOSE_SWEEP = REPO_ROOT / "experiments" / "dose-sweep" / "n50"
neg_steered_n50 = {}
for t in bench.index:
    sel = select_representative(DOSE_SWEEP / t / "run")
    if sel is not None:
        neg_steered_n50[t] = sel["rep_ra_eff"]

cmp = pd.DataFrame({
    "unconstrained": bench["boltz2_unconstrained_ra_eff"],
    "constrained": bench["boltz2_constrained_ra_eff"],
    "neg_steered": pd.Series(neg_steered_n50),
}).reindex(bench.index)

FIGDIR = HERE / "thesis-figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


def readable_log(axis, ticks):
    axis.set_ticks(ticks)
    axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))


SERIES = [
    ("unconstrained", "unrestrained", "#999999"),
    ("constrained", "reference-restrained", "#D55E00"),
    ("neg_steered", "negative-steered", "#0072B2"),
]

best_any = cmp[[c for c, _, _ in SERIES]].min(axis=1)
o = cmp.loc[best_any.sort_values(ascending=False).index]
y = np.arange(len(o))

fig, ax = plt.subplots(figsize=(9.0, 6.8))
for yi, (_, r) in zip(y, o.iterrows()):
    vals = [r[c] for c, _, _ in SERIES if pd.notna(r[c])]
    if vals:
        ax.plot([min(vals), max(vals)], [yi, yi], color="#DDDDDD", lw=1.4, zorder=1)
for col, lab, colour in SERIES:
    ax.scatter(o[col], y, s=48, color=colour, edgecolor="black",
               linewidth=0.5, label=lab, zorder=3)
ax.axvline(RA_EFF_THRESHOLD, ls="--", color="grey", lw=1)
ax.set_xscale("log")
readable_log(ax.xaxis, RA_TICKS)
ax.set_yticks(y)
ax.set_yticklabels(o.index, fontsize=8)
ax.set_ylim(-0.8, len(o) - 0.2)
ax.set_xlabel("Receptor-aligned effector RMSD (Å, log scale)")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
          frameon=False, fontsize=9, borderaxespad=0.0)
fig.savefig(FIGDIR / "model_choice_comparison_3way.png", dpi=200,
            bbox_inches="tight", pad_inches=0.05)

summary_rows = []
for col, lab, _ in SERIES:
    n = int(o[col].notna().sum())
    summary_rows.append({
        "method": lab,
        f"correct /{n}": int((o[col] < RA_EFF_THRESHOLD).sum()),
        "median ra_eff": round(float(o[col].median()), 2) if n else None,
    })
result = pd.DataFrame(summary_rows)
print(result.to_string(index=False))
print(f"\nWrote {FIGDIR / 'model_choice_comparison_3way.png'}")
