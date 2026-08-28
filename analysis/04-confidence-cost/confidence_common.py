"""Shared loading, statistics and plotting for the two confidence documents.

`04-confidence-cost.qmd` runs on the 18-target benchmark, `05-campaign-confidence.qmd`
on the 128-sequence pipeline campaign. They ask the same two questions and must
therefore use identical selection rules, statistics and figures, so all of that
lives here and each document only supplies its own data.

Both read the same file, `raw_per_seed_results.csv`, which the engine writes for
every negative-steering run. One row per (design, seed). The rows whose `design`
starts with `initial` are the COLD START, the unsteered prediction of the input
sequence, made by the same engine with the same settings as every steered design.
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

RA_EFF_THRESHOLD = 5.0   # Angstrom; a pose at or below this is correct

METRICS = {"complex_plddt": "complex pLDDT",
           "interface_plddt": "interface pLDDT",
           "iptm": "ipTM",
           "ipsae_min": "ipSAE (min)"}
UNITS = {"complex_plddt": "pLDDT points", "interface_plddt": "pLDDT points",
         "iptm": "ipTM", "ipsae_min": "ipSAE"}
FMT = {"complex_plddt": "{:+.2f}", "interface_plddt": "{:+.2f}", "iptm": "{:+.3f}",
       "ipsae_min": "{:+.3f}"}
PLDDT_COLS = ["complex_plddt", "interface_plddt"]

OUTCOME_COLOR = {"rescued": "#0072B2",          # cold start wrong -> steered right
                 "already correct": "#009E73",  # right before and after
                 "still wrong": "#D55E00",      # wrong before and after
                 "lost": "#6E6E6E"}             # right before, wrong after
OUTCOME_ORDER = ["rescued", "already correct", "still wrong", "lost"]
# Sequence controls, kept visually distinct from each other and from the four
# outcome colours. Both are Okabe-Ito, as the outcome palette is.
CONTROL_COLOR = {"polyA": "#CC79A7", "scrambled": "#000000"}
INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"
# Plain-number ticks for log RMSD axes, matching the chapter's other log plots.
LOG_TICKS = [0.5, 1, 2, 3, 5, 10, 20, 40]

LEGEND_KW = dict(loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0.0)


def place_legend(ax, handles=None, labels=None, **kw):
    """Beside the axes for a short legend, above the figure for a long one.

    A long legend beside a faceted figure eats the width of a whole panel, so
    anything with more than three entries or a wrapped label goes on top.
    """
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    elif labels is None:
        labels = [h.get_label() for h in handles]
    n = len(labels)
    longest = max((max(len(part) for part in str(x).split("\n")) for x in labels),
                  default=0)
    if n > 3 or longest > 22:
        base = dict(loc="lower center", bbox_to_anchor=(0.5, 1.02),
                    borderaxespad=0.0, frameon=False, ncol=min(n, 3))
        target = ax.get_figure()
        base["bbox_transform"] = target.transFigure
        base["bbox_to_anchor"] = (0.5, 1.0)
        return target.legend(handles, labels, **{**base, **kw})
    return ax.legend(handles, labels, **{**LEGEND_KW, **kw})

RENAME = {"steered_ra_eff_vs_truth": "ra_eff",
          "steered_complex_plddt": "complex_plddt",
          "steered_interface_plddt": "interface_plddt",
          "steered_iptm": "iptm", "steered_ptm": "ptm",
          "steered_ipsae_min": "ipsae_min",
          "steered_true_jaccard": "true_jaccard",
          "steered_total_mutations": "n_mutations"}

KEEP = ["unit", "design", "base", "status", "kind", "seed_index", "n_mutations",
        "ra_eff", "reverted_ra_eff_vs_truth", "reversion_verdict",
        "steered_n_contacts_on_mutated_positions", "steered_receptor_intact",
        "complex_plddt", "interface_plddt", "iptm", "ptm", "ipsae_min",
        "true_jaccard"]


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
    })


def restore_inline_backend(fn):
    """Run `fn` with the notebook's matplotlib backend put back afterwards.

    `make_plots.py` calls matplotlib.use("Agg") at import time. That is right
    for a headless script and wrong inside a notebook kernel, where it detaches
    the figures so nothing reaches the rendered HTML. Anything that imports it
    has to undo it.
    """
    backend = matplotlib.get_backend()
    try:
        return fn()
    finally:
        matplotlib.use(backend)


# ── Loading ──────────────────────────────────────────────────────────────────

def load_per_seed(paths: dict[str, "object"]) -> pd.DataFrame:
    """{unit label -> raw_per_seed_results.csv path} -> one tidy frame.

    "unit" is whatever the document treats as an independent case: a benchmark
    target, or a designed sequence in the campaign. pLDDT is rescaled to the
    conventional 0-100 axis, since a change of 0.032 is easy to misread and 3.2
    pLDDT points is not.
    """
    frames = []
    for unit, path in paths.items():
        d = pd.read_csv(path)
        d["unit"] = unit
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    raw["kind"] = np.where(
        raw["design"].astype(str).str.startswith("initial"), "cold_start", "steered")
    raw["base"] = raw["design"].astype(str).str.replace(r"_s\d+$", "", regex=True)
    raw = raw.rename(columns=RENAME)
    for c in KEEP:
        if c not in raw:
            raw[c] = np.nan
    out = raw[KEEP].copy()
    out[PLDDT_COLS] = out[PLDDT_COLS] * 100.0
    return out


def split(pred: pd.DataFrame):
    """-> (one cold-start row per unit, all usable steered rows)."""
    cold = (pred[(pred.kind == "cold_start") & (pred.design == "initial")]
            .drop_duplicates("unit").set_index("unit"))
    steered = pred[(pred.kind == "steered") & (pred.status == "ok")].copy()
    return cold, steered


# ── Representative-design selection ──────────────────────────────────────────
# Mirrors make_plots.select_representative, but reads only the per-seed CSV so
# the same code serves both documents. Designs are ranked by how many of their
# seeds cleanly pass (reliability first) and then by the median ra_eff of those
# seeds, computed on each seed's reversion-corrected final value.

_REVERTED_FAIL = {"pose_collapses", "new_contamination", "pose_holds_not_intact",
                  "missing_verdict", "unknown"}


def _num(v):
    try:
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


def _final_seed_value(row) -> tuple[float | None, bool]:
    """One per-seed row -> (final ra_eff, did it cleanly pass).

    A seed whose steered mutations landed on or near a protected position goes
    through reversion, and its real final value is the REVERTED structure's
    ra_eff rather than the raw steered prediction.
    """
    verdict = str(row.get("reversion_verdict") or "").strip()
    if verdict == "pose_holds":
        ra = _num(row.get("reverted_ra_eff_vs_truth"))
        return ra, (ra is not None and ra < RA_EFF_THRESHOLD)
    if verdict in _REVERTED_FAIL:
        return _num(row.get("reverted_ra_eff_vs_truth")), False
    # Blank verdict: reversion was correctly skipped (no contamination on the
    # mutated positions), or the row is missing data.
    n_contacts = _num(row.get("steered_n_contacts_on_mutated_positions"))
    intact = _num(row.get("steered_receptor_intact"))
    ra = _num(row.get("ra_eff"))
    if n_contacts == 0 and ra is not None:
        return ra, (intact == 1 and ra < RA_EFF_THRESHOLD)
    return None, False


def _rank(n_pass: int, n_seeds: int) -> int:
    """Reliability-first ordering: all seeds pass, then most, then one, then none."""
    if n_seeds <= 0:
        return 3
    if n_pass == n_seeds:
        return 0
    if n_pass == 0:
        return 3
    return 1 if n_pass > 1 else 2


def select_representative(unit_steered: pd.DataFrame,
                          unit_cold: pd.DataFrame) -> dict | None:
    """Pick one unit's representative design from its per-seed rows.

    Falls back to the cold start's own multi-seed set when steering was
    short-circuited (the cold start already passed on every seed), which is the
    same rule make_plots applies.
    """
    candidates = []
    for base, grp in unit_steered.groupby("base"):
        finals, n_pass = [], 0
        for _, row in grp.iterrows():
            ra, ok = _final_seed_value(row)
            if ra is not None:
                finals.append(ra)
            n_pass += bool(ok)
        if finals:
            candidates.append((_rank(n_pass, len(grp)), float(np.median(finals)),
                               n_pass, len(grp), base))

    if not candidates and len(unit_cold):
        finals = [v for v in (_num(x) for x in unit_cold["ra_eff"]) if v is not None]
        if finals:
            n_pass = sum(f < RA_EFF_THRESHOLD for f in finals)
            candidates.append((_rank(n_pass, len(finals)), float(np.median(finals)),
                               n_pass, len(finals), "initial"))

    if not candidates:
        return None
    rank, median, n_pass, n_seeds, label = min(candidates, key=lambda c: (c[0], c[1]))
    return {"rep_ra_eff": median, "rep_n_pass": n_pass,
            "rep_n_seeds": n_seeds, "rep_design": label}


def build_representative_table(pred: pd.DataFrame) -> pd.DataFrame:
    """One row per unit: cold-start and representative ra_eff and confidence."""
    cold, steered = split(pred)
    cold_all = pred[pred.kind == "cold_start"]
    recs = []
    for unit in cold.index:
        us = steered[steered.unit == unit]
        uc = cold_all[cold_all.unit == unit]
        rep = select_representative(us, uc)
        if rep is None:
            continue
        skipped = rep["rep_design"] == "initial"
        grp = uc if skipped else us[us.base == rep["rep_design"]]
        muts = us["n_mutations"]
        r = {"unit": unit, "rep_design": rep["rep_design"],
             "steering_skipped": skipped, "rep_ra_eff": rep["rep_ra_eff"],
             "rep_n_pass": rep["rep_n_pass"], "rep_n_seeds": rep["rep_n_seeds"],
             "cold_ra_eff": cold.loc[unit, "ra_eff"],
             "n_mutations": float(np.nanmedian(muts)) if len(muts) else np.nan}
        for m in METRICS:
            r[f"cold_{m}"] = cold.loc[unit, m]
            r[f"rep_{m}"] = grp[m].median()
            r[f"d_{m}"] = r[f"rep_{m}"] - r[f"cold_{m}"]
        # Not a confidence metric and so not differenced against the cold start,
        # but carried at the same level so the controls figure can be drawn on
        # representative designs rather than on raw seeds.
        r["rep_true_jaccard"] = pd.to_numeric(grp.get("true_jaccard"),
                                              errors="coerce").median()
        recs.append(r)

    df = pd.DataFrame(recs)
    was, now = df.cold_ra_eff <= RA_EFF_THRESHOLD, df.rep_ra_eff <= RA_EFF_THRESHOLD
    df["outcome"] = np.select([~was & now, was & now, ~was & ~now, was & ~now],
                              OUTCOME_ORDER, default="")
    return df


# ── Statistics ───────────────────────────────────────────────────────────────

def signed_rank_p(diffs, n=20000, seed=0) -> float:
    """Wilcoxon signed-rank test, by sign-flip permutation.

    The statistic is the sum of signed ranks of |d| and the null is built by
    randomly flipping each pair's sign, which is exactly the Wilcoxon
    construction. Done by permutation because scipy is not a dependency here.
    A median statistic is too coarse at n = 15 and a mean is dragged around by
    one or two large-effect units.
    """
    d = np.asarray([x for x in diffs if np.isfinite(x)], float)
    d = d[d != 0]
    if d.size == 0:
        return np.nan
    ranks = pd.Series(np.abs(d)).rank().to_numpy()
    rng = np.random.default_rng(seed)
    obs = abs((np.sign(d) * ranks).sum())
    null = np.abs((rng.choice([-1.0, 1.0], size=(n, d.size)) * ranks).sum(axis=1))
    return float((1 + (null >= obs - 1e-12).sum()) / (n + 1))


def median_ci(diffs, n=10000, seed=1):
    """Bootstrap 95% CI on the median of a set of paired differences."""
    d = np.asarray([x for x in diffs if np.isfinite(x)], float)
    if d.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = np.median(rng.choice(d, size=(n, d.size), replace=True), axis=1)
    return (float(np.median(d)), float(np.percentile(bs, 2.5)),
            float(np.percentile(bs, 97.5)))


def rank_sum_p(a, b, n=20000, seed=0) -> float:
    """Two-sample rank-sum test, by label permutation.

    The unpaired counterpart of `signed_rank_p`, for comparing rescued against
    still-wrong units. These are different units rather than a before/after of
    the same unit, so the paired test does not apply. The statistic is the rank
    sum of group a within the pooled ranks and the null shuffles the labels.
    """
    a = np.asarray([x for x in a if np.isfinite(x)], float)
    b = np.asarray([x for x in b if np.isfinite(x)], float)
    if a.size == 0 or b.size == 0:
        return np.nan
    pooled = np.concatenate([a, b])
    ranks = pd.Series(pooled).rank().to_numpy()
    obs = abs(ranks[:a.size].sum() - a.size * (ranks.sum() / pooled.size))
    rng = np.random.default_rng(seed)
    idx = np.argsort(rng.random((n, pooled.size)), axis=1)[:, :a.size]
    null = np.abs(ranks[idx].sum(axis=1) - a.size * (ranks.sum() / pooled.size))
    return float((1 + (null >= obs - 1e-12).sum()) / (n + 1))


def spearman(a, b) -> float:
    a, b = pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)
    ok = a.notna() & b.notna()
    if ok.sum() < 4:
        return np.nan
    return float(np.corrcoef(a[ok].rank(), b[ok].rank())[0, 1])


def effect_table(entries) -> pd.DataFrame:
    """[(set label, metric key, diffs)] -> tidy effect sizes with intervals."""
    rows = []
    for lab, m, d in entries:
        d = np.asarray(d, float)
        med, lo, hi = median_ci(d)
        f = FMT[m]
        rows.append({"metric": METRICS[m], "set": lab,
                     "n": int(np.isfinite(d).sum()),
                     f"median Δ ({UNITS[m]})": f.format(med),
                     "95% CI": f"[{f.format(lo)}, {f.format(hi)}]",
                     "p": round(signed_rank_p(d), 4)})
    return pd.DataFrame(rows)


# ── Figures ──────────────────────────────────────────────────────────────────

def _ylabel(m, lab, delta=False):
    unit = " (0-100)" if (m in PLDDT_COLS and not delta) else \
           (" (pLDDT points)" if m in PLDDT_COLS else "")
    return (f"Δ {lab}" if delta else lab) + unit


def fig_before_after(repdf: pd.DataFrame, steered_only: pd.DataFrame,
                     outpath, alpha=0.85, lw=2.0):
    """Paired cold start to representative steered, one panel per metric."""
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 4.2))
    for ax, (m, lab) in zip(axes, METRICS.items()):
        for _, r in repdf.iterrows():
            ax.plot([0, 1], [r[f"cold_{m}"], r[f"rep_{m}"]],
                    color=OUTCOME_COLOR[r["outcome"]], linewidth=lw, alpha=alpha,
                    linestyle="--" if r["steering_skipped"] else "-",
                    marker="o", markersize=4.0, zorder=2,
                    markeredgecolor="white", markeredgewidth=0.6)
        # A heavy median slope over the top. At n = 128 the individual lines are
        # a hairball, and the median difference is the number the panel title
        # quotes, so it has to be visible in the plot too. Anchored at the
        # median cold start and offset by the median paired difference, which is
        # not the same as the difference of the two medians.
        d = steered_only[f"d_{m}"].values
        med, _, _ = median_ci(d)
        base = steered_only[f"cold_{m}"].median()
        ax.plot([0, 1], [base, base + med], color=INK, linewidth=3.2,
                marker="o", markersize=6.5, zorder=6,
                markeredgecolor="white", markeredgewidth=1.2,
                solid_capstyle="round")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["cold start", "steered"])
        ax.set_xlim(-0.25, 1.25)
        ax.set_title(f"{lab}\nmedian Δ {FMT[m].format(med)},  p = {signed_rank_p(d):.3f}",
                     fontsize=9)
        ax.set_ylabel(_ylabel(m, lab))
        ax.grid(axis="x", visible=False)

    handles = [Line2D([], [], color=OUTCOME_COLOR[o], linewidth=2.4, marker="o",
                      markersize=5, label=o)
               for o in OUTCOME_ORDER if (repdf.outcome == o).any()]
    if repdf.steering_skipped.any():
        handles.append(Line2D([], [], color=MUTED, linewidth=2.0, linestyle="--",
                              label="steering skipped\n(cold start already passed)"))
    handles.append(Line2D([], [], color=INK, linewidth=3.2, marker="o", markersize=6,
                          label="median (steered units)"))
    place_legend(axes[-1], handles=handles)
    fig.tight_layout()
    fig.savefig(outpath)
    return fig


def fig_dose(sub: pd.DataFrame, outpath, jitter=0.13, size=60):
    """Confidence change against the unit's mutation budget."""
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.8), sharex=True)
    jit = np.random.default_rng(7).uniform(-jitter, jitter, len(sub))
    for ax, (m, lab) in zip(axes, METRICS.items()):
        ax.scatter(sub.n_mutations + jit, sub[f"d_{m}"], s=size, zorder=3,
                   color=[OUTCOME_COLOR[o] for o in sub.outcome],
                   edgecolor="white", linewidth=0.7, alpha=0.85)
        for n_mut, grp in sub.groupby("n_mutations"):
            ax.plot([n_mut - 0.3, n_mut + 0.3], [grp[f"d_{m}"].median()] * 2,
                    color=INK, linewidth=2.2, zorder=4)
        ax.axhline(0, color=MUTED, linestyle="--", linewidth=1.2, zorder=1)
        ax.set_xticks(sorted(sub.n_mutations.dropna().unique()))
        ax.set_xlabel("mutations per design")
        ax.set_ylabel(_ylabel(m, lab, delta=True))
        ax.set_title(f"{lab}\nρ vs budget = {spearman(sub.n_mutations, sub[f'd_{m}']):+.2f}",
                     fontsize=9)
        ax.grid(axis="x", visible=False)

    handles = [Line2D([], [], marker="o", linestyle="", markersize=7,
                      color=OUTCOME_COLOR[o], label=o)
               for o in OUTCOME_ORDER if (sub.outcome == o).any()]
    handles.append(Line2D([], [], color=INK, linewidth=2.2, label="group median"))
    place_legend(axes[-1], handles=handles)
    fig.tight_layout()
    fig.savefig(outpath)
    return fig


def dose_table(sub: pd.DataFrame) -> pd.DataFrame:
    return sub.groupby("n_mutations").agg(
        n_units=("unit", "size"),
        d_complex_plddt=("d_complex_plddt", "median"),
        d_interface_plddt=("d_interface_plddt", "median"),
        d_iptm=("d_iptm", "median"),
        n_rescued=("outcome", lambda s: (s == "rescued").sum()),
    ).round(3)


# ── Effect-size figures ──────────────────────────────────────────────────────
# The before/after and dose figures each show one cohort at a time, which makes
# the benchmark's pLDDT drop look like a property of the method rather than of
# its 15 targets. These two put the paired differences themselves on a common
# axis, so a cohort whose interval spans zero cannot be read as a cost.

CONTRAST_OUTCOMES = ["rescued", "still wrong"]


def _forest_panel(ax, rows, m, jitter=0.10, size=26, seed=11):
    """Draw one metric's paired differences as jittered points + median and CI.

    `rows` is [(label, diffs, colour)], drawn top-down. The CI is the bootstrap
    interval on the median from `median_ci`, and is suppressed below n = 3,
    where resampling a median is degenerate rather than informative.
    """
    rng = np.random.default_rng(seed)
    for i, (lab, diffs, color) in enumerate(rows):
        y = -i
        d = np.asarray([x for x in diffs if np.isfinite(x)], float)
        if not d.size:
            continue
        ax.scatter(d, y + rng.uniform(-jitter, jitter, d.size), s=size, zorder=3,
                   color=color, edgecolor="white", linewidth=0.5, alpha=0.55)
        med, lo, hi = median_ci(d)
        if d.size >= 3:
            ax.plot([lo, hi], [y, y], color=INK, linewidth=2.0, zorder=5,
                    solid_capstyle="round")
        ax.plot([med], [y], marker="D", markersize=6.5, color=INK, zorder=6,
                markeredgecolor="white", markeredgewidth=1.0)
    ax.axvline(0, color=MUTED, linestyle="--", linewidth=1.2, zorder=1)
    ax.set_yticks([-i for i in range(len(rows))])
    ax.set_yticklabels([lab for lab, _, _ in rows])
    ax.set_ylim(-len(rows) + 0.4, 0.6)
    ax.set_xlabel(_ylabel(m, METRICS[m], delta=True))
    ax.grid(axis="y", visible=False)


def _steered(df):
    return df[~df.steering_skipped]


def fig_cohort_effects(cohorts: dict, outpath, stats_in_title=True,
                       legend="right", figsize=(9.0, 6.4)):
    """Cohorts' paired differences on one axis per metric.

    `cohorts` is {label -> representative table}. Only actually-steered units
    are used, since a short-circuited unit has no steered prediction to
    difference against.

    As with `fig_constraint_scatter`, the defaults give the analysis version and
    the thesis version drops the per-panel statistics.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for ax, m in zip(axes.ravel(), METRICS):
        rows = [(f"{lab}\n(n = {len(_steered(df))})", _steered(df)[f"d_{m}"].values,
                 OUTCOME_COLOR["rescued"]) for lab, df in cohorts.items()]
        _forest_panel(ax, rows, m)
        # One cohort per line. Side by side they collide with the neighbouring
        # panel's title long before the axes themselves overlap.
        parts = [f"{lab}: p = {signed_rank_p(_steered(df)[f'd_{m}'].values):.3f}"
                 for lab, df in cohorts.items()]
        ax.set_title(METRICS[m] + ("\n" + "\n".join(parts) if stats_in_title else ""),
                     fontsize=7.5 if stats_in_title else 10)

    handles = [Line2D([], [], color=INK, marker="D", linestyle="", markersize=6,
                      label="median"),
               Line2D([], [], color=INK, linewidth=2.0, label="95% CI (bootstrap)")]
    if legend == "top":
        fig.legend(handles=handles, loc="upper center", ncol=len(handles),
                   frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(h_pad=2.4, rect=(0, 0, 1, 0.95))
    else:
        place_legend(axes[0, 1], handles=handles)
        fig.tight_layout(h_pad=2.4)
    fig.savefig(outpath)
    return fig


def fig_outcome_contrast(cohorts: dict, outpath):
    """Paired differences split by whether steering rescued the pose.

    This is the panel that tests the runner-up-interface explanation directly.
    If steering paid for the pose, the units it moved to the true interface
    would carry the cost and the units left at the wrong one would not.
    """
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.6))
    for ax, m in zip(axes.ravel(), METRICS):
        rows, tests = [], []
        for lab, df in cohorts.items():
            s = _steered(df)
            by = {o: s[s.outcome == o][f"d_{m}"].values for o in CONTRAST_OUTCOMES}
            for o in CONTRAST_OUTCOMES:
                rows.append((f"{lab}\n{o} (n = {len(by[o])})", by[o], OUTCOME_COLOR[o]))
            tests.append(f"{lab}: p = {rank_sum_p(*[by[o] for o in CONTRAST_OUTCOMES]):.3f}")
        _forest_panel(ax, rows, m)
        ax.set_title(f"{METRICS[m]}\n" + "\n".join(tests), fontsize=7.5)

    handles = [Line2D([], [], marker="o", linestyle="", markersize=7,
                      color=OUTCOME_COLOR[o], label=o) for o in CONTRAST_OUTCOMES]
    handles.append(Line2D([], [], color=INK, marker="D", linestyle="", markersize=6,
                          label="median, 95% CI"))
    place_legend(axes[0, 1], handles=handles)
    fig.tight_layout(h_pad=2.4)
    fig.savefig(outpath)
    return fig


def pose_matched_pairs(unconstrained: pd.DataFrame, constrained: pd.DataFrame):
    """Units whose representative design is correctly posed under BOTH variants.

    Comparing a confidence metric across variants only means something when both
    variants are describing the same pose. A unit the constrained variant placed
    correctly and the unconstrained one did not would contribute a difference
    that is about the pose, not about what constraints do to the metric.
    """
    a = unconstrained.set_index("unit")
    b = constrained.set_index("unit")
    shared = a.index.intersection(b.index)
    ok = shared[(a.loc[shared, "rep_ra_eff"] <= RA_EFF_THRESHOLD)
                & (b.loc[shared, "rep_ra_eff"] <= RA_EFF_THRESHOLD)]
    return a.loc[ok], b.loc[ok]


def fig_constraint_scatter(pairs: dict, outpath, stats_in_title=True,
                           metric_in_axes=True, legend="right", size=34):
    """Steered confidence with constraints against without, pose-matched.

    `pairs` is {cohort -> (unconstrained table, constrained table, colour)}.
    Points above the identity line are units the constraints made the model
    more confident about, without the pose having changed.

    The keyword defaults give the analysis version. The thesis version drops the
    the per-panel statistics and the metric name on the axes, which the
    panel title already carries, and moves the legend to a single row on top.
    """
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.4))
    matched = {lab: (*pose_matched_pairs(u, c), col)
               for lab, (u, c, col) in pairs.items()}
    for ax, m in zip(axes.ravel(), METRICS):
        lims, titles = [], []
        for lab, (a, b, col) in matched.items():
            x, y = a[f"rep_{m}"].values, b[f"rep_{m}"].values
            ax.scatter(x, y, s=size, color=col, alpha=0.65, zorder=3,
                       edgecolor="white", linewidth=0.6, label=f"{lab} (n = {len(x)})")
            lims += [v for v in np.concatenate([x, y]) if np.isfinite(v)]
            d = y - x
            titles.append(f"{lab}: median Δ {FMT[m].format(np.median(d))}, "
                          f"p = {signed_rank_p(d):.3f}")
        lo, hi = min(lims), max(lims)
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED,
                linestyle="--", linewidth=1.2, zorder=1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
        stem = f"{METRICS[m]}, " if metric_in_axes else ""
        ax.set_xlabel(f"{stem}negsteer")
        ax.set_ylabel(f"{stem}negsteer + constraints")
        ax.set_title(METRICS[m] + ("\n" + "\n".join(titles) if stats_in_title else ""),
                     fontsize=7.5 if stats_in_title else 10)

    if legend == "top":
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(labels),
                   frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(h_pad=2.4, rect=(0, 0, 1, 0.96))
    else:
        place_legend(axes[0, 1])
        fig.tight_layout(h_pad=2.4)
    fig.savefig(outpath)
    return fig


# ── Validity figures ─────────────────────────────────────────────────────────
# The effect-size figures above characterise what steering costs. These three
# ask the prior question of whether it works for the reason claimed.

def seed_pass_counts(pred: pd.DataFrame) -> pd.DataFrame:
    """One row per design: how many of its seeds cleanly passed.

    Uses the same `_final_seed_value` rule the representative-design selection
    uses, so reliability here is the quantity that ranking is actually built on
    rather than a second, looser definition of a pass.
    """
    steered = pred[(pred.kind == "steered") & (pred.status == "ok")]
    recs = []
    for (unit, base), grp in steered.groupby(["unit", "base"]):
        n_pass = sum(ok for _, ok in (_final_seed_value(r) for _, r in grp.iterrows()))
        recs.append({"unit": unit, "base": base, "n_seeds": len(grp),
                     "n_pass": n_pass, "frac_pass": n_pass / len(grp) if len(grp) else np.nan})
    return pd.DataFrame(recs)


def fig_controls_diagnostic(groups: dict, outpath, panels=("ra_eff", "true_jaccard"),
                            labels=None, unit_name="seeds", figsize=(7.6, 5.6)):
    """Real steered designs against sequence controls, per seed.

    `groups` is {label -> (per-seed frame, colour)}. Every usable steered seed is
    a point, rather than one representative per run, because the controls are
    two sequences and collapsing them to one point each throws away the only
    spread they have.
    """
    labels = labels or {"ra_eff": "receptor-aligned effector RMSD (Å)",
                        "true_jaccard": "interface Jaccard vs true interface"}
    fig, axes = plt.subplots(len(panels), 1, figsize=figsize, sharex=True)
    rng = np.random.default_rng(3)
    for ax, col in zip(np.atleast_1d(axes), panels):
        for i, (lab, (df, colour)) in enumerate(groups.items()):
            v = df[col].astype(float).dropna().values
            if not v.size:
                continue
            ax.scatter(i + rng.uniform(-0.12, 0.12, v.size), v, s=30, color=colour,
                       alpha=0.6, zorder=3, edgecolor="white", linewidth=0.5)
            ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color=INK,
                    linewidth=2.4, zorder=5)
        if col.endswith("ra_eff"):
            ax.axhline(RA_EFF_THRESHOLD, color="#C0392B", linestyle="--",
                       linewidth=1.2, zorder=1)
        ax.set_ylabel(labels.get(col, col))
        ax.grid(axis="x", visible=False)
    ax = np.atleast_1d(axes)[-1]
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{lab}\n(n = {len(df[panels[0]].dropna())} {unit_name})"
                        for lab, (df, _) in groups.items()])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    fig.tight_layout()
    fig.savefig(outpath)
    return fig


def fig_controls_joint(groups: dict, outpath, x="rep_ra_eff", y="rep_true_jaccard",
                       xlabel="receptor-aligned effector RMSD (Å, log scale)",
                       ylabel="interface Jaccard vs true interface",
                       unit_name="sequences", legend="top", size=44,
                       figsize=(6.6, 5.6)):
    """Pose error against interface overlap, in one panel.

    The two-panel version shows each quantity's marginal distribution and hides
    the relationship between them. Here a design that merely moved the effector
    somewhere else and one that moved it onto the intended surface separate
    vertically even when they sit at the same RMSD. RMSD is on a log axis
    because the failures run to 40 Å and would otherwise compress the passes
    into the left edge.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for lab, (df, colour) in groups.items():
        d = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
        ax.scatter(d[x], d[y], s=size, color=colour, alpha=0.7, zorder=3,
                   edgecolor="white", linewidth=0.6,
                   label=f"{lab} (n = {len(d)} {unit_name})")
    ax.axvline(RA_EFF_THRESHOLD, color="#C0392B", linestyle="--", linewidth=1.2,
               zorder=1)
    # A log axis so the failures at 40 A do not compress the passes into the
    # left edge, but with the plain numbers the rest of the chapter uses rather
    # than matplotlib's 10^1 default.
    ax.set_xscale("log")
    lo = min(pd.to_numeric(df[x], errors="coerce").min() for df, _ in groups.values())
    hi = max(pd.to_numeric(df[x], errors="coerce").max() for df, _ in groups.values())
    ticks = [t for t in LOG_TICKS if lo / 1.2 <= t <= hi * 1.2]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend == "top":
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        place_legend(ax)
        fig.tight_layout()
    fig.savefig(outpath)
    return fig


def fig_seed_reliability(groups: dict, outpath, figsize=(7.6, 4.0)):
    """How many of a design's seeds pass, as a distribution rather than a mean.

    A design that passes on every seed and one that passes on a single lucky
    seed have very different value, and a mean pass rate hides which of the two
    a cohort is made of. `groups` is {label -> (per-seed frame, colour)}.
    """
    fig, axes = plt.subplots(1, len(groups), figsize=figsize, sharey=True)
    for ax, (lab, (pred, colour)) in zip(np.atleast_1d(axes), groups.items()):
        sp = seed_pass_counts(pred)
        n_max = int(sp.n_seeds.max()) if len(sp) else 0
        counts = sp.n_pass.value_counts().reindex(range(n_max + 1), fill_value=0)
        ax.bar(counts.index, counts.values, color=colour, edgecolor="white",
               linewidth=0.8, zorder=3)
        ax.set_xticks(range(n_max + 1))
        ax.set_xlabel("seeds passing")
        ax.set_title(f"{lab}\n{len(sp)} designs, {n_max} seeds each", fontsize=9)
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].set_ylabel("designs")
    fig.tight_layout()
    fig.savefig(outpath)
    return fig


REVERSION_ORDER = ["pose_holds", "pose_collapses", "new_contamination",
                   "pose_holds_not_intact"]


def fig_reversion_cascade(groups: dict, outpath, figsize=(7.6, 4.2)):
    """What reversion does to the seeds it is applied to.

    Reversion only runs on seeds whose steering mutations themselves contact the
    effector, since the wet-lab construct will not carry them. This shows how
    many seeds reach that check and what the verdict does to the pose, which is
    the step the Methods flowchart describes but no result quantifies.
    """
    fig, axes = plt.subplots(1, len(groups), figsize=figsize, sharey=True)
    for ax, (lab, (pred, _)) in zip(np.atleast_1d(axes), groups.items()):
        st = pred[(pred.kind == "steered") & (pred.status == "ok")]
        verd = st.reversion_verdict.fillna("").astype(str).str.strip()
        checked = st[verd != ""]
        rows, xt = [], []
        for name, sub, col in [
                ("no reversion\nneeded", st[verd == ""], "ra_eff"),
                ("reverted", checked, "reverted_ra_eff_vs_truth")]:
            v = pd.to_numeric(sub[col], errors="coerce")
            rows.append((name, int((v <= RA_EFF_THRESHOLD).sum()), int(v.notna().sum())))
        for i, (name, n_pass, n_tot) in enumerate(rows):
            ax.bar(i, n_tot, color=GRID, edgecolor="white", linewidth=0.8, zorder=2)
            ax.bar(i, n_pass, color=OUTCOME_COLOR["rescued"], edgecolor="white",
                   linewidth=0.8, zorder=3)
            ax.text(i, n_tot, f"{n_pass}/{n_tot}", ha="center", va="bottom", fontsize=8)
            xt.append(name)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(xt)
        ax.set_title(f"{lab}\n{len(checked)} of {len(st)} seeds needed reversion",
                     fontsize=9)
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].set_ylabel("steered seeds")
    handles = [Line2D([], [], color=OUTCOME_COLOR["rescued"], linewidth=8,
                      label="pose correct (≤ 5 Å)"),
               Line2D([], [], color=GRID, linewidth=8, label="all seeds")]
    place_legend(np.atleast_1d(axes)[-1], handles=handles)
    fig.tight_layout()
    fig.savefig(outpath)
    return fig


def representative_reversion(pred: pd.DataFrame, repdf: pd.DataFrame) -> pd.DataFrame:
    """One row per unit: did its representative design go through reversion?

    The seed-level cascade counts every candidate design the engine generated,
    which is dominated by candidates that were never selected. This restricts to
    the design each unit was actually represented by. A unit counts as reverted
    when at least half of that design's seeds carry a reversion verdict, which
    is the same majority rule the engine applies.
    """
    steered = pred[(pred.kind == "steered") & (pred.status == "ok")]
    recs = []
    for _, r in repdf[~repdf.steering_skipped].iterrows():
        grp = steered[(steered.unit == r["unit"]) & (steered.base == r["rep_design"])]
        if not len(grp):
            continue
        verd = grp.reversion_verdict.fillna("").astype(str).str.strip()
        n_verd = int((verd != "").sum())
        recs.append({"unit": r["unit"], "rep_design": r["rep_design"],
                     "n_seeds": len(grp), "n_with_verdict": n_verd,
                     "reverted": n_verd * 2 >= len(grp),
                     "rep_ra_eff": r["rep_ra_eff"],
                     "pose_correct": r["rep_ra_eff"] <= RA_EFF_THRESHOLD})
    return pd.DataFrame(recs)


def fig_reversion_representative(groups: dict, outpath, figsize=(7.0, 4.2)):
    """The reversion cascade at the level the rest of the analysis uses.

    `groups` is {label -> (per-seed frame, representative table)}. One bar per
    unit group rather than per candidate seed, so the counts are comparable
    with every other representative-level figure.
    """
    fig, axes = plt.subplots(1, len(groups), figsize=figsize, sharey=True)
    for ax, (lab, (pred, repdf)) in zip(np.atleast_1d(axes), groups.items()):
        rr = representative_reversion(pred, repdf)
        bars = [("no reversion\nneeded", rr[~rr.reverted]), ("reverted", rr[rr.reverted])]
        for i, (name, sub) in enumerate(bars):
            n_tot, n_pass = len(sub), int(sub.pose_correct.sum())
            ax.bar(i, n_tot, color=GRID, edgecolor="white", linewidth=0.8, zorder=2)
            ax.bar(i, n_pass, color=OUTCOME_COLOR["rescued"], edgecolor="white",
                   linewidth=0.8, zorder=3)
            ax.text(i, n_tot, f"{n_pass}/{n_tot}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(bars)))
        ax.set_xticklabels([n for n, _ in bars])
        ax.set_title(lab, fontsize=10)
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].set_ylabel("input complexes")
    handles = [Line2D([], [], color=OUTCOME_COLOR["rescued"], linewidth=8,
                      label="pose correct (≤ 5 Å)"),
               Line2D([], [], color=GRID, linewidth=8, label="all input complexes")]
    place_legend(np.atleast_1d(axes)[-1], handles=handles)
    fig.tight_layout()
    fig.savefig(outpath)
    return fig


RANK_METRICS = ["iptm", "ipsae_min"]


# What each arm actually predicted for a unit, which decides whether its two
# points describe the same molecule. Ordered clean cases first.
ARM_GROUPS = ["initial in both", "steered in both", "steered in one arm only"]
ARM_GROUP_COLOR = {"initial in both": "#0072B2",
                   "steered in both": "#009E73",
                   "steered in one arm only": "#E69F00"}


def arm_groups(unconstrained, constrained) -> pd.Series:
    """Unit -> which of ARM_GROUPS it falls in.

    Constraints change the cold start, so a unit can steer in one arm and
    short-circuit in the other. Those units carry a sequence difference on top
    of the constraint difference and cannot be read as a constraint effect. The
    two same-in-both groups can, and they are kept apart rather than pooled so
    that steering is not assumed to be irrelevant before it has been shown.
    """
    a = unconstrained.steering_skipped.astype(bool)
    b = constrained.reindex(unconstrained.index).steering_skipped.astype(bool)
    return pd.Series(np.select([a & b, ~a & ~b],
                               ARM_GROUPS[:2], default=ARM_GROUPS[2]),
                     index=unconstrained.index)


def fig_constraint_level_and_rank(unconstrained, constrained, groups, outpath,
                                  metrics=RANK_METRICS, size=34, figsize=(8.4, 8.0)):
    """What constraints do to an interface metric's level, and to its order.

    One cohort, split by what each arm did to the unit. Where both arms took
    the same route the sequence is identical on the two axes and the whole
    displacement is the constraints; where they diverged it is not.

    The rows answer different questions and both are needed. Inflating every
    score does not on its own stop a metric ranking designs, which needs only
    the order to survive, so the level row is read against the rank row beneath
    it. Ranks are within cohort and computed over all plotted units, not within
    each group, since the ranking a campaign would actually use is over the
    whole cohort.

    `groups` is a Series of ARM_GROUPS values indexed by unit.
    """
    fig, axes = plt.subplots(2, len(metrics), figsize=figsize)
    present = [(g, groups.index[groups == g]) for g in ARM_GROUPS
               if (groups == g).any()]
    for col, m in enumerate(metrics):
        x_raw = unconstrained.loc[groups.index, f"rep_{m}"].astype(float)
        y_raw = constrained.loc[groups.index, f"rep_{m}"].astype(float)
        rows = (("score", x_raw, y_raw, lambda d, _m=m: FMT[_m].format(np.median(d))),
                ("within-cohort rank", x_raw.rank(), y_raw.rank(), None))
        for row, (kind, x, y, fmt) in enumerate(rows):
            ax = axes[row, col]
            titles = []
            for g, idx in present:
                ax.scatter(x[idx], y[idx], s=size, color=ARM_GROUP_COLOR[g],
                           alpha=0.75, zorder=3, edgecolor="white", linewidth=0.6,
                           label=f"{g} (n = {len(idx)})")
                stat = (fmt((y - x)[idx]) if fmt
                        else f"ρ = {spearman(x[idx], y[idx]):+.2f}")
                titles.append(f"{g}: {stat}")
            lims = [v for v in np.concatenate([x.values, y.values]) if np.isfinite(v)]
            lo, hi = min(lims), max(lims)
            pad = 0.05 * (hi - lo)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED,
                    linestyle="--", linewidth=1.2, zorder=1)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("negsteer")
            ax.set_ylabel("negsteer + constraints")
            ax.set_title(f"{METRICS[m]}, {kind}\n" + "\n".join(titles), fontsize=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(h_pad=2.4, rect=(0, 0, 1, 0.95))
    fig.savefig(outpath)
    return fig


def fig_rank_agreement(pairs: dict, outpath, metrics=RANK_METRICS,
                       size=34, figsize=(8.4, 4.2)):
    """Does adding constraints reorder the units, or just raise every score?

    The inflation figure shows the level of each interface metric moving. That
    on its own does not stop the metric being useful, because ranking designs
    only needs the ORDER to survive. This ranks the pose-matched units within
    each cohort under each variant and plots one against the other, so a metric that
    is merely shifted lands on the diagonal and a metric that is scrambled does
    not. Ranks are within cohort, since the two cohorts differ in both size and
    baseline and a pooled rank would mix the two effects.
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    matched = {lab: (*pose_matched_pairs(u, c), col)
               for lab, (u, c, col) in pairs.items()}
    for ax, m in zip(np.atleast_1d(axes), metrics):
        n_max, titles = 0, []
        for lab, (a, b, col) in matched.items():
            ok = a[f"rep_{m}"].notna() & b[f"rep_{m}"].notna()
            x = a.loc[ok, f"rep_{m}"].rank().to_numpy()
            y = b.loc[ok, f"rep_{m}"].rank().to_numpy()
            ax.scatter(x, y, s=size, color=col, alpha=0.65, zorder=3,
                       edgecolor="white", linewidth=0.6, label=f"{lab} (n = {len(x)})")
            n_max = max(n_max, len(x))
            titles.append(f"{lab}: ρ = {spearman(x, y):+.2f}")
        ax.plot([0, n_max + 1], [0, n_max + 1], color=MUTED, linestyle="--",
                linewidth=1.2, zorder=1)
        ax.set_xlim(0, n_max + 1)
        ax.set_ylim(0, n_max + 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("rank, negsteer")
        ax.set_ylabel("rank, negsteer + constraints")
        ax.set_title(f"{METRICS[m]}\n" + "\n".join(titles), fontsize=9)

    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(outpath)
    return fig


def outcome_contrast_table(cohorts: dict, by_budget=(3.0, 6.0)) -> pd.DataFrame:
    """Rescued against still-wrong, pooled and within each populated budget.

    Rescue and mutation budget are not independent, so a pooled difference could
    in principle be a budget difference wearing a rescue label. Stratifying by
    the two well-populated budgets removes that reading.
    """
    rows = []
    for lab, df in cohorts.items():
        s = _steered(df)
        strata = [("pooled", s)] + [(f"{int(b)} mutations", s[s.n_mutations == b])
                                    for b in by_budget]
        for stratum, sub in strata:
            by = {o: sub[sub.outcome == o] for o in CONTRAST_OUTCOMES}
            for m in METRICS:
                a, b_ = (by[o][f"d_{m}"].values for o in CONTRAST_OUTCOMES)
                f = FMT[m]
                rows.append({
                    "cohort": lab, "stratum": stratum, "metric": METRICS[m],
                    "n rescued": len(a), "n still wrong": len(b_),
                    "median Δ rescued": f.format(np.median(a)) if len(a) else "",
                    "median Δ still wrong": f.format(np.median(b_)) if len(b_) else "",
                    "p": round(rank_sum_p(a, b_), 4),
                })
    return pd.DataFrame(rows)
