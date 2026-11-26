#!/usr/bin/env python3
"""
Cold-start and representative ra_eff for all 18 targets, sourced from the
n=50 dose-sweep arm (experiments/dose-sweep/n50/) rather than the shipped
20-set benchmark (experiments/benchmarking/). 50 sets is the tighter bound
negative steering has actually demonstrated on this benchmark (15/18 vs
12/18 at n=20, see 02-model-choice.qmd), so figures meant to show what
steering can do should use it.

cold_start_ra_eff is read directly from each target's own n50
cycle_0/plan.json (steering dose does not change the cold-start prediction
itself, but each run tree carries its own copy, so this stays paired within
one run rather than mixing run trees). That field is seed 0 alone.

cold_start_ra_eff_median is the median over the three cold-start seeds, so
it summarises the unsteered prediction the same way select_representative()
summarises the steered one. Pair it with rep_ra_eff whenever the two are
plotted against each other, or a target that never entered steering comes
out off the y = x diagonal purely from comparing one seed against a median
of three.

rep_ra_eff / rep_n_pass / rep_n_seeds reuse select_representative()
(the reliability-first, reversion-corrected pick used throughout this repo).

No constrained series here: experiments/benchmarking/constrained/ exists
only at n=20; there is no n50 arm for reference-restrained + negative
steering to pair against.

Output: n50_pose_metrics.csv (committed), read by 01-pose-rescue.qmd.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DOSE_SWEEP = REPO_ROOT / "experiments" / "dose-sweep" / "n50"

sys.path.insert(0, str(HERE))
from make_plots import RA_EFF_THRESHOLD, TARGETS, select_representative  # noqa: E402


def main():
    rows = []
    for target in TARGETS:
        rundir = DOSE_SWEEP / target / "run"
        plan_path = rundir / "cycle_0" / "plan.json"

        cold = None
        cold_median = None
        steered = None
        if plan_path.is_file():
            try:
                plan = json.loads(plan_path.read_text())
                cold = plan.get("initial_receptor_aligned_effector_rmsd")
                seeds = [s.get("ra_eff") for s in plan.get("cold_start_seeds", [])]
                seeds = [v for v in seeds if v is not None]
                cold_median = statistics.median(seeds) if seeds else None
                # Steering is skipped when all three cold-start seeds are
                # correctly posed and unmisfolded, which is what
                # cold_start_all_clean records.
                all_clean = plan.get("cold_start_all_clean")
                steered = (not all_clean) if all_clean is not None else None
            except Exception:
                cold = None
                cold_median = None
                steered = None

        sel = select_representative(rundir)
        rep_ra_eff = sel["rep_ra_eff"] if sel else None
        rep_n_pass = sel["rep_n_pass"] if sel else None
        rep_n_seeds = sel["rep_n_seeds"] if sel else None
        rep_design = sel["rep_design"] if sel else None

        change = (rep_ra_eff - cold) if (rep_ra_eff is not None and cold is not None) else None

        rows.append({
            "target": target,
            "steering_triggered": steered,
            "cold_start_ra_eff": cold,
            "cold_start_ra_eff_median": cold_median,
            "rep_ra_eff": rep_ra_eff,
            "change_ra_eff": change,
            "rep_n_pass": rep_n_pass,
            "rep_n_seeds": rep_n_seeds,
            "rep_design": rep_design,
        })

    df = pd.DataFrame(rows)
    out = HERE / "n50_pose_metrics.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    n_pass = int((df["rep_ra_eff"] < RA_EFF_THRESHOLD).sum())
    print(f"\n{n_pass} / {len(df)} pass at n=50")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
