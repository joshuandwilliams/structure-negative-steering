#!/usr/bin/env python3
"""Count correctly-posed targets (< 5 A) at each steering-set dose.

Reads experiments/dose-sweep/n{03,05,10,20,50}/<target>/run/ (results-only,
gitignored, pulled from the cluster) plus dose 0, the cold start each dose's
own cycle_0/plan.json already carries, which does not depend on
negsteer_n_designs and is read once from the n03 arm.

Writes dose_sweep_pass_counts.csv, committed so 01-pose-rescue.qmd renders
without the dose-sweep run trees present.

Usage:
    python3 analysis/01-pose-rescue/dose_sweep_pass_counts.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DOSE_SWEEP = REPO_ROOT / "experiments" / "dose-sweep"
DOSES = (3, 5, 10, 20, 50)
RA_EFF_THRESHOLD = 5.0

sys.path.insert(0, str(HERE))
from make_plots import select_representative  # noqa: E402


def main():
    targets = sorted(p.name for p in (DOSE_SWEEP / f"n{DOSES[0]:02d}").iterdir())

    cold_pass = 0
    for t in targets:
        plan_path = DOSE_SWEEP / f"n{DOSES[0]:02d}" / t / "run" / "cycle_0" / "plan.json"
        cold = json.loads(plan_path.read_text()).get("initial_receptor_aligned_effector_rmsd")
        if cold is not None and cold < RA_EFF_THRESHOLD:
            cold_pass += 1
    rows = [{"dose": 0, "n_pass": cold_pass, "n_total": len(targets)}]

    for dose in DOSES:
        n_pass = 0
        for t in targets:
            rundir = DOSE_SWEEP / f"n{dose:02d}" / t / "run"
            sel = select_representative(rundir)
            if sel is not None and sel["rep_ra_eff"] < RA_EFF_THRESHOLD:
                n_pass += 1
        rows.append({"dose": dose, "n_pass": n_pass, "n_total": len(targets)})

    df = pd.DataFrame(rows)
    out = HERE / "dose_sweep_pass_counts.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
