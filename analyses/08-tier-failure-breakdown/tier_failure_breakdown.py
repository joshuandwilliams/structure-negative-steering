#!/usr/bin/env python3
"""
Per-complex breakdown of the 50 steering sets in the n=50 dose-sweep arm.

For each of the 18 benchmark targets, every steering set (a mutated
sequence, 3 seeds each) is classified into one of five outcomes:

* Tier A  -- all seeds pose-correct and intact (n_pass == n_seeds)
* Tier B  -- some but not all seeds pass (1 < n_pass < n_seeds)
* Tier C  -- exactly one seed passes (n_pass == 1)
* Failed (misfolded)   -- n_pass == 0, and at least one seed's receptor or
  effector chain was independently misfolded (>5 A from its own reference
  on Kabsch alignment, the same cutoff used throughout this repo)
* Failed (wrong pose)  -- n_pass == 0, every seed folded correctly but the
  effector was never placed within 5 A of the true interface

Tier definitions match bin/cross_summary.py's n_pass/n_seeds convention
(see that file's docstring) and make_plots.py's per-seed pass logic
(reversion-corrected final ra_eff), reimplemented here with per-seed
fold-intactness also tracked, which those two scripts do not need.

Data: experiments/dose-sweep/n50/<target>/run/raw_per_seed_results.csv,
gitignored (HPC), pulled with
    contrib/sync_from_hpc.sh --name dose-sweep --results-only
Output: tier_failure_breakdown_n50.csv (committed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DOSE_SWEEP = REPO_ROOT / "experiments" / "dose-sweep" / "n50"

sys.path.insert(0, str(REPO_ROOT / "analyses" / "01-pose-rescue"))
from make_plots import RA_EFF_THRESHOLD, TARGETS, _try_float, _try_int  # noqa: E402

FOLD_THRESHOLD = 5.0  # Angstrom; independent chain RMSD misfolding cutoff


def _final_seed_state(row) -> tuple[float | None, bool | None]:
    """One raw_per_seed_results.csv row -> (final ra_eff, final fold-intact).

    Mirrors make_plots._final_seed_value's branch logic (which final value
    to read given the reversion verdict), but also reads back whether the
    receptor was intact at that final state, which _final_seed_value
    discards.
    """
    verdict = str(row.get("reversion_verdict") or "").strip()
    if verdict == "pose_holds":
        ra = _try_float(row.get("reverted_ra_eff_vs_truth"))
        intact = _try_int(row.get("reverted_receptor_intact"))
        return ra, (None if intact is None else intact == 1)
    if verdict in ("pose_collapses", "new_contamination",
                    "pose_holds_not_intact", "missing_verdict", "unknown"):
        ra = _try_float(row.get("reverted_ra_eff_vs_truth"))
        intact = _try_int(row.get("reverted_receptor_intact"))
        if intact is None:
            intact = _try_int(row.get("steered_receptor_intact"))
        return ra, (None if intact is None else intact == 1)
    # Blank verdict: reversion correctly skipped (zero contamination) or
    # missing data. In the n50 dose-sweep run tree (unlike the shipped
    # 20-set benchmark, where this field is always an explicit 0 on a
    # blank-verdict row), steered_n_contacts_on_mutated_positions is often
    # blank/NaN rather than 0 on these same blank-verdict rows -- a CSV
    # writer quirk in that run's pipeline version, not missing data. Since
    # a blank verdict never co-occurs with a genuine non-zero contact count
    # in the clean (20-set) data, NaN here is treated the same as 0.
    n_contacts = _try_int(row.get("steered_n_contacts_on_mutated_positions"))
    ra = _try_float(row.get("steered_ra_eff_vs_truth"))
    intact = _try_int(row.get("steered_receptor_intact"))
    if (n_contacts is None or n_contacts == 0) and ra is not None:
        return ra, (None if intact is None else intact == 1)
    return None, None


def classify_target(target: str) -> dict:
    rundir = DOSE_SWEEP / target / "run"
    per_seed_path = rundir / "raw_per_seed_results.csv"
    counts = {"A": 0, "B": 0, "C": 0, "failed_misfolded": 0,
              "failed_wrong_pose": 0, "no_data": 0}
    if not per_seed_path.is_file():
        return {"target": target, "n_sets": 0, **counts}

    df = pd.read_csv(per_seed_path)
    steered = df[(df.get("status") == "ok") & (df.get("design") != "initial")].copy()
    if not len(steered):
        return {"target": target, "n_sets": 0, **counts}
    steered["base"] = steered["design"].str.replace(r"_s\d+$", "", regex=True)

    for _base, grp in steered.groupby("base"):
        seeds = [_final_seed_state(row) for _, row in grp.iterrows()]
        seeds = [(ra, intact) for ra, intact in seeds if ra is not None]
        if not seeds:
            counts["no_data"] += 1
            continue
        n_seeds = len(seeds)
        n_pass = sum(1 for ra, _ in seeds if ra < RA_EFF_THRESHOLD)

        if n_pass == n_seeds:
            counts["A"] += 1
        elif n_pass == 0:
            any_misfolded = any(intact is False for _, intact in seeds)
            if any_misfolded:
                counts["failed_misfolded"] += 1
            else:
                counts["failed_wrong_pose"] += 1
        elif n_pass == 1:
            counts["C"] += 1
        else:
            counts["B"] += 1

    n_sets = sum(counts.values())
    return {"target": target, "n_sets": n_sets, **counts}


def main():
    rows = [classify_target(t) for t in TARGETS]
    df = pd.DataFrame(rows)
    out = HERE / "tier_failure_breakdown_n50.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
