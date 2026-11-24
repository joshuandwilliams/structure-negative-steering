#!/usr/bin/env python3
"""Stage Boltz-2's benchmark ra_eff into boltz2_benchmark_comparison.csv.

The unrestrained and reference-restrained series in 02-model-choice.qmd come
from the sibling structure-prediction-benchmarking repo rather than from this
one. That staging used to be manual, and the committed table had drifted from
its source. The restrained column held the minimum ra_eff over all 25
predictions and the unrestrained column matched no rule in the source at all,
so the figure showed a restrained series picked by looking at the answer.

combined_metrics.csv already carries one row per target per variant, each the
most confident of that variant's 25 predictions selected on mean pLDDT, which
is the rule the benchmark chapter reports. Copying that row is all the staging
needed.

Usage:
    python3 analysis/02-model-choice/stage_boltz2_benchmark.py [--metrics PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_METRICS = (HERE.parent.parent.parent / "structure-prediction-benchmarking"
                   / "data" / "metrics" / "combined_metrics.csv")
RA_COL = "rmsd_effector_receptor_aligned"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    args = ap.parse_args()

    df = pd.read_csv(args.metrics)
    df[RA_COL] = pd.to_numeric(df[RA_COL], errors="coerce")

    def series(model, msa):
        sub = df[(df["model"] == model) & (df["msa"] == msa)]
        return sub.set_index("pdb")[RA_COL].round(2)

    out = pd.DataFrame({
        "boltz2_unconstrained_ra_eff": series("boltz2", "no_msa"),
        "boltz2_constrained_ra_eff": series("boltz2_constrained", "no_msa"),
    })
    out.index.name = "target"
    out = out.sort_index().reset_index()

    path = HERE / "boltz2_benchmark_comparison.csv"
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    for col in ("boltz2_unconstrained_ra_eff", "boltz2_constrained_ra_eff"):
        print(f"{col}: {(out[col] < 5).sum()} / {len(out)} under 5 A")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
