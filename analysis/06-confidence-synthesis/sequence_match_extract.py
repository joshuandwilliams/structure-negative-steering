#!/usr/bin/env python3
"""Record whether each campaign sequence's representative design carries the
same steering mutations under both constraint variants.

The constraint comparison in ``06-confidence-synthesis.qmd`` pairs a unit's
representative steered design across the unconstrained and constrained arms.
Those two representatives are not guaranteed to be the same molecule. Boltz-2
constraints change the cold-start prediction, the steering plan is derived from
that prediction, and a unit whose constrained cold start already passes never
steers at all. So a paired difference can be a constraint effect, a sequence
effect, or both.

The mutation identities that settle this are dropped by ``load_per_seed``,
which keeps only the confidence columns. This walks the raw per-seed tables of
both arms, resolves each unit's representative design the same way the analysis
does, and writes the two mutation sets and whether they agree.

Output (next to this file):
  sequence_match.csv   one row per campaign unit

Usage:
  python sequence_match_extract.py [--run-root PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
CONF = ANALYSIS / "04-confidence-cost"          # owns confidence_common.py
CAMPAIGN = ANALYSIS / "05-campaign-confidence"  # owns campaign_data/
HPC = Path(
    "/Volumes/HPC-Home/receptor_design/receptor-resurfacing-pipeline"
    "/experiments/campaigns/pikp1_avrpikf/runs/crystal_full_test_contig"
)
ARMS = {
    "negsteer": ("results", CAMPAIGN / "campaign_data" / "campaign_per_seed.csv"),
    "negsteer+constraints": (
        "results_constrained",
        CAMPAIGN / "campaign_data" / "campaign_per_seed_constrained.csv"),
}
OUT_CSV = HERE / "sequence_match.csv"

sys.path.insert(0, str(CONF))
import confidence_common as cc  # noqa: E402


def mutations_by_design(run_root: Path, arm_dir: str) -> dict[tuple[str, str], str]:
    """{(unit, design) -> mutation string} from one arm's raw per-seed tables.

    Keyed on the design base rather than the seed, since every seed of a design
    carries the same mutations and the representative is chosen at design level.
    """
    out: dict[tuple[str, str], str] = {}
    runs = sorted((run_root / arm_dir / "negative_steering" / "runs").glob(
        "*/raw_per_seed_results.csv"))
    for path in runs:
        unit = path.parent.name
        if "control" in unit:
            continue
        for _, row in pd.read_csv(path).iterrows():
            base = str(row["design"]).rsplit("_s", 1)[0]
            if base == "initial":
                continue
            key = (unit, base)
            if key not in out:
                out[key] = str(row.get("steered_mutations_aa") or "")
    return out


def representative_mutations(run_root: Path) -> pd.DataFrame:
    """One row per unit: each arm's representative mutation set, and agreement.

    A unit whose cold start already passed is not steered, so its representative
    is the unmutated input sequence. That is an empty mutation set, not a
    missing value, and two empty sets agree.
    """
    frames = {}
    for arm, (arm_dir, cache) in ARMS.items():
        rep = cc.build_representative_table(pd.read_csv(cache)).set_index("unit")
        muts = mutations_by_design(run_root, arm_dir)
        frames[arm] = pd.Series(
            {u: "" if r.steering_skipped else muts[(u, r.rep_design)]
             for u, r in rep.iterrows()}, dtype=object)

    df = pd.DataFrame(frames)
    df.columns = ["mutations_negsteer", "mutations_constrained"]
    as_set = df.map(lambda s: frozenset(s.split()))
    df["sequence_matched"] = as_set.mutations_negsteer == as_set.mutations_constrained
    return df.rename_axis("unit").reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, default=HPC,
                    help="campaign run directory holding both arms")
    args = ap.parse_args()
    if not args.run_root.is_dir():
        raise SystemExit(f"run root not found: {args.run_root}")

    df = representative_mutations(args.run_root)
    df.to_csv(OUT_CSV, index=False)
    n = int(df.sequence_matched.sum())
    print(f"{len(df)} units, {n} sequence-matched, {len(df) - n} not -> {OUT_CSV}")


if __name__ == "__main__":
    main()
