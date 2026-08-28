#!/usr/bin/env python3
"""Where the campaign's steering mutations end up relative to the effector.

`compute_metrics.classify_mutation_reliance` calls a steered design
CONTAMINATED when at least one steering mutation is in heavy-atom contact with
the effector in the predicted pose. The wet-lab construct does not carry those
residues, so a contaminated pose is partly held up by a molecule that will never
be made, and the reversion pass exists to check whether it survives without
them.

Contamination is common enough in this campaign to be worth characterising
rather than treating as an edge case. This walks the unconstrained campaign
arm and answers, per steered prediction: whether it is contaminated, how many
mutations are involved, whether reversion was even eligible to run, and where
the contaminating positions sit relative to the two regions that matter --
the RFDiffusion design region, which steering is forbidden to touch, and the
intended receptor-effector interface.

Coordinate systems, which differ between the files this reads:
  * contact residues and mutation positions are 1-based on the receptor chain
  * `<design>_design_region.txt` is 1-based
  * `<design>_true_interface.txt` is 0-based, and is shifted here

Outputs (next to this file):
  campaign_data/contamination_per_prediction.csv

Usage:
  python contamination_extract.py --run-root PATH --indices PATH
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
HPC = Path(
    "/Volumes/HPC-Home/receptor_design/receptor-resurfacing-pipeline"
    "/experiments/campaigns/pikp1_avrpikf/runs/crystal_full_test_contig"
)
OUT_CSV = HERE / "campaign_data" / "contamination_per_prediction.csv"
RA_EFF_THRESHOLD = 5.0   # the gate that decides whether reversion may run


def parse_positions(text: str, base: int) -> set[int]:
    """A design-region or true-interface file -> a set of 1-based positions.

    Both formats are comment-headed lists, but one writes ranges on a single
    line and the other one index per line, and they disagree on base.
    """
    out: set[int] = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        for token in line.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                lo, hi = (int(v) for v in token.split("-", 1))
                out.update(range(lo, hi + 1))
            else:
                out.add(int(token))
    return {p + (1 - base) for p in out}


def load_regions(indices: Path) -> dict[str, dict[str, set[int]]]:
    """{design -> {"design_region": set, "true_interface": set}}, all 1-based."""
    regions: dict[str, dict[str, set[int]]] = {}
    for path in sorted(indices.glob("design_*_*.txt")):
        m = re.match(r"(design_\d+)_(design_region|true_interface)\.txt$", path.name)
        if not m:
            continue
        design, kind = m.groups()
        base = 1 if kind == "design_region" else 0
        regions.setdefault(design, {})[kind] = parse_positions(path.read_text(), base)
    return regions


def parse_int_list(value) -> list[int]:
    return [int(v) for v in re.findall(r"-?\d+", str(value or ""))]


def build(run_root: Path, indices: Path) -> pd.DataFrame:
    regions = load_regions(indices)
    runs = sorted((run_root / "results" / "negative_steering" / "runs").glob(
        "*/raw_per_seed_results.csv"))
    rows = []
    for path in runs:
        unit = path.parent.name
        if "control" in unit:
            continue
        # The RFDiffusion design a sequence came from owns the region files.
        parent = re.match(r"(design_\d+)_seq_\d+$", unit)
        reg = regions.get(parent.group(1), {}) if parent else {}
        design_region = reg.get("design_region", set())
        true_interface = reg.get("true_interface", set())
        for _, r in pd.read_csv(path).iterrows():
            base = str(r["design"]).rsplit("_s", 1)[0]
            if base == "initial":
                continue
            hits = set(parse_int_list(r.get("steered_mutated_contact_positions")))
            muts = [int(p) for p in re.findall(
                r"[A-Z](\d+)[A-Z]", str(r.get("steered_mutations_aa") or ""))]
            ra = pd.to_numeric(r.get("steered_ra_eff_vs_truth"), errors="coerce")
            rows.append({
                "unit": unit, "design": base, "seed_index": r.get("seed_index"),
                "n_mutations": len(muts),
                "n_contaminating": len(hits),
                "contaminated": len(hits) > 0,
                "pose_passes": bool(ra < RA_EFF_THRESHOLD) if pd.notna(ra) else False,
                "reversion_verdict": str(r.get("reversion_verdict") or "").strip(),
                "in_design_region": len(hits & design_region),
                "in_true_interface": len(hits & true_interface),
                "outside_both": len(hits - design_region - true_interface),
                "contaminating_positions": ",".join(str(p) for p in sorted(hits)),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, default=HPC)
    ap.add_argument("--indices", type=Path, default=None,
                    help="negative_steering/indices dir (default: under run root)")
    args = ap.parse_args()
    indices = args.indices or (args.run_root / "results" / "negative_steering" / "indices")
    for p in (args.run_root, indices):
        if not p.is_dir():
            raise SystemExit(f"not found: {p}")

    df = build(args.run_root, indices)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"{len(df)} steered predictions, {int(df.contaminated.sum())} contaminated "
          f"-> {OUT_CSV}")


if __name__ == "__main__":
    main()
