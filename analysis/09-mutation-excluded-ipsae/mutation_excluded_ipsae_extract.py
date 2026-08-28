#!/usr/bin/env python3
"""ipSAE recomputed with the steering mutations removed from the calculation.

The campaign ranks designs on ipSAE. The agroinfiltrated construct does not carry
the steering mutations, so a score those residues contributed to describes a
molecule that will not be made. The shipped implementation offers no protection
against that: ``compute_metrics.compute_ipsae_one_direction`` takes EVERY receptor
residue as a candidate reference residue, thresholds on PAE rather than distance,
and reports the maximum over reference residues. Nothing in it refers to the
intended interface.

Empirically the winning residue is never a mutated one, but that is an observed
outcome rather than a guarantee. This recomputes the metric over restricted
residue sets so the guarantee holds by construction, and reports whether the
restriction changes how the cohort ranks.

Three variants per prediction and per direction:

  full        every receptor residue, reproducing the shipped metric. Present as
              a control: it must agree with the stored ``ipsae_min``, otherwise
              the reimplementation or the file mapping is wrong.
  no_mut      mutated receptor positions dropped, both as reference residues in
              the receptor->effector direction and as partner residues in the
              reverse one, since a mutation contributes to the mean either way.
  interface   only intended-interface receptor residues kept, same both ways.

The ipSAE kernel is byte-faithful to
``receptor-resurfacing-pipeline/bin/compute_metrics.py:57-114``: d0 is computed
per reference residue from that residue's own partner count, clamped at 0.5, and
the reported score is the max over reference residues, not the mean.

Output (next to this file):
  mutation_excluded_ipsae.csv   one row per (arm, unit, design, seed)

Usage:
  python mutation_excluded_ipsae_extract.py --run-root PATH --indices PATH
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
HPC = Path(
    "/Volumes/HPC-Home/receptor_design/receptor-resurfacing-pipeline"
    "/experiments/campaigns/pikp1_avrpikf/runs/crystal_full_test_contig"
)
ARMS = {"negsteer": "results", "negsteer+constraints": "results_constrained"}
PAE_CUTOFF = 10.0
OUT_CSV = HERE / "mutation_excluded_ipsae.csv"


def d0(n: int) -> float:
    """TM-score d0 on the interacting-set size, clamped at 0.5 as upstream."""
    if n < 16:
        return 0.5
    return max(1.24 * ((n - 15) ** (1.0 / 3.0)) - 1.8, 0.5)


def ipsae_direction(pae_ij: np.ndarray, cutoff: float = PAE_CUTOFF) -> float:
    """Max over reference residues of the mean pTM kernel over its partners."""
    best = 0.0
    for i in range(pae_ij.shape[0]):
        below = pae_ij[i][pae_ij[i] < cutoff]
        if below.size == 0:
            continue
        score = float((1.0 / (1.0 + (below / d0(below.size)) ** 2)).mean())
        best = max(best, score)
    return best


def ipsae_min(pae: np.ndarray, n_rec: int, keep: np.ndarray | None) -> float:
    """min over the two directions, with `keep` masking receptor residues.

    `keep` is a boolean mask over receptor residues. It is applied to the
    receptor axis in both directions, so a dropped residue is neither a
    reference residue going one way nor a partner going the other.
    """
    ab = pae[:n_rec, n_rec:]
    ba = pae[n_rec:, :n_rec]
    if keep is not None:
        ab, ba = ab[keep, :], ba[:, keep]
    if ab.size == 0 or ba.size == 0:
        return 0.0
    return round(min(ipsae_direction(ab), ipsae_direction(ba)), 4)


def parse_positions(text: str, base: int) -> set[int]:
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


def load_interfaces(indices: Path) -> dict[str, set[int]]:
    """{RFDiffusion design -> intended interface, 1-based receptor positions}."""
    out = {}
    for path in sorted(indices.glob("design_*_true_interface.txt")):
        design = re.match(r"(design_\d+)_true_interface\.txt$", path.name).group(1)
        out[design] = parse_positions(path.read_text(), base=0)
    return out


def pae_file(run_root: Path, arm_dir: str, unit: str, design: str) -> Path | None:
    """Prediction directory for one steered (design, seed), or None if absent."""
    base = run_root / arm_dir / "negative_steering" / "runs" / unit / "cycle_0"
    sub = base / design if design.startswith("initial") else base / "steered" / design
    p = sub / "boltz_results_input" / "predictions" / "input" / "pae_input_model_0.npz"
    return p if p.exists() else None


def receptor_length(run_root: Path, arm_dir: str, unit: str) -> int | None:
    pj = run_root / arm_dir / "negative_steering" / "runs" / unit / "cycle_0" / "plan.json"
    if not pj.exists():
        return None
    return len(json.loads(pj.read_text()).get("wild_type_receptor_seq", "")) or None


def build(run_root: Path, indices: Path, units: list[str]) -> pd.DataFrame:
    interfaces = load_interfaces(indices)
    rows = []
    for arm, arm_dir in ARMS.items():
        for unit in units:
            n_rec = receptor_length(run_root, arm_dir, unit)
            per_seed = (run_root / arm_dir / "negative_steering" / "runs" / unit
                        / "raw_per_seed_results.csv")
            if n_rec is None or not per_seed.exists():
                continue
            parent = re.match(r"(design_\d+)_seq_\d+$", unit).group(1)
            interface = interfaces.get(parent, set())
            for _, r in pd.read_csv(per_seed).iterrows():
                design = str(r["design"])
                path = pae_file(run_root, arm_dir, unit, design)
                if path is None:
                    continue
                pae = np.load(path)["pae"]
                muts = {int(p) for p in re.findall(
                    r"[A-Z](\d+)[A-Z]", str(r.get("steered_mutations_aa") or ""))}
                idx = np.arange(1, n_rec + 1)          # 1-based receptor positions
                rows.append({
                    "arm": arm, "unit": unit, "design": design,
                    "base": design.rsplit("_s", 1)[0],
                    "seed_index": r.get("seed_index"),
                    "n_mutations": len(muts),
                    "stored_ipsae_min": r.get("steered_ipsae_min"),
                    "ipsae_full": ipsae_min(pae, n_rec, None),
                    "ipsae_no_mut": ipsae_min(pae, n_rec, ~np.isin(idx, list(muts))),
                    "ipsae_interface": ipsae_min(
                        pae, n_rec, np.isin(idx, list(interface))) if interface else np.nan,
                })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, default=HPC)
    ap.add_argument("--indices", type=Path, default=None)
    ap.add_argument("--units", type=Path, default=None,
                    help="file of unit names, one per line; default is all present")
    args = ap.parse_args()
    indices = args.indices or (args.run_root / "results" / "negative_steering" / "indices")
    if args.units:
        units = [u for u in args.units.read_text().split() if u]
    else:
        units = sorted(p.name for p in (
            args.run_root / "results" / "negative_steering" / "runs").iterdir()
            if p.is_dir() and "control" not in p.name)

    df = build(args.run_root, indices, units)
    df.to_csv(OUT_CSV, index=False)
    ok = np.isclose(df.ipsae_full, pd.to_numeric(df.stored_ipsae_min, errors="coerce"),
                    atol=5e-3, equal_nan=True)
    print(f"{len(df)} predictions -> {OUT_CSV}")
    print(f"control: ipsae_full matches the stored value for {int(ok.sum())}/{len(df)}")


if __name__ == "__main__":
    main()
