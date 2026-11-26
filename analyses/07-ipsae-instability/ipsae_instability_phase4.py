#!/usr/bin/env python3
"""The decisive test: does d0 rescaling drive the rank instability?

Three ipSAE variants are computed for the pose-matched units, all from the
same PAE matrices, so the only thing that changes between them is the part of
the formula under test:

  as_is    upstream ipSAE -- per-residue d0, max over residues
  d0_swap  constrained variant rescored with the UNCONSTRAINED variant's per-residue
           d0. Only the length normalisation is frozen; the PAE values are
           the constrained variant's own. If rank agreement jumps, d0 was the
           cause.
  mean     mean over reference residues instead of max, with each variant's own
           d0. Isolates the argmax-residue operator, which is the other way
           the reported number can move without any PAE changing.

Writes ipsae_phase4.csv, one row per pose-matched unit.

Usage:
  python ipsae_instability_phase4.py [--cutoff 10.0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
CONF = ANALYSIS / "04-confidence-cost"          # owns confidence_common.py
CAMPAIGN = ANALYSIS / "05-campaign-confidence"  # owns campaign_data/
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CONF))

import confidence_common as cc  # noqa: E402
from ipsae_instability_extract import (  # noqa: E402
    VARIANTS,
    _d0,
    pae_path,
    receptor_length,
)

OUT = HERE / "ipsae_phase4.csv"
UNCON, CON = "negsteer", "negsteer+constraints"


def per_residue(pae_ij: np.ndarray, cutoff: float,
                d0_override: np.ndarray | None = None):
    """Per-residue ipSAE scores for one direction.

    `d0_override` supplies a d0 per reference residue (the other variant's), in
    which case the residue's own n_below is used only to select which partner
    columns count, not to set the normalisation.
    """
    n_i = pae_ij.shape[0]
    scores = np.zeros(n_i, dtype=float)
    nbelow = np.zeros(n_i, dtype=int)
    for i in range(n_i):
        row = pae_ij[i]
        below = row[row < cutoff]
        nbelow[i] = below.size
        if below.size == 0:
            continue
        d0 = _d0(below.size) if d0_override is None else float(d0_override[i])
        if d0 <= 0:
            continue
        scores[i] = float((1.0 / (1.0 + (below / d0) ** 2)).mean())
    return scores, nbelow


def blocks(results_dir, unit, kind, design, cutoff):
    """(ab, ba) interchain PAE blocks for one prediction, or None."""
    p = pae_path(results_dir, unit, kind, design)
    n_rec = receptor_length(results_dir, unit)
    if p is None or n_rec is None:
        return None
    pae = np.load(p)["pae"]
    return pae[:n_rec, n_rec:], pae[n_rec:, :n_rec]


def rep_rows(per_seed: pd.DataFrame, unit: str, rep_design: str) -> pd.DataFrame:
    """The per-seed rows that the representative median is taken over."""
    if rep_design == "initial":
        return per_seed[(per_seed.unit == unit) & (per_seed.kind == "cold_start")]
    return per_seed[(per_seed.unit == unit) & (per_seed.base == rep_design)
                    & (per_seed.kind == "steered")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=float, default=10.0)
    args = ap.parse_args()

    seed_tab, res_dir = {}, {}
    for variant, (rd, csv_path) in VARIANTS.items():
        seed_tab[variant] = pd.read_csv(csv_path)
        res_dir[variant] = rd
    reps = {variant: cc.build_representative_table(seed_tab[variant]) for variant in VARIANTS}
    a, b = cc.pose_matched_pairs(reps[UNCON], reps[CON])
    units = list(a.index)
    print(f"pose-matched units: {len(units)}", flush=True)

    out = []
    for k, unit in enumerate(units, 1):
        rec = {"unit": unit,
               "rep_design_uncon": a.loc[unit, "rep_design"],
               "rep_design_con": b.loc[unit, "rep_design"],
               "rep_ra_eff_uncon": a.loc[unit, "rep_ra_eff"],
               "rep_ra_eff_con": b.loc[unit, "rep_ra_eff"],
               "ipsae_min_stored_uncon": a.loc[unit, "rep_ipsae_min"],
               "ipsae_min_stored_con": b.loc[unit, "rep_ipsae_min"]}

        # Per-variant per-seed scores. The representative value upstream is the
        # MEDIAN over that base's seeds, so every variant is reduced the same
        # way to stay comparable with the published number.
        per_variant = {}
        for variant in (UNCON, CON):
            rd = res_dir[variant]
            rows = rep_rows(seed_tab[variant], unit, rec[f"rep_design_{'uncon' if variant == UNCON else 'con'}"])
            acc = []
            for r in rows.itertuples(index=False):
                blk = blocks(rd, unit, r.kind, r.design, args.cutoff)
                if blk is None:
                    continue
                ab, ba = blk
                sab, nab = per_residue(ab, args.cutoff)
                sba, nba = per_residue(ba, args.cutoff)
                acc.append({"design": r.design, "ab": ab, "ba": ba,
                            "sab": sab, "sba": sba, "nab": nab, "nba": nba})
            per_variant[variant] = acc

        if not per_variant[UNCON] or not per_variant[CON]:
            rec["status"] = "missing_pae"
            out.append(rec)
            continue

        def med(variant, fn):
            return float(np.median([fn(d) for d in per_variant[variant]]))

        for variant, tag in ((UNCON, "uncon"), (CON, "con")):
            rec[f"ipsae_min_asis_{tag}"] = med(
                variant, lambda d: min(d["sab"].max(), d["sba"].max()))
            rec[f"ipsae_min_mean_{tag}"] = med(
                variant, lambda d: min(d["sab"].mean(), d["sba"].mean()))
            rec[f"n_below_argmax_ab_{tag}"] = med(
                variant, lambda d: d["nab"][int(np.argmax(d["sab"]))])
            rec[f"n_below_argmax_ba_{tag}"] = med(
                variant, lambda d: d["nba"][int(np.argmax(d["sba"]))])
            rec[f"d0_argmax_ab_{tag}"] = med(
                variant, lambda d: _d0(int(d["nab"][int(np.argmax(d["sab"]))])))
            rec[f"L_pairs_{tag}"] = med(
                variant, lambda d: int((d["ab"] < args.cutoff).sum()
                                   + (d["ba"] < args.cutoff).sum()))
            rec[f"mean_pae_passing_{tag}"] = med(
                variant, lambda d: float(np.concatenate(
                    [d["ab"].ravel(), d["ba"].ravel()])[
                        np.concatenate([d["ab"].ravel(), d["ba"].ravel()]) < args.cutoff].mean())
                if ((d["ab"] < args.cutoff).sum() + (d["ba"] < args.cutoff).sum()) else np.nan)

        # d0 swap: rescore the constrained variant using the unconstrained variant's
        # per-residue d0. Seeds are paired by position after sorting, and the
        # unconstrained median-seed d0 profile is used when counts differ, so
        # the swap never depends on an arbitrary seed pairing.
        d0_ab = np.median(np.stack([[_d0(int(n)) for n in d["nab"]]
                                    for d in per_variant[UNCON]]), axis=0)
        d0_ba = np.median(np.stack([[_d0(int(n)) for n in d["nba"]]
                                    for d in per_variant[UNCON]]), axis=0)
        swapped = []
        for d in per_variant[CON]:
            sab, _ = per_residue(d["ab"], args.cutoff, d0_override=d0_ab)
            sba, _ = per_residue(d["ba"], args.cutoff, d0_override=d0_ba)
            swapped.append(min(sab.max(), sba.max()))
        rec["ipsae_min_d0swap_con"] = float(np.median(swapped))
        rec["status"] = "ok"
        out.append(rec)
        if k % 5 == 0:
            print(f"  {k}/{len(units)}", flush=True)

    df = pd.DataFrame(out)
    df.to_csv(OUT, index=False)
    ok = df[df.status == "ok"]
    print(f"\nWROTE {OUT} ({len(df)} rows, {len(ok)} ok)")
    for tag in ("uncon", "con"):
        d = (ok[f"ipsae_min_asis_{tag}"] - ok[f"ipsae_min_stored_{tag}"]).abs()
        print(f"  as_is vs stored [{tag}]: max|delta| = {d.max():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
