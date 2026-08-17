#!/usr/bin/env python3
"""Recompute ipSAE internals from the raw Boltz-2 PAE matrices.

The pipeline stores only the final ipSAE score. The length parameter that
ipSAE rescales by -- d0, derived from the count of partner residues under
the PAE cutoff -- is a local variable in ``compute_metrics.py`` and is
discarded. This walks the retained ``pae_*.npz`` files and rebuilds it, so
the rank instability between the unconstrained and constrained variants can be
attributed (or not) to that rescaling.

The ipSAE reimplementation here is byte-faithful to
``receptor-resurfacing-pipeline/bin/compute_metrics.py:57-114``:

  * d0 is computed PER REFERENCE RESIDUE from that residue's own count of
    partners under the cutoff, not once per chain.
  * the reported score is the MAX over reference residues, not the mean.

Both of those differ from a chain-level reading of the Dunbrack formula and
both are candidate causes of rank instability, so both are measured.

Outputs (next to this file):
  ipsae_internals.csv   one row per prediction, scalars
  ipsae_nbelow.npz      per-residue n_below arrays, keyed "<variant>|<unit>|<design>|<dir>"

Usage:
  python ipsae_instability_extract.py [--cutoff 10.0] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
CONF = ANALYSIS / "04-confidence-cost"          # owns confidence_common.py
CAMPAIGN = ANALYSIS / "05-campaign-confidence"  # owns campaign_data/
HPC = Path(
    "/Volumes/HPC-Home/receptor_design/receptor-resurfacing-pipeline"
    "/experiments/campaigns/pikp1_avrpikf/runs/crystal_full_test_contig"
)
VARIANTS = {
    "negsteer": ("results", CAMPAIGN / "campaign_data" / "campaign_per_seed.csv"),
    "negsteer+constraints": ("results_constrained",
                             CAMPAIGN / "campaign_data" / "campaign_per_seed_constrained.csv"),
}
OUT_CSV = HERE / "ipsae_internals.csv"
OUT_NPZ = HERE / "ipsae_nbelow.npz"


# ── ipSAE, faithful to compute_metrics.py ────────────────────────────────────
def _d0(n: int) -> float:
    """TM-score d0 on the interacting-set size, clamped at 0.5 as upstream."""
    if n < 16:
        return 0.5
    return max(1.24 * ((n - 15) ** (1.0 / 3.0)) - 1.8, 0.5)


def _direction(pae_ij: np.ndarray, cutoff: float) -> dict:
    """One direction of ipSAE plus the internals the hypothesis needs.

    Returns the upstream score (max over residues), the mean-over-residues
    variant, the per-residue n_below array, and the argmax residue's n_below
    and d0 -- the latter being the only d0 that actually reaches the reported
    number, since everything but the argmax residue is discarded.
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
        d0 = _d0(below.size)
        scores[i] = float((1.0 / (1.0 + (below / d0) ** 2)).mean())

    if not scores.size or scores.max() == 0.0:
        arg = int(np.argmax(scores)) if scores.size else -1
        return {"score": 0.0, "score_mean": float(scores.mean()) if scores.size else 0.0,
                "argmax": arg, "n_below_argmax": int(nbelow[arg]) if arg >= 0 else 0,
                "d0_argmax": _d0(int(nbelow[arg])) if arg >= 0 else np.nan,
                "nbelow": nbelow, "scores": scores}

    arg = int(np.argmax(scores))
    return {"score": float(scores.max()),
            "score_mean": float(scores.mean()),
            "argmax": arg,
            "n_below_argmax": int(nbelow[arg]),
            "d0_argmax": _d0(int(nbelow[arg])),
            "nbelow": nbelow,
            "scores": scores}


def analyse(pae: np.ndarray, n_rec: int, cutoff: float) -> dict:
    """Both directions plus whole-interface summaries."""
    ab = pae[:n_rec, n_rec:]
    ba = pae[n_rec:, :n_rec]
    A = _direction(ab, cutoff)
    B = _direction(ba, cutoff)

    inter = np.concatenate([ab.ravel(), ba.ravel()])
    passing = inter[inter < cutoff]
    # Chain-level L: the reading in the original hypothesis, i.e. how many
    # interchain pairs clear the cutoff at all. Not what the code uses, but
    # the quantity the mechanism was framed around, so measured alongside.
    return {
        "ipsae_ab": round(A["score"], 4),
        "ipsae_ba": round(B["score"], 4),
        "ipsae_min": round(min(A["score"], B["score"]), 4),
        "ipsae_ab_mean": A["score_mean"],
        "ipsae_ba_mean": B["score_mean"],
        "ipsae_min_mean": min(A["score_mean"], B["score_mean"]),
        "argmax_ab": A["argmax"], "argmax_ba": B["argmax"],
        "n_below_argmax_ab": A["n_below_argmax"],
        "n_below_argmax_ba": B["n_below_argmax"],
        "d0_argmax_ab": A["d0_argmax"], "d0_argmax_ba": B["d0_argmax"],
        "n_rec": n_rec, "n_eff": pae.shape[0] - n_rec,
        # saturation: does the argmax residue see the WHOLE partner chain?
        "saturated_ab": int(A["n_below_argmax"] == pae.shape[0] - n_rec),
        "saturated_ba": int(B["n_below_argmax"] == n_rec),
        "L_interchain_pairs": int(passing.size),
        "L_frac": float(passing.size / inter.size) if inter.size else np.nan,
        "mean_pae_passing": float(passing.mean()) if passing.size else np.nan,
        "n_interacting_rec": int((A["nbelow"] > 0).sum()),
        "n_interacting_eff": int((B["nbelow"] > 0).sum()),
        "_nbelow_ab": A["nbelow"], "_nbelow_ba": B["nbelow"],
    }


# ── locating the PAE file for one per-seed row ───────────────────────────────
def pae_path(results_dir: str, unit: str, kind: str, design: str) -> Path | None:
    """Prediction dir for a row of the per-seed table.

    Confidence metrics in those tables come from the STEERED prediction even
    on rows that later went through reversion (upstream renames
    ``steered_ipsae_min`` -> ``ipsae_min``; only ra_eff is taken from the
    reverted structure). So reversion directories are never needed here.
    """
    base = HPC / results_dir / "negative_steering" / "runs" / unit / "cycle_0"
    if kind == "cold_start":
        # design is "initial" for seed 0 and "initial_s1"/"initial_s2" for the
        # others, and each seed has its own prediction directory of that name.
        sub = base / design
    else:
        sub = base / "steered" / design
    p = sub / "boltz_results_input" / "predictions" / "input" / "pae_input_model_0.npz"
    return p if p.exists() else None


def receptor_length(results_dir: str, unit: str) -> int | None:
    pj = HPC / results_dir / "negative_steering" / "runs" / unit / "cycle_0" / "plan.json"
    if not pj.exists():
        return None
    return len(json.loads(pj.read_text()).get("wild_type_receptor_seq", "")) or None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=float, default=10.0,
                    help="PAE cutoff; 10.0 matches the ipsae_min column")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap rows per variant")
    args = ap.parse_args()

    rows, arrays = [], {}
    for variant, (results_dir, csv_path) in VARIANTS.items():
        if not csv_path.is_file():
            print(f"MISSING per-seed table: {csv_path}", file=sys.stderr)
            return 2
        tab = pd.read_csv(csv_path)
        if args.limit:
            tab = tab.head(args.limit)
        reclen: dict[str, int | None] = {}
        for n, r in enumerate(tab.itertuples(index=False), 1):
            unit, design, kind = r.unit, r.design, r.kind
            if unit not in reclen:
                reclen[unit] = receptor_length(results_dir, unit)
            n_rec = reclen[unit]
            p = pae_path(results_dir, unit, kind, design)
            if p is None or n_rec is None:
                rows.append({"variant": variant, "unit": unit, "design": design,
                             "base": r.base, "kind": kind,
                             "seed_index": r.seed_index, "found": 0,
                             "ipsae_min_stored": r.ipsae_min})
                continue
            pae = np.load(p)["pae"]
            res = analyse(pae, n_rec, args.cutoff)
            key = f"{variant}|{unit}|{design}"
            arrays[key + "|ab"] = res.pop("_nbelow_ab")
            arrays[key + "|ba"] = res.pop("_nbelow_ba")
            rows.append({"variant": variant, "unit": unit, "design": design,
                         "base": r.base, "kind": kind,
                         "seed_index": r.seed_index, "found": 1,
                         "ipsae_min_stored": r.ipsae_min,
                         "ra_eff": r.ra_eff, **res})
            if n % 100 == 0:
                print(f"  {variant}: {n}/{len(tab)}", flush=True)
        print(f"{variant}: {len(tab)} rows done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    np.savez_compressed(OUT_NPZ, **arrays)

    ok = df[df.found == 1]
    delta = (ok.ipsae_min - ok.ipsae_min_stored).abs()
    print(f"\nWROTE {OUT_CSV} ({len(df)} rows, {len(ok)} with PAE found)")
    print(f"WROTE {OUT_NPZ} ({len(arrays)} arrays)")
    print(f"reproduction check vs stored ipsae_min: max|delta| = {delta.max():.6f}, "
          f"n mismatching >1e-4 = {(delta > 1e-4).sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
