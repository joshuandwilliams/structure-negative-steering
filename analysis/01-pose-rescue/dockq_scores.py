#!/usr/bin/env python3
"""
DockQ scoring for the cold start and representative prediction of each of
the 18 HMA decoy + AVR benchmark targets, sourced from the n=50 dose-sweep
arm (experiments/dose-sweep/n50/) -- the same run tree as
n50_pose_metrics.py's ra_eff figures, so the two pose metrics are paired on
identical structures. Reference (native) complexes are read from
data/reference_complexes/, the single tracked copy every config points at.

ra_eff (receptor-aligned effector RMSD) is the pose metric used everywhere
else in this analysis. DockQ is a second, independent pose metric (Fnat,
LRMSD, iRMSD combined into one 0-1 score) computed here for the same two
structures n50_pose_metrics.py already scores by ra_eff:

* cold start   -> the cold-start seed whose ra_eff is the median of the
                  three, i.e. the structure behind n50_pose_metrics.csv's
                  cold_start_ra_eff_median. Summarising the unsteered
                  prediction the same way as the steered one is what keeps a
                  target that never entered steering on the y = x diagonal.
                  The n50 run trees only kept seed 0's structure for most
                  targets, so non-zero seeds are read from the n=20
                  benchmarking tree, whose cold start is the same prediction
                  (identical per-seed ra_eff for all 18 targets; the two
                  copies of seed 0 agree to the last written decimal).
* representative -> the single structure behind select_representative()'s
                  reported median ra_eff: within that design's retained
                  seeds, the one whose OWN final ra_eff (post-reversion
                  where applicable) is the median value. For an even
                  seed count there is no single median structure, so the
                  seed closest to the reported median is used instead.

DockQ needs its own environment (the DockQ PyPI package pins numpy<2,
incompatible with this repo's numpy 2.4.6), so scoring runs via
`conda run -n dockq-tool DockQ ...` as a subprocess. Create it once with:

    conda create -n dockq-tool python=3.11 -y
    conda run -n dockq-tool pip install DockQ

This script itself runs in the repo's normal analysis environment.

Output: dockq_scores.csv (committed), read by 01-pose-rescue.qmd.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
MODEL_ROOT = REPO_ROOT / "experiments" / "dose-sweep" / "n50"
# The solved complexes, one tracked copy each.
NATIVE_ROOT = REPO_ROOT / "data" / "reference_complexes"
# The n=20 run tree, used only as the fallback source for cold-start seed
# structures the n50 tree did not keep. Its cold start is the same prediction,
# so this is the same structure, not a substitute for it.
BENCH_ROOT = REPO_ROOT / "experiments" / "benchmarking" / "unconstrained"

sys.path.insert(0, str(HERE))
from make_plots import RA_EFF_THRESHOLD, TARGETS, _try_float, _try_int  # noqa: E402

DOCKQ_ENV = "dockq-tool"


def _local_path(hpc_path: str) -> Path:
    """Remap a path recorded from the HPC run to this repo's local checkout.

    Recorded paths look like
    /hpc-home/jowillia/.../structure-negative-steering/experiments/...;
    only the part from "structure-negative-steering/" onward is portable.
    Paths under experiments/benchmarking/ were recorded before the local
    tree was split into unconstrained/ and constrained/ subfolders, so
    "unconstrained/" is reinserted after "benchmarking/" for those.
    """
    marker = "structure-negative-steering/"
    idx = hpc_path.find(marker)
    if idx == -1:
        raise ValueError(f"cannot remap path: {hpc_path}")
    rel = hpc_path[idx + len(marker):]
    bmarker = "experiments/benchmarking/"
    if rel.startswith(bmarker) and not rel.startswith(bmarker + "unconstrained/") \
            and not rel.startswith(bmarker + "constrained/"):
        rel = bmarker + "unconstrained/" + rel[len(bmarker):]
    return REPO_ROOT / rel


def _final_seed_value_and_pdb(row, reversion_results: dict) -> tuple[float | None, Path | None]:
    """One raw_per_seed_results.csv row -> (final ra_eff, final structure pdb).

    Mirrors make_plots._final_seed_value, but also resolves which structure
    file that final ra_eff came from: the reverted prediction when the seed
    went through reversion, otherwise the steered (or initial) prediction.
    """
    verdict = str(row.get("reversion_verdict") or "").strip()
    if verdict == "pose_holds":
        entry = reversion_results.get(str(row.get("design")))
        ra = _try_float(row.get("reverted_ra_eff_vs_truth"))
        pdb = _local_path(entry["reverted_prediction_pdb"]) if entry else None
        return ra, pdb
    if verdict in ("pose_collapses", "new_contamination",
                    "pose_holds_not_intact", "missing_verdict", "unknown"):
        entry = reversion_results.get(str(row.get("design")))
        ra = _try_float(row.get("reverted_ra_eff_vs_truth"))
        pdb = _local_path(entry["reverted_prediction_pdb"]) if entry else None
        return ra, pdb
    # Blank verdict: reversion was correctly skipped (zero contamination on
    # the mutated positions), or the row is missing data.
    n_contacts = _try_int(row.get("steered_n_contacts_on_mutated_positions"))
    ra = _try_float(row.get("steered_ra_eff_vs_truth"))
    if n_contacts == 0 and ra is not None:
        pdb_str = row.get("pdb")
        pdb = _local_path(pdb_str) if isinstance(pdb_str, str) and pdb_str else None
        return ra, pdb
    return None, None


def cold_start_median_structure(target: str) -> tuple[Path | None, float | None]:
    """(pdb path, ra_eff) for the median of a target's three cold-start seeds.

    Seed offset n is written as initial_prediction.pdb for n = 0 and
    initial_prediction_s<n>.pdb otherwise. Falls back to the n=20
    benchmarking tree when the n50 tree did not keep that seed's structure.
    """
    plan_path = MODEL_ROOT / target / "run" / "cycle_0" / "plan.json"
    if not plan_path.is_file():
        return None, None
    seeds = [s for s in json.loads(plan_path.read_text()).get("cold_start_seeds", [])
             if s.get("ra_eff") is not None]
    if not seeds:
        return None, None
    med = statistics.median(s["ra_eff"] for s in seeds)
    pick = min(seeds, key=lambda s: abs(s["ra_eff"] - med))
    offset = int(pick.get("seed_offset", 0))
    name = "initial_prediction.pdb" if offset == 0 else f"initial_prediction_s{offset}.pdb"
    for root in (MODEL_ROOT, BENCH_ROOT):
        pdb = root / target / "run" / "cycle_0" / name
        if pdb.is_file():
            return pdb, pick["ra_eff"]
    return None, pick["ra_eff"]


def representative_structure(rundir: Path) -> tuple[Path | None, float | None]:
    """(pdb path, ra_eff) for the single structure behind select_representative().

    Reproduces select_representative()'s design-group ranking (best
    seed-pass-count group, then lowest median ra_eff), then within the
    winning group picks the seed whose own final ra_eff is closest to that
    median, and resolves that seed's final structure file.
    """
    plan_path = rundir / "cycle_0" / "plan.json"
    per_seed_path = rundir / "raw_per_seed_results.csv"
    reversion_path = rundir / "cycle_0" / "reversion_results.json"

    reversion_results = {}
    if reversion_path.is_file():
        try:
            reversion_results = json.loads(reversion_path.read_text())
        except Exception:
            reversion_results = {}

    candidates = []  # (rank, median, seeds: list[(ra, pdb)])

    if per_seed_path.is_file():
        df = pd.read_csv(per_seed_path)
        steered = df[(df.get("status") == "ok") & (df.get("design") != "initial")].copy()
        if len(steered):
            steered["base"] = steered["design"].str.replace(r"_s\d+$", "", regex=True)
            for base, grp in steered.groupby("base"):
                seeds = []
                n_pass = 0
                for _, row in grp.iterrows():
                    ra, pdb = _final_seed_value_and_pdb(row, reversion_results)
                    if ra is not None:
                        seeds.append((ra, pdb))
                    if ra is not None and ra < RA_EFF_THRESHOLD:
                        n_pass += 1
                if not seeds:
                    continue
                if n_pass == len(grp):
                    rank = 0
                elif n_pass == 0:
                    rank = 3
                else:
                    rank = 1 if n_pass > 1 else 2
                median = sorted(r for r, _ in seeds)[len(seeds) // 2]
                candidates.append((rank, median, seeds))

    if not candidates and plan_path.is_file():
        try:
            cold_start_seeds = json.loads(plan_path.read_text()).get("cold_start_seeds") or []
        except Exception:
            cold_start_seeds = []
        seeds = []
        for s in cold_start_seeds:
            ra = _try_float(s.get("ra_eff"))
            pdb_str = s.get("pdb_path")
            if ra is not None and pdb_str:
                seeds.append((ra, _local_path(pdb_str)))
        if seeds:
            n_pass = sum(1 for r, _ in seeds if r < RA_EFF_THRESHOLD)
            if n_pass == len(seeds):
                rank = 0
            elif n_pass == 0:
                rank = 3
            else:
                rank = 1 if n_pass > 1 else 2
            median = sorted(r for r, _ in seeds)[len(seeds) // 2]
            candidates.append((rank, median, seeds))

    if not candidates:
        return None, None

    candidates.sort(key=lambda c: (c[0], c[1]))
    _, median, seeds = candidates[0]
    # Seed whose own ra_eff is closest to the group's reported median.
    seeds_with_pdb = [(ra, pdb) for ra, pdb in seeds if pdb is not None]
    pool = seeds_with_pdb or seeds
    ra, pdb = min(pool, key=lambda s: abs(s[0] - median))
    return pdb, ra


def run_dockq(model: Path, native: Path) -> float | None:
    """Score model vs native. --allowed_mismatches 6 accommodates negative
    steering's mutated receptor (up to max_mutations=6 substitutions), whose
    sequence otherwise fails DockQ's default homology check against the
    unmutated native chain."""
    if not (model and model.is_file() and native.is_file()):
        return None
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out_path = Path(tmp.name)
    try:
        subprocess.run(
            ["conda", "run", "-n", DOCKQ_ENV, "DockQ", str(model), str(native),
             "--allowed_mismatches", "6", "--json", str(out_path)],
            capture_output=True, text=True, check=False,
        )
        if not out_path.is_file() or out_path.stat().st_size == 0:
            return None
        data = json.loads(out_path.read_text())
        return data.get("GlobalDockQ")
    finally:
        out_path.unlink(missing_ok=True)


def plot_dockq_scatter(df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Which targets ever entered steering comes from n50_pose_metrics.csv, so
    # the two panels of the thesis figure cannot disagree about it.
    flags = pd.read_csv(HERE / "n50_pose_metrics.csv")[["target", "steering_triggered"]]
    g = df.dropna(subset=["cold_start_dockq", "rep_dockq"]).merge(flags, on="target")
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    for triggered, colour, label in [(True, "#0072B2", "Negative steering applied"),
                                     (False, "#009E73", "Correctly posed without steering")]:
        sub = g[g["steering_triggered"] == triggered]
        ax.scatter(sub["cold_start_dockq"], sub["rep_dockq"], color=colour,
                   label=f"{label} (n = {len(sub)})",
                   s=55, edgecolor="black", linewidth=0.4, alpha=0.9, zorder=3)
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1.2, zorder=0)
    for thr, style in [(0.23, ":"), (0.49, ":"), (0.80, ":")]:
        ax.axhline(thr, color="#BBBBBB", linestyle=style, linewidth=0.8, zorder=0)
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.02, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Initial DockQ")
    ax.set_ylabel("Steered DockQ")
    fig.savefig(HERE / "thesis-figures" / "cold_start_vs_steered_dockq.png",
                dpi=450, bbox_inches="tight", pad_inches=0.02)
    print(f"Wrote {HERE / 'thesis-figures' / 'cold_start_vs_steered_dockq.png'}")


def main():
    # Re-drawing the figure should not need the DockQ environment or another
    # pass over every structure, so --plot-only redraws from the committed CSV.
    if "--plot-only" in sys.argv:
        plot_dockq_scatter(pd.read_csv(HERE / "dockq_scores.csv"))
        return

    rows = []
    for target in TARGETS:
        rundir = MODEL_ROOT / target / "run"
        native = NATIVE_ROOT / f"{target}.pdb"
        cold_pdb, cold_ra_eff = cold_start_median_structure(target)

        cold_dockq = run_dockq(cold_pdb, native) if cold_pdb else None
        rep_pdb, rep_ra_eff = representative_structure(rundir)
        rep_dockq = run_dockq(rep_pdb, native) if rep_pdb else None

        rows.append({
            "target": target,
            "cold_start_dockq": cold_dockq,
            "cold_start_ra_eff_at_dockq": cold_ra_eff,
            "cold_start_pdb": str(cold_pdb) if cold_pdb else None,
            "rep_dockq": rep_dockq,
            "rep_ra_eff_at_dockq": rep_ra_eff,
            "rep_pdb": str(rep_pdb) if rep_pdb else None,
        })
        print(f"{target}: cold_start_dockq={cold_dockq}, rep_dockq={rep_dockq}")

    df = pd.DataFrame(rows)
    out = HERE / "dockq_scores.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")
    (HERE / "thesis-figures").mkdir(parents=True, exist_ok=True)
    plot_dockq_scatter(df)


if __name__ == "__main__":
    main()
