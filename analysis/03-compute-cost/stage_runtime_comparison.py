#!/usr/bin/env python3
"""
Stage the compute-cost comparison between negative steering and the
structure-prediction-benchmarking predictors.

Writes ``runtime_comparison.csv``, one row per (method, target) run, combining
two sources that live in different repos:

* negative steering, from this repo's
  ``experiments/benchmarking/unconstrained/<TARGET>/run/``.
  Wall-clock comes from ``run_one_runtime_sec.txt``, the summed seconds of
  every stage of the run (plan, predict-one, collect, reversion, harvest,
  finalize, aggregate and metrics). The number of Boltz-2
  calls is counted from ``raw_per_seed_results.csv``, one row per call plus
  one extra call for every row that triggered a reversion re-prediction, and
  multiplied by ``negsteer_diffusion_samples`` from ``config.yml`` to get
  predictions.
* benchmark predictors, from ``structure-prediction-benchmarking``'s
  ``data/metrics/runtime_stats.csv``. A run there is 5 seeds x 5 diffusion
  samples = 25 predictions for every combo except ESMFold2, which produces 5.
  Wall-clock is ``elapsed_s + search_cpu_s`` for the three models that share
  one pooled ColabFold/MMseqs2 search (``boltz1`` and ``boltz2`` with an MSA,
  and ``colabfold``) -- their own ``elapsed_s`` covers prediction only, so
  without the addition they understate by the search's CPU time, roughly
  176 min each, and read as if they were free. Every other combo uses bare
  ``elapsed_s``: af2m and af3 with an MSA run search and prediction in one
  Nextflow process, so theirs already includes it, and every no-MSA combo has
  ``search_cpu_s`` of zero. This matches what that repo's own compute-cost
  analysis does.

Both sets were run on the same cluster queue with one GPU (``jic-gpu``,
``--gres=gpu:1``), so the wall-clock numbers are comparable. Targets are
restricted to the 18 Tier-1 HMA decoy + AVR complexes that both repos cover,
taken from ``benchmark_summary.csv`` so the two stay in sync.

The per-run and per-prediction columns answer different questions and disagree:
a steering run is an order of magnitude more predictions than a benchmark run,
so it is the most expensive method per run while being mid-pack per prediction.
Both are written; the figure shows both.

``peak_rss_gb`` is staged per cohort, from whatever measured that cohort. See
the note above ``negsteer_memory`` for the tooling difference and why it does
not distort the comparison. Regenerate the steering half with
``collect_negsteer_maxrss.sh``, which must run on the cluster.

Usage:
    python3 analysis/03-compute-cost/stage_runtime_comparison.py
    python3 analysis/03-compute-cost/stage_runtime_comparison.py --bench-repo ../spb
"""

import argparse
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_BENCH_REPO = REPO_ROOT.parent / "structure-prediction-benchmarking"
DEFAULT_CAMPAIGN_REPO = REPO_ROOT.parent / "receptor-resurfacing-pipeline"
# That repo moved this file. Newest location first, so an old checkout still works.
BENCH_RUNTIME_RELS = (
    "data/metrics/runtime_stats.csv",
    "analysis/structure_prediction_benchmark/runtime_stats.csv",
)

# Predictions per benchmark run: 5 seeds x 5 diffusion samples, except ESMFold2,
# which emits 5. Mirrors N_PREDICTIONS in that repo's 07-compute-cost.qmd.
BENCH_PREDICTIONS = {"esmfold2": 5}
BENCH_PREDICTIONS_DEFAULT = 25

# Base model name only. The variant (no MSA / MSA / reference restraints / negative
# steering) is carried in its own column and shown by colour in the figure, so
# the constrained variants collapse onto their parent model's name.
DISPLAY = {
    "af2m": "AlphaFold2-Multimer", "af3": "AlphaFold3",
    "boltz1": "Boltz-1", "boltz1_constrained": "Boltz-1",
    "boltz2": "Boltz-2", "boltz2_constrained": "Boltz-2",
    "chai1": "Chai-1", "colabfold": "ColabFold", "esmfold2": "ESMFold2",
}
USES_REFERENCE = {"boltz1_constrained", "boltz2_constrained"}

# Two negative-steering cohorts on the 18 benchmark targets themselves
# (steering sets 3/5/10/20/50, the dose sweep under experiments/dose-sweep/,
# plus the original 20-set run under experiments/benchmarking/), and a third,
# separate cohort on a different target set entirely (the AVR-PikF campaign,
# 3 steering sets). The dose-sweep and campaign labels both name "3 sets" but
# are NOT the same run and must not collapse together: the dose sweep shares
# targets with the benchmark predictors and is plotted in the cost/memory
# scatter, the campaign does not and stays out of it (see qmd section 3).
NEGSTEER_LABEL = "Boltz-2 + negative steering (steering sets = 20)"
DOSE_SWEEP_DOSES = (3, 5, 10, 50)
DOSE_SWEEP_LABEL = "Boltz-2 + negative steering, dose sweep (steering sets = {n})"
CAMPAIGN_LABEL = "Boltz-2 + negative steering, AVR-PikF campaign (steering sets = 3)"
CAMPAIGN_SURVIVORS_REL = ("experiments/campaigns/pikp1_avrpikf/analyses/survivor_metrics/"
                          "data/crystal_full_test_contig_survivors.csv")
CAMPAIGN_PER_SEED_REL = "analysis/05-campaign-confidence/campaign_data/campaign_per_seed.csv"
CAMPAIGN_DIFFUSION_SAMPLES = 5   # negsteer_diffusion_samples in the run's params.yml

# Peak memory, staged per cohort because the three sets were measured by
# different tooling. The benchmark predictors and the campaign both come from a
# Nextflow trace's peak_rss; the 18-target steering runs were plain sbatch jobs
# with no trace, so theirs is Slurm's MaxRSS, recovered by
# collect_negsteer_maxrss.sh. MaxRSS is sampled by the Slurm accounting plugin
# rather than by Nextflow, so it is not the identical quantity, but the two
# agree closely where they overlap: the campaign's steering runs read 5.1 GB by
# trace against 4.6 GB by MaxRSS for the benchmark steering runs, and unsteered
# Boltz-2 reads 5.0 GB by trace. Treat differences below ~1 GB as noise.
NEGSTEER_MAXRSS_REL = "negsteer_maxrss.csv"
# Peak memory for the dose-sweep cohort, from the Nextflow trace of the single
# head job that ran all 90 (target, dose) units (nf_dosesweep, cluster job
# 22636181, .nextflow-reports/trace.txt). One row per (target, dose), the max
# peak_rss across every task belonging to that unit's chain (PREPARE through
# POSTPROCESS). See dose_sweep_trace/ for how it was extracted.
DOSE_SWEEP_MAXRSS_REL = "dose_sweep_maxrss.csv"
CAMPAIGN_TRACE_MEMORY_REL = "campaign_trace_memory.csv"


def negsteer_memory(path):
    """Peak memory per benchmark target, from Slurm MaxRSS.

    Several targets were retried, so more than one job id can exist per target.
    The run the analysis uses is the one whose Slurm ``Elapsed`` matches the
    ``run_one_runtime_sec.txt`` the wall-clock came from, so pick the job with
    the smallest gap. The gap is a few seconds of sbatch wrapper overhead.
    """
    if not path.exists():
        print(f"  no {path.name}; steering rows will have no memory")
        return {}

    m = pd.read_csv(path)
    m = m[(m["state"] == "COMPLETED") & (m["maxrss_gb"] > 0)].copy()

    def elapsed_s(v):
        parts = str(v).split("-")
        days = int(parts[0]) if len(parts) == 2 else 0
        h, mi, s = parts[-1].split(":")
        return days * 86400 + int(h) * 3600 + int(mi) * 60 + int(s)

    m["gap"] = (m["elapsed"].map(elapsed_s) - m["runtime_sec_recorded"]).abs()
    best = m.sort_values("gap").groupby("target", as_index=False).first()
    if len(best) and best["gap"].max() > 120:
        print(f"  WARNING: worst MaxRSS/runtime gap is {best['gap'].max():.0f}s, "
              f"so a retried target may have matched the wrong job")
    return dict(zip(best["target"], best["maxrss_gb"]))


def campaign_memory(path):
    """Peak memory per campaign unit, from the run's Nextflow trace."""
    if not path.exists():
        print(f"  no {path.name}; campaign rows will have no memory")
        return {}
    t = pd.read_csv(path)
    return dict(zip(t["unit"], t["peak_rss_gb"]))


def diffusion_samples(config_path):
    """negsteer_diffusion_samples from a benchmark target's config.yml."""
    m = re.search(r"^negsteer_diffusion_samples:\s*(\d+)",
                  config_path.read_text(), flags=re.M)
    if not m:
        raise ValueError(f"no negsteer_diffusion_samples in {config_path}")
    return int(m.group(1))


def negsteer_rows(target_dirs, model_label, memory=None):
    """One row per (target -> its run dir): wall-clock, Boltz-2 calls,
    predictions, peak memory.

    target_dirs maps target -> the run/ directory itself, so this covers both
    tree shapes in the repo: experiments/benchmarking/unconstrained/<target>/run
    and experiments/dose-sweep/nNN/<target>/run, the latter with no
    "unconstrained" level and its config.yml one level up from run/.
    """
    memory = memory or {}
    rows = []
    for target, rundir in target_dirs.items():
        runtime_file = rundir / "run_one_runtime_sec.txt"
        per_seed = rundir / "raw_per_seed_results.csv"
        config_path = rundir.parent / "config.yml"
        if not (runtime_file.exists() and per_seed.exists() and config_path.exists()):
            print(f"  skip {target}: missing runtime sidecar, per-seed results or config")
            continue

        seeds = pd.read_csv(per_seed)
        # One Boltz-2 call per row. A row whose mutations needed reverting was
        # re-predicted, which is a second call for that row.
        n_reversions = int(seeds["reversion_verdict"].notna().sum()) \
            if "reversion_verdict" in seeds else 0
        n_calls = len(seeds) + n_reversions
        # A run whose cold start passed on every seed exits before steering, so
        # it only ever paid for the cold-start seeds.
        steered = bool((seeds["status"] != "initial").any())

        samples = diffusion_samples(config_path)
        elapsed_s = float(runtime_file.read_text().strip())
        rows.append({
            "model": model_label,
            "variant": "negsteer",
            "target": target,
            "elapsed_min": elapsed_s / 60.0,
            "n_predictions": n_calls * samples,
            "steered": steered,
            "peak_rss_gb": memory.get(target, float("nan")),
        })
    return rows


def campaign_rows(survivors_csv, per_seed_csv, memory=None):
    """One row per designed sequence of the AVR-PikF campaign (steering sets = 3).

    Wall-clock is ``run_one_runtime_sec`` from the campaign's survivor table,
    keyed by the unit its ``source_passing_summary`` path names. Boltz-2 calls
    are counted from the per-seed table the same way as for the benchmark, one
    per row plus one per reversion re-prediction. The two polyA/scrambled
    control units appear in the survivor table but not the per-seed table, so
    the inner join drops them, which is what we want: the controls ran with
    ``negsteer_controls_n_designs: 1`` and are not the same cost.
    """
    memory = memory or {}
    if not (survivors_csv.exists() and per_seed_csv.exists()):
        print(f"  skip campaign: missing {survivors_csv} or {per_seed_csv}")
        return []

    surv = pd.read_csv(survivors_csv)
    surv["unit"] = surv["source_passing_summary"].str.split("/").str[1]
    elapsed_s = surv.groupby("unit")["run_one_runtime_sec"].median()

    seeds = pd.read_csv(per_seed_csv)
    calls = seeds.groupby("unit").size()
    reversions = seeds.groupby("unit")["reversion_verdict"].apply(lambda s: s.notna().sum())
    total_calls = calls + reversions

    units = elapsed_s.index.intersection(total_calls.index)
    print(f"  campaign: {len(units)} sequences "
          f"({len(elapsed_s) - len(units)} dropped, no per-seed rows)")
    return [{
        "model": CAMPAIGN_LABEL,
        "variant": "negsteer",
        "target": unit,
        "elapsed_min": float(elapsed_s[unit]) / 60.0,
        "n_predictions": int(total_calls[unit]) * CAMPAIGN_DIFFUSION_SAMPLES,
        # A unit with only cold-start calls exited before steering.
        "steered": bool((seeds[seeds["unit"] == unit]["kind"] != "cold_start").any()),
        "peak_rss_gb": memory.get(unit, float("nan")),
    } for unit in units]


# The three predictors that share ONE pooled ColabFold/MMseqs2 search rather
# than running their own. Their elapsed_s is prediction-only; search_cpu_s
# must be added back on to get a standalone, comparable wall-clock. Keyed on
# (model, msa) since only the *_msa row of each needs the addition.
SHARED_SEARCH = {("boltz1", "msa"), ("boltz2", "msa"), ("colabfold", "msa")}


def bench_rows(bench_repo, targets):
    """One row per (model, msa, target) from the benchmarking repo's traces."""
    path = next((bench_repo / rel for rel in BENCH_RUNTIME_RELS
                 if (bench_repo / rel).exists()), None)
    if path is None:
        tried = "\n       ".join(str(bench_repo / rel) for rel in BENCH_RUNTIME_RELS)
        raise SystemExit(f"ERROR: no runtime_stats.csv. Tried:\n       {tried}\n"
                         f"       point --bench-repo at structure-prediction-benchmarking")
    rt = pd.read_csv(path)
    rt = rt[(rt["status"] == "COMPLETED") & (rt["pdb"].isin(targets))]
    rt = rt.dropna(subset=["elapsed_s"])
    rows = []
    for _, r in rt.iterrows():
        elapsed_s = r["elapsed_s"]
        if (r["model"], r["msa"]) in SHARED_SEARCH:
            elapsed_s = elapsed_s + r["search_cpu_s"]
        rows.append({
            "model": DISPLAY.get(r["model"], r["model"]),
            "variant": "restraints" if r["model"] in USES_REFERENCE else r["msa"],
            "target": r["pdb"],
            "elapsed_min": elapsed_s / 60.0,
            "n_predictions": BENCH_PREDICTIONS.get(r["model"],
                                                   BENCH_PREDICTIONS_DEFAULT),
            "steered": False,
            "peak_rss_gb": r.get("peak_rss_gb", float("nan")),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench-tree", type=Path,
                    default=REPO_ROOT / "experiments" / "benchmarking",
                    help="this repo's negative-steering run tree")
    ap.add_argument("--dose-sweep-tree", type=Path,
                    default=REPO_ROOT / "experiments" / "dose-sweep",
                    help="the dose-sweep run tree (experiments/dose-sweep/nNN/<target>/run)")
    ap.add_argument("--bench-repo", type=Path, default=DEFAULT_BENCH_REPO,
                    help="path to the structure-prediction-benchmarking checkout")
    ap.add_argument("--campaign-repo", type=Path, default=DEFAULT_CAMPAIGN_REPO,
                    help="path to the receptor-resurfacing-pipeline checkout")
    ap.add_argument("--summary", type=Path,
                    default=HERE.parent / "01-pose-rescue" / "benchmark_summary.csv",
                    help="benchmark_summary.csv, read for the Tier-1 target list")
    ap.add_argument("--out", type=Path, default=HERE / "runtime_comparison.csv")
    args = ap.parse_args()

    targets = sorted(pd.read_csv(args.summary)["target"].astype(str))
    print(f"Tier-1 targets: {len(targets)}")

    bench_target_dirs = {t: args.bench_tree / "unconstrained" / t / "run" for t in targets}
    rows = negsteer_rows(bench_target_dirs, NEGSTEER_LABEL,
                        negsteer_memory(HERE / NEGSTEER_MAXRSS_REL))

    dose_maxrss_path = HERE / DOSE_SWEEP_MAXRSS_REL
    if dose_maxrss_path.exists():
        dm = pd.read_csv(dose_maxrss_path)
        for n in DOSE_SWEEP_DOSES:
            dose_dir = args.dose_sweep_tree / f"n{n:02d}"
            dose_target_dirs = {t: dose_dir / t / "run" for t in targets}
            dose_memory = dict(zip(dm[dm["dose"] == n]["target"],
                                   dm[dm["dose"] == n]["peak_rss_gb"]))
            rows += negsteer_rows(dose_target_dirs, DOSE_SWEEP_LABEL.format(n=n),
                                  dose_memory)
    else:
        print(f"  no {dose_maxrss_path.name}; dose-sweep rows skipped")

    rows += (campaign_rows(args.campaign_repo / CAMPAIGN_SURVIVORS_REL,
                          REPO_ROOT / CAMPAIGN_PER_SEED_REL,
                          campaign_memory(HERE / CAMPAIGN_TRACE_MEMORY_REL))
           + bench_rows(args.bench_repo, targets))
    df = pd.DataFrame(rows)
    df["sec_per_prediction"] = df["elapsed_min"] * 60.0 / df["n_predictions"]
    df = df.sort_values(["model", "variant", "target"]).reset_index(drop=True)
    df.to_csv(args.out, index=False)

    print(f"\nWrote {args.out}  ({len(df)} runs, {df['model'].nunique()} models)")
    missing = df["peak_rss_gb"].isna().sum()
    if missing:
        print(f"  {missing} of {len(df)} runs have no peak memory")
    summary = (df.groupby(["model", "variant"])
                 .agg(n=("target", "size"),
                      median_min=("elapsed_min", "median"),
                      median_s_per_pred=("sec_per_prediction", "median"),
                      median_peak_rss_gb=("peak_rss_gb", "median"))
                 .round(1).sort_values("median_min"))
    print(summary.to_string())


if __name__ == "__main__":
    main()
