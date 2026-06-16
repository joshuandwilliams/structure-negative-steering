# Negative-steering benchmark analysis

Summary plots and table for the 43-target negative-steering benchmark
(`experiments/benchmarking/`). This analysis is **local-only** — it reads the
result summaries pulled down from the HPC and produces figures that are committed
to the repo so the headline results travel with the code.

## What it measures

Each benchmark target is a known receptor–effector complex. We compare two
predictions of the effector pose against the experimental truth, using
`ra_eff` = receptor-aligned effector RMSD (Å); **lower is better**, and ≤ 5 Å
counts as a rescued (correct) pose:

* **cold start** — the initial Boltz-2 prediction, before steering
  (`run/cycle_0/plan.json::initial_receptor_aligned_effector_rmsd`).
* **representative** — the single representative steered design selected by the
  cross-sequence tiering (`run/cross_sequence_summary.csv::rep_ra_eff_vs_truth_median`).

Targets are grouped into four biologically meaningful categories (hard-coded in
`make_plots.py` as difficulty tiers 1–4, each mapped to a `category` label that
names both parts of the modelled complex, receptor + effector):

| code | category | what it is |
|---|---|---|
| 1 | **HMA decoy + AVR** | isolated integrated HMA decoy domain (Pikp/Pikm/RGA5-HMA…) / AVR effector |
| 2 | **NLR sensor + effector** | single NLR sensor protomer (LRR-C-JID / WRKY / CNL), 1:1 binary complex |
| 3 | **Host target + effector** | genuine host virulence target (OsHIPP19, Exo70F2, PP1c, SPL5…) / effector |
| 4 | **Resistosome protomer + effector** | one receptor+effector pair from an oligomeric NLR resistosome (Sr35 / WRR4A / NRC), *not* the full assembly |

## How to reproduce

The benchmark run outputs live on the HPC and are gitignored. Pull just the
lightweight summaries down (≈ 16 MB, no raw model outputs), then regenerate:

```bash
# 1. populate experiments/benchmarking/<TARGET>/{run,inputs} with summaries only
scripts/sync_from_hpc.sh --name benchmarking --results-only

# 2. regenerate benchmark_summary.csv + figures/
python3 analysis/negsteer_benchmark/make_plots.py
```

`--bench` / `--outdir` override the input tree and output location.

`negsteer_benchmark.qmd` is a Quarto write-up of the same analysis (in the style
of the `uv-agroinfiltration-dataset/analyses/` documents): it reads the committed
`benchmark_summary.csv`, renders every plot inline with a short interpretation,
and re-saves the `figures/`. Render it with:

```bash
quarto render analysis/negsteer_benchmark/negsteer_benchmark.qmd
```

## Outputs (committed)

* `benchmark_summary.csv` — one row per target: tier, category, status, sequence
  length, cold-start ra_eff, representative ra_eff, and the change, plus the full
  representative cross-sequence row.
* `figures/plot1_ra_eff_hist` — stacked histogram of representative ra_eff by category.
* `figures/plot2_ra_eff_change_hist` — histogram of change (representative − cold start).
* `figures/plot3_length_vs_ra_eff_scatter` — combined sequence length vs representative ra_eff.
* `figures/plot4_ra_eff_vs_change_scatter` — representative ra_eff vs its change.
* `figures/plot5_ra_eff_by_tier_boxplot` — representative ra_eff per category.
* `figures/plot6_initial_vs_rep_scatter` — cold-start vs representative ra_eff (y = x diagonal).

Each figure is written as both `.png` and `.svg`.

## Latest result (43/43 targets finished)

* 43 / 43 targets produced a representative steered design.
* Representative ra_eff: min 1.16 Å, median 15.53 Å, max 65.88 Å.
* 14 / 43 reach the ≤ 5 Å pass threshold.
* Steering improved the pose (ra_eff decreased) for 35 targets and worsened it
  for 7 (cold start already correct in the rest).
