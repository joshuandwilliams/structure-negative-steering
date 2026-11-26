# Simplification proposals, one file per script

Read these in number order. Each is independent, so you can accept one and
reject the next.

**All twelve are settled, and this folder is the record rather than a queue.**
Seven were accepted and landed on `main` as separate commits, named in the
table below. Four concluded that the right change is no change, and one was
rejected by its own measurement. Nothing here is waiting to be applied, so the
value left in these files is the reasoning: five of the twelve argue against a
change that looked obvious before it was measured.

## What I measured, and what I did not

Two goals, and they pull in different directions often enough to be worth
separating.

**Shorter and plainer.** Measured as lines, and as how many of those lines are
mechanical: `out["x"] = r.get("x", "")` repeated forty times is forty lines
carrying one idea. The repo-wide count of that pattern is 191 assignments.

**Faster.** Measured by benchmark, not by reasoning. This matters less than it
sounds, and the reason is worth stating once: every stage's wall-clock is
dominated by `singularity exec … boltz predict`, which is minutes to hours.
From `analysis/03-compute-cost/runtime_comparison.csv`, one steered benchmark
target is 19.5 to 263 minutes depending on dose. Python that reads a few
thousand CSV rows is milliseconds against that. So **shaving Python almost
never shows up in a run**, and I have not pretended otherwise in these files.

The exception is algorithmic: a loop that re-reads a file, or re-walks a
directory tree, once per item. Two of those exist and both are in here. They
are the only changes where the speed claim is the point rather than a
footnote.

Where a rewrite would be slower, I say so and do not propose it.

## How each proposal is verified

Every proposal that changes behaviour-bearing code carries two checks before
it is offered:

1. **Fixture equality.** The old and new implementations run over every
   relevant input the repo holds, and their outputs are compared field by
   field. For row builders that is 151 `aggregated_results.csv` rows; for
   structure readers it is the 18 complexes in `data/reference_complexes/`.
2. **Randomised equality.** Thousands of generated inputs covering each
   branch, including the outcome classes and absent keys, since the fixtures
   do not reach every path.

Then the suite runs, and coverage has to stay at 100%. A proposal that fails
any of those is not in this folder.

## Order, and why

`Lines` is the file as it stood when the proposal was written. `Outcome` is
what the proposal's own body concludes, which for three of them is not what
the candidate change was.

| # | File | Lines | Candidate change | Risk | Outcome |
|---|:---|---:|:---|:---|:---|
| 01 | `sum_stage_times.py` | 47 | nothing to do | none | change nothing |
| 02 | `pipeline_thresholds.py` | 158 | nothing to do | none | change nothing |
| 03 | `pathway_labels.py` | 137 | one helper inlined | low | accepted, `1a43ba0` |
| 04 | `compute_metrics.py` sidecar lookup | 1,485 | **4.2× faster**, 25 lines longer | low | accepted, `1686e1a` |
| 05 | `negsteer_coldstart.py` plan lookup | 633 | one file read instead of N | low | accepted, `1688630` |
| 06 | `extract_passing.py` | 629 | the column spec, 243 → 161 | low | accepted, `85acf45` |
| 07 | `negative_steering.py` | 296 | print blocks folded | low | **change nothing** |
| 08 | `prepare_complex_inputs.py` | 329 | writer helpers merged | low | **rejected** |
| 09 | `cross_summary.py` | 940 | tier and rank tables | medium | one factorisation accepted, `b7dadee` |
| 10 | `negsteer_aggregate.py` | 2,580 | the column families, 47 assignments | medium | one change accepted, `1d24652` |
| 11 | `reversion.py` | 1,826 | the argv builders | medium | argv builder accepted, `3cb4f4e` |
| 12 | `boltz2_negative_steering.py` | 3,339 | `cmd_plan`'s reporting | high | **change nothing** |

The order is ascending risk, not ascending size. 01 and 02 come first because
they are the answer to "is every line necessary" being yes, which is worth
knowing before reading eleven files that say otherwise. 04 and 05 come next
because they are the only two with a measured speed win. Then the mechanical
column work, which is the bulk of the saving. 12 is last and is the one I
would leave alone.

## The honest summary

Across all twelve, the available saving is about 1,100 lines of 12,399, under
10%. There is no single large win left: the large wins were the 6,890 lines of
design-campaign code already removed. What remains is column bookkeeping,
which is repetitive rather than complicated, and two directory walks that are
genuinely inefficient.

If you only take two, take 04 and 06.

One correction, made after applying them. 04 first claimed 96× and 24 lines
shorter. 96× was the lookup in isolation rather than `parse_boltz2` end to
end, which is 4.2×, and the file is 25 lines longer rather than shorter
because I estimated the line count instead of measuring it. 06's claim went
the other way: 6.1× in the proposal, 7.6× once applied and measured against
the committed implementation.
