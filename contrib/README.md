# contrib

Files that are not part of the pipeline. `main.nf` and `negsteer run` never
call anything here.

| File | What it does |
|:---|:---|
| `negative_steering.nf` | a drop-in Nextflow module for `receptor-resurfacing-pipeline`, so that repo can call this engine as one process instead of vendoring it |

`negative_steering.nf` is not run here at all. `main.nf` does not include it,
and it is meant to be copied into the downstream repo's `modules/`.

The laptop-to-cluster sync scripts and the Slurm memory-accounting script that
used to sit here belong to the benchmark campaign. They are in
[`negative-steering-analyses`](https://github.com/joshuandwilliams/negative-steering-analyses)
under `contrib/`.
