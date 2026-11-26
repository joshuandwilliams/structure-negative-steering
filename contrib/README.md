# contrib

Convenience scripts that are not part of the pipeline. `main.nf` and
`negsteer run` never call anything here, and nothing here is needed to
reproduce a result.

All three read `site.conf`, so copy `site.conf.example` first.

| Script | What it does |
|:---|:---|
| `sync_to_hpc.sh` | rsync a laptop checkout up to a cluster, protecting `experiments/` from `--delete` |
| `sync_from_hpc.sh` | rsync one experiment tree back down, with `--results-only` to skip raw model output |
| `collect_negsteer_maxrss.sh` | recover peak memory from Slurm accounting for runs you submitted |

`collect_negsteer_maxrss.sh` is kept as the provenance of
`analysis/03-compute-cost/negsteer_maxrss.csv` rather than as something to
run. Slurm's accounting database holds only your own jobs, so it reports
nothing for anyone else.

The sync scripts assume the cluster has no git and that the laptop checkout is
authoritative for code, while `experiments/` is authoritative on the cluster.
That is one way of working rather than the only one. If you use git on both
ends, you do not need them.
