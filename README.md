# structure-negative-steering

A **standalone** home for the Boltz-2 negative-steering engine — so it can be
run and tested on arbitrary inputs in isolation, separate from the full
[`receptor-resurfacing-pipeline`](https://github.com/joshuandwilliams/receptor-resurfacing-pipeline).

The pipeline remains the **single source of truth** for the engine code. This
repo *vendors* the engine from it and keeps it in sync with a pull-only script;
the pipeline is never modified from here.

## Layout

```
structure-negative-steering/
├── engine/                       # VENDORED from the pipeline — do not hand-edit
│   ├── _UPSTREAM.json            # source commit + per-file sha256 (provenance)
│   ├── bin/                      # the import-closure of the negative-steering scripts
│   └── modules/negative_steering.nf
├── scripts/
│   ├── sync_from_pipeline.py     # pull/refresh engine/ from the pipeline (+ --check)
│   ├── sync_to_hpc.sh            # rsync this repo up to the HPC (excludes experiments/)
│   ├── sync_from_hpc.sh          # pull an experiment's results back down
│   ├── run_from_config.slurm.sh  # airgapped-HPC entry point (sbatch; everything in-container)
│   ├── run_from_config.py        # the Python brain (runs IN the container)
│   ├── prepare_inputs.py         # derive FASTAs + interface/design-region from a complex PDB
│   └── run_negative_steering.sh  # drive the engine on a prepared complex (HPC, GPU)
├── experiments/benchmarking/     # tracked per-target test specs (config.yml + ref PDB)
├── docs/running_a_test.md        # end-to-end recipe (e.g. 6G10)
├── tests/
│   ├── test_engine_integrity.py  # vendored files unchanged since last sync (runs anywhere)
│   ├── test_engine_staleness.py  # are we behind the pipeline? (skips if pipeline absent)
│   └── test_smoke.py             # vendored Python compiles; entry points present
├── experiments/                  # gitignored, HPC-authoritative — prior + new runs
├── pyproject.toml
└── .gitignore
```

## Keeping the engine in sync with the pipeline

The engine under `engine/` is a vendored copy. `scripts/sync_from_pipeline.py`
computes the **import closure** of the negative-steering entry points inside the
pipeline's `bin/` (so a newly added sibling import is picked up automatically),
copies those files plus the Nextflow module and shell orchestrator, and records
the upstream commit and a sha256 of every file in `engine/_UPSTREAM.json`.

```bash
# Refresh the vendored engine after you've updated the pipeline:
./scripts/sync_from_pipeline.py
git add engine && git commit -m "Sync engine from pipeline <short-sha>"

# Just check whether you're behind, without writing anything:
./scripts/sync_from_pipeline.py --check

# Point at a pipeline clone somewhere else:
./scripts/sync_from_pipeline.py --pipeline /path/to/receptor-resurfacing-pipeline
```

Two tests back this up:

- **`test_engine_integrity.py`** runs *anywhere* and fails if any vendored file
  was hand-edited (its hash no longer matches `_UPSTREAM.json`). The rule is:
  **never edit `engine/` directly — fix it upstream in the pipeline and re-sync.**
- **`test_engine_staleness.py`** runs `--check` against your local pipeline clone
  and fails if the engine has fallen behind. It **skips** where the pipeline
  isn't present (CI, HPC), so it's a developer convenience, not a hard gate.
  Override the clone location with `NEGSTEER_PIPELINE=/path/...`.

> The engine was first vendored from pipeline commit recorded in
> `engine/_UPSTREAM.json`. At that sync the pipeline working tree was *dirty*,
> so `upstream_dirty: true` is recorded — re-run the sync from a clean checkout
> when convenient to pin a reproducible commit.

## Tests

Mirrors the pipeline / benchmarking-repo convention of pytest marker *tiers*:

```bash
pip install -e ".[test]"
pytest -m local_unit          # laptop tier: integrity, staleness, syntax — no GPU
pytest -m hpc                 # cluster tier: a real steering run (GPU + Boltz)
```

## Running negative steering on a complex (e.g. 6G10)

The engine is a *mid-pipeline* component — it does **not** turn a bare PDB into
results by itself. It needs the two chain sequences plus a true-interface and a
design-region index file; it uses the complex as the **comparison basis** and
outputs predicted structures + steering summary tables.

It's **config-driven** — one YAML per target under `experiments/benchmarking/`,
e.g. [`experiments/benchmarking/6G10/config.yml`](experiments/benchmarking/6G10/config.yml)
(reference PDB, chains, interface mode, steering knobs).

### On the HPC (airgapped — nothing installed, everything in a container)

`run_from_config.slurm.sh` is the entry point. `sbatch` it for the full run — its
header mirrors the pipeline's `NEGSTEER_RUN_ONE` process (`jic-gpu`, 4 CPUs, 12 GB,
6 h, `--gres=gpu:1`); the GPU allocation is required or Boltz fails with
"No supported gpu backend found!". It uses only base-OS tools on the host
(`bash`, `grep`, `singularity`); **all Python runs inside the Boltz image** (which
ships `yaml`/`gemmi`/`biopython`/`numpy`), and the engine run's orchestrator does
its own `singularity exec --nv`.

```bash
./scripts/sync_to_hpc.sh                                       # from the Mac
# then on the HPC:
sbatch scripts/run_from_config.slurm.sh experiments/benchmarking/6G10/config.yml   # full run (GPU)

# CPU-only steps don't need a GPU node — run interactively with bash:
bash scripts/run_from_config.slurm.sh experiments/benchmarking/6G10/config.yml --dry-run
bash scripts/run_from_config.slurm.sh experiments/benchmarking/6G10/config.yml --prepare-only
```

The container defaults to the config's `boltz_container:`; override with
`--container /path/to.img` (see `/hpc-home/jowillia/singularity/` for the library).

### On the Mac (optional — for prepping/inspecting only, no GPU)

```bash
pip install -e '.[experiments]'        # numpy + gemmi + biopython + pyyaml
./scripts/run_from_config.py experiments/benchmarking/6G10/config.yml --dry-run
./scripts/run_from_config.py experiments/benchmarking/6G10/config.yml --prepare-only
```

Under the hood the driver calls `prepare_inputs.py` (derive FASTAs + interface +
design-region) then writes a `launch.sh` that calls `run_negative_steering.sh`.
The interface can be **derived** from heavy-atom contacts or **provided** as a
residue file. Full walkthrough and the design-region caveat:
[`docs/running_a_test.md`](docs/running_a_test.md) and
[`experiments/benchmarking/README.md`](experiments/benchmarking/README.md).

## Experiments

Prior runs (migrated from the old `receptor_design/negative_steering` scratch
folder) and new ones live under `experiments/` — gitignored and HPC-only. See
[`experiments/README.md`](experiments/README.md).
