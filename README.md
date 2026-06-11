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
│   ├── prepare_inputs.py         # derive FASTAs + interface/design-region from a complex PDB
│   └── run_negative_steering.sh  # drive the engine on a prepared complex (HPC, GPU)
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
outputs predicted structures + steering summary tables. Two scripts bridge the
gap:

```bash
pip install -e '.[experiments]'        # numpy + gemmi + biopython (prepare step)

# 1. Derive inputs from the complex (runs anywhere):
./scripts/prepare_inputs.py \
    --complex experiments/6G10/6G10.pdb \
    --outdir  experiments/6G10/inputs \
    --derive --contact-cutoff 5.0      # or: --interface-file <your residues>

# 2. Run the engine (HPC, GPU + Boltz container):
./scripts/sync_to_hpc.sh
./scripts/run_negative_steering.sh \
    --name 6G10_test \
    --complex experiments/6G10/6G10.pdb \
    --inputs-dir experiments/6G10/inputs \
    --boltz-container /path/to/boltz2_negsteer.img \
    --plan-extra-args "--mode mild --n-designs 5 --num-seeds 1"
```

Chains default to receptor=A / effector=B (override with `--receptor-chain` /
`--effector-chain`). The interface can be **derived** from heavy-atom contacts or
**provided** as a residue file. Full walkthrough: [`docs/running_a_test.md`](docs/running_a_test.md).

## Experiments

Prior runs (migrated from the old `receptor_design/negative_steering` scratch
folder) and new ones live under `experiments/` — gitignored and HPC-only. See
[`experiments/README.md`](experiments/README.md).
