# experiments/benchmarking/

Config-driven negative-steering test runs against reference complexes (the same
structures used in `structure-prediction-benchmarking/data/complexes_for_benchmarking`).

One subdirectory per complex (43 of them, mirroring `complexes_for_benchmarking`;
every target is receptor=A / effector=B). Each holds a **tracked** test definition:

```
benchmarking/
├── 6G10/
│   ├── 6G10.pdb              # the reference complex (comparison basis)   [tracked]
│   ├── config.yml            # file, chains, interface mode, steering params [tracked]
│   ├── negsteer_<jobid>.out  # SLURM log (via benchmark_submission.sh)    [gitignored]
│   ├── inputs/               # derived by prepare                          [gitignored]
│   └── run/                  # engine outputs + launch.sh                  [gitignored]
├── 7B1I/  …
└── … (one per complex)
```

Unlike the rest of `experiments/` (HPC-only results), the `config.yml` and the
reference `*.pdb` here **are** version-controlled — they're the reproducible test
spec. Everything generated (SLURM logs, `inputs/`, `run/`) is gitignored and stays
inside its target folder.

The configs are all identical to `6G10/config.yml` apart from `name` and
`reference_pdb` — edit a target's `config.yml` to change its interface mode,
steering knobs, etc.

## Running a target

On the **airgapped HPC**, everything runs in a container. `sbatch` the SLURM
entry point (`run_from_config.slurm.sh`) for the full GPU run; it execs all Python
via `singularity exec` and needs nothing installed on the host:

```bash
./scripts/sync_to_hpc.sh                                       # from the Mac
# then on the HPC:
sbatch scripts/run_from_config.slurm.sh experiments/benchmarking/6G10/config.yml   # full run (GPU)

# CPU-only steps (no GPU node) — run interactively with bash:
bash scripts/run_from_config.slurm.sh experiments/benchmarking/6G10/config.yml --dry-run
bash scripts/run_from_config.slurm.sh experiments/benchmarking/6G10/config.yml --prepare-only
```

To submit **all** targets at once (each job's logs/outputs land in its own
folder), use the batch submitter from the HPC login node:

```bash
./scripts/benchmark_submission.sh --dry-run     # preview the sbatch commands
./scripts/benchmark_submission.sh               # submit all 43
./scripts/benchmark_submission.sh --only 6G10   # just one (repeatable)
```

On the **Mac** (optional, for prepping/inspecting — no GPU), the Python driver
works directly after `pip install -e '.[experiments]'`:
`./scripts/run_from_config.py …/config.yml --prepare-only`.

Add a new target by copying `6G10/`, dropping in its PDB, and editing
`config.yml` (`reference_pdb`, `receptor_chain`, `effector_chain`, …).

## Important: what the "design region" means

Negative steering does **not** mutate the interface to break binding. It mutates
surface residues that the *predicted* structure wrongly places in contact with
the effector — **excluding a protected set**. The `design_region` is part of
that protected set (in the full pipeline it's the RFDiffusion de-novo region
being validated, which must stay untouched so the rescue pose isn't
contaminated).

Consequences for a **native** solved complex like 6G10:

- There is no de-novo region, so the default is `design_region: none` — protect
  only the true interface (`negsteer_protected_set_source: true_interface`).
- A native complex already binds at its true interface, so the predicted "wrong"
  interface overlaps the true interface heavily. The engine detects this
  "almost-already-correct" regime, gentles the mutation count, and leans on its
  second-shell fallback (residues near, but not at, the protected interface).
  That's expected — a native complex is a sanity case, not the designed-sequence
  case the pipeline normally feeds it.

If you instead want to steer a **designed** receptor sequence, point
`reference_pdb` at the design and set `design_region` to the de-novo stretch
(and `negsteer_protected_set_source: design_region_union`).
