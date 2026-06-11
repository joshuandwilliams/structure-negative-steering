# Running negative steering on a solved complex (e.g. 6G10)

The engine is a *mid-pipeline* component: on its own it does **not** take a bare
PDB and do everything. It needs the two chain sequences plus a true-interface and
a design-region index file. `scripts/prepare_inputs.py` derives all of those from
one complex PDB, and `scripts/run_negative_steering.sh` then drives the engine.

What the engine **does** give you: it uses the complex as the **comparison basis**
(`--ground-truth`) and outputs predicted structures, per-sequence CSVs, and a
`cross_sequence_summary.csv`. What it does **not** do: extract sequences or derive
the interface/design-region from a bare PDB — that's what the prepare step is for.

> A real run needs a GPU + the Boltz-2 Singularity image (`boltz2_negsteer.img`).
> **The run is HPC-only.** On the airgapped HPC nothing is installed on the host —
> every Python step runs inside the container.

> **Recommended path — config-driven.** Rather than the manual steps below, put a
> `config.yml` next to the PDB under `experiments/benchmarking/<target>/` and use
> `sbatch scripts/run_from_config.slurm.sh <config>` on the HPC (it runs all Python
> via `singularity exec`, needs nothing installed). See
> [`../experiments/benchmarking/README.md`](../experiments/benchmarking/README.md).
> The manual steps below show what it does under the hood.

## Step 1 — prepare inputs (laptop or HPC)

```bash
pip install -e '.[experiments]'        # numpy + gemmi + biopython

# Copy the complex into its experiment dir (experiments/ is gitignored):
mkdir -p experiments/6G10
cp ../structure-prediction-benchmarking/data/complexes_for_benchmarking/6G10.pdb experiments/6G10/

./scripts/prepare_inputs.py \
    --complex experiments/6G10/6G10.pdb \
    --outdir  experiments/6G10/inputs \
    --derive --contact-cutoff 5.0
```

This writes `experiments/6G10/inputs/{receptor.fasta, effector.fasta,
true_interface.txt, design_region.txt}`. For 6G10 that's receptor chain A
(76 residues), effector chain B (83), and 26 interface residues at 5 Å.

### Interface modes (your two options)

- **Derive from the complex** (default) — receptor residues with a heavy atom
  within `--contact-cutoff` Å of the effector:
  ```bash
  ./scripts/prepare_inputs.py --complex … --outdir … --derive --contact-cutoff 5.0
  ```
- **Provide the interface residues** — a file of 0-based positional indices
  (one per line or comma-separated; `#` comments ignored):
  ```bash
  ./scripts/prepare_inputs.py --complex … --outdir … --interface-file my_interface.txt
  ```

### Chains and design region

- Chains default to **receptor = A, effector = B**; override with
  `--receptor-chain` / `--effector-chain`.
- The **design region** is the receptor stretch steering must NOT mutate (it is
  *protected*, not the mutation target — see
  [`../experiments/benchmarking/README.md`](../experiments/benchmarking/README.md)).
  For a native complex use `--design-region none`; `all` protects the whole chain,
  `--design-region-file FILE` takes an explicit 1-based list.

## Step 2 — run the engine (HPC, GPU)

`run_negative_steering.sh` runs on the host (only `bash` + `singularity`); the
orchestrator it calls does its own `singularity exec --nv` for the engine, so no
host Python packages are needed. **It must run on a GPU node** — under `sbatch`/
`salloc -p jic-gpu --gres=gpu:1`, or just use `sbatch run_from_config.slurm.sh`,
which generates exactly this call inside the right allocation. On a CPU/login node
Boltz fails with "No supported gpu backend found!".

```bash
./scripts/sync_to_hpc.sh           # push code up; then on the HPC:

./scripts/run_negative_steering.sh \
    --name 6G10_test \
    --complex experiments/6G10/6G10.pdb \
    --inputs-dir experiments/6G10/inputs \
    --boltz-container /path/to/containers/boltz2_negsteer.img \
    --plan-extra-args "--mode mild --n-designs 5 --num-seeds 1"
```

Results land in `experiments/6G10_test/run/` (gitignored, HPC-only):
predicted structures, per-design/per-seed metrics, and the steering summary
tables — all scored against `6G10.pdb` as the reference.

Pull a finished run back to the Mac for analysis:

```bash
./scripts/sync_from_hpc.sh --name 6G10_test
```

## Testing multiple inputs

Repeat Step 1+2 per complex (`7B1I`, `7QPX`, …), each in its own
`experiments/<name>/`. The same two scripts handle any two-chain complex; only
the chain IDs and (optionally) the interface mode change.
