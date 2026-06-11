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
> **Prepare can run anywhere** (needs `pip install -e '.[experiments]'`); **the run
> is HPC-only.**

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
- The **design region** (residues steering may mutate) defaults to the interface
  residues. Use `--design-region all` for the whole receptor chain, or
  `--design-region-file FILE` for an explicit 1-based list.

## Step 2 — run the engine (HPC, GPU)

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
