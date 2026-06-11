# experiments/

This directory holds negative-steering runs — both the historical ones
migrated from the old HPC scratch folder (`receptor_design/negative_steering/`)
and any new ones. **The whole tree is gitignored** (only this README is
tracked) and is **HPC-authoritative**: it is never pushed up by
`scripts/sync_to_hpc.sh`, so a code sync can never overwrite or delete results.

## Layout

One subdirectory per experiment, e.g.:

```
experiments/
├── runs/                      # migrated: prior negative-steering run outputs
├── logs/                      # migrated: run logs
├── solved_structures/         # migrated: reference structures used
├── ChimeraX_visualisations/   # migrated: figures
├── resurface_pipeline_test/   # migrated: end-to-end test inputs/outputs
└── notes/                     # migrated: development notes (.md / .docx)
```

(The exact set reflects what was migrated from the scratch folder.)

## Getting results onto your Mac

Results live on the cluster. Pull a single one down for local analysis:

```bash
./scripts/sync_from_hpc.sh --list            # see what's on the HPC
./scripts/sync_from_hpc.sh --name runs        # pull experiments/runs/ down
```

## Provenance of the migrated data

These were produced during development under the receptor-resurfacing-pipeline,
in `/Volumes/HPC-Home/receptor_design/negative_steering/`. The development-only
Python scripts from that folder (waypoint generation, multiseed variance,
version comparison, the sequence registry) were intentionally **not** carried
over — the canonical engine now comes from the pipeline via
`scripts/sync_from_pipeline.py`.
