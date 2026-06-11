# Local patches to the vendored engine

> ⚠️ Normally `engine/` is vendored verbatim from `receptor-resurfacing-pipeline`
> and must **not** be hand-edited (see the top-level README). The patches below
> are deliberate, documented exceptions. They are **reverted** the moment you run
> `scripts/sync_from_pipeline.py` (which re-vendors clean), and
> `tests/test_engine_staleness.py` will flag the engine as diverged from upstream
> until then — that is expected while a patch is live.

## `bin/boltz2_negative_steering.py` — `cmd_predict_one` resume guard

**Added:** a skip-if-already-predicted check at the top of `cmd_predict_one`.

**Why:** the standalone benchmark runs full experimental complexes (up to ~1400
residues) whose jobs can exceed the SLURM walltime mid-sweep. The pipeline never
needs resume (its design fragments finish quickly), so the engine has none. Without
it, a resubmitted job re-runs every Boltz prediction from scratch, discarding hours
of completed work.

**Safety:** the plan stage samples designs with `random.Random(args.seed)`
(deterministic) and creates design dirs with `mkdir(exist_ok=True)` (never wipes),
so a re-run reproduces the same designs and keeps existing predictions. The guard
additionally confirms the on-disk `prediction.pdb` receptor sequence matches the
design's current `receptor.fasta` before skipping; on any mismatch or error it
recomputes. So it cannot reuse a stale prediction.

**To remove:** run `scripts/sync_from_pipeline.py` to restore the clean vendored
engine (and refresh `engine/_UPSTREAM.json`). Do this once the large-complex sweep
has finished.
