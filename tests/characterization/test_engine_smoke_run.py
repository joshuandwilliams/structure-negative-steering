"""HPC tier: assert on a real end-to-end engine run.

Produced by tests/run_smoke_negative_steering.slurm.sh, which drives the same
entry point a benchmark target uses against tests/smoke/config.yml. The whole
point is to reach the code a laptop cannot: the plan and collect stages of
boltz2_negative_steering, the per-cycle loop in boltz2_iterate_steering, the
contamination and reversion harvest in reversion, and extract_passing.

These skip cleanly when no smoke run is present. Point elsewhere with
NEGSTEER_SMOKE_DIR.

What is deliberately NOT asserted: the pose. One design, one seed and one
diffusion sample is far too little sampling to expect a rescue, and asserting
`ra_eff` improved would make the test flaky for a reason unrelated to whether
the engine works. What is asserted is that every stage ran, wrote what it
promises to write, and that the numbers it wrote are internally consistent.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE = Path(os.environ.get("NEGSTEER_SMOKE_DIR", REPO_ROOT / "tests" / "smoke"))
RUN = SMOKE / "run"
CYCLE0 = RUN / "cycle_0"

needs_run = pytest.mark.skipif(
    not (RUN / "raw_per_seed_results.csv").is_file(),
    reason=f"no smoke run under {RUN}. "
           "Run: sbatch tests/run_smoke_negative_steering.slurm.sh")


def _read(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


# ── the prepare stage ────────────────────────────────────────────────────────

@pytest.mark.hpc
@needs_run
def test_prepare_derived_all_four_engine_inputs():
    inputs = SMOKE / "inputs"
    for name in ("receptor.fasta", "effector.fasta",
                 "true_interface.txt", "design_region.txt"):
        assert (inputs / name).is_file(), f"prepare did not write {name}"

    rec = (inputs / "receptor.fasta").read_text().splitlines()[1]
    eff = (inputs / "effector.fasta").read_text().splitlines()[1]
    # 6Q76 is receptor A at 73 residues, effector B at 68.
    assert len(rec) == 73, f"receptor is {len(rec)} residues, expected 73"
    assert len(eff) == 68, f"effector is {len(eff)} residues, expected 68"


# ── boltz2_negative_steering: the plan stage ─────────────────────────────────

@pytest.mark.hpc
@needs_run
def test_plan_stage_predicted_a_cold_start_and_found_the_wrong_interface():
    """cmd_plan runs Boltz once, then locates the interface to steer away from."""
    plan = json.loads((CYCLE0 / "plan.json").read_text())

    rmsd = plan.get("initial_receptor_aligned_effector_rmsd")
    assert rmsd is not None, "plan.json has no cold-start RMSD"
    assert 0.0 < float(rmsd) < 200.0, f"implausible cold-start RMSD {rmsd}"

    seq = plan.get("wild_type_receptor_seq", "")
    assert len(seq) == 73, f"plan recorded a {len(seq)}-residue receptor, expected 73"

    for f in ("true_interface_residues.txt", "wrong_interface_residues.txt"):
        assert (CYCLE0 / f).is_file(), f"plan stage did not write {f}"


@pytest.mark.hpc
@needs_run
def test_steering_mutations_avoid_the_protected_interface():
    """The engine must not mutate the residues it was told to protect.

    This is the central invariant of the method. Steering that mutated the true
    interface would be destroying the binding site rather than removing a
    competing one, and every downstream number would still look plausible.
    """
    true_iface = {
        int(x) for x in (CYCLE0 / "true_interface_residues.txt").read_text().split()
        if x.strip().isdigit()
    }
    assert true_iface, "no true-interface residues recorded"

    rows = _read(RUN / "raw_per_seed_results.csv")
    steered = [r for r in rows if r.get("design") not in ("initial", "", None)]
    if not steered:
        pytest.skip("cold start passed, so no steered design was produced")

    for r in steered:
        chimerax = r.get("steered_mutations_chimerax", "")
        if not chimerax:
            continue
        # Format is "/A:12,45,60", 1-based. The interface file is 0-based.
        positions = {int(p) - 1 for p in chimerax.split(":")[-1].split(",")
                     if p.strip().isdigit()}
        overlap = positions & true_iface
        assert not overlap, (
            f"design {r.get('design')} mutated protected interface residues "
            f"{sorted(overlap)} (0-based)"
        )


# ── boltz2_iterate_steering and extract_passing ──────────────────────────────

@pytest.mark.hpc
@needs_run
def test_every_configured_design_and_seed_produced_a_row():
    """The smoke config asks for 1 design x 1 seed, so that is what must appear.

    The engine tolerates per-design failure, so a design that silently died
    leaves a shorter table rather than an error. Counting is the only check.
    """
    rows = _read(RUN / "raw_per_seed_results.csv")
    assert rows, "raw_per_seed_results.csv is empty"

    initial = [r for r in rows if r.get("design") == "initial"]
    assert initial, "no cold-start rows, so there is no paired baseline"

    seeds = {r["seed_index"] for r in rows}
    assert len(seeds) == 1, f"config asked for 1 seed, table has {len(seeds)}: {seeds}"

    designs = {r["design"] for r in rows if r.get("design") != "initial"}
    assert len(designs) <= 1, \
        f"config asked for 1 design, table has {len(designs)}: {designs}"


@pytest.mark.hpc
@needs_run
def test_every_row_carries_the_metrics_the_analyses_read():
    """A row missing these is invisible to the analyses rather than an error."""
    rows = _read(RUN / "raw_per_seed_results.csv")
    required = ("status", "steered_ra_eff_vs_truth", "seed_index", "design")
    missing = [c for c in required if c not in rows[0]]
    assert not missing, f"raw_per_seed_results.csv lacks {missing}"

    for r in rows:
        v = r.get("steered_ra_eff_vs_truth", "")
        if v not in ("", None):
            assert 0.0 <= float(v) < 500.0, f"implausible ra_eff {v!r}"


@pytest.mark.hpc
@needs_run
def test_passing_summary_and_cross_summary_were_written():
    """extract_passing then cross_summary_cli, the two post-run stages."""
    assert (RUN / "passing_summary.csv").is_file(), "extract_passing wrote nothing"
    xs = RUN / "cross_sequence_summary.csv"
    assert xs.is_file(), "the tiering stage wrote no cross summary"

    rows = _read(xs)
    assert len(rows) == 1, f"expected one representative row, got {len(rows)}"
    assert rows[0]["cross_tier"] in {"A", "B", "C", "none"}
    assert rows[0]["mpnn_sequence"] == "6Q76_smoke", \
        f"cross summary is keyed {rows[0]['mpnn_sequence']!r}, expected 6Q76_smoke"


@pytest.mark.hpc
@needs_run
def test_tier_is_consistent_with_the_passing_gate():
    """With 1 seed only tier A or none can occur, which is worth pinning."""
    row = _read(RUN / "cross_sequence_summary.csv")[0]
    n_passing = len(_read(RUN / "passing_summary.csv"))
    if row["cross_tier"] == "none":
        assert n_passing == 0, "tier none with a populated passing_summary.csv"
    else:
        assert row["cross_tier"] == "A", (
            f"tier {row['cross_tier']} is impossible with num_seeds=1, "
            "which admits only A or none"
        )
        assert n_passing > 0


# ── reversion ────────────────────────────────────────────────────────────────

@pytest.mark.hpc
@needs_run
def test_reversion_stage_recorded_a_verdict():
    """reversion.py runs whether or not any mutation needed reverting.

    Its absence would mean the contamination check never ran, and a
    contaminated design would be scored as a clean rescue.
    """
    produced = [f for f in ("contaminated.json", "reversion_plan.json",
                            "reversion_results.json")
                if (CYCLE0 / f).is_file()]
    assert produced, (
        "the reversion stage left no record. Expected at least one of "
        "contaminated.json, reversion_plan.json, reversion_results.json"
    )


# ── the run is reproducible from what it recorded ────────────────────────────

@pytest.mark.hpc
@needs_run
def test_launch_script_records_the_configured_knobs():
    """launch.sh is the provenance record of what was actually run."""
    launch = RUN / "launch.sh"
    assert launch.is_file(), "no launch.sh recorded"
    body = launch.read_text()
    for flag in ("--n-designs 1", "--num-seeds 1", "--diffusion-samples 1"):
        assert flag in body, f"launch.sh does not carry {flag!r}"
    assert "--boltz-constraints" not in body, \
        "the smoke config is unconstrained but the run carried constraints"


@pytest.mark.hpc
@needs_run
def test_runtime_was_recorded():
    f = RUN / "run_one_runtime_sec.txt"
    assert f.is_file(), "no runtime recorded"
    assert float(f.read_text().strip()) > 0
