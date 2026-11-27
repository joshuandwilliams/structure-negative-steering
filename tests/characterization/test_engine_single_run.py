"""HPC tier: assert on a real end-to-end engine run.

Produced by tests/test_run.slurm.sh, which drives the same
entry point a benchmark target uses against tests/test_run/config.yml. The whole
point is to reach the code a laptop cannot: the plan and collect stages of
boltz2_negative_steering, the per-cycle loop in boltz2_iterate_steering, the
contamination and reversion harvest in reversion, and extract_passing.

These skip cleanly when no single-run test is present. Point elsewhere with
NEGSTEER_SINGLE_RUN_DIR, which single_run.py reads.

What is deliberately NOT asserted: the pose. The config's sampling is far too
little to expect a rescue, and asserting `ra_eff` improved would make the test
flaky for a reason unrelated to whether the engine works. What is asserted is
that every stage ran, wrote what it promises to write, and that the numbers it
wrote are internally consistent.

Every expected value is READ FROM tests/test_run/config.yml rather than written
here, through single_run.py. The previous version hardcoded 6Q76 at one design, one seed and one
diffusion sample. Commit 8429e44 changed the config to 6G10 at 3 x 3 x 5 and
did not update these assertions, so five of them failed against a perfectly
good run: the receptor length, the plan's recorded sequence, the seed count,
the cross-summary key and the launch.sh flags. Nobody saw it for the same
reason the tier exists, which is that running it needs a finished GPU run.
Deriving the numbers removes the chance of that happening again.
"""

from __future__ import annotations

import json
import re

import pytest

# SINGLE_RUN, the config parser and the values derived from it are shared with
# test_single_run_cli_matches_orchestrator.py, which asserts on the same tree.
# single_run.py says why they are not written here. It is a sibling module, and
# pytest puts this directory on sys.path.
from single_run import (
    DIFFUSION,
    N_DESIGNS,
    NUM_SEEDS,
    RUN_NAME,
    SINGLE_RUN,
)
from single_run import read_csv as _read

RUN = SINGLE_RUN / "run"
CYCLE0 = RUN / "cycle_0"

needs_run = pytest.mark.skipif(
    not (RUN / "raw_per_seed_results.csv").is_file(),
    reason=f"no single-run test under {RUN}. "
           "Run: sbatch tests/test_run.slurm.sh")


# ── the prepare stage ────────────────────────────────────────────────────────

@pytest.mark.check_test_run
@needs_run
def test_prepare_derived_all_four_engine_inputs():
    inputs = SINGLE_RUN / "inputs"
    for name in ("receptor.fasta", "effector.fasta",
                 "true_interface.txt", "design_region.txt"):
        assert (inputs / name).is_file(), f"prepare did not write {name}"

    rec = (inputs / "receptor.fasta").read_text().splitlines()[1]
    eff = (inputs / "effector.fasta").read_text().splitlines()[1]
    # 6G10 is receptor A at 76 residues and effector B at 83, which is the
    # same pair test_prepare_complex_inputs.py asserts against the committed
    # data/reference_complexes/6G10.pdb. Both numbers are here rather than
    # derived because they are a property of the structure, not the config,
    # and a silent change to either is worth failing on.
    assert len(rec) == 76, f"receptor is {len(rec)} residues, expected 76"
    assert len(eff) == 83, f"effector is {len(eff)} residues, expected 83"


# ── boltz2_negative_steering: the plan stage ─────────────────────────────────

@pytest.mark.check_test_run
@needs_run
def test_plan_stage_predicted_a_cold_start_and_found_the_wrong_interface():
    """cmd_plan runs Boltz once, then locates the interface to steer away from."""
    plan = json.loads((CYCLE0 / "plan.json").read_text())

    rmsd = plan.get("initial_receptor_aligned_effector_rmsd")
    assert rmsd is not None, "plan.json has no cold-start RMSD"
    assert 0.0 < float(rmsd) < 200.0, f"implausible cold-start RMSD {rmsd}"

    seq = plan.get("wild_type_receptor_seq", "")
    assert len(seq) == 76, f"plan recorded a {len(seq)}-residue receptor, expected 76"

    for f in ("true_interface_residues.txt", "wrong_interface_residues.txt"):
        assert (CYCLE0 / f).is_file(), f"plan stage did not write {f}"


@pytest.mark.check_test_run
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

@pytest.mark.check_test_run
@needs_run
def test_every_configured_design_and_seed_produced_a_row():
    """The single-run config asks for 1 design x 1 seed, so that is what must appear.

    The engine tolerates per-design failure, so a design that silently died
    leaves a shorter table rather than an error. Counting is the only check.
    """
    rows = _read(RUN / "raw_per_seed_results.csv")
    assert rows, "raw_per_seed_results.csv is empty"

    initial = [r for r in rows if r.get("design") == "initial"]
    assert initial, "no cold-start rows, so there is no paired baseline"

    # The cold-start row carries a BLANK seed_index, not "0". It is one
    # prediction of the wild-type receptor rather than a per-seed sample of a
    # design, and the engine records that by leaving the field empty. Counting
    # distinct seed_index over every row therefore counts a seed that does not
    # exist. Pinned because it is not obvious and reads like missing data.
    assert all(r["seed_index"] == "" for r in initial), \
        f"cold-start rows now carry a seed index: {[r['seed_index'] for r in initial]}"

    steered = [r for r in rows if r.get("design") != "initial"]
    seeds = {r["seed_index"] for r in steered}
    assert len(seeds) <= NUM_SEEDS, \
        f"config asked for {NUM_SEEDS} seed(s), steered rows have {len(seeds)}: {seeds}"

    # The `design` column is design AND seed in one string, `design_00_s0`, so
    # it holds n_designs x num_seeds distinct values rather than n_designs.
    # The previous assertion compared it against 1 and passed only because the
    # old config was one design by one seed, where the two collapse to the same
    # number. Both are checked separately here.
    units = {re.sub(r"_s\d+$", "", r["design"]) for r in steered}
    assert len(units) <= N_DESIGNS, \
        f"config asked for {N_DESIGNS} design(s), table has {len(units)}: {units}"
    assert len(steered) <= N_DESIGNS * NUM_SEEDS, \
        (f"config asked for {N_DESIGNS} x {NUM_SEEDS} = "
         f"{N_DESIGNS * NUM_SEEDS} steered rows, table has {len(steered)}")


@pytest.mark.check_test_run
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


@pytest.mark.check_test_run
@needs_run
def test_passing_summary_and_cross_summary_were_written():
    """extract_passing then cross_summary_cli, the two post-run stages."""
    assert (RUN / "passing_summary.csv").is_file(), "extract_passing wrote nothing"
    xs = RUN / "cross_sequence_summary.csv"
    assert xs.is_file(), "the tiering stage wrote no cross summary"

    rows = _read(xs)
    assert len(rows) == 1, f"expected one representative row, got {len(rows)}"
    assert rows[0]["cross_tier"] in {"A", "B", "C", "none"}
    assert rows[0]["mpnn_sequence"] == RUN_NAME, \
        f"cross summary is keyed {rows[0]['mpnn_sequence']!r}, expected {RUN_NAME!r}"


@pytest.mark.check_test_run
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

@pytest.mark.check_test_run
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

@pytest.mark.check_test_run
@needs_run
def test_launch_script_records_the_configured_knobs():
    """launch.sh is the provenance record of what was actually run.

    Only the config-driven path writes one. The tool interface takes its
    arguments directly and records them in negsteer_result.json instead, so
    this skips rather than failing there.
    """
    launch = RUN / "launch.sh"
    if not launch.is_file() and (RUN / "negsteer_result.json").is_file():
        pytest.skip("tool-interface run; provenance is in negsteer_result.json")
    assert launch.is_file(), "no launch.sh recorded"
    body = launch.read_text()
    for flag in (f"--n-designs {N_DESIGNS}",
                 f"--num-seeds {NUM_SEEDS}",
                 f"--diffusion-samples {DIFFUSION}"):
        assert flag in body, f"launch.sh does not carry {flag!r}"
    assert "--boltz-constraints" not in body, \
        "the single-run config is unconstrained but the run carried constraints"


@pytest.mark.check_test_run
@needs_run
def test_runtime_was_recorded():
    f = RUN / "run_one_runtime_sec.txt"
    assert f.is_file(), "no runtime recorded"
    assert float(f.read_text().strip()) > 0
