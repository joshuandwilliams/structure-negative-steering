"""Tests for NegativeSteeringRun.from_workdir, on built run directories.

from_workdir turns a finished run directory into the typed object every
cohort query runs against. All of its grouping comes from directory names:

    steered/<design>_s<seed>/
    reversions/rev_<design>_s<parent_seed>_s<rev_seed>/

A name parsed wrongly groups one design's seeds under another, and the tier
that design is then given is computed over the wrong set of predictions. The
counts stay plausible, so nothing downstream reports an error.

These tests build the directories with the grouping stated up front and
assert the object that comes back matches it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import runtree as rt

REPO_ROOT = Path(__file__).resolve().parents[2]


def _deps() -> bool:
    if importlib.util.find_spec("numpy") is None:
        return False
    return any(importlib.util.find_spec(m) is not None for m in ("gemmi", "Bio"))


pytestmark = pytest.mark.skipif(
    not _deps(), reason="needs numpy plus gemmi or biopython")


@pytest.fixture(scope="module")
def nsr():
    sys.path.insert(0, str(REPO_ROOT / "bin"))
    spec = importlib.util.spec_from_file_location(
        "negative_steering_run", REPO_ROOT / "bin" / "negative_steering_run.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _build(root, **kw):
    return rt.deep(root, **kw)


# ── the cold start is required ───────────────────────────────────────────────

@pytest.mark.logic
def test_a_directory_with_no_cycle_is_none(nsr, tmp_path):
    assert nsr.NegativeSteeringRun.from_workdir(
        workdir=tmp_path, mpnn_sequence_id="seqA") is None


@pytest.mark.logic
def test_a_cycle_with_no_cold_start_is_none(nsr, tmp_path):
    """Without the unsteered prediction there is nothing to compare against,
    so the run cannot be assembled at all."""
    _build(tmp_path, seeds=())
    assert nsr.NegativeSteeringRun.from_workdir(
        workdir=tmp_path, mpnn_sequence_id="seqA") is None


@pytest.mark.logic
def test_the_cold_start_seeds_are_read_in_order(nsr, tmp_path):
    _build(tmp_path, seeds=(0, 1, 2))
    run = nsr.NegativeSteeringRun.from_workdir(
        workdir=tmp_path, mpnn_sequence_id="seqA", num_seeds=3)
    assert run is not None
    assert list(run.cold_start.seed_indices) == [0, 1, 2]
    assert run.mpnn_sequence_id == "seqA"


# ── the row type sidecar ─────────────────────────────────────────────────────

@pytest.mark.logic
def test_the_row_type_defaults_to_steered(nsr, tmp_path):
    _build(tmp_path)
    run = nsr.NegativeSteeringRun.from_workdir(
        workdir=tmp_path, mpnn_sequence_id="seqA")
    assert run.row_type == "steered"


@pytest.mark.logic
def test_a_sidecar_names_the_row_type(nsr, tmp_path):
    """A control read as a steered row would be counted as a result."""
    _build(tmp_path, row_type="control_polyA")
    run = nsr.NegativeSteeringRun.from_workdir(
        workdir=tmp_path, mpnn_sequence_id="seqA")
    assert run.row_type == "control_polyA"


@pytest.mark.logic
def test_an_empty_sidecar_falls_back_to_steered(nsr, tmp_path):
    _build(tmp_path, row_type="")
    run = nsr.NegativeSteeringRun.from_workdir(
        workdir=tmp_path, mpnn_sequence_id="seqA")
    assert run.row_type == "steered"


@pytest.mark.logic
def test_an_explicit_row_type_beats_the_sidecar(nsr, tmp_path):
    _build(tmp_path, row_type="control_polyA")
    run = nsr.NegativeSteeringRun.from_workdir(
        workdir=tmp_path, mpnn_sequence_id="seqA", row_type="control_scrambled")
    assert run.row_type == "control_scrambled"


# ── grouping the steered stage ───────────────────────────────────────────────

@pytest.mark.logic
def test_each_design_gets_its_own_stage_with_its_own_seeds(nsr, tmp_path):
    """This is the grouping that matters. design_00's three seeds and
    design_01's two must not be pooled."""
    _build(tmp_path, seeds=(0,),
           steered=[("design_00", [0, 1, 2], [5, 7]),
                    ("design_01", [0, 1], [9])])
    run = nsr.NegativeSteeringRun.from_workdir(
        workdir=tmp_path, mpnn_sequence_id="seqA", num_seeds=3)
    assert sorted(run.steered) == ["design_00", "design_01"]
    assert list(run.steered["design_00"].seed_indices) == [0, 1, 2]
    assert list(run.steered["design_01"].seed_indices) == [0, 1]


@pytest.mark.logic
def test_the_mutated_positions_come_from_the_designs_own_file(nsr, tmp_path):
    _build(tmp_path, steered=[("design_00", [0], [5, 7]),
                              ("design_01", [0], [21])])
    run = nsr.NegativeSteeringRun.from_workdir(
        workdir=tmp_path, mpnn_sequence_id="seqA")
    assert list(run.steered["design_00"].mutated_positions) == [5, 7]
    assert list(run.steered["design_01"].mutated_positions) == [21]


@pytest.mark.logic
def test_a_design_with_no_mutations_file_gets_an_empty_set_not_a_drop(nsr,
                                                                      tmp_path):
    """Dropping the stage would lose the design entirely. An empty set keeps
    it, and every contamination test over an empty set is correctly false."""
    _build(tmp_path, steered=[("design_00", [0], [])])
    run = nsr.NegativeSteeringRun.from_workdir(
        workdir=tmp_path, mpnn_sequence_id="seqA")
    assert "design_00" in run.steered
    assert list(run.steered["design_00"].mutated_positions) == []


@pytest.mark.logic
def test_a_directory_that_is_not_a_design_slot_is_ignored(nsr, tmp_path):
    cycle = _build(tmp_path, steered=[("design_00", [0], [5])])
    (cycle / "steered" / "notadesign").mkdir()
    (cycle / "steered" / "design_00_sX").mkdir()
    (cycle / "steered" / "stray.txt").write_text("x")
    run = nsr.NegativeSteeringRun.from_workdir(
        workdir=tmp_path, mpnn_sequence_id="seqA")
    assert sorted(run.steered) == ["design_00"]


@pytest.mark.logic
def test_a_seed_missing_its_boltz_output_is_skipped(nsr, tmp_path):
    cycle = _build(tmp_path, steered=[("design_00", [0, 1], [5])])
    # Remove seed 1's confidence, which makes it unscoreable.
    conf = (cycle / "steered" / "design_00_s1" / "boltz_results_input"
            / "predictions" / "input" / "confidence_input_model_0.json")
    conf.unlink()
    run = nsr.NegativeSteeringRun.from_workdir(
        workdir=tmp_path, mpnn_sequence_id="seqA")
    assert list(run.steered["design_00"].seed_indices) == [0]


@pytest.mark.logic
def test_a_design_whose_every_seed_is_unusable_is_dropped(nsr, tmp_path):
    cycle = _build(tmp_path, steered=[("design_00", [0], [5])])
    (cycle / "steered" / "design_00_s0" / "boltz_results_input"
     / "predictions" / "input" / "confidence_input_model_0.json").unlink()
    run = nsr.NegativeSteeringRun.from_workdir(
        workdir=tmp_path, mpnn_sequence_id="seqA")
    assert run.steered == {}


# ── grouping the reversion stage ─────────────────────────────────────────────

@pytest.mark.logic
def test_reversions_are_grouped_by_their_parent_slot(nsr, tmp_path):
    """The name carries three numbers: the design, the seed being reverted,
    and the reversion's own seed. Grouping on the wrong one attributes a
    reversion to a design that never produced it."""
    _build(tmp_path,
           steered=[("design_00", [0, 1], [5])],
           reversions=[("design_00", 0, [0, 1]), ("design_00", 1, [0])])
    run = nsr.NegativeSteeringRun.from_workdir(
        workdir=tmp_path, mpnn_sequence_id="seqA", num_seeds=2)
    assert len(run.reversion) == 2
    for stage in run.reversion.values():
        assert stage.applies_to_designs == ("design_00",)


@pytest.mark.logic
def test_a_reversion_directory_with_an_unparseable_name_is_ignored(nsr,
                                                                   tmp_path):
    cycle = _build(tmp_path, steered=[("design_00", [0], [5])],
                   reversions=[("design_00", 0, [0])])
    (cycle / "reversions" / "rev_nonsense").mkdir()
    run = nsr.NegativeSteeringRun.from_workdir(
        workdir=tmp_path, mpnn_sequence_id="seqA")
    assert len(run.reversion) == 1


@pytest.mark.logic
def test_no_reversions_leaves_the_mapping_empty(nsr, tmp_path):
    _build(tmp_path, steered=[("design_00", [0], [5])])
    run = nsr.NegativeSteeringRun.from_workdir(
        workdir=tmp_path, mpnn_sequence_id="seqA")
    assert run.reversion == {}


# ── reading a different model sample ─────────────────────────────────────────

@pytest.mark.logic
def test_a_different_model_sample_is_read(nsr, tmp_path):
    """Boltz writes several samples per prediction; the run has to take the
    one it was asked for rather than whichever sorts first."""
    cycle_dir = tmp_path / "cycle_0"
    cycle_dir.mkdir(parents=True)
    rt.boltz_outputs(cycle_dir / "initial", sample=2)
    run = nsr.NegativeSteeringRun.from_workdir(
        workdir=tmp_path, mpnn_sequence_id="seqA", model_sample=2)
    assert run is not None
    assert list(run.cold_start.seed_indices) == [0]
    assert nsr.NegativeSteeringRun.from_workdir(
        workdir=tmp_path, mpnn_sequence_id="seqA", model_sample=0) is None
