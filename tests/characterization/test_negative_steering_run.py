"""Unit tests for bin/negative_steering_run.py (Phase 4 Tier 5).

Uses MagicMock PSPs (same pattern as test_stage_result.py) so the cross-
stage aggregation logic is testable without real PDBs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin"))

import negative_steering_run as nsr  # noqa: E402
import pipeline_thresholds as pt  # noqa: E402
import stage_result as sr  # noqa: E402
from position_set import PositionSet  # noqa: E402


def _fake_pred(ra_eff, intact_rmsd, contact_positions=()):
    m = MagicMock()
    m.ra_eff_vs.return_value = ra_eff
    m.independent_receptor_rmsd_vs.return_value = intact_rmsd
    cs = PositionSet(list(contact_positions), chain="A", frame="prediction")
    m.receptor_contact_residues.return_value = cs
    return m


def _fake_conf():
    return MagicMock()


def _make_cold_start(predictions, config_id="design_03_seq_01"):
    return sr.StageResult(
        stage_type="cold_start", config_id=config_id,
        predictions=predictions,
        confidences=[_fake_conf() for _ in predictions],
        seed_indices=list(range(len(predictions))))


def _make_steered(predictions, design_id="design_03",
                  mutated_positions=None):
    if mutated_positions is None:
        mutated_positions = PositionSet([5, 7, 12], chain="A", frame="prediction")
    return sr.StageResult(
        stage_type="steered", config_id=design_id,
        predictions=predictions,
        confidences=[_fake_conf() for _ in predictions],
        seed_indices=list(range(len(predictions))),
        mutated_positions=mutated_positions)


def _make_reversion(predictions, rev_id="rev_design_03_abc",
                    applies_to=("design_03",),
                    mutated_positions=None):
    if mutated_positions is None:
        mutated_positions = PositionSet([5, 7, 12], chain="A", frame="prediction")
    return sr.StageResult(
        stage_type="reversion", config_id=rev_id,
        predictions=predictions,
        confidences=[_fake_conf() for _ in predictions],
        seed_indices=list(range(len(predictions))),
        mutated_positions=mutated_positions,
        applies_to_designs=list(applies_to))


@pytest.fixture
def truth():
    return MagicMock()


@pytest.fixture
def contam():
    return PositionSet([5, 7, 12], chain="A", frame="prediction")


# ── Construction ───────────────────────────────────────────────────


@pytest.mark.logic
class TestConstruction:
    def test_basic(self, tmp_path, truth):
        cs = _make_cold_start([_fake_pred(2.0, 1.0)] * 3)
        run = nsr.NegativeSteeringRun(
            mpnn_sequence_id="design_03_seq_01",
            workdir=tmp_path,
            row_type="steered",
            cold_start=cs,
            truth=truth,
        )
        assert run.mpnn_sequence_id == "design_03_seq_01"
        assert run.num_seeds == 3

    def test_empty_id_raises(self, tmp_path, truth):
        with pytest.raises(ValueError, match="mpnn_sequence_id"):
            nsr.NegativeSteeringRun(
                mpnn_sequence_id="", workdir=tmp_path,
                row_type="steered",
                cold_start=_make_cold_start([_fake_pred(2.0, 1.0)]),
                truth=truth)

    def test_invalid_row_type_raises(self, tmp_path, truth):
        with pytest.raises(ValueError, match="row_type"):
            nsr.NegativeSteeringRun(
                mpnn_sequence_id="x", workdir=tmp_path,
                row_type="bogus",
                cold_start=_make_cold_start([_fake_pred(2.0, 1.0)]),
                truth=truth)

    def test_wrong_cold_start_type_raises(self, tmp_path, truth):
        wrong = _make_steered([_fake_pred(2.0, 1.0)])
        with pytest.raises(ValueError, match="cold_start"):
            nsr.NegativeSteeringRun(
                mpnn_sequence_id="x", workdir=tmp_path,
                row_type="steered", cold_start=wrong, truth=truth)


# ── Cold-start path (no steering ran) ──────────────────────────────


@pytest.mark.logic
class TestColdStartPath:
    def test_all_clean_cold_start_is_tier_A(self, tmp_path, truth):
        cs = _make_cold_start([_fake_pred(2.0, 1.0)] * 3)
        run = nsr.NegativeSteeringRun(
            mpnn_sequence_id="x", workdir=tmp_path, row_type="steered",
            cold_start=cs, truth=truth)
        t = pt.PipelineInternalThresholds.default()
        assert run.outcome(t) == "no_reversion"
        assert run.n_pass(t) == 3
        assert run.n_seeds_total() == 3
        assert run.tier(t) == "A"

    def test_partial_cold_start_pass_is_tier_B(self, tmp_path, truth):
        cs = _make_cold_start(
            [_fake_pred(2.0, 1.0), _fake_pred(10.0, 1.0), _fake_pred(2.0, 1.0)])
        run = nsr.NegativeSteeringRun(
            mpnn_sequence_id="x", workdir=tmp_path, row_type="steered",
            cold_start=cs, truth=truth)
        t = pt.PipelineInternalThresholds.default()
        # 2 clean, 1 fail_structural — but outcome is still no_reversion
        # because steering didn't run; n_pass=2 (only 'clean' counts)
        assert run.outcome(t) == "no_reversion"
        assert run.n_pass(t) == 2
        assert run.tier(t) == "B"

    def test_zero_clean_cold_start_is_tier_none(self, tmp_path, truth):
        cs = _make_cold_start([_fake_pred(10.0, 1.0)] * 3)
        run = nsr.NegativeSteeringRun(
            mpnn_sequence_id="x", workdir=tmp_path, row_type="steered",
            cold_start=cs, truth=truth)
        t = pt.PipelineInternalThresholds.default()
        # The "wrong placement no contamination" case from CL-3 / CL-1
        # outcome=no_reversion BUT tier=none — the bug fix
        assert run.outcome(t) == "no_reversion"
        assert run.n_pass(t) == 0
        assert run.tier(t) == "none"


# ── Steered + reversion path ────────────────────────────────────


@pytest.mark.logic
class TestSteeredOnlyPath:
    """Steering ran but no reversion was triggered (CL-3 rule didn't fire)."""

    def test_all_clean_steered_3of3_tier_A(self, tmp_path, truth):
        cs = _make_cold_start([_fake_pred(10.0, 1.0)] * 3)  # failed cold start
        steered = {
            "design_03": _make_steered(
                [_fake_pred(2.0, 1.0, contact_positions=[100])] * 3),
        }
        run = nsr.NegativeSteeringRun(
            mpnn_sequence_id="x", workdir=tmp_path, row_type="steered",
            cold_start=cs, steered=steered, truth=truth,
            contamination_positions=PositionSet(
                [5, 7, 12], chain="A", frame="prediction"))
        t = pt.PipelineInternalThresholds.default()
        assert run.outcome(t) == "no_reversion"  # reversion didn't run
        assert run.n_pass(t) == 3
        assert run.tier(t) == "A"


@pytest.mark.logic
class TestReversionPath:
    """Steering ran + reversion ran."""

    def test_reversion_pose_holds_is_tier_A(self, tmp_path, truth, contam):
        cs = _make_cold_start([_fake_pred(10.0, 1.0)] * 3)
        # 3 contaminated steered → reversion triggers
        steered = {
            "design_03": _make_steered(
                [_fake_pred(2.0, 1.0, contact_positions=[5])] * 3),
        }
        # Reversion all pose_holds
        reversion = {
            "rev_design_03_abc": _make_reversion(
                [_fake_pred(2.0, 1.0, contact_positions=[100])] * 3,
                applies_to=("design_03",)),
        }
        run = nsr.NegativeSteeringRun(
            mpnn_sequence_id="x", workdir=tmp_path, row_type="steered",
            cold_start=cs, steered=steered, reversion=reversion,
            truth=truth, contamination_positions=contam)
        t = pt.PipelineInternalThresholds.default()
        verdicts = run.per_seed_final_verdicts(t)
        # All 3 should be 'reverted' stage with pose_holds verdict
        assert all(e["stage_run"] == "reverted" for e in verdicts)
        assert all(e["verdict"] == "pose_holds" for e in verdicts)
        assert run.outcome(t) == "pose_holds"
        assert run.n_pass(t) == 3
        assert run.tier(t) == "A"

    def test_reversion_new_contamination_is_tier_none(
        self, tmp_path, truth, contam
    ):
        cs = _make_cold_start([_fake_pred(10.0, 1.0)] * 3)
        steered = {
            "design_03": _make_steered(
                [_fake_pred(2.0, 1.0, contact_positions=[5])] * 3),
        }
        # Reversion all new_contamination
        reversion = {
            "rev_design_03_abc": _make_reversion(
                [_fake_pred(2.0, 1.0, contact_positions=[5])] * 3,
                applies_to=("design_03",)),
        }
        run = nsr.NegativeSteeringRun(
            mpnn_sequence_id="x", workdir=tmp_path, row_type="steered",
            cold_start=cs, steered=steered, reversion=reversion,
            truth=truth, contamination_positions=contam)
        t = pt.PipelineInternalThresholds.default()
        assert run.outcome(t) == "new_contamination"
        assert run.n_pass(t) == 0
        assert run.tier(t) == "none"

    def test_mixed_reversion_outcomes(self, tmp_path, truth, contam):
        """2 pose_holds, 1 new_contamination → mixed; majority is pose_holds."""
        cs = _make_cold_start([_fake_pred(10.0, 1.0)] * 3)
        steered = {
            "design_03": _make_steered(
                [_fake_pred(2.0, 1.0, contact_positions=[5])] * 3),
        }
        reversion = {
            "rev_design_03_abc": _make_reversion(
                [
                    _fake_pred(2.0, 1.0, contact_positions=[100]),  # pose_holds
                    _fake_pred(2.0, 1.0, contact_positions=[100]),  # pose_holds
                    _fake_pred(2.0, 1.0, contact_positions=[5]),    # new_contam
                ],
                applies_to=("design_03",)),
        }
        run = nsr.NegativeSteeringRun(
            mpnn_sequence_id="x", workdir=tmp_path, row_type="steered",
            cold_start=cs, steered=steered, reversion=reversion,
            truth=truth, contamination_positions=contam)
        t = pt.PipelineInternalThresholds.default()
        assert run.outcome(t) == "pose_holds"  # 2 pass > 1 new_contam
        assert run.n_pass(t) == 2
        assert run.tier(t) == "B"


@pytest.mark.logic
class TestReversionForDesign:
    def test_finds_matching_reversion(self, tmp_path, truth, contam):
        cs = _make_cold_start([_fake_pred(2.0, 1.0)])
        rev_a = _make_reversion([_fake_pred(2.0, 1.0)],
                                rev_id="rev_a", applies_to=("design_03",))
        rev_b = _make_reversion([_fake_pred(2.0, 1.0)],
                                rev_id="rev_b", applies_to=("design_04",))
        run = nsr.NegativeSteeringRun(
            mpnn_sequence_id="x", workdir=tmp_path, row_type="steered",
            cold_start=cs, reversion={"rev_a": rev_a, "rev_b": rev_b},
            truth=truth, contamination_positions=contam)
        assert run._reversion_for_design("design_03") is rev_a
        assert run._reversion_for_design("design_04") is rev_b
        assert run._reversion_for_design("design_99") is None

    def test_dedup_one_reversion_applies_to_many(self, tmp_path, truth, contam):
        """Same reversion attempt represents multiple steered designs."""
        cs = _make_cold_start([_fake_pred(2.0, 1.0)])
        rev = _make_reversion([_fake_pred(2.0, 1.0)],
                              applies_to=("design_03", "design_05"))
        run = nsr.NegativeSteeringRun(
            mpnn_sequence_id="x", workdir=tmp_path, row_type="steered",
            cold_start=cs, reversion={"rev": rev},
            truth=truth, contamination_positions=contam)
        # Both designs map to the same reversion
        assert run._reversion_for_design("design_03") is rev
        assert run._reversion_for_design("design_05") is rev


@pytest.mark.logic
class TestImmutability:
    def test_frozen(self, tmp_path, truth):
        cs = _make_cold_start([_fake_pred(2.0, 1.0)])
        run = nsr.NegativeSteeringRun(
            mpnn_sequence_id="x", workdir=tmp_path, row_type="steered",
            cold_start=cs, truth=truth)
        with pytest.raises((AttributeError, Exception)):
            run.mpnn_sequence_id = "new"  # type: ignore


# ── locating Boltz output on disk ────────────────────────────────────────────
#
# The pipeline has written two layouts. Returning (None, None) rather than a
# half-built pair is what keeps a seed out of the run entirely, instead of
# putting a prediction in with no confidence beside it.

def _boltz_tree(root: Path, *, flat=False, conf=True, sample=0):
    pred = root / "boltz_results_input" / "predictions" / "input"
    pdb_dir = pred if flat else pred / "pdb"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    (pdb_dir / f"input_model_{sample}.pdb").write_text("ATOM\n")
    if conf:
        (pred / f"confidence_input_model_{sample}.json").write_text("{}")
    return pred


@pytest.mark.logic
def test_the_nested_pdb_layout_is_found(tmp_path):
    _boltz_tree(tmp_path)
    pdb, conf = nsr._find_boltz_outputs(tmp_path)
    assert pdb is not None and pdb.name == "input_model_0.pdb"
    assert conf is not None


@pytest.mark.logic
def test_the_flat_pdb_layout_is_found_too(tmp_path):
    """Some pipeline versions wrote the PDB without the pdb/ subdirectory."""
    _boltz_tree(tmp_path, flat=True)
    pdb, _ = nsr._find_boltz_outputs(tmp_path)
    assert pdb is not None


@pytest.mark.logic
def test_a_missing_prediction_root_is_none(tmp_path):
    assert nsr._find_boltz_outputs(tmp_path) == (None, None)


@pytest.mark.logic
def test_a_missing_confidence_file_yields_none_for_it(tmp_path):
    _boltz_tree(tmp_path, conf=False)
    pdb, conf = nsr._find_boltz_outputs(tmp_path)
    assert pdb is not None and conf is None


@pytest.mark.logic
def test_a_different_model_sample_is_selected(tmp_path):
    _boltz_tree(tmp_path, sample=3)
    assert nsr._find_boltz_outputs(tmp_path, model_sample=3)[0] is not None
    assert nsr._find_boltz_outputs(tmp_path, model_sample=0)[0] is None


# ── mutations.tsv ────────────────────────────────────────────────────────────

@pytest.mark.logic
def test_mutations_are_read_as_a_designed_frame_position_set(tmp_path):
    f = tmp_path / "mutations.tsv"
    f.write_text("pos1\twt\tmut\n12\tA\tK\n30\tG\tW\n")
    got = nsr._read_mutations_positionset(f, chain="A")
    assert list(got) == [12, 30]
    assert got.frame == "designed"


@pytest.mark.logic
def test_a_missing_mutations_file_is_none(tmp_path):
    assert nsr._read_mutations_positionset(tmp_path / "no.tsv", "A") is None


@pytest.mark.logic
def test_unparseable_mutation_rows_are_skipped(tmp_path):
    f = tmp_path / "mutations.tsv"
    f.write_text("pos1\twt\tmut\nxx\tA\tK\n30\tG\tW\n")
    assert list(nsr._read_mutations_positionset(f, "A")) == [30]


# ── walking the seed directories ─────────────────────────────────────────────

@pytest.mark.logic
def test_seed_zero_is_initial_and_the_rest_are_suffixed(tmp_path):
    """Seed 0 lives in initial/ and seed n in initial_s<n>/. Reading the wrong
    directory would pair a seed index with another seed's structure."""
    for name in ("initial", "initial_s1", "initial_s2"):
        _boltz_tree(tmp_path / name)
    preds, confs, idx = nsr._read_seed_stage_dirs(
        tmp_path, num_seeds=3, receptor_chain="A", effector_chain="B",
        model_sample=0, require_pdb=True)
    assert idx == [0, 1, 2]
    assert len(preds) == len(confs) == 3


@pytest.mark.logic
def test_a_seed_directory_that_is_absent_is_skipped(tmp_path):
    for name in ("initial", "initial_s2"):
        _boltz_tree(tmp_path / name)
    _, _, idx = nsr._read_seed_stage_dirs(
        tmp_path, num_seeds=3, receptor_chain="A", effector_chain="B",
        model_sample=0, require_pdb=True)
    assert idx == [0, 2]


@pytest.mark.logic
def test_a_seed_missing_its_confidence_is_skipped(tmp_path):
    """A prediction with no confidence beside it cannot be scored, so it must
    not enter the run as a seed that merely has blank metrics."""
    _boltz_tree(tmp_path / "initial")
    _boltz_tree(tmp_path / "initial_s1", conf=False)
    _, _, idx = nsr._read_seed_stage_dirs(
        tmp_path, num_seeds=2, receptor_chain="A", effector_chain="B",
        model_sample=0, require_pdb=True)
    assert idx == [0]
