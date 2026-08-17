"""Unit tests for bin/stage_result.py (Phase 4 Tier 4.2).

The verdict logic depends on ra_eff/intact/contact behaviour of
ProteinStructurePrediction, which need real PDB files. To keep the
Tier 4 tests local-only and fast, we use a TestablePSP stand-in that
returns hand-injected metric values. The real PSP wiring is
characterised at hpc-tier when negsteer runs end-to-end.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin"))

import pipeline_thresholds as pt  # noqa: E402
import stage_result as sr  # noqa: E402
from position_set import PositionSet  # noqa: E402


def _fake_pred(ra_eff, intact_rmsd, contact_positions=()):
    """Return a MagicMock that quacks like a ProteinStructurePrediction
    for the methods StageResult uses."""
    m = MagicMock()
    m.ra_eff_vs.return_value = ra_eff
    m.independent_receptor_rmsd_vs.return_value = intact_rmsd
    cs = PositionSet(
        positions=list(contact_positions),
        chain="A", frame="prediction",
    )
    m.receptor_contact_residues.return_value = cs
    # spec=ProteinStructurePrediction is not enforced — the type check
    # for StageResult is on the LIST, not on each element.
    return m


def _fake_conf():
    return MagicMock()


def _truth():
    return MagicMock()


# ── Construction validation ────────────────────────────────────────


@pytest.mark.local_unit
class TestConstruction:
    def test_basic_cold_start(self):
        s = sr.StageResult(
            stage_type="cold_start", config_id="design_03_seq_01",
            predictions=[_fake_pred(2.0, 1.0)],
            confidences=[_fake_conf()],
            seed_indices=[0])
        assert s.stage_type == "cold_start"
        assert s.num_seeds == 1

    def test_invalid_stage_type_raises(self):
        with pytest.raises(ValueError, match="stage_type"):
            sr.StageResult(stage_type="bogus", config_id="c",
                           predictions=[_fake_pred(2.0, 1.0)],
                           confidences=[_fake_conf()], seed_indices=[0])

    def test_empty_config_id_raises(self):
        with pytest.raises(ValueError, match="config_id"):
            sr.StageResult(stage_type="cold_start", config_id="",
                           predictions=[_fake_pred(2.0, 1.0)],
                           confidences=[_fake_conf()], seed_indices=[0])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="confidences length"):
            sr.StageResult(stage_type="cold_start", config_id="c",
                           predictions=[_fake_pred(2.0, 1.0)],
                           confidences=[_fake_conf(), _fake_conf()],
                           seed_indices=[0])

    def test_empty_predictions_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            sr.StageResult(stage_type="cold_start", config_id="c",
                           predictions=[], confidences=[], seed_indices=[])

    def test_cold_start_with_mutated_positions_raises(self):
        mp = PositionSet([5], chain="A", frame="prediction")
        with pytest.raises(ValueError, match="cold_start"):
            sr.StageResult(stage_type="cold_start", config_id="c",
                           predictions=[_fake_pred(2.0, 1.0)],
                           confidences=[_fake_conf()], seed_indices=[0],
                           mutated_positions=mp)

    def test_steered_without_mutated_positions_raises(self):
        with pytest.raises(ValueError, match="requires mutated_positions"):
            sr.StageResult(stage_type="steered", config_id="c",
                           predictions=[_fake_pred(2.0, 1.0)],
                           confidences=[_fake_conf()], seed_indices=[0])

    def test_applies_to_designs_only_for_reversion(self):
        mp = PositionSet([5], chain="A", frame="prediction")
        with pytest.raises(ValueError, match="reversion-only"):
            sr.StageResult(stage_type="steered", config_id="c",
                           predictions=[_fake_pred(2.0, 1.0)],
                           confidences=[_fake_conf()], seed_indices=[0],
                           mutated_positions=mp,
                           applies_to_designs=["design_03"])


# ── Per-seed verdicts ──────────────────────────────────────────────


def _make_cold_start(predictions):
    return sr.StageResult(
        stage_type="cold_start", config_id="design_03_seq_01",
        predictions=predictions,
        confidences=[_fake_conf() for _ in predictions],
        seed_indices=list(range(len(predictions))))


def _make_steered(predictions, mutated_positions):
    return sr.StageResult(
        stage_type="steered", config_id="design_03",
        predictions=predictions,
        confidences=[_fake_conf() for _ in predictions],
        seed_indices=list(range(len(predictions))),
        mutated_positions=mutated_positions)


def _make_reversion(predictions, mutated_positions, applies_to=("design_03",)):
    return sr.StageResult(
        stage_type="reversion", config_id="rev_design_03_abc",
        predictions=predictions,
        confidences=[_fake_conf() for _ in predictions],
        seed_indices=list(range(len(predictions))),
        mutated_positions=mutated_positions,
        applies_to_designs=list(applies_to))


@pytest.mark.local_unit
class TestColdStartVerdicts:
    def test_all_clean(self):
        # ra_eff < 5, intact_rmsd < 5 → "clean"
        preds = [_fake_pred(2.0, 1.0)] * 3
        s = _make_cold_start(preds)
        t = pt.PipelineInternalThresholds.default()
        assert s.per_seed_verdicts(t, _truth()) == ["clean"] * 3

    def test_fail_structural_ra_eff(self):
        # ra_eff >= 5 → fail
        preds = [_fake_pred(10.0, 1.0)] * 3
        s = _make_cold_start(preds)
        t = pt.PipelineInternalThresholds.default()
        assert s.per_seed_verdicts(t, _truth()) == ["fail_structural"] * 3

    def test_fail_structural_intact(self):
        # intact_rmsd >= 5 → fail (receptor not intact)
        preds = [_fake_pred(2.0, 10.0)] * 3
        s = _make_cold_start(preds)
        t = pt.PipelineInternalThresholds.default()
        assert s.per_seed_verdicts(t, _truth()) == ["fail_structural"] * 3

    def test_no_data_when_metrics_none(self):
        preds = [_fake_pred(None, None)]
        s = _make_cold_start(preds)
        t = pt.PipelineInternalThresholds.default()
        assert s.per_seed_verdicts(t, _truth()) == ["no_data"]


@pytest.mark.local_unit
class TestSteeredVerdicts:
    def _setup(self):
        mp = PositionSet([5, 7, 12], chain="A", frame="prediction")
        contam = PositionSet([5, 7, 12], chain="A", frame="prediction")
        return mp, contam

    def test_clean_steered_no_contamination(self):
        mp, contam = self._setup()
        # No contacts on mutated positions → clean_steered
        preds = [_fake_pred(2.0, 1.0, contact_positions=[100, 101])] * 3
        s = _make_steered(preds, mp)
        t = pt.PipelineInternalThresholds.default()
        assert s.per_seed_verdicts(t, _truth(), contam) == ["clean_steered"] * 3

    def test_contaminated(self):
        mp, contam = self._setup()
        # contact at position 5 (in both mutated_positions and contam) → contaminated
        preds = [_fake_pred(2.0, 1.0, contact_positions=[5, 100])] * 3
        s = _make_steered(preds, mp)
        t = pt.PipelineInternalThresholds.default()
        assert s.per_seed_verdicts(t, _truth(), contam) == ["contaminated"] * 3

    def test_pose_collapses_structural_fail(self):
        mp, contam = self._setup()
        preds = [_fake_pred(10.0, 1.0)]
        s = _make_steered(preds, mp)
        t = pt.PipelineInternalThresholds.default()
        assert s.per_seed_verdicts(t, _truth(), contam) == ["pose_collapses"]


@pytest.mark.local_unit
class TestReversionVerdicts:
    def _setup(self):
        mp = PositionSet([5, 7, 12], chain="A", frame="prediction")
        contam = PositionSet([5, 7, 12], chain="A", frame="prediction")
        return mp, contam

    def test_pose_holds(self):
        mp, contam = self._setup()
        preds = [_fake_pred(2.0, 1.0, contact_positions=[100])] * 3
        s = _make_reversion(preds, mp)
        t = pt.PipelineInternalThresholds.default()
        assert s.per_seed_verdicts(t, _truth(), contam) == ["pose_holds"] * 3

    def test_new_contamination(self):
        mp, contam = self._setup()
        preds = [_fake_pred(2.0, 1.0, contact_positions=[5])] * 3
        s = _make_reversion(preds, mp)
        t = pt.PipelineInternalThresholds.default()
        assert s.per_seed_verdicts(t, _truth(), contam) == ["new_contamination"] * 3

    def test_pose_collapses(self):
        mp, contam = self._setup()
        preds = [_fake_pred(10.0, 1.0, contact_positions=[100])]
        s = _make_reversion(preds, mp)
        t = pt.PipelineInternalThresholds.default()
        assert s.per_seed_verdicts(t, _truth(), contam) == ["pose_collapses"]


# ── CL-3 gating rule (the headline test) ──────────────────────────


@pytest.mark.local_unit
class TestCL3SteeredGating:
    """CL-3 rule: reversion runs iff n_contaminated >= ceil(n_correctly_placed / 2)."""

    def _setup(self):
        mp = PositionSet([5, 7, 12], chain="A", frame="prediction")
        contam = PositionSet([5, 7, 12], chain="A", frame="prediction")
        return mp, contam

    def test_3of3_correct_1_contaminated_no_reversion(self):
        """3/3 correctly placed, 1 contaminated → 1 < ceil(3/2)=2 → NO reversion."""
        mp, contam = self._setup()
        preds = [
            _fake_pred(2.0, 1.0, contact_positions=[100]),     # clean
            _fake_pred(2.0, 1.0, contact_positions=[101]),     # clean
            _fake_pred(2.0, 1.0, contact_positions=[5]),       # contaminated
        ]
        s = _make_steered(preds, mp)
        t = pt.PipelineInternalThresholds.default()
        assert s.n_correctly_placed(t, _truth()) == 3
        assert s.n_contaminated(t, _truth(), contam) == 1
        assert s.triggers_next_stage(t, _truth(), contam) is False

    def test_3of3_correct_2_contaminated_reversion_runs(self):
        """3/3 correctly placed, 2 contaminated → 2 >= 2 → reversion."""
        mp, contam = self._setup()
        preds = [
            _fake_pred(2.0, 1.0, contact_positions=[100]),
            _fake_pred(2.0, 1.0, contact_positions=[5]),
            _fake_pred(2.0, 1.0, contact_positions=[7]),
        ]
        s = _make_steered(preds, mp)
        t = pt.PipelineInternalThresholds.default()
        assert s.triggers_next_stage(t, _truth(), contam) is True

    def test_2of3_correct_1_contaminated_reversion_runs(self):
        """2/3 correct (1 wrong placement), 1 contaminated → 1 >= ceil(2/2)=1 → reversion."""
        mp, contam = self._setup()
        preds = [
            _fake_pred(2.0, 1.0, contact_positions=[100]),  # clean
            _fake_pred(2.0, 1.0, contact_positions=[5]),    # contaminated
            _fake_pred(10.0, 1.0),                          # wrong placement
        ]
        s = _make_steered(preds, mp)
        t = pt.PipelineInternalThresholds.default()
        assert s.n_correctly_placed(t, _truth()) == 2
        assert s.n_contaminated(t, _truth(), contam) == 1
        assert s.triggers_next_stage(t, _truth(), contam) is True

    def test_2of3_correct_0_contaminated_no_reversion(self):
        """2/3 correctly placed, 0 contaminated → no reversion (tier-B holds)."""
        mp, contam = self._setup()
        preds = [
            _fake_pred(2.0, 1.0, contact_positions=[100]),
            _fake_pred(2.0, 1.0, contact_positions=[101]),
            _fake_pred(10.0, 1.0),
        ]
        s = _make_steered(preds, mp)
        t = pt.PipelineInternalThresholds.default()
        assert s.triggers_next_stage(t, _truth(), contam) is False

    def test_0_correctly_placed_no_reversion(self):
        """0 correctly placed → nothing to revert."""
        mp, contam = self._setup()
        preds = [_fake_pred(10.0, 1.0)] * 3
        s = _make_steered(preds, mp)
        t = pt.PipelineInternalThresholds.default()
        assert s.n_correctly_placed(t, _truth()) == 0
        assert s.triggers_next_stage(t, _truth(), contam) is False

    def test_3of3_all_contaminated_reversion_runs(self):
        mp, contam = self._setup()
        preds = [
            _fake_pred(2.0, 1.0, contact_positions=[5]),
            _fake_pred(2.0, 1.0, contact_positions=[7]),
            _fake_pred(2.0, 1.0, contact_positions=[12]),
        ]
        s = _make_steered(preds, mp)
        t = pt.PipelineInternalThresholds.default()
        assert s.triggers_next_stage(t, _truth(), contam) is True


# ── Other gating ─────────────────────────────────────────────────


@pytest.mark.local_unit
class TestColdStartGating:
    def test_all_clean_skips_steering(self):
        preds = [_fake_pred(2.0, 1.0)] * 3
        s = _make_cold_start(preds)
        t = pt.PipelineInternalThresholds.default()
        # 3/3 clean → fewer than majority NOT clean → False (skip steering)
        assert s.triggers_next_stage(t, _truth()) is False

    def test_majority_fail_runs_steering(self):
        preds = [_fake_pred(2.0, 1.0), _fake_pred(10.0, 1.0), _fake_pred(10.0, 1.0)]
        s = _make_cold_start(preds)
        t = pt.PipelineInternalThresholds.default()
        assert s.triggers_next_stage(t, _truth()) is True


@pytest.mark.local_unit
class TestReversionGating:
    def test_reversion_never_triggers_next_stage(self):
        mp = PositionSet([5], chain="A", frame="prediction")
        contam = PositionSet([5], chain="A", frame="prediction")
        preds = [_fake_pred(2.0, 1.0, contact_positions=[100])]
        s = _make_reversion(preds, mp)
        t = pt.PipelineInternalThresholds.default()
        assert s.triggers_next_stage(t, _truth(), contam) is False


# ── n_pass ──────────────────────────────────────────────────────


@pytest.mark.local_unit
class TestNPass:
    def test_cold_start_pass_counts_clean(self):
        preds = [_fake_pred(2.0, 1.0), _fake_pred(10.0, 1.0), _fake_pred(2.0, 1.0)]
        s = _make_cold_start(preds)
        t = pt.PipelineInternalThresholds.default()
        assert s.n_pass(t, _truth()) == 2

    def test_steered_pass_counts_clean_steered(self):
        mp = PositionSet([5], chain="A", frame="prediction")
        contam = PositionSet([5], chain="A", frame="prediction")
        preds = [
            _fake_pred(2.0, 1.0, contact_positions=[100]),  # clean_steered
            _fake_pred(2.0, 1.0, contact_positions=[5]),    # contaminated
            _fake_pred(10.0, 1.0),                          # pose_collapses
        ]
        s = _make_steered(preds, mp)
        t = pt.PipelineInternalThresholds.default()
        assert s.n_pass(t, _truth(), contam) == 1

    def test_reversion_pass_counts_pose_holds(self):
        mp = PositionSet([5], chain="A", frame="prediction")
        contam = PositionSet([5], chain="A", frame="prediction")
        preds = [
            _fake_pred(2.0, 1.0, contact_positions=[100]),  # pose_holds
            _fake_pred(2.0, 1.0, contact_positions=[5]),    # new_contamination
        ]
        s = _make_reversion(preds, mp)
        t = pt.PipelineInternalThresholds.default()
        assert s.n_pass(t, _truth(), contam) == 1
