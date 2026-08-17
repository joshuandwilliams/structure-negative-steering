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


@pytest.mark.local_unit
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


@pytest.mark.local_unit
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


@pytest.mark.local_unit
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


@pytest.mark.local_unit
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


@pytest.mark.local_unit
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


@pytest.mark.local_unit
class TestImmutability:
    def test_frozen(self, tmp_path, truth):
        cs = _make_cold_start([_fake_pred(2.0, 1.0)])
        run = nsr.NegativeSteeringRun(
            mpnn_sequence_id="x", workdir=tmp_path, row_type="steered",
            cold_start=cs, truth=truth)
        with pytest.raises((AttributeError, Exception)):
            run.mpnn_sequence_id = "new"  # type: ignore
