"""Unit tests for bin/boltz_confidence.py (Phase 4 Tier 0.3)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin"))

import boltz_confidence as bc  # noqa: E402
import pipeline_thresholds as pt  # noqa: E402

# A real Boltz confidence JSON pulled from the per-module fixture.
_REAL_FIXTURE = (
    Path(__file__).parent / "fixtures" / "boltz_confidence"
    / "confidence_input_model_4.json"
)


def _write_conf_json(path, payload):
    path.write_text(json.dumps(payload))


@pytest.mark.local_unit
class TestConstruction:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            bc.BoltzConfidenceMetrics(tmp_path / "missing.json")

    def test_reads_minimal_json(self, tmp_path):
        cj = tmp_path / "confidence_input_model_0.json"
        _write_conf_json(cj, {"ptm": 0.5, "iptm": 0.3})
        m = bc.BoltzConfidenceMetrics(cj)
        assert m.ptm == 0.5
        assert m.iptm == 0.3


@pytest.mark.local_unit
class TestProperties:
    def test_missing_fields_return_none(self, tmp_path):
        cj = tmp_path / "confidence_input_model_0.json"
        _write_conf_json(cj, {})
        m = bc.BoltzConfidenceMetrics(cj)
        assert m.ptm is None
        assert m.iptm is None
        assert m.complex_plddt is None
        assert m.confidence_score is None

    def test_unparseable_field_returns_none(self, tmp_path):
        cj = tmp_path / "confidence_input_model_0.json"
        _write_conf_json(cj, {"ptm": "not-a-number"})
        m = bc.BoltzConfidenceMetrics(cj)
        assert m.ptm is None

    def test_malformed_json_treats_as_empty(self, tmp_path):
        cj = tmp_path / "bad.json"
        cj.write_text("{not valid")
        m = bc.BoltzConfidenceMetrics(cj)
        assert m.ptm is None
        assert m.iptm is None

    def test_iptm_falls_back_to_protein_iptm(self, tmp_path):
        """Matches compute_metrics.py:1241-1243 fallback behaviour."""
        cj = tmp_path / "confidence_input_model_0.json"
        _write_conf_json(cj, {"protein_iptm": 0.42})
        m = bc.BoltzConfidenceMetrics(cj)
        assert m.iptm == 0.42

    def test_iptm_prefers_iptm_over_protein_iptm(self, tmp_path):
        cj = tmp_path / "confidence_input_model_0.json"
        _write_conf_json(cj, {"iptm": 0.7, "protein_iptm": 0.42})
        m = bc.BoltzConfidenceMetrics(cj)
        assert m.iptm == 0.7

    def test_rounding_to_4dp(self, tmp_path):
        cj = tmp_path / "confidence_input_model_0.json"
        _write_conf_json(cj, {"ptm": 0.123456789})
        m = bc.BoltzConfidenceMetrics(cj)
        assert m.ptm == 0.1235

    def test_chains_ptm_dict(self, tmp_path):
        cj = tmp_path / "confidence_input_model_0.json"
        _write_conf_json(cj, {"chains_ptm": {"0": 0.9, "1": 0.8}})
        m = bc.BoltzConfidenceMetrics(cj)
        d = m.chains_ptm
        assert d == {"0": 0.9, "1": 0.8}

    def test_pair_chains_iptm_nested(self, tmp_path):
        cj = tmp_path / "confidence_input_model_0.json"
        _write_conf_json(cj, {"pair_chains_iptm": {"0": {"1": 0.5}, "1": {"0": 0.45}}})
        m = bc.BoltzConfidenceMetrics(cj)
        d = m.pair_chains_iptm
        assert d == {"0": {"1": 0.5}, "1": {"0": 0.45}}


@pytest.mark.local_unit
class TestLazyLoading:
    def test_json_loaded_only_once(self, tmp_path):
        cj = tmp_path / "confidence_input_model_0.json"
        _write_conf_json(cj, {"ptm": 0.5})
        m = bc.BoltzConfidenceMetrics(cj)
        # Access twice — should not raise even if file is removed after first read
        _ = m.ptm
        cj.unlink()
        _ = m.iptm  # cache means this works
        # And the cached value is what we expect
        assert m.ptm == 0.5


@pytest.mark.local_unit
@pytest.mark.skipif(not _REAL_FIXTURE.is_file(),
                    reason="real fixture not present in this checkout")
class TestRealFixture:
    """End-to-end against a real Boltz confidence JSON from the per-module
    fixture set, to catch any drift between the wrapper and real output."""

    def test_reads_all_top_level_floats(self):
        m = bc.BoltzConfidenceMetrics(_REAL_FIXTURE)
        # From the fixture:
        # confidence_score: 0.7693772912025452 -> 0.7694
        # ptm: 0.658906102180481 -> 0.6589
        # iptm: 0.4295026361942291 -> 0.4295
        # complex_plddt: 0.8543459177017212 -> 0.8543
        assert m.confidence_score == 0.7694
        assert m.ptm == 0.6589
        assert m.iptm == 0.4295
        assert m.complex_plddt == 0.8543

    def test_chains_and_pair_chains_populated(self):
        m = bc.BoltzConfidenceMetrics(_REAL_FIXTURE)
        cp = m.chains_ptm
        assert "0" in cp
        assert "1" in cp
        pci = m.pair_chains_iptm
        assert "0" in pci and "1" in pci["0"]


@pytest.mark.local_unit
class TestConfidenceFlag:
    """Tier 0 confidence_flag: iptm + complex_plddt only (PAE-derived
    checks deferred to Tier 2 when chain lengths are accessible)."""

    def _make(self, tmp_path, payload):
        cj = tmp_path / "confidence_input_model_0.json"
        _write_conf_json(cj, payload)
        return bc.BoltzConfidenceMetrics(cj)

    def test_ok_when_above_thresholds(self, tmp_path):
        m = self._make(tmp_path, {"iptm": 0.5, "complex_plddt": 0.8})
        t = pt.PipelineInternalThresholds.default()
        assert m.confidence_flag(t) == "ok"

    def test_low_iptm(self, tmp_path):
        m = self._make(tmp_path, {"iptm": 0.1, "complex_plddt": 0.8})
        t = pt.PipelineInternalThresholds.default()
        assert m.confidence_flag(t) == "low_iptm"

    def test_low_plddt(self, tmp_path):
        m = self._make(tmp_path, {"iptm": 0.5, "complex_plddt": 0.5})
        t = pt.PipelineInternalThresholds.default()
        assert m.confidence_flag(t) == "low_plddt"

    def test_multiple(self, tmp_path):
        m = self._make(tmp_path, {"iptm": 0.1, "complex_plddt": 0.5})
        t = pt.PipelineInternalThresholds.default()
        assert m.confidence_flag(t) == "multiple"

    def test_missing_fields_count_as_passing(self, tmp_path):
        """When a field is missing, the check for it is skipped (returns
        ok if no other trigger fires).  This matches existing behaviour
        in extract_passing.py:_compute_confidence_flag."""
        m = self._make(tmp_path, {})  # both missing
        t = pt.PipelineInternalThresholds.default()
        assert m.confidence_flag(t) == "ok"


@pytest.mark.local_unit
class TestFromPredictionDir:
    def test_returns_empty_for_dir_with_no_confidences(self, tmp_path):
        result = bc.BoltzConfidenceMetrics.from_prediction_dir(tmp_path)
        assert result == []

    def test_finds_all_samples(self, tmp_path):
        for i in range(3):
            _write_conf_json(
                tmp_path / f"confidence_input_model_{i}.json",
                {"ptm": 0.5 + 0.01 * i},
            )
        result = bc.BoltzConfidenceMetrics.from_prediction_dir(tmp_path)
        assert len(result) == 3
        # Sorted by glob order
        ptms = sorted([m.ptm for m in result])
        assert ptms == [0.5, 0.51, 0.52]

    def test_custom_base_name(self, tmp_path):
        _write_conf_json(
            tmp_path / "confidence_other_model_0.json", {"ptm": 0.9}
        )
        result = bc.BoltzConfidenceMetrics.from_prediction_dir(
            tmp_path, base_name="other"
        )
        assert len(result) == 1
        assert result[0].ptm == 0.9

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            bc.BoltzConfidenceMetrics.from_prediction_dir(tmp_path / "nope")
