"""Unit tests for bin/pipeline_thresholds.py (Phase 4 Tier 0.1)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# bin/ is not a package; insert it on sys.path so we can import.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin"))

import pipeline_thresholds as pt  # noqa: E402


@pytest.mark.local_unit
class TestDefaults:
    """The default() class method returns an instance with the canonical
    research-methodology values currently in use across the codebase."""

    def test_default_returns_instance(self):
        t = pt.PipelineInternalThresholds.default()
        assert isinstance(t, pt.PipelineInternalThresholds)

    def test_default_equivalent_to_no_args(self):
        assert pt.PipelineInternalThresholds.default() == pt.PipelineInternalThresholds()

    def test_default_structural_thresholds(self):
        t = pt.PipelineInternalThresholds.default()
        assert t.steered_ra_eff == 5.0
        assert t.reverted_ra_eff == 5.0
        assert t.cold_start_rmsd_threshold == 6.0
        assert t.intact_threshold == 5.0

    def test_default_confidence_thresholds(self):
        """Match the extract_passing.py:_compute_confidence_flag defaults."""
        t = pt.PipelineInternalThresholds.default()
        assert t.iptm_min == 0.30
        assert t.complex_plddt_min == 0.70
        assert t.ipae_max == 15.0
        assert t.pae_pass_frac_min == 0.10

    def test_default_orthogonal_gate(self):
        """Match the ORTHOG_* constants in orthogonal_metrics_plots.py."""
        t = pt.PipelineInternalThresholds.default()
        assert t.orthogonal_sc_min == 0.55
        assert t.orthogonal_bsa_min == 600.0
        assert t.orthogonal_ddg_max == -30.0
        assert t.orthogonal_af3_ra_max == 5.0

    def test_default_composite_weight(self):
        t = pt.PipelineInternalThresholds.default()
        assert t.composite_ra_eff_weight == 0.05

    def test_default_contact_cutoff(self):
        t = pt.PipelineInternalThresholds.default()
        assert t.contact_cutoff == 4.5

    def test_default_weighted_jaccard_constants(self):
        """Match _WJ_MU / _WJ_TWO_SIGMA_SQ in compute_metrics.py."""
        t = pt.PipelineInternalThresholds.default()
        assert t.weighted_jaccard_mu == 4.0
        assert t.weighted_jaccard_two_sigma_sq == 2.25

    def test_default_sc_threshold(self):
        t = pt.PipelineInternalThresholds.default()
        assert t.sc_threshold == 0.62


@pytest.mark.local_unit
class TestImmutability:
    def test_frozen(self):
        """Cannot assign to a field after construction."""
        t = pt.PipelineInternalThresholds.default()
        with pytest.raises((AttributeError, Exception)):
            t.steered_ra_eff = 99.0  # type: ignore

    def test_with_overrides_returns_new_instance(self):
        original = pt.PipelineInternalThresholds.default()
        modified = original.with_overrides({"steered_ra_eff": 4.5})
        assert modified is not original
        assert modified.steered_ra_eff == 4.5

    def test_with_overrides_leaves_original_unchanged(self):
        original = pt.PipelineInternalThresholds.default()
        original.with_overrides({"steered_ra_eff": 99.0})
        # Re-read original
        assert original.steered_ra_eff == 5.0

    def test_with_overrides_preserves_unrelated_fields(self):
        original = pt.PipelineInternalThresholds.default()
        modified = original.with_overrides({"steered_ra_eff": 4.5})
        # Every other field should be unchanged
        for field_name, value in original.as_dict().items():
            if field_name == "steered_ra_eff":
                assert modified.steered_ra_eff == 4.5
            else:
                assert getattr(modified, field_name) == value


@pytest.mark.local_unit
class TestOverridesValidation:
    def test_unknown_key_raises(self):
        t = pt.PipelineInternalThresholds.default()
        with pytest.raises(TypeError) as exc_info:
            t.with_overrides({"not_a_real_threshold": 42})
        assert "not_a_real_threshold" in str(exc_info.value)

    def test_unknown_key_message_lists_valid_keys(self):
        t = pt.PipelineInternalThresholds.default()
        with pytest.raises(TypeError) as exc_info:
            t.with_overrides({"typo_field": 1.0})
        # The error message should help debugging by listing real fields
        assert "steered_ra_eff" in str(exc_info.value)

    def test_multiple_unknown_keys_all_reported(self):
        t = pt.PipelineInternalThresholds.default()
        with pytest.raises(TypeError) as exc_info:
            t.with_overrides({"bad_a": 1, "bad_b": 2})
        msg = str(exc_info.value)
        assert "bad_a" in msg
        assert "bad_b" in msg


@pytest.mark.local_unit
class TestAsDict:
    def test_as_dict_includes_every_field(self):
        t = pt.PipelineInternalThresholds.default()
        d = t.as_dict()
        # Spot-check a few from each category
        assert "steered_ra_eff" in d
        assert "iptm_min" in d
        assert "orthogonal_sc_min" in d
        assert "composite_ra_eff_weight" in d
        assert "contact_cutoff" in d
        assert "sc_threshold" in d

    def test_as_dict_values_match_attrs(self):
        t = pt.PipelineInternalThresholds.default()
        d = t.as_dict()
        for key, val in d.items():
            assert val == getattr(t, key)


@pytest.mark.local_unit
class TestChainedOverrides:
    """Common usage pattern: layer multiple override sets on top of default."""

    def test_chained_overrides(self):
        base = pt.PipelineInternalThresholds.default()
        relaxed = base.with_overrides({"steered_ra_eff": 6.0})
        very_relaxed = relaxed.with_overrides({"ipae_max": 30.0})
        # All three live independently
        assert base.steered_ra_eff == 5.0 and base.ipae_max == 15.0
        assert relaxed.steered_ra_eff == 6.0 and relaxed.ipae_max == 15.0
        assert very_relaxed.steered_ra_eff == 6.0 and very_relaxed.ipae_max == 30.0
