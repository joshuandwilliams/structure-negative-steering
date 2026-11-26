"""Reversion bookkeeping and passing-row extraction.

reversion decides whether a rescue survived having its contaminating mutations
put back, and extract_passing turns aggregated rows into the passing_summary
the tiering reads.

Both are string and dictionary work. 6G10 supplies a real PDB where a
structure is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin"))

import extract_passing as ep  # noqa: E402
import reversion as rev  # noqa: E402

SIXG10 = (_REPO_ROOT / "data" / "reference_complexes" / "6G10.pdb")
needs_pdb = pytest.mark.skipif(not SIXG10.is_file(), reason="6G10 reference absent")


# ── reversion: the verdict ───────────────────────────────────────────────────

@pytest.mark.logic
def test_a_reverted_design_that_keeps_its_pose_is_pose_holds():
    """The rescue survived putting the contaminating mutations back."""
    label, reason = rev.classify_reversion_verdict(
        steered={"steered_ra_eff_vs_truth": 2.0},
        reverted={"reverted_ra_eff_vs_truth": 2.2, "reverted_intact": 1},
        structural_ra_eff_cutoff=5.0,
        structural_intact_required=False,
        contamination_gating_positions=None,
    )
    assert label == "pose_holds", f"got {label!r} ({reason})"


@pytest.mark.logic
def test_a_reverted_design_that_loses_its_pose_is_not_pose_holds():
    label, reason = rev.classify_reversion_verdict(
        steered={"steered_ra_eff_vs_truth": 2.0},
        reverted={"reverted_ra_eff_vs_truth": 35.0, "reverted_intact": 1},
        structural_ra_eff_cutoff=5.0,
        structural_intact_required=False,
        contamination_gating_positions=None,
    )
    assert label == "pose_collapses", f"got {label!r}"
    assert "35" in reason, f"the reason does not quote the RMSD: {reason!r}"


@pytest.mark.logic
def test_a_missing_reverted_metric_collapses_rather_than_passing():
    """An absent measurement must never be read as a surviving pose."""
    label, reason = rev.classify_reversion_verdict(
        steered={"steered_ra_eff_vs_truth": 2.0},
        reverted={},
        structural_ra_eff_cutoff=5.0,
        structural_intact_required=False,
        contamination_gating_positions=None,
    )
    assert label == "pose_collapses"
    assert "unavailable" in reason


# ── reversion: file helpers ──────────────────────────────────────────────────

@pytest.mark.logic
def test_receptor_fasta_is_read_as_a_bare_sequence(tmp_path):
    f = tmp_path / "receptor.fasta"
    f.write_text(">receptor_reverted\nACDEFGHIK\n")
    assert rev._read_receptor_fasta(f) == "ACDEFGHIK"


@pytest.mark.logic
def test_a_multiline_fasta_is_joined(tmp_path):
    f = tmp_path / "r.fasta"
    f.write_text(">x\nACDEF\nGHIK\n")
    assert rev._read_receptor_fasta(f) == "ACDEFGHIK"


@pytest.mark.logic
@needs_pdb
def test_ca_chain_yields_three_letter_codes():
    """Yields one code per CA ATOM record, so altlocs are not collapsed.

    6G10 chain A has 76 residues but 81 CA records, so this count is atoms
    rather than residues. Anything pairing it with a sequence must dedupe.
    """
    got = list(rev._read_ca_chain(str(SIXG10), "A"))
    assert len(got) == 81, f"read {len(got)} CA records, expected 81"
    assert all(isinstance(x, str) and len(x) == 3 for x in got)


@pytest.mark.logic
def test_locating_a_reverted_prediction_in_an_empty_dir_returns_nothing(tmp_path):
    assert rev._locate_reverted_prediction_pdb(tmp_path) in (None, "")


@pytest.mark.logic
def test_cycle0_plan_is_not_found_outside_a_run_tree(tmp_path):
    assert rev._resolve_cycle0_plan_path(tmp_path) is None


@pytest.mark.logic
def test_remaining_mutations_needs_its_metadata_file(tmp_path):
    """Raises rather than degrading when reversion_metadata.json is absent."""
    with pytest.raises(FileNotFoundError):
        rev._build_remaining_mutations_str(tmp_path, [])


# ── extract_passing ──────────────────────────────────────────────────────────

@pytest.mark.logic
def test_get_prefers_the_primary_column_then_the_fallback():
    assert ep._get({"a": "1", "b": "2"}, "a", "b") == "1"
    assert ep._get({"a": "", "b": "2"}, "a", "b") == "2"
    assert ep._get({}, "a", "b") == ""


@pytest.mark.logic
def test_confidence_flag_is_ok_when_all_medians_pass():
    out = {"ipsae_min_median": 0.5, "pae_pass_frac_median": 0.9,
           "iptm_median": 0.8, "complex_plddt_median": 0.9, "ipae_median": 5.0}
    assert ep._compute_confidence_flag(out, 0.5, 0.4, 0.7, 10.0) == "ok"


@pytest.mark.logic
def test_confidence_flag_names_a_failing_median():
    out = {"ipsae_min_median": 0.5, "pae_pass_frac_median": 0.1,
           "iptm_median": 0.8, "complex_plddt_median": 0.9, "ipae_median": 5.0}
    got = ep._compute_confidence_flag(out, 0.5, 0.4, 0.7, 10.0)
    assert got not in ("", "ok"), f"a failing median was flagged {got!r}"


@pytest.mark.logic
def test_confidence_flag_reports_ok_when_there_are_no_metrics():
    """A row with no metrics is flagged ok rather than blank.

    Worth knowing when reading passing_summary.csv: an ok flag does not by
    itself prove the thresholds were evaluated. boltz2_iterate_steering's
    equivalent returns an empty string in the same situation.
    """
    assert ep._compute_confidence_flag({}, 0.5, 0.4, 0.7, 10.0) == "ok"


@pytest.mark.logic
def test_mutation_summary_returns_majority_all_and_agreement():
    """Three values: the majority set, the union, and whether seeds agreed."""
    rows = [
        {"steered_mutations_chimerax": "/A:5,7"},
        {"steered_mutations_chimerax": "/A:5,7"},
        {"steered_mutations_chimerax": "/A:5"},
    ]
    majority, all_seeds, identical = ep._summarize_mutations_across_seeds(
        rows, "steered_mutations_chimerax", 0.5)
    assert "5" in majority, "position 5 was in every seed and must be in the majority"
    assert "7" in all_seeds, "position 7 appeared and must be in the union"
    assert identical == "0", "seeds differed, so the agreement flag should be 0"


@pytest.mark.logic
def test_identical_seeds_are_reported_as_agreeing():
    rows = [{"steered_mutations_chimerax": "/A:5,7"}] * 3
    majority, all_seeds, identical = ep._summarize_mutations_across_seeds(
        rows, "steered_mutations_chimerax", 0.5)
    assert majority == all_seeds, \
        "when every seed agrees the majority and the union must match"
    assert identical == "1", "identical seeds should set the agreement flag to 1"


@pytest.mark.logic
def test_amino_acid_mutations_are_restricted_to_the_named_positions():
    rows = [{"steered_mutations_aa": "K5E V7D M22R"}] * 2
    got = ep._muts_aa_for_positions(rows, "steered_mutations_aa", [5, 22])
    assert "K5E" in got and "M22R" in got
    assert "V7D" not in got, "an unrequested position leaked into the string"


@pytest.mark.logic
def test_amino_acid_mutations_for_no_positions_is_empty():
    rows = [{"steered_mutations_aa": "K5E"}]
    assert ep._muts_aa_for_positions(rows, "steered_mutations_aa", []) in ("", None)
