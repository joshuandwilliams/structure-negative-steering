"""Contig notation, reversion bookkeeping and passing-row extraction.

contig_utils turns the user-facing contig string into segments and back, and
derives the design region from it. Getting that wrong shifts the protected
residues, which nothing downstream would catch.

reversion decides whether a rescue survived having its contaminating mutations
put back, and extract_passing turns aggregated rows into the passing_summary
the tiering reads.

All three are string and dictionary work. 6G10 supplies a real PDB where a
structure is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin"))

import contig_utils as cu  # noqa: E402
import extract_passing as ep  # noqa: E402
import reversion as rev  # noqa: E402

SIXG10 = (_REPO_ROOT / "experiments" / "benchmarking" / "unconstrained" / "6G10"
          / "6G10.pdb")
needs_pdb = pytest.mark.skipif(not SIXG10.is_file(), reason="6G10 reference absent")


# ── contig_utils: reading a real chain ───────────────────────────────────────

@pytest.mark.local_integration
@needs_pdb
def test_chain_residue_range_matches_the_file():
    """6G10 chain A is author-numbered 187..262."""
    lo, hi = cu.get_chain_residue_range(str(SIXG10), "A")
    assert (lo, hi) == (187, 262), f"got {lo}..{hi}"


@pytest.mark.local_integration
@needs_pdb
def test_absent_chain_gives_no_range_rather_than_raising():
    assert cu.get_chain_residue_range(str(SIXG10), "Z") == (None, None)


@pytest.mark.local_integration
@needs_pdb
def test_sorted_residues_are_unique_and_ordered():
    got = cu.get_chain_residues_sorted(str(SIXG10), "A")
    assert got == sorted(set(got)), "residues came back unsorted or duplicated"
    assert len(got) == 76, f"chain A has {len(got)} residues, expected 76"


@pytest.mark.local_integration
@needs_pdb
def test_effector_chain_is_read_independently():
    got = cu.get_chain_residues_sorted(str(SIXG10), "B")
    assert len(got) == 83, f"chain B has {len(got)} residues, expected 83"


# ── contig_utils: the notation itself ────────────────────────────────────────

@pytest.mark.local_unit
def test_expected_chain_lengths_from_a_resolved_contig():
    rec, eff = cu.get_expected_chain_lengths("A1-10/5/A16-20 B1-30", "A", "B")
    assert rec == 10 + 5 + 5, f"receptor length {rec}, expected 20"
    assert eff == 30, f"effector length {eff}, expected 30"


@pytest.mark.local_unit
def test_design_region_is_the_de_novo_stretch():
    """The inserted segment is designed, the A-prefixed ones are fixed."""
    design, fixed = cu.parse_design_region("A1-10/5/A16-20", "A")
    assert len(design) == 5, f"expected 5 designed positions, got {sorted(design)}"
    assert len(fixed) == 15, f"expected 15 fixed positions, got {len(fixed)}"
    assert not (set(design) & set(fixed)), "a position is both designed and fixed"


@pytest.mark.local_unit
def test_a_wholly_fixed_contig_has_no_design_region():
    design, fixed = cu.parse_design_region("A1-20", "A")
    assert not design
    assert len(fixed) == 20


@pytest.mark.local_unit
def test_parsing_a_block_yields_descriptors_and_the_chain():
    """Returns (segments, chain). Fixed segments carry their bounds."""
    segs, chain = cu.parse_block_segments("A1-10/5/A16-20", "A", None)
    assert chain == "A"
    assert len(segs) == 3, f"expected 3 segments, got {segs}"
    assert segs[0] == ("fixed", "A", 1, 10)
    assert segs[1][0] == "denovo"
    assert segs[2] == ("fixed", "A", 16, 20)


@pytest.mark.local_unit
def test_segments_round_trip_through_the_string_form():
    segs, _ = cu.parse_block_segments("A1-10/5/A16-20", "A", None)
    assert cu.segments_to_string(segs)


# ── reversion: the verdict ───────────────────────────────────────────────────

@pytest.mark.local_unit
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


@pytest.mark.local_unit
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


@pytest.mark.local_unit
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

@pytest.mark.local_unit
def test_receptor_fasta_is_read_as_a_bare_sequence(tmp_path):
    f = tmp_path / "receptor.fasta"
    f.write_text(">receptor_reverted\nACDEFGHIK\n")
    assert rev._read_receptor_fasta(f) == "ACDEFGHIK"


@pytest.mark.local_unit
def test_a_multiline_fasta_is_joined(tmp_path):
    f = tmp_path / "r.fasta"
    f.write_text(">x\nACDEF\nGHIK\n")
    assert rev._read_receptor_fasta(f) == "ACDEFGHIK"


@pytest.mark.local_integration
@needs_pdb
def test_ca_chain_yields_three_letter_codes():
    """Yields one code per CA ATOM record, so altlocs are not collapsed.

    6G10 chain A has 76 residues but 81 CA records, so this count is atoms
    rather than residues. Anything pairing it with a sequence must dedupe.
    """
    got = list(rev._read_ca_chain(str(SIXG10), "A"))
    assert len(got) == 81, f"read {len(got)} CA records, expected 81"
    assert all(isinstance(x, str) and len(x) == 3 for x in got)


@pytest.mark.local_unit
def test_locating_a_reverted_prediction_in_an_empty_dir_returns_nothing(tmp_path):
    assert rev._locate_reverted_prediction_pdb(tmp_path) in (None, "")


@pytest.mark.local_unit
def test_cycle0_plan_is_not_found_outside_a_run_tree(tmp_path):
    assert rev._resolve_cycle0_plan_path(tmp_path) is None


@pytest.mark.local_unit
def test_remaining_mutations_needs_its_metadata_file(tmp_path):
    """Raises rather than degrading when reversion_metadata.json is absent."""
    with pytest.raises(FileNotFoundError):
        rev._build_remaining_mutations_str(tmp_path, [])


# ── extract_passing ──────────────────────────────────────────────────────────

@pytest.mark.local_unit
def test_get_prefers_the_primary_column_then_the_fallback():
    assert ep._get({"a": "1", "b": "2"}, "a", "b") == "1"
    assert ep._get({"a": "", "b": "2"}, "a", "b") == "2"
    assert ep._get({}, "a", "b") == ""


@pytest.mark.local_unit
def test_confidence_flag_is_ok_when_all_medians_pass():
    out = {"ipsae_min_median": 0.5, "pae_pass_frac_median": 0.9,
           "iptm_median": 0.8, "complex_plddt_median": 0.9, "ipae_median": 5.0}
    assert ep._compute_confidence_flag(out, 0.5, 0.4, 0.7, 10.0) == "ok"


@pytest.mark.local_unit
def test_confidence_flag_names_a_failing_median():
    out = {"ipsae_min_median": 0.5, "pae_pass_frac_median": 0.1,
           "iptm_median": 0.8, "complex_plddt_median": 0.9, "ipae_median": 5.0}
    got = ep._compute_confidence_flag(out, 0.5, 0.4, 0.7, 10.0)
    assert got not in ("", "ok"), f"a failing median was flagged {got!r}"


@pytest.mark.local_unit
def test_confidence_flag_reports_ok_when_there_are_no_metrics():
    """A row with no metrics is flagged ok rather than blank.

    Worth knowing when reading passing_summary.csv: an ok flag does not by
    itself prove the thresholds were evaluated. boltz2_iterate_steering's
    equivalent returns an empty string in the same situation.
    """
    assert ep._compute_confidence_flag({}, 0.5, 0.4, 0.7, 10.0) == "ok"


@pytest.mark.local_unit
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


@pytest.mark.local_unit
def test_identical_seeds_are_reported_as_agreeing():
    rows = [{"steered_mutations_chimerax": "/A:5,7"}] * 3
    majority, all_seeds, identical = ep._summarize_mutations_across_seeds(
        rows, "steered_mutations_chimerax", 0.5)
    assert majority == all_seeds, \
        "when every seed agrees the majority and the union must match"
    assert identical == "1", "identical seeds should set the agreement flag to 1"


@pytest.mark.local_unit
def test_amino_acid_mutations_are_restricted_to_the_named_positions():
    rows = [{"steered_mutations_aa": "K5E V7D M22R"}] * 2
    got = ep._muts_aa_for_positions(rows, "steered_mutations_aa", [5, 22])
    assert "K5E" in got and "M22R" in got
    assert "V7D" not in got, "an unrequested position leaked into the string"


@pytest.mark.local_unit
def test_amino_acid_mutations_for_no_positions_is_empty():
    rows = [{"steered_mutations_aa": "K5E"}]
    assert ep._muts_aa_for_positions(rows, "steered_mutations_aa", []) in ("", None)
