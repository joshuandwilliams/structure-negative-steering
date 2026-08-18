"""reversion.py's harvest half: locating predictions and reaching a verdict.

The verdict is the correctness gate on every reported rescue. It decides
whether a steered design that landed a good pose keeps it once the mutations
that were touching the effector are put back:

    pose_collapses      the pose depended on the contaminating mutation
    new_contamination   the reverted pose moved and a mutation landed somewhere
                        that matters
    pose_holds          the rescue is real

Getting this wrong does not raise. It reclassifies a contaminated design as a
clean rescue, which inflates the headline pass rate and is invisible in every
downstream table. Hence the detail.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE = REPO_ROOT / "tests" / "characterization" / "fixtures" / "smoke_run"

_spec = importlib.util.spec_from_file_location(
    "reversion_h", REPO_ROOT / "bin" / "reversion.py")
rev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rev)


def _reverted(**over) -> dict:
    """A passing reverted result; override one field per test."""
    r = {
        "reverted_intact": True,
        "reverted_ra_eff_vs_truth": 2.0,
        "reverted_mutated_contact_positions": [],
        "reverted_independent_receptor_rmsd": 0.8,
        "reverted_independent_effector_rmsd": 0.9,
    }
    r.update(over)
    return r


# ── the verdict ──────────────────────────────────────────────────────────────

@pytest.mark.local_unit
def test_a_clean_reverted_pose_holds():
    verdict, _why = rev.classify_reversion_verdict({}, _reverted(), 5.0)
    assert verdict == "pose_holds"


@pytest.mark.local_unit
def test_a_reverted_pose_worse_than_the_cutoff_collapses():
    """The rescue depended on the residue that was put back."""
    verdict, why = rev.classify_reversion_verdict(
        {}, _reverted(reverted_ra_eff_vs_truth=9.0), 5.0)
    assert verdict == "pose_collapses"
    assert why


@pytest.mark.local_unit
def test_a_reverted_pose_that_is_not_intact_collapses():
    """Intactness is checked again after reversion. A chain that fell apart
    can still report a flattering RMSD, so the structural filter runs first."""
    verdict, why = rev.classify_reversion_verdict(
        {}, _reverted(reverted_intact=False), 5.0)
    assert verdict == "pose_collapses"
    assert "intact" in why


@pytest.mark.local_unit
def test_intactness_can_be_waived():
    verdict, _ = rev.classify_reversion_verdict(
        {}, _reverted(reverted_intact=False), 5.0,
        structural_intact_required=False)
    assert verdict == "pose_holds"


@pytest.mark.local_unit
def test_a_failed_reverted_prediction_collapses_and_says_why():
    """A missing prediction is not a passing one. Treating an absent result as
    'no contamination found' would score every failed reversion as a rescue."""
    verdict, why = rev.classify_reversion_verdict(
        {}, {"error": "boltz died"}, 5.0)
    assert verdict == "pose_collapses"
    assert "boltz died" in why


@pytest.mark.local_unit
def test_a_mutation_contacting_inside_the_gated_region_is_new_contamination():
    verdict, _ = rev.classify_reversion_verdict(
        {}, _reverted(reverted_mutated_contact_positions=[12]), 5.0,
        contamination_gating_positions={10, 11, 12})
    assert verdict == "new_contamination"


@pytest.mark.local_unit
def test_a_mutation_contacting_outside_the_gated_region_is_not():
    """The asymmetry that makes the gate necessary. Contacts outside the
    design region and true interface were already accepted as valid steering
    on the steered pose and deliberately kept in the reverted sequence.
    Flagging them here would contradict the stage that let them through."""
    verdict, _ = rev.classify_reversion_verdict(
        {}, _reverted(reverted_mutated_contact_positions=[99]), 5.0,
        contamination_gating_positions={10, 11, 12})
    assert verdict == "pose_holds"


@pytest.mark.local_unit
def test_without_a_gate_any_mutated_contact_fires():
    """Legacy behaviour, kept for callers that pass no gating set. Pinned so
    the default cannot drift into the gated semantics unnoticed."""
    verdict, _ = rev.classify_reversion_verdict(
        {}, _reverted(reverted_mutated_contact_positions=[99]), 5.0,
        contamination_gating_positions=None)
    assert verdict == "new_contamination"


@pytest.mark.local_unit
def test_the_structural_filter_runs_before_the_contamination_check():
    """A collapsed pose is reported as collapsed even when it also has a
    mutated contact. Reporting it as new_contamination would suggest the pose
    survived and was merely contaminated."""
    verdict, _ = rev.classify_reversion_verdict(
        {}, _reverted(reverted_ra_eff_vs_truth=40.0,
                      reverted_mutated_contact_positions=[12]), 5.0,
        contamination_gating_positions={12})
    assert verdict == "pose_collapses"


# ── locating what the harvest reads ──────────────────────────────────────────

@pytest.mark.local_unit
def test_prediction_pdb_is_preferred_over_other_pdbs(tmp_path):
    d = tmp_path / "rev_design_00"
    (d / "boltz_results_input" / "predictions" / "input").mkdir(parents=True)
    (d / "prediction.pdb").write_text("")
    (d / "boltz_results_input" / "predictions" / "input"
        / "input_model_0.pdb").write_text("")
    assert rev._locate_reverted_prediction_pdb(d).name == "prediction.pdb"


@pytest.mark.local_unit
def test_the_msa_directory_is_never_mistaken_for_a_prediction(tmp_path):
    """Boltz writes PDBs under msa/ too. Picking one would compute metrics
    against an alignment artefact and report them as a pose."""
    d = tmp_path / "rev_design_00"
    (d / "msa").mkdir(parents=True)
    (d / "msa" / "something.pdb").write_text("")
    assert rev._locate_reverted_prediction_pdb(d) is None


@pytest.mark.local_unit
def test_no_prediction_returns_none_rather_than_raising(tmp_path):
    d = tmp_path / "rev_design_00"
    d.mkdir()
    assert rev._locate_reverted_prediction_pdb(d) is None


@pytest.mark.local_integration
def test_chain_lengths_are_read_off_a_real_prediction():
    """6Q76 is receptor A at 73 residues and effector B at 68. Read from the
    smoke run's own output rather than a hand-written stub."""
    pdb = SMOKE / "cycle_0" / "steered" / "design_00" / "prediction.pdb"
    rec, eff = rev._count_pdb_chain_ca_lengths(pdb, "A", "B")
    assert (rec, eff) == (73, 68)


@pytest.mark.local_integration
def test_unknown_chain_ids_fall_back_to_the_two_largest_chains():
    """Chain naming is not guaranteed across predictors. Falling back beats
    returning zero lengths, which would silently skip the metrics."""
    pdb = SMOKE / "cycle_0" / "steered" / "design_00" / "prediction.pdb"
    rec, eff = rev._count_pdb_chain_ca_lengths(pdb, "X", "Y")
    assert rec > 0 and eff > 0
    assert {rec, eff} == {73, 68}


@pytest.mark.local_unit
def test_remaining_mutations_exclude_the_reverted_ones(tmp_path):
    """This string is what the downstream tables report as the design's
    surviving steering. Including a reverted position would credit the design
    with a mutation it no longer carries."""
    d = tmp_path / "rev_design_00"
    d.mkdir()
    (d / "reversion_metadata.json").write_text(json.dumps({
        "cumulative_mutations": [
            {"pos1": 2, "wt": "C", "mut": "W"},
            {"pos1": 9, "wt": "K", "mut": "E"},
            {"pos1": 20, "wt": "Y", "mut": "F"},
        ]}))
    assert rev._build_remaining_mutations_str(d, {9}) == "2,20"


@pytest.mark.local_unit
def test_reverting_everything_leaves_an_empty_string(tmp_path):
    d = tmp_path / "rev_design_00"
    d.mkdir()
    (d / "reversion_metadata.json").write_text(json.dumps({
        "cumulative_mutations": [{"pos1": 2, "wt": "C", "mut": "W"}]}))
    assert rev._build_remaining_mutations_str(d, {2}) == ""


# ── finding cycle_0 from a reversion workdir ─────────────────────────────────

@pytest.mark.local_unit
def test_cycle0_plan_is_found_from_the_workdir_itself(tmp_path):
    wd = tmp_path / "cycle_0"
    wd.mkdir()
    (wd / "plan.json").write_text(json.dumps({"cycle": 0}))
    assert rev._resolve_cycle0_plan_path(wd) == wd / "plan.json"


@pytest.mark.local_unit
def test_a_later_cycles_plan_is_not_mistaken_for_cycle_zero(tmp_path):
    """Cycle 1's plan records different interface residues. Using it would
    gate contamination against the wrong region."""
    wd = tmp_path / "cycle_1"
    wd.mkdir()
    (wd / "plan.json").write_text(json.dumps({"cycle": 1}))
    (tmp_path / "cycle_0").mkdir()
    (tmp_path / "cycle_0" / "plan.json").write_text(json.dumps({"cycle": 0}))
    assert rev._resolve_cycle0_plan_path(wd) == tmp_path / "cycle_0" / "plan.json"


@pytest.mark.local_unit
def test_no_cycle0_plan_anywhere_returns_none(tmp_path):
    wd = tmp_path / "cycle_0"
    wd.mkdir()
    assert rev._resolve_cycle0_plan_path(wd) is None


@pytest.mark.local_integration
def test_the_real_plan_supplies_the_gating_context():
    """The gating set comes from cycle_0's plan. An empty context would
    disable the gate and reclassify accepted steering as contamination."""
    ctx = rev._load_cycle0_plan_context(SMOKE / "cycle_0" / "plan.json", 5.0)
    assert isinstance(ctx, dict)
    assert "true_idx" in ctx
