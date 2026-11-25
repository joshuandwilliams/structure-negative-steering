"""Tests for the reversion pass's sequence surgery and verdict rules.

Reversion exists because a rescued pose that depends on a steering mutation
touching the effector is not a usable prediction. The wet-lab construct is
expressed from the original sequence, so it will not carry that residue.

Two functions decide everything. build_reverted_sequence puts the wild-type
residues back, and classify_reversion_verdict decides whether the result is
kept. Both fail silently when wrong: a reversion applied at the wrong index
produces a valid protein sequence, and a verdict that is too lenient keeps a
pose whose numbers look ordinary.

Positions are 1-based throughout, which is the off-by-one these tests exist
to pin.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def rev():
    sys.path.insert(0, str(REPO_ROOT / "bin"))
    spec = importlib.util.spec_from_file_location(
        "reversion", REPO_ROOT / "bin" / "reversion.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── putting the wild type back ───────────────────────────────────────────────

@pytest.mark.logic
def test_a_position_is_reverted_to_its_wild_type(rev):
    """Position 3 is 1-based, so it is index 2. Off by one here silently
    edits a neighbouring residue."""
    steered = "MAKVL"
    seq, applied = rev.build_reverted_sequence(
        steered, [(3, "G", "K")], [3])
    assert seq == "MAGVL"
    assert applied == [(3, "K", "G")]


@pytest.mark.logic
def test_only_the_requested_positions_move(rev):
    steered = "MAKVL"
    seq, applied = rev.build_reverted_sequence(
        steered, [(3, "G", "K"), (5, "W", "L")], [3])
    assert seq == "MAGVL"            # position 5 is untouched
    assert [p for p, _, _ in applied] == [3]


@pytest.mark.logic
def test_several_positions_revert_together(rev):
    seq, applied = rev.build_reverted_sequence(
        "MAKVL", [(3, "G", "K"), (5, "W", "L")], [5, 3])
    assert seq == "MAGVW"
    assert [p for p, _, _ in applied] == [3, 5]      # reported in order


@pytest.mark.logic
def test_a_position_edited_twice_reverts_to_the_earliest_wild_type(rev):
    """An ancestor edited the position and a descendant re-edited it. The
    true wild type is the first one recorded, not the intermediate."""
    seq, applied = rev.build_reverted_sequence(
        "MAKVL", [(3, "G", "D"), (3, "D", "K")], [3])
    assert seq == "MAGVL"
    assert applied == [(3, "K", "G")]


@pytest.mark.logic
def test_duplicate_requests_are_applied_once(rev):
    seq, applied = rev.build_reverted_sequence("MAKVL", [(3, "G", "K")],
                                               [3, 3, 3])
    assert seq == "MAGVL"
    assert len(applied) == 1


@pytest.mark.logic
def test_nothing_to_revert_returns_the_sequence_unchanged(rev):
    seq, applied = rev.build_reverted_sequence("MAKVL", [(3, "G", "K")], [])
    assert seq == "MAKVL" and applied == []


# ── the refusals ─────────────────────────────────────────────────────────────

@pytest.mark.logic
def test_an_empty_sequence_is_refused(rev):
    with pytest.raises(ValueError, match="steered_seq is empty"):
        rev.build_reverted_sequence("", [(1, "A", "K")], [1])


@pytest.mark.logic
@pytest.mark.parametrize("pos", [0, -1, 6, 100])
def test_a_position_outside_the_sequence_is_refused(rev, pos):
    with pytest.raises(ValueError, match="out of range"):
        rev.build_reverted_sequence("MAKVL", [(pos, "G", "K")], [pos])


@pytest.mark.logic
def test_a_steered_sequence_that_disagrees_with_the_history_is_refused(rev):
    """If the residue sitting at that position is not the one the history
    says was put there, the two inputs describe different designs. Reverting
    anyway would write a wild type over an unrelated residue."""
    with pytest.raises(ValueError, match="steered sequence mismatch"):
        rev.build_reverted_sequence("MAKVL", [(3, "G", "W")], [3])


@pytest.mark.logic
def test_a_position_with_no_history_is_skipped_with_a_warning(rev, capsys):
    """Skipping leaves that residue steered, which the verdict stage then
    judges. Guessing a wild type would invent one."""
    seq, applied = rev.build_reverted_sequence(
        "MAKVL", [(3, "G", "K")], [3, 4])
    assert seq == "MAGVL"
    assert [p for p, _, _ in applied] == [3]
    assert "had no entry in cumulative_mutations" in capsys.readouterr().err


# ── the verdict ──────────────────────────────────────────────────────────────

def _reverted(**kw):
    base = {"reverted_intact": True, "reverted_ra_eff_vs_truth": 2.0,
            "reverted_mutated_contact_positions": []}
    base.update(kw)
    return base


@pytest.mark.logic
def test_a_clean_reverted_pose_holds(rev):
    verdict, _ = rev.classify_reversion_verdict({}, _reverted(), 5.0)
    assert verdict == "pose_holds"


@pytest.mark.logic
def test_a_failed_reverted_prediction_collapses(rev):
    verdict, reason = rev.classify_reversion_verdict(
        {}, {"error": "boltz died"}, 5.0)
    assert verdict == "pose_collapses" and "boltz died" in reason


@pytest.mark.logic
def test_a_reverted_pose_that_is_not_intact_collapses(rev):
    verdict, reason = rev.classify_reversion_verdict(
        {}, _reverted(reverted_intact=False), 5.0)
    assert verdict == "pose_collapses" and "not intact" in reason


@pytest.mark.logic
def test_intactness_can_be_waived(rev):
    verdict, _ = rev.classify_reversion_verdict(
        {}, _reverted(reverted_intact=False), 5.0,
        structural_intact_required=False)
    assert verdict == "pose_holds"


@pytest.mark.logic
def test_a_missing_reverted_rmsd_collapses_rather_than_passing(rev):
    """No measurement is not evidence of a good pose."""
    verdict, reason = rev.classify_reversion_verdict(
        {}, _reverted(reverted_ra_eff_vs_truth=None), 5.0)
    assert verdict == "pose_collapses" and "unavailable" in reason


@pytest.mark.logic
@pytest.mark.parametrize("ra,expected", [
    (4.99, "pose_holds"), (5.0, "pose_collapses"), (5.01, "pose_collapses"),
])
def test_the_cutoff_is_exclusive_at_the_threshold(rev, ra, expected):
    verdict, _ = rev.classify_reversion_verdict(
        {}, _reverted(reverted_ra_eff_vs_truth=ra), 5.0)
    assert verdict == expected


# ── the contamination gate ───────────────────────────────────────────────────
#
# A mutated residue contacting the effector on the reverted pose only counts
# as new contamination if it sits inside the design region or the true
# interface. Positions outside that set already passed the pre-reversion
# filter on the steered pose and are kept deliberately, so flagging them here
# would contradict the earlier stage.

@pytest.mark.logic
def test_a_contact_inside_the_gate_is_new_contamination(rev):
    verdict, reason = rev.classify_reversion_verdict(
        {}, _reverted(reverted_mutated_contact_positions=[12]), 5.0,
        contamination_gating_positions={10, 11, 12})
    assert verdict == "new_contamination" and "12" in reason


@pytest.mark.logic
def test_a_contact_outside_the_gate_still_holds(rev):
    verdict, reason = rev.classify_reversion_verdict(
        {}, _reverted(reverted_mutated_contact_positions=[99]), 5.0,
        contamination_gating_positions={10, 11, 12})
    assert verdict == "pose_holds"
    assert "99" in reason and "accepted" in reason


@pytest.mark.logic
def test_both_kinds_of_contact_report_both_but_fail(rev):
    verdict, reason = rev.classify_reversion_verdict(
        {}, _reverted(reverted_mutated_contact_positions=[12, 99]), 5.0,
        contamination_gating_positions={10, 11, 12})
    assert verdict == "new_contamination"
    assert "12" in reason and "99" in reason


@pytest.mark.logic
def test_the_structural_filter_runs_before_the_contamination_check(rev):
    """A collapsed pose is reported as collapsed even when it also has a
    mutated contact. Reporting it as new_contamination instead would suggest
    the pose survived and was merely contaminated, which is a rescue that
    could be recovered by another reversion rather than one that failed."""
    verdict, _ = rev.classify_reversion_verdict(
        {}, _reverted(reverted_ra_eff_vs_truth=40.0,
                      reverted_mutated_contact_positions=[12]), 5.0,
        contamination_gating_positions={12})
    assert verdict == "pose_collapses"


@pytest.mark.logic
def test_without_a_gate_any_mutated_contact_is_contamination(rev):
    """The pre-gating behaviour, kept for legacy callers."""
    verdict, _ = rev.classify_reversion_verdict(
        {}, _reverted(reverted_mutated_contact_positions=[99]), 5.0,
        contamination_gating_positions=None)
    assert verdict == "new_contamination"


@pytest.mark.logic
def test_an_empty_gate_set_is_not_the_same_as_no_gate(rev):
    """An empty set means nothing is gated in, so every contact is outside
    it. None means no gating at all, so every contact fires. Treating the two
    the same would invert the verdict."""
    contacts = _reverted(reverted_mutated_contact_positions=[99])
    gated, _ = rev.classify_reversion_verdict({}, contacts, 5.0,
                                              contamination_gating_positions=set())
    ungated, _ = rev.classify_reversion_verdict({}, contacts, 5.0,
                                                contamination_gating_positions=None)
    assert gated == "pose_holds"
    assert ungated == "new_contamination"


# ── reading the steered sequence ─────────────────────────────────────────────
#
# This is the sequence the reversion is applied to. A reader that silently
# accepted two records would splice two designs together and the surgery
# would then land on the wrong residues.

@pytest.mark.logic
def test_a_single_record_fasta_is_joined_into_one_sequence(rev, tmp_path):
    f = tmp_path / "receptor.fasta"
    f.write_text(">receptor\nMAKV\nLGGW\n")
    assert rev._read_receptor_fasta(f) == "MAKVLGGW"


@pytest.mark.logic
def test_blank_lines_are_ignored(rev, tmp_path):
    f = tmp_path / "receptor.fasta"
    f.write_text(">receptor\n\nMAKV\n\nLGGW\n\n")
    assert rev._read_receptor_fasta(f) == "MAKVLGGW"


@pytest.mark.logic
def test_two_records_are_refused(rev, tmp_path):
    f = tmp_path / "receptor.fasta"
    f.write_text(">a\nMAKV\n>b\nLGGW\n")
    with pytest.raises(ValueError, match="multiple FASTA records"):
        rev._read_receptor_fasta(f)


@pytest.mark.logic
def test_a_file_with_no_header_is_refused(rev, tmp_path):
    f = tmp_path / "receptor.fasta"
    f.write_text("MAKVLGGW\n")
    with pytest.raises(ValueError, match="no FASTA header"):
        rev._read_receptor_fasta(f)


@pytest.mark.logic
def test_a_header_with_no_sequence_is_refused(rev, tmp_path):
    f = tmp_path / "receptor.fasta"
    f.write_text(">receptor\n\n")
    with pytest.raises(ValueError, match="no sequence"):
        rev._read_receptor_fasta(f)


# ── reading CA residues off one chain ────────────────────────────────────────

def _pdb(path, rows):
    """rows: (chain, resnum, resname, atom)"""
    lines = []
    for i, (chain, rn, name, atom) in enumerate(rows, 1):
        lines.append(
            f"ATOM  {i:5d}  {atom:<3s} {name} {chain}{rn:4d}"
            f"       0.000   0.000   0.000  1.00  0.00")
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.mark.logic
def test_only_ca_atoms_on_the_named_chain_are_yielded(rev, tmp_path):
    p = _pdb(tmp_path / "x.pdb", [
        ("A", 1, "ALA", "CA"), ("A", 1, "ALA", "CB"),
        ("B", 1, "GLY", "CA"), ("A", 2, "TRP", "CA")])
    assert list(rev._read_ca_chain(p, "A")) == ["ALA", "TRP"]
    assert list(rev._read_ca_chain(p, "B")) == ["GLY"]


@pytest.mark.logic
def test_a_chain_that_is_absent_yields_nothing(rev, tmp_path):
    p = _pdb(tmp_path / "x.pdb", [("A", 1, "ALA", "CA")])
    assert list(rev._read_ca_chain(p, "Z")) == []


# ── locating cycle 0 from a reversion directory ──────────────────────────────

@pytest.mark.logic
def test_a_cycle_zero_plan_in_the_workdir_is_used(rev, tmp_path):
    (tmp_path / "plan.json").write_text('{"cycle": 0}')
    assert rev._resolve_cycle0_plan_path(tmp_path) == tmp_path / "plan.json"


@pytest.mark.logic
def test_a_plan_for_a_later_cycle_is_not_mistaken_for_cycle_zero(rev,
                                                                 tmp_path):
    """Reading a cycle-1 plan as cycle 0 would take the true interface from
    the wrong cycle."""
    (tmp_path / "plan.json").write_text('{"cycle": 1}')
    nested = tmp_path / "cycle_0"
    nested.mkdir()
    (nested / "plan.json").write_text('{"cycle": 0}')
    assert rev._resolve_cycle0_plan_path(tmp_path) is None


@pytest.mark.logic
def test_cycle_zero_is_found_one_or_two_levels_up(rev, tmp_path):
    root = tmp_path / "run"
    (root / "cycle_0").mkdir(parents=True)
    (root / "cycle_0" / "plan.json").write_text("{}")
    deep = root / "cycle_0" / "reversions" / "rev_design_00_s0_s0"
    deep.mkdir(parents=True)
    assert rev._resolve_cycle0_plan_path(deep.parent) == \
        root / "cycle_0" / "plan.json"


@pytest.mark.logic
def test_an_unparseable_workdir_plan_falls_through_to_the_search(rev,
                                                                 tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "plan.json").write_text("{not json")
    (root / "cycle_0").mkdir()
    (root / "cycle_0" / "plan.json").write_text("{}")
    sub = root / "sub"
    sub.mkdir()
    assert rev._resolve_cycle0_plan_path(sub) == root / "cycle_0" / "plan.json"


@pytest.mark.logic
def test_no_plan_anywhere_is_none(rev, tmp_path):
    assert rev._resolve_cycle0_plan_path(tmp_path) is None


# ── the plan context ─────────────────────────────────────────────────────────

@pytest.mark.logic
def test_the_context_defaults_when_there_is_no_plan(rev):
    ctx = rev._load_cycle0_plan_context(None, 5.0)
    assert ctx["true_idx"] == [] and ctx["pred_rec"] == "A"
    assert ctx["pred_eff"] == "B" and ctx["contact_cutoff"] == 5.0


@pytest.mark.logic
def test_the_context_is_read_out_of_the_plan(rev, tmp_path):
    import json
    p = tmp_path / "plan.json"
    p.write_text(json.dumps({
        "true_interface_idx": [1, 2], "initial_wrong_interface_idx": [40],
        "wild_type_receptor_seq": "MAKV", "pred_receptor_chain": "C",
        "pred_effector_chain": "D", "contact_cutoff": 8.0}))
    ctx = rev._load_cycle0_plan_context(p, 5.0)
    assert ctx["true_idx"] == [1, 2] and ctx["wrong_idx"] == [40]
    assert ctx["rec_seq"] == "MAKV"
    assert ctx["pred_rec"] == "C" and ctx["pred_eff"] == "D"
    assert ctx["contact_cutoff"] == 8.0


@pytest.mark.logic
def test_an_unparseable_cutoff_keeps_the_default(rev, tmp_path):
    import json
    p = tmp_path / "plan.json"
    p.write_text(json.dumps({"contact_cutoff": "wide"}))
    assert rev._load_cycle0_plan_context(p, 5.0)["contact_cutoff"] == 5.0


@pytest.mark.logic
def test_an_unreadable_plan_gives_the_defaults(rev, tmp_path):
    p = tmp_path / "plan.json"
    p.write_text("{not json")
    assert rev._load_cycle0_plan_context(p, 5.0)["pred_rec"] == "A"


# ── finding the reverted prediction ──────────────────────────────────────────

@pytest.mark.logic
def test_the_flat_prediction_is_preferred(rev, tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "input_model_0.pdb").write_text("ATOM\n")
    (tmp_path / "prediction.pdb").write_text("ATOM\n")
    assert rev._locate_reverted_prediction_pdb(tmp_path).name == "prediction.pdb"


@pytest.mark.logic
def test_model_zero_is_preferred_when_there_is_no_flat_copy(rev, tmp_path):
    (tmp_path / "input_model_3.pdb").write_text("ATOM\n")
    (tmp_path / "input_model_0.pdb").write_text("ATOM\n")
    assert rev._locate_reverted_prediction_pdb(tmp_path).name == \
        "input_model_0.pdb"


@pytest.mark.logic
def test_cached_references_are_not_mistaken_for_predictions(rev, tmp_path):
    """An msa/ directory and a reference.pdb are inputs, not output."""
    (tmp_path / "msa").mkdir()
    (tmp_path / "msa" / "x.pdb").write_text("ATOM\n")
    (tmp_path / "reference.pdb").write_text("ATOM\n")
    assert rev._locate_reverted_prediction_pdb(tmp_path) is None


@pytest.mark.logic
def test_no_pdb_at_all_is_none(rev, tmp_path):
    assert rev._locate_reverted_prediction_pdb(tmp_path) is None


# ── chain lengths off the reverted PDB ───────────────────────────────────────

@pytest.mark.logic
def test_the_named_chains_are_counted(rev, tmp_path):
    p = _pdb(tmp_path / "x.pdb",
             [("A", i, "ALA", "CA") for i in range(1, 4)]
             + [("B", i, "GLY", "CA") for i in range(1, 3)])
    assert rev._count_pdb_chain_ca_lengths(p, "A", "B") == (3, 2)


@pytest.mark.logic
def test_unnamed_chains_fall_back_to_the_two_largest(rev, tmp_path):
    """A prediction whose chains were renamed still has to yield two
    lengths, or the per-seed jaccard guard cannot run at all."""
    p = _pdb(tmp_path / "x.pdb",
             [("X", i, "ALA", "CA") for i in range(1, 5)]
             + [("Y", i, "GLY", "CA") for i in range(1, 3)])
    rec, eff = rev._count_pdb_chain_ca_lengths(p, "A", "B")
    assert {rec, eff} == {4, 2}
