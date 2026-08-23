"""reversion.py's staging half: write_reversion_plan and its helpers.

Reversion is the correctness guard on the whole method. When a steered design
lands a good pose, but does so using a mutated residue that is itself touching
the effector, the pose is an artefact: the wet-lab construct carries the
original sequence and will not have that residue. Reversion re-predicts with
those mutations put back, and the verdict decides whether the rescue survives.

Getting this wrong does not raise. It silently scores contaminated designs as
clean rescues, which inflates every headline number. That is why it is tested
in detail despite needing no GPU.

The single-run test on the cluster produced zero contaminated designs, so it cannot
exercise any of this. These build the inputs directly instead.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "reversion", REPO_ROOT / "bin" / "reversion.py")
rev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rev)

# A 20-residue receptor is enough: the mutated positions are what matter.
WILD_TYPE = "ACDEFGHIKLMNPQRSTVWY"


def _design(tmp_path: Path, label: str, steered_seq: str) -> Path:
    """A steered design directory as the collect stage leaves it."""
    d = tmp_path / "cycle_0" / "steered" / label
    d.mkdir(parents=True, exist_ok=True)
    (d / "receptor.fasta").write_text(f">receptor\n{steered_seq}\n")
    return d


def _contaminated(label: str, design_dir: Path, muts, revert, **extra) -> dict:
    """One entry of the contaminated list, in the shape the planner expects."""
    entry = {
        "label": label,
        "design_idx": 0,
        "design_workdir": str(design_dir),
        "cumulative_mutations": muts,      # [(pos1, wt, mut), ...]
        "positions_to_revert": revert,     # [pos1, ...]
        "steered_ra_eff_vs_truth": 3.2,
        "steered_contact_residues": [5, 9],
    }
    entry.update(extra)
    return entry


def _meta(**over) -> dict:
    m = {
        "effector_seq": "MKTAYIAKQR",
        "effector_template_cif": None,
        "pred_receptor_chain": "A",
        "pred_effector_chain": "B",
        "base_seed": 0,
        "num_seeds": 1,
        "boltz_constraints_block": None,
    }
    m.update(over)
    return m


def _plan(workdir: Path) -> dict:
    return json.loads((workdir / "reversion_plan.json").read_text())


# ── building the reverted sequence ───────────────────────────────────────────

@pytest.mark.local_unit
def test_reverting_puts_the_wild_type_residue_back():
    steered = "AWDEFGHIKLMNPQRSTVWY"          # C2W
    seq, applied = rev.build_reverted_sequence(
        steered, [(2, "C", "W")], [2])
    assert seq == WILD_TYPE
    assert applied == [(2, "W", "C")]


@pytest.mark.local_unit
def test_positions_are_one_based():
    """Off by one here reverts the wrong residue and still returns a sequence
    of the right length, so nothing downstream would notice."""
    steered = "WCDEFGHIKLMNPQRSTVWY"          # A1W
    seq, _ = rev.build_reverted_sequence(steered, [(1, "A", "W")], [1])
    assert seq == WILD_TYPE


@pytest.mark.local_unit
def test_only_the_listed_positions_revert():
    """A design usually carries several steering mutations and only some are
    contaminating. Reverting all of them would undo the steering entirely."""
    steered = "AWDEFGHIKLMNPQRSTVWW"          # C2W and Y20W
    seq, applied = rev.build_reverted_sequence(
        steered, [(2, "C", "W"), (20, "Y", "W")], [2])
    assert seq == "AC" + steered[2:]
    assert [p for p, _, _ in applied] == [2]


@pytest.mark.local_unit
def test_the_earliest_wild_type_wins_when_a_position_was_edited_twice():
    """Across cycles a position can be mutated more than once. The true wild
    type is the first one recorded; using the later one would revert to
    another cycle's mutation rather than to the original."""
    steered = "AKDEFGHIKLMNPQRSTVWY"
    seq, _ = rev.build_reverted_sequence(
        steered, [(2, "C", "W"), (2, "W", "K")], [2])
    assert seq[1] == "C"


@pytest.mark.local_unit
@pytest.mark.parametrize("steered,muts,revert", [
    ("",                    [(2, "C", "W")], [2]),      # empty sequence
    ("AWDEFGHIKLMNPQRSTVWY", [(2, "C", "W")], [99]),    # out of range
    ("AQDEFGHIKLMNPQRSTVWY", [(2, "C", "W")], [2]),     # residue disagrees
])
def test_inconsistent_input_raises_rather_than_guessing(steered, muts, revert):
    """Each of these means the caller's idea of the sequence and the engine's
    have diverged. Continuing would produce a plausible wrong sequence."""
    with pytest.raises(ValueError):
        rev.build_reverted_sequence(steered, muts, revert)


# ── staging the plan ─────────────────────────────────────────────────────────

@pytest.mark.local_integration
def test_one_contaminated_design_stages_one_prediction(tmp_path):
    wd = tmp_path / "cycle_0"
    wd.mkdir(parents=True)
    d = _design(tmp_path, "design_00", "AWDEFGHIKLMNPQRSTVWY")

    rev.write_reversion_plan(
        wd, [_contaminated("design_00", d, [(2, "C", "W")], [2])], _meta())

    plan = _plan(wd)
    assert plan["n_contaminated"] == 1
    assert plan["n_staged"] == 1
    entry = plan["entries"][0]
    assert entry["label"] == "design_00"

    rev_dir = wd / "reversions" / "rev_design_00"
    assert (rev_dir / "receptor.fasta").read_text().splitlines()[1] == WILD_TYPE
    assert (rev_dir / "reversion_metadata.json").is_file()
    assert list(rev_dir.glob("*.yaml")), "no Boltz YAML staged"


@pytest.mark.local_integration
def test_designs_reverting_to_the_same_sequence_are_predicted_once(tmp_path):
    """The dedup that makes reversion affordable. Two contaminated designs
    differing only outside the reverted positions collapse to one prediction,
    and both labels must still be recorded or one loses its verdict."""
    wd = tmp_path / "cycle_0"
    wd.mkdir(parents=True)
    a = _design(tmp_path, "design_00", "AWDEFGHIKLMNPQRSTVWY")
    b = _design(tmp_path, "design_01", "AWDEFGHIKLMNPQRSTVWY")

    rev.write_reversion_plan(wd, [
        _contaminated("design_00", a, [(2, "C", "W")], [2]),
        _contaminated("design_01", b, [(2, "C", "W")], [2]),
    ], _meta())

    plan = _plan(wd)
    assert plan["n_contaminated"] == 2
    assert plan["n_staged"] == 1, "the duplicate was predicted twice"
    assert set(plan["entries"][0]["all_labels"]) == {"design_00", "design_01"}


@pytest.mark.local_integration
def test_designs_reverting_to_different_sequences_are_not_merged(tmp_path):
    wd = tmp_path / "cycle_0"
    wd.mkdir(parents=True)
    a = _design(tmp_path, "design_00", "AWDEFGHIKLMNPQRSTVWY")   # C2W
    b = _design(tmp_path, "design_01", "AWDEFGHIKLMNPQRSTVWW")   # C2W + Y20W

    rev.write_reversion_plan(wd, [
        _contaminated("design_00", a, [(2, "C", "W")], [2]),
        _contaminated("design_01", b, [(2, "C", "W"), (20, "Y", "W")], [2]),
    ], _meta())

    assert _plan(wd)["n_staged"] == 2


@pytest.mark.local_integration
def test_multi_seed_stages_one_prediction_per_seed(tmp_path):
    """Reverted predictions are sampled as many times as steered ones, or the
    comparison is one seed against three."""
    wd = tmp_path / "cycle_0"
    wd.mkdir(parents=True)
    d = _design(tmp_path, "design_00", "AWDEFGHIKLMNPQRSTVWY")

    rev.write_reversion_plan(
        wd, [_contaminated("design_00", d, [(2, "C", "W")], [2])],
        _meta(num_seeds=3))

    assert _plan(wd)["n_staged"] == 3
    for s in range(3):
        assert (wd / "reversions" / f"rev_design_00_s{s}").is_dir()


@pytest.mark.local_integration
def test_reverted_seeds_cannot_collide_with_steered_seeds(tmp_path):
    """Boltz seeds are offset into the 100000+ range. A collision would make a
    reverted prediction reuse a steered one's sampling, and the comparison
    between them would be circular."""
    wd = tmp_path / "cycle_0"
    wd.mkdir(parents=True)
    d = _design(tmp_path, "design_00", "AWDEFGHIKLMNPQRSTVWY")

    rev.write_reversion_plan(
        wd, [_contaminated("design_00", d, [(2, "C", "W")], [2])],
        _meta(base_seed=7, num_seeds=2))

    seeds = [e.get("boltz_seed") for e in _plan(wd)["entries"]]
    assert all(s >= 100000 for s in seeds), seeds
    assert len(set(seeds)) == len(seeds), "two reverted predictions share a seed"


@pytest.mark.local_integration
def test_a_design_with_no_receptor_fasta_is_skipped_not_fatal(tmp_path):
    """One unreadable design must not cost the whole cohort its reversion
    pass. The engine tolerates per-design failure everywhere else too."""
    wd = tmp_path / "cycle_0"
    wd.mkdir(parents=True)
    good = _design(tmp_path, "design_00", "AWDEFGHIKLMNPQRSTVWY")
    missing = tmp_path / "cycle_0" / "steered" / "design_99"
    missing.mkdir(parents=True)

    rev.write_reversion_plan(wd, [
        _contaminated("design_99", missing, [(2, "C", "W")], [2]),
        _contaminated("design_00", good, [(2, "C", "W")], [2]),
    ], _meta())

    plan = _plan(wd)
    assert plan["n_staged"] == 1
    assert plan["entries"][0]["label"] == "design_00"


@pytest.mark.local_integration
def test_a_design_whose_sequence_contradicts_its_history_is_skipped(tmp_path):
    """Same tolerance, for the case build_reverted_sequence rejects."""
    wd = tmp_path / "cycle_0"
    wd.mkdir(parents=True)
    d = _design(tmp_path, "design_00", "AQDEFGHIKLMNPQRSTVWY")  # not W at 2

    rev.write_reversion_plan(
        wd, [_contaminated("design_00", d, [(2, "C", "W")], [2])], _meta())
    assert _plan(wd)["n_staged"] == 0


@pytest.mark.local_integration
def test_no_contaminated_designs_still_writes_a_plan(tmp_path):
    """This is the common case, and the one the single-run test hit. The harvest
    reads reversion_plan.json unconditionally, so its absence would turn a
    normal outcome into a crash."""
    wd = tmp_path / "cycle_0"
    wd.mkdir(parents=True)
    rev.write_reversion_plan(wd, [], _meta())

    plan = _plan(wd)
    assert plan["n_contaminated"] == 0
    assert plan["n_staged"] == 0
    assert plan["entries"] == []


@pytest.mark.local_integration
def test_the_manifest_records_what_was_reverted_and_from_what(tmp_path):
    """The harvest and every downstream table read these fields. A missing one
    surfaces as a blank column rather than an error."""
    wd = tmp_path / "cycle_0"
    wd.mkdir(parents=True)
    d = _design(tmp_path, "design_00", "AWDEFGHIKLMNPQRSTVWY")

    rev.write_reversion_plan(
        wd, [_contaminated("design_00", d, [(2, "C", "W")], [2])], _meta())

    meta = json.loads(
        (wd / "reversions" / "rev_design_00" / "reversion_metadata.json").read_text())
    assert meta["positions_to_revert"] == [2]
    assert meta["applied_reversions"] == [
        {"pos1": 2, "steered": "W", "wild_type": "C"}]
    assert meta["steered_ra_eff_vs_truth"] == 3.2
    assert meta["cumulative_mutations"] == [{"pos1": 2, "wt": "C", "mut": "W"}]
    assert Path(meta["steered_prediction_pdb"]).name == "prediction.pdb"
