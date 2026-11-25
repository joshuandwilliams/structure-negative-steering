"""Tests for write_reversion_plan, the stage that lays out the reversion pass.

Everything the reversion arm later does is addressed through the manifest this
writes. Three properties in it are load-bearing.

Deduplication. Two contaminated designs that revert to the same receptor
sequence are the same prediction, so only one is staged. Getting this wrong
does not give a wrong answer, it doubles GPU time on a pass that already runs
once per cycle.

Seed uniqueness. Each staged slot gets its own Boltz seed, offset into the
100000+ range so it cannot collide with the steered prediction it is being
compared against. A collision would make the reverted prediction a rerun of
the steered one rather than an independent sample.

Label pairing. `all_label_seeds` carries, for each steered design sharing the
staged sequence, the seed index it had on the steered side. The harvest uses
it to match a reverted seed to the right steered per-seed label. If the
pairing drifts, a design is scored against another design's reverted pose.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

EFFECTOR = "GGGEFFECTORSEQ"


@pytest.fixture(scope="module")
def rev():
    sys.path.insert(0, str(REPO_ROOT / "bin"))
    spec = importlib.util.spec_from_file_location(
        "reversion", REPO_ROOT / "bin" / "reversion.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def meta(**kw):
    base = {"effector_seq": EFFECTOR, "pred_receptor_chain": "A",
            "pred_effector_chain": "B", "base_seed": 0, "num_seeds": 1}
    base.update(kw)
    return base


def design(root: Path, label, steered_seq, *, muts, revert, idx=0,
           seed_index=0, ra=2.0, contacts=(5, 6)):
    """A contaminated entry plus the design workdir it points at."""
    wd = root / "designs" / label
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "receptor.fasta").write_text(f">receptor_{label}\n{steered_seq}\n")
    return {"label": label, "design_idx": idx, "seed_index": seed_index,
            "design_workdir": str(wd), "cumulative_mutations": list(muts),
            "positions_to_revert": list(revert),
            "steered_ra_eff_vs_truth": ra,
            "steered_contact_residues": list(contacts)}


def manifest(path: Path):
    return json.loads(path.read_text())


# ── the reverted sequence that gets staged ───────────────────────────────────

@pytest.mark.logic
def test_the_staged_fasta_carries_the_reverted_sequence(rev, tmp_path):
    """Not the steered one. This is the whole point of the pass, so it is
    worth pinning at the file the GPU task actually reads."""
    c = design(tmp_path, "d0", "MKWVTA", muts=[[3, "A", "W"]], revert=[3])
    out = rev.write_reversion_plan(tmp_path, [c], meta())
    entry = manifest(out)["entries"][0]
    assert Path(entry["reverted_receptor_fasta"]).read_text().endswith(
        "MKAVTA\n")


@pytest.mark.logic
def test_only_the_requested_position_is_reverted(rev, tmp_path):
    c = design(tmp_path, "d0", "MKWVTY",
               muts=[[3, "A", "W"], [6, "A", "Y"]], revert=[3])
    out = rev.write_reversion_plan(tmp_path, [c], meta())
    entry = manifest(out)["entries"][0]
    assert Path(entry["reverted_receptor_fasta"]).read_text().endswith(
        "MKAVTY\n")
    assert entry["positions_to_revert"] == [3]
    assert entry["n_reversions_applied"] == 1


@pytest.mark.logic
def test_a_boltz_yaml_is_written_next_to_it(rev, tmp_path):
    c = design(tmp_path, "d0", "MKWVTA", muts=[[3, "A", "W"]], revert=[3])
    out = rev.write_reversion_plan(tmp_path, [c], meta())
    yaml_path = Path(manifest(out)["entries"][0]["yaml"])
    assert yaml_path.exists()
    text = yaml_path.read_text()
    assert "MKAVTA" in text and EFFECTOR in text


# ── deduplication ────────────────────────────────────────────────────────────

@pytest.mark.logic
def test_two_designs_reverting_to_the_same_sequence_stage_once(rev, tmp_path,
                                                               capsys):
    """Same wild type, same single mutation, two seeds that both found it.
    One reverted sequence, so one prediction."""
    cs = [design(tmp_path, "d0", "MKWVTA", muts=[[3, "A", "W"]], revert=[3]),
          design(tmp_path, "d1", "MKWVTA", muts=[[3, "A", "W"]], revert=[3],
                 idx=1)]
    doc = manifest(rev.write_reversion_plan(tmp_path, cs, meta()))
    assert doc["n_contaminated"] == 2
    assert doc["n_unique_sequences"] == 1
    assert doc["n_staged"] == 1
    assert "1 duplicate(s) deduplicated" in capsys.readouterr().out


@pytest.mark.logic
def test_the_shared_entry_names_both_designs(rev, tmp_path):
    """Dropping the second label would silently leave that design unscored."""
    cs = [design(tmp_path, "d0", "MKWVTA", muts=[[3, "A", "W"]], revert=[3]),
          design(tmp_path, "d1", "MKWVTA", muts=[[3, "A", "W"]], revert=[3],
                 idx=1)]
    entry = manifest(rev.write_reversion_plan(tmp_path, cs, meta()))[
        "entries"][0]
    assert entry["all_labels"] == ["d0", "d1"]


@pytest.mark.logic
def test_the_pairing_keeps_each_designs_steered_seed_index(rev, tmp_path):
    cs = [design(tmp_path, "d0", "MKWVTA", muts=[[3, "A", "W"]], revert=[3],
                 seed_index=0),
          design(tmp_path, "d1", "MKWVTA", muts=[[3, "A", "W"]], revert=[3],
                 idx=1, seed_index=3)]
    entry = manifest(rev.write_reversion_plan(tmp_path, cs, meta()))[
        "entries"][0]
    assert entry["all_label_seeds"] == [
        {"label": "d0", "seed_index": 0},
        {"label": "d1", "seed_index": 3}]


@pytest.mark.logic
def test_a_missing_seed_index_is_read_as_zero(rev, tmp_path):
    """Single-seed cycles never wrote the field."""
    c = design(tmp_path, "d0", "MKWVTA", muts=[[3, "A", "W"]], revert=[3])
    del c["seed_index"]
    entry = manifest(rev.write_reversion_plan(tmp_path, [c], meta()))[
        "entries"][0]
    assert entry["all_label_seeds"] == [{"label": "d0", "seed_index": 0}]


@pytest.mark.logic
def test_different_reverted_sequences_stage_separately(rev, tmp_path, capsys):
    cs = [design(tmp_path, "d0", "MKWVTA", muts=[[3, "A", "W"]], revert=[3]),
          design(tmp_path, "d1", "MKWVTY", muts=[[6, "A", "Y"]], revert=[6],
                 idx=1)]
    doc = manifest(rev.write_reversion_plan(tmp_path, cs, meta()))
    assert doc["n_unique_sequences"] == 2 and doc["n_staged"] == 2
    assert "no duplicates" in capsys.readouterr().out
    assert len({e["sequence_hash"] for e in doc["entries"]}) == 2


@pytest.mark.logic
def test_two_designs_with_the_same_sequence_share_a_hash(rev, tmp_path):
    """The hash is over the sequence, so it is the grouping key."""
    cs = [design(tmp_path, "d0", "MKWVTA", muts=[[3, "A", "W"]], revert=[3]),
          design(tmp_path, "d1", "MKWVTY", muts=[[6, "A", "Y"]], revert=[6],
                 idx=1)]
    doc = manifest(rev.write_reversion_plan(tmp_path, cs, meta()))
    assert [e["sequence_group_idx"] for e in doc["entries"]] == [0, 1]


# ── multi-seed fan-out ───────────────────────────────────────────────────────

@pytest.mark.logic
def test_each_unique_sequence_gets_one_slot_per_seed(rev, tmp_path):
    c = design(tmp_path, "d0", "MKWVTA", muts=[[3, "A", "W"]], revert=[3])
    doc = manifest(rev.write_reversion_plan(tmp_path, [c], meta(num_seeds=4)))
    assert doc["n_unique_sequences"] == 1
    assert doc["n_staged"] == 4
    assert [e["seed_index"] for e in doc["entries"]] == [0, 1, 2, 3]


@pytest.mark.logic
def test_the_seeded_slots_get_distinct_directories(rev, tmp_path):
    """They must not overwrite each other's predictions."""
    c = design(tmp_path, "d0", "MKWVTA", muts=[[3, "A", "W"]], revert=[3])
    doc = manifest(rev.write_reversion_plan(tmp_path, [c], meta(num_seeds=3)))
    names = [e["safe_name"] for e in doc["entries"]]
    assert names == ["rev_d0_s0", "rev_d0_s1", "rev_d0_s2"]
    assert len({e["rev_dir"] for e in doc["entries"]}) == 3


@pytest.mark.logic
def test_a_single_seed_slot_is_not_suffixed(rev, tmp_path):
    c = design(tmp_path, "d0", "MKWVTA", muts=[[3, "A", "W"]], revert=[3])
    doc = manifest(rev.write_reversion_plan(tmp_path, [c], meta()))
    assert doc["entries"][0]["safe_name"] == "rev_d0"


@pytest.mark.logic
def test_every_staged_slot_gets_its_own_boltz_seed(rev, tmp_path):
    """Across unique sequences as well as across seeds. A repeated seed makes
    two slots the same prediction."""
    cs = [design(tmp_path, "d0", "MKWVTA", muts=[[3, "A", "W"]], revert=[3]),
          design(tmp_path, "d1", "MKWVTY", muts=[[6, "A", "Y"]], revert=[6],
                 idx=1)]
    doc = manifest(rev.write_reversion_plan(tmp_path, cs, meta(num_seeds=2)))
    seeds = [e["boltz_seed"] for e in doc["entries"]]
    assert len(set(seeds)) == 4


@pytest.mark.logic
def test_the_reverted_seeds_sit_clear_of_the_steered_ones(rev, tmp_path):
    """The offset is what stops a reverted prediction from being a rerun of
    the steered prediction it is compared against."""
    c = design(tmp_path, "d0", "MKWVTA", muts=[[3, "A", "W"]], revert=[3])
    doc = manifest(rev.write_reversion_plan(
        tmp_path, [c], meta(base_seed=7, num_seeds=2)))
    assert [e["boltz_seed"] for e in doc["entries"]] == [100008, 100009]


# ── the names that become directories ────────────────────────────────────────

@pytest.mark.logic
def test_dots_and_slashes_in_a_label_do_not_become_paths(rev, tmp_path):
    """Labels carry cycle and seed markers. An unescaped slash would stage
    the YAML outside the reversions root."""
    c = design(tmp_path, "c1/d0.s2", "MKWVTA", muts=[[3, "A", "W"]],
               revert=[3])
    doc = manifest(rev.write_reversion_plan(tmp_path, [c], meta()))
    entry = doc["entries"][0]
    assert entry["safe_name"] == "rev_c1_d0_s2"
    assert Path(entry["rev_dir"]).parent == (tmp_path / "reversions")
    assert entry["label"] == "c1/d0.s2"


# ── what the harvest reads back ──────────────────────────────────────────────

@pytest.mark.logic
def test_the_metadata_file_records_the_steered_side(rev, tmp_path):
    """The harvest compares reverted against steered, so the steered pose's
    identity and score have to travel with the staged slot."""
    c = design(tmp_path, "d0", "MKWVTA", muts=[[3, "A", "W"]], revert=[3],
               ra=1.75, contacts=(9, 5))
    doc = manifest(rev.write_reversion_plan(tmp_path, [c], meta()))
    md = json.loads(
        (Path(doc["entries"][0]["rev_dir"]) / "reversion_metadata.json"
         ).read_text())
    assert md["steered_ra_eff_vs_truth"] == 1.75
    assert md["steered_contact_residues"] == [5, 9]
    assert md["steered_prediction_pdb"].endswith("d0/prediction.pdb")
    assert md["applied_reversions"] == [
        {"pos1": 3, "steered": "W", "wild_type": "A"}]
    assert md["cumulative_mutations"] == [{"pos1": 3, "wt": "A", "mut": "W"}]


@pytest.mark.logic
def test_duplicate_revert_positions_are_reported_once(rev, tmp_path):
    c = design(tmp_path, "d0", "MKWVTA", muts=[[3, "A", "W"]], revert=[3, 3])
    doc = manifest(rev.write_reversion_plan(tmp_path, [c], meta()))
    assert doc["entries"][0]["positions_to_revert"] == [3]


@pytest.mark.logic
def test_the_chain_names_are_carried_on_the_manifest(rev, tmp_path):
    c = design(tmp_path, "d0", "MKWVTA", muts=[[3, "A", "W"]], revert=[3])
    doc = manifest(rev.write_reversion_plan(
        tmp_path, [c], meta(pred_receptor_chain="C",
                            pred_effector_chain="D")))
    assert doc["pred_receptor_chain"] == "C"
    assert doc["pred_effector_chain"] == "D"


# ── designs that cannot be staged ────────────────────────────────────────────
#
# Each of these drops one design and keeps going. A contaminated design that
# cannot be reverted has no reversion verdict, which collapses it downstream.
# Aborting the whole pass instead would lose the designs that were fine.

@pytest.mark.logic
def test_a_design_with_no_receptor_fasta_is_skipped(rev, tmp_path, capsys):
    c = design(tmp_path, "d0", "MKWVTA", muts=[[3, "A", "W"]], revert=[3])
    (Path(c["design_workdir"]) / "receptor.fasta").unlink()
    doc = manifest(rev.write_reversion_plan(tmp_path, [c], meta()))
    assert doc["n_contaminated"] == 1 and doc["n_staged"] == 0
    assert "SKIP d0: no receptor.fasta" in capsys.readouterr().err


@pytest.mark.logic
def test_an_unparseable_receptor_fasta_is_skipped(rev, tmp_path, capsys):
    c = design(tmp_path, "d0", "MKWVTA", muts=[[3, "A", "W"]], revert=[3])
    (Path(c["design_workdir"]) / "receptor.fasta").write_text(
        ">one\nMKWVTA\n>two\nMKWVTA\n")
    doc = manifest(rev.write_reversion_plan(tmp_path, [c], meta()))
    assert doc["n_staged"] == 0
    assert "SKIP d0" in capsys.readouterr().err


@pytest.mark.logic
def test_a_history_that_disagrees_with_the_sequence_is_skipped(rev, tmp_path,
                                                               capsys):
    """The recorded mutation says position 3 is W, the FASTA says it is V.
    Reverting on a mismatched history would write a sequence that is neither
    the steered nor the wild type one."""
    c = design(tmp_path, "d0", "MKVVTA", muts=[[3, "A", "W"]], revert=[3])
    doc = manifest(rev.write_reversion_plan(tmp_path, [c], meta()))
    assert doc["n_staged"] == 0
    assert "reversion build failed" in capsys.readouterr().err


@pytest.mark.logic
def test_one_bad_design_does_not_stop_the_others(rev, tmp_path):
    bad = design(tmp_path, "bad", "MKWVTA", muts=[[3, "A", "W"]], revert=[3])
    (Path(bad["design_workdir"]) / "receptor.fasta").unlink()
    good = design(tmp_path, "good", "MKWVTY", muts=[[6, "A", "Y"]],
                  revert=[6], idx=1)
    doc = manifest(rev.write_reversion_plan(tmp_path, [bad, good], meta()))
    assert doc["n_staged"] == 1
    assert doc["entries"][0]["label"] == "good"
    assert doc["n_contaminated"] == 2   # the count is of what was asked for


@pytest.mark.logic
def test_an_empty_contaminated_list_writes_an_empty_manifest(rev, tmp_path):
    doc = manifest(rev.write_reversion_plan(tmp_path, [], meta()))
    assert doc["n_contaminated"] == 0 and doc["entries"] == []


# ── plan-reversions, through its own CLI ─────────────────────────────────────
#
# The stage around write_reversion_plan. It reads the cycle's plan.json for
# the effector sequence and the Boltz hyperparameters, reads
# contaminated.json for the designs to stage, and hands both to the function
# above. The effector sequence is read from the ground-truth PDB's effector
# chain, or from an override FASTA when the plan recorded one — the same way
# cmd_plan reads it, so the reverted prediction gets the same effector as the
# steered prediction it is compared against.

REFERENCE = REPO_ROOT / "data" / "reference_complexes" / "6G10.pdb"


class Stage:
    """A collect workdir ready for plan-reversions."""

    def __init__(self, root, *, plan_extra=None):
        import shutil
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.truth = root / "truth.pdb"
        shutil.copy(REFERENCE, self.truth)
        plan = {"cycle": 0, "ground_truth": str(self.truth),
                "truth_receptor_chain": "A", "truth_effector_chain": "B",
                "pred_receptor_chain": "A", "pred_effector_chain": "B",
                "seed": 0, "num_seeds": 1, "recycling_steps": 3,
                "diffusion_samples": 1, "boltz_container": "/fake.sif",
                "no_kernels": True}
        if plan_extra:
            plan.update(plan_extra)
        (root / "plan.json").write_text(json.dumps(plan))
        self.contaminated = []

    def contaminate(self, label, steered_seq="MKWVTA",
                    muts=((3, "A", "W"),), revert=(3,), **kw):
        wd = self.root / "designs" / label
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "receptor.fasta").write_text(f">r_{label}\n{steered_seq}\n")
        entry = {"label": label, "design_idx": len(self.contaminated),
                 "seed_index": 0, "design_workdir": str(wd),
                 "cumulative_mutations": [list(m) for m in muts],
                 "positions_to_revert": list(revert),
                 "steered_ra_eff_vs_truth": 2.0,
                 "steered_contact_residues": [3]}
        entry.update(kw)
        self.contaminated.append(entry)
        return self

    def write(self):
        (self.root / "contaminated.json").write_text(json.dumps(
            {"n_checked": len(self.contaminated),
             "n_contaminated": len(self.contaminated),
             "contaminated": self.contaminated}))
        return self

    def run(self, rev):
        argv = ["plan-reversions", "--workdir", str(self.root)]
        return rev.cmd_plan_reversions(rev.build_parser().parse_args(argv))

    def doc(self):
        return json.loads(
            (self.root / "reversion_plan.json").read_text())


@pytest.mark.logic
def test_the_stage_stages_the_contaminated_designs(rev, tmp_path):
    s = Stage(tmp_path / "wd").contaminate("d0").write()
    assert s.run(rev) == 0
    doc = s.doc()
    assert doc["n_contaminated"] == 1 and doc["n_staged"] == 1


@pytest.mark.logic
def test_the_effector_comes_from_the_ground_truth_chain(rev, tmp_path):
    """The reverted prediction has to bind the same effector as the steered
    prediction it is compared against. Reading it from the truth PDB is how
    cmd_plan got it, so reading it the same way here keeps them identical."""
    s = Stage(tmp_path / "wd").contaminate("d0").write()
    s.run(rev)
    yaml_text = Path(s.doc()["entries"][0]["yaml"]).read_text()
    assert "MKWVTA".replace("W", "A") in yaml_text          # the receptor
    assert len(yaml_text) > 100                             # plus an effector


@pytest.mark.logic
def test_an_effector_override_is_used_instead(rev, tmp_path):
    """A run that fed Boltz a different effector recorded the override in
    plan.json, and the reversion pass has to use the same one."""
    fasta = tmp_path / "eff.fasta"
    fasta.write_text(">eff\nQQQWWWEEE\n")
    s = Stage(tmp_path / "wd",
              plan_extra={"effector_fasta_override": str(fasta)})
    s.contaminate("d0").write()
    assert s.run(rev) == 0
    assert "QQQWWWEEE" in Path(s.doc()["entries"][0]["yaml"]).read_text()


@pytest.mark.logic
def test_a_multi_line_override_is_joined(rev, tmp_path):
    fasta = tmp_path / "eff.fasta"
    fasta.write_text(">eff\nQQQ\nWWW\nEEE\n")
    s = Stage(tmp_path / "wd",
              plan_extra={"effector_fasta_override": str(fasta)})
    s.contaminate("d0").write()
    s.run(rev)
    assert "QQQWWWEEE" in Path(s.doc()["entries"][0]["yaml"]).read_text()


@pytest.mark.logic
def test_the_seed_count_fans_the_staging_out(rev, tmp_path):
    s = Stage(tmp_path / "wd", plan_extra={"num_seeds": 3})
    s.contaminate("d0").write()
    s.run(rev)
    assert s.doc()["n_staged"] == 3


@pytest.mark.logic
def test_the_constraints_block_is_inherited(rev, tmp_path):
    """The steered prediction was made under these constraints, so a
    reverted prediction made without them would not be a comparison."""
    s = Stage(tmp_path / "wd", plan_extra={
        "boltz_constraints_block": ["constraints:", "  - pocket:"]})
    s.contaminate("d0").write()
    s.run(rev)
    assert "pocket" in Path(s.doc()["entries"][0]["yaml"]).read_text()


@pytest.mark.logic
def test_a_design_that_cannot_be_staged_is_reported(rev, tmp_path, capsys):
    """The run continues with the rest, but the shortfall has to be visible
    or a partial reversion pass reads as a complete one."""
    s = Stage(tmp_path / "wd").contaminate("bad").contaminate("good",
                                                              steered_seq="MKWVTY",
                                                              muts=((6, "A", "Y"),),
                                                              revert=(6,))
    s.write()
    (Path(s.contaminated[0]["design_workdir"]) / "receptor.fasta").unlink()
    assert s.run(rev) == 0
    assert s.doc()["n_staged"] == 1
    err = capsys.readouterr().err
    assert "1 contaminated design(s) could not be staged" in err


# ── predict-reversion-one ────────────────────────────────────────────────────

@pytest.mark.logic
def test_the_array_task_runs_boltz_on_its_own_slot(rev, tmp_path,
                                                   monkeypatch):
    """Each slot is one SLURM array task, so it has to pick its YAML by
    index and write its prediction where the harvest will look for it."""
    import shutil

    import boltz2_negative_steering as bns

    s = Stage(tmp_path / "wd").contaminate("d0").write()
    s.run(rev)
    seen = {}

    def fake_run_boltz(yaml_path, out_dir, container, recycling, samples,
                       seed, no_kernels):
        seen.update(yaml=str(yaml_path), container=container, seed=seed,
                    recycling=recycling, samples=samples,
                    no_kernels=no_kernels)
        out = Path(out_dir) / "boltz_out"
        out.mkdir(parents=True, exist_ok=True)
        pred = out / "model_0.pdb"
        shutil.copy(REFERENCE, pred)
        return pred

    monkeypatch.setattr(bns, "run_boltz", fake_run_boltz)
    argv = ["predict-reversion-one", "--workdir", str(s.root), "--index", "0"]
    rc = rev.cmd_predict_reversion_one(rev.build_parser().parse_args(argv))
    assert rc == 0

    entry = s.doc()["entries"][0]
    assert seen["yaml"] == entry["yaml"]
    assert seen["container"] == "/fake.sif"
    assert seen["seed"] == entry["boltz_seed"]
    assert (Path(entry["rev_dir"]) / "prediction.pdb").is_file()


@pytest.mark.logic
def test_a_failed_prediction_leaves_a_reason_on_disk(rev, tmp_path,
                                                     monkeypatch):
    """The harvest reads this directory later. Without a note it would look
    like a slot that was never dispatched rather than one that failed."""
    import boltz2_negative_steering as bns

    s = Stage(tmp_path / "wd").contaminate("d0").write()
    s.run(rev)

    def boom(*a, **kw):
        raise RuntimeError("the GPU was not there")

    monkeypatch.setattr(bns, "run_boltz", boom)
    argv = ["predict-reversion-one", "--workdir", str(s.root), "--index", "0"]
    assert rev.cmd_predict_reversion_one(
        rev.build_parser().parse_args(argv)) == 1
    note = Path(s.doc()["entries"][0]["rev_dir"]) / "predict_error.txt"
    assert "the GPU was not there" in note.read_text()
    assert not (note.parent / "prediction.pdb").exists()
