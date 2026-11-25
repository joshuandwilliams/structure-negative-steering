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
