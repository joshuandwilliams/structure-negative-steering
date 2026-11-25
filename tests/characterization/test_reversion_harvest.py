"""Tests for harvest_reversion_results and its helpers.

This is where the reversion arm produces its answer. For every staged slot it
measures the reverted pose against the ground truth and against its own
steered parent, re-runs the contamination check on what the reversion left in
place, and writes one record per label. Those records are what
classify_reversion_verdict reads, so an error here does not show up as a
crash. It shows up as a design reported as rescued, or as collapsed, on the
strength of another design's numbers.

Three things are pinned because each has broken before or would break
silently.

Per-seed pairing. Before the multi-seed fix the loop wrote `results[label]`
with the same label for every seed, so only the last seed's record survived
and the rest of the GPU work was discarded. The harvest now keys records by
(canonical label, seed index) and pairs each steered label with the reverted
record at *its* seed index.

The remaining-mutation set. A reverted design still carries every mutation the
pass did not revert, and those are the ones that can re-contaminate the rescue
pose. Computing the set as "all cumulative mutations" or as "none" both give a
wrong contamination verdict in opposite directions.

The receptor-length guard on the Jaccard. The guard used to compare amino
acids against cycle_0's wild-type receptor, which differs from the reverted
sequence at every surviving mutation, so it raised on every pose_holds row and
the Jaccards came back blank. Length is the invariant that actually matters
for index alignment, and that is what is checked now.

The structural work is real here, against data/reference_complexes/6G10.pdb,
so binding_rmsds runs its actual Kabsch fit. compute_metrics.py is stubbed.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE = REPO_ROOT / "data" / "reference_complexes" / "6G10.pdb"


def _deps() -> bool:
    return all(importlib.util.find_spec(m) is not None
               for m in ("numpy", "Bio"))


pytestmark = pytest.mark.skipif(
    not _deps(),
    reason="needs numpy and biopython (pip install -e '.[experiments]')")


@pytest.fixture(scope="module")
def rev():
    sys.path.insert(0, str(REPO_ROOT / "bin"))
    spec = importlib.util.spec_from_file_location(
        "reversion", REPO_ROOT / "bin" / "reversion.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── the stub ─────────────────────────────────────────────────────────────────
#
# It answers from a file in the prediction directory, so each slot can be made
# to return different metrics, fail, or hang.

STUB = """\
import csv, json, sys
argv = sys.argv[1:]
out = argv[argv.index("--output-csv") + 1]
pdir = argv[argv.index("--prediction-dir") + 1]
json.dump(argv, open(pdir + "/argv.json", "w"))
vals = {}
try:
    for line in open(pdir + "/stub_values.txt").read().split():
        k, v = line.split("=", 1)
        vals[k] = v
except FileNotFoundError:
    pass
if vals.get("FAIL"):
    sys.stderr.write("stub refused\\n")
    sys.exit(5)
fields = ["model_name", "iptm", "ptm", "complex_plddt", "ipsae_min",
          "n_contact_residues", "contact_residues", "interface_plddt",
          "intact_core", "weighted_jaccard", "ipae"]
rec = {k: vals.get(k, "") for k in fields}
rec["model_name"] = "boltz2"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerow(rec)
"""

DEFAULT_VALUES = ("iptm=0.8 ptm=0.7 complex_plddt=88 ipsae_min=0.6 ipae=3.0 "
                  "n_contact_residues=3 contact_residues=5,26,44 "
                  "interface_plddt=81 intact_core=1 weighted_jaccard=0.55")


class Harvest:
    """A collect workdir with staged reversion slots around 6G10.

    Every PDB is a copy of the reference complex, so binding_rmsds reports
    near-zero and the structural path runs for real without the test having
    to assert on a particular RMSD.
    """

    def __init__(self, root: Path, *, plan=None):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.truth = root / "truth.pdb"
        shutil.copy(REFERENCE, self.truth)
        self.script = root / "stub_cm.py"
        self.script.write_text(STUB)
        self.manifest = {"pred_receptor_chain": "A",
                         "pred_effector_chain": "B", "entries": []}
        if plan is not None:
            (root / "plan.json").write_text(json.dumps(
                {"cycle": 0, **plan}))

    def slot(self, label, *, seed_index=0, members=None, values=DEFAULT_VALUES,
             positions_to_revert=(3,), cumulative=((3, "A", "W"),),
             prediction=True, steered=True, name=None):
        name = name or f"rev_{label}_s{seed_index}"
        rev_dir = self.root / "reversions" / name
        rev_dir.mkdir(parents=True, exist_ok=True)
        if prediction:
            shutil.copy(REFERENCE, rev_dir / "prediction.pdb")
        if values is not None:
            (rev_dir / "stub_values.txt").write_text(values)
        (rev_dir / "reversion_metadata.json").write_text(json.dumps(
            {"cumulative_mutations": [{"pos1": p, "wt": w, "mut": m}
                                      for p, w, m in cumulative]}))

        design_workdir = self.root / "designs" / label
        design_workdir.mkdir(parents=True, exist_ok=True)
        if steered:
            shutil.copy(REFERENCE, design_workdir / "prediction.pdb")

        entry = {"label": label, "seed_index": seed_index,
                 "rev_dir": str(rev_dir),
                 "design_workdir": str(design_workdir),
                 "positions_to_revert": list(positions_to_revert)}
        if members is None:
            entry["all_label_seeds"] = [{"label": label,
                                         "seed_index": seed_index}]
        else:
            entry["all_label_seeds"] = [dict(m) for m in members]
        self.manifest["entries"].append(entry)
        return self

    def write(self):
        (self.root / "reversion_plan.json").write_text(
            json.dumps(self.manifest))
        return self

    def run(self, rev, **kw):
        return rev.harvest_reversion_results(
            workdir=self.root, ground_truth_pdb=self.truth,
            truth_receptor_chain="A", truth_effector_chain="B",
            compute_metrics_script=kw.pop("script", self.script), **kw)

    def argv(self, name):
        return json.loads(
            (self.root / "reversions" / name / "argv.json").read_text())


@pytest.fixture
def h(tmp_path):
    return Harvest(tmp_path / "wd")


# ── the refusal ──────────────────────────────────────────────────────────────

@pytest.mark.logic
def test_harvesting_without_a_manifest_is_an_error(rev, h):
    """Unlike the CLI wrapper, which no-ops permissively, the library call
    raises. A caller that reached here without staging has a bug."""
    with pytest.raises(FileNotFoundError, match="run plan-reversions first"):
        h.run(rev)


@pytest.mark.logic
def test_an_empty_manifest_harvests_to_nothing(rev, h):
    assert h.write().run(rev) == {}


# ── the successful record ────────────────────────────────────────────────────

@pytest.mark.logic
def test_a_reverted_copy_of_the_truth_reports_a_held_pose(rev, h):
    """Every PDB here is the same complex, so the reverted pose sits on the
    ground truth and the receptor is intact by construction."""
    res = h.slot("d0").write().run(rev)
    rec = res["d0"]
    assert rec["reverted_ra_eff_vs_truth"] == pytest.approx(0.0, abs=1e-6)
    assert rec["reverted_independent_receptor_rmsd"] == pytest.approx(
        0.0, abs=1e-6)
    assert rec["reverted_intact"] is True
    assert "error" not in rec


@pytest.mark.logic
def test_the_drift_from_the_steered_parent_is_measured_separately(rev, h):
    """vs_truth says whether the rescue worked. vs_steered says whether it is
    the same pose, which is a different question and reported separately."""
    rec = h.slot("d0").write().run(rev)["d0"]
    assert rec["reverted_ra_eff_vs_steered"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.logic
def test_a_displaced_effector_is_reported_at_its_displacement(rev, h,
                                                              tmp_path):
    """The structural path is real, not stubbed, so a known displacement has
    to come back as that displacement."""
    h.slot("d0").write()
    rev_pdb = Path(h.manifest["entries"][0]["rev_dir"]) / "prediction.pdb"
    _translate_chain(rev_pdb, "B", (7.5, 0.0, 0.0))
    rec = h.run(rev)["d0"]
    assert rec["reverted_ra_eff_vs_truth"] == pytest.approx(7.5, abs=1e-3)
    assert rec["reverted_independent_receptor_rmsd"] == pytest.approx(
        0.0, abs=1e-6)


@pytest.mark.logic
def test_a_broken_receptor_is_not_intact(rev, h):
    """Intactness gates the verdict, so a receptor whose fold has come apart
    cannot be reported as held. The distortion here is non-rigid, because
    independent_receptor_rmsd is a Kabsch fit and a rigid move of the whole
    chain reports zero by construction."""
    h.slot("d0").write()
    rev_pdb = Path(h.manifest["entries"][0]["rev_dir"]) / "prediction.pdb"
    _translate_chain(rev_pdb, "A", (0.0, 0.0, 25.0), every_other=True)
    rec = h.run(rev)["d0"]
    assert rec["reverted_independent_receptor_rmsd"] > 5.0
    assert rec["reverted_intact"] is False


@pytest.mark.logic
def test_the_confidence_suite_is_carried_onto_the_record(rev, h):
    rec = h.slot("d0").write().run(rev)["d0"]
    assert rec["reverted_iptm"] == 0.8
    assert rec["reverted_ptm"] == 0.7
    assert rec["reverted_complex_plddt"] == 88.0
    assert rec["reverted_ipsae_min"] == 0.6
    assert rec["reverted_interface_plddt"] == 81.0
    assert rec["reverted_intact_core"] == 1
    assert rec["reverted_weighted_jaccard"] == 0.55


@pytest.mark.logic
def test_a_metric_the_stub_left_blank_becomes_none(rev, h):
    """Not zero. A blank ipSAE means it could not be computed, and zero is a
    real score that would be averaged into the medians."""
    rec = h.slot("d0", values="iptm=0.8").write().run(rev)["d0"]
    assert rec["reverted_ipsae_min"] is None
    assert rec["reverted_ptm"] is None


@pytest.mark.logic
def test_an_unparseable_metric_becomes_none_rather_than_raising(rev, h):
    rec = h.slot("d0", values="iptm=NA ptm=0.7 n_contact_residues=many"
                 ).write().run(rev)["d0"]
    assert rec["reverted_iptm"] is None
    assert rec["reverted_n_contact_residues"] is None
    assert rec["reverted_ptm"] == 0.7


# ── the contamination check on the reverted pose ─────────────────────────────

@pytest.mark.logic
def test_a_surviving_mutation_in_contact_is_new_contamination(rev, h):
    """Position 26 was not reverted and contacts the effector in the reverted
    pose, so the rescue is held by a residue the construct will not carry."""
    rec = (h.slot("d0", positions_to_revert=(3,),
                  cumulative=((3, "A", "W"), (26, "A", "Y")))
           .write().run(rev))["d0"]
    assert rec["reverted_mutated_contact_positions"] == [26]
    assert rec["reverted_n_contacts_on_mutated_positions"] == 1


@pytest.mark.logic
def test_a_reverted_position_in_contact_is_not_contamination(rev, h):
    """Position 3 is back to wild type, so a contact there is a contact the
    construct will reproduce."""
    rec = (h.slot("d0", positions_to_revert=(3, 5),
                  cumulative=((3, "A", "W"), (5, "A", "Y")))
           .write().run(rev))["d0"]
    assert rec["reverted_mutated_contact_positions"] == []
    assert rec["reverted_n_contacts_on_mutated_positions"] == 0


@pytest.mark.logic
def test_a_surviving_mutation_away_from_the_interface_is_clean(rev, h):
    rec = (h.slot("d0", positions_to_revert=(3,),
                  cumulative=((3, "A", "W"), (99, "A", "Y")))
           .write().run(rev))["d0"]
    assert rec["reverted_mutated_contact_positions"] == []


@pytest.mark.logic
def test_only_the_surviving_mutations_are_handed_to_compute_metrics(rev, h):
    """The reverted positions carry wild type again, so passing them would
    ask the contamination check about residues that are no longer there."""
    (h.slot("d0", positions_to_revert=(3,),
            cumulative=((3, "A", "W"), (26, "A", "Y"), (44, "A", "F")))
     .write().run(rev))
    argv = h.argv("rev_d0_s0")
    assert argv[argv.index("--mutated-positions") + 1] == "26,44"


@pytest.mark.logic
def test_a_fully_reverted_design_passes_no_mutation_flag(rev, h):
    (h.slot("d0", positions_to_revert=(3,), cumulative=((3, "A", "W"),))
     .write().run(rev))
    assert "--mutated-positions" not in h.argv("rev_d0_s0")


@pytest.mark.logic
def test_the_ground_truth_is_passed_so_the_native_metrics_populate(rev, h):
    """Without --native-pdb, intact_core and weighted_jaccard come out blank
    on the reverted side while the steered side has them, which makes the
    two stages uncomparable."""
    h.slot("d0").write().run(rev)
    argv = h.argv("rev_d0_s0")
    assert argv[argv.index("--native-pdb") + 1] == str(h.truth)


@pytest.mark.logic
def test_the_chain_lengths_come_off_the_reverted_pdb(rev, h):
    h.slot("d0").write().run(rev)
    argv = h.argv("rev_d0_s0")
    i = argv.index("--chain-lengths")
    assert argv[i + 1:i + 3] == ["81", "86"]


@pytest.mark.logic
def test_the_cutoffs_are_forwarded(rev, h):
    h.slot("d0").write().run(rev, pae_cutoff=15.0, contact_cutoff=4.0)
    argv = h.argv("rev_d0_s0")
    assert argv[argv.index("--pae-cutoff") + 1] == "15.0"
    assert argv[argv.index("--contact-cutoff") + 1] == "4.0"


# ── the effector atom filter ─────────────────────────────────────────────────

@pytest.mark.logic
def test_the_cycle_zero_plan_is_passed_as_the_atom_filter(rev, tmp_path):
    """Contact detection on the reverted pose has to use the same effector
    atoms as the steered pose, or the two contact sets are not comparable."""
    h = Harvest(tmp_path / "wd", plan={"effector_interface_atoms": {"B": [1]}})
    h.slot("d0").write().run(rev)
    argv = h.argv("rev_d0_s0")
    assert argv[argv.index("--effector-atom-filter-json") + 1] == str(
        h.root / "plan.json")


@pytest.mark.logic
def test_no_locatable_plan_omits_the_filter(rev, h):
    """Legacy runs and unusual layouts fall back to unfiltered detection
    rather than failing the harvest."""
    h.slot("d0").write().run(rev)
    assert "--effector-atom-filter-json" not in h.argv("rev_d0_s0")


# ── per-seed pairing ─────────────────────────────────────────────────────────

@pytest.mark.logic
def test_each_seed_gets_its_own_record(rev, h):
    """The bug this replaced wrote results[label] once per seed, so only the
    last seed survived and the rest of the GPU work was thrown away."""
    (h.slot("d0", seed_index=0, members=[{"label": "d0_s0",
                                          "seed_index": 0}])
     .slot("d0", seed_index=1, members=[{"label": "d0_s1",
                                         "seed_index": 1}])
     .write())
    res = h.run(rev)
    assert set(res) == {"d0_s0", "d0_s1"}


@pytest.mark.logic
def test_a_steered_label_is_paired_with_its_own_seed(rev, h):
    """Two seeds of the same reverted sequence, each a different pose. The
    wrong pairing scores a design against the other seed's pose."""
    (h.slot("shared", seed_index=0,
            members=[{"label": "a_s0", "seed_index": 0},
                     {"label": "b_s0", "seed_index": 0}], name="rev_s0")
     .slot("shared", seed_index=1,
           members=[{"label": "a_s1", "seed_index": 1},
                    {"label": "b_s1", "seed_index": 1}], name="rev_s1")
     .write())
    # Displace seed 1's effector so the two seeds are distinguishable.
    _translate_chain(h.root / "reversions" / "rev_s1" / "prediction.pdb",
                     "B", (12.0, 0.0, 0.0))
    res = h.run(rev)
    for lbl in ("a_s0", "b_s0"):
        assert res[lbl]["reverted_ra_eff_vs_truth"] == pytest.approx(
            0.0, abs=1e-6)
    for lbl in ("a_s1", "b_s1"):
        assert res[lbl]["reverted_ra_eff_vs_truth"] == pytest.approx(
            12.0, abs=1e-3)


@pytest.mark.logic
def test_a_steered_seed_whose_reverted_seed_is_absent_is_not_paired(rev, h):
    """Pairing is on seed index, so a steered label at seed 7 against a
    reverted slot at seed 0 is not a pair and gets no record. Pairing it
    anyway would score that design against another seed's pose."""
    (h.slot("shared", seed_index=0,
            members=[{"label": "a_s0", "seed_index": 0},
                     {"label": "a_s7", "seed_index": 7}])
     .write())
    res = h.run(rev)
    assert set(res) == {"a_s0"}


@pytest.mark.logic
def test_a_matching_seed_with_no_harvested_record_gets_an_error_stub(rev):
    """A blank would be read as a design with no verdict for an unknown
    reason. The stub names the seed whose harvest is missing. The harvest
    loop always writes a record for its own slot, so this is reachable only
    by calling the pairing step on records from elsewhere."""
    manifest = {"entries": [
        {"label": "shared", "seed_index": 2,
         "all_label_seeds": [{"label": "a_s2", "seed_index": 2}]}]}
    res = rev._pair_per_seed_records_to_steered_labels({}, manifest)
    assert "canonical=shared" in res["a_s2"]["error"]
    assert "seed_index=2" in res["a_s2"]["error"]


@pytest.mark.logic
def test_a_legacy_manifest_without_the_pairing_field_still_harvests(rev, h):
    """Manifests written before the multi-seed fix carry all_labels only."""
    h.slot("d0", members=[]).write()
    entry = h.manifest["entries"][0]
    del entry["all_label_seeds"]
    entry["all_labels"] = ["d0", "d0_shared"]
    (h.root / "reversion_plan.json").write_text(json.dumps(h.manifest))
    res = h.run(rev)
    assert set(res) == {"d0", "d0_shared"}


@pytest.mark.logic
def test_a_legacy_manifest_with_no_labels_at_all_uses_the_canonical(rev, h):
    h.slot("d0", members=[]).write()
    entry = h.manifest["entries"][0]
    del entry["all_label_seeds"]
    (h.root / "reversion_plan.json").write_text(json.dumps(h.manifest))
    assert set(h.run(rev)) == {"d0"}


@pytest.mark.logic
def test_the_shared_records_are_separate_copies(rev, h):
    """Downstream annotates these records per label, so a shared dict would
    let one label's annotation appear on another's."""
    h.slot("shared", members=[{"label": "a", "seed_index": 0},
                              {"label": "b", "seed_index": 0}]).write()
    res = h.run(rev)
    res["a"]["verdict"] = "rescued"
    assert "verdict" not in res["b"]


# ── failures land on the record ──────────────────────────────────────────────

@pytest.mark.logic
def test_a_slot_with_no_prediction_is_an_error_record(rev, h):
    """The GPU task for that slot never finished, or was pre-empted."""
    rec = h.slot("d0", prediction=False).write().run(rev)["d0"]
    assert "no reverted prediction PDB found" in rec["error"]
    assert "reverted_ra_eff_vs_truth" not in rec


@pytest.mark.logic
def test_a_prediction_that_cannot_be_aligned_scores_nan_not_zero(rev, h,
                                                                 capsys):
    """binding_rmsds reports an unusable pair as nan on stdout rather than
    raising, so the record is built and carries nan. That matters because nan
    fails every `< threshold` comparison downstream, which is the safe
    direction, whereas a 0.0 would read as a perfect pose."""
    h.slot("d0").write()
    (Path(h.manifest["entries"][0]["rev_dir"]) / "prediction.pdb").write_text(
        "not a pdb at all\n")
    rec = h.run(rev)["d0"]
    ra = rec["reverted_ra_eff_vs_truth"]
    assert ra != ra                      # nan
    assert rec["reverted_intact"] is False
    assert "receptor chain 'A' not found" in capsys.readouterr().out


@pytest.mark.logic
def test_an_unexpected_alignment_exception_becomes_an_error_record(rev, h,
                                                                   monkeypatch):
    """The nan path above covers the failures binding_rmsds anticipates. The
    guard is for the ones it does not, and it has to drop the one slot rather
    than the whole harvest."""
    import boltz2_negative_steering as bns

    def boom(**kw):
        raise RuntimeError("numpy fell over")

    monkeypatch.setattr(bns, "binding_rmsds", boom)
    rec = h.slot("d0").write().run(rev)["d0"]
    assert "binding_rmsds vs truth failed" in rec["error"]
    assert "numpy fell over" in rec["error"]


@pytest.mark.logic
def test_a_missing_steered_parent_leaves_the_drift_unknown(rev, h, capsys):
    """The rescue question is about the truth, so the record is still usable
    without the steered comparison. A missing parent must not take the slot
    down, and the unknown drift is written as null rather than as a number."""
    rec = h.slot("d0", steered=False).write().run(rev)["d0"]
    assert rec["reverted_ra_eff_vs_steered"] is None
    assert rec["reverted_ra_eff_vs_truth"] == pytest.approx(0.0, abs=1e-6)
    assert "Design PDB not found" in capsys.readouterr().out


@pytest.mark.logic
def test_a_failing_compute_metrics_is_an_error_record(rev, h):
    rec = h.slot("d0", values="FAIL=1").write().run(rev)["d0"]
    assert "compute_metrics.py failed rc=5" in rec["error"]
    assert "stub refused" in rec["error"]


@pytest.mark.logic
def test_one_failed_slot_does_not_stop_the_others(rev, h):
    (h.slot("bad", values="FAIL=1").slot("good").write())
    res = h.run(rev)
    assert "error" in res["bad"]
    assert res["good"]["reverted_iptm"] == 0.8


# ── what is written to disk ──────────────────────────────────────────────────

@pytest.mark.logic
def test_the_results_are_persisted_beside_the_manifest(rev, h):
    """kickoff-finalize reads this file rather than re-running the harvest."""
    h.slot("d0").write().run(rev)
    doc = json.loads((h.root / "reversion_results.json").read_text())
    assert doc["d0"]["reverted_intact"] is True


@pytest.mark.logic
def test_the_per_seed_file_keeps_the_seed_structure(rev, h):
    """The flat file collapses to one record per steered label. The per-seed
    file is what a post-hoc analysis needs to see the spread across seeds."""
    (h.slot("d0", seed_index=0, members=[{"label": "d0_s0", "seed_index": 0}])
     .slot("d0", seed_index=1, members=[{"label": "d0_s1", "seed_index": 1}])
     .write().run(rev))
    doc = json.loads(
        (h.root / "reversion_results_per_seed.json").read_text())
    assert set(doc["d0"]) == {"0", "1"}


# ── the per-seed Jaccard ─────────────────────────────────────────────────────

@pytest.mark.logic
def test_without_a_cycle_zero_interface_the_jaccards_stay_blank(rev, h):
    """Blank rather than zero. Zero is a real Jaccard, meaning the design
    found none of the true interface."""
    rec = h.slot("d0").write().run(rev)["d0"]
    assert rec["reverted_true_jaccard"] == ""
    assert rec["reverted_n_design_interface_residues"] == ""


@pytest.mark.logic
def test_the_jaccard_is_computed_against_the_cycle_zero_interface(rev,
                                                                 tmp_path):
    h = Harvest(tmp_path / "wd", plan={
        "true_interface_idx": list(range(0, 40)),
        "initial_wrong_interface_idx": list(range(60, 70)),
        "pred_receptor_chain": "A", "pred_effector_chain": "B",
        "contact_cutoff": 5.0})
    rec = h.slot("d0").write().run(rev)["d0"]
    assert isinstance(rec["reverted_n_design_interface_residues"], int)
    assert rec["reverted_n_design_interface_residues"] > 0
    assert 0.0 <= rec["reverted_true_jaccard"] <= 1.0
    assert isinstance(rec["reverted_n_shared_true"], int)
    assert 0.0 <= rec["reverted_wrong_jaccard"] <= 1.0


@pytest.mark.logic
def test_surviving_mutations_do_not_block_the_jaccard(rev, tmp_path):
    """The guard used to compare amino acids against cycle_0's wild-type
    receptor sequence. A reverted design differs from it at every surviving
    mutation, so the guard raised on every pose_holds row and the Jaccards
    came back blank. Only the length has to match."""
    h = Harvest(tmp_path / "wd", plan={
        "true_interface_idx": list(range(0, 40)),
        "wild_type_receptor_seq": "Q" * 81,     # right length, wrong residues
        "pred_receptor_chain": "A", "pred_effector_chain": "B"})
    rec = h.slot("d0").write().run(rev)["d0"]
    assert isinstance(rec["reverted_true_jaccard"], float)


@pytest.mark.logic
def test_a_receptor_of_the_wrong_length_blanks_the_jaccard(rev, tmp_path,
                                                           capsys):
    """Index N of the contact set has to mean the same residue as index N of
    cycle_0's interface. A different chain length means they do not, and the
    Jaccard would be a number computed on misaligned indices."""
    h = Harvest(tmp_path / "wd", plan={
        "true_interface_idx": list(range(0, 40)),
        "wild_type_receptor_seq": "Q" * 200,
        "pred_receptor_chain": "A", "pred_effector_chain": "B"})
    rec = h.slot("d0").write().run(rev)["d0"]
    assert rec["reverted_true_jaccard"] == ""
    err = capsys.readouterr().err
    assert "Receptor chain length mismatch" in err


@pytest.mark.logic
def test_an_unreadable_cycle_zero_plan_falls_back_to_the_defaults(rev,
                                                                  tmp_path):
    h = Harvest(tmp_path / "wd")
    (h.root / "plan.json").write_text("{not json")
    rec = h.slot("d0").write().run(rev)["d0"]
    assert rec["reverted_true_jaccard"] == ""
    assert "error" not in rec


@pytest.mark.logic
def test_an_unparseable_contact_cutoff_keeps_the_default(rev, tmp_path):
    h = Harvest(tmp_path / "wd", plan={"contact_cutoff": "wide"})
    ctx = rev._load_cycle0_plan_context(h.root / "plan.json", 5.0)
    assert ctx["contact_cutoff"] == 5.0


# ── locating the reverted prediction ─────────────────────────────────────────

@pytest.mark.logic
def test_a_flat_prediction_is_preferred_over_a_nested_one(rev, tmp_path):
    """prediction.pdb is the copy the engine chose. The nested Boltz outputs
    may include models the engine rejected."""
    d = tmp_path / "rev"
    (d / "predictions" / "x" / "pdb").mkdir(parents=True)
    (d / "predictions" / "x" / "pdb" / "x_model_0.pdb").write_text("")
    (d / "prediction.pdb").write_text("")
    assert rev._locate_reverted_prediction_pdb(d).name == "prediction.pdb"


@pytest.mark.logic
def test_model_zero_is_preferred_when_there_is_no_flat_copy(rev, tmp_path):
    d = tmp_path / "rev"
    d.mkdir()
    (d / "x_model_3.pdb").write_text("")
    (d / "x_model_0.pdb").write_text("")
    assert rev._locate_reverted_prediction_pdb(d).name == "x_model_0.pdb"


@pytest.mark.logic
def test_the_cached_reference_is_not_mistaken_for_a_prediction(rev, tmp_path):
    """reference.pdb is the ground truth copied in for alignment. Scoring it
    against itself would report a perfect rescue for every design."""
    d = tmp_path / "rev"
    d.mkdir()
    (d / "reference.pdb").write_text("")
    assert rev._locate_reverted_prediction_pdb(d) is None


@pytest.mark.logic
def test_msa_files_are_not_mistaken_for_predictions(rev, tmp_path):
    d = tmp_path / "rev"
    (d / "msa").mkdir(parents=True)
    (d / "msa" / "chain_A.pdb").write_text("")
    assert rev._locate_reverted_prediction_pdb(d) is None


# ── chain lengths ────────────────────────────────────────────────────────────

@pytest.mark.logic
def test_the_named_chains_are_counted(rev):
    assert rev._count_pdb_chain_ca_lengths(REFERENCE, "A", "B") == (81, 86)


@pytest.mark.logic
def test_unnamed_chains_fall_back_to_the_two_largest(rev):
    """Not every prediction uses the plan's chain letters."""
    rec, eff = rev._count_pdb_chain_ca_lengths(REFERENCE, "X", "Y")
    assert sorted((rec, eff), reverse=True) == [86, 81]


# ── the remaining-mutation string ────────────────────────────────────────────

@pytest.mark.logic
def test_the_remaining_set_excludes_the_reverted_positions(rev, tmp_path):
    d = tmp_path / "rev"
    d.mkdir()
    (d / "reversion_metadata.json").write_text(json.dumps(
        {"cumulative_mutations": [{"pos1": 44, "wt": "A", "mut": "F"},
                                  {"pos1": 3, "wt": "A", "mut": "W"},
                                  {"pos1": 26, "wt": "A", "mut": "Y"}]}))
    assert rev._build_remaining_mutations_str(d, {3}) == "26,44"


@pytest.mark.logic
def test_a_fully_reverted_design_has_no_remaining_set(rev, tmp_path):
    d = tmp_path / "rev"
    d.mkdir()
    (d / "reversion_metadata.json").write_text(json.dumps(
        {"cumulative_mutations": [{"pos1": 3, "wt": "A", "mut": "W"}]}))
    assert rev._build_remaining_mutations_str(d, {3}) == ""


@pytest.mark.logic
def test_a_cold_start_has_no_remaining_set(rev, tmp_path):
    d = tmp_path / "rev"
    d.mkdir()
    (d / "reversion_metadata.json").write_text("{}")
    assert rev._build_remaining_mutations_str(d, set()) == ""


# ── a timeout on the metrics call ────────────────────────────────────────────

@pytest.mark.logic
def test_a_timeout_is_returned_as_a_message_not_an_exception(rev, tmp_path):
    """The harvest runs unattended inside the container, so a hung slot has
    to become one error record rather than an aborted pass."""
    import subprocess
    import unittest.mock as mock
    with mock.patch.object(
            subprocess, "run",
            side_effect=subprocess.TimeoutExpired(cmd="x", timeout=600)):
        out = rev._run_compute_metrics_for_reverted(
            ["x"], tmp_path / "nope.csv")
    assert out == "compute_metrics.py timed out"


@pytest.mark.logic
def test_a_run_that_writes_no_csv_is_returned_as_a_message(rev, tmp_path):
    script = tmp_path / "quiet.py"
    script.write_text("import sys\nsys.exit(0)\n")
    out = rev._run_compute_metrics_for_reverted(
        [sys.executable, str(script)], tmp_path / "nope.csv")
    assert out == "compute_metrics.py produced no CSV"


@pytest.mark.logic
def test_an_empty_csv_is_returned_as_a_message(rev, tmp_path):
    csv_path = tmp_path / "m.csv"
    csv_path.write_text("model_name,iptm\n")
    script = tmp_path / "quiet.py"
    script.write_text("import sys\nsys.exit(0)\n")
    out = rev._run_compute_metrics_for_reverted(
        [sys.executable, str(script)], csv_path)
    assert out == "compute_metrics.py CSV was empty"


@pytest.mark.logic
def test_an_unreadable_csv_is_returned_as_a_message(rev, tmp_path):
    d = tmp_path / "m.csv"
    d.mkdir()
    script = tmp_path / "quiet.py"
    script.write_text("import sys\nsys.exit(0)\n")
    out = rev._run_compute_metrics_for_reverted(
        [sys.executable, str(script)], d)
    assert out.startswith("could not read")


# ── helpers ──────────────────────────────────────────────────────────────────

def _translate_chain(pdb: Path, chain: str, delta, *, every_other=False):
    """Shift one chain's atoms, rewriting the fixed PDB coordinate columns."""
    out = []
    n = 0
    for line in pdb.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[21] == chain:
            n += 1
            if not (every_other and n % 2):
                xyz = [float(line[30:38]) + delta[0],
                       float(line[38:46]) + delta[1],
                       float(line[46:54]) + delta[2]]
                line = (line[:30] + f"{xyz[0]:8.3f}{xyz[1]:8.3f}"
                        f"{xyz[2]:8.3f}" + line[54:])
        out.append(line)
    pdb.write_text("\n".join(out) + "\n")
