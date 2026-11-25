"""Tests for reversion's build-contaminated stage decision logic.

This stage decides which steered designs get a reversion run. Reversion asks
whether a good pose survives without the mutations that were touching the
effector, so it is only a meaningful question for a design whose steered pose
is already good. Running it on a design whose pose is far out instead asks
whether Boltz happens to find the right answer on a fresh seed, which is a
different question and one the pipeline does not want answered here.

The filter that enforces that, and the refusals around it, are what these
tests cover. The per-candidate loop shells out to compute_metrics.py for
contact residues, which needs structures and a container, so it is not
exercised here.
"""

from __future__ import annotations

import importlib.util
import json
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


class Args:
    def __init__(self, **kw):
        kw.setdefault("compute_metrics_script", None)
        self.__dict__.update(kw)


def candidate(name, *, ra=2.0, idx=0, pdb=None):
    return {"design": name, "design_idx": idx,
            "receptor_aligned_effector_rmsd_vs_truth": ra,
            "pdb": pdb or f"/nonexistent/{name}/prediction.pdb"}


def build(root: Path, *, candidates=(), exhausted=False, rmsd_threshold=5.0,
          plan_extra=None, prefilter_extra=None, cycle=0):
    """A cycle-0 workdir with just the two files this stage reads."""
    root.mkdir(parents=True, exist_ok=True)
    plan = {"cycle": cycle, "rmsd_threshold": rmsd_threshold,
            "pred_receptor_chain": "A", "pred_effector_chain": "B"}
    if plan_extra:
        plan.update(plan_extra)
    (root / "plan.json").write_text(json.dumps(plan))

    pre = {"cycle": cycle, "parent_pathway": None,
           "intact_candidates": list(candidates)}
    if exhausted:
        pre["exhausted"] = True
    if prefilter_extra:
        pre.update(prefilter_extra)
    (root / "prefilter.json").write_text(json.dumps(pre))
    return root


def contaminated(root: Path):
    return json.loads((root / "contaminated.json").read_text())


# ── the refusals ─────────────────────────────────────────────────────────────

@pytest.mark.logic
def test_a_workdir_with_no_plan_is_refused(rev, tmp_path, capsys):
    (tmp_path / "prefilter.json").write_text("{}")
    assert rev.cmd_build_contaminated(Args(workdir=tmp_path)) == 2
    assert "plan.json not found" in capsys.readouterr().err


@pytest.mark.logic
def test_a_workdir_with_no_prefilter_is_refused(rev, tmp_path, capsys):
    (tmp_path / "plan.json").write_text("{}")
    assert rev.cmd_build_contaminated(Args(workdir=tmp_path)) == 2
    assert "prefilter first" in capsys.readouterr().err


@pytest.mark.logic
def test_a_missing_compute_metrics_script_is_refused(rev, tmp_path, capsys):
    """Without it no contact residues can be computed, so every design would
    come back uncontaminated, which reads as nothing needing reversion."""
    build(tmp_path, candidates=[candidate("design_00")])
    assert rev.cmd_build_contaminated(Args(
        workdir=tmp_path,
        compute_metrics_script=tmp_path / "nope.py")) == 2
    assert "compute_metrics.py not found" in capsys.readouterr().err


# ── the exhausted short circuit ──────────────────────────────────────────────

@pytest.mark.logic
def test_an_exhausted_pathway_writes_an_empty_record(rev, tmp_path, capsys):
    """It still has to write the file. A missing contaminated.json reads
    downstream as a stage that never ran."""
    build(tmp_path, exhausted=True)
    assert rev.cmd_build_contaminated(Args(workdir=tmp_path)) == 0
    doc = contaminated(tmp_path)
    assert doc["n_checked"] == 0 and doc["n_contaminated"] == 0
    assert doc["contaminated"] == []
    assert "nothing to do" in capsys.readouterr().out


# ── which candidates are checked ─────────────────────────────────────────────

@pytest.mark.logic
def test_only_candidates_inside_the_threshold_are_checked(rev, tmp_path,
                                                          capsys):
    """Reversion asks whether a good pose survives without its contaminating
    mutations. On a pose that is already far out that question is not the one
    being asked."""
    build(tmp_path, candidates=[candidate("good", ra=2.0),
                                candidate("bad", ra=40.0, idx=1)])
    assert rev.cmd_build_contaminated(Args(workdir=tmp_path)) == 0
    out = capsys.readouterr().out
    assert "2 intact candidates, 1 pass" in out
    assert "1 skipped" in out


@pytest.mark.logic
def test_the_threshold_comes_from_the_plan(rev, tmp_path, capsys):
    build(tmp_path, rmsd_threshold=10.0,
          candidates=[candidate("a", ra=8.0)])
    assert rev.cmd_build_contaminated(Args(workdir=tmp_path)) == 0
    assert "1 pass ra_eff < 10.0" in capsys.readouterr().out


@pytest.mark.logic
@pytest.mark.parametrize("ra,checked", [(4.99, 1), (5.0, 0), (5.01, 0)])
def test_the_threshold_is_exclusive(rev, tmp_path, capsys, ra, checked):
    build(tmp_path, candidates=[candidate("a", ra=ra)])
    assert rev.cmd_build_contaminated(Args(workdir=tmp_path)) == 0
    assert f"{checked} pass" in capsys.readouterr().out


@pytest.mark.logic
def test_an_unparseable_rmsd_is_treated_as_failing(rev, tmp_path, capsys):
    """A missing measurement must not let a design into the reversion pass."""
    build(tmp_path, candidates=[candidate("a", ra="n/a")])
    assert rev.cmd_build_contaminated(Args(workdir=tmp_path)) == 0
    assert "0 pass" in capsys.readouterr().out


@pytest.mark.logic
def test_no_intact_candidates_is_an_empty_record_not_a_failure(rev, tmp_path):
    build(tmp_path, candidates=[])
    assert rev.cmd_build_contaminated(Args(workdir=tmp_path)) == 0
    assert contaminated(tmp_path)["n_contaminated"] == 0


# ── candidates that cannot be checked ────────────────────────────────────────

@pytest.mark.logic
def test_a_candidate_with_no_pdb_on_disk_is_skipped_with_a_warning(
        rev, tmp_path, capsys):
    build(tmp_path, candidates=[candidate("a", pdb="/nope/x.pdb")])
    assert rev.cmd_build_contaminated(Args(workdir=tmp_path)) == 0
    assert "pdb not found" in capsys.readouterr().err
    assert contaminated(tmp_path)["n_contaminated"] == 0


@pytest.mark.logic
@pytest.mark.parametrize("missing", ["design", "pdb"])
def test_a_candidate_missing_a_required_field_is_skipped(rev, tmp_path,
                                                         capsys, missing):
    cand = candidate("a")
    cand[missing] = ""
    build(tmp_path, candidates=[cand])
    assert rev.cmd_build_contaminated(Args(workdir=tmp_path)) == 0
    assert "missing fields" in capsys.readouterr().err


# ── plan-reversions ──────────────────────────────────────────────────────────
#
# This stage stages one reverted sequence per contaminated design. The
# permissive paths matter more than the happy one: every downstream stage
# reads reversion_plan.json, so an absent file is a crash later rather than a
# no-op now. Writing an empty manifest is what lets a run with nothing
# contaminated finish cleanly.

@pytest.mark.logic
def test_plan_reversions_refuses_without_a_plan(rev, tmp_path, capsys):
    assert rev.cmd_plan_reversions(Args(workdir=tmp_path)) == 2
    assert "plan.json not found" in capsys.readouterr().err


@pytest.mark.logic
def test_no_contaminated_file_writes_an_empty_manifest(rev, tmp_path, capsys):
    """build-contaminated may legitimately never have written one."""
    build(tmp_path, candidates=[])
    assert rev.cmd_plan_reversions(Args(workdir=tmp_path)) == 0
    doc = json.loads((tmp_path / "reversion_plan.json").read_text())
    assert doc["n_contaminated"] == 0 and doc["entries"] == []
    assert "no contaminated.json" in capsys.readouterr().out


@pytest.mark.logic
def test_the_chain_names_survive_into_an_empty_manifest(rev, tmp_path):
    """Downstream reads the chains from this file, so an empty manifest that
    dropped them would default to A/B on a run that used other names."""
    build(tmp_path, candidates=[],
          plan_extra={"pred_receptor_chain": "C", "pred_effector_chain": "D"})
    assert rev.cmd_plan_reversions(Args(workdir=tmp_path)) == 0
    doc = json.loads((tmp_path / "reversion_plan.json").read_text())
    assert doc["pred_receptor_chain"] == "C"
    assert doc["pred_effector_chain"] == "D"


@pytest.mark.logic
def test_an_empty_contaminated_list_also_writes_an_empty_manifest(rev,
                                                                  tmp_path,
                                                                  capsys):
    build(tmp_path, candidates=[])
    (tmp_path / "contaminated.json").write_text(json.dumps(
        {"n_checked": 3, "n_contaminated": 0, "contaminated": []}))
    assert rev.cmd_plan_reversions(Args(workdir=tmp_path)) == 0
    assert json.loads((tmp_path / "reversion_plan.json").read_text())[
        "entries"] == []
    assert "no contaminated designs" in capsys.readouterr().out


# ── predict-reversion-one ────────────────────────────────────────────────────

@pytest.mark.logic
def test_predict_one_refuses_without_a_plan(rev, tmp_path, capsys):
    assert rev.cmd_predict_reversion_one(
        Args(workdir=tmp_path, index=0)) == 2
    assert "plan.json not found" in capsys.readouterr().err


@pytest.mark.logic
def test_predict_one_refuses_without_a_manifest(rev, tmp_path, capsys):
    build(tmp_path, candidates=[])
    assert rev.cmd_predict_reversion_one(
        Args(workdir=tmp_path, index=0)) == 2
    assert "run plan-reversions first" in capsys.readouterr().err


@pytest.mark.logic
@pytest.mark.parametrize("idx", [-1, 0, 5])
def test_an_index_with_no_work_exits_cleanly(rev, tmp_path, capsys, idx):
    """These run as SLURM array tasks. A slot with no work must exit 0, or an
    over-provisioned array reports failures for tasks that had nothing to do.
    """
    build(tmp_path, candidates=[])
    (tmp_path / "reversion_plan.json").write_text(json.dumps({"entries": []}))
    assert rev.cmd_predict_reversion_one(
        Args(workdir=tmp_path, index=idx)) == 0
    assert "nothing to do" in capsys.readouterr().out


# ── harvest-reversions ───────────────────────────────────────────────────────

@pytest.mark.logic
def test_harvest_refuses_without_a_plan(rev, tmp_path, capsys):
    assert rev.cmd_harvest_reversions(
        Args(workdir=tmp_path, compute_metrics_script=None)) == 2
    assert "plan.json not found" in capsys.readouterr().err


@pytest.mark.logic
def test_harvest_with_no_manifest_writes_an_empty_results_file(rev, tmp_path,
                                                               capsys):
    """kickoff-finalize reads this file. Without it every intact design would
    be dropped for having no verdict, rather than flowing through as-is."""
    build(tmp_path, candidates=[])
    assert rev.cmd_harvest_reversions(
        Args(workdir=tmp_path, compute_metrics_script=None)) == 0
    assert (tmp_path / "reversion_results.json").read_text().strip() == "{}"
    assert "no reversion_plan.json" in capsys.readouterr().out


@pytest.mark.logic
def test_harvest_refuses_a_missing_compute_metrics_script(rev, tmp_path,
                                                          capsys):
    build(tmp_path, candidates=[])
    (tmp_path / "reversion_plan.json").write_text(json.dumps({"entries": []}))
    assert rev.cmd_harvest_reversions(Args(
        workdir=tmp_path,
        compute_metrics_script=tmp_path / "nope.py")) == 2
    assert "compute_metrics.py not found" in capsys.readouterr().err


# ── the per-candidate loop and the CL-3 majority gate ────────────────────────
#
# The loop shells out to compute_metrics.py once per surviving candidate to
# get the steered pose's contact residues, intersects them with that design's
# cumulative mutation history, and records the intersection as the positions
# reversion must revert.
#
# The gate on top of it is the part worth pinning. Reversion runs for a design
# only when at least half of its correctly-placed seeds are contaminated. The
# pre-Phase-4 behaviour fired on any single contaminated seed, which sent
# designs whose contamination was a one-seed artefact into a reversion pass
# that cost a full prediction each. A below-majority design is left alone, and
# its contaminated seeds still count as failures in n_pass at the aggregate
# level — they are simply not worth a reversion.
#
# The subprocess is real. The loop runs sys.executable itself, so the stub is
# a Python file that reads the contacts it should report out of the prediction
# directory.

STUB_CONTACTS = """\
import csv, sys
argv = sys.argv[1:]
out = argv[argv.index("--output-csv") + 1]
pdir = argv[argv.index("--prediction-dir") + 1]
try:
    contacts = open(pdir + "/stub_contacts.txt").read().strip()
except FileNotFoundError:
    contacts = ""
if contacts == "FAIL":
    sys.stderr.write("stub refused\\n")
    sys.exit(7)
if contacts == "NOCSV":
    sys.exit(0)
if contacts == "EMPTY":
    open(out, "w").write("model_name,contact_residues\\n")
    sys.exit(0)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model_name", "contact_residues",
                                      "n_contact_residues"])
    w.writeheader()
    w.writerow({"model_name": "boltz2", "contact_residues": contacts,
                "n_contact_residues": len([c for c in contacts.split(",")
                                           if c])})
"""


def _pdb(rec=12, eff=7) -> str:
    lines = []
    n = 0
    for chain, count in (("A", rec), ("B", eff)):
        for i in range(1, count + 1):
            n += 1
            lines.append(
                f"ATOM  {n:5d}  CA  ALA {chain}{i:4d}    "
                f"{i:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 80.00\n")
    return "".join(lines)


class Run:
    """A cycle-0 collect workdir with predictions on disk."""

    def __init__(self, root: Path, *, threshold=5.0):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.script = root / "stub_cm.py"
        self.script.write_text(STUB_CONTACTS)
        (root / "plan.json").write_text(json.dumps({
            "cycle": 0, "rmsd_threshold": threshold,
            "pred_receptor_chain": "A", "pred_effector_chain": "B",
            "effector_interface_residues": [1, 2]}))
        self.candidates = []

    def design(self, slot, *, seed=0, ra=2.0, muts=((3, "A", "W"),),
               contacts="3,26", pdb=True, rec=12, eff=7, name=None):
        """One per-seed prediction of one design slot."""
        d = self.root / (name or f"{slot}_s{seed}")
        d.mkdir(parents=True, exist_ok=True)
        if pdb:
            (d / "prediction.pdb").write_text(_pdb(rec, eff))
        (d / "stub_contacts.txt").write_text(contacts)
        if muts is not None:
            rows = "".join(f"{p}\t{w}\t{m}\n" for p, w, m in muts)
            (d / "mutations.tsv").write_text("pos1\twt\tmut\n" + rows)
        self.candidates.append({
            "design": slot, "design_idx": len(self.candidates),
            "seed_index": seed,
            "receptor_aligned_effector_rmsd_vs_truth": ra,
            "pdb": str(d / "prediction.pdb")})
        return self

    def write(self):
        (self.root / "prefilter.json").write_text(json.dumps({
            "cycle": 0, "parent_pathway": None,
            "intact_candidates": self.candidates}))
        return self

    def run(self, rev, **kw):
        args = Args(workdir=self.root, pae_cutoff=10.0, contact_cutoff=5.0,
                    compute_metrics_script=self.script)
        args.__dict__.update(kw)
        return rev.cmd_build_contaminated(args)

    def doc(self):
        return json.loads((self.root / "contaminated.json").read_text())


@pytest.mark.logic
def test_the_positions_to_revert_are_the_mutations_in_contact(rev, tmp_path):
    """Position 3 is both mutated and contacting, so the rescue pose depends
    on a residue the construct will not carry. Position 26 contacts but was
    never mutated, and position 9 was mutated but does not contact."""
    run = (Run(tmp_path / "wd")
           .design("design_00", muts=((3, "A", "W"), (9, "A", "Y")),
                   contacts="3,26").write())
    assert run.run(rev) == 0
    entry = run.doc()["contaminated"][0]
    assert entry["positions_to_revert"] == [3]
    assert entry["steered_contact_residues"] == [3, 26]


@pytest.mark.logic
def test_a_design_with_no_mutation_in_contact_is_not_contaminated(rev,
                                                                   tmp_path):
    run = (Run(tmp_path / "wd")
           .design("design_00", muts=((9, "A", "Y"),), contacts="3,26")
           .write())
    assert run.run(rev) == 0
    assert run.doc()["n_contaminated"] == 0
    assert run.doc()["n_checked"] == 1


@pytest.mark.logic
def test_a_cold_start_with_no_mutations_can_never_be_contaminated(rev,
                                                                  tmp_path):
    run = Run(tmp_path / "wd").design("initial", muts=(),
                                      contacts="3,26").write()
    assert run.run(rev) == 0
    assert run.doc()["n_contaminated"] == 0


@pytest.mark.logic
def test_the_full_mutation_history_travels_with_the_entry(rev, tmp_path):
    """The reversion pass reverts against this history, so a dropped
    mutation would be reverted to the wrong residue or not at all."""
    run = (Run(tmp_path / "wd")
           .design("design_00", muts=((3, "A", "W"), (9, "G", "Y")),
                   contacts="3").write())
    run.run(rev)
    entry = run.doc()["contaminated"][0]
    assert entry["cumulative_mutations"] == [[3, "A", "W"], [9, "G", "Y"]]
    assert entry["steered_ra_eff_vs_truth"] == 2.0
    assert entry["design_workdir"].endswith("design_00_s0")


@pytest.mark.logic
def test_a_cycle_zero_design_keeps_its_flat_label(rev, tmp_path):
    """parse_pathway_label cannot read "design_NN", so cycle 0 reads the
    design's own mutations.tsv rather than walking ancestors it has none of."""
    run = Run(tmp_path / "wd").design("design_07", contacts="3").write()
    run.run(rev)
    assert run.doc()["contaminated"][0]["label"] == "design_07"


# ── the majority gate ────────────────────────────────────────────────────────

@pytest.mark.logic
def test_a_single_contaminated_seed_out_of_three_does_not_trigger(rev,
                                                                  tmp_path):
    """One seed in three is a prediction artefact rather than a property of
    the design, and reverting it costs a full prediction. This is the gate
    that replaced the per-seed trigger."""
    run = (Run(tmp_path / "wd")
           .design("design_00", seed=0, contacts="3")
           .design("design_00", seed=1, contacts="26")
           .design("design_00", seed=2, contacts="26")
           .write())
    assert run.run(rev) == 0
    assert run.doc()["n_contaminated"] == 0


@pytest.mark.logic
def test_two_contaminated_seeds_out_of_three_trigger(rev, tmp_path):
    """ceil(3/2) is 2, so two is the majority."""
    run = (Run(tmp_path / "wd")
           .design("design_00", seed=0, contacts="3")
           .design("design_00", seed=1, contacts="3")
           .design("design_00", seed=2, contacts="26")
           .write())
    run.run(rev)
    assert run.doc()["n_contaminated"] == 2


@pytest.mark.logic
def test_only_the_contaminated_seeds_are_queued(rev, tmp_path):
    """The clean seeds of a triggering design need no reversion, so staging
    them would spend a prediction on a question already answered."""
    run = (Run(tmp_path / "wd")
           .design("design_00", seed=0, contacts="3")
           .design("design_00", seed=1, contacts="3")
           .design("design_00", seed=2, contacts="26")
           .write())
    run.run(rev)
    assert {e["seed_index"] for e in run.doc()["contaminated"]} == {0, 1}


@pytest.mark.logic
def test_one_contaminated_seed_out_of_two_triggers(rev, tmp_path):
    """ceil(2/2) is 1, so an even split counts as a majority. Half the seeds
    of a design depending on a residue that will not exist is enough."""
    run = (Run(tmp_path / "wd")
           .design("design_00", seed=0, contacts="3")
           .design("design_00", seed=1, contacts="26")
           .write())
    run.run(rev)
    assert run.doc()["n_contaminated"] == 1


@pytest.mark.logic
def test_a_single_seed_design_triggers_on_that_seed(rev, tmp_path):
    """ceil(1/2) is 1. Most runs are single-seed, so this is the common
    case and the gate must not suppress it."""
    run = Run(tmp_path / "wd").design("design_00", contacts="3").write()
    run.run(rev)
    assert run.doc()["n_contaminated"] == 1


@pytest.mark.logic
def test_each_design_slot_is_gated_separately(rev, tmp_path):
    """The grouping key is the slot, not the run. One design's clean seeds
    must not raise another design's threshold."""
    run = (Run(tmp_path / "wd")
           .design("design_00", seed=0, contacts="3")
           .design("design_00", seed=1, contacts="3")
           .design("design_01", seed=0, contacts="26")
           .design("design_01", seed=1, contacts="26")
           .write())
    run.run(rev)
    labels = {e["label"] for e in run.doc()["contaminated"]}
    assert labels == {"design_00"}


@pytest.mark.logic
def test_the_gating_decision_is_reported(rev, tmp_path, capsys):
    run = (Run(tmp_path / "wd")
           .design("design_00", contacts="3")
           .design("design_01", contacts="26")
           .write())
    run.run(rev)
    assert "1 / 2 designs trigger reversion" in capsys.readouterr().out


# ── candidates the loop cannot evaluate ──────────────────────────────────────
#
# Each of these drops one candidate and keeps going, which also leaves
# n_checked short of the candidate count and so keeps the scratch directory
# for post-mortem.

@pytest.mark.logic
def test_a_candidate_whose_prediction_is_gone_is_skipped(rev, tmp_path,
                                                          capsys):
    run = Run(tmp_path / "wd").design("design_00", pdb=False).write()
    assert run.run(rev) == 0
    assert run.doc()["n_contaminated"] == 0
    assert "pdb not found" in capsys.readouterr().err


@pytest.mark.logic
def test_n_checked_counts_candidates_entered_not_evaluated(rev, tmp_path):
    """It is the count that passed the ra_eff gate, so a candidate the loop
    could not evaluate still appears in it. Reading it as an evaluated count
    would overstate how much of the run was actually checked."""
    run = (Run(tmp_path / "wd")
           .design("design_00", pdb=False)
           .design("design_01", ra=40.0)
           .design("design_02", contacts="3")
           .write())
    run.run(rev)
    assert run.doc()["n_checked"] == 2


@pytest.mark.logic
def test_a_candidate_with_no_design_name_is_skipped(rev, tmp_path, capsys):
    run = Run(tmp_path / "wd").design("design_00", contacts="3").write()
    run.candidates[0]["design"] = ""
    run.write()
    assert run.run(rev) == 0
    assert run.doc()["n_contaminated"] == 0
    assert "missing fields" in capsys.readouterr().err


@pytest.mark.logic
def test_a_single_chain_prediction_is_skipped(rev, tmp_path, capsys):
    """One chain means there is no interface, so there are no contacts to
    intersect the mutations with."""
    run = Run(tmp_path / "wd").design("design_00", eff=0,
                                      contacts="3").write()
    assert run.run(rev) == 0
    assert run.doc()["n_contaminated"] == 0
    assert "could not determine chain lengths" in capsys.readouterr().err


@pytest.mark.logic
def test_unnamed_chains_fall_back_to_the_two_largest(rev, tmp_path):
    """A prediction that did not use the plan's chain letters is still
    evaluable, because only the two lengths are needed."""
    run = Run(tmp_path / "wd").design("design_00", contacts="3").write()
    (run.root / "plan.json").write_text(json.dumps({
        "cycle": 0, "rmsd_threshold": 5.0,
        "pred_receptor_chain": "X", "pred_effector_chain": "Y"}))
    assert run.run(rev) == 0
    assert run.doc()["n_contaminated"] == 1


@pytest.mark.logic
@pytest.mark.parametrize("marker,message", [
    ("FAIL", "rc=7"),
    ("NOCSV", "produced no CSV"),
    ("EMPTY", "CSV was empty"),
])
def test_a_metrics_failure_drops_that_candidate_only(rev, tmp_path, capsys,
                                                      marker, message):
    run = (Run(tmp_path / "wd")
           .design("design_00", contacts=marker)
           .design("design_01", contacts="3")
           .write())
    assert run.run(rev) == 0
    assert message in capsys.readouterr().err
    assert {e["label"] for e in run.doc()["contaminated"]} == {"design_01"}


@pytest.mark.logic
def test_the_scratch_directory_is_kept_when_a_candidate_was_dropped(rev,
                                                                    tmp_path):
    """The per-candidate CSVs are the only record of what compute_metrics.py
    saw for the candidate that failed."""
    run = Run(tmp_path / "wd").design("design_00", contacts="FAIL").write()
    run.run(rev)
    assert (run.root / "contamination_scratch").is_dir()


@pytest.mark.logic
def test_the_scratch_directory_is_removed_on_a_clean_run(rev, tmp_path):
    run = Run(tmp_path / "wd").design("design_00", contacts="3").write()
    run.run(rev)
    assert not (run.root / "contamination_scratch").exists()


# ── what reaches compute_metrics.py ──────────────────────────────────────────

@pytest.mark.logic
def test_the_mutated_positions_are_forwarded(rev, tmp_path):
    """Without them the contamination columns come back blank and every
    design reads as clean."""
    STUB_ARGV = STUB_CONTACTS.replace(
        'with open(out, "w", newline="") as f:',
        'import json\n'
        'json.dump(argv, open(pdir + "/argv.json", "w"))\n'
        'with open(out, "w", newline="") as f:')
    run = (Run(tmp_path / "wd")
           .design("design_00", muts=((3, "A", "W"), (9, "G", "Y")),
                   contacts="3").write())
    run.script.write_text(STUB_ARGV)
    run.run(rev)
    argv = json.loads(
        (run.root / "design_00_s0" / "argv.json").read_text())
    assert argv[argv.index("--mutated-positions") + 1] == "3,9"
    assert argv[argv.index("--pae-cutoff") + 1] == "10.0"
    assert argv[argv.index("--contact-cutoff") + 1] == "5.0"
    assert argv[argv.index("--effector-atom-filter-json") + 1] == str(
        run.root / "plan.json")


# ── failures inside the per-candidate loop ───────────────────────────────────
#
# Each of these drops one candidate and keeps going. The loop is the only
# place a design's mutation history is read, so a design whose history cannot
# be read has no mutated positions, and the contamination check then has
# nothing to intersect the contact set against.

@pytest.mark.logic
def test_an_empty_mutations_file_is_a_warning_not_a_crash(rev, tmp_path,
                                                           capsys):
    """read_mutations_tsv_full skips the header with next(f), which raises
    StopIteration on a file that exists and is empty. That is what a task
    killed between creating the file and writing its header leaves behind.
    Without the guard the whole stage would stop on one such design."""
    run = Run(tmp_path / "wd").design("design_00", contacts="3").write()
    (run.root / "design_00_s0" / "mutations.tsv").write_text("")
    assert run.run(rev) == 0
    assert "read_mutations_tsv_full failed" in capsys.readouterr().err
    assert run.doc()["n_contaminated"] == 0


@pytest.mark.logic
def test_a_design_with_no_readable_history_is_treated_as_unmutated(rev,
                                                                    tmp_path):
    """Not as contaminated. An unreadable history means the positions are
    unknown, and guessing them would revert residues the design may not
    carry."""
    run = Run(tmp_path / "wd").design("design_00", contacts="3").write()
    (run.root / "design_00_s0" / "mutations.tsv").write_text("")
    run.run(rev)
    assert run.doc()["contaminated"] == []


@pytest.mark.logic
def test_a_metrics_call_that_hangs_drops_that_candidate(rev, tmp_path,
                                                         capsys, monkeypatch):
    """The call is capped at 600 seconds. One design that hangs must not
    hold the stage past its own walltime, because the candidates after it
    would then never be checked at all."""
    import subprocess as sp

    real = sp.run

    def timing_out(cmd, *a, **kw):
        if any("stub_cm" in str(c) for c in cmd):
            raise sp.TimeoutExpired(cmd=cmd, timeout=600)
        return real(cmd, *a, **kw)

    monkeypatch.setattr(sp, "run", timing_out)
    run = (Run(tmp_path / "wd")
           .design("design_00", contacts="3")
           .design("design_01", contacts="3")
           .write())
    assert run.run(rev) == 0
    assert "timed out" in capsys.readouterr().err
    assert run.doc()["n_contaminated"] == 0


@pytest.mark.logic
def test_a_scratch_directory_that_cannot_be_removed_is_left_alone(
        rev, tmp_path, monkeypatch):
    """Tidying up is not worth failing a stage that has already produced its
    answer. contaminated.json is written before the cleanup."""
    import shutil

    def boom(*a, **kw):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(shutil, "rmtree", boom)
    run = Run(tmp_path / "wd").design("design_00", contacts="3").write()
    assert run.run(rev) == 0
    assert run.doc()["n_contaminated"] == 1
    assert (run.root / "contamination_scratch").is_dir()
