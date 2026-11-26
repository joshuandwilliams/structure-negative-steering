"""The harvest-reversions stage end to end, through its own CLI.

This is where a reversion run becomes a verdict. The stage harvests every
staged reverted prediction, then classifies each one against its steered
parent as pose_holds, pose_collapses or new_contamination. Those three words
are what the thesis reports, so the classification has to be right for the
right reason rather than merely populated.

Two things in between the harvest and the verdict are easy to get wrong and
invisible when they are.

The contamination gating set. A mutated residue contacting the effector on
the reverted pose only counts as new contamination if it sits inside the
design region or the true interface. Positions outside that set already passed
the contamination filter on the steered pose and were kept deliberately, so
flagging them here would contradict the stage that let them through. The set
is built from plan.json's design_region_idx and true_interface_idx, which are
0-based, while reverted_mutated_contact_positions is 1-based. The conversion
is one line and an off-by-one in it silently moves the gate by one residue.

The steered row. classify_reversion_verdict needs the steered pose's score to
compare against, and it comes from contaminated.json rather than from the
harvest. A label missing there is classified against an empty steered row.

compute_metrics.py is stubbed. The structural work runs for real against
data/reference_complexes/6G10.pdb.
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
    reason="needs numpy and biopython (pip install -e '.[runtime]')")


@pytest.fixture(scope="module")
def rev():
    sys.path.insert(0, str(REPO_ROOT / "bin"))
    spec = importlib.util.spec_from_file_location(
        "reversion", REPO_ROOT / "bin" / "reversion.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


STUB = """\
import csv, sys
argv = sys.argv[1:]
out = argv[argv.index("--output-csv") + 1]
pdir = argv[argv.index("--prediction-dir") + 1]
try:
    contacts = open(pdir + "/stub_contacts.txt").read().strip()
except FileNotFoundError:
    contacts = ""
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model_name", "contact_residues",
                                      "n_contact_residues", "iptm"])
    w.writeheader()
    w.writerow({"model_name": "boltz2", "contact_residues": contacts,
                "n_contact_residues": len([c for c in contacts.split(",")
                                           if c]),
                "iptm": "0.8"})
"""


class Harvest:
    """A collect workdir with staged reversions and a cycle-0 plan."""

    def __init__(self, root: Path, *, design_region=(), true_interface=()):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.truth = root / "truth.pdb"
        shutil.copy(REFERENCE, self.truth)
        self.script = root / "stub_cm.py"
        self.script.write_text(STUB)
        (root / "plan.json").write_text(json.dumps({
            "cycle": 0, "ground_truth": str(self.truth),
            "truth_receptor_chain": "A", "truth_effector_chain": "B",
            "pred_receptor_chain": "A", "pred_effector_chain": "B",
            "design_region_idx": list(design_region),
            "true_interface_idx": list(true_interface)}))
        self.manifest = {"pred_receptor_chain": "A",
                         "pred_effector_chain": "B", "entries": []}
        self.contaminated = []

    def slot(self, label, *, contacts="", revert=(3,),
             cumulative=((3, "A", "W"), (26, "A", "Y")), steered_ra=2.0,
             displace=None, in_contaminated=True):
        rev_dir = self.root / "reversions" / f"rev_{label}"
        rev_dir.mkdir(parents=True, exist_ok=True)
        pdb = rev_dir / "prediction.pdb"
        shutil.copy(REFERENCE, pdb)
        if displace is not None:
            _translate_chain(pdb, "B", (float(displace), 0.0, 0.0))
        (rev_dir / "stub_contacts.txt").write_text(contacts)
        (rev_dir / "reversion_metadata.json").write_text(json.dumps(
            {"cumulative_mutations": [{"pos1": p, "wt": w, "mut": m}
                                      for p, w, m in cumulative]}))

        design_workdir = self.root / "designs" / label
        design_workdir.mkdir(parents=True, exist_ok=True)
        shutil.copy(REFERENCE, design_workdir / "prediction.pdb")

        self.manifest["entries"].append({
            "label": label, "seed_index": 0, "rev_dir": str(rev_dir),
            "design_workdir": str(design_workdir),
            "positions_to_revert": list(revert),
            "all_label_seeds": [{"label": label, "seed_index": 0}]})
        if in_contaminated:
            self.contaminated.append({
                "label": label, "steered_ra_eff_vs_truth": steered_ra})
        return self

    def write(self):
        (self.root / "reversion_plan.json").write_text(
            json.dumps(self.manifest))
        (self.root / "contaminated.json").write_text(json.dumps(
            {"n_checked": len(self.manifest["entries"]),
             "n_contaminated": len(self.contaminated),
             "contaminated": self.contaminated}))
        return self

    def run(self, rev, *extra):
        argv = ["harvest-reversions", "--workdir", str(self.root),
                "--compute-metrics-script", str(self.script), *extra]
        args = rev.build_parser().parse_args(argv)
        return rev.cmd_harvest_reversions(args)

    def verdicts(self):
        return json.loads(
            (self.root / "reversion_results.json").read_text())


def _translate_chain(pdb: Path, chain: str, delta):
    out = []
    for line in pdb.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[21] == chain:
            xyz = [float(line[30:38]) + delta[0],
                   float(line[38:46]) + delta[1],
                   float(line[46:54]) + delta[2]]
            line = (line[:30] + f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                    + line[54:])
        out.append(line)
    pdb.write_text("\n".join(out) + "\n")


# ── the three verdicts ───────────────────────────────────────────────────────

@pytest.mark.logic
def test_a_reverted_pose_on_the_truth_with_no_new_contact_holds(rev,
                                                                tmp_path):
    """The rescue worked. The pose the construct will express is the pose
    that was scored."""
    h = Harvest(tmp_path / "wd", true_interface=range(0, 40))
    assert h.slot("d0", contacts="").write().run(rev) == 0
    assert h.verdicts()["d0"]["verdict"] == "pose_holds"


@pytest.mark.logic
def test_a_reverted_pose_that_moved_too_far_collapses(rev, tmp_path):
    """Removing the mutations lost the pose, so the steered result was held
    in place by residues the construct will not carry."""
    h = Harvest(tmp_path / "wd", true_interface=range(0, 40))
    h.slot("d0", displace=20.0).write()
    assert h.run(rev) == 0
    assert h.verdicts()["d0"]["verdict"] == "pose_collapses"


@pytest.mark.logic
def test_a_surviving_mutation_inside_the_gate_is_new_contamination(rev,
                                                                   tmp_path):
    """Position 26 was not reverted, contacts the effector in the reverted
    pose, and sits inside the true interface. So the rescue pose is still
    held by a residue the construct will not carry."""
    h = Harvest(tmp_path / "wd", true_interface=[25])      # 0-based 25 = 26
    h.slot("d0", contacts="26", revert=(3,)).write()
    assert h.run(rev) == 0
    v = h.verdicts()["d0"]
    assert v["verdict"] == "new_contamination"
    assert "26" in v["reason"]


# ── the gating set and its coordinate system ─────────────────────────────────

@pytest.mark.logic
def test_the_gate_is_converted_from_zero_based_to_one_based(rev, tmp_path):
    """plan.json's index sets are 0-based; the reverted contact positions are
    1-based residue numbers. Both sides of this test use the same contact at
    residue 26, and only the plan index moves. The one that names 0-based 25
    gates it in, the one that names 0-based 26 does not."""
    hit = Harvest(tmp_path / "hit", true_interface=[25])
    hit.slot("d0", contacts="26").write().run(rev)
    assert hit.verdicts()["d0"]["verdict"] == "new_contamination"

    miss = Harvest(tmp_path / "miss", true_interface=[26])
    miss.slot("d0", contacts="26").write().run(rev)
    assert miss.verdicts()["d0"]["verdict"] == "pose_holds"


@pytest.mark.logic
def test_a_contact_outside_the_gate_does_not_fire(rev, tmp_path):
    """That position already passed the contamination filter on the steered
    pose and was kept deliberately. Flagging it here would contradict the
    stage that let it through."""
    h = Harvest(tmp_path / "wd", true_interface=range(0, 10))
    h.slot("d0", contacts="26").write().run(rev)
    v = h.verdicts()["d0"]
    assert v["verdict"] == "pose_holds"
    assert "accepted" in v["reason"]


@pytest.mark.logic
def test_the_design_region_is_part_of_the_gate(rev, tmp_path):
    """Not just the true interface. The design region is the stretch being
    validated, so a surviving mutation contacting from inside it invalidates
    the rescue the same way."""
    h = Harvest(tmp_path / "wd", design_region=[25], true_interface=[0])
    h.slot("d0", contacts="26").write().run(rev)
    assert h.verdicts()["d0"]["verdict"] == "new_contamination"


@pytest.mark.logic
def test_the_gate_is_the_union_of_both_sets(rev, tmp_path, capsys):
    h = Harvest(tmp_path / "wd", design_region=[0, 1], true_interface=[1, 2])
    h.slot("d0").write().run(rev)
    assert "contamination gating set: 3 positions" in capsys.readouterr().out


@pytest.mark.logic
def test_a_plan_with_neither_set_falls_back_to_an_ungated_check(rev,
                                                                tmp_path,
                                                                capsys):
    """Legacy runs recorded neither. Ungated means any surviving mutated
    contact fires, which over-reports contamination rather than missing it."""
    h = Harvest(tmp_path / "wd")
    h.slot("d0", contacts="26").write().run(rev)
    assert h.verdicts()["d0"]["verdict"] == "new_contamination"
    assert "legacy ungated contamination check" in capsys.readouterr().out


# ── the steered row ──────────────────────────────────────────────────────────

@pytest.mark.logic
def test_the_steered_score_comes_from_contaminated_json(rev, tmp_path):
    """The harvest knows the reverted pose. The steered pose it is being
    compared against is recorded by build-contaminated, so that file has to
    be read rather than inferred."""
    h = Harvest(tmp_path / "wd", true_interface=range(0, 40))
    h.slot("d0", steered_ra=1.25).write().run(rev)
    assert h.verdicts()["d0"]["verdict"] == "pose_holds"


@pytest.mark.logic
def test_a_label_absent_from_contaminated_json_is_still_classified(rev,
                                                                    tmp_path):
    """With an empty steered row. A verdict on a reverted pose that cannot
    be compared to its parent is still better than no row at all, because
    a missing row reads as a design the harvest never reached."""
    h = Harvest(tmp_path / "wd", true_interface=range(0, 40))
    h.slot("d0", in_contaminated=False).write().run(rev)
    assert h.verdicts()["d0"]["verdict"]


@pytest.mark.logic
def test_a_missing_contaminated_json_does_not_stop_the_harvest(rev, tmp_path):
    h = Harvest(tmp_path / "wd", true_interface=range(0, 40))
    h.slot("d0").write()
    (h.root / "contaminated.json").unlink()
    assert h.run(rev) == 0
    assert "d0" in h.verdicts()


# ── the written results ──────────────────────────────────────────────────────

@pytest.mark.logic
def test_the_verdict_is_written_alongside_the_metrics(rev, tmp_path):
    """kickoff-finalize reads one file for both, so a verdict without its
    metrics could not be audited and metrics without a verdict would be
    dropped for having no outcome."""
    h = Harvest(tmp_path / "wd", true_interface=range(0, 40))
    h.slot("d0").write().run(rev)
    row = h.verdicts()["d0"]
    assert row["verdict"] == "pose_holds"
    assert row["reason"]
    assert row["reverted_iptm"] == 0.8
    assert row["reverted_intact"] is True


@pytest.mark.logic
def test_every_label_gets_a_row(rev, tmp_path):
    h = Harvest(tmp_path / "wd", true_interface=range(0, 40))
    (h.slot("d0").slot("d1", displace=20.0).slot("d2", contacts="26")
     .write().run(rev))
    assert set(h.verdicts()) == {"d0", "d1", "d2"}


@pytest.mark.logic
def test_each_verdict_is_printed(rev, tmp_path, capsys):
    """The stage runs unattended on the cluster, so the log is the only
    record of why a design got the verdict it did."""
    h = Harvest(tmp_path / "wd", true_interface=range(0, 40))
    h.slot("d0").write().run(rev)
    out = capsys.readouterr().out
    assert "harvesting 1 reverted prediction(s)" in out
    assert "verdict=pose_holds" in out


# ── the cutoff ───────────────────────────────────────────────────────────────

@pytest.mark.logic
def test_the_structural_cutoff_is_configurable(rev, tmp_path):
    """The same reverted pose is a hold or a collapse depending on it, which
    is why it is a flag rather than a constant."""
    h = Harvest(tmp_path / "wd", true_interface=range(0, 40))
    h.slot("d0", displace=7.5).write()
    h.run(rev)
    assert h.verdicts()["d0"]["verdict"] == "pose_collapses"
    h.run(rev, "--structural-ra-eff-cutoff", "10")
    assert h.verdicts()["d0"]["verdict"] == "pose_holds"


@pytest.mark.logic
def test_the_cutoff_is_exclusive_at_the_threshold(rev, tmp_path):
    """A pose exactly at the cutoff collapses. Every gate in this pipeline
    is < rather than <=, so a result reported at 5.0 is a failure."""
    h = Harvest(tmp_path / "wd", true_interface=range(0, 40))
    h.slot("d0", displace=5.0).write()
    h.run(rev, "--structural-ra-eff-cutoff", "5.0")
    assert h.verdicts()["d0"]["verdict"] == "pose_collapses"


# ── resolving compute_metrics.py ─────────────────────────────────────────────

@pytest.mark.logic
def test_the_metrics_script_defaults_to_this_script_s_sibling(rev, tmp_path):
    """Nextflow invokes the stage by absolute path with no flags beyond the
    workdir, so the default has to resolve without being told. The sibling
    is where the container stages every bin/ script."""
    h = Harvest(tmp_path / "wd", true_interface=range(0, 40))
    h.slot("d0").write()
    argv = ["harvest-reversions", "--workdir", str(h.root)]
    args = rev.build_parser().parse_args(argv)
    assert args.compute_metrics_script is None
    assert rev.cmd_harvest_reversions(args) == 0
    assert h.verdicts()["d0"]["verdict"] in ("pose_holds", "pose_collapses",
                                             "new_contamination")
