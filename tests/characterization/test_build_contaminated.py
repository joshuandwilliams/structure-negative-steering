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


@pytest.mark.logic
def test_a_later_cycle_with_no_cycle_zero_plan_is_refused(rev, tmp_path,
                                                          capsys):
    root = tmp_path / "run"
    wd = root / "cycle_1" / "pathway_c0d00"
    build(wd, candidates=[candidate("design_00")], cycle=1,
          plan_extra={"experiment_root": str(root)})
    assert rev.cmd_build_contaminated(Args(workdir=wd)) == 2
    assert "cannot resolve chain info" in capsys.readouterr().err


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
