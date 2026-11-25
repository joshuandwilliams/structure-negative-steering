"""Tests for bin/negsteer_coldstart.py's helpers and its first two stages.

Cold start decides which unsteered predictions survive into steering. Two
things here are load-bearing and quiet.

The intact filter drops a candidate whose own chain RMSD says it misfolded.
Letting a misfolded prediction through means steering is scored against a
structure that was never a complex, and the number it produces looks ordinary.

maximin_select picks which candidates go forward, and its tie-break reads
`_active_ra_eff_for_selection` where reversion has set one, falling back to the
steered value. Reading the steered value on a reverted row selects on a pose
that was discarded, and nothing downstream can tell.

`kickoff-distances` calls into boltz2_negative_steering for the pose distance,
which needs numpy and Bio.Align inside the container. That one call is stubbed.
Everything around it is stdlib and runs here, including the two refusals that
exist because a missing upstream file otherwise reads as "no candidates".
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE = REPO_ROOT / "bin" / "negsteer_coldstart.py"


@pytest.fixture(scope="module")
def cs():
    sys.path.insert(0, str(REPO_ROOT / "bin"))
    spec = importlib.util.spec_from_file_location("negsteer_coldstart", MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


# ── maximin selection ────────────────────────────────────────────────────────

@pytest.mark.logic
def test_the_most_distant_candidates_win(cs):
    cands = [{"min_dist_to_upstream": d, "receptor_aligned_effector_rmsd_vs_truth": 1.0,
              "design": f"d{d}"} for d in (1.0, 5.0, 3.0)]
    assert [c["design"] for c in cs.maximin_select(cands, 2)] == ["d5.0", "d3.0"]


@pytest.mark.logic
def test_a_tie_on_distance_breaks_on_lower_ra_eff(cs):
    cands = [
        {"min_dist_to_upstream": 2.0, "receptor_aligned_effector_rmsd_vs_truth": 9.0,
         "design": "far"},
        {"min_dist_to_upstream": 2.0, "receptor_aligned_effector_rmsd_vs_truth": 1.0,
         "design": "near"},
    ]
    assert [c["design"] for c in cs.maximin_select(cands, 2)] == ["near", "far"]


@pytest.mark.logic
def test_the_tie_break_prefers_the_post_reversion_value(cs):
    """A pose_holds row carries _active_ra_eff_for_selection. Selecting on the
    steered value instead would rank on a pose that was discarded."""
    cands = [
        {"min_dist_to_upstream": 2.0, "design": "reverted",
         "receptor_aligned_effector_rmsd_vs_truth": 9.0,
         "_active_ra_eff_for_selection": 1.0},
        {"min_dist_to_upstream": 2.0, "design": "plain",
         "receptor_aligned_effector_rmsd_vs_truth": 5.0},
    ]
    assert [c["design"] for c in cs.maximin_select(cands, 2)] == ["reverted", "plain"]


@pytest.mark.logic
def test_a_missing_or_unusable_ra_eff_sorts_last(cs):
    cands = [
        {"min_dist_to_upstream": 1.0, "design": "none"},
        {"min_dist_to_upstream": 1.0, "design": "bad",
         "receptor_aligned_effector_rmsd_vs_truth": "not-a-number"},
        {"min_dist_to_upstream": 1.0, "design": "good",
         "receptor_aligned_effector_rmsd_vs_truth": 2.0},
    ]
    assert cs.maximin_select(cands, 1)[0]["design"] == "good"


@pytest.mark.logic
def test_asking_for_more_than_there_are_returns_them_all(cs):
    cands = [{"min_dist_to_upstream": 1.0, "design": "a"}]
    assert len(cs.maximin_select(cands, 5)) == 1
    assert cs.maximin_select([], 3) == []


# ── small helpers ────────────────────────────────────────────────────────────

@pytest.mark.logic
def test_is_nan_handles_non_numbers_without_raising(cs):
    assert cs._is_nan(float("nan")) is True
    assert cs._is_nan(1.0) is False
    assert cs._is_nan(None) is False
    assert cs._is_nan("x") is False


@pytest.mark.logic
def test_reverted_confidence_copies_only_what_is_present(cs):
    """A blank stays blank. Writing a None would turn a missing metric into a
    value the aggregator then reports."""
    field = cs._REVERTED_CONFIDENCE_FIELDS[0]
    dst = {}
    cs._copy_reverted_confidence(dst, {field: 0.5, "unrelated": 1})
    assert dst == {field: 0.5}
    cs._copy_reverted_confidence(dst, {field: None})
    assert dst == {field: 0.5}


@pytest.mark.logic
def test_cycle_statistics_writes_a_header_once_then_appends(cs, tmp_path):
    for n in (1, 2):
        cs.append_cycle_statistics(tmp_path, 0, "c0d00", n, n, n, n,
                                   [1.0, 2.0], [3.0])
    lines = (tmp_path / "cycle_statistics.csv").read_text().splitlines()
    assert lines[0].startswith("cycle,pathway")
    assert len(lines) == 3
    assert "1.000;2.000" in lines[1]


# ── kickoff-distances ────────────────────────────────────────────────────────

def _cycle0(tmp_path, plan: dict, *, with_pdb=True) -> Path:
    c0 = tmp_path / "cycle_0"
    c0.mkdir(parents=True, exist_ok=True)
    (c0 / "plan.json").write_text(json.dumps(plan))
    if with_pdb:
        (c0 / "initial_prediction.pdb").write_text("ATOM\n")
    return c0


def _design(c0: Path, idx: int, *, status="ok", ra=2.0, rec=1.0, eff=1.0,
            result_extra=None) -> dict:
    d = c0 / f"design_{idx:02d}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "prediction.pdb").write_text("ATOM\n")
    r = {"design": f"design_{idx:02d}", "status": status,
         "receptor_aligned_effector_rmsd": ra,
         "independent_receptor_rmsd": rec,
         "independent_effector_rmsd": eff}
    if result_extra:
        r.update(result_extra)
    (d / "result.json").write_text(json.dumps(r))
    return {"dir": str(d), "index": idx, "name": f"design_{idx:02d}"}


@pytest.mark.logic
def test_distances_refuses_without_a_cycle_zero(cs, tmp_path, capsys):
    rc = cs.cmd_kickoff_distances(Args(experiment_root=tmp_path))
    assert rc == 2
    assert "no cycle_0" in capsys.readouterr().err


@pytest.mark.logic
def test_distances_refuses_without_a_plan(cs, tmp_path, capsys):
    (tmp_path / "cycle_0").mkdir()
    assert cs.cmd_kickoff_distances(Args(experiment_root=tmp_path)) == 2
    assert "no plan.json" in capsys.readouterr().err


@pytest.mark.logic
def test_distances_refuses_without_the_initial_prediction(cs, tmp_path, capsys):
    _cycle0(tmp_path, {"pred_receptor_chain": "A", "pred_effector_chain": "B",
                       "designs": []}, with_pdb=False)
    assert cs.cmd_kickoff_distances(Args(experiment_root=tmp_path)) == 2
    assert "missing" in capsys.readouterr().err


@pytest.mark.logic
def test_a_skipped_cold_start_writes_an_empty_distances_file(cs, tmp_path):
    """The cold start already passed, so there is nothing to steer away from."""
    c0 = _cycle0(tmp_path, {"skip_steering": True})
    assert cs.cmd_kickoff_distances(Args(experiment_root=tmp_path)) == 0
    doc = json.loads((c0 / "kickoff_distances.json").read_text())
    assert doc == {"skip_steering": True, "n_generated": 0, "candidates": []}


@pytest.mark.logic
def test_distances_scores_each_design_against_the_cold_start(cs, tmp_path,
                                                             monkeypatch):
    c0 = _cycle0(tmp_path, {})
    designs = [_design(c0, 0, ra=2.0), _design(c0, 1, ra=7.0)]
    (c0 / "plan.json").write_text(json.dumps(
        {"pred_receptor_chain": "A", "pred_effector_chain": "B",
         "designs": designs}))
    monkeypatch.setattr(cs, "compute_pose_distance",
                        lambda *a, **k: 4.25)
    assert cs.cmd_kickoff_distances(Args(experiment_root=tmp_path)) == 0
    doc = json.loads((c0 / "kickoff_distances.json").read_text())
    assert doc["n_generated"] == 2
    assert [c["min_dist_to_upstream"] for c in doc["candidates"]] == [4.25, 4.25]
    assert doc["candidates"][0]["receptor_aligned_effector_rmsd_vs_truth"] == 2.0
    # The whole result.json rides along for the aggregator.
    assert doc["candidates"][0]["result"]["design"] == "design_00"


@pytest.mark.logic
@pytest.mark.parametrize("broken", [
    {"status": "failed"},
    {"ra": None}, {"rec": None}, {"eff": None},
])
def test_a_design_without_usable_metrics_is_dropped(cs, tmp_path, monkeypatch,
                                                    broken):
    c0 = _cycle0(tmp_path, {})
    designs = [_design(c0, 0, **broken)]
    (c0 / "plan.json").write_text(json.dumps(
        {"pred_receptor_chain": "A", "pred_effector_chain": "B",
         "designs": designs}))
    monkeypatch.setattr(cs, "compute_pose_distance", lambda *a, **k: 1.0)
    assert cs.cmd_kickoff_distances(Args(experiment_root=tmp_path)) == 0
    doc = json.loads((c0 / "kickoff_distances.json").read_text())
    assert doc["candidates"] == []
    assert doc["n_generated"] == 1          # counted, but not carried forward


@pytest.mark.logic
def test_a_design_with_no_result_json_is_dropped(cs, tmp_path, monkeypatch):
    c0 = _cycle0(tmp_path, {})
    d = c0 / "design_00"
    d.mkdir()
    (c0 / "plan.json").write_text(json.dumps(
        {"pred_receptor_chain": "A", "pred_effector_chain": "B",
         "designs": [{"dir": str(d), "index": 0, "name": "design_00"}]}))
    monkeypatch.setattr(cs, "compute_pose_distance", lambda *a, **k: 1.0)
    assert cs.cmd_kickoff_distances(Args(experiment_root=tmp_path)) == 0
    assert json.loads((c0 / "kickoff_distances.json").read_text())["candidates"] == []


@pytest.mark.logic
def test_a_failed_distance_becomes_null_rather_than_killing_the_stage(
        cs, tmp_path, monkeypatch, capsys):
    c0 = _cycle0(tmp_path, {})
    designs = [_design(c0, 0)]
    (c0 / "plan.json").write_text(json.dumps(
        {"pred_receptor_chain": "A", "pred_effector_chain": "B",
         "designs": designs}))

    def boom(*a, **k):
        raise RuntimeError("no numpy")
    monkeypatch.setattr(cs, "compute_pose_distance", boom)
    assert cs.cmd_kickoff_distances(Args(experiment_root=tmp_path)) == 0
    doc = json.loads((c0 / "kickoff_distances.json").read_text())
    assert doc["candidates"][0]["min_dist_to_upstream"] is None
    assert "pose distance failed" in capsys.readouterr().err


@pytest.mark.logic
def test_a_nan_distance_is_written_as_null(cs, tmp_path, monkeypatch):
    """JSON has no NaN, so writing one produces a file json.load rejects."""
    c0 = _cycle0(tmp_path, {})
    designs = [_design(c0, 0)]
    (c0 / "plan.json").write_text(json.dumps(
        {"pred_receptor_chain": "A", "pred_effector_chain": "B",
         "designs": designs}))
    monkeypatch.setattr(cs, "compute_pose_distance", lambda *a, **k: float("nan"))
    assert cs.cmd_kickoff_distances(Args(experiment_root=tmp_path)) == 0
    raw = (c0 / "kickoff_distances.json").read_text()
    assert "NaN" not in raw
    assert json.loads(raw)["candidates"][0]["min_dist_to_upstream"] is None


# ── kickoff-prefilter ────────────────────────────────────────────────────────

def _distances(c0: Path, candidates, **extra):
    doc = {"n_generated": len(candidates), "candidates": candidates}
    doc.update(extra)
    (c0 / "kickoff_distances.json").write_text(json.dumps(doc))


def _cand(rec, eff, name="d"):
    return {"design": name, "independent_receptor_rmsd": rec,
            "independent_effector_rmsd": eff, "min_dist_to_upstream": 1.0}


@pytest.mark.logic
def test_prefilter_refuses_without_a_cycle_zero(cs, tmp_path, capsys):
    assert cs.cmd_kickoff_prefilter(Args(experiment_root=tmp_path)) == 2
    assert "no cycle_0" in capsys.readouterr().err


@pytest.mark.logic
def test_prefilter_records_why_it_could_not_run(cs, tmp_path, capsys):
    """Phase 1 failing must not read downstream as 'no candidates'."""
    c0 = tmp_path / "cycle_0"
    c0.mkdir()
    assert cs.cmd_kickoff_prefilter(Args(experiment_root=tmp_path)) == 1
    assert "error" in json.loads((c0 / "prefilter.json").read_text())
    assert "kickoff-distances" in capsys.readouterr().err


@pytest.mark.logic
def test_prefilter_carries_a_skipped_cold_start_through(cs, tmp_path):
    c0 = tmp_path / "cycle_0"
    c0.mkdir()
    _distances(c0, [], skip_steering=True)
    assert cs.cmd_kickoff_prefilter(Args(experiment_root=tmp_path)) == 0
    doc = json.loads((c0 / "prefilter.json").read_text())
    assert doc["skip_steering"] is True and doc["exhausted"] is True
    assert doc["n_intact"] == 0


@pytest.mark.logic
def test_a_misfolded_candidate_is_filtered_out(cs, tmp_path):
    """Either chain over the threshold means the prediction is not a complex,
    so scoring steering against it would be scoring against nothing."""
    c0 = tmp_path / "cycle_0"
    c0.mkdir()
    over = cs._INTACT_THRESHOLD + 1.0
    _distances(c0, [_cand(1.0, 1.0, "intact"),
                    _cand(over, 1.0, "rec_broken"),
                    _cand(1.0, over, "eff_broken")])
    assert cs.cmd_kickoff_prefilter(Args(experiment_root=tmp_path)) == 0
    doc = json.loads((c0 / "prefilter.json").read_text())
    assert doc["n_generated"] == 3
    assert [c["design"] for c in doc["intact_candidates"]] == ["intact"]
    assert doc["intact_candidates"][0]["receptor_intact"] is True


@pytest.mark.logic
def test_a_candidate_exactly_on_the_threshold_is_kept(cs, tmp_path):
    c0 = tmp_path / "cycle_0"
    c0.mkdir()
    t = cs._INTACT_THRESHOLD
    _distances(c0, [_cand(t, t, "edge")])
    assert cs.cmd_kickoff_prefilter(Args(experiment_root=tmp_path)) == 0
    doc = json.loads((c0 / "prefilter.json").read_text())
    assert [c["design"] for c in doc["intact_candidates"]] == ["edge"]


# ── the CLI surface ──────────────────────────────────────────────────────────

@pytest.mark.logic
def test_every_stage_is_reachable_from_the_parser(cs):
    """The three stage names are what bin/negative_steering_run_one.sh calls."""
    parser = cs.build_parser()
    cases = {
        "kickoff-distances": [],
        "kickoff-prefilter": [],
        "kickoff-finalize": ["--max-cycles", "0"],
    }
    for stage, extra in cases.items():
        args = parser.parse_args([stage, "--experiment-root", "/tmp/x"] + extra)
        assert callable(args.func)


@pytest.mark.logic
def test_main_dispatches_to_the_named_stage(cs, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv",
                        ["negsteer_coldstart.py", "kickoff-prefilter",
                         "--experiment-root", str(tmp_path)])
    assert cs.main() == 2        # no cycle_0, so the stage refuses


# ── kickoff-finalize ─────────────────────────────────────────────────────────
#
# Verdict routing. Every design that was flagged contaminated has to end up
# either kept with its reverted sequence in place, or dropped with a verdict
# recorded. The failure this guards against is a contaminated design with no
# verdict being kept, which puts an unvalidated sequence into the passing set
# and nothing downstream can tell it apart from a validated one.

def _finalize_args(root, **kw):
    base = dict(experiment_root=root, novelty_cutoff=0.0, max_passing=10,
                max_cycles=0)
    base.update(kw)
    return Args(**base)


def _prefilter(c0: Path, candidates, **extra):
    doc = {"cycle": 0, "parent_pathway": None, "n_generated": len(candidates),
           "n_intact": len(candidates), "intact_candidates": candidates}
    doc.update(extra)
    (c0 / "prefilter.json").write_text(json.dumps(doc))


def _icand(c0: Path, name, *, ra=2.0, dist=5.0):
    d = c0 / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "prediction.pdb").write_text("ATOM\n")
    return {"design": name, "pdb": str(d / "prediction.pdb"),
            "receptor_aligned_effector_rmsd_vs_truth": ra,
            "independent_receptor_rmsd": 1.0, "independent_effector_rmsd": 1.0,
            "min_dist_to_upstream": dist}


@pytest.mark.logic
def test_finalize_refuses_without_a_cycle_zero(cs, tmp_path, capsys):
    assert cs.cmd_kickoff_finalize(_finalize_args(tmp_path)) == 2
    assert "no cycle_0" in capsys.readouterr().err


@pytest.mark.logic
def test_finalize_refuses_without_a_prefilter(cs, tmp_path, capsys):
    (tmp_path / "cycle_0").mkdir()
    assert cs.cmd_kickoff_finalize(_finalize_args(tmp_path)) == 1
    assert "kickoff-prefilter first" in capsys.readouterr().err


@pytest.mark.logic
@pytest.mark.parametrize("flag", ["exhausted", "skip_steering"])
def test_an_exhausted_cycle_stops_without_writing_passing(cs, tmp_path, flag):
    c0 = tmp_path / "cycle_0"
    c0.mkdir()
    _prefilter(c0, [], **{flag: True})
    assert cs.cmd_kickoff_finalize(_finalize_args(tmp_path)) == 0
    assert not (c0 / "passing.json").exists()


@pytest.mark.logic
def test_an_uncontaminated_design_passes_straight_through(cs, tmp_path):
    c0 = tmp_path / "cycle_0"
    c0.mkdir()
    _prefilter(c0, [_icand(c0, "design_00")])
    assert cs.cmd_kickoff_finalize(_finalize_args(tmp_path)) == 0
    doc = json.loads((c0 / "passing.json").read_text())
    assert [c["design"] for c in doc["selected"]] == ["design_00"]
    assert doc["dropped_labels"] == []


@pytest.mark.logic
def test_a_contaminated_design_with_no_verdict_is_dropped(cs, tmp_path, capsys):
    """Keeping it would put an unvalidated sequence into the passing set."""
    c0 = tmp_path / "cycle_0"
    c0.mkdir()
    _prefilter(c0, [_icand(c0, "design_00")])
    (c0 / "contaminated.json").write_text(
        json.dumps({"contaminated": [{"label": "design_00"}]}))
    assert cs.cmd_kickoff_finalize(_finalize_args(tmp_path)) == 0
    doc = json.loads((c0 / "passing.json").read_text())
    assert doc["selected"] == []
    assert doc["dropped_labels"] == ["design_00"]
    assert doc["all_candidates"][0]["reversion_verdict"] == "missing_verdict"
    assert "no verdict" in capsys.readouterr().err


@pytest.mark.logic
def test_pose_holds_rewrites_the_receptor_and_keeps_the_design(cs, tmp_path):
    """The kept design must carry the reverted sequence, not the steered one."""
    c0 = tmp_path / "cycle_0"
    c0.mkdir()
    cand = _icand(c0, "design_00")
    _prefilter(c0, [cand])
    fasta = c0 / "reverted.fasta"
    fasta.write_text(">x\nMKVL\nAAGG\n")
    (c0 / "reversion_plan.json").write_text(json.dumps(
        {"entries": [{"label": "design_00",
                      "reverted_receptor_fasta": str(fasta)}]}))
    (c0 / "reversion_results.json").write_text(json.dumps(
        {"design_00": {"verdict": "pose_holds",
                       "reverted_ra_eff_vs_truth": 1.5,
                       "reverted_independent_receptor_rmsd": 1.0,
                       "reverted_independent_effector_rmsd": 1.0}}))
    assert cs.cmd_kickoff_finalize(_finalize_args(tmp_path)) == 0

    design_dir = Path(cand["pdb"]).parent
    assert (design_dir / "receptor.fasta").read_text() == \
        ">receptor_reverted\nMKVLAAGG\n"
    applied = json.loads((design_dir / "reversion_applied.json").read_text())
    assert applied["reverted_sequence"] == "MKVLAAGG"

    doc = json.loads((c0 / "passing.json").read_text())
    kept = doc["selected"][0]
    assert kept["reversion_verdict"] == "pose_holds"
    assert kept["reverted_ra_eff_vs_truth"] == 1.5
    # The steered value stays in the base column; selection uses the reverted.
    assert kept["receptor_aligned_effector_rmsd_vs_truth"] == 2.0
    assert kept["_active_ra_eff_for_selection"] == 1.5


@pytest.mark.logic
def test_pose_holds_whose_reverted_prediction_misfolded_is_dropped(cs, tmp_path):
    c0 = tmp_path / "cycle_0"
    c0.mkdir()
    _prefilter(c0, [_icand(c0, "design_00")])
    (c0 / "reversion_results.json").write_text(json.dumps(
        {"design_00": {"verdict": "pose_holds",
                       "reverted_ra_eff_vs_truth": 1.5,
                       "reverted_independent_receptor_rmsd": 9.0,
                       "reverted_independent_effector_rmsd": 1.0}}))
    assert cs.cmd_kickoff_finalize(_finalize_args(tmp_path)) == 0
    doc = json.loads((c0 / "passing.json").read_text())
    assert doc["selected"] == []
    assert doc["all_candidates"][0]["reversion_verdict"] == "pose_holds_not_intact"


@pytest.mark.logic
def test_a_missing_reverted_fasta_warns_but_still_routes(cs, tmp_path, capsys):
    c0 = tmp_path / "cycle_0"
    c0.mkdir()
    _prefilter(c0, [_icand(c0, "design_00")])
    (c0 / "reversion_results.json").write_text(json.dumps(
        {"design_00": {"verdict": "pose_holds",
                       "reverted_ra_eff_vs_truth": 1.5,
                       "reverted_independent_receptor_rmsd": 1.0,
                       "reverted_independent_effector_rmsd": 1.0}}))
    assert cs.cmd_kickoff_finalize(_finalize_args(tmp_path)) == 0
    assert "reverted FASTA not found" in capsys.readouterr().err
    doc = json.loads((c0 / "passing.json").read_text())
    assert [c["design"] for c in doc["selected"]] == ["design_00"]


@pytest.mark.logic
@pytest.mark.parametrize("verdict", ["pose_collapses", "new_contamination"])
def test_a_failing_verdict_drops_the_design_and_keeps_its_numbers(
        cs, tmp_path, verdict):
    """The row survives into all_candidates so the aggregator can show it."""
    c0 = tmp_path / "cycle_0"
    c0.mkdir()
    _prefilter(c0, [_icand(c0, "design_00")])
    (c0 / "reversion_results.json").write_text(json.dumps(
        {"design_00": {"verdict": verdict, "reason": "because",
                       "reverted_ra_eff_vs_truth": 8.0,
                       "reverted_independent_receptor_rmsd": 2.0,
                       "reverted_independent_effector_rmsd": 3.0}}))
    assert cs.cmd_kickoff_finalize(_finalize_args(tmp_path)) == 0
    doc = json.loads((c0 / "passing.json").read_text())
    assert doc["selected"] == []
    row = doc["all_candidates"][0]
    assert row["reversion_verdict"] == verdict and row["reversion_dropped"] == 1
    assert row["reverted_ra_eff_vs_truth"] == 8.0
    assert doc[f"{verdict}_count"] == 1


@pytest.mark.logic
def test_an_unknown_verdict_is_dropped_rather_than_kept(cs, tmp_path, capsys):
    c0 = tmp_path / "cycle_0"
    c0.mkdir()
    _prefilter(c0, [_icand(c0, "design_00")])
    (c0 / "reversion_results.json").write_text(json.dumps(
        {"design_00": {"verdict": "something_new"}}))
    assert cs.cmd_kickoff_finalize(_finalize_args(tmp_path)) == 0
    doc = json.loads((c0 / "passing.json").read_text())
    assert doc["selected"] == []
    assert doc["all_candidates"][0]["reversion_verdict"] == "something_new"
    assert "unknown verdict" in capsys.readouterr().out


@pytest.mark.logic
def test_a_blank_verdict_is_recorded_as_unknown(cs, tmp_path):
    c0 = tmp_path / "cycle_0"
    c0.mkdir()
    _prefilter(c0, [_icand(c0, "design_00")])
    (c0 / "reversion_results.json").write_text(json.dumps({"design_00": {}}))
    assert cs.cmd_kickoff_finalize(_finalize_args(tmp_path)) == 0
    doc = json.loads((c0 / "passing.json").read_text())
    assert doc["all_candidates"][0]["reversion_verdict"] == "unknown"


@pytest.mark.logic
def test_the_novelty_cutoff_excludes_poses_too_close_to_the_cold_start(
        cs, tmp_path):
    c0 = tmp_path / "cycle_0"
    c0.mkdir()
    _prefilter(c0, [_icand(c0, "near", dist=0.5), _icand(c0, "far", dist=9.0)])
    assert cs.cmd_kickoff_finalize(_finalize_args(tmp_path, novelty_cutoff=1.0)) == 0
    doc = json.loads((c0 / "passing.json").read_text())
    assert doc["n_novel"] == 1
    assert [c["design"] for c in doc["selected"]] == ["far"]


@pytest.mark.logic
def test_a_candidate_with_no_distance_is_not_novel(cs, tmp_path):
    """A failed pose distance must not count as maximally novel."""
    c0 = tmp_path / "cycle_0"
    c0.mkdir()
    cand = _icand(c0, "design_00")
    cand["min_dist_to_upstream"] = None
    _prefilter(c0, [cand])
    assert cs.cmd_kickoff_finalize(_finalize_args(tmp_path, novelty_cutoff=1.0)) == 0
    assert json.loads((c0 / "passing.json").read_text())["n_novel"] == 0


@pytest.mark.logic
def test_max_passing_caps_what_is_selected(cs, tmp_path):
    c0 = tmp_path / "cycle_0"
    c0.mkdir()
    _prefilter(c0, [_icand(c0, f"design_{i:02d}", dist=float(i))
                    for i in range(5)])
    assert cs.cmd_kickoff_finalize(_finalize_args(tmp_path, max_passing=2)) == 0
    doc = json.loads((c0 / "passing.json").read_text())
    assert doc["n_selected"] == 2
    assert [c["design"] for c in doc["selected"]] == ["design_04", "design_03"]


@pytest.mark.logic
def test_finalize_appends_a_statistics_row(cs, tmp_path):
    c0 = tmp_path / "cycle_0"
    c0.mkdir()
    _prefilter(c0, [_icand(c0, "design_00")])
    assert cs.cmd_kickoff_finalize(_finalize_args(tmp_path)) == 0
    rows = (tmp_path / "cycle_statistics.csv").read_text().splitlines()
    assert len(rows) == 2 and rows[1].startswith("0,cycle_0,1,1,1,1")


@pytest.mark.logic
def test_nothing_selected_still_exits_zero(cs, tmp_path, capsys):
    c0 = tmp_path / "cycle_0"
    c0.mkdir()
    _prefilter(c0, [])
    assert cs.cmd_kickoff_finalize(_finalize_args(tmp_path)) == 0
    assert "nothing to iterate" in capsys.readouterr().out


@pytest.mark.logic
def test_asking_for_follow_up_cycles_says_they_are_not_implemented(
        cs, tmp_path, capsys):
    """--max-cycles is kept because callers pass it, and any value means 0."""
    c0 = tmp_path / "cycle_0"
    c0.mkdir()
    _prefilter(c0, [_icand(c0, "design_00")])
    assert cs.cmd_kickoff_finalize(_finalize_args(tmp_path, max_cycles=3)) == 0
    assert "not implemented" in capsys.readouterr().err


# ── the container-only seam, and the last refusal ────────────────────────────

@pytest.mark.logic
def test_a_rewrite_that_fails_warns_and_keeps_going(cs, tmp_path, capsys,
                                                    monkeypatch):
    """One unwritable design directory must not abandon the other verdicts."""
    c0 = tmp_path / "cycle_0"
    c0.mkdir()
    cand = _icand(c0, "design_00")
    _prefilter(c0, [cand])
    fasta = c0 / "reverted.fasta"
    fasta.write_text(">x\nMKVL\n")
    (c0 / "reversion_plan.json").write_text(json.dumps(
        {"entries": [{"label": "design_00",
                      "reverted_receptor_fasta": str(fasta)}]}))
    (c0 / "reversion_results.json").write_text(json.dumps(
        {"design_00": {"verdict": "pose_holds",
                       "reverted_ra_eff_vs_truth": 1.5,
                       "reverted_independent_receptor_rmsd": 1.0,
                       "reverted_independent_effector_rmsd": 1.0}}))

    real = Path.write_text

    def refuse(self, *a, **k):
        if self.name == "receptor.fasta":
            raise OSError("read-only")
        return real(self, *a, **k)
    monkeypatch.setattr(Path, "write_text", refuse)

    assert cs.cmd_kickoff_finalize(_finalize_args(tmp_path)) == 0
    assert "failed to rewrite" in capsys.readouterr().err


@pytest.mark.logic
def test_the_intact_threshold_falls_back_when_the_catalogue_is_absent(cs):
    """The module keeps a literal so it stays importable without the
    thresholds catalogue, and the two must agree."""
    sys.path.insert(0, str(REPO_ROOT / "bin"))
    from pipeline_thresholds import PipelineInternalThresholds
    assert cs._INTACT_THRESHOLD == PipelineInternalThresholds.default().intact_threshold
    assert cs._INTACT_THRESHOLD == 5.0


@pytest.mark.logic
def test_the_boltz_module_is_imported_once_and_cached(cs, monkeypatch):
    """_lazy_bns defers a numpy and Bio.Align import to call time, so the
    stdlib-only stages can run outside the container."""
    monkeypatch.setattr(cs, "_BNS_MODULE", None)
    first = cs._lazy_bns()
    assert first is cs._lazy_bns()
    assert hasattr(first, "compute_binding_rmsds")


@pytest.mark.logic
def test_pose_distance_reads_the_receptor_aligned_effector_rmsd(cs, monkeypatch,
                                                                tmp_path):
    seen = {}

    class FakeBns:
        @staticmethod
        def compute_binding_rmsds(cand, ref, **kw):
            seen.update(kw, cand=cand, ref=ref)
            return {"receptor_aligned_effector_rmsd": 3.5}

    monkeypatch.setattr(cs, "_lazy_bns", lambda: FakeBns)
    got = cs.compute_pose_distance(tmp_path / "c.pdb", tmp_path / "r.pdb",
                                   "A", "B")
    assert got == 3.5
    # Both sides are predictions, so every chain argument is the same pair.
    assert seen["rec_chain"] == seen["des_rec_chain"] == "A"
    assert seen["eff_chain"] == seen["des_eff_chain"] == "B"


@pytest.mark.logic
def test_a_missing_rmsd_becomes_nan(cs, monkeypatch, tmp_path):
    class FakeBns:
        @staticmethod
        def compute_binding_rmsds(*a, **k):
            return {}
    monkeypatch.setattr(cs, "_lazy_bns", lambda: FakeBns)
    assert cs._is_nan(cs.compute_pose_distance(
        tmp_path / "c.pdb", tmp_path / "r.pdb", "A", "B"))


@pytest.mark.logic
def test_the_module_still_imports_without_the_thresholds_catalogue(tmp_path):
    """The fallback exists so this module can be imported outside the
    container. Loading it with pipeline_thresholds blocked proves the literal
    is reached, and that it agrees with the catalogue's value."""
    import builtins
    real_import = builtins.__import__

    def blocked(name, *a, **k):
        if name == "pipeline_thresholds":
            raise ImportError("blocked for this test")
        return real_import(name, *a, **k)

    saved = sys.modules.pop("pipeline_thresholds", None)
    saved_self = sys.modules.pop("negsteer_coldstart", None)
    builtins.__import__ = blocked
    try:
        spec = importlib.util.spec_from_file_location("negsteer_coldstart_nopt",
                                                      MODULE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod._INTACT_THRESHOLD == 5.0
    finally:
        builtins.__import__ = real_import
        sys.modules.pop("negsteer_coldstart_nopt", None)
        if saved is not None:
            sys.modules["pipeline_thresholds"] = saved
        if saved_self is not None:
            sys.modules["negsteer_coldstart"] = saved_self
