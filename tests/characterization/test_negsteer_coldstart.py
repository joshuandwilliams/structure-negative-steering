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
