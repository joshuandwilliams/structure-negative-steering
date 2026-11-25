"""The last reachable guards in compute_metrics.py and negsteer_aggregate.py.

Four things are pinned here, and each one changes a number or a path rather
than only exercising a line.

The two import fallbacks. Both scripts read their constants from
`pipeline_thresholds` and carry literals for the case where that import fails,
which is what a partially staged container looks like. A fallback out of step
with the real thresholds would silently retune a published metric, so both
tests assert the literal and assert it equals `PipelineInternalThresholds`.
These run at import, before coverage attaches to a test, so the module is
re-executed with the import blocked.

negsteer_aggregate's sys.path repair, by the technique in
test_modules_repair_sys_path.py. Nextflow invokes the script by absolute path
with no PYTHONPATH, so without the repair the sibling imports on the next line
fail and the stage dies before doing any work.

The two malformed-row guards that decide an ordering. `_sort_key` and
`_rank_continuous` are nested in their commands, so they are lifted out of the
enclosing code object and called directly. Neither has free variables. What
the tests record is where a row the code cannot read ends up, which is at the
top for an unreadable cycle and at the bottom for an unreadable rank.

compute-final-metrics' two real edges. The default `compute_metrics.py` path,
which is how the pipeline invokes the stage, and the scratch-directory
cleanup, which must not fail a stage that has already written its answer.

The guards these tests do not cover are the ones upstream arithmetic makes
impossible. They carry `pragma: no cover` naming the invariant, because a test
that replaces the invariant to reach them would assert nothing a reader could
use.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import shutil
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BIN = REPO_ROOT / "bin"
sys.path.insert(0, str(BIN))


@pytest.fixture(scope="module")
def agg():
    spec = importlib.util.spec_from_file_location(
        "negsteer_aggregate", BIN / "negsteer_aggregate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _reexec(name: str, monkeypatch):
    """Re-execute a bin/ module from source, under its own name.

    The same shape as test_modules_repair_sys_path.py. The spec carries the
    real filename, so the executed lines are attributed to the file on disk,
    and registering the module in sys.modules before the exec is what lets it
    refer to itself. monkeypatch restores sys.modules afterwards.
    """
    spec = importlib.util.spec_from_file_location(name, BIN / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _nested(fn, name):
    """Lift a nested function of `fn` out as a callable of its own.

    Both functions reached this way have no free variables, so the code
    object needs no closure. It comes straight from the enclosing function,
    so the lines executed are the ones in the file.
    """
    code = next(c for c in fn.__code__.co_consts
                if isinstance(c, types.CodeType) and c.co_name == name)
    return types.FunctionType(code, fn.__globals__, name)


# ── the two import fallbacks ─────────────────────────────────────────────────

@pytest.mark.logic
def test_the_weighted_jaccard_parameters_fall_back_to_their_literals(
        monkeypatch):
    """The try/except runs at import, so the fallback cannot be reached from
    an already-imported module. Re-executing the source with
    `pipeline_thresholds` blocked is the only way in.

    The two literals are the Gaussian's mean and twice its variance, giving a
    contact pair the weight exp(-(d - 4)^2 / 2.25). They have to equal
    PipelineInternalThresholds' values. A divergence would retune the
    weighted Jaccard on any container where the import fails, and the
    affected cohort would stop being comparable with the rest without
    anything in the output saying so.
    """
    from pipeline_thresholds import PipelineInternalThresholds

    expected = PipelineInternalThresholds.default()
    monkeypatch.setattr(sys, "path", list(sys.path))
    with mock.patch.dict(sys.modules, {"pipeline_thresholds": None}):
        module = _reexec("compute_metrics", monkeypatch)

    assert module._WJ_MU == 4.0
    assert module._WJ_TWO_SIGMA_SQ == 2.25
    assert (expected.weighted_jaccard_mu,
            expected.weighted_jaccard_two_sigma_sq) == (4.0, 2.25)
    assert module._gaussian_contact_weight(4.0) == 1.0


@pytest.mark.logic
def test_the_intact_threshold_falls_back_to_five_angstroms(monkeypatch):
    """Same technique, same reason.

    5.0 Å is the receptor-intact cutoff the whole outcome taxonomy rests on.
    _classify_outcome compares every reverted receptor RMSD against it, so a
    fallback out of step with PipelineInternalThresholds would move the
    pose_holds boundary and change which designs the thesis reports as
    rescues.
    """
    from pipeline_thresholds import PipelineInternalThresholds

    expected = PipelineInternalThresholds.default().intact_threshold
    with mock.patch.dict(sys.modules, {"pipeline_thresholds": None}):
        module = _reexec("negsteer_aggregate", monkeypatch)

    assert module._INTACT_THRESHOLD == 5.0
    assert expected == 5.0


# ── the sys.path repair ──────────────────────────────────────────────────────

@pytest.mark.logic
def test_the_aggregate_script_repairs_sys_path_and_still_imports(monkeypatch):
    """With bin/ absent from sys.path, the module has to put it back.

    Nextflow invokes negsteer_aggregate.py by absolute path inside the
    container with no PYTHONPATH set. The `from pathway_labels import ...` on
    the next line depends on this repair, so a missing insert kills the stage
    at import, before any work is done. The same ten lines in the other bin/
    modules are covered in test_modules_repair_sys_path.py by this technique.
    """
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(BIN)])
    assert str(BIN) not in sys.path

    module = _reexec("negsteer_aggregate", monkeypatch)

    assert str(BIN) in sys.path
    assert module.SCRIPT_DIR == BIN
    assert hasattr(module, "cmd_aggregate")


# ── cmd_aggregate's sort key ─────────────────────────────────────────────────
#
# _sort_key is nested in cmd_aggregate and runs on rows cmd_aggregate itself
# built, so neither int() in it can raise on real input. The guards are worth
# a test anyway, because each one decides where an unreadable row lands, and
# a row sorted to the wrong end of all_results_multicycle.csv is read as a
# result of that quality.

@pytest.mark.logic
def test_an_unreadable_cycle_sorts_with_the_first_cycle(agg):
    """A row whose `cycle` cannot be read is treated as cycle 0, which puts
    it at the top of the table rather than raising inside `rows.sort` and
    losing every row in the run, including the good ones.

    The design name still decides the second key, so the initial baseline
    stays above the steered designs of its cycle.
    """
    sort_key = _nested(agg.cmd_aggregate, "_sort_key")
    assert sort_key({"cycle": "later", "design": "design_00",
                     "rank_by_composite_score": 2}) == (0, 1, 2)
    assert sort_key({"cycle": "", "design": "initial",
                     "rank_by_composite_score": ""}) == (0, 0, 10**9)


@pytest.mark.logic
def test_an_unreadable_composite_rank_sorts_last(agg):
    """A row whose `rank_by_composite_score` cannot be read gets 10**9, the
    same sentinel as a blank rank.

    That puts it below every ranked row instead of above them, which matters
    because the top of this table is what gets inspected first.
    """
    sort_key = _nested(agg.cmd_aggregate, "_sort_key")
    ranked = sort_key({"cycle": "0", "design": "design_00",
                       "rank_by_composite_score": 1})
    unreadable = sort_key({"cycle": "0", "design": "design_01",
                           "rank_by_composite_score": "n/a"})
    assert unreadable == (0, 1, 10**9)
    assert ranked < unreadable


# ── the per-sequence ranking adapter ─────────────────────────────────────────

@pytest.fixture
def rank_continuous(agg):
    return _nested(agg.cmd_aggregate_per_sequence, "_rank_continuous")


@pytest.mark.logic
@pytest.mark.parametrize("bad", ["inf", "-inf", "nan", "Infinity"])
def test_a_non_finite_median_is_not_a_rankable_value(rank_continuous, bad):
    """_aggregate_per_sequence builds every *_median column through
    _agg_continuous, which drops non-finite seed values before taking the
    median, so a median reaching here is finite in production.

    None rather than the float is what the caller needs, because it means
    "this row cannot be ranked". An inf would sort as the worst row rather
    than as no row, and a nan would make the sort order depend on which
    comparison happened first.
    """
    assert rank_continuous({"m": bad}, "m") is None
    assert not math.isfinite(float(bad))


@pytest.mark.logic
@pytest.mark.parametrize("bad", ["n/a", [], {}, object()])
def test_an_unparseable_median_is_not_a_rankable_value(rank_continuous, bad):
    """The same answer for a value float() cannot read at all, covering both
    error types it raises, a string it cannot parse and an object it will not
    take.

    An unrankable row is dropped from the ranking and keeps its blank rank
    column. It is not dropped from the table.
    """
    assert rank_continuous({"m": bad}, "m") is None


# ── compute-final-metrics' two edges ─────────────────────────────────────────

def _run_tree(root: Path) -> Path:
    """An experiment root with one row the final-metrics filter rejects.

    steered_receptor_intact=0 fails --require-intact, so no row costs a
    compute_metrics.py subprocess and the stage still runs end to end.
    """
    (root / "cycle_0" / "design_00").mkdir(parents=True)
    (root / "cycle_0" / "plan.json").write_text(json.dumps(
        {"cycle": 0, "pred_receptor_chain": "A", "pred_effector_chain": "B"}))
    pdb = root / "cycle_0" / "design_00" / "prediction.pdb"
    pdb.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n")
    with (root / "all_results_multicycle.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "cycle", "design", "pathway", "pdb",
            "steered_ra_eff_vs_truth", "steered_receptor_intact",
            "reversion_dropped"])
        w.writeheader()
        w.writerow({"cycle": "0", "design": "design_00",
                    "pathway": "cycle_0", "pdb": str(pdb),
                    "steered_ra_eff_vs_truth": "2.0",
                    "steered_receptor_intact": "0",
                    "reversion_dropped": "0"})
    return root


@pytest.mark.logic
def test_compute_metrics_defaults_to_the_script_beside_this_one(agg, tmp_path,
                                                               capsys):
    """The flag is simply left off. Every other test of this stage points
    --compute-metrics-script at a stub, so the default had never run.

    This is how the pipeline invokes the stage. Both scripts are staged into
    the same bin/ directory in the container, and the default resolves
    against __file__ rather than the working directory, because the stage is
    run from the experiment root. A default that resolved relatively would
    refuse every real invocation.
    """
    root = _run_tree(tmp_path / "exp")
    args = agg.build_parser().parse_args(
        ["compute-final-metrics", "--experiment-root", str(root)])
    assert args.compute_metrics_script is None

    assert agg.cmd_compute_final_metrics(args) == 0
    captured = capsys.readouterr()
    assert "compute_metrics.py not found" not in captured.err
    assert "Passing: 0" in captured.out

    sibling = Path(agg.__file__).resolve().parent / "compute_metrics.py"
    assert sibling.is_file()

    out = list(csv.DictReader(
        (root / "all_results_multicycle_with_metrics.csv").open()))
    assert [r["design"] for r in out] == ["design_00"]
    assert out[0]["steered_iptm"] == ""


@pytest.mark.logic
def test_a_scratch_directory_that_cannot_be_removed_is_left_behind(
        agg, tmp_path, monkeypatch, capsys):
    """rmtree is the last statement of a clean run and the output CSV is
    already written, so a failure here must cost the tidy-up and nothing
    else.

    The exit code stays 0 because no row failed. Reporting 1 would make the
    pipeline rerun hours of finished GPU work over a directory it could not
    delete. The run is still a clean one, so the "kept for inspection" line
    stays unprinted.
    """
    def boom(*a, **kw):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(shutil, "rmtree", boom)
    root = _run_tree(tmp_path / "exp")
    args = agg.build_parser().parse_args(
        ["compute-final-metrics", "--experiment-root", str(root),
         "--compute-metrics-script", str(BIN / "compute_metrics.py")])

    assert agg.cmd_compute_final_metrics(args) == 0
    assert (root / "final_metrics_scratch").is_dir()
    assert "scratch dir kept for inspection" not in capsys.readouterr().out
    assert (root / "all_results_multicycle_with_metrics.csv").is_file()
