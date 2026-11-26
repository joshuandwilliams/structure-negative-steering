"""Tests for negsteer_aggregate's `aggregate` stage, on built run trees.

This stage flattens a finished run into all_results_multicycle.csv, which is
what every analysis in analyses/ reads. A row lost here is a prediction that
silently never happened, and a column renamed wrongly is a number attributed
to the wrong metric. Neither shows up as an error.

The trees are built by runtree.py with the values stated up front, so each
test asserts the aggregator reports what was put in rather than that it
agrees with its own previous output.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import runtree as rt

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE = REPO_ROOT / "bin" / "negsteer_aggregate.py"


@pytest.fixture(scope="module")
def agg():
    sys.path.insert(0, str(REPO_ROOT / "bin"))
    spec = importlib.util.spec_from_file_location("negsteer_aggregate", MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _rows(path: Path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ── locating cycle 0 ─────────────────────────────────────────────────────────

@pytest.mark.logic
def test_a_tree_with_no_plan_is_refused(agg, tmp_path, capsys):
    """Without plan.json there is no cycle 0, and writing an empty table
    would read as a run that produced nothing."""
    assert agg.cmd_aggregate(Args(experiment_root=tmp_path)) == 2
    assert "no cycle_0 plan.json" in capsys.readouterr().err


@pytest.mark.logic
def test_a_path_that_is_not_a_directory_is_refused(agg, tmp_path, capsys):
    f = tmp_path / "file.txt"
    f.write_text("x")
    assert agg.cmd_aggregate(Args(experiment_root=f)) == 2
    assert "not a directory" in capsys.readouterr().err


@pytest.mark.logic
@pytest.mark.parametrize("nested", [True, False])
def test_both_cycle_zero_layouts_are_found(agg, tmp_path, nested, capsys):
    """A single-cycle run writes cycle-0 files in the root; a multi-cycle run
    nests them. Both still exist on disk."""
    rt.build(tmp_path, nested=nested,
             steered_rows=[rt.steered_row("design_00")],
             design_names=["design_00"])
    assert agg.cmd_aggregate(Args(experiment_root=tmp_path)) == 0
    out = capsys.readouterr().out
    assert ("nested" if nested else "flat") in out


# ── the flattened table ──────────────────────────────────────────────────────

@pytest.mark.logic
def test_every_steered_row_reaches_the_table(agg, tmp_path):
    names = [f"design_{i:02d}" for i in range(4)]
    rt.build(tmp_path, steered_rows=[rt.steered_row(n, rank=i + 1)
                                     for i, n in enumerate(names)],
             design_names=names)
    assert agg.cmd_aggregate(Args(experiment_root=tmp_path)) == 0
    rows = _rows(tmp_path / "all_results_multicycle.csv")
    # Row order is the aggregator's ranking, not the input order, so the
    # property under test is that none is lost.
    assert sorted(r["design"] for r in rows) == sorted(names)


@pytest.mark.logic
def test_the_steered_metrics_are_renamed_not_recomputed(agg, tmp_path):
    """The aggregator renames the engine's columns to steered_*. A value that
    changes on the way through would be a number attributed to the wrong
    metric in every downstream figure."""
    rt.build(tmp_path,
             steered_rows=[rt.steered_row("design_00", ra_eff=3.25,
                                          ind_rec=1.5, ind_eff=2.5)],
             design_names=["design_00"])
    assert agg.cmd_aggregate(Args(experiment_root=tmp_path)) == 0
    row = _rows(tmp_path / "all_results_multicycle.csv")[0]
    assert float(row["steered_ra_eff_vs_truth"]) == 3.25
    assert float(row["steered_independent_receptor_rmsd"]) == 1.5
    assert float(row["steered_independent_effector_rmsd"]) == 2.5


@pytest.mark.logic
def test_the_pathways_index_is_written(agg, tmp_path):
    rt.build(tmp_path, steered_rows=[rt.steered_row("design_00")],
             design_names=["design_00"])
    assert agg.cmd_aggregate(Args(experiment_root=tmp_path)) == 0
    paths = json.loads((tmp_path / "pathways.json").read_text())
    assert [p["label"] for p in paths] == ["cycle_0"]
    assert paths[0]["n_designs"] == 1


@pytest.mark.logic
def test_an_empty_steered_table_still_produces_a_file(agg, tmp_path):
    """A run where nothing passed is a result, not a failure."""
    rt.build(tmp_path, steered_rows=[], design_names=[])
    assert agg.cmd_aggregate(Args(experiment_root=tmp_path)) == 0
    assert (tmp_path / "all_results_multicycle.csv").exists()


# ── the reconstructed initial row ────────────────────────────────────────────

@pytest.mark.logic
def test_a_run_with_no_steered_csv_reconstructs_its_initial_row(agg, tmp_path,
                                                                capsys):
    """A cold start that already passed writes no steered_results.csv. Without
    the reconstruction that complex would be absent from the comparison table
    entirely, which reads as a complex that was never run."""
    rt.build(tmp_path, write_steered_csv=False,
             plan_doc=rt.plan(initial_ra_eff=2.4, skip_steering=True))
    assert agg.cmd_aggregate(Args(experiment_root=tmp_path)) == 0
    rows = _rows(tmp_path / "all_results_multicycle.csv")
    assert [r["design"] for r in rows] == ["initial"]
    assert float(rows[0]["steered_ra_eff_vs_truth"]) == 2.4
    assert "reconstructing initial row" in capsys.readouterr().out


@pytest.mark.logic
def test_the_reconstructed_row_marks_a_broken_fold_as_not_intact(agg, tmp_path):
    rt.build(tmp_path, write_steered_csv=False,
             plan_doc=rt.plan(initial_ra_eff=2.4, initial_ind_rec=9.0))
    assert agg.cmd_aggregate(Args(experiment_root=tmp_path)) == 0
    assert _rows(tmp_path / "all_results_multicycle.csv")[0][
        "steered_receptor_intact"] == "0"


@pytest.mark.logic
def test_the_reconstructed_row_leaves_intactness_blank_when_unknown(agg,
                                                                    tmp_path):
    """An old run with no independent RMSDs must not be reported as intact."""
    rt.build(tmp_path, write_steered_csv=False,
             plan_doc=rt.plan(initial_ra_eff=2.4, initial_ind_rec=None,
                              initial_ind_eff=None))
    assert agg.cmd_aggregate(Args(experiment_root=tmp_path)) == 0
    assert _rows(tmp_path / "all_results_multicycle.csv")[0][
        "steered_receptor_intact"] == ""


@pytest.mark.logic
def test_an_empty_wrong_interface_gives_a_zero_jaccard_not_a_blank(agg,
                                                                   tmp_path):
    """Only the truth set is required. An empty predicted-wrong set is a real
    answer, 0 overlap, rather than missing data."""
    rt.build(tmp_path, write_steered_csv=False,
             plan_doc=rt.plan(initial_ra_eff=2.4, initial_wrong_interface_idx=[]))
    assert agg.cmd_aggregate(Args(experiment_root=tmp_path)) == 0
    row = _rows(tmp_path / "all_results_multicycle.csv")[0]
    assert float(row["steered_true_jaccard"]) == 0.0


# ── merging the kickoff selection ────────────────────────────────────────────

@pytest.mark.logic
def test_kickoff_distances_and_selection_are_merged_onto_the_rows(agg,
                                                                  tmp_path):
    rt.build(
        tmp_path,
        steered_rows=[rt.steered_row("design_00"), rt.steered_row("design_01")],
        design_names=["design_00", "design_01"],
        passing=rt.passing_doc(
            [rt.candidate("design_00", 0, min_dist_to_upstream=6.5),
             rt.candidate("design_01", 1, min_dist_to_upstream=1.0)],
            selected_idx=[0]))
    assert agg.cmd_aggregate(Args(experiment_root=tmp_path)) == 0
    rows = {r["design"]: r for r in _rows(tmp_path / "all_results_multicycle.csv")}
    assert float(rows["design_00"]["min_dist_to_upstream"]) == 6.5
    assert rows["design_00"]["selected_for_next_cycle"] == "1"
    assert rows["design_01"]["selected_for_next_cycle"] == "0"


@pytest.mark.logic
def test_a_reverted_row_keeps_its_steered_values_in_the_base_columns(agg,
                                                                     tmp_path):
    """The reverted numbers live only in reverted_* keys. Overwriting the base
    columns would make every table report a pose that was never predicted from
    the steered sequence."""
    rt.build(
        tmp_path,
        steered_rows=[rt.steered_row("design_00", ra_eff=8.0)],
        design_names=["design_00"],
        passing=rt.passing_doc([rt.candidate(
            "design_00", 0, reversion_verdict="pose_holds",
            reverted_ra_eff_vs_truth=1.5,
            reverted_sequence_used_as_parent=1)]))
    assert agg.cmd_aggregate(Args(experiment_root=tmp_path)) == 0
    row = _rows(tmp_path / "all_results_multicycle.csv")[0]
    assert float(row["steered_ra_eff_vs_truth"]) == 8.0
    assert float(row["reverted_ra_eff_vs_truth"]) == 1.5
    assert row["reversion_verdict"] == "pose_holds"


@pytest.mark.logic
def test_a_dropped_row_carries_its_verdict(agg, tmp_path):
    rt.build(
        tmp_path,
        steered_rows=[rt.steered_row("design_00")],
        design_names=["design_00"],
        passing=rt.passing_doc([rt.candidate(
            "design_00", 0, reversion_verdict="pose_collapses",
            reversion_dropped=1)]))
    assert agg.cmd_aggregate(Args(experiment_root=tmp_path)) == 0
    row = _rows(tmp_path / "all_results_multicycle.csv")[0]
    assert row["reversion_verdict"] == "pose_collapses"
    assert row["reversion_dropped"] == "1"
