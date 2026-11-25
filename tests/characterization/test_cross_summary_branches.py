"""Tests for cross_summary.py's input guards, sidecars and CLI refusals.

This module decides, for every MPNN sequence in a cohort, which single design
represents it and which tier it lands in. The tiers are what the thesis reports
per target. A sequence pushed into the wrong tier is a wrong result, and a
sequence dropped from the table disappears from every proportion computed over
the cohort.

Three classes of behaviour are pinned here.

The input readers. A malformed or unreadable passing_summary.csv must become a
stub row under the default policy and must abort under --strict. Silently
dropping it shrinks the cohort while every count still looks plausible.

The sidecars. row_type.txt is the only thing that marks a run as a Task 6
negative control. A control read as a design joins the cross-rank ranking and
competes with real steered designs, which is exactly the comparison the
controls exist to calibrate.

The CLI. Every refusal returns 2. A refusal that returned 0 would write a
table missing the sequences it could not parse.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE = REPO_ROOT / "bin" / "cross_summary.py"


@pytest.fixture(scope="module")
def cs():
    sys.path.insert(0, str(REPO_ROOT / "bin"))
    spec = importlib.util.spec_from_file_location("cross_summary", MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


PASSING_COLUMNS = [
    "design", "n_pass", "n_seeds", "ra_eff_vs_truth_median",
    "true_jaccard_median", "rank_by_composite_score", "outcome",
    "canonical_pdb",
]


def prow(design="design_00", *, n_pass=3, n_seeds=3, ra=2.0, tj=0.8, rank=1,
         outcome="no_reversion", pdb=""):
    return {"design": design, "n_pass": str(n_pass), "n_seeds": str(n_seeds),
            "ra_eff_vs_truth_median": str(ra),
            "true_jaccard_median": str(tj),
            "rank_by_composite_score": str(rank),
            "outcome": outcome, "canonical_pdb": pdb}


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def write_passing(path: Path, rows):
    return write_csv(path, PASSING_COLUMNS, rows)


def read_out(path: Path):
    with open(path) as f:
        return list(csv.DictReader(f))


def deny_read(path: Path) -> None:
    """Make `path` unreadable, or skip the test if permissions do not bite."""
    path.chmod(0o000)
    if os.access(path, os.R_OK):
        path.chmod(0o644)
        pytest.skip("file permissions do not restrict this user")


# ── sibling imports ──────────────────────────────────────────────────────────
#
# cross_summary.py calls extract_passing.extract_row for the tier-none
# fallback. Nextflow invokes the script by absolute path with no PYTHONPATH, so
# the script has to put its own bin/ directory on sys.path itself. If it did
# not, the fallback would fail at import time and every empty passing_summary
# would lose its representative.

@pytest.mark.logic
def test_the_script_adds_its_own_bin_directory_to_the_import_path(monkeypatch):
    """Loaded with bin/ absent from sys.path, the script still finds its
    sibling. Nextflow runs it by absolute path and sets no PYTHONPATH."""
    bin_dir = str(REPO_ROOT / "bin")
    monkeypatch.setattr(sys, "path",
                        [p for p in sys.path if p != bin_dir])
    monkeypatch.delitem(sys.modules, "extract_passing", raising=False)
    spec = importlib.util.spec_from_file_location("cross_summary_unpathed",
                                                  MODULE)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)
    assert bin_dir in sys.path
    assert callable(mod._extract_passing_row)


# ── the numeric coercions ────────────────────────────────────────────────────
#
# Both helpers are fed CSV text in production and already-parsed numbers by
# in-process callers. The two routes must agree, and a nan must be rejected
# rather than ranked. nan compares false against everything, so a nan that
# reached the sort would place a failed prediction at an arbitrary rank.

@pytest.mark.logic
def test_a_number_parses_the_same_as_its_csv_text(cs):
    assert cs._try_float(2) == 2.0
    assert cs._try_float("2") == 2.0
    assert cs._try_int(3) == 3
    assert cs._try_int("3") == 3


@pytest.mark.logic
@pytest.mark.parametrize("v", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_number_is_rejected_not_ranked(cs, v):
    assert cs._try_float(v) is None


@pytest.mark.logic
def test_an_absent_value_is_none_rather_than_zero(cs):
    """A blank or missing metric must come back as None. Zero ra_eff is a
    perfect pose, so coercing an absent value to 0.0 would rank a prediction
    that never landed at the top of the cohort."""
    assert cs._try_float(None) is None
    assert cs._try_float("") is None
    assert cs._try_int(None) is None
    assert cs._try_int("") is None


# ── reading the per-sequence inputs ──────────────────────────────────────────
#
# One passing_summary.csv per sequence. A file that cannot be decoded is a
# truncated or half-written run, which happens when a task is killed mid-write.
# The default policy records it as an empty stub so the sequence still appears.
# --strict aborts instead, which is what a publication run should do.

@pytest.mark.logic
def test_an_undecodable_input_becomes_an_empty_stub_by_default(cs, tmp_path,
                                                               capsys):
    """A half-written CSV must not remove the sequence from the cohort. The
    row survives with no representative, so the cohort base stays right."""
    path = tmp_path / "seqA" / "passing_summary.csv"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"design,n_pass\n\xff\xfe,3\n")
    rows, paths, fieldnames = cs._read_passing_summary_inputs(
        [("seqA", path)], strict=False)
    assert rows == {"seqA": []}
    assert paths == {"seqA": path}
    assert fieldnames == []
    assert "failed to read" in capsys.readouterr().err


@pytest.mark.logic
def test_an_undecodable_input_aborts_under_strict(cs, tmp_path):
    path = tmp_path / "seqA" / "passing_summary.csv"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"design,n_pass\n\xff\xfe,3\n")
    with pytest.raises(cs._AggregateInputError, match="failed to read"):
        cs._read_passing_summary_inputs([("seqA", path)], strict=True)


@pytest.mark.logic
def test_an_undecodable_input_makes_aggregate_exit_two_under_strict(cs,
                                                                    tmp_path):
    path = tmp_path / "seqA" / "passing_summary.csv"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"design,n_pass\n\xff\xfe,3\n")
    out = tmp_path / "cross.csv"
    assert cs.aggregate(sequences=[("seqA", path)], output_path=out,
                        strict=True) == 2
    assert not out.exists()


@pytest.mark.logic
def test_an_unreadable_input_is_handled_like_an_undecodable_one(cs, tmp_path,
                                                                capsys):
    path = write_passing(tmp_path / "seqA" / "passing_summary.csv", [prow()])
    deny_read(path)
    try:
        rows, _, _ = cs._read_passing_summary_inputs(
            [("seqA", path)], strict=False)
    finally:
        path.chmod(0o644)
    assert rows == {"seqA": []}
    assert "failed to read" in capsys.readouterr().err


@pytest.mark.logic
def test_differing_headers_are_unioned_in_first_file_order(cs, tmp_path,
                                                           capsys):
    """Two inputs from different pipeline versions must not lose a column.
    The first file's header fixes the order and the extras are appended, so a
    rep_* column never silently drops the value under it."""
    first = write_csv(tmp_path / "seqA" / "passing_summary.csv",
                      ["design", "n_pass"], [{"design": "d", "n_pass": "3"}])
    second = write_csv(tmp_path / "seqB" / "passing_summary.csv",
                       ["design", "n_pass", "ipsae_median"],
                       [{"design": "d", "n_pass": "3",
                         "ipsae_median": "0.4"}])
    _, _, fieldnames = cs._read_passing_summary_inputs(
        [("seqA", first), ("seqB", second)], strict=False)
    assert fieldnames == ["design", "n_pass", "ipsae_median"]
    assert "differ from the first" in capsys.readouterr().err


@pytest.mark.logic
def test_a_duplicate_name_is_refused_before_anything_is_read(cs, tmp_path):
    """The second row would overwrite the first in a dict keyed by name."""
    path = write_passing(tmp_path / "seqA" / "passing_summary.csv", [prow()])
    with pytest.raises(cs._AggregateInputError, match="duplicate"):
        cs._read_passing_summary_inputs([("seqA", path), ("seqA", path)],
                                        strict=False)


# ── the tier-none fallback ───────────────────────────────────────────────────
#
# An empty passing_summary.csv means every prediction failed the 5 Å
# eligibility gate. The sibling aggregated_results.csv is then read raw and its
# best composite row becomes the representative, still at tier none. Showing
# the best of bad attempts is more informative than a blank row, and the tier
# stripe already tells the reader it did not pass.

@pytest.mark.logic
def test_the_fallback_sources_ra_eff_per_stage_and_jaccard_from_steered(cs,
                                                                        tmp_path):
    """A pose_holds row's ra_eff comes from reverted_, everything else from
    steered_. true_jaccard always comes from steered_. The reverted pose is
    the final design, but the interface identity belongs to the cold start.

    Constructed so the two policies disagree. The pose_holds row scores
    0.3 - 0.05*2.0 = 0.2 on the correct keys and 0.3 - 0.05*9.0 = -0.15 on the
    steered ones, against 0.5 - 0.05*8.0 = 0.1 for the other row.
    """
    p = write_passing(tmp_path / "seqA" / "passing_summary.csv", [])
    write_csv(p.parent / "aggregated_results.csv",
              ["design", "outcome", "steered_ra_eff_vs_truth_median",
               "reverted_ra_eff_vs_truth_median",
               "steered_true_jaccard_median"],
              [{"design": "cold", "outcome": "no_reversion",
                "steered_ra_eff_vs_truth_median": "8.0",
                "reverted_ra_eff_vs_truth_median": "",
                "steered_true_jaccard_median": "0.5"},
               {"design": "held", "outcome": "pose_holds",
                "steered_ra_eff_vs_truth_median": "9.0",
                "reverted_ra_eff_vs_truth_median": "2.0",
                "steered_true_jaccard_median": "0.3"}])
    rep = cs._pick_rep_from_aggregated(p)
    assert rep["design"] == "held"
    assert rep["_picked_tier"] == "none"
    assert rep["_fallback_composite"] == pytest.approx(0.2)


@pytest.mark.logic
def test_rows_missing_a_composite_component_give_no_fallback(cs, tmp_path):
    """A row with no ra_eff cannot be scored, so it must not be picked by
    default. Picking it would stamp an empty metric onto the cohort table."""
    p = write_passing(tmp_path / "seqA" / "passing_summary.csv", [])
    write_csv(p.parent / "aggregated_results.csv",
              ["design", "outcome", "steered_ra_eff_vs_truth_median",
               "steered_true_jaccard_median"],
              [{"design": "a", "outcome": "no_reversion",
                "steered_ra_eff_vs_truth_median": "",
                "steered_true_jaccard_median": "0.5"},
               {"design": "b", "outcome": "no_reversion",
                "steered_ra_eff_vs_truth_median": "3.0",
                "steered_true_jaccard_median": ""}])
    assert cs._pick_rep_from_aggregated(p) is None


@pytest.mark.logic
def test_a_permission_denied_sibling_aggregate_warns_and_gives_no_fallback(
        cs, tmp_path, capsys):
    p = write_passing(tmp_path / "seqA" / "passing_summary.csv", [])
    agg = write_csv(p.parent / "aggregated_results.csv", ["design"],
                    [{"design": "a"}])
    deny_read(agg)
    try:
        assert cs._pick_rep_from_aggregated(p) is None
    finally:
        agg.chmod(0o644)
    assert "could not read" in capsys.readouterr().err


@pytest.mark.logic
def test_the_fallback_columns_are_added_to_the_output_header(cs, tmp_path):
    """extract_row emits columns a thin passing_summary header never had. They
    must be unioned in, or the CSV writer drops the fallback's only metrics."""
    p = write_csv(tmp_path / "seqA" / "passing_summary.csv", ["design"], [])
    write_csv(p.parent / "aggregated_results.csv",
              ["design", "outcome", "steered_ra_eff_vs_truth_median",
               "steered_true_jaccard_median"],
              [{"design": "best", "outcome": "no_reversion",
                "steered_ra_eff_vs_truth_median": "6.0",
                "steered_true_jaccard_median": "0.4"}])
    out = tmp_path / "cross.csv"
    assert cs.aggregate(sequences=[("seqA", p)], output_path=out,
                        strict=False) == 0
    row = read_out(out)[0]
    assert row["cross_tier"] == "none"
    assert row["rep_design"] == "best"
    # 0.4 - 0.05 * 6.0 = 0.1
    assert row["cross_composite_score"] == "0.100000"
    assert row["rep_ra_eff_vs_truth_median"] == "6.0"
    assert row["cross_rank_by_composite"] == ""


# ── the runtime sidecar ──────────────────────────────────────────────────────
#
# run_one_runtime_sec.txt carries a sequence's cumulative wall clock. It is the
# only source for a tier-none sequence, whose passing_summary.csv has no row to
# read the column off. A junk value must read as absent, not as a runtime.

@pytest.mark.logic
def test_a_non_numeric_runtime_sidecar_reads_as_absent(cs, tmp_path):
    """Returning the raw text would put a non-number in a seconds column and
    break every per-sequence cost figure drawn from it."""
    p = write_passing(tmp_path / "seqA" / "passing_summary.csv", [])
    (p.parent / "run_one_runtime_sec.txt").write_text("not-a-number\n")
    assert cs._resolve_run_one_runtime_sec(None, p) == ""


@pytest.mark.logic
def test_an_unreadable_runtime_sidecar_reads_as_absent(cs, tmp_path):
    p = write_passing(tmp_path / "seqA" / "passing_summary.csv", [])
    (p.parent / "run_one_runtime_sec.txt").mkdir()
    assert cs._resolve_run_one_runtime_sec(None, p) == ""


@pytest.mark.logic
def test_a_numeric_runtime_sidecar_is_used_when_no_row_has_one(cs, tmp_path):
    p = write_passing(tmp_path / "seqA" / "passing_summary.csv", [])
    (p.parent / "run_one_runtime_sec.txt").write_text(" 1234.5 \n")
    assert cs._resolve_run_one_runtime_sec({}, p) == "1234.5"


# ── the row_type sidecar ─────────────────────────────────────────────────────
#
# row_type.txt is written beside passing_summary.csv by NEGSTEER_CONTROLS and
# is the only marker separating a scrambled or poly-A control from a steered
# design. A control read as steered enters the cross-rank ranking, so it
# competes with the designs it was built to calibrate against. Absent or
# unreadable defaults to 'steered', which keeps pre-controls workdirs working.

@pytest.mark.logic
def test_a_control_sidecar_is_read_verbatim(cs, tmp_path):
    p = write_passing(tmp_path / "seqA" / "passing_summary.csv", [prow()])
    (p.parent / "row_type.txt").write_text("control_scrambled\n")
    assert cs._read_row_type_sidecar(p) == "control_scrambled"


@pytest.mark.logic
@pytest.mark.parametrize("content", ["", "   \n"])
def test_an_empty_sidecar_falls_back_to_steered(cs, tmp_path, content):
    """An empty file is a failed write, not a declaration of a new row type."""
    p = write_passing(tmp_path / "seqA" / "passing_summary.csv", [prow()])
    (p.parent / "row_type.txt").write_text(content)
    assert cs._read_row_type_sidecar(p) == "steered"


@pytest.mark.logic
def test_an_unreadable_sidecar_falls_back_to_steered(cs, tmp_path):
    p = write_passing(tmp_path / "seqA" / "passing_summary.csv", [prow()])
    (p.parent / "row_type.txt").mkdir()
    assert cs._read_row_type_sidecar(p) == "steered"


@pytest.mark.logic
def test_no_sidecar_at_all_is_steered(cs, tmp_path):
    p = write_passing(tmp_path / "seqA" / "passing_summary.csv", [prow()])
    assert cs._read_row_type_sidecar(p) == "steered"


@pytest.mark.logic
def test_any_sidecar_text_is_taken_as_a_row_type(cs, tmp_path):
    """Current behaviour, and a gap. The sidecar is not checked against the
    known row types, so a typo such as 'steerd' makes a production design
    non-steered. It then drops out of the cross-rank ranking with no warning.
    """
    p = write_passing(tmp_path / "seqA" / "passing_summary.csv", [prow()])
    (p.parent / "row_type.txt").write_text("steerd\n")
    assert cs._read_row_type_sidecar(p) == "steerd"
    out = tmp_path / "cross.csv"
    assert cs.aggregate(sequences=[("seqA", p)], output_path=out,
                        strict=False) == 0
    assert read_out(out)[0]["cross_rank_by_composite"] == ""


@pytest.mark.logic
def test_a_control_is_kept_out_of_the_cross_rank_ranking(cs, tmp_path, capsys):
    """Controls are diagnostic. Ranking one against the steered designs is
    the comparison they exist to calibrate, so it must not happen. The control
    here has the better pose, so a bug would put it at rank 1."""
    seqs = []
    for name, ra, row_type in (("seqA", 4.0, "steered"),
                               ("ctrl", 1.0, "control_polyA")):
        p = write_passing(tmp_path / name / "passing_summary.csv",
                          [prow(ra=ra)])
        (p.parent / "row_type.txt").write_text(row_type)
        seqs.append((name, p))
    out = tmp_path / "cross.csv"
    assert cs.aggregate(sequences=seqs, output_path=out, strict=False) == 0
    by = {r["mpnn_sequence"]: r for r in read_out(out)}
    assert by["ctrl"]["row_type"] == "control_polyA"
    assert by["ctrl"]["cross_rank_by_composite"] == ""
    assert by["ctrl"]["cross_rank_by_ra_eff"] == ""
    assert by["ctrl"]["cross_composite_score"] == ""
    assert by["seqA"]["cross_rank_by_composite"] == "1"
    assert by["seqA"]["cross_rank_by_ra_eff"] == "1"
    err = capsys.readouterr().err
    assert "row_type breakdown: control_polyA=1, steered=1" in err
    assert "2 sequences, 1 placed in tiers A/B/C" in err
    # The tier line is labelled "steered only" and counts only steered
    # records, so it agrees with the n_ranked count on the line above it.
    # It used to count every record, which put this tier-A control into the
    # A total while its cross_rank_by_composite stayed blank.
    assert "tier breakdown (steered only): A=1 B=0 C=0 none=0" in err


@pytest.mark.logic
def test_an_all_steered_cohort_prints_no_row_type_breakdown(cs, tmp_path,
                                                             capsys):
    p = write_passing(tmp_path / "seqA" / "passing_summary.csv", [prow()])
    out = tmp_path / "cross.csv"
    assert cs.aggregate(sequences=[("seqA", p)], output_path=out,
                        strict=False) == 0
    assert "row_type breakdown" not in capsys.readouterr().err


# ── aggregate's own refusal ──────────────────────────────────────────────────

@pytest.mark.logic
def test_an_empty_sequence_list_is_refused(cs, tmp_path, capsys):
    """Writing a header-only table here would read as 'nothing passed'."""
    out = tmp_path / "cross.csv"
    assert cs.aggregate(sequences=[], output_path=out, strict=False) == 2
    assert not out.exists()
    assert "no inputs supplied" in capsys.readouterr().err


# ── the CLI ──────────────────────────────────────────────────────────────────
#
# main() turns flags into the (name, path) list aggregate consumes. Each
# refusal returns 2. Returning 0 would publish a table missing whatever could
# not be parsed, with no sign that anything was left out.

@pytest.mark.logic
def test_a_path_with_no_inferable_name_is_refused(cs, monkeypatch, capsys,
                                                   tmp_path):
    """The name becomes the key of every row, so only the safe character set
    is accepted rather than sanitised."""
    monkeypatch.setattr(sys, "argv", [
        "cross_summary.py",
        "--passing-summary", str(tmp_path / "bad name!" / "passing_summary.csv"),
        "--output", str(tmp_path / "out.csv"),
    ])
    assert cs.main() == 2
    assert "unexpected characters" in capsys.readouterr().err


@pytest.mark.logic
def test_two_flags_naming_the_same_sequence_are_refused(cs, monkeypatch,
                                                         capsys, tmp_path):
    """Both would be keyed 'seqA' and the second would overwrite the first."""
    p = write_passing(tmp_path / "seqA" / "passing_summary.csv", [prow()])
    monkeypatch.setattr(sys, "argv", [
        "cross_summary.py",
        "--passing-summary", f"seqA={p}",
        "--passing-summary", f"seqA={p}",
        "--output", str(tmp_path / "out.csv"),
    ])
    assert cs.main() == 2
    assert "duplicate sequence name 'seqA'" in capsys.readouterr().err


@pytest.mark.logic
def test_a_passing_summary_dir_that_is_not_a_directory_is_refused(cs,
                                                                   monkeypatch,
                                                                   capsys,
                                                                   tmp_path):
    stray = tmp_path / "stray.txt"
    stray.write_text("x\n")
    monkeypatch.setattr(sys, "argv", [
        "cross_summary.py",
        "--passing-summary-dir", str(stray),
        "--output", str(tmp_path / "out.csv"),
    ])
    assert cs.main() == 2
    assert "not a directory" in capsys.readouterr().err


@pytest.mark.logic
def test_a_run_tree_enumerates_subdirectories_and_skips_the_rest(cs,
                                                                  monkeypatch,
                                                                  capsys,
                                                                  tmp_path):
    """One flag per cohort instead of one per sequence. A stray file must not
    become a sequence, a directory whose name is not a valid identifier is
    skipped with a warning, and a sequence already named explicitly is not
    added twice."""
    runs = tmp_path / "runs"
    explicit = write_passing(runs / "seqA" / "passing_summary.csv", [prow()])
    write_passing(runs / "seqB" / "passing_summary.csv",
                  [prow(n_pass=2, n_seeds=3)])
    (runs / "bad name!").mkdir(parents=True)
    (runs / "stray.txt").write_text("x\n")
    out = tmp_path / "out.csv"
    monkeypatch.setattr(sys, "argv", [
        "cross_summary.py",
        "--passing-summary", str(explicit),
        "--passing-summary-dir", str(runs),
        "--output", str(out),
    ])
    assert cs.main() == 0
    rows = read_out(out)
    assert sorted(r["mpnn_sequence"] for r in rows) == ["seqA", "seqB"]
    assert {r["mpnn_sequence"]: r["cross_tier"] for r in rows} == {
        "seqA": "A", "seqB": "B"}
    assert "skipping" in capsys.readouterr().err


@pytest.mark.logic
def test_the_published_runs_dir_is_resolved_before_it_is_stamped(cs,
                                                                  monkeypatch,
                                                                  tmp_path):
    """A relative value would be stamped verbatim into rep_canonical_pdb, then
    resolve against each downstream container task's own work dir rather than
    the launch dir. The file would be unreadable where it is read."""
    p = write_passing(tmp_path / "seqA" / "passing_summary.csv",
                      [prow(pdb="/scratch/work/b6/afc/seqA/cycle_0/x.pdb")])
    (tmp_path / "published").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "cross_summary.py",
        "--passing-summary", f"seqA={p}",
        "--published-runs-dir", "published",
        "--output", "out.csv",
    ])
    assert cs.main() == 0
    got = read_out(tmp_path / "out.csv")[0]["rep_canonical_pdb"]
    assert got == str((tmp_path / "published" / "seqA" / "cycle_0"
                       / "x.pdb").resolve())
