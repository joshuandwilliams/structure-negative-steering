"""Tests for bin/cross_summary_cli.py's argument handling.

The sequence name is what every row in the cross-sequence summary is keyed on.
Deriving it from the wrong path component silently relabels a whole run, and
the numbers still look right, so the two naming modes and the explicit
NAME=PATH override are pinned here.

The refusals matter for the same reason. A CLI that accepts no inputs and
exits 0 writes an empty summary, which reads as "nothing passed".
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE = REPO_ROOT / "bin" / "cross_summary_cli.py"


@pytest.fixture(scope="module")
def cli():
    sys.path.insert(0, str(REPO_ROOT / "bin"))
    spec = importlib.util.spec_from_file_location("cross_summary_cli", MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── the name/path argument ───────────────────────────────────────────────────

@pytest.mark.logic
def test_parent_mode_names_a_run_after_its_own_directory(cli):
    name, path = cli._parse_passing_summary_arg(
        "runs/seqA/passing_summary.csv", "parent")
    assert name == "seqA"
    assert path == Path("runs/seqA/passing_summary.csv")


@pytest.mark.logic
def test_grandparent_mode_climbs_one_more_level(cli):
    """Some layouts put the summary in a per-stage subdirectory."""
    name, path = cli._parse_passing_summary_arg(
        "runs/seqA/final/passing_summary.csv", "grandparent")
    assert name == "seqA"
    assert path == Path("runs/seqA/final/passing_summary.csv")


@pytest.mark.logic
def test_an_explicit_name_overrides_either_mode(cli):
    for mode in ("parent", "grandparent"):
        name, path = cli._parse_passing_summary_arg(
            "myname=runs/seqA/passing_summary.csv", mode)
        assert name == "myname"
        assert path == Path("runs/seqA/passing_summary.csv")


@pytest.mark.logic
def test_only_the_first_equals_splits_the_argument(cli):
    """A path may legitimately contain '=', so the split is not greedy."""
    name, path = cli._parse_passing_summary_arg("n=/tmp/a=b/summary.csv", "parent")
    assert name == "n"
    assert path == Path("/tmp/a=b/summary.csv")


# ── main ─────────────────────────────────────────────────────────────────────

@pytest.mark.logic
def test_no_inputs_is_an_error_not_an_empty_summary(cli, monkeypatch, capsys,
                                                    tmp_path):
    """Exiting 0 here would write an empty summary that reads as 'none passed'."""
    monkeypatch.setattr(sys, "argv", [
        "cross_summary_cli.py", "--output", str(tmp_path / "out.csv")])
    assert cli.main() == 2
    assert "passing-summary" in capsys.readouterr().err


@pytest.mark.logic
def test_main_forwards_to_the_cohort_emitter(cli, monkeypatch, tmp_path):
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(cli.DesignCohort, "emit_cross_summary_from_dirs",
                        staticmethod(fake))
    monkeypatch.setattr(sys, "argv", [
        "cross_summary_cli.py",
        "--passing-summary", f"seqA={tmp_path / 'a.csv'}",
        "--output", str(tmp_path / "out.csv"),
    ])
    assert cli.main() == 0
    assert seen, "the CLI did not reach the emitter"


@pytest.mark.logic
def test_published_runs_dir_is_resolved_to_an_absolute_path(cli, monkeypatch,
                                                            tmp_path):
    """It is compared against run paths later, so a relative value would not
    match and every run would look unpublished."""
    seen = {}
    monkeypatch.setattr(cli.DesignCohort, "emit_cross_summary_from_dirs",
                        staticmethod(lambda **kw: seen.update(kw) or 0))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "published").mkdir()
    monkeypatch.setattr(sys, "argv", [
        "cross_summary_cli.py",
        "--passing-summary", "seqA=a.csv",
        "--published-runs-dir", "published",
        "--output", "out.csv",
    ])
    assert cli.main() == 0
    assert seen["published_runs_dir"].is_absolute()
