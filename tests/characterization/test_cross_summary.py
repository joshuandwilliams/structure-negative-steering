"""Golden-output tests for the cross-sequence summary chain.

``cross_summary_cli.py`` is the CLI the Nextflow module and the SLURM entry point
both invoke. It routes through ``design_cohort.DesignCohort`` into
``cross_summary.aggregate``, which reads a run's ``passing_summary.csv``
and picks one representative steered design, assigning its tier from
``n_pass``/``n_seeds``.

These tests pin that whole chain against output the real benchmark runs
produced. The committed fixtures cover every tier:

    9IP6  A     all seeds pass
    6G10  B     most seeds pass
    6G11  C     one seed passes
    5A6W  none  no seed passes

Each fixture directory holds ``passing_summary.csv`` and the siblings the
aggregator reads next to it, ``aggregated_results.csv`` and
``run_one_runtime_sec.txt``. Those siblings matter. When no seed passes, the
tier-none fallback fills the representative block from
``aggregated_results.csv``, so a fixture without it silently produces empty
columns rather than failing.

Two columns are excluded from the comparison. ``source_passing_summary`` and
``rep_canonical_pdb`` record where the input lived, so they hold the cluster
path in the committed output and a local path here. Every computed column must
match exactly.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = REPO_ROOT / "bin" / "cross_summary_cli.py"
FIXTURES = Path(__file__).parent / "fixtures" / "cross_summary"

# Provenance columns. They record an input location rather than a result.
PATH_COLUMNS = {"source_passing_summary", "rep_canonical_pdb"}

TARGETS = [("9IP6", "A"), ("6G10", "B"), ("6G11", "C"), ("5A6W", "none")]


def _read(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


sys.path.insert(0, str(REPO_ROOT / "bin"))


def _main(*args: str) -> int:
    """Drive cross_summary_cli in process so coverage can see it."""
    import cross_summary_cli as csv2
    with mock.patch.object(sys, "argv", ["cross_summary_cli.py", *args]):
        rc = csv2.main()
    return 0 if rc is None else rc


def _run_cli(target: str, out_path: Path) -> None:
    passing = FIXTURES / target / "passing_summary.csv"
    rc = _main("--passing-summary", f"{target}={passing}", "--output", str(out_path))
    assert rc == 0, f"cross_summary_cli failed for {target} with rc={rc}"


@pytest.mark.local_integration
@pytest.mark.parametrize("target,expected_tier", TARGETS)
def test_cross_summary_matches_committed_run(target, expected_tier, tmp_path):
    """Regenerating a real run's cross summary reproduces every computed column."""
    out = tmp_path / "cross_sequence_summary.csv"
    _run_cli(target, out)

    got = _read(out)
    want = _read(FIXTURES / target / "expected_cross_summary.csv")

    assert len(got) == len(want) == 1, f"{target}: expected exactly one row"
    assert got[0].keys() == want[0].keys(), f"{target}: column set changed"

    mismatches = [
        f"{col}: got {got[0][col]!r}, want {want[0][col]!r}"
        for col in want[0]
        if col not in PATH_COLUMNS and got[0][col] != want[0][col]
    ]
    assert not mismatches, f"{target} diverged from the committed run:\n  " + \
        "\n  ".join(mismatches)


@pytest.mark.local_integration
@pytest.mark.parametrize("target,expected_tier", TARGETS)
def test_cross_tier_is_assigned_from_seed_pass_count(target, expected_tier, tmp_path):
    """Tier follows n_pass/n_seeds, so it is asserted against the known answer.

    A test that only checked the tier column was non-empty would pass even if
    every target were mislabelled.
    """
    out = tmp_path / "cross_sequence_summary.csv"
    _run_cli(target, out)
    row = _read(out)[0]

    assert row["cross_tier"] == expected_tier, (
        f"{target}: tier {row['cross_tier']!r}, expected {expected_tier!r}"
    )

    n_pass, n_seeds = int(row["rep_n_pass"]), int(row["rep_n_seeds"])
    if expected_tier == "A":
        assert n_pass == n_seeds
    elif expected_tier == "B":
        assert 1 < n_pass < n_seeds
    elif expected_tier == "C":
        assert n_pass == 1
    else:
        assert n_pass == 0


@pytest.mark.local_unit
def test_missing_passing_summary_degrades_to_tier_none(tmp_path):
    """A missing input warns and emits a tier-none row rather than failing.

    This is deliberate. The pipeline aggregates many sequences at once and one
    absent file must not kill the batch. The consequence is that a silently
    incomplete cohort looks like a healthy one, so callers have to assert the
    expected sequence count themselves rather than trusting a zero exit.
    """
    out = tmp_path / "out.csv"
    rc = _main("--passing-summary", f"NOPE={tmp_path / 'absent.csv'}", "--output", str(out))
    assert rc == 0
    rows = _read(out)
    assert len(rows) == 1 and rows[0]["cross_tier"] == "none"


@pytest.mark.local_unit
def test_every_emitted_row_is_counted(tmp_path):
    """The row count matches the number of sequences asked for.

    The completeness check the tier-none fallback makes necessary.
    """
    out = tmp_path / "out.csv"
    args = ["--output", str(out)]
    for t in ("9IP6", "6G10", "6G11"):
        args += ["--passing-summary", f"{t}={FIXTURES / t / 'passing_summary.csv'}"]
    assert _main(*args) == 0
    rows = _read(out)
    assert len(rows) == 3, f"asked for 3 sequences, got {len(rows)} rows"
    assert {r["mpnn_sequence"] for r in rows} == {"9IP6", "6G10", "6G11"}
