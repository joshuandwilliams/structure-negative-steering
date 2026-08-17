"""run_one_runtime_sec.txt is the per-run cost the compute-cost analysis reads.

Under the old design one SLURM job ran every stage in sequence and recorded its
own elapsed time. The stages are separate Nextflow tasks now, so each appends
its seconds to .stage_times and this sums them. Getting the sum wrong would not
fail a run. It would quietly move a published cost number, which is why these
pin the parsing rather than only the happy path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "sum_stage_times", REPO_ROOT / "bin" / "sum_stage_times.py")
sst = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sst)


@pytest.mark.local_unit
def test_sums_every_stage(tmp_path: Path) -> None:
    f = tmp_path / ".stage_times"
    f.write_text("prepare 12\nplan 300\npredict_one_0 45\ncollect 3\n")
    assert sst.total_seconds(f) == 360


@pytest.mark.local_unit
def test_a_run_with_no_stage_times_costs_zero_rather_than_raising(tmp_path: Path) -> None:
    """A target that failed before its first stage has no file. The pipeline
    must still write a runtime, or the analysis loses the row entirely."""
    assert sst.total_seconds(tmp_path / "absent") == 0


@pytest.mark.local_unit
def test_an_empty_file_is_zero(tmp_path: Path) -> None:
    f = tmp_path / ".stage_times"
    f.write_text("")
    assert sst.total_seconds(f) == 0


@pytest.mark.local_unit
@pytest.mark.parametrize("line", [
    "prepare",              # a stage name with no time
    "prepare 12 extra",     # a stage name with a space in it
    "prepare abc",          # a non-numeric time
    "prepare 1.5",          # seconds are integers; $SECONDS never emits a float
    "",                     # a blank line
])
def test_a_malformed_line_is_skipped_not_counted(tmp_path: Path, line: str) -> None:
    """Stages append with >> from bash, so a partial write is possible. One bad
    line must not take the whole runtime with it."""
    f = tmp_path / ".stage_times"
    f.write_text(f"prepare 10\n{line}\ncollect 5\n")
    assert sst.total_seconds(f) == 15


@pytest.mark.local_unit
def test_a_stage_that_restarted_is_counted_once_per_attempt(tmp_path: Path) -> None:
    """A retried GPU task appends a second line. Both attempts really were
    paid for, so both count. This is deliberate, not an accident of parsing."""
    f = tmp_path / ".stage_times"
    f.write_text("predict_one_0 100\npredict_one_0 100\n")
    assert sst.total_seconds(f) == 200


@pytest.mark.local_unit
def test_main_writes_the_total_with_a_trailing_newline(tmp_path: Path) -> None:
    stage_times = tmp_path / ".stage_times"
    stage_times.write_text("prepare 7\nplan 8\n")
    out = tmp_path / "run_one_runtime_sec.txt"

    assert sst.main(["--stage-times", str(stage_times), "--output", str(out)]) == 0
    assert out.read_text() == "15\n"
    assert float(out.read_text().strip()) == 15.0


@pytest.mark.local_unit
def test_main_writes_a_zero_rather_than_nothing_when_the_file_is_missing(
        tmp_path: Path) -> None:
    """test_engine_smoke_run asserts run_one_runtime_sec.txt parses as a float.
    Writing no file at all would fail that instead of reporting a zero."""
    out = tmp_path / "run_one_runtime_sec.txt"
    assert sst.main(["--stage-times", str(tmp_path / "absent"),
                     "--output", str(out)]) == 0
    assert out.read_text() == "0\n"
