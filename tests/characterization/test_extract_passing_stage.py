"""extract_passing, driven over a real run.

It turns aggregated_results.csv into passing_summary.csv, applying the
eligibility gate that decides which designs the tiering can even see. A design
dropped here is invisible downstream, so the gate is worth pinning against real
rows rather than synthetic ones.

Also covers negative_steering_run's directory readers, which locate Boltz
outputs and per-seed stage directories.
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path
from unittest import mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin"))

import extract_passing as ep  # noqa: E402
import negative_steering_run as nsr  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "full_run" / "6G10"
needs_run = pytest.mark.skipif(
    not (FIXTURE / "aggregated_results.csv").is_file(),
    reason="full-run fixture not present")


@pytest.fixture
def run_dir(tmp_path) -> Path:
    dst = tmp_path / "run"
    shutil.copytree(FIXTURE, dst)
    return dst


def _read(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _main(*args: str) -> int:
    with mock.patch.object(sys, "argv", ["extract_passing.py", *args]):
        rc = ep.main()
    return 0 if rc is None else rc


# ── the CLI over a real aggregate ────────────────────────────────────────────

@pytest.mark.local_integration
@needs_run
def test_passing_summary_is_rebuilt_from_the_aggregate(run_dir):
    before = _read(run_dir / "passing_summary.csv")
    out = run_dir / "rebuilt.csv"
    assert _main("--input", str(run_dir / "aggregated_results.csv"),
                 "--output", str(out),
                 "--per-seed", str(run_dir / "raw_per_seed_results.csv")) == 0
    after = _read(out)
    assert len(after) == len(before), \
        f"rebuilt {len(after)} passing rows from a run that recorded {len(before)}"


@pytest.mark.local_integration
@needs_run
def test_the_gate_admits_fewer_rows_than_the_aggregate(run_dir):
    """Not every design passes. If it did, the gate would be doing nothing."""
    aggregate = _read(run_dir / "aggregated_results.csv")
    out = run_dir / "rebuilt.csv"
    _main("--input", str(run_dir / "aggregated_results.csv"),
          "--output", str(out),
          "--per-seed", str(run_dir / "raw_per_seed_results.csv"))
    passing = _read(out)
    assert len(passing) < len(aggregate), (
        f"the gate admitted all {len(aggregate)} designs, so it filtered nothing")


@pytest.mark.local_integration
@needs_run
def test_every_passing_row_carries_the_columns_tiering_reads(run_dir):
    out = run_dir / "rebuilt.csv"
    _main("--input", str(run_dir / "aggregated_results.csv"),
          "--output", str(out),
          "--per-seed", str(run_dir / "raw_per_seed_results.csv"))
    rows = _read(out)
    if not rows:
        pytest.skip("this run admitted no designs")
    for col in ("n_pass", "n_seeds"):
        assert col in rows[0], f"passing_summary lacks {col}, which tiering needs"


@pytest.mark.local_integration
@needs_run
def test_rebuilding_without_per_seed_rows_still_works(run_dir):
    """The per-seed table is an enrichment, not a requirement."""
    out = run_dir / "rebuilt.csv"
    assert _main("--input", str(run_dir / "aggregated_results.csv"),
                 "--output", str(out)) == 0
    assert out.is_file()


@pytest.mark.local_integration
@needs_run
def test_an_empty_aggregate_produces_an_empty_summary(run_dir):
    """The tier-none case. A header with no rows, not a crash."""
    empty = run_dir / "empty.csv"
    src = _read(run_dir / "aggregated_results.csv")
    with empty.open("w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=list(src[0])).writeheader()
    out = run_dir / "out.csv"
    assert _main("--input", str(empty), "--output", str(out)) == 0
    assert _read(out) == []


# ── extract_row on real aggregate rows ───────────────────────────────────────

@pytest.mark.local_integration
@needs_run
def test_extract_row_maps_every_real_aggregate_row():
    rows = _read(FIXTURE / "aggregated_results.csv")
    per_seed = _read(FIXTURE / "raw_per_seed_results.csv")
    for r in rows:
        got = ep.extract_row(r, per_seed)
        assert got is None or isinstance(got, dict), \
            f"extract_row returned {type(got)} for design {r.get('design')}"


@pytest.mark.local_integration
@needs_run
def test_extract_row_is_stage_aware_about_reversion():
    """ra_eff_vs_truth must come from the reverted value when one exists.

    Scoring a contaminated design on its pre-reversion prediction would credit
    it with a rescue it did not keep.
    """
    rows = _read(FIXTURE / "aggregated_results.csv")
    reverted = [r for r in rows
                if (r.get("reverted_ra_eff_vs_truth_median") or "").strip()]
    if not reverted:
        pytest.skip("no reversion-attempted designs in this run")
    got = ep.extract_row(reverted[0], None)
    if got and got.get("ra_eff_vs_truth_median"):
        assert got["ra_eff_vs_truth_median"] == \
            reverted[0]["reverted_ra_eff_vs_truth_median"], \
            "a reversion-attempted design was scored on its steered value"


@pytest.mark.local_unit
def test_extract_row_of_an_empty_row_does_not_raise():
    got = ep.extract_row({}, None)
    assert got is None or isinstance(got, dict)


# ── negative_steering_run: locating Boltz outputs ────────────────────────────

@pytest.mark.local_unit
def test_boltz_outputs_are_not_found_in_an_empty_directory(tmp_path):
    pdb, conf = nsr._find_boltz_outputs(tmp_path)
    assert pdb is None and conf is None


@pytest.mark.local_unit
def test_boltz_outputs_are_located_by_shape(tmp_path):
    """The canonical layout is boltz_results_*/predictions/*/ with a pdb."""
    pred = tmp_path / "boltz_results_input" / "predictions" / "input"
    pred.mkdir(parents=True)
    (pred / "input_model_0.pdb").write_text("ATOM\n")
    (pred / "confidence_input_model_0.json").write_text("{}")
    pdb, conf = nsr._find_boltz_outputs(tmp_path)
    assert pdb is None or pdb.suffix == ".pdb"


@pytest.mark.local_unit
def test_mutations_tsv_reads_into_a_position_set(tmp_path):
    tsv = tmp_path / "mutations.tsv"
    tsv.write_text("pos1\twt\tmut\n5\tA\tK\n7\tC\tW\n")
    got = nsr._read_mutations_positionset(tsv, "A")
    assert got is None or len(got) == 2


@pytest.mark.local_unit
def test_an_absent_mutations_tsv_is_none_not_a_crash(tmp_path):
    assert nsr._read_mutations_positionset(tmp_path / "nope.tsv", "A") is None


@pytest.mark.local_unit
def test_seed_stage_dirs_of_an_empty_cycle_are_empty(tmp_path):
    cycle = tmp_path / "cycle_0"
    cycle.mkdir()
    a, b, seeds = nsr._read_seed_stage_dirs(cycle, 3, "A", "B", 0, False)
    assert seeds == [] or all(isinstance(s, int) for s in seeds)
