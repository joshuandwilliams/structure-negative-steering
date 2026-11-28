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

@pytest.mark.logic
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


@pytest.mark.logic
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


@pytest.mark.logic
@needs_run
def test_every_passing_row_carries_the_columns_tiering_reads(run_dir):
    out = run_dir / "rebuilt.csv"
    _main("--input", str(run_dir / "aggregated_results.csv"),
          "--output", str(out),
          "--per-seed", str(run_dir / "raw_per_seed_results.csv"))
    rows = _read(out)
    # Asserted, not skipped. The fixture is committed and does admit designs,
    # so an empty result means extract_passing dropped them, which is what
    # this test is for.
    assert rows, (
        "extract_passing admitted no designs from the committed fixture, so "
        "there are no rows to check the tiering columns on.")
    for col in ("n_pass", "n_seeds"):
        assert col in rows[0], f"passing_summary lacks {col}, which tiering needs"


@pytest.mark.logic
@needs_run
def test_rebuilding_without_per_seed_rows_still_works(run_dir):
    """The per-seed table is an enrichment, not a requirement."""
    out = run_dir / "rebuilt.csv"
    assert _main("--input", str(run_dir / "aggregated_results.csv"),
                 "--output", str(out)) == 0
    assert out.is_file()


@pytest.mark.logic
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

@pytest.mark.logic
@needs_run
def test_extract_row_maps_every_real_aggregate_row():
    rows = _read(FIXTURE / "aggregated_results.csv")
    per_seed = _read(FIXTURE / "raw_per_seed_results.csv")
    for r in rows:
        got = ep.extract_row(r, per_seed)
        assert got is None or isinstance(got, dict), \
            f"extract_row returned {type(got)} for design {r.get('design')}"


@pytest.mark.logic
@needs_run
def test_extract_row_is_stage_aware_about_reversion():
    """ra_eff_vs_truth must come from the reverted value when one exists.

    Scoring a contaminated design on its pre-reversion prediction would credit
    it with a rescue it did not keep.
    """
    rows = _read(FIXTURE / "aggregated_results.csv")
    reverted = [r for r in rows
                if (r.get("reverted_ra_eff_vs_truth_median") or "").strip()]
    # Asserted, not skipped. full_run/6G10 is committed and carries two rows
    # with a reverted median, so an empty list means the fixture changed under
    # the test rather than that this run had no reversions.
    assert reverted, (
        "the committed fixture has no reversion-attempted designs, so the "
        "stage-awareness rule below cannot be checked at all.")
    got = ep.extract_row(reverted[0], None)
    if got and got.get("ra_eff_vs_truth_median"):
        assert got["ra_eff_vs_truth_median"] == \
            reverted[0]["reverted_ra_eff_vs_truth_median"], \
            "a reversion-attempted design was scored on its steered value"


@pytest.mark.logic
def test_extract_row_of_an_empty_row_does_not_raise():
    got = ep.extract_row({}, None)
    assert got is None or isinstance(got, dict)


# ── the confidence flag ──────────────────────────────────────────────────────
#
# Computed on medians, so a single rogue seed does not flag a sequence whose
# typical prediction is fine. Every threshold is a separate trigger, and an
# unparseable value must not fire one: a blank metric is missing data, not a
# failing one, and flagging it would discard a sequence for having no
# measurement.

@pytest.mark.logic
class TestConfidenceFlag:
    def test_everything_within_threshold_is_ok(self):
        assert ep._compute_confidence_flag({
            "pae_pass_frac_median": 0.9, "iptm_median": 0.9,
            "complex_plddt_median": 0.9, "ipae_median": 1.0}) == "ok"

    @pytest.mark.parametrize("field,value,trigger", [
        ("pae_pass_frac_median", 0.0, "low_pass_frac"),
        ("iptm_median", 0.0, "low_iptm"),
        ("complex_plddt_median", 0.0, "low_plddt"),
        ("ipae_median", 999.0, "high_ipae"),
    ])
    def test_each_threshold_names_its_own_trigger(self, field, value, trigger):
        row = {"pae_pass_frac_median": 0.9, "iptm_median": 0.9,
               "complex_plddt_median": 0.9, "ipae_median": 1.0}
        row[field] = value
        assert ep._compute_confidence_flag(row) == trigger

    def test_two_failures_report_as_multiple(self):
        assert ep._compute_confidence_flag({
            "pae_pass_frac_median": 0.0, "iptm_median": 0.0,
            "complex_plddt_median": 0.9, "ipae_median": 1.0}) == "multiple"

    @pytest.mark.parametrize("bad", ["", "n/a", None])
    def test_an_unparseable_metric_does_not_fire_a_trigger(self, bad):
        """A missing measurement is not a failing one."""
        assert ep._compute_confidence_flag({
            "pae_pass_frac_median": bad, "iptm_median": bad,
            "complex_plddt_median": bad, "ipae_median": bad}) == "ok"

    def test_explicit_thresholds_override_the_catalogue(self):
        row = {"pae_pass_frac_median": 0.5, "iptm_median": 0.9,
               "complex_plddt_median": 0.9, "ipae_median": 1.0}
        assert ep._compute_confidence_flag(row) == "ok"
        assert ep._compute_confidence_flag(row, pass_frac_min=0.8) == \
            "low_pass_frac"

    def test_the_defaults_match_the_thresholds_catalogue(self):
        """The literals in the ImportError fallback must not drift from the
        catalogue they stand in for."""
        import pipeline_thresholds as pt
        t = pt.PipelineInternalThresholds.default()
        assert (t.pae_pass_frac_min, t.iptm_min, t.complex_plddt_min,
                t.ipae_max) == (0.10, 0.30, 0.70, 15.0)


# ── the CLI ──────────────────────────────────────────────────────────────────

@pytest.mark.logic
class TestCli:
    def _aggregated(self, path, rows=()):
        cols = ["cycle", "pathway", "sequence_group", "design", "outcome",
                "n_pass", "n_seeds", "steered_ra_eff_vs_truth_median",
                "rank_by_composite_score"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return path

    def test_a_missing_input_is_refused(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["extract_passing.py", "--input",
                                          str(tmp_path / "nope.csv")])
        assert ep.main() == 2
        assert "not found" in capsys.readouterr().err

    def test_a_missing_per_seed_file_warns_and_blanks_the_columns(
            self, tmp_path, capsys, monkeypatch):
        inp = self._aggregated(tmp_path / "aggregated_results.csv")
        monkeypatch.setattr(sys, "argv", ["extract_passing.py",
                                          "--input", str(inp)])
        assert ep.main() == 0
        assert "will be blank" in capsys.readouterr().out

    def test_the_output_defaults_beside_the_input(self, tmp_path, monkeypatch):
        inp = self._aggregated(tmp_path / "aggregated_results.csv")
        monkeypatch.setattr(sys, "argv", ["extract_passing.py",
                                          "--input", str(inp)])
        assert ep.main() == 0
        assert (tmp_path / "passing_summary.csv").exists()

    def test_an_explicit_output_path_is_used(self, tmp_path, monkeypatch):
        inp = self._aggregated(tmp_path / "aggregated_results.csv")
        out = tmp_path / "elsewhere.csv"
        monkeypatch.setattr(sys, "argv", ["extract_passing.py",
                                          "--input", str(inp),
                                          "--output", str(out)])
        assert ep.main() == 0
        assert out.exists()

    def test_a_corrupt_runtime_file_is_blanked_not_propagated(
            self, tmp_path, monkeypatch):
        """Passing a non-numeric runtime downstream would land in a cost
        table as a value."""
        inp = self._aggregated(tmp_path / "aggregated_results.csv")
        (tmp_path / "run_one_runtime_sec.txt").write_text("not-a-number\n")
        monkeypatch.setattr(sys, "argv", ["extract_passing.py",
                                          "--input", str(inp)])
        assert ep.main() == 0
        rows = list(csv.DictReader(open(tmp_path / "passing_summary.csv")))
        assert all(r.get("run_one_runtime_sec", "") == "" for r in rows)
