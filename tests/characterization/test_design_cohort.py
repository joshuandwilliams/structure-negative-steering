"""Unit tests for bin/design_cohort.py (Phase 4 Tier 6)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin"))

import design_cohort as dc  # noqa: E402
import negative_steering_run as nsr  # noqa: E402
import pipeline_thresholds as pt  # noqa: E402
import stage_result as sr  # noqa: E402
from position_set import PositionSet  # noqa: E402


def _fake_pred(ra_eff, intact_rmsd, contact_positions=()):
    m = MagicMock()
    m.ra_eff_vs.return_value = ra_eff
    m.independent_receptor_rmsd_vs.return_value = intact_rmsd
    cs = PositionSet(list(contact_positions), chain="A", frame="prediction")
    m.receptor_contact_residues.return_value = cs
    return m


def _fake_conf():
    return MagicMock()


def _truth():
    return MagicMock()


def _make_run(tmp_path, mpnn_id, row_type="steered", n_clean=3, n_seeds=3):
    """Build a NegativeSteeringRun on the cold-start path with `n_clean`
    of `n_seeds` clean predictions."""
    preds = []
    for i in range(n_seeds):
        if i < n_clean:
            preds.append(_fake_pred(2.0, 1.0))  # clean
        else:
            preds.append(_fake_pred(10.0, 1.0))  # fail_structural
    cs = sr.StageResult(
        stage_type="cold_start", config_id=mpnn_id,
        predictions=preds,
        confidences=[_fake_conf() for _ in preds],
        seed_indices=list(range(n_seeds)))
    return nsr.NegativeSteeringRun(
        mpnn_sequence_id=mpnn_id, workdir=tmp_path,
        row_type=row_type, cold_start=cs, truth=_truth())


# ── Construction ───────────────────────────────────────────────────


@pytest.mark.logic
class TestConstruction:
    def test_basic(self, tmp_path):
        cohort = dc.DesignCohort(runs=[
            _make_run(tmp_path, "design_03_seq_01"),
            _make_run(tmp_path, "design_03_seq_02"),
        ])
        assert len(cohort) == 2

    def test_duplicate_ids_raise(self, tmp_path):
        with pytest.raises(ValueError, match="duplicate"):
            dc.DesignCohort(runs=[
                _make_run(tmp_path, "x"),
                _make_run(tmp_path, "x"),
            ])

    def test_wrong_type_in_runs_raises(self):
        with pytest.raises(TypeError):
            dc.DesignCohort(runs=["not a NSR"])

    def test_iterator(self, tmp_path):
        cohort = dc.DesignCohort(runs=[
            _make_run(tmp_path, "a"), _make_run(tmp_path, "b"),
        ])
        ids = [r.mpnn_sequence_id for r in cohort]
        assert ids == ["a", "b"]


# ── Lookup ────────────────────────────────────────────────────────


@pytest.mark.logic
class TestLookup:
    def test_get_existing(self, tmp_path):
        a = _make_run(tmp_path, "a")
        cohort = dc.DesignCohort(runs=[a, _make_run(tmp_path, "b")])
        assert cohort.get("a") is a

    def test_get_missing_returns_none(self, tmp_path):
        cohort = dc.DesignCohort(runs=[_make_run(tmp_path, "a")])
        assert cohort.get("zzz") is None


# ── Row-type filtering ────────────────────────────────────────────


@pytest.mark.logic
class TestRowTypeFiltering:
    def test_steered_and_controls(self, tmp_path):
        cohort = dc.DesignCohort(runs=[
            _make_run(tmp_path, "s1", row_type="steered"),
            _make_run(tmp_path, "s2", row_type="steered"),
            _make_run(tmp_path, "cs", row_type="control_scrambled"),
            _make_run(tmp_path, "cp", row_type="control_polyA"),
        ])
        assert len(cohort.steered_runs()) == 2
        assert len(cohort.control_runs()) == 2
        assert cohort.n_steered() == 2
        assert cohort.n_controls() == 2


# ── Tier breakdown ────────────────────────────────────────────────


@pytest.mark.logic
class TestTierBreakdown:
    def test_full_breakdown(self, tmp_path):
        cohort = dc.DesignCohort(runs=[
            _make_run(tmp_path, "a", n_clean=3, n_seeds=3),  # tier A
            _make_run(tmp_path, "b", n_clean=2, n_seeds=3),  # tier B
            _make_run(tmp_path, "c", n_clean=1, n_seeds=3),  # tier C
            _make_run(tmp_path, "d", n_clean=0, n_seeds=3),  # tier none
        ])
        t = pt.PipelineInternalThresholds.default()
        breakdown = cohort.tier_breakdown(t)
        assert breakdown == {"A": 1, "B": 1, "C": 1, "none": 1}

    def test_excludes_controls(self, tmp_path):
        cohort = dc.DesignCohort(runs=[
            _make_run(tmp_path, "a", n_clean=3, row_type="steered"),
            _make_run(tmp_path, "cs", n_clean=3, row_type="control_scrambled"),
        ])
        t = pt.PipelineInternalThresholds.default()
        assert cohort.tier_breakdown(t) == {"A": 1, "B": 0, "C": 0, "none": 0}


# ── by_tier and survivors ─────────────────────────────────────────


@pytest.mark.logic
class TestByTierAndSurvivors:
    @pytest.fixture
    def cohort(self, tmp_path):
        return dc.DesignCohort(runs=[
            _make_run(tmp_path, "a", n_clean=3),  # A
            _make_run(tmp_path, "b", n_clean=2),  # B
            _make_run(tmp_path, "c", n_clean=1),  # C
            _make_run(tmp_path, "d", n_clean=0),  # none
        ])

    def test_by_tier(self, cohort):
        t = pt.PipelineInternalThresholds.default()
        a_runs = cohort.by_tier("A", t)
        assert len(a_runs) == 1
        assert a_runs[0].mpnn_sequence_id == "a"

    def test_survivors_are_abc(self, cohort):
        t = pt.PipelineInternalThresholds.default()
        survivors = cohort.survivors(t)
        ids = {r.mpnn_sequence_id for r in survivors}
        assert ids == {"a", "b", "c"}  # "d" (tier none) excluded


# ── Ranking ───────────────────────────────────────────────────────


@pytest.mark.logic
class TestRanking:
    def test_ranked_by_composite(self, tmp_path):
        cohort = dc.DesignCohort(runs=[
            _make_run(tmp_path, "tier_c", n_clean=1),  # C
            _make_run(tmp_path, "tier_a", n_clean=3),  # A
            _make_run(tmp_path, "tier_none", n_clean=0),  # none
            _make_run(tmp_path, "tier_b", n_clean=2),  # B
        ])
        t = pt.PipelineInternalThresholds.default()
        ranked = cohort.ranked_by_composite(t)
        # Order should be A, B, C, none
        assert [r.mpnn_sequence_id for r in ranked] == [
            "tier_a", "tier_b", "tier_c", "tier_none"
        ]


@pytest.mark.logic
class TestImmutability:
    def test_frozen(self, tmp_path):
        cohort = dc.DesignCohort(runs=[_make_run(tmp_path, "x")])
        with pytest.raises((AttributeError, Exception)):
            cohort.runs = ()  # type: ignore


# ── the two emitters and the directory factory ───────────────────────────────
#
# Both emitters hand a (name, csv path) list to cross_summary.aggregate. The
# name keys every row of the cross-sequence summary, so a duplicate silently
# overwrites a run and a wrongly derived one relabels it. aggregate itself is
# stubbed here: what is under test is the list these build.

@pytest.mark.logic
def test_to_cross_summary_csv_names_each_run_by_its_sequence_id(monkeypatch,
                                                                tmp_path):
    seen = {}
    import cross_summary
    monkeypatch.setattr(cross_summary, "aggregate",
                        lambda **kw: seen.update(kw) or 0)

    cohort = dc.DesignCohort(runs=(_make_run(tmp_path / "seqA", "seqA"),
                                   _make_run(tmp_path / "seqB", "seqB")))
    assert cohort.to_cross_summary_csv(tmp_path / "out.csv") == 0
    assert [n for n, _ in seen["sequences"]] == ["seqA", "seqB"]
    assert seen["sequences"][0][1].name == "passing_summary.csv"


def _seqdir(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "passing_summary.csv").write_text("x\n")
    return d


@pytest.mark.logic
def test_from_dirs_walks_one_subdirectory_per_sequence(monkeypatch, tmp_path):
    seen = {}
    import cross_summary
    monkeypatch.setattr(cross_summary, "aggregate",
                        lambda **kw: seen.update(kw) or 0)
    runs = tmp_path / "runs"
    for n in ("seqB", "seqA"):
        _seqdir(runs, n)
    (runs / "a_file.txt").write_text("not a dir")
    assert dc.DesignCohort.emit_cross_summary_from_dirs(
        runs_dirs=[runs], output_path=tmp_path / "o.csv") == 0
    assert [n for n, _ in seen["sequences"]] == ["seqA", "seqB"]


@pytest.mark.logic
def test_explicit_pairs_come_first_and_win_a_name_clash(monkeypatch, tmp_path):
    """A name seen twice must not produce two rows for one sequence."""
    seen = {}
    import cross_summary
    monkeypatch.setattr(cross_summary, "aggregate",
                        lambda **kw: seen.update(kw) or 0)
    runs = tmp_path / "runs"
    _seqdir(runs, "seqA")
    explicit = tmp_path / "elsewhere.csv"
    assert dc.DesignCohort.emit_cross_summary_from_dirs(
        runs_dirs=[runs], output_path=tmp_path / "o.csv",
        explicit_pairs=[("seqA", explicit), ("seqA", explicit)]) == 0
    assert seen["sequences"] == [("seqA", explicit)]


@pytest.mark.logic
def test_a_directory_that_does_not_exist_is_skipped(monkeypatch, tmp_path):
    seen = {}
    import cross_summary
    monkeypatch.setattr(cross_summary, "aggregate",
                        lambda **kw: seen.update(kw) or 0)
    runs = tmp_path / "runs"
    _seqdir(runs, "seqA")
    assert dc.DesignCohort.emit_cross_summary_from_dirs(
        runs_dirs=[runs, tmp_path / "nope"],
        output_path=tmp_path / "o.csv") == 0
    assert [n for n, _ in seen["sequences"]] == ["seqA"]


@pytest.mark.logic
def test_a_subdirectory_whose_name_is_not_a_sequence_id_is_skipped(
        monkeypatch, tmp_path):
    """The name becomes a column key, so only the safe character set passes."""
    seen = {}
    import cross_summary
    monkeypatch.setattr(cross_summary, "aggregate",
                        lambda **kw: seen.update(kw) or 0)
    runs = tmp_path / "runs"
    _seqdir(runs, "seqA")
    _seqdir(runs, "bad name!")
    assert dc.DesignCohort.emit_cross_summary_from_dirs(
        runs_dirs=[runs], output_path=tmp_path / "o.csv") == 0
    assert [n for n, _ in seen["sequences"]] == ["seqA"]


@pytest.mark.logic
def test_no_sequences_at_all_is_an_error_not_an_empty_summary(tmp_path):
    assert dc.DesignCohort.emit_cross_summary_from_dirs(
        runs_dirs=[tmp_path / "nope"], output_path=tmp_path / "o.csv") == 2


@pytest.mark.logic
def test_from_runs_directory_skips_files_and_unloadable_runs(monkeypatch,
                                                             tmp_path):
    runs = tmp_path / "runs"
    for n in ("seqA", "seqB"):
        _seqdir(runs, n)
    (runs / "stray.txt").write_text("x")

    def fake_from_workdir(workdir, mpnn_sequence_id, **kw):
        # seqB fails to load, which must drop it rather than the cohort.
        if mpnn_sequence_id == "seqB":
            return None
        return _make_run(workdir, mpnn_sequence_id)

    monkeypatch.setattr(nsr.NegativeSteeringRun, "from_workdir",
                        staticmethod(fake_from_workdir))
    cohort = dc.DesignCohort.from_runs_directory(runs)
    assert len(cohort.runs) == 1


@pytest.mark.logic
def test_repr_reports_both_counts(monkeypatch):
    cohort = dc.DesignCohort(runs=())
    monkeypatch.setattr(type(cohort), "n_steered", lambda self: 4)
    monkeypatch.setattr(type(cohort), "n_controls", lambda self: 2)
    r = repr(cohort)
    assert "n_steered=4" in r and "n_controls=2" in r
