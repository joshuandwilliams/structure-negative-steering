"""The two plot modules, driven end to end.

negsteer_plots and negsteer_within_sequence_plots are 1,427 statements between
them and were entirely untested. They need no GPU. They read a cohort CSV and a
runs directory and write PNGs, all of which is reproducible from committed
fixtures.

The fixture is four real benchmark runs spanning every tier: 6G10 (B), 6G11 (C),
9IP6 (A) and 5A6W (none). A cohort with only one tier would leave the branches
that colour and group by tier unexercised.

Figures are asserted to exist and to be non-trivial. Pixel comparison is
deliberately avoided, since matplotlib output shifts between versions and would
make these fail for reasons unrelated to the code.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin"))

FIXTURES = Path(__file__).parent / "fixtures" / "cohort"
RUNS = FIXTURES / "runs"
COHORT_CSV = FIXTURES / "cross_sequence_summary.csv"

COHORT_PLOTS = _REPO_ROOT / "bin" / "negsteer_plots.py"
WITHIN_PLOTS = _REPO_ROOT / "bin" / "negsteer_within_sequence_plots.py"

# A PNG smaller than this is an axes frame with nothing drawn on it.
MIN_PNG_BYTES = 5_000

needs_fixture = pytest.mark.skipif(
    not COHORT_CSV.is_file(), reason="cohort fixture not present")


def _run(script: Path, *args: str) -> None:
    """Drive a plot CLI in process.

    Deliberately not a subprocess. These modules are 1,427 statements and a
    subprocess run exercises them without coverage being able to see it, which
    is how they came to look untested while working fine.
    """
    import negsteer_plots as npl
    import negsteer_within_sequence_plots as nwp

    mod = npl if script.name == "negsteer_plots.py" else nwp
    with mock.patch.object(sys, "argv", [script.name, *args]):
        rc = mod.main()
    assert rc in (0, None), f"{script.name} returned {rc}"


def _pngs(d: Path) -> list[Path]:
    return sorted(d.glob("*.png"))


# ── the cohort plots ─────────────────────────────────────────────────────────

@pytest.mark.local_integration
@needs_fixture
def test_cohort_plots_render_every_figure(tmp_path):
    _run(COHORT_PLOTS, "--csv", str(COHORT_CSV), "--runs-dir", str(RUNS),
         "--outdir", str(tmp_path))

    produced = {p.name for p in _pngs(tmp_path)}
    expected = {
        "negsteer_tier_landscape.png",
        "negsteer_seed_outcomes_heatmap.png",
        "negsteer_seed_outcomes_bars.png",
        "negsteer_ra_eff_vs_jaccard.png",
        "negsteer_filter_cascade.png",
        "negsteer_controls_diagnostic.png",
        "negsteer_mutation_impact.png",
    }
    assert expected <= produced, f"missing figures: {sorted(expected - produced)}"


@pytest.mark.local_integration
@needs_fixture
def test_cohort_plots_are_not_blank(tmp_path):
    """An empty-data path still writes a PNG, so size is the check that matters."""
    _run(COHORT_PLOTS, "--csv", str(COHORT_CSV), "--runs-dir", str(RUNS),
         "--outdir", str(tmp_path))
    for p in _pngs(tmp_path):
        assert p.stat().st_size > MIN_PNG_BYTES, \
            f"{p.name} is {p.stat().st_size} bytes, which is an empty plot"


@pytest.mark.local_integration
@needs_fixture
def test_cohort_plots_run_without_a_runs_dir(tmp_path):
    """The per-seed figures degrade rather than failing when runs/ is absent."""
    _run(COHORT_PLOTS, "--csv", str(COHORT_CSV), "--outdir", str(tmp_path))
    assert _pngs(tmp_path), "no figures at all were produced"


@pytest.mark.local_integration
@needs_fixture
@pytest.mark.parametrize("flag,value", [
    ("--complex-plddt-min", "0.8"),
    ("--ipae-max", "10"),
    ("--pae-pass-frac-min", "0.5"),
    ("--iptm-min", "0.4"),
])
def test_confidence_thresholds_are_accepted(tmp_path, flag, value):
    """Each threshold drives the filter-cascade figure."""
    _run(COHORT_PLOTS, "--csv", str(COHORT_CSV), "--runs-dir", str(RUNS),
         "--outdir", str(tmp_path), flag, value)
    assert (tmp_path / "negsteer_filter_cascade.png").is_file()


@pytest.mark.local_integration
@needs_fixture
def test_input_design_region_is_accepted(tmp_path):
    """The mutation-impact figure maps mutations onto the input PDB positions."""
    region = tmp_path / "region.txt"
    region.write_text("1,2,3,4,5\n")
    _run(COHORT_PLOTS, "--csv", str(COHORT_CSV), "--runs-dir", str(RUNS),
         "--outdir", str(tmp_path / "out"),
         "--input-design-region", str(region), "--pdb-length", "76")
    assert (tmp_path / "out" / "negsteer_mutation_impact.png").is_file()


# ── the within-sequence plots ────────────────────────────────────────────────

@pytest.mark.local_integration
@needs_fixture
def test_within_sequence_plots_render_every_figure(tmp_path):
    _run(WITHIN_PLOTS, "--runs-dir", str(RUNS),
         "--cross-summary-csv", str(COHORT_CSV), "--outdir", str(tmp_path))

    produced = {p.name for p in _pngs(tmp_path)}
    assert produced, "no figures produced"
    for name in ("negsteer_weighted_vs_true_jaccard.png",
                 "negsteer_composite_vs_confidence.png"):
        assert name in produced, f"missing {name}"


@pytest.mark.local_integration
@needs_fixture
def test_within_sequence_plots_are_not_blank(tmp_path):
    _run(WITHIN_PLOTS, "--runs-dir", str(RUNS),
         "--cross-summary-csv", str(COHORT_CSV), "--outdir", str(tmp_path))
    for p in _pngs(tmp_path):
        assert p.stat().st_size > MIN_PNG_BYTES, \
            f"{p.name} is {p.stat().st_size} bytes, which is an empty plot"


# ── the loaders, in process ──────────────────────────────────────────────────

@pytest.mark.local_integration
@needs_fixture
def test_cohort_loader_reads_every_sequence():
    import negsteer_plots as npl

    rows = npl.load_unified_cohort(COHORT_CSV, RUNS, False)
    assert len(rows) == 4, f"loaded {len(rows)} sequences, expected 4"
    tiers = {r.get("cross_tier") for r in rows}
    assert tiers == {"A", "B", "C", "none"}, f"fixture lost a tier: {tiers}"


@pytest.mark.local_integration
@needs_fixture
def test_seed_outcome_loader_classifies_every_row():
    import negsteer_plots as npl

    seeds = npl.load_seed_outcomes(RUNS, COHORT_CSV, True)
    assert seeds, "no per-seed rows loaded from the runs directory"
    for r in seeds:
        stage, outcome = npl.classify_seed(r)
        assert stage, f"row has no stage: {r.get('design')}"
        assert outcome, f"row has no outcome: {r.get('design')}"


@pytest.mark.local_integration
@needs_fixture
def test_within_sequence_loader_reads_the_runs():
    import negsteer_within_sequence_plots as nwp

    rows = nwp.load_cohort_seeds(RUNS, COHORT_CSV)
    assert rows, "no rows loaded"
    names = {nwp._design_id(r["sequence"]) if "sequence" in r else "" for r in rows}
    assert names, "no design identifiers derived"


# ── degenerate input ─────────────────────────────────────────────────────────

@pytest.mark.local_unit
def test_empty_plot_helper_still_writes_a_file(tmp_path):
    """Every figure has an empty-data path, so it must produce something."""
    import negsteer_plots as npl

    out = tmp_path / "empty.png"
    npl._make_empty_plot("nothing to show", str(out))
    assert out.is_file() and out.stat().st_size > 0


@pytest.mark.local_integration
@needs_fixture
def test_a_cohort_with_one_sequence_still_plots(tmp_path):
    """A single-row cohort exercises the degenerate axis-scaling paths."""
    import csv

    rows = list(csv.DictReader(COHORT_CSV.open()))
    single = tmp_path / "one.csv"
    with single.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerow(rows[0])

    _run(COHORT_PLOTS, "--csv", str(single), "--runs-dir", str(RUNS),
         "--outdir", str(tmp_path / "out"))
    assert _pngs(tmp_path / "out"), "a one-sequence cohort produced no figures"
