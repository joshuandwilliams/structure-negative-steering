"""The three entry points must agree on what compute-final-metrics is told.

There are three ways to run this engine, and each one passes its own flags to
`negsteer_aggregate.py compute-final-metrics`:

    bin/negative_steering_run_one.sh   the shell orchestrator, which produced
                                       the committed benchmark
    nextflow.config                    the workflow, for many targets
    negsteer/cli.py                    `negsteer run`, the tool interface

Two of those defaults had drifted, and the result was not an error. It was a
table that still looked like a table.

Run on the cluster against tests/test_run/config.yml, the orchestrator and the
CLI produced per-seed CSVs agreeing bit-for-bit on ra_eff and differing on 24
of 84 columns, with 23 of them blank on the CLI side: every confidence metric,
every ipSAE column, the interface pLDDT, the weighted Jaccard and the
contamination check. Either drift alone was enough to cause it.

    --metric-column   the CLI said receptor_aligned_effector_rmsd, which is
                      the per-design column name. compute-final-metrics
                      filters all_results_multicycle.csv, which carries
                      steered_ra_eff_vs_truth. The column was therefore absent,
                      _passes_final_metrics_filter read None for every row and
                      rejected all of them, and the stage exited 0.

    --populate-all    the orchestrator has it on, the workflow and the CLI had
                      it off, so the filter above was not bypassed either.

Nothing caught it. test_negsteer_cli.py compares the stage SEQUENCE, not the
flags, and test_single_run_cli_matches_orchestrator.py compared column names
rather than values until this was found.

So these assert on the defaults themselves, in the laptop tier, where they are
actually run. The reference is negsteer_aggregate.py's own argparse, because
that is the stage being configured.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from negsteer.cli import build_parser

REPO_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR = REPO_ROOT / "bin" / "negative_steering_run_one.sh"
NF_CONFIG = REPO_ROOT / "nextflow.config"
AGGREGATE = REPO_ROOT / "bin" / "negsteer_aggregate.py"

# The knobs where a disagreement changes what lands in the published table,
# rather than only how long the run takes.
TRACKED = ("metric_column", "populate_all")


MINIMAL = (
    "run", "--name", "x",
    "--receptor-fasta", "r", "--effector-fasta", "e",
    "--reference-pdb", "p", "--design-region", "d", "--true-interface", "t",
    "--outdir", "o",
)


def _parse_minimal(*extra: str) -> argparse.Namespace:
    """The shortest argv `negsteer run` accepts, so the result is the defaults."""
    return build_parser().parse_args([*MINIMAL, *extra])


def _aggregate_defaults() -> dict[str, object]:
    """compute-final-metrics' own argparse defaults, read from the source.

    Parsed as text rather than by importing: this module runs in the laptop
    tier, and importing the aggregator pulls in the whole stage. The two
    patterns below are anchored on `pcfm.add_argument`, which is the
    compute-final-metrics subparser and nothing else.
    """
    text = AGGREGATE.read_text()
    out: dict[str, object] = {}

    m = re.search(r'pcfm\.add_argument\(\s*"--metric-column",\s*'
                  r'default="([^"]+)"', text)
    assert m, "could not read compute-final-metrics' --metric-column default"
    out["metric_column"] = m.group(1)

    m = re.search(r'pcfm\.add_argument\(\s*"--populate-all",[^)]*?'
                  r'default=(True|False)', text, re.S)
    assert m, "could not read compute-final-metrics' --populate-all default"
    out["populate_all"] = m.group(1) == "True"
    return out


def _orchestrator_defaults() -> dict[str, object]:
    """run_one.sh's initial PP_* assignments, before the arg loop."""
    text = ORCHESTRATOR.read_text()
    out: dict[str, object] = {}

    m = re.search(r'^PP_METRIC_COLUMN="([^"]+)"', text, re.M)
    assert m, "run_one.sh no longer sets PP_METRIC_COLUMN"
    out["metric_column"] = m.group(1)

    m = re.search(r'^PP_POPULATE_ALL=([01])', text, re.M)
    assert m, "run_one.sh no longer sets PP_POPULATE_ALL"
    out["populate_all"] = m.group(1) == "1"
    return out


def _cli_defaults() -> dict[str, object]:
    """Read off the built parser, so this tracks the real argparse."""
    return {k: getattr(_parse_minimal(), k) for k in TRACKED}


@pytest.fixture(scope="module")
def defaults() -> dict[str, dict[str, object]]:
    return {"aggregate": _aggregate_defaults(),
            "orchestrator": _orchestrator_defaults(),
            "cli": _cli_defaults()}


@pytest.mark.logic
@pytest.mark.parametrize("path", ("orchestrator", "cli"))
def test_the_metric_column_matches_the_stage_default(defaults, path):
    """The column name is a value, so it is compared against the stage's own
    default, which is the column cmd_aggregate actually writes."""
    assert defaults[path]["metric_column"] == defaults["aggregate"]["metric_column"], (
        f"{path} filters on {defaults[path]['metric_column']!r}, but "
        f"compute-final-metrics defaults to "
        f"{defaults['aggregate']['metric_column']!r}. A column the table does "
        f"not have makes every row fail the filter.")


@pytest.mark.logic
@pytest.mark.parametrize("path", ("orchestrator", "cli"))
def test_every_entry_point_populates_every_row_by_default(defaults, path):
    """On, because that is what produced the committed benchmark.

    run_one.sh has had PP_POPULATE_ALL=1 from the start, and
    experiments/benchmarking/unconstrained/5A6W/run/launch.sh in the analyses
    repository passes no populate flag, so the benchmark inherited it. Its
    per-seed tables carry metrics on all 60 steered rows. The workflow and the
    CLI both defaulted off, which would have written tables the analyses read
    as absent data rather than as a measured null.
    """
    assert defaults[path]["populate_all"] is True, (
        f"{path} defaults populate_all off. compute-final-metrics then writes "
        f"confidence and interface metrics only on rows passing the ra_eff "
        f"and intact filter, and the holes are not distinguishable from a "
        f"measured zero downstream.")


@pytest.mark.logic
def test_the_workflow_populates_every_row_too(defaults):
    """nextflow.config is checked separately because the workflow has no
    argparse. modules/negsteer_stages.nf renders the flag from
    params.postprocess_populate_all, so the default is the config literal."""
    m = re.search(r'^\s*postprocess_populate_all\s*=\s*(true|false)',
                  NF_CONFIG.read_text(), re.M)
    assert m, "nextflow.config no longer declares postprocess_populate_all"
    assert m.group(1) == "true", (
        "the workflow defaults postprocess_populate_all off, so a cohort run "
        "would write tables with holes where the benchmark has values")


@pytest.mark.logic
def test_the_stage_itself_must_keep_defaulting_off(defaults):
    """The invariant that makes "off" expressible at all.

    All three entry points say off by OMITTING --populate-all:

        bin/negative_steering_run_one.sh   POPULATE_FLAG=() unless PP=1
        modules/negsteer_stages.nf         params ? '--populate-all' : ''
        negsteer/cli.py                    [...] if a.populate_all else []

    The flag is store_true, so there is no --no-populate-all on the stage.
    Flipping the stage's own default to True would therefore make every
    entry point's off switch silently do nothing, including run_one.sh's
    documented --no-postprocess-populate-all.
    """
    assert defaults["aggregate"]["populate_all"] is False, (
        "compute-final-metrics now defaults --populate-all on. Every entry "
        "point turns it off by omitting the flag, so none of them can turn it "
        "off any more. Give the stage an explicit --no-populate-all before "
        "changing this.")
    for path, pattern in (
            ("run_one.sh", r"POPULATE_FLAG=\(--populate-all\)"),
            ("negsteer_stages.nf", r"\?\s*'--populate-all'\s*:\s*''"),
            ("cli.py", r'\["--populate-all"\] if a\.populate_all else \[\]'),
    ):
        body = {"run_one.sh": ORCHESTRATOR,
                "negsteer_stages.nf": REPO_ROOT / "modules" / "negsteer_stages.nf",
                "cli.py": REPO_ROOT / "negsteer" / "cli.py"}[path].read_text()
        assert re.search(pattern, body), (
            f"{path} no longer passes --populate-all by omission, so the "
            f"assertion above no longer protects anything. Re-read it.")


@pytest.mark.logic
def test_the_metric_column_is_one_the_filtered_table_actually_has(defaults):
    """The fault that made this a blank table rather than a loud failure.

    compute-final-metrics filters all_results_multicycle.csv. The column it
    filters on has to be a column cmd_aggregate writes into that file, or every
    row fails on a missing key. Both committed fixtures carry the real header,
    so they are the check.
    """
    import csv

    col = defaults["aggregate"]["metric_column"]
    checked = 0
    for p in sorted((REPO_ROOT / "tests" / "characterization"
                     / "fixtures").rglob("all_results_multicycle.csv")):
        with p.open() as fh:
            header = next(csv.reader(fh))
        assert col in header, (
            f"{p.relative_to(REPO_ROOT)} has no column {col!r}, so the "
            f"default filter would reject every row of it. Columns with "
            f"'rmsd' or 'ra_eff': "
            + ", ".join(c for c in header if "rmsd" in c or "ra_eff" in c))
        checked += 1
    assert checked, "no all_results_multicycle.csv fixture, so this asserted nothing"


@pytest.mark.logic
def test_populate_all_can_still_be_turned_off_on_the_cli():
    """It is on by default, not forced. The cost is real on a big cohort, and
    the orchestrator has always had --no-postprocess-populate-all."""
    assert _parse_minimal("--no-populate-all").populate_all is False
    assert _parse_minimal().populate_all is True
    assert "--no-postprocess-populate-all" in ORCHESTRATOR.read_text()


# The guard that makes a wrong --metric-column loud rather than silent is
# asserted in test_final_metrics_stage.py, where the stage fixtures live:
# test_a_metric_column_the_table_lacks_is_refused.
