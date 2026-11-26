"""Tests for the compute-final-metrics stage, driven through its own CLI.

This is the stage that turns all_results_multicycle.csv into the table the
analyses read. It is opt-in and slow, one compute_metrics.py subprocess per
passing design, so it is also the stage most likely to be killed part way
through and resumed. Three guarantees make that safe, and all three are
asserted here.

The output is a superset of the input. Every row of all_results_multicycle.csv
appears in the output, filtered rows included, with blank metric columns. A
stage that emitted only the passing rows would quietly become the input to the
next analysis and lose the rest of the run.

The table on disk is readable after every computed row. A job killed mid-loop
leaves a valid CSV whose finished rows carry metrics, which is exactly what
--skip-existing reads back on the next attempt.

A failing row is a recorded failure, not a dead stage. The exit code is 1 when
any row failed and the scratch directory stays for inspection; on a clean run
it is removed.

The subprocess here is real, with `--python-executable` pointed at this
interpreter and `--compute-metrics-script` at a stub. Only the metric
arithmetic is stubbed, and test_pae_metrics.py covers that directly.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def agg():
    sys.path.insert(0, str(REPO_ROOT / "bin"))
    spec = importlib.util.spec_from_file_location(
        "negsteer_aggregate", REPO_ROOT / "bin" / "negsteer_aggregate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# The stub answers whatever it is asked, with values chosen so each flag
# threshold can be crossed one at a time.
STUB = """\
import csv, sys
argv = sys.argv[1:]
out = argv[argv.index("--output-csv") + 1]
pdir = argv[argv.index("--prediction-dir") + 1]
vals = dict(line.split("=", 1) for line in
            open(pdir + "/stub_values.txt").read().split())
fields = ["model_name", "iptm", "ptm", "complex_plddt", "ipsae_min",
          "pae_pass_frac", "ipae", "avg_plddt"]
rec = {k: vals.get(k, "") for k in fields}
rec["model_name"] = argv[argv.index("--model") + 1]
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerow(rec)
"""

STUB_FAILS_ON = """\
import csv, sys
argv = sys.argv[1:]
out = argv[argv.index("--output-csv") + 1]
pdir = argv[argv.index("--prediction-dir") + 1]
if "FAIL" in open(pdir + "/stub_values.txt").read():
    sys.stderr.write("stub refused this one\\n")
    sys.exit(4)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model_name", "iptm"])
    w.writeheader()
    w.writerow({"model_name": "boltz2", "iptm": "0.8"})
"""

BASE_COLUMNS = ["cycle", "design", "pathway", "pdb",
                "steered_ra_eff_vs_truth", "steered_receptor_intact",
                "reversion_dropped"]


def _pdb_text(rec=12, eff=7):
    lines = []
    n = 0
    for chain, count in (("A", rec), ("B", eff)):
        for i in range(1, count + 1):
            n += 1
            lines.append(
                f"ATOM  {n:5d}  CA  ALA {chain}{i:4d}    "
                f"{i:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 80.00\n")
    return "".join(lines)


class Run:
    """An experiment root laid out the way `aggregate` leaves one."""

    def __init__(self, root: Path, *, nested=True, plan_extra=None):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        plan = {"cycle": 0, "pred_receptor_chain": "A",
                "pred_effector_chain": "B"}
        if plan_extra:
            plan.update(plan_extra)
        pdir = root / "cycle_0" if nested else root
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "plan.json").write_text(json.dumps(plan))
        self.rows = []

    def design(self, name, *, ra=2.0, intact=1, dropped=0, cycle=0,
               values="iptm=0.8 ptm=0.7 complex_plddt=88 ipsae_min=0.6 "
                      "pae_pass_frac=0.9 ipae=3.0 avg_plddt=85",
               with_pdb=True):
        d = self.root / f"cycle_{cycle}" / name
        pdb = d / "prediction.pdb"
        if with_pdb:
            d.mkdir(parents=True, exist_ok=True)
            pdb.write_text(_pdb_text())
            (d / "stub_values.txt").write_text(values)
        self.rows.append({
            "cycle": str(cycle), "design": name,
            "pathway": f"cycle_{cycle}", "pdb": str(pdb),
            "steered_ra_eff_vs_truth": "" if ra is None else str(ra),
            "steered_receptor_intact": str(intact),
            "reversion_dropped": str(dropped)})
        return self

    def write(self):
        out = self.root / "all_results_multicycle.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=BASE_COLUMNS)
            w.writeheader()
            w.writerows(self.rows)
        return self

    @property
    def output(self):
        return self.root / "all_results_multicycle_with_metrics.csv"

    def read(self):
        return list(csv.DictReader(self.output.open()))


def stub(root: Path, body=STUB, name="stub_cm.py") -> Path:
    p = root / name
    p.write_text(body)
    return p


def invoke(agg, run: Run, *extra, script=None):
    argv = ["compute-final-metrics",
            "--experiment-root", str(run.root),
            "--compute-metrics-script",
            str(script if script is not None else stub(run.root.parent)),
            "--python-executable", sys.executable,
            *extra]
    args = agg.build_parser().parse_args(argv)
    return agg.cmd_compute_final_metrics(args)


# ── the refusals ─────────────────────────────────────────────────────────────

@pytest.mark.logic
def test_a_root_that_is_not_a_directory_is_refused(agg, tmp_path, capsys):
    run = Run(tmp_path / "exp")
    argv = ["compute-final-metrics", "--experiment-root",
            str(tmp_path / "nope")]
    assert agg.cmd_compute_final_metrics(
        agg.build_parser().parse_args(argv)) == 2
    assert "not a directory" in capsys.readouterr().err
    assert run.root.exists()


@pytest.mark.logic
def test_a_root_with_no_aggregate_output_is_refused(agg, tmp_path, capsys):
    """This stage reads what `aggregate` wrote. Running it first on an
    un-aggregated root would otherwise write an empty metrics table that
    looks like a finished run with nothing passing."""
    run = Run(tmp_path / "exp")
    assert invoke(agg, run) == 2
    assert "run `aggregate` first" in capsys.readouterr().err


@pytest.mark.logic
def test_a_missing_compute_metrics_script_is_refused(agg, tmp_path, capsys):
    run = Run(tmp_path / "exp").design("d0").write()
    assert invoke(agg, run, script=tmp_path / "absent.py") == 2
    assert "compute_metrics.py not found" in capsys.readouterr().err


@pytest.mark.logic
def test_a_root_with_no_plan_anywhere_is_refused(agg, tmp_path, capsys):
    run = Run(tmp_path / "exp").design("d0").write()
    (run.root / "cycle_0" / "plan.json").unlink()
    assert invoke(agg, run) == 2
    assert "no cycle_0 plan.json" in capsys.readouterr().err


@pytest.mark.logic
def test_a_single_cycle_run_keeps_its_plan_in_the_root(agg, tmp_path):
    """`--n-cycles 1` runs are flat rather than nested under cycle_0/."""
    run = Run(tmp_path / "exp", nested=False).design("d0").write()
    assert invoke(agg, run) == 0
    assert run.read()[0]["steered_iptm"] == "0.8"


# ── the output is a superset of the input ────────────────────────────────────

@pytest.mark.logic
def test_every_input_row_appears_in_the_output(agg, tmp_path):
    """Including the ones that were filtered out. This table replaces
    all_results_multicycle.csv downstream, so a short one loses designs."""
    run = (Run(tmp_path / "exp")
           .design("passes", ra=2.0)
           .design("too_far", ra=40.0)
           .design("broken", ra=1.0, intact=0)
           .design("dropped", ra=1.0, dropped=1)
           .write())
    assert invoke(agg, run) == 0
    rows = run.read()
    assert [r["design"] for r in rows] == ["passes", "too_far", "broken",
                                           "dropped"]


@pytest.mark.logic
def test_a_filtered_row_gets_blank_metrics_not_missing_columns(agg, tmp_path):
    run = (Run(tmp_path / "exp").design("passes").design("too_far", ra=40.0)
           .write())
    invoke(agg, run)
    far = [r for r in run.read() if r["design"] == "too_far"][0]
    assert far["steered_iptm"] == ""
    assert far["steered_error"] == ""
    assert far["steered_ipsae_min"] == ""


@pytest.mark.logic
def test_the_base_columns_survive_the_join(agg, tmp_path):
    run = Run(tmp_path / "exp").design("d0", ra=1.25).write()
    invoke(agg, run)
    row = run.read()[0]
    assert row["steered_ra_eff_vs_truth"] == "1.25"
    assert row["pathway"] == "cycle_0"


@pytest.mark.logic
def test_an_empty_input_table_writes_a_header_only_output(agg, tmp_path):
    run = Run(tmp_path / "exp").write()
    assert invoke(agg, run) == 0
    assert run.read() == []
    assert run.output.read_text().strip() != ""


# ── the filter and its reporting ─────────────────────────────────────────────

@pytest.mark.logic
def test_only_the_passing_rows_are_computed(agg, tmp_path, capsys):
    run = (Run(tmp_path / "exp").design("a").design("b", ra=40.0)
           .design("c").write())
    invoke(agg, run)
    out = capsys.readouterr().out
    assert "Read 3 rows" in out
    assert "Passing: 2" in out
    assert "computed now    : 2" in out
    assert "filtered out    : 1" in out


@pytest.mark.logic
def test_populate_all_reaches_every_row_including_the_dropped_ones(agg,
                                                                    tmp_path):
    """Per-seed aggregation takes a median across seeds. A dropped seed left
    without metrics is a hole in that column, which biases the median rather
    than widening it, so --populate-all overrides the verdict routing too."""
    run = (Run(tmp_path / "exp").design("far", ra=40.0)
           .design("broken", intact=0).design("dropped", dropped=1).write())
    assert invoke(agg, run, "--populate-all") == 0
    by = {r["design"]: r for r in run.read()}
    assert by["far"]["steered_iptm"] == "0.8"
    assert by["broken"]["steered_iptm"] == "0.8"
    assert by["dropped"]["steered_iptm"] == "0.8"


@pytest.mark.logic
def test_the_threshold_is_configurable(agg, tmp_path):
    run = Run(tmp_path / "exp").design("d0", ra=7.5).write()
    assert invoke(agg, run, "--rmsd-threshold", "10") == 0
    assert run.read()[0]["steered_iptm"] == "0.8"


@pytest.mark.logic
def test_the_filter_column_is_configurable(agg, tmp_path):
    """Different analyses rank on different metrics."""
    run = Run(tmp_path / "exp").design("d0", ra=40.0).write()
    assert invoke(agg, run, "--metric-column", "cycle",
                  "--rmsd-threshold", "1") == 0
    assert run.read()[0]["steered_iptm"] == "0.8"


@pytest.mark.logic
def test_the_intact_requirement_can_be_waived(agg, tmp_path):
    run = Run(tmp_path / "exp").design("d0", intact=0).write()
    assert invoke(agg, run, "--no-require-intact") == 0
    assert run.read()[0]["steered_iptm"] == "0.8"


@pytest.mark.logic
def test_a_skip_steering_run_includes_its_baseline(agg, tmp_path, capsys):
    """With no steered designs the cold-start row is the only row, so the
    ra_eff filter has nothing to select between and is not applied."""
    run = Run(tmp_path / "exp", plan_extra={"skip_steering": True})
    run.design("initial", ra=40.0).write()
    assert invoke(agg, run) == 0
    assert run.read()[0]["steered_iptm"] == "0.8"
    assert "skip_steering=true" in capsys.readouterr().out


# ── the confidence flag ──────────────────────────────────────────────────────
#
# These are the thresholds the ipSAE-instability analysis is about. The flag is derived after the
# loop so a resumed row and a fresh row are classified the same way.

@pytest.mark.logic
def test_a_confident_row_is_flagged_ok(agg, tmp_path):
    run = Run(tmp_path / "exp").design("d0").write()
    invoke(agg, run)
    assert run.read()[0]["steered_confidence_flag"] == "ok"


@pytest.mark.logic
@pytest.mark.parametrize("values,flag", [
    ("iptm=0.8 ipsae_min=0.6 pae_pass_frac=0.01 ipae=3.0 complex_plddt=88",
     "low_pass_frac"),
    ("iptm=0.1 ipsae_min=0.6 pae_pass_frac=0.9 ipae=3.0 complex_plddt=88",
     "low_iptm"),
    ("iptm=0.8 ipsae_min=0.6 pae_pass_frac=0.9 ipae=3.0 complex_plddt=0.2",
     "low_plddt"),
    ("iptm=0.8 ipsae_min=0.6 pae_pass_frac=0.9 ipae=40 complex_plddt=88",
     "high_ipae"),
])
def test_one_threshold_crossed_names_that_threshold(agg, tmp_path, values,
                                                    flag):
    run = Run(tmp_path / "exp").design("d0", values=values).write()
    invoke(agg, run)
    assert run.read()[0]["steered_confidence_flag"] == flag


@pytest.mark.logic
def test_two_thresholds_crossed_is_reported_as_multiple(agg, tmp_path):
    """The individual name would be misleading, because fixing that one
    metric would not make the row confident."""
    run = Run(tmp_path / "exp").design(
        "d0", values="iptm=0.1 ipsae_min=0.6 pae_pass_frac=0.01 ipae=3.0 "
                     "complex_plddt=88").write()
    invoke(agg, run)
    assert run.read()[0]["steered_confidence_flag"] == "multiple"


@pytest.mark.logic
def test_the_thresholds_are_configurable(agg, tmp_path):
    run = Run(tmp_path / "exp").design("d0").write()
    invoke(agg, run, "--flag-iptm-min", "0.95")
    assert run.read()[0]["steered_confidence_flag"] == "low_iptm"


@pytest.mark.logic
def test_a_row_with_no_metrics_has_no_flag(agg, tmp_path):
    """Blank, not "ok". A filtered row was never assessed, and calling it ok
    would put it in the confident bucket of every downstream count."""
    run = Run(tmp_path / "exp").design("far", ra=40.0).write()
    invoke(agg, run)
    assert run.read()[0]["steered_confidence_flag"] == ""


@pytest.mark.logic
def test_the_pae_cutoff_used_is_recorded_on_every_row(agg, tmp_path):
    """The ipSAE columns are meaningless without it, and the 10 Å and 15 Å
    variants are the subject of their own analysis."""
    run = Run(tmp_path / "exp").design("d0").design("far", ra=40.0).write()
    invoke(agg, run, "--pae-cutoff", "15")
    assert {r["steered_pae_cutoff_used"] for r in run.read()} == {"15.0"}


# ── failures ─────────────────────────────────────────────────────────────────

@pytest.mark.logic
def test_a_failing_row_does_not_stop_the_others(agg, tmp_path, capsys):
    run = (Run(tmp_path / "exp").design("bad", values="FAIL")
           .design("good", values="ok=1").write())
    assert invoke(agg, run, script=stub(tmp_path, STUB_FAILS_ON)) == 1
    by = {r["design"]: r for r in run.read()}
    assert "exited with 4" in by["bad"]["steered_error"]
    assert by["good"]["steered_iptm"] == "0.8"
    assert "failed          : 1" in capsys.readouterr().out


@pytest.mark.logic
def test_a_row_whose_prediction_is_gone_is_recorded(agg, tmp_path):
    """all_results_multicycle.csv outlives the predictions it names, because
    the Boltz outputs are the bulk of a run and get cleaned first."""
    run = Run(tmp_path / "exp").design("d0", with_pdb=False).write()
    assert invoke(agg, run) == 1
    assert "pdb not found" in run.read()[0]["steered_error"]


@pytest.mark.logic
def test_the_scratch_directory_is_kept_when_a_row_failed(agg, tmp_path):
    """The per-row CSVs are the only record of what compute_metrics.py saw."""
    run = Run(tmp_path / "exp").design("bad", values="FAIL").write()
    invoke(agg, run, script=stub(tmp_path, STUB_FAILS_ON))
    assert (run.root / "final_metrics_scratch").is_dir()


@pytest.mark.logic
def test_the_scratch_directory_is_removed_on_a_clean_run(agg, tmp_path):
    run = Run(tmp_path / "exp").design("d0").write()
    assert invoke(agg, run) == 0
    assert not (run.root / "final_metrics_scratch").exists()


# ── resume ───────────────────────────────────────────────────────────────────

@pytest.mark.logic
def test_the_table_is_readable_after_every_computed_row(agg, tmp_path):
    """A job killed mid-loop has to leave a valid CSV. The stub deletes its
    own prediction after the first row, so the second row fails and the
    first row's incremental write is what is on disk."""
    run = (Run(tmp_path / "exp").design("first").design("second").write())
    (run.root / "cycle_0" / "second" / "prediction.pdb").unlink()
    assert invoke(agg, run) == 1
    rows = run.read()
    assert rows[0]["steered_iptm"] == "0.8"
    assert "pdb not found" in rows[1]["steered_error"]


@pytest.mark.logic
def test_a_second_pass_reuses_the_first_passs_metrics(agg, tmp_path, capsys):
    run = Run(tmp_path / "exp").design("d0").write()
    assert invoke(agg, run, "--skip-existing") == 0
    # Break the stub so the second pass cannot possibly recompute.
    assert invoke(agg, run, "--skip-existing",
                  script=stub(tmp_path, "import sys\nsys.exit(9)\n",
                              name="broken.py")) == 0
    assert run.read()[0]["steered_iptm"] == "0.8"
    out = capsys.readouterr().out
    assert "Resume: 1 rows already have metrics" in out
    assert "resumed         : 1" in out


@pytest.mark.logic
def test_resume_is_on_by_default(agg, tmp_path, capsys):
    """The stage costs one subprocess per passing design, so the common case
    is finishing an interrupted run rather than starting one."""
    run = Run(tmp_path / "exp").design("d0").write()
    invoke(agg, run)
    capsys.readouterr()
    invoke(agg, run)
    assert "resumed         : 1" in capsys.readouterr().out


@pytest.mark.logic
def test_no_skip_existing_forces_a_recompute(agg, tmp_path, capsys):
    """Which is what makes a changed cutoff take effect rather than being
    silently resumed past."""
    run = Run(tmp_path / "exp").design("d0").write()
    invoke(agg, run)
    capsys.readouterr()
    invoke(agg, run, "--no-skip-existing")
    out = capsys.readouterr().out
    assert "computed now    : 1" in out
    assert "resumed         : 0" in out


@pytest.mark.logic
def test_a_resumed_row_is_flagged_on_this_runs_thresholds(agg, tmp_path):
    """The flag derivation runs after the loop precisely so resumed and
    fresh rows are classified the same way. Were it inside the loop, a
    resumed row would keep the previous run's flag."""
    run = Run(tmp_path / "exp").design("d0").write()
    invoke(agg, run, "--skip-existing")
    assert run.read()[0]["steered_confidence_flag"] == "ok"
    invoke(agg, run, "--skip-existing", "--flag-iptm-min", "0.95")
    assert run.read()[0]["steered_confidence_flag"] == "low_iptm"


@pytest.mark.logic
def test_an_explicit_output_path_is_honoured(agg, tmp_path):
    run = Run(tmp_path / "exp").design("d0").write()
    out = tmp_path / "elsewhere.csv"
    assert invoke(agg, run, "--output-csv", str(out)) == 0
    assert list(csv.DictReader(out.open()))[0]["steered_iptm"] == "0.8"
    assert not run.output.exists()
