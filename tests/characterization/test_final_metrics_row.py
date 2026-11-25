"""Tests for compute-final-metrics' per-row worker and its passing filter.

`_process_passing_row` is the only place in the pipeline that attaches
confidence metrics to a result row. It shells out to compute_metrics.py once
per passing design, so every failure mode of that call has to land as a
recorded `steered_error` on the row rather than as an exception. A row that
raises takes the whole stage down after hours of per-row subprocess work; a
row that returns silently blank looks like a design with no confidence
metrics, which is indistinguishable from a design the stage never reached.

The subprocess is real here. `python_executable` points at this interpreter
and `compute_metrics_script` at a stub that writes whatever CSV the test
wants, so the argv construction, the timeout, the non-zero exit and the CSV
read-back are all exercised through the same subprocess.run the pipeline uses.
What the stub stands in for is only the metric arithmetic, which
test_pae_metrics.py covers directly.

Two details in the argv are load-bearing and have bug history. The mutation
set is read from `steered_mutations_chimerax`, the post-rename column; reading
only the legacy name blanked the mutated-contact columns on every row. And the
sidecar lookup points `--prediction-dir` at Boltz's nested output rather than
at prediction.pdb's parent, without which ptm, iptm and PAE all come back
zero and only pLDDT survives via the B-factor fallback.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

METRIC_KEYS = ["iptm", "ptm", "complex_plddt", "ipsae_min"]
PREFIX = "steered_"
PREFIXED = [PREFIX + k for k in METRIC_KEYS]


@pytest.fixture(scope="module")
def agg():
    sys.path.insert(0, str(REPO_ROOT / "bin"))
    spec = importlib.util.spec_from_file_location(
        "negsteer_aggregate", REPO_ROOT / "bin" / "negsteer_aggregate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class Args:
    """Only the attributes _process_passing_row reads."""

    def __init__(self, **kw):
        self.skip_existing = False
        self.pae_cutoff = 10.0
        self.contact_cutoff = 4.5
        self.subprocess_timeout = 60
        self.verbose = False
        self.python_executable = sys.executable
        self.populate_all = False
        self.require_intact = True
        self.__dict__.update(kw)


# ── stub compute_metrics scripts ─────────────────────────────────────────────
#
# Each writes to the --output-csv it is handed. The worker passes that flag
# positionally-free, so the stubs parse argv rather than assuming an order.

ARGV_PREAMBLE = """\
import csv, json, sys
argv = sys.argv[1:]
def flag(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default
out = flag("--output-csv")
"""

STUB_OK = ARGV_PREAMBLE + """
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model_name", "iptm", "ptm",
                                      "complex_plddt", "ipsae_min"])
    w.writeheader()
    w.writerow({"model_name": "pred_model_0", "iptm": "0.81", "ptm": "0.74",
                "complex_plddt": "88.2", "ipsae_min": "0.63"})
"""

STUB_RECORD_ARGV = ARGV_PREAMBLE + """
json.dump(argv, open(str(out) + ".argv.json", "w"))
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model_name", "iptm"])
    w.writeheader()
    w.writerow({"model_name": "pred_model_0", "iptm": "0.5"})
"""

STUB_FAILS = """\
import sys
sys.stderr.write("the container was not found\\n")
sys.exit(3)
"""

STUB_WRITES_NOTHING = "import sys\nsys.exit(0)\n"

STUB_WRITES_HEADER_ONLY = ARGV_PREAMBLE + """
open(out, "w").write("model_name,iptm\\n")
"""

STUB_WRITES_RUBBISH = ARGV_PREAMBLE + """
import os
os.mkdir(out)
"""

STUB_HANGS = "import time\ntime.sleep(30)\n"

STUB_TWO_MODELS = ARGV_PREAMBLE + """
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model_name", "iptm"])
    w.writeheader()
    w.writerow({"model_name": "someone_elses_model", "iptm": "0.11"})
    w.writerow({"model_name": "pred_model_0", "iptm": "0.99"})
"""


@pytest.fixture
def tree(tmp_path):
    """A design directory with a Boltz-shaped prediction, plus a cycle-0 plan.

    The nested `predictions/<stem>/pdb/<stem>_model_0.pdb` layout is what the
    real engine leaves behind, and the sidecar JSON/NPZ files sit beside the
    nested copy rather than beside prediction.pdb.
    """
    design = tmp_path / "designs" / "d0"
    nested = design / "predictions" / "pred" / "pdb"
    nested.mkdir(parents=True)
    pdb = _two_chain_pdb(rec=12, eff=7)
    (nested / "pred_model_0.pdb").write_text(pdb)
    (nested / "confidence_pred_model_0.json").write_text("{}")
    (design / "prediction.pdb").write_text(pdb)

    truth = tmp_path / "truth.pdb"
    truth.write_text(pdb)
    plan_path = tmp_path / "cycle_0" / "plan.json"
    plan_path.parent.mkdir()
    plan = {"cycle": 0, "ground_truth": str(truth),
            "effector_interface_atoms": {"B": [1, 2]}}
    plan_path.write_text(json.dumps(plan))

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return {"root": tmp_path, "design": design, "scratch": scratch,
            "plan": plan, "plan_path": plan_path, "truth": truth}


def _two_chain_pdb(*, rec: int, eff: int) -> str:
    lines = []
    n = 0
    for chain, count in (("A", rec), ("B", eff)):
        for i in range(1, count + 1):
            n += 1
            lines.append(
                f"ATOM  {n:5d}  CA  ALA {chain}{i:4d}    "
                f"{i:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 80.00\n")
    return "".join(lines)


def stub(root: Path, body: str, name="stub_cm.py") -> Path:
    p = root / name
    p.write_text(body)
    return p


def row(**kw):
    base = {"cycle": "0", "design": "d0", "pathway": "cycle_0",
            "receptor_aligned_effector_rmsd_vs_truth": "2.0"}
    base.update(kw)
    return base


def run(agg, tree, script, *, args=None, r=None, existing=None,
        rec_chain="A", eff_chain="B"):
    return agg._process_passing_row(
        row=r if r is not None else row(pdb=str(tree["design"]
                                               / "prediction.pdb")),
        args=args or Args(),
        prefixed_metric_keys=PREFIXED,
        metric_keys_raw=METRIC_KEYS,
        metric_key_prefix=PREFIX,
        existing_by_key=existing or {},
        scratch_dir=tree["scratch"],
        compute_metrics_script=script,
        cycle_0_plan=tree["plan"],
        cycle_0_plan_path=tree["plan_path"],
        pred_rec_chain=rec_chain,
        pred_eff_chain=eff_chain,
        model_name="boltz2",
        metric_col="receptor_aligned_effector_rmsd_vs_truth",
        passing_progress=1,
        n_passing_total=1,
    )


# ── the successful path ──────────────────────────────────────────────────────

@pytest.mark.logic
def test_the_metrics_land_on_the_row_under_the_steered_prefix(agg, tree):
    out, status = run(agg, tree, stub(tree["root"], STUB_OK))
    assert status == "computed"
    assert out["steered_iptm"] == "0.81"
    assert out["steered_ptm"] == "0.74"
    assert out["steered_complex_plddt"] == "88.2"
    assert out["steered_ipsae_min"] == "0.63"
    assert out["steered_error"] == ""


@pytest.mark.logic
def test_the_base_columns_are_carried_through_untouched(agg, tree):
    r = row(pdb=str(tree["design"] / "prediction.pdb"), mutations_aa="A5W")
    out, _ = run(agg, tree, stub(tree["root"], STUB_OK), r=r)
    assert out["design"] == "d0"
    assert out["mutations_aa"] == "A5W"


@pytest.mark.logic
def test_the_row_matching_the_prediction_stem_is_chosen(agg, tree):
    """A CSV with more than one model must not be read positionally."""
    out, _ = run(agg, tree, stub(tree["root"], STUB_TWO_MODELS))
    assert out["steered_iptm"] == "0.99"


@pytest.mark.logic
def test_an_unrecognised_stem_falls_back_to_the_first_row(agg, tree):
    """Better a metric from the only row present than a blank column."""
    body = STUB_TWO_MODELS.replace("pred_model_0", "nothing_matches")
    out, status = run(agg, tree, stub(tree["root"], body))
    assert status == "computed"
    assert out["steered_iptm"] == "0.11"


# ── what gets handed to compute_metrics.py ───────────────────────────────────

@pytest.mark.logic
def test_the_prediction_dir_points_at_the_nested_sidecars(agg, tree):
    """prediction.pdb's own directory has no confidence JSON or PAE next to
    it, so pointing there returns ptm=0, iptm=0, pae=0 with only pLDDT
    surviving through the B-factor fallback."""
    argv = _argv_of(agg, tree)
    pred_dir = Path(argv[argv.index("--prediction-dir") + 1])
    assert (pred_dir / "confidence_pred_model_0.json").exists()
    assert pred_dir != tree["design"]


@pytest.mark.logic
def test_a_prediction_with_no_nested_copy_uses_its_own_directory(agg, tree):
    """Not every prediction came out of Boltz's nested layout."""
    flat = tree["root"] / "flat"
    flat.mkdir()
    (flat / "prediction.pdb").write_text(_two_chain_pdb(rec=12, eff=7))
    argv = _argv_of(agg, tree, pdb=flat / "prediction.pdb")
    assert argv[argv.index("--prediction-dir") + 1] == str(flat)


@pytest.mark.logic
def test_the_chain_lengths_are_counted_off_the_pdb_that_is_parsed(agg, tree):
    """Handing compute_metrics.py the wrong lengths misaligns every PAE
    block, so the numbers come back plausible and wrong."""
    argv = _argv_of(agg, tree)
    i = argv.index("--chain-lengths")
    assert argv[i + 1:i + 3] == ["12", "7"]


@pytest.mark.logic
def test_the_post_rename_mutation_column_is_forwarded(agg, tree):
    """Bug history. Reading only `mutations_chimerax`, which aggregate
    renames away, ran every row without --mutated-positions and blanked the
    mutated-contact columns across the whole table."""
    argv = _argv_of(agg, tree, extra={
        "steered_mutations_chimerax": "/A:5,26,44"})
    assert argv[argv.index("--mutated-positions") + 1] == "/A:5,26,44"


@pytest.mark.logic
def test_the_legacy_mutation_column_is_still_read(agg, tree):
    """For any path that bypasses the rename."""
    argv = _argv_of(agg, tree, extra={"mutations_chimerax": "/A:9"})
    assert argv[argv.index("--mutated-positions") + 1] == "/A:9"


@pytest.mark.logic
def test_the_post_rename_column_wins_when_both_are_present(agg, tree):
    argv = _argv_of(agg, tree, extra={
        "steered_mutations_chimerax": "/A:5", "mutations_chimerax": "/A:99"})
    assert argv[argv.index("--mutated-positions") + 1] == "/A:5"


@pytest.mark.logic
def test_a_cold_start_passes_no_mutation_flag(agg, tree):
    """A blank set is not the same as an empty string argument."""
    argv = _argv_of(agg, tree, extra={"steered_mutations_chimerax": ""})
    assert "--mutated-positions" not in argv


@pytest.mark.logic
def test_the_ground_truth_comes_from_the_cycle_zero_plan(agg, tree):
    argv = _argv_of(agg, tree)
    assert argv[argv.index("--native-pdb") + 1] == str(tree["truth"])


@pytest.mark.logic
def test_a_plan_with_no_ground_truth_omits_the_flag(agg, tree):
    """The native-PDB metrics come out blank rather than the row failing."""
    tree["plan"] = {"cycle": 0}
    argv = _argv_of(agg, tree)
    assert "--native-pdb" not in argv


@pytest.mark.logic
def test_the_effector_atom_filter_is_the_plan_itself(agg, tree):
    """compute_metrics.py reads the key out of plan.json, so there is no
    materialised filter file to go stale."""
    argv = _argv_of(agg, tree)
    assert argv[argv.index("--effector-atom-filter-json") + 1] == str(
        tree["plan_path"])


@pytest.mark.logic
def test_the_chain_names_and_cutoffs_are_passed_through(agg, tree):
    argv = _argv_of(agg, tree, args=Args(pae_cutoff=12.5,
                                         contact_cutoff=5.0))
    assert argv[argv.index("--receptor-chain") + 1] == "A"
    assert argv[argv.index("--effector-chain") + 1] == "B"
    assert argv[argv.index("--pae-cutoff") + 1] == "12.5"
    assert argv[argv.index("--contact-cutoff") + 1] == "5.0"


@pytest.mark.logic
def test_no_reference_pdb_is_requested(agg, tree):
    """Structural RMSDs come from all_results_multicycle.csv, which already
    uses the correct truth-to-prediction chain mapping. Asking for them here
    would produce a second set under a different mapping."""
    assert "--reference-pdb" not in _argv_of(agg, tree)


def _argv_of(agg, tree, *, pdb=None, extra=None, args=None):
    r = row(pdb=str(pdb or tree["design"] / "prediction.pdb"))
    if extra:
        r.update(extra)
    run(agg, tree, stub(tree["root"], STUB_RECORD_ARGV), r=r, args=args)
    hits = sorted(tree["scratch"].glob("*.argv.json"))
    assert len(hits) == 1, f"expected one recorded call, got {hits}"
    return json.loads(hits[0].read_text())


# ── chain lengths that cannot be determined ──────────────────────────────────

@pytest.mark.logic
def test_unnamed_chains_fall_back_to_the_two_largest(agg, tree):
    """Some predictions do not use the plan's chain letters."""
    argv = _argv_of(agg, tree)      # warm the scratch dir layout
    assert argv  # sanity
    (tree["scratch"] / "cycle0_d0.csv.argv.json").unlink()
    r = row(pdb=str(tree["design"] / "prediction.pdb"))
    out, status = agg._process_passing_row(
        row=r, args=Args(), prefixed_metric_keys=PREFIXED,
        metric_keys_raw=METRIC_KEYS, metric_key_prefix=PREFIX,
        existing_by_key={}, scratch_dir=tree["scratch"],
        compute_metrics_script=stub(tree["root"], STUB_RECORD_ARGV),
        cycle_0_plan=tree["plan"], cycle_0_plan_path=tree["plan_path"],
        pred_rec_chain="X", pred_eff_chain="Y", model_name="boltz2",
        metric_col="receptor_aligned_effector_rmsd_vs_truth",
        passing_progress=1, n_passing_total=1)
    assert status == "computed"
    argv = json.loads(
        (tree["scratch"] / "cycle0_d0.csv.argv.json").read_text())
    i = argv.index("--chain-lengths")
    assert argv[i + 1:i + 3] == ["12", "7"]


@pytest.mark.logic
def test_a_single_chain_prediction_is_recorded_as_a_failure(agg, tree, capsys):
    """One chain means there is no interface, so every PAE-derived metric is
    undefined. That is a result about the prediction, not a crash."""
    solo = tree["root"] / "solo"
    solo.mkdir()
    (solo / "prediction.pdb").write_text(_two_chain_pdb(rec=9, eff=0))
    out, status = run(agg, tree, stub(tree["root"], STUB_OK),
                      r=row(pdb=str(solo / "prediction.pdb")))
    assert status == "failed"
    assert "could not determine chain lengths" in out["steered_error"]
    assert out["steered_iptm"] == ""


# ── failures that must land on the row ───────────────────────────────────────

@pytest.mark.logic
def test_a_missing_pdb_is_recorded_rather_than_raised(agg, tree):
    out, status = run(agg, tree, stub(tree["root"], STUB_OK),
                      r=row(pdb=str(tree["root"] / "gone.pdb")))
    assert status == "failed"
    assert out["steered_error"].startswith("pdb not found")


@pytest.mark.logic
def test_a_blank_pdb_column_is_recorded(agg, tree):
    out, status = run(agg, tree, stub(tree["root"], STUB_OK), r=row(pdb=""))
    assert status == "failed"
    assert "pdb not found" in out["steered_error"]


@pytest.mark.logic
def test_a_non_zero_exit_keeps_the_stderr_tail(agg, tree):
    """The stage runs unattended, so the reason has to be on the row."""
    out, status = run(agg, tree, stub(tree["root"], STUB_FAILS))
    assert status == "failed"
    assert "exited with 3" in out["steered_error"]
    assert "container was not found" in out["steered_error"]


@pytest.mark.logic
def test_a_missing_script_is_recorded_rather_than_raised(agg, tree):
    out, status = run(agg, tree, tree["root"] / "not_here.py")
    assert status == "failed"
    assert out["steered_error"]


@pytest.mark.logic
def test_a_timeout_is_recorded_with_its_limit(agg, tree):
    args = Args(subprocess_timeout=1)
    out, status = run(agg, tree, stub(tree["root"], STUB_HANGS), args=args)
    assert status == "failed"
    assert "timed out after 1s" in out["steered_error"]


@pytest.mark.logic
def test_a_script_that_writes_no_csv_is_recorded(agg, tree):
    out, status = run(agg, tree, stub(tree["root"], STUB_WRITES_NOTHING))
    assert status == "failed"
    assert "produced no CSV" in out["steered_error"]


@pytest.mark.logic
def test_an_empty_csv_is_recorded(agg, tree):
    out, status = run(agg, tree, stub(tree["root"], STUB_WRITES_HEADER_ONLY))
    assert status == "failed"
    assert "CSV was empty" in out["steered_error"]


@pytest.mark.logic
def test_an_unreadable_csv_is_recorded(agg, tree):
    out, status = run(agg, tree, stub(tree["root"], STUB_WRITES_RUBBISH))
    assert status == "failed"
    assert "could not read" in out["steered_error"]


@pytest.mark.logic
def test_every_metric_column_exists_even_on_a_failure(agg, tree):
    """A missing column and a blank one are different downstream. The CSV
    writer takes its fieldnames from the rows, so a row short of a column
    would drop it for the whole table."""
    out, _ = run(agg, tree, stub(tree["root"], STUB_FAILS))
    assert all(k in out for k in PREFIXED)


# ── resume ───────────────────────────────────────────────────────────────────

@pytest.mark.logic
def test_resume_takes_the_cached_metrics_without_running_anything(agg, tree):
    key = str(tree["design"] / "prediction.pdb")
    cached = {"pdb": key, "steered_iptm": "0.42", "steered_ptm": "0.40",
              "steered_complex_plddt": "70.0", "steered_ipsae_min": "0.30"}
    out, status = run(agg, tree, tree["root"] / "deliberately_absent.py",
                      args=Args(skip_existing=True),
                      existing={key: cached})
    assert status == "resumed"
    assert out["steered_iptm"] == "0.42"


@pytest.mark.logic
def test_resume_keeps_the_fresh_base_row_not_the_cached_one(agg, tree):
    """The cached CSV predates whatever columns this run added, so overlaying
    the cached metrics onto the fresh row is the only safe direction. The
    other way round silently reinstates stale base columns."""
    key = str(tree["design"] / "prediction.pdb")
    cached = {"pdb": key, "design": "an_old_name", "steered_iptm": "0.42"}
    r = row(pdb=key, rank_by_truth="3")
    out, status = run(agg, tree, tree["root"] / "absent.py",
                      args=Args(skip_existing=True), r=r, existing={key: cached})
    assert status == "resumed"
    assert out["design"] == "d0"
    assert out["rank_by_truth"] == "3"
    assert out["steered_iptm"] == "0.42"


@pytest.mark.logic
def test_a_blank_cached_metric_does_not_overwrite(agg, tree):
    key = str(tree["design"] / "prediction.pdb")
    r = row(pdb=key, steered_ptm="0.9")
    out, _ = run(agg, tree, tree["root"] / "absent.py",
                 args=Args(skip_existing=True), r=r,
                 existing={key: {"pdb": key, "steered_iptm": "0.42",
                                 "steered_ptm": ""}})
    assert out["steered_ptm"] == "0.9"


@pytest.mark.logic
def test_a_cached_failure_reason_is_preserved(agg, tree):
    """Otherwise a resumed run looks like it never attempted the row."""
    key = str(tree["design"] / "prediction.pdb")
    out, _ = run(agg, tree, tree["root"] / "absent.py",
                 args=Args(skip_existing=True),
                 existing={key: {"pdb": key, "steered_iptm": "0.42",
                                 "steered_error": "pdb not found"}})
    assert out["steered_error"] == "pdb not found"


@pytest.mark.logic
def test_without_skip_existing_the_cache_is_ignored(agg, tree):
    key = str(tree["design"] / "prediction.pdb")
    out, status = run(agg, tree, stub(tree["root"], STUB_OK),
                      existing={key: {"pdb": key, "steered_iptm": "0.42"}})
    assert status == "computed"
    assert out["steered_iptm"] == "0.81"


# ── the resume key ───────────────────────────────────────────────────────────

@pytest.mark.logic
def test_the_resume_key_is_the_pdb_path(agg):
    """Not `pathway`. Every cycle-0 design gets the literal string "cycle_0"
    from aggregate, so keying on it collides across the whole first cycle and
    resumes every design from one cached row."""
    assert agg._resume_key_for_row(
        {"pdb": "/p/d0/prediction.pdb", "pathway": "cycle_0"}
    ) == "/p/d0/prediction.pdb"


@pytest.mark.logic
def test_the_key_falls_back_to_the_pathway_then_to_empty(agg):
    assert agg._resume_key_for_row({"pathway": "cycle_2.d1"}) == "cycle_2.d1"
    assert agg._resume_key_for_row({}) == ""


# ── loading the previous output ──────────────────────────────────────────────

@pytest.mark.logic
def test_a_row_with_no_metrics_is_not_treated_as_done(agg, tmp_path):
    """It is a row the previous run reached and failed on, or never reached.
    Either way there is nothing to resume from."""
    out = tmp_path / "metrics.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pdb"] + PREFIXED)
        w.writeheader()
        w.writerow({"pdb": "/p/a.pdb", "steered_iptm": "0.8"})
        w.writerow({"pdb": "/p/b.pdb"})
    loaded = agg._load_existing_metrics_rows(out, PREFIXED)
    assert set(loaded) == {"/p/a.pdb"}


@pytest.mark.logic
def test_one_populated_metric_is_enough(agg, tmp_path):
    out = tmp_path / "metrics.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pdb"] + PREFIXED)
        w.writeheader()
        w.writerow({"pdb": "/p/a.pdb", "steered_ipsae_min": "0.1"})
    assert set(agg._load_existing_metrics_rows(out, PREFIXED)) == {"/p/a.pdb"}


@pytest.mark.logic
def test_a_row_with_no_key_is_dropped(agg, tmp_path):
    out = tmp_path / "metrics.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pdb"] + PREFIXED)
        w.writeheader()
        w.writerow({"pdb": "", "steered_iptm": "0.8"})
    assert agg._load_existing_metrics_rows(out, PREFIXED) == {}


@pytest.mark.logic
def test_no_previous_output_is_an_empty_cache(agg, tmp_path):
    assert agg._load_existing_metrics_rows(tmp_path / "nope.csv",
                                           PREFIXED) == {}


# ── which rows get metrics at all ────────────────────────────────────────────

@pytest.mark.logic
def test_a_dropped_row_is_skipped_by_default(agg):
    """Confidence metrics on a prediction reversion already discarded are
    wasted subprocess time."""
    r = {"reversion_dropped": "1", "steered_receptor_intact": "1",
         "ra": "1.0"}
    assert agg._passes_final_metrics_filter(r, Args(), "ra", 5.0, False) \
        is False


@pytest.mark.logic
def test_populate_all_includes_the_dropped_rows(agg):
    """Per-seed aggregation takes a median across every seed, so a dropped
    seed left as a hole biases the aggregate rather than widening it."""
    r = {"reversion_dropped": "1", "steered_receptor_intact": "0",
         "ra": "99.0"}
    assert agg._passes_final_metrics_filter(
        r, Args(populate_all=True), "ra", 5.0, False) is True


@pytest.mark.logic
def test_an_unparseable_dropped_flag_does_not_drop_the_row(agg):
    r = {"reversion_dropped": "maybe", "steered_receptor_intact": "1",
         "ra": "1.0"}
    assert agg._passes_final_metrics_filter(r, Args(), "ra", 5.0, False) \
        is True


@pytest.mark.logic
@pytest.mark.parametrize("v,want", [("4.99", True), ("5.0", False),
                                    ("5.01", False)])
def test_the_metric_threshold_is_exclusive(agg, v, want):
    r = {"steered_receptor_intact": "1", "ra": v}
    assert agg._passes_final_metrics_filter(
        r, Args(), "ra", 5.0, False) is want


@pytest.mark.logic
def test_a_missing_metric_is_not_a_passing_row(agg):
    r = {"steered_receptor_intact": "1", "ra": ""}
    assert agg._passes_final_metrics_filter(r, Args(), "ra", 5.0, False) \
        is False


@pytest.mark.logic
def test_a_broken_receptor_is_excluded(agg):
    r = {"steered_receptor_intact": "0", "ra": "1.0"}
    assert agg._passes_final_metrics_filter(r, Args(), "ra", 5.0, False) \
        is False


@pytest.mark.logic
@pytest.mark.parametrize("intact", ["", "yes", None])
def test_an_unreadable_intact_flag_excludes_the_row(agg, intact):
    """Unlike the dropped flag, this one fails closed. An unknown intactness
    cannot be asserted, and the metrics would describe a broken receptor."""
    r = {"steered_receptor_intact": intact, "ra": "1.0"}
    assert agg._passes_final_metrics_filter(r, Args(), "ra", 5.0, False) \
        is False


@pytest.mark.logic
def test_the_intact_requirement_can_be_waived(agg):
    r = {"steered_receptor_intact": "0", "ra": "1.0"}
    assert agg._passes_final_metrics_filter(
        r, Args(require_intact=False), "ra", 5.0, False) is True


# ── the skip-steering baseline exemption ─────────────────────────────────────
#
# A run with skip_steering has no steered designs, so its cycle-0 baselines
# are the whole result. They are exempt from the metric threshold, because a
# baseline that misses the threshold is the finding rather than a reject.

@pytest.mark.logic
def test_a_skip_steering_baseline_is_exempt_from_the_threshold(agg):
    r = {"design": "initial", "cycle": "0",
         "steered_receptor_intact": "1", "ra": "99.0"}
    assert agg._passes_final_metrics_filter(r, Args(), "ra", 5.0, True) \
        is True


@pytest.mark.logic
def test_every_cold_start_seed_is_exempt(agg):
    """Multi-seed cold starts write initial_s1, initial_s2 and so on."""
    for design in ("initial", "initial_s1", "initial_s7"):
        r = {"design": design, "cycle": "0",
             "steered_receptor_intact": "1", "ra": "99.0"}
        assert agg._passes_final_metrics_filter(
            r, Args(), "ra", 5.0, True) is True


@pytest.mark.logic
def test_the_exemption_still_requires_an_intact_receptor(agg):
    r = {"design": "initial", "cycle": "0",
         "steered_receptor_intact": "0", "ra": "1.0"}
    assert agg._passes_final_metrics_filter(r, Args(), "ra", 5.0, True) \
        is False


@pytest.mark.logic
def test_the_exemption_does_not_reach_a_later_cycle(agg):
    """Cycle-1 and later rows never have design=="initial", so a row that
    claims to is not a baseline."""
    r = {"design": "initial", "cycle": "1",
         "steered_receptor_intact": "1", "ra": "99.0"}
    assert agg._passes_final_metrics_filter(r, Args(), "ra", 5.0, True) \
        is False


@pytest.mark.logic
def test_without_skip_steering_a_baseline_is_filtered_like_any_row(agg):
    r = {"design": "initial", "cycle": "0",
         "steered_receptor_intact": "1", "ra": "99.0"}
    assert agg._passes_final_metrics_filter(r, Args(), "ra", 5.0, False) \
        is False


# ── the written table ────────────────────────────────────────────────────────

@pytest.mark.logic
def test_the_metrics_csv_keeps_the_canonical_column_order(agg, tmp_path):
    """Downstream readers and the committed thesis tables both index by name,
    but a stable order is what makes a diff of two runs readable."""
    out = tmp_path / "m.csv"
    agg._write_metrics_csv(out, [
        {"steered_iptm": "0.8", "design": "d0", "cycle": "0"},
        {"steered_iptm": "0.7", "design": "d1", "cycle": "0"}])
    header = out.read_text().splitlines()[0].split(",")
    assert header.index("cycle") < header.index("design")
    assert header.index("design") < header.index("steered_iptm")


@pytest.mark.logic
def test_rows_with_different_columns_still_write(agg, tmp_path):
    """A failed row has no metric values but the same columns, and a resumed
    row may carry one the fresh rows do not."""
    out = tmp_path / "m.csv"
    agg._write_metrics_csv(out, [{"design": "d0", "steered_iptm": "0.8"},
                                 {"design": "d1", "extra": "x"}])
    rows = list(csv.DictReader(out.open()))
    assert [r["design"] for r in rows] == ["d0", "d1"]
