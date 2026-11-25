"""Two guards nothing else reaches, plus the panel guard that nothing can.

bin/reversion.py repairs its own sys.path in six places, once at module level
and once in each of five stages. Nextflow invokes the script by absolute path
with no PYTHONPATH, so `bin/` is not on sys.path and the sibling imports
(boltz2_negative_steering, pathway_labels, reversion itself) would all fail
without the repair. The same comment sits over the same two lines in
bin/cross_summary.py, which does it once at module level.

Every other test imports the module first, which puts `bin/` on sys.path and
makes `if str(script_dir) not in sys.path` false for the rest of the session.
So none of the six inserts has ever run under test, and a stage that stopped
repairing its path would still pass the whole suite while failing in the
container on the first import. Each test here removes `bin/` from sys.path and
then calls one stage. The stage's own lazy import is what proves the repair
happened, because the insert is the only thing that puts the path back.

The two uncovered lines in bin/negsteer_within_sequence_plots.py are different.
Neither can be reached, and each is now pragma'd, so what is left to pin is the
invariant behind one and the defect behind the other.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from unittest import mock

import matplotlib
import pytest

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]
BIN = REPO_ROOT / "bin"
REFERENCE = REPO_ROOT / "data" / "reference_complexes" / "6G10.pdb"

sys.path.insert(0, str(BIN))

from test_harvest_reversions_cli import Harvest as CliHarvest  # noqa: E402
from test_plot_branches import _seed_rec  # noqa: E402
from test_plot_data_policy import (  # noqa: E402
    PASSING,
    _agg,
    _cross,
    _seq,
    _write_raw,
)
from test_reversion_harvest import Harvest as LibHarvest  # noqa: E402
from test_reversion_plan_staging import Stage, design, meta  # noqa: E402


def _deps() -> bool:
    return all(importlib.util.find_spec(m) is not None
               for m in ("numpy", "Bio"))


needs_structure = pytest.mark.skipif(
    not _deps(),
    reason="needs numpy and biopython (pip install -e '.[experiments]')")


@pytest.fixture(scope="module")
def rev():
    """bin/reversion.py executed with `bin/` absent from sys.path.

    This is the module-level repair at :1130, and re-executing the source
    under a changed sys.path is the only way to reach it, because by the time
    any test runs the module is already imported and the guard is false.
    The sibling import on the next line is what proves it ran.
    """
    saved = list(sys.path)
    sys.path[:] = [p for p in saved if p != str(BIN)]
    try:
        assert str(BIN) not in sys.path
        spec = importlib.util.spec_from_file_location(
            "reversion", BIN / "reversion.py")
        mod = importlib.util.module_from_spec(spec)
        # Registered before the exec, so the module's own `import reversion`
        # inside its stages resolves to this object rather than re-executing
        # the file.
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        assert str(BIN) in sys.path, "reversion.py did not put bin/ back"
        yield mod
    finally:
        sys.path[:] = saved


@pytest.fixture(scope="module")
def nwp():
    import negsteer_within_sequence_plots
    return negsteer_within_sequence_plots


def _without_bin(monkeypatch) -> None:
    """Take `bin/` off sys.path for one test, the way the container has it.

    monkeypatch rebinds sys.path to a new list, so the stage's own insert
    lands in that list and the real sys.path comes back at teardown.
    """
    monkeypatch.setattr(sys, "path",
                        [p for p in sys.path if p != str(BIN)])
    assert str(BIN) not in sys.path


# ── the module-level repair ──────────────────────────────────────────────────

@pytest.mark.logic
def test_the_module_puts_bin_back_on_sys_path_as_it_loads(rev):
    """read_mutations_tsv_full comes from pathway_labels, imported two lines
    after the insert. It is bound here only because the insert ran, so its
    presence is the proof rather than sys.path's contents."""
    assert str(BIN) in sys.path
    assert rev.SCRIPT_DIR == BIN
    assert callable(rev.read_mutations_tsv_full)
    assert callable(rev._locate_boltz_sidecar_pdb)


# ── write_reversion_plan ─────────────────────────────────────────────────────

@pytest.mark.logic
def test_write_reversion_plan_repairs_its_path_before_staging(rev, tmp_path,
                                                              monkeypatch):
    """write_boltz_yaml lives in boltz2_negative_steering and is imported
    inside the function. Without the insert the stage would raise
    ModuleNotFoundError before writing a single YAML, so a manifest that
    names a real YAML is the proof."""
    _without_bin(monkeypatch)
    c = design(tmp_path, "d0", "MKWVTA", muts=[[3, "A", "W"]], revert=[3])
    out = rev.write_reversion_plan(tmp_path, [c], meta())
    doc = json.loads(out.read_text())
    assert doc["n_staged"] == 1
    entry = doc["entries"][0]
    assert Path(entry["reverted_receptor_fasta"]).read_text().endswith(
        "MKAVTA\n")
    assert Path(entry["yaml"]).is_file()


# ── harvest_reversion_results ────────────────────────────────────────────────

@needs_structure
@pytest.mark.logic
def test_harvest_reversion_results_repairs_its_path_before_measuring(
        rev, tmp_path, monkeypatch):
    """binding_rmsds and find_contact_residues_heavy are imported inside the
    function. The RMSDs below are computed by them, so a record carrying them
    could not exist if the insert had not run first."""
    _without_bin(monkeypatch)
    h = LibHarvest(tmp_path / "wd")
    res = h.slot("d0").write().run(rev)
    rec = res["d0"]
    assert rec["reverted_ra_eff_vs_truth"] == pytest.approx(0.0, abs=1e-6)
    assert rec["reverted_intact"] is True
    assert "error" not in rec


# ── plan-reversions ──────────────────────────────────────────────────────────

@needs_structure
@pytest.mark.logic
def test_cmd_plan_reversions_repairs_its_path_before_reading_the_effector(
        rev, tmp_path, monkeypatch):
    """The stage imports both `reversion` itself and get_chain_sequence after
    the insert. The effector sequence in the staged YAML comes out of the
    ground-truth PDB through get_chain_sequence, so a YAML with one in it is
    the proof."""
    _without_bin(monkeypatch)
    s = Stage(tmp_path / "wd").contaminate("d0").write()
    assert s.run(rev) == 0
    doc = s.doc()
    assert doc["n_staged"] == 1
    yaml_text = Path(doc["entries"][0]["yaml"]).read_text()
    assert "MKAVTA" in yaml_text
    assert len(yaml_text) > 100


# ── predict-reversion-one ────────────────────────────────────────────────────

@needs_structure
@pytest.mark.logic
def test_cmd_predict_reversion_one_repairs_its_path_before_calling_boltz(
        rev, tmp_path, monkeypatch):
    """run_boltz is imported after the insert. boltz2_negative_steering is
    imported here first, while `bin/` is still on sys.path, so the stub is in
    place on the module object the stage will reach through its own import.
    """
    import boltz2_negative_steering as bns

    s = Stage(tmp_path / "wd").contaminate("d0").write()
    assert s.run(rev) == 0

    seen = {}

    def fake_run_boltz(yaml_path, out_dir, container, recycling, samples,
                       seed, no_kernels):
        seen["yaml"] = str(yaml_path)
        out = Path(out_dir) / "boltz_out"
        out.mkdir(parents=True, exist_ok=True)
        pred = out / "model_0.pdb"
        shutil.copy(REFERENCE, pred)
        return pred

    monkeypatch.setattr(bns, "run_boltz", fake_run_boltz)
    _without_bin(monkeypatch)
    argv = ["predict-reversion-one", "--workdir", str(s.root), "--index", "0"]
    assert rev.cmd_predict_reversion_one(
        rev.build_parser().parse_args(argv)) == 0

    entry = s.doc()["entries"][0]
    assert seen["yaml"] == entry["yaml"]
    assert (Path(entry["rev_dir"]) / "prediction.pdb").is_file()


# ── harvest-reversions ───────────────────────────────────────────────────────

@needs_structure
@pytest.mark.logic
def test_cmd_harvest_reversions_repairs_its_path_before_harvesting(
        rev, tmp_path, monkeypatch):
    """The stage imports `reversion` by name after the insert and calls
    harvest_reversion_results through it. A written verdict is the proof,
    because the import is the only route to the function that produces it."""
    _without_bin(monkeypatch)
    h = CliHarvest(tmp_path / "wd", true_interface=range(0, 40))
    assert h.slot("d0", contacts="").write().run(rev) == 0
    assert h.verdicts()["d0"]["verdict"] == "pose_holds"


# ══════════════════════════════════════════════════════════════════════════
# negsteer_within_sequence_plots
# ══════════════════════════════════════════════════════════════════════════

# One seed carrying every metric the confidence grid panels, so all ten
# panels exist and the 3x4 grid has two cells left over.
TEN_METRICS = {"avg_plddt": 88.0, "interface_plddt": 0.81, "iptm": 0.8,
               "ptm": 0.7, "actifptm": 0.65, "pae_pass_frac": 0.9,
               "ipsae_min": 0.6, "ipsae_min_15": 0.62, "ipae": 3.0,
               "pae_mean": 5.0}


@pytest.mark.logic
def test_the_spare_grid_cells_are_blanked_rather_than_labelled_no_data(
        nwp, tmp_path, monkeypatch):
    """The invariant behind the pragma on plot_composite_vs_confidence's
    `if not xs` branch.

    Ten metrics lay out as three rows of four, so two cells are spare. A
    spare cell is an index at or above n and takes set_axis_off. Every index
    below n names a metric that metrics_present kept, which it kept only
    because some sequence has that metric, so the panel has at least one
    point. Nothing is left to put "no data" on.

    The figure is captured through plt.subplots rather than read back off
    disk, because an axes labelled "no data" and an axes with its frame off
    are not distinguishable in a PNG. ax.texts holds ax.text artists only,
    and a title is not one of them.
    """
    seen = {}
    real_subplots = nwp.plt.subplots

    def spy(*args, **kwargs):
        fig, axes = real_subplots(*args, **kwargs)
        seen["fig"] = fig
        return fig, axes

    monkeypatch.setattr(nwp.plt, "subplots", spy)
    nwp.plot_composite_vs_confidence(
        {"design_0_seq_1": [_seed_rec(**TEN_METRICS)]},
        {"design_0_seq_1": 0.7}, {"design_0_seq_1": "A"},
        str(tmp_path / "conf.png"))

    fig = seen["fig"]
    assert len(fig.axes) == 12, "the grid shape changed, so re-check the count"
    assert [ax.axison for ax in fig.axes].count(False) == 2
    assert [t.get_text() for ax in fig.axes for t in ax.texts] == [], \
        "a panel was labelled no data, so the branch is reachable after all"


@pytest.mark.logic
def test_every_figure_that_drew_reports_the_path_it_wrote(nwp, tmp_path,
                                                          capsys):
    """main reports a figure by the truth of its return value, so each plot_*
    has to return True once it has saved.

    All five used to fall off the end on success and return None, so the log
    read "(skipped — no data)" for five figures that had just written a PNG.
    Nextflow publishes a fixed figure list, so the stage would then be read as
    having produced nothing on every successful run. The sibling module
    bin/negsteer_plots.py returns True from all seven of its figures, which is
    the contract main was written against.
    """
    runs = tmp_path / "runs"
    d = _seq(runs, "design_0_seq_1", [_agg("0", ra=2.0, tj=0.8)])
    _write_raw(d, [dict(PASSING, sequence_group="0", seed_index="0",
                        steered_true_jaccard="0.8",
                        steered_weighted_jaccard="0.75")])
    cross = tmp_path / "cross.csv"
    _cross(cross, [{"mpnn_sequence": "design_0_seq_1", "cross_tier": "A",
                    "rep_sequence_group": "0"}])
    out = tmp_path / "out"
    argv = ["negsteer_within_sequence_plots.py", "--runs-dir", str(runs),
            "--cross-summary-csv", str(cross), "--outdir", str(out)]
    with mock.patch.object(sys, "argv", argv):
        assert nwp.main() == 0

    text = capsys.readouterr().out
    assert len(list(out.glob("*.png"))) == 5
    assert text.count("→") == 5, f"a drawn figure was not reported:\n{text}"
    assert "(skipped — no data)" not in text
    assert "FAILED" not in text
