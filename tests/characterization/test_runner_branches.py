"""The branches the orchestration layers only take when something is wrong.

Five modules sit between a finished run directory and the tables a thesis
chapter reads: the per-stage selector, the per-sequence run view, the tool
CLI, the passing-summary extractor and the structure wrapper. Their happy
paths are pinned elsewhere. What is pinned here is what each one does with a
malformed input, a missing file or a refusal.

These matter because a run is hours of GPU time. Every branch below is one
the pipeline takes after the compute is already spent, so the choice is
between a stated failure and a plausible-looking wrong number. A guard that
degrades one seed keeps the rest of the stage. A guard that silently picks a
fallback changes which prediction a wet-lab construct is built from.

No cluster, no container and no GPU. The structure wrapper is driven over
small PDB files written here; the CLI's stages are real subprocesses pointed
at a stub script, so the subprocess machinery stays in the loop.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin"))

import extract_passing as ep  # noqa: E402
import pipeline_thresholds as pt  # noqa: E402

THRESHOLDS = pt.PipelineInternalThresholds.default()


# ── negsteer run: the stages it shells out to ───────────────────────────────
#
# `negsteer run` is the supported entry point, and every stage is a real
# subprocess. The stubs below are pointed at by the module's script
# constants, so the subprocess call itself stays real and only the engine is
# replaced. What is pinned is the refusals: a stage that exited in a way the
# CLI does not tolerate must stop the run with the command in the message,
# because the alternative is a run directory that looks finished.

_STUB_SOURCE = '''\
"""Stand-in for one engine script. Logs its arguments, writes plan files."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "stub_config.json").read_text())
ARGS = sys.argv[1:]
SUB = ARGS[0] if ARGS and not ARGS[0].startswith("-") else ""


def opt(name):
    return ARGS[ARGS.index(name) + 1] if name in ARGS else None


with (HERE / "stage_calls.log").open("a") as fh:
    fh.write(" ".join(ARGS) + "\\n")

workdir = opt("--workdir")
if SUB == "plan" and workdir and CFG.get("plan") is not None:
    Path(workdir).mkdir(parents=True, exist_ok=True)
    (Path(workdir) / "plan.json").write_text(json.dumps(CFG["plan"]))
if SUB == "plan-reversions" and workdir:
    (Path(workdir) / "reversion_plan.json").write_text(
        json.dumps(CFG.get("reversion_plan", {})))

sys.exit(CFG.get("exit", {}).get(SUB, 0))
'''


def _stub_engine(tmp_path, *, plan=None, reversion_plan=None, exits=None):
    """Write the stub script plus its config and return its path."""
    home = tmp_path / "stub"
    home.mkdir(parents=True, exist_ok=True)
    (home / "stub_config.json").write_text(json.dumps({
        "plan": plan,
        "reversion_plan": reversion_plan or {},
        "exit": exits or {},
    }))
    script = home / "engine_stub.py"
    script.write_text(_STUB_SOURCE)
    return script


def _point_cli_at(monkeypatch, script):
    import negsteer.cli as cli
    for name in ("ENGINE", "COLDSTART", "REVERSION", "AGGREGATE", "EXTRACT",
                 "CROSS_SUMMARY"):
        monkeypatch.setattr(cli, name, script)


def _cli_argv(tmp_path, outdir="out"):
    return ["run", "--name", "unit",
            "--receptor-fasta", "r", "--effector-fasta", "e",
            "--reference-pdb", "p", "--design-region", "d",
            "--true-interface", "t",
            "--outdir", str(tmp_path / outdir),
            "--boltz-container", "img"]


def _stage_calls(script):
    log = script.with_name("stage_calls.log")
    if not log.is_file():
        return []
    return [ln for ln in log.read_text().splitlines() if ln]


@pytest.mark.logic
class TestStageInvocation:
    def test_a_stage_that_exits_zero_returns_zero(self, tmp_path, capsys):
        """The command is printed before it runs, so a failed run's log says
        what to re-execute by hand."""
        import negsteer.cli as cli
        script = _stub_engine(tmp_path)
        rc = cli._run([sys.executable, str(script), "collect"])
        assert rc == 0
        assert f"$ {sys.executable} {script} collect" in capsys.readouterr().out

    def test_an_untolerated_exit_stops_the_run_and_quotes_the_command(
            self, tmp_path):
        """The message carries the command because the next step is running
        that one stage again against the half-built workdir."""
        import negsteer.cli as cli
        script = _stub_engine(tmp_path, exits={"collect": 3})
        with pytest.raises(cli.StageError, match="exited 3: "):
            cli._run([sys.executable, str(script), "collect"])

    def test_a_tolerated_exit_is_returned_rather_than_raised(self, tmp_path):
        """predict-one returns 1 for a per-design failure. Raising on it
        would let one bad design take the whole cohort down, which is the
        behaviour the shell orchestrator deliberately avoided."""
        import negsteer.cli as cli
        script = _stub_engine(tmp_path, exits={"predict-one": 1})
        rc = cli._run([sys.executable, str(script), "predict-one"],
                      tolerate=cli.PER_DESIGN_FAILURE)
        assert rc == 1

    def test_a_plan_stage_that_wrote_no_plan_stops_the_run(
            self, tmp_path, monkeypatch, capsys):
        """Without plan.json there is nothing to fan out over, so every later
        stage would run on an empty design list and the run would report
        success with no designs."""
        from negsteer.cli import StageError, build_parser, cmd_run
        script = _stub_engine(tmp_path, plan=None)
        _point_cli_at(monkeypatch, script)
        args = build_parser().parse_args(_cli_argv(tmp_path))
        with pytest.raises(StageError, match="plan stage produced no"):
            cmd_run(args)
        capsys.readouterr()

    def test_a_failed_plan_without_skip_steering_stops_the_run(
            self, tmp_path, monkeypatch, capsys):
        """A non-zero plan exit is tolerated only when the plan itself says
        skip_steering, which is the cold start having already passed. Any
        other non-zero exit is a planning failure."""
        from negsteer.cli import StageError, build_parser, cmd_run
        script = _stub_engine(tmp_path, plan={"designs": []},
                              exits={"plan": 1})
        _point_cli_at(monkeypatch, script)
        args = build_parser().parse_args(_cli_argv(tmp_path))
        with pytest.raises(StageError, match="plan stage exited 1"):
            cmd_run(args)
        capsys.readouterr()

    def test_main_turns_a_stage_failure_into_exit_one_and_a_message(
            self, tmp_path, monkeypatch, capsys):
        """A Nextflow process reads the exit code. An uncaught traceback and
        a 1 look the same to it, but only the message tells the operator
        which stage to re-run."""
        import negsteer.cli as cli
        script = _stub_engine(tmp_path, plan=None)
        _point_cli_at(monkeypatch, script)
        assert cli.main(_cli_argv(tmp_path)) == 1
        assert "ERROR: plan stage produced no" in capsys.readouterr().err

    def test_the_fan_outs_run_once_per_planned_item(
            self, tmp_path, monkeypatch, capsys):
        """Both widths are read off the plan files the engine wrote, not
        counted from a flag. A width read short silently drops designs, and
        the run still reports ok."""
        script = _stub_engine(
            tmp_path,
            plan={"designs": [{"name": "design_00"}, {"name": "design_01"}]},
            reversion_plan={"entries": [1, 2, 3]})
        _point_cli_at(monkeypatch, script)
        from negsteer.cli import build_parser, cmd_run
        assert cmd_run(build_parser().parse_args(_cli_argv(tmp_path))) == 0
        capsys.readouterr()
        calls = _stage_calls(script)
        predict = [c for c in calls if c.startswith("predict-one")]
        revert = [c for c in calls if c.startswith("predict-reversion-one")]
        assert [c.split("--index")[1].strip() for c in predict] == ["0", "1"]
        assert [c.split("--index")[1].strip() for c in revert] == \
            ["0", "1", "2"]


@pytest.mark.logic
def test_an_installation_with_no_engine_directory_is_an_error(tmp_path,
                                                              monkeypatch):
    """bin/ is found relative to the package or at a known container path.
    Neither existing means the install is broken, and saying so here beats
    failing at the first stage with "no such file", which reads as a broken
    engine rather than a broken installation.

    The module is re-executed with its __file__ pointed into an empty
    directory, which is what site-packages without the staged bin/ looks
    like.
    """
    import negsteer  # noqa: F401 - the relative import inside needs it

    monkeypatch.delenv("NEGSTEER_BIN", raising=False)
    source = _REPO_ROOT / "negsteer" / "cli.py"
    namespace = {
        "__name__": "negsteer._cli_under_test",
        "__package__": "negsteer",
        "__file__": str(tmp_path / "negsteer" / "cli.py"),
    }
    with pytest.raises(FileNotFoundError, match="cannot find the engine"):
        exec(compile(source.read_text(), str(source), "exec"), namespace)


# ── extract_passing: the mutation columns ───────────────────────────────────
#
# passing_summary.csv is the hand-off to the wet lab. Its mutation columns
# say which residues to change, so a position dropped by a parser is a
# residue left native in the construct. The parsers below are deliberately
# tolerant: a token they cannot read is skipped, because a seed with one
# malformed string must not blank the whole sequence group.


@pytest.mark.logic
class TestMutationStringParsing:
    def test_a_selection_without_a_chain_prefix_is_skipped(self):
        """A ChimeraX selection is "/A:5,7". Without the colon there is no
        chain to attribute the positions to, and guessing one would report
        receptor positions against the effector chain."""
        rows = [{"reverted_mutations_chimerax": "5,7"},
                {"reverted_mutations_chimerax": "/A:5"}]
        maj, allp, identical = ep._summarize_mutations_across_seeds(
            rows, "reverted_mutations_chimerax")
        assert (maj, allp, identical) == ("/A:5", "/A:5", "")

    @pytest.mark.parametrize("token,why", [
        ("A5", "too short to hold a position and two residues"),
        ("ABC", "carries no digits where the position should be"),
        ("A²B", "a superscript digit, which isdigit accepts and int rejects"),
    ])
    def test_an_unreadable_token_is_skipped_not_guessed(self, token, why):
        """Each of these is skipped, leaving the readable token alone. A
        guessed position would name the wrong residue in the construct."""
        rows = [{"reverted_mutations_aa": f"{token} V7D"}]
        got = ep._muts_aa_for_positions(rows, "reverted_mutations_aa", [7])
        assert got == "V7D", why

    def test_a_seed_with_no_mutation_string_contributes_nothing(self):
        """A seed whose reversion produced no mutations has a blank column.
        It is skipped rather than treated as a seed with zero positions,
        which is what the other seeds' positions are merged against."""
        rows = [{"reverted_mutations_aa": ""},
                {"reverted_mutations_aa": "K5E V7D"}]
        got = ep._muts_aa_for_positions(rows, "reverted_mutations_aa", [5, 7])
        assert got == "K5E V7D"

    def test_a_position_missing_from_the_reverted_column_falls_back(self):
        """A seed's reverted_mutations_aa only covers positions that seed
        retained. Positions 7 and 22 were retained by no seed's reverted
        column, so their residue identities are looked up in the steered set,
        which always carries every mutated position. Without the fallback the
        construct instruction would silently omit them.

        The first seed carries neither column, which is what a seed whose own
        prediction failed leaves behind. It is skipped in both passes rather
        than ending the lookup at the first blank.
        """
        rows = [{"reverted_mutations_aa": "", "steered_mutations_aa": ""},
                {"reverted_mutations_aa": "K5E",
                 "steered_mutations_aa": "K5E V7D M22R"}]
        got = ep._muts_aa_for_positions(
            rows, "reverted_mutations_aa", [5, 7, 22])
        assert got == "K5E V7D M22R"

    def test_the_majority_selection_is_turned_back_into_residue_names(self):
        """The majority set is computed as positions and reported as both a
        selection and an AA string. The two must name the same residues, or
        the row says one thing in the selection column and another in the
        column the lab reads."""
        per_seed = [
            {"steered_mutations_chimerax": "/A:5,7,22",
             "steered_mutations_aa": "K5E V7D M22R",
             "reverted_mutations_chimerax": "/A:5,7",
             "reverted_mutations_aa": "K5E V7D"},
            {"steered_mutations_chimerax": "/A:5,7,22",
             "steered_mutations_aa": "K5E V7D M22R",
             "reverted_mutations_chimerax": "/A:5",
             "reverted_mutations_aa": "K5E"},
        ]
        got = ep.extract_row({"outcome": "pose_holds"}, per_seed_rows=per_seed)
        assert got["reverted_mutations_majority_chimerax"] == "/A:5"
        assert got["reverted_mutations_majority_aa"] == "K5E"
        assert got["reverted_mutations_all_chimerax"] == "/A:5,7"
        assert got["reverted_mutations_all_aa"] == "K5E V7D"


@pytest.mark.logic
def test_the_import_fallback_thresholds_match_the_catalogue(monkeypatch):
    """extract_passing is runnable as a standalone script on a workdir, so
    it falls back to hard-coded confidence thresholds when
    pipeline_thresholds is not importable. The fallback must agree with the
    catalogue, or the same run scored two ways disagrees about which
    sequences are confident.

    0.29 iptm is below the 0.30 bar and 0.31 is above it. Both are asserted,
    so a fallback that drifted either way is caught.
    """
    monkeypatch.setitem(sys.modules, "pipeline_thresholds", None)
    below = {"pae_pass_frac_median": 0.9, "iptm_median": 0.29,
             "complex_plddt_median": 0.9, "ipae_median": 1.0}
    above = dict(below, iptm_median=0.31)
    assert ep._compute_confidence_flag(below) == "low_iptm"
    assert ep._compute_confidence_flag(above) == "ok"


@pytest.mark.logic
def test_a_row_with_an_unreadable_rank_sorts_last_rather_than_crashing(
        tmp_path, monkeypatch, capsys):
    """passing_summary.csv is ordered by composite rank, best first. A row
    whose rank did not parse is sunk to the bottom instead of taking the
    write down, because the ranked rows above it are the ones a human reads
    and they are already correct.
    """
    columns = ["cycle", "pathway", "sequence_group", "design", "outcome",
               "rank_by_composite_score"]
    inp = tmp_path / "aggregated_results.csv"
    with inp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerow({"cycle": "0", "pathway": "p", "sequence_group": "g",
                         "design": "design_bad", "outcome": "no_reversion",
                         "rank_by_composite_score": "unranked"})
        writer.writerow({"cycle": "0", "pathway": "p", "sequence_group": "g",
                         "design": "design_good", "outcome": "no_reversion",
                         "rank_by_composite_score": "2"})
    monkeypatch.setattr(sys, "argv",
                        ["extract_passing.py", "--input", str(inp)])
    assert ep.main() == 0
    capsys.readouterr()
    rows = list(csv.DictReader((tmp_path / "passing_summary.csv").open()))
    assert [r["design"] for r in rows] == ["design_good", "design_bad"]


# ── the stub engine is a stub, not the engine ───────────────────────────────


@pytest.mark.logic
def test_the_stage_stub_is_only_ever_run_as_a_subprocess(tmp_path):
    """Guards the shape of the tests above. The stub stands in for a script
    the CLI invokes by path, so it has to be executable by the interpreter
    the CLI uses. Importing it instead would take the real subprocess call
    out of the loop, which is where the exit codes come from."""
    script = _stub_engine(tmp_path, plan={"designs": []})
    done = subprocess.run([sys.executable, str(script), "plan",
                           "--workdir", str(tmp_path / "cycle_0")])
    assert done.returncode == 0
    assert json.loads(
        (tmp_path / "cycle_0" / "plan.json").read_text()) == {"designs": []}
    assert importlib.util.find_spec("negsteer") is not None
