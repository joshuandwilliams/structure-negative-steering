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
from unittest.mock import MagicMock

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin"))

import extract_passing as ep  # noqa: E402
import negative_steering_run as nsr  # noqa: E402
import pipeline_thresholds as pt  # noqa: E402
import protein_structure_prediction as psp  # noqa: E402
import runtree as rt  # noqa: E402
import stage_result as sr  # noqa: E402
from position_set import PositionSet  # noqa: E402

THRESHOLDS = pt.PipelineInternalThresholds.default()


def _fake_pred(ra_eff, intact_rmsd, contact_positions=()):
    """A stand-in for ProteinStructurePrediction with injected metrics.

    Same pattern as test_stage_result.py. The verdict logic under test reads
    four numbers off a prediction; supplying them directly is what keeps
    these tests free of real PDBs.
    """
    m = MagicMock()
    m.ra_eff_vs.return_value = ra_eff
    m.independent_receptor_rmsd_vs.return_value = intact_rmsd
    m.receptor_contact_residues.return_value = PositionSet(
        list(contact_positions), chain="A", frame="prediction")
    return m


def _confs(predictions):
    return [MagicMock() for _ in predictions]


def _cold_start(predictions, config_id="seqA"):
    return sr.StageResult(
        stage_type="cold_start", config_id=config_id,
        predictions=predictions, confidences=_confs(predictions),
        seed_indices=list(range(len(predictions))))


def _steered(predictions, *, seeds=None, mutated=(5, 7, 12),
             config_id="seqA__design_00"):
    seeds = list(range(len(predictions))) if seeds is None else list(seeds)
    return sr.StageResult(
        stage_type="steered", config_id=config_id,
        predictions=predictions, confidences=_confs(predictions),
        seed_indices=seeds,
        mutated_positions=PositionSet(list(mutated), chain="A",
                                      frame="prediction"))


def _reversion(predictions, *, seeds=None, mutated=(5, 7, 12),
               applies_to=("design_00",), config_id="seqA__rev_design_00_s0"):
    seeds = list(range(len(predictions))) if seeds is None else list(seeds)
    return sr.StageResult(
        stage_type="reversion", config_id=config_id,
        predictions=predictions, confidences=_confs(predictions),
        seed_indices=seeds,
        mutated_positions=PositionSet(list(mutated), chain="A",
                                      frame="prediction"),
        applies_to_designs=list(applies_to))


def _contam(positions=(5, 7, 12)):
    return PositionSet(list(positions), chain="A", frame="prediction")


def _run(cold, *, workdir, steered=None, reversion=None, truth=None,
         row_type="steered"):
    return nsr.NegativeSteeringRun(
        mpnn_sequence_id="seqA", workdir=workdir, row_type=row_type,
        cold_start=cold, steered=steered or {}, reversion=reversion or {},
        truth=truth, contamination_positions=_contam())


# ── StageResult.canonical_prediction ────────────────────────────────────────
#
# The canonical prediction is the one seed per stage that becomes the design's
# representative: its PDB is the file a construct is designed from, and its
# ra_eff is the number the cohort table reports. Choosing it by ra_eff alone
# would pick a seed whose receptor fold collapsed, because a collapsed
# receptor can give a flatteringly low receptor-aligned effector RMSD. The
# rule is therefore pass-equivalent seeds first, lowest ra_eff within them.


@pytest.mark.logic
class TestCanonicalPrediction:
    def test_a_passing_seed_beats_a_lower_ra_eff_that_failed(self):
        """Seed 0 has the lowest ra_eff in the stage and a collapsed
        receptor. Picking it would report 1.0 A for a design whose fold did
        not survive, and hand the wet lab that PDB."""
        collapsed = _fake_pred(1.0, 10.0)
        worse_clean = _fake_pred(3.0, 1.0)
        best_clean = _fake_pred(2.5, 1.0)
        stage = _cold_start([collapsed, worse_clean, best_clean])
        got = stage.canonical_prediction(THRESHOLDS, MagicMock())
        assert got is best_clean

    def test_with_nothing_passing_the_lowest_ra_eff_overall_is_taken(self):
        """A stage where every seed failed still needs a representative, so
        the cohort row describes the least-bad attempt rather than nothing."""
        worse = _fake_pred(9.0, 10.0)
        better = _fake_pred(7.0, 10.0)
        stage = _cold_start([worse, better])
        assert stage.canonical_prediction(THRESHOLDS, MagicMock()) is better

    def test_with_no_ra_eff_anywhere_the_first_seed_stands_in(self):
        """Every seed unmeasurable means the comparison cannot be made. The
        first seed keeps the stage in the table instead of dropping it, and
        its blank metrics are what mark it unscored."""
        first = _fake_pred(None, None)
        second = _fake_pred(None, None)
        stage = _cold_start([first, second])
        assert stage.canonical_prediction(THRESHOLDS, MagicMock()) is first

    def test_the_steered_selector_prefers_a_clean_steered_seed(self):
        """Passing means stage-specific. A contaminated seed with the lowest
        ra_eff must not represent a steered design, because contamination is
        the thing the stage exists to detect."""
        contaminated = _fake_pred(1.0, 1.0, contact_positions=[5])
        clean = _fake_pred(4.0, 1.0, contact_positions=[100])
        stage = _steered([contaminated, clean])
        got = stage.canonical_prediction(THRESHOLDS, MagicMock(), _contam())
        assert got is clean

    def test_an_unmeasurable_seed_is_not_counted_as_correctly_placed(self):
        """n_correctly_placed is the denominator of the CL-3 majority gate.
        Counting a seed with no metrics would raise the bar reversion has to
        clear, and reversion would then not run on a contaminated design."""
        stage = _steered([_fake_pred(2.0, 1.0), _fake_pred(None, None),
                          _fake_pred(2.0, 1.0)])
        assert stage.n_correctly_placed(THRESHOLDS, MagicMock()) == 2


# ── NegativeSteeringRun construction guards ─────────────────────────────────
#
# A NegativeSteeringRun holds one cold start, a dict of steered stages and a
# dict of reversions. The stage_type on each has to match the slot it sits
# in, because every verdict method branches on it. A reversion stage filed
# under `steered` would be scored against the steered vocabulary, so its
# pose_holds seeds would be read as neither passing nor failing.


@pytest.mark.logic
class TestConstructionTypeGuards:
    def test_a_cold_start_that_is_not_a_stage_result_is_refused(self, tmp_path):
        with pytest.raises(TypeError, match="cold_start must be a StageResult"):
            _run({"not": "a stage"}, workdir=tmp_path)

    def test_a_steered_slot_holding_something_else_is_refused(self, tmp_path):
        with pytest.raises(TypeError, match=r"steered\['design_00'\]"):
            _run(_cold_start([_fake_pred(2.0, 1.0)]), workdir=tmp_path,
                 steered={"design_00": "not a stage"})

    def test_a_steered_slot_holding_the_wrong_stage_type_is_refused(self, tmp_path):
        wrong = _cold_start([_fake_pred(2.0, 1.0)])
        with pytest.raises(ValueError, match="stage_type='steered'"):
            _run(_cold_start([_fake_pred(2.0, 1.0)]), workdir=tmp_path,
                 steered={"design_00": wrong})

    def test_a_reversion_slot_holding_something_else_is_refused(self, tmp_path):
        with pytest.raises(TypeError, match=r"reversion\['rev_0'\]"):
            _run(_cold_start([_fake_pred(2.0, 1.0)]), workdir=tmp_path,
                 reversion={"rev_0": 17})

    def test_a_reversion_slot_holding_a_steered_stage_is_refused(self, tmp_path):
        """The verdict vocabularies differ between the two stages. A steered
        stage scored as a reversion would report new_contamination where it
        meant contaminated, and the outcome label would change."""
        with pytest.raises(ValueError, match="stage_type='reversion'"):
            _run(_cold_start([_fake_pred(2.0, 1.0)]), workdir=tmp_path,
                 reversion={"rev_0": _steered([_fake_pred(2.0, 1.0)])})


# ── Cross-stage per-seed aggregation ────────────────────────────────────────
#
# A seed's final verdict is its reversion verdict when reversion ran for that
# design, and its steered verdict otherwise. The two stages are paired by
# seed index. A reversion that lost one seed to a failed job therefore leaves
# that seed with its steered verdict, which is the honest answer: nothing
# reverted it, so nothing changed about it.


@pytest.mark.logic
class TestPerSeedFinalVerdicts:
    def test_a_seed_the_reversion_never_covered_keeps_its_steered_verdict(
            self, tmp_path):
        """Seed 1's reversion job never landed. Giving it seed 0's reverted
        verdict would credit a rescue that was never computed for it, and
        n_pass would count a seed that is still contaminated."""
        steered = _steered(
            [_fake_pred(2.0, 1.0, contact_positions=[5]),      # contaminated
             _fake_pred(2.0, 1.0, contact_positions=[100])],   # clean_steered
            seeds=[0, 1])
        reversion = _reversion(
            [_fake_pred(2.0, 1.0, contact_positions=[100])], seeds=[0])
        run = _run(_cold_start([_fake_pred(2.0, 1.0)]), workdir=tmp_path,
                   steered={"design_00": steered},
                   reversion={"rev_design_00_s0": reversion},
                   truth=MagicMock())
        got = run.per_seed_final_verdicts(THRESHOLDS)
        assert [(e["seed_index"], e["stage_run"], e["verdict"]) for e in got] == [
            (0, "reverted", "pose_holds"),
            (1, "steered", "clean_steered"),
        ]

    def test_verdicts_without_a_truth_structure_are_refused(self, tmp_path):
        """Every verdict is a comparison against the reference complex. With
        no reference, ra_eff is unmeasurable and every seed would come back
        no_data, which reads as a failed run rather than a missing input."""
        run = _run(_cold_start([_fake_pred(2.0, 1.0)]), workdir=tmp_path,
                   truth=None)
        with pytest.raises(ValueError, match="require a 'truth'"):
            run.per_seed_final_verdicts(THRESHOLDS)


# ── Outcome labels and their stated reason ──────────────────────────────────
#
# `outcome` is the label the cohort table carries per sequence, and
# `outcome_reason` is the sentence written beside it. The reason has to name
# why the label was chosen, because the three no-reversion cases look
# identical in the table and mean different things: steering never ran,
# steering ran and found no contamination to revert, or reversion ran.


@pytest.mark.logic
class TestOutcomeAndReason:
    def test_collapsed_reversions_are_reported_as_pose_collapses(self, tmp_path):
        """No seed holds its pose after reversion. Reporting pose_holds here
        would put a design in the passing tier whose receptor moved 10 A."""
        steered = _steered([_fake_pred(2.0, 1.0, contact_positions=[5])])
        reversion = _reversion([_fake_pred(10.0, 1.0,
                                           contact_positions=[100])])
        run = _run(_cold_start([_fake_pred(2.0, 1.0)]), workdir=tmp_path,
                   steered={"design_00": steered},
                   reversion={"rev_design_00_s0": reversion},
                   truth=MagicMock())
        assert run.outcome(THRESHOLDS) == "pose_collapses"

    def test_the_cold_start_path_says_steering_never_ran(self, tmp_path):
        run = _run(_cold_start([_fake_pred(2.0, 1.0)]), workdir=tmp_path,
                   truth=MagicMock())
        assert run.outcome(THRESHOLDS) == "no_reversion"
        assert run.outcome_reason(THRESHOLDS) == \
            "cold-start path: no steering ran"

    def test_steering_without_reversion_names_the_cl3_rule(self, tmp_path):
        """Same `no_reversion` label as the cold-start path, different cause.
        Here steering did run and no design cleared the CL-3 majority, so the
        reason has to distinguish it or the two cases pool in the summary."""
        steered = _steered([_fake_pred(2.0, 1.0, contact_positions=[100])])
        run = _run(_cold_start([_fake_pred(2.0, 1.0)]), workdir=tmp_path,
                   steered={"design_00": steered}, truth=MagicMock())
        assert run.outcome(THRESHOLDS) == "no_reversion"
        assert run.outcome_reason(THRESHOLDS) == (
            "steering ran but no design had a majority of its "
            "correctly-placed seeds contaminated, so no reversion "
            "was triggered")

    def test_a_mixed_reversion_reports_its_verdict_counts(self, tmp_path):
        """One seed held its pose and one picked up new contamination. The
        reason is the tally, sorted by verdict name so it is stable between
        runs and comparable across sequences."""
        steered = _steered([_fake_pred(2.0, 1.0, contact_positions=[5]),
                            _fake_pred(2.0, 1.0, contact_positions=[7])],
                           seeds=[0, 1])
        reversion = _reversion(
            [_fake_pred(2.0, 1.0, contact_positions=[100]),   # pose_holds
             _fake_pred(2.0, 1.0, contact_positions=[5])],    # new_contamination
            seeds=[0, 1])
        run = _run(_cold_start([_fake_pred(2.0, 1.0)]), workdir=tmp_path,
                   steered={"design_00": steered},
                   reversion={"rev_design_00_s0": reversion},
                   truth=MagicMock())
        assert run.outcome_reason(THRESHOLDS) == \
            "new_contamination=1; pose_holds=1"

    def test_the_repr_names_the_sequence_and_its_stage_counts(self, tmp_path):
        """These objects are inspected one at a time while debugging a
        cohort. The id and the two stage counts are what identify which
        sequence is in hand."""
        run = _run(_cold_start([_fake_pred(2.0, 1.0)]), workdir=tmp_path,
                   steered={"design_00": _steered([_fake_pred(2.0, 1.0)])},
                   reversion={"rev_design_00_s0":
                              _reversion([_fake_pred(2.0, 1.0)])},
                   truth=MagicMock())
        text = repr(run)
        assert "mpnn_sequence_id='seqA'" in text
        assert "row_type='steered'" in text
        assert "n_steered_designs=1" in text
        assert "n_reversions=1" in text


# ── from_workdir over a damaged run directory ───────────────────────────────
#
# from_workdir is the only reader between a finished run tree and every
# cohort query. A job that died leaves a prediction directory with no Boltz
# output, and a directory name it cannot parse carries no grouping. Both have
# to be skipped rather than guessed at, because a guess silently files one
# design's seeds under another and the tier is then computed over the wrong
# set of predictions.


@pytest.mark.logic
class TestFromWorkdirDamage:
    def test_a_cold_start_seed_with_no_output_is_skipped_either_way(
            self, tmp_path):
        """require_pdb=False does not make an unreadable seed usable. There
        are no files to construct a prediction from, so the seed is dropped
        and the surviving seeds still form the stage."""
        cycle = rt.deep(tmp_path, seeds=(0, 1))
        (cycle / "initial_s1" / "boltz_results_input" / "predictions"
         / "input" / "confidence_input_model_0.json").unlink()
        run = nsr.NegativeSteeringRun.from_workdir(
            workdir=tmp_path, mpnn_sequence_id="seqA", num_seeds=2,
            require_pdb=False)
        assert list(run.cold_start.seed_indices) == [0]

    def test_a_file_among_the_reversion_directories_is_ignored(self, tmp_path):
        """The reversion root is iterated to find prediction directories. A
        stray log or tarball beside them is not one."""
        cycle = rt.deep(tmp_path, steered=[("design_00", [0], [5])],
                        reversions=[("design_00", 0, [0])])
        (cycle / "reversions" / "slurm-1234.out").write_text("log\n")
        run = nsr.NegativeSteeringRun.from_workdir(
            workdir=tmp_path, mpnn_sequence_id="seqA")
        assert len(run.reversion) == 1

    def test_a_reversion_seed_with_no_output_is_dropped_from_its_group(
            self, tmp_path):
        """Reversion seeds are pooled by parent slot. A seed whose job died
        must leave the group smaller rather than appear with no metrics,
        because n_pass is counted out of the seeds present."""
        cycle = rt.deep(tmp_path, steered=[("design_00", [0], [5])],
                        reversions=[("design_00", 0, [0, 1])])
        (cycle / "reversions" / "rev_design_00_s0_s1" / "boltz_results_input"
         / "predictions" / "input" / "confidence_input_model_0.json").unlink()
        run = nsr.NegativeSteeringRun.from_workdir(
            workdir=tmp_path, mpnn_sequence_id="seqA")
        assert list(run.reversion["rev_design_00_s0"].seed_indices) == [0]

    def test_a_reversion_group_with_no_usable_seed_is_dropped_entirely(
            self, tmp_path):
        """An empty group cannot be built as a StageResult, and filing it
        with zero predictions would make the design look reverted when no
        reversion prediction exists."""
        cycle = rt.deep(tmp_path, steered=[("design_00", [0], [5])],
                        reversions=[("design_00", 0, [0])])
        (cycle / "reversions" / "rev_design_00_s0_s0" / "boltz_results_input"
         / "predictions" / "input" / "confidence_input_model_0.json").unlink()
        run = nsr.NegativeSteeringRun.from_workdir(
            workdir=tmp_path, mpnn_sequence_id="seqA")
        assert run.reversion == {}

    def test_the_runtime_sidecar_is_read_as_a_number(self, tmp_path):
        """run_one_runtime_sec is the GPU cost of this sequence. It is read
        here and carried into the cost table, so it has to be a float rather
        than the file's text."""
        rt.deep(tmp_path)
        (tmp_path / "run_one_runtime_sec.txt").write_text("1234.5\n")
        run = nsr.NegativeSteeringRun.from_workdir(
            workdir=tmp_path, mpnn_sequence_id="seqA")
        assert run.run_one_runtime_sec == 1234.5

    def test_a_corrupt_runtime_sidecar_is_none_not_a_crash(self, tmp_path):
        """A truncated sidecar is what a killed job leaves. None means "not
        measured", which a cost table can report as blank. Raising would lose
        the whole sequence over a timing file."""
        rt.deep(tmp_path)
        (tmp_path / "run_one_runtime_sec.txt").write_text("killed\n")
        run = nsr.NegativeSteeringRun.from_workdir(
            workdir=tmp_path, mpnn_sequence_id="seqA")
        assert run.run_one_runtime_sec is None


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


# ── ProteinStructurePrediction: unreadable and incomplete structures ────────
#
# Every metric in the pipeline is derived from these files. A file that will
# not parse has to be named as such, because the alternative is a chain
# sequence read as empty and a prediction scored against nothing. A chain
# that is absent is a different case: the contact set is honestly empty, and
# the geometry helpers return None rather than a number.


def _write_pdb(path: Path, lines) -> Path:
    path.write_text("".join(lines) + "END\n")
    return path


def _atom(serial, name, res, chain, resnum, x, y, z):
    return (f"ATOM  {serial:>5}  {name:<3}{res:>4} {chain}{resnum:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C\n")


@pytest.mark.logic
class TestStructureLoading:
    def test_a_missing_gemmi_names_itself_and_the_file(self, tmp_path,
                                                       monkeypatch):
        """This runs inside the Boltz container, where gemmi is present. On a
        bare host it is not, and the error has to name gemmi rather than
        surfacing as an empty sequence."""
        pdb = _write_pdb(tmp_path / "x.pdb",
                         [_atom(1, "CA", "ALA", "A", 1, 0, 0, 0)])
        monkeypatch.setitem(sys.modules, "gemmi", None)
        pred = psp.ProteinStructurePrediction(
            path=pdb, receptor_chain="A", effector_chain="B")
        with pytest.raises(ImportError, match="gemmi is required"):
            pred.chain_sequence("A")

    def test_an_unparseable_file_is_an_os_error_naming_the_path(self, tmp_path):
        """A truncated or half-written prediction is what a killed Boltz job
        leaves. The path is in the message because the caller is iterating
        over hundreds of them."""
        pytest.importorskip("gemmi")
        bad = tmp_path / "prediction.unknownext"
        bad.write_text("not a structure\n")
        pred = psp.ProteinStructurePrediction(
            path=bad, receptor_chain="A", effector_chain="B")
        with pytest.raises(OSError, match="failed to read structure"):
            pred.chain_sequence("A")

    def test_a_reused_residue_number_is_counted_once(self, tmp_path):
        """Alternate conformations and insertions can repeat a residue
        number. Counting it twice lengthens the receptor sequence, and every
        position index derived from that sequence shifts by one, which moves
        the protected set off the interface."""
        pytest.importorskip("gemmi")
        pdb = _write_pdb(tmp_path / "dup.pdb", [
            _atom(1, "CA", "ALA", "A", 1, 0, 0, 0),
            _atom(2, "CA", "VAL", "A", 1, 1, 0, 0),
            _atom(3, "CA", "SER", "A", 2, 2, 0, 0),
        ])
        pred = psp.ProteinStructurePrediction(
            path=pdb, receptor_chain="A", effector_chain="B")
        assert pred.chain_sequence("A") == "AS"


@pytest.mark.logic
class TestMissingChainGeometry:
    def _single_chain(self, tmp_path, name="single.pdb"):
        return _write_pdb(tmp_path / name, [
            _atom(1, "CA", "ALA", "A", 1, 0, 0, 0),
            _atom(2, "CA", "VAL", "A", 2, 3, 0, 0),
            _atom(3, "CA", "SER", "A", 3, 6, 0, 0),
        ])

    def test_a_prediction_with_no_effector_has_no_contacts(self, tmp_path):
        """A monomer prediction has no interface. An empty contact set in the
        prediction frame is the right answer, and it makes every
        contamination test over it correctly false."""
        pytest.importorskip("gemmi")
        pred = psp.ProteinStructurePrediction(
            path=self._single_chain(tmp_path), receptor_chain="A",
            effector_chain="B")
        got = pred.receptor_contact_residues(4.5)
        assert got.as_sorted_list() == []
        assert got.frame == "prediction"

    def test_two_interfaceless_predictions_have_zero_overlap(self, tmp_path):
        """Jaccard of two empty sets is undefined. Zero is the reported
        value, and a division would raise instead."""
        pytest.importorskip("gemmi")
        one = psp.ProteinStructurePrediction(
            path=self._single_chain(tmp_path, "a.pdb"), receptor_chain="A",
            effector_chain="B")
        two = psp.ProteinStructurePrediction(
            path=self._single_chain(tmp_path, "b.pdb"), receptor_chain="A",
            effector_chain="B")
        assert one.jaccard_interface_vs(two, 4.5) == 0.0


@pytest.mark.logic
class TestAlignmentOnNonFiniteCoordinates:
    """A PDB can carry a residue with no resolved coordinates, which reads
    back as nan. The Kabsch SVD then fails to converge. None means "not
    measured", which the aggregator writes as a blank metric. A number from
    a failed alignment would be read as a real RMSD."""

    def test_kabsch_returns_none_when_the_alignment_does_not_converge(self):
        bad = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, np.nan]])
        good = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
        assert psp._kabsch_rmsd(bad, good) is None

    def test_the_receptor_aligned_effector_rmsd_returns_none_too(self):
        """ra_eff is the headline metric. A nan receptor would otherwise
        propagate into the number the thesis reports."""
        bad_rec = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, np.nan]])
        rec = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
        eff = np.array([[5.0, 0, 0], [6, 0, 0], [5, 1, 0]])
        assert psp._ra_eff_rmsd(bad_rec, eff, rec, eff) is None


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
