"""The tool interface: `negsteer run`.

The point of these is not that the CLI parses arguments. It is that the CLI
runs the SAME nine stages, in the same order, with the same arguments as
bin/negative_steering_run_one.sh, which produced the committed benchmark.
A caller swapping the shell orchestrator for this must get the same science.

The stage commands are captured from a --dry-run rather than executed, so
these need no GPU, no container and no Boltz.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from negsteer import OUTPUT_CONTRACT_VERSION, OUTPUT_FILES, RESULT_MANIFEST
from negsteer.cli import build_parser, cmd_run

REPO_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR = REPO_ROOT / "bin" / "negative_steering_run_one.sh"

# The stage sequence, as the shell orchestrator runs it. Read off
# negative_steering_run_one.sh, which is the reference implementation.
EXPECTED_STAGES = [
    ("boltz2_negative_steering.py", "plan"),
    ("boltz2_negative_steering.py", "collect"),
    ("boltz2_iterate_steering.py", "kickoff-distances"),
    ("boltz2_iterate_steering.py", "kickoff-prefilter"),
    ("boltz2_iterate_steering.py", "build-contaminated"),
    ("boltz2_iterate_steering.py", "plan-reversions"),
    ("boltz2_iterate_steering.py", "harvest-reversions"),
    ("boltz2_iterate_steering.py", "kickoff-finalize"),
    ("boltz2_iterate_steering.py", "aggregate"),
    ("boltz2_iterate_steering.py", "compute-final-metrics"),
    ("boltz2_iterate_steering.py", "aggregate-per-sequence"),
    ("extract_passing.py", None),
    ("cross_summary_cli.py", None),
]


def _invoke(tmp_path: Path, capsys, *extra: str):
    """Run the CLI in dry mode and return (exit code, captured stage commands)."""
    argv = [
        "run",
        "--name", "unit",
        "--receptor-fasta", str(tmp_path / "r.fasta"),
        "--effector-fasta", str(tmp_path / "e.fasta"),
        "--reference-pdb", str(tmp_path / "ref.pdb"),
        "--design-region", str(tmp_path / "dr.txt"),
        "--true-interface", str(tmp_path / "ti.txt"),
        "--outdir", str(tmp_path / "out"),
        "--boltz-container", str(tmp_path / "img.img"),
        "--dry-run",
        *extra,
    ]
    args = build_parser().parse_args(argv)
    rc = cmd_run(args)
    lines = [ln.strip()[2:] for ln in capsys.readouterr().out.splitlines()
             if ln.strip().startswith("$ ")]
    return rc, lines


def _stage_of(cmd: str) -> tuple[str, str | None]:
    parts = cmd.split()
    script = Path(parts[1]).name
    sub = parts[2] if len(parts) > 2 and not parts[2].startswith("-") else None
    return script, sub


@pytest.mark.local_unit
def test_the_stage_sequence_matches_the_shell_orchestrator(tmp_path, capsys):
    """The one assertion that protects the science. If this fails, the tool and
    the runner that produced the benchmark no longer do the same thing."""
    rc, cmds = _invoke(tmp_path, capsys)
    assert rc == 0
    assert [_stage_of(c) for c in cmds] == EXPECTED_STAGES


@pytest.mark.local_unit
def test_every_stage_the_orchestrator_runs_is_one_the_cli_runs_too():
    """Guards the direction the list above cannot: a stage added to the shell
    orchestrator later, and not to the CLI, would silently diverge."""
    shell = ORCHESTRATOR.read_text()
    for _script, sub in EXPECTED_STAGES:
        if sub:
            assert re.search(rf"\b{re.escape(sub)}\b", shell), \
                f"{sub} is in the CLI but not in the orchestrator"


@pytest.mark.local_unit
def test_finalize_is_pinned_to_a_single_cycle(tmp_path, capsys):
    """--max-cycles 0 is what stops the engine submitting a follow-up cycle.
    Dropping it would make the tool spawn SLURM jobs its caller never asked
    for, from inside a task the caller thinks is one job."""
    _, cmds = _invoke(tmp_path, capsys)
    finalize = next(c for c in cmds if "kickoff-finalize" in c)
    assert "--max-cycles 0" in finalize


@pytest.mark.local_unit
def test_an_unset_knob_is_not_forwarded(tmp_path, capsys):
    """The engine owns its defaults. Restating them here would fork them, and
    the fork would be invisible until the engine's default changed."""
    _, cmds = _invoke(tmp_path, capsys)
    plan = next(c for c in cmds if " plan " in c)
    for absent in ("--n-designs", "--num-seeds", "--diffusion-samples",
                   "--max-mutations", "--seed"):
        assert absent not in plan, f"{absent} was forwarded without being set"


@pytest.mark.local_unit
def test_a_set_knob_reaches_the_plan_stage(tmp_path, capsys):
    _, cmds = _invoke(tmp_path, capsys, "--n-designs", "7", "--boltz-constraints")
    plan = next(c for c in cmds if " plan " in c)
    assert "--n-designs 7" in plan
    assert "--boltz-constraints" in plan


@pytest.mark.local_unit
def test_plan_extra_args_is_forwarded_verbatim(tmp_path, capsys):
    """The escape hatch. A caller must be able to reach an engine flag this
    CLI does not name, or the CLI becomes a bottleneck on the engine."""
    _, cmds = _invoke(tmp_path, capsys, "--plan-extra-args", "--mode aggressive")
    plan = next(c for c in cmds if " plan " in c)
    assert "--mode aggressive" in plan


@pytest.mark.local_unit
def test_it_refuses_to_run_outside_a_container_without_an_image(tmp_path, monkeypatch):
    """Failing here is a clear error. Failing later is a Boltz stack trace
    about a missing GPU backend, which reads as a cluster problem."""
    for var in ("SINGULARITY_NAME", "SINGULARITY_CONTAINER",
                "APPTAINER_NAME", "APPTAINER_CONTAINER"):
        monkeypatch.delenv(var, raising=False)
    args = build_parser().parse_args([
        "run", "--name", "x",
        "--receptor-fasta", "r", "--effector-fasta", "e",
        "--reference-pdb", "p", "--design-region", "d", "--true-interface", "t",
        "--outdir", str(tmp_path / "out"), "--dry-run",
    ])
    assert cmd_run(args) == 2


@pytest.mark.local_unit
def test_inside_a_container_the_running_image_is_used(tmp_path, monkeypatch, capsys):
    """The caller should not have to name an image it is already running
    inside. The runtime exports the path, so use it."""
    monkeypatch.setenv("SINGULARITY_CONTAINER", "/images/negsteer.img")
    args = build_parser().parse_args([
        "run", "--name", "x",
        "--receptor-fasta", "r", "--effector-fasta", "e",
        "--reference-pdb", "p", "--design-region", "d", "--true-interface", "t",
        "--outdir", str(tmp_path / "out"), "--dry-run",
    ])
    assert cmd_run(args) == 0
    assert "--boltz-container /images/negsteer.img" in capsys.readouterr().out


@pytest.mark.local_unit
@pytest.mark.parametrize("missing", [
    "--name", "--receptor-fasta", "--effector-fasta",
    "--reference-pdb", "--design-region", "--true-interface", "--outdir",
])
def test_every_required_input_is_required(missing):
    argv = ["run"]
    for flag in ("--name", "--receptor-fasta", "--effector-fasta",
                 "--reference-pdb", "--design-region", "--true-interface",
                 "--outdir"):
        if flag != missing:
            argv += [flag, "x"]
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


# ── the output contract ──────────────────────────────────────────────────────

@pytest.mark.local_unit
def test_the_manifest_is_written_and_names_every_contract_file(tmp_path, capsys):
    """A caller reads the manifest instead of globbing. If a name here changes
    without the contract version changing, the caller breaks silently."""
    _invoke(tmp_path, capsys)
    # --dry-run skips the write, so exercise the real path with a live outdir.
    args = build_parser().parse_args([
        "run", "--name", "unit",
        "--receptor-fasta", "r", "--effector-fasta", "e",
        "--reference-pdb", "p", "--design-region", "d", "--true-interface", "t",
        "--outdir", str(tmp_path / "live"),
        "--boltz-container", "img", "--dry-run",
    ])
    args.dry_run = False
    # The engine writes plan.json; stand in for it, since the stages are stubbed.
    (tmp_path / "live" / "cycle_0").mkdir(parents=True)
    (tmp_path / "live" / "cycle_0" / "plan.json").write_text('{"designs": []}')

    import negsteer.cli as cli
    calls = []

    def fake_run(argv, *, tolerate=None, dry=False):
        calls.append(argv)
        return 0

    orig, cli._run = cli._run, fake_run
    try:
        assert cmd_run(args) == 0
    finally:
        cli._run = orig

    manifest = json.loads((tmp_path / "live" / RESULT_MANIFEST).read_text())
    assert manifest["contract_version"] == OUTPUT_CONTRACT_VERSION
    assert manifest["name"] == "unit"
    assert set(manifest["outputs"]) == set(OUTPUT_FILES)
    # Nothing actually ran, so every output is reported absent rather than
    # asserted present. Reporting presence is the contract.
    assert manifest["outputs"]["passing_summary"] is None


@pytest.mark.local_unit
def test_the_runtime_file_is_written_before_extract_passing_reads_it(
        tmp_path, capsys):
    """extract_passing injects run_one_runtime_sec into passing_summary.csv,
    so a runtime stamped after it would be silently dropped from the table."""
    _, cmds = _invoke(tmp_path, capsys)
    order = [i for i, c in enumerate(cmds) if "extract_passing" in c]
    assert order, "extract_passing never ran"
    # The stamp happens between aggregate-per-sequence and extract_passing.
    per_seq = next(i for i, c in enumerate(cmds) if "aggregate-per-sequence" in c)
    assert per_seq < order[0]


@pytest.mark.local_unit
def test_tiering_failure_does_not_fail_the_run(tmp_path, monkeypatch):
    """The engine results are already on disk by stage 9. A tiering hiccup
    must not turn a completed run into a failed one."""
    import negsteer.cli as cli

    def flaky(argv, *, tolerate=None, dry=False):
        if "cross_summary_cli" in " ".join(str(a) for a in argv):
            raise cli.StageError("tiering blew up")
        return 0

    args = build_parser().parse_args([
        "run", "--name", "x",
        "--receptor-fasta", "r", "--effector-fasta", "e",
        "--reference-pdb", "p", "--design-region", "d", "--true-interface", "t",
        "--outdir", str(tmp_path / "out"), "--boltz-container", "img",
    ])
    (tmp_path / "out" / "cycle_0").mkdir(parents=True)
    (tmp_path / "out" / "cycle_0" / "plan.json").write_text('{"designs": []}')
    monkeypatch.setattr(cli, "_run", flaky)
    assert cmd_run(args) == 0

    manifest = json.loads((tmp_path / "out" / RESULT_MANIFEST).read_text())
    assert manifest["tiering_succeeded"] is False
    assert manifest["status"] == "ok"


# ── locating the engine ──────────────────────────────────────────────────────

@pytest.mark.local_unit
def test_bin_is_found_in_a_source_checkout():
    from negsteer.cli import BIN
    assert (BIN / "boltz2_negative_steering.py").is_file()


@pytest.mark.local_unit
def test_an_explicit_override_wins(tmp_path, monkeypatch):
    """The installed package sits in site-packages, nowhere near bin/, so the
    lookup cannot assume they are siblings. NEGSTEER_BIN is the escape hatch
    for a layout neither default anticipates."""
    fake = tmp_path / "engine"
    fake.mkdir()
    (fake / "boltz2_negative_steering.py").write_text("")
    monkeypatch.setenv("NEGSTEER_BIN", str(fake))

    import importlib

    import negsteer.cli as cli
    assert importlib.reload(cli).BIN == fake
    monkeypatch.delenv("NEGSTEER_BIN")
    importlib.reload(cli)


@pytest.mark.local_unit
def test_an_override_pointing_nowhere_is_an_error_not_a_fallback(tmp_path, monkeypatch):
    """Falling through to the source checkout would run a different engine
    than the caller asked for, and a typo would look like it worked."""
    monkeypatch.setenv("NEGSTEER_BIN", str(tmp_path / "nope"))
    import importlib

    import negsteer.cli as cli
    with pytest.raises(FileNotFoundError, match="NEGSTEER_BIN is set to"):
        cli._find_bin()
    monkeypatch.delenv("NEGSTEER_BIN")
    importlib.reload(cli)


# ── the CLI satisfies the engine's own argument requirements ─────────────────

def _required_flags(subcommand: str) -> set[str]:
    """Flags the engine marks required=True for a subcommand.

    Read from the source rather than by importing, because the engine pulls in
    numpy and gemmi at module scope and this must run with neither.
    """
    src = (REPO_ROOT / "bin" / "boltz2_negative_steering.py").read_text()
    start = src.index(f'"{subcommand}"')
    end = src.find("add_parser(", start + 1)
    block = src[start:end if end != -1 else len(src)]
    return set(re.findall(
        r'add_argument\(\s*"(--[a-z0-9-]+)"[^)]*required\s*=\s*True', block))


@pytest.mark.local_unit
def test_the_plan_invocation_satisfies_every_flag_the_engine_requires(
        tmp_path, capsys, monkeypatch):
    """The gap that let a broken command reach the cluster.

    The stage-sequence test asserts the CLI runs `plan`. It says nothing about
    whether plan will accept the arguments. --boltz-container is required by
    the engine unconditionally, including inside a container where it goes
    unused, and omitting it failed 20 seconds into a GPU job rather than here.
    """
    monkeypatch.delenv("SINGULARITY_CONTAINER", raising=False)
    _, cmds = _invoke(tmp_path, capsys)
    plan = next(c for c in cmds if " plan " in c)

    required = _required_flags("plan")
    assert required, "found no required flags; the parser shape changed"
    missing = [f for f in required if f not in plan]
    assert not missing, f"plan requires {missing}, which the CLI does not pass"


@pytest.mark.local_unit
def test_the_engine_really_does_require_a_container_even_in_one():
    """Pins the surprise itself. If the engine ever relaxes this, the extra
    argument above becomes unnecessary rather than wrong, and this says so."""
    assert "--boltz-container" in _required_flags("plan")


@pytest.mark.local_unit
def test_the_cli_can_set_every_plan_knob_the_config_driver_can(tmp_path, capsys):
    """The omission the GPU comparison caught, generalised.

    negsteer_mode was absent from the CLI, so a caller could not ask for the
    benchmark's 'mild' and silently got the engine's 'strong' instead. Same
    positions mutated, different residues, and a 2.2 A difference in ra_eff.
    Nothing failed; the answer was just quietly different.

    Every knob scripts/negative_steering.py forwards to plan must therefore be
    reachable from the CLI. Flags exempt below are structural rather than
    scientific: paths and chain ids the CLI names differently, and control flow
    it owns.
    """
    STRUCTURAL = {
        "--workdir", "--index", "--skip-steering", "--boltz-container",
        "--ground-truth", "--receptor", "--effector",
        "--receptor-fasta", "--effector-fasta",
        "--true-interface-indices", "--true-interface-indices-file",
        "--design-region-indices-file",
        "--effector-template",  # the default; --no-effector-template negates it
    }
    src = (REPO_ROOT / "bin" / "boltz2_negative_steering.py").read_text()
    start = src.index('"plan"')
    end = src.find("add_parser(", start + 1)
    plan_flags = set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"',
                                src[start:end if end != -1 else len(src)]))

    cli_flags = {a.option_strings[0]
                 for a in build_parser()._subparsers._group_actions[0]
                 .choices["run"]._actions if a.option_strings}

    missing = sorted(plan_flags - STRUCTURAL - cli_flags)
    assert not missing, (
        f"plan accepts {missing}, which the CLI gives no way to set. "
        "A caller cannot reach them, and the engine default applies silently.")


@pytest.mark.local_unit
def test_the_two_rmsd_thresholds_go_to_different_stages(tmp_path, capsys):
    """They are separate config keys and mean different things. The plan one
    decides whether steering runs at all; the postprocess one only scores what
    came out. Sending one value to both would couple two decisions."""
    _, cmds = _invoke(tmp_path, capsys,
                      "--rmsd-threshold", "4.0",
                      "--postprocess-rmsd-threshold", "6.0")
    plan = next(c for c in cmds if " plan " in c)
    metrics = next(c for c in cmds if "compute-final-metrics" in c)
    assert "--rmsd-threshold 4.0" in plan
    assert "--rmsd-threshold 6.0" in metrics
