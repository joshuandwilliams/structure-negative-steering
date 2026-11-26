"""Tests for the config-driven runner, bin/negative_steering.py.

This is the half of the runner that reads a target's ``config.yml`` and writes
``run/launch.sh``. Everything it does is a pure translation from config keys to
an engine command line, so it is testable without a GPU or a container.

The engine invocation is asserted against the known answer rather than against
"a command was produced". A test that only checked launch.sh was non-empty
would pass even if every steering knob were dropped.
"""

from __future__ import annotations

import importlib.util
import json
import shlex
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER = REPO_ROOT / "bin" / "negative_steering.py"


@pytest.fixture(scope="module")
def drv():
    spec = importlib.util.spec_from_file_location("negative_steering_driver", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_config(tmp_path: Path, **overrides) -> Path:
    """A minimal valid config, plus whatever the test wants to change."""
    cfg = {
        "name": "TEST",
        "reference_pdb": "./ref.pdb",
        "receptor_chain": "A",
        "effector_chain": "B",
        "boltz_container": "/fake/boltz.img",
    }
    cfg.update(overrides)
    lines = []
    for k, v in cfg.items():
        if isinstance(v, bool):
            lines.append(f"{k}: {'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            lines.append(f"{k}: {v}")
        else:
            lines.append(f'{k}: "{v}"')
    path = tmp_path / "config.yml"
    path.write_text("\n".join(lines) + "\n")
    (tmp_path / "ref.pdb").write_text("ATOM      1  CA  ALA A   1       0.0 0.0 0.0\n")
    return path


# ── config validation ────────────────────────────────────────────────────────

@pytest.mark.logic
def test_missing_required_key_is_rejected(drv, tmp_path, capsys):
    cfg = tmp_path / "config.yml"
    cfg.write_text('name: "X"\nreceptor_chain: "A"\n')
    assert drv.main([str(cfg)]) == 2
    assert "missing required keys" in capsys.readouterr().err


@pytest.mark.logic
def test_absent_config_is_rejected(drv, tmp_path, capsys):
    assert drv.main([str(tmp_path / "nope.yml")]) == 2
    assert "config not found" in capsys.readouterr().err


@pytest.mark.logic
def test_absent_reference_pdb_is_rejected(drv, tmp_path, capsys):
    cfg = _write_config(tmp_path)
    (tmp_path / "ref.pdb").unlink()
    assert drv.main([str(cfg)]) == 2
    assert "reference_pdb not found" in capsys.readouterr().err


@pytest.mark.logic
def test_non_mapping_config_is_rejected(drv, tmp_path):
    cfg = tmp_path / "config.yml"
    cfg.write_text("- just\n- a\n- list\n")
    with pytest.raises(SystemExit, match="did not parse to a mapping"):
        drv.main([str(cfg)])


@pytest.mark.logic
def test_provide_interface_mode_needs_a_file(drv, tmp_path):
    cfg = _write_config(tmp_path, interface_mode="provide")
    with pytest.raises(SystemExit, match="interface_residues_file"):
        drv.main([str(cfg), "--dry-run"])


# ── the engine command line ──────────────────────────────────────────────────

@pytest.mark.logic
def test_steering_knobs_reach_the_engine(drv, tmp_path):
    """Every negsteer_* key becomes the matching engine flag."""
    cfg = _write_config(
        tmp_path, negsteer_mode="mild", negsteer_n_designs=20,
        negsteer_num_seeds=3, negsteer_max_mutations=6,
        negsteer_candidate_pool_size=12, negsteer_diffusion_samples=5,
        negsteer_recycling_steps=3, negsteer_seed=0,
        negsteer_rmsd_threshold=5.0, negsteer_contact_cutoff=5.0,
        negsteer_protected_set_source="true_interface",
    )
    data = drv._load_yaml(cfg)
    plan = drv._build_plan_extra_args(data)

    for flag, value in [
        ("--mode", "mild"), ("--n-designs", "20"), ("--num-seeds", "3"),
        ("--max-mutations", "6"), ("--candidate-pool-size", "12"),
        ("--diffusion-samples", "5"), ("--recycling-steps", "3"),
        ("--seed", "0"), ("--rmsd-threshold", "5.0"),
        ("--contact-cutoff", "5.0"),
        ("--protected-set-source", "true_interface"),
    ]:
        assert f"{flag} {value}" in plan, f"{flag} did not reach the engine"


@pytest.mark.logic
@pytest.mark.parametrize("key,flag,when", [
    ("negsteer_boltz_constraints", "--boltz-constraints", True),
    ("negsteer_no_kernels", "--no-kernels", True),
])
def test_opt_in_flags_appear_only_when_set(drv, tmp_path, key, flag, when):
    on = drv._build_plan_extra_args(drv._load_yaml(_write_config(tmp_path, **{key: when})))
    off = drv._build_plan_extra_args(drv._load_yaml(_write_config(tmp_path, **{key: not when})))
    assert flag in on
    assert flag not in off


@pytest.mark.logic
def test_the_effector_template_is_forwarded_only_when_the_config_asks(
        drv, tmp_path):
    """Opt-in on both sides. The engine's own default is off, so a config
    that says false, or omits the key, must forward no flag at all rather
    than restate the default."""
    on = drv._build_plan_extra_args(drv._load_yaml(
        _write_config(tmp_path, negsteer_effector_template=True)))
    off = drv._build_plan_extra_args(drv._load_yaml(
        _write_config(tmp_path, negsteer_effector_template=False)))
    assert "--effector-template" in on
    assert "--effector-template" not in off
    assert "--no-effector-template" not in on
    assert "--no-effector-template" not in off


@pytest.mark.logic
def test_relative_paths_resolve_against_the_config_directory(drv, tmp_path):
    cfg = _write_config(tmp_path)
    data = drv._load_yaml(cfg)
    cmd = drv._build_run_cmd(data, tmp_path, tmp_path / "inputs",
                             tmp_path / "run", "/fake/boltz.img")
    gt = cmd[cmd.index("--ground-truth") + 1]
    assert Path(gt).is_absolute() and Path(gt) == tmp_path / "ref.pdb"


@pytest.mark.logic
def test_derive_mode_passes_the_contact_cutoff(drv, tmp_path):
    cfg = _write_config(tmp_path, interface_contact_cutoff=4.5)
    cmd = drv._build_prepare_cmd(drv._load_yaml(cfg), tmp_path, tmp_path / "inputs")
    assert "--derive" in cmd
    assert cmd[cmd.index("--contact-cutoff") + 1] == "4.5"


@pytest.mark.logic
def test_design_region_file_is_used_when_it_is_not_a_keyword(drv, tmp_path):
    cfg = _write_config(tmp_path, design_region="./my_region.txt")
    cmd = drv._build_prepare_cmd(drv._load_yaml(cfg), tmp_path, tmp_path / "inputs")
    assert "--design-region-file" in cmd
    assert "--design-region" not in cmd


# ── interface ablation ───────────────────────────────────────────────────────

@pytest.mark.logic
def test_no_ablation_by_default(drv, tmp_path):
    cmd = drv._build_prepare_cmd(drv._load_yaml(_write_config(tmp_path)),
                                 tmp_path, tmp_path / "inputs")
    assert "--ablate" not in cmd


@pytest.mark.logic
@pytest.mark.parametrize("mode", ["alanine", "scrambled"])
def test_ablation_mode_reaches_prepare(drv, tmp_path, mode):
    cfg = _write_config(tmp_path, interface_ablation=mode)
    cmd = drv._build_prepare_cmd(drv._load_yaml(cfg), tmp_path, tmp_path / "inputs")
    assert cmd[cmd.index("--ablate") + 1] == mode
    assert "--ablate-seed" not in cmd


@pytest.mark.logic
def test_ablation_seed_is_forwarded_when_set(drv, tmp_path):
    cfg = _write_config(tmp_path, interface_ablation="scrambled", ablation_seed=99)
    cmd = drv._build_prepare_cmd(drv._load_yaml(cfg), tmp_path, tmp_path / "inputs")
    assert cmd[cmd.index("--ablate-seed") + 1] == "99"


@pytest.mark.logic
def test_unknown_ablation_mode_is_rejected(drv, tmp_path):
    cfg = _write_config(tmp_path, interface_ablation="shuffled")
    with pytest.raises(SystemExit) as exc:
        drv._build_prepare_cmd(drv._load_yaml(cfg), tmp_path, tmp_path / "inputs")
    assert "interface_ablation" in str(exc.value)


@pytest.mark.logic
def test_ablation_does_not_touch_the_ground_truth(drv, tmp_path):
    """The control depends on the reference PDB staying the native complex."""
    cfg = _write_config(tmp_path, interface_ablation="alanine")
    loaded = drv._load_yaml(cfg)
    run = drv._build_run_cmd(loaded, tmp_path, tmp_path / "inputs",
                             tmp_path / "run", "/fake.img")
    assert run[run.index("--ground-truth") + 1] == str(tmp_path / "ref.pdb")
    assert "--ablate" not in run


# ── launch.sh ────────────────────────────────────────────────────────────────

@pytest.mark.logic
def test_launch_script_is_executable_and_runs_the_orchestrator(drv, tmp_path):
    run_dir = tmp_path / "run"
    cmd = ["bash", "/x/negative_steering_run_one.sh", "--seq-name", "a b"]
    launch = drv._write_launch_script(run_dir, cmd)
    assert launch.exists() and launch.stat().st_mode & 0o111
    body = launch.read_text()
    assert body.startswith("#!/bin/bash")
    assert "set -euo pipefail" in body
    # The name with a space must survive as one argument.
    assert shlex.split(body.splitlines()[-1])[1:] == cmd


@pytest.mark.logic
def test_dry_run_writes_nothing(drv, tmp_path, capsys):
    cfg = _write_config(tmp_path)
    assert drv.main([str(cfg), "--dry-run"]) == 0
    assert not (tmp_path / "run").exists()
    assert not (tmp_path / "inputs").exists()
    assert "prepare inputs" in capsys.readouterr().out


@pytest.mark.logic
def test_dry_run_reports_when_no_container_is_configured(drv, tmp_path, capsys):
    cfg = tmp_path / "config.yml"
    cfg.write_text('name: "T"\nreference_pdb: "./ref.pdb"\n'
                   'receptor_chain: "A"\neffector_chain: "B"\n')
    (tmp_path / "ref.pdb").write_text("ATOM\n")
    assert drv.main([str(cfg), "--dry-run"]) == 0
    assert "no boltz_container" in capsys.readouterr().out


@pytest.mark.logic
def test_missing_prepared_inputs_are_detected(drv, tmp_path):
    with pytest.raises(SystemExit, match="prepare did not produce"):
        drv._check_inputs_present(tmp_path)


# ── the full main() path ─────────────────────────────────────────────────────

@pytest.mark.logic
def test_full_run_prepares_inputs_then_writes_launch(drv, tmp_path, monkeypatch, capsys):
    """The happy path: prepare is invoked, then launch.sh is written from it."""
    cfg = _write_config(tmp_path)
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        out = Path(cmd[cmd.index("--outdir") + 1])
        out.mkdir(parents=True, exist_ok=True)
        for name in drv.INPUT_FILES:
            (out / name).write_text("stub\n")
        return None

    monkeypatch.setattr(drv.subprocess, "run", fake_run)
    assert drv.main([str(cfg)]) == 0

    assert len(calls) == 1 and str(drv.PREPARE) in calls[0]
    launch = tmp_path / "run" / "launch.sh"
    assert launch.is_file()
    assert "negative_steering_run_one.sh" in launch.read_text()
    assert "LAUNCH_SCRIPT=" in capsys.readouterr().out


@pytest.mark.logic
def test_prepare_only_stops_before_launch(drv, tmp_path, monkeypatch, capsys):
    cfg = _write_config(tmp_path)

    def fake_run(cmd, check):
        out = Path(cmd[cmd.index("--outdir") + 1])
        out.mkdir(parents=True, exist_ok=True)
        for name in drv.INPUT_FILES:
            (out / name).write_text("stub\n")

    monkeypatch.setattr(drv.subprocess, "run", fake_run)
    assert drv.main([str(cfg), "--prepare-only"]) == 0
    assert not (tmp_path / "run" / "launch.sh").exists()
    assert "prepare-only" in capsys.readouterr().out


def _stub_prepare(drv, monkeypatch):
    """Let the prepare step appear to succeed, so the container check is what
    the test reaches."""
    def fake_run(cmd, check):
        out = Path(cmd[cmd.index("--outdir") + 1])
        out.mkdir(parents=True, exist_ok=True)
        for name in drv.INPUT_FILES:
            (out / name).write_text("stub\n")
    monkeypatch.setattr(drv.subprocess, "run", fake_run)


def _minimal_config(tmp_path):
    cfg = tmp_path / "config.yml"
    cfg.write_text('name: "T"\nreference_pdb: "./ref.pdb"\n'
                   'receptor_chain: "A"\neffector_chain: "B"\n')
    (tmp_path / "ref.pdb").write_text("ATOM\n")
    return cfg


@pytest.mark.logic
def test_run_without_a_container_is_rejected_after_prepare(drv, tmp_path, monkeypatch, capsys):
    """The last link in the chain. --boltz-container, then the config, then
    NEGSTEER_BOLTZ_IMG, and with none of the three there is no image to run
    Boltz in, so launch.sh cannot be written.

    The environment is cleared on purpose. Without that this passes or fails
    depending on whether whoever runs it has a site.conf loaded."""
    monkeypatch.delenv("NEGSTEER_BOLTZ_IMG", raising=False)
    cfg = _minimal_config(tmp_path)
    _stub_prepare(drv, monkeypatch)

    assert drv.main([str(cfg)]) == 2
    err = capsys.readouterr().err
    assert "no Boltz image" in err
    assert "NEGSTEER_BOLTZ_IMG" in err, "the message must name the way out"


@pytest.mark.logic
def test_the_environment_supplies_the_container_when_the_config_does_not(
        drv, tmp_path, monkeypatch):
    """A config that leaves boltz_container null relies on this fallback. It
    is what makes a whole tree of such configs runnable from one site setting."""
    monkeypatch.setenv("NEGSTEER_BOLTZ_IMG", "/img/boltz.img")
    cfg = _minimal_config(tmp_path)
    _stub_prepare(drv, monkeypatch)

    assert drv.main([str(cfg)]) == 0
    assert "/img/boltz.img" in (tmp_path / "run" / "launch.sh").read_text()


@pytest.mark.logic
def test_the_config_beats_the_environment(drv, tmp_path, monkeypatch):
    """A config that names an image means it, so the site default must not
    silently replace it."""
    monkeypatch.setenv("NEGSTEER_BOLTZ_IMG", "/img/site.img")
    cfg = tmp_path / "config.yml"
    cfg.write_text('name: "T"\nreference_pdb: "./ref.pdb"\n'
                   'receptor_chain: "A"\neffector_chain: "B"\n'
                   'boltz_container: "/img/from_config.img"\n')
    (tmp_path / "ref.pdb").write_text("ATOM\n")
    _stub_prepare(drv, monkeypatch)

    assert drv.main([str(cfg)]) == 0
    launch = (tmp_path / "run" / "launch.sh").read_text()
    assert "/img/from_config.img" in launch
    assert "/img/site.img" not in launch


@pytest.mark.logic
def test_missing_orchestrator_is_rejected(drv, tmp_path, monkeypatch, capsys):
    cfg = _write_config(tmp_path)
    monkeypatch.setattr(drv, "ORCHESTRATOR", tmp_path / "absent.sh")
    assert drv.main([str(cfg)]) == 2
    assert "orchestrator not found" in capsys.readouterr().err


@pytest.mark.logic
def test_yaml_absence_is_reported_with_guidance(drv, tmp_path, monkeypatch):
    """Without PyYAML the driver says where it ships, rather than tracebacking."""
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) \
        else __builtins__.__import__

    def blocked(name, *a, **k):
        if name == "yaml":
            raise ImportError("no yaml")
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", blocked)
    with pytest.raises(SystemExit, match="PyYAML is required"):
        drv._load_yaml(_write_config(tmp_path))


@pytest.mark.logic
def test_provide_interface_mode_passes_the_file(drv, tmp_path):
    iface = tmp_path / "iface.txt"
    iface.write_text("1\n2\n3\n")
    cfg = _write_config(tmp_path, interface_mode="provide",
                        interface_residues_file="./iface.txt")
    cmd = drv._build_prepare_cmd(drv._load_yaml(cfg), tmp_path, tmp_path / "inputs")
    assert "--interface-file" in cmd
    assert cmd[cmd.index("--interface-file") + 1] == str(iface)


# ── --emit-json ──────────────────────────────────────────────────────────────
#
# main.nf reads this instead of parsing the YAML itself, so every config key
# and its default is defined in exactly one place. A key that stops being
# emitted, or a default that drifts, changes what the workflow runs without
# changing any config file.

@pytest.mark.logic
def test_emit_json_carries_every_key_the_workflow_reads(drv, tmp_path, capsys):
    cfg = _write_config(tmp_path)
    assert drv.main([str(cfg), "--emit-json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert set(doc) == {
        "name", "config", "cfg_dir", "inputs_dir", "run_dir", "container",
        "plan_args", "postprocess_rmsd_threshold",
        "postprocess_metric_column", "postprocess_contact_cutoff",
    }
    assert doc["name"] == "TEST"
    assert doc["container"] == "/fake/boltz.img"


@pytest.mark.logic
def test_emit_json_defaults_match_the_engine_defaults(drv, tmp_path, capsys):
    """These are the values a config does not have to state, so they are the
    ones that can drift unnoticed."""
    cfg = _write_config(tmp_path)
    assert drv.main([str(cfg), "--emit-json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["postprocess_rmsd_threshold"] == "5.0"
    assert doc["postprocess_metric_column"] == "steered_ra_eff_vs_truth"
    assert doc["postprocess_contact_cutoff"] == "5.0"


@pytest.mark.logic
def test_emit_json_takes_the_config_over_the_default(drv, tmp_path, capsys):
    cfg = _write_config(tmp_path,
                        negsteer_postprocess_rmsd_threshold=3.5,
                        negsteer_postprocess_metric_column="other_column",
                        negsteer_postprocess_contact_cutoff=7.0)
    assert drv.main([str(cfg), "--emit-json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["postprocess_rmsd_threshold"] == "3.5"
    assert doc["postprocess_metric_column"] == "other_column"
    assert doc["postprocess_contact_cutoff"] == "7.0"


@pytest.mark.logic
def test_emit_json_plan_args_name_the_derived_input_files(drv, tmp_path,
                                                          capsys):
    cfg = _write_config(tmp_path)
    assert drv.main([str(cfg), "--emit-json"]) == 0
    plan = json.loads(capsys.readouterr().out)["plan_args"]
    for flag in ("--ground-truth", "--receptor", "--effector",
                 "--receptor-fasta", "--effector-fasta",
                 "--true-interface-indices-file",
                 "--design-region-indices-file"):
        assert flag in plan
    assert "true_interface.txt" in plan and "design_region.txt" in plan


@pytest.mark.logic
def test_emit_json_falls_back_to_the_directory_name(drv, tmp_path, capsys):
    """A config with no name is keyed on its folder, which is what the run
    directory is named after."""
    cfg = _write_config(tmp_path)
    cfg.write_text("\n".join(
        ln for ln in cfg.read_text().splitlines() if not ln.startswith("name:")))
    assert drv.main([str(cfg), "--emit-json"]) == 0
    assert json.loads(capsys.readouterr().out)["name"] == tmp_path.name


@pytest.mark.logic
def test_emit_json_leaves_the_container_blank_when_there_is_none(drv, tmp_path,
                                                                 capsys):
    cfg = _write_config(tmp_path)
    cfg.write_text("\n".join(
        ln for ln in cfg.read_text().splitlines()
        if not ln.startswith("boltz_container:")))
    assert drv.main([str(cfg), "--emit-json"]) == 0
    assert json.loads(capsys.readouterr().out)["container"] == ""
