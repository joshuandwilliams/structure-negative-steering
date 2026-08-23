"""The dose sweep must differ from the benchmark in dose and nothing else.

scripts/make_dose_sweep.py copies each tracked unconstrained benchmark config and
changes three lines: `name`, `reference_pdb` and `negsteer_n_designs`. If anything
else drifted, the sweep would stop being a dose contrast and become a confound,
and every number it produced would still look plausible.

The generator asserts this when it writes. This asserts it on what is committed,
which is what will actually run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP = REPO_ROOT / "experiments" / "dose-sweep"
BENCH = REPO_ROOT / "experiments" / "benchmarking" / "unconstrained"
LAUNCHER = REPO_ROOT / "scripts" / "run_dose_sweep.slurm.sh"

CONFIGS = sorted(SWEEP.glob("*/*/config.yml"))
needs_sweep = pytest.mark.skipif(not CONFIGS, reason="dose sweep not generated")

# Only these may differ from the benchmark config they came from.
MUTABLE = ("name:", "reference_pdb:", "negsteer_n_designs:")


@pytest.fixture(scope="module")
def drv():
    spec = importlib.util.spec_from_file_location(
        "drv", REPO_ROOT / "scripts" / "negative_steering.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ids(p: Path) -> str:
    return f"{p.parents[1].name}/{p.parent.name}"


@pytest.mark.local_integration
@needs_sweep
def test_the_sweep_covers_every_target_at_every_dose() -> None:
    targets = {p.parent.name for p in BENCH.glob("*/config.yml")}
    doses = {p.parents[1].name for p in CONFIGS}
    assert targets, "benchmark tree missing"
    for dose in doses:
        got = {p.parent.name for p in SWEEP.glob(f"{dose}/*/config.yml")}
        assert got == targets, f"{dose} covers {sorted(got ^ targets)} differently"


@pytest.mark.local_integration
@needs_sweep
@pytest.mark.parametrize("cfg", CONFIGS, ids=_ids)
def test_only_the_dose_lines_differ_from_the_benchmark(cfg: Path) -> None:
    target = cfg.parent.name
    original = (BENCH / target / "config.yml").read_text().splitlines()
    # The generated file carries a header block before the copied body.
    body = cfg.read_text().splitlines()
    body = body[len(body) - len(original):]
    assert len(body) == len(original), f"{cfg}: body length changed"
    for a, b in zip(original, body):
        if a != b:
            assert b.startswith(MUTABLE), f"{cfg}: unexpected change\n  - {a}\n  + {b}"


@pytest.mark.local_integration
@needs_sweep
@pytest.mark.parametrize("cfg", CONFIGS, ids=_ids)
def test_each_config_translates_and_asks_for_its_own_dose(cfg: Path, drv) -> None:
    dose = int(cfg.parents[1].name.lstrip("n"))
    c = drv._load_yaml(cfg)
    cmd = drv._build_run_cmd(c, cfg.parent, cfg.parent / "inputs",
                             cfg.parent / "run", "/fake.img")
    plan = cmd[cmd.index("--plan-extra-args") + 1]
    assert f"--n-designs {dose}" in plan
    # The sweep is the unconstrained arm. Restraints here would make it a
    # different experiment wearing the same name.
    assert "--boltz-constraints" not in plan
    assert Path(cmd[cmd.index("--ground-truth") + 1]).resolve().is_file()


@pytest.mark.local_integration
@needs_sweep
@pytest.mark.parametrize("cfg", CONFIGS, ids=_ids)
def test_run_names_are_unique_and_carry_the_dose(cfg: Path, drv) -> None:
    dose = cfg.parents[1].name
    name = drv._load_yaml(cfg)["name"]
    assert name == f"{cfg.parent.name}_{dose}", (
        "the run name is the engine's --seq-name and lands in every summary. "
        "Two runs sharing one would be indistinguishable downstream.")


@pytest.mark.local_unit
def test_the_launcher_delegates_and_asks_for_no_gpu() -> None:
    text = LAUNCHER.read_text()
    assert "run_workflow.slurm.sh" in text, "must not duplicate the bootstrap"
    header = [ln for ln in text.splitlines() if ln.startswith("#SBATCH")]
    assert not any("gres" in ln or "jic-gpu" in ln for ln in header), \
        "the head process only submits and polls"
