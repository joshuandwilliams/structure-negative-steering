"""The bash container resolver must agree with a real YAML parser.

scripts/resolve_boltz_container.sh reads `boltz_container:` out of a config
with grep and sed, because the stage that needs the answer is the one that
cannot use a parser: the airgapped host has no PyYAML, and PyYAML lives in the
container the answer points at.

Text extraction that quietly disagreed with the parser would send a run to a
different image than the config names, and every number it produced would still
look plausible. So the two are asserted equal on every config in the repo.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESOLVER = REPO_ROOT / "scripts" / "resolve_boltz_container.sh"


def _resolve(config: Path, override: str = "") -> str:
    out = subprocess.run(
        ["bash", str(RESOLVER), str(config), override],
        capture_output=True, text=True, check=True,
    )
    return out.stdout


def _all_configs() -> list[Path]:
    found = sorted((REPO_ROOT / "experiments" / "benchmarking").glob("*/*/config.yml"))
    found += sorted((REPO_ROOT / "tests").glob("*/config.yml"))
    return found


@pytest.mark.local_integration
@pytest.mark.parametrize(
    "config", _all_configs(), ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}"
)
def test_bash_resolution_matches_pyyaml(config: Path) -> None:
    parsed = yaml.safe_load(config.read_text()).get("boltz_container", "")
    assert _resolve(config) == (parsed or "")


@pytest.mark.local_unit
def test_override_wins_over_the_config(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yml"
    cfg.write_text('boltz_container: "/from/config.img"\n')
    assert _resolve(cfg, "/from/cli.img") == "/from/cli.img"


@pytest.mark.local_unit
def test_absent_key_and_absent_file_resolve_to_nothing(tmp_path: Path) -> None:
    """An empty answer is not an error. CI has PyYAML and no singularity, so
    READ_CONFIG's host-python branch is the one that has to take over."""
    cfg = tmp_path / "config.yml"
    cfg.write_text("name: nothing_here\n")
    assert _resolve(cfg) == ""
    assert _resolve(tmp_path / "missing.yml") == ""


@pytest.mark.local_unit
@pytest.mark.parametrize("line", [
    'boltz_container: "/q/double.img"',
    "boltz_container: '/q/single.img'",
    "boltz_container: /q/bare.img",
    "boltz_container:    /q/spaced.img   ",
    "boltz_container: /q/commented.img   # the image the pipeline bundles",
    'boltz_container: "/q/commented2.img"  # quoted, with a comment after',
    'boltz_container: "/q/hash#in#path.img"',
])
def test_quoting_and_spacing_are_stripped(tmp_path: Path, line: str) -> None:
    cfg = tmp_path / "config.yml"
    cfg.write_text(f"name: x\n{line}\n")
    expected = yaml.safe_load(cfg.read_text())["boltz_container"]
    assert _resolve(cfg) == expected
