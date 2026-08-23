"""Every param the module reads must be declared where Nextflow will find it.

On the Nextflow the cluster runs, 25.10.4, a `params.x = ...` assignment in
main.nf's body is NOT visible to an included module. `params.repo_root` worked
because it is declared in nextflow.config's `params {}` block. `params.max_passing`
was declared in main.nf's body, resolved to nothing, and rendered into the harvest
command as the literal string "null". argparse rejected it, and the run died after
it had already spent 45 GPU-minutes on the predictions.

Three of the four undeclared params happened to be harmless. `boltz_container` was
read as `?: ''`, and `allow_host_python` and `postprocess_populate_all` were
compared against a truthy value, so null behaved like the intended default. That is
luck, not design, and it is why this asserts on all of them rather than the one
that broke.

The live signal is "WARN: Access to undefined parameter" in the run log.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_NF = REPO_ROOT / "main.nf"
CONFIG = REPO_ROOT / "nextflow.config"
STAGES_NF = REPO_ROOT / "modules" / "negsteer_stages.nf"

# Set by Nextflow itself, not by us.
BUILT_IN = {"outdir", "workDir"}


def _declared_in_config() -> set[str]:
    text = CONFIG.read_text()
    start = text.index("\nparams {")
    depth, i = 0, start + 1
    while True:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    body = text[start:i]
    return set(re.findall(r"^\s{4}(\w+)\s*=", body, flags=re.M))


def _strip_comments(text: str) -> str:
    """Prose about params is not a reference to one."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//.*$", "", text, flags=re.M)


def _referenced(path: Path) -> set[str]:
    return set(re.findall(r"params\.(\w+)", _strip_comments(path.read_text()))) - BUILT_IN


@pytest.mark.local_unit
def test_main_nf_assigns_no_params_in_its_body() -> None:
    """An assignment here is silently invisible to the module."""
    offenders = re.findall(r"^\s*params\.(\w+)\s*=", MAIN_NF.read_text(), flags=re.M)
    assert not offenders, (
        f"main.nf assigns {offenders} in its body. Declare them in "
        "nextflow.config's `params {{}}` block, or the module reads them as null."
    )


@pytest.mark.local_unit
@pytest.mark.parametrize("source", [STAGES_NF, MAIN_NF], ids=lambda p: p.name)
def test_every_referenced_param_is_declared(source: Path) -> None:
    missing = sorted(_referenced(source) - _declared_in_config())
    assert not missing, (
        f"{source.name} reads {missing}, which nextflow.config does not declare. "
        "They will render as the string 'null' inside a task script."
    )
