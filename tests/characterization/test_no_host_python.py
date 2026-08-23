"""No stage may run Python on the host.

The HPC is airgapped and nothing is installed on it. Every version the science
depends on is the one built into the Boltz image, so a stage that calls the
host's `python3` is running against whatever that machine happens to carry.
scripts/negative_steering.slurm.sh, the runner that produced the committed
benchmark, honours this. It uses only bash, grep and singularity on the host.

The Nextflow rewrite did not. Five call sites ran Python directly, and the
first cluster run died on the first of them, because the driver needs PyYAML and
the host has stdlib only. The other four would have failed later, or worse,
succeeded against a different interpreter than the one the run was built on.

One exception is allowed and it is gated. READ_CONFIG has no stub, so under
-stub-run it executes the real driver, and CI has no image to execute it inside.
That path needs --allow_host_python, which is never set on the cluster.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STAGES_NF = REPO_ROOT / "modules" / "negsteer_stages.nf"

HOST_PYTHON_EXCEPTION = "NEGSTEER_READ_CONFIG"


def _blocks() -> dict[str, dict[str, str]]:
    """Each process split into its script: and stub: halves."""
    text = STAGES_NF.read_text()
    out: dict[str, dict[str, str]] = {}
    for chunk in re.split(r"^process\s+", text, flags=re.M)[1:]:
        name = chunk.split("{", 1)[0].strip()
        body = chunk.split("{", 1)[1]
        script, _, stub = body.partition("\n    stub:")
        script = script.partition("script:")[2]
        out[name] = {"script": script, "stub": stub}
    return out


def _logical_lines(block: str) -> list[str]:
    """Join Nextflow's `\\\\` continuations, so a wrapped singularity call is
    one command rather than two lines that look unrelated."""
    joined = re.sub(r"\\\\\s*\n\s*", " ", block)
    return [ln.strip() for ln in joined.splitlines() if ln.strip()]


def _bare_python_calls(block: str) -> list[str]:
    calls = []
    for line in _logical_lines(block):
        if re.search(r"(?<![\w/])python3?\s", line) and "singularity exec" not in line:
            calls.append(line)
    return calls


@pytest.mark.local_unit
@pytest.mark.parametrize("process", sorted(_blocks()))
def test_script_blocks_run_python_only_in_the_container(process: str) -> None:
    bare = _bare_python_calls(_blocks()[process]["script"])
    if process == HOST_PYTHON_EXCEPTION:
        pytest.skip("gated exception, asserted by its own test below")
    assert not bare, (
        f"{process} runs Python on the host: {bare}. Wrap it in "
        f"`singularity exec --bind ... ${{container}} python`."
    )


@pytest.mark.local_unit
def test_the_one_host_python_call_is_gated_behind_an_opt_in() -> None:
    script = _blocks()[HOST_PYTHON_EXCEPTION]["script"]
    assert _bare_python_calls(script), "exception no longer needed, delete it"
    assert "allow_host_python" in script
    # Declared in nextflow.config, not main.nf. A params assignment in the main
    # script's body is invisible to the module on the cluster's Nextflow, so the
    # gate would read as null and the branch would be unreachable by accident
    # rather than by design. See tests/test_nextflow_params.py.
    assert "allow_host_python = false" in (REPO_ROOT / "nextflow.config").read_text()


@pytest.mark.local_unit
@pytest.mark.parametrize("process", sorted(_blocks()))
def test_stub_blocks_need_neither_container_nor_python(process: str) -> None:
    """A stub run is a laptop check of the DAG shape. It has no image, so a
    stub that reached for one would fail for a reason unrelated to wiring."""
    stub = _blocks()[process]["stub"]
    assert "singularity" not in stub, f"{process}'s stub needs no container"
    assert not _bare_python_calls(stub), f"{process}'s stub should not need Python"
