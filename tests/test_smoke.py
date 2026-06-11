"""Standalone smoke checks for the negative-steering engine.

The real engine needs a GPU + Boltz container, so an end-to-end run is an
HPC-tier test (see the placeholder at the bottom). What we *can* check on a
laptop with no heavy dependencies installed:

  - every vendored Python file is syntactically valid (byte-compiles);
  - the shell orchestrator and Nextflow module are present and non-empty;
  - the documented entry-point scripts exist in the vendored engine.
"""

from __future__ import annotations

import py_compile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_DIR = REPO_ROOT / "engine"
BIN_DIR = ENGINE_DIR / "bin"

ENTRY_POINTS = [
    "bin/negative_steering_run_one.sh",
    "bin/boltz2_negative_steering.py",
    "bin/boltz2_iterate_steering.py",
    "modules/negative_steering.nf",
]


@pytest.mark.local_unit
@pytest.mark.parametrize("py", sorted(BIN_DIR.glob("*.py")), ids=lambda p: p.name)
def test_engine_python_compiles(py: Path) -> None:
    """Vendored Python is at least syntactically valid (no run-time deps needed)."""
    try:
        py_compile.compile(str(py), doraise=True)
    except py_compile.PyCompileError as exc:
        pytest.fail(f"{py.name} failed to compile:\n{exc}")


@pytest.mark.local_unit
@pytest.mark.parametrize("rel", ENTRY_POINTS)
def test_entry_points_present(rel: str) -> None:
    path = ENGINE_DIR / rel
    assert path.exists(), f"engine entry point missing: {rel}"
    assert path.stat().st_size > 0, f"engine entry point is empty: {rel}"


@pytest.mark.hpc
@pytest.mark.skip(reason="end-to-end negative-steering run requires GPU + Boltz container; "
                         "run on the HPC against an experiments/ input")
def test_end_to_end_negative_steering() -> None:
    """Placeholder for a real standalone run on the cluster.

    Wire this to a small fixture input under experiments/ and assert the
    expected cross_sequence_summary.csv is produced.
    """
