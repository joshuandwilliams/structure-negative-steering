"""Every bin/ module puts its own directory on sys.path before importing a sibling.

Nextflow invokes these scripts by absolute path inside the container, with no
PYTHONPATH set. A module that imported a sibling without first repairing
sys.path would fail at import, and the stage would die before doing any work.
So each one opens with

    _SCRIPT_DIR = Path(__file__).resolve().parent
    if str(_SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPT_DIR))

The guard is false in an ordinary test run, because importing the module put
bin/ on sys.path already. These lines were therefore marked `pragma: no cover
- import order`, which was not true. Removing bin/ from sys.path and
re-executing the module reaches them, and that is what this file does.

The assertion is the point rather than the coverage. Each module has to leave
bin/ on sys.path and still expose its public names, because the import of its
siblings happens on the next line and depends on the repair having worked.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
BIN = _REPO_ROOT / "bin"

# One name per module that must survive the re-execution. A module whose
# sibling import failed would raise, so reaching the assertion at all is most
# of the test. The name proves the module finished executing.
MODULES = [
    ("cross_summary_cli", "main"),
    ("design_cohort", "DesignCohort"),
    ("designed_backbone", "DesignedBackbone"),
    ("designed_sequence", "DesignedSequence"),
    ("negative_steering_run", "NegativeSteeringRun"),
    ("negsteer_coldstart", "main"),
    ("pathway_labels", "read_mutations_tsv_full"),
    ("position_set", "PositionSet"),
    ("protein_structure_prediction", "ProteinStructurePrediction"),
    ("stage_result", "StageResult"),
]


@pytest.mark.logic
@pytest.mark.parametrize("name,public", MODULES, ids=[m for m, _ in MODULES])
def test_the_module_repairs_sys_path_and_still_imports(name, public,
                                                        monkeypatch):
    """With bin/ absent from sys.path, the module has to put it back.

    The module is re-executed under its own name, registered in sys.modules
    before the exec so dataclasses can resolve the classes it defines. The
    monkeypatch restores sys.path and sys.modules afterwards, so no later
    test sees the altered interpreter.
    """
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(BIN)])
    assert str(BIN) not in sys.path

    spec = importlib.util.spec_from_file_location(name, BIN / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)

    assert str(BIN) in sys.path, f"{name} did not put bin/ back on sys.path"
    assert hasattr(module, public), f"{name} has no {public}"


@pytest.mark.logic
def test_a_module_already_on_the_path_does_not_duplicate_it(monkeypatch):
    """The guard is there so a repeated import cannot grow sys.path without
    bound. A stage that imports several of these modules in one process would
    otherwise add bin/ once per module."""
    monkeypatch.setattr(sys, "path", [str(BIN)] + list(sys.path))
    before = sys.path.count(str(BIN))

    spec = importlib.util.spec_from_file_location(
        "position_set", BIN / "position_set.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "position_set", module)
    spec.loader.exec_module(module)

    assert sys.path.count(str(BIN)) == before
