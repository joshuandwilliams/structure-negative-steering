"""Structural checks that need no dependencies and no run.

Three things worth catching before anything heavier runs. Every engine
Python file byte-compiles, the entry points the pipeline invokes by path
still exist, and the README flowchart still describes the workflow it
claims to. A real end-to-end run is an HPC-tier test, in
characterization/test_engine_smoke_run.py.
"""

from __future__ import annotations

import py_compile
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = REPO_ROOT / "bin"
STAGES_NF = REPO_ROOT / "modules" / "negsteer_stages.nf"
FLOWCHART = REPO_ROOT / "docs" / "pipeline_flowchart.svg"

# Paths hard-coded in modules/negsteer_stages.nf. Renaming one without
# updating the module leaves a pipeline that fails only at that stage.
ENTRY_POINTS = [
    "main.nf",
    "modules/negsteer_stages.nf",
    "bin/boltz2_negative_steering.py",
    "bin/boltz2_iterate_steering.py",
    "bin/cross_summary_cli.py",
    "bin/extract_passing.py",
    "bin/sum_stage_times.py",
    "scripts/negative_steering.py",
]


@pytest.mark.local_unit
@pytest.mark.parametrize("py", sorted(BIN_DIR.glob("*.py")), ids=lambda p: p.name)
def test_engine_python_compiles(py: Path) -> None:
    """Engine Python is at least syntactically valid, with no deps installed."""
    try:
        py_compile.compile(str(py), doraise=True)
    except py_compile.PyCompileError as exc:
        pytest.fail(f"{py.name} failed to compile:\n{exc}")


@pytest.mark.local_unit
@pytest.mark.parametrize("rel", ENTRY_POINTS)
def test_entry_points_present(rel: str) -> None:
    path = REPO_ROOT / rel
    assert path.exists(), f"entry point missing: {rel}"
    assert path.stat().st_size > 0, f"entry point is empty: {rel}"


@pytest.mark.local_unit
def test_every_process_the_module_defines_is_wired_into_main() -> None:
    """A process defined but never included is dead code, which is how the
    superseded module survived the Nextflow rewrite."""
    defined = set(re.findall(r"^process (NEGSTEER_\w+)", STAGES_NF.read_text(), re.M))
    included = set(re.findall(r"(NEGSTEER_\w+);", (REPO_ROOT / "main.nf").read_text()))
    assert defined == included, (
        f"defined but not included: {sorted(defined - included)}; "
        f"included but not defined: {sorted(included - defined)}"
    )


# ── the README flowchart ─────────────────────────────────────────────────────
# A figure that drifts from the code is worse than no figure, because it is
# still believed. These pin the two things a reader takes from it.

def _flowchart_process_names() -> set[str]:
    svg = FLOWCHART.read_text()
    labels = re.findall(r">([A-Z][A-Z_]{3,})(?:\s*&#215;\s*[NM])?</text>", svg)
    return set(labels)


def _labels_by_process() -> dict[str, str]:
    text = STAGES_NF.read_text()
    out = {}
    for block in re.split(r"^process ", text, flags=re.M)[1:]:
        name = block.split("{", 1)[0].strip().removeprefix("NEGSTEER_")
        label = re.search(r"label '(\w+)'", block)
        out[name] = label.group(1) if label else ""
    return out


@pytest.mark.local_unit
def test_flowchart_shows_exactly_the_processes_that_exist() -> None:
    assert _flowchart_process_names() == set(_labels_by_process())


@pytest.mark.local_unit
def test_flowchart_colours_the_gpu_stages_as_gpu_stages() -> None:
    """Blue means a Boltz-2 call, which is where the cost is. Mislabelling
    a CPU stage blue misrepresents what the run spends its GPU hours on."""
    svg = FLOWCHART.read_text()
    gpu_fill, cpu_fill = "#e3f0fd", "#eaf5ea"

    for name, label in _labels_by_process().items():
        # The box fill is the nearest preceding rect for that process' text.
        idx = svg.index(f">{name}")
        preceding = svg[:idx]
        fill = re.findall(r'fill="(#e3f0fd|#eaf5ea)"', preceding)[-1]
        if label == "gpu":
            assert fill == gpu_fill, f"{name} is a GPU stage but drawn as CPU"
        else:
            assert fill == cpu_fill, f"{name} is a {label} stage but drawn as GPU"
