"""bin/negative_steering_run_one.sh, driven against a stubbed singularity.

This script is the whole per-sequence pipeline and it is on the critical path of
every run, but coverage measures bin/*.py only, so none of it was exercised. A
flag-forwarding or loop-bound defect here fails on a GPU node minutes into a job
rather than in CI.

The technique is the one that catches those: put a fake `singularity` on PATH
that records its argv and fabricates the files the real one would have written.
No GPU, no container, no cluster. Each test asserts on what the script actually
invoked, so argument forwarding, the two loop bounds and the failure policy are
all pinned.

Requires bash >= 4.4. The script expands empty arrays under `set -u`, which is
an error in bash 3.2, still the system bash on macOS. environment.yml pins
bash 5.2.37 so the conda environment supplies it; CI runs bash 5 already.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "bin" / "negative_steering_run_one.sh"


def _modern_bash() -> str | None:
    """The active conda environment first, then PATH.

    environment.yml pins bash 5.2.37 precisely so this is self-contained, so
    look there before anything system-wide. On Linux and CI there is no conda
    prefix to find and PATH already supplies bash 5.
    """
    for cand in (str(Path(sys.prefix) / "bin" / "bash"), shutil.which("bash")):
        if not cand or not Path(cand).exists():
            continue
        out = subprocess.run([cand, "-c", "echo $BASH_VERSINFO"],
                             capture_output=True, text=True).stdout.strip()
        if out.isdigit() and int(out) >= 4:
            return cand
    return None


BASH = _modern_bash()
needs_bash = pytest.mark.skipif(
    BASH is None, reason="needs bash >= 4.4; environment.yml pins it, so create/update that env")

# Records every call as one tab-separated line, and fabricates whatever the real
# singularity would have produced. Control it through the STUB_* environment.
STUB = r"""#!/usr/bin/env bash
printf '%s\t' "$@" >> "$STUB_LOG"
printf '\n'       >> "$STUB_LOG"

# The engine sub-command is the token after the .py path.
sub=""
for ((i=1; i<=$#; i++)); do
  case "${!i}" in *.py) j=$((i+1)); sub="${!j}"; break ;; esac
done

workdir=""
for ((i=1; i<=$#; i++)); do
  if [[ "${!i}" == "--workdir" ]]; then j=$((i+1)); workdir="${!j}"; fi
done

if [[ -n "${STUB_FAIL_SUB:-}" && "$sub" == "$STUB_FAIL_SUB" ]]; then
  exit "${STUB_FAIL_RC:-1}"
fi

case "$sub" in
  plan)            [[ -n "${STUB_PLAN:-}"      ]] && cp "$STUB_PLAN"      "$workdir/plan.json" ;;
  plan-reversions) [[ -n "${STUB_REVERSIONS:-}" ]] && cp "$STUB_REVERSIONS" "$workdir/reversion_plan.json" ;;
esac
exit 0
"""


@pytest.fixture
def rig(tmp_path):
    """A runnable environment: fake inputs, fake bin/, stubbed singularity."""
    for name in ("gt.pdb", "receptor.fasta", "effector.fasta",
                 "true_interface.txt", "design_region.txt"):
        (tmp_path / name).write_text("x\n")

    bindir = tmp_path / "bin"
    bindir.mkdir()
    for py in ("boltz2_negative_steering.py", "negsteer_coldstart.py",
               "negsteer_aggregate.py", "reversion.py", "extract_passing.py"):
        (bindir / py).write_text("")
    container = tmp_path / "boltz.sif"
    container.write_text("")

    stubdir = tmp_path / "stubs"
    stubdir.mkdir()
    sing = stubdir / "singularity"
    sing.write_text(STUB)
    sing.chmod(0o755)

    log = tmp_path / "calls.log"
    log.write_text("")

    class Rig:
        root = tmp_path
        workdir = tmp_path / "work"
        calls_file = log

        def plan(self, *, skip_steering=False, n_designs=0):
            p = tmp_path / "plan_src.json"
            p.write_text(json.dumps({"skip_steering": skip_steering,
                                     "designs": [{} for _ in range(n_designs)]}))
            return p

        def reversions(self, n):
            p = tmp_path / "rev_src.json"
            p.write_text(json.dumps({"entries": [{} for _ in range(n)]}))
            return p

        def run(self, *extra, env=None, expect=None):
            cmd = [BASH, str(SCRIPT),
                   "--seq-name", "SEQ1",
                   "--ground-truth", str(tmp_path / "gt.pdb"),
                   "--receptor-chain", "A", "--effector-chain", "B",
                   "--receptor-fasta", str(tmp_path / "receptor.fasta"),
                   "--effector-fasta", str(tmp_path / "effector.fasta"),
                   "--true-interface-indices-file", str(tmp_path / "true_interface.txt"),
                   "--design-region-indices-file", str(tmp_path / "design_region.txt"),
                   "--workdir", str(self.workdir),
                   "--bin-dir", str(bindir),
                   "--boltz-container", str(container),
                   *extra]
            e = {**os.environ, "STUB_LOG": str(log), "PATH": f"{stubdir}:{os.environ['PATH']}"}
            e.update(env or {})
            r = subprocess.run(cmd, capture_output=True, text=True, env=e)
            if expect is not None:
                assert r.returncode == expect, (
                    f"rc={r.returncode}\n--- stdout\n{r.stdout}\n--- stderr\n{r.stderr}")
            return r

        @property
        def calls(self) -> list[list[str]]:
            return [ln.split("\t")[:-1]
                    for ln in log.read_text().splitlines() if ln.strip()]

        def sub_calls(self, sub: str) -> list[list[str]]:
            """Calls whose engine sub-command is `sub`."""
            out = []
            for c in self.calls:
                for i, tok in enumerate(c):
                    if tok.endswith(".py"):
                        if i + 1 < len(c) and c[i + 1] == sub:
                            out.append(c)
                        break
            return out

    return Rig()


# --------------------------------------------------------------------------
# Argument handling
# --------------------------------------------------------------------------

@pytest.mark.local_unit
@needs_bash
def test_unknown_argument_is_rejected(rig) -> None:
    r = rig.run("--not-a-flag", "x", expect=2)
    assert "unknown arg: --not-a-flag" in r.stderr


@pytest.mark.local_unit
@needs_bash
def test_missing_required_arguments_are_all_named(rig) -> None:
    r = subprocess.run([BASH, str(SCRIPT), "--seq-name", "SEQ1"],
                       capture_output=True, text=True)
    assert r.returncode == 2
    assert "missing required args" in r.stderr
    # Naming every missing flag at once beats one round trip per flag.
    for flag in ("--ground-truth", "--workdir", "--bin-dir", "--boltz-container"):
        assert flag in r.stderr


@pytest.mark.local_unit
@needs_bash
def test_a_missing_input_file_is_caught_before_any_gpu_work(rig) -> None:
    (rig.root / "gt.pdb").unlink()
    r = rig.run(expect=2)
    assert "input file not found" in r.stderr
    assert rig.calls == [], "must not reach the container"


@pytest.mark.local_unit
@needs_bash
def test_a_missing_engine_script_is_caught(rig) -> None:
    (rig.root / "bin" / "extract_passing.py").unlink()
    r = rig.run(expect=2)
    assert "required Python script not found" in r.stderr
    assert rig.calls == []


@pytest.mark.local_unit
@needs_bash
def test_a_missing_container_is_caught(rig) -> None:
    (rig.root / "boltz.sif").unlink()
    r = rig.run(expect=2)
    assert "Boltz container not found" in r.stderr


# --------------------------------------------------------------------------
# What gets invoked
# --------------------------------------------------------------------------

@pytest.mark.local_unit
@needs_bash
def test_gpu_stages_request_nv_and_cpu_stages_do_not(rig) -> None:
    """--nv on a CPU stage would take a GPU for work that cannot use one."""
    rig.run(env={"STUB_PLAN": str(rig.plan(n_designs=1))}, expect=0)
    for c in rig.sub_calls("plan") + rig.sub_calls("predict-one"):
        assert "--nv" in c
    for sub in ("collect", "aggregate", "harvest-reversions", "kickoff-finalize"):
        for c in rig.sub_calls(sub):
            assert "--nv" not in c, f"{sub} asked for a GPU"


@pytest.mark.local_unit
@needs_bash
def test_bind_paths_are_deduplicated(rig) -> None:
    """Every input here shares one parent. Binding it repeatedly is a bug."""
    rig.run(env={"STUB_PLAN": str(rig.plan())}, expect=0)
    binds = [c[i + 1] for c in rig.calls[:1] for i, t in enumerate(c) if t == "--bind"]
    assert len(binds) == len(set(binds)), f"duplicate binds: {binds}"


@pytest.mark.local_unit
@needs_bash
def test_plan_extra_args_are_word_split_and_forwarded(rig) -> None:
    rig.run("--plan-extra-args", "--n-designs 7 --mode mild",
            env={"STUB_PLAN": str(rig.plan())}, expect=0)
    plan = rig.sub_calls("plan")[0]
    assert plan[-4:] == ["--n-designs", "7", "--mode", "mild"]


@pytest.mark.local_unit
@needs_bash
def test_postprocess_thresholds_are_forwarded(rig) -> None:
    rig.run("--postprocess-rmsd-threshold", "3.5",
            "--postprocess-metric-column", "custom_col",
            "--postprocess-contact-cutoff", "4.0",
            env={"STUB_PLAN": str(rig.plan())}, expect=0)
    c = rig.sub_calls("compute-final-metrics")[0]
    assert c[c.index("--rmsd-threshold") + 1] == "3.5"
    assert c[c.index("--metric-column") + 1] == "custom_col"
    assert c[c.index("--contact-cutoff") + 1] == "4.0"
    assert "--populate-all" in c


@pytest.mark.local_unit
@needs_bash
def test_populate_all_can_be_turned_off(rig) -> None:
    rig.run("--no-postprocess-populate-all",
            env={"STUB_PLAN": str(rig.plan())}, expect=0)
    assert "--populate-all" not in rig.sub_calls("compute-final-metrics")[0]


# --------------------------------------------------------------------------
# Loop bounds
# --------------------------------------------------------------------------

@pytest.mark.local_unit
@needs_bash
def test_predict_one_runs_once_per_design_with_ascending_indices(rig) -> None:
    rig.run(env={"STUB_PLAN": str(rig.plan(n_designs=4))}, expect=0)
    calls = rig.sub_calls("predict-one")
    assert [c[c.index("--index") + 1] for c in calls] == ["0", "1", "2", "3"]


@pytest.mark.local_unit
@needs_bash
def test_skip_steering_runs_no_predictions(rig) -> None:
    """The cold start already passed. Predicting anything would waste the GPU."""
    r = rig.run(env={"STUB_PLAN": str(rig.plan(skip_steering=True, n_designs=5))},
                expect=0)
    assert rig.sub_calls("predict-one") == []
    assert "skip_steering=true" in r.stdout
    # The rest of the pipeline still has to run, or there is no summary.
    assert rig.sub_calls("aggregate") and rig.sub_calls("extract_passing") == []


@pytest.mark.local_unit
@needs_bash
def test_zero_designs_predicts_nothing_but_still_finishes(rig) -> None:
    r = rig.run(env={"STUB_PLAN": str(rig.plan(n_designs=0))}, expect=0)
    assert rig.sub_calls("predict-one") == []
    assert "n_designs=0" in r.stdout


@pytest.mark.local_unit
@needs_bash
def test_reversions_run_once_per_staged_entry(rig) -> None:
    rig.run(env={"STUB_PLAN": str(rig.plan(n_designs=1)),
                 "STUB_REVERSIONS": str(rig.reversions(3))}, expect=0)
    calls = rig.sub_calls("predict-reversion-one")
    assert [c[c.index("--index") + 1] for c in calls] == ["0", "1", "2"]


@pytest.mark.local_unit
@needs_bash
def test_no_reversion_plan_means_no_reversion_predictions(rig) -> None:
    r = rig.run(env={"STUB_PLAN": str(rig.plan(n_designs=1))}, expect=0)
    assert rig.sub_calls("predict-reversion-one") == []
    assert "no reversions staged" in r.stdout


# --------------------------------------------------------------------------
# Failure policy
# --------------------------------------------------------------------------

@pytest.mark.local_unit
@needs_bash
def test_a_failing_plan_without_skip_steering_is_fatal(rig) -> None:
    r = rig.run(env={"STUB_FAIL_SUB": "plan", "STUB_FAIL_RC": "1"}, expect=1)
    assert "plan stage exited 1" in r.stderr


@pytest.mark.local_unit
@needs_bash
def test_a_failing_plan_with_skip_steering_is_tolerated(rig) -> None:
    """plan exits non-zero on a skipped cold start, having written plan.json."""
    src = rig.plan(skip_steering=True)
    cycle0 = rig.workdir / "cycle_0"
    cycle0.mkdir(parents=True)
    (cycle0 / "plan.json").write_text(src.read_text())
    r = rig.run(env={"STUB_FAIL_SUB": "plan", "STUB_FAIL_RC": "3"}, expect=0)
    assert "skip_steering=true — continuing" in r.stdout


@pytest.mark.local_unit
@needs_bash
def test_a_plan_that_writes_nothing_is_fatal(rig) -> None:
    r = rig.run(expect=1)
    assert "did not produce" in r.stderr


@pytest.mark.local_unit
@needs_bash
def test_one_failing_design_does_not_kill_the_run(rig) -> None:
    """rc=1 is a per-design failure. Losing 19 good designs to it would be worse."""
    r = rig.run(env={"STUB_PLAN": str(rig.plan(n_designs=3)),
                     "STUB_FAIL_SUB": "predict-one", "STUB_FAIL_RC": "1"}, expect=0)
    assert len(rig.sub_calls("predict-one")) == 3
    assert "0 ok, 3 failed" in r.stdout


@pytest.mark.local_unit
@needs_bash
def test_an_invalid_index_is_fatal(rig) -> None:
    """rc>=2 means the loop bound disagrees with plan.json. Never tolerate that."""
    r = rig.run(env={"STUB_PLAN": str(rig.plan(n_designs=3)),
                     "STUB_FAIL_SUB": "predict-one", "STUB_FAIL_RC": "2"}, expect=1)
    assert "exited 2 (fatal)" in r.stderr
    assert len(rig.sub_calls("predict-one")) == 1, "must stop at the first fatal"


@pytest.mark.local_unit
@needs_bash
def test_a_failing_cpu_stage_is_fatal(rig) -> None:
    r = rig.run(env={"STUB_PLAN": str(rig.plan()), "STUB_FAIL_SUB": "collect"}, expect=1)
    assert "collect stage failed" in r.stderr


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

@pytest.mark.local_unit
@needs_bash
def test_the_runtime_stamp_is_written(rig) -> None:
    """extract_passing reads this into passing_summary.csv, and thence into the
    cross-sequence summary. A missing file loses per-sequence timing."""
    rig.run(env={"STUB_PLAN": str(rig.plan())}, expect=0)
    stamp = rig.workdir / "run_one_runtime_sec.txt"
    assert stamp.is_file()
    assert stamp.read_text().strip().isdigit()


@pytest.mark.local_unit
@needs_bash
def test_the_stages_run_in_order(rig) -> None:
    """Each stage consumes what the previous one wrote, so order is correctness."""
    expected = [
        "plan", "predict-one", "collect", "kickoff-distances", "kickoff-prefilter",
        "build-contaminated", "plan-reversions", "harvest-reversions",
        "kickoff-finalize", "aggregate", "compute-final-metrics",
        "aggregate-per-sequence"]
    rig.run(env={"STUB_PLAN": str(rig.plan(n_designs=1))}, expect=0)
    seen: list[str] = []
    for c in rig.calls:
        for i, tok in enumerate(c):
            if tok.endswith(".py") and i + 1 < len(c):
                if c[i + 1] in expected and c[i + 1] not in seen:
                    seen.append(c[i + 1])
                break
    assert seen == expected, f"stage order drifted\n  got: {seen}"
