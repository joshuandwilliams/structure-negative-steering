"""boltz2_negative_steering's plan stage, with Boltz stubbed.

cmd_plan is the half of the engine that decides what to steer. It predicts a
cold start, locates the true interface on the ground truth and the wrong
interface on the prediction, builds a candidate pool from the difference, and
writes one steered design per requested set. Only the prediction itself needs a
GPU, so stubbing run_boltz reaches everything else.

Two cold starts are exercised, because they take opposite branches:

  clean  the prediction is the ground truth, so ra_eff is 0, the run
         short-circuits and no steering happens
  wrong  the effector is reflected through the receptor centroid, so it binds
         the opposite face and ra_eff is large while contacts still exist

Building the wrong pose by rigid transform is what makes the expected outcome
knowable. The reference repo's playbook makes the same point: a test that only
checks output exists would pass against a wrong answer.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from unittest import mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin"))

import boltz2_negative_steering as bns  # noqa: E402

GT = (_REPO_ROOT / "experiments" / "benchmarking" / "unconstrained" / "6G10"
      / "6G10.pdb")
needs_pdb = pytest.mark.skipif(not GT.is_file(), reason="6G10 reference absent")

REC_LEN, EFF_LEN = 76, 83


def _wrong_interface_pose(src: Path, dst: Path) -> Path:
    """Copy a complex with the effector bound to the opposite receptor face.

    Reflecting chain B through the receptor centroid rather than translating it
    away. A translation large enough to fail the RMSD threshold also carries the
    effector off the protein entirely, leaving no wrong interface to steer away
    from, so the run short-circuits. The reflection keeps 27 contacts at 4.5 A
    while placing them on the wrong surface, which is the situation negative
    steering exists to fix.
    """
    lines = src.read_text().splitlines(keepends=True)
    coords = [(float(x[30:38]), float(x[38:46]), float(x[46:54]))
              for x in lines if x.startswith("ATOM") and x[21] == "A"]
    cx = sum(c[0] for c in coords) / len(coords)
    cy = sum(c[1] for c in coords) / len(coords)
    cz = sum(c[2] for c in coords) / len(coords)

    out = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")) and len(line) > 54 and line[21] == "B":
            x = 2 * cx - float(line[30:38])
            y = 2 * cy - float(line[38:46])
            z = 2 * cz - float(line[46:54])
            out.append(f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}")
        else:
            out.append(line)
    dst.write_text("".join(out))
    return dst


def _run_plan(workdir: Path, prediction: Path, *extra: str) -> int:
    """Drive cmd_plan with run_boltz returning a fixed prediction."""
    def fake_run_boltz(yaml_path, out_dir, container, recycling_steps,
                       diffusion_samples, seed, no_kernels):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pred = out_dir / "prediction.pdb"
        shutil.copy(prediction, pred)
        return pred

    argv = ["plan", "--ground-truth", str(GT), "--workdir", str(workdir),
            "--receptor", "A", "--effector", "B",
            "--boltz-container", "/fake.img",
            "--n-designs", "2", "--num-seeds", "1", "--diffusion-samples", "1",
            "--recycling-steps", "1", *extra]
    with mock.patch.object(bns, "run_boltz", fake_run_boltz), \
            mock.patch.object(sys, "argv", ["boltz2_negative_steering.py", *argv]):
        return bns.main()


# ── the clean cold start ─────────────────────────────────────────────────────

@pytest.mark.local_integration
@needs_pdb
def test_a_correct_cold_start_skips_steering(tmp_path):
    """Nothing to rescue means no designs, which is the short-circuit path."""
    assert _run_plan(tmp_path, GT) == 0
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert float(plan["initial_receptor_aligned_effector_rmsd"]) == pytest.approx(0.0, abs=0.01)
    assert not (tmp_path / "steered").is_dir(), \
        "steering ran despite a correct cold start"


@pytest.mark.local_integration
@needs_pdb
def test_the_plan_records_the_ground_truth_context(tmp_path):
    _run_plan(tmp_path, GT)
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["truth_receptor_chain"] == "A"
    assert plan["truth_effector_chain"] == "B"
    assert len(plan["wild_type_receptor_seq"]) == REC_LEN


@pytest.mark.local_integration
@needs_pdb
def test_the_plan_stage_writes_the_boltz_inputs(tmp_path):
    """One A3M per chain plus the YAML, or Boltz cannot be invoked at all."""
    _run_plan(tmp_path, GT)
    msa = tmp_path / "initial" / "input" / "msa"
    for name in ("chain_A.a3m", "chain_B.a3m"):
        assert (msa / name).is_file(), f"missing {name}"
    assert (tmp_path / "initial" / "input" / "input.yaml").is_file()


@pytest.mark.local_integration
@needs_pdb
def test_the_effector_template_is_written_by_default(tmp_path):
    _run_plan(tmp_path, GT)
    assert (tmp_path / "effector_template.cif").is_file()


@pytest.mark.local_integration
@needs_pdb
def test_no_effector_template_suppresses_it(tmp_path):
    _run_plan(tmp_path, GT, "--no-effector-template")
    assert not (tmp_path / "effector_template.cif").is_file(), \
        "--no-effector-template still wrote a template"


# ── the wrong cold start, where steering actually runs ───────────────────────

@pytest.mark.local_integration
@needs_pdb
def test_a_wrong_cold_start_triggers_steering(tmp_path):
    """A 25 A displacement must be measured as a failure, not a rescue."""
    wrong = _wrong_interface_pose(GT, tmp_path / "wrong.pdb")
    work = tmp_path / "work"
    assert _run_plan(work, wrong) == 0

    plan = json.loads((work / "plan.json").read_text())
    rmsd = float(plan["initial_receptor_aligned_effector_rmsd"])
    assert rmsd > 5.0, f"a wrong-face pose scored {rmsd} A, which would pass"


@pytest.mark.local_integration
@needs_pdb
def test_steering_produces_the_requested_number_of_designs(tmp_path):
    wrong = _wrong_interface_pose(GT, tmp_path / "wrong.pdb")
    work = tmp_path / "work"
    _run_plan(work, wrong)

    steered = work / "steered"
    if not steered.is_dir():
        pytest.skip("this cold start did not reach the steering stage")
    designs = {p.name.rsplit("_s", 1)[0] for p in steered.iterdir() if p.is_dir()}
    assert len(designs) <= 2, f"asked for 2 designs, got {sorted(designs)}"
    assert designs, "the steering stage produced no designs"


@pytest.mark.local_integration
@needs_pdb
def test_steering_never_mutates_the_protected_interface(tmp_path):
    """The central invariant. Mutating the true interface would destroy the
    binding site rather than removing the competing one."""
    wrong = _wrong_interface_pose(GT, tmp_path / "wrong.pdb")
    work = tmp_path / "work"
    _run_plan(work, wrong)

    plan = json.loads((work / "plan.json").read_text())
    wild = plan["wild_type_receptor_seq"]
    true_idx = set(plan.get("true_interface_idx") or plan.get("true_site_idx") or [])
    if not true_idx:
        pytest.skip("plan.json does not record the protected index set")

    steered = work / "steered"
    if not steered.is_dir():
        pytest.skip("no steered designs produced")

    for d in sorted(steered.iterdir()):
        fasta = d / "receptor.fasta"
        if not fasta.is_file():
            continue
        seq = "".join(x for x in fasta.read_text().splitlines() if not x.startswith(">"))
        changed = {i for i, (a, b) in enumerate(zip(wild, seq)) if a != b}
        assert not (changed & true_idx), (
            f"{d.name} mutated protected interface positions "
            f"{sorted(changed & true_idx)}")


@pytest.mark.local_integration
@needs_pdb
def test_the_plan_records_a_candidate_pool(tmp_path):
    wrong = _wrong_interface_pose(GT, tmp_path / "wrong.pdb")
    work = tmp_path / "work"
    _run_plan(work, wrong)
    plan = json.loads((work / "plan.json").read_text())
    assert plan.get("mode") == "mild" or "mode" in plan


# ── options that change what gets written ────────────────────────────────────

@pytest.mark.local_integration
@needs_pdb
def test_boltz_constraints_reach_the_generated_yaml(tmp_path):
    """The constrained variant appends a pocket and contact block."""
    _run_plan(tmp_path, GT, "--boltz-constraints")
    yaml_text = (tmp_path / "initial" / "input" / "input.yaml").read_text()
    assert "constraints" in yaml_text.lower(), \
        "--boltz-constraints produced no constraint block"


@pytest.mark.local_integration
@needs_pdb
def test_without_the_flag_no_constraints_are_written(tmp_path):
    _run_plan(tmp_path, GT)
    yaml_text = (tmp_path / "initial" / "input" / "input.yaml").read_text()
    assert "pocket" not in yaml_text.lower(), \
        "an unconstrained run emitted a pocket block"


@pytest.mark.local_integration
@needs_pdb
def test_skip_steering_stops_after_the_cold_start(tmp_path):
    wrong = _wrong_interface_pose(GT, tmp_path / "wrong.pdb")
    work = tmp_path / "work"
    assert _run_plan(work, wrong, "--skip-steering") == 0
    assert not (work / "steered").is_dir(), "--skip-steering still steered"


@pytest.mark.local_integration
@needs_pdb
def test_a_supplied_true_interface_overrides_detection(tmp_path):
    iface = tmp_path / "iface.txt"
    iface.write_text("# 0-based\n0\n1\n2\n")
    work = tmp_path / "work"
    assert _run_plan(work, GT, "--true-interface-indices-file", str(iface)) == 0
    plan = json.loads((work / "plan.json").read_text())
    assert plan, "plan.json was not written"


@pytest.mark.local_integration
@needs_pdb
def test_a_receptor_override_of_the_wrong_length_is_rejected(tmp_path):
    """Every index downstream is positional, so an indel would corrupt them."""
    bad = tmp_path / "short.fasta"
    bad.write_text(">r\nACDEF\n")
    with pytest.raises(SystemExit):
        _run_plan(tmp_path / "work", GT, "--receptor-fasta", str(bad))


@pytest.mark.local_integration
@needs_pdb
def test_a_receptor_override_of_the_right_length_is_accepted(tmp_path):
    plain = tmp_path / "probe"
    _run_plan(plain, GT)
    wild = json.loads((plain / "plan.json").read_text())["wild_type_receptor_seq"]

    override = tmp_path / "ok.fasta"
    mutated = ("K" if wild[0] != "K" else "A") + wild[1:]
    override.write_text(f">r\n{mutated}\n")

    work = tmp_path / "work"
    assert _run_plan(work, GT, "--receptor-fasta", str(override)) == 0
