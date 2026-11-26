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

GT = (_REPO_ROOT / "data" / "reference_complexes" / "6G10.pdb")
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

@pytest.mark.logic
@needs_pdb
def test_a_correct_cold_start_skips_steering(tmp_path):
    """Nothing to rescue means no designs, which is the short-circuit path."""
    assert _run_plan(tmp_path, GT) == 0
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert float(plan["initial_receptor_aligned_effector_rmsd"]) == pytest.approx(0.0, abs=0.01)
    assert not (tmp_path / "steered").is_dir(), \
        "steering ran despite a correct cold start"


@pytest.mark.logic
@needs_pdb
def test_the_plan_records_the_ground_truth_context(tmp_path):
    _run_plan(tmp_path, GT)
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["truth_receptor_chain"] == "A"
    assert plan["truth_effector_chain"] == "B"
    assert len(plan["wild_type_receptor_seq"]) == REC_LEN


@pytest.mark.logic
@needs_pdb
def test_the_plan_stage_writes_the_boltz_inputs(tmp_path):
    """One A3M per chain plus the YAML, or Boltz cannot be invoked at all."""
    _run_plan(tmp_path, GT)
    msa = tmp_path / "initial" / "input" / "msa"
    for name in ("chain_A.a3m", "chain_B.a3m"):
        assert (msa / name).is_file(), f"missing {name}"
    assert (tmp_path / "initial" / "input" / "input.yaml").is_file()


@pytest.mark.logic
@needs_pdb
def test_no_effector_template_is_written_by_default(tmp_path):
    """The template needs the ground-truth complex, so a novel target cannot
    use it. Off is therefore the only safe default, and every committed
    result was produced without one."""
    _run_plan(tmp_path, GT)
    assert not (tmp_path / "effector_template.cif").is_file(), \
        "the plan stage wrote a template without being asked"


@pytest.mark.logic
@needs_pdb
def test_effector_template_opts_in(tmp_path):
    _run_plan(tmp_path, GT, "--effector-template")
    assert (tmp_path / "effector_template.cif").is_file(), \
        "--effector-template wrote no template"


@pytest.mark.logic
@needs_pdb
def test_no_effector_template_still_suppresses_it(tmp_path):
    """Redundant now, and kept because a caller may still pass it."""
    _run_plan(tmp_path, GT, "--no-effector-template")
    assert not (tmp_path / "effector_template.cif").is_file(), \
        "--no-effector-template still wrote a template"


# ── the wrong cold start, where steering actually runs ───────────────────────

@pytest.mark.logic
@needs_pdb
def test_a_wrong_cold_start_triggers_steering(tmp_path):
    """A 25 A displacement must be measured as a failure, not a rescue."""
    wrong = _wrong_interface_pose(GT, tmp_path / "wrong.pdb")
    work = tmp_path / "work"
    assert _run_plan(work, wrong) == 0

    plan = json.loads((work / "plan.json").read_text())
    rmsd = float(plan["initial_receptor_aligned_effector_rmsd"])
    assert rmsd > 5.0, f"a wrong-face pose scored {rmsd} A, which would pass"


@pytest.mark.logic
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


@pytest.mark.logic
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


@pytest.mark.logic
@needs_pdb
def test_the_plan_records_a_candidate_pool(tmp_path):
    wrong = _wrong_interface_pose(GT, tmp_path / "wrong.pdb")
    work = tmp_path / "work"
    _run_plan(work, wrong)
    plan = json.loads((work / "plan.json").read_text())
    assert plan.get("mode") == "mild" or "mode" in plan


# ── options that change what gets written ────────────────────────────────────

@pytest.mark.logic
@needs_pdb
def test_boltz_constraints_reach_the_generated_yaml(tmp_path):
    """The constrained variant appends a pocket and contact block."""
    _run_plan(tmp_path, GT, "--boltz-constraints")
    yaml_text = (tmp_path / "initial" / "input" / "input.yaml").read_text()
    assert "constraints" in yaml_text.lower(), \
        "--boltz-constraints produced no constraint block"


@pytest.mark.logic
@needs_pdb
def test_without_the_flag_no_constraints_are_written(tmp_path):
    _run_plan(tmp_path, GT)
    yaml_text = (tmp_path / "initial" / "input" / "input.yaml").read_text()
    assert "pocket" not in yaml_text.lower(), \
        "an unconstrained run emitted a pocket block"


@pytest.mark.logic
@needs_pdb
def test_skip_steering_stops_after_the_cold_start(tmp_path):
    wrong = _wrong_interface_pose(GT, tmp_path / "wrong.pdb")
    work = tmp_path / "work"
    assert _run_plan(work, wrong, "--skip-steering") == 0
    assert not (work / "steered").is_dir(), "--skip-steering still steered"


@pytest.mark.logic
@needs_pdb
def test_a_supplied_true_interface_overrides_detection(tmp_path):
    iface = tmp_path / "iface.txt"
    iface.write_text("# 0-based\n0\n1\n2\n")
    work = tmp_path / "work"
    assert _run_plan(work, GT, "--true-interface-indices-file", str(iface)) == 0
    plan = json.loads((work / "plan.json").read_text())
    assert plan, "plan.json was not written"


@pytest.mark.logic
@needs_pdb
def test_a_receptor_override_of_the_wrong_length_is_rejected(tmp_path):
    """Every index downstream is positional, so an indel would corrupt them."""
    bad = tmp_path / "short.fasta"
    bad.write_text(">r\nACDEF\n")
    with pytest.raises(SystemExit):
        _run_plan(tmp_path / "work", GT, "--receptor-fasta", str(bad))


@pytest.mark.logic
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


# ── what the method is and is not blind to ───────────────────────────────────

@pytest.mark.logic
def test_every_protected_set_source_includes_the_true_interface():
    """The scope of the method's claim, pinned to the code that sets it.

    The wrong interface is read off the prediction, so that half needs no
    answer. The protected set is not blind: both sources include the true
    interface, design_region_union being true interface UNION design region
    rather than the design region alone. So the residues the engine may mutate
    are always the predicted contacts minus a truth-derived set.

    The README and the analyses state this. If a source is ever added that
    excludes the true interface, the claim becomes wrong in the other
    direction and the prose needs revisiting, which is what this catches.
    """
    src = (_REPO_ROOT / "bin" / "boltz2_negative_steering.py").read_text()
    start = src.index("protected_source = getattr")
    end = src.index('plan["protected_set_source"]')
    assignments = src[start:end].split("protected_set =")[1:]

    assert assignments, "the protected-set block moved; this test is stale"
    for branch in assignments:
        head = branch.split("\n")[0]
        assert "true_set" in head, (
            f"a protected-set branch omits the true interface: {head.strip()!r}. "
            "If that is deliberate, the blindness claim in README.md and "
            "analysis/02-model-choice needs updating with it.")


# ── the overlapping cold start, where there is nothing to steer away from ────
#
# A third cold start, built by rotating the effector 180 degrees about the axis
# through the receptor centroid and the interface centroid. The effector stays
# on the correct face but in the wrong orientation, so ra_eff is 22.7 A while
# 86% of the wrong-interface residues are also true-interface residues.
#
# This is the pathological regime the plan stage has to recognise. Almost every
# candidate is protected, so steering would have to mutate the binding site to
# have any effect at all. The engine detects it, halves the mutation count,
# and in the limit switches to sampling mode, which reseeds the wild type
# rather than designing anything.

def _overlapping_pose(src: Path, dst: Path, degrees: float = 180.0) -> Path:
    import math

    import numpy as np

    lines = src.read_text().splitlines(keepends=True)

    def xyz(line):
        return np.array([float(line[30:38]), float(line[38:46]),
                         float(line[46:54])])

    rec = np.array([xyz(x) for x in lines
                    if x.startswith("ATOM") and x[21] == "A"])
    eff_lines = [x for x in lines
                 if x.startswith(("ATOM", "HETATM")) and x[21] == "B"]
    eff = np.array([xyz(x) for x in eff_lines])
    near = np.linalg.norm(eff[:, None, :] - rec[None, :, :],
                          axis=2).min(axis=1) < 8.0
    centre = eff[near].mean(axis=0) if near.any() else eff.mean(axis=0)

    axis = centre - rec.mean(axis=0)
    axis = axis / np.linalg.norm(axis)
    theta = math.radians(degrees)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)

    out = []
    for line in lines:
        if (line.startswith(("ATOM", "HETATM")) and len(line) > 54
                and line[21] == "B"):
            p = R @ (xyz(line) - centre) + centre
            out.append(f"{line[:30]}{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}"
                       f"{line[54:]}")
        else:
            out.append(line)
    dst.write_text("".join(out))
    return dst


@pytest.fixture
def overlapping(tmp_path):
    pytest.importorskip("numpy")
    return _overlapping_pose(GT, tmp_path / "overlapping.pdb")


def _plan_of(work: Path) -> dict:
    return json.loads((work / "plan.json").read_text())


@pytest.mark.logic
@needs_pdb
def test_the_overlapping_pose_is_a_failure_on_the_right_face(tmp_path,
                                                             overlapping):
    """The premise of the next few tests. Without both halves holding, they
    would be testing a different situation than the one they describe."""
    work = tmp_path / "work"
    assert _run_plan(work, overlapping) == 0
    plan = _plan_of(work)
    assert float(plan["initial_receptor_aligned_effector_rmsd"]) > 5.0
    assert plan["interfaces_overlap"] > 0.5


@pytest.mark.logic
@needs_pdb
def test_an_overlapping_interface_halves_the_mutation_count(tmp_path,
                                                            overlapping):
    """An aggressive mutation count is marginal when the cold start is
    already on the right face, and the surviving candidates are second-shell
    residues rather than direct contacts."""
    work = tmp_path / "work"
    _run_plan(work, overlapping, "--max-mutations", "6")
    plan = _plan_of(work)
    assert plan.get("max_mutations_reduced_from") == 6
    assert plan["max_mutations_reduced_to"] <= 3
    assert plan["max_mutations_reduced_reason"]


@pytest.mark.logic
@needs_pdb
def test_a_non_overlapping_interface_keeps_the_requested_count(tmp_path):
    """The reduction is a response to the pathology, not a default."""
    wrong = _wrong_interface_pose(GT, tmp_path / "wrong.pdb")
    work = tmp_path / "work"
    _run_plan(work, wrong, "--max-mutations", "6")
    plan = _plan_of(work)
    assert plan["interfaces_overlap"] <= 0.5
    assert "max_mutations_reduced_from" not in plan


@pytest.mark.logic
@needs_pdb
def test_the_overlap_is_recorded_whichever_way_it_falls(tmp_path,
                                                        overlapping):
    """It is the number the pathology is diagnosed from, so it belongs in
    plan.json whether or not it crossed the threshold."""
    wrong = _wrong_interface_pose(GT, tmp_path / "wrong.pdb")
    for pose, name in ((wrong, "a"), (overlapping, "b")):
        work = tmp_path / name
        _run_plan(work, pose)
        assert 0.0 <= _plan_of(work)["interfaces_overlap"] <= 1.0


# ── the protected set ────────────────────────────────────────────────────────

@pytest.mark.logic
@needs_pdb
def test_the_design_region_is_protected_alongside_the_true_interface(
        tmp_path):
    """The design region is the stretch being validated. A steering mutation
    inside it contaminates the rescue pose by construction, because the
    construct carries the designed sequence and not the steered one."""
    region = tmp_path / "region.txt"
    region.write_text("# 1-based\n30-45\n")
    wrong = _wrong_interface_pose(GT, tmp_path / "wrong.pdb")
    work = tmp_path / "work"
    assert _run_plan(work, wrong, "--design-region-indices-file",
                     str(region)) == 0
    plan = _plan_of(work)
    assert plan["protected_set_source"] == "design_region_union"
    protected = set(plan["protected_set_idx"])
    assert set(range(29, 45)) <= protected          # 1-based 30-45 → 0-based
    assert set(plan["true_interface_idx"]) <= protected


@pytest.mark.logic
@needs_pdb
def test_no_design_region_falls_back_to_the_true_interface(tmp_path):
    """The default source asks for the union, and most runs have no region
    file. Falling back is what keeps that default usable."""
    wrong = _wrong_interface_pose(GT, tmp_path / "wrong.pdb")
    work = tmp_path / "work"
    _run_plan(work, wrong)
    plan = _plan_of(work)
    assert plan["protected_set_idx"] == sorted(plan["true_interface_idx"])


@pytest.mark.logic
@needs_pdb
def test_the_legacy_source_ignores_the_design_region(tmp_path):
    """Kept so a run can be reproduced as it was before the union became the
    default."""
    region = tmp_path / "region.txt"
    region.write_text("30-45\n")
    wrong = _wrong_interface_pose(GT, tmp_path / "wrong.pdb")
    work = tmp_path / "work"
    _run_plan(work, wrong, "--protected-set-source", "true_interface",
              "--design-region-indices-file", str(region))
    plan = _plan_of(work)
    assert plan["protected_set_source"] == "true_interface"
    assert plan["protected_set_idx"] == sorted(plan["true_interface_idx"])


@pytest.mark.logic
@needs_pdb
def test_a_design_region_file_that_is_absent_is_refused(tmp_path):
    """Silently proceeding with no region would leave it unprotected, and
    the run would look like it had been protected. A wrong cold start is
    needed to reach the check, because a clean one short-circuits first."""
    wrong = _wrong_interface_pose(GT, tmp_path / "wrong.pdb")
    work = tmp_path / "work"
    assert _run_plan(work, wrong, "--design-region-indices-file",
                     str(tmp_path / "nope.txt")) == 1


@pytest.mark.logic
@needs_pdb
def test_a_design_region_outside_the_receptor_is_refused(tmp_path):
    """The indices are positional, so one past the end means the file was
    derived against a different receptor."""
    region = tmp_path / "region.txt"
    region.write_text("900-910\n")
    wrong = _wrong_interface_pose(GT, tmp_path / "wrong.pdb")
    work = tmp_path / "work"
    assert _run_plan(work, wrong, "--design-region-indices-file",
                     str(region)) == 1


@pytest.mark.logic
@needs_pdb
def test_no_steering_mutation_lands_in_the_design_region(tmp_path):
    """The point of protecting it. This is the same invariant as the true
    interface test above, applied to the other protected block."""
    region = tmp_path / "region.txt"
    region.write_text("30-45\n")
    wrong = _wrong_interface_pose(GT, tmp_path / "wrong.pdb")
    work = tmp_path / "work"
    _run_plan(work, wrong, "--design-region-indices-file", str(region))
    steered = work / "steered"
    if not steered.is_dir():
        pytest.skip("this cold start did not reach the steering stage")
    wild = _plan_of(work)["wild_type_receptor_seq"]
    region_idx = set(range(29, 45))
    for d in sorted(steered.iterdir()):
        fasta = d / "receptor.fasta"
        if not fasta.is_file():
            continue
        seq = "".join(x for x in fasta.read_text().splitlines()
                      if not x.startswith(">"))
        changed = {i for i, (a, b) in enumerate(zip(wild, seq)) if a != b}
        assert not (changed & region_idx), (
            f"{d.name} mutated design-region positions "
            f"{sorted(changed & region_idx)}")


# ── the multi-seed cold start ────────────────────────────────────────────────

@pytest.mark.logic
@needs_pdb
def test_extra_cold_start_seeds_are_run_and_recorded(tmp_path):
    """One cold start cannot distinguish a prediction that is reliably wrong
    from one Boltz happens to get wrong on that seed. The per-seed rows are
    what the median across seeds is taken over."""
    wrong = _wrong_interface_pose(GT, tmp_path / "wrong.pdb")
    work = tmp_path / "work"
    assert _run_plan(work, wrong, "--num-seeds", "3") == 0
    plan = _plan_of(work)
    assert len(plan["cold_start_seeds"]) == 3
    assert plan["cold_start_n_clean"] == 0
    assert plan["cold_start_all_clean"] is False


@pytest.mark.logic
@needs_pdb
def test_the_extra_seeds_get_distinct_boltz_seeds(tmp_path):
    """Same seed means the same prediction, which would make the confirmation
    a copy of the first run rather than a second sample."""
    wrong = _wrong_interface_pose(GT, tmp_path / "wrong.pdb")
    work = tmp_path / "work"
    _run_plan(work, wrong, "--num-seeds", "3")
    records = _plan_of(work)["cold_start_seeds"]
    seeds = [r["boltz_seed"] for r in records]
    assert len(set(seeds)) == 3


@pytest.mark.logic
@needs_pdb
def test_skip_steering_does_not_pay_for_extra_seeds(tmp_path):
    """The extra seeds exist to confirm a steering decision. With steering
    off there is no decision to confirm, so the GPU time is wasted."""
    wrong = _wrong_interface_pose(GT, tmp_path / "wrong.pdb")
    work = tmp_path / "work"
    assert _run_plan(work, wrong, "--num-seeds", "3", "--skip-steering") == 0
    assert len(_plan_of(work)["cold_start_seeds"]) == 1


# ── the candidate pool ───────────────────────────────────────────────────────

@pytest.mark.logic
@needs_pdb
def test_a_pool_smaller_than_the_mutation_count_caps_the_count(tmp_path):
    """Capping keeps protection strict. The alternative, loosening the
    protected set to reach the requested count, puts steering mutations
    inside the region being validated, which contaminates the rescue pose by
    construction. Fewer mutations beats contaminated ones."""
    wrong = _wrong_interface_pose(GT, tmp_path / "wrong.pdb")
    work = tmp_path / "work"
    assert _run_plan(work, wrong, "--max-mutations", "40") == 0
    plan = _plan_of(work)
    assert plan["max_mutations_reduced_from"] == 40
    assert 1 <= plan["max_mutations_effective"] < 40
    assert plan["protected_set_idx"] == sorted(plan["true_interface_idx"])


@pytest.mark.logic
@needs_pdb
def test_the_recorded_mutation_count_is_the_effective_one(tmp_path):
    """Later cycles and the reversion pass read plan["max_mutations"], so on
    a reduced run it has to be the count that was actually used. The number
    the user asked for moves to max_mutations_requested."""
    wrong = _wrong_interface_pose(GT, tmp_path / "wrong.pdb")
    work = tmp_path / "work"
    _run_plan(work, wrong, "--max-mutations", "40")
    plan = _plan_of(work)
    assert plan["max_mutations_requested"] == 40
    assert plan["max_mutations"] == plan["max_mutations_effective"]
    assert plan["max_mutations"] < 40


@pytest.mark.logic
@needs_pdb
def test_an_unreduced_run_records_the_requested_count(tmp_path):
    wrong = _wrong_interface_pose(GT, tmp_path / "wrong.pdb")
    work = tmp_path / "work"
    _run_plan(work, wrong, "--max-mutations", "3")
    plan = _plan_of(work)
    assert plan["max_mutations"] == 3
    assert plan["max_mutations_effective"] == 3
    assert "max_mutations_requested" not in plan


@pytest.mark.logic
@needs_pdb
def test_the_pool_is_widened_to_hold_the_mutation_count(tmp_path):
    """--candidate-pool-size is a floor, not a ceiling. A pool narrower than
    max_mutations could not supply enough distinct positions for one design,
    so every design would draw the same ones."""
    wrong = _wrong_interface_pose(GT, tmp_path / "wrong.pdb")
    work = tmp_path / "work"
    _run_plan(work, wrong, "--max-mutations", "12",
              "--candidate-pool-size", "4")
    plan = _plan_of(work)
    assert len(plan["candidate_pool"]) >= plan["max_mutations_effective"]


@pytest.mark.logic
@needs_pdb
def test_a_fully_protected_interface_falls_back_to_the_second_shell(tmp_path):
    """Every direct candidate protected is not the same as nothing to do.
    The fallback looks one shell out, at residues near the predicted effector
    but outside both the wrong interface and the protected set."""
    region = tmp_path / "region.txt"
    region.write_text("1-70\n")
    wrong = _wrong_interface_pose(GT, tmp_path / "wrong.pdb")
    work = tmp_path / "work"
    rc = _run_plan(work, wrong, "--design-region-indices-file", str(region),
                   "--max-mutations", "6")
    plan = _plan_of(work)
    assert plan["candidate_pool"] == []
    assert plan["second_shell_fallback_used"] is False
    assert rc == 1


@pytest.mark.logic
@needs_pdb
def test_an_unsteerable_run_still_writes_the_cold_start_row(tmp_path):
    """The cold start is a result even when steering cannot proceed. Losing
    it would drop the target from the comparison entirely rather than
    recording that it had nothing steerable."""
    region = tmp_path / "region.txt"
    region.write_text("1-70\n")
    wrong = _wrong_interface_pose(GT, tmp_path / "wrong.pdb")
    work = tmp_path / "work"
    _run_plan(work, wrong, "--design-region-indices-file", str(region))
    rows = (work / "steered_results.csv").read_text().splitlines()
    assert len(rows) == 2, "expected a header and the cold-start row"


# ── sampling mode, where there is nothing left to steer ──────────────────────
#
# Four conditions have to hold at once: the candidate pool is empty, a wrong
# interface exists, it overlaps the true interface by more than half, and the
# second-shell fallback found fewer than three residues. The engine's answer is
# not to give up. It re-predicts the wild-type sequence under several seeds,
# because Boltz's diffusion is stochastic and a cold start already on the right
# face may land closer on another roll.
#
# The overlapping pose supplies the overlap. A design region covering the whole
# receptor empties both the pool and the second shell, which is what a target
# whose wrong interface sits entirely inside the protected region looks like.

@pytest.fixture
def unsteerable(tmp_path, overlapping):
    """An overlapping cold start with every receptor position protected."""
    region = tmp_path / "whole_receptor.txt"
    region.write_text(f"1-{REC_LEN}\n")
    return overlapping, region


@pytest.mark.logic
@needs_pdb
def test_sampling_mode_needs_all_four_conditions(tmp_path, unsteerable):
    """The premise of the tests below. Without all four the engine either
    steers or refuses, and neither exercises this path."""
    pose, region = unsteerable
    work = tmp_path / "work"
    assert _run_plan(work, pose, "--design-region-indices-file",
                     str(region)) == 0
    plan = _plan_of(work)
    assert plan["sampling_mode"] is True
    assert plan["candidate_pool"] == []
    assert plan["initial_wrong_interface_idx"]
    assert plan["interfaces_overlap"] > 0.5
    assert plan["second_shell_fallback_used"] is False


@pytest.mark.logic
@needs_pdb
def test_sampling_mode_does_not_skip_steering(tmp_path, unsteerable):
    """skip_steering is deliberately left unset, because predict-one still
    has work to do. The empty candidate pool is how the design loop knows
    not to mutate anything."""
    pose, region = unsteerable
    work = tmp_path / "work"
    _run_plan(work, pose, "--design-region-indices-file", str(region))
    plan = _plan_of(work)
    assert plan.get("skip_steering") is not True
    assert plan["designs"], "sampling mode produced no design slots"


@pytest.mark.logic
@needs_pdb
def test_sampling_mode_records_why_it_fired(tmp_path, unsteerable):
    """A run that mutated nothing looks like a failed run from its outputs
    alone. The reason is the only thing distinguishing the two."""
    pose, region = unsteerable
    work = tmp_path / "work"
    _run_plan(work, pose, "--design-region-indices-file", str(region))
    reason = _plan_of(work)["sampling_mode_reason"]
    assert "interfaces overlap" in reason
    assert "second-shell fallback produced < 3 candidates" in reason


@pytest.mark.logic
@needs_pdb
def test_every_sampling_mode_design_is_the_wild_type(tmp_path, unsteerable):
    """One unique sequence, replicated. A mutation here would be a steering
    edit at a protected position, which is the thing the pool being empty
    says cannot be done."""
    pose, region = unsteerable
    work = tmp_path / "work"
    _run_plan(work, pose, "--design-region-indices-file", str(region))
    plan = _plan_of(work)
    assert plan["n_unique_sequences"] == 1
    wild = plan["wild_type_receptor_seq"]
    steered = work / "steered"
    for d in sorted(p for p in steered.iterdir() if p.is_dir()):
        seq = "".join(x for x in (d / "receptor.fasta").read_text().splitlines()
                      if not x.startswith(">"))
        assert seq == wild, f"{d.name} is not the wild-type sequence"
        assert not (d / "mutations.tsv").exists() or \
            (d / "mutations.tsv").read_text().strip().count("\n") == 0


@pytest.mark.logic
@needs_pdb
def test_the_prediction_count_is_slots_times_seeds(tmp_path, unsteerable):
    """Outside sampling mode the count is unique sequences times seeds, and
    there is only one unique sequence here. Using that would collapse the
    run to a single prediction and defeat the re-sampling."""
    pose, region = unsteerable
    work = tmp_path / "work"
    _run_plan(work, pose, "--design-region-indices-file", str(region),
              "--n-designs", "2", "--num-seeds", "3")
    plan = _plan_of(work)
    assert plan["n_unique_sequences"] == 1
    assert plan["total_predictions"] == 6


@pytest.mark.logic
@needs_pdb
def test_each_sampling_mode_slot_gets_its_own_seed(tmp_path, unsteerable):
    """Identical seeds would make the re-predictions identical, so the
    sampling variance the mode exists to exploit would never appear."""
    pose, region = unsteerable
    work = tmp_path / "work"
    _run_plan(work, pose, "--design-region-indices-file", str(region),
              "--n-designs", "3", "--num-seeds", "1")
    designs = _plan_of(work)["designs"]
    seeds = [d.get("boltz_seed", d.get("seed")) for d in designs]
    assert len(designs) == 3
    assert len(set(seeds)) == len(seeds), f"repeated seed in {seeds}"


@pytest.mark.logic
@needs_pdb
def test_the_interface_logs_are_not_written_in_sampling_mode(tmp_path,
                                                              unsteerable):
    """Both files list the wrong interface against the protected set. In
    sampling mode every wrong-interface residue is protected, so the files
    would carry no information."""
    pose, region = unsteerable
    work = tmp_path / "work"
    _run_plan(work, pose, "--design-region-indices-file", str(region))
    assert not (work / "wrong_interface_residues.txt").exists()
    assert not (work / "true_interface_residues.txt").exists()
