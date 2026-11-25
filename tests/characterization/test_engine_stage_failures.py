"""What one failed stage costs the engine, and what it leaves behind.

cmd_plan and cmd_predict_one wrap every stage of the pass in its own handler.
Some of those handlers abort the run, because the stage they guard produces the
indices everything downstream is positional against. Others absorb the failure
and carry on with an explicit blank, because the unit that failed is one seed or
one design out of many. Which handler does which is the engine's whole failure
policy, and a happy-path run never shows it.

Nearly all of it is reachable from a real input. A ligand sharing the effector
chain makes the mmCIF template writer emit `label_seq_id .` and raise. Two
chains 100 Å apart leave no pocket for the constraints block. A ground truth
whose effector is written as HETATM parses for its sequence and then has no
effector residues for the ATOM-only contact reader. A prediction that came back
with one chain cannot be scanned for a wrong interface. A stub run_boltz that
raises is an input rather than a mock, because the harness has to supply the
Boltz call either way.

One handler resists a real input and says so in its own docstring.

Every ground truth built here is small and hand-placed, so the expected outcome
is known before the engine runs.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from unittest import mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin"))

import boltz2_negative_steering as bns  # noqa: E402

GT = _REPO_ROOT / "data" / "reference_complexes" / "6G10.pdb"
needs_pdb = pytest.mark.skipif(not GT.is_file(), reason="6G10 reference absent")

REC_LEN = 76

ONE_TO_THREE = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY",
    "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN",
    "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER", "T": "THR", "V": "VAL",
    "W": "TRP", "Y": "TYR",
}


# ── hand-built ground truths ─────────────────────────────────────────────────
# A receptor of ALA and an effector of GLY, both strung along x. `separation`
# is the gap in y, so whether the two chains touch is set by one number.

def _atom(serial, name, resname, chain, resnum, x, y, z,
          record="ATOM  ", element="C"):
    return (f"{record}{serial:5d} {name:<4s} {resname:>3s} {chain}{resnum:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n")


def _two_chain_pdb(path: Path, *, separation: float = 4.0, n: int = 10,
                   effector_ligand: str | None = None,
                   effector_as_hetatm: bool = False) -> Path:
    """A two-chain complex of `n` residues per chain, `separation` Å apart."""
    lines, serial = [], 1
    for i in range(n):
        for name, dx, dy, element in (("N", 0.0, 0.0, "N"), ("CA", 0.5, 0.0, "C"),
                                      ("C", 1.0, 0.0, "C"), ("CB", 0.5, 1.0, "C")):
            lines.append(_atom(serial, name, "ALA", "A", i + 1,
                               3.8 * i + dx, dy, 0.0, element=element))
            serial += 1
    record = "HETATM" if effector_as_hetatm else "ATOM  "
    for i in range(n):
        for name, dx, element in (("N", 0.0, "N"), ("CA", 0.5, "C"),
                                  ("C", 1.0, "C")):
            lines.append(_atom(serial, name, "GLY", "B", i + 1,
                               3.8 * i + dx, separation, 0.0,
                               record=record, element=element))
            serial += 1
    if effector_ligand:
        lines.append(_atom(serial, "S", effector_ligand, "B", 500,
                           5.0, separation + 2.0, 2.0,
                           record="HETATM", element="S"))
    lines.append("END\n")
    path.write_text("".join(lines))
    return path


# ── poses derived from the reference complex ─────────────────────────────────

def _wrong_interface_pose(src: Path, dst: Path) -> Path:
    """The effector reflected through the receptor centroid.

    The standard cold start for these tests. It keeps real contacts while
    putting them on the opposite face, so the run cannot short-circuit.
    """
    lines = src.read_text().splitlines(keepends=True)
    co = [(float(x[30:38]), float(x[38:46]), float(x[46:54]))
          for x in lines if x.startswith("ATOM") and x[21] == "A"]
    cx, cy, cz = (sum(c[i] for c in co) / len(co) for i in range(3))
    out = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")) and len(line) > 54 and line[21] == "B":
            out.append(f"{line[:30]}{2 * cx - float(line[30:38]):8.3f}"
                       f"{2 * cy - float(line[38:46]):8.3f}"
                       f"{2 * cz - float(line[46:54]):8.3f}{line[54:]}")
        else:
            out.append(line)
    dst.write_text("".join(out))
    return dst


def _without_the_effector(src: Path, dst: Path) -> Path:
    """The receptor alone, which is what a half-finished Boltz run returns."""
    dst.write_text("".join(
        line for line in src.read_text().splitlines(keepends=True)
        if not (line.startswith(("ATOM", "HETATM")) and len(line) > 21
                and line[21] == "B")))
    return dst


def _effector_pushed_away(src: Path, dst: Path, dx: float = 300.0) -> Path:
    """The effector translated off the receptor entirely, so no contacts."""
    out = []
    for line in src.read_text().splitlines(keepends=True):
        if line.startswith(("ATOM", "HETATM")) and len(line) > 54 and line[21] == "B":
            out.append(f"{line[:30]}{float(line[30:38]) + dx:8.3f}{line[38:]}")
        else:
            out.append(line)
    dst.write_text("".join(out))
    return dst


def _relabel_receptor(pdb: Path, sequence: str, chain: str = "A") -> None:
    """Rewrite chain `chain`'s residue names to spell `sequence`.

    Coordinates are untouched, so every geometric metric is unchanged. The
    engine refuses a prediction whose receptor sequence is not the one it asked
    for, so a stub that stands in for a successful Boltz call has to do this.
    """
    lines = pdb.read_text().splitlines(keepends=True)
    order, seen = [], set()
    for line in lines:
        if line.startswith("ATOM") and line[21] == chain:
            key = line[22:27]
            if key not in seen:
                seen.add(key)
                order.append(key)
    mapping = {k: ONE_TO_THREE.get(aa, "ALA") for k, aa in zip(order, sequence)}
    out = []
    for line in lines:
        if line.startswith("ATOM") and line[21] == chain and line[22:27] in mapping:
            out.append(f"{line[:17]}{mapping[line[22:27]]:>3}{line[20:]}")
        else:
            out.append(line)
    pdb.write_text("".join(out))


def _requested_sequence(out_dir: Path) -> str | None:
    """The receptor.fasta the plan staged next to this prediction, if any."""
    for candidate in (out_dir, out_dir.parent, out_dir.parent.parent):
        f = candidate / "receptor.fasta"
        if f.is_file():
            return "".join(x for x in f.read_text().splitlines()
                           if not x.startswith(">"))
    return None


# ── the Boltz stand-in ───────────────────────────────────────────────────────

def _stub_boltz(scratch: Path, prediction: Path, *, relabel: bool = True,
                fail_on: tuple[int, ...] = ()):
    """A run_boltz stand-in returning `prediction` from a scratch path.

    The engine copies the returned file to out_dir/prediction.pdb, so writing it
    there first makes the copy a same-file error. `fail_on` lists the 1-based
    call numbers that raise instead, which is how a GPU failure arrives.
    """
    counter = [0]

    def fake(yaml_path, out_dir, container, recycling_steps,
             diffusion_samples, seed, no_kernels):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        counter[0] += 1
        if counter[0] in fail_on:
            raise RuntimeError(f"boltz exited 1 on call {counter[0]}")
        d = scratch / f"pred_{counter[0]}"
        d.mkdir(parents=True, exist_ok=True)
        pred = d / "prediction.pdb"
        shutil.copy(prediction, pred)
        if relabel:
            requested = _requested_sequence(out_dir)
            if requested:
                _relabel_receptor(pred, requested)
        return pred

    return fake, counter


def _engine(fake_boltz, *argv) -> int:
    with mock.patch.object(bns, "run_boltz", fake_boltz), \
            mock.patch.object(sys, "argv",
                              ["boltz2_negative_steering.py", *argv]):
        return bns.main()


def _plan(work: Path, ground_truth: Path, fake_boltz, *extra: str) -> int:
    return _engine(fake_boltz, "plan",
                   "--ground-truth", str(ground_truth), "--workdir", str(work),
                   "--receptor", "A", "--effector", "B",
                   "--boltz-container", "/fake.img",
                   "--n-designs", "2", "--num-seeds", "1",
                   "--diffusion-samples", "1", "--recycling-steps", "1",
                   *extra)


def _plan_of(work: Path) -> dict:
    return json.loads((work / "plan.json").read_text())


def _staged_plan(root: Path, *extra: str):
    """A plan on the wrong-interface cold start, with one design staged."""
    wrong = _wrong_interface_pose(GT, root / "wrong.pdb")
    work = root / "work"
    fake, _ = _stub_boltz(root / "scratch", wrong)
    assert _engine(fake, "plan", "--ground-truth", str(GT),
                   "--workdir", str(work), "--receptor", "A", "--effector", "B",
                   "--boltz-container", "/fake.img", "--n-designs", "1",
                   "--num-seeds", "1", "--diffusion-samples", "1",
                   "--recycling-steps", "1", *extra) == 0
    return work, wrong


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — the plan stages that abort the run
# ─────────────────────────────────────────────────────────────────────────────
# Each of these four produces something every later index is positional
# against. There is no partial answer to carry forward, so the handler prints
# the stage that failed and returns 1 before any GPU time is spent on the
# designs.

@pytest.mark.logic
def test_a_chain_under_twenty_residues_warns_about_the_chain_ids(tmp_path, capsys):
    """Ten residues is a plausible chain and an implausible receptor.

    The usual cause is a chain ID naming a ligand or a peptide tag rather than
    the protein, which produces a complete run against the wrong molecule. The
    warning is the only place that is visible, so the run continues.
    """
    gt = _two_chain_pdb(tmp_path / "small.pdb")
    fake, _ = _stub_boltz(tmp_path / "scratch", gt)
    assert _plan(tmp_path / "work", gt, fake) == 0
    assert "short sequence(s)" in capsys.readouterr().out
    assert _plan_of(tmp_path / "work")["wild_type_receptor_seq"] == "A" * 10


@pytest.mark.logic
def test_a_ligand_in_the_effector_chain_stops_the_run(tmp_path, capsys):
    """A real malformed ground truth, and the commonest one.

    Many PDBs give a bound ligand the protein's chain ID. gemmi reads the
    effector sequence from it without complaint, so the plan gets as far as the
    template, where the writer emits `label_seq_id .` for the ligand row and the
    validation refuses it. AF3 and Boltz would both reject that template 30
    minutes into a GPU job, so the run has to stop here instead.
    """
    gt = _two_chain_pdb(tmp_path / "lig.pdb", effector_ligand="SO4")
    assert bns.get_chain_sequence(gt, "B") == "G" * 10, \
        "the ligand already broke sequence extraction, so this tests nothing"
    fake, counter = _stub_boltz(tmp_path / "scratch", gt)

    assert _plan(tmp_path / "work", gt, fake) == 1
    assert "Effector template extraction failed" in capsys.readouterr().out
    assert counter[0] == 0, "GPU time was spent after the template failed"
    assert not (tmp_path / "work" / "plan.json").exists()


@pytest.mark.logic
def test_two_chains_that_never_touch_stop_the_constrained_run(tmp_path, capsys):
    """100 Å apart leaves no pocket and no contact pairs to restrain.

    An empty constraints block would be written into the YAML and the
    prediction would run unconstrained, which is a different experiment
    reported under the constrained label.
    """
    gt = _two_chain_pdb(tmp_path / "far.pdb", separation=100.0)
    fake, counter = _stub_boltz(tmp_path / "scratch", gt)

    assert _plan(tmp_path / "work", gt, fake,
                 "--boltz-constraints", "--no-effector-template") == 1
    assert "Benchmark-style Boltz constraints failed" in capsys.readouterr().out
    assert counter[0] == 0, "GPU time was spent after the constraints failed"
    assert not (tmp_path / "work" / "initial").exists(), \
        "a YAML was written with no constraints in it"


@pytest.mark.logic
@needs_pdb
def test_a_failed_initial_prediction_stops_the_run(tmp_path, capsys):
    """The cold start is the baseline every design's delta is measured from.

    Without it there is nothing to steer away from and nothing to compare
    against, so the run cannot continue on a partial answer.
    """
    fake, _ = _stub_boltz(tmp_path / "scratch", GT, fail_on=(1,))
    assert _plan(tmp_path / "work", GT, fake) == 1
    assert "Initial prediction failed" in capsys.readouterr().out
    assert not (tmp_path / "work" / "plan.json").exists()


@pytest.mark.logic
def test_an_effector_written_as_hetatm_stops_true_interface_detection(tmp_path,
                                                                      capsys):
    """gemmi reads HETATM amino acids as a polymer. The contact reader does not.

    So the sequence comes back for both chains and the ground truth looks fine,
    then heavy-atom contact detection finds no effector residues at all. The
    true interface is the protected set, and an empty one would let steering
    mutate the real binding site, so this aborts rather than proceeding.
    """
    gt = _two_chain_pdb(tmp_path / "hetb.pdb", effector_as_hetatm=True)
    assert bns.get_chain_sequence(gt, "B") == "G" * 10, \
        "HETATM already broke sequence extraction, so this tests nothing"
    fake, _ = _stub_boltz(tmp_path / "scratch", gt)

    assert _plan(tmp_path / "work", gt, fake, "--no-effector-template") == 1
    assert "True-interface detection failed" in capsys.readouterr().out
    assert not (tmp_path / "work" / "plan.json").exists()


@pytest.mark.logic
@needs_pdb
def test_a_prediction_missing_its_effector_stops_wrong_interface_detection(
        tmp_path, capsys):
    """A single-chain prediction is what a half-finished Boltz run leaves.

    The three RMSDs come back nan, which is tolerated, so the run reaches the
    wrong-interface scan. That scan is what the candidate pool is built from,
    and a prediction with no effector has no wrong interface to measure.
    """
    pred = _without_the_effector(GT, tmp_path / "receptor_only.pdb")
    fake, _ = _stub_boltz(tmp_path / "scratch", pred)

    assert _plan(tmp_path / "work", GT, fake) == 1
    out = capsys.readouterr().out
    assert "Wrong-interface detection failed" in out
    assert "True-interface detection failed" not in out, \
        "detection failed on the ground truth, not on the prediction"
    assert not (tmp_path / "work" / "plan.json").exists()


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — the plan stages that cost one unit
# ─────────────────────────────────────────────────────────────────────────────
# An extra cold-start seed and the effector interface mask are both one input
# to a decision taken over many. Aborting the run over either would throw away
# the seeds that did land, so each failure is recorded as an explicit blank and
# the run continues.

@pytest.mark.logic
@needs_pdb
def test_a_failed_extra_cold_start_seed_still_records_its_seed(tmp_path, capsys):
    """One seed of N is one sample, not the run.

    The nan row is what keeps the aggregator seeing N seeds. Dropping it would
    make a sequence that failed on one seed look like a sequence that was only
    ever sampled on the others, and the all-clean policy counts clean seeds
    against the total.
    """
    wrong = _wrong_interface_pose(GT, tmp_path / "wrong.pdb")
    work = tmp_path / "work"
    fake, counter = _stub_boltz(tmp_path / "scratch", wrong, fail_on=(2,))

    assert _engine(fake, "plan", "--ground-truth", str(GT),
                   "--workdir", str(work), "--receptor", "A", "--effector", "B",
                   "--boltz-container", "/fake.img", "--n-designs", "2",
                   "--num-seeds", "3", "--diffusion-samples", "1",
                   "--recycling-steps", "1") == 0
    assert counter[0] == 3, "the third seed was skipped after the second failed"
    assert "seed 1: FAILED" in capsys.readouterr().out

    seeds = _plan_of(work)["cold_start_seeds"]
    assert len(seeds) == 3, "the failed seed was dropped from the record"
    failed = seeds[1]
    assert failed["seed_offset"] == 1
    for key in ("ra_eff", "ind_rec", "ind_eff"):
        assert failed[key] != failed[key], f"{key} was not nan"
    assert failed["pdb_path"] == "", "a path was recorded for a missing PDB"
    assert not (work / "initial_prediction_s1.pdb").exists()
    assert _plan_of(work)["designs"], "the run gave up after one failed seed"


@pytest.mark.logic
@needs_pdb
def test_an_unimportable_compute_metrics_empties_the_effector_mask(tmp_path,
                                                                   monkeypatch,
                                                                   capsys):
    """The mask narrows later contamination checks, and it is optional.

    No input reaches this handler. compute_effector_interface_residues catches
    its own parse failures and returns an empty set, so only the lazy import
    above it can raise. The import sits inside the try precisely because
    compute_metrics is a sibling script in bin/ rather than an installed
    package, and a None entry in sys.modules is how Python itself signals an
    unimportable module.

    The three plan keys have to come out empty rather than absent. Downstream
    reads them unconditionally and treats empty as "no filter", which is the
    legacy unfiltered behaviour.
    """
    monkeypatch.setitem(sys.modules, "compute_metrics", None)
    wrong = _wrong_interface_pose(GT, tmp_path / "wrong.pdb")
    work = tmp_path / "work"
    fake, _ = _stub_boltz(tmp_path / "scratch", wrong)

    assert _plan(work, GT, fake) == 0
    out = capsys.readouterr().out
    assert "could not compute effector interface residue mask" in out
    assert "fall back to unfiltered" in out

    plan = _plan_of(work)
    assert plan["effector_interface_residues"] == []
    assert plan["effector_interface_atoms"] == []
    assert plan["effector_interface_atom_cutoff_angstroms"] is None
    assert plan["designs"], "an optional mask took the whole run down"


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — the plan's own diagnostics
# ─────────────────────────────────────────────────────────────────────────────
# Four warnings about a run that will finish and report less than it looks
# like. Each one names the input that caused it, because the output alone does
# not distinguish a degenerate run from a failed one.

@pytest.mark.logic
@needs_pdb
def test_a_prediction_with_no_contacts_has_no_interface_to_steer(tmp_path,
                                                                 capsys):
    """An effector 300 Å off the receptor gives an empty wrong interface.

    That is not the same as a protected one. There is nothing to steer away
    from, so sampling mode does not apply either, and the skip reason has to
    distinguish the two or the target looks like it had candidates that were
    all filtered out.
    """
    pred = _effector_pushed_away(GT, tmp_path / "apart.pdb")
    work = tmp_path / "work"
    fake, _ = _stub_boltz(tmp_path / "scratch", pred)

    assert _plan(work, GT, fake) == 1
    assert "No interface residues found within" in capsys.readouterr().out
    plan = _plan_of(work)
    assert plan["initial_wrong_interface_idx"] == []
    assert plan["skip_reason"] == "no interface residues found"
    assert plan["skip_steering"] is True
    assert (work / "steered_results.csv").is_file(), \
        "the cold start was dropped instead of reported"


@pytest.mark.logic
@needs_pdb
def test_conservative_mode_warns_that_every_design_will_be_identical(tmp_path,
                                                                     capsys):
    """conservative is a lookup table, so a pool it must use whole is one design.

    --max-mutations 40 caps to the pool size, which makes every draw take every
    position. The substitution for each one is then fixed, so the N designs are
    one design repeated, and the run reports N GPU jobs' worth of replicates as
    N designs.
    """
    wrong = _wrong_interface_pose(GT, tmp_path / "wrong.pdb")
    work = tmp_path / "work"
    fake, _ = _stub_boltz(tmp_path / "scratch", wrong)

    assert _plan(work, GT, fake, "--mode", "conservative",
                 "--max-mutations", "40") == 0
    assert "will produce N IDENTICAL designs" in capsys.readouterr().out
    plan = _plan_of(work)
    assert len(plan["candidate_pool"]) <= plan["max_mutations_effective"]
    assert plan["n_unique_sequences"] == 1


@pytest.mark.logic
@needs_pdb
def test_a_pool_with_too_little_diversity_reports_how_many_it_found(tmp_path,
                                                                    capsys):
    """Two designs asked for, one sequence possible.

    The retry budget is ten draws per design, so exhausting it means the
    combinatorial space is smaller than the request rather than the RNG being
    unlucky. The count found is what tells the operator to widen the pool.
    """
    wrong = _wrong_interface_pose(GT, tmp_path / "wrong.pdb")
    work = tmp_path / "work"
    fake, _ = _stub_boltz(tmp_path / "scratch", wrong)

    assert _plan(work, GT, fake, "--mode", "conservative",
                 "--max-mutations", "40") == 0
    out = capsys.readouterr().out
    assert "only 1 unique sequences found after 20 attempts (requested 2)" in out
    plan = _plan_of(work)
    assert plan["n_unique_sequences"] == 1 < plan["n_designs"]


@pytest.mark.logic
@needs_pdb
def test_duplicate_draws_are_deduplicated_rather_than_predicted_twice(tmp_path,
                                                                      capsys):
    """A pool of two positions mutated one at a time gives two sequences.

    alanine is deterministic per position, so the second draw repeats the first
    whenever it lands on the same position. The dedup is what stops two GPU
    slots predicting one sequence, and the collision count is the only record
    that the draws were retried. The RNG is seeded from --seed, so which seed
    collides is fixed rather than flaky.
    """
    wrong = _wrong_interface_pose(GT, tmp_path / "wrong.pdb")
    work = tmp_path / "work"
    fake, _ = _stub_boltz(tmp_path / "scratch", wrong)

    assert _plan(work, GT, fake, "--mode", "alanine", "--max-mutations", "1",
                 "--candidate-pool-size", "2") == 0
    assert "collision(s) deduplicated" in capsys.readouterr().out
    plan = _plan_of(work)
    assert plan["n_unique_sequences"] == plan["n_designs"] == 2, \
        "this is the branch where the request was met, collisions and all"
    sequences = {
        "".join(x for x in (Path(d["dir"]) / "receptor.fasta").read_text()
                .splitlines() if not x.startswith(">"))
        for d in plan["designs"]
    }
    assert len(sequences) == 2, "a duplicate survived the dedup"


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — predict-one, where one design is the unit
# ─────────────────────────────────────────────────────────────────────────────
# Every design is an independent array task. A failure inside one must cost
# that design and no other, because the array's other tasks are already
# running and the run is scored over whichever designs landed.

@pytest.mark.logic
@needs_pdb
def test_an_unreadable_resumed_prediction_is_recomputed(tmp_path):
    """A truncated prediction.pdb beside a finished result.json.

    That pair is what a task killed during the copy leaves behind. The resume
    check reads the structure's receptor sequence to confirm the prediction
    belongs to this design, and an unparseable file makes that read raise. The
    handler has to fall through to a fresh prediction, because treating the
    unreadable file as a match would score the design off a truncated
    structure.
    """
    work, wrong = _staged_plan(tmp_path)
    fake, counter = _stub_boltz(tmp_path / "scratch2", wrong)
    assert _engine(fake, "predict-one", "--workdir", str(work),
                   "--index", "0") == 0
    assert counter[0] == 1

    design = Path(_plan_of(work)["designs"][0]["dir"])
    assert (design / "result.json").is_file()
    (design / "prediction.pdb").write_text("")

    assert _engine(fake, "predict-one", "--workdir", str(work),
                   "--index", "0") == 0
    assert counter[0] == 2, "an unreadable prediction was reused as a resume"
    wanted = "".join(x for x in (design / "receptor.fasta").read_text().splitlines()
                     if not x.startswith(">"))
    assert bns.get_chain_sequence(design / "prediction.pdb", "A") == wanted, \
        "the truncated prediction was left on disk"


@pytest.mark.logic
@needs_pdb
def test_a_failed_same_site_check_leaves_the_jaccards_nan(tmp_path, capsys):
    """A prediction of the wild type where the design asked for a mutant.

    A real failure mode, and the reason the check passes expected_rec_seq. It
    happens when a stale YAML or a stale cached prediction is picked up, and
    contact detection refuses the mismatch rather than returning indices
    against the wrong sequence.

    The three RMSDs are measured by sequence alignment and survive the
    mismatch, so the design still has a result. Only the Jaccards are lost, and
    nan is what says so. A zero would read downstream as a measured complete
    mismatch between the steered and the wrong interface.
    """
    work, wrong = _staged_plan(tmp_path)
    fake, _ = _stub_boltz(tmp_path / "scratch2", wrong, relabel=False)

    assert _engine(fake, "predict-one", "--workdir", str(work),
                   "--index", "0") == 0
    assert "same-site check failed" in capsys.readouterr().err

    design = Path(_plan_of(work)["designs"][0]["dir"])
    result = json.loads((design / "result.json").read_text())
    assert result["status"] == "ok", "a lost Jaccard failed the whole design"
    assert result["receptor_aligned_effector_rmsd"] is not None
    assert result["wrong_jaccard"] is None, "nan was not serialised to null"
    assert result["true_jaccard"] is None
    assert result["n_shared_wrong"] == 0
    assert result["n_shared_true"] == 0
    assert result["steered_interface_idx"] == []


@pytest.mark.logic
@needs_pdb
def test_a_failed_prediction_is_one_failed_design_not_a_failed_run(tmp_path,
                                                                   capsys):
    """A design whose Boltz call raises still writes a result.

    collect reads one result.json per design. An absent file and a failed one
    are different things, and only the failed one carries the reason, so the
    handler writes the nan row itself rather than leaving the task silent. The
    exit code is 1, which is what marks the array task as failed without
    stopping its siblings.
    """
    work, _ = _staged_plan(tmp_path)
    fake, _ = _stub_boltz(tmp_path / "scratch2", GT, fail_on=(1,))

    assert _engine(fake, "predict-one", "--workdir", str(work),
                   "--index", "0") == 1
    assert "FAILED:" in capsys.readouterr().err

    design = Path(_plan_of(work)["designs"][0]["dir"])
    result = json.loads((design / "result.json").read_text())
    assert result["status"].startswith("failed: ")
    assert result["receptor_aligned_effector_rmsd"] is None
    assert result["independent_receptor_rmsd"] is None
    assert result["independent_effector_rmsd"] is None
    assert result["pdb"] == ""
    assert result["design"] == _plan_of(work)["designs"][0]["name"], \
        "the failed row does not name the design it belongs to"


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — collect, aggregating over seeds that may all have failed
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.logic
@needs_pdb
def test_a_sequence_group_with_no_measured_seed_aggregates_to_blanks(tmp_path):
    """Two seeds of one sequence, neither of which produced a result.

    The mean of nothing is not zero. Writing a number here would put a measured
    value in the comparison table for a sequence that was never scored, and
    every downstream filter on these columns is numeric. The blank keeps the
    group's row, with n_seeds recording that two seeds were asked for.
    """
    wrong = _wrong_interface_pose(GT, tmp_path / "wrong.pdb")
    work = tmp_path / "work"
    fake, _ = _stub_boltz(tmp_path / "scratch", wrong)
    assert _engine(fake, "plan", "--ground-truth", str(GT),
                   "--workdir", str(work), "--receptor", "A", "--effector", "B",
                   "--boltz-container", "/fake.img", "--n-designs", "1",
                   "--num-seeds", "2", "--diffusion-samples", "1",
                   "--recycling-steps", "1") == 0

    designs = _plan_of(work)["designs"]
    assert len(designs) == 2, f"expected one sequence over two seeds, got {designs}"
    for d in designs:
        assert not (Path(d["dir"]) / "result.json").exists()

    assert _engine(fake, "collect", "--workdir", str(work)) == 0
    with (work / "steered_results_aggregate.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    row = rows[0]
    assert row["n_seeds"] == "2"
    for key in ("receptor_aligned_effector_rmsd", "independent_receptor_rmsd",
                "independent_effector_rmsd"):
        for suffix in ("_mean", "_std", "_min", "_max"):
            assert row[f"{key}{suffix}"] == "", \
                f"{key}{suffix} carries a number for a group with no seeds"
    assert row["n_intact_seeds"] == "0"
    assert row["any_seed_intact"] == "0"
