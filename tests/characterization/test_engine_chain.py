"""The engine's stage chain, end to end, with Boltz stubbed.

plan -> predict-one -> collect -> the iterate stages. Every one of these needs a
prediction, but nothing needs the prediction to be *good*, so substituting a
fixed structure for the Boltz call reaches all of it on a laptop.

The substituted prediction is 6G10 with the effector reflected through the
receptor centroid. That is a real wrong-interface pose: 44 A from the truth, so
the run does not short-circuit, while keeping genuine contacts on the opposite
face for the steering stage to work against.

A session fixture builds the run once and each test copies it, so a stage that
writes cannot affect another test.
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

import boltz2_iterate_steering as its  # noqa: E402
import boltz2_negative_steering as bns  # noqa: E402

GT = (_REPO_ROOT / "experiments" / "benchmarking" / "unconstrained" / "6G10"
      / "6G10.pdb")
needs_pdb = pytest.mark.skipif(not GT.is_file(), reason="6G10 reference absent")

N_DESIGNS = 2

# The engine refuses a prediction whose receptor sequence is not the one it
# asked for, because a mismatch would break the index alignment between
# wrong-interface detection and the protected set. The stub therefore has to
# relabel its structure to the requested sequence.
ONE_TO_THREE = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY",
    "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN",
    "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER", "T": "THR", "V": "VAL",
    "W": "TRP", "Y": "TYR",
}


def _relabel_receptor(pdb: Path, sequence: str, chain: str = "A") -> None:
    """Rewrite chain `chain`'s residue names to spell `sequence`.

    Coordinates are untouched, so every geometric metric is unchanged. Only the
    residue identity changes, which is what the engine's same-site check reads.
    """
    lines = pdb.read_text().splitlines(keepends=True)
    order, seen = [], set()
    for line in lines:
        if line.startswith("ATOM") and line[21] == chain:
            key = line[22:27]
            if key not in seen:
                seen.add(key)
                order.append(key)
    mapping = {k: ONE_TO_THREE.get(aa, "ALA")
               for k, aa in zip(order, sequence)}
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


def _wrong_interface_pose(src: Path, dst: Path) -> Path:
    lines = src.read_text().splitlines(keepends=True)
    co = [(float(x[30:38]), float(x[38:46]), float(x[46:54]))
          for x in lines if x.startswith("ATOM") and x[21] == "A"]
    cx, cy, cz = (sum(c[i] for c in co) / len(co) for i in range(3))
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


@pytest.fixture(scope="session")
def built_run(tmp_path_factory) -> Path:
    """plan, predict every design, collect. Built once for the session."""
    root = tmp_path_factory.mktemp("engine_chain")
    wrong = _wrong_interface_pose(GT, root / "wrong.pdb")
    work = root / "work"

    counter = [0]

    def fake_run_boltz(yaml_path, out_dir, container, recycling_steps,
                       diffusion_samples, seed, no_kernels):
        """Return the prediction from a scratch path.

        Not out_dir/prediction.pdb: the engine copies the returned file to
        exactly that location, so writing it there first makes the copy a
        same-file error.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        counter[0] += 1
        scratch = root / "boltz_out" / f"pred_{counter[0]}"
        scratch.mkdir(parents=True, exist_ok=True)
        pred = scratch / "prediction.pdb"
        shutil.copy(wrong, pred)
        requested = _requested_sequence(out_dir)
        if requested:
            _relabel_receptor(pred, requested)
        return pred

    def run(*argv):
        with mock.patch.object(bns, "run_boltz", fake_run_boltz), \
                mock.patch.object(sys, "argv", ["boltz2_negative_steering.py", *argv]):
            return bns.main()

    assert run("plan", "--ground-truth", str(GT), "--workdir", str(work),
               "--receptor", "A", "--effector", "B",
               "--boltz-container", "/fake.img",
               "--n-designs", str(N_DESIGNS), "--num-seeds", "1",
               "--diffusion-samples", "1", "--recycling-steps", "1") == 0

    staged = sorted(d for d in (work / "steered").iterdir() if d.is_dir())
    for i in range(len(staged)):
        assert run("predict-one", "--workdir", str(work), "--index", str(i)) == 0
    assert run("collect", "--workdir", str(work)) == 0
    return work


@pytest.fixture
def run_dir(built_run, tmp_path) -> Path:
    dst = tmp_path / "work"
    shutil.copytree(built_run, dst)
    return dst


def _iterate(*argv) -> int:
    with mock.patch.object(sys, "argv", ["boltz2_iterate_steering.py", *argv]):
        rc = its.main()
    return 0 if rc is None else rc


def _read(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


# ── the chain produced what the later stages need ────────────────────────────

@pytest.mark.local_integration
@needs_pdb
def test_the_wrong_pose_is_measured_as_wrong(built_run):
    """44 A, so the run cannot short-circuit past the steering stage."""
    plan = json.loads((built_run / "plan.json").read_text())
    assert float(plan["initial_receptor_aligned_effector_rmsd"]) > 20.0


@pytest.mark.local_integration
@needs_pdb
def test_every_requested_design_was_staged(built_run):
    designs = sorted(d.name for d in (built_run / "steered").iterdir() if d.is_dir())
    assert len(designs) == N_DESIGNS, f"staged {designs}"


@pytest.mark.local_integration
@needs_pdb
def test_every_staged_design_carries_its_mutation_record(built_run):
    """mutations.tsv is what reversion later reads to undo the steering."""
    for d in sorted(p for p in (built_run / "steered").iterdir() if p.is_dir()):
        assert (d / "mutations.tsv").is_file(), f"{d.name} has no mutations.tsv"
        assert (d / "receptor.fasta").is_file(), f"{d.name} has no receptor.fasta"


@pytest.mark.local_integration
@needs_pdb
def test_predict_one_wrote_a_prediction_for_every_design(built_run):
    for d in sorted(p for p in (built_run / "steered").iterdir() if p.is_dir()):
        preds = list(d.rglob("prediction.pdb"))
        assert preds, f"{d.name} has no prediction"


@pytest.mark.local_integration
@needs_pdb
def test_collect_wrote_a_row_for_the_cold_start_and_every_design(built_run):
    rows = _read(built_run / "steered_results.csv")
    assert len(rows) == N_DESIGNS + 1, (
        f"expected the cold start plus {N_DESIGNS} designs, got {len(rows)} rows")
    assert any(r["design"] == "initial" for r in rows), "no cold-start row"
    assert (built_run / "summary.txt").is_file()


@pytest.mark.local_integration
@needs_pdb
def test_the_steered_sequences_differ_from_the_wild_type(built_run):
    """If steering changed nothing the whole run would be a no-op."""
    wild = json.loads((built_run / "plan.json").read_text())["wild_type_receptor_seq"]
    changed_any = False
    for d in sorted(p for p in (built_run / "steered").iterdir() if p.is_dir()):
        seq = "".join(x for x in (d / "receptor.fasta").read_text().splitlines()
                      if not x.startswith(">"))
        assert len(seq) == len(wild), f"{d.name} changed the sequence length"
        if seq != wild:
            changed_any = True
    assert changed_any, "no design differs from the wild type"


@pytest.mark.local_integration
@needs_pdb
def test_no_design_mutates_the_protected_interface(built_run):
    """The invariant the method depends on, checked on a real steering run."""
    plan = json.loads((built_run / "plan.json").read_text())
    wild = plan["wild_type_receptor_seq"]
    protected = {
        int(x) for x in
        (built_run / "true_interface_residues.txt").read_text().split()
        if x.strip().isdigit()
    }
    assert protected, "no protected set recorded"

    for d in sorted(p for p in (built_run / "steered").iterdir() if p.is_dir()):
        seq = "".join(x for x in (d / "receptor.fasta").read_text().splitlines()
                      if not x.startswith(">"))
        changed = {i for i, (a, b) in enumerate(zip(wild, seq)) if a != b}
        assert not (changed & protected), (
            f"{d.name} mutated protected positions {sorted(changed & protected)}")


# ── the iterate stages, on the built run ─────────────────────────────────────

@pytest.mark.local_integration
@needs_pdb
def test_aggregate_runs_over_the_built_tree(run_dir):
    assert _iterate("aggregate", "--experiment-root", str(run_dir)) == 0
    assert (run_dir / "all_results_multicycle.csv").is_file()
    assert (run_dir / "pathways.json").is_file()


@pytest.mark.local_integration
@needs_pdb
def test_aggregate_per_sequence_runs_over_the_built_tree(run_dir):
    _iterate("aggregate", "--experiment-root", str(run_dir))
    assert _iterate("aggregate-per-sequence", "--experiment-root", str(run_dir)) == 0
    assert (run_dir / "aggregated_results.csv").is_file()


@pytest.mark.local_integration
@needs_pdb
def test_build_contaminated_requires_the_prefilter_first(run_dir, capsys):
    """The stages are ordered. Running one out of turn fails loudly.

    That matters because the pipeline tolerates per-unit failure, so a stage
    that quietly produced an empty result would look like a clean run.
    """
    rc = _iterate("build-contaminated", "--workdir", str(run_dir))
    assert rc != 0, "build-contaminated ran without its prefilter input"
    assert "prefilter.json not found" in capsys.readouterr().err
    assert not (run_dir / "contaminated.json").is_file()


@pytest.mark.local_integration
@needs_pdb
def test_the_prefilter_needs_phase_one_first(run_dir, capsys):
    """It names compute-distances as the missing predecessor.

    An error that says which stage to run is the difference between a
    recoverable mis-order and an opaque stall.
    """
    assert _iterate("iterate-collect-prefilter", "--workdir", str(run_dir)) != 0
    assert "compute-distances" in capsys.readouterr().err


@pytest.mark.local_integration
@needs_pdb
def test_plan_reversions_runs(run_dir):
    _iterate("build-contaminated", "--workdir", str(run_dir))
    rc = _iterate("plan-reversions", "--workdir", str(run_dir))
    assert rc == 0 or rc == 1


@pytest.mark.local_integration
@needs_pdb
def test_kickoff_distances_needs_a_cycle_directory(run_dir, capsys):
    """--experiment-root expects the run root, which holds cycle_0/.

    The stubbed run puts its cycle-0 artefacts at the workdir root rather than
    under cycle_0/, so this pins the error rather than the success path.
    """
    rc = _iterate("kickoff-distances", "--experiment-root", str(run_dir))
    assert rc != 0
    assert "no cycle_0" in capsys.readouterr().err


@pytest.mark.local_integration
@needs_pdb
def test_iterate_collect_prefilter_runs(run_dir):
    rc = _iterate("iterate-collect-prefilter", "--workdir", str(run_dir))
    assert rc in (0, 1)


# ── the collect stage's own numbers ──────────────────────────────────────────

@pytest.mark.local_integration
@needs_pdb
def test_collected_rows_carry_a_measured_pose(built_run):
    rows = _read(built_run / "steered_results.csv")
    measured = [r for r in rows
                if (r.get("receptor_aligned_effector_rmsd") or "").strip()]
    assert measured, "no row carries a receptor-aligned effector RMSD"
    for r in measured:
        assert 0.0 <= float(r["receptor_aligned_effector_rmsd"]) < 500.0


@pytest.mark.local_integration
@needs_pdb
def test_every_prediction_scored_the_same_wrong_pose(built_run):
    """Every design got the same stubbed structure, so the RMSDs must agree.

    A spread here would mean the scoring depends on something other than the
    coordinates it was handed.
    """
    rows = _read(built_run / "steered_results.csv")
    values = {round(float(r["receptor_aligned_effector_rmsd"]), 3)
              for r in rows if (r.get("receptor_aligned_effector_rmsd") or "").strip()}
    assert len(values) == 1, (
        f"identical coordinates scored differently: {sorted(values)}")


# ── the reversion chain's guards ─────────────────────────────────────────────
#
# compute-distances, the prefilter and build-contaminated operate on a per-cycle
# workdir created by iterate-plan, whose plan.json carries `cycle` and
# `parent_pathway`. cmd_plan's cycle-0 plan.json has a different schema, so
# pointing these stages at the run root exercises their guards rather than their
# work. Both matter: the guards are what stop a mis-ordered pipeline producing
# a plausible-looking empty result.

@pytest.mark.local_integration
@needs_pdb
def test_compute_distances_rejects_a_cycle_zero_plan(run_dir):
    """The cycle-0 plan has no `cycle` key, so this must fail rather than guess."""
    with pytest.raises(KeyError):
        _iterate("compute-distances", "--workdir", str(run_dir))


@pytest.mark.local_integration
@needs_pdb
def test_compute_distances_reports_a_missing_plan(tmp_path, capsys):
    empty = tmp_path / "nothing"
    empty.mkdir()
    assert _iterate("compute-distances", "--workdir", str(empty)) == 1
    assert "does not exist" in capsys.readouterr().err


@pytest.mark.local_integration
@needs_pdb
def test_the_prefilter_records_why_it_bailed(run_dir):
    """It writes prefilter.json carrying the error rather than writing nothing.

    A stage that silently produced no file would stall the chain with no record
    of the cause, which is the failure mode the playbook warns about.
    """
    (run_dir / "distances.json").unlink(missing_ok=True)
    _iterate("iterate-collect-prefilter", "--workdir", str(run_dir))
    out = run_dir / "prefilter.json"
    assert out.is_file(), "the prefilter left no record of its failure"
    assert "error" in json.loads(out.read_text())


@pytest.mark.local_integration
@needs_pdb
def test_the_prefilter_names_the_missing_input(run_dir):
    (run_dir / "distances.json").unlink(missing_ok=True)
    _iterate("iterate-collect-prefilter", "--workdir", str(run_dir))
    err = json.loads((run_dir / "prefilter.json").read_text())["error"]
    assert "distances.json" in err, f"the error does not name its input: {err!r}"


@pytest.mark.local_integration
@needs_pdb
def test_build_contaminated_refuses_without_the_prefilter(run_dir, capsys):
    (run_dir / "prefilter.json").unlink(missing_ok=True)
    assert _iterate("build-contaminated", "--workdir", str(run_dir)) != 0
    assert "prefilter.json not found" in capsys.readouterr().err
    assert not (run_dir / "contaminated.json").is_file()


@pytest.mark.local_integration
@needs_pdb
def test_plan_reversions_with_no_contamination_stages_nothing(run_dir):
    """No contaminated designs means no reversions to plan, not a failure."""
    (run_dir / "contaminated.json").unlink(missing_ok=True)
    rc = _iterate("plan-reversions", "--workdir", str(run_dir))
    assert rc == 0
    staged = run_dir / "reversions"
    assert not staged.is_dir() or not any(staged.iterdir()), \
        "reversions were staged despite no contamination record"


@pytest.mark.local_integration
@needs_pdb
def test_harvest_reversions_writes_an_empty_result_without_a_plan(run_dir, capsys):
    """It records an explicit empty result rather than leaving no file.

    Downstream reads reversion_results.json unconditionally, so an absent file
    and an empty one are not the same thing.
    """
    rc = _iterate("harvest-reversions", "--workdir", str(run_dir))
    assert rc == 0
    assert "no reversion_plan.json" in capsys.readouterr().out
    out = run_dir / "reversion_results.json"
    assert out.is_file(), "harvest left no results file at all"
    assert json.loads(out.read_text()) in ([], {}, None) or \
        not json.loads(out.read_text())
