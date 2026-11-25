"""The engine's branches that a happy-path run never reaches.

boltz2_negative_steering is one file holding the whole steering engine. The
stage chain is covered elsewhere (test_plan_stage, test_engine_chain), but a
run that works never touches the parser fallbacks, the Boltz output-discovery
logic, the second-shell candidate pool or the cold-start-only CSV. Those are
exactly the paths that decide what a difficult complex reports, so leaving them
unexercised means the awkward cases are the untested ones.

Nothing here needs a GPU. The only stubbed edge is the Boltz invocation itself,
and it is stubbed as a real executable on PATH so the module's own
subprocess.run, argument assembly and output globbing all still run.

Every expected value is derived from the geometry or the formula stated in the
docstring, not recorded from a run.
"""

from __future__ import annotations

import csv
import os
import random
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin"))

import boltz2_negative_steering as bns  # noqa: E402

GT = _REPO_ROOT / "data" / "reference_complexes" / "6G10.pdb"
needs_pdb = pytest.mark.skipif(not GT.is_file(), reason="6G10 reference absent")

REC_LEN, EFF_LEN = 76, 83


# ── small PDB builders ───────────────────────────────────────────────────────
# Hand-built PDB text, so the correct answer is known before the reader runs.
# Columns follow the PDB spec exactly, including the element column at 77-78,
# because both readers in this module branch on it.

def _atom(serial, name, resname, chain, resnum, x, y, z, alt=" ", element="",
          record="ATOM  "):
    return (f"{record}{serial:5d} {name:<4s}{alt}{resname:>3s} {chain}{resnum:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n")


def _write_pdb(path: Path, lines) -> Path:
    path.write_text("".join(lines))
    return path


def _residue(chain, seq_index, atoms, resnum=None, resname="ALA", one="A"):
    return bns.ResidueAtoms(
        chain=chain,
        seq_index=seq_index,
        resnum=seq_index + 1 if resnum is None else resnum,
        resname=resname,
        one=one,
        atoms={k: np.asarray(v, dtype=float) for k, v in atoms.items()},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — sequence extraction, and the agreement between the two parsers
# ─────────────────────────────────────────────────────────────────────────────
# extract_sequences tries gemmi and falls back to BioPython. Which one runs
# depends on what imported in the container, so the two must return the same
# chains for the same file. If they disagreed, the receptor length and
# therefore every positional index in the plan would depend on the image the
# job happened to land in.

@pytest.mark.logic
@needs_pdb
def test_biopython_reads_the_reference_complex_chains():
    """6G10 is receptor A of 76 residues and effector B of 83."""
    chains = bns.extract_sequences_biopython(str(GT))
    assert [(c["id"], c["length"]) for c in chains] == [("A", REC_LEN), ("B", EFF_LEN)]
    assert all(len(c["sequence"]) == c["length"] for c in chains)


@pytest.mark.logic
@needs_pdb
def test_the_two_parsers_return_identical_sequences():
    """gemmi and BioPython must not disagree, because which one runs is chance.

    The plan's protected set, candidate pool and surface flags are all 0-based
    positions into the receptor sequence. A one-residue difference between the
    parsers would shift every one of them, and the run would still complete.
    """
    by_gemmi = bns.extract_sequences_gemmi(str(GT))
    by_bio = bns.extract_sequences_biopython(str(GT))
    assert [c["id"] for c in by_gemmi] == [c["id"] for c in by_bio]
    for g, b in zip(by_gemmi, by_bio):
        assert g["sequence"] == b["sequence"], f"chain {g['id']} differs between parsers"


@pytest.mark.logic
def test_biopython_skips_hetero_residues(tmp_path):
    """Waters and ligands share a chain ID with the protein in many PDBs.

    Counting them would inflate the receptor length and shift every index.
    """
    pdb = _write_pdb(tmp_path / "het.pdb", [
        _atom(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, element="C"),
        _atom(2, "CA", "GLY", "A", 2, 3.8, 0.0, 0.0, element="C"),
        _atom(3, "CA", "SER", "A", 3, 7.6, 0.0, 0.0, element="C"),
        _atom(4, "O", "HOH", "A", 4, 9.0, 2.0, 0.0, element="O", record="HETATM"),
    ])
    chains = bns.extract_sequences_biopython(str(pdb))
    assert [(c["id"], c["sequence"]) for c in chains] == [("A", "AGS")]


@pytest.mark.logic
def test_an_unknown_residue_name_becomes_x(tmp_path):
    """An unrecognised three-letter code must hold its position, not vanish.

    Dropping it would shorten the receptor and shift every index after it.
    Selenomethionine is in the table and maps to M, so the unknown case needs
    a code that genuinely is not.
    """
    assert "XYZ" not in bns.THREE_TO_ONE
    pdb = _write_pdb(tmp_path / "unk.pdb", [
        _atom(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, element="C"),
        _atom(2, "CA", "XYZ", "A", 2, 3.8, 0.0, 0.0, element="C"),
        _atom(3, "CA", "GLY", "A", 3, 7.6, 0.0, 0.0, element="C"),
    ])
    assert bns.extract_sequences_biopython(str(pdb))[0]["sequence"] == "AXG"
    assert bns.THREE_TO_ONE["MSE"] == "M"


@pytest.mark.logic
@needs_pdb
def test_the_mmcif_parser_is_chosen_by_the_file_extension(tmp_path):
    """A .cif template has to go through MMCIFParser, not PDBParser."""
    cif = bns.extract_effector_template_cif(GT, "B", "C", tmp_path / "eff.cif")
    chains = bns.extract_sequences_biopython(str(cif))
    assert [(c["id"], c["length"]) for c in chains] == [("C", EFF_LEN)]


@pytest.mark.logic
@needs_pdb
def test_extract_sequences_falls_back_to_biopython_without_gemmi(monkeypatch):
    """The cluster image may not carry gemmi. The answer must not change.

    Blocking the import with a None entry in sys.modules is how Python itself
    signals an unimportable module, so the fallback runs for the real reason.
    """
    monkeypatch.setitem(sys.modules, "gemmi", None)
    chains = bns.extract_sequences(str(GT))
    assert [(c["id"], c["length"]) for c in chains] == [("A", REC_LEN), ("B", EFF_LEN)]


@pytest.mark.logic
@needs_pdb
def test_extract_sequences_raises_when_neither_parser_is_available(monkeypatch):
    """With no parser the run must stop, not plan against an empty receptor."""
    monkeypatch.setitem(sys.modules, "gemmi", None)
    monkeypatch.setitem(sys.modules, "Bio.PDB", None)
    with pytest.raises(RuntimeError, match="Neither gemmi nor BioPython"):
        bns.extract_sequences(str(GT))


@pytest.mark.logic
def test_extract_sequences_raises_on_a_file_that_is_not_a_structure(tmp_path):
    """A truncated or wrong-format ground truth must fail loudly at plan time.

    Both parsers read this without raising and return no chains, so the empty
    result rather than an exception is what has to be caught.
    """
    junk = tmp_path / "junk.pdb"
    junk.write_text("not a structure at all\n")
    with pytest.raises(RuntimeError, match="Neither gemmi nor BioPython"):
        bns.extract_sequences(str(junk))


@pytest.mark.logic
def test_both_parsers_failing_outright_is_reported_as_one_error(tmp_path, capsys):
    """A missing file makes both parsers raise, and both failures are named.

    The two warnings are the only record of why the ground truth could not be
    read. The single RuntimeError is what the plan stage turns into exit 1.
    """
    absent = tmp_path / "absent.pdb"
    with pytest.raises(RuntimeError, match="Neither gemmi nor BioPython"):
        bns.extract_sequences(str(absent))
    err = capsys.readouterr().err
    assert "gemmi failed" in err
    assert "BioPython failed" in err


@pytest.mark.logic
@needs_pdb
def test_get_chain_sequence_names_the_chains_it_did_find():
    """A wrong --receptor letter is the commonest operator error here."""
    with pytest.raises(ValueError, match=r"Available chains: \['A', 'B'\]"):
        bns.get_chain_sequence(GT, "Z")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — the effector structural template
# ─────────────────────────────────────────────────────────────────────────────
# Every prediction in a run reuses one single-chain mmCIF of the effector so
# the effector fold is pinned and only the receptor sequence varies. The
# template has to be relabelled to the prediction's chain ID, because Boltz is
# always handed chains A and B whatever the ground truth calls them.

@pytest.mark.logic
@needs_pdb
def test_the_template_chain_is_relabelled_to_the_prediction_chain(tmp_path):
    """Truth chain B becomes prediction chain C here, atoms unchanged.

    A template labelled with the truth chain ID would not match any chain in
    the Boltz input, and the effector fold would go unconstrained.
    """
    cif = bns.extract_effector_template_cif(GT, "B", "C", tmp_path / "eff.cif")
    chains = bns.extract_sequences_gemmi(str(cif))
    assert [c["id"] for c in chains] == ["C"]
    assert chains[0]["length"] == EFF_LEN
    truth_eff = bns.get_chain_sequence(GT, "B")
    assert chains[0]["sequence"] == truth_eff, "relabelling altered the sequence"


@pytest.mark.logic
@needs_pdb
def test_a_missing_effector_chain_is_caught_before_the_cif_is_written(tmp_path):
    out = tmp_path / "eff.cif"
    with pytest.raises(ValueError, match="chains present: \\['A', 'B'\\]"):
        bns.extract_effector_template_cif(GT, "Z", "B", out)
    assert not out.exists(), "a malformed template was left on disk"


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — run_boltz: command assembly and output discovery
# ─────────────────────────────────────────────────────────────────────────────
# Boltz's output layout has changed across versions, so run_boltz globs the
# whole out_dir for *.pdb rather than tracking the layouts. The rules it
# applies decide which of several diffusion samples is scored for the rest of
# the run, so they are worth pinning. The Boltz call itself is replaced by a
# stub executable on PATH, which keeps the module's own subprocess.run,
# argument list and globbing in the loop.

_STUB_SOURCE = '''
import os
import sys

argv = sys.argv[1:]
with open(os.environ["STUB_ARGV"], "w") as fh:
    fh.write("\\n".join(argv))

out_dir = argv[argv.index("--out_dir") + 1]
for rel in os.environ.get("STUB_FILES", "").split("|"):
    if not rel:
        continue
    target = os.path.join(out_dir, rel)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w") as fh:
        fh.write("ATOM      1  CA  ALA A   1       0.000   0.000   0.000\\n")

sys.stdout.write("stub boltz stdout\\n")
sys.stderr.write("stub boltz stderr\\n")
sys.exit(int(os.environ.get("STUB_EXIT", "0")))
'''


def _install_stub(tmp_path, monkeypatch, name, *, files=(), exit_code=0):
    """Put an executable called `name` first on PATH and return its argv log.

    The shim is a two-line shell script that re-executes this interpreter on a
    small Python stub, so no assumption is made about what is installed.
    """
    bindir = tmp_path / "stubbin"
    bindir.mkdir(exist_ok=True)
    stub_py = bindir / "stub_boltz.py"
    stub_py.write_text(_STUB_SOURCE)
    shim = bindir / name
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{stub_py}" "$@"\n')
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    argv_log = tmp_path / f"{name}_argv.txt"
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("STUB_ARGV", str(argv_log))
    monkeypatch.setenv("STUB_FILES", "|".join(files))
    monkeypatch.setenv("STUB_EXIT", str(exit_code))
    return argv_log


def _outside_container(monkeypatch):
    for var in ("SINGULARITY_NAME", "SINGULARITY_CONTAINER",
                "APPTAINER_NAME", "APPTAINER_CONTAINER"):
        monkeypatch.delenv(var, raising=False)


def _call_run_boltz(tmp_path, out_dir, *, no_kernels=False, container="/img.sif"):
    yaml_path = tmp_path / "input" / "input.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("version: 1\n")
    return bns.run_boltz(
        yaml_path, out_dir, container,
        recycling_steps=3, diffusion_samples=1, seed=7, no_kernels=no_kernels,
    )


@pytest.mark.logic
def test_outside_a_container_the_call_is_wrapped_in_singularity(tmp_path, monkeypatch):
    """The wrapper must request the GPU and bind both the input and output dirs.

    Without --nv the job runs on CPU and takes days. Without the binds Boltz
    cannot see the YAML it was told to predict.
    """
    _outside_container(monkeypatch)
    argv_log = _install_stub(tmp_path, monkeypatch, "singularity",
                             files=("predictions/x_model_0.pdb",))
    out_dir = tmp_path / "out"
    got = _call_run_boltz(tmp_path, out_dir, container="/img.sif")

    argv = argv_log.read_text().splitlines()
    assert argv[:3] == ["exec", "--nv", "--bind"]
    assert "/img.sif" in argv
    assert str(out_dir.resolve()) in " ".join(argv)
    assert str((tmp_path / "input").resolve()) in " ".join(argv)
    # The container path separates the singularity flags from the boltz call.
    assert argv[argv.index("/img.sif") + 1:argv.index("/img.sif") + 3] == ["boltz", "predict"]
    assert got == out_dir / "predictions" / "x_model_0.pdb"


@pytest.mark.logic
def test_inside_a_container_boltz_is_called_directly(tmp_path, monkeypatch):
    """Nested singularity exec is unsupported, so the wrapper must be dropped.

    The submit script already runs this module inside the benchmark image, so
    the in-container branch is the one the cluster actually takes.
    """
    monkeypatch.setenv("SINGULARITY_NAME", "benchmark.sif")
    argv_log = _install_stub(tmp_path, monkeypatch, "boltz",
                             files=("predictions/x_model_0.pdb",))
    out_dir = tmp_path / "out"
    got = _call_run_boltz(tmp_path, out_dir)

    argv = argv_log.read_text().splitlines()
    assert "exec" not in argv and "--nv" not in argv
    assert argv[0] == "predict"
    assert got.name == "x_model_0.pdb"


@pytest.mark.logic
def test_the_boltz_flags_are_the_ones_the_experiment_depends_on(tmp_path, monkeypatch):
    """--use_potentials is what makes the effector template's force:true bite.

    --output_format pdb is what the readers in this module expect, and
    --write_full_pae is what compute_metrics needs for ipSAE. Losing any of
    them produces a run that finishes and cannot be scored.
    """
    monkeypatch.setenv("SINGULARITY_NAME", "benchmark.sif")
    argv_log = _install_stub(tmp_path, monkeypatch, "boltz",
                             files=("x_model_0.pdb",))
    _call_run_boltz(tmp_path, tmp_path / "out")
    argv = argv_log.read_text().splitlines()
    for flag in ("--use_potentials", "--write_full_pae", "--override"):
        assert flag in argv, f"{flag} was dropped"
    assert argv[argv.index("--output_format") + 1] == "pdb"
    assert argv[argv.index("--seed") + 1] == "7"
    assert argv[argv.index("--recycling_steps") + 1] == "3"
    assert argv[argv.index("--num_workers") + 1] == "0"
    assert "--no_kernels" not in argv


@pytest.mark.logic
def test_no_kernels_is_forwarded_when_asked_for(tmp_path, monkeypatch):
    """The flag exists for GPUs whose compute capability the kernels reject."""
    monkeypatch.setenv("SINGULARITY_NAME", "benchmark.sif")
    argv_log = _install_stub(tmp_path, monkeypatch, "boltz", files=("x_model_0.pdb",))
    _call_run_boltz(tmp_path, tmp_path / "out", no_kernels=True)
    assert "--no_kernels" in argv_log.read_text().splitlines()


@pytest.mark.logic
def test_run_boltz_creates_the_output_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("SINGULARITY_NAME", "benchmark.sif")
    _install_stub(tmp_path, monkeypatch, "boltz", files=("x_model_0.pdb",))
    out_dir = tmp_path / "deep" / "nested" / "out"
    assert not out_dir.exists()
    _call_run_boltz(tmp_path, out_dir)
    assert out_dir.is_dir()


@pytest.mark.logic
@pytest.mark.parametrize("written,expected", [
    # model_0 wins over a higher-numbered sample.
    (("predictions/x_model_1.pdb", "predictions/x_model_0.pdb"), "x_model_0.pdb"),
    # rank_0 is the other naming Boltz has used for the top sample.
    (("predictions/x_rank_2.pdb", "predictions/x_rank_0.pdb"), "x_rank_0.pdb"),
    # With no rank marker at all, fall back to name order.
    (("predictions/zz.pdb", "predictions/aa.pdb"), "aa.pdb"),
])
def test_the_top_ranked_sample_is_the_one_returned(tmp_path, monkeypatch,
                                                   written, expected):
    """Whichever PDB comes back is the only one scored for the whole run.

    Returning sample 1 instead of sample 0 would silently report Boltz's
    second-best pose as the design's result.
    """
    monkeypatch.setenv("SINGULARITY_NAME", "benchmark.sif")
    _install_stub(tmp_path, monkeypatch, "boltz", files=written)
    got = _call_run_boltz(tmp_path, tmp_path / "out")
    assert got.name == expected


@pytest.mark.logic
def test_a_pdb_cached_under_msa_is_not_mistaken_for_a_prediction(tmp_path, monkeypatch):
    """Some Boltz versions cache reference structures under msa/.

    Alphabetically msa/ sorts before predictions/, so without the filter the
    cached reference would be scored as the prediction and every design would
    look identical to its template.
    """
    monkeypatch.setenv("SINGULARITY_NAME", "benchmark.sif")
    _install_stub(tmp_path, monkeypatch, "boltz",
                  files=("msa/aaa_reference.pdb", "predictions/x_model_0.pdb"))
    got = _call_run_boltz(tmp_path, tmp_path / "out")
    assert got.name == "x_model_0.pdb"
    assert "msa" not in got.parts


@pytest.mark.logic
def test_only_an_msa_pdb_counts_as_no_output(tmp_path, monkeypatch):
    monkeypatch.setenv("SINGULARITY_NAME", "benchmark.sif")
    _install_stub(tmp_path, monkeypatch, "boltz", files=("msa/aaa_reference.pdb",))
    with pytest.raises(RuntimeError, match="No PDB output found"):
        _call_run_boltz(tmp_path, tmp_path / "out")


@pytest.mark.logic
def test_a_nonzero_exit_raises_and_reports_the_code(tmp_path, monkeypatch, capsys):
    """A failed prediction must not be read as a prediction with no output.

    The two conditions have different causes, and the exit code is the only
    thing that distinguishes them in the job log.
    """
    monkeypatch.setenv("SINGULARITY_NAME", "benchmark.sif")
    _install_stub(tmp_path, monkeypatch, "boltz", exit_code=3)
    with pytest.raises(RuntimeError, match=r"exit 3"):
        _call_run_boltz(tmp_path, tmp_path / "out")
    err = capsys.readouterr().err
    assert "stub boltz stderr" in err, "Boltz's own stderr was swallowed"


@pytest.mark.logic
def test_a_successful_run_with_no_pdb_dumps_the_directory_tree(tmp_path, monkeypatch,
                                                               capsys):
    """Boltz reporting success with no PDB is a layout change, not a crash.

    The tree is the only evidence of where the output actually went, and it is
    printed once per failure rather than requiring the run to be repeated.
    """
    monkeypatch.setenv("SINGULARITY_NAME", "benchmark.sif")
    _install_stub(tmp_path, monkeypatch, "boltz",
                  files=("predictions/x.cif", "predictions/confidence.json"))
    out_dir = tmp_path / "out"
    with pytest.raises(RuntimeError, match="after Boltz reported success"):
        _call_run_boltz(tmp_path, out_dir)
    err = capsys.readouterr().err
    assert "x.cif" in err and "confidence.json" in err, "the tree was not dumped"
    assert "stub boltz stdout" in err


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — the second-shell candidate pool
# ─────────────────────────────────────────────────────────────────────────────
# When the wrong interface sits entirely inside the protected true site there
# is nothing to mutate, and the plan stage falls back to this pool. It scores
# every other receptor residue by the shortest distance from its Cα to any
# effector heavy atom, keeps those within the radius and surface-exposed, and
# returns nothing below three candidates rather than padding the pool with
# residues too far away to perturb binding.
#
# The geometry here is a line of receptor Cαs on the x axis with one effector
# atom at the origin, so each residue's score is just its x coordinate.

_CENTROID_X = 3.25   # mean of x = 1.0, 1.5, ... 5.5


def _line_receptor(n=10, start=1.0, step=0.5, with_cb=True):
    """n receptor residues with Cα at x = start + i*step, y = z = 0.

    Each Cβ points directly away from the chain's own centroid, which is what
    surface exposure means, so every residue passes the Cβ-vector test.
    """
    residues = []
    for i in range(n):
        x = start + step * i
        atoms = {"CA": (x, 0.0, 0.0)}
        if with_cb:
            direction = -1.5 if x < _CENTROID_X else 1.5
            atoms["CB"] = (x + direction, 0.0, 0.0)
        residues.append(_residue("A", i, atoms))
    return residues


def _point_effector():
    """One effector residue with a single heavy atom at the origin."""
    return [_residue("B", 0, {"CA": (0.0, 0.0, 0.0)}, resname="GLY", one="G")]


@pytest.mark.logic
def test_the_second_shell_pool_takes_the_closest_residues_to_the_effector():
    """Distances are 1.0, 1.5, ... 5.5 Å. Residues 0-1 are the wrong interface
    and residue 2 is protected, leaving 3-9 eligible. The pool size is 6, so
    the six closest survive and residue 9 at 5.5 Å is dropped.

    Ranking by distance to the effector rather than to the wrong interface is
    the whole point of this version. The earlier one could pick residues near
    the wrong interface in receptor space but nowhere near the effector.
    """
    cache = {i: True for i in range(10)}
    got = bns._second_shell_fallback(
        _line_receptor(), _point_effector(),
        predicted_wrong_idx=[0, 1], true_set={2},
        surface_flags_cache=cache, radius=6.0, pool_size=6, min_pool_size=3,
    )
    assert got == [3, 4, 5, 6, 7, 8]


@pytest.mark.logic
@pytest.mark.parametrize("radius,expected", [
    (2.0, [0, 1, 2]),             # x = 1.0, 1.5, 2.0
    (3.0, [0, 1, 2, 3, 4]),       # x = 1.0 ... 3.0
])
def test_the_radius_is_the_reach_of_a_side_chain(radius, expected):
    """6 Å is the default because a bulky substitution can plausibly reach that
    far. A residue outside the radius cannot affect effector contacts however
    it is mutated, so it must not enter the pool."""
    cache = {i: True for i in range(10)}
    got = bns._second_shell_fallback(
        _line_receptor(), _point_effector(),
        predicted_wrong_idx=[], true_set=set(),
        surface_flags_cache=cache, radius=radius, pool_size=6, min_pool_size=3,
    )
    assert got == expected


@pytest.mark.logic
def test_buried_residues_are_dropped_from_the_second_shell_pool():
    """Residues 3-5 are marked buried, so the pool becomes 6-9.

    Mutating a buried residue changes the fold rather than the interface, which
    shows up as a broken receptor RMSD and wastes the design slot.
    """
    cache = {i: True for i in range(10)}
    cache.update({3: False, 4: False, 5: False})
    got = bns._second_shell_fallback(
        _line_receptor(), _point_effector(),
        predicted_wrong_idx=[0, 1], true_set={2},
        surface_flags_cache=cache, radius=6.0, pool_size=6, min_pool_size=3,
    )
    assert got == [6, 7, 8, 9]


@pytest.mark.logic
def test_a_residue_without_a_ca_cannot_be_scored():
    """Residue 5 keeps only its Cβ, so it has no distance and drops out.

    The remaining six eligible residues exactly fill the pool.
    """
    receptor = _line_receptor()
    receptor[5] = _residue("A", 5, {"CB": (3.5, 0.0, 0.0)})
    cache = {i: True for i in range(10)}
    got = bns._second_shell_fallback(
        receptor, _point_effector(),
        predicted_wrong_idx=[0, 1], true_set={2},
        surface_flags_cache=cache, radius=6.0, pool_size=6, min_pool_size=3,
    )
    assert got == [3, 4, 6, 7, 8, 9]


@pytest.mark.logic
def test_fewer_than_three_candidates_returns_nothing():
    """Residues 2-7 are protected and 0-1 are the wrong interface, so only 8
    and 9 remain. Two candidates is not a steering experiment.

    Returning them anyway would produce designs that cannot move the effector
    and a results table that looks like a genuine failure to steer. The empty
    list makes the plan skip steering and score the cold start instead, which
    is what happened on 7B1I.
    """
    cache = {i: True for i in range(10)}
    got = bns._second_shell_fallback(
        _line_receptor(), _point_effector(),
        predicted_wrong_idx=[0, 1], true_set={2, 3, 4, 5, 6, 7},
        surface_flags_cache=cache, radius=6.0, pool_size=6, min_pool_size=3,
    )
    assert got == []


@pytest.mark.logic
def test_an_effector_with_no_atoms_yields_no_candidates():
    """No effector geometry means no distance to score against."""
    cache = {i: True for i in range(10)}
    assert bns._second_shell_fallback(
        _line_receptor(), [], predicted_wrong_idx=[], true_set=set(),
        surface_flags_cache=cache, radius=6.0, pool_size=6, min_pool_size=3,
    ) == []
    assert bns._second_shell_fallback(
        _line_receptor(), [_residue("B", 0, {})],
        predicted_wrong_idx=[], true_set=set(),
        surface_flags_cache=cache, radius=6.0, pool_size=6, min_pool_size=3,
    ) == []


@pytest.mark.logic
def test_an_empty_cache_is_filled_by_computing_the_exposure():
    """The cache is an optimisation, not the source of truth.

    Every Cβ in this fixture points away from the chain centroid, so all the
    scored residues are exposed and the pool matches the cached-flag case.
    """
    cache: dict = {}
    got = bns._second_shell_fallback(
        _line_receptor(), _point_effector(),
        predicted_wrong_idx=[0, 1], true_set={2},
        surface_flags_cache=cache, radius=6.0, pool_size=6, min_pool_size=3,
    )
    assert got == [3, 4, 5, 6, 7, 8]
    assert cache == {i: True for i in range(3, 10)}, \
        "only the scored residues should be cached"


@pytest.mark.logic
def test_glycine_without_a_cb_is_never_a_second_shell_candidate():
    """No Cβ means no side-chain direction and nothing useful to substitute."""
    cache: dict = {}
    got = bns._second_shell_fallback(
        _line_receptor(with_cb=False), _point_effector(),
        predicted_wrong_idx=[0, 1], true_set={2},
        surface_flags_cache=cache, radius=6.0, pool_size=6, min_pool_size=3,
    )
    assert got == []
    assert set(cache.values()) == {False}


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — the cold-start-only results CSV
# ─────────────────────────────────────────────────────────────────────────────
# Every skip_steering exit writes this one row. It is the entire reported
# result for that workdir, so the aggregator, compute-final-metrics and
# compare_versions all read it. An empty file here produces zero rows in the
# comparison table with no error anywhere.

def _only_row(workdir: Path) -> dict:
    with open(workdir / "steered_results.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1, f"expected exactly one row, got {len(rows)}"
    return rows[0]


@pytest.mark.logic
def test_the_cold_start_row_carries_its_metrics_and_zero_mutations(tmp_path):
    """rank 0, design "initial", 0 mutations, and delta 0 against itself.

    Receptor and effector are both intact at 1.0 and 1.5 Å, well under the 5 Å
    cutoffs, so receptor_intact is 1.
    """
    bns._write_initial_only_csv(tmp_path, {}, 12.5, 1.0, 1.5)
    row = _only_row(tmp_path)
    assert row["rank"] == "0"
    assert row["design"] == "initial"
    assert row["total_mutations"] == "0"
    assert row["status"] == "initial"
    assert float(row["receptor_aligned_effector_rmsd"]) == pytest.approx(12.5)
    assert float(row["delta_receptor_aligned_effector_rmsd"]) == 0.0
    assert row["receptor_intact"] == "1"


@pytest.mark.logic
def test_a_broken_fold_marks_the_cold_start_as_not_intact(tmp_path):
    """9 Å receptor RMSD is past the 5 Å cutoff, so the fold did not survive."""
    bns._write_initial_only_csv(tmp_path, {}, 12.5, 9.0, 1.5)
    assert _only_row(tmp_path)["receptor_intact"] == "0"


@pytest.mark.logic
def test_a_nan_metric_is_written_blank_not_as_nan(tmp_path):
    """A prediction that never landed must not be read as a measured value.

    nan fails every comparison, so writing it into the numeric columns would
    drop the row out of downstream filters silently. The blank is explicit.
    """
    nan = float("nan")
    bns._write_initial_only_csv(tmp_path, {}, nan, nan, nan)
    row = _only_row(tmp_path)
    assert row["receptor_aligned_effector_rmsd"] == ""
    assert row["independent_receptor_rmsd"] == ""
    assert row["independent_effector_rmsd"] == ""
    assert row["receptor_intact"] == "0"


@pytest.mark.logic
def test_without_a_truth_interface_the_jaccard_columns_are_blank(tmp_path):
    """The --skip-steering diagnostic path runs no interface analysis at all.

    Blank is the honest value there. A 0 would be read downstream as a measured
    complete mismatch.
    """
    bns._write_initial_only_csv(tmp_path, {}, 12.5, 1.0, 1.5)
    row = _only_row(tmp_path)
    for col in ("wrong_jaccard", "n_shared_wrong", "true_jaccard",
                "n_shared_true", "n_design_interface_residues"):
        assert row[col] == "", f"{col} should be blank without a truth interface"
    assert row["pdb"] == "", "no prediction on disk, so no path"


@pytest.mark.logic
def test_an_undetectable_interface_scores_zero_rather_than_blank(tmp_path):
    """An empty detected interface gives true_jaccard 0, not a missing value.

    This is the P0 jaccard-empty-wrong fix. Treating empty-wrong as "could not
    compute" propagated a missing true_jaccard into cross_summary's
    rep_true_jaccard_median, the composite score and every plot. The wrong
    jaccard of a cold start against itself is nan when there were no contacts.
    """
    (tmp_path / "initial_prediction.pdb").write_text("")   # parses to no residues
    plan = {"true_interface_idx": [1, 2, 3],
            "pred_receptor_chain": "A", "pred_effector_chain": "B"}
    bns._write_initial_only_csv(tmp_path, plan, 12.5, 1.0, 1.5)
    row = _only_row(tmp_path)
    assert float(row["true_jaccard"]) == 0.0
    assert row["n_shared_true"] == "0"
    assert row["n_design_interface_residues"] == "0"
    assert row["wrong_jaccard"] == "nan"
    assert row["pdb"].endswith("initial_prediction.pdb")


@pytest.mark.logic
def test_a_recorded_interface_in_the_plan_is_used_as_is(tmp_path):
    """Two of the three cold-start contacts are on the true site.

    jaccard of {1,2,9} against {1,2,3} is 2 shared over 4 unique, so 0.5.
    """
    plan = {"true_interface_idx": [1, 2, 3],
            "initial_wrong_interface_idx": [1, 2, 9]}
    bns._write_initial_only_csv(tmp_path, plan, 12.5, 1.0, 1.5)
    row = _only_row(tmp_path)
    assert float(row["true_jaccard"]) == pytest.approx(0.5)
    assert row["n_shared_true"] == "2"
    assert row["n_design_interface_residues"] == "3"
    assert float(row["wrong_jaccard"]) == 1.0, "a cold start matches itself exactly"


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — the surface-exposure heuristic
# ─────────────────────────────────────────────────────────────────────────────
# A residue is a mutation candidate only if its side chain points away from
# the local core, i.e. out at the predicted partner. The test is the sign of
# the dot product between the Cα→Cβ vector and the Cα→local-centroid vector.
# The degenerate cases decide whether an isolated or centrally placed residue
# is offered for mutation at all.

@pytest.mark.logic
@pytest.mark.parametrize("atoms", [
    {"CA": (0.0, 0.0, 0.0)},                    # glycine: no Cβ
    {"CB": (0.0, 0.0, 1.5)},                    # no Cα to measure from
    {"CA": (0.0, 0.0, 0.0), "CB": (0.0, 0.0, 0.0)},   # degenerate Cα-Cβ vector
])
def test_a_residue_with_no_side_chain_direction_is_not_exposed(atoms):
    """Glycine is treated as buried because there is no side chain to swap.

    Mutating it would be a backbone change, which is not what steering does.
    """
    r = _residue("A", 0, atoms, resname="GLY", one="G")
    assert bns.is_surface_exposed(r, [r]) is False


@pytest.mark.logic
def test_a_side_chain_pointing_away_from_the_neighbour_is_exposed():
    """The neighbour sits at +5 z and the Cβ points to -z, so the dot is -1."""
    r = _residue("A", 0, {"CA": (0.0, 0.0, 0.0), "CB": (0.0, 0.0, -1.5)})
    neighbour = _residue("A", 1, {"CA": (0.0, 0.0, 5.0)})
    assert bns.is_surface_exposed(r, [r, neighbour]) is True

    inward = _residue("A", 0, {"CA": (0.0, 0.0, 0.0), "CB": (0.0, 0.0, 1.5)})
    assert bns.is_surface_exposed(inward, [inward, neighbour]) is False


@pytest.mark.logic
def test_a_neighbour_without_a_ca_is_ignored_not_counted():
    """A residue whose Cα is missing contributes no direction information.

    Counting it as the origin would drag the local centroid towards (0,0,0)
    and flip the verdict for residues far from the origin.
    """
    r = _residue("A", 0, {"CA": (0.0, 0.0, 0.0), "CB": (0.0, 0.0, -1.5)})
    neighbour = _residue("A", 1, {"CA": (0.0, 0.0, 5.0)})
    cb_only = _residue("A", 2, {"CB": (0.0, 0.0, -9.0)})
    assert bns.is_surface_exposed(r, [r, neighbour, cb_only]) is True


@pytest.mark.logic
def test_a_residue_with_no_neighbour_within_ten_angstrom_is_exposed():
    """Nothing nearby means nothing to be buried by."""
    r = _residue("A", 0, {"CA": (0.0, 0.0, 0.0), "CB": (0.0, 0.0, 1.5)})
    far = _residue("A", 1, {"CA": (0.0, 0.0, 40.0)})
    assert bns.is_surface_exposed(r, [r, far]) is True


@pytest.mark.logic
def test_a_residue_at_the_centroid_of_its_neighbours_is_exposed():
    """Two neighbours at +5 and -5 x put the centroid on the Cα itself.

    There is then no inward direction to compare against, so the heuristic
    cannot call the residue buried and defaults to exposed.
    """
    r = _residue("A", 0, {"CA": (0.0, 0.0, 0.0), "CB": (0.0, 0.0, 1.5)})
    left = _residue("A", 1, {"CA": (-5.0, 0.0, 0.0)})
    right = _residue("A", 2, {"CA": (5.0, 0.0, 0.0)})
    assert bns.is_surface_exposed(r, [r, left, right]) is True


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — the index files that define the protected set
# ─────────────────────────────────────────────────────────────────────────────
# The true-interface indices are the residues steering must not touch, and the
# design region is where RFDiffusion was allowed to build. A single off-by-one
# in either silently poisons the protected set for the whole run, so both
# parsers validate against the receptor length at plan time.

@pytest.mark.logic
def test_the_two_true_interface_sources_are_mutually_exclusive(tmp_path):
    """Supplying both leaves it ambiguous which one defined the protected set."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        bns._load_true_interface_indices("1,2", tmp_path / "idx.txt", 76)


@pytest.mark.logic
def test_a_missing_true_interface_file_is_an_error_not_an_empty_set(tmp_path):
    """An empty protected set would let steering mutate the real binding site."""
    with pytest.raises(FileNotFoundError, match="not found"):
        bns._load_true_interface_indices(None, tmp_path / "absent.txt", 76)


@pytest.mark.logic
def test_non_integer_true_interface_tokens_are_rejected():
    """1-based ranges and residue names are both plausible operator mistakes."""
    with pytest.raises(ValueError, match="Could not parse"):
        bns._load_true_interface_indices("1,2,ARG12", None, 76)


@pytest.mark.logic
def test_an_empty_true_interface_argument_is_rejected():
    """An explicitly supplied but empty set means the file or flag is wrong."""
    with pytest.raises(ValueError, match="parsed to an empty list"):
        bns._load_true_interface_indices("", None, 76)


@pytest.mark.logic
def test_no_true_interface_source_returns_none_not_an_empty_list():
    """None and [] mean different things to the caller.

    None is "fall back to detecting the interface from the ground truth".
    [] would be "the truth interface is empty", which is never right.
    """
    assert bns._load_true_interface_indices(None, None, 76) is None


@pytest.mark.logic
def test_a_chimerax_selection_prefix_is_stripped(tmp_path):
    """ChimeraX copies selections as /A:5,26,44. Pasting that is normal.

    The positions are 1-based there, so they come back as 4, 25, 43.
    """
    f = tmp_path / "region.txt"
    f.write_text("/A:5,26,44\n")
    assert bns._load_design_region_indices(f, 76) == [4, 25, 43]


@pytest.mark.logic
def test_a_design_region_range_is_expanded(tmp_path):
    """33-36 is four 1-based positions, which are 0-based 32 to 35."""
    f = tmp_path / "region.txt"
    f.write_text("# contigs\n33-36, 74\n")
    assert bns._load_design_region_indices(f, 76) == [32, 33, 34, 35, 73]


@pytest.mark.logic
def test_a_reversed_design_region_range_is_normalised(tmp_path):
    """36-33 means the same four residues as 33-36, not an empty range."""
    f = tmp_path / "region.txt"
    f.write_text("36-33\n")
    assert bns._load_design_region_indices(f, 76) == [32, 33, 34, 35]


@pytest.mark.logic
def test_empty_tokens_and_inline_comments_are_skipped(tmp_path):
    """Trailing commas and inline notes survive hand-editing of these files."""
    f = tmp_path / "region.txt"
    f.write_text(",,5,  # the loop\n\n ,7,\n")
    assert bns._load_design_region_indices(f, 76) == [4, 6]


@pytest.mark.logic
def test_malformed_design_region_tokens_are_warned_about_and_dropped(tmp_path,
                                                                     capsys):
    """A junk token must not take the valid positions down with it.

    The warning is what tells the operator the file needs fixing, because the
    run otherwise proceeds with a smaller design region than intended.
    """
    f = tmp_path / "region.txt"
    f.write_text("5,foo-bar,baz,7\n")
    assert bns._load_design_region_indices(f, 76) == [4, 6]
    err = capsys.readouterr().err
    assert "malformed range 'foo-bar'" in err
    assert "non-integer position 'baz'" in err


@pytest.mark.logic
def test_an_out_of_range_design_region_position_raises(tmp_path):
    """Position 99 on a 76-residue receptor means the file is for another run."""
    f = tmp_path / "region.txt"
    f.write_text("99\n")
    with pytest.raises(ValueError, match=r"out of range \[1, 76\]"):
        bns._load_design_region_indices(f, 76)


@pytest.mark.logic
def test_no_design_region_file_means_no_design_region():
    assert bns._load_design_region_indices(None, 76) == []


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — the structure readers and the RMSD primitives
# ─────────────────────────────────────────────────────────────────────────────
# read_ca_atoms and read_residue_heavy_atoms must assign the same seq_index to
# the same residue, because contact detection uses one and the protected set
# uses the other. The skip rules they share are what keep them in step.

@pytest.mark.logic
def test_only_the_first_altloc_of_a_residue_is_read(tmp_path):
    """Altloc B of residue 1 sits 15 Å away. Counting it would add a residue.

    Both readers key on (chain, resnum, icode), so a second altloc would also
    consume a seq_index and shift every index after it.
    """
    pdb = _write_pdb(tmp_path / "alt.pdb", [
        _atom(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, alt="A", element="C"),
        _atom(2, "CA", "SER", "A", 1, 9.0, 9.0, 9.0, alt="B", element="C"),
        _atom(3, "CA", "GLY", "A", 2, 3.8, 0.0, 0.0, element="C"),
    ])
    entries = bns.read_ca_atoms(pdb)
    assert [(e.one, e.seq_index) for e in entries] == [("A", 0), ("G", 1)]
    assert bns._read_ca_seq_from_chain(pdb, "A") == "AG"
    assert bns._read_ca_seq_from_chain(pdb, "B") == ""


@pytest.mark.logic
def test_a_blank_altloc_followed_by_altloc_a_keeps_the_first(tmp_path):
    """Both labels pass the altloc filter, so the dedup is what separates them.

    PDBs that label one conformer blank and the next A occur in the wild. The
    reader keeps the first in file order, which is the convention the rest of
    the pipeline assumes when it compares seq_index values across files.
    """
    pdb = _write_pdb(tmp_path / "dup.pdb", [
        _atom(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, element="C"),
        _atom(2, "CA", "SER", "A", 1, 7.0, 0.0, 0.0, alt="A", element="C"),
        _atom(3, "CA", "GLY", "A", 2, 3.8, 0.0, 0.0, element="C"),
    ])
    entries = bns.read_ca_atoms(pdb)
    assert [(e.one, e.seq_index) for e in entries] == [("A", 0), ("G", 1)]
    assert entries[0].xyz == pytest.approx(np.zeros(3))


@pytest.mark.logic
def test_hydrogens_are_dropped_from_the_heavy_atom_reader(tmp_path):
    """Contact distances are minima over heavy atoms only.

    Including hydrogens would shorten every contact by roughly 1 Å and inflate
    the detected interface at a fixed cutoff. The element column is used when
    present and the atom name when it is not, because both occur in the wild.
    """
    pdb = _write_pdb(tmp_path / "h.pdb", [
        _atom(1, "CA", "GLY", "A", 1, 0.0, 0.0, 0.0, element="C"),
        _atom(2, "H", "GLY", "A", 1, 0.5, 0.5, 0.0, element="H"),
        _atom(3, "HB2", "GLY", "A", 1, 0.9, 0.9, 0.0),          # no element column
        _atom(4, "CB", "GLY", "A", 1, 5.0, 5.0, 5.0, alt="B", element="C"),
        _atom(5, "O", "GLY", "A", 1, 1.2, 0.0, 0.0, element="O"),
    ])
    residues = bns.read_residue_heavy_atoms(pdb)
    assert len(residues) == 1
    assert sorted(residues[0].atoms) == ["CA", "O"], \
        "a hydrogen or a second altloc leaked into the heavy-atom set"


@pytest.mark.logic
def test_an_unreadable_pdb_returns_no_chains_rather_than_raising(tmp_path):
    """compute_binding_rmsds calls this on paths Boltz may never have written.

    An empty dict becomes a None metric, which collect writes as a blank. A
    raise here would abort the whole collect stage over one missing file.
    """
    assert bns._read_ca_seq_by_chain(str(tmp_path / "absent.pdb")) == {}


@pytest.mark.logic
def test_contact_detection_names_the_chain_it_could_not_find(tmp_path):
    """A chain-ID mistake must fail before it becomes an empty interface."""
    pdb = _write_pdb(tmp_path / "one.pdb", [
        _atom(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, element="C"),
        _atom(2, "CA", "GLY", "A", 2, 3.8, 0.0, 0.0, element="C"),
    ])
    with pytest.raises(ValueError, match="receptor chain Z"):
        bns.find_contact_residues_heavy(pdb, "Z", "A", 4.5)
    with pytest.raises(ValueError, match="effector chain B"):
        bns.find_contact_residues_heavy(pdb, "A", "B", 4.5)


@pytest.mark.logic
def test_benchmark_constraints_refuse_a_chain_with_no_ca_atoms(tmp_path):
    pdb = _write_pdb(tmp_path / "two.pdb", [
        _atom(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, element="C"),
        _atom(2, "CA", "GLY", "B", 1, 4.0, 0.0, 0.0, element="C"),
    ])
    with pytest.raises(ValueError, match="receptor chain Z"):
        bns.build_benchmark_style_constraints_block(pdb, "Z", "B", "A", "B")
    with pytest.raises(ValueError, match="effector chain Z"):
        bns.build_benchmark_style_constraints_block(pdb, "A", "Z", "A", "B")


@pytest.mark.logic
def test_benchmark_constraints_refuse_two_chains_that_never_touch(tmp_path):
    """500 Å apart yields no pocket and no contact pairs.

    An empty constraints block would be written into the Boltz YAML and the
    prediction would run unconstrained, which is a different experiment.
    """
    lines = [_atom(i + 1, "CA", "ALA", "A", i + 1, 3.8 * i, 0.0, 0.0, element="C")
             for i in range(5)]
    lines += [_atom(i + 6, "CA", "GLY", "B", i + 1, 500.0 + 3.8 * i, 0.0, 0.0,
                    element="C") for i in range(5)]
    pdb = _write_pdb(tmp_path / "far.pdb", lines)
    with pytest.raises(ValueError, match="No benchmark-style constraints"):
        bns.build_benchmark_style_constraints_block(pdb, "A", "B", "A", "B")


@pytest.mark.logic
def test_alignment_of_an_empty_sequence_pairs_nothing():
    """A chain Boltz did not emit gives an empty sequence, not a crash."""
    assert bns._seqalign_pair_indices("", "ACDEF") == ([], [])
    assert bns._seqalign_pair_indices("ACDEF", "") == ([], [])


@pytest.mark.logic
def test_an_alignment_failure_pairs_nothing_and_warns(capsys):
    """Characters outside BLOSUM62 make the aligner raise.

    Zero pairs propagate to a None RMSD, which collect writes as a blank. A
    raise would abort collect for every remaining design.
    """
    assert bns._seqalign_pair_indices("@@@@", "@@@@") == ([], [])
    assert "Sequence alignment failed" in capsys.readouterr().out


@pytest.mark.logic
def test_kabsch_on_no_points_is_nan_with_an_identity_transform():
    """nan is the only honest RMSD for zero aligned pairs.

    The identity rotation and zero translation mean a caller that applies the
    transform anyway moves nothing rather than scattering the coordinates.
    """
    rmsd, R, t, n = bns._kabsch_align(np.zeros((0, 3)), np.zeros((0, 3)))
    assert rmsd != rmsd, "expected nan"
    assert n == 0
    assert np.allclose(R, np.eye(3))
    assert np.allclose(t, np.zeros(3))


@pytest.mark.logic
def test_kabsch_truncates_to_the_shorter_point_set():
    """Three paired points plus one stray. The stray must not enter the fit.

    P is Q shifted 10 Å along x, so the fit is exact and the translation
    recovers the shift. A fourth point 99 Å away would wreck the RMSD if it
    were included.
    """
    Q = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    P = np.vstack([Q + np.array([10.0, 0.0, 0.0]), [[99.0, 99.0, 99.0]]])
    rmsd, R, t, n = bns._kabsch_align(P, Q)
    assert n == 3
    assert rmsd == pytest.approx(0.0, abs=1e-9)
    assert t == pytest.approx(np.array([-10.0, 0.0, 0.0]), abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 — plan-stage input validation
# ─────────────────────────────────────────────────────────────────────────────
# These checks run before any GPU time is spent. Every one of them guards an
# input that would otherwise produce a finished run with meaningless indices.

def _plan_argv(workdir: Path, ground_truth: Path, *extra: str):
    return ["boltz2_negative_steering.py", "plan",
            "--ground-truth", str(ground_truth),
            "--workdir", str(workdir),
            "--receptor", "A", "--effector", "B",
            "--boltz-container", "/fake.img",
            "--n-designs", "1", "--num-seeds", "1",
            "--diffusion-samples", "1", "--recycling-steps", "1",
            *extra]


@pytest.mark.logic
def test_plan_exits_nonzero_when_the_ground_truth_will_not_parse(tmp_path,
                                                                 monkeypatch):
    """An exit code, not a traceback, so the pipeline stage fails cleanly."""
    junk = tmp_path / "junk.pdb"
    junk.write_text("not a structure\n")
    monkeypatch.setattr(sys, "argv", _plan_argv(tmp_path / "work", junk))
    assert bns.main() == 1


@pytest.mark.logic
@needs_pdb
def test_a_missing_receptor_fasta_stops_the_run(tmp_path, monkeypatch):
    """Silently using the ground-truth sequence instead would run the wrong
    experiment and report it as the override."""
    monkeypatch.setattr(sys, "argv", _plan_argv(
        tmp_path / "work", GT, "--receptor-fasta", str(tmp_path / "absent.fa")))
    with pytest.raises(SystemExit) as exc:
        bns.main()
    assert exc.value.code == 1


@pytest.mark.logic
@needs_pdb
@pytest.mark.parametrize("content", ["", ">header only\n"])
def test_an_empty_receptor_fasta_stops_the_run(tmp_path, monkeypatch, content):
    """A header with no residues is the usual way these files end up empty."""
    fa = tmp_path / "rec.fa"
    fa.write_text(content)
    monkeypatch.setattr(sys, "argv", _plan_argv(
        tmp_path / "work", GT, "--receptor-fasta", str(fa)))
    with pytest.raises(SystemExit) as exc:
        bns.main()
    assert exc.value.code == 1


@pytest.mark.logic
@needs_pdb
def test_a_different_length_override_stops_the_run(tmp_path, monkeypatch):
    """Every index in the plan is positional, so indels cannot be tolerated.

    A 75-residue override against a 76-residue truth would shift the protected
    set by one from the insertion onwards.
    """
    fa = tmp_path / "rec.fa"
    fa.write_text(">short\n" + "A" * (REC_LEN - 1) + "\n")
    monkeypatch.setattr(sys, "argv", _plan_argv(
        tmp_path / "work", GT, "--receptor-fasta", str(fa)))
    with pytest.raises(SystemExit) as exc:
        bns.main()
    assert exc.value.code == 1


@pytest.mark.logic
def test_an_unknown_steering_mode_is_rejected():
    """The mode decides the substitution chemistry, so a typo must not default."""
    with pytest.raises(ValueError, match="Unknown steering mode"):
        bns.pick_mutant_residue("K", "agressive", random.Random(0))
