"""The engine's defensive guards, and the inputs that reach them.

boltz2_negative_steering's happy path never runs these lines. Each one decides
what a malformed ground truth, a repeated chain label or an almost-correct cold
start reports, so leaving them unexercised means the awkward cases are the
untested ones.

test_engine_branches already covers this module's small helpers and
test_plan_stage drives cmd_plan on 6G10. Nothing here repeats either, and the
fixture builders are imported from the first rather than rewritten.

Several of these lines cannot be reached by an input alone, so every docstring
names the technique it needed and why a real input could not do the job.

Five guards inside extract_effector_template_cif carry a pragma instead of a
test. They re-read the mmCIF gemmi wrote on the line above, so reaching them
means making gemmi's writer emit something invalid, which tests gemmi rather
than this module. The two guards in that function that a real ground truth
does reach are tested, the missing polymer entity below and the non-polymer
label_seq_id in test_engine_stage_failures.
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin"))

import boltz2_negative_steering as bns  # noqa: E402
from test_engine_branches import (  # noqa: E402
    _atom,
    _line_receptor,
    _point_effector,
    _residue,
    _write_pdb,
)

_ENGINE = _REPO_ROOT / "bin" / "boltz2_negative_steering.py"


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — the BioPython import guard
# ─────────────────────────────────────────────────────────────────────────────
# Every residue pairing in this module goes through BioPython's
# PairwiseAligner. Without it there is no RMSD and no contact comparison, so
# the module refuses to load at all rather than failing later per design.

@pytest.mark.logic
def test_without_biopython_the_module_refuses_to_load(monkeypatch, capsys):
    """Technique 5, re-executing the source with Bio.Align blocked.

    The guard runs at import, before coverage can attribute anything to a
    normal import of this module, and BioPython is installed in every
    environment that runs these tests. A None entry in sys.modules is how
    Python itself marks a module as unimportable, so the except branch fires
    for the real reason.

    Exit 1 rather than a traceback is what makes the SLURM step fail cleanly,
    and naming the singularity image is what makes the failure actionable.
    """
    monkeypatch.setitem(sys.modules, "Bio.Align", None)
    namespace = {
        "__name__": "boltz2_negative_steering_under_test",
        "__file__": str(_ENGINE),
    }
    with pytest.raises(SystemExit) as exc:
        exec(compile(_ENGINE.read_text(), str(_ENGINE), "exec"), namespace)

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "BioPython is required" in err
    assert "singularity image" in err
    assert "Bio.Align" in err, "the ImportError itself was not reported"


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — alignments that pair nothing
# ─────────────────────────────────────────────────────────────────────────────
# compute_binding_rmsds pairs receptor and effector chains independently by
# global alignment. Zero pairs for either chain means no Kabsch fit, so all
# three metrics have to come back None and collect writes them blank. A raise
# instead would abort collect for every remaining design in the array.

_REC_NAMES = ["ALA", "GLY", "SER", "CYS", "LEU"]
_EFF_NAMES = ["TRP", "TYR", "PHE", "VAL"]


def _two_chain_pdb(path: Path, rec_names, eff_names) -> Path:
    """Cα-only PDB with receptor chain A and effector chain B.

    Residue names are given as three-letter codes, so a code whose one-letter
    form is outside BLOSUM62's alphabet can be planted in either chain.
    """
    lines = []
    serial = 1
    for chain, names, y in (("A", rec_names, 0.0), ("B", eff_names, 10.0)):
        for i, name in enumerate(names):
            lines.append(_atom(serial, "CA", name, chain, i + 1,
                               3.8 * i, y, 0.0, element="C"))
            serial += 1
    return _write_pdb(path, lines)


@pytest.mark.logic
def test_a_pyrrolysine_in_the_receptor_pairs_no_residues(tmp_path, capsys):
    """Technique 1, a ground truth carrying a residue BLOSUM62 has no row for.

    PYL is in THREE_TO_ONE and maps to O, which is not in BLOSUM62's alphabet,
    so PairwiseAligner raises on the whole chain and the receptor pairs come
    back empty. Selenocysteine, U, does the same.

    All three metrics must then be None. A number computed from a partial
    pairing would be reported as a measured RMSD and ranked against the real
    ones.
    """
    rec = ["ALA", "PYL", "GLY", "SER", "LEU"]
    assert bns.THREE_TO_ONE["PYL"] == "O"
    pred = _two_chain_pdb(tmp_path / "pred.pdb", rec, _EFF_NAMES)
    design = _two_chain_pdb(tmp_path / "design.pdb", rec, _EFF_NAMES)

    assert bns.compute_binding_rmsds(str(pred), str(design)) == {
        "receptor_aligned_effector_rmsd": None,
        "independent_receptor_rmsd": None,
        "independent_effector_rmsd": None,
    }
    out = capsys.readouterr().out
    assert "Receptor sequence alignment produced zero pairs" in out
    assert "not in the alphabet" in out, "the aligner's own reason was swallowed"


@pytest.mark.logic
def test_a_selenocysteine_in_the_effector_pairs_no_residues(tmp_path, capsys):
    """Technique 1. The receptor aligns, so only the effector guard can fire.

    Both chains are checked separately because either can be the unreadable
    one. A receptor-only check would let an unpaired effector through to the
    rigid transform, which would then be applied to coordinates nothing had
    paired.
    """
    eff = ["TRP", "SEC", "TYR", "VAL"]
    assert bns.THREE_TO_ONE["SEC"] == "U"
    pred = _two_chain_pdb(tmp_path / "pred.pdb", _REC_NAMES, eff)
    design = _two_chain_pdb(tmp_path / "design.pdb", _REC_NAMES, eff)

    assert bns.compute_binding_rmsds(str(pred), str(design)) == {
        "receptor_aligned_effector_rmsd": None,
        "independent_receptor_rmsd": None,
        "independent_effector_rmsd": None,
    }
    out = capsys.readouterr().out
    assert "Effector sequence alignment produced zero pairs" in out
    assert "Receptor sequence alignment produced zero pairs" not in out, \
        "the receptor chain is readable, so its guard must not have fired"


@pytest.mark.logic
def test_an_aligner_that_returns_no_alignment_pairs_nothing(monkeypatch):
    """Technique 3, on the cached BioPython aligner object.

    PairwiseAligner in global mode returns at least one alignment for any pair
    of sequences it accepts, so no input makes the count zero. An input it
    rejects makes align() raise instead, which the two tests above cover. The
    stub stands in for an aligner that finds nothing, and what is pinned is
    that zero alignments give zero pairs rather than an IndexError on alns[0].
    """
    class _NoAlignments:
        def align(self, seq_a, seq_b):
            return []

    monkeypatch.setattr(bns, "_PAIRWISE_ALIGNER", _NoAlignments())
    assert bns._seqalign_pair_indices("ACDEF", "ACDEF") == ([], [])


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — a chain label that appears twice
# ─────────────────────────────────────────────────────────────────────────────
# Both sequence parsers keep a `seen` set and skip a chain ID they have
# already recorded. Neither parser hands back a repeated ID on its default
# path, because both merge the parts as they read, so the guards decide what
# happens to a structure that reaches them already split.

def _split_chain_pdb(path: Path) -> Path:
    """Chain A, then chain B, then two more residues labelled chain A."""
    return _write_pdb(path, [
        _atom(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, element="C"),
        _atom(2, "CA", "GLY", "A", 2, 3.8, 0.0, 0.0, element="C"),
        _atom(3, "CA", "SER", "B", 1, 20.0, 0.0, 0.0, element="C"),
        _atom(4, "CA", "TRP", "A", 10, 40.0, 0.0, 0.0, element="C"),
        _atom(5, "CA", "TYR", "A", 11, 43.8, 0.0, 0.0, element="C"),
    ])


@pytest.mark.logic
def test_gemmi_reports_only_the_first_part_of_a_repeated_chain(tmp_path,
                                                               monkeypatch):
    """Technique 3, on gemmi's reader option rather than on its output.

    gemmi merges same-named chain parts as it reads, so the default path
    returns chain A once holding all four residues and the guard never fires.
    merge_chain_parts=False is gemmi's own switch for keeping the parts apart,
    which is the structure the guard is written against.

    The second part is then dropped rather than appended, so the receptor
    comes back two residues long instead of four. That matters because every
    index in the plan is positional, and a silently shortened receptor shifts
    all of them.
    """
    import gemmi

    pdb = _split_chain_pdb(tmp_path / "split.pdb")
    merged = bns.extract_sequences_gemmi(str(pdb))
    assert [(c["id"], c["sequence"]) for c in merged] == [("A", "AGWY"), ("B", "S")]

    real_read = gemmi.read_structure
    monkeypatch.setattr(gemmi, "read_structure",
                        lambda path, **kw: real_read(path, merge_chain_parts=False))
    split = bns.extract_sequences_gemmi(str(pdb))
    assert [(c["id"], c["sequence"]) for c in split] == [("A", "AG"), ("B", "S")]


@pytest.mark.logic
def test_biopython_reports_only_the_first_chain_with_a_given_id(tmp_path,
                                                                monkeypatch):
    """Technique 3, on BioPython's parser.

    Bio.PDB's structure builder folds a discontinuous chain into the one
    already in the model and warns, so no PDB makes the parser hand back chain
    A twice. The structure here is built by the real parser and then has chain
    B relabelled to A, which is the state the guard is written against.

    The whole relabelled chain disappears from the result, serine included.
    The gemmi path drops the second part the same way, which is what keeps the
    two parsers interchangeable.
    """
    from Bio import BiopythonWarning
    from Bio.PDB import PDBParser

    pdb = _split_chain_pdb(tmp_path / "split.pdb")
    structure = PDBParser(QUIET=True).get_structure("s", str(pdb))
    with pytest.warns(BiopythonWarning):
        structure[0]["B"].id = "A"
    monkeypatch.setattr(PDBParser, "get_structure",
                        lambda self, *args, **kwargs: structure)

    chains = bns.extract_sequences_biopython(str(pdb))
    assert [(c["id"], c["sequence"]) for c in chains] == [("A", "AGWY")]


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — the effector template's polymer entity
# ─────────────────────────────────────────────────────────────────────────────
# extract_effector_template_cif sets Entity.full_sequence and
# Residue.label_seq by hand, because gemmi populates neither from a PDB input
# and AF3 needs both. Both are set per polymer residue, so a chain with no
# polymer at all fails here, before the write. A chain that is a polymer plus
# a ligand gets past this and fails the label_seq_id check after the write,
# which test_engine_stage_failures covers. The five checks between them only
# re-read what gemmi just wrote, and they carry a pragma.

@pytest.mark.logic
def test_an_effector_chain_holding_only_a_ligand_has_no_polymer_entity(tmp_path):
    """Technique 1, a ground truth whose effector chain carries no residues.

    Chain B here is a zinc and nothing else, which happens when an operator
    names the ion chain as the effector. setup_entities() files it as a
    non-polymer entity, so get_polymer() is empty and get_entity_of returns
    None.

    Failing here costs 100 ms. The alternative is a template with no sequence,
    which AF3 rejects 30 minutes into a GPU job. No file is left behind, so a
    retry cannot pick up a half-written template.

    A chain holding a polymer alongside a ligand is the neighbouring case. It
    has an entity, so it gets past this guard and fails the label_seq_id check
    after the write instead.
    """
    pdb = _write_pdb(tmp_path / "ligand.pdb", [
        _atom(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, element="C"),
        _atom(2, "CA", "GLY", "A", 2, 3.8, 0.0, 0.0, element="C"),
        _atom(3, "ZN", "ZN", "B", 1, 10.0, 0.0, 0.0, element="ZN",
              record="HETATM"),
    ])
    out = tmp_path / "eff.cif"
    with pytest.raises(RuntimeError, match="No Entity associated with polymer"):
        bns.extract_effector_template_cif(pdb, "B", "C", out)
    assert not out.exists(), "a template with no sequence was left on disk"


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — the two per-residue skips
# ─────────────────────────────────────────────────────────────────────────────
# Both loops drop a residue they cannot measure rather than failing. What
# matters in each case is that the cost is one residue and not the whole
# complex.

@pytest.mark.logic
def test_a_residue_with_no_atoms_costs_one_residue_not_the_complex(tmp_path,
                                                                    monkeypatch):
    """Technique 4, on the module's own reader.

    read_residue_heavy_atoms attaches an atom in the same statement that
    creates the residue, so no file it reads produces an atom-less one. The
    guard still decides a real outcome, because np.stack on an empty atom list
    raises and that would abort contact detection for the whole complex. The
    skip is what is under test, not the reader.

    The effector atom sits 2 Å from the one real receptor residue, inside the
    4.5 Å cutoff, so the surviving contact is the proof that the loop carried
    on past the empty residue.
    """
    residues = [
        _residue("A", 0, {}),
        _residue("A", 1, {"CA": (0.0, 0.0, 0.0)}),
        _residue("B", 0, {"CA": (2.0, 0.0, 0.0)}, resname="GLY", one="G"),
    ]
    monkeypatch.setattr(bns, "read_residue_heavy_atoms", lambda path: residues)

    contacts = bns.find_contact_residues_heavy(tmp_path / "unused.pdb",
                                               "A", "B", 4.5)
    assert contacts == [(1, pytest.approx(2.0))]


class _HiddenOnFirstPass(list):
    """A residue list that yields nothing the first time it is iterated.

    _second_shell_fallback iterates pred_rec_residues twice, once to index it
    by seq_index and once to score it against the effector. Hiding the first
    pass makes the two disagree, which is the only way an index can be scored
    and then fail to map back to a residue.
    """

    def __init__(self, items):
        super().__init__(items)
        self._passes = 0

    def __iter__(self):
        self._passes += 1
        if self._passes == 1:
            return iter(())
        return super().__iter__()


@pytest.mark.logic
def test_a_scored_index_with_no_residue_behind_it_is_dropped():
    """Technique 2, a direct call with an argument cmd_plan would never pass.

    cmd_plan passes a plain list, so its two passes always agree and the guard
    cannot fire there. Driving the function directly is the honest way to pin
    what it does when they disagree, which is to drop the index rather than
    cache an exposure flag it has no residue to compute.

    All ten receptor Cαs are within the 6 Å radius of the effector atom, so
    all ten are scored and all ten reach the guard. The empty pool and the
    untouched cache together show that none of them was kept on a guess.
    """
    cache: dict = {}
    got = bns._second_shell_fallback(
        _HiddenOnFirstPass(_line_receptor()), _point_effector(),
        predicted_wrong_idx=[], true_set=set(),
        surface_flags_cache=cache, radius=6.0, pool_size=6, min_pool_size=3,
    )
    assert got == []
    assert cache == {}


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — cmd_plan recording a surviving second-shell pool
# ─────────────────────────────────────────────────────────────────────────────
# The naive pool and the fallback measure different distances, which is what
# leaves room between them. A naive candidate is a receptor residue whose
# closest heavy atom is within --contact-cutoff (4.5 Å) of the effector. A
# second-shell candidate is one whose Cα is within 6 Å. A residue in the
# 4.5-6 Å band is therefore eligible for the fallback and was never a direct
# candidate, so protecting every direct candidate does not empty the fallback.
#
# The complex below is built rather than borrowed, because 6G10's second shell
# collapses with its naive pool and no flag combination separates them.

_ROW = 24          # receptor residues per row
_SPACING = 3.8     # Å along x


def _receptor_lines():
    """Two rows of 24 alanines, the front row at z=0 and the back at z=-5.

    Each front-row Cβ points to +z, away from the back row, so the Cα→Cβ
    vector opposes the Cα→local-centroid vector and every front-row residue
    passes the surface test. Back-row Cαs are 6.4 Å or more from the nearest
    effector atom, which is outside the 6 Å fallback radius, so they can never
    enter the pool.
    """
    lines = []
    serial, resnum = 1, 1
    for z, cb_z in ((0.0, 1.5), (-5.0, -6.5)):
        for i in range(_ROW):
            x = _SPACING * i
            lines.append(_atom(serial, "CA", "ALA", "A", resnum, x, 0.0, z,
                               element="C"))
            lines.append(_atom(serial + 1, "CB", "ALA", "A", resnum, x, 0.0,
                               cb_z, element="C"))
            serial += 2
            resnum += 1
    return lines


def _effector_lines(y_offset):
    """Five glycines 4.0 Å above front-row residues 5 to 9, one 5.2 Å above 14.

    y_offset lifts the whole effector, which is how the ground truth is given
    a different binding mode from the prediction without touching the
    receptor or either sequence.
    """
    lines = []
    serial, resnum = 2001, 1
    for k in range(5):
        lines.append(_atom(serial, "CA", "GLY", "B", resnum,
                           _SPACING * (5 + k), 4.0 + y_offset, 0.0,
                           element="C"))
        serial += 1
        resnum += 1
    lines.append(_atom(serial, "CA", "GLY", "B", resnum,
                       _SPACING * 14, 5.2 + y_offset, 0.0, element="C"))
    return lines


def _complex(path: Path, y_offset: float) -> Path:
    return _write_pdb(path, _receptor_lines() + _effector_lines(y_offset))


def _run_plan(monkeypatch, workdir: Path, truth: Path, prediction: Path,
              *extra: str) -> int:
    """Drive cmd_plan with run_boltz returning a fixed prediction.

    Only the prediction needs a GPU, so this is the same substitution
    test_plan_stage makes. Everything else in the stage runs for real.
    """
    def fake_run_boltz(yaml_path, out_dir, container, recycling_steps,
                       diffusion_samples, seed, no_kernels):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pred = out_dir / "prediction.pdb"
        shutil.copy(prediction, pred)
        return pred

    argv = ["boltz2_negative_steering.py", "plan",
            "--ground-truth", str(truth), "--workdir", str(workdir),
            "--receptor", "A", "--effector", "B",
            "--boltz-container", "/fake.img",
            "--n-designs", "1", "--num-seeds", "1",
            "--diffusion-samples", "1", "--recycling-steps", "1",
            "--no-effector-template", *extra]
    monkeypatch.setattr(bns, "run_boltz", fake_run_boltz)
    monkeypatch.setattr(sys, "argv", argv)
    return bns.main()


@pytest.mark.logic
def test_a_surviving_second_shell_pool_is_recorded_as_the_fallback(tmp_path,
                                                                   monkeypatch,
                                                                   capsys):
    """Technique 1, a pose whose naive pool is empty and whose second shell
    is not. The stage runs end to end with only run_boltz substituted.

    Front-row residues 5 to 9 are the wrong interface, all exactly 4.0 Å from
    an effector Cα, and --true-interface-indices protects all five, so the
    naive pool comes out empty. Residues 4 and 10 are 5.52 Å from the nearest
    effector atom and residue 14 is 5.2 Å, each outside the 4.5 Å contact
    cutoff and inside the 6 Å fallback radius, so the second shell is exactly
    {4, 10, 14}. Three is the minimum the fallback accepts.

    The ground truth holds the same two chains with the effector lifted 40 Å
    in y, so the receptor fit is the identity and the cold-start RMSD is 40.0
    by construction. That is what stops the run short-circuiting as a clean
    cold start before any pool is built.

    Recording the fallback is what later suppresses sampling_mode and labels
    the pool in the plan. A pool built by the fallback but recorded as a
    direct one would be read downstream as evidence the wrong interface was
    mutatable, which is the opposite of what happened.
    """
    truth = _complex(tmp_path / "truth.pdb", 40.0)
    prediction = _complex(tmp_path / "pred.pdb", 0.0)
    work = tmp_path / "work"

    rc = _run_plan(monkeypatch, work, truth, prediction,
                   "--true-interface-indices", "5,6,7,8,9")
    out = capsys.readouterr().out
    assert rc == 0, out

    plan = json.loads((work / "plan.json").read_text())
    assert plan["initial_wrong_interface_idx"] == [5, 6, 7, 8, 9]
    assert plan["initial_receptor_aligned_effector_rmsd"] == pytest.approx(40.0)
    assert plan["second_shell_fallback_used"] is True
    assert plan["second_shell_pool_size"] == 3
    assert plan["candidate_pool"] == [4, 10, 14]
    assert "Second-shell pool: 3 residues" in out
    assert "[second-shell fallback]" in out


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — the invariants the pragmas claim
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.logic
def test_no_conservative_substitution_maps_a_residue_to_itself():
    """Two guards carry a pragma on the strength of this, the alanine fallback
    in pick_mutant_residue and the self-mutation skip in
    make_steered_sequence. Both become reachable the moment an entry here maps
    a residue to itself, so the claim is asserted rather than trusted.

    A self-mapping entry would make conservative mode record a mutation that
    changed nothing, which still counts towards total_mutations and towards
    every downstream mutation-burden figure.
    """
    self_mapped = sorted(k for k, v in bns.CONSERVATIVE_SUBSTITUTIONS.items()
                         if k == v)
    assert self_mapped == []
    rng = random.Random(0)
    for wt in bns.CONSERVATIVE_SUBSTITUTIONS:
        assert bns.pick_mutant_residue(wt, "conservative", rng) != wt


# ── a known limit of the mmCIF path ──────────────────────────────────────────

@pytest.mark.logic
def test_an_mmcif_chain_split_across_subchains_comes_back_short(tmp_path):
    """Pinned as it stands, because it is a trap rather than a decision.

    gemmi merges two polymer label_asym_id subchains that share one
    auth_asym_id into a single chain, but get_polymer() yields only the first
    subchain. The sequence therefore comes back short with no warning, and
    every index downstream is positional, so a short receptor shifts the
    protected set and the candidate pool.

    Nothing hits this today. The only mmCIF this pipeline writes is the
    single-chain effector template, and the PDB path is unaffected because
    gemmi keeps all the residues there. The assertion is here so that anyone
    who makes extract_sequences read a multi-subchain CIF sees it fail.
    """
    cif = tmp_path / "split.cif"
    cif.write_text(
        "data_probe\n"
        "loop_\n"
        "_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n"
        "_atom_site.label_atom_id\n_atom_site.label_alt_id\n"
        "_atom_site.label_comp_id\n_atom_site.label_asym_id\n"
        "_atom_site.label_entity_id\n_atom_site.label_seq_id\n"
        "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
        "_atom_site.auth_seq_id\n_atom_site.auth_asym_id\n"
        "ATOM 1 C CA . ALA A 1 1 0.000 0.000 0.000 1 A\n"
        "ATOM 2 C CA . GLY A 1 2 3.800 0.000 0.000 2 A\n"
        "ATOM 3 C CA . TRP C 2 1 7.600 0.000 0.000 3 A\n"
        "ATOM 4 C CA . TYR C 2 2 11.400 0.000 0.000 4 A\n")

    got = bns.extract_sequences_gemmi(str(cif))
    assert [(c["id"], c["sequence"]) for c in got] == [("A", "AG")], \
        "extract_sequences_gemmi now reads every subchain, so update the " \
        "known-limit note in its docstring"
