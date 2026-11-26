"""compute_metrics' guards, filters and degradations.

The arithmetic of the confidence metrics is pinned in test_pae_metrics.py and
test_compute_metrics.py, and the happy path through parse_boltz2 and main is
pinned in test_compute_metrics_cli.py. What is left is everything those take
for granted. The column discipline of the two PDB parsers, the effector-atom
restriction the contamination check rests on, the sequence guard on
find_contact_residues_heavy, and what each structural metric returns when its
inputs do not cooperate.

These paths matter because the script runs unattended once per design inside
the container, after hours of GPU time that cannot be recovered. A guard that
degrades writes a usable row. A guard that degrades the wrong way writes a row
that reads as a result and is not one.

6G10 is the real two-chain complex tracked in this repo. Chain A is the
receptor at 76 residues numbered 187-262, chain B the effector at 83 residues
numbered 31-113. Heavy-atom contacts give 26 receptor residues at 5.0 A and 23
at 4.5 A. Everything smaller is hand-built so its answer is derivable.
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

import compute_metrics as cm  # noqa: E402

np = pytest.importorskip("numpy")

GT = (_REPO_ROOT / "data" / "reference_complexes" / "6G10.pdb")
needs_pdb = pytest.mark.skipif(not GT.is_file(), reason="6G10 reference absent")

REC, EFF = "A", "B"
REC_LEN, EFF_LEN = 76, 83
TOTAL = REC_LEN + EFF_LEN

# Chain B residues within 5 A of chain A, and the rest of chain B. Derived
# from 6G10 itself, so the two sets partition the effector exactly.
EFF_INTERFACE = [39, 40, 41, 42, 43, 44, 46, 47, 49, 50, 52, 53, 56, 64, 65,
                 66, 67, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 81, 84]
EFF_BACK = sorted(set(range(31, 114)) - set(EFF_INTERFACE))


def _atom(serial, name, resname, chain, resseq, x, y, z,
          b=90.0, element="C", altloc=" ", icode=" "):
    """One PDB ATOM record in fixed columns.

    Both parsers under test slice by column rather than by field, so a helper
    that lays the columns out correctly is the only way to exercise them.
    """
    return (f"ATOM  {serial:5d} {name:<4s}{altloc:1s}{resname:>3s} "
            f"{chain:1s}{resseq:4d}{icode:1s}   "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b:6.2f}          {element:>2s}")


def _pdb(path: Path, lines) -> str:
    path.write_text("\n".join(lines) + "\nEND\n")
    return str(path)


class _UnusableFilter:
    """An effector filter that fails when it is iterated.

    compute_interface_contacts guards only TypeError from list(), so anything
    else escapes and lands in parse_boltz2's contact-analysis guard.
    """

    def __iter__(self):
        raise ValueError("effector filter is unusable")


def _tree(root: Path, structure: Path, *, plddt=None, pae=None,
          corrupt: tuple[str, ...] = ()) -> str:
    """A Boltz-2 output directory, same shape as test_compute_metrics_cli's.

    plddt and pae override the sidecar arrays so their shapes can be varied.
    Naming a sidecar in `corrupt` writes bytes np.load refuses instead.
    """
    job = root / "predictions" / "input"
    for sub in ("pae", "plddt", "confidence"):
        (job / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy(structure, job / "input_model_0.pdb")

    if "plddt" in corrupt:
        (job / "plddt" / "plddt_input_model_0.npz").write_bytes(b"not an npz")
    else:
        np.savez(job / "plddt" / "plddt_input_model_0.npz",
                 plddt=np.full(TOTAL, 0.9) if plddt is None else plddt)

    if "pae" in corrupt:
        (job / "pae" / "pae_input_model_0.npz").write_bytes(b"not an npz")
    else:
        np.savez(job / "pae" / "pae_input_model_0.npz",
                 pae=np.full((TOTAL, TOTAL), 2.0) if pae is None else pae)

    (job / "confidence" / "confidence_input_model_0.json").write_text(
        json.dumps({"ptm": 0.66, "iptm": 0.43}))
    return str(root)


def _parse(pred_dir, **kw) -> list[dict]:
    opts = dict(pae_cutoff=10.0, receptor_chain=REC, effector_chain=EFF,
                mutated_positions=None, contact_cutoff=5.0,
                effector_atom_filter=None, native_pdb=None,
                intact_plddt_threshold=50.0, intact_rmsd_threshold=5.0,
                weighted_jaccard_pair_cutoff=8.0)
    opts.update(kw)
    return cm.parse_boltz2(pred_dir, [REC_LEN, EFF_LEN], **opts)


def _run_cli(pred_dir, csv_path: Path, *extra: str):
    argv = ["compute_metrics.py",
            "--prediction-dir", str(pred_dir),
            "--chain-lengths", str(REC_LEN), str(EFF_LEN),
            "--receptor-chain", REC, "--effector-chain", EFF,
            "--model", "boltz2", "--output-csv", str(csv_path), *extra]
    with mock.patch.object(sys, "argv", argv):
        return cm.main()


def _cli_rows(csv_path: Path) -> list[dict]:
    with csv_path.open() as fh:
        return list(csv.DictReader(fh))


# ── the PAE guards the arithmetic tests do not reach ─────────────────────────
#
# A PAE matrix whose size disagrees with --chain-lengths is a real occurrence.
# The flag is set once per campaign while the matrix comes from whatever Boltz
# wrote for this design. actifPTM recovers the second chain length from the
# matrix when it can, and refuses when it cannot, so the two cases have
# different consequences for a row.

@pytest.mark.logic
def test_actifptm_rederives_the_second_chain_length_from_the_matrix():
    """A 45x45 matrix declared as 30 + 20 is scored as 30 + 15.

    With every inter-chain entry at 2 A both whole chains interact, so d0
    comes from the partner set size. B->A uses d0(30) = 1.24 * 15^(1/3) - 1.8
    = 1.258 and scores 1 / (1 + (2 / 1.258)^2) = 0.2835, which beats A->B's
    d0(15) = 0.5 floor. 0.2835 is therefore the recovered answer, and it is
    the one a row gets rather than a zero.
    """
    got = cm.compute_actifptm(np.full((45, 45), 2.0), [30, 20], cutoff=10.0)
    assert got == pytest.approx(0.2835, abs=1e-4)


@pytest.mark.logic
def test_actifptm_refuses_a_matrix_smaller_than_the_first_chain():
    """No recovery is possible once the receptor alone overruns the matrix.

    0.0 is indistinguishable from a hopeless interface in the CSV, so a
    mis-sized matrix reads as a failed design rather than a failed parse.
    """
    assert cm.compute_actifptm(np.full((5, 5), 2.0), [30, 20],
                               cutoff=10.0) == 0.0


@pytest.mark.logic
def test_the_pass_fraction_of_an_empty_inter_chain_block_is_zero():
    """A declared chain of zero length leaves no inter-chain pairs at all.

    The sizes agree so the mismatch guard does not fire, and the fraction has
    no denominator. 0.0 is the answer rather than a ZeroDivisionError.
    """
    assert cm.compute_pae_pass_frac(np.ones((2, 2)), [2, 0], 10.0) == 0.0


@pytest.mark.logic
def test_a_one_dimensional_pae_slice_is_returned_unaveraged():
    """The kernel is per-reference-residue, so a single row must not collapse.

    Averaging a 1-d slice would return one number for the whole chain and
    actifPTM's max over residues would have nothing to maximise over.
    """
    d0 = cm._ipsae_d0(100)
    got = cm._pae_to_ptm_score(np.array([0.0, 1.0]), 100)
    assert got.tolist() == pytest.approx(
        [1.0, 1.0 / (1.0 + (1.0 / d0) ** 2)])


# ── the heavy-atom parsers' column discipline ────────────────────────────────
#
# Both parsers slice fixed PDB columns rather than splitting on whitespace,
# and both decide what counts as a heavy atom. Two decisions matter. An
# alternate conformation must not contribute a second copy of a residue, and
# a hydrogen must not count towards a heavy-atom contact, including in files
# whose element column is blank. Letting hydrogens through would shrink every
# apparent contact distance by about 1 A and over-report contamination.

def _noisy_pdb(tmp_path: Path) -> str:
    return _pdb(tmp_path / "noisy.pdb", [
        _atom(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0),
        _atom(2, "CA", "ALA", "A", 1, 9.0, 9.0, 9.0, altloc="B"),
        _atom(3, "HA", "ALA", "A", 1, 0.1, 0.0, 0.0, element=" "),
        _atom(4, "1HB", "ALA", "A", 1, 0.2, 0.0, 0.0, element=" "),
        _atom(5, "HA", "ALA", "A", 1, 0.3, 0.0, 0.0, element="H"),
        _atom(6, "CB", "ALA", "A", 1, 0.5, 0.0, 0.0, element=" "),
        _atom(7, "CA", "ALA", "B", 1, 3.0, 0.0, 0.0),
    ])


@pytest.mark.logic
def test_altlocs_and_hydrogens_are_excluded_from_the_heavy_atoms(tmp_path):
    """Seven ATOM records reduce to two receptor atoms and one effector atom.

    Dropped are altloc B, a hydrogen named by its element column, and two
    hydrogens recognised only from their names because that column is blank.
    """
    got = cm._parse_heavy_atoms_by_chain(_noisy_pdb(tmp_path))
    assert got["A"] == [(1, 0.0, 0.0, 0.0), (1, 0.5, 0.0, 0.0)]
    assert got["B"] == [(1, 3.0, 0.0, 0.0)]


@pytest.mark.logic
def test_the_same_exclusions_apply_to_the_positional_parser(tmp_path):
    """find_contact_residues_heavy carries its own inline copy of the rules.

    It reports the closest heavy-atom distance, so the answer is 2.5 A from
    CB at x=0.5 rather than 2.7 A from the hydrogen at x=0.3.
    """
    got = cm.find_contact_residues_heavy(_noisy_pdb(tmp_path), REC, EFF, 5.0)
    assert got == [(0, 2.5)]


@pytest.mark.logic
@pytest.mark.parametrize("parser", ["by_chain", "positional"])
def test_an_unparseable_residue_number_drops_that_atom(tmp_path, parser):
    """A record whose number or coordinates will not parse is skipped.

    Only that record goes. Abandoning the file would lose a prediction over
    one malformed line, and both parsers are the only route to the
    contamination check.
    """
    path = _pdb(tmp_path / "bad.pdb", [
        _atom(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0).replace("A   1", "A  ??", 1),
        _atom(2, "CA", "ALA", "A", 2, 0.0, 0.0, 0.0),
        _atom(3, "CA", "ALA", "B", 1, 3.0, 0.0, 0.0),
    ])
    if parser == "by_chain":
        assert cm._parse_heavy_atoms_by_chain(path)["A"] == [(2, 0.0, 0.0, 0.0)]
    else:
        assert cm.find_contact_residues_heavy(path, REC, EFF, 5.0) == [(0, 3.0)]


@pytest.mark.logic
def test_an_unreadable_pdb_is_a_warning_and_an_empty_map(tmp_path, capsys):
    """compute_interface_contacts reads an empty map as a missing chain.

    The consequence is a row with zero contacts, which the contamination
    check then reports as clean. The warning on stdout is the only signal
    that no structure was read at all.
    """
    assert cm._parse_heavy_atoms_by_chain(tmp_path / "absent.pdb") == {}
    assert "failed to parse heavy atoms" in capsys.readouterr().out


# ── which effector residues a contact may count against ─────────────────────
#
# compute_effector_interface_residues defines the effector's functional
# binding surface from the ground truth. Contact detection in a prediction is
# then restricted to those residues, so a steering mutation touching the back
# of the effector is not counted as contamination. It is not contamination.
# The construct's binding is unaffected by it.
#
# The returned indices are 1-based positional, not ground-truth numbers. A
# RFDiffusion ground truth numbers the effector from 1000-something while
# Boltz renumbers every chain from 1, so a filter in ground-truth numbering
# would match nothing in the prediction and silently restrict to nothing.

def _tiny_gt(tmp_path: Path) -> str:
    """Three receptor and three effector CA atoms, 6 A apart in y.

    The effector carries RFDiffusion-style numbering from 1001. Receptor
    position 1 sits at the origin, so with an 8 A cutoff effector positions 1
    (6.00 A) and 2 (7.81 A) are interface and position 3 (11.66 A) is not.
    """
    return _pdb(tmp_path / "gt.pdb", [
        _atom(1, "CA", "GLY", "A", 1, 0.0, 0.0, 0.0),
        _atom(2, "CA", "GLY", "A", 2, 5.0, 0.0, 0.0),
        _atom(3, "CA", "GLY", "A", 3, 10.0, 0.0, 0.0),
        _atom(4, "CA", "GLY", "B", 1001, 0.0, 6.0, 0.0),
        _atom(5, "CA", "GLY", "B", 1002, 5.0, 6.0, 0.0),
        _atom(6, "CA", "GLY", "B", 1003, 10.0, 6.0, 0.0),
    ])


@pytest.mark.logic
def test_the_interface_residues_come_back_in_prediction_numbering(tmp_path):
    """{1, 2}, not {1001, 1002}.

    Returning ground-truth numbers would make the filter match no effector
    atom in any Boltz prediction, which restricts the contamination check to
    nothing and reports every contaminated design as clean.
    """
    got = cm.compute_effector_interface_residues(
        _tiny_gt(tmp_path), [1], REC, EFF, 8.0)
    assert got == {1, 2}


@pytest.mark.logic
def test_the_cutoff_decides_the_interface_set(tmp_path):
    """7.81 A is inside an 8 A cutoff and outside a 7 A one."""
    gt = _tiny_gt(tmp_path)
    assert cm.compute_effector_interface_residues(gt, [1], REC, EFF, 7.0) == {1}
    assert cm.compute_effector_interface_residues(gt, [1], REC, EFF, 12.0) == \
        {1, 2, 3}


@pytest.mark.logic
def test_a_malformed_ground_truth_line_drops_only_that_residue(tmp_path):
    """Receptor position 1 is the first line that parses, not the first line.

    A skipped record shifts every positional index after it, so the receptor
    interface indices the caller supplies refer to different residues. The
    parser cannot detect that, which is why the ground truth is a committed
    reference rather than a per-run artefact.
    """
    gt = _pdb(tmp_path / "bad_gt.pdb", [
        _atom(1, "CA", "GLY", "A", 1, 0.0, 0.0, 0.0).replace("A   1", "A ABC", 1),
        _atom(2, "CA", "GLY", "A", 1, 0.0, 0.0, 0.0),
        _atom(3, "CA", "GLY", "A", 2, 5.0, 0.0, 0.0),
        _atom(4, "CA", "GLY", "B", 1001, 0.0, 6.0, 0.0),
        _atom(5, "CA", "GLY", "B", 1002, 0.0, 6.0, 0.0).replace(
            "   0.000   6.000", "   X.XXX   6.000", 1),
    ])
    assert cm.compute_effector_interface_residues(gt, [1], REC, EFF, 8.0) == {1}


@pytest.mark.logic
@pytest.mark.parametrize("gt,iface", [
    (None, [1]),          # no ground truth was wired through
    ("", [1]),            # the plan carried a blank path
    ("tiny", []),         # no receptor interface to measure against
    ("tiny", [99]),       # interface indices past the end of the receptor
    ("absent", [1]),      # the path does not exist
    ("rec_only", [1]),    # the effector chain is missing from the file
])
def test_every_failure_degrades_to_the_empty_set(tmp_path, gt, iface):
    """Empty is the documented "no filter" signal, so callers fall back to
    unrestricted effector atoms. Over-reporting contamination costs a
    reversion pass. Under-reporting it puts an invalid pose in the results.
    """
    if gt == "tiny":
        gt = _tiny_gt(tmp_path)
    elif gt == "absent":
        gt = str(tmp_path / "absent.pdb")
    elif gt == "rec_only":
        gt = _pdb(tmp_path / "rec.pdb",
                  [_atom(1, "CA", "GLY", "A", 1, 0.0, 0.0, 0.0)])
    assert cm.compute_effector_interface_residues(gt, iface, REC, EFF, 8.0) \
        == set()


@pytest.mark.logic
def test_an_absent_ground_truth_says_so_on_stdout(tmp_path, capsys):
    """The empty set is also the success-with-nothing-found answer, so the
    warning is the only way to tell a missing file from a real absence."""
    cm.compute_effector_interface_residues(
        str(tmp_path / "absent.pdb"), [1], REC, EFF, 8.0)
    assert "failed to parse ground truth" in capsys.readouterr().out


# ── the effector filter inside compute_interface_contacts ────────────────────
#
# This is the function the contamination check rests on. The filter restricts
# which effector atoms a receptor contact may be counted against, and it is
# loaded from plan.json, so it arrives in whatever shape the cycle that wrote
# it used. Two shapes are accepted and one distinction is load-bearing. A
# filter that names no real residue restricts to nothing, while a filter that
# parses to nothing at all is treated as no filter.

@pytest.mark.logic
@needs_pdb
def test_the_effector_interface_filter_reproduces_the_whole_interface():
    """6G10's own 29 interface effector residues recover all 26 contacts.

    Restricting to the binding surface loses nothing, which is what makes the
    restriction safe. Every one of the 26 receptor contacts is against an
    effector residue that is genuinely at the interface.
    """
    unfiltered = cm.compute_interface_contacts(str(GT), REC, EFF, 5.0)
    filtered = cm.compute_interface_contacts(
        str(GT), REC, EFF, 5.0, effector_atom_filter=set(EFF_INTERFACE))
    assert len(unfiltered) == 26
    assert filtered == unfiltered


@pytest.mark.logic
@needs_pdb
def test_the_effector_back_face_contributes_no_contacts():
    """The other 54 effector residues find nothing at 5 A.

    This is the whole purpose of the filter. A mutation reaching the back of
    the effector is not contamination, because the wet-lab construct's
    binding is unaffected by it.
    """
    assert cm.compute_interface_contacts(
        str(GT), REC, EFF, 5.0, effector_atom_filter=set(EFF_BACK)) == set()


@pytest.mark.logic
@needs_pdb
def test_the_legacy_atom_pair_filter_collapses_to_its_residue_set():
    """{(39, "any"), ...} is read as {39, ...}.

    Older plan.json files wrote [resseq, atom_name] pairs against a Ca-only
    ground truth. Honouring the atom name would restrict to Ca atoms and find
    almost no heavy-atom contacts, so only the residue number is used and the
    atom name is discarded whatever it says.
    """
    pairs = {(r, "any") for r in EFF_INTERFACE}
    assert cm.compute_interface_contacts(
        str(GT), REC, EFF, 5.0, effector_atom_filter=pairs) == \
        cm.compute_interface_contacts(str(GT), REC, EFF, 5.0)


@pytest.mark.logic
@needs_pdb
def test_a_filter_naming_only_absent_residues_restricts_to_nothing():
    """A real restriction that matches nothing yields no contacts.

    6G10's effector is numbered 31-113, so 9999 is a residue the prediction
    does not have. The filter is non-empty, so it is applied, and the answer
    is zero contacts rather than a silent fallback to everything.
    """
    assert cm.compute_interface_contacts(
        str(GT), REC, EFF, 5.0, effector_atom_filter={9999}) == set()


@pytest.mark.logic
@needs_pdb
@pytest.mark.parametrize("flt", [
    5,                      # not iterable at all
    {"xyz"},                # a string is neither an int nor a pair
    {("CA", 39)},           # a pair with the atom name first
    {(None, "CA")},         # a pair with no residue number
    set(),                  # nothing in it
])
def test_a_filter_that_parses_to_nothing_is_read_as_no_filter(flt):
    """All 26 contacts come back, because an unparseable filter is dropped.

    The asymmetry with the previous test is deliberate and worth stating.
    A filter naming residue 9999 restricts to nothing, while a filter naming
    nothing restricts to everything. Only the second is a fallback, so a
    plan.json whose filter key is the wrong type over-reports contamination
    rather than under-reporting it.
    """
    assert len(cm.compute_interface_contacts(
        str(GT), REC, EFF, 5.0, effector_atom_filter=flt)) == 26


@pytest.mark.logic
@needs_pdb
def test_contacts_are_empty_when_a_chain_letter_is_absent():
    """The chain letters come from plan.json's pred_*_chain, so getting them
    wrong is a configuration error. It reads as a clean design."""
    assert cm.compute_interface_contacts(str(GT), REC, "Z", 5.0) == set()
    assert cm.compute_interface_contacts(str(GT), "Z", EFF, 5.0) == set()


@pytest.mark.logic
@needs_pdb
def test_the_committed_contact_counts_hold_at_both_cutoffs():
    """23 receptor residues at 4.5 A and 26 at 5.0 A.

    The contamination check compares the mutated positions against this set,
    so the cutoff decides which designs are called contaminated.
    """
    assert len(cm.compute_interface_contacts(str(GT), REC, EFF, 4.5)) == 23
    assert len(cm.compute_interface_contacts(str(GT), REC, EFF, 5.0)) == 26


# ── find_contact_residues_heavy's sequence guard ─────────────────────────────
#
# expected_rec_seq is an optional check that the walked residue count matches
# what the caller believes the receptor to be. It catches PDB-vs-FASTA
# inconsistencies that would silently misalign the positional indices
# the design-region file is written in.
#
# What it checks is length and nothing else. That matters, because comparing a
# reverted design against cycle_0's wild-type sequence raised on every design
# once. A reverted design still carries its surviving mutations, so its
# sequence differs from the wild type while its length does not. A guard that
# compared residue identity would reject every one of them.

@pytest.mark.logic
@needs_pdb
def test_the_sequence_guard_accepts_any_sequence_of_the_right_length():
    """76 residues of alanine pass against 6G10's chain A.

    The guard is a length check, so a reverted design carrying surviving
    mutations passes. Pinning this is what stops the check being tightened
    into an identity comparison that rejects every reverted design.
    """
    got = cm.find_contact_residues_heavy(
        str(GT), REC, EFF, 5.0, expected_rec_seq="A" * REC_LEN)
    assert len(got) == 26


@pytest.mark.logic
@needs_pdb
def test_the_sequence_guard_rejects_a_length_mismatch():
    """A wrong length raises and names both counts.

    Misaligned positional indices would send the reversion pass at the wrong
    residues without anything looking wrong, so this is the one case where
    failing beats degrading.
    """
    with pytest.raises(ValueError) as exc:
        cm.find_contact_residues_heavy(
            str(GT), REC, EFF, 5.0, expected_rec_seq="A" * 75)
    message = str(exc.value)
    assert "(76)" in message and "(75)" in message


@pytest.mark.logic
@needs_pdb
def test_a_missing_chain_returns_no_contacts_before_the_guard_runs():
    """The empty-chain exit comes first, so a wrong chain letter is silent
    even with expected_rec_seq supplied."""
    assert cm.find_contact_residues_heavy(
        str(GT), REC, "Z", 5.0, expected_rec_seq="A" * REC_LEN) == []
    assert cm.find_contact_residues_heavy(str(GT), "Z", EFF, 5.0) == []


# ── the weighted Jaccard's degenerate cases ──────────────────────────────────
#
# weighted_jaccard scores the model's Cb-Cb contact map against the native's
# under a Gaussian weight peaking at 4 A. Its return is (value, error), and
# which of the three empty answers comes back decides what the CSV says.
# A None value writes a blank cell. So does a NaN. A 0.0 writes a zero, which
# downstream reads as a real score of nothing.

@pytest.mark.logic
def test_a_residue_with_neither_cb_nor_ca_still_consumes_an_index(tmp_path):
    """Positions 0 and 2 come back from a three-residue chain.

    The keys are positional so that model and native stay comparable when
    their numbering differs. An anchorless residue that did not advance the
    index would shift every pair after it and compare the wrong residues.
    """
    gemmi = pytest.importorskip("gemmi")
    path = _pdb(tmp_path / "anchorless.pdb", [
        _atom(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, element="N"),
        _atom(2, "CA", "ALA", "A", 1, 1.0, 0.0, 0.0),
        _atom(3, "CB", "ALA", "A", 1, 2.0, 0.0, 0.0),
        _atom(4, "N", "ALA", "A", 2, 5.0, 0.0, 0.0, element="N"),
        _atom(5, "O", "ALA", "A", 2, 6.0, 0.0, 0.0, element="O"),
        _atom(6, "CA", "GLY", "A", 3, 10.0, 0.0, 0.0),
    ])
    got = cm._collect_chain_cb_positions(gemmi.read_structure(path), "A")
    assert [idx for idx, _xyz, _name in got] == [0, 2]


@pytest.mark.logic
def test_an_absent_chain_gives_an_empty_pair_map(tmp_path):
    """_build_weighted_pair_map returns {} rather than raising, which is what
    makes the two empty-map outcomes below reachable."""
    gemmi = pytest.importorskip("gemmi")
    path = _pdb(tmp_path / "one.pdb",
                [_atom(1, "CB", "ALA", "A", 1, 0.0, 0.0, 0.0)])
    st = gemmi.read_structure(path)
    assert cm._build_weighted_pair_map(st, "Y", "Z", 8.0) == {}


@pytest.mark.logic
@needs_pdb
def test_two_empty_pair_maps_give_nan_and_no_error():
    """Neither chain letter exists, so there is no interface to compare.

    NaN says "not computed" while 0.0 would say "computed, no overlap". The
    parser writes a blank cell for NaN, so the distinction survives into the
    CSV.
    """
    value, error = cm.compute_weighted_jaccard(str(GT), str(GT), "X", "Y", 8.0)
    assert error is None
    assert np.isnan(value)


@pytest.mark.logic
@needs_pdb
def test_one_empty_pair_map_scores_zero(tmp_path):
    """A native with no effector chain scores 0.0, not NaN.

    The model has an interface and the native has none, so the overlap really
    is zero. That is a result about the pose rather than a missing
    computation.
    """
    pytest.importorskip("gemmi")
    rec_only = tmp_path / "rec_only.pdb"
    rec_only.write_text("".join(
        line for line in GT.read_text().splitlines(keepends=True)
        if not (line.startswith(("ATOM", "HETATM")) and line[21] == EFF)))
    value, error = cm.compute_weighted_jaccard(
        str(GT), str(rec_only), REC, EFF, 8.0)
    assert (value, error) == (0.0, None)


@pytest.mark.logic
def test_pairs_too_far_apart_to_carry_weight_score_zero(tmp_path):
    """Two chains 200 A apart, admitted by a 1000 A pair cutoff.

    Every weight is exp(-(196)^2 / 2.25), which underflows to exactly zero,
    so the union is non-empty and its denominator is not. 0.0 is returned
    rather than a division by zero. This is also why the default pair cutoff
    is 8 A, which drops such pairs before they reach the sum.
    """
    pytest.importorskip("gemmi")
    far = _pdb(tmp_path / "far.pdb", [
        _atom(1, "CB", "ALA", "A", 1, 0.0, 0.0, 0.0),
        _atom(2, "CB", "ALA", "B", 1, 200.0, 0.0, 0.0),
    ])
    assert cm.compute_weighted_jaccard(far, far, REC, EFF, 1000.0) == \
        (0.0, None)
    value, error = cm.compute_weighted_jaccard(far, far, REC, EFF, 8.0)
    assert error is None and np.isnan(value)


@pytest.mark.logic
@needs_pdb
def test_an_unloadable_native_is_reported_as_an_error_string():
    """(None, "structure_load_failed:...") rather than an exception.

    The parser turns a None value into a blank cell and keeps the row, so a
    native path that has gone missing costs one column rather than the
    prediction.
    """
    value, error = cm.compute_weighted_jaccard(
        str(GT), str(GT.parent / "absent.pdb"), REC, EFF, 8.0)
    assert value is None
    assert error.startswith("structure_load_failed:")


@pytest.mark.logic
@needs_pdb
def test_a_pair_map_failure_is_reported_separately_from_a_load_failure():
    """The error string names which stage failed.

    Nothing reads these strings automatically, so they only have to be
    distinguishable by eye when a column comes back blank for a whole cohort.
    """
    value, error = cm.compute_weighted_jaccard(
        str(GT), str(GT), REC, EFF, "not-a-number")
    assert value is None
    assert error == "pair_map_failed:ValueError"


# ── the structural metrics without gemmi ─────────────────────────────────────
#
# All three native-dependent metrics import gemmi inside the function rather
# than at module scope, so compute_metrics still runs where gemmi is absent.
# Removing gemmi from sys.modules is the only way to reach that, and it is
# worth reaching. A container built without gemmi would otherwise lose every
# prediction instead of three columns.

@pytest.mark.logic
@needs_pdb
def test_every_gemmi_metric_degrades_when_gemmi_cannot_be_imported():
    with mock.patch.dict(sys.modules, {"gemmi": None}):
        wj_value, wj_error = cm.compute_weighted_jaccard(
            str(GT), str(GT), REC, EFF, 8.0)
        ic_flag, ic_rmsd, ic_error = cm.compute_intact_core(
            str(GT), str(GT), 50.0, 5.0, REC, EFF)
        ifp_value, ifp_error = cm.compute_interface_plddt(
            str(GT), REC, EFF, 5.0)

    assert (wj_value, ic_flag, ic_rmsd, ifp_value) == (None, None, None, None)
    for error in (wj_error, ic_error, ifp_error):
        assert error.startswith("gemmi_import_failed:")


@pytest.mark.logic
@needs_pdb
def test_an_unloadable_structure_stops_the_intact_core_without_raising():
    pytest.importorskip("gemmi")
    flag, rmsd, error = cm.compute_intact_core(
        str(GT), str(GT.parent / "absent.pdb"), 50.0, 5.0, REC, EFF)
    assert (flag, rmsd) == (None, None)
    assert error.startswith("structure_load_failed:")


# ── interface pLDDT's five exits ─────────────────────────────────────────────
#
# interface_plddt is the one native-free structural metric, so it is populated
# on every row and its failure modes are the ones seen most often. It reads
# the B-factor column, which Boltz fills with per-residue pLDDT.

@pytest.mark.logic
def test_chains_that_are_neither_receptor_nor_effector_are_ignored(tmp_path):
    """A third chain in the file contributes no atoms to either side.

    A uniform B-factor of 90 has to come back as 0.9, because the thresholds
    downstream are on 0-1 while Boltz writes pLDDT on 0-100. Hydrogens are
    excluded here too, by element rather than by name.
    """
    pytest.importorskip("gemmi")
    path = _pdb(tmp_path / "third.pdb", [
        _atom(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0),
        _atom(2, "CA", "ALA", "B", 1, 3.0, 0.0, 0.0),
        _atom(3, "CA", "ALA", "C", 1, 50.0, 0.0, 0.0, b=10.0),
        _atom(4, "H", "ALA", "A", 1, 0.3, 0.0, 0.0, b=10.0, element="H"),
    ])
    assert cm.compute_interface_plddt(path, REC, EFF, 5.0) == (0.9, None)


@pytest.mark.logic
@needs_pdb
def test_interface_plddt_names_each_reason_it_could_not_be_computed(tmp_path):
    """Four distinct diagnostics, all written to the CSV as the same blank.

    The strings are only visible in the stage log, so a blank column has to be
    read together with it. chain_not_found is a configuration error while
    no_interface_residues is a real result about the pose, and the CSV cannot
    tell them apart.
    """
    pytest.importorskip("gemmi")
    far = _pdb(tmp_path / "far.pdb", [
        _atom(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0),
        _atom(2, "CA", "ALA", "B", 1, 50.0, 0.0, 0.0),
    ])
    no_ca = _pdb(tmp_path / "no_ca.pdb", [
        _atom(1, "CB", "ALA", "A", 1, 0.0, 0.0, 0.0),
        _atom(2, "CB", "ALA", "B", 1, 3.0, 0.0, 0.0),
    ])
    assert cm.compute_interface_plddt(
        str(tmp_path / "absent.pdb"), REC, EFF, 5.0)[1].startswith(
            "structure_load_failed:")
    assert cm.compute_interface_plddt(str(GT), "X", "Y", 5.0) == \
        (None, "chain_not_found")
    assert cm.compute_interface_plddt(far, REC, EFF, 5.0) == \
        (None, "no_interface_residues")
    assert cm.compute_interface_plddt(no_ca, REC, EFF, 5.0) == \
        (None, "no_interface_cas")


# ── sidecar shapes and failures inside parse_boltz2 ──────────────────────────
#
# Boltz has written the pLDDT and PAE arrays in more than one shape across
# versions, and a sidecar that never finished writing is a real occurrence on
# a preempted node. The parser accepts the shapes and degrades past the
# failures, because the alternative is losing the prediction.

@pytest.mark.logic
@needs_pdb
def test_a_two_dimensional_plddt_array_is_flattened(tmp_path):
    """A (1, N) array averages to the same 90.0 as a flat one.

    Without the flatten the mean would still be 0.9 here, so the shape only
    matters for the 0-1 rescale guard, which reads arr.max().
    """
    out = _tree(tmp_path / "out", GT, plddt=np.full((1, TOTAL), 0.9))
    assert _parse(out)[0]["avg_plddt"] == pytest.approx(90.0)


@pytest.mark.logic
@needs_pdb
def test_a_three_dimensional_pae_array_takes_its_first_slice(tmp_path):
    """Some Boltz versions write PAE with a leading sample axis."""
    out = _tree(tmp_path / "out", GT, pae=np.full((1, TOTAL, TOTAL), 2.0))
    row = _parse(out)[0]
    assert row["pae_mean"] == pytest.approx(2.0)
    assert row["ipae"] == pytest.approx(2.0)


@pytest.mark.logic
@needs_pdb
def test_an_unreadable_plddt_npz_falls_back_to_the_b_factor_column(
        tmp_path, capsys):
    """6G10's real B-factors average 19.03, and that is what the row reports.

    The fallback reads whatever is in that column without knowing whether it
    is a pLDDT. For a crystal structure it is not, so an exact pose comes
    back as avg_plddt 19.03 and fails every confidence threshold. The warning
    on stdout is the only way to tell that from a genuinely poor prediction.
    """
    out = _tree(tmp_path / "out", GT, corrupt=("plddt",))
    row = _parse(out)[0]
    assert row["avg_plddt"] == cm.plddt_from_pdb(str(GT))
    assert row["avg_plddt"] == pytest.approx(19.03)
    assert "plddt parse failed" in capsys.readouterr().out


@pytest.mark.logic
@needs_pdb
def test_an_unreadable_pae_npz_leaves_every_pae_metric_at_zero(
        tmp_path, capsys):
    """pae_mean, ipae and ipsae_min all come back 0.0, not blank.

    A prediction whose PAE never landed is therefore indistinguishable in the
    CSV from one whose interface really is hopeless.
    """
    out = _tree(tmp_path / "out", GT, corrupt=("pae",))
    row = _parse(out)[0]
    assert (row["pae_mean"], row["ipae"], row["ipsae_min"]) == (0.0, 0.0, 0.0)
    assert "pae parse failed" in capsys.readouterr().out


@pytest.mark.logic
@needs_pdb
def test_a_failed_contact_analysis_reports_the_design_as_clean(
        tmp_path, capsys):
    """Zero contacts and zero mutated contacts, for a mutation at the
    interface.

    Position 220 is one of 6G10's 26 contact residues, so the honest answer
    is contaminated. The guard writes 0 instead, which reads as clean, and
    only the warning on stdout distinguishes the two. Nothing downstream
    reads that warning, so a contact-analysis failure admits an invalid pose
    to the results.
    """
    out = _tree(tmp_path / "out", GT)
    row = _parse(out, effector_atom_filter=_UnusableFilter(),
                 mutated_positions=[220])[0]
    assert row["n_contact_residues"] == 0
    assert row["contact_residues"] == ""
    assert row["n_contacts_on_mutated_positions"] == 0
    assert "contact analysis failed" in capsys.readouterr().out


@pytest.mark.logic
@needs_pdb
def test_both_native_dependent_columns_blank_when_the_native_is_unloadable(
        tmp_path):
    """A None weighted Jaccard becomes an empty cell, not the string None."""
    out = _tree(tmp_path / "out", GT)
    row = _parse(out, native_pdb=str(tmp_path / "absent.pdb"))[0]
    assert row["weighted_jaccard"] == ""
    assert row["intact_core"] == ""


@pytest.mark.logic
@needs_pdb
def test_a_nan_weighted_jaccard_becomes_an_empty_cell_too(tmp_path):
    """NaN must not reach the CSV.

    A nan written to the column would fail every comparison downstream while
    still looking like a number, so the aggregators would read it as a score
    rather than as absent.
    """
    out = _tree(tmp_path / "out", GT)
    row = _parse(out, receptor_chain="X", effector_chain="Y",
                 native_pdb=str(GT))[0]
    assert row["weighted_jaccard"] == ""
    assert row["interface_plddt"] == ""


# ── main's filter loading and its last resort ────────────────────────────────
#
# main reads the effector filter from plan.json, resolves the native path and
# then calls the parser inside a try. Each of those can fail on a real run,
# and the stage that invoked it reports success either way, so what main
# writes when it fails is what a campaign is read from.

@pytest.mark.logic
@needs_pdb
@pytest.mark.parametrize("payload", [
    "5",                                  # a bare scalar, neither dict nor list
    '[["x", "CA"]]',                      # a pair with an unparseable number
    '{"effector_interface_residues": []}',  # the key is there and empty
])
def test_a_filter_that_yields_no_residues_runs_on_all_effector_atoms(
        tmp_path, payload):
    """All 26 contacts, and a line on stdout saying the filter was empty.

    Running unfiltered over-reports contamination, which costs a reversion
    pass. Treating an empty filter as a restriction to nothing would report
    every design clean, which is the failure that matters.
    """
    out = _tree(tmp_path / "out", GT)
    flt = tmp_path / "filter.json"
    flt.write_text(payload)
    csv_path = tmp_path / "m.csv"
    _run_cli(out, csv_path, "--effector-atom-filter-json", str(flt))
    assert int(_cli_rows(csv_path)[0]["n_contact_residues"]) == 26


@pytest.mark.logic
@needs_pdb
def test_the_empty_filter_is_announced(tmp_path, capsys):
    out = _tree(tmp_path / "out", GT)
    flt = tmp_path / "filter.json"
    flt.write_text("5")
    _run_cli(out, tmp_path / "m.csv", "--effector-atom-filter-json", str(flt))
    assert "loaded but empty" in capsys.readouterr().out


@pytest.mark.logic
@needs_pdb
def test_a_native_path_that_does_not_exist_is_a_warning_not_an_exit(
        tmp_path, capsys):
    """The row is still written, with both native-dependent columns blank.

    Exiting would lose the confidence metrics as well, and those need no
    native at all.
    """
    out = _tree(tmp_path / "out", GT)
    csv_path = tmp_path / "m.csv"
    _run_cli(out, csv_path, "--native-pdb", str(tmp_path / "absent.pdb"))
    row = _cli_rows(csv_path)[0]
    assert (row["intact_core"], row["weighted_jaccard"]) == ("", "")
    assert float(row["iptm"]) == pytest.approx(0.43)
    assert "does not exist" in capsys.readouterr().out


@pytest.mark.logic
@needs_pdb
def test_a_parser_exception_writes_a_header_only_csv(tmp_path, capsys):
    """A non-numeric PAE array escapes the parser's own guards.

    main catches it, prints the traceback and writes the header alone. The
    callers read an absent file as the subprocess having failed, so a
    header-only CSV is how nothing-to-report is distinguished from that. The
    cost is that a design lost this way is indistinguishable from a
    prediction directory that was empty.
    """
    out = _tree(tmp_path / "out", GT,
                pae=np.array([["a", "b"], ["c", "d"]]))
    csv_path = tmp_path / "m.csv"
    _run_cli(out, csv_path)
    assert csv_path.read_text().splitlines() == [",".join(cm.CSV_FIELDS)]
    assert "Parser failed" in capsys.readouterr().out


# ── the position parser's empty tokens ───────────────────────────────────────
