"""Tests for bin/contig_utils.py.

A contig string says which receptor residues RFDiffusion keeps and which it
rebuilds. Every failure here is quiet. A block that resolves against the wrong
residue numbers hands RFDiffusion a valid-looking contig that fixes the wrong
stretch, and the run completes. A design region derived one residue out
protects the wrong position, and negative steering then mutates a residue it
was told to keep.

So the assertions are on the residues themselves, not on the shape of the
output. The one place the module does raise, a contig naming a chain the PDB
does not have, is tested too: RFDiffusion's own failure for that case is an
assertion reading "(<chain>, 1) is not in pdb file!", which says nothing about
the contig that caused it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE = REPO_ROOT / "bin" / "contig_utils.py"


@pytest.fixture(scope="module")
def cu():
    sys.path.insert(0, str(REPO_ROOT / "bin"))
    spec = importlib.util.spec_from_file_location("contig_utils", MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _pdb(path: Path, chains: dict[str, list[int]]) -> Path:
    """A minimal PDB carrying one CA per residue number given."""
    lines = []
    serial = 1
    for chain, resnums in chains.items():
        for r in resnums:
            lines.append(
                f"ATOM  {serial:5d}  CA  ALA {chain}{r:4d}"
                f"      0.000   0.000   0.000  1.00  0.00           C"
            )
            serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


# ── reading residue numbers off a PDB ────────────────────────────────────────

@pytest.mark.logic
def test_chain_range_and_sorted_residues(cu, tmp_path):
    p = _pdb(tmp_path / "x.pdb", {"A": [5, 7, 6, 10], "B": [1, 2]})
    assert cu.get_chain_residue_range(p, "A") == (5, 10)
    assert cu.get_chain_residues_sorted(p, "A") == [5, 6, 7, 10]
    assert cu.get_chain_residue_range(p, "B") == (1, 2)


@pytest.mark.logic
def test_an_absent_chain_is_none_rather_than_an_error(cu, tmp_path):
    p = _pdb(tmp_path / "x.pdb", {"A": [1, 2]})
    assert cu.get_chain_residue_range(p, "Z") == (None, None)
    assert cu.get_chain_residues_sorted(p, "Z") == []


@pytest.mark.logic
def test_a_malformed_residue_number_is_skipped(cu, tmp_path):
    """A non-numeric residue field must not take the whole chain down."""
    p = tmp_path / "x.pdb"
    p.write_text(
        "ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00  0.00\n"
        "ATOM      2  CA  ALA A  xx      0.000   0.000   0.000  1.00  0.00\n"
        "ATOM      3  CA  ALA A   3      0.000   0.000   0.000  1.00  0.00\n"
        "END\n")
    assert cu.get_chain_residues_sorted(p, "A") == [1, 3]


# ── parsing a block ──────────────────────────────────────────────────────────

@pytest.mark.logic
def test_fixed_denovo_and_break_segments(cu):
    descs, chain = cu.parse_block_segments("A1-10/5-5/A21-30/0")
    assert chain == "A"
    assert descs == [("fixed", "A", 1, 10), ("denovo", "5-5"),
                     ("fixed", "A", 21, 30), ("break",)]


@pytest.mark.logic
def test_a_bare_chain_letter_is_passthrough_without_a_pdb(cu):
    assert cu.parse_block_segments("B") == ([("passthrough", "B")], "B")


@pytest.mark.logic
def test_a_bare_chain_letter_resolves_against_a_pdb(cu, tmp_path):
    p = _pdb(tmp_path / "x.pdb", {"B": [3, 4, 5]})
    assert cu.parse_block_segments("B", pdb_path=p) == ([("fixed", "B", 3, 5)], "B")


@pytest.mark.logic
def test_a_bare_chain_letter_absent_from_the_pdb_stays_passthrough(cu, tmp_path):
    p = _pdb(tmp_path / "x.pdb", {"A": [1]})
    assert cu.parse_block_segments("B", pdb_path=p) == ([("passthrough", "B")], "B")


@pytest.mark.logic
def test_an_unparseable_block_says_which_block(cu):
    with pytest.raises(ValueError, match="parse failed"):
        cu.parse_block_segments("A1-")


@pytest.mark.logic
@pytest.mark.parametrize("empty", ["", "   "])
def test_an_empty_block_is_an_error(cu, empty):
    """from_string raises on empty input, so there is no empty-spec path."""
    with pytest.raises(ValueError, match="parse failed"):
        cu.parse_block_segments(empty)


# ── remapping onto real residue numbers ──────────────────────────────────────

@pytest.mark.logic
def test_matching_numbering_is_left_alone(cu, tmp_path):
    p = _pdb(tmp_path / "x.pdb", {"A": list(range(1, 21))})
    descs = [("fixed", "A", 1, 10)]
    assert cu.remap_segments_to_pdb(descs, "A", p) is descs


@pytest.mark.logic
def test_segments_are_walked_onto_the_pdb_when_numbering_differs(cu, tmp_path):
    """The contig counts from 1, the structure starts at 101."""
    p = _pdb(tmp_path / "x.pdb", {"A": list(range(101, 121))})
    out = cu.remap_segments_to_pdb([("fixed", "A", 1, 10)], "A", p)
    assert out == [("fixed", "A", 101, 110)]


@pytest.mark.logic
def test_a_denovo_gap_advances_the_walk_by_its_own_length(cu, tmp_path):
    """The second fixed block must land after the rebuilt stretch, not on it."""
    p = _pdb(tmp_path / "x.pdb", {"A": list(range(101, 126))})
    out = cu.remap_segments_to_pdb(
        [("fixed", "A", 1, 10), ("denovo", "5-5"), ("fixed", "A", 16, 25)], "A", p)
    assert out == [("fixed", "A", 101, 110), ("denovo", "5-5"),
                   ("fixed", "A", 116, 125)]


@pytest.mark.logic
def test_a_variable_denovo_takes_the_residues_left_over(cu, tmp_path):
    """One variable gap is solved from the chain length, not from its minimum."""
    p = _pdb(tmp_path / "x.pdb", {"A": list(range(101, 131))})
    out = cu.remap_segments_to_pdb(
        [("fixed", "A", 1, 10), ("denovo", "2-20"), ("fixed", "A", 30, 39)], "A", p)
    # 30 residues, 20 fixed, so the gap is 10 and the tail starts at 121.
    assert out[2] == ("fixed", "A", 121, 130)


@pytest.mark.logic
def test_several_variable_denovos_fall_back_to_their_minimums(cu, tmp_path):
    """With more than one unknown the length is underdetermined."""
    p = _pdb(tmp_path / "x.pdb", {"A": list(range(101, 141))})
    out = cu.remap_segments_to_pdb(
        [("fixed", "A", 1, 5), ("denovo", "3-9"), ("fixed", "A", 20, 24),
         ("denovo", "4-8"), ("fixed", "A", 40, 44)], "A", p)
    assert out[2] == ("fixed", "A", 109, 113)   # 5 fixed + 3 minimum
    assert out[4] == ("fixed", "A", 118, 122)   # + 5 fixed + 4 minimum


@pytest.mark.logic
def test_a_break_segment_survives_remapping_unchanged(cu, tmp_path):
    p = _pdb(tmp_path / "x.pdb", {"A": list(range(101, 111))})
    out = cu.remap_segments_to_pdb(
        [("fixed", "A", 1, 5), ("break",), ("fixed", "A", 6, 10)], "A", p)
    assert out[1] == ("break",)


@pytest.mark.logic
def test_no_block_chain_is_a_no_op(cu, tmp_path):
    p = _pdb(tmp_path / "x.pdb", {"A": [1]})
    descs = [("denovo", "5-5")]
    assert cu.remap_segments_to_pdb(descs, None, p) is descs


@pytest.mark.logic
def test_a_contig_naming_a_chain_the_pdb_lacks_exits_with_the_reason(cu, tmp_path):
    """RFDiffusion's own error for this says nothing about the contig."""
    p = _pdb(tmp_path / "x.pdb", {"A": [1, 2, 3]})
    with pytest.raises(SystemExit) as exc:
        cu.remap_segments_to_pdb([("fixed", "Z", 1, 10)], "Z", p)
    msg = str(exc.value)
    assert "Z" in msg and "no atoms" in msg


@pytest.mark.logic
def test_a_missing_chain_with_no_fixed_segments_passes_through(cu, tmp_path):
    p = _pdb(tmp_path / "x.pdb", {"A": [1]})
    descs = [("denovo", "5-5")]
    assert cu.remap_segments_to_pdb(descs, "Z", p) is descs


@pytest.mark.logic
def test_a_single_residue_fixed_segment_reads_back_singular(cu, tmp_path):
    """The error text writes A7 rather than A7-7."""
    p = _pdb(tmp_path / "x.pdb", {"A": [1]})
    with pytest.raises(SystemExit) as exc:
        cu.remap_segments_to_pdb([("fixed", "Z", 7, 7)], "Z", p)
    assert "Z7" in str(exc.value) and "Z7-7" not in str(exc.value)


# ── round trip ───────────────────────────────────────────────────────────────

@pytest.mark.logic
def test_segments_to_string_round_trips_every_segment_type(cu):
    descs = [("fixed", "A", 1, 10), ("denovo", "5-9"), ("break",),
             ("passthrough", "B")]
    assert cu.segments_to_string(descs) == "A1-10/5-9/0/B"


@pytest.mark.logic
def test_resolve_contigs_expands_a_bare_chain(cu, tmp_path):
    p = _pdb(tmp_path / "x.pdb", {"A": list(range(1, 11)), "B": list(range(1, 6))})
    assert cu.resolve_contigs("A1-10 B", p) == "A1-10 B1-5"


@pytest.mark.logic
def test_resolve_contigs_warns_on_a_chain_the_pdb_lacks(cu, tmp_path, capsys):
    p = _pdb(tmp_path / "x.pdb", {"A": [1, 2]})
    assert cu.resolve_contigs("Z", p) == "Z"
    assert "Chain 'Z' not found" in capsys.readouterr().err


@pytest.mark.logic
@pytest.mark.parametrize("sep", [",", ":", " "])
def test_blocks_may_be_separated_three_ways(cu, tmp_path, sep):
    p = _pdb(tmp_path / "x.pdb", {"A": list(range(1, 11)), "B": list(range(1, 6))})
    assert cu.resolve_contigs(f"A1-10{sep}B", p) == "A1-10 B1-5"


@pytest.mark.logic
def test_resolve_contigs_remaps_against_the_pdb(cu, tmp_path):
    p = _pdb(tmp_path / "x.pdb", {"A": list(range(101, 121))})
    assert cu.resolve_contigs("A1-10", p) == "A101-110"


# ── expected lengths ─────────────────────────────────────────────────────────

@pytest.mark.logic
def test_expected_lengths_count_each_chain_separately(cu):
    assert cu.get_expected_chain_lengths("A1-10 B1-5", "A", "B") == (10, 5)


@pytest.mark.logic
def test_a_denovo_range_contributes_its_midpoint(cu):
    # 10 fixed plus (4+8)//2 = 6.
    assert cu.get_expected_chain_lengths("A1-10/4-8", "A", "B") == (16, 0)


@pytest.mark.logic
def test_a_bare_denovo_number_contributes_itself(cu):
    assert cu.get_expected_chain_lengths("A1-10/7", "A", "B") == (17, 0)


@pytest.mark.logic
def test_a_single_residue_reference_counts_as_one(cu):
    assert cu.get_expected_chain_lengths("A5", "A", "B") == (1, 0)


@pytest.mark.logic
def test_break_markers_and_blanks_are_not_counted(cu):
    assert cu.get_expected_chain_lengths("A1-10/0", "A", "B") == (10, 0)


@pytest.mark.logic
def test_chain_letters_are_matched_case_insensitively(cu):
    assert cu.get_expected_chain_lengths("a1-10 b1-5", "A", "B") == (10, 5)


# ── the design region ────────────────────────────────────────────────────────

@pytest.mark.logic
def test_design_region_is_the_gap_between_fixed_segments(cu):
    design, fixed = cu.parse_design_region("A1-10/5-5/A16-25", "A")
    assert design == set(range(11, 16))
    assert fixed == set(range(1, 11)) | set(range(16, 26))


@pytest.mark.logic
def test_adjacent_fixed_segments_leave_no_design_region(cu):
    design, fixed = cu.parse_design_region("A1-10/A11-20", "A")
    assert design == set()
    assert fixed == set(range(1, 21))


@pytest.mark.logic
def test_two_gaps_are_both_found(cu):
    design, _ = cu.parse_design_region("A1-5/3-3/A9-12/4-4/A17-20", "A")
    assert design == set(range(6, 9)) | set(range(13, 17))


@pytest.mark.logic
def test_the_receptor_chain_is_matched_case_insensitively(cu):
    design, fixed = cu.parse_design_region("a1-10/5-5/a16-25", "A")
    assert design == set(range(11, 16))


@pytest.mark.logic
def test_an_unparseable_contig_gives_empty_sets(cu):
    assert cu.parse_design_region("A1-", "A") == (set(), set())


@pytest.mark.logic
def test_a_chain_not_in_the_contig_gives_empty_sets(cu):
    assert cu.parse_design_region("A1-10/5-5/A16-25", "Z") == (set(), set())


@pytest.mark.logic
def test_a_chain_with_no_fixed_segments_gives_empty_sets(cu):
    assert cu.parse_design_region("A", "A") == (set(), set())


# ── a passthrough segment inside a block ─────────────────────────────────────
#
# A bare chain letter after a slash, as in "A1-10/B", is a PassthroughSegment
# rather than a whole passthrough block. It resolves against the PDB on its
# own chain, not the block's, so a mix-up here silently fixes the wrong chain.

@pytest.mark.logic
def test_a_passthrough_segment_resolves_on_its_own_chain(cu, tmp_path):
    p = _pdb(tmp_path / "x.pdb", {"A": list(range(1, 11)), "B": list(range(4, 9))})
    descs, chain = cu.parse_block_segments("A1-10/B", pdb_path=p)
    assert chain == "A"
    assert descs == [("fixed", "A", 1, 10), ("fixed", "B", 4, 8)]


@pytest.mark.logic
def test_a_passthrough_segment_stays_unresolved_without_a_pdb(cu):
    descs, _ = cu.parse_block_segments("A1-10/B")
    assert descs == [("fixed", "A", 1, 10), ("passthrough", "B")]


@pytest.mark.logic
def test_a_passthrough_segment_whose_chain_is_absent_stays_passthrough(cu, tmp_path):
    p = _pdb(tmp_path / "x.pdb", {"A": list(range(1, 11))})
    descs, _ = cu.parse_block_segments("A1-10/Z", pdb_path=p)
    assert descs == [("fixed", "A", 1, 10), ("passthrough", "Z")]


@pytest.mark.logic
def test_chain_range_skips_a_malformed_residue_number(cu, tmp_path):
    """get_chain_residue_range has its own parse guard, separate from
    get_chain_residues_sorted's."""
    p = tmp_path / "x.pdb"
    p.write_text(
        "ATOM      1  CA  ALA A   2      0.000   0.000   0.000  1.00  0.00\n"
        "ATOM      2  CA  ALA A  ??      0.000   0.000   0.000  1.00  0.00\n"
        "ATOM      3  CA  ALA A   9      0.000   0.000   0.000  1.00  0.00\n"
        "END\n")
    assert cu.get_chain_residue_range(p, "A") == (2, 9)
