"""Tests for compute_metrics' position parsers and the contamination check.

classify_mutation_reliance is the authoritative contamination check for the
whole pipeline. If a steering mutation is contacting the effector in the
predicted pose, the wet-lab construct will not carry that residue, so the
pose's metrics are invalid unless reversion recovers them. Missing one
contaminated position lets an invalid pose through as a result.

Everything it compares comes out of these parsers, and both CLI position
arguments are 1-based residue numbers. A parser that drops a token, or
expands a range off by one, changes which positions are checked without
changing the shape of anything.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("numpy") is None, reason="needs numpy")


@pytest.fixture(scope="module")
def cm():
    sys.path.insert(0, str(REPO_ROOT / "bin"))
    spec = importlib.util.spec_from_file_location(
        "compute_metrics", REPO_ROOT / "bin" / "compute_metrics.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── the strict comma list ────────────────────────────────────────────────────

@pytest.mark.logic
@pytest.mark.parametrize("spec,want", [
    ("5,26,44", [5, 26, 44]),
    ("/A:5,26,44", [5, 26, 44]),          # ChimeraX spec, chain stripped
    (" 5 , 26 ", [5, 26]),
    ("44,5,26", [5, 26, 44]),             # comes back sorted
    ("5,5,26", [5, 26]),                  # and deduplicated
    ("", []),
    (None, []),
    ("5,,26", [5, 26]),                   # empty tokens skipped
])
def test_mutated_positions_parse(cm, spec, want):
    assert cm.parse_mutated_positions(spec) == want


@pytest.mark.logic
def test_a_non_integer_is_skipped_with_a_warning_not_a_failure(cm, capsys):
    """A partial list plus a warning beats aborting mid-run, because the run
    is hours of GPU time and the remaining positions are still checkable."""
    assert cm.parse_mutated_positions("5,x,26") == [5, 26]
    assert "non-integer" in capsys.readouterr().out


@pytest.mark.logic
def test_ranges_are_not_expanded_in_the_strict_list(cm, capsys):
    """Mutation sets are small and explicit, so '30-78' there would be a
    typo rather than a range. int() rejects it and it is skipped."""
    assert cm.parse_mutated_positions("5,30-78") == [5]


@pytest.mark.logic
def test_a_leading_slash_without_a_colon_is_not_treated_as_a_chain(cm, capsys):
    """The chain prefix is only stripped when a colon follows, so a stray
    slash makes that one token unparseable rather than silently removing a
    character from a position."""
    assert cm.parse_mutated_positions("/5,26") == [26]
    assert "non-integer" in capsys.readouterr().out


# ── the range-capable list ───────────────────────────────────────────────────

@pytest.mark.logic
@pytest.mark.parametrize("spec,want", [
    ("5,26", [5, 26]),
    ("30-33", [30, 31, 32, 33]),             # inclusive at both ends
    ("5,30-32,92", [5, 30, 31, 32, 92]),
    ("/A:5,30-32", [5, 30, 31, 32]),
    ("5, 30 - 32", [5, 30, 31, 32]),         # whitespace inside a token
    ("33-30", [30, 31, 32, 33]),             # reversed range still works
    ("7-7", [7]),
    ("", []),
    (None, []),
])
def test_position_lists_parse(cm, spec, want):
    assert cm.parse_position_list(spec) == want


@pytest.mark.logic
def test_overlapping_ranges_are_merged(cm):
    assert cm.parse_position_list("30-32,31-33") == [30, 31, 32, 33]


@pytest.mark.logic
def test_a_malformed_range_is_skipped_with_a_warning(cm, capsys):
    assert cm.parse_position_list("30-x,5") == [5]
    assert "malformed range" in capsys.readouterr().out


@pytest.mark.logic
def test_a_non_integer_outside_a_range_is_skipped(cm, capsys):
    assert cm.parse_position_list("5,nope,7") == [5, 7]
    assert "non-integer" in capsys.readouterr().out


# ── the file form ────────────────────────────────────────────────────────────

@pytest.mark.logic
def test_a_positions_file_reads_lines_comments_and_ranges(cm, tmp_path):
    f = tmp_path / "pos.txt"
    f.write_text("# a header\n5, 7\n30-32   # inline comment\n\n92\n")
    assert cm.load_positions_file(f) == [5, 7, 30, 31, 32, 92]


@pytest.mark.logic
def test_an_empty_or_absent_path_is_an_empty_list(cm, tmp_path):
    assert cm.load_positions_file(None) == []
    assert cm.load_positions_file("") == []


@pytest.mark.logic
def test_a_file_of_only_comments_is_an_empty_list(cm, tmp_path):
    f = tmp_path / "pos.txt"
    f.write_text("# nothing here\n\n")
    assert cm.load_positions_file(f) == []


# ── the contamination check ──────────────────────────────────────────────────

@pytest.mark.logic
def test_a_mutation_contacting_the_effector_is_reported(cm):
    """This is the check the reversion pass acts on, and the positions it
    returns are exactly the ones reversion must revert."""
    n, positions = cm.classify_mutation_reliance({5, 9, 12}, [5, 12, 40])
    assert n == 2
    assert positions == [5, 12]


@pytest.mark.logic
def test_no_overlap_is_a_clean_design(cm):
    assert cm.classify_mutation_reliance({1, 2}, [40, 41]) == (0, [])


@pytest.mark.logic
def test_no_mutations_at_all_is_clean(cm):
    """A cold start has no mutations, so it can never be contaminated."""
    assert cm.classify_mutation_reliance({1, 2, 3}, []) == (0, [])
    assert cm.classify_mutation_reliance({1, 2, 3}, None) == (0, [])


@pytest.mark.logic
def test_the_returned_positions_are_sorted(cm):
    _, positions = cm.classify_mutation_reliance({40, 5, 12}, [40, 12, 5])
    assert positions == [5, 12, 40]


# ── pLDDT off the B-factor column ────────────────────────────────────────────

@pytest.mark.logic
def test_plddt_averages_the_ca_b_factors(cm, tmp_path):
    """Only CA atoms count. Including side-chain atoms would weight residues
    by how many atoms they have."""
    p = tmp_path / "x.pdb"
    p.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 80.00\n"
        "ATOM      2  CB  ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
        "ATOM      3  CA  ALA A   2       0.000   0.000   0.000  1.00 90.00\n")
    assert cm.plddt_from_pdb(p) == 85.0


@pytest.mark.logic
def test_a_file_with_no_ca_atoms_is_zero(cm, tmp_path):
    p = tmp_path / "x.pdb"
    p.write_text("HETATM    1  O   HOH A   1       0.0 0.0 0.0  1.00 50.00\n")
    assert cm.plddt_from_pdb(p) == 0.0


@pytest.mark.logic
def test_an_unreadable_file_is_zero_rather_than_an_error(cm, tmp_path):
    assert cm.plddt_from_pdb(tmp_path / "nope.pdb") == 0.0


@pytest.mark.logic
def test_an_unparseable_b_factor_is_skipped(cm, tmp_path):
    p = tmp_path / "x.pdb"
    p.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00   abc\n"
        "ATOM      2  CA  ALA A   2       0.000   0.000   0.000  1.00 90.00\n")
    assert cm.plddt_from_pdb(p) == 90.0
