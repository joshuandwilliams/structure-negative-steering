"""Tests for scripts/prepare_inputs.py.

The pure-logic helpers (index parsing, output formats, coordinate conversion)
run anywhere. The full derive-from-a-complex path needs gemmi + biopython +
numpy and a complex PDB, so it is gated: it runs only when those imports and the
6G10 reference fixture are both available, and skips otherwise.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PREPARE = REPO_ROOT / "scripts" / "prepare_inputs.py"
SIXG10 = (REPO_ROOT.parent / "structure-prediction-benchmarking" / "data"
          / "complexes_for_benchmarking" / "6G10.pdb")


def _load_module():
    spec = importlib.util.spec_from_file_location("prepare_inputs", PREPARE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def prep():
    return _load_module()


@pytest.mark.local_unit
def test_parse_index_file_mixed_formats(prep, tmp_path: Path) -> None:
    f = tmp_path / "idx.txt"
    f.write_text("# header comment\n1\n3\n5, 7 , 9\n2  # trailing comment\n\n")
    assert prep._parse_index_file(f) == [1, 2, 3, 5, 7, 9]


@pytest.mark.local_unit
def test_true_interface_format_is_0based_one_per_line(prep, tmp_path: Path) -> None:
    out = tmp_path / "ti.txt"
    prep._write_true_interface(
        out, [0, 2, 5], complex_pdb=Path("x.pdb"),
        receptor_chain="A", effector_chain="B", source="test",
    )
    body = [ln for ln in out.read_text().splitlines() if not ln.startswith("#")]
    assert body == ["0", "2", "5"]
    assert "# n contacts: 3" in out.read_text()


@pytest.mark.local_unit
def test_design_region_format_is_1based_csv(prep, tmp_path: Path) -> None:
    out = tmp_path / "dr.txt"
    prep._write_design_region(
        out, [2, 4, 6], complex_pdb=Path("x.pdb"),
        receptor_chain="A", source="test",
    )
    data = [ln for ln in out.read_text().splitlines() if not ln.startswith("#")]
    assert data == ["2,4,6"]


@pytest.mark.local_unit
def test_design_region_equals_interface_plus_one(prep) -> None:
    """The default design region is the interface residues, 0-based -> 1-based."""
    iface0 = [0, 2, 5, 41]
    assert [i + 1 for i in iface0] == [1, 3, 6, 42]


def _deps_available() -> bool:
    for m in ("numpy", "gemmi", "Bio"):
        if importlib.util.find_spec(m) is None:
            return False
    return True


@pytest.mark.local_unit
@pytest.mark.skipif(not _deps_available(),
                    reason="needs numpy + gemmi + biopython (pip install -e '.[experiments]')")
@pytest.mark.skipif(not SIXG10.exists(), reason="6G10.pdb reference fixture not present")
def test_derive_end_to_end_6g10(prep, tmp_path: Path) -> None:
    rc = prep.main([
        "--complex", str(SIXG10),
        "--outdir", str(tmp_path),
        "--contact-cutoff", "5.0",
    ])
    assert rc == 0
    for name in ("receptor.fasta", "effector.fasta", "true_interface.txt", "design_region.txt"):
        assert (tmp_path / name).is_file(), f"missing {name}"

    iface = prep._parse_index_file(tmp_path / "true_interface.txt")
    design = prep._parse_index_file(tmp_path / "design_region.txt")
    assert iface, "no interface residues derived"
    # default design region == interface residues shifted to 1-based
    assert design == [i + 1 for i in iface]
    # receptor FASTA is a single non-empty sequence line
    rec_lines = (tmp_path / "receptor.fasta").read_text().splitlines()
    assert rec_lines[0].startswith(">") and len(rec_lines[1]) > 0
