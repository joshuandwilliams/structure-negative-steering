"""Tests for scripts/prepare_complex_inputs.py.

The adapter between a solved complex and the four files the engine's CLI
demands. The derivation is asserted against 6G10's known answer, and every
rejection path is exercised, because a silent wrong answer here would be
copied into every downstream metric.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PREPARE = REPO_ROOT / "scripts" / "prepare_complex_inputs.py"
SIXG10 = (REPO_ROOT / "experiments" / "benchmarking" / "unconstrained" / "6G10"
          / "6G10.pdb")

OUTPUTS = ("receptor.fasta", "effector.fasta", "true_interface.txt",
           "design_region.txt")


def _deps_available() -> bool:
    """numpy, plus a structure parser. The engine takes gemmi or biopython."""
    if importlib.util.find_spec("numpy") is None:
        return False
    return any(importlib.util.find_spec(m) is not None for m in ("gemmi", "Bio"))


needs_deps = pytest.mark.skipif(
    not _deps_available(),
    reason="needs numpy plus gemmi or biopython (pip install -e '.[experiments]')")
needs_fixture = pytest.mark.skipif(
    not SIXG10.is_file(), reason="6G10 reference not present")


@pytest.fixture(scope="module")
def prep():
    spec = importlib.util.spec_from_file_location("prepare_complex_inputs", PREPARE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── index-file parsing ───────────────────────────────────────────────────────

@pytest.mark.local_unit
@pytest.mark.parametrize("text,expected", [
    ("1\n2\n3\n", [1, 2, 3]),
    ("1,2,3\n", [1, 2, 3]),
    ("3 1 2\n", [1, 2, 3]),
    ("# a comment\n1\n\n2  # trailing\n", [1, 2]),
    ("2\n1\n2\n", [1, 2]),          # sorted and deduplicated
    ("", []),
])
def test_index_file_grammar(prep, tmp_path, text, expected):
    f = tmp_path / "idx.txt"
    f.write_text(text)
    assert prep._parse_index_file(f) == expected


# ── the derivation, against 6G10's known answer ──────────────────────────────

@pytest.mark.local_integration
@needs_deps
@needs_fixture
def test_derive_reproduces_the_committed_inputs(prep, tmp_path):
    """6G10 is receptor A at 76 residues, effector B at 83, 26 contacts at 5 A."""
    assert prep.main(["--complex", str(SIXG10), "--outdir", str(tmp_path),
                      "--derive", "--contact-cutoff", "5.0",
                      "--design-region", "none"]) == 0
    for name in OUTPUTS:
        assert (tmp_path / name).is_file(), f"missing {name}"

    rec = (tmp_path / "receptor.fasta").read_text().splitlines()[1]
    eff = (tmp_path / "effector.fasta").read_text().splitlines()[1]
    iface = prep._parse_index_file(tmp_path / "true_interface.txt")
    design = prep._parse_index_file(tmp_path / "design_region.txt")

    assert len(rec) == 76, f"receptor chain A should be 76 residues, got {len(rec)}"
    assert len(eff) == 83, f"effector chain B should be 83 residues, got {len(eff)}"
    assert len(iface) == 26, f"expected 26 interface residues, got {len(iface)}"
    assert max(iface) < len(rec), "interface index runs past the receptor chain"
    assert design == [], "design_region none should protect only the true interface"


@pytest.mark.local_integration
@needs_deps
@needs_fixture
def test_design_region_all_covers_the_whole_receptor(prep, tmp_path):
    assert prep.main(["--complex", str(SIXG10), "--outdir", str(tmp_path),
                      "--derive", "--design-region", "all"]) == 0
    rec = (tmp_path / "receptor.fasta").read_text().splitlines()[1]
    assert prep._parse_index_file(tmp_path / "design_region.txt") == \
        list(range(1, len(rec) + 1))


@pytest.mark.local_integration
@needs_deps
@needs_fixture
def test_design_region_interface_is_the_interface_shifted_to_one_based(prep, tmp_path):
    """The two files use different index bases, which is easy to get wrong."""
    assert prep.main(["--complex", str(SIXG10), "--outdir", str(tmp_path),
                      "--derive", "--design-region", "interface"]) == 0
    iface0 = prep._parse_index_file(tmp_path / "true_interface.txt")
    design1 = prep._parse_index_file(tmp_path / "design_region.txt")
    assert design1 == [i + 1 for i in iface0]


@pytest.mark.local_integration
@needs_deps
@needs_fixture
def test_provided_interface_file_is_used_verbatim(prep, tmp_path):
    iface = tmp_path / "iface.txt"
    iface.write_text("0,2,5,41\n")
    out = tmp_path / "out"
    assert prep.main(["--complex", str(SIXG10), "--outdir", str(out),
                      "--interface-file", str(iface)]) == 0
    assert prep._parse_index_file(out / "true_interface.txt") == [0, 2, 5, 41]


@pytest.mark.local_integration
@needs_deps
@needs_fixture
def test_provided_design_region_file_is_used_verbatim(prep, tmp_path):
    region = tmp_path / "region.txt"
    region.write_text("4,5,6\n")
    out = tmp_path / "out"
    assert prep.main(["--complex", str(SIXG10), "--outdir", str(out),
                      "--derive", "--design-region-file", str(region)]) == 0
    assert prep._parse_index_file(out / "design_region.txt") == [4, 5, 6]


# ── rejection paths ──────────────────────────────────────────────────────────

@pytest.mark.local_unit
def test_absent_complex_is_rejected(prep, tmp_path, capsys):
    assert prep.main(["--complex", str(tmp_path / "nope.pdb"),
                      "--outdir", str(tmp_path)]) == 2
    assert "complex PDB not found" in capsys.readouterr().err


@pytest.mark.local_integration
@needs_deps
@needs_fixture
def test_absent_interface_file_is_rejected(prep, tmp_path, capsys):
    assert prep.main(["--complex", str(SIXG10), "--outdir", str(tmp_path),
                      "--interface-file", str(tmp_path / "nope.txt")]) == 2
    assert "--interface-file not found" in capsys.readouterr().err


@pytest.mark.local_integration
@needs_deps
@needs_fixture
def test_absent_design_region_file_is_rejected(prep, tmp_path, capsys):
    assert prep.main(["--complex", str(SIXG10), "--outdir", str(tmp_path),
                      "--derive",
                      "--design-region-file", str(tmp_path / "nope.txt")]) == 2
    assert "--design-region-file not found" in capsys.readouterr().err


@pytest.mark.local_integration
@needs_deps
@needs_fixture
def test_out_of_range_interface_index_is_rejected(prep, tmp_path, capsys):
    """An index past the chain end would silently mis-place the protected set."""
    iface = tmp_path / "iface.txt"
    iface.write_text("0,1,9999\n")
    assert prep.main(["--complex", str(SIXG10), "--outdir", str(tmp_path / "o"),
                      "--interface-file", str(iface)]) == 2
    assert "out of range" in capsys.readouterr().err


@pytest.mark.local_integration
@needs_deps
@needs_fixture
@pytest.mark.parametrize("flag", ["--receptor-chain", "--effector-chain"])
def test_absent_chain_is_rejected(prep, tmp_path, flag):
    """A chain that is not in the file stops the run, naming what was available.

    The engine's own parser raises before this script's empty-sequence guard is
    reached, so the guard below it is unreachable and marked as such.
    """
    with pytest.raises(ValueError, match=r"Chain Z not found.*Available chains"):
        prep.main(["--complex", str(SIXG10), "--outdir", str(tmp_path),
                   flag, "Z", "--derive"])
