"""Tests for bin/pathway_labels.py.

Two readers live there. read_mutations_tsv_full reads one design's mutation
record; a walk that silently returns [] reports a steered design as unmutated,
which reads as the cold start rather than as a missing file.

The sidecar locator is the other. The pipeline copies Boltz's chosen PDB out to
`<design>/prediction.pdb`, but the confidence and PAE files keep Boltz's own
nested layout and are keyed off the original stem. Pointing a metrics parser at
`prediction.pdb` therefore finds no confidence at all, and the metrics come back
as ptm=0, iptm=0 and pae=0 with only pLDDT surviving through the B-factor
fallback. Every layout the locator has to recognise is built here.

The dotted pathway-label scheme and the ancestry walk that indexed it were
removed with the cycle-1+ reader; see archive/cycle1plus_aggregation.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE = REPO_ROOT / "bin" / "pathway_labels.py"


@pytest.fixture(scope="module")
def pl():
    spec = importlib.util.spec_from_file_location("pathway_labels", MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── mutations.tsv ────────────────────────────────────────────────────────────

@pytest.mark.logic
def test_missing_tsv_is_empty_not_an_error(pl, tmp_path):
    assert pl.read_mutations_tsv_full(tmp_path / "nope.tsv") == []


@pytest.mark.logic
def test_reads_positions_wt_and_mut(pl, tmp_path):
    f = tmp_path / "mutations.tsv"
    f.write_text("pos1\twt\tmut\n12\tA\tK\n30\tg\tw\n")
    # 1-based positions are preserved and codes are upper-cased.
    assert pl.read_mutations_tsv_full(f) == [(12, "A", "K"), (30, "G", "W")]


@pytest.mark.logic
def test_unparseable_rows_are_dropped_not_guessed(pl, tmp_path):
    f = tmp_path / "mutations.tsv"
    f.write_text("pos1\twt\tmut\n"
                 "\tA\tK\n"        # no position
                 "notanint\tA\tK\n"  # position is not a number
                 "5\tA\n"          # too few columns
                 "6\t\tK\n"        # no wild type
                 "7\tA\t\n"        # no mutant
                 "8\tA\tK\n")      # the only good row
    assert pl.read_mutations_tsv_full(f) == [(8, "A", "K")]


# ── the Boltz sidecar locator ────────────────────────────────────────────────

def _boltz_output(d: Path, stem: str, *, confidence: bool,
                  layered: bool = False) -> Path:
    """Boltz's own nested layout under a design directory."""
    pdb_dir = d / "predictions" / stem / "pdb"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    pdb = pdb_dir / f"{stem}_model_0.pdb"
    pdb.write_text("ATOM\n")
    if confidence:
        if layered:
            conf = pdb_dir.parent / "confidence"
            conf.mkdir(parents=True, exist_ok=True)
            (conf / f"confidence_{pdb.stem}.json").write_text("{}")
        else:
            (pdb_dir / f"confidence_{pdb.stem}.json").write_text("{}")
    return pdb


@pytest.mark.logic
def test_locator_finds_the_nested_pdb_beside_its_confidence(pl, tmp_path):
    d = tmp_path / "design_00"
    d.mkdir()
    flat = d / "prediction.pdb"
    flat.write_text("ATOM\n")
    want = _boltz_output(d, "seq", confidence=True)
    assert pl._locate_boltz_sidecar_pdb(flat) == want


@pytest.mark.logic
def test_locator_reads_the_layered_confidence_directory(pl, tmp_path):
    d = tmp_path / "design_00"
    d.mkdir()
    flat = d / "prediction.pdb"
    flat.write_text("ATOM\n")
    want = _boltz_output(d, "seq", confidence=True, layered=True)
    assert pl._locate_boltz_sidecar_pdb(flat) == want


@pytest.mark.logic
def test_locator_never_returns_the_flat_copy_itself(pl, tmp_path):
    """Returning prediction.pdb would send the metrics parser back where it
    started, and it would find no confidence there."""
    d = tmp_path / "design_00"
    d.mkdir()
    flat = d / "prediction.pdb"
    flat.write_text("ATOM\n")
    assert pl._locate_boltz_sidecar_pdb(flat) is None


@pytest.mark.logic
def test_locator_falls_back_to_a_nested_pdb_without_confidence(pl, tmp_path):
    d = tmp_path / "design_00"
    d.mkdir()
    flat = d / "prediction.pdb"
    flat.write_text("ATOM\n")
    want = _boltz_output(d, "seq", confidence=False)
    assert pl._locate_boltz_sidecar_pdb(flat) == want


@pytest.mark.logic
def test_locator_prefers_a_pdb_that_has_confidence(pl, tmp_path):
    d = tmp_path / "design_00"
    d.mkdir()
    flat = d / "prediction.pdb"
    flat.write_text("ATOM\n")
    _boltz_output(d, "aaa", confidence=False)
    want = _boltz_output(d, "bbb", confidence=True)
    assert pl._locate_boltz_sidecar_pdb(flat) == want


@pytest.mark.logic
def test_locator_skips_cached_reference_structures(pl, tmp_path):
    """An msa/ directory holds reference structures, not predictions."""
    d = tmp_path / "design_00"
    d.mkdir()
    flat = d / "prediction.pdb"
    flat.write_text("ATOM\n")
    msa = d / "msa"
    msa.mkdir()
    (msa / "reference.pdb").write_text("ATOM\n")
    assert pl._locate_boltz_sidecar_pdb(flat) is None


@pytest.mark.logic
def test_cold_start_uses_its_own_initial_directory(pl, tmp_path):
    """Globbing the experiment root would pick up unrelated design PDBs."""
    initial = tmp_path / "initial"
    initial.mkdir()
    want = _boltz_output(initial, "seq", confidence=True)
    decoy = tmp_path / "cycle_0" / "design_00"
    decoy.mkdir(parents=True)
    (decoy / "other.pdb").write_text("ATOM\n")
    flat = tmp_path / "initial_prediction.pdb"
    flat.write_text("ATOM\n")
    assert pl._locate_boltz_sidecar_pdb(flat) == want


@pytest.mark.logic
def test_each_cold_start_seed_uses_its_own_directory(pl, tmp_path):
    """Seed 2 must not resolve to seed 1's structure."""
    for n in (1, 2):
        (tmp_path / f"initial_s{n}").mkdir()
        _boltz_output(tmp_path / f"initial_s{n}", f"seed{n}", confidence=True)
    flat = tmp_path / "initial_prediction_s2.pdb"
    flat.write_text("ATOM\n")
    got = pl._locate_boltz_sidecar_pdb(flat)
    assert got is not None and "initial_s2" in got.parts


@pytest.mark.logic
def test_cold_start_falls_back_to_the_root_when_there_is_no_seed_dir(pl, tmp_path):
    want = _boltz_output(tmp_path, "seq", confidence=True)
    flat = tmp_path / "initial_prediction_s3.pdb"
    flat.write_text("ATOM\n")
    assert pl._locate_boltz_sidecar_pdb(flat) == want


@pytest.mark.logic
def test_cold_start_without_an_initial_dir_falls_back_to_the_root(pl, tmp_path):
    want = _boltz_output(tmp_path, "seq", confidence=True)
    flat = tmp_path / "initial_prediction.pdb"
    flat.write_text("ATOM\n")
    assert pl._locate_boltz_sidecar_pdb(flat) == want
