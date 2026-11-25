"""Tests for bin/pathway_labels.py.

Two jobs live here. Pathway labels encode which design in which cycle produced
a sequence, and the cumulative-mutation walk reads the edits back out of every
ancestor. A label that parses wrongly mislabels a mutation set, and a walk that
silently returns [] reports a steered design as unmutated, which reads as the
cold start rather than as a missing file.

The sidecar locator is the other half. The pipeline copies Boltz's chosen PDB
out to `<design>/prediction.pdb`, but the confidence and PAE files keep Boltz's
own nested layout and are keyed off the original stem. Pointing a metrics
parser at `prediction.pdb` therefore finds no confidence at all. Every layout
the engine writes is exercised here on a built tree, because the bug this
guards against is silent.
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


# ── labels ───────────────────────────────────────────────────────────────────

@pytest.mark.logic
@pytest.mark.parametrize("parent,cycle,idx,want", [
    (None, 0, 0, "c0d00"),
    ("", 0, 7, "c0d07"),
    ("c0d03", 1, 12, "c0d03.c1d12"),
    ("c0d03.c1d12", 2, 5, "c0d03.c1d12.c2d05"),
])
def test_make_pathway_label(pl, parent, cycle, idx, want):
    assert pl.make_pathway_label(parent, cycle, idx) == want


@pytest.mark.logic
def test_label_round_trips(pl):
    label = pl.make_pathway_label(pl.make_pathway_label(None, 0, 3), 1, 12)
    assert pl.parse_pathway_label(label) == [(0, 3), (1, 12)]


@pytest.mark.logic
def test_design_index_is_zero_padded_but_parses_as_an_int(pl):
    """The label pads to two digits, so a 3 and an 03 must not become
    different pathways."""
    assert pl.make_pathway_label(None, 0, 3) == "c0d03"
    assert pl.parse_pathway_label("c0d03") == [(0, 3)]
    assert pl.parse_pathway_label("c0d3") == [(0, 3)]


@pytest.mark.logic
@pytest.mark.parametrize("bad", ["", "x0d01", "c0x01", "nonsense"])
def test_a_malformed_segment_raises(pl, bad):
    """Returning a default here would attribute mutations to the wrong design."""
    with pytest.raises(ValueError):
        pl.parse_pathway_label(bad)


@pytest.mark.logic
def test_parent_of_a_leaf_is_none(pl):
    assert pl.parent_pathway_label("c0d03") is None
    assert pl.parent_pathway_label("c0d03.c1d12") == "c0d03"
    assert pl.parent_pathway_label("c0d03.c1d12.c2d05") == "c0d03.c1d12"


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


# ── the cumulative walk ──────────────────────────────────────────────────────

def _design(root: Path, cycle: int, pathway: str | None, idx: int,
            muts: list[tuple[int, str, str]], flat: bool = False) -> None:
    """Write a plan.json and a mutations.tsv the way the engine lays them out."""
    import json
    wd = root if flat else (root / f"cycle_{cycle}" if pathway is None
                            else root / f"cycle_{cycle}" / f"pathway_{pathway}")
    d_dir = wd / f"design_{idx:02d}"
    d_dir.mkdir(parents=True, exist_ok=True)
    rows = "pos1\twt\tmut\n" + "".join(f"{p}\t{w}\t{m}\n" for p, w, m in muts)
    (d_dir / "mutations.tsv").write_text(rows)
    plan = wd / "plan.json"
    designs = json.loads(plan.read_text())["designs"] if plan.exists() else []
    while len(designs) <= idx:
        designs.append({"dir": str(d_dir)})
    designs[idx] = {"dir": str(d_dir)}
    plan.write_text(json.dumps({"designs": designs}))


@pytest.mark.logic
def test_cycle_zero_walk_returns_that_design_only(pl, tmp_path):
    _design(tmp_path, 0, None, 3, [(10, "A", "K")])
    assert pl.read_cumulative_mutations(tmp_path, "c0d03") == [(10, "A", "K")]


@pytest.mark.logic
def test_walk_collects_ancestors_oldest_first(pl, tmp_path):
    """A cycle-1 design carries its parent's edits as well as its own."""
    _design(tmp_path, 0, None, 3, [(10, "A", "K")])
    _design(tmp_path, 1, "c0d03", 7, [(44, "S", "D")])
    assert pl.read_cumulative_mutations(tmp_path, "c0d03.c1d07") == [
        (10, "A", "K"), (44, "S", "D")]


@pytest.mark.logic
def test_walk_falls_back_to_a_flat_cycle_zero_layout(pl, tmp_path):
    """Older runs wrote plan.json at the experiment root, not under cycle_0."""
    _design(tmp_path, 0, None, 1, [(5, "V", "E")], flat=True)
    assert pl.read_cumulative_mutations(tmp_path, "c0d01") == [(5, "V", "E")]


@pytest.mark.logic
def test_a_missing_plan_contributes_nothing_rather_than_raising(pl, tmp_path):
    assert pl.read_cumulative_mutations(tmp_path, "c0d00") == []


@pytest.mark.logic
def test_a_design_index_the_plan_does_not_have_is_skipped(pl, tmp_path):
    _design(tmp_path, 0, None, 0, [(1, "A", "K")])
    # Index 9 is not in that plan, so the walk yields nothing for it.
    assert pl.read_cumulative_mutations(tmp_path, "c0d09") == []


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
