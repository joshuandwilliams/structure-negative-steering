"""Tests for bin/derive_design_region.py.

Writes the de novo stretch RFDiffusion built, which the engine must not touch.
Its sibling derive_true_interface.py writes the protected set from a different
source field through the same transform, and the two are cross-checked against
each other: the true interface has to sit inside the design region on a
resurfacing design.

The coordinate systems differ, and that is where the quiet failures are. The
derivation works in 0-based positional indices, the cross-check compares in
0-based, and the file on disk is 1-based. One off-by-one anywhere protects the
wrong residues.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE = REPO_ROOT / "bin" / "derive_design_region.py"


@pytest.fixture(scope="module")
def ddr():
    spec = importlib.util.spec_from_file_location("derive_design_region", MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _entry(design=(12, 13, 14), order=(10, 11, 12, 13, 14), name="design_3"):
    return {"design": name,
            "per_design_design_residues": list(design),
            "receptor_position_order": list(order)}


def _metrics(tmp_path, *entries):
    p = tmp_path / "rfdiffusion_metrics.json"
    p.write_text(json.dumps({"designs": list(entries)}))
    return p


# ── identifying a design ─────────────────────────────────────────────────────

@pytest.mark.logic
@pytest.mark.parametrize("raw,bare", [
    ("design_3", "design_3"), ("design_3.pdb", "design_3"), ("3", "design_3"),
])
def test_design_ids_normalise(ddr, raw, bare):
    assert ddr.normalise_design_id(raw) == {"bare": bare,
                                            "with_pdb": f"{bare}.pdb"}


@pytest.mark.logic
@pytest.mark.parametrize("asked", ["design_3", "design_3.pdb", "3"])
def test_any_spelling_finds_the_entry(ddr, asked):
    assert ddr.find_design_entry({"designs": [_entry()]}, asked)["design"] == \
        "design_3"


@pytest.mark.logic
def test_a_missing_designs_array_is_an_error(ddr):
    with pytest.raises(ValueError, match="designs"):
        ddr.find_design_entry({"designs": "nope"}, "design_3")


@pytest.mark.logic
def test_an_unmatched_design_lists_the_alternatives(ddr):
    metrics = {"designs": [_entry(name=f"design_{i}") for i in range(3)]}
    with pytest.raises(KeyError) as exc:
        ddr.find_design_entry(metrics, "design_9")
    assert "design_0" in str(exc.value)


@pytest.mark.logic
def test_a_long_list_is_truncated_with_a_count(ddr):
    metrics = {"designs": [_entry(name=f"design_{i}") for i in range(25)]}
    with pytest.raises(KeyError) as exc:
        ddr.find_design_entry(metrics, "design_99")
    assert "(25 total)" in str(exc.value)


# ── the derivation ───────────────────────────────────────────────────────────

@pytest.mark.logic
def test_indices_are_positions_in_the_walk(ddr):
    assert ddr.derive_positional_indices(
        _entry(design=(12, 14), order=(10, 11, 12, 13, 14))) == [2, 4]


@pytest.mark.logic
@pytest.mark.parametrize("field", ["per_design_design_residues",
                                   "receptor_position_order"])
def test_a_missing_field_names_itself(ddr, field):
    e = _entry()
    del e[field]
    with pytest.raises(ValueError, match=field):
        ddr.derive_positional_indices(e)


@pytest.mark.logic
def test_an_empty_design_region_is_an_error(ddr):
    with pytest.raises(ValueError, match="no residues to extract"):
        ddr.derive_positional_indices(_entry(design=()))


@pytest.mark.logic
def test_an_empty_position_order_is_an_error(ddr):
    with pytest.raises(ValueError, match="cannot derive"):
        ddr.derive_positional_indices(_entry(order=()))


@pytest.mark.logic
def test_a_design_residue_absent_from_the_walk_aborts(ddr):
    """Dropping it would shrink the design region, leaving part of the de novo
    stretch unprotected."""
    with pytest.raises(ValueError) as exc:
        ddr.derive_positional_indices(_entry(design=(12, 99), order=(10, 12)))
    assert "99" in str(exc.value) and "different atom walks" in str(exc.value)


@pytest.mark.logic
def test_many_missing_residues_are_truncated(ddr):
    with pytest.raises(ValueError) as exc:
        ddr.derive_positional_indices(
            _entry(design=tuple(range(100, 130)), order=(100,)))
    assert "..." in str(exc.value)


# ── the cross-check ──────────────────────────────────────────────────────────

@pytest.mark.logic
def test_cross_check_is_advisory_when_the_plan_is_absent(ddr, tmp_path, capsys):
    assert ddr.cross_check_with_plan(tmp_path / "no.json", [0, 1]) is True
    assert "not found" in capsys.readouterr().err


@pytest.mark.logic
def test_cross_check_is_advisory_when_the_plan_is_unreadable(ddr, tmp_path,
                                                             capsys):
    p = tmp_path / "plan.json"
    p.write_text("{not json")
    assert ddr.cross_check_with_plan(p, [0]) is True
    assert "failed to read" in capsys.readouterr().err


@pytest.mark.logic
def test_cross_check_is_skipped_without_a_true_interface(ddr, tmp_path, capsys):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps({"designs": []}))
    assert ddr.cross_check_with_plan(p, [0]) is True
    assert "no true_interface_idx" in capsys.readouterr().err


@pytest.mark.logic
def test_an_interface_inside_the_design_region_passes(ddr, tmp_path, capsys):
    """Both sides are 0-based here, unlike the file this module writes."""
    p = tmp_path / "plan.json"
    p.write_text(json.dumps({"true_interface_idx": [1, 2]}))
    assert ddr.cross_check_with_plan(p, [0, 1, 2, 3]) is True
    assert "subset" in capsys.readouterr().err


@pytest.mark.logic
def test_an_interface_reaching_outside_is_reported(ddr, tmp_path, capsys):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps({"true_interface_idx": [1, 9]}))
    assert ddr.cross_check_with_plan(p, [0, 1, 2]) is False
    err = capsys.readouterr().err
    assert "OUTSIDE" in err and "9" in err


# ── formatting ───────────────────────────────────────────────────────────────

@pytest.mark.logic
def test_contiguous_runs_collapse_into_ranges(ddr):
    out = ddr.format_output([30, 31, 32, 35, 36, 37], Path("/m.json"), "d", True)
    assert out.splitlines()[-1] == "30-32,35-37"


@pytest.mark.logic
def test_a_lone_position_is_written_bare(ddr):
    out = ddr.format_output([5, 9, 10], Path("/m.json"), "d", True)
    assert out.splitlines()[-1] == "5,9-10"


@pytest.mark.logic
def test_a_trailing_single_position_is_written_bare(ddr):
    out = ddr.format_output([1, 2, 7], Path("/m.json"), "d", True)
    assert out.splitlines()[-1] == "1-2,7"


@pytest.mark.logic
def test_no_ranges_writes_a_plain_list(ddr):
    out = ddr.format_output([30, 31, 32], Path("/m.json"), "d", False)
    assert out.splitlines()[-1] == "30,31,32"


@pytest.mark.logic
def test_an_empty_list_writes_an_empty_body(ddr):
    out = ddr.format_output([], Path("/m.json"), "d", True)
    assert out.splitlines()[-1] == ""


@pytest.mark.logic
def test_the_header_records_its_provenance_and_coordinate_system(ddr):
    out = ddr.format_output([1], Path("/m.json"), "design_3", True)
    head = "\n".join(ln for ln in out.splitlines() if ln.startswith("#"))
    assert "/m.json" in head and "design_3" in head and "1-based" in head


# ── the CLI ──────────────────────────────────────────────────────────────────

def _run(ddr, monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["derive_design_region.py"] + argv)
    return ddr.main()


@pytest.mark.logic
def test_cli_writes_one_based_ranges(ddr, tmp_path, monkeypatch):
    """Indices 2,3,4 are positions 3,4,5 in the file."""
    m = _metrics(tmp_path, _entry())
    out = tmp_path / "region.txt"
    assert _run(ddr, monkeypatch, ["--metrics-json", str(m), "--design", "3",
                                   "--output", str(out)]) == 0
    assert out.read_text().splitlines()[-1] == "3-5"


@pytest.mark.logic
def test_cli_honours_no_ranges(ddr, tmp_path, monkeypatch):
    m = _metrics(tmp_path, _entry())
    out = tmp_path / "region.txt"
    assert _run(ddr, monkeypatch, ["--metrics-json", str(m), "--design", "3",
                                   "--output", str(out), "--no-ranges"]) == 0
    assert out.read_text().splitlines()[-1] == "3,4,5"


@pytest.mark.logic
def test_cli_reports_a_region_with_gaps(ddr, tmp_path, monkeypatch, capsys):
    m = _metrics(tmp_path, _entry(design=(10, 14), order=(10, 11, 12, 13, 14)))
    assert _run(ddr, monkeypatch, ["--metrics-json", str(m), "--design", "3",
                                   "--output", str(tmp_path / "o.txt")]) == 0
    assert "with gaps" in capsys.readouterr().err


@pytest.mark.logic
def test_cli_reports_a_contiguous_region(ddr, tmp_path, monkeypatch, capsys):
    m = _metrics(tmp_path, _entry())
    assert _run(ddr, monkeypatch, ["--metrics-json", str(m), "--design", "3",
                                   "--output", str(tmp_path / "o.txt")]) == 0
    assert "contiguous" in capsys.readouterr().err


@pytest.mark.logic
def test_cli_rejects_a_missing_metrics_file(ddr, tmp_path, monkeypatch, capsys):
    assert _run(ddr, monkeypatch, ["--metrics-json", str(tmp_path / "no.json"),
                                   "--design", "3",
                                   "--output", str(tmp_path / "o.txt")]) == 2
    assert "not found" in capsys.readouterr().err


@pytest.mark.logic
def test_cli_rejects_unparseable_json(ddr, tmp_path, monkeypatch, capsys):
    m = tmp_path / "m.json"
    m.write_text("{not json")
    assert _run(ddr, monkeypatch, ["--metrics-json", str(m), "--design", "3",
                                   "--output", str(tmp_path / "o.txt")]) == 2
    assert "failed to parse" in capsys.readouterr().err


@pytest.mark.logic
def test_cli_rejects_an_unknown_design(ddr, tmp_path, monkeypatch, capsys):
    m = _metrics(tmp_path, _entry())
    assert _run(ddr, monkeypatch, ["--metrics-json", str(m), "--design", "99",
                                   "--output", str(tmp_path / "o.txt")]) == 2
    assert "ERROR" in capsys.readouterr().err


@pytest.mark.logic
def test_cli_rejects_an_underivable_entry(ddr, tmp_path, monkeypatch, capsys):
    m = _metrics(tmp_path, _entry(design=()))
    assert _run(ddr, monkeypatch, ["--metrics-json", str(m), "--design", "3",
                                   "--output", str(tmp_path / "o.txt")]) == 2
    assert "ERROR" in capsys.readouterr().err


@pytest.mark.logic
def test_cli_runs_the_cross_check_when_asked(ddr, tmp_path, monkeypatch, capsys):
    m = _metrics(tmp_path, _entry())
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"true_interface_idx": [2, 3]}))
    assert _run(ddr, monkeypatch, [
        "--metrics-json", str(m), "--design", "3",
        "--output", str(tmp_path / "o.txt"),
        "--cross-check-plan", str(plan)]) == 0
    assert "cross-check" in capsys.readouterr().err
