"""Tests for bin/derive_true_interface.py.

This writes the protected set. Negative steering mutates the receptor's
predicted contacts minus whatever this file names, so an index that is one out
tells the engine to mutate a residue it was supposed to keep, and the run
reports success either way.

The derivation is an intersection of two lists written by different parts of
RFDiffusion's filter step. The module's own guard for the case that matters,
a contact residue that does not appear in the position order, is the one that
says the two lists describe different atom walks. Those are the assertions
here, along with the 0-based/1-based boundary that the cross-check crosses.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE = REPO_ROOT / "bin" / "derive_true_interface.py"


@pytest.fixture(scope="module")
def dti():
    spec = importlib.util.spec_from_file_location("derive_true_interface", MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _entry(contacts=(12, 14), order=(10, 11, 12, 13, 14), name="design_3"):
    return {"design": name,
            "receptor_contact_residues": list(contacts),
            "receptor_position_order": list(order)}


def _metrics(tmp_path, *entries, filename="rfdiffusion_metrics.json"):
    p = tmp_path / filename
    p.write_text(json.dumps({"designs": list(entries)}))
    return p


# ── identifying a design ─────────────────────────────────────────────────────

@pytest.mark.logic
@pytest.mark.parametrize("raw,bare", [
    ("design_3", "design_3"), ("design_3.pdb", "design_3"),
    ("3", "design_3"), (" 3 ", "design_3"), (3, "design_3"),
])
def test_design_ids_normalise_to_one_form(dti, raw, bare):
    got = dti.normalise_design_id(raw)
    assert got == {"bare": bare, "with_pdb": f"{bare}.pdb"}


@pytest.mark.logic
@pytest.mark.parametrize("key", ["design_3", "design_3.pdb"])
@pytest.mark.parametrize("asked", ["design_3", "design_3.pdb", "3"])
def test_any_spelling_finds_the_entry(dti, key, asked):
    found = dti.find_design_entry({"designs": [_entry(name=key)]}, asked)
    assert found["design"] == key


@pytest.mark.logic
def test_the_name_field_is_accepted_as_well_as_design(dti):
    found = dti.find_design_entry(
        {"designs": [{"name": "design_7", "receptor_contact_residues": [1]}]},
        "design_7")
    assert found["name"] == "design_7"


@pytest.mark.logic
def test_a_missing_designs_array_says_what_it_got(dti):
    with pytest.raises(ValueError, match="no top-level 'designs' array"):
        dti.find_design_entry({"designs": {}}, "design_3")


@pytest.mark.logic
def test_an_unmatched_design_lists_what_is_available(dti):
    metrics = {"designs": [_entry(name=f"design_{i}") for i in range(3)]}
    with pytest.raises(KeyError) as exc:
        dti.find_design_entry(metrics, "design_9")
    msg = str(exc.value)
    assert "design_9" in msg and "design_0" in msg


@pytest.mark.logic
def test_a_long_design_list_is_truncated_with_a_count(dti):
    metrics = {"designs": [_entry(name=f"design_{i}") for i in range(25)]}
    with pytest.raises(KeyError) as exc:
        dti.find_design_entry(metrics, "design_99")
    assert "(25 total)" in str(exc.value)


# ── the derivation itself ────────────────────────────────────────────────────

@pytest.mark.logic
def test_indices_are_positions_in_the_order_not_residue_numbers(dti):
    """The whole point of the transform: 12 and 14 are residue numbers, and
    what comes back is where they sit in the walk."""
    assert dti.derive_positional_indices(
        _entry(contacts=(12, 14), order=(10, 11, 12, 13, 14))) == [2, 4]


@pytest.mark.logic
def test_indices_come_back_sorted_whatever_order_the_contacts_were_in(dti):
    assert dti.derive_positional_indices(
        _entry(contacts=(14, 10, 12), order=(10, 11, 12, 13, 14))) == [0, 2, 4]


@pytest.mark.logic
@pytest.mark.parametrize("field", ["receptor_contact_residues",
                                   "receptor_position_order"])
def test_a_missing_field_names_itself(dti, field):
    e = _entry()
    del e[field]
    with pytest.raises(ValueError, match=field):
        dti.derive_positional_indices(e)


@pytest.mark.logic
def test_no_contacts_is_an_error_that_explains_the_two_causes(dti):
    with pytest.raises(ValueError, match="too strict|not a binder"):
        dti.derive_positional_indices(_entry(contacts=()))


@pytest.mark.logic
def test_an_empty_position_order_is_an_error(dti):
    with pytest.raises(ValueError, match="cannot derive"):
        dti.derive_positional_indices(_entry(order=()))


@pytest.mark.logic
def test_a_contact_absent_from_the_walk_aborts(dti):
    """Silently dropping it would shrink the protected set, which is the
    failure that lets steering mutate the binding site."""
    with pytest.raises(ValueError) as exc:
        dti.derive_positional_indices(_entry(contacts=(12, 99),
                                             order=(10, 11, 12)))
    msg = str(exc.value)
    assert "99" in msg and "different atom walks" in msg


@pytest.mark.logic
def test_many_missing_contacts_are_truncated(dti):
    with pytest.raises(ValueError) as exc:
        dti.derive_positional_indices(
            _entry(contacts=tuple(range(100, 130)), order=(100, 101)))
    assert "..." in str(exc.value)


# ── the cross-check ──────────────────────────────────────────────────────────

@pytest.mark.logic
def test_cross_check_is_advisory_when_the_file_is_absent(dti, tmp_path, capsys):
    assert dti.cross_check_design_region(tmp_path / "nope.txt", [0, 1]) is True
    assert "not found" in capsys.readouterr().err


@pytest.mark.logic
def test_cross_check_compares_in_one_based_space(dti, tmp_path, capsys):
    """The interface is 0-based and the region file is 1-based. Index 0 is
    position 1, so a region of 1-5 contains it."""
    f = tmp_path / "region.txt"
    f.write_text("# a comment\n1-5\n")
    assert dti.cross_check_design_region(f, [0, 1, 2]) is True
    assert "subset" in capsys.readouterr().err


@pytest.mark.logic
def test_cross_check_reports_positions_outside_the_region(dti, tmp_path, capsys):
    f = tmp_path / "region.txt"
    f.write_text("1-3\n")
    assert dti.cross_check_design_region(f, [0, 1, 9]) is False
    err = capsys.readouterr().err
    assert "OUTSIDE" in err and "10" in err      # index 9 is position 10


@pytest.mark.logic
def test_cross_check_reads_commas_ranges_and_bare_numbers(dti, tmp_path):
    f = tmp_path / "region.txt"
    f.write_text("1,2, 3-5\n6\n")
    assert dti.cross_check_design_region(f, [0, 1, 2, 3, 4, 5]) is True


@pytest.mark.logic
def test_cross_check_skips_unparseable_tokens(dti, tmp_path):
    f = tmp_path / "region.txt"
    f.write_text("1-3, x-y, notanint, 4\n")
    assert dti.cross_check_design_region(f, [0, 3]) is True


@pytest.mark.logic
def test_a_region_file_with_nothing_parseable_is_skipped(dti, tmp_path, capsys):
    f = tmp_path / "region.txt"
    f.write_text("# only a comment\n\n")
    assert dti.cross_check_design_region(f, [0]) is True
    assert "no parseable positions" in capsys.readouterr().err


@pytest.mark.logic
def test_an_unreadable_region_file_is_skipped(dti, tmp_path, capsys):
    d = tmp_path / "region.txt"
    d.mkdir()                                     # a directory, not a file
    assert dti.cross_check_design_region(d, [0]) is True
    assert "failed to read" in capsys.readouterr().err


# ── output and the CLI ───────────────────────────────────────────────────────

@pytest.mark.logic
def test_output_carries_its_provenance_then_one_index_per_line(dti):
    out = dti.format_output([2, 4], Path("/m.json"), "design_3", 2)
    head = [ln for ln in out.splitlines() if ln.startswith("#")]
    body = [ln for ln in out.splitlines() if not ln.startswith("#")]
    assert any("/m.json" in ln for ln in head)
    assert any("design_3" in ln for ln in head)
    assert any("0-based" in ln for ln in head)
    assert body == ["2", "4"]


def _run(dti, monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["derive_true_interface.py"] + argv)
    return dti.main()


@pytest.mark.logic
def test_cli_writes_the_indices(dti, tmp_path, monkeypatch):
    m = _metrics(tmp_path, _entry())
    out = tmp_path / "iface.txt"
    assert _run(dti, monkeypatch, ["--metrics-json", str(m),
                                   "--design", "3", "--output", str(out)]) == 0
    body = [ln for ln in out.read_text().splitlines() if not ln.startswith("#")]
    assert body == ["2", "4"]


@pytest.mark.logic
def test_cli_rejects_a_missing_metrics_file(dti, tmp_path, monkeypatch, capsys):
    assert _run(dti, monkeypatch, ["--metrics-json", str(tmp_path / "no.json"),
                                   "--design", "3",
                                   "--output", str(tmp_path / "o.txt")]) == 2
    assert "not found" in capsys.readouterr().err


@pytest.mark.logic
def test_cli_rejects_unparseable_json(dti, tmp_path, monkeypatch, capsys):
    m = tmp_path / "m.json"
    m.write_text("{not json")
    assert _run(dti, monkeypatch, ["--metrics-json", str(m), "--design", "3",
                                   "--output", str(tmp_path / "o.txt")]) == 2
    assert "failed to parse" in capsys.readouterr().err


@pytest.mark.logic
def test_cli_rejects_an_unknown_design(dti, tmp_path, monkeypatch, capsys):
    m = _metrics(tmp_path, _entry())
    assert _run(dti, monkeypatch, ["--metrics-json", str(m), "--design", "99",
                                   "--output", str(tmp_path / "o.txt")]) == 2
    assert "ERROR" in capsys.readouterr().err


@pytest.mark.logic
def test_cli_rejects_a_derivation_that_cannot_run(dti, tmp_path, monkeypatch,
                                                  capsys):
    m = _metrics(tmp_path, _entry(contacts=()))
    assert _run(dti, monkeypatch, ["--metrics-json", str(m), "--design", "3",
                                   "--output", str(tmp_path / "o.txt")]) == 2
    assert "ERROR" in capsys.readouterr().err


@pytest.mark.logic
def test_cli_runs_the_cross_check_when_asked(dti, tmp_path, monkeypatch, capsys):
    m = _metrics(tmp_path, _entry())
    region = tmp_path / "region.txt"
    region.write_text("1-10\n")
    assert _run(dti, monkeypatch, [
        "--metrics-json", str(m), "--design", "3",
        "--output", str(tmp_path / "o.txt"),
        "--cross-check-design-region", str(region)]) == 0
    assert "cross-check" in capsys.readouterr().err
