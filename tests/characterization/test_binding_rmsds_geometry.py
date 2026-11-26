"""Geometry tests for compute_binding_rmsds, on constructed placements.

receptor_aligned_effector_rmsd is the gating metric for the whole project. A
pose passes at under 5 A and fails above it, and every figure, every tier and
every verdict downstream rests on that one number.

It is computed by fitting the receptors with Kabsch and applying that
transform rigidly to the effector. Two things can go wrong there and neither
shows up in a test that only checks the output exists:

  - The rotation can be applied on the wrong side. R @ v and v @ R are both
    valid-looking matrix products, and the wrong one throws an effector tens
    of angstroms off while still returning a float.
  - The receptor fit can leak into the effector measurement, so a complex
    that was merely rotated in space reports a displaced effector.

So these tests construct the answer rather than observing it. A reference
complex is read from data/reference_complexes/, a known rigid transform is
applied to one chain or to both, and the metric is asserted against the
displacement that was applied. A whole-complex transform must report zero,
because aligning the receptors undoes it. An effector-only translation of
d angstroms must report d.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE = REPO_ROOT / "data" / "reference_complexes" / "6G10.pdb"


def _deps() -> bool:
    return all(importlib.util.find_spec(m) is not None
               for m in ("numpy", "Bio"))


pytestmark = pytest.mark.skipif(
    not _deps(), reason="needs numpy and biopython (pip install -e '.[runtime]')")


@pytest.fixture(scope="module")
def bns():
    sys.path.insert(0, str(REPO_ROOT / "bin"))
    spec = importlib.util.spec_from_file_location(
        "boltz2_negative_steering", REPO_ROOT / "bin" / "boltz2_negative_steering.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _rotation(axis: str, degrees: float):
    """A right-handed rotation matrix about one axis."""
    import numpy as np
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _write_transformed(src: Path, dst: Path, chains, R=None, t=None,
                       about_centroid=False) -> Path:
    """Copy a PDB, applying `R` then `t` to the named chains' atoms.

    Columns are rewritten in place at the fixed PDB offsets, so every other
    field, including the residue numbering the alignment pairs on, is
    untouched.
    """
    import numpy as np
    R = np.eye(3) if R is None else np.asarray(R, dtype=float)
    t = np.zeros(3) if t is None else np.asarray(t, dtype=float)
    lines = src.read_text().splitlines()

    centre = np.zeros(3)
    if about_centroid:
        pts = [[float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]
               for ln in lines
               if ln.startswith(("ATOM", "HETATM")) and ln[21] in chains]
        centre = np.asarray(pts, dtype=float).mean(axis=0)

    out = []
    for ln in lines:
        if not ln.startswith(("ATOM", "HETATM")) or ln[21] not in chains:
            out.append(ln)
            continue
        v = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        v = R @ (v - centre) + centre + t
        out.append(f"{ln[:30]}{v[0]:8.3f}{v[1]:8.3f}{v[2]:8.3f}{ln[54:]}")
    dst.write_text("\n".join(out) + "\n")
    return dst


# ── the constructed placements ───────────────────────────────────────────────

@pytest.mark.logic
def test_a_structure_against_itself_is_zero_everywhere(bns, tmp_path):
    r = bns.compute_binding_rmsds(str(REFERENCE), str(REFERENCE),
                                  rec_chain="A", eff_chain="B")
    assert r["receptor_aligned_effector_rmsd"] == pytest.approx(0.0, abs=1e-3)
    assert r["independent_receptor_rmsd"] == pytest.approx(0.0, abs=1e-3)
    assert r["independent_effector_rmsd"] == pytest.approx(0.0, abs=1e-3)


@pytest.mark.logic
def test_a_rigid_move_of_the_whole_complex_reports_zero(bns, tmp_path):
    """This is the test that catches the rotation being applied on the wrong
    side. Aligning the receptors undoes a whole-complex transform exactly, so
    the effector must come back to zero. Under R @ v where v @ R was meant,
    the receptor still fits and the effector lands tens of angstroms away.
    """
    moved = _write_transformed(REFERENCE, tmp_path / "moved.pdb",
                               chains={"A", "B"},
                               R=_rotation("y", 37.0), t=[12.0, -5.0, 3.0])
    r = bns.compute_binding_rmsds(str(moved), str(REFERENCE),
                                  rec_chain="A", eff_chain="B")
    assert r["receptor_aligned_effector_rmsd"] == pytest.approx(0.0, abs=1e-2)
    assert r["independent_receptor_rmsd"] == pytest.approx(0.0, abs=1e-2)
    assert r["independent_effector_rmsd"] == pytest.approx(0.0, abs=1e-2)


@pytest.mark.logic
@pytest.mark.parametrize("d", [2.0, 7.5, 25.0])
def test_an_effector_moved_by_d_reports_d(bns, tmp_path, d):
    """A pure translation of the effector alone, receptor untouched. Every
    paired atom moves by exactly d, so the RMSD is exactly d."""
    moved = _write_transformed(REFERENCE, tmp_path / f"eff_{d}.pdb",
                               chains={"B"}, t=[d, 0.0, 0.0])
    r = bns.compute_binding_rmsds(str(moved), str(REFERENCE),
                                  rec_chain="A", eff_chain="B")
    assert r["receptor_aligned_effector_rmsd"] == pytest.approx(d, abs=1e-2)
    # Its own fold is unchanged, and so is the receptor's.
    assert r["independent_effector_rmsd"] == pytest.approx(0.0, abs=1e-2)
    assert r["independent_receptor_rmsd"] == pytest.approx(0.0, abs=1e-2)


@pytest.mark.logic
def test_the_threshold_separates_a_pass_from_a_fail(bns, tmp_path):
    """The 5 A line is what every tier and figure rests on, so a displacement
    either side of it must land either side of it."""
    near = _write_transformed(REFERENCE, tmp_path / "near.pdb",
                              chains={"B"}, t=[4.0, 0.0, 0.0])
    far = _write_transformed(REFERENCE, tmp_path / "far.pdb",
                             chains={"B"}, t=[6.0, 0.0, 0.0])
    a = bns.compute_binding_rmsds(str(near), str(REFERENCE), "A", "B")
    b = bns.compute_binding_rmsds(str(far), str(REFERENCE), "A", "B")
    assert a["receptor_aligned_effector_rmsd"] < 5.0
    assert b["receptor_aligned_effector_rmsd"] > 5.0


@pytest.mark.logic
def test_an_effector_rotated_in_place_moves_without_deforming(bns, tmp_path):
    """Rotating the effector about its own centroid leaves its fold intact
    but changes the binding mode, which is the case the gating metric exists
    to catch and the independent effector RMSD must not."""
    spun = _write_transformed(REFERENCE, tmp_path / "spun.pdb",
                              chains={"B"}, R=_rotation("z", 90.0),
                              about_centroid=True)
    r = bns.compute_binding_rmsds(str(spun), str(REFERENCE), "A", "B")
    assert r["receptor_aligned_effector_rmsd"] > 5.0
    assert r["independent_effector_rmsd"] == pytest.approx(0.0, abs=1e-2)
    assert r["independent_receptor_rmsd"] == pytest.approx(0.0, abs=1e-2)


@pytest.mark.logic
def test_moving_the_receptor_alone_also_displaces_the_effector(bns, tmp_path):
    """Aligning the receptors undoes the receptor's own move, which then
    carries the effector the same distance the other way."""
    moved = _write_transformed(REFERENCE, tmp_path / "rec.pdb",
                               chains={"A"}, t=[9.0, 0.0, 0.0])
    r = bns.compute_binding_rmsds(str(moved), str(REFERENCE), "A", "B")
    assert r["independent_receptor_rmsd"] == pytest.approx(0.0, abs=1e-2)
    assert r["receptor_aligned_effector_rmsd"] == pytest.approx(9.0, abs=1e-2)


@pytest.mark.logic
def test_chain_arguments_are_honoured_independently(bns, tmp_path):
    """The prediction's chains and the reference's are named separately, so a
    swap would silently compare the receptor against the effector."""
    r = bns.compute_binding_rmsds(str(REFERENCE), str(REFERENCE),
                                  rec_chain="A", eff_chain="B",
                                  des_rec_chain="A", des_eff_chain="B")
    assert r["receptor_aligned_effector_rmsd"] == pytest.approx(0.0, abs=1e-3)


# ── the refusals ─────────────────────────────────────────────────────────────

@pytest.mark.logic
@pytest.mark.parametrize("which", ["pred", "design"])
def test_a_missing_file_returns_all_none(bns, tmp_path, capsys, which):
    missing = str(tmp_path / "nope.pdb")
    args = (missing, str(REFERENCE)) if which == "pred" else (str(REFERENCE), missing)
    r = bns.compute_binding_rmsds(*args, rec_chain="A", eff_chain="B")
    assert set(r.values()) == {None}
    assert "not found" in capsys.readouterr().out


@pytest.mark.logic
@pytest.mark.parametrize("kw", [
    {"rec_chain": "Z"}, {"eff_chain": "Z"},
    {"des_rec_chain": "Z"}, {"des_eff_chain": "Z"},
])
def test_a_chain_that_is_not_there_returns_all_none(bns, capsys, kw):
    """Returning zeros here would read as a perfect pose."""
    base = {"rec_chain": "A", "eff_chain": "B"}
    base.update(kw)
    r = bns.compute_binding_rmsds(str(REFERENCE), str(REFERENCE), **base)
    assert set(r.values()) == {None}
    assert "not found" in capsys.readouterr().out


@pytest.mark.logic
def test_an_empty_path_returns_all_none(bns):
    assert set(bns.compute_binding_rmsds("", str(REFERENCE),
                                         rec_chain="A", eff_chain="B").values()) \
        == {None}
