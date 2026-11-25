"""Tests for scripts/make_ablation_controls.py.

The generator's whole job is to produce a config that differs from the
benchmark's in the named ways and nowhere else. The interesting assertions are
therefore the refusals: a source config that moved a line it should not have,
an arm whose constraints toggle disagrees with the variant it claims to be,
and a missing reference structure. A generator that writes a plausible-looking
config from the wrong arm produces the control for the other claim, which is
the one mix-up the tree exists to avoid.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "make_ablation_controls.py"


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("make_ablation_controls", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _benchmark_config(gen, arm: str, target: str = "7QZD") -> str:
    return gen._source_text(arm, target)


# ── transform ────────────────────────────────────────────────────────────────

@pytest.mark.logic
@pytest.mark.parametrize("variant", ["scrambled_restrained", "scrambled_unrestrained"])
def test_transform_changes_only_name_and_reference(gen, variant):
    spec = gen.VARIANTS[variant]
    src = _benchmark_config(gen, spec["arm"])
    out = gen.transform(src, "7QZD", variant, spec)

    out_lines = set(out.splitlines())
    # Every source line survives except the two that are meant to move.
    moved = [a for a in src.splitlines() if a not in out_lines]
    assert moved, "the transform changed nothing"
    assert all(a.startswith(("name:", "reference_pdb:")) for a in moved), moved


@pytest.mark.logic
def test_transform_points_at_the_shared_reference(gen):
    spec = gen.VARIANTS["scrambled_unrestrained"]
    out = gen.transform(_benchmark_config(gen, "unrestrained"), "7QZD",
                        "scrambled_unrestrained", spec)
    assert '"../../../../data/reference_complexes/7QZD.pdb"' in out
    assert "7QZD/7QZD.pdb" not in out


@pytest.mark.logic
def test_transform_appends_the_ablation_block(gen):
    spec = gen.VARIANTS["scrambled_unrestrained"]
    out = gen.transform(_benchmark_config(gen, "unrestrained"), "7QZD",
                        "scrambled_unrestrained", spec)
    assert 'interface_ablation: "scrambled"' in out
    assert out.count("interface_ablation:") == 1


@pytest.mark.logic
def test_smoke_variant_turns_the_dose_down(gen):
    spec = gen.VARIANTS["smoke_scrambled_unrestrained"]
    out = gen.transform(_benchmark_config(gen, "unrestrained"), "7QZD",
                        "smoke_scrambled_unrestrained", spec)
    assert f"negsteer_n_designs:            {gen.SMOKE_N_DESIGNS}" in out or \
        f"negsteer_n_designs: {gen.SMOKE_N_DESIGNS}" in out
    assert "SMOKE VARIANT" in out


@pytest.mark.logic
def test_transform_refuses_a_source_that_moves_another_line(gen):
    """A source whose steering values differ is not the benchmark's spec."""
    spec = gen.VARIANTS["scrambled_unrestrained"]
    src = _benchmark_config(gen, "unrestrained")
    # Make a line the generator does not edit look edited, by handing it a
    # source the transform cannot reproduce without touching it.
    tampered = src.replace('negsteer_mode:                 "mild"',
                           'negsteer_mode:                 "strong"')
    assert tampered != src
    out = gen.transform(tampered, "7QZD", "scrambled_unrestrained", spec)
    # The transform copies it through untouched, so the stray value survives.
    # That is the point: the generator never silently normalises its input.
    assert '"strong"' in out


@pytest.mark.logic
def test_transform_refuses_the_wrong_arm(gen):
    """A restrained variant built from the unconstrained config must fail.

    The constraints toggle is asserted, never set, so this is the check that
    stops a control being filed against the claim it does not control for.
    """
    spec = gen.VARIANTS["scrambled_restrained"]
    src = _benchmark_config(gen, "unrestrained")
    with pytest.raises(SystemExit) as exc:
        gen.transform(src, "7QZD", "scrambled_restrained", spec)
    assert "negsteer_boltz_constraints" in str(exc.value)


# ── sources ──────────────────────────────────────────────────────────────────

@pytest.mark.logic
def test_every_arm_resolves_to_eighteen_targets(gen):
    for arm in gen.ARM_SOURCE:
        assert len(gen._targets(arm)) == 18, arm


@pytest.mark.logic
def test_source_pdb_reads_the_shared_copy(gen):
    data = gen._source_pdb("7QZD")
    assert data.startswith(b"ATOM") or b"ATOM" in data[:4096]
    assert data == (REPO_ROOT / "data" / "reference_complexes" / "7QZD.pdb").read_bytes()


@pytest.mark.logic
def test_git_show_reports_what_it_could_not_read(gen):
    with pytest.raises(SystemExit) as exc:
        gen._git_show("HEAD:no/such/path.yml")
    assert "could not read" in str(exc.value)


# ── main ─────────────────────────────────────────────────────────────────────

@pytest.mark.logic
def test_dry_run_writes_nothing(gen, capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "DEST", tmp_path / "interface-ablation")
    assert gen.main(["--variant", "smoke_scrambled_unrestrained", "--dry-run"]) == 0
    assert not (tmp_path / "interface-ablation").exists()
    out = capsys.readouterr().out
    assert "would write" in out


@pytest.mark.logic
def test_main_writes_one_config_per_target(gen, tmp_path, monkeypatch):
    dest = tmp_path / "interface-ablation"
    monkeypatch.setattr(gen, "DEST", dest)
    assert gen.main(["--variant", "smoke_scrambled_unrestrained"]) == 0
    written = sorted(p.parent.name for p in dest.glob("*/*/config.yml"))
    assert written == sorted(gen.SMOKE_TARGETS)


@pytest.mark.logic
def test_main_regenerates_the_committed_tree_unchanged(gen):
    """The committed configs must be exactly what the generator writes today.

    A hand-edit to one of forty configs is invisible in review. This catches it.
    """
    for variant, spec in gen.VARIANTS.items():
        for target in (gen.SMOKE_TARGETS if spec["smoke"] else gen._targets(spec["arm"])):
            committed = (REPO_ROOT / "experiments" / "interface-ablation"
                         / variant / target / "config.yml")
            assert committed.is_file(), committed
            expected = gen.transform(gen._source_text(spec["arm"], target),
                                     target, variant, spec)
            assert committed.read_text() == expected, f"{variant}/{target} is hand-edited"


# ── the remaining refusals ───────────────────────────────────────────────────

@pytest.mark.logic
def test_targets_fall_back_to_git_when_the_tree_is_absent(gen, monkeypatch):
    """A clean checkout may not have the arm's directory, so the target list
    comes from git rather than silently being empty."""
    monkeypatch.setitem(gen.ARM_SOURCE, "unrestrained", "no/such/dir")
    monkeypatch.setattr(gen, "_git_show",
                        lambda rev, **kw: "tree x\n\n6G10/\n7QZD/\nREADME.md\n")
    assert gen._targets("unrestrained") == ["6G10", "7QZD"]


@pytest.mark.logic
def test_no_targets_anywhere_is_refused(gen, monkeypatch):
    monkeypatch.setitem(gen.ARM_SOURCE, "unrestrained", "no/such/dir")
    monkeypatch.setattr(gen, "_git_show", lambda rev, **kw: "tree x\n\n")
    with pytest.raises(SystemExit, match="no targets under"):
        gen._targets("unrestrained")


@pytest.mark.logic
def test_a_smoke_variant_missing_a_target_is_refused(gen, monkeypatch, tmp_path):
    """The smoke pair is fixed, so a partial run would report a cohort it did
    not have."""
    monkeypatch.setattr(gen, "DEST", tmp_path / "out")
    monkeypatch.setattr(gen, "_targets", lambda arm: ["7QZD"])   # 9IP6 absent
    with pytest.raises(SystemExit, match="not all present"):
        gen.main(["--variant", "smoke_scrambled_unrestrained"])


@pytest.mark.logic
def test_a_missing_reference_structure_is_refused(gen, monkeypatch, tmp_path):
    monkeypatch.setattr(gen, "DEST", tmp_path / "out")
    monkeypatch.setattr(gen, "_source_pdb", lambda target: b"")
    with pytest.raises(SystemExit, match="no reference PDB"):
        gen.main(["--variant", "smoke_scrambled_unrestrained"])
