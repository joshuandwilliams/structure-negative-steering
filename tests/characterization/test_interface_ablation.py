"""Tests for the interface ablation in bin/prepare_complex_inputs.py.

The ablation is the control for the constrained arm, so a silent no-op here
would read as the method working. The permutation and the alanine substitution
are pure functions of a sequence and an index list, so most of this needs no
structure file, and the one CLI test asserts that the ablated sequence is what
lands in receptor.fasta while the interface indices stay untouched.
"""

from __future__ import annotations

import collections
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PREPARE = REPO_ROOT / "bin" / "prepare_complex_inputs.py"
REFERENCE = REPO_ROOT / "data" / "reference_complexes" / "7QZD.pdb"

SEQ = "MKVLAAGGWWDDEEKKRRTT"
IDX = [2, 3, 4, 10, 11, 12]


def _deps_available() -> bool:
    if importlib.util.find_spec("numpy") is None:
        return False
    return any(importlib.util.find_spec(m) is not None for m in ("gemmi", "Bio"))


needs_deps = pytest.mark.skipif(
    not _deps_available(),
    reason="needs numpy plus gemmi or biopython (pip install -e '.[experiments]')")


@pytest.fixture(scope="module")
def prep():
    spec = importlib.util.spec_from_file_location("prepare_complex_inputs", PREPARE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── alanine ──────────────────────────────────────────────────────────────────

@pytest.mark.logic
def test_alanine_hits_every_selected_position(prep):
    out = prep.ablate_alanine(SEQ, IDX)
    assert all(out[i] == "A" for i in IDX)


@pytest.mark.logic
def test_alanine_leaves_everything_else_alone(prep):
    out = prep.ablate_alanine(SEQ, IDX)
    assert len(out) == len(SEQ)
    assert all(out[i] == SEQ[i] for i in range(len(SEQ)) if i not in set(IDX))


# ── scramble ─────────────────────────────────────────────────────────────────

@pytest.mark.logic
def test_scramble_preserves_length_and_composition(prep):
    out = prep.ablate_scramble(SEQ, IDX, seed=7)
    assert len(out) == len(SEQ)
    assert collections.Counter(out) == collections.Counter(SEQ)


@pytest.mark.logic
def test_scramble_leaves_everything_else_alone(prep):
    out = prep.ablate_scramble(SEQ, IDX, seed=7)
    assert all(out[i] == SEQ[i] for i in range(len(SEQ)) if i not in set(IDX))


@pytest.mark.logic
def test_scramble_is_deterministic_in_the_seed(prep):
    assert prep.ablate_scramble(SEQ, IDX, seed=7) == prep.ablate_scramble(SEQ, IDX, seed=7)
    assert prep.ablate_scramble(SEQ, IDX, seed=7) != prep.ablate_scramble(SEQ, IDX, seed=8)


@pytest.mark.logic
def test_scramble_does_not_return_the_input_when_it_can_avoid_it(prep):
    """A permutation that happened to be the identity would be a silent no-op."""
    for seed in range(40):
        out = prep.ablate_scramble(SEQ, IDX, seed=seed)
        assert any(out[i] != SEQ[i] for i in IDX), f"seed {seed} changed nothing"


@pytest.mark.logic
def test_scramble_of_identical_residues_is_a_no_op(prep):
    """Nothing can be done with one distinct residue, and it must not hang trying."""
    seq = "MAAAK"
    assert prep.ablate_scramble(seq, [1, 2, 3], seed=1) == seq


@pytest.mark.logic
def test_empty_selection_changes_nothing(prep):
    assert prep.ablate_scramble(SEQ, [], seed=1) == SEQ
    assert prep.ablate_alanine(SEQ, []) == SEQ


@pytest.mark.logic
def test_default_seed_is_per_target_and_stable(prep):
    a = prep.default_ablation_seed(Path("/wherever/7QZD.pdb"))
    b = prep.default_ablation_seed(Path("/elsewhere/7QZD.pdb"))
    c = prep.default_ablation_seed(Path("/wherever/9IP6.pdb"))
    assert a == b
    assert a != c


# ── the CLI, against a real complex ──────────────────────────────────────────

@needs_deps
@pytest.mark.parametrize("mode", ["alanine", "scrambled"])
@pytest.mark.logic
def test_cli_writes_the_ablated_receptor_and_keeps_the_interface(prep, tmp_path, mode):
    base = tmp_path / "base"
    abl = tmp_path / mode
    common = ["--complex", str(REFERENCE), "--receptor-chain", "A",
              "--effector-chain", "B", "--derive", "--contact-cutoff", "5.0",
              "--design-region", "none"]
    assert prep.main(common + ["--outdir", str(base)]) == 0
    assert prep.main(common + ["--outdir", str(abl), "--ablate", mode]) == 0

    native = (base / "receptor.fasta").read_text().splitlines()[1]
    ablated = (abl / "receptor.fasta").read_text().splitlines()[1]
    idx = [int(ln) for ln in (abl / "true_interface.txt").read_text().splitlines()
           if not ln.startswith("#")]

    assert (base / "true_interface.txt").read_text().splitlines()[-1] == \
        (abl / "true_interface.txt").read_text().splitlines()[-1]
    assert len(ablated) == len(native)
    assert ablated != native
    assert all(ablated[i] == native[i] for i in range(len(native)) if i not in set(idx))
    if mode == "alanine":
        assert all(ablated[i] == "A" for i in idx)
    else:
        assert collections.Counter(ablated) == collections.Counter(native)
    assert (abl / "interface_ablation.txt").is_file()
    assert not (base / "interface_ablation.txt").exists()


@needs_deps
@pytest.mark.logic
def test_ablation_record_lists_every_interface_position(prep, tmp_path):
    out = tmp_path / "scrambled"
    assert prep.main(["--complex", str(REFERENCE), "--outdir", str(out),
                      "--receptor-chain", "A", "--effector-chain", "B",
                      "--derive", "--contact-cutoff", "5.0",
                      "--design-region", "none", "--ablate", "scrambled"]) == 0
    idx = [int(ln) for ln in (out / "true_interface.txt").read_text().splitlines()
           if not ln.startswith("#")]
    rows = [ln for ln in (out / "interface_ablation.txt").read_text().splitlines()
            if not ln.startswith("#")]
    assert len(rows) == len(idx)
    assert [int(r.split("\t")[0]) for r in rows] == [i + 1 for i in idx]


# ── the CLI's refusals ───────────────────────────────────────────────────────

@needs_deps
@pytest.mark.logic
def test_ablation_needs_a_non_empty_interface(prep, tmp_path):
    """With no contacts there is nothing to ablate, so the control is empty.

    Returning 0 here would hand the pipeline an unmodified receptor labelled
    as ablated, which reads as the method working.
    """
    out = tmp_path / "none"
    rc = prep.main(["--complex", str(REFERENCE), "--outdir", str(out),
                    "--receptor-chain", "A", "--effector-chain", "B",
                    "--derive", "--contact-cutoff", "0.01",
                    "--design-region", "none", "--ablate", "scrambled"])
    assert rc == 2
    assert not (out / "interface_ablation.txt").exists()


@needs_deps
@pytest.mark.logic
def test_a_stale_ablation_record_is_removed_when_ablation_is_off(prep, tmp_path):
    """An unablated rerun into the same folder must not inherit the record."""
    out = tmp_path / "reused"
    common = ["--complex", str(REFERENCE), "--receptor-chain", "A",
              "--effector-chain", "B", "--derive", "--contact-cutoff", "5.0",
              "--design-region", "none", "--outdir", str(out)]
    assert prep.main(common + ["--ablate", "scrambled"]) == 0
    assert (out / "interface_ablation.txt").is_file()
    assert prep.main(common) == 0
    assert not (out / "interface_ablation.txt").exists()


@needs_deps
@pytest.mark.logic
def test_a_scramble_that_changes_nothing_is_refused(prep, tmp_path, monkeypatch):
    """A permutation of identical residues is a no-op, not a control."""
    out = tmp_path / "noop"
    monkeypatch.setattr(prep, "ablate_scramble", lambda seq, idx, seed: seq)
    rc = prep.main(["--complex", str(REFERENCE), "--outdir", str(out),
                    "--receptor-chain", "A", "--effector-chain", "B",
                    "--derive", "--contact-cutoff", "5.0",
                    "--design-region", "none", "--ablate", "scrambled"])
    assert rc == 2
