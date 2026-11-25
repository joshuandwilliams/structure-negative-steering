"""The YAML this repo writes has to match where Boltz actually reads keys.

Boltz 2.2.1 parses an input YAML in `boltz/data/parse/schema.py`. Two facts
from that file decide the shape of everything here.

`parse_boltz_schema` reads templates from the schema root and nowhere else::

    template_schema = schema.get("templates", [])      # schema.py:1603

and a chain entry is inspected only for its entity type::

    for item in schema["sequences"]:                    # schema.py:1018
        entity_type = next(iter(item.keys())).lower()   # schema.py:1020

So the only key of a chain entry that is ever looked at is the one naming the
entity, `protein`. Anything else placed inside the entry is dropped, and no
error is raised, because nothing iterates the entry's keys to validate them.

This repo emitted `templates:` inside the effector's chain entry until
2026-11-25, so every prediction it had produced ran with no template. Boltz's
own record of the run says so: both committed fixtures under
`tests/characterization/fixtures/single_run/.../processed/records/input.json`
carry `"templates": []`.

These tests pin the root placement. A regression would again be silent at run
time, which is the whole reason to assert it here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin"))

import boltz2_negative_steering as eng  # noqa: E402

REC = "MKWVTAAGKQ"
EFF = "QDTLYFCNER"


def _write(tmp_path, **kw):
    """Emit one YAML and hand back the parsed document."""
    path = eng.write_boltz_yaml(
        tmp_path, REC, EFF, "A", "B", **kw)
    return yaml.safe_load(path.read_text())


@pytest.mark.logic
def test_the_template_is_a_root_key_not_a_chain_key(tmp_path):
    """`schema.get("templates", [])` reads the root, so the block has to be
    a sibling of `sequences:`. The second assertion is the actual bug: a
    block under the chain entry parses as valid YAML and is then ignored."""
    cif = tmp_path / "eff.cif"
    cif.write_text("data_x\n")
    doc = _write(tmp_path, effector_template_cif=cif)

    assert "templates" in doc
    for entry in doc["sequences"]:
        assert list(entry) == ["protein"]
        assert "templates" not in entry["protein"]


@pytest.mark.logic
def test_the_template_entry_carries_every_key_boltz_reads(tmp_path):
    """Boltz reads `cif`, `chain_id`, `template_id`, `force` and
    `threshold` from each template entry. `chain_id` is the chain in this
    prediction and `template_id` the chain inside the CIF, and
    `extract_effector_template_cif` relabels the CIF chain to the
    prediction's effector chain, so here the two are both B."""
    cif = tmp_path / "eff.cif"
    cif.write_text("data_x\n")
    doc = _write(tmp_path, effector_template_cif=cif,
                 template_threshold=1.0)

    assert len(doc["templates"]) == 1
    entry = doc["templates"][0]
    assert entry["cif"] == str(cif.resolve())
    assert entry["chain_id"] == "B"
    assert entry["template_id"] == "B"
    assert entry["force"] is True
    assert entry["threshold"] == 1.0


@pytest.mark.logic
def test_force_is_never_emitted_without_a_threshold(tmp_path):
    """boltz 2.2.1 schema.py:1694-1700 raises when `force` is set and
    `threshold` is absent, so the two keys have to travel together."""
    cif = tmp_path / "eff.cif"
    cif.write_text("data_x\n")
    for threshold in (0.5, 1.0, 4.0):
        doc = _write(tmp_path, effector_template_cif=cif,
                     template_threshold=threshold)
        entry = doc["templates"][0]
        assert entry["force"] is True
        assert entry["threshold"] == threshold


@pytest.mark.logic
def test_no_template_key_appears_when_the_template_is_disabled(tmp_path):
    """`--no-effector-template` passes None. An empty `templates: []` would
    also be accepted by Boltz, but omitting the key keeps the sequence-only
    YAML identical to what the benchmarking repo writes."""
    doc = _write(tmp_path)
    assert "templates" not in doc


@pytest.mark.logic
def test_the_template_does_not_disturb_the_constraints_block(tmp_path):
    """Both are root keys, so emitting one must not nest the other. The
    constraint rendering is exercised elsewhere; this only pins that the
    two blocks stay siblings."""
    cif = tmp_path / "eff.cif"
    cif.write_text("data_x\n")
    doc = _write(tmp_path, effector_template_cif=cif,
                 constraints_block=[
                     "  - contact:",
                     "      token1: [A, 1]",
                     "      token2: [B, 1]",
                     "      max_distance: 6.0",
                     "      force: false",
                 ])

    assert set(doc) == {"version", "sequences", "templates", "constraints"}
    assert doc["constraints"][0]["contact"]["max_distance"] == 6.0


@pytest.mark.logic
def test_the_receptor_never_gets_a_template(tmp_path):
    """Only the receptor is mutated, so templating it would pin the surface
    the steering is supposed to move. Every template entry names the
    effector chain."""
    cif = tmp_path / "eff.cif"
    cif.write_text("data_x\n")
    doc = _write(tmp_path, effector_template_cif=cif)

    chains = {e["chain_id"] for e in doc["templates"]}
    assert chains == {"B"}
