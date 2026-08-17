"""derive_true_interface and derive_design_region, driven in process.

Both read `rfdiffusion_metrics.json` and write an index file the engine consumes.
They are 277 statements and were untested. Neither needs a GPU.

They exist because a designed receptor chain is Ca-only, so heavy-atom contacts
cannot be computed and the contact set has to be precomputed upstream. That is
what makes them different from scripts/prepare_complex_inputs.py, which reads a
solved complex.

The two write different coordinate systems, which is the thing most likely to
be got wrong. true_interface is 0-based, design_region is 1-based.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin"))

import derive_design_region as ddr  # noqa: E402
import derive_true_interface as dti  # noqa: E402

METRICS = (Path(__file__).parent / "fixtures" / "rfdiffusion"
           / "rfdiffusion_metrics.json")

needs_fixture = pytest.mark.skipif(
    not METRICS.is_file(), reason="rfdiffusion_metrics.json fixture not present")


def _main(mod, *args: str) -> int:
    with mock.patch.object(sys, "argv", [mod.__name__, *args]):
        rc = mod.main()
    return 0 if rc is None else rc


def _indices(path: Path) -> list[int]:
    """Parse an index file. derive_design_region emits ranges like 33-46,74-79."""
    out: list[int] = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        for tok in line.replace(",", " ").split():
            if "-" in tok[1:]:
                lo, hi = tok.split("-", 1)
                out.extend(range(int(lo), int(hi) + 1))
            elif tok.lstrip("-").isdigit():
                out.append(int(tok))
    return sorted(set(out))


@pytest.fixture(scope="module")
def a_design() -> str:
    data = json.loads(METRICS.read_text())
    designs = data["designs"] if isinstance(data, dict) else data
    first = designs[0] if isinstance(designs, list) else next(iter(designs.values()))
    return str(first.get("design") or first.get("name"))


# ── derive_true_interface ────────────────────────────────────────────────────

@pytest.mark.local_integration
@needs_fixture
def test_true_interface_is_written_and_zero_based(tmp_path, a_design):
    out = tmp_path / "iface.txt"
    assert _main(dti, "--metrics-json", str(METRICS),
                 "--design", a_design, "--output", str(out)) == 0
    idx = _indices(out)
    assert idx, "no interface residues written"
    assert min(idx) >= 0, "0-based indices must not be negative"
    assert idx == sorted(set(idx)), "indices should be sorted and unique"


@pytest.mark.local_integration
@needs_fixture
def test_true_interface_header_records_its_coordinate_system(tmp_path, a_design):
    """The file is consumed by a flag that assumes 0-based, so it must say so."""
    out = tmp_path / "iface.txt"
    _main(dti, "--metrics-json", str(METRICS), "--design", a_design,
          "--output", str(out))
    header = "\n".join(x for x in out.read_text().splitlines() if x.startswith("#"))
    assert "0-based" in header, f"header does not state the index base:\n{header}"


@pytest.mark.local_integration
@needs_fixture
@pytest.mark.parametrize("suffix", ["", ".pdb"])
def test_design_can_be_named_with_or_without_the_extension(tmp_path, a_design, suffix):
    stem = a_design[:-4] if a_design.endswith(".pdb") else a_design
    out = tmp_path / f"iface{suffix or 'bare'}.txt"
    assert _main(dti, "--metrics-json", str(METRICS),
                 "--design", stem + suffix, "--output", str(out)) == 0
    assert _indices(out)


@pytest.mark.local_integration
@needs_fixture
def test_unknown_design_is_rejected(tmp_path, capsys):
    """A typo must fail loudly rather than writing an empty interface."""
    out = tmp_path / "iface.txt"
    assert _main(dti, "--metrics-json", str(METRICS),
                 "--design", "design_does_not_exist", "--output", str(out)) == 2
    assert "No design entry matching" in capsys.readouterr().err
    assert not out.exists(), "a rejected design still wrote an output file"


@pytest.mark.local_unit
def test_absent_metrics_json_is_rejected(tmp_path, capsys):
    assert _main(dti, "--metrics-json", str(tmp_path / "nope.json"),
                 "--design", "design_0", "--output", str(tmp_path / "o.txt")) == 2
    assert "not found" in capsys.readouterr().err


# ── derive_design_region ─────────────────────────────────────────────────────

@pytest.mark.local_integration
@needs_fixture
def test_design_region_is_written_and_one_based(tmp_path, a_design):
    out = tmp_path / "region.txt"
    assert _main(ddr, "--metrics-json", str(METRICS),
                 "--design", a_design, "--output", str(out)) == 0
    idx = _indices(out)
    assert idx, "no design-region positions written"
    assert min(idx) >= 1, "1-based indices must start at 1, not 0"


@pytest.mark.local_integration
@needs_fixture
def test_design_region_header_records_its_coordinate_system(tmp_path, a_design):
    out = tmp_path / "region.txt"
    _main(ddr, "--metrics-json", str(METRICS), "--design", a_design,
          "--output", str(out))
    header = "\n".join(x for x in out.read_text().splitlines() if x.startswith("#"))
    assert "1-based" in header, f"header does not state the index base:\n{header}"


@pytest.mark.local_integration
@needs_fixture
def test_no_ranges_writes_every_position_individually(tmp_path, a_design):
    """Ranges are a display convenience; --no-ranges must not change the set."""
    ranged = tmp_path / "ranged.txt"
    flat = tmp_path / "flat.txt"
    _main(ddr, "--metrics-json", str(METRICS), "--design", a_design,
          "--output", str(ranged))
    _main(ddr, "--metrics-json", str(METRICS), "--design", a_design,
          "--output", str(flat), "--no-ranges")
    assert flat.is_file()
    assert _indices(ranged) == _indices(flat), \
        "--no-ranges changed which positions are in the design region"


@pytest.mark.local_integration
@needs_fixture
def test_the_two_derivations_use_different_index_bases(tmp_path, a_design):
    """0-based interface against 1-based design region.

    Confusing the two would shift the protected set by one residue, which no
    downstream check would catch.
    """
    iface = tmp_path / "iface.txt"
    region = tmp_path / "region.txt"
    _main(dti, "--metrics-json", str(METRICS), "--design", a_design,
          "--output", str(iface))
    _main(ddr, "--metrics-json", str(METRICS), "--design", a_design,
          "--output", str(region))

    assert "0-based" in iface.read_text()
    assert "1-based" in region.read_text()


@pytest.mark.local_integration
@needs_fixture
def test_existing_output_is_overwritten_not_appended(tmp_path, a_design):
    out = tmp_path / "iface.txt"
    out.write_text("# stale content\n999999\n")
    _main(dti, "--metrics-json", str(METRICS), "--design", a_design,
          "--output", str(out))
    assert 999999 not in _indices(out), "stale content survived the rewrite"
