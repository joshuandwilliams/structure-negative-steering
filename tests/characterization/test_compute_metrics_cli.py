"""compute_metrics driven over a synthesised Boltz output directory.

parse_boltz2 and main are the entry points the engine calls after every
prediction. They walk a Boltz output tree, read the PAE, pLDDT and confidence
sidecars, score the pose against the native complex and write one metrics row.
Nothing about that needs a GPU, only a directory shaped the way Boltz shapes
one, which is cheap to build.

The prediction is 6G10 itself, so the pose is exactly right and every geometric
metric has a knowable value. A second case uses the effector reflected through
the receptor centroid, where the pose is wrong by construction.

parse_boltz2 does not compute an RMSD. It reports confidence, contacts,
weighted Jaccard and the intact-core flag. The receptor-aligned effector RMSD
is added later by the collect stage, so the pose is checked here through the
weighted Jaccard instead.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from unittest import mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bin"))

import compute_metrics as cm  # noqa: E402

np = pytest.importorskip("numpy")

GT = (_REPO_ROOT / "experiments" / "benchmarking" / "unconstrained" / "6G10"
      / "6G10.pdb")
needs_pdb = pytest.mark.skipif(not GT.is_file(), reason="6G10 reference absent")

REC_LEN, EFF_LEN = 76, 83
TOTAL = REC_LEN + EFF_LEN


def _wrong_pose(src: Path, dst: Path) -> Path:
    lines = src.read_text().splitlines(keepends=True)
    co = [(float(x[30:38]), float(x[38:46]), float(x[46:54]))
          for x in lines if x.startswith("ATOM") and x[21] == "A"]
    cx, cy, cz = (sum(c[i] for c in co) / len(co) for i in range(3))
    out = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")) and len(line) > 54 and line[21] == "B":
            x = 2 * cx - float(line[30:38])
            y = 2 * cy - float(line[38:46])
            z = 2 * cz - float(line[46:54])
            out.append(f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}")
        else:
            out.append(line)
    dst.write_text("".join(out))
    return dst


def _boltz_output(root: Path, structure: Path, *, inter_pae: float = 3.0) -> Path:
    """A directory shaped the way Boltz-2 shapes one.

    output/predictions/<job>/ holds the model PDB plus pae/, plddt/ and
    confidence/ sidecars, which is what parse_boltz2 walks.
    """
    job = root / "predictions" / "input"
    job.mkdir(parents=True, exist_ok=True)
    shutil.copy(structure, job / "input_model_0.pdb")

    pae = np.full((TOTAL, TOTAL), 1.0, dtype=float)
    pae[:REC_LEN, REC_LEN:] = inter_pae
    pae[REC_LEN:, :REC_LEN] = inter_pae
    (job / "pae").mkdir(exist_ok=True)
    np.savez(job / "pae" / "pae_input_model_0.npz", pae=pae)

    (job / "plddt").mkdir(exist_ok=True)
    np.savez(job / "plddt" / "plddt_input_model_0.npz",
             plddt=np.full(TOTAL, 0.9, dtype=float))

    (job / "confidence").mkdir(exist_ok=True)
    (job / "confidence" / "confidence_input_model_0.json").write_text(json.dumps({
        "confidence_score": 0.77, "ptm": 0.66, "iptm": 0.43,
        "complex_plddt": 0.85, "complex_iplddt": 0.80,
    }))
    return root


def _parse(pred_dir: Path, native: Path | None = None) -> list[dict]:
    return cm.parse_boltz2(
        str(pred_dir), [REC_LEN, EFF_LEN], 10.0, "A", "B",
        None, 5.0, None, str(native) if native else None, 50.0, 5.0, 8.0)


# ── parse_boltz2 over a correct pose ─────────────────────────────────────────

@pytest.mark.local_integration
@needs_pdb
def test_a_boltz_directory_is_parsed_into_metric_rows(tmp_path):
    out = _boltz_output(tmp_path / "out", GT)
    rows = _parse(out, GT)
    assert rows, "parse_boltz2 returned nothing for a well-formed directory"


@pytest.mark.local_integration
@needs_pdb
def test_a_perfect_pose_scores_a_weighted_jaccard_of_one(tmp_path):
    """Scoring the native against itself is the fixed point of the metric."""
    rows = _parse(_boltz_output(tmp_path / "out", GT), GT)
    assert float(rows[0]["weighted_jaccard"]) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.local_integration
@needs_pdb
def test_a_wrong_pose_scores_a_low_weighted_jaccard(tmp_path):
    """The reflected effector shares almost no contacts with the truth."""
    wrong = _wrong_pose(GT, tmp_path / "wrong.pdb")
    rows = _parse(_boltz_output(tmp_path / "out", wrong), GT)
    assert float(rows[0]["weighted_jaccard"]) < 0.2, \
        f"a reflected effector scored {rows[0]['weighted_jaccard']}"


@pytest.mark.local_integration
@needs_pdb
def test_the_contact_count_is_reported(tmp_path):
    rows = _parse(_boltz_output(tmp_path / "out", GT), GT)
    assert int(rows[0]["n_contact_residues"]) > 0
    assert rows[0]["contact_residues"], "no contact residue list recorded"


@pytest.mark.local_integration
@needs_pdb
def test_the_confidence_sidecar_is_read(tmp_path):
    out = _boltz_output(tmp_path / "out", GT)
    rows = _parse(out, GT)
    assert float(rows[0]["iptm"]) == pytest.approx(0.43)
    assert float(rows[0]["ptm"]) == pytest.approx(0.66)


@pytest.mark.local_integration
@needs_pdb
def test_pae_derived_metrics_reach_the_row(tmp_path):
    out = _boltz_output(tmp_path / "out", GT, inter_pae=2.0)
    rows = _parse(out, GT)
    for col in ("ipsae_min", "ipae", "pae_pass_frac", "actifptm"):
        assert rows[0][col] not in (None, ""), f"{col} was not produced"


@pytest.mark.local_integration
@needs_pdb
def test_a_confident_interface_outscores_a_hopeless_one(tmp_path):
    """The whole point of the PAE metrics, checked end to end."""
    good = _parse(_boltz_output(tmp_path / "g", GT, inter_pae=1.0), GT)
    bad = _parse(_boltz_output(tmp_path / "b", GT, inter_pae=30.0), GT)

    g = float(good[0]["ipsae_min"])
    b = float(bad[0]["ipsae_min"])
    assert g > b, f"a hopeless interface ({b}) outscored a confident one ({g})"


@pytest.mark.local_integration
@needs_pdb
def test_an_empty_directory_yields_no_rows(tmp_path):
    empty = tmp_path / "empty"
    (empty / "predictions" / "input").mkdir(parents=True)
    assert _parse(empty, GT) == []


@pytest.mark.local_integration
@needs_pdb
def test_parsing_without_a_native_still_produces_confidence(tmp_path):
    """Cold-start rows have no comparison basis but still carry confidence."""
    out = _boltz_output(tmp_path / "out", GT)
    rows = _parse(out, None)
    assert rows, "no rows produced without a native reference"


# ── the CLI ──────────────────────────────────────────────────────────────────

@pytest.mark.local_integration
@needs_pdb
def test_the_cli_writes_a_metrics_csv(tmp_path):
    out = _boltz_output(tmp_path / "out", GT)
    csv_path = tmp_path / "metrics.csv"
    argv = ["compute_metrics.py",
            "--prediction-dir", str(out),
            "--chain-lengths", str(REC_LEN), str(EFF_LEN),
            "--receptor-chain", "A", "--effector-chain", "B",
            "--model", "boltz2",
            "--native-pdb", str(GT),
            "--output-csv", str(csv_path)]
    with mock.patch.object(sys, "argv", argv):
        rc = cm.main()
    assert rc in (0, None), f"compute_metrics exited {rc}"
    assert csv_path.is_file(), "no metrics CSV written"
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "the metrics CSV has no rows"


@pytest.mark.local_integration
@needs_pdb
def test_the_cli_accepts_mutated_positions(tmp_path):
    """Mutated positions drive the mutation-reliance column."""
    out = _boltz_output(tmp_path / "out", GT)
    csv_path = tmp_path / "metrics.csv"
    argv = ["compute_metrics.py",
            "--prediction-dir", str(out),
            "--chain-lengths", str(REC_LEN), str(EFF_LEN),
            "--receptor-chain", "A", "--effector-chain", "B",
            "--model", "boltz2",
            "--native-pdb", str(GT),
            "--mutated-positions", "5,7,22",
            "--output-csv", str(csv_path)]
    with mock.patch.object(sys, "argv", argv):
        rc = cm.main()
    assert rc in (0, None)
    assert csv_path.is_file()
