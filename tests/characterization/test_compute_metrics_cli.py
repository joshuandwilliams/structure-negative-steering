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

GT = (_REPO_ROOT / "data" / "reference_complexes" / "6G10.pdb")
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

@pytest.mark.logic
@needs_pdb
def test_a_boltz_directory_is_parsed_into_metric_rows(tmp_path):
    out = _boltz_output(tmp_path / "out", GT)
    rows = _parse(out, GT)
    assert rows, "parse_boltz2 returned nothing for a well-formed directory"


@pytest.mark.logic
@needs_pdb
def test_a_perfect_pose_scores_a_weighted_jaccard_of_one(tmp_path):
    """Scoring the native against itself is the fixed point of the metric."""
    rows = _parse(_boltz_output(tmp_path / "out", GT), GT)
    assert float(rows[0]["weighted_jaccard"]) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.logic
@needs_pdb
def test_a_wrong_pose_scores_a_low_weighted_jaccard(tmp_path):
    """The reflected effector shares almost no contacts with the truth."""
    wrong = _wrong_pose(GT, tmp_path / "wrong.pdb")
    rows = _parse(_boltz_output(tmp_path / "out", wrong), GT)
    assert float(rows[0]["weighted_jaccard"]) < 0.2, \
        f"a reflected effector scored {rows[0]['weighted_jaccard']}"


@pytest.mark.logic
@needs_pdb
def test_the_contact_count_is_reported(tmp_path):
    rows = _parse(_boltz_output(tmp_path / "out", GT), GT)
    assert int(rows[0]["n_contact_residues"]) > 0
    assert rows[0]["contact_residues"], "no contact residue list recorded"


@pytest.mark.logic
@needs_pdb
def test_the_confidence_sidecar_is_read(tmp_path):
    out = _boltz_output(tmp_path / "out", GT)
    rows = _parse(out, GT)
    assert float(rows[0]["iptm"]) == pytest.approx(0.43)
    assert float(rows[0]["ptm"]) == pytest.approx(0.66)


@pytest.mark.logic
@needs_pdb
def test_pae_derived_metrics_reach_the_row(tmp_path):
    out = _boltz_output(tmp_path / "out", GT, inter_pae=2.0)
    rows = _parse(out, GT)
    for col in ("ipsae_min", "ipae", "pae_pass_frac", "actifptm"):
        assert rows[0][col] not in (None, ""), f"{col} was not produced"


@pytest.mark.logic
@needs_pdb
def test_a_confident_interface_outscores_a_hopeless_one(tmp_path):
    """The whole point of the PAE metrics, checked end to end."""
    good = _parse(_boltz_output(tmp_path / "g", GT, inter_pae=1.0), GT)
    bad = _parse(_boltz_output(tmp_path / "b", GT, inter_pae=30.0), GT)

    g = float(good[0]["ipsae_min"])
    b = float(bad[0]["ipsae_min"])
    assert g > b, f"a hopeless interface ({b}) outscored a confident one ({g})"


@pytest.mark.logic
@needs_pdb
def test_an_empty_directory_yields_no_rows(tmp_path):
    empty = tmp_path / "empty"
    (empty / "predictions" / "input").mkdir(parents=True)
    assert _parse(empty, GT) == []


@pytest.mark.logic
@needs_pdb
def test_parsing_without_a_native_still_produces_confidence(tmp_path):
    """Cold-start rows have no comparison basis but still carry confidence."""
    out = _boltz_output(tmp_path / "out", GT)
    rows = _parse(out, None)
    assert rows, "no rows produced without a native reference"


# ── the CLI ──────────────────────────────────────────────────────────────────

@pytest.mark.logic
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


@pytest.mark.logic
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


# ── missing sidecars degrade, they do not fail ───────────────────────────────
#
# The script runs unattended once per design inside the container. A
# prediction whose PAE array never landed still has to yield its pLDDT and
# its contacts, because the hour of GPU time that produced it is not
# recoverable.
#
# What it degrades to is worth stating plainly: the absent metrics come back
# as 0.0, not blank. A prediction whose PAE was not found is therefore
# indistinguishable from one whose interface really is that bad. That is
# exactly why both callers locate Boltz's nested sidecars rather than
# pointing --prediction-dir at prediction.pdb's parent, where none of them
# sit — doing so returns ptm=0, iptm=0 and pae=0 with only pLDDT surviving
# through the B-factor fallback.

def _strip(root: Path, *kinds: str) -> Path:
    """Remove whole sidecar directories from a built Boltz tree."""
    job = root / "predictions" / "input"
    for kind in kinds:
        shutil.rmtree(job / kind)
    return root


@pytest.mark.logic
@needs_pdb
def test_a_prediction_with_no_pae_still_reports_its_plddt(tmp_path):
    out = _strip(_boltz_output(tmp_path / "out", GT), "pae")
    rows = _parse(out, GT)
    assert float(rows[0]["pae_mean"]) == 0.0
    assert float(rows[0]["avg_plddt"]) == pytest.approx(90.0, abs=1e-6)


@pytest.mark.logic
@needs_pdb
def test_a_prediction_with_no_confidence_json_still_reports_its_pae(tmp_path):
    out = _strip(_boltz_output(tmp_path / "out", GT), "confidence")
    rows = _parse(out, GT)
    assert float(rows[0]["ptm"]) == 0.0
    assert float(rows[0]["pae_mean"]) > 0.0


@pytest.mark.logic
@needs_pdb
def test_a_bare_pdb_still_reports_its_contacts(tmp_path):
    """Contacts come off the structure alone, so they survive every missing
    sidecar. They are also what the contamination check reads."""
    out = _strip(_boltz_output(tmp_path / "out", GT),
                 "pae", "plddt", "confidence")
    rows = _parse(out, GT)
    assert int(rows[0]["n_contact_residues"]) > 0


@pytest.mark.logic
@needs_pdb
def test_an_unreadable_confidence_json_is_a_warning_not_a_failure(tmp_path,
                                                                  capsys):
    out = _boltz_output(tmp_path / "out", GT)
    (out / "predictions" / "input" / "confidence"
     / "confidence_input_model_0.json").write_text("{not json")
    rows = _parse(out, GT)
    assert float(rows[0]["ptm"]) == 0.0
    assert "conf parse failed" in capsys.readouterr().out


@pytest.mark.logic
@needs_pdb
def test_a_cached_reference_in_the_tree_is_not_scored(tmp_path):
    """reference.pdb is the ground truth copied in for alignment. Scoring it
    would report a perfect prediction for every design in the run."""
    out = _boltz_output(tmp_path / "out", GT)
    shutil.copy(GT, out / "predictions" / "input" / "reference.pdb")
    rows = _parse(out, GT)
    assert [r["model_name"] for r in rows] == ["input_model_0"]


# ── the effector atom filter ─────────────────────────────────────────────────
#
# Contact detection restricted to the effector's own interface atoms. Without
# it a steering mutation touching the back of the effector counts as
# contamination, which it is not: the construct's binding is unaffected.
#
# The callers point this at cycle_0/plan.json directly, so the loader has to
# accept a whole plan dict as well as a bare list. A materialised filter file
# would be a second copy that can drift from the plan it came from.

def _run_cli(out: Path, csv_path: Path, *extra: str):
    argv = ["compute_metrics.py",
            "--prediction-dir", str(out),
            "--chain-lengths", str(REC_LEN), str(EFF_LEN),
            "--receptor-chain", "A", "--effector-chain", "B",
            "--model", "boltz2", "--output-csv", str(csv_path), *extra]
    with mock.patch.object(sys, "argv", argv):
        return cm.main()


def _cli_rows(csv_path: Path) -> list[dict]:
    with csv_path.open() as fh:
        return list(csv.DictReader(fh))


@pytest.mark.logic
@needs_pdb
def test_a_plan_json_can_be_pointed_at_directly(tmp_path):
    out = _boltz_output(tmp_path / "out", GT)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(
        {"cycle": 0, "effector_interface_residues": [1, 2, 3]}))
    csv_path = tmp_path / "m.csv"
    _run_cli(out, csv_path, "--effector-atom-filter-json", str(plan))
    assert int(_cli_rows(csv_path)[0]["n_contact_residues"]) >= 0


@pytest.mark.logic
@needs_pdb
def test_a_bare_residue_list_is_accepted(tmp_path):
    out = _boltz_output(tmp_path / "out", GT)
    f = tmp_path / "filter.json"
    f.write_text(json.dumps([1, 2, 3]))
    csv_path = tmp_path / "m.csv"
    _run_cli(out, csv_path, "--effector-atom-filter-json", str(f))
    assert _cli_rows(csv_path)


@pytest.mark.logic
@needs_pdb
def test_the_legacy_atom_pair_list_is_accepted(tmp_path):
    """Earlier runs wrote [resseq, atom_name] pairs. They are collapsed to a
    residue set internally, so an old plan.json still works."""
    out = _boltz_output(tmp_path / "out", GT)
    f = tmp_path / "filter.json"
    f.write_text(json.dumps([[1, "CA"], [2, "any"]]))
    csv_path = tmp_path / "m.csv"
    _run_cli(out, csv_path, "--effector-atom-filter-json", str(f))
    assert _cli_rows(csv_path)


@pytest.mark.logic
@needs_pdb
def test_a_filter_restricted_to_nothing_finds_no_contacts(tmp_path):
    """The filter is a restriction, so a filter naming no real residue has
    to restrict to nothing rather than being read as no filter at all."""
    out = _boltz_output(tmp_path / "out", GT)
    f = tmp_path / "filter.json"
    f.write_text(json.dumps({"effector_interface_residues": [9999]}))
    csv_path = tmp_path / "m.csv"
    _run_cli(out, csv_path, "--effector-atom-filter-json", str(f))
    assert int(_cli_rows(csv_path)[0]["n_contact_residues"]) == 0


@pytest.mark.logic
@needs_pdb
def test_an_unreadable_filter_falls_back_to_unfiltered_detection(tmp_path):
    """Over-reporting contamination is the conservative direction. Failing
    loses the prediction."""
    out = _boltz_output(tmp_path / "out", GT)
    f = tmp_path / "filter.json"
    f.write_text("{not json")
    csv_path = tmp_path / "m.csv"
    _run_cli(out, csv_path, "--effector-atom-filter-json", str(f))
    assert int(_cli_rows(csv_path)[0]["n_contact_residues"]) > 0


# ── the contamination columns ────────────────────────────────────────────────

@pytest.mark.logic
@needs_pdb
def test_without_a_mutation_list_the_contamination_positions_are_blank(
        tmp_path):
    """A cold start has no mutations, so there is nothing to be contaminated
    by. The count is 0 rather than blank, so a reader has to use the position
    list to tell "no mutations" from "mutations, none at the interface"."""
    out = _boltz_output(tmp_path / "out", GT)
    csv_path = tmp_path / "m.csv"
    _run_cli(out, csv_path)
    row = _cli_rows(csv_path)[0]
    assert row["mutated_contact_positions"] == ""
    assert row["n_contacts_on_mutated_positions"] == "0"


@pytest.mark.logic
@needs_pdb
def test_a_mutation_at_the_interface_is_reported_as_contamination(tmp_path):
    """This is the authoritative check the reversion pass acts on, and the
    positions it names are exactly the ones reversion must revert."""
    out = _boltz_output(tmp_path / "out", GT)
    base = tmp_path / "base.csv"
    _run_cli(out, base, "--contact-cutoff", "4.5")
    contacts = [int(p) for p in
                _cli_rows(base)[0]["contact_residues"].split(",") if p]
    assert contacts, "the reference complex has no interface to test with"

    hit = tmp_path / "hit.csv"
    _run_cli(out, hit, "--contact-cutoff", "4.5",
             "--mutated-positions", f"/A:{contacts[0]}")
    row = _cli_rows(hit)[0]
    assert row["mutated_contact_positions"] == str(contacts[0])
    assert int(row["n_contacts_on_mutated_positions"]) == 1


@pytest.mark.logic
@needs_pdb
def test_a_mutation_away_from_the_interface_is_clean(tmp_path):
    out = _boltz_output(tmp_path / "out", GT)
    base = tmp_path / "base.csv"
    _run_cli(out, base, "--contact-cutoff", "4.5")
    contacts = {int(p) for p in
                _cli_rows(base)[0]["contact_residues"].split(",") if p}
    clean = max(contacts) + 500

    miss = tmp_path / "miss.csv"
    _run_cli(out, miss, "--contact-cutoff", "4.5",
             "--mutated-positions", f"/A:{clean}")
    row = _cli_rows(miss)[0]
    assert row["mutated_contact_positions"] == ""
    assert int(row["n_contacts_on_mutated_positions"]) == 0


@pytest.mark.logic
@needs_pdb
def test_a_wider_cutoff_finds_at_least_as_many_contacts(tmp_path):
    out = _boltz_output(tmp_path / "out", GT)
    narrow, wide = tmp_path / "n.csv", tmp_path / "w.csv"
    _run_cli(out, narrow, "--contact-cutoff", "4.0")
    _run_cli(out, wide, "--contact-cutoff", "8.0")
    assert (int(_cli_rows(wide)[0]["n_contact_residues"])
            >= int(_cli_rows(narrow)[0]["n_contact_residues"]))


@pytest.mark.logic
@needs_pdb
def test_the_contact_cutoff_is_recorded_but_the_pae_cutoff_is_not(tmp_path):
    """An asymmetry in the schema. contact_cutoff_used travels with the
    contact columns, so a row can be read without knowing how it was made.
    The ipSAE columns have no equivalent, so a bare compute_metrics CSV does
    not say which PAE cutoff produced them. negsteer_aggregate supplies
    steered_pae_cutoff_used from its own --pae-cutoff, which is correct but
    only because the two flags are set from one place."""
    out = _boltz_output(tmp_path / "out", GT)
    csv_path = tmp_path / "m.csv"
    _run_cli(out, csv_path, "--pae-cutoff", "15", "--contact-cutoff", "4.5")
    row = _cli_rows(csv_path)[0]
    assert float(row["contact_cutoff_used"]) == 4.5
    assert "pae_cutoff_used" not in row


# ── the pLDDT-trimmed intact core ────────────────────────────────────────────
#
# The core is the receptor trimmed to residues above a pLDDT threshold, fitted
# between model and native with Kabsch, then scored as a whole-complex Ca
# RMSD. Boltz writes per-residue pLDDT into the B-factor column, which is
# where the trim reads it from. The reference complex is a crystal structure,
# so its own B-factors are real B-factors in the 20-40 range and every pLDDT
# threshold rejects them — which is why these tests rewrite that column.

def _with_plddt(src: Path, dst: Path, value: float) -> Path:
    out = []
    for line in src.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 66:
            line = f"{line[:60]}{value:6.2f}{line[66:]}"
        out.append(line)
    dst.write_text("\n".join(out) + "\n")
    return dst


@pytest.mark.logic
@needs_pdb
def test_a_confident_prediction_of_the_native_is_intact(tmp_path):
    """Same structure, so the fit is exact and the RMSD is zero. A test that
    only checked the column was populated would pass against a transposed
    rotation."""
    pred = _with_plddt(GT, tmp_path / "pred.pdb", 90.0)
    rows = _parse(_boltz_output(tmp_path / "out", pred), GT)
    assert str(rows[0]["intact_core"]) == "1"
    assert float(rows[0]["weighted_jaccard"]) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.logic
@needs_pdb
def test_a_prediction_the_plddt_filter_rejects_has_no_intact_core(tmp_path):
    """Below the threshold there are fewer than three residues to fit, and a
    three-point Kabsch fit on a hand-picked core would report whatever RMSD
    it liked. Blank is the honest answer."""
    pred = _with_plddt(GT, tmp_path / "pred.pdb", 20.0)
    rows = _parse(_boltz_output(tmp_path / "out", pred), GT)
    assert rows[0]["intact_core"] in ("", None)


@pytest.mark.logic
@needs_pdb
def test_the_threshold_decides_which_residues_form_the_core(tmp_path):
    """Same structure and same native, two thresholds. One keeps the core and
    one rejects it, which is what shows the trim is doing the work rather
    than the structure."""
    pred = _with_plddt(GT, tmp_path / "pred.pdb", 60.0)
    out = _boltz_output(tmp_path / "out", pred)
    kept = cm.parse_boltz2(str(out), [REC_LEN, EFF_LEN], 10.0, "A", "B",
                           None, 5.0, None, str(GT), 50.0, 5.0, 8.0)
    strict = cm.parse_boltz2(str(out), [REC_LEN, EFF_LEN], 10.0, "A", "B",
                             None, 5.0, None, str(GT), 80.0, 5.0, 8.0)
    assert str(kept[0]["intact_core"]) == "1"
    assert strict[0]["intact_core"] in ("", None)


@pytest.mark.logic
@needs_pdb
def test_a_displaced_effector_is_not_an_intact_complex(tmp_path):
    """intact_core is a whole-complex RMSD after fitting on the receptor
    core, so an effector on the wrong face fails it even though the receptor
    itself is perfect."""
    wrong = _with_plddt(_wrong_pose(GT, tmp_path / "wrong.pdb"),
                        tmp_path / "pred.pdb", 90.0)
    rows = _parse(_boltz_output(tmp_path / "out", wrong), GT)
    assert str(rows[0]["intact_core"]) == "0"


@pytest.mark.logic
@needs_pdb
def test_the_interface_plddt_needs_no_native(tmp_path):
    """It is read off the model's own B-factor column, so unlike the other
    two it is populated on every row. A uniform 90 there has to come back as
    0.9, because the thresholds downstream are on 0-1 while Boltz writes
    0-100."""
    pred = _with_plddt(GT, tmp_path / "pred.pdb", 90.0)
    rows = _parse(_boltz_output(tmp_path / "out", pred), None)
    assert float(rows[0]["interface_plddt"]) == pytest.approx(0.9, abs=1e-3)


# ── the CSV as an interface ──────────────────────────────────────────────────

@pytest.mark.logic
@needs_pdb
def test_every_declared_column_is_present_in_the_header(tmp_path):
    """Both callers copy columns by name out of this header. One that
    disappears blanks that metric for a whole campaign, and the stage that
    asked for it reports success either way."""
    out = _boltz_output(tmp_path / "out", GT)
    csv_path = tmp_path / "m.csv"
    _run_cli(out, csv_path)
    assert csv_path.read_text().splitlines()[0].split(",") == list(
        cm.CSV_FIELDS)


@pytest.mark.logic
@needs_pdb
def test_a_directory_with_no_predictions_still_writes_a_csv(tmp_path):
    """The callers read an absent file as the subprocess having failed, so
    nothing-to-report has to be a file with a header and no rows."""
    empty = tmp_path / "empty"
    (empty / "predictions" / "input").mkdir(parents=True)
    csv_path = tmp_path / "m.csv"
    _run_cli(empty, csv_path)
    assert csv_path.is_file()
    assert _cli_rows(csv_path) == []


@pytest.mark.logic
@needs_pdb
def test_a_prediction_directory_that_does_not_exist_is_refused(tmp_path):
    """Unlike a missing sidecar, nothing here can be salvaged, and writing an
    empty CSV would be indistinguishable from a run with no predictions."""
    csv_path = tmp_path / "m.csv"
    with pytest.raises(SystemExit) as exc:
        _run_cli(tmp_path / "gone", csv_path)
    assert exc.value.code == 1
    assert not csv_path.exists()


@pytest.mark.logic
@needs_pdb
def test_the_output_directory_is_created_if_it_is_absent(tmp_path):
    """The callers hand it a path under a scratch directory they have not
    necessarily made yet."""
    out = _boltz_output(tmp_path / "out", GT)
    csv_path = tmp_path / "nested" / "deeper" / "m.csv"
    _run_cli(out, csv_path)
    assert len(_cli_rows(csv_path)) == 1
