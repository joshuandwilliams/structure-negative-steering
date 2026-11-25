"""Tests for cross_summary.aggregate, on built per-sequence inputs.

This is the last stage of a run. It reads one passing_summary.csv per MPNN
sequence, picks the single design that represents each, and writes the
cross-sequence table that every cohort figure is drawn from.

The pick is reliability first: a design where more seeds independently land
the same correct pose outranks one that merely has a lower median by chance.
Inverting that, or breaking ties on the composite score instead of the median
ra_eff, changes which design is reported for a sequence without changing any
count, so nothing downstream looks wrong.

Inputs are built here with the tier and the median stated up front, so each
test asserts the pick it constructed rather than agreeing with whatever the
code did last time.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE = REPO_ROOT / "bin" / "cross_summary.py"


@pytest.fixture(scope="module")
def cs():
    sys.path.insert(0, str(REPO_ROOT / "bin"))
    spec = importlib.util.spec_from_file_location("cross_summary", MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


PASSING_COLUMNS = [
    "design", "n_pass", "n_seeds", "ra_eff_vs_truth_median",
    "true_jaccard_median", "rank_by_composite_score", "outcome",
    "rep_canonical_pdb", "total_mutations_median",
]


def prow(design, *, n_pass=3, n_seeds=3, ra=2.0, tj=0.8, rank=1,
         outcome="ok", pdb="", **extra):
    row = {"design": design, "n_pass": str(n_pass), "n_seeds": str(n_seeds),
           "ra_eff_vs_truth_median": str(ra),
           "true_jaccard_median": str(tj),
           "rank_by_composite_score": str(rank),
           "outcome": outcome, "rep_canonical_pdb": pdb,
           "total_mutations_median": "3"}
    row.update({k: str(v) for k, v in extra.items()})
    return row


def write_passing(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PASSING_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def read_out(path: Path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ── tiering ──────────────────────────────────────────────────────────────────

@pytest.mark.logic
@pytest.mark.parametrize("n_pass,n_seeds,tier", [
    (3, 3, "A"), (2, 3, "B"), (1, 3, "C"), (0, 3, "none"),
    (1, 1, "A"), (0, 0, "none"),
])
def test_the_tier_comes_from_the_seed_counts(cs, n_pass, n_seeds, tier):
    assert cs._tier_for_row(prow("d", n_pass=n_pass, n_seeds=n_seeds)) == tier


@pytest.mark.logic
@pytest.mark.parametrize("row", [
    {"n_pass": "", "n_seeds": "3"},
    {"n_pass": "2", "n_seeds": ""},
    {"n_pass": "x", "n_seeds": "3"},
])
def test_missing_counts_land_in_no_tier_rather_than_guessing(cs, row):
    assert cs._tier_for_row(row) == "none"


# ── the within-sequence pick ─────────────────────────────────────────────────

@pytest.mark.logic
def test_reliability_beats_a_lower_median(cs):
    """The whole point of tier-first. A 3/3 design with a worse median must
    still represent the sequence."""
    rows = [prow("sloppy", n_pass=1, n_seeds=3, ra=1.0),
            prow("reliable", n_pass=3, n_seeds=3, ra=4.0)]
    assert cs._pick_representative(rows)["design"] == "reliable"


@pytest.mark.logic
def test_within_a_tier_the_lower_median_wins(cs):
    rows = [prow("worse", n_pass=3, n_seeds=3, ra=4.0),
            prow("better", n_pass=3, n_seeds=3, ra=1.5)]
    assert cs._pick_representative(rows)["design"] == "better"


@pytest.mark.logic
def test_the_tie_break_is_the_median_not_the_composite(cs):
    """A design with a better interface overlap but a meaningfully worse pose
    must not win. Ranking on the composite here would pick 'overlap'."""
    rows = [prow("overlap", n_pass=3, n_seeds=3, ra=4.5, tj=0.99),
            prow("pose", n_pass=3, n_seeds=3, ra=1.0, tj=0.10)]
    assert cs._pick_representative(rows)["design"] == "pose"


@pytest.mark.logic
def test_the_pick_is_annotated_with_its_tier(cs):
    got = cs._pick_representative([prow("d", n_pass=2, n_seeds=3)])
    assert got["_picked_tier"] == "B"


@pytest.mark.logic
def test_no_rows_means_no_representative(cs):
    assert cs._pick_representative([]) is None


@pytest.mark.logic
def test_rows_that_are_all_untiered_have_no_representative(cs):
    assert cs._pick_representative([prow("d", n_pass=0, n_seeds=3)]) is None


@pytest.mark.logic
def test_an_unparseable_median_sorts_last(cs):
    rows = [prow("bad", n_pass=3, n_seeds=3, ra="x"),
            prow("good", n_pass=3, n_seeds=3, ra=9.0)]
    assert cs._pick_representative(rows)["design"] == "good"


# ── the composite score ──────────────────────────────────────────────────────

@pytest.mark.logic
def test_the_composite_is_jaccard_minus_a_twentieth_of_the_rmsd(cs):
    assert cs._composite_score(prow("d", ra=10.0, tj=0.9)) == pytest.approx(0.4)


@pytest.mark.logic
@pytest.mark.parametrize("missing", ["ra_eff_vs_truth_median",
                                     "true_jaccard_median"])
def test_a_composite_missing_a_component_is_none(cs, missing):
    row = prow("d")
    row[missing] = ""
    assert cs._composite_score(row) is None


# ── aggregate, end to end ────────────────────────────────────────────────────

@pytest.mark.logic
def test_one_row_per_sequence_is_written(cs, tmp_path):
    seqs = []
    for name, ra in (("seqA", 1.0), ("seqB", 3.0)):
        p = write_passing(tmp_path / name / "passing_summary.csv",
                          [prow("design_00", ra=ra)])
        seqs.append((name, p))
    out = tmp_path / "cross.csv"
    assert cs.aggregate(sequences=seqs, output_path=out, strict=False) == 0
    rows = read_out(out)
    assert sorted(r["mpnn_sequence"] for r in rows) == ["seqA", "seqB"]


@pytest.mark.logic
def test_cross_ranks_order_the_sequences(cs, tmp_path):
    """The better pose must rank first by ra_eff across sequences."""
    seqs = []
    for name, ra in (("seqA", 9.0), ("seqB", 1.0)):
        p = write_passing(tmp_path / name / "passing_summary.csv",
                          [prow("design_00", ra=ra)])
        seqs.append((name, p))
    out = tmp_path / "cross.csv"
    assert cs.aggregate(sequences=seqs, output_path=out, strict=False) == 0
    by = {r["mpnn_sequence"]: r for r in read_out(out)}
    assert by["seqB"]["cross_rank_by_ra_eff"] == "1"
    assert by["seqA"]["cross_rank_by_ra_eff"] == "2"


@pytest.mark.logic
def test_the_chosen_tier_reaches_the_output(cs, tmp_path):
    p = write_passing(tmp_path / "seqA" / "passing_summary.csv",
                      [prow("design_00", n_pass=2, n_seeds=3)])
    out = tmp_path / "cross.csv"
    assert cs.aggregate(sequences=[("seqA", p)], output_path=out,
                        strict=False) == 0
    assert read_out(out)[0]["cross_tier"] == "B"


@pytest.mark.logic
def test_a_missing_input_is_a_stub_row_by_default(cs, tmp_path, capsys):
    """A sequence whose run did not finish must still appear, or the cohort
    silently shrinks and every proportion is computed over the wrong base."""
    out = tmp_path / "cross.csv"
    assert cs.aggregate(sequences=[("seqA", tmp_path / "nope.csv")],
                        output_path=out, strict=False) == 0
    rows = read_out(out)
    assert [r["mpnn_sequence"] for r in rows] == ["seqA"]
    assert "not found" in capsys.readouterr().err


@pytest.mark.logic
def test_a_missing_input_is_fatal_under_strict(cs, tmp_path):
    out = tmp_path / "cross.csv"
    assert cs.aggregate(sequences=[("seqA", tmp_path / "nope.csv")],
                        output_path=out, strict=True) == 2


@pytest.mark.logic
def test_a_duplicate_sequence_name_is_refused(cs, tmp_path):
    """Two rows keyed the same would overwrite one run with another."""
    p = write_passing(tmp_path / "seqA" / "passing_summary.csv",
                      [prow("design_00")])
    out = tmp_path / "cross.csv"
    assert cs.aggregate(sequences=[("seqA", p), ("seqA", p)],
                        output_path=out, strict=False) == 2


@pytest.mark.logic
def test_an_empty_passing_summary_still_yields_a_row(cs, tmp_path):
    """Nothing passed is a result for that sequence, not an absence."""
    p = write_passing(tmp_path / "seqA" / "passing_summary.csv", [])
    out = tmp_path / "cross.csv"
    assert cs.aggregate(sequences=[("seqA", p)], output_path=out,
                        strict=False) == 0
    assert [r["mpnn_sequence"] for r in read_out(out)] == ["seqA"]


# ── rewriting work-dir paths to the published tree ───────────────────────────
#
# NEGSTEER_RUN_ONE stamps Nextflow's per-task work/<hash>/ path into the
# canonical_pdb columns. That tree is scratch: `nextflow clean` deletes it,
# and a later task cannot read it without binding that exact path. The same
# file always exists under the published runs directory, so the path is
# rewritten there. Every failure mode returns the path unchanged, because a
# best-effort rewrite must never mangle a path it does not understand.

@pytest.mark.logic
def test_a_work_path_is_rewritten_under_the_published_tree(cs, tmp_path):
    raw = "/scratch/work/b6/afc123/seqA/cycle_0/steered/x.pdb"
    got = cs._rewrite_workdir_path_to_published(raw, "seqA", tmp_path)
    assert got == str(tmp_path / "seqA" / "cycle_0" / "steered" / "x.pdb")


@pytest.mark.logic
def test_the_sequence_name_is_matched_as_a_segment_not_a_substring(cs,
                                                                   tmp_path):
    """A directory merely containing the name must not be split on."""
    raw = "/scratch/work/b6/seqA_backup/cycle_0/x.pdb"
    assert cs._rewrite_workdir_path_to_published(raw, "seqA", tmp_path) == raw


@pytest.mark.logic
@pytest.mark.parametrize("raw,seq,pub", [
    ("", "seqA", "SET"),                      # blank path
    ("/a/seqA/x.pdb", "seqA", None),          # no published dir
    ("/a/b/x.pdb", "", "SET"),                # no sequence name
    ("/a/b/x.pdb", "seqA", "SET"),            # name not in the path
    ("/a/seqA", "seqA", "SET"),               # nothing after the name
])
def test_every_unrewritable_path_comes_back_unchanged(cs, tmp_path, raw, seq,
                                                      pub):
    published = tmp_path if pub == "SET" else None
    assert cs._rewrite_workdir_path_to_published(raw, seq, published) == raw


# ── scored metadata ──────────────────────────────────────────────────────────

@pytest.mark.logic
def test_scored_metadata_is_keyed_the_way_sequences_are_named(cs, tmp_path):
    p = tmp_path / "scored.csv"
    p.write_text("design,seq,corrected_receptor\n7,3,MKVL\n")
    got = cs._load_scored_metadata(p)
    assert list(got) == ["design_7_seq_3"]


@pytest.mark.logic
def test_metadata_rows_missing_a_key_component_are_skipped(cs, tmp_path):
    p = tmp_path / "scored.csv"
    p.write_text("design,seq,corrected_receptor\n,3,A\n7,,B\n8,1,C\n")
    assert list(cs._load_scored_metadata(p)) == ["design_8_seq_1"]


@pytest.mark.logic
def test_no_metadata_path_is_an_empty_lookup(cs):
    assert cs._load_scored_metadata(None) == {}


@pytest.mark.logic
def test_a_missing_metadata_file_warns_and_blanks_the_columns(cs, tmp_path,
                                                              capsys):
    assert cs._load_scored_metadata(tmp_path / "nope.csv") == {}
    assert "not found" in capsys.readouterr().err


@pytest.mark.logic
def test_metadata_columns_are_joined_onto_the_output(cs, tmp_path):
    meta = tmp_path / "scored.csv"
    meta.write_text("design,seq,corrected_receptor,designed_residues,"
                    "native_residues\n7,3,MKVL,AAA,GGG\n")
    p = write_passing(tmp_path / "design_7_seq_3" / "passing_summary.csv",
                      [prow("design_00")])
    out = tmp_path / "cross.csv"
    assert cs.aggregate(sequences=[("design_7_seq_3", p)], output_path=out,
                        strict=False, scored_metadata_path=meta) == 0
    row = read_out(out)[0]
    assert row["corrected_receptor"] == "MKVL"
    assert row["designed_residues"] == "AAA"


@pytest.mark.logic
def test_a_sequence_with_no_metadata_gets_blank_columns(cs, tmp_path):
    meta = tmp_path / "scored.csv"
    meta.write_text("design,seq,corrected_receptor\n9,9,ZZZ\n")
    p = write_passing(tmp_path / "seqA" / "passing_summary.csv",
                      [prow("design_00")])
    out = tmp_path / "cross.csv"
    assert cs.aggregate(sequences=[("seqA", p)], output_path=out,
                        strict=False, scored_metadata_path=meta) == 0
    assert read_out(out)[0]["corrected_receptor"] == ""


# ── the CLI entry point ──────────────────────────────────────────────────────

@pytest.mark.logic
def test_no_inputs_at_all_is_refused(cs, capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv",
                        ["cross_summary.py", "--output", str(tmp_path / "o.csv")])
    assert cs.main() == 2
    assert "no inputs found" in capsys.readouterr().err


# ── deriving the sequence name from a path ───────────────────────────────────
#
# The name keys every row of the cross-sequence table. Inferring it from the
# wrong path component relabels a whole run, and the numbers stay plausible.

@pytest.mark.logic
def test_parent_and_grandparent_modes_read_different_components(cs, tmp_path):
    p = tmp_path / "seqA" / "final" / "passing_summary.csv"
    p.parent.mkdir(parents=True)
    p.write_text("x\n")
    assert cs._infer_seq_name(p, "parent") == "final"
    assert cs._infer_seq_name(p, "grandparent") == "seqA"


@pytest.mark.logic
def test_an_unknown_mode_is_refused(cs, tmp_path):
    p = tmp_path / "seqA" / "passing_summary.csv"
    p.parent.mkdir(parents=True)
    p.write_text("x\n")
    with pytest.raises(ValueError, match="unknown seq-name inference mode"):
        cs._infer_seq_name(p, "greatgrandparent")


@pytest.mark.logic
def test_a_name_with_unexpected_characters_is_refused(cs, tmp_path):
    """It becomes a column key, so only the safe character set is allowed."""
    p = tmp_path / "bad name!" / "passing_summary.csv"
    p.parent.mkdir(parents=True)
    p.write_text("x\n")
    with pytest.raises(ValueError, match="unexpected characters"):
        cs._infer_seq_name(p, "parent")


@pytest.mark.logic
def test_an_explicit_name_is_taken_when_the_path_exists(cs, tmp_path):
    p = write_passing(tmp_path / "whatever" / "passing_summary.csv",
                      [prow("design_00")])
    name, got = cs._parse_passing_summary_arg(f"myname={p}", "parent")
    assert name == "myname" and got == p


@pytest.mark.logic
def test_an_equals_in_a_path_does_not_split_it(cs, tmp_path):
    """Only the first '=' separates, and only when the head looks like a name
    and the tail is a real file. A path containing '=' stays whole."""
    d = tmp_path / "a=b" / "seqA"
    p = write_passing(d / "passing_summary.csv", [prow("design_00")])
    name, got = cs._parse_passing_summary_arg(str(p), "parent")
    assert name == "seqA" and got == p


@pytest.mark.logic
def test_a_bare_path_infers_its_name(cs, tmp_path):
    p = write_passing(tmp_path / "seqA" / "passing_summary.csv",
                      [prow("design_00")])
    assert cs._parse_passing_summary_arg(str(p), "parent")[0] == "seqA"


# ── the tier-none fallback ───────────────────────────────────────────────────
#
# When every prediction failed the eligibility gate, passing_summary.csv has
# no rows. The cohort plot still needs something for that sequence, so the
# sibling aggregated_results.csv is read and its best composite row used. A
# sequence that vanishes instead changes the base every proportion is over.

@pytest.mark.logic
def test_no_sibling_aggregate_means_no_fallback(cs, tmp_path):
    p = write_passing(tmp_path / "seqA" / "passing_summary.csv", [])
    assert cs._pick_rep_from_aggregated(p) is None


@pytest.mark.logic
def test_an_empty_sibling_aggregate_means_no_fallback(cs, tmp_path):
    p = write_passing(tmp_path / "seqA" / "passing_summary.csv", [])
    (p.parent / "aggregated_results.csv").write_text(
        "design,steered_ra_eff_vs_truth_median,steered_true_jaccard_median\n")
    assert cs._pick_rep_from_aggregated(p) is None


@pytest.mark.logic
def test_an_unreadable_sibling_aggregate_warns_and_falls_through(cs, tmp_path,
                                                                 capsys):
    p = write_passing(tmp_path / "seqA" / "passing_summary.csv", [])
    (p.parent / "aggregated_results.csv").mkdir()   # a directory, not a file
    assert cs._pick_rep_from_aggregated(p) is None
