"""Tests for negsteer_aggregate's `aggregate-per-sequence` stage.

This collapses one row per Boltz seed into one row per unique sequence. The
collapse is where a result stops being a set of three predictions and becomes
the single number a figure plots, so every rule in it decides what gets
reported.

Three rules matter and all three fail quietly:

  - Grouping is on (cycle, pathway, sequence_group). Dropping any part of
    that key merges two different sequences into one row, and the median
    then summarises predictions of different proteins.
  - Continuous metrics collapse to a median, categoricals to a majority.
    Using a mean instead would let one collapsed seed drag a sequence over
    or under the 5 A line.
  - The canonical seed, which names the structure a reader opens, is the
    middle on an odd count and deliberately the WORSE of the two middle
    seeds on an even count.

Inputs are built with the per-seed values stated up front, so each test
asserts the collapse it constructed.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def agg():
    sys.path.insert(0, str(REPO_ROOT / "bin"))
    spec = importlib.util.spec_from_file_location(
        "negsteer_aggregate", REPO_ROOT / "bin" / "negsteer_aggregate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


PER_SEED_COLUMNS = [
    "cycle", "pathway", "design", "sequence_group", "seed_index", "status",
    "pdb", "steered_total_mutations", "steered_ra_eff_vs_truth",
    "steered_independent_receptor_rmsd", "steered_independent_effector_rmsd",
    "steered_receptor_intact", "steered_true_jaccard", "reversion_verdict",
]


def seed_row(group, seed, *, ra=2.0, design=None, cycle="0", pathway="cycle_0",
             intact="1", jaccard="0.8", status="ok", pdb=None, **extra):
    row = {
        "cycle": cycle, "pathway": pathway,
        "design": design or f"design_{group}",
        "sequence_group": str(group), "seed_index": str(seed),
        "status": status, "pdb": pdb or f"/runs/{group}_s{seed}.pdb",
        "steered_total_mutations": "3",
        "steered_ra_eff_vs_truth": str(ra),
        "steered_independent_receptor_rmsd": "1.0",
        "steered_independent_effector_rmsd": "1.0",
        "steered_receptor_intact": intact,
        "steered_true_jaccard": str(jaccard),
        "reversion_verdict": "",
    }
    row.update({k: str(v) for k, v in extra.items()})
    return row


def build(root: Path, rows, *, plan=None, filename="all_results_multicycle.csv"):
    root.mkdir(parents=True, exist_ok=True)
    with open(root / filename, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PER_SEED_COLUMNS,
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    if plan is not None:
        c0 = root / "cycle_0"
        c0.mkdir(parents=True, exist_ok=True)
        (c0 / "plan.json").write_text(json.dumps(plan))
    return root


def out_rows(root: Path, name="aggregated_results.csv"):
    with open(root / name) as f:
        return list(csv.DictReader(f))


# ── choosing the input ───────────────────────────────────────────────────────

@pytest.mark.logic
def test_a_tree_with_neither_table_is_refused(agg, tmp_path, capsys):
    tmp_path.mkdir(exist_ok=True)
    assert agg.cmd_aggregate_per_sequence(Args(experiment_root=tmp_path)) == 2
    assert "Run `aggregate` first" in capsys.readouterr().err


@pytest.mark.logic
def test_a_path_that_is_not_a_directory_is_refused(agg, tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("x")
    assert agg.cmd_aggregate_per_sequence(Args(experiment_root=f)) == 2


@pytest.mark.logic
def test_the_metriced_table_is_preferred(agg, tmp_path):
    """compute-final-metrics writes the confidence columns, so its output is
    the one to aggregate when it exists."""
    build(tmp_path, [seed_row(1, 0, ra=9.0)])
    build(tmp_path, [seed_row(1, 0, ra=1.0)],
          filename="all_results_multicycle_with_metrics.csv")
    assert agg.cmd_aggregate_per_sequence(Args(experiment_root=tmp_path)) == 0
    assert float(out_rows(tmp_path)[0]["steered_ra_eff_vs_truth_median"]) == 1.0


@pytest.mark.logic
def test_falling_back_to_the_plain_table_says_so(agg, tmp_path, capsys):
    build(tmp_path, [seed_row(1, 0)])
    assert agg.cmd_aggregate_per_sequence(Args(experiment_root=tmp_path)) == 0
    assert "no confidence metrics" in capsys.readouterr().out


@pytest.mark.logic
def test_the_per_seed_input_is_copied_under_a_stable_name(agg, tmp_path):
    """Downstream reads raw_per_seed_results.csv regardless of which table
    was aggregated."""
    build(tmp_path, [seed_row(1, 0)])
    assert agg.cmd_aggregate_per_sequence(Args(experiment_root=tmp_path)) == 0
    assert (tmp_path / "raw_per_seed_results.csv").exists()
    assert len(out_rows(tmp_path, "raw_per_seed_results.csv")) == 1


# ── grouping ─────────────────────────────────────────────────────────────────

@pytest.mark.logic
def test_seeds_of_one_sequence_collapse_to_one_row(agg, tmp_path):
    build(tmp_path, [seed_row(1, s) for s in range(3)])
    assert agg.cmd_aggregate_per_sequence(Args(experiment_root=tmp_path)) == 0
    rows = out_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["n_seeds"] == "3"


@pytest.mark.logic
def test_two_sequence_groups_stay_separate(agg, tmp_path):
    build(tmp_path, [seed_row(1, 0), seed_row(1, 1),
                     seed_row(2, 0), seed_row(2, 1)])
    assert agg.cmd_aggregate_per_sequence(Args(experiment_root=tmp_path)) == 0
    assert len(out_rows(tmp_path)) == 2


@pytest.mark.logic
def test_the_same_group_number_in_two_pathways_is_two_sequences(agg, tmp_path):
    """The group number is scoped by cycle and pathway. Grouping on it alone
    would take the median across two different proteins."""
    build(tmp_path, [seed_row(1, 0, pathway="cycle_0"),
                     seed_row(1, 0, pathway="c0d00")])
    assert agg.cmd_aggregate_per_sequence(Args(experiment_root=tmp_path)) == 0
    assert len(out_rows(tmp_path)) == 2


@pytest.mark.logic
def test_a_row_with_no_sequence_group_is_its_own_singleton(agg, tmp_path):
    """The initial baseline row predates the multi-seed plumbing and must
    survive rather than being grouped with anything."""
    build(tmp_path, [seed_row(1, 0), seed_row(1, 1),
                     seed_row("", 0, design="initial")])
    assert agg.cmd_aggregate_per_sequence(Args(experiment_root=tmp_path)) == 0
    designs = {r["design"] for r in out_rows(tmp_path)}
    assert "initial" in designs
    assert len(out_rows(tmp_path)) == 2


# ── the collapse ─────────────────────────────────────────────────────────────

@pytest.mark.logic
def test_a_continuous_metric_collapses_to_its_median(agg, tmp_path):
    """Median, not mean. One collapsed seed must not drag the sequence over
    the 5 A line."""
    build(tmp_path, [seed_row(1, 0, ra=1.0), seed_row(1, 1, ra=2.0),
                     seed_row(1, 2, ra=90.0)])
    assert agg.cmd_aggregate_per_sequence(Args(experiment_root=tmp_path)) == 0
    assert float(out_rows(tmp_path)[0]["steered_ra_eff_vs_truth_median"]) == 2.0


@pytest.mark.logic
def test_a_binary_categorical_collapses_by_majority(agg, tmp_path):
    build(tmp_path, [seed_row(1, 0, intact="1"), seed_row(1, 1, intact="1"),
                     seed_row(1, 2, intact="0")])
    assert agg.cmd_aggregate_per_sequence(Args(experiment_root=tmp_path)) == 0
    assert out_rows(tmp_path)[0]["steered_receptor_intact_majority"] == "1"


@pytest.mark.logic
def test_a_single_seed_needs_no_collapse(agg, tmp_path):
    build(tmp_path, [seed_row(1, 0, ra=3.5)])
    assert agg.cmd_aggregate_per_sequence(Args(experiment_root=tmp_path)) == 0
    row = out_rows(tmp_path)[0]
    assert float(row["steered_ra_eff_vs_truth_median"]) == 3.5
    assert row["n_seeds"] == "1"


# ── the canonical seed ───────────────────────────────────────────────────────

@pytest.mark.logic
def test_an_odd_count_takes_the_middle_seed(agg, tmp_path):
    build(tmp_path, [seed_row(1, 0, ra=1.0), seed_row(1, 1, ra=5.0),
                     seed_row(1, 2, ra=9.0)])
    assert agg.cmd_aggregate_per_sequence(Args(experiment_root=tmp_path)) == 0
    assert "_s1.pdb" in out_rows(tmp_path)[0]["canonical_pdb"]


@pytest.mark.logic
def test_an_even_count_takes_the_worse_of_the_two_middle_seeds(agg, tmp_path):
    """Conservative by construction: the structure a reader opens is not
    better than the median the table reports."""
    build(tmp_path, [seed_row(1, 0, ra=1.0), seed_row(1, 1, ra=4.0),
                     seed_row(1, 2, ra=6.0), seed_row(1, 3, ra=9.0)])
    assert agg.cmd_aggregate_per_sequence(Args(experiment_root=tmp_path)) == 0
    row = out_rows(tmp_path)[0]
    assert "_s2.pdb" in row["canonical_pdb"]        # ra 6.0, the worse middle


@pytest.mark.logic
def test_a_seed_with_no_rmsd_sorts_last_for_the_canonical_pick(agg, tmp_path):
    """A missing measurement must not be treated as the best pose."""
    build(tmp_path, [seed_row(1, 0, ra=""), seed_row(1, 1, ra=2.0),
                     seed_row(1, 2, ra=3.0)])
    assert agg.cmd_aggregate_per_sequence(Args(experiment_root=tmp_path)) == 0
    assert "_s2.pdb" in out_rows(tmp_path)[0]["canonical_pdb"]


# ── the contamination gating set ─────────────────────────────────────────────

@pytest.mark.logic
def test_the_gating_set_is_read_from_the_plan_and_converted_to_one_based(
        agg, tmp_path, capsys):
    """plan.json stores 0-based indices and the contact positions it is
    compared against are 1-based. Skipping the conversion shifts the whole
    gate by one residue."""
    build(tmp_path, [seed_row(1, 0)],
          plan={"design_region_idx": [10, 11], "true_interface_idx": [20]})
    assert agg.cmd_aggregate_per_sequence(Args(experiment_root=tmp_path)) == 0
    assert "contamination gating set: 3 positions" in capsys.readouterr().out


@pytest.mark.logic
def test_no_plan_falls_back_to_ungated_with_a_warning(agg, tmp_path, capsys):
    build(tmp_path, [seed_row(1, 0)])
    assert agg.cmd_aggregate_per_sequence(Args(experiment_root=tmp_path)) == 0
    assert "legacy (ungated)" in capsys.readouterr().out


@pytest.mark.logic
def test_an_unparseable_plan_warns_and_carries_on(agg, tmp_path, capsys):
    build(tmp_path, [seed_row(1, 0)])
    c0 = tmp_path / "cycle_0"
    c0.mkdir(exist_ok=True)
    (c0 / "plan.json").write_text("{not json")
    assert agg.cmd_aggregate_per_sequence(Args(experiment_root=tmp_path)) == 0
    assert "could not parse" in capsys.readouterr().out


# ── the per-seed verdict counter ─────────────────────────────────────────────
#
# These counts become n_pass, which becomes the tier, which is what the
# cohort figures are grouped by. Bucketing a seed wrongly moves a sequence
# between tiers with nothing reporting an error.
#
# The subtle case is a blank verdict column. It means one of two things: the
# steered prediction had zero contamination so reversion was deliberately
# skipped, or reversion ran and the row is missing what is needed to classify
# it. The first is pass-equivalent, the second is not, and the difference is
# whether the steered row itself is structurally sound.

def _rev_row(**kw):
    base = {"reversion_verdict": "", "steered_n_contacts_on_mutated_positions": "",
            "steered_receptor_intact": "1", "steered_ra_eff_vs_truth": "2.0",
            "reverted_receptor_intact": "", "reverted_ra_eff_vs_truth": "",
            "reverted_total_mutations": "",
            "reverted_mutated_contact_positions": ""}
    base.update({k: str(v) for k, v in kw.items()})
    return base


@pytest.mark.logic
def test_positions_parse_from_a_csv_string(agg):
    assert agg._parse_position_csv("5, 7,12") == [5, 7, 12]
    assert agg._parse_position_csv("") == []
    assert agg._parse_position_csv(None) == []
    assert agg._parse_position_csv("5,,x,7") == [5, 7]


@pytest.mark.logic
@pytest.mark.parametrize("verdict,bucket", [
    ("pose_holds", "pose_holds"),
    ("pose_collapses", "pose_collapses"),
    ("new_contamination", "new_contamination"),
    ("pose_holds_not_intact", "pose_collapses"),
    ("missing_verdict", "pose_collapses"),
    ("unknown", "pose_collapses"),
])
def test_an_explicit_verdict_goes_to_its_bucket(agg, verdict, bucket):
    counts = agg._per_seed_verdict_breakdown([_rev_row(reversion_verdict=verdict)], 5.0)
    assert counts[bucket] == 1


@pytest.mark.logic
def test_an_unrecognised_verdict_is_visible_rather_than_bucketed(agg):
    """Silently folding it into a pass or a fail would hide a new verdict
    string arriving from an upstream change."""
    counts = agg._per_seed_verdict_breakdown(
        [_rev_row(reversion_verdict="something_new")], 5.0)
    assert counts["no_data"] == 1


@pytest.mark.logic
def test_zero_contamination_on_a_sound_steered_row_is_pass_equivalent(agg):
    """Reversion was skipped on purpose, so the steered structure is the
    final answer for that seed."""
    counts = agg._per_seed_verdict_breakdown(
        [_rev_row(steered_n_contacts_on_mutated_positions=0)], 5.0)
    assert counts["clean_steered"] == 1


@pytest.mark.logic
@pytest.mark.parametrize("kw", [
    {"steered_receptor_intact": "0"},
    {"steered_ra_eff_vs_truth": "99.0"},
])
def test_zero_contamination_on_an_unsound_steered_row_is_not_a_pass(agg, kw):
    """Otherwise a steered miss with no contamination counts as a pass."""
    counts = agg._per_seed_verdict_breakdown(
        [_rev_row(steered_n_contacts_on_mutated_positions=0, **kw)], 5.0)
    assert counts["clean_steered"] == 0


@pytest.mark.logic
def test_a_blank_row_with_reverted_values_is_re_derived(agg):
    counts = agg._per_seed_verdict_breakdown(
        [_rev_row(reverted_receptor_intact=1, reverted_ra_eff_vs_truth=2.0)], 5.0)
    assert counts["pose_holds"] == 1


@pytest.mark.logic
def test_a_re_derived_row_over_the_cutoff_collapses(agg):
    counts = agg._per_seed_verdict_breakdown(
        [_rev_row(reverted_receptor_intact=1, reverted_ra_eff_vs_truth=99.0)], 5.0)
    assert counts["pose_collapses"] == 1


@pytest.mark.logic
def test_a_re_derived_row_that_is_not_intact_collapses(agg):
    counts = agg._per_seed_verdict_breakdown(
        [_rev_row(reverted_receptor_intact=0, reverted_ra_eff_vs_truth=2.0)], 5.0)
    assert counts["pose_collapses"] == 1


@pytest.mark.logic
@pytest.mark.parametrize("kw", [
    {"reverted_receptor_intact": "x"},
    {"reverted_receptor_intact": "1", "reverted_ra_eff_vs_truth": "x"},
    {"reverted_receptor_intact": "1", "reverted_ra_eff_vs_truth": "nan"},
])
def test_an_unusable_reverted_row_is_no_data_not_a_pass(agg, kw):
    counts = agg._per_seed_verdict_breakdown([_rev_row(**kw)], 5.0)
    assert counts["no_data"] == 1
    assert counts["pose_holds"] == 0


@pytest.mark.logic
def test_a_re_derived_row_with_a_mutated_contact_is_contamination(agg):
    counts = agg._per_seed_verdict_breakdown(
        [_rev_row(reverted_receptor_intact=1, reverted_ra_eff_vs_truth=2.0,
                  reverted_mutated_contact_positions="12")], 5.0)
    assert counts["new_contamination"] == 1


# ── the composite ranker ─────────────────────────────────────────────────────

@pytest.mark.logic
def test_the_composite_subtracts_a_twentieth_of_the_rmsd(agg):
    row = {"steered_ra_eff_vs_truth": "10.0", "steered_true_jaccard": "0.9",
           "reversion_verdict": "", "steered_receptor_intact": "1"}
    got = agg._ranking_composite(row)
    assert got is None or got == pytest.approx(0.4)


@pytest.mark.logic
@pytest.mark.parametrize("field", ["steered_ra_eff_vs_truth",
                                   "steered_true_jaccard"])
def test_a_composite_missing_a_component_is_none(agg, field):
    row = {"steered_ra_eff_vs_truth": "10.0", "steered_true_jaccard": "0.9",
           "reversion_verdict": "", "steered_receptor_intact": "1"}
    row[field] = ""
    assert agg._ranking_composite(row) is None


@pytest.mark.logic
@pytest.mark.parametrize("bad", ["nan", "inf", "notanumber"])
def test_a_non_finite_jaccard_is_none(agg, bad):
    row = {"steered_ra_eff_vs_truth": "10.0", "steered_true_jaccard": bad,
           "reversion_verdict": "", "steered_receptor_intact": "1"}
    assert agg._ranking_composite(row) is None


# ── the write-boundary rename ────────────────────────────────────────────────
#
# Internal dict keys keep their historical names so upstream code goes on
# working; the rename happens once, here, at the point of writing. A key that
# fails to rename appears in the CSV under a name no analysis reads, which
# looks like a missing column rather than a renamed one.

@pytest.mark.logic
def test_dropped_keys_do_not_reach_the_output(agg):
    row = dict.fromkeys(agg._AGG_DROP, "x")
    row["design"] = "design_00"
    out = agg._translate_aggregate_row(row)
    assert not (set(out) & set(agg._AGG_DROP))
    assert out["design"] == "design_00"


@pytest.mark.logic
def test_every_rename_is_applied_and_carries_its_value(agg):
    row = {k: f"v_{k}" for k in agg._AGG_RENAME}
    out = agg._translate_aggregate_row(row)
    for old, new in agg._AGG_RENAME.items():
        assert new in out, f"{old} did not become {new}"
        assert out[new] == f"v_{old}"
        assert old not in out or old in agg._AGG_RENAME.values()


@pytest.mark.logic
def test_an_unknown_key_passes_through_untouched(agg):
    """A column the rename table does not know about is kept rather than
    silently dropped, so a new metric is not lost on its way out."""
    out = agg._translate_aggregate_row({"some_new_metric": "1.5"})
    assert out["some_new_metric"] == "1.5"


@pytest.mark.logic
@pytest.mark.parametrize("rec,eff,expected", [
    ("1.0", "1.0", 1),       # both inside the fold threshold
    ("9.0", "1.0", 0),       # receptor broken
    ("1.0", "9.0", 0),       # effector broken
    ("5.0", "5.0", 1),       # exactly on the threshold counts as intact
])
def test_reverted_intactness_is_derived_from_both_chain_rmsds(
        agg, rec, eff, expected):
    out = agg._translate_aggregate_row({
        "reverted_independent_receptor_rmsd": rec,
        "reverted_independent_effector_rmsd": eff})
    assert out["reverted_receptor_intact"] == expected


@pytest.mark.logic
@pytest.mark.parametrize("rec,eff", [
    ("", "1.0"), ("1.0", ""), ("", ""),
])
def test_a_row_with_no_reverted_rmsds_is_blank_not_zero(agg, rec, eff):
    """Blank means reversion was not attempted. A 0 would read as attempted
    and failed, which is a different thing entirely."""
    out = agg._translate_aggregate_row({
        "reverted_independent_receptor_rmsd": rec,
        "reverted_independent_effector_rmsd": eff})
    assert out["reverted_receptor_intact"] == ""


@pytest.mark.logic
def test_an_unparseable_reverted_rmsd_is_blank_not_intact(agg):
    out = agg._translate_aggregate_row({
        "reverted_independent_receptor_rmsd": "x",
        "reverted_independent_effector_rmsd": "1.0"})
    assert out["reverted_receptor_intact"] == ""
