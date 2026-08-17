"""HPC tier: pin the shape and completeness of a finished benchmark run.

These read `experiments/benchmarking/<variant>/<target>/run/`, which is
gitignored and produced on the cluster. They skip cleanly when the tree is
absent, so a clean checkout does not fail. Pull a run down first with

    scripts/sync_from_hpc.sh --name benchmarking --results-only

Why this tier exists. The engine writes many outputs per run and the pipeline
deliberately tolerates per-unit failure, so a truncated or partial run looks
like a healthy one from the outside. `cross_summary_cli.py` will happily emit a
tier-none row for a sequence whose input never arrived. The completeness
assertions below are the check the playbook calls for: assert the expected
number of groups, not merely that output exists.

Point somewhere other than the repo's own tree with NEGSTEER_BENCH_DIR.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH = Path(os.environ.get(
    "NEGSTEER_BENCH_DIR", REPO_ROOT / "experiments" / "benchmarking"))

VARIANTS = ("unconstrained", "constrained")

# Written by the engine for every run that reaches the end.
REQUIRED_RUN_FILES = (
    "passing_summary.csv",
    "cross_sequence_summary.csv",
    "aggregated_results.csv",
    "raw_per_seed_results.csv",
    "all_results_multicycle.csv",
    "run_one_runtime_sec.txt",
)
# plan.json is written by every run. summary.txt is not: a target whose cold
# start already passed on every seed short-circuits before the steering summary
# is written, and some steered runs do not emit it either.
REQUIRED_CYCLE_FILES = ("plan.json",)


def _targets(variant: str) -> list[Path]:
    root = BENCH / variant
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if (p / "run").is_dir())


def _all_targets() -> list[tuple[str, Path]]:
    return [(v, t) for v in VARIANTS for t in _targets(v)]


ALL = _all_targets()
needs_runs = pytest.mark.skipif(
    not ALL,
    reason=f"no finished runs under {BENCH}. "
           "Pull one with scripts/sync_from_hpc.sh --name benchmarking --results-only")


def _read(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


# ── completeness ─────────────────────────────────────────────────────────────

@pytest.mark.hpc
@needs_runs
def test_both_variants_cover_the_same_targets():
    """A target present in one variant must be present in the other.

    The two variants are paired per target. A missing one silently halves a
    comparison rather than failing it.
    """
    unc = {p.name for p in _targets("unconstrained")}
    con = {p.name for p in _targets("constrained")}
    assert unc, "no unconstrained runs found"
    assert unc == con, (
        f"variants diverge. unconstrained only: {sorted(unc - con)}, "
        f"constrained only: {sorted(con - unc)}"
    )


@pytest.mark.hpc
@needs_runs
@pytest.mark.parametrize("variant,target", ALL, ids=lambda x: getattr(x, "name", x))
def test_run_carries_every_expected_output(variant, target):
    """Every stage that should have written a file did."""
    run = target / "run"
    missing = [f for f in REQUIRED_RUN_FILES if not (run / f).is_file()]
    assert not missing, f"{variant}/{target.name}: run/ is missing {missing}"

    cycle0 = run / "cycle_0"
    assert cycle0.is_dir(), f"{variant}/{target.name}: no cycle_0/"
    missing = [f for f in REQUIRED_CYCLE_FILES if not (cycle0 / f).is_file()]
    assert not missing, f"{variant}/{target.name}: cycle_0/ is missing {missing}"


@pytest.mark.hpc
@needs_runs
@pytest.mark.parametrize("variant,target", ALL, ids=lambda x: getattr(x, "name", x))
def test_cross_summary_has_exactly_one_representative(variant, target):
    """One run yields one tiered representative, never zero and never several."""
    rows = _read(target / "run" / "cross_sequence_summary.csv")
    assert len(rows) == 1, \
        f"{variant}/{target.name}: {len(rows)} representative rows, expected 1"
    assert rows[0]["cross_tier"] in {"A", "B", "C", "none"}, \
        f"{variant}/{target.name}: unknown tier {rows[0]['cross_tier']!r}"


@pytest.mark.hpc
@needs_runs
@pytest.mark.parametrize("variant,target", ALL, ids=lambda x: getattr(x, "name", x))
def test_tier_none_means_nothing_cleared_the_passing_gate(variant, target):
    """Tier comes from passing_summary.csv, so tier none means it had no rows.

    The representative block on a tier-none row does NOT come from the passing
    gate. It is filled by the fallback from aggregated_results.csv, a best of a
    bad lot diagnostic, so its rep_n_pass can be non-zero while the tier is
    none. Asserting rep_n_pass == 0 there would be asserting the wrong thing.
    """
    row = _read(target / "run" / "cross_sequence_summary.csv")[0]
    n_passing_rows = len(_read(target / "run" / "passing_summary.csv"))
    if row["cross_tier"] == "none":
        assert n_passing_rows == 0, (
            f"{variant}/{target.name}: tier none but passing_summary.csv has "
            f"{n_passing_rows} rows"
        )
    else:
        assert n_passing_rows > 0, (
            f"{variant}/{target.name}: tier {row['cross_tier']} with an empty "
            "passing_summary.csv"
        )


@pytest.mark.hpc
@needs_runs
@pytest.mark.parametrize("variant,target", ALL, ids=lambda x: getattr(x, "name", x))
def test_graded_tiers_agree_with_the_seed_pass_count(variant, target):
    """For A, B and C the tier must follow n_pass/n_seeds exactly."""
    row = _read(target / "run" / "cross_sequence_summary.csv")[0]
    tier = row["cross_tier"]
    if tier == "none":
        pytest.skip("tier none is covered by the passing-gate test")
    n_pass, n_seeds = int(row["rep_n_pass"]), int(row["rep_n_seeds"])
    expected = ("A" if n_pass == n_seeds else
                "B" if 1 < n_pass < n_seeds else
                "C" if n_pass == 1 else "none")
    assert tier == expected, (
        f"{variant}/{target.name}: tier {tier} but {n_pass}/{n_seeds} seeds passed, "
        f"which is tier {expected}"
    )


# ── the cold start ───────────────────────────────────────────────────────────

@pytest.mark.hpc
@needs_runs
@pytest.mark.parametrize("variant,target", ALL, ids=lambda x: getattr(x, "name", x))
def test_plan_records_a_usable_cold_start(variant, target):
    """The cold-start RMSD is the baseline every result is measured against."""
    plan = json.loads((target / "run" / "cycle_0" / "plan.json").read_text())
    rmsd = plan.get("initial_receptor_aligned_effector_rmsd")
    assert rmsd is not None, f"{variant}/{target.name}: plan.json has no cold-start RMSD"
    assert 0.0 < float(rmsd) < 200.0, \
        f"{variant}/{target.name}: implausible cold-start RMSD {rmsd}"
    assert plan.get("wild_type_receptor_seq"), \
        f"{variant}/{target.name}: plan.json records no receptor sequence"


@pytest.mark.hpc
@needs_runs
@pytest.mark.parametrize("variant,target", ALL, ids=lambda x: getattr(x, "name", x))
def test_per_seed_results_contain_the_unsteered_baseline(variant, target):
    """The `initial` rows are what make the before/after a paired comparison."""
    rows = _read(target / "run" / "raw_per_seed_results.csv")
    assert rows, f"{variant}/{target.name}: raw_per_seed_results.csv is empty"
    initial = [r for r in rows if r.get("design") == "initial"]
    assert initial, f"{variant}/{target.name}: no cold-start rows in the per-seed table"
    seeds = {r["seed_index"] for r in initial}
    assert len(seeds) >= 1, f"{variant}/{target.name}: cold start has no seed index"


@pytest.mark.hpc
@needs_runs
@pytest.mark.parametrize("variant,target", ALL, ids=lambda x: getattr(x, "name", x))
def test_recorded_runtime_is_positive(variant, target):
    """run_one_runtime_sec.txt feeds the compute-cost analysis."""
    raw = (target / "run" / "run_one_runtime_sec.txt").read_text().strip()
    assert raw, f"{variant}/{target.name}: runtime file is empty"
    assert float(raw) > 0, f"{variant}/{target.name}: non-positive runtime {raw!r}"


# ── the constrained variant is what it claims to be ──────────────────────────

@pytest.mark.hpc
@needs_runs
@pytest.mark.parametrize("variant,target", ALL, ids=lambda x: getattr(x, "name", x))
def test_launch_script_matches_the_variant_directory(variant, target):
    """A run under constrained/ must actually have carried --boltz-constraints.

    The two variants differ only by that flag. If a run landed in the wrong
    directory the comparison between them is meaningless, and nothing else
    would catch it.
    """
    launch = target / "run" / "launch.sh"
    if not launch.is_file():
        pytest.skip(f"{variant}/{target.name}: no launch.sh recorded")
    body = launch.read_text()
    has_flag = "--boltz-constraints" in body
    assert has_flag == (variant == "constrained"), (
        f"{variant}/{target.name}: launch.sh "
        f"{'has' if has_flag else 'lacks'} --boltz-constraints"
    )
