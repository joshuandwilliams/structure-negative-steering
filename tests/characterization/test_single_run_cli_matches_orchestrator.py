"""The tool and the runner behind the benchmark produce the same science.

test_negsteer_cli.py asserts the CLI emits the same stage commands as
bin/negative_steering_run_one.sh. That is an argv-level claim. It cannot catch
a difference in what those commands do once a GPU is involved.

This closes that gap. Both paths run tests/test_run/config.yml on the same
hardware with the same seed, and their run directories are compared:

    sbatch tests/test_run.slurm.sh            -> run/
    sbatch tests/test_run.slurm.sh --via-cli  -> run_cli/

Skips cleanly unless both exist, so it costs nothing when only one has run.
The tree comes from single_run.py, so NEGSTEER_SINGLE_RUN_DIR points this and
test_engine_single_run.py at the same place. It used to read
NEGSTEER_SINGLE_RUN_BASE, which nothing sets, so on a relocated tree it looked
at the checkout, found nothing and skipped.

On tolerance: Boltz is seeded, but CUDA kernel scheduling is not bit-exact
across runs, so structural metrics are compared with a tolerance rather than
for equality. The tolerance is deliberately tight. It is there to absorb
floating-point drift, not a different pose. Anything categorical, meaning row
counts, statuses, mutation positions and tier, must match exactly, because
nondeterminism cannot explain a difference in those.
"""

from __future__ import annotations

import json

import pytest

# Shared with test_engine_single_run.py, which asserts on the same tree.
# single_run.py is a sibling module and pytest puts this directory on sys.path.
from single_run import RUN_NAME, SINGLE_RUN
from single_run import read_csv as _read

ORCH = SINGLE_RUN / "run"
CLI = SINGLE_RUN / "run_cli"

# Kernel scheduling moves the last decimal place, not the answer.
RMSD_TOL_A = 0.15

needs_both = pytest.mark.skipif(
    not ((ORCH / "raw_per_seed_results.csv").is_file()
         and (CLI / "raw_per_seed_results.csv").is_file()),
    reason=f"needs both {ORCH} and {CLI}. Run the single-run job twice, once with "
           "--via-cli.")


@pytest.mark.check_test_run
def test_the_two_run_directories_are_not_the_same_tree():
    """Not a comparison, a guard on the comparison.

    tests/test_run.slurm.sh runs the --via-cli job against a `.cli_view`
    directory whose `run` is a symlink to `run_cli`, so that
    test_engine_single_run.py can assert on the CLI output without learning a
    second layout. Point this module at that view and ORCH resolves to the CLI
    run, CLI resolves to nothing, and every test below either skips or compares
    the CLI run against itself and passes for free.

    This runs unconditionally, so a tree that cannot support the comparison
    fails here rather than passing quietly below.
    """
    if not ORCH.exists():
        pytest.skip(f"no run/ under {SINGLE_RUN}")

    wrong_dir = (f"Point NEGSTEER_SINGLE_RUN_DIR at the directory holding both "
                 f"run/ and run_cli/, not at the .cli_view shim. It is at "
                 f"{SINGLE_RUN}.")

    # The shim's only possible shape: run/ is a link whose target is run_cli/.
    assert ORCH.resolve().name != "run_cli", (
        f"{ORCH} resolves to {ORCH.resolve()}, so the orchestrator side is "
        f"really the CLI run. {wrong_dir}")

    if CLI.exists():
        assert ORCH.resolve() != CLI.resolve(), (
            f"{ORCH} and {CLI} are one tree at {ORCH.resolve()}, so this "
            f"module would compare a run against itself. {wrong_dir}")


def _key(row: dict) -> tuple[str, str]:
    return row.get("design", ""), row.get("seed_index", "")


@pytest.mark.check_test_run
@needs_both
def test_both_paths_produced_the_same_rows():
    """A different row count means one path ran a different number of designs,
    which no amount of GPU nondeterminism explains."""
    orch = {_key(r) for r in _read(ORCH / "raw_per_seed_results.csv")}
    cli = {_key(r) for r in _read(CLI / "raw_per_seed_results.csv")}
    assert orch == cli, (
        f"only the orchestrator produced {sorted(orch - cli)}; "
        f"only the CLI produced {sorted(cli - orch)}")


@pytest.mark.check_test_run
@needs_both
def test_both_paths_wrote_the_same_columns():
    """A missing column is invisible to the analyses rather than an error, so
    a column the tool drops would silently degrade every downstream table."""
    orch = set(_read(ORCH / "raw_per_seed_results.csv")[0])
    cli = set(_read(CLI / "raw_per_seed_results.csv")[0])
    assert orch == cli, f"column difference: {orch ^ cli}"


# Columns whose values are allowed to differ, each for a named reason. Anything
# not listed here must match, because nothing else about the two runs differs.
ALLOWED_TO_DIFFER = {
    # The run directory is literally named differently, run/ against run_cli/,
    # so every absolute path inside it differs. The filenames are compared
    # instead, below.
    "pdb",
}


# Continuous metrics are compared with a relative tolerance, for the reason in
# the module docstring: Boltz is seeded but CUDA kernel scheduling is not
# bit-exact. Measured on the 2026-11-28 pair, 9 of 10 rows matched to the last
# digit and one differed by 0.01 pLDDT points, 80.60 against 80.59, which is
# the 4th significant figure. complex_plddt is the same quantity over 100, so
# that is one value reported twice rather than two independent differences.
#
# 1e-3 relative absorbs that and nothing larger. The absolute floor covers
# columns that sit near zero, where a relative tolerance is meaningless: the
# ipSAE columns are legitimately 0.0 on this target.
REL_TOL = 1e-3
ABS_TOL = 1e-4


def _num(v):
    """float, or None when the text is not a number."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


@pytest.mark.check_test_run
@needs_both
def test_every_column_agrees_except_the_run_paths():
    """The assertion the other comparisons were missing.

    Until this existed, this module compared column NAMES, statuses, ra_eff,
    mutations and tier. The first cluster comparison, on 2026-11-27, passed
    every one of those while 24 of 84 columns actually differed: 23 were blank
    on the CLI side, including every confidence metric, every ipSAE column, the
    interface pLDDT, the weighted Jaccard and the contamination check.

    Two default divergences caused it, both since fixed and both now pinned by
    test_postprocess_defaults.py. This is the assertion that would have caught
    them on the data rather than on the source.

    A value against a blank is always a failure, whatever the tolerance. That
    is the shape the defect took, and absent data is not a small numeric
    difference.
    """
    orch = {_key(r): r for r in _read(ORCH / "raw_per_seed_results.csv")}
    cli = {_key(r): r for r in _read(CLI / "raw_per_seed_results.csv")}
    columns = [c for c in _read(ORCH / "raw_per_seed_results.csv")[0]
               if c not in ALLOWED_TO_DIFFER]
    assert columns, "no comparable columns, so this asserted nothing"

    missing, differing = {}, {}
    for col in columns:
        for k in orch:
            a = (orch[k].get(col) or "").strip()
            b = (cli[k].get(col) or "").strip()
            if (a == "") != (b == ""):
                missing.setdefault(col, []).append((k, a, b))
                continue
            if a == b:
                continue
            x, y = _num(a), _num(b)
            if x is None or y is None:
                differing.setdefault(col, []).append((k, a, b))
            elif abs(x - y) > max(ABS_TOL, REL_TOL * max(abs(x), abs(y))):
                differing.setdefault(col, []).append((k, a, b))

    assert not missing, (
        "a column is populated on one path and blank on the other:\n"
        + "\n".join(
            f"  {c}: {len(rows)}/{len(orch)} rows, e.g. {rows[0][0]} "
            f"orch={rows[0][1]!r} cli={rows[0][2]!r}"
            for c, rows in sorted(missing.items()))
        + "\n\nA blank is not a measured null. The analyses read it as "
          "absent data.")

    assert not differing, (
        f"columns disagree by more than {REL_TOL:g} relative:\n"
        + "\n".join(
            f"  {c}: {len(rows)}/{len(orch)} rows, e.g. {rows[0][0]} "
            f"orch={rows[0][1]!r} cli={rows[0][2]!r}"
            for c, rows in sorted(differing.items())))


@pytest.mark.check_test_run
@needs_both
def test_the_pdb_filenames_match_even_though_the_paths_cannot():
    """`pdb` is exempt above because the two run directories have different
    names. The basename still has to match, or the two paths predicted
    different things."""
    orch = {_key(r): r for r in _read(ORCH / "raw_per_seed_results.csv")}
    for k, row in {_key(r): r for r in _read(
            CLI / "raw_per_seed_results.csv")}.items():
        a = (row.get("pdb") or "").rsplit("/", 1)[-1]
        b = (orch[k].get("pdb") or "").rsplit("/", 1)[-1]
        assert a == b, f"{k}: predicted {a!r} against {b!r}"


@pytest.mark.check_test_run
@needs_both
def test_the_confidence_metrics_are_present_on_both_sides():
    """Stated separately from the comparison above, because two runs that both
    wrote nothing would agree.

    compute-final-metrics is the only thing that writes these. The CLI skipped
    them for a whole run while agreeing on every other assertion in this file.
    """
    required = ("steered_iptm", "steered_complex_plddt",
                "steered_interface_plddt", "steered_ipsae_min",
                "steered_n_contacts_on_mutated_positions")
    for label, path in (("orchestrator", ORCH), ("CLI", CLI)):
        rows = [r for r in _read(path / "raw_per_seed_results.csv")
                if r.get("design") != "initial"]
        assert rows, f"{label} produced no steered rows"
        for col in required:
            filled = sum(1 for r in rows if (r.get(col) or "").strip())
            assert filled == len(rows), (
                f"{label}: {col} is blank on {len(rows) - filled} of "
                f"{len(rows)} steered rows. compute-final-metrics did not run "
                f"on them, so the analyses will read absent data.")


@pytest.mark.check_test_run
@needs_both
def test_every_row_agrees_on_status():
    orch = {_key(r): r for r in _read(ORCH / "raw_per_seed_results.csv")}
    for row in _read(CLI / "raw_per_seed_results.csv"):
        assert row.get("status") == orch[_key(row)].get("status"), \
            f"{_key(row)}: status differs"


@pytest.mark.check_test_run
@needs_both
def test_the_pose_agrees_within_tolerance():
    """The number the whole method is judged on. If ra_eff differs by more
    than kernel noise, the tool is not doing what the benchmark runner did."""
    orch = {_key(r): r for r in _read(ORCH / "raw_per_seed_results.csv")}
    compared = 0
    for row in _read(CLI / "raw_per_seed_results.csv"):
        a, b = row.get("steered_ra_eff_vs_truth"), orch[_key(row)].get(
            "steered_ra_eff_vs_truth")
        if a in ("", None) or b in ("", None):
            continue
        assert abs(float(a) - float(b)) <= RMSD_TOL_A, (
            f"{_key(row)}: ra_eff {a} via the CLI against {b} via the "
            f"orchestrator, beyond the {RMSD_TOL_A} A tolerance")
        compared += 1
    assert compared, "no comparable ra_eff values, so this asserted nothing"


@pytest.mark.check_test_run
@needs_both
def test_the_same_residues_were_mutated():
    """Steering picks positions from a seeded ranking. Different positions
    mean a different steering decision, not a different rounding."""
    orch = {_key(r): r for r in _read(ORCH / "raw_per_seed_results.csv")}
    for row in _read(CLI / "raw_per_seed_results.csv"):
        assert row.get("steered_mutations_chimerax", "") == \
            orch[_key(row)].get("steered_mutations_chimerax", ""), \
            f"{_key(row)}: different residues mutated"


@pytest.mark.check_test_run
@needs_both
def test_the_representative_gets_the_same_tier():
    """Tier is what the cohort-level analyses key on."""
    a = _read(ORCH / "cross_sequence_summary.csv")
    b = _read(CLI / "cross_sequence_summary.csv")
    assert len(a) == len(b) == 1
    assert a[0]["cross_tier"] == b[0]["cross_tier"]


@pytest.mark.check_test_run
@needs_both
def test_only_the_cli_run_carries_a_result_manifest():
    """The manifest is the tool's contract and the orchestrator has no such
    thing. Its absence on one side is the expected asymmetry, and asserting it
    stops the comparison being read as "these are interchangeable"."""
    manifest = CLI / "negsteer_result.json"
    assert manifest.is_file(), "the CLI run wrote no negsteer_result.json"
    assert not (ORCH / "negsteer_result.json").exists()

    payload = json.loads(manifest.read_text())
    assert payload["name"] == RUN_NAME
    assert payload["status"] in {"ok", "skipped_steering"}
    assert payload["runtime_seconds"] > 0
