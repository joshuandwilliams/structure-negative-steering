# 10 — `bin/negsteer_aggregate.py`, 2,580 lines

**Proposal: one change, 24 lines out. The bigger-looking targets do not hold
up.**

This is the largest file after the engine and it has the most literal
assignments in `bin/`, 47. I expected the biggest win here. It is not.

## Where the 47 assignments are

| Function | Assignments | Lines |
|:---|---:|---:|
| `cmd_aggregate_per_sequence` | 14 | 419 |
| `cmd_aggregate` | 9 | 291 |
| `_process_passing_row` | 9 | 202 |
| `_populate_reverted_mutations` | 8 | 191 |

Unlike `extract_passing.extract_row`, these are **not** a column-copying
block. They are scattered through long functions, one or two at a time, each
setting a different thing at a different point. There is no list of names to
factor out, because the assignments do not form a group.

## The one real repetition: eight identical failure returns

`_process_passing_row` has this shape eight times, at lines 2009, 2045, 2109,
2114, 2124, 2130, 2141 and 2147:

```python
msg = f"pdb not found: {pdb_path_str}"
print(f"    ERROR: {msg}")
out_row["steered_error"] = msg
return out_row, "failed"
```

Four lines, eight times, differing only in the message and in whether the
print carries the progress prefix. A closure over `out_row` collapses each to
one line:

```python
def fail(msg, *, prefix=False):
    """Record the failure on the row and return the caller's result.

    One failure costs one design rather than the whole stage: the row comes
    back with steered_error set and the metric columns blank, which the
    aggregate CSV carries so the loss is visible rather than silent.
    """
    where = (f"  [{passing_progress}/{n_passing_total}] {pathway_label}: SKIP"
             if prefix else "    ERROR:")
    print(f"{where} ({msg})" if prefix else f"{where} {msg}")
    out_row["steered_error"] = msg
    return out_row, "failed"
```

Then:

```python
if pdb_path is None or not pdb_path.exists():
    return fail(f"pdb not found: {pdb_path_str}", prefix=True)
...
if not cm_rows:
    return fail("compute_metrics.py CSV was empty")
```

**24 lines out**, and the invariant that every failure path leaves the row
recoverable is stated once rather than implied eight times.

The one case that keeps its own code is the `returncode != 0` branch, because
it also prints `proc.stdout` under `--verbose`. Forcing that through the
helper would mean a second keyword argument used once.

## What I rejected

**`cmd_aggregate_per_sequence`, 419 lines, 14 assignments.** I read it looking
for a column group. The 14 assignments set `pathway`, `parent_pathway`,
`cycle`, `min_dist_to_upstream` and nine per-seed aggregates, at six different
points, interleaved with the grouping logic that decides what a sequence
group even is. There is no shared rule to name. Breaking the function up is a
different proposal from this one, and a riskier one, because the grouping
logic is what decides which prediction a thesis figure reports.

**The three copies of the `compute_metrics.py` argv builder.** Earlier
analysis flagged `negsteer_aggregate.py:2073`, `reversion.py:884` and
`reversion.py:1298` as three places building the same subprocess command line.
They share eight flags, but each adds different ones and two of them are in a
different file, so factoring means a new shared module for one function. See
11 for why I did not.

## Speed

Measured nothing here, because nothing changes in a loop. The `fail` closure
is created once per call to `_process_passing_row`, which also spawns a
subprocess running `compute_metrics.py` — tens of milliseconds against a
closure's microsecond.

## Is every line necessary

The eight failure paths are. Each one is a different thing going wrong with
one design: a missing PDB, a subprocess timeout, a non-zero exit, a missing
CSV, an unreadable CSV, an empty CSV. They exist because the alternative is
losing the whole aggregate stage after the GPU time is already spent, and the
message distinguishes them in the CSV so a failure can be diagnosed without
rerunning. Collapsing them into one generic `except Exception` would save
another 10 lines and lose that.
