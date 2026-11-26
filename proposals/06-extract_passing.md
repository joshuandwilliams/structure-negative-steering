# 06 — `bin/extract_passing.py`, 629 lines

**Proposal: accept.** 6.1× faster per row, 243 → 161 lines on the part that
changes, and it closes a defect that grows `sys.path` without bound.

This file is where I started, and the benchmark found something I was not
looking for. Read the second change first: it is the important one.

Code for all three parts is in `proposals/code/06-*.py`.

---

## Change 2 first: `_compute_confidence_flag` re-imports on every call

```python
if pass_frac_min is None or iptm_min is None or ...:
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from pipeline_thresholds import PipelineInternalThresholds
        _t = PipelineInternalThresholds.default()
```

This runs **once per row**. Three consequences, in order of seriousness.

**It grows `sys.path` without bound.** There is no `if not in sys.path`
guard, so every call appends another copy of `bin/`. Measured: 500 calls add
500 duplicate entries. Every import after that searches a longer path.

The repo already asserts this must not happen. From
`tests/characterization/test_modules_repair_sys_path.py:66`:

> `test_a_module_already_on_the_path_does_not_duplicate_it` — "The guard is
> there so a repeated import cannot grow `sys.path` without bound."

That test covers the module-level inserts. This function has the same bug and
no guard.

**It dominates the runtime.** Profiling `extract_row` over the fixture rows:
`_compute_confidence_flag` is 1.634 s of 2.005 s cumulative, **82%**. Inside
it, `Path.resolve()` → `realpath` → `_joinrealpath` is about 1.0 s, **half of
`extract_row`'s total time**, spent re-resolving `__file__`.

**The fix** resolves the thresholds once at import and turns the four
near-identical `try`/`float`/`compare` blocks into one table:

```python
_CONFIDENCE_CHECKS = (
    ("pae_pass_frac_median", "<", _DEFAULT_PASS_FRAC_MIN, "low_pass_frac", 1),
    ("iptm_median", "<", _DEFAULT_IPTM_MIN, "low_iptm", 1),
    ("complex_plddt_median", "<", _DEFAULT_COMPLEX_PLDDT_MIN, "low_plddt", 1),
    ("ipae_median", ">", _DEFAULT_IPAE_MAX, "high_ipae", 0),
)
```

The last column is the value used when the metric is absent, chosen so a
missing metric does not trigger. That was implicit in the old code's
`out.get("iptm_median", 1)` versus `out.get("ipae_median", 0)`, and it is
easy to get backwards, so it is now written down.

| | Current | Proposed |
|:---|---:|---:|
| per call | 66.89 µs | 0.81 µs |
| `sys.path` entries added per 500 calls | 500 | 0 |

**82× faster**, and 24,000 randomised cases — every combination of the four
metrics at threshold boundaries, absent, empty, and non-numeric, with and
without explicit overrides — give identical flags.

---

## Change 1: the column spec

`extract_row` is 152 lines for 62 statements, and 39 of those statements are
`out["x"] = r.get("x", "")`. The 46 column names are also written out a
second time in `OUTPUT_FIELDS`, so 93 name-spellings for 47 columns.

One ordered list of `(column, rule)` replaces both, and `OUTPUT_FIELDS` is
derived from it:

```python
OUTPUT_FIELDS = [c + "_median" if rule is CONF else c for c, rule in COLUMNS]
```

That matches the committed 47-entry list exactly, in order. Each column name
is written once, and the two lists can no longer drift — which matters in one
direction: `csv.DictWriter` raises on a key that is not a fieldname, but
writes a blank for a fieldname the row never set, so a column `extract_row`
forgot was silent.

`extract_row`'s body becomes one loop per rule, six loops:

```python
out = {}
for c in _AS_IS_COLS:
    out[c] = r.get(c, "")
for c in _STEERED_COLS:
    out[c] = r.get("steered_" + c, "")
...
```

### I tried three forms and measured all of them

This is where the speed constraint changed the answer:

| Form | µs/row | vs current |
|:---|---:|---:|
| current, 39 straight assignments | 87.1 | 1.00× |
| branch per column inside one loop | 104.0 | **1.19× slower** |
| `out.update({...})` per rule | 113.0 | **1.30× slower** |
| plain `for` per rule, assigning into `out` | 102.2 | 1.00× |

The first two are the obvious ways to write it and both are meaningfully
slower: a 47-iteration `if`/`elif` chain costs more than straight-line
assignment, and `.update()` builds an intermediate dict and merges it, which
is two passes. Only the plain per-rule loop is free, because it does exactly
what the current code does with the column name coming from a tuple instead
of a literal.

So the proposed form is the third one. I would have shipped the first without
the benchmark.

---

## Change 3: the mutation columns

The seven `SEEDS` columns are filled in an `if per_seed_rows:` branch and
blanked in an `else:` that spells all seven out again. `dict.fromkeys` does
the blanking, and the whole thing moves to its own function so the docstring
can say why the steered set takes the first seed while the reverted set is
summarised across seeds.

---

## Combined effect

| | Current | Proposed |
|:---|---:|---:|
| `extract_row` per row | 91.7 µs | 15.0 µs (**6.1×**) |
| `extract_row` lines | 152 | 40 |
| spec + `extract_row` lines | 243 | 161 |
| column names written | 93 | 47 |

## Verification

- 151 `aggregated_results.csv` rows from the fixtures, with and without seed
  rows: identical.
- 4,000 randomised rows across all six outcome classes with randomly absent
  keys on both the `steered_` and `reverted_` sides: identical.
- 24,000 randomised confidence-flag cases: identical.
- `OUTPUT_FIELDS` derived from `COLUMNS` equals the committed list, in order.

## Is every line necessary

The `try`/`except (ValueError, TypeError)` around each `float()` is. The
columns arrive from a CSV, so `""` and `"nan"` are both possible, and a flag
computation that raises would fail the stage after the GPU time is spent. The
`continue` on a parse failure is the same decision as before: an unreadable
metric is not evidence of low confidence.
