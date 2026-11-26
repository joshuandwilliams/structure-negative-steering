# 09 — `bin/cross_summary.py`, 940 lines

**Proposal: accept one small factorisation. Separately, a defect I found
while verifying it, which is your call rather than mine. Read the defect
section first.**

---

## The defect: tied ranks are not reproducible

`_assign_cross_ranks` breaks ties with `id(rec)`:

```python
def _composite_rank_key(item):
    comp, rec = item
    tier_order = _TIER_ORDER[rec["cross_tier"]]
    # Negated so higher composites come first within a tier.  id()
    # breaks ties as stably as Python's own sort would.
    return (tier_order, -comp, id(rec))
```

`id()` is the object's memory address. For two records with the same tier and
the same composite, the rank order therefore depends on where Python happened
to allocate the dicts.

Demonstrated: four records with identical tier and identical metrics, ranked
three times in one process on freshly built objects.

```
ranks: [3, 2, 4, 1]
ranks: [3, 1, 4, 2]
ranks: [2, 3, 4, 1]
```

Three different answers for the same input. `cross_rank_by_composite` and
`cross_rank_by_ra_eff` are columns in the committed
`cross_sequence_summary.csv`, so a rerun of a cohort containing tied rows
emits different ranks and the file does not reproduce.

The comment says `id()` "breaks ties as stably as Python's own sort would".
That is not right. Python's sort is stable, meaning it **preserves input
order** for equal keys. Using `id()` as a third key replaces that stability
with an arbitrary order, so the comment describes the opposite of what the
code does.

**The fix is to delete the third key.** `list.sort` is stable, so dropping
`id(rec)` makes tied records keep the order they arrived in, which is the
order the passing-summary inputs were read in, which is deterministic. That is
one line removed and it makes the output reproducible.

I have not folded that into the proposal below, because it **changes the
committed output** for any cohort with ties, and that is your call rather than
mine. For the benchmark the question is narrow: each target is one sequence,
so there is nothing to tie. For a campaign cohort with 128 sequences it is
live.

---

## The factorisation

The two rankings are the same five lines twice:

```python
def _composite_rank_key(item):
    comp, rec = item
    tier_order = _TIER_ORDER[rec["cross_tier"]]
    return (tier_order, -comp, id(rec))
scored_composite.sort(key=_composite_rank_key)
for rank_idx, (_, rec) in enumerate(scored_composite, start=1):
    rec["cross_rank_by_composite"] = rank_idx

def _ra_rank_key(item):
    ra, rec = item
    tier_order = _TIER_ORDER[rec["cross_tier"]]
    return (tier_order, ra, id(rec))
scored_ra.sort(key=_ra_rank_key)
for rank_idx, (_, rec) in enumerate(scored_ra, start=1):
    rec["cross_rank_by_ra_eff"] = rank_idx
```

One helper, called twice:

```python
def _stamp_ranks(scored, column: str) -> None:
    """Sort (value, rec) pairs tier-first and stamp 1-based ranks.

    `value` is already signed so that ascending order is "better first", so
    the caller negates a composite and leaves an ra_eff alone.
    """
    scored.sort(key=lambda pair: (_TIER_ORDER[pair[1]["cross_tier"]],
                                  pair[0], id(pair[1])))
    for rank_idx, (_value, rec) in enumerate(scored, start=1):
        rec[column] = rank_idx
```

with `scored_composite.append((-comp, rec))` and `scored_ra.append((ra, rec))`
carrying the sign. 16 lines → 10, and the "better first" convention is stated
once instead of implied twice.

**What I did not merge.** My first attempt replaced the whole function with a
loop over the two metrics. It was wrong: the two rankings share a filter
(`row_type == "steered"` and `cross_tier in ("A","B","C")`) but the composite
branch also computes a derived value from two columns and stamps
`cross_composite_score` as a side effect. A generic loop lost all three. The
filter and the side effect stay written out.

## Verification

3,000 randomised cohorts of 1 to 8 records across all four tiers, all three
row types, with empty, `nan` and numeric metrics: **identical**, with one
condition — metric values must be unique.

With ties allowed, 152 of 3,000 differed. That is not the factorisation: it is
the `id()` tie-break above, comparing two deep copies at different addresses.
Finding that is the only reason this file has a defect section.

## Speed

Not measured, and I would not expect a difference. Both forms do one
`list.sort` per ranking. A cohort is 1 record for the benchmark and 128 for a
campaign.

## Is every line necessary

The `rec.get("row_type", "steered") != "steered"` filter is. Control and
cold-start rows are in the same record list and must not be ranked against
steered designs — the docstring on `_tier_for_row` explains that controls
"sit outside the cohort ranking by design". Removing the filter would rank a
polyA control alongside a real design.
