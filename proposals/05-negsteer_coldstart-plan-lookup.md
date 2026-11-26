# 05 — `bin/negsteer_coldstart.py`, the reversion-plan lookup

**Proposal: accept.** The second of the two changes where speed is the point.
40× at the benchmark's dose, 77× at n=50, and four lines shorter.

## The problem

Inside `cmd_kickoff_finalize`'s loop over candidates, for every candidate whose
verdict is `pose_holds`:

```python
rp_path = cycle0_dir / "reversion_plan.json"
if rp_path.exists():
    rp = json.loads(rp_path.read_text())
    for entry in rp.get("entries", []):
        if entry.get("label") == label:
            reverted_fasta = Path(entry["reverted_receptor_fasta"])
            break
```

`cycle0_dir` is fixed for the whole stage, so this re-reads and re-parses the
same JSON file once per candidate, then linearly scans its entries. Both costs
grow with the number of designs, so the whole thing is O(n²) in designs.

## The change

Read it once before the loop and index by label:

```python
# reversion_plan.json is read once rather than per candidate: the file is
# the same for the whole stage, and scanning its entries per candidate made
# the lookup quadratic in the number of designs.
reverted_fasta_by_label = {}
rp_path = cycle0_dir / "reversion_plan.json"
if rp_path.exists():
    for entry in json.loads(rp_path.read_text()).get("entries", []):
        reverted_fasta_by_label.setdefault(
            entry.get("label"), entry.get("reverted_receptor_fasta"))
```

and in the loop:

```python
fasta = reverted_fasta_by_label.get(label)
reverted_fasta = Path(fasta) if fasta else None
```

`setdefault` rather than `[...] =` is deliberate: the old loop `break`s on the
first matching label, so a duplicated label took the first entry. `setdefault`
keeps that. Plain assignment would silently take the last, which is a
different answer on a malformed plan.

## Speed, measured

Entries are 3 per design, one per seed, which is what `write_reversion_plan`
emits. Timed over the full candidate list:

| Designs | Plan entries | Current | Proposed | Speedup |
|---:|---:|---:|---:|---:|
| 3 | 9 | 1.6 ms | 0.20 ms | 8× |
| 20 | 60 | 14.1 ms | 0.35 ms | 40× |
| 50 | 150 | 46.2 ms | 0.60 ms | 77× |

20 designs is `negsteer_n_designs` in the committed benchmark configs; 50 is
the n50 dose-sweep arm. The results were asserted equal inside the benchmark,
so the measurement and the correctness check are the same run.

Same honest context as 04: 46 ms is nothing against a run of hours. It is
worth taking because the code is shorter and the quadratic term goes away, so
it will not surprise anyone at a larger dose.

## Is every line necessary

The `rp_path.exists()` guard is. A run where no design was contaminated writes
no `reversion_plan.json`, and `cmd_kickoff_finalize` runs regardless, so the
missing file is an expected case rather than an error. Without the guard that
run raises instead of finalising.

The `if fasta else None` is, for the same reason: a label with no plan entry
is the `missing_verdict` path a few lines above, which is handled rather than
fatal.
