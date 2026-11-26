# 03 — `bin/pathway_labels.py`, 137 lines

**Proposal: accept.** 137 → 110 lines, and 1.25 to 1.58× faster on the
function that does the work.

Both changes are in `_locate_boltz_sidecar_pdb`. `read_mutations_tsv_full` is
left alone: like `sum_stage_times`, it is guard conditions and those are the
interesting part.

## What changes

### 1. The directory routing, 19 lines → 8

The current code picks the directory to search with a three-branch
`if/elif/else`, where two branches differ only in how they build the suffix:

```python
if prediction_pdb.name == "initial_prediction.pdb":
    experiment_root = prediction_pdb.parent
    initial_dir = experiment_root / "initial"
    if initial_dir.is_dir():
        design_dir = initial_dir
    else:
        design_dir = experiment_root
elif (prediction_pdb.name.startswith("initial_prediction_s")
      and prediction_pdb.name.endswith(".pdb")):
    experiment_root = prediction_pdb.parent
    suffix = prediction_pdb.name[len("initial_prediction_"):-len(".pdb")]
    seed_dir = experiment_root / f"initial_{suffix}"
    if seed_dir.is_dir():
        design_dir = seed_dir
    else:
        design_dir = experiment_root
else:
    design_dir = prediction_pdb.parent
```

The two initial cases are one case. `initial_prediction.pdb` has an empty
suffix and `initial_prediction_s2.pdb` has `_s2`, so slicing after
`initial_prediction` rather than after `initial_prediction_` gives both:

```python
def _sidecar_search_dir(prediction_pdb: Path) -> Path:
    """Where to look for the nested model PDB.

    A design's sidecars are under the design directory. The cold-start
    baselines are not: boltz2_negative_steering.py writes
    `<root>/initial_prediction.pdb` with sidecars in `<root>/initial/`, and
    `<root>/initial_prediction_s<N>.pdb` with sidecars in
    `<root>/initial_s<N>/`. Routing each to its own seed directory is what
    stops the glob picking up a sibling seed's PDBs.
    """
    name = prediction_pdb.name
    if not name.startswith("initial_prediction") or not name.endswith(".pdb"):
        return prediction_pdb.parent
    root = prediction_pdb.parent
    suffix = name[len("initial_prediction"):-len(".pdb")]   # "" or "_s1"
    seed_dir = root / f"initial{suffix}"
    return seed_dir if seed_dir.is_dir() else root
```

Pulling it out as its own function is what makes the docstring possible. In
place it was a comment block in the middle of a longer function.

### 2. `resolve()` hoisted out of the loop — this is the speed part

```python
for p in design_dir.rglob("*.pdb"):
    if p.resolve() == prediction_pdb.resolve():      # ← right-hand side
        continue                                     #   never changes
```

`Path.resolve()` makes syscalls. `prediction_pdb` is the function's argument,
so the right-hand side is the same on every iteration. Hoisting it:

```python
target = prediction_pdb.resolve()
candidates = [p for p in search_dir.rglob("*.pdb")
              if p.resolve() != target and "msa" not in p.parts]
```

Measured over 20 calls each, on a tree with N nested PDBs:

| Nested PDBs | Current | Proposed | Speedup |
|---:|---:|---:|---:|
| 5 | 2.13 ms | 1.63 ms | 1.31× |
| 25 | 6.07 ms | 4.86 ms | 1.25× |
| 125 | 26.92 ms | 17.08 ms | 1.58× |

Small in absolute terms. It is in here because it costs nothing to take: the
list comprehension is also the shorter form.

### 3. The duplicated sort key, two copies → one named function

```python
with_conf.sort(key=lambda p: ("model_0" not in p.name and "rank_0" not in p.name, str(p)))
...
candidates.sort(key=lambda p: ("model_0" not in p.name and "rank_0" not in p.name, str(p)))
return candidates[0]
```

Two identical five-line lambdas, each followed by taking element zero. That is
`min` with a named key:

```python
def _sidecar_rank(pdb: Path) -> tuple:
    """Sort key preferring Boltz's first-ranked model."""
    return ("model_0" not in pdb.name and "rank_0" not in pdb.name, str(pdb))

with_conf = [p for p in candidates if _has_confidence(p)]
return min(with_conf or candidates, key=_sidecar_rank)
```

`min` rather than `sort` also stops doing an O(n log n) sort to read one
element. `with_conf or candidates` is the whole of the preference rule: use
the ones with a confidence JSON if there are any, else all of them.

`_has_confidence` moves from a nested function to a module-level one, because
it closes over nothing and is six lines of `or` rather than two `if`s and a
`return False`.

## Verification

- Every `.pdb` in `tests/characterization/fixtures`, 4 of them: identical
  results.
- Synthetic trees covering all three routing branches × sidecar directory
  present or absent, 6 cases: identical results.

## Is every line necessary

In the proposed version, yes. The one thing I considered cutting and did not
is the `"msa" not in p.parts` filter. It skips cached reference structures
that Boltz writes under `msa/`, and without it a cached reference can win the
`_sidecar_rank` comparison and send `compute_metrics` at the wrong structure.
The comment on it says "skip cached reference structures", which is the
reason, so it stays.
